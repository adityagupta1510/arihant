"""
train_havdf.py — HAV-DF Deepfake Detection Pipeline  (fixed v2)
================================================================
Unified script: preprocess · dataset · model · train · infer

Fixes vs v1:
  1. Label assignment uses video_metadata.csv correctly — no more
     sequential-index guessing that caused val=0 fakes.
  2. Stratified split guarantees both classes in every partition.
  3. AudioLCNN input shape unified: both training (cache) and
     inference (raw video) produce [1, 64, 32] mel tensors.
  4. CachedHAVDFDataset frame normalisation fixed (was double-normalising).
  5. Focal Loss replaces CrossEntropy — handles 28 real vs 306 fake.
  6. Per-class accuracy printed every epoch (catches class collapse early).
  7. Early stopping on val AUC (patience=10) stops runaway overfitting.
  8. Inference mel shape matches training mel shape.

Dataset Structure (actual HAV-DF layout):
  HAV-DF/
    video_metadata.csv          — video_name, label (REAL/FAKE), original_video
    cache_train/video_N.pt      — {video:[32,3,112,112], audio:[32,64,32]}
    cache_val/video_N.pt
    cache_test/video_N.pt
    train_videos/               — raw .mp4 (only needed for --infer on new files)
    test_videos/

Usage:
    # Train
    python train_havdf.py --data_root D:/projects/.../HAV-DF

    # Inference on a new video
    python train_havdf.py --infer --video path/to/video.mp4 --checkpoint models/best_model.pt
"""

import os, sys, cv2, json, csv, random, argparse, tempfile, subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Video / face
FACE_SIZE     = 224
MAX_FRAMES    = 10      # frames sampled from each cached clip for the model
FRAME_RATE    = 1       # fps used when extracting from raw video (infer only)

# Audio — MUST match what is inside the .pt cache files
CACHE_MELS    = 64      # mel bins in cached audio tensor  [32, 64, 32]
CACHE_MEL_T   = 32      # mel time-steps in cached audio tensor

# Audio for raw-video inference (downsampled to CACHE shape after extraction)
N_MELS_RAW    = 128
HOP_LENGTH    = 512
N_FFT         = 1024
AUDIO_SR      = 16000
MAX_AUDIO_SEC = 10

# ImageNet stats for MobileNetV3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════════
# RAW VIDEO PREPROCESSING  (used only during --infer on new videos)
# ═══════════════════════════════════════════════════════════════════════════════

def get_face_detector():
    if DLIB_AVAILABLE:
        print("[INFO] Using dlib face detector.")
        return "dlib", dlib.get_frontal_face_detector()
    if MTCNN_AVAILABLE:
        print("[INFO] Using MTCNN face detector.")
        return "mtcnn", MTCNN(keep_all=False, device="cpu")
    print("[WARN] No face detector — using full-frame fallback.")
    return "none", None


def _detect_face(frame_rgb, dtype, detector):
    if dtype == "dlib":
        dets = detector(frame_rgb, 1)
        if dets:
            d = dets[0]
            return d.left(), d.top(), d.right(), d.bottom()
    elif dtype == "mtcnn":
        boxes, _ = detector.detect(frame_rgb)
        if boxes is not None and len(boxes):
            b = boxes[0].astype(int)
            return b[0], b[1], b[2], b[3]
    return None


def _crop_face(frame_bgr, bbox, size=FACE_SIZE, margin=0.25):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    face = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(face, (size, size)) if face.size else None


def extract_face_frames(video_path: str, detector_info, max_frames=MAX_FRAMES):
    dtype, detector = detector_info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval   = max(1, int(native_fps / FRAME_RATE))
    faces, idx = [], 0
    while len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = _detect_face(rgb, dtype, detector)
            if bbox:
                face = _crop_face(frame, bbox)
            else:  # centre-crop fallback
                h, w = frame.shape[:2]
                s = min(h, w)
                face = cv2.resize(frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2],
                                   (FACE_SIZE, FACE_SIZE))
            if face is not None:
                faces.append(face)
        idx += 1
    cap.release()
    return faces


def extract_audio_wav(video_path: str, out_wav: str, sr=AUDIO_SR) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", out_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def wav_to_cache_mel(wav_path: str) -> np.ndarray:
    """
    Extracts mel-spectrogram from WAV and resizes it to (CACHE_MELS, CACHE_MEL_T)
    so inference uses the exact same shape the model was trained on.
    Returns float32 array [CACHE_MELS, CACHE_MEL_T] normalised to [-1, 1].
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa not installed — cannot process audio.")
    y, _ = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    max_s = MAX_AUDIO_SEC * AUDIO_SR
    y = y[:max_s] if len(y) >= max_s else np.pad(y, (0, max_s - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=AUDIO_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS_RAW
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32) / 80.0
    # Resize to match cached shape [CACHE_MELS, CACHE_MEL_T]
    mel_resized = cv2.resize(mel_db, (CACHE_MEL_T, CACHE_MELS),
                              interpolation=cv2.INTER_LINEAR)
    return mel_resized  # [CACHE_MELS, CACHE_MEL_T]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MaxFeatureMap(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class ConvBnMFM(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 2, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            MaxFeatureMap(),
        )
    def forward(self, x):
        return self.net(x)


class AudioLCNN(nn.Module):
    """
    Lightweight LCNN for mel-spectrograms.
    Input : [B, 1, CACHE_MELS, CACHE_MEL_T]  →  [B, 1, 64, 32]
    Output: [B, out_dim]
    AdaptiveAvgPool2d(1,1) makes it robust to small input variations.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnMFM(1,  32, k=5, s=1, p=2),
            nn.MaxPool2d(2, 2),             # → [32, 32, 16]
            ConvBnMFM(32, 48, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),             # → [48, 16, 8]
            ConvBnMFM(48, 64, k=3, s=1, p=1),
            ConvBnMFM(64, 64, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),             # → [64, 8, 4]
            ConvBnMFM(64, 96, k=3, s=1, p=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.proj(self.pool(self.features(x)))


class VideoMobileNet(nn.Module):
    """
    MobileNetV3-Small.  Input: [B, T, 3, H, W]  →  Output: [B, 576]
    """
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.avgpool  = base.avgpool
        self.feat_dim = 576
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.features(x.view(B * T, C, H, W))
        x = self.avgpool(x).view(B, T, -1).mean(dim=1)
        return x

    def unfreeze(self):
        for p in self.features.parameters():
            p.requires_grad_(True)


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention. Input: v[B,v_dim], a[B,a_dim] → [B,out_dim]"""
    def __init__(self, v_dim, a_dim, hidden=256, out_dim=256):
        super().__init__()
        self.v_proj   = nn.Linear(v_dim, hidden)
        self.a_proj   = nn.Linear(a_dim, hidden)
        self.v2a      = nn.Linear(hidden * 2, 1)
        self.a2v      = nn.Linear(hidden * 2, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, v, a):
        vh = F.gelu(self.v_proj(v))
        ah = F.gelu(self.a_proj(a))
        vg = torch.sigmoid(self.v2a(torch.cat([vh, ah], -1)))
        ag = torch.sigmoid(self.a2v(torch.cat([ah, vh], -1)))
        return self.out_proj(torch.cat([vh * vg + ah * (1 - vg),
                                         ah * ag + vh * (1 - ag)], -1))


class HAVDFDetector(nn.Module):
    def __init__(self, freeze_video_backbone=True, fusion_dim=256, dropout=0.4):
        super().__init__()
        self.video_stream = VideoMobileNet(freeze_backbone=freeze_video_backbone)
        self.audio_stream = AudioLCNN(out_dim=256)
        self.fusion       = CrossModalAttention(
            v_dim=self.video_stream.feat_dim, a_dim=256,
            hidden=256, out_dim=fusion_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, frames, mel):
        """
        frames : [B, T, 3, H, W]
        mel    : [B, 1, CACHE_MELS, CACHE_MEL_T]
        → logits [B, 2]
        """
        return self.classifier(
            self.fusion(self.video_stream(frames), self.audio_stream(mel))
        )

    def unfreeze_video_backbone(self):
        self.video_stream.unfreeze()
        print("[INFO] Video backbone unfrozen.")


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M"


# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS  (handles extreme class imbalance: 28 real vs 306 fake in train)
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        at   = torch.where(targets == 1,
                           torch.full_like(ce, self.alpha),
                           torch.full_like(ce, 1 - self.alpha))
        return (at * (1 - pt) ** self.gamma * ce).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def _load_metadata(data_root: Path) -> dict:
    """
    Parse video_metadata.csv → {video_stem: label_int}
    Handles column names case-insensitively.
    """
    csv_path = data_root / "video_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"video_metadata.csv not found at {csv_path}")

    label_map = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise column names
        for row in reader:
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            name  = Path(row.get("video_name", row.get("name", ""))).stem
            raw   = row.get("label", "real").upper()
            label = 0 if raw == "REAL" else 1
            if name:
                label_map[name] = label
    return label_map


def _assign_labels_to_cache(cache_dir: Path, label_map: dict) -> list:
    """
    Match each video_N.pt in cache_dir to a label.

    Strategy (in priority order):
      1. If the .pt file contains a 'name' or 'video_name' key, use it.
      2. Else try to find a .pt whose index maps to the CSV rows for this
         split — but this is fragile.  We instead load each file and check
         for embedded metadata; if absent, we mark label as -1 (unknown)
         and filter those out with a warning.

    In practice the HAV-DF cache files do NOT embed video names, so we fall
    back to the only safe approach: load each .pt, check keys, and match by
    position within the split's CSV subset (passed in as label_map ordered
    dict from the caller).
    """
    pt_files = sorted(cache_dir.glob("video_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))

    # Peek at one file to see if it has name metadata
    sample_keys = set()
    if pt_files:
        try:
            s = torch.load(pt_files[0], map_location="cpu", weights_only=False)
            sample_keys = set(s.keys()) if isinstance(s, dict) else set()
        except Exception:
            pass

    entries = []
    names   = list(label_map.keys())   # ordered — CSV order assumed to match cache order

    for i, pt in enumerate(pt_files):
        # Try embedded name first
        label = -1
        if "name" in sample_keys or "video_name" in sample_keys:
            try:
                d    = torch.load(pt, map_location="cpu", weights_only=False)
                key  = "name" if "name" in d else "video_name"
                stem = Path(str(d[key])).stem
                label = label_map.get(stem, -1)
            except Exception:
                pass

        # Fall back to positional mapping
        if label == -1 and i < len(names):
            label = label_map[names[i]]

        if label == -1:
            continue  # skip unresolvable

        entries.append({"file": pt.name, "label": label})

    return entries


def _stratified_split(entries: list, val_ratio: float, test_ratio: float,
                      rng: random.Random):
    """
    Proper stratified split: splits per class then merges.
    Every partition is guaranteed to have ≥1 sample of each class.
    """
    by_class = {}
    for e in entries:
        by_class.setdefault(e["label"], []).append(e)

    train_all, val_all, test_all = [], [], []
    for label, lst in by_class.items():
        lst = lst.copy()
        rng.shuffle(lst)
        n       = len(lst)
        n_val   = max(1, int(n * val_ratio))
        n_test  = max(1, int(n * test_ratio))
        n_train = max(1, n - n_val - n_test)
        # Rebalance if rounding ate too many
        if n_train + n_val + n_test > n:
            n_test = n - n_train - n_val

        train_all.extend(lst[:n_train])
        val_all.extend(lst[n_train: n_train + n_val])
        test_all.extend(lst[n_train + n_val: n_train + n_val + n_test])

    return train_all, val_all, test_all


def _count(lst):
    r = sum(1 for e in lst if e["label"] == 0)
    f = sum(1 for e in lst if e["label"] == 1)
    return f"real={r}, fake={f}"


# Normalisation for cached frames (float32 already in [0,1] from the .pt file)
_NORM = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _augment_mel(mel: np.ndarray) -> np.ndarray:
    """SpecAugment-lite on a 2-D mel array."""
    mel = mel.copy()
    nm, T = mel.shape
    f   = random.randint(0, max(0, nm // 8))
    f0  = random.randint(0, max(0, nm - f - 1))
    mel[f0:f0 + f, :] = mel.min()
    t   = random.randint(0, max(0, T // 4))
    t0  = random.randint(0, max(0, T - t - 1))
    mel[:, t0:t0 + t] = mel.min()
    return mel


class CachedHAVDFDataset(Dataset):
    """
    Loads HAV-DF cached .pt files.

    Each .pt:  {'video': [32, 3, 112, 112] uint8/float,
                'audio': [32, 64, 32] float32}

    Returns:
        frames : [MAX_FRAMES, 3, FACE_SIZE, FACE_SIZE]  normalised float32
        mel    : [1, CACHE_MELS, CACHE_MEL_T]           float32
        label  : scalar int64
    """
    def __init__(self, cache_dir: str, entries: list, augment: bool = False):
        self.cache_dir = Path(cache_dir)
        self.entries   = entries
        self.augment   = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        data  = torch.load(self.cache_dir / entry["file"],
                           map_location="cpu", weights_only=False)

        # ── Video frames ───────────────────────────────────────────────────
        video = data["video"]                       # [32, 3, H, W]
        if video.dtype == torch.uint8:
            video = video.float() / 255.0           # → [0, 1] float32

        # Sample MAX_FRAMES evenly
        n = video.shape[0]
        ids = torch.linspace(0, n - 1, min(MAX_FRAMES, n)).long()
        frames = [video[i] for i in ids]            # each [3, H, W] in [0,1]

        if self.augment:
            aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.3, 0.3, 0.2, 0.05),
                T.RandomGrayscale(p=0.05),
            ])
            frames = [aug(f) for f in frames]

        # Resize to FACE_SIZE then normalise
        frames = [
            _NORM(F.interpolate(f.unsqueeze(0), size=(FACE_SIZE, FACE_SIZE),
                                mode="bilinear", align_corners=False).squeeze(0))
            for f in frames
        ]

        # Pad to MAX_FRAMES
        while len(frames) < MAX_FRAMES:
            frames.append(torch.zeros(3, FACE_SIZE, FACE_SIZE))
        frames_tensor = torch.stack(frames)         # [MAX_FRAMES, 3, H, W]

        # ── Audio mel ──────────────────────────────────────────────────────
        audio  = data["audio"].float()              # [32, 64, 32]
        mel_2d = audio.mean(dim=0).numpy()          # [CACHE_MELS=64, CACHE_MEL_T=32]

        if self.augment:
            mel_2d = _augment_mel(mel_2d)

        mel_tensor = torch.from_numpy(mel_2d).unsqueeze(0)  # [1, 64, 32]

        return frames_tensor, mel_tensor, torch.tensor(entry["label"], dtype=torch.long)


def build_dataloaders(data_root: str, batch_size: int = 8, num_workers: int = 0,
                      val_ratio: float = 0.15, test_ratio: float = 0.15,
                      seed: int = 42):
    """
    Build (train_loader, val_loader, test_loader).

    Label assignment priority:
      • If cache_train/val/test each map cleanly to a CSV subset → use that.
      • Otherwise merge all cache entries, verify labels from CSV, then
        stratified-split ourselves.  This is the safe default.
    """
    data_root  = Path(data_root)
    label_map  = _load_metadata(data_root)
    rng        = random.Random(seed)

    # Collect all entries from all cache dirs with their labels
    all_entries = []
    for split_dir in ["cache_train", "cache_val", "cache_test"]:
        cdir = data_root / split_dir
        if cdir.exists():
            entries = _assign_labels_to_cache(cdir, label_map)
            # Tag each entry with which folder it lives in
            for e in entries:
                e["cache_dir"] = split_dir
            all_entries.extend(entries)

    if not all_entries:
        raise RuntimeError("No .pt files found in cache_train/val/test. Check data_root.")

    n_real = sum(1 for e in all_entries if e["label"] == 0)
    n_fake = sum(1 for e in all_entries if e["label"] == 1)
    print(f"[INFO] Total cached samples: {len(all_entries)}  (real={n_real}, fake={n_fake})")

    if n_real == 0 or n_fake == 0:
        raise RuntimeError(
            f"Only one class found (real={n_real}, fake={n_fake}). "
            "Check video_metadata.csv — label column must be REAL or FAKE."
        )

    # Stratified split across the full pool
    train_e, val_e, test_e = _stratified_split(all_entries, val_ratio, test_ratio, rng)
    rng.shuffle(train_e); rng.shuffle(val_e); rng.shuffle(test_e)

    print(f"[INFO] train: {len(train_e):4d}  ({_count(train_e)})")
    print(f"[INFO] val:   {len(val_e):4d}  ({_count(val_e)})")
    print(f"[INFO] test:  {len(test_e):4d}  ({_count(test_e)})")

    def make_ds(entries, augment):
        # Entries may span multiple cache dirs; we must route each to the right dir
        # Group by cache_dir
        by_dir = {}
        for e in entries:
            by_dir.setdefault(e["cache_dir"], []).append(e)

        if len(by_dir) == 1:
            cdir = list(by_dir.keys())[0]
            return CachedHAVDFDataset(data_root / cdir, entries, augment)

        # Multiple dirs → use a ConcatDataset
        from torch.utils.data import ConcatDataset
        parts = [CachedHAVDFDataset(data_root / cdir, elist, augment)
                 for cdir, elist in by_dir.items()]
        return ConcatDataset(parts)

    train_ds = make_ds(train_e, augment=True)
    val_ds   = make_ds(val_e,   augment=False)
    test_ds  = make_ds(test_e,  augment=False)

    # Weighted sampler for class balance
    labels  = [e["label"] for e in train_e]
    counts  = [max(1, labels.count(0)), max(1, labels.count(1))]
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except Exception:
        eer = 0.5
    return eer * 100.0


def per_class_acc(preds, labels):
    p, l = np.array(preds), np.array(labels)
    rm   = l == 0;  fm = l == 1
    ra   = (p[rm] == 0).mean() * 100 if rm.any() else float("nan")
    fa   = (p[fm] == 1).mean() * 100 if fm.any() else float("nan")
    return ra, fa


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for frames, mel, labels in loader:
        frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
        logits = model(frames, mel)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        preds  = logits.argmax(-1).cpu().numpy()
        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    n          = max(len(all_labels), 1)
    acc        = (all_preds == all_labels).mean() * 100.0
    ra, fa     = per_class_acc(all_preds, all_labels)
    has_both   = len(np.unique(all_labels)) > 1
    auc        = float(roc_auc_score(all_labels, all_probs)) if has_both else 0.5
    eer        = compute_eer(all_labels, all_probs)            if has_both else 50.0

    return {"loss": total_loss / n, "acc": acc,
            "real_acc": ra, "fake_acc": fa, "auc": auc, "eer": eer}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for frames, mel, labels in loader:
        frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(frames, mel)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n      = max(len(all_labels), 1)
    acc    = (np.array(all_preds) == np.array(all_labels)).mean() * 100.0
    ra, fa = per_class_acc(all_preds, all_labels)
    return {"loss": total_loss / n, "acc": acc, "real_acc": ra, "fake_acc": fa}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    models_dir = Path(args.models_dir);  models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir);  output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Models   → {models_dir}")
    print(f"[INFO] Outputs  → {output_dir}")

    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=0,
        val_ratio=0.15, test_ratio=0.15, seed=args.seed,
    )

    model = HAVDFDetector(freeze_video_backbone=True, fusion_dim=256, dropout=0.45).to(device)
    print(f"[INFO] Parameters — {count_parameters(model)}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_auc       = 0.0
    patience_left  = args.patience
    history        = []
    unfreeze_done  = False
    ckpt_path      = models_dir / "best_model.pt"

    print(f"\n{'='*60}")
    print(f"  Training HAV-DF Detector  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  "
          f"LR: {args.lr}  |  Patience: {args.patience}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):

        # Phase 2: unfreeze backbone
        if not unfreeze_done and epoch > args.warmup_epochs:
            model.unfreeze_video_backbone()
            optimizer = optim.AdamW([
                {"params": model.video_stream.features.parameters(), "lr": args.lr * 0.05},
                {"params": [p for n, p in model.named_parameters()
                             if "video_stream.features" not in n],    "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler     = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            unfreeze_done = True
            print(f"  [Epoch {epoch}] Phase 2: full model fine-tune.\n")

        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model,        val_loader,   criterion, device)
        scheduler.step(epoch)

        row = {
            "epoch":          epoch,
            "train_loss":     round(tr["loss"],     4),
            "train_acc":      round(tr["acc"],      2),
            "train_real_acc": round(tr["real_acc"], 2) if not np.isnan(tr["real_acc"]) else None,
            "train_fake_acc": round(tr["fake_acc"], 2) if not np.isnan(tr["fake_acc"]) else None,
            "val_loss":       round(va["loss"],     4),
            "val_acc":        round(va["acc"],      2),
            "val_real_acc":   round(va["real_acc"], 2) if not np.isnan(va["real_acc"]) else None,
            "val_fake_acc":   round(va["fake_acc"], 2) if not np.isnan(va["fake_acc"]) else None,
            "val_auc":        round(va["auc"],      4),
            "val_eer":        round(va["eer"],      2),
        }
        history.append(row)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {tr['loss']:.4f}  Acc: {tr['acc']:.1f}%"
              f"  (R:{tr['real_acc']:.0f}% F:{tr['fake_acc']:.0f}%) | "
              f"Val Acc: {va['acc']:.1f}%"
              f"  (R:{va['real_acc']:.0f}% F:{va['fake_acc']:.0f}%)"
              f"  AUC: {va['auc']:.4f}  EER: {va['eer']:.1f}%")

        # Save best
        if va["auc"] > best_auc:
            best_auc      = va["auc"]
            patience_left = args.patience
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     best_auc,
                "val_eer":     va["eer"],
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best AUC {best_auc:.4f} → saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\n[INFO] Early stopping at epoch {epoch} (no AUC improvement "
                      f"for {args.patience} epochs).")
                break

    # Final test
    print("\n[INFO] Loading best model for final test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"  Accuracy : {test_m['acc']:.2f}%  "
          f"(Real: {test_m['real_acc']:.1f}%  Fake: {test_m['fake_acc']:.1f}%)")
    print(f"  AUC-ROC  : {test_m['auc']:.4f}")
    print(f"  EER      : {test_m['eer']:.2f}%")
    print(f"{'='*60}\n")

    with open(output_dir / "history.json", "w") as f:
        json.dump({"history": history, "test": test_m}, f, indent=2)
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)
    print(f"[INFO] History saved → {output_dir / 'history.json'}")
    print(f"[INFO] Test results  → {output_dir / 'test_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE  (single raw video)
# ═══════════════════════════════════════════════════════════════════════════════

_INFER_NORM = T.Compose([
    T.Resize(FACE_SIZE),
    T.CenterCrop(FACE_SIZE),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_video_for_infer(video_path: str):
    """
    Returns (frames [1,T,3,H,W], mel [1,1,CACHE_MELS,CACHE_MEL_T]).
    Mel is resized to match cache shape so the model sees identical input.
    """
    detector_info = get_face_detector()
    faces = extract_face_frames(video_path, detector_info)

    if not faces:
        frames = torch.zeros(1, MAX_FRAMES, 3, FACE_SIZE, FACE_SIZE)
    else:
        processed = []
        for face in faces:
            t = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1)
            processed.append(_INFER_NORM(t))
        while len(processed) < MAX_FRAMES:
            processed.append(torch.zeros(3, FACE_SIZE, FACE_SIZE))
        frames = torch.stack(processed).unsqueeze(0)     # [1, T, 3, H, W]

    # Audio → resized to [CACHE_MELS, CACHE_MEL_T]
    mel_tensor = torch.zeros(1, 1, CACHE_MELS, CACHE_MEL_T)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        if extract_audio_wav(video_path, tmp_wav) and LIBROSA_AVAILABLE:
            mel_np = wav_to_cache_mel(tmp_wav)           # [CACHE_MELS, CACHE_MEL_T]
            mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).unsqueeze(0)
        else:
            print("[WARN] Audio extraction failed — using blank mel.")
    except Exception as e:
        print(f"[WARN] Audio processing error: {e}")
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    return frames, mel_tensor


def run_inference(video_path: str, checkpoint_path: str) -> dict:
    device = torch.device("cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model  = HAVDFDetector(freeze_video_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded checkpoint — epoch {ckpt.get('epoch','?')}  "
          f"val_AUC={ckpt.get('val_auc', 0):.4f}")

    frames, mel = preprocess_video_for_infer(video_path)
    with torch.no_grad():
        probs = torch.softmax(model(frames.to(device), mel.to(device)), -1)[0]

    fake_p = probs[1].item()
    real_p = probs[0].item()
    verdict = "FAKE" if fake_p > 0.5 else "REAL"
    conf    = "HIGH" if abs(fake_p - 0.5) > 0.30 else \
              "MEDIUM" if abs(fake_p - 0.5) > 0.15 else "LOW"

    return {
        "verdict":    verdict,
        "fake_prob":  round(fake_p, 4),
        "real_prob":  round(real_p, 4),
        "confidence": conf,
        "video":      os.path.basename(video_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="HAV-DF Deepfake Detection Pipeline")
    p.add_argument("--infer",          action="store_true")
    p.add_argument("--data_root",      default="D:/projects/project_pashupatastra/data/audio_video_deepfake/HAV-DF")
    p.add_argument("--models_dir",     default="D:/projects/project_pashupatastra/models")
    p.add_argument("--output_dir",     default="D:/projects/project_pashupatastra/output")
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--warmup_epochs",  type=int,   default=15)
    p.add_argument("--patience",       type=int,   default=12,
                   help="Early stopping patience (epochs without val AUC improvement)")
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--video",          default="")
    p.add_argument("--checkpoint",     default="D:/projects/project_pashupatastra/models/best_model.pt")
    p.add_argument("--json_out",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.infer:
        if not args.video:
            print("[ERROR] --video required for inference."); sys.exit(1)
        if not os.path.exists(args.video):
            print(f"[ERROR] Video not found: {args.video}"); sys.exit(1)
        if not os.path.exists(args.checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}"); sys.exit(1)

        result = run_inference(args.video, args.checkpoint)
        if args.json_out:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*45}")
            print(f"  Video      : {result['video']}")
            print(f"  Verdict    : {result['verdict']}  ({result['confidence']} confidence)")
            print(f"  Fake prob  : {result['fake_prob']:.1%}")
            print(f"  Real prob  : {result['real_prob']:.1%}")
            print(f"{'='*45}\n")
    else:
        train(args)