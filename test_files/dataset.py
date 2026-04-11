"""
dataset.py — PyTorch Dataset for HAV-DF processed outputs.

Each sample returns:
  frames : Tensor [MAX_FRAMES, 3, 224, 224]  float32, normalized
  mel    : Tensor [1, N_MELS, T]             float32
  label  : Tensor scalar                     int64 (0=real, 1=fake)
"""

import json
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

# ── Constants matching preprocess.py ──────────────────────────────────────────
MAX_FRAMES = 10
FACE_SIZE  = 224
N_MELS     = 128
MEL_T      = 313   # time frames for 10 sec audio @ sr=16000, hop=512

# ── ImageNet normalization for the video backbone ──────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_frame_transform(augment: bool = False):
    ops = []
    if augment:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.RandomRotation(degrees=10),
            T.RandomResizedCrop(FACE_SIZE, scale=(0.85, 1.0)),
        ]
    else:
        ops += [T.CenterCrop(FACE_SIZE)]
    ops += [
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(ops)


class HAVDFDataset(Dataset):
    """
    Loads preprocessed HAV-DF samples from manifest.json.

    Args:
        processed_dir : path to output_dir from preprocess.py
        split         : 'train' | 'val' | 'test' (uses pre-split manifest keys)
        augment       : apply video augmentation (train only)
        manifest_path : override default manifest.json location
    """

    def __init__(self, processed_dir: str, entries: list,
                 augment: bool = False):
        self.root      = Path(processed_dir)
        self.entries   = entries
        self.augment   = augment
        self.frame_tfm = build_frame_transform(augment)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        vid_dir = self.root / entry["path"]

        # ── Load face frames ───────────────────────────────────────────────
        frame_paths = sorted(vid_dir.glob("frame_*.npy"))
        frames = []
        for fp in frame_paths[:MAX_FRAMES]:
            arr = np.load(fp)           # [H, W, 3] BGR uint8
            arr = arr[:, :, ::-1].copy() # BGR → RGB
            t = torch.from_numpy(arr).permute(2, 0, 1)   # [3, H, W] uint8
            t = self.frame_tfm(t)        # [3, 224, 224] float32
            frames.append(t)

        # Pad to MAX_FRAMES with zeros if needed
        while len(frames) < MAX_FRAMES:
            frames.append(torch.zeros(3, FACE_SIZE, FACE_SIZE))

        frames = torch.stack(frames)  # [MAX_FRAMES, 3, 224, 224]

        # ── Load mel-spectrogram ───────────────────────────────────────────
        mel_path = vid_dir / "mel.npy"
        if mel_path.exists():
            mel = np.load(mel_path).astype(np.float32)   # [N_MELS, T]
            # Pad or truncate T dimension
            if mel.shape[1] < MEL_T:
                mel = np.pad(mel, ((0, 0), (0, MEL_T - mel.shape[1])))
            else:
                mel = mel[:, :MEL_T]
            if self.augment:
                mel = self._augment_mel(mel)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, N_MELS, MEL_T]
        else:
            mel_tensor = torch.zeros(1, N_MELS, MEL_T)

        label = torch.tensor(entry["label"], dtype=torch.long)
        return frames, mel_tensor, label

    # ── Mel augmentation ──────────────────────────────────────────────────────
    @staticmethod
    def _augment_mel(mel: np.ndarray) -> np.ndarray:
        """SpecAugment-lite: random frequency & time masking."""
        mel = mel.copy()
        n_mels, T = mel.shape

        # Frequency masking (mask up to 15 mel bins)
        f = random.randint(0, 15)
        f0 = random.randint(0, max(0, n_mels - f - 1))
        mel[f0:f0 + f, :] = mel.min()

        # Time masking (mask up to 30 time frames)
        t = random.randint(0, 30)
        t0 = random.randint(0, max(0, T - t - 1))
        mel[:, t0:t0 + t] = mel.min()

        return mel


# ── Train / Val / Test split utility ──────────────────────────────────────────
def load_splits(processed_dir: str, val_ratio: float = 0.15,
                test_ratio: float = 0.15, seed: int = 42):
    """
    Loads manifest.json and returns (train_entries, val_entries, test_entries).
    Stratified split by label.
    """
    manifest_path = Path(processed_dir) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    rng = random.Random(seed)

    real_entries = [m for m in manifest if m["label"] == 0]
    fake_entries = [m for m in manifest if m["label"] == 1]

    def split_list(lst):
        lst = lst.copy()
        rng.shuffle(lst)
        n      = len(lst)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]

    real_tr, real_val, real_te = split_list(real_entries)
    fake_tr, fake_val, fake_te = split_list(fake_entries)

    train = real_tr + fake_tr
    val   = real_val + fake_val
    test  = real_te + fake_te

    rng.shuffle(train)
    print(f"[INFO] Split — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def build_dataloaders(processed_dir: str, batch_size: int = 8,
                      num_workers: int = 0, seed: int = 42):
    """
    Returns (train_loader, val_loader, test_loader).
    Uses WeightedRandomSampler for class imbalance (308 fake vs 200 real).
    """
    train_entries, val_entries, test_entries = load_splits(processed_dir, seed=seed)

    train_ds = HAVDFDataset(processed_dir, train_entries, augment=True)
    val_ds   = HAVDFDataset(processed_dir, val_entries,   augment=False)
    test_ds  = HAVDFDataset(processed_dir, test_entries,  augment=False)

    # Weighted sampler for class balance
    labels     = [e["label"] for e in train_entries]
    class_counts = [labels.count(0), labels.count(1)]
    weights    = [1.0 / class_counts[l] for l in labels]
    sampler    = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    import sys
    processed_dir = sys.argv[1] if len(sys.argv) > 1 else "./processed"
    tr, va, te = build_dataloaders(processed_dir, batch_size=4)
    frames, mel, labels = next(iter(tr))
    print(f"frames: {frames.shape}, mel: {mel.shape}, labels: {labels}")
