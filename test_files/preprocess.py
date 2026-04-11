"""
preprocess.py — HAV-DF Dataset Preprocessing
Extracts:
  - Face crops from video frames (via dlib or MTCNN fallback)
  - Mel-spectrograms from audio tracks (via librosa)

Usage:
    python preprocess.py --data_root /path/to/HAV-DF --output_dir ./processed
"""

import os
import cv2
import json
import argparse
import numpy as np
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
import subprocess

# ── Optional face detection backends ─────────────────────────────────────────
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
FRAME_RATE       = 1        # frames per second to sample (1 fps is enough for CPU)
FACE_SIZE        = 224      # face crop resolution (H x W)
N_MELS           = 128      # mel-spectrogram bins
HOP_LENGTH       = 512
N_FFT            = 1024
AUDIO_SR         = 16000    # resample all audio to 16 kHz
MAX_AUDIO_SEC    = 10       # truncate/pad audio to this length
MAX_FRAMES       = 10       # max face frames per video


# ── Face detector factory ──────────────────────────────────────────────────────
def get_face_detector():
    if DLIB_AVAILABLE:
        detector = dlib.get_frontal_face_detector()
        print("[INFO] Using dlib face detector.")
        return ("dlib", detector)
    elif MTCNN_AVAILABLE:
        detector = MTCNN(keep_all=False, device="cpu")
        print("[INFO] Using MTCNN face detector.")
        return ("mtcnn", detector)
    else:
        print("[WARN] No face detector found. Using full-frame fallback.")
        return ("none", None)


def detect_face_dlib(frame_rgb, detector):
    dets = detector(frame_rgb, 1)
    if len(dets) == 0:
        return None
    d = dets[0]
    return (d.left(), d.top(), d.right(), d.bottom())


def detect_face_mtcnn(frame_rgb, detector):
    boxes, _ = detector.detect(frame_rgb)
    if boxes is None or len(boxes) == 0:
        return None
    b = boxes[0].astype(int)
    return (b[0], b[1], b[2], b[3])


def crop_face(frame_bgr, bbox, size=FACE_SIZE, margin=0.25):
    """Crop face with margin, resize to size x size."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))


# ── Audio extraction ───────────────────────────────────────────────────────────
def extract_audio_from_video(video_path: str, out_wav: str, sr: int = AUDIO_SR):
    """Use ffmpeg to extract audio from video."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",               # mono
        "-ar", str(sr),            # resample
        "-vn",                     # no video
        out_wav
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def wav_to_melspectrogram(wav_path: str, sr: int = AUDIO_SR,
                           n_mels: int = N_MELS, max_sec: int = MAX_AUDIO_SEC):
    """
    Returns mel-spectrogram as float32 numpy array [n_mels, T].
    Truncated or zero-padded to max_sec * sr samples.
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    max_samples = max_sec * sr
    if len(y) >= max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Normalize to [-1, 1]
    mel_db = mel_db / 80.0  # librosa dB range is typically [-80, 0]
    return mel_db  # [n_mels, T]


# ── Video frame extraction ─────────────────────────────────────────────────────
def extract_face_frames(video_path: str, detector_info, fps: int = FRAME_RATE,
                         max_frames: int = MAX_FRAMES):
    """
    Extracts up to max_frames face crops from video at given fps.
    Returns list of numpy arrays [H, W, 3] (BGR, uint8).
    """
    dtype, detector = detector_info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(native_fps / fps))

    faces = []
    frame_idx = 0
    while len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = None
            if dtype == "dlib":
                bbox = detect_face_dlib(frame_rgb, detector)
            elif dtype == "mtcnn":
                bbox = detect_face_mtcnn(frame_rgb, detector)

            if bbox is not None:
                face = crop_face(frame, bbox)
            else:
                # fallback: center crop
                h, w = frame.shape[:2]
                s = min(h, w)
                y0, x0 = (h - s) // 2, (w - s) // 2
                face = cv2.resize(frame[y0:y0+s, x0:x0+s], (FACE_SIZE, FACE_SIZE))

            if face is not None:
                faces.append(face)
        frame_idx += 1

    cap.release()
    return faces


# ── Main processing ────────────────────────────────────────────────────────────
def process_dataset(data_root: str, output_dir: str):
    """
    Expected dataset structure (adjust as needed for actual HAV-DF layout):
      data_root/
        real/   *.mp4
        fake/   *.mp4

    Outputs:
      output_dir/
        real/<video_id>/
          frame_000.npy  ... face crops [224, 224, 3]
          mel.npy        ... mel-spectrogram [128, T]
        fake/<video_id>/
          ...
        manifest.json   ... list of {id, label, n_frames, has_audio}
    """
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_info = get_face_detector()
    manifest = []
    tmp_wav = "/tmp/havdf_tmp_audio.wav"

    for label_name, label_int in [("real", 0), ("fake", 1)]:
        src_dir = data_root / label_name
        if not src_dir.exists():
            print(f"[WARN] {src_dir} not found, skipping.")
            continue

        videos = sorted(src_dir.glob("*.mp4")) + sorted(src_dir.glob("*.avi"))
        print(f"\n[INFO] Processing {len(videos)} {label_name} videos...")

        for vpath in tqdm(videos, desc=label_name):
            vid_id = vpath.stem
            out_vid_dir = output_dir / label_name / vid_id
            out_vid_dir.mkdir(parents=True, exist_ok=True)

            # ── Face frames ────────────────────────────────────────────────
            frames_done = False
            n_frames    = 0
            frame_npys  = list(out_vid_dir.glob("frame_*.npy"))
            if frame_npys:
                n_frames    = len(frame_npys)
                frames_done = True
            else:
                faces = extract_face_frames(str(vpath), detector_info)
                for i, face in enumerate(faces):
                    np.save(out_vid_dir / f"frame_{i:03d}.npy", face)
                n_frames    = len(faces)
                frames_done = n_frames > 0

            # ── Audio mel-spec ─────────────────────────────────────────────
            has_audio  = False
            mel_path   = out_vid_dir / "mel.npy"
            if mel_path.exists():
                has_audio = True
            else:
                ok = extract_audio_from_video(str(vpath), tmp_wav)
                if ok and os.path.exists(tmp_wav):
                    try:
                        mel = wav_to_melspectrogram(tmp_wav)
                        np.save(mel_path, mel)
                        has_audio = True
                    except Exception as e:
                        print(f"[WARN] Audio mel failed for {vid_id}: {e}")

            manifest.append({
                "id":        vid_id,
                "label":     label_int,
                "label_str": label_name,
                "n_frames":  n_frames,
                "has_audio": has_audio,
                "path":      str(out_vid_dir.relative_to(output_dir))
            })

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total  = len(manifest)
    n_real = sum(1 for m in manifest if m["label"] == 0)
    n_fake = sum(1 for m in manifest if m["label"] == 1)
    print(f"\n[DONE] Processed {total} videos: {n_real} real, {n_fake} fake.")
    print(f"       Manifest saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True, help="Path to HAV-DF dataset root")
    parser.add_argument("--output_dir", default="./processed", help="Where to save processed data")
    args = parser.parse_args()
    process_dataset(args.data_root, args.output_dir)
