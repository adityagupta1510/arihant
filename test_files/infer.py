"""
infer.py — Run HAV-DF Detector on a new video file.

Usage:
    python infer.py --video path/to/video.mp4 --checkpoint ./runs/exp1/best_model.pt

Output:
    {
      "verdict"    : "FAKE" | "REAL",
      "fake_prob"  : 0.93,
      "real_prob"  : 0.07,
      "confidence" : "HIGH" | "MEDIUM" | "LOW"
    }
"""

import os
import sys
import json
import argparse
import tempfile
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    get_face_detector, extract_face_frames,
    extract_audio_from_video, wav_to_melspectrogram
)
from model import HAVDFDetector
from dataset import (
    IMAGENET_MEAN, IMAGENET_STD, FACE_SIZE,
    MAX_FRAMES, N_MELS, MEL_T
)


# ── Transform for inference (no augmentation) ─────────────────────────────────
FRAME_TRANSFORM = T.Compose([
    T.CenterCrop(FACE_SIZE),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(checkpoint_path: str, device: torch.device) -> HAVDFDetector:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = HAVDFDetector(freeze_video_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val AUC: {ckpt.get('val_auc', '?'):.4f})")
    return model


def preprocess_video(video_path: str):
    """Returns (frames_tensor, mel_tensor) ready for model."""
    detector_info = get_face_detector()

    # ── Video ──────────────────────────────────────────────────────────────
    faces = extract_face_frames(video_path, detector_info, fps=1, max_frames=MAX_FRAMES)
    if not faces:
        print("[WARN] No faces extracted — using blank video tensor.")
        frames = torch.zeros(1, MAX_FRAMES, 3, FACE_SIZE, FACE_SIZE)
    else:
        processed = []
        for face in faces:
            face_rgb = face[:, :, ::-1].copy()            # BGR → RGB
            t = torch.from_numpy(face_rgb).permute(2, 0, 1)
            t = FRAME_TRANSFORM(t)
            processed.append(t)
        while len(processed) < MAX_FRAMES:
            processed.append(torch.zeros(3, FACE_SIZE, FACE_SIZE))
        frames = torch.stack(processed).unsqueeze(0)       # [1, T, 3, H, W]

    # ── Audio ──────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    ok = extract_audio_from_video(video_path, tmp_wav)
    if ok and os.path.exists(tmp_wav):
        try:
            mel = wav_to_melspectrogram(tmp_wav)              # [N_MELS, T]
            if mel.shape[1] < MEL_T:
                mel = np.pad(mel, ((0, 0), (0, MEL_T - mel.shape[1])))
            else:
                mel = mel[:, :MEL_T]
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # [1, 1, N_MELS, T]
        except Exception as e:
            print(f"[WARN] Audio processing failed: {e}. Using blank mel.")
            mel_tensor = torch.zeros(1, 1, N_MELS, MEL_T)
        finally:
            os.unlink(tmp_wav)
    else:
        print("[WARN] Could not extract audio. Using blank mel tensor.")
        mel_tensor = torch.zeros(1, 1, N_MELS, MEL_T)

    return frames, mel_tensor


def interpret_confidence(fake_prob: float) -> str:
    if fake_prob >= 0.80 or fake_prob <= 0.20:
        return "HIGH"
    elif fake_prob >= 0.65 or fake_prob <= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


def run_inference(video_path: str, checkpoint_path: str) -> dict:
    device = torch.device("cpu")
    model  = load_model(checkpoint_path, device)

    print(f"[INFO] Preprocessing: {video_path}")
    frames, mel = preprocess_video(video_path)
    frames = frames.to(device)
    mel    = mel.to(device)

    with torch.no_grad():
        logits = model(frames, mel)
        probs  = F.softmax(logits, dim=-1)[0]

    fake_prob = probs[1].item()
    real_prob = probs[0].item()
    verdict   = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = interpret_confidence(fake_prob)

    result = {
        "verdict":    verdict,
        "fake_prob":  round(fake_prob, 4),
        "real_prob":  round(real_prob, 4),
        "confidence": confidence,
        "video":      os.path.basename(video_path),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="HAV-DF Deepfake Inference")
    parser.add_argument("--video",      required=True,  help="Path to input video")
    parser.add_argument("--checkpoint", required=True,  help="Path to best_model.pt")
    parser.add_argument("--json",       action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    result = run_inference(args.video, args.checkpoint)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*40}")
        print(f"  Video    : {result['video']}")
        print(f"  Verdict  : {result['verdict']}  ({result['confidence']} confidence)")
        print(f"  Fake prob: {result['fake_prob']:.1%}")
        print(f"  Real prob: {result['real_prob']:.1%}")
        print(f"{'='*40}\n")

    return result


if __name__ == "__main__":
    main()
