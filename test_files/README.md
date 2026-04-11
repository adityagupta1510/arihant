# HAV-DF Multimodal Deepfake Detector

Dual-stream audio+video deepfake detection for the **Hindi Audio-Video Deepfake (HAV-DF)** dataset.  
Detects face-swap, lip-sync, and voice-cloning manipulations using cross-modal fusion.

---

## Architecture

```
Video Stream                          Audio Stream
─────────────────────────────         ──────────────────────────────
Input: [B, T, 3, 224, 224]           Input: [B, 1, 128, 313]
       ↓                                     ↓
MobileNetV3-Small (pretrained)        Lightweight LCNN
Per-frame features                    (Max Feature Map activations)
Temporal mean pooling                        ↓
       ↓                               [B, 256]
  [B, 576]
               ↘               ↙
           Cross-Modal Attention Fusion
                   [B, 256]
               MLP Classifier
              real (0) / fake (1)
```

**Design choices for CPU deployment:**
- MobileNetV3-Small: 2.5M params, fast inference
- LCNN over mel-spectrograms: proven in anti-spoofing literature
- Cross-attention fusion: captures audio-video sync anomalies
- Total trainable params: ~4–5M (frozen backbone phase 1), ~7M (phase 2)

---

## Setup

```bash
# 1. Clone / download this project
cd havdf/

# 2. Install dependencies
pip install -r requirements.txt

# Note for dlib on Linux:
sudo apt-get install cmake build-essential
pip install dlib

# Note for dlib on macOS:
brew install cmake
pip install dlib
```

---

## Dataset Structure

Expected layout after downloading HAV-DF from Kaggle:
```
HAV-DF/
  real/
    video_001.mp4
    video_002.mp4
    ...
  fake/
    video_001.mp4
    ...
```

---

## Usage

### Step 1 — Preprocess

Extract face crops and mel-spectrograms from all videos:

```bash
python preprocess.py \
  --data_root /path/to/HAV-DF \
  --output_dir ./processed
```

Outputs:
```
processed/
  real/<video_id>/
    frame_000.npy   # face crop [224, 224, 3]
    frame_001.npy
    ...
    mel.npy         # mel-spectrogram [128, 313]
  fake/<video_id>/
    ...
  manifest.json
```

**Runtime:** ~2–4 hours on CPU for 508 videos (face detection is the bottleneck).

---

### Step 2 — Train

```bash
python train.py \
  --processed_dir ./processed \
  --output_dir ./runs/exp1 \
  --epochs 50 \
  --warmup_epochs 10 \
  --batch_size 8 \
  --lr 1e-3
```

**Two-phase training:**
- **Epochs 1–10** (warm-up): MobileNetV3 backbone frozen. Only fusion + classifier + LCNN train.
- **Epochs 11–50** (fine-tune): Full model trains. Backbone LR = 0.1× main LR.

**Expected metrics** (HAV-DF, small dataset):
| Metric | Target |
|--------|--------|
| AUC-ROC | > 0.85 |
| EER | < 15% |
| Accuracy | > 80% |

Checkpoints and `history.json` saved to `--output_dir`.

---

### Step 3 — Inference

```bash
python infer.py \
  --video /path/to/new_video.mp4 \
  --checkpoint ./runs/exp1/best_model.pt
```

Output:
```
========================================
  Video    : suspect_video.mp4
  Verdict  : FAKE  (HIGH confidence)
  Fake prob: 93.2%
  Real prob: 6.8%
========================================
```

For JSON output (pipe-friendly):
```bash
python infer.py --video video.mp4 --checkpoint best_model.pt --json
```

---

## Tips for Small Datasets (508 videos)

1. **Preprocessing augmentation** is applied during training: horizontal flip, color jitter, SpecAugment.
2. **WeightedRandomSampler** handles class imbalance (308 fake vs 200 real) automatically.
3. **Label smoothing** (0.1) reduces overconfidence on small data.
4. If AUC plateaus, try:
   - Reducing `--lr` to `5e-4`
   - Increasing `--warmup_epochs` to `15`
   - Adding more fake/real augmentations in `dataset.py`

---

## File Overview

| File | Purpose |
|------|---------|
| `preprocess.py` | Extract face frames + mel-spectrograms from raw videos |
| `dataset.py` | PyTorch Dataset + train/val/test split + WeightedSampler |
| `model.py` | HAVDFDetector: VideoMobileNet + AudioLCNN + CrossModalAttention |
| `train.py` | Two-phase training loop with AUC/EER metrics |
| `infer.py` | Single-video inference with confidence output |
| `requirements.txt` | Python dependencies |

---

## Citation

If you use the HAV-DF dataset, please cite:

```
Kaur, S., Buhari, M., Khandelwal, N., Tyagi, P., & Sharma, K. (2024).
Hindi Audio-Video Deepfake (HAV-DF): A Hindi Language-Based Audio-Video Deepfake Dataset.
arXiv:2411.15457
```
