"""
model.py — Dual-Stream Multimodal Deepfake Detector
  
Architecture:
  ┌─────────────────────────────┐   ┌──────────────────────────────┐
  │  VIDEO STREAM               │   │  AUDIO STREAM                │
  │  MobileNetV3-Small backbone │   │  Lightweight CNN (LCNN-style)│
  │  Per-frame features         │   │  on mel-spectrogram          │
  │  Temporal mean pooling      │   │                              │
  └────────────┬────────────────┘   └──────────────┬───────────────┘
               │  [B, 576]                          │  [B, 256]
               └────────────────┬───────────────────┘
                        Cross-Modal Attention
                             [B, 256]
                          MLP Classifier
                         real (0) / fake (1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ── Helpers ───────────────────────────────────────────────────────────────────
class MaxFeatureMap(nn.Module):
    """Max Feature Map activation used in LCNN."""
    def forward(self, x):
        assert x.shape[1] % 2 == 0, "Channel dim must be even for MFM"
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class ConvBnMFM(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MaxFeatureMap()

    def forward(self, x):
        return self.mfm(self.bn(self.conv(x)))


# ── Audio Stream: Lightweight LCNN ────────────────────────────────────────────
class AudioLCNN(nn.Module):
    """
    Lightweight CNN on mel-spectrograms inspired by LCNN (Wu et al., 2018).
    Input : [B, 1, N_MELS, T]  (128 × 313)
    Output: [B, 256]
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnMFM(1,   32, k=5, s=1, p=2),   # [B, 32, 128, 313]
            nn.MaxPool2d(2, 2),                   # [B, 32, 64, 156]
            ConvBnMFM(32,  48, k=3, s=1, p=1),   # [B, 48, 64, 156]
            nn.MaxPool2d(2, 2),                   # [B, 48, 32, 78]
            ConvBnMFM(48,  64, k=3, s=1, p=1),   # [B, 64, 32, 78]
            ConvBnMFM(64,  64, k=3, s=1, p=1),   # [B, 64, 32, 78]
            nn.MaxPool2d(2, 2),                   # [B, 64, 16, 39]
            ConvBnMFM(64,  96, k=3, s=1, p=1),   # [B, 96, 16, 39]
            nn.MaxPool2d(2, 2),                   # [B, 96, 8, 19]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.proj(x)


# ── Video Stream: MobileNetV3-Small ───────────────────────────────────────────
class VideoMobileNet(nn.Module):
    """
    MobileNetV3-Small pretrained on ImageNet.
    Processes each frame independently, then mean-pools across time.
    Input : [B, T, 3, 224, 224]
    Output: [B, feat_dim]   (feat_dim = 576 by default)
    """
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Drop the classifier; keep features (outputs [B, 576, 1, 1] after avgpool)
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.feat_dim   = 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)          # [B*T, 3, H, W]
        x = self.features(x)                 # [B*T, 576, 7, 7]
        x = self.avgpool(x)                  # [B*T, 576, 1, 1]
        x = x.view(B, T, -1)                 # [B, T, 576]
        x = x.mean(dim=1)                    # [B, 576]  temporal mean pool
        return x


# ── Cross-Modal Attention Fusion ──────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention between video and audio embeddings.
    Projects both to a shared dim, then:
      - audio attends to video
      - video attends to audio
    Outputs concatenated attended embeddings → linear projection.

    Input : v [B, v_dim], a [B, a_dim]
    Output: [B, out_dim]
    """
    def __init__(self, v_dim: int, a_dim: int, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hidden)
        self.a_proj = nn.Linear(a_dim, hidden)

        # Cross-attention weights
        self.v2a_attn = nn.Linear(hidden * 2, 1)   # video attends to audio
        self.a2v_attn = nn.Linear(hidden * 2, 1)   # audio attends to video

        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, v, a):
        v_h = F.gelu(self.v_proj(v))   # [B, hidden]
        a_h = F.gelu(self.a_proj(a))   # [B, hidden]

        # Video conditioned on audio
        v_gate = torch.sigmoid(self.v2a_attn(torch.cat([v_h, a_h], dim=-1)))
        v_attended = v_h * v_gate + a_h * (1 - v_gate)

        # Audio conditioned on video
        a_gate = torch.sigmoid(self.a2v_attn(torch.cat([a_h, v_h], dim=-1)))
        a_attended = a_h * a_gate + v_h * (1 - a_gate)

        fused = torch.cat([v_attended, a_attended], dim=-1)  # [B, hidden*2]
        return self.out_proj(fused)                           # [B, out_dim]


# ── Full Model ────────────────────────────────────────────────────────────────
class HAVDFDetector(nn.Module):
    """
    Full dual-stream deepfake detector.

    Args:
        freeze_video_backbone : freeze MobileNetV3 weights (good for very small datasets)
        fusion_dim            : dimension of cross-modal fusion output
        dropout               : classifier dropout
    """
    def __init__(self, freeze_video_backbone: bool = True,
                 fusion_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.video_stream  = VideoMobileNet(freeze_backbone=freeze_video_backbone)
        self.audio_stream  = AudioLCNN(out_dim=256)
        self.fusion        = CrossModalAttention(
            v_dim=self.video_stream.feat_dim,
            a_dim=256,
            hidden=256,
            out_dim=fusion_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),   # logits: [real, fake]
        )

    def forward(self, frames, mel):
        """
        Args:
            frames : [B, T, 3, 224, 224]
            mel    : [B, 1, N_MELS, T_mel]
        Returns:
            logits : [B, 2]
        """
        v_feat = self.video_stream(frames)   # [B, 576]
        a_feat = self.audio_stream(mel)      # [B, 256]
        fused  = self.fusion(v_feat, a_feat) # [B, 256]
        return self.classifier(fused)        # [B, 2]

    def predict_proba(self, frames, mel):
        """Returns softmax probability of fake class (index 1)."""
        with torch.no_grad():
            logits = self.forward(frames, mel)
            return F.softmax(logits, dim=-1)[:, 1]

    def unfreeze_video_backbone(self):
        """Call after initial warm-up epochs to fine-tune backbone."""
        for p in self.video_stream.features.parameters():
            p.requires_grad_(True)
        print("[INFO] Video backbone unfrozen.")


# ── Parameter count utility ───────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M"


if __name__ == "__main__":
    model = HAVDFDetector(freeze_video_backbone=True)
    print(model)
    print(count_parameters(model))

    # Dummy forward pass
    B, T = 2, 10
    frames = torch.randn(B, T, 3, 224, 224)
    mel    = torch.randn(B, 1, 128, 313)
    logits = model(frames, mel)
    print(f"Output logits shape: {logits.shape}")   # [2, 2]
