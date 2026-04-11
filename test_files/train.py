"""
train.py — Training loop for HAVDFDetector

Features:
  - Two-phase training: warm-up (frozen backbone) → fine-tune (full model)
  - Mixed loss: CrossEntropy + label smoothing
  - Metrics: Accuracy, AUC-ROC, EER (Equal Error Rate) — standard spoof detection
  - Cosine LR schedule with warm restarts
  - Checkpoint: saves best model by val AUC

Usage:
    python train.py --processed_dir ./processed --output_dir ./runs/exp1
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from dataset import build_dataloaders
from model import HAVDFDetector, count_parameters


# ── EER utility ───────────────────────────────────────────────────────────────
def compute_eer(y_true, y_score):
    """Equal Error Rate: FRR = FAR point on ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    # Find crossing
    eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    return eer * 100.0   # as percentage


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for frames, mel, labels in loader:
        frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
        logits = model(frames, mel)
        loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = (all_preds == all_labels).mean() * 100.0
    auc      = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    eer      = compute_eer(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 50.0

    return {"loss": avg_loss, "acc": acc, "auc": auc, "eer": eer}


# ── Training step ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for frames, mel, labels in loader:
        frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(frames, mel)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)

    return {"loss": total_loss / n, "acc": correct / n * 100.0}


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        args.processed_dir,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = HAVDFDetector(
        freeze_video_backbone=True,   # Phase 1: backbone frozen
        fusion_dim=256,
        dropout=0.4,
    ).to(device)
    print(f"[INFO] Parameters — {count_parameters(model)}")

    # ── Loss: label smoothing helps on small datasets ──────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc      = 0.0
    history       = []
    unfreeze_done = False

    print(f"\n{'='*60}")
    print(f"  Training HAV-DF Detector   [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):

        # ── Phase 2: unfreeze backbone after warm-up ───────────────────────
        if not unfreeze_done and epoch > args.warmup_epochs:
            model.unfreeze_video_backbone()
            # Re-create optimizer with lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.video_stream.features.parameters(), "lr": args.lr * 0.1},
                {"params": [p for n, p in model.named_parameters()
                             if "video_stream.features" not in n], "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            unfreeze_done = True
            print(f"  [Epoch {epoch}] Phase 2: Fine-tuning full model.\n")

        # ── Train ─────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        row = {
            "epoch":      epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc":  round(train_metrics["acc"],  2),
            "val_loss":   round(val_metrics["loss"],   4),
            "val_acc":    round(val_metrics["acc"],    2),
            "val_auc":    round(val_metrics["auc"],    4),
            "val_eer":    round(val_metrics["eer"],    2),
        }
        history.append(row)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {row['train_loss']:.4f}  Acc: {row['train_acc']:.1f}% | "
              f"Val Loss: {row['val_loss']:.4f}  Acc: {row['val_acc']:.1f}%  "
              f"AUC: {row['val_auc']:.4f}  EER: {row['val_eer']:.1f}%")

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     best_auc,
                "val_eer":     val_metrics["eer"],
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best AUC {best_auc:.4f} — saved to {ckpt_path}")

    # ── Final test evaluation ──────────────────────────────────────────────
    print("\n[INFO] Loading best model for test evaluation...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"  Accuracy : {test_metrics['acc']:.2f}%")
    print(f"  AUC-ROC  : {test_metrics['auc']:.4f}")
    print(f"  EER      : {test_metrics['eer']:.2f}%")
    print(f"{'='*60}\n")

    # ── Save history ──────────────────────────────────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)
    print(f"[INFO] Training history saved to {out_dir / 'history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",  required=True,       help="Output dir from preprocess.py")
    parser.add_argument("--output_dir",     default="./runs/exp1")
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--warmup_epochs",  type=int, default=10, help="Epochs before unfreezing backbone")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
