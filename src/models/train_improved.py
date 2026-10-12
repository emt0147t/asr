"""
Controlled Training Pipeline for Vietnamese Keyword Spotting Models.
Supports comparing:
  - Architecture: baseline (VietnameseKWSNet), temporal_cnn (VietnameseTemporalKWSNet), tcresnet (VietnameseTCResNet)
  - Augmentation: baseline (time shift + scale) vs advanced (SpecAugment + Noise mixing + Time shift)
Trains strictly on train_manifest_v2.csv and validates on val_manifest_v2.csv (speaker-disjoint).
Held-out test_manifest_v2.csv is NOT used during training or checkpoint selection.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout and sys.path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
else:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import os
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, recall_score

from src.models.kws_architectures import VietnameseKWSNet, VietnameseTemporalKWSNet, VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VietnameseKWSDatasetV2(Dataset):
    """
    Dataset with advanced Keyword Spotting augmentations:
      - Variable time shifts (+- 100 ms)
      - Dynamic background noise mixing (from SILENCE pool at SNR 10-25 dB)
      - SpecAugment (frequency and time masking)
      - Volume perturbation
    """
    def __init__(self, manifest_csv: Path, is_train: bool = True, augment_level: str = "advanced"):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment_level = augment_level if is_train else "none"
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)

        # Preload audio into memory (< 100 MB total)
        self.audio_cache = []
        self.noise_cache = []

        for idx, row in self.df.iterrows():
            wav_path = Path(row["processed_path"])
            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                elif len(audio) > 16000:
                    audio = audio[:16000]
                self.audio_cache.append(audio)
                if row["class_label"] == "SILENCE":
                    self.noise_cache.append(audio)
            except Exception:
                zeros = np.zeros(16000, dtype=np.float32)
                self.audio_cache.append(zeros)

        if not self.noise_cache:
            self.noise_cache.append(np.zeros(16000, dtype=np.float32))

    def __len__(self):
        return len(self.df)

    def _apply_spec_augment(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Applies SpecAugment on (1, n_mels=40, time=101):
          - Frequency masking: mask up to 6 mel bands
          - Time masking: mask up to 10 time frames
        """
        # Frequency masking
        f_mask_len = random.randint(0, 6)
        if f_mask_len > 0:
            f0 = random.randint(0, 40 - f_mask_len)
            log_mel[:, f0:f0 + f_mask_len, :] = 0.0

        # Time masking
        t_mask_len = random.randint(0, 10)
        if t_mask_len > 0:
            t0 = random.randint(0, 101 - t_mask_len)
            log_mel[:, :, t0:t0 + t_mask_len] = 0.0

        return log_mel

    def __getitem__(self, idx):
        audio = self.audio_cache[idx].copy()
        label = int(self.df.iloc[idx]["class_id"])

        if self.augment_level == "baseline":
            # Baseline augmentation: small shift, volume scale, light white noise
            shift = np.random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]
            scale = np.random.uniform(0.8, 1.2)
            audio = np.clip(audio * scale, -1.0, 1.0)
            if np.random.rand() > 0.5:
                noise = np.random.randn(len(audio)) * np.random.uniform(0.001, 0.008)
                audio = np.clip(audio + noise, -1.0, 1.0)

        elif self.augment_level == "advanced":
            # Advanced augmentation for robust KWS:
            # 1. Temporal shift (+- 1600 samples = +- 100 ms)
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]

            # 2. Volume scaling (0.7x to 1.3x)
            scale = np.random.uniform(0.7, 1.3)
            audio = audio * scale

            # 3. Dynamic background noise mixing (30% probability)
            if label != 5 and np.random.rand() < 0.40:
                noise_sample = random.choice(self.noise_cache)
                speech_power = np.mean(audio ** 2) + 1e-9
                noise_power = np.mean(noise_sample ** 2) + 1e-9
                target_snr_db = np.random.uniform(10.0, 25.0)
                target_snr_linear = 10.0 ** (target_snr_db / 10.0)
                noise_gain = np.sqrt(speech_power / (noise_power * target_snr_linear))
                audio = audio + noise_sample * noise_gain

            audio = np.clip(audio, -1.0, 1.0)

        # Compute log-mel spectrogram
        audio_tensor = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel = self.extractor(audio_tensor)  # (1, 40, 101)

        if self.augment_level == "advanced" and np.random.rand() < 0.50:
            log_mel = self._apply_spec_augment(log_mel)

        return log_mel.squeeze(0), label


def train_model(arch: str = "temporal_cnn",
                augment_level: str = "advanced",
                epochs: int = 30,
                batch_size: int = 64,
                lr: float = 1e-3,
                seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"TRAINING EXPERIMENT: Arch = {arch.upper()} | Augment = {augment_level.upper()}")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | LR: {lr}")
    print("=" * 65)

    train_csv = Path("data/manifests/train_manifest_v2.csv")
    val_csv = Path("data/manifests/val_manifest_v2.csv")

    train_dataset = VietnameseKWSDatasetV2(train_csv, is_train=True, augment_level=augment_level)
    val_dataset = VietnameseKWSDatasetV2(val_csv, is_train=False, augment_level="none")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train Dataset: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"Val Dataset:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    # Model instantiation
    if arch == "baseline":
        model = VietnameseKWSNet(num_classes=6, dropout_p=0.3).to(device)
    elif arch == "temporal_cnn":
        model = VietnameseTemporalKWSNet(num_classes=6, dropout_p=0.3).to(device)
    elif arch == "tcresnet":
        model = VietnameseTCResNet(num_classes=6, dropout_p=0.3).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"kws_{arch}_{augment_level}.pt"

    best_val_f1 = -1.0
    best_trai_recall = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Train phase
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_targets = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_targets.extend(y.cpu().numpy())

        scheduler.step()
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        train_f1 = f1_score(all_train_targets, all_train_preds, average="macro", zero_division=0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)
        
        # Per-class recall in validation (0=TRAI, 1=PHAI, 2=LEN, 3=XUONG)
        per_class_recalls = recall_score(all_val_targets, all_val_preds, average=None, labels=range(6), zero_division=0)
        val_trai_rec = per_class_recalls[0]
        val_phai_rec = per_class_recalls[1]

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | "
              f"TRAI-Rec: {val_trai_rec:.3f} PHAI-Rec: {val_phai_rec:.3f}", flush=True)

        # Save best model based on speaker-disjoint validation Macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_trai_recall = val_trai_rec
            best_metrics = {
                "epoch": epoch,
                "arch": arch,
                "augment_level": augment_level,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "val_trai_recall": val_trai_rec,
                "val_phai_recall": val_phai_rec,
                "per_class_recalls": per_class_recalls.tolist()
            }
            torch.save({
                "epoch": epoch,
                "arch": arch,
                "augment_level": augment_level,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "val_trai_recall": val_trai_rec
            }, best_ckpt_path)

    print(f"\n[TRAINING FINISHED] Best Val Macro-F1: {best_val_f1:.4f} (TRAI Recall: {best_trai_recall:.3f})")
    print(f"Checkpoint saved: {best_ckpt_path}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="temporal_cnn", choices=["baseline", "temporal_cnn", "tcresnet"])
    parser.add_argument("--augment_level", type=str, default="advanced", choices=["baseline", "advanced"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        arch=args.arch,
        augment_level=args.augment_level,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
