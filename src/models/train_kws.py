"""
Baseline Training Pipeline for Vietnamese KWS 6-Class Model.
Trains on data/manifests/train_manifest.csv and validates on data/manifests/val_manifest.csv.
Saves best model weights and training history.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout and sys.path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import os
import time
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.models.kws_cnn import VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor


class VietnameseKWSDataset(Dataset):
    """
    Dataset loading preprocessed 1.0s 16kHz WAV files and extracting Log-Mel spectrograms.
    """
    def __init__(self, manifest_csv: Path, is_train: bool = True, augment: bool = True):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment = augment and is_train
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
        
        # Preload audio into memory if total dataset fits easily in RAM (< 100 MB)
        self.audio_cache = []
        for idx, row in self.df.iterrows():
            wav_path = Path(row["processed_path"])
            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                elif len(audio) > 16000:
                    audio = audio[:16000]
                self.audio_cache.append(audio)
            except Exception:
                self.audio_cache.append(np.zeros(16000, dtype=np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio = self.audio_cache[idx].copy()
        label = int(self.df.iloc[idx]["class_id"])

        # Augmentation for training
        if self.augment:
            # 1. Random time shift (+- 800 samples = +- 50 ms)
            shift = np.random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]
            
            # 2. Random volume scaling (0.8x to 1.2x)
            scale = np.random.uniform(0.8, 1.2)
            audio = np.clip(audio * scale, -1.0, 1.0)

            # 3. Add light Gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.randn(len(audio)) * np.random.uniform(0.001, 0.008)
                audio = np.clip(audio + noise, -1.0, 1.0)

        # Convert to tensor and compute log-mel
        audio_tensor = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel = self.extractor(audio_tensor)  # (1, 40, 101)

        return log_mel.squeeze(0), label


def train_baseline(epochs: int = 25, batch_size: int = 64, lr: float = 1e-3):
    print("=" * 60)
    print("TRAINING VIETNAMESE KWS CNN BASELINE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on Device: {device}")

    train_csv = Path("data/manifests/train_manifest.csv")
    val_csv = Path("data/manifests/val_manifest.csv")

    train_dataset = VietnameseKWSDataset(train_csv, is_train=True, augment=True)
    val_dataset = VietnameseKWSDataset(val_csv, is_train=False, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train Dataset: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"Val Dataset:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    model = VietnameseKWSNet(num_classes=6, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_kws_model.pt"

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Train phase
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []

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
        train_f1 = f1_score(all_train_targets, all_train_preds, average="macro")

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

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
        val_f1 = f1_score(all_val_targets, all_val_preds, average="macro")
        t_epoch = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} [{t_epoch:.1f}s]: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.1f}%, Train F1={train_f1:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.1f}%, Val F1={val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "num_classes": 6
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val Macro-F1 = {val_f1:.4f})")

    # Save training history
    with open(ckpt_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[TRAINING FINISHED] Best Val Macro-F1: {best_val_f1:.4f} saved to {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    train_baseline(epochs=20, batch_size=64, lr=1e-3)
