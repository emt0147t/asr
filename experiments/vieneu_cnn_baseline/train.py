"""
Phase 26: Four-Class CNN Baseline Training Pipeline for VieNeu-TTS Dataset.
Trains a compact 2D-CNN (VietnameseKWSNet) on 680 synthetic training utterances
and validates on 160 validation utterances across disjoint synthetic voice presets.

Classes:
  0: TRAI (Trái)
  1: PHAI (Phải)
  2: LEN (Lên)
  3: XUONG (Xuống)
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

LABEL_TO_ID = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3}
ID_TO_LABEL = {0: "TRAI", 1: "PHAI", 2: "LEN", 3: "XUONG"}
CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG"]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VieNeuKWSDataset(Dataset):
    """Loads 16 kHz 1.0s WAVs and computes log-Mel spectrograms."""
    def __init__(self, manifest_csv: str, is_train: bool = True, augment: bool = True, seed: int = 42):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment = augment and is_train
        self.seed = seed
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160)

        # Preload 16,000-sample audio clips into memory for high-speed GPU training
        self.audio_data = []
        self.labels = []
        self.sample_ids = []

        for idx, row in self.df.iterrows():
            kws_rel = row["kws_wav_path"]
            kws_full = os.path.join(BASE_DIR, kws_rel)
            data, sr = sf.read(kws_full, dtype="float32")
            if len(data) < 16000:
                data = np.pad(data, (0, 16000 - len(data)))
            elif len(data) > 16000:
                data = data[:16000]

            self.audio_data.append(data)
            self.labels.append(LABEL_TO_ID[row["class_label"]])
            self.sample_ids.append(row["sample_id"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.audio_data[idx].copy()
        label = self.labels[idx]

        # Mild training-only augmentation
        if self.augment:
            # 1. Subtle time shift (+- 400 samples = +- 25ms, within safe silent margins)
            shift = np.random.randint(-400, 401)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]

            # 2. Subtle gain scaling (0.9x to 1.1x)
            scale = np.random.uniform(0.90, 1.10)
            audio = np.clip(audio * scale, -0.99, 0.99)

        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            feat = self.extractor(audio_t)  # (1, 40, 101)

        return feat.squeeze(0), label

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for feats, targets in loader:
        feats = feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError("Non-finite loss detected during training.")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1

def evaluate_val(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            targets = targets.to(device)

            logits = model(feats)
            loss = criterion(logits, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError("Non-finite loss detected during validation.")

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1

def run_training(seed: int, config_path: str = None):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(exp_dir, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Starting 4-Class CNN Baseline Training (Seed {seed}) on {device}")
    print("=" * 70)

    train_manifest = os.path.join(BASE_DIR, cfg["splits"]["train_manifest"])
    val_manifest = os.path.join(BASE_DIR, cfg["splits"]["val_manifest"])

    train_ds = VieNeuKWSDataset(train_manifest, is_train=True, augment=True, seed=seed)
    val_ds = VieNeuKWSDataset(val_manifest, is_train=False, augment=False, seed=seed)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_ds)} train clips and {len(val_ds)} validation clips.")

    # 4-Class Compact CNN
    model = VietnameseKWSNet(num_classes=4, dropout_p=cfg["model"]["dropout_p"])
    model.to(device)
    param_count = get_model_param_count(model)
    print(f"Initialized VietnameseKWSNet (4 classes). Trainable parameters: {param_count:,}")

    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = -1
    best_val_acc = -1.0
    patience = cfg["training"]["early_stopping_patience"]
    min_epochs = cfg["training"]["min_epochs"]
    patience_counter = 0

    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    logs_dir = os.path.join(exp_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    best_ckpt_path = os.path.join(checkpoints_dir, f"best_checkpoint_seed{seed}.pt")
    history_path = os.path.join(logs_dir, f"history_seed{seed}.json")

    history = {
        "seed": seed,
        "device": str(device),
        "param_count": param_count,
        "epochs": []
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = evaluate_val(model, val_loader, criterion, device)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "train_f1": float(tr_f1),
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
            "val_f1": float(va_f1)
        }
        history["epochs"].append(epoch_record)

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% F1: {tr_f1*100:.1f}% | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.1f}% F1: {va_f1*100:.1f}%", flush=True)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_val_acc = va_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": va_f1,
                "val_acc": va_acc,
                "val_loss": va_loss,
                "num_classes": 4,
                "class_names": CLASS_NAMES,
                "label_to_id": LABEL_TO_ID,
                "param_count": param_count
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val F1: {va_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if epoch >= min_epochs and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

    elapsed_time = time.time() - t0
    history["total_time_sec"] = elapsed_time
    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = best_val_f1
    history["best_val_acc"] = best_val_acc

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Training Complete for Seed {seed} in {elapsed_time:.1f}s")
    print(f"Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_f1*100:.2f}% | Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved: {best_ckpt_path}")
    print("=" * 70)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "checkpoint_path": best_ckpt_path,
        "elapsed_time": elapsed_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-class CNN baseline on VieNeu-TTS data")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--all-seeds", action="store_true", help="Run multi-seed training (42, 43, 44)")
    args = parser.parse_args()

    if args.all_seeds:
        seeds = [42, 43, 44]
        summaries = []
        for s in seeds:
            res = run_training(s)
            summaries.append(res)
        print("\nMulti-Seed Training Summary:")
        for r in summaries:
            print(f"Seed {r['seed']}: Best Val F1 = {r['best_val_macro_f1']*100:.2f}%, Best Val Acc = {r['best_val_acc']*100:.2f}% (Epoch {r['best_epoch']})")
    else:
        run_training(args.seed)
