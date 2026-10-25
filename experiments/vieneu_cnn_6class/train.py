"""
Phase 29: Six-Class CNN Baseline Training Pipeline for VieNeu-TTS Dataset.
Trains a compact 2D-CNN (VietnameseKWSNet) on 1,020 synthetic training utterances
(170 per class across 6 classes) and validates on 240 validation utterances across
disjoint synthetic voice presets and disjoint SILENCE conditions.

Classes:
  0: TRAI
  1: PHAI
  2: LEN
  3: XUONG
  4: UNKNOWN
  5: SILENCE
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

LABEL_TO_ID = {
    "TRAI": 0,
    "PHAI": 1,
    "LEN": 2,
    "XUONG": 3,
    "UNKNOWN": 4,
    "SILENCE": 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VieNeuKWS6ClassDataset(Dataset):
    """Loads 16 kHz 1.0s WAVs and computes log-Mel spectrograms for 6 classes."""
    def __init__(self, manifest_csv: str, is_train: bool = True, augment: bool = True, seed: int = 42):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment = augment and is_train
        self.seed = seed
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0)

        # Preload audio into memory
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

        # Training-only mild augmentation
        if self.augment:
            # 1. Subtle time shift (+- 400 samples = +- 25ms, within safe margins)
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
            feat = self.extractor(audio_t)  # (1, 1, 40, 101)

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

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1

def train_baseline_seed(seed: int, config: dict, exp_dir: str):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print(f"Starting 6-Class CNN Baseline Training (Seed {seed}) on {device}")
    print("=" * 70, flush=True)

    splits_cfg = config["splits"]
    train_manifest = os.path.join(BASE_DIR, splits_cfg["train_manifest"])
    val_manifest = os.path.join(BASE_DIR, splits_cfg["val_manifest"])

    train_ds = VieNeuKWS6ClassDataset(train_manifest, is_train=True, augment=True, seed=seed)
    val_ds = VieNeuKWS6ClassDataset(val_manifest, is_train=False, augment=False, seed=seed)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_ds)} train clips and {len(val_ds)} validation clips.", flush=True)

    # Initialize 6-class model from scratch
    model = VietnameseKWSNet(num_classes=6, dropout_p=config["model"]["dropout_p"])
    model.to(device)
    param_count = get_model_param_count(model)
    print(f"Initialized VietnameseKWSNet (6 classes). Trainable parameters: {param_count:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    logs_dir = os.path.join(exp_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    best_val_acc = 0.0
    history = []
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_val(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "train_f1": float(tr_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1)
        })

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            best_ckpt_path = os.path.join(checkpoints_dir, f"best_checkpoint_seed{seed}.pt")
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": float(val_f1),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "num_classes": 6,
                "class_names": CLASS_NAMES,
                "label_to_id": LABEL_TO_ID,
                "param_count": param_count,
                "config": config
            }, best_ckpt_path)
            best_marker = f"  --> Saved new best checkpoint at epoch {epoch} (Val F1: {val_f1*100:.2f}%)"
        else:
            patience_counter += 1
            best_marker = ""

        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% F1: {tr_f1*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1*100:.1f}%{best_marker}", flush=True)

        if epoch >= config["training"]["min_epochs"] and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).", flush=True)
            break

    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"Training Complete for Seed {seed} in {elapsed:.1f}s")
    print(f"Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_f1*100:.2f}% | Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved: {best_ckpt_path}")
    print("=" * 70, flush=True)

    # Save training history
    log_file = os.path.join(logs_dir, f"history_seed{seed}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "device": str(device),
            "param_count": param_count,
            "epochs": history,
            "total_time_sec": elapsed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_f1),
            "best_val_acc": float(best_val_acc)
        }, f, indent=2)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "checkpoint_path": best_ckpt_path
    }

def main():
    parser = argparse.ArgumentParser(description="Train 6-Class KWS CNN Baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (42, 43, 44)")
    parser.add_argument("--all-seeds", action="store_true", help="Train sequentially across seeds 42, 43, 44")
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(exp_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.all_seeds:
        seeds = config["training"]["seeds"]
        results = []
        for s in seeds:
            res = train_baseline_seed(s, config, exp_dir)
            results.append(res)
        print("\nMulti-Seed Training Summary:")
        for r in results:
            print(f"Seed {r['seed']}: Best Val F1 = {r['best_val_f1']*100:.2f}%, Best Val Acc = {r['best_val_acc']*100:.2f}% (Epoch {r['best_epoch']})")
    else:
        train_baseline_seed(args.seed, config, exp_dir)

if __name__ == "__main__":
    main()
