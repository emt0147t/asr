#!/usr/bin/env python3
"""
Phase 25 Factorial Experiment Training Script
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/train_phase25_factorial.py

Executes a single condition and seed in the 2x2 Factorial Experiment:
- Factor D: D0 (VieNeu Turbo 100%) vs D1 (VieNeu Turbo 90% + VieNeu Nano 10% matched)
- Factor H: H0 (Standard 50:50 Target/REJECT batch prior) vs H1 (Hard-Negative 50:50 Target/REJECT with 50:50 OOV/Acoustic reject split)
- Seeds: 42, 43, 44

Architecture: CorrectedVietnameseKWSNet (Exact Baseline Scale: 138,469 parameters)
Inference Decision Threshold: theta_cmd = 0.20
"""

import sys
import io
import time
import json
import hashlib
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from sklearn.metrics import precision_recall_fscore_support

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
OUTPUT_BASE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase25"

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# ARCHITECTURE & FRONTEND (EXACT E1 BASELINE SCALE)
# -----------------------------------------------------------------------------
class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=20.0,
            f_max=8000.0,
        )

    def forward(self, x):
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)
        return log_mel


class CorrectedVietnameseKWSNet(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (40, 101) -> (20, 50)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (20, 50) -> (10, 25)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (10, 25) -> (5, 12)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))      # (5, 12) -> (1, 1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))

        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)
        return logits


# -----------------------------------------------------------------------------
# DATASET & SAMPLERS
# -----------------------------------------------------------------------------
class FactorialKWSDataset(Dataset):
    def __init__(self, manifest_csv: Path, is_training: bool = False, target_length: int = 16000):
        self.is_training = is_training
        self.target_length = target_length
        self.df = pd.read_csv(manifest_csv)

        # Preload audio into memory
        self.cached_audios = []
        self.labels = []
        self.reject_subgroups = []

        for idx, row in self.df.iterrows():
            wav_path = PROJECT_ROOT / row["audio_path"]
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)

            if len(audio) < target_length:
                pad = target_length - len(audio)
                audio = np.pad(audio, (0, pad), mode="constant")
            elif len(audio) > target_length:
                audio = audio[:target_length]

            max_val = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            if max_val > 1.0:
                audio = audio / max_val

            self.cached_audios.append(audio.astype(np.float32))
            self.labels.append(CLASS_TO_IDX[row["label"]])
            self.reject_subgroups.append(str(row.get("reject_subgroup", "none")))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.cached_audios[idx].copy()
        label = self.labels[idx]

        if self.is_training:
            # 1. Random time shift (-50ms to +50ms)
            shift = random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode="constant")[:self.target_length]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode="constant")[-shift:self.target_length - shift]

            # 2. Random gain (-3dB to +3dB)
            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = audio * gain

            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

        return torch.from_numpy(audio.astype(np.float32)), label


class FactorialBatchSampler:
    """
    Controlled Batch Sampler ensuring exact 50:50 Target/REJECT batch prior.
    H0: 32 Target + 32 REJECT (from all REJECT pool)
    H1: 32 Target + 16 OOV REJECT + 16 Acoustic REJECT
    """
    def __init__(self, dataset: FactorialKWSDataset, mode: str = "H0", batch_size: int = 64, num_batches: int = 85, seed: int = 42):
        self.dataset = dataset
        self.mode = mode.upper()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)

        # Separate indices
        df = dataset.df
        self.target_indices = df[df["label"].isin(["TRAI", "PHAI", "LEN", "XUONG"])].index.to_numpy()
        self.reject_all = df[df["label"] == "REJECT"].index.to_numpy()
        self.reject_oov = df[(df["label"] == "REJECT") & (df["reject_subgroup"] == "oov_word")].index.to_numpy()
        self.reject_acoustic = df[(df["label"] == "REJECT") & (df["reject_subgroup"] != "oov_word")].index.to_numpy()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            t_batch = self.rng.choice(self.target_indices, size=32, replace=True)
            if self.mode == "H0":
                r_batch = self.rng.choice(self.reject_all, size=32, replace=True)
            elif self.mode == "H1":
                oov_batch = self.rng.choice(self.reject_oov, size=16, replace=True)
                ac_batch = self.rng.choice(self.reject_acoustic, size=16, replace=True)
                r_batch = np.concatenate([oov_batch, ac_batch])
            else:
                raise ValueError(f"Unknown mode {self.mode}")

            batch = np.concatenate([t_batch, r_batch])
            self.rng.shuffle(batch)
            yield batch.tolist()


def apply_spec_augment(specs: torch.Tensor, freq_mask_max: int = 4, time_mask_max: int = 10) -> torch.Tensor:
    b, c, num_mels, num_frames = specs.shape
    masked = specs.clone()
    for i in range(b):
        f = random.randint(0, freq_mask_max)
        if f > 0:
            f0 = random.randint(0, num_mels - f)
            masked[i, :, f0 : f0 + f, :] = 0.0

        t = random.randint(0, time_mask_max)
        if t > 0:
            t0 = random.randint(0, num_frames - t)
            masked[i, :, :, t0 : t0 + t] = 0.0
    return masked


# -----------------------------------------------------------------------------
# EVALUATION AT THETA_CMD = 0.20
# -----------------------------------------------------------------------------
def evaluate_model(model: nn.Module, frontend: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_targets = []
    all_preds = []
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.25], device=device))

    with torch.no_grad():
        for waveforms, targets in data_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            specs = frontend(waveforms)
            logits = model(specs)
            loss = ce_loss_fn(logits, targets)
            total_loss += loss.item() * len(targets)

            # Apply theta_cmd = 0.20 decision rule
            probs = F.softmax(logits, dim=-1)  # shape (B, 5)
            # Classes: 0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT
            target_probs = probs[:, :4]
            max_p, max_idx = torch.max(target_probs, dim=-1)

            # If max target prob >= 0.20, predict target class; else REJECT (class 4)
            preds = torch.where(max_p >= 0.20, max_idx, torch.tensor(4, device=device))

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute metrics
    val_loss = total_loss / len(y_true)
    val_acc = float(np.mean(y_true == y_pred))

    # Per-class P, R, F1
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    # Target macro-F1 across 4 target classes
    val_macro_f1 = float(np.mean(f1[:4]))

    metrics = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "per_class": {
            CLASSES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i])
            }
            for i in range(5)
        }
    }
    return metrics


# -----------------------------------------------------------------------------
# MAIN FACTORIAL TRAINING FUNCTION
# -----------------------------------------------------------------------------
def train_factorial_run(condition: str, seed: int):
    print("=" * 80)
    print(f"STARTING PHASE 25 RUN: Condition = {condition}, Seed = {seed}")
    print("=" * 80)

    parts = condition.split("_")
    factor_d = parts[0].upper()
    factor_h = parts[1].upper()

    assert factor_d in ["D0", "D1"], f"Invalid Factor D: {factor_d}"
    assert factor_h in ["H0", "H1"], f"Invalid Factor H: {factor_h}"

    train_manifest_path = MANIFESTS_DIR / ("D0_train_v1.csv" if factor_d == "D0" else "D1_matched_train_v2.csv")
    val_manifest_path = MANIFESTS_DIR / "validation_v1.csv"

    out_dir = OUTPUT_BASE_DIR / f"{condition}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    print(f"Loading datasets into RAM...")
    print(f"  Train: {train_manifest_path.name} ({compute_sha256(train_manifest_path)})")
    print(f"  Val:   {val_manifest_path.name} ({compute_sha256(val_manifest_path)})")

    train_dataset = FactorialKWSDataset(train_manifest_path, is_training=True)
    val_dataset = FactorialKWSDataset(val_manifest_path, is_training=False)

    batch_sampler = FactorialBatchSampler(
        dataset=train_dataset,
        mode=factor_h,
        batch_size=64,
        num_batches=85,
        seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model & frontend
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized CorrectedVietnameseKWSNet with {total_params:,} trainable parameters.")
    assert total_params == 138469, f"Parameter mismatch: {total_params} != 138469"

    # Loss, Optimizer, Scheduler
    ce_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.25], device=device)
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    max_epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    patience = 8
    patience_counter = 0

    log_records = []
    t0_train = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            # Extract features & apply spec augment
            with torch.no_grad():
                specs = frontend(waveforms)
            specs = apply_spec_augment(specs)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(labels)
            train_samples += len(labels)

        scheduler.step()
        epoch_train_loss = train_loss_total / train_samples

        # Validation evaluation at theta_cmd = 0.20
        val_metrics = evaluate_model(model, frontend, val_loader, device)
        val_macro_f1 = val_metrics["val_macro_f1"]
        val_acc = val_metrics["val_accuracy"]
        val_loss = val_metrics["val_loss"]

        log_records.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_acc, 5),
            "val_macro_f1": round(val_macro_f1, 5),
            "lr": round(optimizer.param_groups[0]["lr"], 7)
        })

        is_best = val_macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_metrics": val_metrics,
                "condition": condition,
                "seed": seed
            }, out_dir / "best_model.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}{' [BEST]' if is_best else ''}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    total_duration = time.perf_counter() - t0_train
    best_ckpt_path = out_dir / "best_model.pt"
    ckpt_sha = compute_sha256(best_ckpt_path)

    # Save log CSV
    pd.DataFrame(log_records).to_csv(out_dir / "training_log.csv", index=False)

    # Save summary JSON
    summary = {
        "condition": condition,
        "seed": seed,
        "factor_d": factor_d,
        "factor_h": factor_h,
        "total_epochs": epoch,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_metrics["val_accuracy"],
        "best_val_loss": best_metrics["val_loss"],
        "best_per_class": best_metrics["per_class"],
        "training_duration_seconds": round(total_duration, 2),
        "checkpoint_sha256": ckpt_sha,
        "train_manifest_sha256": compute_sha256(train_manifest_path),
        "val_manifest_sha256": compute_sha256(val_manifest_path)
    }

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Condition {condition} Seed {seed} completed in {total_duration:.1f}s.")
    print(f"       Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_macro_f1:.4f} | Checkpoint SHA: {ckpt_sha[:16]}...")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 25 Factorial Trainer")
    parser.add_argument("--condition", type=str, required=True, choices=["D0_H0", "D0_H1", "D1_H0", "D1_H1"])
    parser.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    args = parser.parse_args()

    train_factorial_run(args.condition, args.seed)


if __name__ == "__main__":
    main()
