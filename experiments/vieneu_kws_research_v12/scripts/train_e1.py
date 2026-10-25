"""
Controlled Training Script for Experiment E1 (Baseline Scale Retraining on Dataset v12).
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/train_e1.py

Hypothesis:
Retraining the baseline-scale architecture (CorrectedVietnameseKWSNet, 138,469 trainable parameters)
on provenance-controlled, acoustically balanced Dataset v12 significantly reduces false acceptance
on conversational speech and unseen OOV confusers ('cửa', 'ghế') while maintaining >=95% clean
keyword recall, proving that dataset quality resolves live-mic instability without architectural expansion.

Mandatory Constraints:
- Architecture: CorrectedVietnameseKWSNet (138,469 trainable parameters)
- Train manifest: experiments/vieneu_kws_research_v12/manifests/train_manifest.csv (5,440 samples)
- Val manifest: experiments/vieneu_kws_research_v12/manifests/val_manifest.csv (1,280 samples)
- Test manifest: LOCKED (zero access during training, selection, or tuning)
- Checkpoint output: experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt
"""

import sys
import io
import time
import json
import hashlib
import random
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
TRAIN_MANIFEST = V12_DIR / "manifests" / "train_manifest.csv"
VAL_MANIFEST = V12_DIR / "manifests" / "val_manifest.csv"
CHECKPOINTS_DIR = V12_DIR / "checkpoints"
REPORTS_DIR = V12_DIR / "reports"
CONFIGS_DIR = V12_DIR / "configs"

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
# ARCHITECTURE & FRONTEND
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
# DATASET & ON-THE-FLY AUGMENTATION
# -----------------------------------------------------------------------------
class KWSDatasetV12(Dataset):
    def __init__(self, manifest_csv: Path, is_training: bool = False, target_length: int = 16000):
        self.is_training = is_training
        self.target_length = target_length
        self.df = pd.read_csv(manifest_csv)

        # Preload audio into RAM for high-throughput training
        self.cached_audios = []
        self.labels = []
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

            # Peak clamping ONLY if max > 1.0 (remediated conditional clamping)
            max_val = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            if max_val > 1.0:
                audio = audio / max_val

            self.cached_audios.append(audio.astype(np.float32))
            self.labels.append(CLASS_TO_IDX[row["label"]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.cached_audios[idx].copy()
        label = self.labels[idx]

        if self.is_training:
            # 1. Random time shift (-50ms to +50ms = -800 to +800 samples)
            shift = random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode="constant")[:self.target_length]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode="constant")[-shift:self.target_length - shift]

            # 2. Random gain (-3dB to +3dB)
            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = audio * gain

            # Conditional clamping if gain pushed max over 1.0
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

        return torch.from_numpy(audio.astype(np.float32)), label


def apply_spec_augment(specs: torch.Tensor, freq_mask_max: int = 4, time_mask_max: int = 10) -> torch.Tensor:
    """Applies frequency and time masking to log-mel spectrogram batch."""
    b, c, num_mels, num_frames = specs.shape
    masked = specs.clone()
    for i in range(b):
        # Frequency masking
        f = random.randint(0, freq_mask_max)
        if f > 0:
            f0 = random.randint(0, num_mels - f)
            masked[i, :, f0 : f0 + f, :] = 0.0

        # Time masking
        t = random.randint(0, time_mask_max)
        if t > 0:
            t0 = random.randint(0, num_frames - t)
            masked[i, :, :, t0 : t0 + t] = 0.0

    return masked


# -----------------------------------------------------------------------------
# TRAINING ROUTINE
# -----------------------------------------------------------------------------
def train_e1(
    seed: int = 42,
    batch_size: int = 64,
    max_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 8,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  STARTING CONTROLLED EXPERIMENT E1 TRAINING (exp_v12_e1_baseline_scale)")
    print(f"  Architecture: CorrectedVietnameseKWSNet (Exact Baseline Scale)")
    print(f"  Device: {device_name}")
    print(f"  Seed: {seed}")
    print("=" * 80)

    set_seed(seed)
    device = torch.device(device_name)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cryptographic hashes of input manifests
    train_sha = compute_sha256(TRAIN_MANIFEST)
    val_sha = compute_sha256(VAL_MANIFEST)
    print(f"Train Manifest ({TRAIN_MANIFEST.name}): SHA-256 = {train_sha}")
    print(f"Val Manifest   ({VAL_MANIFEST.name}):   SHA-256 = {val_sha}")

    # Load datasets
    print("\nLoading training and validation datasets into memory...")
    t0_load = time.perf_counter()
    train_dataset = KWSDatasetV12(TRAIN_MANIFEST, is_training=True)
    val_dataset = KWSDatasetV12(VAL_MANIFEST, is_training=False)
    print(f"Datasets loaded in {time.perf_counter() - t0_load:.2f}s (Train: {len(train_dataset)}, Val: {len(val_dataset)})")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and frontend
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())
    print(f"Model instantiated: Trainable Parameters = {trainable_params:,} (Buffers = {total_buffers})")
    assert trainable_params == 138469, f"Architecture parameter count mismatch: expected 138,469, got {trainable_params}"

    # Class weights: REJECT = 1.25, CMDs = 1.0
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.25], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rates": [],
        "per_class_f1": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = CHECKPOINTS_DIR / f"exp_v12_e1_baseline_scale_seed{seed}.pt"

    t0_train = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for audio_batch, labels in train_loader:
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                specs = frontend(audio_batch)
                # Apply SpecAugment in training
                specs = apply_spec_augment(specs, freq_mask_max=4, time_mask_max=10)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += len(labels)

        scheduler.step()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_confusion = np.zeros((5, 5), dtype=int)

        with torch.no_grad():
            for val_audio, val_labels in val_loader:
                val_audio = val_audio.to(device)
                val_labels = val_labels.to(device)

                specs = frontend(val_audio)
                logits = model(specs)
                loss = criterion(logits, val_labels)

                val_running_loss += loss.item() * len(val_labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == val_labels).sum().item()
                val_total += len(val_labels)

                for tl, pl in zip(val_labels.cpu().numpy(), preds.cpu().numpy()):
                    val_confusion[tl, pl] += 1

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Compute validation per-class F1 and macro F1
        f1_scores = {}
        for ci, cname in enumerate(CLASSES):
            tp = val_confusion[ci, ci]
            fp = val_confusion[:, ci].sum() - tp
            fn = val_confusion[ci, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            f1_scores[cname] = round(float(f1), 4)

        val_macro_f1 = float(np.mean(list(f1_scores.values())))
        epoch_sec = time.perf_counter() - t_epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))
        history["learning_rates"].append(current_lr)
        history["per_class_f1"].append(f1_scores)

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.1f}%, Macro-F1: {val_macro_f1:.4f}) | "
            f"Time: {epoch_sec:.1f}s"
        )

        # Early stopping and best model saving based on val_macro_f1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "architecture": "CorrectedVietnameseKWSNet",
                "trainable_parameters": trainable_params,
                "train_manifest_sha256": train_sha,
                "val_manifest_sha256": val_sha,
                "classes": CLASSES,
                "seed": seed,
            }, best_ckpt_path)
            print(f"  >>> Best model saved at epoch {epoch} (Val Macro-F1: {best_val_macro_f1:.4f}) -> {best_ckpt_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement in val_macro_f1 for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    total_training_time_sec = time.perf_counter() - t0_train
    print(f"\nTraining completed in {total_training_time_sec:.1f}s ({total_training_time_sec/60.0:.2f} min).")
    print(f"Best Val Macro-F1: {best_val_macro_f1:.4f} achieved at Epoch {best_epoch}.")

    # Save training history JSON
    history_file = REPORTS_DIR / "e1_training_history.json"
    history_payload = {
        "experiment_id": "exp_v12_e1_baseline_scale",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "total_training_time_seconds": round(total_training_time_sec, 2),
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": compute_sha256(best_ckpt_path),
        "history": history,
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"Training history saved to: {history_file}")

    # Plot learning curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plot_learning_curves(history, best_epoch, REPORTS_DIR / "e1_learning_curves.png")
    except ImportError:
        print("matplotlib not available; learning curves plot skipped.")

    # Save E1 configuration metadata
    save_e1_config(train_sha, val_sha, best_ckpt_path, history_payload["checkpoint_sha256"], best_val_macro_f1, best_epoch)

    return best_ckpt_path, history_payload


def plot_learning_curves(history: Dict[str, Any], best_epoch: int, output_png: Path):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange", linewidth=2)
    ax1.axvline(x=best_epoch, color="tab:green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("E1 Learning Curves: Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Macro-F1 plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="tab:blue", linestyle=":", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="tab:purple", linestyle=":", linewidth=1.5)
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="tab:green", linewidth=2)
    ax2.axvline(x=best_epoch, color="tab:green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("E1 Validation Metrics: Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Learning curve plot saved to: {output_png}")


def save_e1_config(train_sha: str, val_sha: str, ckpt_path: Path, ckpt_sha: str, best_f1: float, best_epoch: int):
    config_data = {
        "experiment_id": "exp_v12_e1_baseline_scale",
        "description": "Controlled Experiment E1: Baseline-scale retraining on Dataset v12",
        "timestamp": "2026-10-23",
        "model_architecture": {
            "name": "CorrectedVietnameseKWSNet",
            "trainable_parameters": 138469,
            "non_trainable_buffers": 580,
            "total_state_dict_elements": 139049,
            "conv_channels": [32, 64, 64, 128],
            "linear_layers": [[128, 64], [64, 5]],
            "dropout": 0.30,
        },
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR",
            "batch_size": 64,
            "max_epochs": 40,
            "class_weights": [1.0, 1.0, 1.0, 1.0, 1.25],
            "seed": 42,
        },
        "governance_hashes": {
            "train_manifest_sha256": train_sha,
            "val_manifest_sha256": val_sha,
            "e1_checkpoint_path": str(ckpt_path),
            "e1_checkpoint_sha256": ckpt_sha,
        },
        "best_validation_result": {
            "epoch": best_epoch,
            "val_macro_f1": best_f1,
        },
    }
    cfg_file = CONFIGS_DIR / "exp_v12_e1_baseline_scale.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"E1 experiment config saved to: {cfg_file}")


if __name__ == "__main__":
    train_e1()
