"""
Unified Training and Evaluation Script for Phase E2 Capacity Experiment.
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/train_eval_phase_e2.py

Models:
1. Baseline: CorrectedVietnameseKWSNet (138,469 trainable parameters)
2. Scaled E2: ScaledVietnameseKWSNet_E2 (304,277 trainable parameters)

Seeds: [42, 43, 44]
Dataset:
- Training: manifests/train_manifest.csv (5,440 samples)
- Validation: manifests/validation_v1.csv (1,280 samples)
Operating Decision: theta_cmd = 0.20
"""

import sys
import io
import time
import json
import hashlib
import random
import os
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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
TRAIN_MANIFEST = V12_DIR / "manifests" / "train_manifest.csv"
VAL_MANIFEST = V12_DIR / "manifests" / "validation_v1.csv"
CHECKPOINTS_DIR = V12_DIR / "checkpoints"
REPORTS_DIR = V12_DIR / "reports"

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
# ARCHITECTURES
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
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

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


class ScaledVietnameseKWSNet_E2(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(192)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(192, 64)
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
# DATASET & AUGMENTATION
# -----------------------------------------------------------------------------
class KWSDatasetV12(Dataset):
    def __init__(self, manifest_csv: Path, is_training: bool = False, target_length: int = 16000):
        self.is_training = is_training
        self.target_length = target_length
        self.df = pd.read_csv(manifest_csv)

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
            shift = random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode="constant")[:self.target_length]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode="constant")[-shift:self.target_length - shift]

            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = audio * gain

            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

        return torch.from_numpy(audio.astype(np.float32)), label


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
# TRAINING WORKFLOW
# -----------------------------------------------------------------------------
def run_training(
    model_family: str,
    seed: int,
    train_dataset: KWSDatasetV12,
    val_dataset: KWSDatasetV12,
    device: torch.device,
    batch_size: int = 64,
    max_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 8,
) -> Dict[str, Any]:
    set_seed(seed)
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    if model_family == "baseline_138k":
        model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == 138469, f"Baseline parameter mismatch: {trainable_params}"
        ckpt_path = CHECKPOINTS_DIR / f"baseline_138k_seed{seed}.pt"
    elif model_family == "e2_scaled_304k":
        model = ScaledVietnameseKWSNet_E2(num_classes=5, dropout_p=0.30).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == 304277, f"E2 parameter mismatch: {trainable_params}"
        ckpt_path = CHECKPOINTS_DIR / f"e2_scaled_304k_seed{seed}.pt"
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.25], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for audio_batch, labels in train_loader:
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                specs = frontend(audio_batch)
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

        # Validation
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

        # 5-class macro F1
        f1_list = []
        for ci in range(5):
            tp = val_confusion[ci, ci]
            fp = val_confusion[:, ci].sum() - tp
            fn = val_confusion[ci, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            f1_list.append(f1)
        val_macro_f1 = float(np.mean(f1_list))

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
                "train_loss": train_loss,
                "train_acc": train_acc,
                "architecture": model.__class__.__name__,
                "trainable_parameters": trainable_params,
                "model_family": model_family,
                "seed": seed,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    training_time = time.perf_counter() - t0
    ckpt_sha = compute_sha256(ckpt_path)

    return {
        "model_family": model_family,
        "seed": seed,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "training_time_sec": round(training_time, 2),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
    }


# -----------------------------------------------------------------------------
# BENCHMARK LATENCY & MEMORY
# -----------------------------------------------------------------------------
def benchmark_model(model: nn.Module, frontend: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    frontend.eval()

    # Latency test on CPU
    cpu_device = torch.device("cpu")
    m_cpu = model.to(cpu_device)
    f_cpu = frontend.to(cpu_device)
    dummy_audio = torch.randn(1, 16000, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = m_cpu(f_cpu(dummy_audio))

    # CPU Latency
    iters = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m_cpu(f_cpu(dummy_audio))
    cpu_latency_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Return model to primary device
    m_dev = model.to(device)
    f_dev = frontend.to(device)
    dummy_dev = dummy_audio.to(device)

    # Warmup on device
    with torch.no_grad():
        for _ in range(50):
            _ = m_dev(f_dev(dummy_dev))

    if device.type == "cuda":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = m_dev(f_dev(dummy_dev))
        torch.cuda.synchronize()
        dev_latency_ms = ((time.perf_counter() - t0) / iters) * 1000.0
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        dev_latency_ms = cpu_latency_ms
        vram_mb = 0.0

    return {
        "cpu_latency_ms": round(cpu_latency_ms, 3),
        "device_latency_ms": round(dev_latency_ms, 3),
        "vram_mb": round(vram_mb, 2),
    }


# -----------------------------------------------------------------------------
# DETAILED VALIDATION EVALUATION
# -----------------------------------------------------------------------------
def evaluate_checkpoint(
    ckpt_path: Path,
    model_class: Any,
    df_val: pd.DataFrame,
    waveforms_tensor: torch.Tensor,
    y_true: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_class(num_classes=5).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frontend = LogMelFrontend().to(device)
    frontend.eval()

    batch_size = 64
    all_logits = []
    with torch.no_grad():
        for b_start in range(0, len(waveforms_tensor), batch_size):
            b_wav = waveforms_tensor[b_start : b_start + batch_size]
            specs = frontend(b_wav)
            logits = model(specs)
            all_logits.append(logits.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits_cat, dim=-1).numpy()

    # Rule 1: theta_cmd = 0.20
    target_probs = probs[:, :4]
    max_target_p = np.max(target_probs, axis=-1)
    max_target_idx = np.argmax(target_probs, axis=-1)
    pred_theta_020 = np.where(max_target_p >= 0.20, max_target_idx, 4)

    # Rule 2: Standard 5-class Argmax
    pred_argmax = np.argmax(probs, axis=-1)

    def calc_metrics(y_pred: np.ndarray) -> Dict[str, Any]:
        n_correct = int(np.sum(y_true == y_pred))
        acc = float(n_correct / len(y_true))
        p, r, f1, s = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

        macro_f1_4cmd = float(np.mean(f1[:4]))
        macro_f1_5class = float(np.mean(f1))
        min_cmd_recall = float(np.min(r[:4]))

        reject_mask = (y_true == 4)
        oov_fa_count = int(np.sum(y_pred[reject_mask] != 4))
        oov_fa_rate = float(oov_fa_count / np.sum(reject_mask))
        reject_precision = float(p[4])
        reject_recall = float(r[4])

        return {
            "total_samples": len(y_true),
            "correct_samples": n_correct,
            "accuracy": round(acc, 4),
            "macro_f1_4cmd": round(macro_f1_4cmd, 4),
            "macro_f1_5class": round(macro_f1_5class, 4),
            "min_command_recall": round(min_cmd_recall, 4),
            "oov_fa_count": oov_fa_count,
            "oov_fa_rate": round(oov_fa_rate, 4),
            "reject_precision": round(reject_precision, 4),
            "reject_recall": round(reject_recall, 4),
            "confusion_matrix": cm,
            "per_class": {
                CLASSES[i]: {
                    "precision": round(float(p[i]), 4),
                    "recall": round(float(r[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(s[i]),
                }
                for i in range(5)
            },
        }

    benchmarks = benchmark_model(model, frontend, device)
    file_size_bytes = os.path.getsize(ckpt_path)

    return {
        "checkpoint_epoch": ckpt["epoch"],
        "train_loss": round(float(ckpt.get("train_loss", 0.0)), 4),
        "train_acc": round(float(ckpt.get("train_acc", 0.0)), 4),
        "val_loss": round(float(ckpt.get("val_loss", 0.0)), 4),
        "val_acc_during_train": round(float(ckpt.get("val_acc", 0.0)), 4),
        "checkpoint_size_kb": round(file_size_bytes / 1024.0, 1),
        "benchmarks": benchmarks,
        "metrics_theta_020": calc_metrics(pred_theta_020),
        "metrics_argmax": calc_metrics(pred_argmax),
    }


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PHASE E2 CONTROLLED CAPACITY EXPERIMENT (138k Baseline vs 304k E2)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Loading Training and Validation Audio into Memory...")
    train_dataset = KWSDatasetV12(TRAIN_MANIFEST, is_training=True)
    val_dataset = KWSDatasetV12(VAL_MANIFEST, is_training=False)
    print(f"Loaded Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    df_val = pd.read_csv(VAL_MANIFEST)
    y_true = np.array([CLASS_TO_IDX[row["label"]] for _, row in df_val.iterrows()])

    # Preload validation waveforms tensor for high-speed evaluation
    val_waveforms = [val_dataset.cached_audios[i] for i in range(len(val_dataset))]
    val_waveforms_tensor = torch.from_numpy(np.array(val_waveforms, dtype=np.float32)).to(device)

    runs_to_execute = [
        ("baseline_138k", 42, CorrectedVietnameseKWSNet),
        ("baseline_138k", 43, CorrectedVietnameseKWSNet),
        ("baseline_138k", 44, CorrectedVietnameseKWSNet),
        ("e2_scaled_304k", 42, ScaledVietnameseKWSNet_E2),
        ("e2_scaled_304k", 43, ScaledVietnameseKWSNet_E2),
        ("e2_scaled_304k", 44, ScaledVietnameseKWSNet_E2),
    ]

    ledger_rows = []
    detailed_results = {}

    print("\n[Step 2] Executing Controlled Training Runs...")
    for model_family, seed, model_cls in runs_to_execute:
        run_id = f"{model_family}_seed{seed}"
        print(f"\n---> Starting Run: {run_id} ({model_family}, seed={seed})")

        train_res = run_training(
            model_family=model_family,
            seed=seed,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            batch_size=64,
            max_epochs=40,
            lr=1e-3,
            weight_decay=0.01,
            patience=8,
        )

        print(f"  Training finished in {train_res['training_time_sec']}s (Best Epoch: {train_res['best_epoch']})")
        ckpt_path = Path(train_res["checkpoint_path"])

        print(f"  Evaluating Checkpoint on validation_v1.csv (theta_cmd = 0.20)...")
        eval_res = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            model_class=model_cls,
            df_val=df_val,
            waveforms_tensor=val_waveforms_tensor,
            y_true=y_true,
            device=device,
        )

        m_020 = eval_res["metrics_theta_020"]
        m_arg = eval_res["metrics_argmax"]
        bench = eval_res["benchmarks"]

        print(f"  [theta_cmd = 0.20] Acc: {m_020['accuracy']*100:.2f}% | 4-Cmd F1: {m_020['macro_f1_4cmd']*100:.2f}% | Min-R: {m_020['min_command_recall']*100:.2f}% | OOV FAR: {m_020['oov_fa_rate']*100:.2f}% | Rej Prec: {m_020['reject_precision']*100:.2f}%")
        print(f"  [Standard Argmax] Acc: {m_arg['accuracy']*100:.2f}% | 4-Cmd F1: {m_arg['macro_f1_4cmd']*100:.2f}% | Min-R: {m_arg['min_command_recall']*100:.2f}% | OOV FAR: {m_arg['oov_fa_rate']*100:.2f}% | Rej Prec: {m_arg['reject_precision']*100:.2f}%")
        print(f"  [Latency & Memory] CPU: {bench['cpu_latency_ms']:.2f} ms | GPU: {bench['device_latency_ms']:.2f} ms | VRAM: {bench['vram_mb']:.1f} MB | Size: {eval_res['checkpoint_size_kb']:.1f} KB")

        row = {
            "run_id": run_id,
            "model_family": model_family,
            "seed": seed,
            "trainable_parameters": train_res["trainable_params"],
            "best_epoch": train_res["best_epoch"],
            "epochs_trained": train_res["epochs_trained"],
            "training_time_sec": train_res["training_time_sec"],
            "train_loss": eval_res["train_loss"],
            "train_acc": eval_res["train_acc"],
            "val_loss": eval_res["val_loss"],
            "theta_020_accuracy": m_020["accuracy"],
            "theta_020_4cmd_macro_f1": m_020["macro_f1_4cmd"],
            "theta_020_5class_macro_f1": m_020["macro_f1_5class"],
            "theta_020_min_command_recall": m_020["min_command_recall"],
            "theta_020_oov_fa_count": m_020["oov_fa_count"],
            "theta_020_oov_fa_rate": m_020["oov_fa_rate"],
            "theta_020_reject_precision": m_020["reject_precision"],
            "argmax_accuracy": m_arg["accuracy"],
            "argmax_4cmd_macro_f1": m_arg["macro_f1_4cmd"],
            "argmax_5class_macro_f1": m_arg["macro_f1_5class"],
            "argmax_min_command_recall": m_arg["min_command_recall"],
            "argmax_oov_fa_rate": m_arg["oov_fa_rate"],
            "argmax_reject_precision": m_arg["reject_precision"],
            "cpu_latency_ms": bench["cpu_latency_ms"],
            "gpu_latency_ms": bench["device_latency_ms"],
            "checkpoint_size_kb": eval_res["checkpoint_size_kb"],
            "checkpoint_sha256": train_res["checkpoint_sha256"],
        }
        ledger_rows.append(row)
        detailed_results[run_id] = {
            "training_meta": train_res,
            "evaluation": eval_res,
        }

    # Save PHASE_E2_RUN_LEDGER.csv
    df_ledger = pd.DataFrame(ledger_rows)
    root_ledger_path = PROJECT_ROOT / "PHASE_E2_RUN_LEDGER.csv"
    rep_ledger_path = REPORTS_DIR / "PHASE_E2_RUN_LEDGER.csv"
    df_ledger.to_csv(root_ledger_path, index=False)
    df_ledger.to_csv(rep_ledger_path, index=False)
    print(f"\n[Saved] Run ledger saved to: {root_ledger_path}")

    # Save detailed JSON
    results_json_path = REPORTS_DIR / "phase_e2_evaluation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"[Saved] Detailed results saved to: {results_json_path}")


if __name__ == "__main__":
    main()
