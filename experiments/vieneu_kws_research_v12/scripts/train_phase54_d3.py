#!/usr/bin/env python3
"""
Phase 54: D3 Balanced-Sensitivity Candidate Training Runner
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/train_phase54_d3.py

Executes the pre-registered D3 training comparison matrix:
- D3_A: Symmetric weights [1.0, 1.0, 1.0, 1.0, 1.0], 32 targets : 16 OOV : 16 acoustic
- D3_B: Cmd-boost weights [1.10, 1.10, 1.10, 1.10, 1.0], 36 targets : 14 OOV : 14 acoustic
- D3_C: Balanced weights [1.05, 1.05, 1.05, 1.05, 1.0], 34 targets : 15 OOV : 15 acoustic, min-recall penalty
Seed: 42
Architecture Scale: Fixed 138,469 parameters (CorrectedVietnameseKWSNet)
Evaluation Threshold: theta_cmd = 0.20
"""

import sys
import os
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
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
OUTPUT_BASE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase54_d3"

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
# ARCHITECTURE (138,469 PARAMS)
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


# -----------------------------------------------------------------------------
# DATASET & SAMPLER
# -----------------------------------------------------------------------------
class D3KWSDataset(Dataset):
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


class D3BatchSampler:
    def __init__(self, dataset: D3KWSDataset, n_target: int = 32, n_oov: int = 16, n_acoustic: int = 16, num_batches: int = 85, seed: int = 42):
        self.dataset = dataset
        self.n_target = n_target
        self.n_oov = n_oov
        self.n_acoustic = n_acoustic
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)

        df = dataset.df
        self.target_indices = df[df["label"].isin(["TRAI", "PHAI", "LEN", "XUONG"])].index.to_numpy()
        self.reject_oov = df[(df["label"] == "REJECT") & (
            (df["reject_subgroup"] == "oov_word") | 
            (df.get("source_type", "").isin(["edge_tts_oov_clean", "edge_tts_oov_noise_augmented"]))
        )].index.to_numpy()
        self.reject_acoustic = df[(df["label"] == "REJECT") & (~df.index.isin(self.reject_oov))].index.to_numpy()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            t_batch = self.rng.choice(self.target_indices, size=self.n_target, replace=True)
            oov_batch = self.rng.choice(self.reject_oov, size=self.n_oov, replace=True)
            ac_batch = self.rng.choice(self.reject_acoustic, size=self.n_acoustic, replace=True)
            batch = np.concatenate([t_batch, oov_batch, ac_batch])
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


def evaluate_model(model: nn.Module, frontend: nn.Module, data_loader: DataLoader, ce_weights: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_targets = []
    all_preds = []
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weights)

    with torch.no_grad():
        for waveforms, targets in data_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            specs = frontend(waveforms)
            logits = model(specs)
            loss = ce_loss_fn(logits, targets)
            total_loss += loss.item() * len(targets)

            probs = F.softmax(logits, dim=-1)
            target_probs = probs[:, :4]
            max_p, max_idx = torch.max(target_probs, dim=-1)

            preds = torch.where(max_p >= 0.20, max_idx, torch.tensor(4, device=device))

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    val_loss = total_loss / len(y_true)
    val_acc = float(np.mean(y_true == y_pred))

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    val_macro_f1 = float(np.mean(f1[:4]))
    overall_macro_f1 = float(np.mean(f1))
    min_cmd_recall = float(np.min(r[:4]))

    # OOV FAR on validation (REJECT is label 4, support = 640)
    reject_mask = (y_true == 4)
    oov_fa_count = int(np.sum(y_pred[reject_mask] != 4))
    oov_fa_rate = float(oov_fa_count / np.sum(reject_mask))
    oov_precision = float(p[4])

    metrics = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "overall_macro_f1": overall_macro_f1,
        "min_cmd_recall": min_cmd_recall,
        "oov_fa_rate": oov_fa_rate,
        "oov_rejection_precision": oov_precision,
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


def train_d3_condition(condition: str, seed: int = 42) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"STARTING PHASE 54 D3 TRAINING: Condition = {condition}, Seed = {seed}")
    print("=" * 80)

    # Condition configuration
    if condition == "D3_A":
        ce_weights_list = [1.0, 1.0, 1.0, 1.0, 1.0]
        n_target, n_oov, n_ac = 32, 16, 16
        use_min_recall_obj = False
    elif condition == "D3_B":
        ce_weights_list = [1.10, 1.10, 1.10, 1.10, 1.0]
        n_target, n_oov, n_ac = 36, 14, 14
        use_min_recall_obj = False
    elif condition == "D3_C":
        ce_weights_list = [1.05, 1.05, 1.05, 1.05, 1.0]
        n_target, n_oov, n_ac = 34, 15, 15
        use_min_recall_obj = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    train_manifest = MANIFESTS_DIR / "D2_matched_train_v1.csv"
    val_manifest = MANIFESTS_DIR / "validation_v1.csv"

    out_dir = OUTPUT_BASE_DIR / f"{condition}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = D3KWSDataset(train_manifest, is_training=True)
    val_dataset = D3KWSDataset(val_manifest, is_training=False)

    sampler = D3BatchSampler(
        dataset=train_dataset,
        n_target=n_target,
        n_oov=n_oov,
        n_acoustic=n_ac,
        num_batches=85,
        seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    frontend = LogMelFrontend().to(device)
    frontend.eval()

    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 138469, f"Parameter mismatch: {total_params} != 138469"

    ce_weights = torch.tensor(ce_weights_list, device=device)
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    max_epochs = 35
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_score = -1.0
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

            with torch.no_grad():
                specs = frontend(waveforms)
            specs = apply_spec_augment(specs, freq_mask_max=4, time_mask_max=10)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(labels)
            train_samples += len(labels)

        scheduler.step()
        train_loss = train_loss_total / train_samples

        # Validation evaluation at locked theta_cmd = 0.20
        val_res = evaluate_model(model, frontend, val_loader, ce_weights, device)

        if use_min_recall_obj:
            val_selection_score = val_res["val_macro_f1"] * (val_res["min_cmd_recall"] ** 0.25)
        else:
            val_selection_score = val_res["val_macro_f1"]

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_res["val_loss"],
            "val_accuracy": val_res["val_accuracy"],
            "val_macro_f1": val_res["val_macro_f1"],
            "min_cmd_recall": val_res["min_cmd_recall"],
            "selection_score": val_selection_score,
            "oov_fa_rate": val_res["oov_fa_rate"]
        })

        improved = (val_selection_score > best_score + 1e-4)
        if improved:
            best_score = val_selection_score
            best_epoch = epoch
            best_metrics = val_res
            patience_counter = 0

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "condition": condition,
                "seed": seed,
                "val_macro_f1": val_res["val_macro_f1"],
                "total_parameters": total_params
            }, out_dir / "best_checkpoint.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Macro-F1: {val_res['val_macro_f1']*100:.2f}% | Min-R: {val_res['min_cmd_recall']*100:.1f}% | Score: {val_selection_score:.4f} | Pat: {patience_counter}/{patience}{' *BEST*' if improved else ''}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    duration_s = time.perf_counter() - t0_train
    ckpt_path = out_dir / "best_checkpoint.pt"
    ckpt_sha = compute_sha256(ckpt_path)

    # Save training log and summary
    pd.DataFrame(log_records).to_csv(out_dir / "training_log.csv", index=False)

    summary = {
        "condition": condition,
        "seed": seed,
        "ce_weights": ce_weights_list,
        "batch_sampler": {"n_target": n_target, "n_oov": n_oov, "n_acoustic": n_ac},
        "use_min_recall_obj": use_min_recall_obj,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_metrics": best_metrics,
        "total_epochs": epoch,
        "duration_seconds": duration_s,
        "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": ckpt_sha,
        "train_manifest_sha256": compute_sha256(train_manifest),
        "val_manifest_sha256": compute_sha256(val_manifest)
    }

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run {condition} completed in {duration_s:.1f}s. Checkpoint SHA-256: {ckpt_sha}")
    return summary


def main():
    print("=" * 80)
    print("PHASE 54: EXECUTING D3 CONTROLLED TRAINING MATRIX (SEEDS: 42)")
    print("=" * 80)

    conditions = ["D3_A", "D3_B", "D3_C"]
    run_summaries = []

    for cond in conditions:
        res = train_d3_condition(cond, seed=42)
        run_summaries.append(res)

    # Compile run ledger
    ledger_records = []
    print("\n" + "=" * 80)
    print("PHASE 54 D3 RUN LEDGER & VALIDATION GATE AUDIT")
    print("=" * 80)

    for r in run_summaries:
        m = r["best_metrics"]
        macro_f1 = m["val_macro_f1"]
        min_rec = m["min_cmd_recall"]
        oov_far = m["oov_fa_rate"]
        oov_prec = m["oov_rejection_precision"]

        # Audit against pre-registered gate:
        # 1. Macro-F1 >= 92.0%
        # 2. Min Recall >= 90.0%
        # 3. OOV FAR <= 10.0%
        # 4. Rejection Precision >= 95.0%
        p_f1 = macro_f1 >= 0.92
        p_rec = min_rec >= 0.90
        p_far = oov_far <= 0.10
        p_prec = oov_prec >= 0.95
        all_passed = bool(p_f1 and p_rec and p_far and p_prec)

        # Composite selection score
        sel_score = macro_f1 * min_rec

        rec = {
            "condition": r["condition"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "val_macro_f1": macro_f1,
            "min_cmd_recall": min_rec,
            "oov_fa_rate": oov_far,
            "oov_rejection_precision": oov_prec,
            "trai_recall": m["per_class"]["TRAI"]["recall"],
            "phai_recall": m["per_class"]["PHAI"]["recall"],
            "len_recall": m["per_class"]["LEN"]["recall"],
            "xuong_recall": m["per_class"]["XUONG"]["recall"],
            "composite_selection_score": sel_score,
            "gate_f1_pass": p_f1,
            "gate_min_recall_pass": p_rec,
            "gate_oov_far_pass": p_far,
            "gate_oov_prec_pass": p_prec,
            "all_criteria_passed": all_passed,
            "checkpoint_sha256": r["checkpoint_sha256"]
        }
        ledger_records.append(rec)

        print(f"\nCondition: {r['condition']}")
        print(f"  Best Epoch:           {r['best_epoch']}")
        print(f"  Val 4-Cmd Macro-F1:   {macro_f1*100:.2f}% (>= 92.0%: {p_f1})")
        print(f"  Min Command Recall:   {min_rec*100:.2f}% (>= 90.0%: {p_rec})")
        print(f"  OOV False Alarm Rate: {oov_far*100:.2f}% (<= 10.0%: {p_far})")
        print(f"  Rejection Precision:  {oov_prec*100:.2f}% (>= 95.0%: {p_prec})")
        print(f"  Per-Class Recalls:    TRAI: {m['per_class']['TRAI']['recall']*100:.1f}%, PHAI: {m['per_class']['PHAI']['recall']*100:.1f}%, LEN: {m['per_class']['LEN']['recall']*100:.1f}%, XUONG: {m['per_class']['XUONG']['recall']*100:.1f}%")
        print(f"  Selection Score:      {sel_score:.4f}")
        print(f"  All Gate Criteria:    {'PASS' if all_passed else 'FAIL'}")

    df_ledger = pd.DataFrame(ledger_records)
    ledger_csv_path = REPORTS_DIR / "PHASE54_D3_RUN_LEDGER.csv"
    df_ledger.to_csv(ledger_csv_path, index=False)
    print(f"\nSaved D3 Run Ledger to: {ledger_csv_path}")


if __name__ == "__main__":
    main()
