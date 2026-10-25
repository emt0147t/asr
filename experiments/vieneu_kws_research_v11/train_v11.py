#!/usr/bin/env python3
"""
Stage C: Training Script for Vietnamese KWS v11 Candidates.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v11/train_v11.py
"""

import os
import sys
import json
import csv
import random
import time
import hashlib
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V11_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11"
DATA_DIR = V11_DIR / "data" / "splits"
CKPT_DIR = V11_DIR / "checkpoints"
CONFIGS_DIR = V11_DIR / "configs"

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]


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
            f_max=8000.0
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


class BoundedDatasetV11(Dataset):
    def __init__(self, manifest_path, is_training=False, target_length=16000,
                 w_cmd=1.0, w_prefix_hn=1.0, w_hn=1.0, w_sil=1.0):
        self.manifest_path = Path(manifest_path)
        self.is_training = is_training
        self.target_length = target_length
        self.w_cmd = w_cmd
        self.w_prefix_hn = w_prefix_hn
        self.w_hn = w_hn
        self.w_sil = w_sil
        self.rows = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

        # Preload audio into memory to eliminate disk I/O bottlenecks during multi-epoch training
        self.cached_audios = []
        for r in self.rows:
            wav_path = PROJECT_ROOT / r["kws_wav_path"]
            audio, sr = sf.read(wav_path, dtype="float32")
            if len(audio) < self.target_length:
                pad = self.target_length - len(audio)
                audio = np.pad(audio, (0, pad), mode="constant")
            elif len(audio) > self.target_length:
                audio = audio[:self.target_length]
            self.cached_audios.append(audio.astype(np.float32))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio = self.cached_audios[idx].copy()

        class_id = int(row["bounded_class_id"])
        fam = row.get("confusion_family", "")

        # Sample weight assignment
        if class_id < 4:
            sample_weight = self.w_cmd
        elif fam in ["target_prefix_compound", "target_suffix_compound"]:
            sample_weight = self.w_prefix_hn
        elif fam in ["minimal_pair_tone", "minimal_pair_consonant", "minimal_pair_coda", "phonetic_phrase_context"]:
            sample_weight = self.w_hn
        else:
            sample_weight = self.w_sil

        if self.is_training:
            # 1. Random time shift (-80ms to +80ms)
            shift = random.randint(-1280, 1280)
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode="constant")[:self.target_length]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode="constant")[-shift:self.target_length - shift]

            # 2. Gain variation (-3dB to +3dB)
            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = audio * gain

            # 3. Additive noise (SNR 12 to 28 dB)
            if random.random() < 0.5:
                noise = np.random.randn(*audio.shape).astype(np.float32)
                sig_rms = np.sqrt(np.mean(audio**2) + 1e-9)
                snr_db = random.uniform(12.0, 28.0)
                noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
                audio = audio + noise * noise_rms

        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        return torch.from_numpy(audio).float(), class_id, float(sample_weight)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_on_validation(model, frontend, val_loader, device, theta_cmd=0.40):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for audios, labels, _ in val_loader:
            audios = audios.to(device)
            specs = frontend(audios)
            logits = model(specs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            for p, l in zip(probs, labels.numpy()):
                all_probs.append(p)
                all_labels.append(l)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Apply decision rule with theta_cmd
    pred_classes = []
    for p in all_probs:
        cmd_probs = p[:4]
        argmax_cmd = int(np.argmax(cmd_probs))
        max_cmd_prob = cmd_probs[argmax_cmd]
        if max_cmd_prob >= theta_cmd:
            pred_classes.append(argmax_cmd)
        else:
            pred_classes.append(4)  # REJECT
    pred_classes = np.array(pred_classes)

    # Metrics
    cmd_mask = (all_labels < 4)
    non_target_mask = (all_labels == 4)

    cmd_total = np.sum(cmd_mask)
    cmd_correct = np.sum(pred_classes[cmd_mask] == all_labels[cmd_mask])
    cmd_recall = float(cmd_correct / cmd_total) if cmd_total > 0 else 0.0

    per_kw_recall = {}
    for cid, cname in enumerate(TARGET_CLASSES):
        m = (all_labels == cid)
        per_kw_recall[cname] = float(np.sum(pred_classes[m] == cid) / np.sum(m)) if np.sum(m) > 0 else 0.0

    nt_total = np.sum(non_target_mask)
    nt_fa = np.sum(pred_classes[non_target_mask] < 4)
    nt_far = float(nt_fa / nt_total) if nt_total > 0 else 0.0

    overall_acc = float(np.sum(pred_classes == all_labels) / len(all_labels))

    score = cmd_recall - 2.0 * nt_far
    if any(r < 0.85 for r in per_kw_recall.values()):
        score -= 0.50

    return {
        "score": score,
        "cmd_recall": cmd_recall,
        "per_keyword_recall": per_kw_recall,
        "non_target_far": nt_far,
        "accuracy": overall_acc,
        "probs": all_probs,
        "labels": all_labels
    }


def train_single_run(variant_name, weights_dict, seed, epochs=25, batch_size=32, lr=1e-3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_csv = DATA_DIR / "train_manifest.csv"
    val_csv = DATA_DIR / "val_manifest.csv"

    train_ds = BoundedDatasetV11(train_csv, is_training=True, **weights_dict)
    val_ds = BoundedDatasetV11(val_csv, is_training=False, **weights_dict)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -999.0
    best_weights = None
    best_metrics = None
    best_epoch = 0

    print(f"\n>>> Training [{variant_name}] Seed {seed} on {device} ({epochs} epochs) <<<")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for audios, labels, weights in train_loader:
            audios, labels, weights = audios.to(device), labels.to(device), weights.to(device)
            specs = frontend(audios)

            # SpecAugment
            if random.random() < 0.5:
                f_mask = random.randint(1, 8)
                f_start = random.randint(0, 40 - f_mask)
                specs[:, :, f_start:f_start+f_mask, :] = 0.0
            if random.random() < 0.5:
                t_mask = random.randint(1, 15)
                t_start = random.randint(0, 101 - t_mask)
                specs[:, :, :, t_start:t_start+t_mask] = 0.0

            optimizer.zero_grad()
            logits = model(specs)

            loss_per_sample = F.cross_entropy(logits, labels, label_smoothing=0.05, reduction="none")
            loss = torch.mean(loss_per_sample * weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        scheduler.step()
        avg_loss = total_loss / total_samples

        # Validation evaluation at default theta=0.40
        val_res = evaluate_on_validation(model, frontend, val_loader, device, theta_cmd=0.40)
        score = val_res["score"]

        if score > best_score:
            best_score = score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_res
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f} | "
                  f"Val Recall: {val_res['cmd_recall']*100:.1f}%, FAR: {val_res['non_target_far']*100:.1f}%, "
                  f"Score: {score:.4f} (Best: {best_score:.4f} @ Ep {best_epoch})")

    # Save best checkpoint
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_filename = f"{variant_name}_seed{seed}.pt"
    ckpt_path = CKPT_DIR / ckpt_filename

    checkpoint_payload = {
        "candidate_id": f"{variant_name}_seed{seed}",
        "variant_name": variant_name,
        "seed": seed,
        "architecture": "CorrectedVietnameseKWSNet",
        "num_classes": 5,
        "classes": CLASSES,
        "best_epoch": best_epoch,
        "val_score": best_score,
        "val_metrics_at_0.40": {
            "cmd_recall": best_metrics["cmd_recall"],
            "per_keyword_recall": best_metrics["per_keyword_recall"],
            "non_target_far": best_metrics["non_target_far"],
            "accuracy": best_metrics["accuracy"]
        },
        "model_state_dict": best_weights
    }
    torch.save(checkpoint_payload, ckpt_path)
    ckpt_hash = compute_sha256(ckpt_path)
    print(f"[OK] Saved best checkpoint: {ckpt_filename} (SHA256: {ckpt_hash[:16]}...)")

    # Threshold calibration on validation set
    model.load_state_dict(best_weights)
    model.eval()

    all_probs = best_metrics["probs"]
    all_labels = best_metrics["labels"]
    cmd_mask = (all_labels < 4)
    nt_mask = (all_labels == 4)

    threshold_sweep = [round(t, 2) for t in np.arange(0.20, 0.75, 0.05)]
    calib_results = []
    best_calib_th = 0.40
    best_calib_recall = 0.0
    best_calib_far = 1.0

    for th in threshold_sweep:
        preds = []
        for p in all_probs:
            cprobs = p[:4]
            argmax_c = int(np.argmax(cprobs))
            if cprobs[argmax_c] >= th:
                preds.append(argmax_c)
            else:
                preds.append(4)
        preds = np.array(preds)

        rec = float(np.sum(preds[cmd_mask] == all_labels[cmd_mask]) / np.sum(cmd_mask))
        far = float(np.sum(preds[nt_mask] < 4) / np.sum(nt_mask))
        per_kw = {TARGET_CLASSES[i]: float(np.sum(preds[all_labels == i] == i) / np.sum(all_labels == i)) for i in range(4)}

        calib_results.append({
            "theta": th,
            "cmd_recall": rec,
            "non_target_far": far,
            "per_keyword_recall": per_kw
        })

        # Selection rule: max recall while far <= 0.05 and all kw >= 0.90
        if far <= 0.05 and all(v >= 0.90 for v in per_kw.values()):
            if rec > best_calib_recall or (rec == best_calib_recall and far < best_calib_far):
                best_calib_th = th
                best_calib_recall = rec
                best_calib_far = far

    # If no threshold achieved far <= 0.05, pick min FAR with recall >= 0.95
    if best_calib_recall == 0.0:
        candidates = [c for c in calib_results if c["cmd_recall"] >= 0.95]
        if candidates:
            best_c = min(candidates, key=lambda x: x["non_target_far"])
            best_calib_th = best_c["theta"]
            best_calib_recall = best_c["cmd_recall"]
            best_calib_far = best_c["non_target_far"]
        else:
            best_calib_th = 0.40

    print(f"Optimal validation-calibrated threshold: theta_cmd* = {best_calib_th} "
          f"(Val Recall: {best_calib_recall*100:.2f}%, Val FAR: {best_calib_far*100:.2f}%)")

    return {
        "candidate_id": f"{variant_name}_seed{seed}",
        "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": ckpt_hash,
        "best_epoch": best_epoch,
        "optimal_theta": best_calib_th,
        "val_metrics_at_optimal_theta": {
            "cmd_recall": best_calib_recall,
            "non_target_far": best_calib_far
        },
        "sweep_records": calib_results
    }


def main():
    print("=" * 80)
    print("STAGE C: TRAINING PREDECLARED V11 CANDIDATES")
    print("=" * 80)

    with open(CONFIGS_DIR / "candidate_configs.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    variants = config["candidate_variants"]
    seeds = config["seeds"]

    all_run_summaries = []

    for vname, vparams in variants.items():
        weights_dict = {
            "w_cmd": vparams["w_cmd"],
            "w_prefix_hn": vparams["w_prefix_hn"],
            "w_hn": vparams["w_hn"],
            "w_sil": vparams["w_sil"]
        }
        for s in seeds:
            res = train_single_run(vname, weights_dict, seed=s, epochs=25, batch_size=32, lr=1e-3)
            all_run_summaries.append(res)

    # Save calibration summary
    calib_summary_path = CONFIGS_DIR / "calibration_summary.json"
    with open(calib_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_run_summaries, f, indent=2)
    print(f"\n[OK] Saved validation calibration summary: {calib_summary_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
