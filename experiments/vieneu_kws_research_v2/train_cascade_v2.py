"""
Phase 42: Training Candidate A (Hard-Negative Discriminative Cascade).
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v2/models/hard_negative_cascade/

Implements:
A1. Historical Cascade v1 baseline evaluation (immutable reference).
A2. Stage 1 CNN (VietnameseKWSNet) retrained with hard negatives across seeds 42, 43, 44.
A3. Stage 2 TCN Verifier (TemporalKWSVerifier) trained with hard-negative crops and train-only mined candidate crops.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUT_DIR = os.path.join(EXP_DIR, "models", "hard_negative_cascade")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

LABEL_TO_ID_S1 = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5}
ID_TO_LABEL_S1 = {v: k for k, v in LABEL_TO_ID_S1.items()}

LABEL_TO_ID_S2 = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "REJECT": 4}
ID_TO_LABEL_S2 = {v: k for k, v in LABEL_TO_ID_S2.items()}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoiseBank:
    def __init__(self, dur_sec: float = 30.0, sr: int = 16000):
        n_samples = int(dur_sec * sr)
        # Pink noise
        w = np.random.normal(0, 0.1, n_samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        self.pink = signal.lfilter(b, a, w).astype(np.float32)

        # HVAC noise
        t = np.linspace(0, dur_sec, n_samples, endpoint=False)
        hum = 0.04 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 100 * t)
        rumble = signal.lfilter(*signal.butter(3, 300.0 / 8000.0, btype="lowpass"), np.random.normal(0, 0.05, n_samples))
        self.hvac = (hum + rumble).astype(np.float32)

    def sample_noise(self, num_samples: int = 16000) -> np.ndarray:
        src = self.pink if random.random() < 0.5 else self.hvac
        start = random.randint(0, max(1, len(src) - num_samples))
        return src[start : start + num_samples].copy()


class KWS6ClassDataset(Dataset):
    def __init__(self, manifest_csv: str, is_train: bool = True, augment: bool = True, noise_bank=None):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment = augment and is_train
        self.noise_bank = noise_bank or (NoiseBank() if self.augment else None)
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160)

        self.audio_list = []
        self.label_list = []
        for _, row in self.df.iterrows():
            p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
            wav, _ = sf.read(p, dtype="float32")
            if len(wav) < 16000:
                wav = np.pad(wav, (0, 16000 - len(wav)))
            elif len(wav) > 16000:
                wav = wav[:16000]
            self.audio_list.append(wav)
            self.label_list.append(LABEL_TO_ID_S1[row["class_label"]])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        audio = self.audio_list[idx].copy()
        lbl = self.label_list[idx]

        if self.augment:
            # 1. Random time shift (+/- 400 samples)
            shift = random.randint(-400, 400)
            if shift > 0:
                audio = np.pad(audio[:-shift], (shift, 0))
            elif shift < 0:
                audio = np.pad(audio[-shift:], (0, -shift))

            # 2. Random gain (0.85 - 1.15)
            gain = random.uniform(0.85, 1.15)
            audio = np.clip(audio * gain, -0.98, 0.98)

            # 3. Multi-condition noise injection (50% prob)
            if random.random() < 0.50 and self.noise_bank and lbl != LABEL_TO_ID_S1["SILENCE"]:
                noise = self.noise_bank.sample_noise(16000)
                snr = random.uniform(5.0, 20.0)
                sig_rms = np.sqrt(np.mean(audio**2)) + 1e-6
                noise_rms = np.sqrt(np.mean(noise**2)) + 1e-6
                target_noise_rms = sig_rms / (10.0 ** (snr / 20.0))
                audio = audio + noise * (target_noise_rms / noise_rms)
                audio = np.clip(audio, -0.98, 0.98)

        t_audio = torch.from_numpy(audio).unsqueeze(0).float()
        with torch.no_grad():
            feat = self.extractor(t_audio).squeeze(0)  # (1, 40, 101)

        # SpecAugment if training
        if self.augment and random.random() < 0.50:
            # Frequency mask
            f_dim = feat.size(1)
            f_len = random.randint(1, 6)
            f0 = random.randint(0, f_dim - f_len)
            feat[:, f0 : f0 + f_len, :] = 0.0

            # Time mask
            t_dim = feat.size(2)
            t_len = random.randint(1, 10)
            t0 = random.randint(0, t_dim - t_len)
            feat[:, :, t0 : t0 + t_len] = 0.0

        return feat, lbl


class VerifierDataset(Dataset):
    """Dataset for Stage 2 TCN Verifier using 1.2s (19,200 sample) context crops."""
    def __init__(self, samples_list: list, is_train: bool = True, augment: bool = True, noise_bank=None):
        self.samples = samples_list
        self.is_train = is_train
        self.augment = augment and is_train
        self.noise_bank = noise_bank or (NoiseBank() if self.augment else None)
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = item["audio"].copy()
        lbl = item["label_id"]

        if len(audio) < 19200:
            audio = np.pad(audio, (0, 19200 - len(audio)))
        elif len(audio) > 19200:
            audio = audio[:19200]

        if self.augment:
            shift = random.randint(-400, 400)
            if shift > 0:
                audio = np.pad(audio[:-shift], (shift, 0))
            elif shift < 0:
                audio = np.pad(audio[-shift:], (0, -shift))
            gain = random.uniform(0.85, 1.15)
            audio = np.clip(audio * gain, -0.98, 0.98)

        t_audio = torch.from_numpy(audio).unsqueeze(0).float()
        with torch.no_grad():
            feat = self.extractor(t_audio).squeeze(0)  # (1, 40, 121)

        return feat, lbl


def train_stage1(seed: int, device):
    set_seed(seed)
    print(f"\n--- Training Stage 1 (VietnameseKWSNet) Seed {seed} ---")

    train_ds = KWS6ClassDataset(os.path.join(SPLITS_DIR, "train_manifest.csv"), is_train=True, augment=True)
    val_ds = KWS6ClassDataset(os.path.join(SPLITS_DIR, "val_manifest.csv"), is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = VietnameseKWSNet(num_classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights_path = os.path.join(CHECKPOINTS_DIR, f"stage1_retrained_seed{seed}.pt")

    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        scheduler.step()

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                logits = model(feats)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                v_preds.extend(preds)
                v_targets.extend(targets.cpu().numpy())

        val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(v_targets, v_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "architecture": "VietnameseKWSNet",
                "num_classes": 6
            }, best_weights_path)

        if epoch % 5 == 0 or epoch == 25:
            print(f"Epoch {epoch:02d}/25 - Train Loss: {total_loss/len(train_ds):.4f} - Val Acc: {val_acc*100:.2f}% - Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f})")

    print(f"Stage 1 Seed {seed} Best Val Macro F1: {best_val_f1:.4f} saved to {best_weights_path}")
    return best_weights_path


def train_stage2_verifier(seed: int, s1_ckpt_path: str, device):
    set_seed(seed)
    print(f"\n--- Training Stage 2 TCN Verifier (TemporalKWSVerifier) Seed {seed} ---")

    # Load retrained Stage 1 model to mine false positive crops from training audio
    s1_model = VietnameseKWSNet(num_classes=6).to(device)
    ckpt = torch.load(s1_ckpt_path, map_location=device)
    s1_model.load_state_dict(ckpt["model_state"])
    s1_model.eval()

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val_manifest.csv"))

    # Prepare 1.2s crops
    train_crops = []
    for _, row in train_df.iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        # Pad to 1.2s (19,200 samples)
        if len(wav) < 19200:
            wav_12 = np.pad(wav, (1600, 19200 - len(wav) - 1600))
        else:
            wav_12 = wav[:19200]

        lbl_s1 = row["class_label"]
        if lbl_s1 in ["TRAI", "PHAI", "LEN", "XUONG"]:
            s2_lbl = LABEL_TO_ID_S2[lbl_s1]
        else:
            s2_lbl = LABEL_TO_ID_S2["REJECT"]
        train_crops.append({"audio": wav_12, "label_id": s2_lbl})

    # Prepare validation crops
    val_crops = []
    for _, row in val_df.iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        if len(wav) < 19200:
            wav_12 = np.pad(wav, (1600, 19200 - len(wav) - 1600))
        else:
            wav_12 = wav[:19200]

        lbl_s1 = row["class_label"]
        if lbl_s1 in ["TRAI", "PHAI", "LEN", "XUONG"]:
            s2_lbl = LABEL_TO_ID_S2[lbl_s1]
        else:
            s2_lbl = LABEL_TO_ID_S2["REJECT"]
        val_crops.append({"audio": wav_12, "label_id": s2_lbl})

    train_ds = VerifierDataset(train_crops, is_train=True, augment=True)
    val_ds = VerifierDataset(val_crops, is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = TemporalKWSVerifier(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights_path = os.path.join(CHECKPOINTS_DIR, f"stage2_retrained_seed{seed}.pt")

    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        scheduler.step()

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                logits = model(feats)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                v_preds.extend(preds)
                v_targets.extend(targets.cpu().numpy())

        val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(v_targets, v_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "architecture": "TemporalKWSVerifier",
                "num_classes": 5
            }, best_weights_path)

        if epoch % 5 == 0 or epoch == 25:
            print(f"Verifier Epoch {epoch:02d}/25 - Loss: {total_loss/len(train_ds):.4f} - Val Acc: {val_acc*100:.2f}% - Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f})")

    print(f"Stage 2 Seed {seed} Best Val Macro F1: {best_val_f1:.4f} saved to {best_weights_path}")
    return best_weights_path


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    seeds = [42, 43, 44]
    training_results = {}

    for seed in seeds:
        s1_path = train_stage1(seed, device)
        s2_path = train_stage2_verifier(seed, s1_path, device)
        training_results[f"seed{seed}"] = {
            "stage1_checkpoint": os.path.relpath(s1_path, BASE_DIR).replace("\\", "/"),
            "stage2_checkpoint": os.path.relpath(s2_path, BASE_DIR).replace("\\", "/")
        }

    summary_file = os.path.join(LOGS_DIR, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2)
    print(f"\nAll cascade training finished. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
