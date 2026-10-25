"""
Phase 37: Train a Compact Temporal Verification Model (Stage 2)
Using only training-split source utterances, hard negatives, edge-aligned crops,
and mined Stage 1 false triggers on training audio.

Operates across three independent random seeds: 42, 43, 44.
Saves best validation checkpoints based on validation macro F1.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_robustness_campaign.confidence_policy import (
    apply_confidence_policy,
    TARGET_COMMAND_INDICES,
    UNKNOWN_INDEX,
    SILENCE_INDEX
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")
LOGS_DIR = os.path.join(EXP_DIR, "logs")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
VERIFIER_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
VERIFIER_LABEL_TO_ID = {c: i for i, c in enumerate(VERIFIER_CLASSES)}
VERIFIER_ID_TO_LABEL = {i: c for i, c in enumerate(VERIFIER_CLASSES)}

# -------------------------------------------------------------
# 1. Model Architecture: TemporalKWSVerifier (TCN)
# -------------------------------------------------------------

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class TemporalKWSVerifier(nn.Module):
    """
    Compact Temporal Convolutional Network (TCN) for Stage 2 Keyword Verification.
    Input: Log-Mel Spectrogram (Batch, 1, 40, T) where T=121 (1.2 seconds of audio)
    Output: Logits (Batch, 5) -> [TRAI, PHAI, LEN, XUONG, REJECT]
    """
    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        # Frequency reduction frontend: preserves frame time T, reduces 40 mel bins to 1
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 40 -> 20
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 20 -> 10
            nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))    # 10 -> 1
        )

        # Temporal TCN sequence backbone with exponentially increasing dilation
        self.tcn = nn.Sequential(
            TCNResidualBlock(64, 64, dilation=1, dropout=0.1),
            TCNResidualBlock(64, 64, dilation=2, dropout=0.1),
            TCNResidualBlock(64, 128, dilation=4, dropout=0.1),
            TCNResidualBlock(128, 128, dilation=8, dropout=0.1),
        )

        # Statistical temporal pooling + linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat2d = self.frontend(x)         # (B, 64, 1, T)
        feat1d = feat2d.squeeze(2)        # (B, 64, T)
        tcn_out = self.tcn(feat1d)        # (B, 128, T)

        # Statistical pooling over time (mean and std captures temporal dynamics)
        mean_pool = torch.mean(tcn_out, dim=-1)   # (B, 128)
        std_pool = torch.std(tcn_out, dim=-1)     # (B, 128)
        stat_pool = torch.cat([mean_pool, std_pool], dim=-1)  # (B, 256)

        logits = self.classifier(stat_pool)       # (B, 5)
        return logits


# -------------------------------------------------------------
# 2. Hard Negative Mining and Dataset Construction
# -------------------------------------------------------------

def extract_or_pad_crop(audio: np.ndarray, target_samples: int = 19200, offset_samples: int = 0) -> np.ndarray:
    """Embeds an audio clip into a target_samples (1.2s) array with specified offset."""
    out = np.zeros(target_samples, dtype=np.float32)
    n = len(audio)
    if n >= target_samples:
        start = max(0, min(offset_samples, n - target_samples))
        return audio[start : start + target_samples].copy()
    else:
        max_offset = target_samples - n
        start = max(0, min(offset_samples, max_offset))
        out[start : start + n] = audio
        return out


def mine_stage1_hard_negatives(stage1_model, extractor, device, df_train, sample_rate=16000, target_crop_samples=19200):
    """
    Runs Stage 1 on a continuous stream of training non-target utterances to mine false alarms.
    Zero validation or test audio is used.
    """
    stage1_model.eval()
    unk_rows = df_train[df_train["class_label"] == "UNKNOWN"].to_dict("records")
    sil_rows = df_train[df_train["class_label"] == "SILENCE"].to_dict("records")

    rng = random.Random(9999)
    np_rng = np.random.RandomState(9999)

    # Stitch a continuous non-target training stream
    blocks = [np.zeros(8000, dtype=np.float32)]
    curr_samples = 8000
    segment_markers = []

    for _ in range(300):
        choice = rng.choice(["unk", "unk", "pause", "sil"])
        if choice == "unk" and unk_rows:
            u = rng.choice(unk_rows)
            full_p = os.path.join(BASE_DIR, u["kws_wav_path"])
            u_data, _ = sf.read(full_p, dtype="float32")
            blocks.append(u_data)
            segment_markers.append((curr_samples, curr_samples + len(u_data), u["sample_id"]))
            curr_samples += len(u_data)
        elif choice == "sil" and sil_rows:
            s = rng.choice(sil_rows)
            full_p = os.path.join(BASE_DIR, s["kws_wav_path"])
            s_data, _ = sf.read(full_p, dtype="float32")
            blocks.append(s_data)
            curr_samples += len(s_data)
        else:
            gap = np_rng.normal(0, 0.001, int(np_rng.uniform(0.5, 2.0) * sample_rate)).astype(np.float32)
            blocks.append(gap)
            curr_samples += len(gap)

    full_stream = np.concatenate(blocks)
    stream_len = len(full_stream)
    print(f"Mined training stream constructed: {stream_len / sample_rate:.1f} seconds ({stream_len} samples)")

    # Run Stage 1 sliding window with candidate generation threshold (theta=0.50, k=2)
    mined_crops = []
    hop_samples = 1600
    win_samples = 16000
    num_windows = (stream_len - win_samples) // hop_samples + 1

    chunk_size = 128
    consecutive_count = 0
    current_streak_class = None
    lockout_until_idx = -1

    for w_start_idx in range(0, num_windows, chunk_size):
        w_end_idx = min(num_windows, w_start_idx + chunk_size)
        batch_windows = []
        w_indices = []

        for w in range(w_start_idx, w_end_idx):
            s_idx = w * hop_samples
            e_idx = s_idx + win_samples
            batch_windows.append(full_stream[s_idx:e_idx])
            w_indices.append(w)

        batch_t = torch.from_numpy(np.array(batch_windows, dtype=np.float32)).float().to(device)
        with torch.no_grad():
            feats = extractor(batch_t)
            probs = F.softmax(stage1_model(feats), dim=1).cpu().numpy()

        preds, confs, _ = apply_confidence_policy(probs, threshold=0.50)

        for i, w in enumerate(w_indices):
            pred_c = preds[i]
            conf = confs[i]

            if w < lockout_until_idx:
                consecutive_count = 0
                current_streak_class = None
                continue

            if pred_c in TARGET_COMMAND_INDICES:
                if pred_c == current_streak_class:
                    consecutive_count += 1
                else:
                    current_streak_class = pred_c
                    consecutive_count = 1

                if consecutive_count >= 2:
                    # Stage 1 triggered a false alarm on non-target training audio!
                    # Extract 1.2s crop around the trigger window
                    w_start_sample = w * hop_samples
                    crop_center_sample = w_start_sample + (win_samples // 2)
                    crop_start_sample = max(0, crop_center_sample - (target_crop_samples // 2))
                    crop_end_sample = crop_start_sample + target_crop_samples

                    if crop_end_sample <= stream_len:
                        crop_data = full_stream[crop_start_sample:crop_end_sample]
                    else:
                        crop_data = np.pad(full_stream[crop_start_sample:], (0, crop_end_sample - stream_len))

                    mined_crops.append(crop_data)
                    lockout_until_idx = w + int(0.60 * sample_rate / hop_samples)
                    current_streak_class = None
                    consecutive_count = 0
            else:
                consecutive_count = 0
                current_streak_class = None

    print(f"Total hard negative crops mined from Stage 1 triggers on training stream: {len(mined_crops)}")
    return mined_crops


def build_verifier_training_dataset(df_train, stage1_model, extractor, device, seed=42):
    """
    Builds the dataset for Stage 2 verifier training.
    All crops strictly originate from df_train.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    crops = []
    labels = []
    metadata = []

    target_len = 19200 # 1.2s at 16kHz

    # 1. Positives: Target Commands with timing jitter
    target_rows = df_train[df_train["class_label"].isin(TARGET_CLASSES)].to_dict("records")
    for r in target_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        lbl_id = VERIFIER_LABEL_TO_ID[r["class_label"]]

        # Jitter crop 1: Center-left (offset ~800 to 1600 samples)
        offset1 = rng.randint(400, 1600)
        c1 = extract_or_pad_crop(audio, target_len, offset1)
        crops.append(c1)
        labels.append(lbl_id)
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "positive_jitter_1"})

        # Jitter crop 2: Center-right (offset ~1600 to 2800 samples)
        offset2 = rng.randint(1600, 2800)
        c2 = extract_or_pad_crop(audio, target_len, offset2)
        crops.append(c2)
        labels.append(lbl_id)
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "positive_jitter_2"})

    # 2. Hard Negatives: Partial / Edge-Aligned Incomplete Commands
    # Take first 45% or second 45% of genuine command -> must be rejected
    for r in target_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")

        # First half crop (onset only)
        half_len = int(len(audio) * rng.uniform(0.40, 0.55))
        first_half = audio[:half_len]
        c_fh = extract_or_pad_crop(first_half, target_len, offset_samples=rng.randint(400, 2000))
        crops.append(c_fh)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "hard_neg_onset_only"})

        # Second half crop (coda only)
        second_half = audio[len(audio) - half_len :]
        c_sh = extract_or_pad_crop(second_half, target_len, offset_samples=rng.randint(400, 2000))
        crops.append(c_sh)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "hard_neg_coda_only"})

    # 3. Non-Target Speech: Full UNKNOWN utterances
    unk_rows = df_train[df_train["class_label"] == "UNKNOWN"].to_dict("records")
    for r in unk_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        offset = rng.randint(400, 2400)
        c = extract_or_pad_crop(audio, target_len, offset)
        crops.append(c)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "unknown_speech"})

    # 4. SILENCE / Ambient Background
    sil_rows = df_train[df_train["class_label"] == "SILENCE"].to_dict("records")
    for r in sil_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        offset = rng.randint(0, 1600)
        c = extract_or_pad_crop(audio, target_len, offset)
        crops.append(c)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "ambient_silence"})

    # 5. Mined Stage 1 False Triggers
    mined = mine_stage1_hard_negatives(stage1_model, extractor, device, df_train)
    for idx, m_crop in enumerate(mined):
        crops.append(m_crop)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": f"mined_stage1_fa_{idx}", "voice_id": "mined_stream", "type": "mined_stage1_trigger"})

    return np.array(crops, dtype=np.float32), np.array(labels, dtype=np.int64), metadata


def build_verifier_val_dataset(df_val, seed=777):
    """Builds validation crops from df_val (unseen validation voices)."""
    rng = random.Random(seed)
    crops = []
    labels = []
    metadata = []
    target_len = 19200

    # 1. Target Commands (160 crops)
    target_rows = df_val[df_val["class_label"].isin(TARGET_CLASSES)].to_dict("records")
    for r in target_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        offset = rng.randint(800, 2000)
        c = extract_or_pad_crop(audio, target_len, offset)
        crops.append(c)
        labels.append(VERIFIER_LABEL_TO_ID[r["class_label"]])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "val_positive"})

    # 2. UNKNOWN Speech (80 crops)
    unk_rows = df_val[df_val["class_label"] == "UNKNOWN"].to_dict("records")
    for r in unk_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        offset = rng.randint(800, 2000)
        c = extract_or_pad_crop(audio, target_len, offset)
        crops.append(c)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "val_unknown"})

    # 3. SILENCE (80 crops)
    sil_rows = df_val[df_val["class_label"] == "SILENCE"].to_dict("records")
    for r in sil_rows:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        offset = rng.randint(0, 1600)
        c = extract_or_pad_crop(audio, target_len, offset)
        crops.append(c)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "val_silence"})

    # 4. Edge-aligned partials from validation targets (80 crops)
    sampled_tgts = rng.sample(target_rows, 80)
    for r in sampled_tgts:
        full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
        audio, _ = sf.read(full_p, dtype="float32")
        half_len = int(len(audio) * 0.50)
        c = extract_or_pad_crop(audio[:half_len], target_len, rng.randint(400, 1600))
        crops.append(c)
        labels.append(VERIFIER_LABEL_TO_ID["REJECT"])
        metadata.append({"source_id": r["sample_id"], "voice_id": r["voice_id"], "type": "val_partial_chop"})

    return np.array(crops, dtype=np.float32), np.array(labels, dtype=np.int64), metadata


# -------------------------------------------------------------
# 3. PyTorch Dataset with Dynamic Audio Augmentation
# -------------------------------------------------------------

class VerifierCropDataset(Dataset):
    def __init__(self, crops: np.ndarray, labels: np.ndarray, is_training: bool = True, seed: int = 42):
        self.crops = crops
        self.labels = labels
        self.is_training = is_training
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        audio = self.crops[idx].copy()
        label = self.labels[idx]

        if self.is_training:
            # Random gain
            gain = self.rng.uniform(0.85, 1.15)
            audio = audio * gain

            # Random time shift (+- 800 samples)
            shift = self.rng.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio[shift:], (0, shift))
            elif shift < 0:
                audio = np.pad(audio[:shift], (-shift, 0))

            # Dynamic noise injection (50% prob)
            if self.rng.rand() < 0.50:
                noise_type = self.rng.choice(["white", "pink", "hvac", "babble"])
                snr_db = self.rng.uniform(5.0, 20.0)
                n_samples = len(audio)

                if noise_type == "white":
                    noise = self.rng.normal(0, 0.1, n_samples).astype(np.float32)
                elif noise_type == "pink":
                    w = self.rng.normal(0, 0.1, n_samples)
                    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
                    noise = signal.lfilter(b, a, w).astype(np.float32)
                elif noise_type == "hvac":
                    t = np.linspace(0, 1.2, n_samples, endpoint=False)
                    hum = 0.03 * np.sin(2 * np.pi * 50 * t) + 0.015 * np.sin(2 * np.pi * 100 * t)
                    rumble = signal.lfilter(*signal.butter(3, 300.0 / 8000.0, btype="lowpass"), self.rng.normal(0, 0.04, n_samples))
                    noise = (hum + rumble).astype(np.float32)
                else:
                    noise = self.rng.normal(0, 0.05, n_samples).astype(np.float32)

                s_rms = np.sqrt(np.mean(audio**2)) + 1e-8
                n_rms = np.sqrt(np.mean(noise**2)) + 1e-8
                scale = (s_rms * (10.0 ** (-snr_db / 20.0))) / n_rms
                audio = audio + noise * scale
                peak = np.max(np.abs(audio))
                if peak > 0.98:
                    audio = audio * (0.98 / peak)

        return torch.from_numpy(audio).float(), torch.tensor(label, dtype=torch.long)


# -------------------------------------------------------------
# 4. Training Engine
# -------------------------------------------------------------

def train_verifier_for_seed(seed: int, train_crops, train_labels, val_crops, val_labels, extractor, device, config):
    print(f"\n=======================================================")
    print(f"Starting Training: Stage 2 Temporal Verifier (Seed {seed})")
    print(f"=======================================================")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = VerifierCropDataset(train_crops, train_labels, is_training=True, seed=seed)
    val_ds = VerifierCropDataset(val_crops, val_labels, is_training=False, seed=seed)

    batch_size = config["stage2"]["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TemporalKWSVerifier(num_classes=5, dropout=0.2).to(device)

    # Class weights: 5 classes (TRAI, PHAI, LEN, XUONG, REJECT)
    # Give target commands slightly higher weight to maintain high recall
    class_weights = torch.tensor([1.2, 1.2, 1.2, 1.2, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = config["stage2"]["training"]["learning_rate"]
    weight_decay = config["stage2"]["training"]["weight_decay"]
    num_epochs = config["stage2"]["training"]["num_epochs"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_target_recall": [],
        "val_reject_recall": [],
        "lr": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_verifier_seed{seed}.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []

        for audio_b, target_b in train_loader:
            audio_b = audio_b.to(device)
            target_b = target_b.to(device)

            with torch.no_grad():
                feats = extractor(audio_b)
                # Apply SpecAugment during training
                if random.random() < 0.50:
                    t_mask = random.randint(1, 10)
                    t_start = random.randint(0, max(0, feats.shape[-1] - t_mask))
                    feats[..., t_start : t_start + t_mask] = 0.0

            logits = model(feats)
            loss = criterion(logits, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_targets.extend(target_b.cpu().numpy())

        scheduler.step()
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        mean_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for audio_b, target_b in val_loader:
                audio_b = audio_b.to(device)
                target_b = target_b.to(device)
                feats = extractor(audio_b)
                logits = model(feats)
                loss = criterion(logits, target_b)

                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_targets.extend(target_b.cpu().numpy())

        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)
        mean_val_loss = float(np.mean(val_losses))

        # Target recall (classes 0..3) vs REJECT recall (class 4)
        cm = confusion_matrix(all_val_targets, all_val_preds, labels=list(range(5)))
        tgt_recs = [cm[i, i] / max(1, np.sum(cm[i, :])) for i in range(4)]
        mean_tgt_rec = float(np.mean(tgt_recs))
        reject_rec = float(cm[4, 4] / max(1, np.sum(cm[4, :])))

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(mean_train_loss)
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(mean_val_loss)
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["val_target_recall"].append(mean_tgt_rec)
        history["val_reject_recall"].append(reject_rec)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {mean_train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {mean_val_loss:.4f} Acc: {val_acc*100:.1f}% MacroF1: {val_f1*100:.2f}% | "
              f"TgtRec: {mean_tgt_rec*100:.1f}% RejRec: {reject_rec*100:.1f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "val_target_recall": mean_tgt_rec,
                "val_reject_recall": reject_rec,
                "class_names": VERIFIER_CLASSES,
                "input_samples": 19200
            }, best_ckpt_path)

    print(f"--> Best Checkpoint Seed {seed}: Epoch {best_epoch} with Val Macro F1: {best_val_f1*100:.2f}%")
    print(f"    Saved to: {best_ckpt_path}")

    # Save log history
    log_p = os.path.join(LOGS_DIR, f"history_verifier_seed{seed}.json")
    with open(log_p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_ckpt_path, best_val_f1


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    cfg_p = os.path.join(EXP_DIR, "config.json")
    with open(cfg_p, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Environment Device: {device}")

    extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
    ).to(device)

    # Load frozen manifests
    train_manifest_p = os.path.join(BASE_DIR, config["dataset"]["train_manifest"])
    val_manifest_p = os.path.join(BASE_DIR, config["dataset"]["val_manifest"])
    df_train = pd.read_csv(train_manifest_p)
    df_val = pd.read_csv(val_manifest_p)

    print(f"Dataset v2: Train Manifest rows: {len(df_train)} | Val Manifest rows: {len(df_val)}")

    # Load frozen Stage 1 model (Seed 42) for mining hard negatives on training audio
    s1_ckpt_p = os.path.join(BASE_DIR, config["stage1"]["checkpoints"]["seed42"])
    s1_ckpt = torch.load(s1_ckpt_p, map_location="cpu", weights_only=False)
    stage1_model = VietnameseKWSNet(num_classes=6)
    stage1_model.load_state_dict(s1_ckpt["model_state_dict"])
    stage1_model.to(device)
    stage1_model.eval()

    # Build verifier training and validation datasets
    print("\n--- Constructing Verifier Training Crops ---")
    train_crops, train_labels, train_meta = build_verifier_training_dataset(df_train, stage1_model, extractor, device, seed=42)
    print(f"Generated {len(train_crops)} training crops from {len(df_train)} unique training source utterances.")
    lbl_counts = {VERIFIER_CLASSES[i]: int(np.sum(train_labels == i)) for i in range(5)}
    print(f"Training crop distribution: {lbl_counts}")

    print("\n--- Constructing Verifier Validation Crops ---")
    val_crops, val_labels, val_meta = build_verifier_val_dataset(df_val, seed=777)
    print(f"Generated {len(val_crops)} validation crops from {len(df_val)} unique validation source utterances.")
    val_counts = {VERIFIER_CLASSES[i]: int(np.sum(val_labels == i)) for i in range(5)}
    print(f"Validation crop distribution: {val_counts}")

    # Train across seeds 42, 43, 44
    seeds = config["stage2"]["training"]["seeds"]
    results = {}
    for seed in seeds:
        ckpt_p, val_f1 = train_verifier_for_seed(
            seed, train_crops, train_labels, val_crops, val_labels, extractor, device, config
        )
        results[f"seed{seed}"] = {"checkpoint": ckpt_p, "best_val_macro_f1": val_f1}

    print("\n=======================================================")
    print("All Stage 2 Verifier Training Runs Completed!")
    print(json.dumps(results, indent=2))
    print("=======================================================")


if __name__ == "__main__":
    main()
