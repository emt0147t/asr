"""
Phase 47: Recall-Preserving Hard-Negative Cascade Training (Candidate A).
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v3/

Implements:
1. Stage 1 (VietnameseKWSNet, 138,534 params):
   - 6 classes: TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE.
   - Positive-preserving loss weighting: w_target = 1.25, w_unknown = 1.0, w_silence = 1.0.
   - SpecAugment + SNR background mixing + temporal jitter.
2. Stage 2 (TemporalKWSVerifier, 340,773 params):
   - 5 classes: TRAI, PHAI, LEN, XUONG, REJECT.
   - 1.2s temporal crops with multi-offset positive jitter.
   - Multi-source hard-negative rejection pool (unknowns, silences, partial commands, mined Stage 1 false alarms).
   - Positive-preserving loss weighting: w_target = 1.5, w_reject = 0.8 to prevent over-rejection.
3. Multi-seed training across seeds 42, 43, 44.
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import f1_score, accuracy_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

DATA_DIR = os.path.join(EXP_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(EXP_DIR, "models", "hard_negative_cascade_v3")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
LOGS_DIR = os.path.join(MODELS_DIR, "logs")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
LABEL_TO_ID_6 = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5}
LABEL_TO_ID_S2 = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "REJECT": 4}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoiseBank:
    def __init__(self):
        self.noise_types = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]

    def sample_noise(self, length: int) -> np.ndarray:
        t = random.choice(self.noise_types)
        rng = np.random.RandomState()
        if t == "near_silence":
            return rng.normal(0, 0.0008, length).astype(np.float32)
        elif t == "hvac_hum":
            time_arr = np.arange(length) / 16000.0
            return (0.012 * np.sin(2 * np.pi * 50.0 * time_arr) + rng.normal(0, 0.005, length)).astype(np.float32)
        elif t == "office_ambient":
            return rng.normal(0, 0.008, length).astype(np.float32)
        else:
            return rng.normal(0, 0.010, length).astype(np.float32)


class KWS6ClassDataset(Dataset):
    def __init__(self, manifest_csv: str, is_train: bool = True, augment: bool = True):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment = augment and is_train
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160)
        self.noise_bank = NoiseBank() if self.augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        audio, _ = sf.read(p, dtype="float32")
        lbl = LABEL_TO_ID_6[row["class_label"]]

        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        elif len(audio) > 16000:
            audio = audio[:16000]

        if self.augment:
            shift = random.randint(-600, 600)
            if shift > 0:
                audio = np.pad(audio[:-shift], (shift, 0))
            elif shift < 0:
                audio = np.pad(audio[-shift:], (0, -shift))

            gain = random.uniform(0.85, 1.15)
            audio = np.clip(audio * gain, -0.98, 0.98)

            if random.random() < 0.40:
                noise = self.noise_bank.sample_noise(16000)
                snr = random.uniform(8.0, 22.0)
                sig_rms = np.sqrt(np.mean(audio**2)) + 1e-6
                noise_rms = np.sqrt(np.mean(noise**2)) + 1e-6
                audio = audio + noise * (sig_rms / (10.0 ** (snr / 20.0)) / noise_rms)
                audio = np.clip(audio, -0.98, 0.98)

        t_audio = torch.from_numpy(audio).unsqueeze(0).float()
        with torch.no_grad():
            feat = self.extractor(t_audio).squeeze(0)  # (1, 40, 101)

        if self.augment and random.random() < 0.40:
            f_dim = feat.size(1)
            f_len = random.randint(1, 5)
            f0 = random.randint(0, f_dim - f_len)
            feat[:, f0 : f0 + f_len, :] = 0.0

            t_dim = feat.size(2)
            t_len = random.randint(1, 8)
            t0 = random.randint(0, t_dim - t_len)
            feat[:, :, t0 : t0 + t_len] = 0.0

        return feat, lbl


class VerifierDataset(Dataset):
    """Dataset for Stage 2 TCN Verifier using 1.2s (19,200 sample) context crops."""
    def __init__(self, samples_list: list, is_train: bool = True, augment: bool = True):
        self.samples = samples_list
        self.is_train = is_train
        self.augment = augment and is_train
        self.noise_bank = NoiseBank() if self.augment else None
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

    # Positive-preserving loss weights: higher weight on 4 target commands
    class_weights = torch.tensor([1.25, 1.25, 1.25, 1.25, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_score = -1.0
    best_weights_path = os.path.join(CHECKPOINTS_DIR, f"stage1_recall_preserving_seed{seed}.pt")

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

        v_targets = np.array(v_targets)
        v_preds = np.array(v_preds)

        # Compute target command recall specifically
        cmd_mask = v_targets < 4
        cmd_rec = np.mean(v_preds[cmd_mask] == v_targets[cmd_mask]) if np.sum(cmd_mask) > 0 else 0.0
        val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(v_targets, v_preds)

        # Composite score prioritizing target command recall
        composite_score = 0.70 * cmd_rec + 0.30 * val_f1

        if composite_score > best_val_score:
            best_val_score = composite_score
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state": model.state_dict(),
                "val_macro_f1": float(val_f1),
                "val_accuracy": float(val_acc),
                "val_target_recall": float(cmd_rec),
                "architecture": "VietnameseKWSNet",
                "num_classes": 6
            }, best_weights_path)

        if epoch % 5 == 0 or epoch == 25:
            print(f"Epoch {epoch:02d}/25 - Loss: {total_loss/len(train_ds):.4f} - Cmd Rec: {cmd_rec*100:.1f}% - Val F1: {val_f1:.4f} (Best Score: {best_val_score:.4f})")

    print(f"Stage 1 Seed {seed} Best Weights saved to {best_weights_path}")
    return best_weights_path


def mine_stage1_proposals(stage1_model, extractor, device, df_train):
    """Mines false alarms generated by retrained Stage 1 on continuous training speech."""
    stage1_model.eval()
    unk_rows = df_train[df_train["class_label"] == "UNKNOWN"].to_dict("records")
    sil_rows = df_train[df_train["class_label"] == "SILENCE"].to_dict("records")

    rng = random.Random(12345)
    np_rng = np.random.RandomState(12345)

    blocks = [np.zeros(8000, dtype=np.float32)]
    for _ in range(200):
        c = rng.choice(["unk", "unk", "sil", "gap"])
        if c == "unk" and unk_rows:
            r = rng.choice(unk_rows)
            p = os.path.join(BASE_DIR, r["kws_wav_path"])
            a, _ = sf.read(p, dtype="float32")
            blocks.append(a)
        elif c == "sil" and sil_rows:
            r = rng.choice(sil_rows)
            p = os.path.join(BASE_DIR, r["kws_wav_path"])
            a, _ = sf.read(p, dtype="float32")
            blocks.append(a)
        else:
            blocks.append(np_rng.normal(0, 0.001, int(np_rng.uniform(0.5, 1.5) * 16000)).astype(np.float32))

    full_stream = np.concatenate(blocks)
    stream_len = len(full_stream)
    hop_samples = 1600
    win_samples = 16000
    num_windows = (stream_len - win_samples) // hop_samples + 1

    chunk_size = 128
    consecutive_count = 0
    streak_class = None
    mined_crops = []

    for w_start in range(0, num_windows, chunk_size):
        w_end = min(num_windows, w_start + chunk_size)
        batch = [full_stream[w * hop_samples : w * hop_samples + win_samples] for w in range(w_start, w_end)]
        batch_t = torch.from_numpy(np.array(batch, dtype=np.float32)).float().to(device)

        with torch.no_grad():
            feats = extractor(batch_t)
            probs = F.softmax(stage1_model(feats), dim=1).cpu().numpy()

        for i, w in enumerate(range(w_start, w_end)):
            p = probs[i]
            tgt_p = p[:4]
            max_c = int(np.argmax(tgt_p))
            max_val = float(tgt_p[max_c])

            if max_val >= 0.50 and max_val > p[4] and max_val > p[5]:
                if max_c == streak_class:
                    consecutive_count += 1
                else:
                    streak_class = max_c
                    consecutive_count = 1

                if consecutive_count >= 2:
                    # Extract centered 1.2s crop around the trigger window
                    w_s = (w - 1) * hop_samples - 800
                    c_start = max(0, w_s)
                    c_end = c_start + 19200
                    if c_end <= stream_len:
                        crop = full_stream[c_start:c_end]
                    else:
                        crop = np.pad(full_stream[c_start:], (0, c_end - stream_len))
                    mined_crops.append({"audio": crop, "label_id": LABEL_TO_ID_S2["REJECT"]})
                    consecutive_count = 0
                    streak_class = None
            else:
                consecutive_count = 0
                streak_class = None

    print(f"Mined {len(mined_crops)} Stage 1 false trigger crops from continuous training speech.")
    return mined_crops


def train_stage2_verifier(seed: int, s1_ckpt_path: str, device):
    set_seed(seed)
    print(f"\n--- Training Stage 2 TCN Verifier (TemporalKWSVerifier) Seed {seed} ---")

    s1_model = VietnameseKWSNet(num_classes=6).to(device)
    ckpt = torch.load(s1_ckpt_path, map_location=device, weights_only=False)
    s1_model.load_state_dict(ckpt["model_state"])
    s1_model.eval()

    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160).to(device)

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val_manifest.csv"))

    target_len = 19200
    rng = random.Random(seed)

    # 1. Build training crops
    train_crops = []

    # Positives with multiple jitter offsets to make verifier robust to timing
    for _, row in train_df[train_df["is_target_command"] == True].iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        lbl_id = LABEL_TO_ID_S2[row["class_label"]]

        for jitter in [800, 1600, 2400]:
            crop = np.zeros(target_len, dtype=np.float32)
            n = len(wav)
            pad_front = jitter
            crop[pad_front : pad_front + min(n, target_len - pad_front)] = wav[: min(n, target_len - pad_front)]
            train_crops.append({"audio": crop, "label_id": lbl_id})

    # Non-target speech (full UNKNOWN)
    for _, row in train_df[train_df["class_label"] == "UNKNOWN"].iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        crop = np.zeros(target_len, dtype=np.float32)
        n = min(len(wav), target_len)
        crop[:n] = wav[:n]
        train_crops.append({"audio": crop, "label_id": LABEL_TO_ID_S2["REJECT"]})

    # Ambient silence
    for _, row in train_df[train_df["class_label"] == "SILENCE"].iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        crop = np.zeros(target_len, dtype=np.float32)
        n = min(len(wav), target_len)
        crop[:n] = wav[:n]
        train_crops.append({"audio": crop, "label_id": LABEL_TO_ID_S2["REJECT"]})

    # Partial commands (must be rejected)
    for _, row in train_df[train_df["is_target_command"] == True].iloc[::2].iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        half_len = int(len(wav) * 0.45)
        crop = np.zeros(target_len, dtype=np.float32)
        crop[1600 : 1600 + half_len] = wav[:half_len]
        train_crops.append({"audio": crop, "label_id": LABEL_TO_ID_S2["REJECT"]})

    # Mined Stage 1 false triggers
    mined_crops = mine_stage1_proposals(s1_model, extractor, device, train_df)
    train_crops.extend(mined_crops)

    # 2. Build validation crops
    val_crops = []
    for _, row in val_df.iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        wav, _ = sf.read(p, dtype="float32")
        crop = np.zeros(target_len, dtype=np.float32)
        n = min(len(wav), target_len)
        crop[1600 : 1600 + min(n, target_len - 1600)] = wav[: min(n, target_len - 1600)]

        lbl_s1 = row["class_label"]
        if lbl_s1 in ["TRAI", "PHAI", "LEN", "XUONG"]:
            s2_lbl = LABEL_TO_ID_S2[lbl_s1]
        else:
            s2_lbl = LABEL_TO_ID_S2["REJECT"]
        val_crops.append({"audio": crop, "label_id": s2_lbl})

    train_ds = VerifierDataset(train_crops, is_train=True, augment=True)
    val_ds = VerifierDataset(val_crops, is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = TemporalKWSVerifier(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    # Positive-preserving weights for Stage 2: penalize false rejection heavily!
    s2_weights = torch.tensor([1.5, 1.5, 1.5, 1.5, 0.8], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=s2_weights)

    best_val_score = -1.0
    best_weights_path = os.path.join(CHECKPOINTS_DIR, f"stage2_recall_preserving_seed{seed}.pt")

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

        v_targets = np.array(v_targets)
        v_preds = np.array(v_preds)

        # Target command recall on verifier
        cmd_mask = v_targets < 4
        cmd_rec = np.mean(v_preds[cmd_mask] == v_targets[cmd_mask]) if np.sum(cmd_mask) > 0 else 0.0
        rej_mask = v_targets == 4
        rej_rec = np.mean(v_preds[rej_mask] == v_targets[rej_mask]) if np.sum(rej_mask) > 0 else 0.0
        val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)

        # Composite score strictly guaranteeing high target command recall (weight 0.70)
        score = 0.70 * cmd_rec + 0.30 * rej_rec

        if score > best_val_score:
            best_val_score = score
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state": model.state_dict(),
                "val_macro_f1": float(val_f1),
                "val_target_recall": float(cmd_rec),
                "val_reject_recall": float(rej_rec),
                "composite_score": float(score),
                "architecture": "TemporalKWSVerifier",
                "num_classes": 5
            }, best_weights_path)

        if epoch % 5 == 0 or epoch == 25:
            print(f"Verifier Epoch {epoch:02d}/25 - Loss: {total_loss/len(train_ds):.4f} - Cmd Rec: {cmd_rec*100:.1f}% - Rej Rec: {rej_rec*100:.1f}% - Val F1: {val_f1:.4f} (Best Score: {best_val_score:.4f})")

    print(f"Stage 2 Seed {seed} Best Weights saved to {best_weights_path}")
    return best_weights_path


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for Candidate A training:", device)

    seeds = [42, 43, 44]
    results = {}

    for seed in seeds:
        print(f"\n========================================================")
        print(f"Training Candidate A (Recall-Preserving Cascade) Seed {seed}")
        print(f"========================================================")

        s1_ckpt = train_stage1(seed, device)
        s2_ckpt = train_stage2_verifier(seed, s1_ckpt, device)

        results[f"seed{seed}"] = {
            "stage1_checkpoint": os.path.relpath(s1_ckpt, BASE_DIR).replace("\\", "/"),
            "stage2_checkpoint": os.path.relpath(s2_ckpt, BASE_DIR).replace("\\", "/")
        }

    summary_file = os.path.join(LOGS_DIR, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll Candidate A training completed. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
