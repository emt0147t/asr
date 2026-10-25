"""
Phase 34: Training Six-Class Vietnamese KWS CNN v2 with Train-Only Robustness Augmentation.
Implements:
  - Architecture: VietnameseKWSNet (compact 6-class CNN, 107,334 parameters)
  - Train-Only Robustness Augmentation:
    * Time shift (+/- 400 samples) and gain scaling (0.85 - 1.15)
    * Multi-condition noise injection (50% prob: white, pink, HVAC, babble at 5-20 dB SNR)
    * SpecAugment (time & frequency masking)
  - Balanced class weighting: [2.0, 2.0, 2.0, 2.0, 1.0, 1.0] (for 170 vs 340 train samples)
  - Checkpoint selection: Validation macro-F1 on untouched clean validation set
  - Repetitions across seeds 42, 43, 44
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")
LOGS_DIR = os.path.join(EXP_DIR, "logs")

LABEL_TO_ID = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5}
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

class NoiseSourceBank:
    """Provides cached background noise tracks for fast on-the-fly train-only augmentation."""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        dur_samples = sample_rate * 30 # 30 seconds
        
        # 1. White noise
        self.white = np.random.normal(0, 0.1, dur_samples).astype(np.float32)
        
        # 2. Pink noise
        w = np.random.normal(0, 0.1, dur_samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        self.pink = signal.lfilter(b, a, w).astype(np.float32)
        
        # 3. HVAC noise (hum + low rumble)
        t = np.linspace(0, 30.0, dur_samples, endpoint=False)
        hum = 0.04 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 100 * t)
        rumble = signal.lfilter(*signal.butter(3, 300.0 / 8000.0, btype="lowpass"), np.random.normal(0, 0.05, dur_samples))
        self.hvac = (hum + rumble).astype(np.float32)
        
        # 4. Multi-talker babble
        babble_p = os.path.join(BASE_DIR, "experiments/vieneu_kws_robustness_campaign/babble_sources/synthetic_vietnamese_babble_30s.wav")
        if os.path.exists(babble_p):
            self.babble, _ = sf.read(babble_p, dtype="float32")
        else:
            self.babble = self.pink.copy()
            
    def sample_noise(self, noise_type: str, num_samples: int = 16000) -> np.ndarray:
        if noise_type == "white":
            src = self.white
        elif noise_type == "pink":
            src = self.pink
        elif noise_type == "hvac":
            src = self.hvac
        elif noise_type == "babble":
            src = self.babble
        else:
            src = self.white
            
        start_idx = np.random.randint(0, max(1, len(src) - num_samples))
        return src[start_idx : start_idx + num_samples].copy()

class VieNeuKWS6ClassDatasetV2(Dataset):
    """Loads 16 kHz 1.0s WAVs with train-only robustness augmentation."""
    def __init__(self, manifest_csv: str, is_train: bool = True, augment: bool = True,
                 noise_bank: NoiseSourceBank = None, seed: int = 42):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.augment = augment and is_train
        self.noise_bank = noise_bank or (NoiseSourceBank() if self.augment else None)
        self.seed = seed
        self.extractor = LogMelFeatureExtractor(
            sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
        )
        
        # Preload audio into memory
        self.audio_data = []
        self.labels = []
        self.sample_ids = []
        self.onsets = []
        self.durs = []
        self.is_silences = []
        
        for _, row in self.df.iterrows():
            kws_rel = row["kws_wav_path"]
            kws_full = os.path.join(BASE_DIR, kws_rel) if not os.path.isabs(kws_rel) else kws_rel
            data, sr = sf.read(kws_full, dtype="float32")
            if len(data) < 16000:
                data = np.pad(data, (0, 16000 - len(data)))
            elif len(data) > 16000:
                data = data[:16000]
                
            self.audio_data.append(data)
            self.labels.append(LABEL_TO_ID[row["class_label"]])
            self.sample_ids.append(row["sample_id"])
            self.onsets.append(float(row.get("kws_onset_sec", 0.15)))
            self.durs.append(float(row.get("kws_active_speech_duration_sec", 0.65)))
            self.is_silences.append(row["class_label"] == "SILENCE")
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        audio = self.audio_data[idx].copy()
        label = self.labels[idx]
        onset_s = self.onsets[idx]
        dur_s = self.durs[idx]
        is_sil = self.is_silences[idx]
        
        if self.augment:
            # 1. Subtle time shift (+/- 400 samples = +/- 25ms)
            shift = np.random.randint(-400, 401)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]
                
            # 2. Gain scaling (0.85x to 1.15x)
            scale = np.random.uniform(0.85, 1.15)
            audio = audio * scale
            
            # 3. Dynamic multi-condition noise injection (50% prob)
            if np.random.rand() < 0.50 and self.noise_bank:
                n_type = np.random.choice(["white", "pink", "hvac", "babble"])
                snr = np.random.uniform(5.0, 20.0)
                noise_seg = self.noise_bank.sample_noise(n_type, 16000)
                
                if not is_sil:
                    onset_idx = int(max(0, min(15000, onset_s * 16000)))
                    end_idx = int(min(16000, onset_idx + max(1000, dur_s * 16000)))
                    speech_active = audio[onset_idx:end_idx]
                    speech_rms = np.sqrt(np.mean(speech_active**2)) + 1e-8
                    
                    noise_active = noise_seg[onset_idx:end_idx]
                    noise_rms = np.sqrt(np.mean(noise_active**2)) + 1e-8
                    
                    target_noise_rms = speech_rms * (10.0 ** (-snr / 20.0))
                    scale_factor = target_noise_rms / noise_rms
                    audio = audio + noise_seg * scale_factor
                else:
                    audio = audio + noise_seg * 0.01
                    
            audio = np.clip(audio, -0.98, 0.98)
            
        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            feat = self.extractor(audio_t).squeeze(0) # (1, 40, 101)
            
        # 4. SpecAugment (training only, 50% prob)
        if self.augment and np.random.rand() < 0.50:
            # Time mask
            t_mask_len = np.random.randint(1, 11)
            t_start = np.random.randint(0, max(1, 101 - t_mask_len))
            feat[:, :, t_start : t_start + t_mask_len] = 0.0
            
            # Freq mask
            f_mask_len = np.random.randint(1, 7)
            f_start = np.random.randint(0, max(1, 40 - f_mask_len))
            feat[:, f_start : f_start + f_mask_len, :] = 0.0
            
        return feat, label

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
    all_probs = []
    
    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            targets = targets.to(device)
            
            logits = model(feats)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    
    # Per-class metrics
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(6)))
    class_rec = {}
    for cid, cname in ID_TO_LABEL.items():
        total_c = np.sum(np.array(all_targets) == cid)
        class_rec[cname] = float(cm[cid, cid] / total_c) if total_c > 0 else 0.0
        
    # UNKNOWN FA rate into target commands
    unk_mask = (np.array(all_targets) == LABEL_TO_ID["UNKNOWN"])
    if np.sum(unk_mask) > 0:
        unk_preds = np.array(all_preds)[unk_mask]
        unk_fa_rate = float(np.sum(np.isin(unk_preds, [0, 1, 2, 3])) / np.sum(unk_mask))
    else:
        unk_fa_rate = 0.0
        
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "per_class_recall": class_rec,
        "unknown_false_activation_rate": unk_fa_rate,
        "confusion_matrix": cm.tolist()
    }

def train_model(seed: int, config: dict):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Training VietnameseKWSNet v2 — Seed {seed} on {device}")
    print(f"=======================================================")
    
    train_manifest = os.path.join(BASE_DIR, config["dataset"]["train_manifest"])
    val_manifest = os.path.join(BASE_DIR, config["dataset"]["val_manifest"])
    
    noise_bank = NoiseSourceBank(sample_rate=16000)
    train_ds = VieNeuKWS6ClassDatasetV2(train_manifest, is_train=True, augment=True, noise_bank=noise_bank, seed=seed)
    val_ds = VieNeuKWS6ClassDatasetV2(val_manifest, is_train=False, augment=False, seed=seed)
    
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    
    print(f"Loaded {len(train_ds)} training samples and {len(val_ds)} validation samples.")
    
    model = VietnameseKWSNet(num_classes=6).to(device)
    
    # Class weights [2.0, 2.0, 2.0, 2.0, 1.0, 1.0] to balance 170 targets vs 340 negatives
    weights_t = torch.tensor(config["training"]["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["training"]["scheduler"]["mode"],
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        min_lr=config["training"]["scheduler"]["min_lr"]
    )
    
    num_epochs = config["training"]["num_epochs"]
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_checkpoint_seed{seed}.pt")
    
    epoch_logs = []
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_val(model, val_loader, criterion, device)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_unk_fa = val_metrics["unknown_false_activation_rate"]
        
        scheduler.step(val_f1)
        cur_lr = optimizer.param_groups[0]["lr"]
        
        is_best = False
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            is_best = True
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "val_metrics": val_metrics,
                "config": config
            }, best_ckpt_path)
            
        dur = time.time() - t0
        best_marker = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:02d}/{num_epochs:02d} ({dur:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% F1: {tr_f1*100:.2f}% | Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}% UNK FA: {val_unk_fa*100:.1f}% | LR: {cur_lr:.6f}{best_marker}")
        
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_macro_f1": tr_f1,
            "val_loss": val_metrics["loss"],
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_unknown_fa_rate": val_unk_fa,
            "is_best": is_best
        })
        
    print(f"\nSeed {seed} Training Finished! Best Validation Macro-F1: {best_val_macro_f1*100:.2f}% at Epoch {best_epoch}")
    print(f"Saved best checkpoint to: {best_ckpt_path}")
    
    # Save log
    log_path = os.path.join(LOGS_DIR, f"training_log_seed{seed}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "epoch_logs": epoch_logs
        }, f, indent=2)
        
    return best_ckpt_path, best_val_macro_f1, best_epoch

def run_all_seeds():
    cfg_path = os.path.join(EXP_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    summary = {}
    for seed in config["training"]["seeds"]:
        ckpt, val_f1, ep = train_model(seed, config)
        summary[f"seed{seed}"] = {
            "checkpoint": os.path.relpath(ckpt, BASE_DIR).replace("\\", "/"),
            "best_epoch": ep,
            "val_macro_f1": val_f1
        }
        
    summary_path = os.path.join(EXP_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved overall training summary to: {summary_path}")

if __name__ == "__main__":
    run_all_seeds()
