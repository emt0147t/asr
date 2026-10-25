"""
Training Pipeline for Synthetic Vietnamese KWS V3.
Enforces:
  - Training strictly on train_manifest_6class.csv (14 speakers, 952 rows)
  - Model selection strictly on val_manifest_6class.csv (3 speakers, 216 rows)
  - Training-only augmentations:
      * Resampling pitch/speed perturbation (scaling F0 and formants by 0.88 - 1.14)
      * Temporal shift (+- 50 ms)
      * Amplitude scaling (0.85 - 1.15)
      * SpecAugment (freq and time masking)
  - Zero holdout tuning: Historical and fresh holdouts are NOT accessed during training.
"""

import sys
import os
import time
import json
import random
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, f1_score, accuracy_score

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet, VietnameseTemporalTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from src.acquisition.campaign_validator import validate_split_manifests_preflight

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
CLASS_ID_MAP_6 = {name: i for i, name in enumerate(CLASS_NAMES_6)}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Synthetic6ClassDatasetV3(Dataset):
    """
    Dataset for 6-class synthetic keyword spotting V3.
    Applies training-only pitch/speed perturbation, SpecAugment, and temporal jittering.
    """
    def __init__(self, manifest_csv: Path, is_train: bool = True, use_resample: bool = True, use_specaugment: bool = True):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.use_resample = use_resample and is_train
        self.use_specaugment = use_specaugment and is_train
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
        
        # Preload into memory
        self.audio_cache = []
        for idx, row in self.df.iterrows():
            wav_path = Path(row["processed_path"])
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            elif len(audio) > 16000:
                audio = audio[:16000]
            self.audio_cache.append(audio)

    def __len__(self):
        return len(self.df)

    def _apply_spec_augment(self, log_mel: torch.Tensor) -> torch.Tensor:
        # Frequency masking (1 to 4 bands)
        f_len = random.randint(1, 4)
        f0 = random.randint(0, 40 - f_len)
        log_mel[:, f0:f0 + f_len, :] = 0.0

        # Time masking (1 to 6 frames)
        t_len = random.randint(1, 6)
        t0 = random.randint(0, 101 - t_len)
        log_mel[:, :, t0:t0 + t_len] = 0.0
        return log_mel

    def __getitem__(self, idx):
        audio = self.audio_cache[idx].copy()
        label = int(self.df.iloc[idx]["class_id"])
        
        if self.is_train:
            # 1. Pitch / Speed perturbation (resampling)
            # Only apply to speech classes (0..4), not pure silence (5)
            if self.use_resample and label != 5 and random.random() < 0.6:
                rate = random.uniform(0.88, 1.14)
                n_new = int(len(audio) * rate)
                res = signal.resample(audio, n_new)
                if len(res) < 16000:
                    audio = np.pad(res, (0, 16000 - len(res)))
                else:
                    audio = res[:16000]

            # 2. Subtle temporal jitter (+- 50 ms)
            shift = random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]

            # 3. Volume scale
            scale = random.uniform(0.85, 1.15)
            audio = np.clip(audio * scale, -1.0, 1.0)

        audio_tensor = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel = self.extractor(audio_tensor)  # (1, 40, 101)

        if self.use_specaugment and random.random() < 0.5:
            log_mel = self._apply_spec_augment(log_mel)

        return log_mel.squeeze(0), label


def train_model(
    model_class,
    model_name: str,
    train_csv: Path,
    val_csv: Path,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    ckpt_path: Path = Path("best.pt"),
    use_resample: bool = True,
    use_specaugment: bool = True
):
    train_ds = Synthetic6ClassDatasetV3(train_csv, is_train=True, use_resample=use_resample, use_specaugment=use_specaugment)
    val_ds = Synthetic6ClassDatasetV3(val_csv, is_train=False, use_resample=False, use_specaugment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = model_class(num_classes=6, dropout_p=0.25).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=======================================================")
    print(f"Training {model_name} ({param_count:,} params)")
    print(f"Augmentations: resample={use_resample}, specaugment={use_specaugment}")
    print(f"=======================================================")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = -1.0
    best_val_metrics = {}
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        all_tr_preds, all_tr_targets = [], []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item() * len(y)
            all_tr_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_tr_targets.extend(y.cpu().numpy())
            
        scheduler.step()
        tr_loss /= len(train_ds)
        tr_acc = accuracy_score(all_tr_targets, all_tr_preds)
        
        # Validation on unseen speakers SPK_CALIB_015..017
        model.eval()
        va_loss = 0.0
        all_va_preds, all_va_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item() * len(y)
                all_va_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_va_targets.extend(y.cpu().numpy())
                
        va_loss /= len(val_ds)
        va_acc = accuracy_score(all_va_targets, all_va_preds)
        va_f1 = f1_score(all_va_targets, all_va_preds, average="macro", zero_division=0)
        
        # Per class recall
        prec, rec, f1_cls, _ = precision_recall_fscore_support(all_va_targets, all_va_preds, labels=list(range(6)), zero_division=0)
        
        if epoch % 5 == 0 or epoch == epochs or va_f1 > best_val_f1:
            print(f"[{model_name}] Ep {epoch:02d}/{epochs:02d} | Tr Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.3f} F1: {va_f1:.4f} (TRAI: {rec[0]:.2f}, PHAI: {rec[1]:.2f})")
            
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_val_metrics = {
                "epoch": epoch,
                "val_accuracy": va_acc,
                "val_macro_f1": va_f1,
                "val_per_class_recall": rec.tolist(),
                "val_per_class_f1": f1_cls.tolist(),
            }
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_class": model.__class__.__name__,
                "num_classes": 6,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": va_f1,
                "val_acc": va_acc,
                "best_val_metrics": best_val_metrics,
                "class_names": CLASS_NAMES_6
            }, ckpt_path)
            
    wall_sec = time.time() - t0
    print(f"[{model_name}] Finished in {wall_sec:.2f}s | Best Val Macro-F1: {best_val_f1:.4f} | Saved: {ckpt_path}")
    return model, ckpt_path, best_val_metrics, wall_sec


def run_training():
    v3_dir = Path("experiments/synthetic_kws_v3")
    v3_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = v3_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # We use the vetted manifests from V2 for train & val:
    # Train: 952 rows (SPK_CALIB_001..014)
    # Val: 216 rows (SPK_CALIB_015..017)
    train_src = Path("experiments/synthetic_kws_v2/train_manifest_6class.csv")
    val_src = Path("experiments/synthetic_kws_v2/val_manifest_6class.csv")
    
    train_dst = v3_dir / "train_manifest_6class.csv"
    val_dst = v3_dir / "val_manifest_6class.csv"
    
    df_tr = pd.read_csv(train_src)
    df_va = pd.read_csv(val_src)
    df_tr.to_csv(train_dst, index=False)
    df_va.to_csv(val_dst, index=False)
    
    print(f"Copied V3 manifests:")
    print(f"  Train: {train_dst} ({len(df_tr)} rows, {df_tr['speaker_id'].nunique()} speakers)")
    print(f"  Val:   {val_dst} ({len(df_va)} rows, {df_va['speaker_id'].nunique()} speakers)")
    
    device = torch.device("cpu")
    
    # Model 1: VietnameseTCResNet (standard TC-ResNet-14 with pitch perturbation)
    set_seed(42)
    ckpt1 = ckpt_dir / "best_tcresnet14_v3.pt"
    m1, _, val_m1, wall1 = train_model(
        VietnameseTCResNet, "TC-ResNet-14-V3", train_dst, val_dst,
        epochs=25, batch_size=32, lr=1e-3, device=device, ckpt_path=ckpt1,
        use_resample=True, use_specaugment=True
    )
    
    # Model 2: VietnameseTemporalTCResNet (Temporal split pooling TC-ResNet-14 with pitch perturbation)
    set_seed(42)
    ckpt2 = ckpt_dir / "best_temporal_tcresnet14_v3.pt"
    m2, _, val_m2, wall2 = train_model(
        VietnameseTemporalTCResNet, "Temporal-TC-ResNet-14-V3", train_dst, val_dst,
        epochs=25, batch_size=32, lr=1e-3, device=device, ckpt_path=ckpt2,
        use_resample=True, use_specaugment=True
    )
    
    summary = {
        "model_comparison_on_validation": {
            "tcresnet14_v3": {
                "checkpoint": str(ckpt1),
                "val_macro_f1": val_m1["val_macro_f1"],
                "val_accuracy": val_m1["val_accuracy"],
                "trai_val_recall": val_m1["val_per_class_recall"][0],
                "phai_val_recall": val_m1["val_per_class_recall"][1],
                "wall_time_sec": wall1
            },
            "temporal_tcresnet14_v3": {
                "checkpoint": str(ckpt2),
                "val_macro_f1": val_m2["val_macro_f1"],
                "val_accuracy": val_m2["val_accuracy"],
                "trai_val_recall": val_m2["val_per_class_recall"][0],
                "phai_val_recall": val_m2["val_per_class_recall"][1],
                "wall_time_sec": wall2
            }
        }
    }
    
    with open(v3_dir / "validation_model_selection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nModel selection summary written to {v3_dir / 'validation_model_selection.json'}")


if __name__ == "__main__":
    run_training()
