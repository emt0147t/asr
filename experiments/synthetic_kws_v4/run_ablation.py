"""
Controlled Ablation Experiment for Synthetic Vietnamese KWS V4.
Compares 3 arms under identical architecture (TC-ResNet-14), optimizer, seeds, and epochs:
  - Arm A (Control): Existing Generator + V3 Augmentation (Resampling + SpecAugment)
  - Arm B: Onset-Aware Generator + Standard Augmentation (NO Resampling, SpecAugment only)
  - Arm C: Onset-Aware Generator + V3 Augmentation (Onset-aware + Resampling + SpecAugment)
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

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from src.acquisition.campaign_validator import validate_split_manifests_preflight

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AblationDataset(Dataset):
    def __init__(self, manifest_csv: Path, is_train: bool = True, use_resample: bool = True, use_specaugment: bool = True):
        self.df = pd.read_csv(manifest_csv)
        self.is_train = is_train
        self.use_resample = use_resample and is_train
        self.use_specaugment = use_specaugment and is_train
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
        
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
        f_len = random.randint(1, 4)
        f0 = random.randint(0, 40 - f_len)
        log_mel[:, f0:f0 + f_len, :] = 0.0

        t_len = random.randint(1, 6)
        t0 = random.randint(0, 101 - t_len)
        log_mel[:, :, t0:t0 + t_len] = 0.0
        return log_mel

    def __getitem__(self, idx):
        audio = self.audio_cache[idx].copy()
        label = int(self.df.iloc[idx]["class_id"])
        
        if self.is_train:
            if self.use_resample and label != 5 and random.random() < 0.6:
                rate = random.uniform(0.88, 1.14)
                n_new = int(len(audio) * rate)
                res = signal.resample(audio, n_new)
                if len(res) < 16000:
                    audio = np.pad(res, (0, 16000 - len(res)))
                else:
                    audio = res[:16000]

            shift = random.randint(-800, 800)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:16000]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:16000 - shift]

            scale = random.uniform(0.85, 1.15)
            audio = np.clip(audio * scale, -1.0, 1.0)

        audio_tensor = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel = self.extractor(audio_tensor)

        if self.use_specaugment and random.random() < 0.5:
            log_mel = self._apply_spec_augment(log_mel)

        return log_mel.squeeze(0), label


def train_single_arm(
    arm_name: str,
    train_csv: Path,
    val_csv: Path,
    use_resample: bool,
    ckpt_path: Path,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
):
    set_seed(42)
    train_ds = AblationDataset(train_csv, is_train=True, use_resample=use_resample, use_specaugment=True)
    val_ds = AblationDataset(val_csv, is_train=False, use_resample=False, use_specaugment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = VietnameseTCResNet(num_classes=6, dropout_p=0.25).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = -1.0
    best_metrics = {}
    t0 = time.time()
    
    print(f"\n=================================================================")
    print(f"TRAINING {arm_name} | use_resample={use_resample}")
    print(f"Train Manifest: {train_csv.name} | Val Manifest: {val_csv.name}")
    print(f"=================================================================")
    
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
        
        # Validation
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
        prec, rec, f1_cls, _ = precision_recall_fscore_support(all_va_targets, all_va_preds, labels=list(range(6)), zero_division=0)
        
        if epoch % 5 == 0 or epoch == epochs or va_f1 > best_val_f1:
            print(f"[{arm_name}] Ep {epoch:02d}/{epochs:02d} | Tr Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.3f} F1: {va_f1:.4f} (TRAI: {rec[0]:.2f}, PHAI: {rec[1]:.2f})")
            
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_metrics = {
                "arm_name": arm_name,
                "best_epoch": epoch,
                "val_accuracy": float(va_acc),
                "val_macro_f1": float(va_f1),
                "val_precision": [float(p) for p in prec],
                "val_recall": [float(r) for r in rec],
                "val_f1": [float(f) for f in f1_cls],
                "trai_val_recall": float(rec[0]),
                "phai_val_recall": float(rec[1]),
                "use_resample": use_resample,
                "checkpoint": str(ckpt_path)
            }
            torch.save({
                "arm_name": arm_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": 6,
                "best_metrics": best_metrics
            }, ckpt_path)
            
    wall_sec = time.time() - t0
    best_metrics["wall_time_sec"] = wall_sec
    print(f"[{arm_name}] Done in {wall_sec:.2f}s | Best Val Macro-F1: {best_val_f1:.4f} | Saved: {ckpt_path}")
    return model, ckpt_path, best_metrics


def run_controlled_ablation():
    base_dir = Path("experiments/synthetic_kws_v4")
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Manifest paths
    tr_ctrl = Path("experiments/synthetic_kws_v3/train_manifest_6class.csv")
    va_ctrl = Path("experiments/synthetic_kws_v3/val_manifest_6class.csv")
    
    tr_onset = base_dir / "train_manifest_onset_aware.csv"
    va_onset = base_dir / "val_manifest_onset_aware.csv"
    
    assert tr_ctrl.exists() and va_ctrl.exists(), "Control manifests missing"
    assert tr_onset.exists() and va_onset.exists(), "Onset-aware manifests missing. Run build_onset_aware_data.py first"
    
    device = torch.device("cpu")
    ablation_results = {}
    
    # Arm A: Control (Existing Generator + V3 Augmentation)
    ckpt_a = ckpt_dir / "arm_a_control.pt"
    _, _, res_a = train_single_arm(
        "Arm A (Existing Generator + V3 Augmentation)",
        train_csv=tr_ctrl, val_csv=va_ctrl,
        use_resample=True, ckpt_path=ckpt_a,
        epochs=25, device=device
    )
    ablation_results["Arm A (Control)"] = res_a
    
    # Arm B: Onset-Aware Generator WITHOUT Resample Augmentation
    ckpt_b = ckpt_dir / "arm_b_onset_no_resample.pt"
    _, _, res_b = train_single_arm(
        "Arm B (Onset-Aware Generator + SpecAugment Only)",
        train_csv=tr_onset, val_csv=va_onset,
        use_resample=False, ckpt_path=ckpt_b,
        epochs=25, device=device
    )
    ablation_results["Arm B (Onset-Aware, No Resample)"] = res_b
    
    # Arm C: Onset-Aware Generator WITH Resample Augmentation
    ckpt_c = ckpt_dir / "arm_c_onset_with_resample.pt"
    _, _, res_c = train_single_arm(
        "Arm C (Onset-Aware Generator + V3 Augmentation)",
        train_csv=tr_onset, val_csv=va_onset,
        use_resample=True, ckpt_path=ckpt_c,
        epochs=25, device=device
    )
    ablation_results["Arm C (Onset-Aware + Resample)"] = res_c
    
    out_json = base_dir / "ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAblation completed! Results saved to {out_json}")


if __name__ == "__main__":
    run_controlled_ablation()
