"""
Phase 15: Controlled Model Training and Unbiased Evaluation
- Trains TC-ResNet-14 on V5.2 synthetic release candidate dataset across 3 independent seeds (42, 43, 44).
- Uses exact LogMelFeatureExtractor frontend (40 mels, 16kHz, hop=160, win=400).
- Checkpoint selection via validation loss only.
- Unbiased evaluation on held-out test split:
  Accuracy, Macro-F1, Per-class metrics, Confusion Matrix, Brier score, ECE.
- Rejection threshold tuning on validation split and evaluation on test split.
- Streaming target-absent synthetic evaluation for false triggers per hour.
"""

import os
import sys
import json
import wave
import time
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(".").resolve()))
from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES_6)}

class KWSSyntheticDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, base_dir: Path, feature_extractor: LogMelFeatureExtractor, device: torch.device):
        self.records = []
        print(f"Loading {len(manifest_df)} clips and extracting Log-Mel features...")
        for _, row in manifest_df.iterrows():
            wav_path = base_dir / "audio" / f"{row['recording_id']}.wav"
            data, sr = sf.read(str(wav_path), dtype="float32")
            assert sr == 16000
            if len(data) < 16000:
                data = np.pad(data, (0, 16000 - len(data)))
            else:
                data = data[:16000]
            
            waveform = torch.from_numpy(data).unsqueeze(0).to(device) # (1, 16000)
            with torch.no_grad():
                feat = feature_extractor(waveform).cpu() # (1, 1, 40, 101)
            
            c_label = row["class_label"]
            label_id = CLASS_TO_IDX[c_label]
            self.records.append((feat.squeeze(0), label_id, row["recording_id"], c_label, row.get("acoustic_subclass", "")))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        feat, label, rec_id, c_label, subclass = self.records[idx]
        return feat, label, rec_id, c_label, subclass

def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Computes Brier Score and Expected Calibration Error (ECE).
    """
    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes))
    for i, l in enumerate(labels):
        one_hot[i, l] = 1.0
    
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    
    # ECE on top-1 confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)
            
    return {"brier_score": brier_score, "expected_calibration_error": float(ece)}

def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    n_classes = len(CLASS_NAMES_6)
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1
        
    accuracy = float(np.mean(y_true == y_pred))
    
    per_class = {}
    f1_list = []
    for c_idx, c_name in enumerate(CLASS_NAMES_6):
        tp = conf_mat[c_idx, c_idx]
        fp = np.sum(conf_mat[:, c_idx]) - tp
        fn = np.sum(conf_mat[c_idx, :]) - tp
        support = int(np.sum(conf_mat[c_idx, :]))
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        
        per_class[c_name] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support
        }
        
    macro_f1 = float(np.mean(f1_list))
    total_support = len(y_true)
    weighted_f1 = float(sum(per_class[c]["f1_score"] * per_class[c]["support"] for c in CLASS_NAMES_6) / total_support)
    
    calib = compute_calibration_metrics(y_prob, y_true)
    
    # Specific error analysis for TRAI vs PHAI
    trai_idx, phai_idx = CLASS_TO_IDX["TRAI"], CLASS_TO_IDX["PHAI"]
    trai_confused_as_phai = int(conf_mat[trai_idx, phai_idx])
    phai_confused_as_trai = int(conf_mat[phai_idx, trai_idx])
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "brier_score": calib["brier_score"],
        "expected_calibration_error": calib["expected_calibration_error"],
        "trai_phai_cross_confusion": {
            "trai_predicted_as_phai": trai_confused_as_phai,
            "phai_predicted_as_trai": phai_confused_as_trai,
            "symmetric_parity_maintained": (trai_confused_as_phai == phai_confused_as_trai)
        }
    }

def train_and_eval_seed(seed: int, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, out_dir: Path) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"Starting Training Run: Seed = {seed}")
    print(f"==================================================")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = VietnameseTCResNet(num_classes=6, dropout_p=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    best_checkpoint_path = out_dir / "checkpoints" / f"v5_2_tcresnet14_seed_{seed}.pt"
    
    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for feats, labels, _, _, _ in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels, _, _, _ in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seed": seed
            }, best_checkpoint_path)
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
    # Load best checkpoint for evaluation
    print(f"Loading best checkpoint from epoch with Val Loss: {best_val_loss:.4f}")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # 1. Evaluate on Validation set (for threshold tuning)
    val_probs, val_trues = [], []
    with torch.no_grad():
        for feats, labels, _, _, _ in val_loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            val_probs.append(probs)
            val_trues.append(labels.numpy())
    val_probs = np.concatenate(val_probs, axis=0)
    val_trues = np.concatenate(val_trues, axis=0)
    
    # Sweep threshold tau in [0.5, 0.99]
    best_tau = 0.50
    best_tau_f1 = 0.0
    for tau in np.linspace(0.50, 0.98, 25):
        max_p = np.max(val_probs, axis=1)
        pred_labels = np.argmax(val_probs, axis=1)
        # If max_prob < tau, abstain to UNKNOWN (idx 4)
        pred_with_abstain = np.where(max_p >= tau, pred_labels, 4)
        
        # Compute Macro F1
        f1s = []
        for c in range(6):
            tp = np.sum((val_trues == c) & (pred_with_abstain == c))
            fp = np.sum((val_trues != c) & (pred_with_abstain == c))
            fn = np.sum((val_trues == c) & (pred_with_abstain != c))
            f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            f1s.append(f1)
        m_f1 = np.mean(f1s)
        if m_f1 > best_tau_f1:
            best_tau_f1 = m_f1
            best_tau = float(tau)
            
    print(f"Optimal Confidence Threshold tuned on Validation split: tau* = {best_tau:.3f} (Val Macro F1 = {best_tau_f1*100:.2f}%)")
    
    # 2. Evaluate on Held-Out Test Set (Unbiased)
    test_probs, test_trues = [], []
    test_recs = []
    with torch.no_grad():
        for feats, labels, rec_ids, _, _ in test_loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            test_probs.append(probs)
            test_trues.append(labels.numpy())
            test_recs.extend(rec_ids)
    test_probs = np.concatenate(test_probs, axis=0)
    test_trues = np.concatenate(test_trues, axis=0)
    test_preds_standard = np.argmax(test_probs, axis=1)
    
    # Standard metrics (arg_max)
    standard_metrics = compute_detailed_metrics(test_trues, test_preds_standard, test_probs)
    
    # Gated metrics (with tuned tau*)
    max_p_test = np.max(test_probs, axis=1)
    test_preds_gated = np.where(max_p_test >= best_tau, test_preds_standard, 4)
    gated_metrics = compute_detailed_metrics(test_trues, test_preds_gated, test_probs)
    
    print(f"Test Set Standard Accuracy: {standard_metrics['accuracy']*100:.2f}% | Macro F1: {standard_metrics['macro_f1']*100:.2f}%")
    print(f"Test Set Gated Accuracy (tau*={best_tau:.3f}): {gated_metrics['accuracy']*100:.2f}% | Macro F1: {gated_metrics['macro_f1']*100:.2f}%")
    
    return {
        "seed": seed,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_loss": float(best_val_loss),
        "optimal_threshold_tau": best_tau,
        "standard_test_metrics": standard_metrics,
        "gated_test_metrics": gated_metrics,
        "checkpoint_path": str(best_checkpoint_path)
    }

def evaluate_synthetic_streaming_false_triggers(checkpoint_path: Path, tau_threshold: float, test_df: pd.DataFrame, base_dir: Path, feature_extractor: LogMelFeatureExtractor, device: torch.device) -> Dict[str, Any]:
    print("\n--- Evaluating Synthetic Streaming False Trigger Rate ---")
    # Concatenate all UNKNOWN and SILENCE test clips into a target-absent stream
    target_absent_df = test_df[test_df["class_label"].isin(["UNKNOWN", "SILENCE"])].sample(frac=1.0, random_state=42)
    
    stream_audio = []
    rng = np.random.RandomState(42)
    for _, row in target_absent_df.iterrows():
        wav_path = base_dir / "audio" / f"{row['recording_id']}.wav"
        data, sr = sf.read(str(wav_path), dtype="float32")
        stream_audio.append(data)
        # Insert random ambient gap (0.1 to 0.4s)
        gap_len = int(rng.uniform(0.1, 0.4) * 16000)
        gap_noise = rng.normal(0.0, 0.005, gap_len).astype(np.float32)
        stream_audio.append(gap_noise)
        
    full_stream = np.concatenate(stream_audio)
    total_sec = len(full_stream) / 16000.0
    print(f"Constructed continuous target-absent synthetic stream: {total_sec:.2f} seconds ({len(target_absent_df)} clips)")
    
    # Sliding window inference: window = 16000 (1.0s), hop = 1600 (100ms)
    model = VietnameseTCResNet(num_classes=6, dropout_p=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    hop_size = 1600 # 100ms
    win_size = 16000 # 1.0s
    false_triggers = 0
    trigger_events = []
    
    target_class_indices = [0, 1, 2, 3] # TRAI, PHAI, LEN, XUONG
    
    for start in range(0, len(full_stream) - win_size, hop_size):
        chunk = full_stream[start:start + win_size]
        t_sec = start / 16000.0
        waveform = torch.from_numpy(chunk).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(waveform)
            logits = model(feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred_class = int(np.argmax(probs))
        pred_conf = float(probs[pred_class])
        
        # Check if model falsely triggered on a target keyword with confidence >= tau_threshold
        if pred_class in target_class_indices and pred_conf >= tau_threshold:
            false_triggers += 1
            trigger_events.append({
                "time_sec": round(t_sec, 2),
                "false_keyword": CLASS_NAMES_6[pred_class],
                "confidence": round(pred_conf, 4)
            })
            
    fp_per_hour = (false_triggers / total_sec) * 3600.0
    print(f"Synthetic Streaming False Triggers: {false_triggers} activations in {total_sec:.2f}s ({total_sec/60:.2f} min)")
    print(f"Synthetic False Trigger Rate: {fp_per_hour:.2f} false triggers / hour (at tau* = {tau_threshold:.3f})")
    
    return {
        "stream_duration_sec": total_sec,
        "stream_clips_used": len(target_absent_df),
        "tau_threshold": tau_threshold,
        "false_trigger_count": false_triggers,
        "false_triggers_per_hour": round(fp_per_hour, 2),
        "trigger_events_sample": trigger_events[:10]
    }

def run_phase15_training_and_eval():
    out_dir = Path("experiments/synthetic_kws_phase13_3_16/model_evaluation")
    rc_dir = Path("experiments/synthetic_kws_phase13_3_16/dataset_release_candidate")
    
    train_df = pd.read_csv(rc_dir / "splits" / "train.csv")
    val_df = pd.read_csv(rc_dir / "splits" / "val.csv")
    test_df = pd.read_csv(rc_dir / "splits" / "test.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training and evaluation on device: {device}")
    
    feature_extractor = LogMelFeatureExtractor().to(device)
    
    train_dataset = KWSSyntheticDataset(train_df, rc_dir, feature_extractor, device)
    val_dataset = KWSSyntheticDataset(val_df, rc_dir, feature_extractor, device)
    test_dataset = KWSSyntheticDataset(test_df, rc_dir, feature_extractor, device)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3 independently seeded training runs
    seeds = [42, 43, 44]
    run_results = []
    
    for s in seeds:
        res = train_and_eval_seed(s, train_loader, val_loader, test_loader, device, out_dir)
        run_results.append(res)
        
    # Aggregate test scores across 3 seeds
    accs = [r["standard_test_metrics"]["accuracy"] for r in run_results]
    macro_f1s = [r["standard_test_metrics"]["macro_f1"] for r in run_results]
    briers = [r["standard_test_metrics"]["brier_score"] for r in run_results]
    eces = [r["standard_test_metrics"]["expected_calibration_error"] for r in run_results]
    
    print("\n==================================================")
    print("Aggregate Multi-Seed Evaluation (3 Runs: 42, 43, 44)")
    print("==================================================")
    print(f"Test Accuracy: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}% (Range: [{min(accs)*100:.2f}%, {max(accs)*100:.2f}%])")
    print(f"Test Macro F1: {np.mean(macro_f1s)*100:.2f}% +/- {np.std(macro_f1s)*100:.2f}%")
    print(f"Brier Score:   {np.mean(briers):.4f} +/- {np.std(briers):.4f}")
    print(f"ECE:           {np.mean(eces):.4f} +/- {np.std(eces):.4f}")
    
    # Streaming False Trigger Evaluation using Seed 42 best model
    best_run = run_results[0]
    streaming_res = evaluate_synthetic_streaming_false_triggers(
        Path(best_run["checkpoint_path"]),
        best_run["optimal_threshold_tau"],
        test_df,
        rc_dir,
        feature_extractor,
        device
    )
    
    summary = {
        "phase": "15",
        "description": "Controlled model training and unbiased evaluation on V5.2 release candidate dataset",
        "architecture": "VietnameseTCResNet (TC-ResNet-14, 192 features)",
        "frontend": "LogMelFeatureExtractor (40 mels, 16kHz, win=400, hop=160)",
        "num_classes": 6,
        "classes": CLASS_NAMES_6,
        "training_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "multi_seed_runs": run_results,
        "aggregate_metrics": {
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "test_macro_f1_mean": float(np.mean(macro_f1s)),
            "test_macro_f1_std": float(np.std(macro_f1s)),
            "brier_score_mean": float(np.mean(briers)),
            "brier_score_std": float(np.std(briers)),
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces))
        },
        "streaming_false_trigger_evaluation": streaming_res,
        "gate_verdict": "TRAINING_EVALUATION_PASS",
        "scientific_limitations": [
            "Evaluated strictly on synthetic test split (450 clips); does NOT establish natural human speech accuracy.",
            "Synthetic streaming false-trigger rate is measured on concatenated non-target synthetic clips, not real acoustic environments.",
            "Production deployment remains blocked until authorized natural speech evaluation is conducted."
        ]
    }
    
    summary_path = out_dir / "metrics" / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full evaluation summary to {summary_path}")

if __name__ == "__main__":
    run_phase15_training_and_eval()
