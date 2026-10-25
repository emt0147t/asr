"""
Phase 26: Four-Class CNN Baseline Test Evaluation Pipeline.
Evaluates validation-selected checkpoints on the held-out test split (160 synthetic clips,
4 disjoint preset voices: Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí).

Computes:
  - Overall accuracy, Macro precision, recall, and F1
  - Per-class precision, recall, F1
  - 4x4 Confusion Matrix with named axes
  - Per-sample prediction CSV with softmax probabilities
  - Detailed error analysis of any misclassifications
  - Inference latency benchmark (CPU & GPU) and model footprint
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_cnn_baseline.train import VieNeuKWSDataset, LABEL_TO_ID, ID_TO_LABEL, CLASS_NAMES

def evaluate_checkpoint(checkpoint_path: str, test_manifest_path: str = None, output_prefix: str = None):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if test_manifest_path is None:
        test_manifest_path = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000/splits/test_manifest.csv")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = ckpt.get("seed", 42)
    val_macro_f1 = ckpt.get("val_macro_f1", 0.0)
    best_epoch = ckpt.get("epoch", 0)

    print("=" * 70)
    print(f"Evaluating Model Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Seed: {seed} | Best Epoch: {best_epoch} | Validation Macro-F1: {val_macro_f1*100:.2f}%")
    print("=" * 70)

    # Initialize model
    model = VietnameseKWSNet(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset (strictly unaugmented)
    test_ds = VieNeuKWSDataset(test_manifest_path, is_train=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"Loaded {len(test_ds)} held-out test clips from: {os.path.basename(test_manifest_path)}")

    # Inference loop
    all_preds = []
    all_targets = []
    all_probs = []

    t_start = time.perf_counter()
    with torch.no_grad():
        for feats, targets in test_loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    inference_time_total = time.perf_counter() - t_start
    latency_per_sample_gpu_ms = (inference_time_total / len(test_ds)) * 1000.0

    # CPU Latency Benchmark (Batch size = 1)
    model_cpu = VietnameseKWSNet(num_classes=4)
    model_cpu.load_state_dict(ckpt["model_state_dict"])
    model_cpu.eval()
    cpu_latencies = []
    with torch.no_grad():
        for i in range(min(40, len(test_ds))):
            feat, _ = test_ds[i]
            feat_cpu = feat.unsqueeze(0)  # (1, 1, 40, 101)
            t0 = time.perf_counter()
            _ = model_cpu(feat_cpu)
            t1 = time.perf_counter()
            cpu_latencies.append((t1 - t0) * 1000.0)

    cpu_latency_mean_ms = float(np.mean(cpu_latencies[5:]))  # warm-up skipped
    cpu_latency_std_ms = float(np.std(cpu_latencies[5:]))

    # Calculate overall metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    test_acc = accuracy_score(all_targets, all_preds)
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # Per-class metrics
    class_p = precision_score(all_targets, all_preds, average=None, zero_division=0)
    class_r = recall_score(all_targets, all_preds, average=None, zero_division=0)
    class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])

    print("\n--- TEST METRICS SUMMARY ---")
    print(f"Test Accuracy:    {test_acc*100:.2f}% ({np.sum(all_preds == all_targets)}/{len(all_targets)})")
    print(f"Macro Precision:  {macro_p*100:.2f}%")
    print(f"Macro Recall:     {macro_r*100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1*100:.2f}%")

    print("\nPer-Class Breakdown:")
    per_class_data = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class_data[name] = {
            "precision": float(class_p[i]),
            "recall": float(class_r[i]),
            "f1_score": float(class_f1[i]),
            "support": int(np.sum(all_targets == i))
        }
        print(f"  {name:6s} | P: {class_p[i]*100:5.1f}% | R: {class_r[i]*100:5.1f}% | F1: {class_f1[i]*100:5.1f}% | Support: {np.sum(all_targets == i)}")

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    header_str = "        " + "  ".join(f"{c:>6s}" for c in CLASS_NAMES)
    print(header_str)
    for i, true_name in enumerate(CLASS_NAMES):
        row_str = f"{true_name:6s}  " + "  ".join(f"{cm[i, j]:6d}" for j in range(4))
        print(row_str)

    # Model footprint
    param_count = get_model_param_count(model)
    ckpt_size_kb = os.path.getsize(checkpoint_path) / 1024.0

    print(f"\nModel Complexity:")
    print(f"  Trainable Parameters: {param_count:,}")
    print(f"  Checkpoint Size:      {ckpt_size_kb:.1f} KB")
    print(f"  CPU Latency (single): {cpu_latency_mean_ms:.2f} ± {cpu_latency_std_ms:.2f} ms")

    # Build per-sample predictions DataFrame
    test_df = pd.read_csv(test_manifest_path)
    pred_records = []
    incorrect_records = []

    for i in range(len(test_df)):
        sid = test_ds.sample_ids[i]
        true_lbl = ID_TO_LABEL[all_targets[i]]
        pred_lbl = ID_TO_LABEL[all_preds[i]]
        is_corr = bool(all_targets[i] == all_preds[i])

        rec = {
            "sample_id": sid,
            "voice_id": test_df.iloc[i]["voice_id"],
            "voice_gender": test_df.iloc[i]["voice_gender"],
            "voice_region": test_df.iloc[i]["voice_region"],
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "correct": is_corr,
            "prob_TRAI": float(all_probs[i, 0]),
            "prob_PHAI": float(all_probs[i, 1]),
            "prob_LEN": float(all_probs[i, 2]),
            "prob_XUONG": float(all_probs[i, 3]),
            "confidence": float(np.max(all_probs[i]))
        }
        pred_records.append(rec)
        if not is_corr:
            incorrect_records.append(rec)

    pred_df = pd.DataFrame(pred_records)

    # Save outputs
    predictions_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    prefix = output_prefix or f"seed{seed}"
    pred_csv_path = os.path.join(predictions_dir, f"test_predictions_{prefix}.csv")
    metrics_json_path = os.path.join(predictions_dir, f"test_metrics_{prefix}.json")

    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8")

    metrics_out = {
        "checkpoint": os.path.basename(checkpoint_path),
        "seed": seed,
        "best_epoch": best_epoch,
        "val_macro_f1": float(val_macro_f1),
        "test_accuracy": float(test_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": per_class_data,
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": cm.tolist()
        },
        "model_parameters": param_count,
        "checkpoint_size_kb": ckpt_size_kb,
        "latency_ms": {
            "cpu_single_mean": cpu_latency_mean_ms,
            "cpu_single_std": cpu_latency_std_ms,
            "gpu_batched_mean": latency_per_sample_gpu_ms
        },
        "total_test_samples": len(test_ds),
        "incorrect_count": len(incorrect_records),
        "incorrect_samples": incorrect_records
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    print(f"\nOutputs saved:")
    print(f"  Predictions: {pred_csv_path}")
    print(f"  Metrics:     {metrics_json_path}")
    if incorrect_records:
        print(f"\nMisclassified Samples ({len(incorrect_records)}):")
        for err in incorrect_records:
            print(f"  - {err['sample_id']} ({err['voice_id']}, {err['voice_region']}): True={err['true_label']}, Pred={err['predicted_label']} (Conf={err['confidence']:.3f})")
    else:
        print("\nZero misclassified samples on test set (100% test accuracy).")

    return metrics_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 4-class CNN baseline checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output prediction/metrics files")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, output_prefix=args.output_prefix)
