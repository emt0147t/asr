"""
Phase 29: Six-Class CNN Baseline Test Evaluation Pipeline.
Evaluates validation-selected checkpoints on the held-out six-class test split
(240 clips: 40 TRAI, 40 PHAI, 40 LEN, 40 XUONG, 40 UNKNOWN, 40 SILENCE).

Computes:
  - Overall accuracy, Macro precision, recall, and F1
  - Per-class precision, recall, F1, and support for all 6 classes
  - 6x6 Confusion Matrix with fixed canonical label order
  - Detailed error analysis:
      * Target commands misclassified as UNKNOWN
      * UNKNOWN speech misclassified as Target commands
      * SILENCE misclassified as Target commands
      * Target commands misclassified as SILENCE
  - SILENCE diagnostic performance across the 3 out-of-condition test groups
  - Per-sample prediction CSV with softmax probability distributions
  - Inference latency benchmark (CPU single-sample & GPU batched) and model footprint
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
from experiments.vieneu_cnn_6class.train import VieNeuKWS6ClassDataset, LABEL_TO_ID, ID_TO_LABEL, CLASS_NAMES

def evaluate_checkpoint(checkpoint_path: str, test_manifest_path: str = None, output_prefix: str = None):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if test_manifest_path is None:
        test_manifest_path = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_6class/splits/test_manifest.csv")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = ckpt.get("seed", 42)
    val_macro_f1 = ckpt.get("val_macro_f1", 0.0)
    best_epoch = ckpt.get("epoch", 0)

    print("=" * 70)
    print(f"Evaluating 6-Class Model Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Seed: {seed} | Best Epoch: {best_epoch} | Validation Macro-F1: {val_macro_f1*100:.2f}%")
    print("=" * 70)

    # Initialize model
    model = VietnameseKWSNet(num_classes=6)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load held-out test split (240 clips)
    test_ds = VieNeuKWS6ClassDataset(test_manifest_path, is_train=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"Loaded {len(test_ds)} held-out test clips from: {os.path.basename(test_manifest_path)}")

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
    model_cpu = VietnameseKWSNet(num_classes=6)
    model_cpu.load_state_dict(ckpt["model_state_dict"])
    model_cpu.eval()
    cpu_latencies = []
    with torch.no_grad():
        for i in range(min(50, len(test_ds))):
            feat, _ = test_ds[i]
            feat_cpu = feat.unsqueeze(0)  # (1, 1, 40, 101)
            t0 = time.perf_counter()
            _ = model_cpu(feat_cpu)
            t1 = time.perf_counter()
            cpu_latencies.append((t1 - t0) * 1000.0)

    cpu_latency_mean_ms = float(np.mean(cpu_latencies[5:]))  # warm-up skipped
    cpu_latency_std_ms = float(np.std(cpu_latencies[5:]))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    test_acc = accuracy_score(all_targets, all_preds)
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4, 5])

    # Per-class metrics
    per_class_data = {}
    for idx, cname in enumerate(CLASS_NAMES):
        c_mask = (all_targets == idx)
        support = int(np.sum(c_mask))
        p = precision_score(all_targets == idx, all_preds == idx, zero_division=0)
        r = recall_score(all_targets == idx, all_preds == idx, zero_division=0)
        f = f1_score(all_targets == idx, all_preds == idx, zero_division=0)
        per_class_data[cname] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": support
        }

    print("\n--- SIX-CLASS TEST METRICS SUMMARY ---")
    print(f"Test Accuracy:    {test_acc*100:.2f}% ({np.sum(all_preds == all_targets)}/{len(test_ds)})")
    print(f"Macro Precision:  {macro_p*100:.2f}%")
    print(f"Macro Recall:     {macro_r*100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1*100:.2f}%")

    print("\nPer-Class Breakdown:")
    for cname in CLASS_NAMES:
        d = per_class_data[cname]
        print(f"  {cname:7s} | P: {d['precision']*100:5.1f}% | R: {d['recall']*100:5.1f}% | F1: {d['f1']*100:5.1f}% | Support: {d['support']}")

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    header = "          " + "".join([f"{c:>8s}" for c in CLASS_NAMES])
    print(header)
    for r_idx, r_name in enumerate(CLASS_NAMES):
        row_str = f"{r_name:8s}  " + "".join([f"{cm[r_idx, c_idx]:8d}" for c_idx in range(6)])
        print(row_str)

    # Save per-sample predictions
    predictions_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    prefix = output_prefix or f"seed{seed}"
    pred_csv_path = os.path.join(predictions_dir, f"test_predictions_{prefix}.csv")
    metrics_json_path = os.path.join(predictions_dir, f"test_metrics_{prefix}.json")

    test_df = pd.read_csv(test_manifest_path)
    records = []
    incorrect_records = []

    target_as_unknown = []
    unknown_as_target = []
    silence_as_target = []
    target_as_silence = []

    TARGET_INDICES = {0, 1, 2, 3}

    for i in range(len(test_ds)):
        sid = test_df.iloc[i]["sample_id"]
        true_lbl = ID_TO_LABEL[all_targets[i]]
        pred_lbl = ID_TO_LABEL[all_preds[i]]
        conf = float(all_probs[i, all_preds[i]])
        correct = bool(all_targets[i] == all_preds[i])

        rec = {
            "sample_id": sid,
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "confidence": conf,
            "prob_TRAI": float(all_probs[i, 0]),
            "prob_PHAI": float(all_probs[i, 1]),
            "prob_LEN": float(all_probs[i, 2]),
            "prob_XUONG": float(all_probs[i, 3]),
            "prob_UNKNOWN": float(all_probs[i, 4]),
            "prob_SILENCE": float(all_probs[i, 5]),
            "correct": correct,
            "voice_id": test_df.iloc[i]["voice_id"],
            "model_condition": test_df.iloc[i].get("model_checkpoint", "standard")
        }
        records.append(rec)

        if not correct:
            incorrect_records.append(rec)
            t_idx = all_targets[i]
            p_idx = all_preds[i]
            if t_idx in TARGET_INDICES and p_idx == 4:
                target_as_unknown.append(rec)
            elif t_idx == 4 and p_idx in TARGET_INDICES:
                unknown_as_target.append(rec)
            elif t_idx == 5 and p_idx in TARGET_INDICES:
                silence_as_target.append(rec)
            elif t_idx in TARGET_INDICES and p_idx == 5:
                target_as_silence.append(rec)

    pd.DataFrame(records).to_csv(pred_csv_path, index=False, encoding="utf-8")

    # Diagnostic on SILENCE out-of-condition breakdown
    silence_sub_df = pd.DataFrame(records)[pd.DataFrame(records)["true_label"] == "SILENCE"]
    silence_conditions = silence_sub_df["model_condition"].value_counts().to_dict()
    silence_cond_acc = {}
    for cond in silence_conditions.keys():
        sub_c = silence_sub_df[silence_sub_df["model_condition"] == cond]
        c_acc = (sub_c["correct"]).mean()
        silence_cond_acc[cond] = {
            "total": len(sub_c),
            "correct": int(sub_c["correct"].sum()),
            "accuracy": float(c_acc)
        }

    param_count = get_model_param_count(model)
    ckpt_size_kb = os.path.getsize(checkpoint_path) / 1024.0

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
        "silence_out_of_condition_diagnostics": silence_cond_acc,
        "error_categories": {
            "target_misclassified_as_unknown": len(target_as_unknown),
            "unknown_misclassified_as_target": len(unknown_as_target),
            "silence_misclassified_as_target": len(silence_as_target),
            "target_misclassified_as_silence": len(target_as_silence)
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
    print(f"  Parameters:  {param_count:,} ({ckpt_size_kb:.1f} KB)")
    print(f"  CPU Latency: {cpu_latency_mean_ms:.2f} ± {cpu_latency_std_ms:.2f} ms")

    print("\nSILENCE Diagnostic Performance (Out-of-Condition Analysis):")
    for cond, st in silence_cond_acc.items():
        print(f"  - Condition '{cond}': {st['correct']}/{st['total']} correct ({st['accuracy']*100:.1f}%)")

    if incorrect_records:
        print(f"\nTotal Misclassified Samples ({len(incorrect_records)}):")
        for err in incorrect_records:
            print(f"  - {err['sample_id']} (Voice: {err['voice_id']}): True={err['true_label']}, Pred={err['predicted_label']} (Conf={err['confidence']:.3f})")
    else:
        print("\nZero misclassified samples on test set (100% test accuracy).")

    return metrics_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 6-class CNN baseline checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output prediction/metrics files")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, output_prefix=args.output_prefix)
