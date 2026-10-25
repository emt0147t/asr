#!/usr/bin/env python3
"""
Phase 64: Forensic Validation Audit & Ground-Truth Metric Reproduction Runner
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_phase64_forensic_validation_audit.py

Executes definitive ground-truth evaluation across all 8 neural checkpoints on:
experiments/vieneu_kws_research_v12/manifests/validation_v1.csv (N=1,280)

Evaluates under both:
1. Operational Decision Rule (theta_cmd = 0.20): max(p[:4]) >= 0.20 -> argmax(p[:4]) else REJECT
2. Standard 5-Class Argmax Rule: argmax(p[:5])
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
MANIFEST_PATH = V12_DIR / "manifests" / "validation_v1.csv"
REPORTS_DIR = V12_DIR / "reports"

sys.path.append(str(V12_DIR / "scripts"))
from train_phase54_d3 import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX

CHECKPOINTS = {
    "Baseline_v11_seed42": PROJECT_ROOT / "experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt",
    "Baseline_E1_seed42": V12_DIR / "checkpoints/exp_v12_e1_baseline_scale_seed42.pt",
    "Phase28_D2_H1_Candidate": V12_DIR / "candidates/phase28_d2_candidate/candidate_checkpoint.pt",
    "Phase26_Candidate_D0_H1_seed44": V12_DIR / "candidates/phase26_candidate_d0_h1_seed44/candidate_checkpoint.pt",
    "Phase28_D0_H1_Candidate": V12_DIR / "candidates/phase28_candidate/candidate_checkpoint.pt",
    "Candidate_D3_A_seed42": V12_DIR / "training_phase54_d3/D3_A_seed42/best_checkpoint.pt",
    "Candidate_D3_B_seed42": V12_DIR / "training_phase54_d3/D3_B_seed42/best_checkpoint.pt",
    "Candidate_D3_C_seed42": V12_DIR / "training_phase54_d3/D3_C_seed42/best_checkpoint.pt"
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    n_total = len(y_true)
    n_correct = int(np.sum(y_true == y_pred))
    acc = float(n_correct / n_total)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    macro_f1_4cmd = float(np.mean(f1[:4]))
    macro_f1_5class = float(np.mean(f1))
    min_cmd_recall = float(np.min(r[:4]))

    # OOV FAR on REJECT class (label 4, support = 640)
    reject_mask = (y_true == 4)
    oov_fa_count = int(np.sum(y_pred[reject_mask] != 4))
    oov_fa_rate = float(oov_fa_count / np.sum(reject_mask))
    reject_precision = float(p[4])
    reject_recall = float(r[4])

    # Command overall accuracy (labels 0-3, support = 640)
    cmd_mask = (y_true < 4)
    cmd_correct = int(np.sum((y_true == y_pred) & cmd_mask))
    cmd_acc = float(cmd_correct / np.sum(cmd_mask))

    return {
        "total_samples": n_total,
        "correct_samples": n_correct,
        "accuracy": acc,
        "macro_f1_4cmd": macro_f1_4cmd,
        "macro_f1_5class": macro_f1_5class,
        "min_command_recall": min_cmd_recall,
        "command_overall_accuracy": cmd_acc,
        "oov_fa_count": oov_fa_count,
        "oov_total": int(np.sum(reject_mask)),
        "oov_fa_rate": oov_fa_rate,
        "reject_precision": reject_precision,
        "reject_recall": reject_recall,
        "confusion_matrix": cm,
        "per_class": {
            CLASSES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i])
            }
            for i in range(5)
        }
    }


def main():
    print("=" * 80)
    print("PHASE 64: FORENSIC VALIDATION AUDIT (N=1,280 ALIGNED SAMPLES)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Check manifest
    if not MANIFEST_PATH.exists():
        print(f"FATAL: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    manifest_sha = compute_sha256(MANIFEST_PATH)
    print(f"Manifest: {MANIFEST_PATH} (SHA-256: {manifest_sha})")
    df_val = pd.read_csv(MANIFEST_PATH)
    print(f"Total rows: {len(df_val)}")
    print("Class distribution:")
    for c in CLASSES:
        cnt = int(np.sum(df_val["label"] == c))
        print(f"  {c:<8s}: {cnt}")

    y_true = np.array([CLASS_TO_IDX[row["label"]] for _, row in df_val.iterrows()])

    # Frontend
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    # Preload and pad all audio waveforms to RAM
    print("\nLoading 1,280 audio waveforms into RAM...")
    waveforms = []
    for idx, row in df_val.iterrows():
        p = Path(row["audio_path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        wav, sr = sf.read(str(p), dtype="float32")
        if len(wav) < 16000:
            wav = np.pad(wav, (0, 16000 - len(wav)))
        else:
            wav = wav[:16000]
        waveforms.append(wav)
    waveforms_tensor = torch.from_numpy(np.array(waveforms, dtype=np.float32)).to(device)
    print("Waveforms loaded successfully.")

    # Master prediction table
    df_aligned = df_val.copy()

    master_results = {
        "metadata": {
            "phase": "Phase 64 Forensic Validation Audit",
            "manifest_path": str(MANIFEST_PATH),
            "manifest_sha256": manifest_sha,
            "total_samples": len(df_val),
            "class_counts": {c: int(np.sum(df_val["label"] == c)) for c in CLASSES}
        },
        "models": {}
    }

    summary_rows = []

    print("\n" + "=" * 80)
    print("RUNNING FORWARD INFERENCE FOR ALL 8 CHECKPOINTS")
    print("=" * 80)

    for m_name, ckpt_path in CHECKPOINTS.items():
        print(f"\nEvaluating: {m_name}")
        ckpt_sha = compute_sha256(ckpt_path)
        print(f"  Path: {ckpt_path.relative_to(PROJECT_ROOT)}")
        print(f"  SHA-256: {ckpt_sha}")

        # Instantiate model
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CorrectedVietnameseKWSNet(num_classes=5).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Batch inference
        batch_size = 64
        all_logits = []
        with torch.no_grad():
            for b_start in range(0, len(waveforms_tensor), batch_size):
                b_wav = waveforms_tensor[b_start : b_start + batch_size]
                specs = frontend(b_wav)
                logits = model(specs)
                all_logits.append(logits.cpu())

        logits_cat = torch.cat(all_logits, dim=0)
        probs = F.softmax(logits_cat, dim=-1).numpy()

        # 1. Operational Rule: theta_cmd = 0.20
        target_probs = probs[:, :4]
        max_target_p = np.max(target_probs, axis=-1)
        max_target_idx = np.argmax(target_probs, axis=-1)
        pred_theta_020 = np.where(max_target_p >= 0.20, max_target_idx, 4)

        # 2. Standard 5-Class Argmax Rule
        pred_argmax = np.argmax(probs, axis=-1)

        # Record to aligned dataframe
        df_aligned[f"{m_name}_pred_theta_020"] = [CLASSES[i] for i in pred_theta_020]
        df_aligned[f"{m_name}_correct_theta_020"] = (pred_theta_020 == y_true)
        df_aligned[f"{m_name}_pred_argmax"] = [CLASSES[i] for i in pred_argmax]
        df_aligned[f"{m_name}_correct_argmax"] = (pred_argmax == y_true)
        for c_idx, c_name in enumerate(CLASSES):
            df_aligned[f"{m_name}_p_{c_name}"] = probs[:, c_idx]

        # Compute metrics
        res_theta_020 = evaluate_predictions(y_true, pred_theta_020)
        res_argmax = evaluate_predictions(y_true, pred_argmax)

        master_results["models"][m_name] = {
            "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": ckpt_sha,
            "decision_rule_theta_020": res_theta_020,
            "standard_argmax_rule": res_argmax
        }

        # Print comparison
        print(f"  [theta_cmd = 0.20] Acc: {res_theta_020['accuracy']*100:.2f}% ({res_theta_020['correct_samples']}/1280) | 4-Cmd F1: {res_theta_020['macro_f1_4cmd']*100:.2f}% | Min-R: {res_theta_020['min_command_recall']*100:.2f}% | OOV FAR: {res_theta_020['oov_fa_rate']*100:.2f}% | Rej Prec: {res_theta_020['reject_precision']*100:.2f}%")
        print(f"  [Standard Argmax] Acc: {res_argmax['accuracy']*100:.2f}% ({res_argmax['correct_samples']}/1280) | 4-Cmd F1: {res_argmax['macro_f1_4cmd']*100:.2f}% | Min-R: {res_argmax['min_command_recall']*100:.2f}% | OOV FAR: {res_argmax['oov_fa_rate']*100:.2f}% | Rej Prec: {res_argmax['reject_precision']*100:.2f}%")

        summary_rows.append({
            "model_name": m_name,
            "rule": "theta_cmd_0.20",
            "correct_samples": res_theta_020["correct_samples"],
            "total_samples": 1280,
            "accuracy_pct": round(res_theta_020["accuracy"] * 100, 2),
            "macro_f1_4cmd_pct": round(res_theta_020["macro_f1_4cmd"] * 100, 2),
            "macro_f1_5class_pct": round(res_theta_020["macro_f1_5class"] * 100, 2),
            "min_command_recall_pct": round(res_theta_020["min_command_recall"] * 100, 2),
            "trai_recall_pct": round(res_theta_020["per_class"]["TRAI"]["recall"] * 100, 2),
            "phai_recall_pct": round(res_theta_020["per_class"]["PHAI"]["recall"] * 100, 2),
            "len_recall_pct": round(res_theta_020["per_class"]["LEN"]["recall"] * 100, 2),
            "xuong_recall_pct": round(res_theta_020["per_class"]["XUONG"]["recall"] * 100, 2),
            "oov_fa_count": res_theta_020["oov_fa_count"],
            "oov_fa_rate_pct": round(res_theta_020["oov_fa_rate"] * 100, 2),
            "reject_precision_pct": round(res_theta_020["reject_precision"] * 100, 2),
            "reject_recall_pct": round(res_theta_020["reject_recall"] * 100, 2)
        })
        summary_rows.append({
            "model_name": m_name,
            "rule": "standard_argmax",
            "correct_samples": res_argmax["correct_samples"],
            "total_samples": 1280,
            "accuracy_pct": round(res_argmax["accuracy"] * 100, 2),
            "macro_f1_4cmd_pct": round(res_argmax["macro_f1_4cmd"] * 100, 2),
            "macro_f1_5class_pct": round(res_argmax["macro_f1_5class"] * 100, 2),
            "min_command_recall_pct": round(res_argmax["min_command_recall"] * 100, 2),
            "trai_recall_pct": round(res_argmax["per_class"]["TRAI"]["recall"] * 100, 2),
            "phai_recall_pct": round(res_argmax["per_class"]["PHAI"]["recall"] * 100, 2),
            "len_recall_pct": round(res_argmax["per_class"]["LEN"]["recall"] * 100, 2),
            "xuong_recall_pct": round(res_argmax["per_class"]["XUONG"]["recall"] * 100, 2),
            "oov_fa_count": res_argmax["oov_fa_count"],
            "oov_fa_rate_pct": round(res_argmax["oov_fa_rate"] * 100, 2),
            "reject_precision_pct": round(res_argmax["reject_precision"] * 100, 2),
            "reject_recall_pct": round(res_argmax["reject_recall"] * 100, 2)
        })

    # Save outputs
    aligned_csv = REPORTS_DIR / "PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv"
    master_json = REPORTS_DIR / "PHASE64_VALIDATION_MASTER_METRICS.json"
    reconciled_csv = REPORTS_DIR / "PHASE64_VALIDATION_RECONCILED_METRICS.csv"

    df_aligned.to_csv(aligned_csv, index=False)
    print(f"\nSaved aligned prediction records to {aligned_csv}")

    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    print(f"Saved master JSON metrics to {master_json}")

    pd.DataFrame(summary_rows).to_csv(reconciled_csv, index=False)
    print(f"Saved reconciled summary CSV to {reconciled_csv}")

    print("\nPhase 64 Forensic Validation Audit completed successfully!")


if __name__ == "__main__":
    main()
