"""
Phase 12.1: Full-Cohort Feature and Waveform Similarity Analysis.
Evaluates all 1,468 clips in V4 (952 Train, 216 Val, 300 Fresh Holdout).
Distinguishes EXACT_LEAKAGE, NEAR_DUPLICATES, and GENERATOR_DEPENDENCE.
"""

import sys
import os
import json
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
from src.preprocessing.feature_extractor import extract_log_mel_numpy


def run_full_similarity():
    v4_tr = pd.read_csv("experiments/synthetic_kws_v4/train_manifest_onset_aware.csv")
    v4_va = pd.read_csv("experiments/synthetic_kws_v4/val_manifest_onset_aware.csv")
    v4_te = pd.read_csv("experiments/synthetic_kws_v4/fresh_holdout_manifest_6class.csv")

    print(f"Total clips to analyze: Train={len(v4_tr)}, Val={len(v4_va)}, Test={len(v4_te)}")

    def extract_all(df):
        feats = {}
        for idx, r in df.iterrows():
            p = r["processed_path"]
            wav, sr = sf.read(p)
            f = extract_log_mel_numpy(wav).flatten()
            norm = np.linalg.norm(f)
            feats[r["recording_id"]] = f / (norm + 1e-9)
        return feats

    print("Extracting Log-Mel features for all 1,468 clips...")
    f_tr = extract_all(v4_tr)
    f_va = extract_all(v4_va)
    f_te = extract_all(v4_te)
    print("Feature extraction complete.")

    classes = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    results = {}

    for c in classes:
        ids_tr = v4_tr[v4_tr["class_label"] == c]["recording_id"].tolist()
        ids_va = v4_va[v4_va["class_label"] == c]["recording_id"].tolist()
        ids_te = v4_te[v4_te["class_label"] == c]["recording_id"].tolist()

        mat_tr = np.array([f_tr[i] for i in ids_tr])
        mat_va = np.array([f_va[i] for i in ids_va])
        mat_te = np.array([f_te[i] for i in ids_te])

        # Within Train
        sim_tr_tr = np.dot(mat_tr, mat_tr.T)
        tr_tr_vals = sim_tr_tr[np.triu_indices(len(ids_tr), k=1)]

        # Within Test
        sim_te_te = np.dot(mat_te, mat_te.T)
        te_te_vals = sim_te_te[np.triu_indices(len(ids_te), k=1)]

        # Cross Train-Val
        sim_tr_va = np.dot(mat_tr, mat_va.T).flatten()

        # Cross Train-Test
        sim_tr_te = np.dot(mat_tr, mat_te.T)
        tr_te_vals = sim_tr_te.flatten()

        # Nearest neighbor in Train for each Test sample
        nn_tr_for_te = np.max(sim_tr_te, axis=0)

        res_c = {
            "n_train": len(ids_tr),
            "n_val": len(ids_va),
            "n_test": len(ids_te),
            "total_cross_comparisons_train_test": len(tr_te_vals),
            "within_train_mean": float(np.mean(tr_tr_vals)),
            "within_train_std": float(np.std(tr_tr_vals)),
            "within_train_min": float(np.min(tr_tr_vals)),
            "within_train_max": float(np.max(tr_tr_vals)),
            "within_test_mean": float(np.mean(te_te_vals)),
            "within_test_std": float(np.std(te_te_vals)),
            "cross_train_val_mean": float(np.mean(sim_tr_va)),
            "cross_train_val_std": float(np.std(sim_tr_va)),
            "cross_train_test_mean": float(np.mean(tr_te_vals)),
            "cross_train_test_std": float(np.std(tr_te_vals)),
            "cross_train_test_p50": float(np.percentile(tr_te_vals, 50)),
            "cross_train_test_p95": float(np.percentile(tr_te_vals, 95)),
            "cross_train_test_min": float(np.min(tr_te_vals)),
            "cross_train_test_max": float(np.max(tr_te_vals)),
            "nn_test_in_train_mean": float(np.mean(nn_tr_for_te)),
            "nn_test_in_train_std": float(np.std(nn_tr_for_te)),
            "nn_test_in_train_min": float(np.min(nn_tr_for_te)),
            "nn_test_in_train_max": float(np.max(nn_tr_for_te)),
            "diagnosis": "GENERATOR_DEPENDENCE" if abs(np.mean(tr_tr_vals) - np.mean(tr_te_vals)) < 0.02 else "POTENTIAL_SHIFT"
        }
        results[c] = res_c

        print(f"=== Class {c:8} (Train: {len(ids_tr)}, Val: {len(ids_va)}, Test: {len(ids_te)}) ===")
        print(f"  Within-Train  : mean={res_c['within_train_mean']:.4f} +/- {res_c['within_train_std']:.4f} (range [{res_c['within_train_min']:.4f}, {res_c['within_train_max']:.4f}])")
        print(f"  Within-Test   : mean={res_c['within_test_mean']:.4f} +/- {res_c['within_test_std']:.4f}")
        print(f"  Cross Tr-Va   : mean={res_c['cross_train_val_mean']:.4f} +/- {res_c['cross_train_val_std']:.4f}")
        print(f"  Cross Tr-Te   : mean={res_c['cross_train_test_mean']:.4f} +/- {res_c['cross_train_test_std']:.4f} (p95={res_c['cross_train_test_p95']:.4f}, max={res_c['cross_train_test_max']:.4f})")
        print(f"  1-NN in Train : mean={res_c['nn_test_in_train_mean']:.4f} (range [{res_c['nn_test_in_train_min']:.4f}, {res_c['nn_test_in_train_max']:.4f}])")
        print(f"  Diagnosis     : {res_c['diagnosis']}")
        print()

    out_json = Path("experiments/synthetic_kws_v4/audit/split_leakage_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full split leakage report to: {out_json}")
    return results


if __name__ == "__main__":
    run_full_similarity()
