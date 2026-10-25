#!/usr/bin/env python3
"""
Phase 25 Factorial Experiment Statistical Analysis
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/analyze_phase25_factorial.py

Performs ANOVA / factorial decomposition and statistical quantification:
- Factor D (Training Source Distribution: D0 Turbo vs D1 Turbo+Nano)
- Factor H (Negative Sampling: H0 Standard vs H1 Hard-Negative)
- Interaction Effect: D x H
- Per-class precision, recall, F1 analysis
- Model Selection & Candidate Sign-Off
"""

import sys
import io
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase25"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"

CONDITIONS = ["D0_H0", "D0_H1", "D1_H0", "D1_H1"]
SEEDS = [42, 43, 44]
CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]


def main():
    print("=" * 80)
    print("PHASE 25 FACTORIAL EXPERIMENT STATISTICAL ANALYSIS")
    print("=" * 80)

    # 1. Load run summaries
    records = []
    per_class_records = []

    for cond in CONDITIONS:
        for seed in SEEDS:
            summary_file = OUTPUT_DIR / f"{cond}_seed{seed}" / "run_summary.json"
            if not summary_file.exists():
                raise FileNotFoundError(f"Missing summary file: {summary_file}")

            with open(summary_file, "r", encoding="utf-8") as f:
                s = json.load(f)

            records.append({
                "condition": cond,
                "factor_d": s["factor_d"],
                "factor_h": s["factor_h"],
                "seed": seed,
                "best_epoch": s["best_epoch"],
                "val_macro_f1": s["best_val_macro_f1"],
                "val_accuracy": s["best_val_accuracy"],
                "val_loss": s["best_val_loss"],
                "training_duration_s": s["training_duration_seconds"],
                "checkpoint_sha256": s["checkpoint_sha256"]
            })

            for c in CLASSES:
                pc = s["best_per_class"][c]
                per_class_records.append({
                    "condition": cond,
                    "factor_d": s["factor_d"],
                    "factor_h": s["factor_h"],
                    "seed": seed,
                    "class": c,
                    "precision": pc["precision"],
                    "recall": pc["recall"],
                    "f1": pc["f1"]
                })

    df = pd.DataFrame(records)
    df_pc = pd.DataFrame(per_class_records)

    print("\n--- Summary by Condition (Mean ± Std over 3 seeds) ---")
    cond_stats = {}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        f1_vals = sub["val_macro_f1"].values
        acc_vals = sub["val_accuracy"].values
        loss_vals = sub["val_loss"].values

        cond_stats[cond] = {
            "val_macro_f1_mean": float(np.mean(f1_vals)),
            "val_macro_f1_std": float(np.std(f1_vals, ddof=1)),
            "val_accuracy_mean": float(np.mean(acc_vals)),
            "val_accuracy_std": float(np.std(acc_vals, ddof=1)),
            "val_loss_mean": float(np.mean(loss_vals)),
            "val_loss_std": float(np.std(loss_vals, ddof=1)),
            "seeds": [int(x) for x in sub["seed"].values],
            "runs": sub.to_dict(orient="records")
        }
        print(f"  {cond:<6} | Val Macro-F1: {np.mean(f1_vals)*100:.2f}% ± {np.std(f1_vals, ddof=1)*100:.2f}% | Val Acc: {np.mean(acc_vals)*100:.2f}% ± {np.std(acc_vals, ddof=1)*100:.2f}% | Val Loss: {np.mean(loss_vals):.4f}")

    # 2. Factorial Main Effects and Interaction Analysis
    # Factor D: D1 vs D0
    d0_f1 = df[df["factor_d"] == "D0"]["val_macro_f1"].values
    d1_f1 = df[df["factor_d"] == "D1"]["val_macro_f1"].values
    effect_d = float(np.mean(d1_f1) - np.mean(d0_f1))
    t_stat_d, p_val_d = stats.ttest_ind(d1_f1, d0_f1, equal_var=False)

    # Factor H: H1 vs H0
    h0_f1 = df[df["factor_h"] == "H0"]["val_macro_f1"].values
    h1_f1 = df[df["factor_h"] == "H1"]["val_macro_f1"].values
    effect_h = float(np.mean(h1_f1) - np.mean(h0_f1))
    t_stat_h, p_val_h = stats.ttest_ind(h1_f1, h0_f1, equal_var=False)

    # Interaction Effect (D x H)
    # Mean(D1H1) - Mean(D1H0) - (Mean(D0H1) - Mean(D0H0))
    m_d0h0 = np.mean(df[df["condition"] == "D0_H0"]["val_macro_f1"])
    m_d0h1 = np.mean(df[df["condition"] == "D0_H1"]["val_macro_f1"])
    m_d1h0 = np.mean(df[df["condition"] == "D1_H0"]["val_macro_f1"])
    m_d1h1 = np.mean(df[df["condition"] == "D1_H1"]["val_macro_f1"])
    interaction_dh = float((m_d1h1 - m_d1h0) - (m_d0h1 - m_d0h0))

    print("\n--- Factorial Decomposition (Macro-F1) ---")
    print(f"  Factor D (Nano Treatment):      {effect_d*100:+.2f}% (t={t_stat_d:.3f}, p={p_val_d:.4f})")
    print(f"  Factor H (Hard-Negative Prior):  {effect_h*100:+.2f}% (t={t_stat_h:.3f}, p={p_val_h:.4f})")
    print(f"  Interaction (D x H):             {interaction_dh*100:+.2f}%")

    # 3. Per-class Analysis
    print("\n--- Per-Class Performance Summary (Mean F1 across seeds) ---")
    per_class_summary = {}
    for c in CLASSES:
        per_class_summary[c] = {}
        for cond in CONDITIONS:
            sub = df_pc[(df_pc["condition"] == cond) & (df_pc["class"] == c)]
            mean_f1 = float(np.mean(sub["f1"]))
            std_f1 = float(np.std(sub["f1"], ddof=1))
            mean_rec = float(np.mean(sub["recall"]))
            mean_prec = float(np.mean(sub["precision"]))
            per_class_summary[c][cond] = {
                "f1_mean": mean_f1,
                "f1_std": std_f1,
                "recall_mean": mean_rec,
                "precision_mean": mean_prec
            }
        print(f"  {c:<8} | D0_H0: {per_class_summary[c]['D0_H0']['f1_mean']*100:.2f}% | D0_H1: {per_class_summary[c]['D0_H1']['f1_mean']*100:.2f}% | D1_H0: {per_class_summary[c]['D1_H0']['f1_mean']*100:.2f}% | D1_H1: {per_class_summary[c]['D1_H1']['f1_mean']*100:.2f}%")

    # 4. Best Condition and Candidate Checkpoint Identification
    # Rank conditions by mean val_macro_f1
    ranked_conds = sorted(cond_stats.items(), key=lambda x: x[1]["val_macro_f1_mean"], reverse=True)
    best_condition_name = ranked_conds[0][0]
    best_cond_mean_f1 = ranked_conds[0][1]["val_macro_f1_mean"]

    # Best single checkpoint across all 12 runs
    best_run = df.sort_values(by="val_macro_f1", ascending=False).iloc[0].to_dict()

    print("\n--- Ranking & Best Candidate Selection ---")
    for rank, (c_name, c_data) in enumerate(ranked_conds, 1):
        print(f"  Rank {rank}: {c_name:<6} -> Val Macro-F1: {c_data['val_macro_f1_mean']*100:.2f}% ± {c_data['val_macro_f1_std']*100:.2f}%")

    print(f"\nWinning Condition: {best_condition_name} ({best_cond_mean_f1*100:.2f}%)")
    print(f"Best Single Run:   {best_run['condition']}_seed{best_run['seed']} (Macro-F1: {best_run['val_macro_f1']*100:.2f}%, Checkpoint SHA: {best_run['checkpoint_sha256'][:16]}...)")

    # 5. Export comprehensive results json
    results_json = {
        "metadata": {
            "experiment": "Phase 25 2x2 Factorial Experiment",
            "date": "2026-10-24",
            "architecture": "CorrectedVietnameseKWSNet (138,469 params)",
            "decision_threshold": 0.20,
            "validation_manifest": "validation_v1.csv (1,280 samples)",
            "total_runs": len(df)
        },
        "condition_statistics": cond_stats,
        "factorial_effects": {
            "factor_d_main_effect": effect_d,
            "factor_d_t_stat": t_stat_d,
            "factor_d_p_value": p_val_d,
            "factor_h_main_effect": effect_h,
            "factor_h_t_stat": t_stat_h,
            "factor_h_p_value": p_val_h,
            "interaction_dh_effect": interaction_dh
        },
        "per_class_summary": per_class_summary,
        "ranking": [
            {"rank": i + 1, "condition": c[0], "macro_f1_mean": c[1]["val_macro_f1_mean"], "macro_f1_std": c[1]["val_macro_f1_std"]}
            for i, c in enumerate(ranked_conds)
        ],
        "winning_condition": best_condition_name,
        "best_single_run": best_run
    }

    out_json_path = REPORTS_DIR / "phase25_factorial_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nSaved statistical analysis results to: {out_json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
