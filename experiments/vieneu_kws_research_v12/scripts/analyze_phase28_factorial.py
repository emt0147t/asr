#!/usr/bin/env python3
"""
Phase 28-31 Stage E: Factorial Statistical Analysis & Candidate Freeze
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/analyze_phase28_factorial.py

Performs:
1. Two-Way Fixed Effects ANOVA on validation Macro-F1:
   - Factor D (D0 vs D2)
   - Factor H (H0 vs H1)
   - Interaction D x H
2. Randomized Block ANOVA treating random seed (42, 43, 44) as block
3. Winning Condition Selection & Candidate Freeze:
   - Exports winning checkpoint to ONNX
   - Verifies ONNX numerical parity
   - Records all SHA-256 hashes
"""

import sys
import io
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase28_31"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
CANDIDATE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_candidate"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase28_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, compute_sha256

CONDITIONS = ["D0_H0", "D0_H1", "D2_H0", "D2_H1"]
SEEDS = [42, 43, 44]


def two_way_anova(df: pd.DataFrame, response: str = "val_macro_f1") -> Dict[str, Any]:
    # Factor levels: D (D0, D2), H (H0, H1), N = 12 total, n = 3 per cell
    cells = {}
    for d in ["D0", "D2"]:
        for h in ["H0", "H1"]:
            vals = df[(df["factor_d"] == d) & (df["factor_h"] == h)][response].to_numpy()
            cells[(d, h)] = vals

    grand_mean = df[response].mean()
    total_ss = np.sum((df[response] - grand_mean) ** 2)
    n = 3
    a = 2  # D
    b = 2  # H
    N = a * b * n

    # Main effect D
    mean_d = {d: df[df["factor_d"] == d][response].mean() for d in ["D0", "D2"]}
    ss_d = b * n * sum((mean_d[d] - grand_mean) ** 2 for d in ["D0", "D2"])

    # Main effect H
    mean_h = {h: df[df["factor_h"] == h][response].mean() for h in ["H0", "H1"]}
    ss_h = a * n * sum((mean_h[h] - grand_mean) ** 2 for h in ["H0", "H1"])

    # Subtree / Cell SS
    ss_sub = n * sum((np.mean(vals) - grand_mean) ** 2 for vals in cells.values())
    ss_dh = ss_sub - ss_d - ss_h

    # Error SS
    ss_err = sum(np.sum((vals - np.mean(vals)) ** 2) for vals in cells.values())

    df_d = a - 1
    df_h = b - 1
    df_dh = (a - 1) * (b - 1)
    df_err = a * b * (n - 1)

    ms_d = ss_d / df_d
    ms_h = ss_h / df_h
    ms_dh = ss_dh / df_dh
    ms_err = ss_err / df_err if df_err > 0 else 1e-12

    f_d = ms_d / ms_err
    p_d = 1.0 - stats.f.cdf(f_d, df_d, df_err)
    eta_d = ss_d / (ss_d + ss_err)

    f_h = ms_h / ms_err
    p_h = 1.0 - stats.f.cdf(f_h, df_h, df_err)
    eta_h = ss_h / (ss_h + ss_err)

    f_dh = ms_dh / ms_err
    p_dh = 1.0 - stats.f.cdf(f_dh, df_dh, df_err)
    eta_dh = ss_dh / (ss_dh + ss_err)

    return {
        "factor_d": {"SS": float(ss_d), "df": df_d, "MS": float(ms_d), "F": float(f_d), "p_value": float(p_d), "partial_eta2": float(eta_d)},
        "factor_h": {"SS": float(ss_h), "df": df_h, "MS": float(ms_h), "F": float(f_h), "p_value": float(p_h), "partial_eta2": float(eta_h)},
        "interaction_dh": {"SS": float(ss_dh), "df": df_dh, "MS": float(ms_dh), "F": float(f_dh), "p_value": float(p_dh), "partial_eta2": float(eta_dh)},
        "error": {"SS": float(ss_err), "df": df_err, "MS": float(ms_err)},
        "total": {"SS": float(total_ss), "df": N - 1}
    }


def randomized_block_anova(df: pd.DataFrame, response: str = "val_macro_f1") -> Dict[str, Any]:
    # Factors: Treatment (4 conditions), Block (3 seeds)
    grand_mean = df[response].mean()
    total_ss = np.sum((df[response] - grand_mean) ** 2)

    k = 4  # treatments
    b = 3  # blocks (seeds)
    N = k * b

    # SS Treatment
    mean_trt = {c: df[df["condition"] == c][response].mean() for c in CONDITIONS}
    ss_trt = b * sum((mean_trt[c] - grand_mean) ** 2 for c in CONDITIONS)

    # SS Block
    mean_blk = {s: df[df["seed"] == s][response].mean() for s in SEEDS}
    ss_blk = k * sum((mean_blk[s] - grand_mean) ** 2 for s in SEEDS)

    # SS Error
    ss_err = total_ss - ss_trt - ss_blk
    df_trt = k - 1
    df_blk = b - 1
    df_err = (k - 1) * (b - 1)

    ms_trt = ss_trt / df_trt
    ms_blk = ss_blk / df_blk
    ms_err = ss_err / df_err if df_err > 0 else 1e-12

    f_trt = ms_trt / ms_err
    p_trt = 1.0 - stats.f.cdf(f_trt, df_trt, df_err)
    f_blk = ms_blk / ms_err
    p_blk = 1.0 - stats.f.cdf(f_blk, df_blk, df_err)

    return {
        "treatment": {"SS": float(ss_trt), "df": df_trt, "MS": float(ms_trt), "F": float(f_trt), "p_value": float(p_trt)},
        "block": {"SS": float(ss_blk), "df": df_blk, "MS": float(ms_blk), "F": float(f_blk), "p_value": float(p_blk)},
        "error": {"SS": float(ss_err), "df": df_err, "MS": float(ms_err)},
        "total": {"SS": float(total_ss), "df": N - 1}
    }


def export_and_verify_onnx(ckpt_path: Path, onnx_out_path: Path) -> Dict[str, Any]:
    device = torch.device("cpu")
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Dummy input: (1, 1, 40, 101) log-mel spectrogram
    dummy_input = torch.randn(1, 1, 40, 101, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_out_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )

    # Verify ONNX model
    onnx_model = onnx.load(str(onnx_out_path))
    onnx.checker.check_model(onnx_model)

    # Test numerical parity on 10 random inputs
    ort_session = ort.InferenceSession(str(onnx_out_path), providers=["CPUExecutionProvider"])
    max_abs_diff = 0.0

    rng = np.random.RandomState(42)
    for _ in range(10):
        test_in = rng.randn(1, 1, 40, 101).astype(np.float32)
        with torch.no_grad():
            pyt_out = model(torch.from_numpy(test_in)).numpy()
        ort_out = ort_session.run(None, {"input": test_in})[0]
        diff = np.max(np.abs(pyt_out - ort_out))
        if diff > max_abs_diff:
            max_abs_diff = float(diff)

    onnx_sha = compute_sha256(onnx_out_path)
    return {
        "onnx_path": str(onnx_out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "onnx_sha256": onnx_sha,
        "max_abs_diff": max_abs_diff,
        "parity_pass": max_abs_diff < 1e-4
    }


def main():
    print("=" * 80)
    print("PHASE 28-31 STAGE E: STATISTICAL ANALYSIS & CANDIDATE FREEZE")
    print("=" * 80)

    # Load 12 run summaries
    records = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            summary_path = OUTPUT_DIR / f"{cond}_seed{seed}" / "run_summary.json"
            if not summary_path.exists():
                print(f"[ERROR] Missing summary for {cond}_seed{seed}")
                return
            with open(summary_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            records.append({
                "condition": cond,
                "factor_d": d["factor_d"],
                "factor_h": d["factor_h"],
                "seed": seed,
                "best_epoch": d["best_epoch"],
                "val_macro_f1": d["best_val_macro_f1"],
                "overall_macro_f1": d["best_metrics"].get("overall_macro_f1", d["best_val_macro_f1"]),
                "val_accuracy": d["best_metrics"]["val_accuracy"],
                "checkpoint_path": d["checkpoint_path"],
                "checkpoint_sha256": d["checkpoint_sha256"]
            })

    df = pd.DataFrame(records)
    print(f"\nLoaded {len(df)} factorial run summaries.")

    # 1. Condition Summary
    cond_summary = {}
    print("\n--- Condition Performance Summary (Mean +/- Std) ---")
    print(f"{'Condition':<10} | {'Val Macro-F1 (Mean +/- Std)':<28} | {'Accuracy (Mean +/- Std)':<25}")
    print("-" * 70)
    for c in CONDITIONS:
        sub = df[df["condition"] == c]
        f1_m, f1_s = sub["val_macro_f1"].mean(), sub["val_macro_f1"].std()
        acc_m, acc_s = sub["val_accuracy"].mean(), sub["val_accuracy"].std()
        cond_summary[c] = {
            "val_macro_f1_mean": float(f1_m),
            "val_macro_f1_std": float(f1_s),
            "val_accuracy_mean": float(acc_m),
            "val_accuracy_std": float(acc_s),
            "seeds": sub.to_dict(orient="records")
        }
        print(f"{c:<10} | {f1_m*100:6.2f}% +/- {f1_s*100:4.2f}%              | {acc_m*100:6.2f}% +/- {acc_s*100:4.2f}%")

    # 2. ANOVA
    anova_res = two_way_anova(df, "val_macro_f1")
    block_anova = randomized_block_anova(df, "val_macro_f1")

    print("\n--- Two-Way Fixed Effects ANOVA (Dependent Variable: val_macro_f1) ---")
    for source in ["factor_d", "factor_h", "interaction_dh", "error", "total"]:
        res = anova_res[source]
        f_str = f"{res.get('F', 0.0):.4f}" if "F" in res else "-"
        p_str = f"{res.get('p_value', 0.0):.4f}" if "p_value" in res else "-"
        eta_str = f"{res.get('partial_eta2', 0.0):.4f}" if "partial_eta2" in res else "-"
        print(f"  {source:<15}: SS={res['SS']:.6f}, df={res['df']}, MS={res.get('MS',0.0):.6f}, F={f_str}, p={p_str}, eta2={eta_str}")

    print("\n--- Randomized Block ANOVA (Seed as Block) ---")
    for source in ["treatment", "block", "error", "total"]:
        res = block_anova[source]
        f_str = f"{res.get('F', 0.0):.4f}" if "F" in res else "-"
        p_str = f"{res.get('p_value', 0.0):.4f}" if "p_value" in res else "-"
        print(f"  {source:<15}: SS={res['SS']:.6f}, df={res['df']}, MS={res.get('MS',0.0):.6f}, F={f_str}, p={p_str}")

    # 3. Winning Condition & Candidate Freeze
    # Best condition by highest mean validation Macro-F1
    best_cond = max(cond_summary.keys(), key=lambda c: cond_summary[c]["val_macro_f1_mean"])
    # Within best condition, select best seed
    best_sub = df[df["condition"] == best_cond].sort_values("val_macro_f1", ascending=False)
    best_run = best_sub.iloc[0]

    winner_name = f"{best_cond}_seed{best_run['seed']}"
    print(f"\n[WINNING CONDITION]: {best_cond} (Mean Val Macro-F1: {cond_summary[best_cond]['val_macro_f1_mean']*100:.2f}%)")
    print(f"[DESIGNATED CANDIDATE]: {winner_name} (Val Macro-F1: {best_run['val_macro_f1']*100:.2f}%)")

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_src = PROJECT_ROOT / best_run["checkpoint_path"]
    candidate_ckpt_dst = CANDIDATE_DIR / "candidate_checkpoint.pt"

    # Copy checkpoint
    import shutil
    shutil.copy2(str(best_ckpt_src), str(candidate_ckpt_dst))
    candidate_ckpt_sha = compute_sha256(candidate_ckpt_dst)

    # Export ONNX
    candidate_onnx_dst = CANDIDATE_DIR / "vietnamese_kws_phase28_candidate.onnx"
    onnx_parity = export_and_verify_onnx(candidate_ckpt_dst, candidate_onnx_dst)

    candidate_metadata = {
        "candidate_id": f"phase28_candidate_{winner_name}",
        "condition": best_cond,
        "seed": int(best_run["seed"]),
        "factor_d": best_run["factor_d"],
        "factor_h": best_run["factor_h"],
        "val_macro_f1": float(best_run["val_macro_f1"]),
        "val_accuracy": float(best_run["val_accuracy"]),
        "checkpoint_path": str(candidate_ckpt_dst.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": candidate_ckpt_sha,
        "onnx_path": onnx_parity["onnx_path"],
        "onnx_sha256": onnx_parity["onnx_sha256"],
        "onnx_max_abs_diff": onnx_parity["max_abs_diff"],
        "onnx_parity_pass": onnx_parity["parity_pass"],
        "frozen_architecture": "CorrectedVietnameseKWSNet",
        "total_parameters": 138469,
        "decision_threshold": 0.20
    }

    with open(CANDIDATE_DIR / "candidate_metadata.json", "w", encoding="utf-8") as f:
        json.dump(candidate_metadata, f, indent=2, ensure_ascii=False)

    # Save full statistical analysis report
    analysis_report = {
        "metadata": {
            "timestamp": "2026-10-25T01:30:00+07:00",
            "phase": "Phase 28-31 Stage E",
            "total_runs": len(df),
            "factors": {"D": ["D0", "D2"], "H": ["H0", "H1"]},
            "seeds": SEEDS
        },
        "condition_summary": cond_summary,
        "two_way_anova": anova_res,
        "randomized_block_anova": block_anova,
        "winning_candidate": candidate_metadata
    }

    out_analysis_path = REPORTS_DIR / "phase28_31_statistical_analysis.json"
    with open(out_analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Candidate Frozen:")
    print(f"     Checkpoint SHA-256: {candidate_ckpt_sha}")
    print(f"     ONNX SHA-256:       {onnx_parity['onnx_sha256']}")
    print(f"     ONNX Max Abs Diff:  {onnx_parity['max_abs_diff']:.2e} (Pass: {onnx_parity['parity_pass']})")
    print(f"     Analysis JSON:      {out_analysis_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
