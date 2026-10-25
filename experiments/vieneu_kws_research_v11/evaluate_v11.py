#!/usr/bin/env python3
"""
Stage C: Independent Evaluation of Baseline and v11 Candidates on Fresh Locked Test.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v11/evaluate_v11.py
"""

import os
import sys
import json
import csv
import time
import math
import hashlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V11_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11"
DATA_DIR = V11_DIR / "data" / "splits"
CKPT_DIR = V11_DIR / "checkpoints"
CONFIGS_DIR = V11_DIR / "configs"
PRED_DIR = V11_DIR / "predictions"
REPORTS_DIR = V11_DIR / "reports"

BASELINE_CKPT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v9" / "artifacts" / "checkpoints" / "variant_w10_seed42.pt"

from experiments.vieneu_kws_research_v11.train_v11 import (
    LogMelFrontend,
    CorrectedVietnameseKWSNet,
    CLASSES,
    TARGET_CLASSES,
    compute_sha256
)


def wilson_interval(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(lower, 4), round(upper, 4)


def load_model(ckpt_path: Path, device: torch.device):
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    payload = torch.load(ckpt_path, map_location=device)
    model = CorrectedVietnameseKWSNet(num_classes=5).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def evaluate_model_on_test(model, frontend, test_rows, device, theta_cmd=0.40):
    cm = defaultdict(lambda: defaultdict(int))
    predictions = []

    # Category breakdown for FAR
    fam_denominators = defaultdict(int)
    fam_fa = defaultdict(int)

    latencies = []

    with torch.no_grad():
        for r in test_rows:
            wav_path = PROJECT_ROOT / r["kws_wav_path"]
            audio, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000:
                raise ValueError(f"Expected 16kHz, got {sr}")
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
            elif len(audio) > 16000:
                audio = audio[:16000]

            x = torch.from_numpy(audio).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            specs = frontend(x)
            logits = model(specs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)

            cmd_probs = probs[:4]
            raw_argmax = int(np.argmax(cmd_probs))
            max_prob = float(cmd_probs[raw_argmax])

            if max_prob >= theta_cmd:
                pred_class_id = raw_argmax
                pred_label = TARGET_CLASSES[raw_argmax]
                status = "ACCEPTED"
            else:
                pred_class_id = 4
                pred_label = "REJECT"
                status = "REJECTED_LOW_CONFIDENCE"

            gt_label = r["bounded_label"]
            gt_id = int(r["bounded_class_id"])
            fam = r["confusion_family"]

            cm[gt_label][pred_label] += 1

            if gt_label == "REJECT":
                fam_denominators[fam] += 1
                if pred_label != "REJECT":
                    fam_fa[fam] += 1

            predictions.append({
                "sample_id": r["sample_id"],
                "transcription": r["transcription"],
                "confusion_family": fam,
                "voice_id": r["voice_id"],
                "ground_truth": gt_label,
                "predicted_label": pred_label,
                "confidence": round(max_prob, 4),
                "rejection_status": status,
                "is_correct": (gt_label == pred_label),
                "prob_TRAI": round(float(probs[0]), 4),
                "prob_PHAI": round(float(probs[1]), 4),
                "prob_LEN": round(float(probs[2]), 4),
                "prob_XUONG": round(float(probs[3]), 4),
                "prob_REJECT": round(float(probs[4]), 4)
            })

    # Metric computations
    cmd_correct = sum(cm[c][c] for c in TARGET_CLASSES)
    cmd_total = sum(sum(cm[c].values()) for c in TARGET_CLASSES)
    macro_recall = float(cmd_correct / cmd_total) if cmd_total > 0 else 0.0

    per_kw_recall = {}
    per_kw_ci = {}
    for c in TARGET_CLASSES:
        tot = sum(cm[c].values())
        cor = cm[c][c]
        rec = float(cor / tot) if tot > 0 else 0.0
        per_kw_recall[c] = rec
        per_kw_ci[c] = wilson_interval(cor, tot)

    nt_total = sum(cm["REJECT"].values())
    nt_fa = sum(cm["REJECT"][c] for c in TARGET_CLASSES)
    nt_far = float(nt_fa / nt_total) if nt_total > 0 else 0.0
    nt_far_ci = wilson_interval(nt_fa, nt_total)

    total_correct = cmd_correct + cm["REJECT"]["REJECT"]
    accuracy = float(total_correct / len(test_rows))

    # FAR per family
    far_by_fam = {}
    for fam, tot in fam_denominators.items():
        fa = fam_fa[fam]
        far_by_fam[fam] = {
            "false_accepts": fa,
            "total": tot,
            "far": round(fa / tot, 4) if tot > 0 else 0.0
        }

    # Criteria evaluation
    all_kw_pass = all(per_kw_recall[c] >= 0.95 for c in TARGET_CLASSES)
    macro_pass = (macro_recall >= 0.95)
    far_pass = (nt_far <= 0.05)
    passed_all = (all_kw_pass and macro_pass and far_pass)

    return {
        "theta_cmd": theta_cmd,
        "sample_counts": {
            "total_test_clips": len(test_rows),
            "command_clips": cmd_total,
            "non_target_clips": nt_total
        },
        "metrics": {
            "mean_command_recall": round(macro_recall, 4),
            "mean_command_recall_ci95": wilson_interval(cmd_correct, cmd_total),
            "per_keyword_recall": {c: round(per_kw_recall[c], 4) for c in TARGET_CLASSES},
            "per_keyword_recall_ci95": per_kw_ci,
            "non_target_far": round(nt_far, 4),
            "non_target_far_ci95": nt_far_ci,
            "non_target_false_accepts_count": nt_fa,
            "accuracy": round(accuracy, 4),
            "far_breakdown_by_family": far_by_fam
        },
        "confusion_matrix": {r: dict(cm[r]) for r in TARGET_CLASSES + ["REJECT"]},
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "std": round(float(np.std(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2)
        },
        "acceptance_criteria": {
            "mean_command_recall_ge_95": macro_pass,
            "all_keywords_ge_95": all_kw_pass,
            "non_target_far_le_05": far_pass,
            "verdict": "PASSED_BOUNDED_UTTERANCE_TARGETS" if passed_all else "FAILED_BOUNDED_UTTERANCE_TARGETS"
        },
        "predictions": predictions
    }


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE C: LOCKED TEST EVALUATION OF BASELINE & V11 CANDIDATES")
    print("=" * 80)

    test_csv = DATA_DIR / "test_manifest.csv"
    assert test_csv.exists(), f"Missing test manifest: {test_csv}"
    with open(test_csv, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    print(f"Loaded {len(test_rows)} fresh locked test clips from {test_csv.name}")

    device = torch.device("cpu")
    torch.set_num_threads(4)
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    # Load calibration summary
    calib_json = CONFIGS_DIR / "calibration_summary.json"
    optimal_thresholds = {}
    if calib_json.exists():
        with open(calib_json, "r", encoding="utf-8") as f:
            for item in json.load(f):
                optimal_thresholds[item["candidate_id"]] = item["optimal_theta"]

    # Candidate registry
    candidates = []

    # 1. Baseline comparator
    assert BASELINE_CKPT.exists(), f"Baseline checkpoint missing: {BASELINE_CKPT}"
    candidates.append({
        "candidate_id": "frozen_baseline_variant_w10_seed42",
        "ckpt_path": BASELINE_CKPT,
        "eval_theta": 0.40,
        "is_baseline": True
    })

    # 2. v11 Trained Candidates
    for ckpt_file in sorted(CKPT_DIR.glob("*.pt")):
        cid = ckpt_file.stem
        calib_th = optimal_thresholds.get(cid, 0.40)
        candidates.append({
            "candidate_id": cid,
            "ckpt_path": ckpt_file,
            "eval_theta": calib_th,
            "is_baseline": False
        })

    all_evaluation_results = {}

    for cand in candidates:
        cid = cand["candidate_id"]
        cpath = cand["ckpt_path"]
        theta = cand["eval_theta"]

        print(f"\n--- Evaluating Candidate: {cid} (Threshold: {theta}) ---")
        model, _ = load_model(cpath, device)
        res = evaluate_model_on_test(model, frontend, test_rows, device, theta_cmd=theta)
        res["checkpoint_path"] = str(cpath.relative_to(PROJECT_ROOT)).replace("\\", "/")
        res["checkpoint_sha256"] = compute_sha256(cpath)

        # Save predictions CSV
        pred_csv_path = PRED_DIR / f"{cid}_locked_test_predictions.csv"
        fieldnames = list(res["predictions"][0].keys())
        with open(pred_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(res["predictions"])

        # Remove raw prediction rows from JSON summary
        res_summary = {k: v for k, v in res.items() if k != "predictions"}
        all_evaluation_results[cid] = res_summary

        m = res_summary["metrics"]
        print(f"Mean Command Recall: {m['mean_command_recall']*100:.2f}% (CI: {m['mean_command_recall_ci95']})")
        print(f"Per-Keyword Recall:  {m['per_keyword_recall']}")
        print(f"Non-Target FAR:      {m['non_target_far']*100:.2f}% (CI: {m['non_target_far_ci95']}, FA: {m['non_target_false_accepts_count']}/480)")
        print(f"Accuracy:            {m['accuracy']*100:.2f}%")
        print(f"Verdict:             {res_summary['acceptance_criteria']['verdict']}")

    # Save summary report JSON
    summary_path = REPORTS_DIR / "locked_test_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluation_results, f, indent=2)
    print(f"\n[OK] Saved locked test evaluation summary: {summary_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
