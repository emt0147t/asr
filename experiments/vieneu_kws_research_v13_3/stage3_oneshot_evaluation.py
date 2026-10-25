#!/usr/bin/env python3
"""
Phase 13.3 - Stage 3: One-Shot Independent Confirmatory Evaluation.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_3/stage3_oneshot_evaluation.py

Executes one-shot evaluation of the frozen candidate CNN on the locked 1,000-clip dataset:
- Checkpoint: variant_v11_hard_negative_weighted_seed42.pt (SHA-256: 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2)
- Frontend: Remediated Phase 13.1 (n_fft=512, win_length=400, hop_length=160, 40 Slaney bands, peak clamping if max > 1.0)
- Decision Threshold: theta_cmd = 0.20 (Frozen)

STRICT GOVERNANCE:
- One-shot execution: exactly one pass over the locked 1,000-clip dataset.
- Zero tuning or adaptation based on test results.
- Full reporting of per-class metrics, confusion matrix, category FAR, and CPU latency.
- Evaluation against preregistered criteria.
"""

import os
import sys
import json
import csv
import time
import math
import hashlib
import numpy as np
import torch
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MANIFEST_CSV = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "confirmatory_evaluation_manifest.csv"
MANIFEST_JSON = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "confirmatory_evaluation_manifest.json"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v13_1"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
EXPECTED_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
DEFAULT_THETA = 0.20

PREDICTIONS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "reports"
PREDICTIONS_CSV = PREDICTIONS_DIR / "confirmatory_predictions.csv"
PREDICTIONS_JSON = PREDICTIONS_DIR / "confirmatory_predictions.json"
METRICS_JSON = REPORTS_DIR / "confirmatory_evaluation_metrics.json"

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
ALL_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = float(successes) / float(total)
    denominator = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denominator
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2)))) / denominator
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(p_hat, 4), round(lower, 4), round(upper, 4)


def run_oneshot_evaluation():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 13.3 - STAGE 3: ONE-SHOT INDEPENDENT CONFIRMATORY EVALUATION")
    print("=" * 80)

    # 1. Verify Checkpoint Immobility
    actual_sha = compute_sha256(CHECKPOINT_PATH)
    assert actual_sha == EXPECTED_SHA256, f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_SHA256}\nActual:   {actual_sha}"
    print(f"[OK] Checkpoint Verified: {actual_sha}")

    # 2. Verify Manifest
    assert MANIFEST_CSV.exists(), f"Manifest CSV missing: {MANIFEST_CSV}"
    manifest_sha = compute_sha256(MANIFEST_CSV)
    print(f"[OK] Manifest CSV Verified: {manifest_sha}")

    # 3. Load Engine
    from experiments.vieneu_kws_research_v13_1.bundle.infer import KWSInferenceEngine
    print(f"Loading KWSInferenceEngine with theta_cmd = {DEFAULT_THETA}...")
    engine = KWSInferenceEngine(checkpoint_path=CHECKPOINT_PATH, theta_cmd=DEFAULT_THETA, device="cpu")
    print("[OK] Inference Engine loaded on CPU.")

    # 4. Load Manifest Records
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_records = list(reader)

    assert len(manifest_records) == 1000, f"Expected exactly 1000 records, got {len(manifest_records)}"
    print(f"Evaluating {len(manifest_records)} frozen clips in a single immutable pass...")

    predictions = []
    latencies = []

    # 5. Execute One-Shot Evaluation Loop
    t_eval_start = time.time()
    for idx, r in enumerate(manifest_records):
        wav_path = PROJECT_ROOT / r["rel_path"]
        assert wav_path.exists(), f"WAV missing: {wav_path}"

        pred_res = engine.predict(wav_path)
        latencies.append(pred_res["latency_ms"])

        pred_record = {
            "clip_id": r["clip_id"],
            "rel_path": r["rel_path"],
            "ground_truth": r["ground_truth"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "speaker_id": r["speaker_id"],
            "voice_name": r["voice_name"],
            "voice_accent": r["voice_accent"],
            "intended_text": r["intended_text"],
            "predicted_class": pred_res["predicted_class"],
            "confidence": pred_res["confidence"],
            "rejection_status": pred_res["rejection_status"],
            "raw_argmax_class": pred_res["raw_argmax_class"],
            "theta_cmd": pred_res["theta_cmd"],
            "correct": (pred_res["predicted_class"] == r["ground_truth"]),
            "prob_TRAI": pred_res["probabilities"]["TRAI"],
            "prob_PHAI": pred_res["probabilities"]["PHAI"],
            "prob_LEN": pred_res["probabilities"]["LEN"],
            "prob_XUONG": pred_res["probabilities"]["XUONG"],
            "prob_REJECT": pred_res["probabilities"]["REJECT"],
            "latency_ms": pred_res["latency_ms"]
        }
        predictions.append(pred_record)

        if (idx + 1) % 200 == 0:
            print(f"  - Evaluated {idx + 1}/1000 clips...")

    t_eval_elapsed = time.time() - t_eval_start
    print(f"[OK] One-shot inference completed in {t_eval_elapsed:.2f}s.")

    # 6. Save Predictions (CSV and JSON)
    pred_keys = list(predictions[0].keys())
    with open(PREDICTIONS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_keys)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"[OK] Predictions CSV written: {PREDICTIONS_CSV.name}")

    with open(PREDICTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[OK] Predictions JSON written: {PREDICTIONS_JSON.name}")

    # 7. Compute Rigorous Evaluation Metrics
    total_clips = len(predictions)
    total_correct = sum(1 for p in predictions if p["correct"])
    overall_accuracy, acc_low, acc_high = wilson_score_interval(total_correct, total_clips)

    # Confusion Matrix: rows = ground_truth, cols = predicted_class
    cm = {gt: {pr: 0 for pr in ALL_CLASSES} for gt in ALL_CLASSES}
    for p in predictions:
        cm[p["ground_truth"]][p["predicted_class"]] += 1

    # Per-Command Recall and Precision
    command_metrics = {}
    total_cmd_tp = 0
    total_cmd_count = 0

    for cmd in TARGET_CLASSES:
        tp = cm[cmd][cmd]
        fn = sum(cm[cmd][pr] for pr in ALL_CLASSES if pr != cmd)
        fp = sum(cm[gt][cmd] for gt in ALL_CLASSES if gt != cmd)
        cmd_total = tp + fn  # should be 50

        recall, rec_low, rec_high = wilson_score_interval(tp, cmd_total)
        prec, prec_low, prec_high = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0, 0.0)

        total_cmd_tp += tp
        total_cmd_count += cmd_total

        command_metrics[cmd] = {
            "total_clips": cmd_total,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "recall": recall,
            "recall_ci_95": [rec_low, rec_high],
            "precision": prec,
            "precision_ci_95": [prec_low, prec_high],
            "passed_target_95": (recall >= 0.95)
        }

    macro_cmd_recall, mrec_low, mrec_high = wilson_score_interval(total_cmd_tp, total_cmd_count)

    # Non-Target False Alarm Rate (FAR) Overall & By Category
    non_target_preds = [p for p in predictions if p["ground_truth"] == "REJECT"]
    total_non_targets = len(non_target_preds)  # 800
    total_false_alarms = sum(1 for p in non_target_preds if p["predicted_class"] != "REJECT")
    overall_far, far_low, far_high = wilson_score_interval(total_false_alarms, total_non_targets)

    # Category FAR Breakdown
    category_far = {}
    for cat in [
        "phonetic_minimal_pairs",
        "syntactic_carrier_collisions",
        "general_conversational_non_targets",
        "acoustic_background_and_silence"
    ]:
        cat_preds = [p for p in non_target_preds if p["category"] == cat]
        cat_total = len(cat_preds)
        cat_fa = sum(1 for p in cat_preds if p["predicted_class"] != "REJECT")
        far_val, c_low, c_high = wilson_score_interval(cat_fa, cat_total)
        category_far[cat] = {
            "total_clips": cat_total,
            "false_accepts": cat_fa,
            "true_rejects": cat_total - cat_fa,
            "far": far_val,
            "far_ci_95": [c_low, c_high]
        }

    # Latency Metrics
    latencies_np = np.array(latencies)
    mean_lat = float(np.mean(latencies_np))
    median_lat = float(np.median(latencies_np))
    p95_lat = float(np.percentile(latencies_np, 95))
    max_lat = float(np.max(latencies_np))

    # Evaluate Acceptance Criteria
    c_trai = command_metrics["TRAI"]["recall"] >= 0.95
    c_phai = command_metrics["PHAI"]["recall"] >= 0.95
    c_len = command_metrics["LEN"]["recall"] >= 0.95
    c_xuong = command_metrics["XUONG"]["recall"] >= 0.95
    c_cmd_macro = macro_cmd_recall >= 0.95
    c_far = overall_far <= 0.05
    c_acc = overall_accuracy >= 0.95
    c_latency = (mean_lat <= 50.0) and (p95_lat <= 50.0)

    all_criteria_passed = (
        c_trai and c_phai and c_len and c_xuong and
        c_cmd_macro and c_far and c_acc and c_latency
    )

    verdict_status = "CONFIRMATORY_CRITERIA_PASSED" if all_criteria_passed else "CONFIRMATORY_CRITERIA_FAILED"

    metrics_summary = {
        "verdict": verdict_status,
        "all_criteria_passed": all_criteria_passed,
        "eval_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_sha256": actual_sha,
        "manifest_csv_sha256": manifest_sha,
        "operating_threshold": DEFAULT_THETA,
        "total_clips": total_clips,
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_count": f"{total_correct}/{total_clips}",
        "overall_accuracy_ci_95": [acc_low, acc_high],
        "aggregate_command_recall": macro_cmd_recall,
        "aggregate_command_recall_count": f"{total_cmd_tp}/{total_cmd_count}",
        "aggregate_command_recall_ci_95": [mrec_low, mrec_high],
        "command_metrics": command_metrics,
        "non_target_far": overall_far,
        "non_target_false_alarms": f"{total_false_alarms}/{total_non_targets}",
        "non_target_far_ci_95": [far_low, far_high],
        "category_far_breakdown": category_far,
        "confusion_matrix": cm,
        "cpu_latency_ms": {
            "mean": round(mean_lat, 2),
            "median": round(median_lat, 2),
            "p95": round(p95_lat, 2),
            "max": round(max_lat, 2)
        },
        "acceptance_criteria_evaluation": {
            "trai_recall_ge_95": {"actual": command_metrics["TRAI"]["recall"], "target": 0.95, "passed": c_trai},
            "phai_recall_ge_95": {"actual": command_metrics["PHAI"]["recall"], "target": 0.95, "passed": c_phai},
            "len_recall_ge_95": {"actual": command_metrics["LEN"]["recall"], "target": 0.95, "passed": c_len},
            "xuong_recall_ge_95": {"actual": command_metrics["XUONG"]["recall"], "target": 0.95, "passed": c_xuong},
            "macro_cmd_recall_ge_95": {"actual": macro_cmd_recall, "target": 0.95, "passed": c_cmd_macro},
            "non_target_far_le_05": {"actual": overall_far, "target": 0.05, "passed": c_far},
            "overall_accuracy_ge_95": {"actual": overall_accuracy, "target": 0.95, "passed": c_acc},
            "latency_mean_p95_le_50ms": {"mean_actual": round(mean_lat, 2), "p95_actual": round(p95_lat, 2), "target": 50.0, "passed": c_latency}
        }
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metrics JSON written: {METRICS_JSON.name}")

    print("\n" + "=" * 80)
    print("PHASE 13.3 EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy:           {overall_accuracy*100:.2f}% ({total_correct}/{total_clips}) [CI: {acc_low*100:.1f}% - {acc_high*100:.1f}%]")
    print(f"Aggregate Command Recall:   {macro_cmd_recall*100:.2f}% ({total_cmd_tp}/{total_cmd_count}) [CI: {mrec_low*100:.1f}% - {mrec_high*100:.1f}%]")
    print(f"  - TRAI Recall:            {command_metrics['TRAI']['recall']*100:.2f}% ({command_metrics['TRAI']['true_positives']}/50)")
    print(f"  - PHAI Recall:            {command_metrics['PHAI']['recall']*100:.2f}% ({command_metrics['PHAI']['true_positives']}/50)")
    print(f"  - LEN Recall:             {command_metrics['LEN']['recall']*100:.2f}% ({command_metrics['LEN']['true_positives']}/50)")
    print(f"  - XUONG Recall:           {command_metrics['XUONG']['recall']*100:.2f}% ({command_metrics['XUONG']['true_positives']}/50)")
    print(f"Non-Target False Alarm Rate: {overall_far*100:.2f}% ({total_false_alarms}/{total_non_targets}) [CI: {far_low*100:.1f}% - {far_high*100:.1f}%]")
    print(f"  - Phonetic Foils FAR:     {category_far['phonetic_minimal_pairs']['far']*100:.2f}% ({category_far['phonetic_minimal_pairs']['false_accepts']}/200)")
    print(f"  - Carrier Collisions FAR: {category_far['syntactic_carrier_collisions']['far']*100:.2f}% ({category_far['syntactic_carrier_collisions']['false_accepts']}/200)")
    print(f"  - Conversational FAR:     {category_far['general_conversational_non_targets']['far']*100:.2f}% ({category_far['general_conversational_non_targets']['false_accepts']}/300)")
    print(f"  - Background/Silence FAR: {category_far['acoustic_background_and_silence']['far']*100:.2f}% ({category_far['acoustic_background_and_silence']['false_accepts']}/100)")
    print(f"CPU Latency (Single Clip):  Mean={mean_lat:.2f}ms, p95={p95_lat:.2f}ms, Max={max_lat:.2f}ms")
    print(f"\nFinal Confirmatory Verdict: {verdict_status}")
    print("=" * 80)

    return metrics_summary


if __name__ == "__main__":
    run_oneshot_evaluation()
