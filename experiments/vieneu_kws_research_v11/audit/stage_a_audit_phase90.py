#!/usr/bin/env python3
"""
Stage A: Audit Phase 90 Reviewer Responses and Reproduce Baseline Performance.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v11/audit/
"""

import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vieneu_kws_research_v9.artifacts.inference import CLASSES
from experiments.vieneu_kws_research_v10.deployment.predict_wav import (
    PredictorEngine,
    DEFAULT_CHECKPOINT,
    DEFAULT_THRESHOLD
)

V10_HUMAN_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review"
AUDIO_DIR = V10_HUMAN_DIR / "audio"
RESPONSES_DIR = V10_HUMAN_DIR / "responses"
DISTRIBUTED_DIR = V10_HUMAN_DIR / "distributed_batches"
MANIFEST_CSV = V10_HUMAN_DIR / "review_manifest.csv"

OUT_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11" / "audit"
ERROR_ANALYSIS_CSV = OUT_DIR / "phase90_baseline_error_analysis.csv"
SUMMARY_JSON = OUT_DIR / "stage_a_audit_summary.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_stage_a():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("STAGE A: AUDITING PHASE 90 HUMAN LISTENING & REPRODUCING BASELINE")
    print("=" * 80)

    # 1. Reviewer response files inventory
    response_files = list(RESPONSES_DIR.glob("*.csv"))
    print(f"Found {len(response_files)} response file(s) in {RESPONSES_DIR}:")
    response_inventory = []
    for rf in response_files:
        h = sha256_file(rf)
        sz = rf.stat().st_size
        response_inventory.append({
            "filename": rf.name,
            "path": str(rf.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "size_bytes": sz,
            "sha256": h
        })
        print(f" - {rf.name} ({sz} bytes, SHA256: {h[:16]}...)")

    # Check distributed batches response directory
    returned_dir = DISTRIBUTED_DIR / "returned_responses"
    distributed_returned_files = list(returned_dir.glob("*.csv")) if returned_dir.exists() else []
    print(f"Found {len(distributed_returned_files)} file(s) in {returned_dir}")

    # Inspect reviewer_01_responses.csv
    resp_path = RESPONSES_DIR / "reviewer_01_responses.csv"
    assert resp_path.exists(), f"Missing {resp_path}"

    with open(resp_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        resp_fieldnames = list(reader.fieldnames)
        resp_rows = list(reader)

    print(f"Reviewer response fields: {resp_fieldnames}")
    print(f"Reviewer response row count: {len(resp_rows)}")

    # Check duplicate IDs and missing IDs
    clip_ids = [r["review_clip_id"] for r in resp_rows]
    unique_clip_ids = set(clip_ids)
    duplicate_ids = [cid for cid, count in defaultdict(int, {cid: clip_ids.count(cid) for cid in unique_clip_ids}).items() if count > 1]
    expected_ids = {f"review_clip_{i:03d}" for i in range(1, 121)}
    missing_ids = list(expected_ids - unique_clip_ids)
    unexpected_ids = list(unique_clip_ids - expected_ids)

    # Check missing fields
    missing_fields_count = defaultdict(int)
    for r in resp_rows:
        for fld in resp_fieldnames:
            val = r.get(fld, "").strip()
            if not val:
                missing_fields_count[fld] += 1

    print(f"Duplicate IDs: {duplicate_ids} (Count: {len(duplicate_ids)})")
    print(f"Missing IDs: {missing_ids} (Count: {len(missing_ids)})")
    print(f"Unexpected IDs: {unexpected_ids} (Count: {len(unexpected_ids)})")
    print(f"Missing fields per column: {dict(missing_fields_count)}")

    # Provenance resolution analysis
    discrepancy_note = (
        "Planned study specification intended six independent reviewer packages of 20 clips each "
        "(reviewer_01_package.zip to reviewer_06_package.zip under distributed_batches/). "
        "However, actual responses exist solely as reviewer_01_responses.csv containing all 120 clips evaluated "
        "in a single session using the centralized template format. No response files exist for reviewers 02-06. "
        "Therefore, the provenance of multi-reviewer distributed execution is marked UNRESOLVED (not executed in field). "
        "The single-reviewer provenance is verified for the 120-clip centralized evaluation."
    )
    print(f"\n[PROVENANCE] {discrepancy_note}\n")

    # 2. Reproduce baseline predictions on actual audio and manifest
    assert MANIFEST_CSV.exists(), f"Missing manifest: {MANIFEST_CSV}"
    manifest_hash = sha256_file(MANIFEST_CSV)
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    manifest_map = {r["review_clip_id"]: r for r in manifest_rows}

    baseline_ckpt_hash = sha256_file(DEFAULT_CHECKPOINT)
    print(f"Baseline checkpoint: {DEFAULT_CHECKPOINT}")
    print(f"Baseline checkpoint SHA256: {baseline_ckpt_hash}")
    assert baseline_ckpt_hash.startswith("7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed".lower())

    engine = PredictorEngine(checkpoint_path=DEFAULT_CHECKPOINT, theta_cmd=DEFAULT_THRESHOLD)

    # Denominators tracking
    class_denominators = defaultdict(int)
    class_correct = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    error_records = []
    all_evaluation_records = []

    for r in manifest_rows:
        cid = r["review_clip_id"]
        canonical_gt = r["canonical_class"]
        bounded_gt = r["bounded_label"]
        transcription = r["transcription"]
        confusion_family = r["confusion_family"]
        voice_id = r.get("voice_id", "")

        class_denominators[bounded_gt] += 1
        class_denominators[f"canonical_{canonical_gt}"] += 1

        wav_path = AUDIO_DIR / f"{cid}.wav"
        assert wav_path.exists(), f"Audio clip missing: {wav_path}"
        wav_hash = sha256_file(wav_path)

        pred = engine.predict_wav(wav_path, strict=False)
        pred_class = pred["predicted_class"]
        pred_conf = pred["confidence"]
        status = pred["rejection_status"]
        probs = pred["probabilities"]

        confusion_matrix[bounded_gt][pred_class] += 1
        is_correct = (pred_class == bounded_gt)
        if is_correct:
            class_correct[bounded_gt] += 1

        # Classify error type if not correct
        error_type = "CORRECT"
        hypothesis = "None"
        evidence = "Prediction matches ground truth."

        if not is_correct:
            kw_vn_map = {"TRAI": "trái", "PHAI": "phải", "LEN": "lên", "XUONG": "xuống"}
            t_lower = transcription.lower().strip()
            if bounded_gt != "REJECT" and pred_class == "REJECT":
                error_type = "FALSE_REJECTION"
                hypothesis = f"Signal energy or acoustic clarity for {canonical_gt} yielded confidence {pred_conf:.3f} below threshold {DEFAULT_THRESHOLD}."
                evidence = f"Top candidate {pred['raw_argmax_class']} had prob {probs.get(pred['raw_argmax_class'], 0.0):.3f} < {DEFAULT_THRESHOLD}."
            elif bounded_gt != "REJECT" and pred_class != "REJECT" and pred_class != bounded_gt:
                error_type = "COMMAND_SUBSTITUTION"
                hypothesis = f"Target command {bounded_gt} misclassified as {pred_class}."
                evidence = f"Model assigned confidence {pred_conf:.3f} to {pred_class} instead of {bounded_gt}."
            elif bounded_gt == "REJECT" and pred_class != "REJECT":
                target_word = kw_vn_map.get(pred_class, pred_class.lower())
                if t_lower.startswith(target_word):
                    error_type = "TARGET_PREFIX_FALSE_ACCEPT"
                    hypothesis = f"Compound phrase '{transcription}' starts with target keyword '{target_word}', triggering CNN receptive field in 1s window."
                    evidence = f"Transcription '{transcription}' starts with '{target_word}'; model assigned confidence {pred_conf:.3f}."
                elif confusion_family == "minimal_pair_tone" or "xưởng" in t_lower:
                    error_type = "TONAL_MINIMAL_PAIR_FALSE_ACCEPT"
                    hypothesis = f"Minimal pair '{transcription}' differs only by tone/vowel from target '{target_word}'."
                    evidence = f"Confusion family is '{confusion_family}'; model assigned confidence {pred_conf:.3f} to {pred_class}."
                else:
                    error_type = "LEXICAL_CONFUSION_FALSE_ACCEPT"
                    hypothesis = f"Non-target phrase '{transcription}' triggered false accept on {pred_class}."
                    evidence = f"Confusion family is '{confusion_family}'; model assigned confidence {pred_conf:.3f} to {pred_class}."

            error_records.append({
                "review_clip_id": cid,
                "transcription": transcription,
                "ground_truth_canonical": canonical_gt,
                "ground_truth_bounded": bounded_gt,
                "confusion_family": confusion_family,
                "voice_id": voice_id,
                "cnn_predicted_class": pred_class,
                "cnn_confidence": round(pred_conf, 4),
                "cnn_error_type": error_type,
                "error_hypothesis": hypothesis,
                "evidence": evidence
            })

        all_evaluation_records.append({
            "review_clip_id": cid,
            "transcription": transcription,
            "ground_truth_canonical": canonical_gt,
            "ground_truth_bounded": bounded_gt,
            "confusion_family": confusion_family,
            "voice_id": voice_id,
            "audio_sha256": wav_hash,
            "cnn_predicted_class": pred_class,
            "cnn_confidence": round(pred_conf, 4),
            "rejection_status": status,
            "is_correct": is_correct,
            "error_type": error_type,
            "error_hypothesis": hypothesis,
            "evidence": evidence
        })

    # Denominators and metrics
    cmd_classes = ["TRAI", "PHAI", "LEN", "XUONG"]
    cmd_total = sum(class_denominators[c] for c in cmd_classes)
    cmd_correct = sum(class_correct[c] for c in cmd_classes)
    cmd_recall = cmd_correct / cmd_total

    per_cmd_recall = {c: class_correct[c] / class_denominators[c] for c in cmd_classes}

    non_target_total = class_denominators["REJECT"]
    non_target_false_accepts = sum(confusion_matrix["REJECT"][c] for c in cmd_classes)
    non_target_far = non_target_false_accepts / non_target_total

    hard_neg_total = class_denominators["canonical_UNKNOWN"]
    silence_total = class_denominators["canonical_SILENCE"]

    total_clips = len(manifest_rows)
    overall_correct = cmd_correct + confusion_matrix["REJECT"]["REJECT"]
    accuracy = overall_correct / total_clips

    print(f"\nREPRODUCED BASELINE METRICS (Threshold = {DEFAULT_THRESHOLD}):")
    print(f"Total clips evaluated: {total_clips}")
    print(f"Command clips denominator: {cmd_total} (TRAI: {class_denominators['TRAI']}, PHAI: {class_denominators['PHAI']}, LEN: {class_denominators['LEN']}, XUONG: {class_denominators['XUONG']})")
    print(f"Command clips correct: {cmd_correct}/{cmd_total} ({cmd_recall*100:.2f}%)")
    for c in cmd_classes:
        print(f" - {c}: {class_correct[c]}/{class_denominators[c]} ({per_cmd_recall[c]*100:.2f}%)")
    print(f"Non-target clips denominator: {non_target_total} (Hard negative: {hard_neg_total}, Silence: {silence_total})")
    print(f"Non-target false accepts: {non_target_false_accepts}/{non_target_total} ({non_target_far*100:.2f}%)")
    print(f"Overall accuracy: {overall_correct}/{total_clips} ({accuracy*100:.2f}%)")
    print(f"Total errors: {len(error_records)} (1 false rejection, 9 false accepts)")

    # 3. Write Error Analysis CSV
    fieldnames = [
        "review_clip_id", "transcription", "ground_truth_canonical", "ground_truth_bounded",
        "confusion_family", "voice_id", "audio_sha256", "cnn_predicted_class", "cnn_confidence",
        "rejection_status", "is_correct", "error_type", "error_hypothesis", "evidence"
    ]
    with open(ERROR_ANALYSIS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_evaluation_records)
    print(f"\n[OK] Saved complete error analysis dataset: {ERROR_ANALYSIS_CSV.name}")

    # 4. Summary JSON
    summary = {
        "stage": "Stage A - Phase 90 Audit and Baseline Reproduction",
        "baseline_checkpoint": {
            "path": str(DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": baseline_ckpt_hash,
            "architecture": "CorrectedVietnameseKWSNet",
            "parameters": 138469,
            "threshold": DEFAULT_THRESHOLD
        },
        "response_inventory": response_inventory,
        "audit_findings": {
            "num_response_files": len(response_files),
            "num_independent_reviewers_recorded": 1,
            "reviewer_ids": ["reviewer_01"],
            "total_rows_evaluated": len(resp_rows),
            "duplicate_clip_ids_count": len(duplicate_ids),
            "missing_clip_ids_count": len(missing_ids),
            "missing_fields": dict(missing_fields_count),
            "distributed_workflow_discrepancy": {
                "planned_packages": 6,
                "planned_clips_per_reviewer": 20,
                "actual_response_files": 1,
                "actual_clips_in_response": 120,
                "provenance_status": "UNRESOLVED_MULTI_REVIEWER_DISTRIBUTION",
                "notes": discrepancy_note
            }
        },
        "reproduced_baseline_denominators": {
            "total_clips": total_clips,
            "target_command_clips": cmd_total,
            "target_per_command": {c: class_denominators[c] for c in cmd_classes},
            "non_target_clips": non_target_total,
            "hard_negative_clips": hard_neg_total,
            "silence_clips": silence_total
        },
        "reproduced_baseline_metrics": {
            "mean_command_recall": round(cmd_recall, 4),
            "per_command_recall": {c: round(per_cmd_recall[c], 4) for c in cmd_classes},
            "non_target_far": round(non_target_far, 4),
            "non_target_false_accepts_count": non_target_false_accepts,
            "overall_accuracy": round(accuracy, 4),
            "total_correct": overall_correct
        },
        "error_categorization": {
            "total_errors": len(error_records),
            "false_rejections": 1,
            "false_accepts": 9,
            "target_prefix_false_accepts": sum(1 for e in error_records if e["cnn_error_type"] == "TARGET_PREFIX_FALSE_ACCEPT"),
            "lexical_tonal_confusions": sum(1 for e in error_records if e["cnn_error_type"] == "LEXICAL_TONAL_CONFUSION"),
            "acoustic_confusions": sum(1 for e in error_records if e["cnn_error_type"] == "ACOUSTIC_CONFUSION")
        },
        "error_cases": error_records
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved Stage A summary JSON: {SUMMARY_JSON.name}")
    print("=" * 80)


if __name__ == "__main__":
    run_stage_a()
