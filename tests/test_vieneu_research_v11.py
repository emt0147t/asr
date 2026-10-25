#!/usr/bin/env python3
"""
Unit and Regression Test Suite for Vietnamese KWS Research Phase 11.
Workspace: D:\trainidentificationaudio
Target: tests/test_vieneu_research_v11.py
"""

import os
import sys
import csv
import json
import hashlib
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V11_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11"
DATA_DIR = V11_DIR / "data" / "splits"
AUDIT_DIR = V11_DIR / "audit"
CKPT_DIR = V11_DIR / "checkpoints"

BASELINE_CKPT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v9" / "artifacts" / "checkpoints" / "variant_w10_seed42.pt"
BASELINE_HASH = "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed"

P90_RESP_CSV = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review" / "responses" / "reviewer_01_responses.csv"
P90_RESP_HASH = "d8f9d6414e21183ca6417cb04a611ef0d0a8082987bcce54c4e704953c8cf458"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_historical_checkpoint_and_response_immutability():
    """Verifies baseline checkpoint and Phase 90 response hashes remain uncompromised."""
    assert BASELINE_CKPT.exists(), f"Missing baseline checkpoint: {BASELINE_CKPT}"
    h_ckpt = sha256_of_file(BASELINE_CKPT)
    assert h_ckpt.lower() == BASELINE_HASH.lower(), f"Baseline checkpoint hash mismatch: {h_ckpt}"

    assert P90_RESP_CSV.exists(), f"Missing Phase 90 response file: {P90_RESP_CSV}"
    h_resp = sha256_of_file(P90_RESP_CSV)
    assert h_resp.lower() == P90_RESP_HASH.lower(), f"Phase 90 response hash mismatch: {h_resp}"


def test_stage_a_audit_reproduction_denominators():
    """Verifies reproduced baseline counts and denominators from Stage A."""
    summary_path = AUDIT_DIR / "stage_a_audit_summary.json"
    assert summary_path.exists(), f"Missing Stage A summary: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    denoms = summary["reproduced_baseline_denominators"]
    assert denoms["total_clips"] == 120
    assert denoms["target_command_clips"] == 40
    assert denoms["target_per_command"]["TRAI"] == 10
    assert denoms["target_per_command"]["PHAI"] == 10
    assert denoms["target_per_command"]["LEN"] == 10
    assert denoms["target_per_command"]["XUONG"] == 10
    assert denoms["non_target_clips"] == 80
    assert denoms["hard_negative_clips"] == 65
    assert denoms["silence_clips"] == 15

    metrics = summary["reproduced_baseline_metrics"]
    assert metrics["mean_command_recall"] == 0.9750
    assert metrics["per_command_recall"]["TRAI"] == 0.90
    assert metrics["per_command_recall"]["PHAI"] == 1.00
    assert metrics["per_command_recall"]["LEN"] == 1.00
    assert metrics["per_command_recall"]["XUONG"] == 1.00
    assert metrics["non_target_far"] == 0.1125
    assert metrics["non_target_false_accepts_count"] == 9
    assert metrics["overall_accuracy"] == 0.9167
    assert metrics["total_correct"] == 110


def test_stage_a_error_analysis_classification():
    """Verifies that the error analysis CSV exists and accurately classifies the 10 errors."""
    err_csv = AUDIT_DIR / "phase90_baseline_error_analysis.csv"
    assert err_csv.exists(), f"Missing error analysis CSV: {err_csv}"
    with open(err_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120

    errors = [r for r in rows if r["is_correct"].strip().lower() == "false"]
    assert len(errors) == 10, f"Expected exactly 10 baseline errors, got {len(errors)}"

    # Check that error types include target-prefix false accepts, tonal minimal pairs, lexical confusion, and command substitution
    err_types = {e["error_type"] for e in errors}
    assert "TARGET_PREFIX_FALSE_ACCEPT" in err_types
    assert "TONAL_MINIMAL_PAIR_FALSE_ACCEPT" in err_types
    assert "COMMAND_SUBSTITUTION" in err_types


def test_split_leakage_and_template_isolation():
    """Ensures test, val, and train splits have zero text leakage and disjoint voices."""
    test_manifest = DATA_DIR / "test_manifest.csv"
    val_manifest = DATA_DIR / "val_manifest.csv"
    train_manifest = DATA_DIR / "train_manifest.csv"

    if not (test_manifest.exists() and val_manifest.exists() and train_manifest.exists()):
        pytest.skip("Dataset manifests not yet generated.")

    with open(test_manifest, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(train_manifest, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    # Extract non-target texts (excluding commands and silence)
    test_nt_texts = {r["transcription"].strip().lower() for r in test_rows if r["bounded_label"] == "REJECT" and r["transcription"] != "[silence]"}
    val_nt_texts = {r["transcription"].strip().lower() for r in val_rows if r["bounded_label"] == "REJECT" and r["transcription"] != "[silence]"}
    train_nt_texts = {r["transcription"].strip().lower() for r in train_rows if r["bounded_label"] == "REJECT" and r["transcription"] != "[silence]"}

    # Assert 0 leakage
    assert len(test_nt_texts.intersection(val_nt_texts)) == 0, "Test-Val text leakage!"
    assert len(test_nt_texts.intersection(train_nt_texts)) == 0, "Test-Train text leakage!"
    assert len(val_nt_texts.intersection(train_nt_texts)) == 0, "Val-Train text leakage!"

    # Voice isolation for test
    test_voices = {r["voice_id"] for r in test_rows if r["canonical_class"] != "SILENCE"}
    train_voices = {r["voice_id"] for r in train_rows if r["canonical_class"] != "SILENCE"}
    assert len(test_voices.intersection(train_voices)) == 0, "Test voices leaked into train!"


def test_duplicate_sample_ids_and_paths():
    """Ensures sample IDs and audio paths are strictly unique within each split."""
    for split in ["test", "val", "train"]:
        manifest_path = DATA_DIR / f"{split}_manifest.csv"
        if not manifest_path.exists():
            pytest.skip(f"Manifest {manifest_path} not ready.")
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        sids = [r["sample_id"] for r in rows]
        assert len(sids) == len(set(sids)), f"Duplicate sample_ids found in {split} split!"

        paths = [r["kws_wav_path"] for r in rows]
        assert len(paths) == len(set(paths)), f"Duplicate audio paths found in {split} split!"


def test_label_mapping_contracts():
    """Ensures bounded label policies adhere to the bounded-utterance specification."""
    for split in ["test", "val", "train"]:
        manifest_path = DATA_DIR / f"{split}_manifest.csv"
        if not manifest_path.exists():
            pytest.skip(f"Manifest {manifest_path} not ready.")
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for r in rows:
            gt_label = r["bounded_label"]
            gt_id = int(r["bounded_class_id"])
            if gt_label in ["TRAI", "PHAI", "LEN", "XUONG"]:
                assert gt_id in [0, 1, 2, 3]
            else:
                assert gt_label == "REJECT"
                assert gt_id == 4


def test_v11_governance_invariants():
    """Guarantees that no forbidden literal strings or raw recordings are referenced in v11 files."""
    forbidden = "data/" + "raw/" + "field_recordings"
    for p in V11_DIR.rglob("*"):
        if p.is_file() and p.suffix in [".py", ".md", ".json"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert forbidden not in text, f"Forbidden path string detected in {p}"


def test_deterministic_evaluation():
    """Verifies that running inference twice on identical audio yields byte-level deterministic outputs."""
    from experiments.vieneu_kws_research_v10.deployment.predict_wav import PredictorEngine
    engine = PredictorEngine(checkpoint_path=BASELINE_CKPT, theta_cmd=0.40)

    test_wav = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review" / "audio" / "review_clip_001.wav"
    assert test_wav.exists(), f"Missing test clip: {test_wav}"

    pred1 = engine.predict_wav(test_wav, strict=False)
    pred2 = engine.predict_wav(test_wav, strict=False)

    assert pred1["predicted_class"] == pred2["predicted_class"]
    assert pred1["confidence"] == pred2["confidence"]
    assert pred1["probabilities"] == pred2["probabilities"]


def test_locked_test_denominators_if_ready():
    """Verifies fresh locked test split has at least 400 non-target clips and 160 command clips."""
    test_manifest = DATA_DIR / "test_manifest.csv"
    if not test_manifest.exists():
        pytest.skip("Test manifest not yet generated.")
    with open(test_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 640
    cmd_count = sum(1 for r in rows if r["bounded_label"] != "REJECT")
    nt_count = sum(1 for r in rows if r["bounded_label"] == "REJECT")
    assert cmd_count == 160
    assert nt_count == 480
    assert nt_count >= 400, "Non-target test set must have at least 400 clips!"
