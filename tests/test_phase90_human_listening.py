"""
Unit and Regression Test Suite for Phase 90: Human-Listening Validation.
Workspace: D:\trainidentificationaudio
Validates:
- Integrity and completeness of submitted human listening responses (120 clips)
- Human perceptual ratings (Intelligibility MOS, Naturalness MOS, Pronunciation Correctness)
- Human keyword spotting performance (Command Recall >= 90%, FAR <= 2%)
- Frozen CNN candidate performance on blinded review clips
- Human-CNN concordance and Cohen's Kappa (kappa >= 0.75)
- Governance status updates: HUMAN_LISTENER_VERIFIED while retaining NOT_VALIDATED_ON_NATURAL_SPEECH
"""

import os
import sys
import csv
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HUMAN_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review"
SUMMARY_JSON = HUMAN_DIR / "phase90_review_quality_summary.json"
JOINED_CSV = HUMAN_DIR / "phase90_human_adjudication.csv"
VALIDATED_CSV = HUMAN_DIR / "phase90_validated_responses.csv"
RESPONSES_CSV = HUMAN_DIR / "responses" / "reviewer_01_responses.csv"


def test_phase90_intake_and_manifest_integrity():
    """Verify that all 120 clips have completed responses and valid adjudication mapping."""
    assert RESPONSES_CSV.exists(), "reviewer_01_responses.csv missing"
    assert VALIDATED_CSV.exists(), "phase90_validated_responses.csv missing"
    assert JOINED_CSV.exists(), "phase90_human_adjudication.csv missing"
    assert SUMMARY_JSON.exists(), "phase90_review_quality_summary.json missing"

    with open(RESPONSES_CSV, "r", encoding="utf-8-sig") as f:
        resp_rows = list(csv.DictReader(f))
    assert len(resp_rows) == 120, f"Expected 120 responses, got {len(resp_rows)}"

    # Confirm all 120 clips have non-empty perceived text
    for r in resp_rows:
        assert r.get("perceived_text", "").strip(), f"Empty perceived_text in clip {r.get('review_clip_id')}"

    with open(JOINED_CSV, "r", encoding="utf-8-sig") as f:
        joined_rows = list(csv.DictReader(f))
    assert len(joined_rows) == 120


def test_phase90_human_perceptual_metrics():
    """Verify that human perceptual ratings demonstrate intelligible, natural synthetic speech."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    mos = summary["human_perceptual_mos"]
    # Overall Intelligibility must be high (>= 3.5 out of 5)
    assert mos["mean_intelligibility"] >= 3.5, f"Low intelligibility: {mos['mean_intelligibility']}"
    # Overall Naturalness must be acceptable (>= 3.5 out of 5)
    assert mos["mean_naturalness"] >= 3.5, f"Low naturalness: {mos['mean_naturalness']}"
    # Pronunciation correctness must be >= 95%
    assert mos["pronunciation_acceptability_pct"] >= 95.0, f"Low pronunciation: {mos['pronunciation_acceptability_pct']}"


def test_phase90_human_kws_accuracy():
    """Verify that human listener achieves high recall on target commands and 0% false alarms."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    h_kws = summary["human_kws_recognition"]
    assert h_kws["command_recall"] >= 0.90, f"Human recall below 90%: {h_kws['command_recall']}"
    assert h_kws["non_target_far"] <= 0.02, f"Human FAR above 2%: {h_kws['non_target_far']}"
    assert h_kws["five_class_accuracy"] >= 0.95, f"Human accuracy below 95%: {h_kws['five_class_accuracy']}"
    assert h_kws["silence_recognition_rate"] == 1.0, "Human listener failed to recognize silence"


def test_phase90_cnn_model_performance():
    """Verify that frozen CNN candidate performs within expected research bounds on review clips."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    cnn = summary["cnn_model_performance"]
    assert cnn["command_recall"] >= 0.95, f"CNN recall below 95%: {cnn['command_recall']}"
    assert cnn["five_class_accuracy"] >= 0.90, f"CNN accuracy below 90%: {cnn['five_class_accuracy']}"


def test_phase90_human_model_concordance():
    """Verify substantial agreement between human listener and CNN model."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    conc = summary["human_model_concordance"]
    assert conc["agreement_rate"] >= 0.85, f"Agreement rate below 85%: {conc['agreement_rate']}"
    assert conc["cohens_kappa"] >= 0.75, f"Cohen's kappa below 0.75: {conc['cohens_kappa']}"


def test_phase90_governance_status():
    """Verify that governance status correctly reflects validated listening while preserving other banners."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    gov = summary["governance_status"]
    assert gov["human_listener_verification"] == "HUMAN_LISTENER_VERIFIED"
    assert gov["natural_speech_validation"] == "NOT_VALIDATED_ON_NATURAL_SPEECH"
    assert gov["continuous_open_mic_verdict"] == "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
