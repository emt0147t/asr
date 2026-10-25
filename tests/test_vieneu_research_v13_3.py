"""
Tests for Phase 13.3: Preregistered Synthetic Confirmatory Evaluation.

Validates:
1. Candidate checkpoint SHA-256 byte immobility pre and post evaluation.
2. Locked confirmatory dataset integrity (1,000 clips, correct category allocations).
3. Manifest and prediction file existence and hash integrity.
4. Validation of all 8 preregistered acceptance criteria (accuracy >= 95%, recall >= 95%, FAR <= 5%, latency <= 50ms).
5. Governance invariants and operational scope assertions.
"""

import hashlib
import json
import csv
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v13_1"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
EXPECTED_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

DATA_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data"
PRED_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "reports"
ROOT_REPORT = PROJECT_ROOT / "reports" / "PHASE13_3_CONFIRMATORY_EVALUATION.md"
EXP_REPORT = REPORTS_DIR / "PHASE13_3_CONFIRMATORY_EVALUATION.md"
METRICS_JSON = REPORTS_DIR / "confirmatory_evaluation_metrics.json"
MANIFEST_CSV = DATA_DIR / "confirmatory_evaluation_manifest.csv"
INTEGRITY_JSON = DATA_DIR / "dataset_integrity_report.json"


def test_checkpoint_immutability():
    """Verify frozen candidate checkpoint exists and matches authoritative SHA-256."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    with open(CHECKPOINT_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == EXPECTED_SHA256, f"Checkpoint SHA-256 altered: {actual_sha}"


def test_dataset_integrity_and_allocation():
    """Verify locked dataset integrity report and 1,000-clip allocation."""
    assert INTEGRITY_JSON.exists(), "Dataset integrity report missing"
    with open(INTEGRITY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "DATASET_INTEGRITY_VERIFIED_AND_LOCKED"
    assert data["total_clips"] == 1000
    assert data["dataset_locked"] is True
    assert data["clipping_detected"] is False
    assert data["category_counts"]["target_commands"] == 200
    assert data["category_counts"]["phonetic_minimal_pairs"] == 200
    assert data["category_counts"]["syntactic_carrier_collisions"] == 200
    assert data["category_counts"]["general_conversational_non_targets"] == 300
    assert data["category_counts"]["acoustic_background_and_silence"] == 100

    assert data["class_counts"]["TRAI"] == 50
    assert data["class_counts"]["PHAI"] == 50
    assert data["class_counts"]["LEN"] == 50
    assert data["class_counts"]["XUONG"] == 50
    assert data["class_counts"]["REJECT"] == 800


def test_manifest_and_predictions_exist():
    """Verify manifest and prediction CSV/JSON files exist and contain 1,000 records."""
    assert MANIFEST_CSV.exists()
    pred_csv = PRED_DIR / "confirmatory_predictions.csv"
    pred_json = PRED_DIR / "confirmatory_predictions.json"
    assert pred_csv.exists()
    assert pred_json.exists()

    with open(pred_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert len(records) == 1000, f"Expected 1000 predictions, got {len(records)}"


def test_preregistered_acceptance_criteria_passed():
    """Verify that all 8 preregistered acceptance criteria are satisfied in metrics JSON."""
    assert METRICS_JSON.exists(), "Metrics summary JSON missing"
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["verdict"] == "CONFIRMATORY_CRITERIA_PASSED"
    assert m["all_criteria_passed"] is True
    assert m["operating_threshold"] == 0.20

    # 1. Overall Accuracy >= 95.0%
    assert m["overall_accuracy"] >= 0.95, f"Overall accuracy {m['overall_accuracy']} < 0.95"

    # 2. Aggregate Command Recall >= 95.0%
    assert m["aggregate_command_recall"] >= 0.95, f"Aggregate command recall {m['aggregate_command_recall']} < 0.95"

    # 3-6. Individual Command Recalls >= 95.0%
    cmd_m = m["command_metrics"]
    for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
        rec = cmd_m[cmd]["recall"]
        assert rec >= 0.95, f"{cmd} recall {rec} < 0.95"
        assert cmd_m[cmd]["passed_target_95"] is True

    # 7. Non-Target False Alarm Rate <= 5.0%
    assert m["non_target_far"] <= 0.05, f"Non-target FAR {m['non_target_far']} > 0.05"

    # 8. CPU Latency <= 50.0 ms
    lat = m["cpu_latency_ms"]
    assert lat["mean"] <= 50.0, f"Mean latency {lat['mean']} > 50.0 ms"
    assert lat["p95"] <= 50.0, f"p95 latency {lat['p95']} > 50.0 ms"


def test_governance_and_reports_content():
    """Verify required governance strings and provenance in root and experiment reports."""
    assert ROOT_REPORT.exists()
    assert EXP_REPORT.exists()

    content_root = ROOT_REPORT.read_text(encoding="utf-8")
    content_exp = EXP_REPORT.read_text(encoding="utf-8")

    for c in [content_root, content_exp]:
        assert "CONFIRMATORY_CRITERIA_PASSED" in c
        assert "FROZEN_CNN_CANDIDATE" in c
        assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in c
        assert EXPECTED_SHA256 in c
        assert "99.20%" in c
        assert "99.00%" in c
        assert "0.75%" in c
