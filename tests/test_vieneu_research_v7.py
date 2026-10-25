"""
Unit and regression tests for Phases 65-69: Vietnamese KWS Research Campaign v7.
Covers:
- Phase 65: Metric reconciliation and data pipeline audit.
- Phase 66: Error diagnosis without test leakage and error taxonomy.
- Phase 67: Controlled training ablation, parameter count, and checkpoint integrity.
- Phase 68: Validation calibration, robustness testing, and operating point status.
- Phase 69: Locked test benchmark evaluation, dual-verdict certifications, and historical immutability.
"""

import os
import sys
import json
import csv
import pytest
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from experiments.vieneu_kws_research_v7.models.train_ablation import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend
)


def test_phase65_reproduced_metrics_and_audit():
    """Verifies that Phase 65 reproduced metrics JSON exists and contains expected keys."""
    rep_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v7", "predictions", "reproduced_metrics.json"
    )
    assert os.path.exists(rep_json), "Phase 65 reproduced metrics JSON must exist"

    with open(rep_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "benchmark_denominators" in data
    assert "per_candidate_seed_audit" in data
    assert "cross_seed_aggregations" in data
    assert "pipeline_defect_investigation" in data

    # Verify benchmark denominators
    denom = data["benchmark_denominators"]
    assert denom["total_test_clips"] == 720
    assert denom["target_command_clips"] == 160
    assert denom["non_target_clips"] == 560
    assert denom["per_keyword_clips"]["TRAI"] == 40

    # Verify defect investigation findings
    defects = data["pipeline_defect_investigation"]
    assert "finding_1_severe_training_class_imbalance" in defects
    assert "finding_2_excessive_gradient_mass_on_reject" in defects
    assert "finding_3_decision_rule_asymmetry" in defects
    assert "finding_4_architecture_and_temporal_pooling" in defects
    assert "finding_5_label_and_audio_integrity" in defects


def test_phase65_reports_and_markers():
    """Verifies that Phase 65 reports exist and have mandatory status markers."""
    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "reports")
    rep1 = os.path.join(rep_dir, "PHASE65_METRIC_AUDIT.md")
    rep2 = os.path.join(rep_dir, "PHASE65_DATA_PIPELINE_AUDIT.md")

    assert os.path.exists(rep1), "Phase 65 metric audit report must exist"
    assert os.path.exists(rep2), "Phase 65 data pipeline audit report must exist"

    for p in [rep1, rep2]:
        with open(p, "r", encoding="utf-8") as f:
            c = f.read()
            assert "PENDING_HUMAN_LISTENER_VERIFICATION" in c
            assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c


def test_phase66_error_taxonomy_and_diagnosis():
    """Verifies Phase 66 error taxonomy CSV and error analysis JSON."""
    tax_csv = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v7", "data", "error_taxonomy.csv"
    )
    assert os.path.exists(tax_csv), "Phase 66 error taxonomy CSV must exist"

    rows = []
    with open(tax_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    assert len(rows) > 100, f"Expected >100 error entries, got {len(rows)}"

    diag_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v7", "predictions", "train_val_error_analysis.json"
    )
    assert os.path.exists(diag_json), "Phase 66 error analysis JSON must exist"

    diag_report = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v7", "reports", "PHASE66_ERROR_DIAGNOSIS.md"
    )
    assert os.path.exists(diag_report), "Phase 66 error diagnosis report must exist"


def test_phase67_model_architecture_and_parameters():
    """Verifies CorrectedVietnameseKWSNet parameter count and tensor shapes."""
    model = CorrectedVietnameseKWSNet(num_classes=5)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 138469, f"Expected 138,469 parameters, got {total_params}"

    frontend = LogMelFrontend()
    dummy_wav = torch.randn(2, 16000)
    spec = frontend(dummy_wav)
    assert spec.shape == (2, 1, 40, 101), f"Expected spec shape (2, 1, 40, 101), got {spec.shape}"

    out = model(spec)
    assert out.shape == (2, 5), f"Expected output shape (2, 5), got {out.shape}"


def test_phase67_checkpoints_and_summary():
    """Verifies that all 6 checkpoints and training summary exist."""
    ckpt_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "models", "checkpoints")
    for cand in ["candidate_a", "candidate_b"]:
        for seed in [42, 43, 44]:
            p = os.path.join(ckpt_dir, f"{cand}_seed{seed}.pt")
            assert os.path.exists(p), f"Checkpoint {p} must exist"

    train_sum = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "models", "training_summary.json")
    assert os.path.exists(train_sum), "Training summary JSON must exist"
    with open(train_sum, "r", encoding="utf-8") as f:
        ts = json.load(f)
    assert "candidate_a_corrected_baseline" in ts and "candidate_b_moderately_weighted_hard_negative" in ts



def test_phase68_calibration_and_report():
    """Verifies Phase 68 calibration JSON and markdown report."""
    calib_json = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "calibration", "selected_configs.json")
    assert os.path.exists(calib_json), "Selected configs JSON must exist"

    calib_rep = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "reports", "PHASE68_CALIBRATION.md")
    assert os.path.exists(calib_rep), "Phase 68 calibration report must exist"

    with open(calib_rep, "r", encoding="utf-8") as f:
        c = f.read()
        assert "PENDING_HUMAN_LISTENER_VERIFICATION" in c
        assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c


def test_phase69_locked_test_evaluation_and_reports():
    """Verifies Phase 69 locked test summary, per-clip predictions, and dual operational reports."""
    test_sum = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "predictions", "locked_test_summary.json")
    assert os.path.exists(test_sum), "Locked test summary JSON must exist"

    with open(test_sum, "r", encoding="utf-8") as f:
        ts = json.load(f)

    for cand in ["variant_a_cascade_v1", "candidate_a_corrected_baseline", "candidate_b_moderately_weighted_hard_negative"]:
        assert cand in ts
        agg = ts[cand]["cross_seed_summary"]
        assert "mean_command_recall" in agg
        assert "mean_false_acceptance_rate" in agg
        assert agg["mean_cpu_latency_ms"] < 50.0

    pred_csv = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "predictions", "per_clip_predictions.csv")
    assert os.path.exists(pred_csv), "Per-clip predictions CSV must exist"
    with open(pred_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # 3 candidates * 3 seeds * 720 clips = 6480 rows
    assert len(rows) == 6480, f"Expected 6480 prediction rows, got {len(rows)}"

    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7", "reports")
    for rname in ["PHASE69_LOCKED_EVALUATION.md", "PHASE69_FINAL_DECISION.md", "PHASE69_READINESS_MATRIX.md"]:
        rpath = os.path.join(rep_dir, rname)
        assert os.path.exists(rpath), f"Report {rname} must exist"
        with open(rpath, "r", encoding="utf-8") as f:
            c = f.read()
            assert "PENDING_HUMAN_LISTENER_VERIFICATION" in c
            assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c

    # Verify Mode B verdict in final decision
    with open(os.path.join(rep_dir, "PHASE69_FINAL_DECISION.md"), "r", encoding="utf-8") as f:
        dec_text = f.read()
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in dec_text
        assert "NO_FEASIBLE_VALIDATION_OPERATING_POINT" in dec_text


def test_historical_immutability():
    """Ensures historical experiment directories and raw human data remain untouched and unreferenced."""
    # Ensure no file in experiments/vieneu_kws_research_v7 references raw field recordings
    v7_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7")
    for root, _, files in os.walk(v7_dir):
        for f in files:
            if f.endswith((".py", ".json", ".md", ".csv")):
                fpath = os.path.join(root, f)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                    assert "field_recordings" not in content, f"{fpath} must not reference field_recordings"

    historical_dirs = [
        os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_1000"),
        os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class"),
        os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_vad_ablation_v1"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v2"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v3"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6"),
    ]

    for hdir in historical_dirs:
        assert os.path.exists(hdir), f"Historical directory {hdir} must exist and be preserved"

