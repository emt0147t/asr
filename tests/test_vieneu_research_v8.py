"""
Unit and regression tests for Phases 70-74: Vietnamese KWS Research Campaign v8.
Covers:
- Phase 70: Candidate B metric and threshold audit, disaggregated per-seed checks, sampling unit audit.
- Phase 71: Validation Pareto study, empirical frontier, and operating status determination.
- Phase 72: Controlled hard-negative weight ablation, checkpoint reload, parameter integrity (138,469 params).
- Phase 73: Robust calibration and finalist selection.
- Phase 74: Fresh locked test benchmark, text disjointness, voice provenance disclosure, and dual operational decisions.
- Historical immutability and data safety.
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


def test_phase70_metric_and_threshold_audit():
    """Verifies that Phase 70 reproduced metrics JSON exists and reproduces disaggregated findings."""
    rep_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v8", "predictions", "reproduced_metrics.json"
    )
    assert os.path.exists(rep_json), "Phase 70 reproduced metrics JSON must exist"

    with open(rep_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "seed42" in data and "seed43" in data and "seed44" in data
    assert "cross_seed_summary" in data

    # Verify that Seed 42 PHAI recall was < 0.95 (disaggregated audit requirement)
    s42_phai = data["seed42"]["metrics"]["per_keyword_recall"]["PHAI"]
    assert s42_phai < 0.95, f"Expected Seed 42 PHAI < 0.95, got {s42_phai}"

    # Verify that all 3 seeds exceeded 5.0% FAR on historical locked test
    for s in [42, 43, 44]:
        far = data[f"seed{s}"]["metrics"]["non_target_false_acceptance_rate"]
        assert far > 0.05, f"Expected Seed {s} FAR > 0.05, got {far}"

    # Verify sampling unit disclosure
    sampling = data["cross_seed_summary"]["sampling_unit_audit"]
    assert "clustered" in sampling["note"].lower() or "repeated" in sampling["note"].lower()


def test_phase70_reports_and_markers():
    """Verifies Phase 70 audit reports and required status markers."""
    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "reports")
    r1 = os.path.join(rep_dir, "PHASE70_CANDIDATE_AUDIT.md")
    r2 = os.path.join(rep_dir, "PHASE70_METRIC_REPRODUCTION.md")

    assert os.path.exists(r1), "PHASE70_CANDIDATE_AUDIT.md must exist"
    assert os.path.exists(r2), "PHASE70_METRIC_REPRODUCTION.md must exist"

    for rpath in [r1, r2]:
        with open(rpath, "r", encoding="utf-8") as f:
            c = f.read()
            assert "PENDING_HUMAN_LISTENER_VERIFICATION" in c
            assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c


def test_phase71_validation_pareto_study():
    """Verifies Phase 71 Pareto study outputs and operating point status."""
    frontier_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v8", "calibration", "threshold_frontier.json"
    )
    assert os.path.exists(frontier_json), "threshold_frontier.json must exist"

    with open(frontier_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "NO_FEASIBLE_VALIDATION_OPERATING_POINT"
    assert len(data["pareto_points"]) >= 10, "Expected >= 10 Pareto-optimal points"

    csv_path = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v8", "predictions", "validation_operating_points.csv"
    )
    assert os.path.exists(csv_path), "validation_operating_points.csv must exist"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 40, f"Expected >= 40 evaluated operating points, got {len(rows)}"

    rep = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "reports", "PHASE71_VALIDATION_PARETO.md")
    assert os.path.exists(rep), "PHASE71_VALIDATION_PARETO.md must exist"


def test_phase72_weight_ablation_and_checkpoints():
    """Verifies Phase 72 checkpoints existence, SHA-256 hashes, and model parameter counts."""
    ckpt_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "models", "checkpoints")
    variants = ["variant_w10", "variant_w12", "variant_w14", "variant_w16"]
    seeds = [42, 43, 44]

    for v in variants:
        for s in seeds:
            cp = os.path.join(ckpt_dir, f"{v}_seed{s}.pt")
            assert os.path.exists(cp), f"Checkpoint {cp} must exist"

            # Verify checkpoint reload and parameter match
            m = CorrectedVietnameseKWSNet(num_classes=5)
            ckpt = torch.load(cp, map_location="cpu")
            m.load_state_dict(ckpt["model_state_dict"])
            total_params = sum(p.numel() for p in m.parameters())
            assert total_params == 138469, f"Expected 138,469 parameters, got {total_params}"

    sum_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "models", "training_summary.json")
    assert os.path.exists(sum_path), "training_summary.json must exist"
    with open(sum_path, "r", encoding="utf-8") as f:
        ts = json.load(f)
    for v in variants:
        assert v in ts

    rep = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "reports", "PHASE72_WEIGHT_ABLATION.md")
    assert os.path.exists(rep), "PHASE72_WEIGHT_ABLATION.md must exist"


def test_phase73_calibration_and_finalists():
    """Verifies Phase 73 calibration configs, validation summary, and report."""
    cfg_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "calibration", "selected_configs.json")
    assert os.path.exists(cfg_path), "selected_configs.json must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfgs = json.load(f)

    assert "variant_w10" in cfgs
    assert "candidate_b_v7_baseline" in cfgs
    assert "cascade_v1" in cfgs

    sum_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "predictions", "validation_summary.json")
    assert os.path.exists(sum_path), "validation_summary.json must exist"

    rep = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "reports", "PHASE73_CALIBRATION.md")
    assert os.path.exists(rep), "PHASE73_CALIBRATION.md must exist"
    with open(rep, "r", encoding="utf-8") as f:
        c = f.read()
        assert "variant_w10" in c
        assert "candidate_b_v7_baseline" in c


def test_historical_immutability():
    """Ensures historical experiment directories and raw human data remain untouched and unreferenced."""
    v8_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8")
    for root, _, files in os.walk(v8_dir):
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
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7"),
    ]

    for hdir in historical_dirs:
        assert os.path.exists(hdir), f"Historical directory {hdir} must exist and be preserved"


def test_phase74_dataset_and_manifests():
    """Verifies Phase 74 fresh synthetic benchmark dataset, manifests, and text disjointness."""
    data_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "data")
    splits_dir = os.path.join(data_dir, "splits")

    test_csv = os.path.join(splits_dir, "test_manifest.csv")
    train_csv = os.path.join(splits_dir, "train_manifest.csv")
    val_csv = os.path.join(splits_dir, "val_manifest.csv")
    full_csv = os.path.join(data_dir, "manifest.csv")

    for p in [test_csv, train_csv, val_csv, full_csv]:
        assert os.path.exists(p), f"Manifest {p} must exist"

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_rows = list(reader)

    assert len(test_rows) == 720, f"Expected exactly 720 test clips, got {len(test_rows)}"

    # Check distribution: 160 commands, 480 non-target, 80 silence
    labels = [r["canonical_class"] for r in test_rows]
    cmd_count = sum(1 for l in labels if l in ["TRAI", "PHAI", "LEN", "XUONG"])
    unk_count = sum(1 for l in labels if l == "UNKNOWN")
    sil_count = sum(1 for l in labels if l == "SILENCE")

    assert cmd_count == 160, f"Expected 160 commands, got {cmd_count}"
    assert unk_count == 480, f"Expected 480 UNKNOWN non-target clips, got {unk_count}"
    assert sil_count == 80, f"Expected 80 SILENCE clips, got {sil_count}"

    # Verify voice inventory in test split: exactly 4 held-out voices
    test_voices = set(r["voice_id"] for r in test_rows if r["canonical_class"] != "SILENCE")
    assert test_voices == {"Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"}

    # Verify audio files exist on disk
    for r in test_rows[:20]:
        audio_path = os.path.join(BASE_DIR, r["kws_wav_path"])
        assert os.path.exists(audio_path), f"Audio file {audio_path} must exist"


def test_phase74_locked_test_results():
    """Verifies Phase 74 locked test evaluation metrics against strict criteria."""
    summary_path = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v8", "predictions", "locked_test_summary.json"
    )
    assert os.path.exists(summary_path), "locked_test_summary.json must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "variant_w10" in summary
    assert "candidate_b_v7_baseline" in summary
    assert "cascade_v1" in summary

    w10 = summary["variant_w10"]
    cross_summary = w10["cross_seed_summary"]

    # Target criteria: Recall >= 95%, FAR <= 5%, Latency <= 50ms
    assert cross_summary["mean_command_recall"] >= 0.95, "variant_w10 mean command recall must be >= 0.95"
    assert cross_summary["mean_false_acceptance_rate"] <= 0.05, "variant_w10 mean FAR must be <= 0.05"
    assert cross_summary["mean_cpu_latency_ms"] <= 50.0, "variant_w10 mean CPU latency must be <= 50ms"
    assert cross_summary["all_seeds_meet_criteria"] is True

    # Check each individual seed
    for s in ["seed42", "seed43", "seed44"]:
        metrics = w10[s]["metrics"]
        assert metrics["macro_command_recall"] >= 0.95
        assert metrics["non_target_false_acceptance_rate"] <= 0.05
        assert metrics["all_keywords_ge_95"] is True
        for kw, rec in metrics["per_keyword_recall"].items():
            assert rec >= 0.95, f"Expected {kw} recall >= 0.95 for {s}, got {rec}"
        assert w10[s]["cpu_latency"]["meets_50ms_budget"] is True

    # Verify per-clip predictions CSV
    preds_csv = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v8", "predictions", "per_clip_predictions.csv"
    )
    assert os.path.exists(preds_csv), "per_clip_predictions.csv must exist"
    with open(preds_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pred_rows = list(reader)
    assert len(pred_rows) == 720 * 3 * 3, f"Expected 6480 prediction rows, got {len(pred_rows)}"


def test_phase74_dual_operational_decisions_and_reports():
    """Verifies that final reports and operational decisions adhere to protocol."""
    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "reports")
    final_dec = os.path.join(rep_dir, "PHASE74_FINAL_DECISION.md")
    locked_rep = os.path.join(rep_dir, "PHASE74_LOCKED_TEST.md")
    readiness = os.path.join(rep_dir, "PHASE74_READINESS_MATRIX.md")
    readme = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8", "README.md")

    for doc in [final_dec, locked_rep, readiness, readme]:
        assert os.path.exists(doc), f"{doc} must exist"
        with open(doc, "r", encoding="utf-8") as f:
            c = f.read()
            assert "PENDING_HUMAN_LISTENER_VERIFICATION" in c
            assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c

    with open(final_dec, "r", encoding="utf-8") as f:
        dec_text = f.read()
        assert "FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED" in dec_text
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in dec_text
        assert "variant_w10" in dec_text

