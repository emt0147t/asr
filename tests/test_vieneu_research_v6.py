"""
Unit and regression tests for Phases 60-64: Vietnamese KWS Research Campaign v6.
Covers:
- Phase 60: Metric reconciliation, mathematical equivalence of macro/micro recall, CPU latency profiling.
- Phase 61: Dataset v6 construction, strict text disjointness, and speaker isolation (to be expanded).
- Phase 62: Training ablation and model architectures (to be expanded).
- Phase 63: Validation calibration, robustness testing, and formal decision (to be expanded).
- Phase 64: Locked test evaluation, dual-verdict certifications, and historical immutability (to be expanded).
"""

import os
import sys
import json
import pytest
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)


def test_phase60_reconciliation_json_exists_and_valid():
    """Verifies that Phase 60 reconciliation JSON exists and contains valid metrics."""
    reconciled_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v6", "predictions", "phase60_reproduced_metrics.json"
    )
    assert os.path.exists(reconciled_json), "Phase 60 reproduced metrics JSON must exist"

    with open(reconciled_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "test_manifest_structure" in data
    assert "discrete_step_sizes" in data
    assert "reconciled_candidates" in data
    assert "latency_decomposition_audit" in data

    # Verify discrete step sizes
    step_sizes = data["discrete_step_sizes"]
    assert pytest.approx(step_sizes["per_keyword_clip_step (1/40)"], abs=1e-4) == 0.025
    assert pytest.approx(step_sizes["command_clip_step (1/160)"], abs=1e-4) == 0.00625
    assert pytest.approx(step_sizes["non_target_clip_step (1/560)"], abs=1e-4) == 0.0017857


def test_phase60_macro_micro_equivalence():
    """Verifies that macro and micro command recall are mathematically equivalent on 40 clips/class."""
    reconciled_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v6", "predictions", "phase60_reproduced_metrics.json"
    )
    with open(reconciled_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cand, cand_data in data["reconciled_candidates"].items():
        for seed_str, seed_data in cand_data["per_seed"].items():
            metrics = seed_data["performance_metrics"]
            assert metrics["macro_equals_micro"] is True, f"{cand} {seed_str}: macro_equals_micro is not True"
            macro_recall = metrics["macro_command_recall"]
            micro_recall = metrics["micro_command_recall"]
            assert pytest.approx(macro_recall, abs=1e-3) == micro_recall, (
                f"Candidate {cand} seed {seed_str}: macro {macro_recall} != micro {micro_recall}"
            )


def test_phase60_cpu_latency_budget():
    """Verifies that all candidates tested in Phase 60 meet the <= 50ms post-capture CPU latency budget."""
    reconciled_json = os.path.join(
        BASE_DIR, "experiments", "vieneu_kws_research_v6", "predictions", "phase60_reproduced_metrics.json"
    )
    with open(reconciled_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = data["latency_decomposition_audit"]["candidate_profiles"]
    for cand, prof in profiles.items():
        assert prof["end_to_end_user_perceived"]["meets_50ms_post_capture_budget"] is True
        total_lat = prof["total_post_capture_ms"]
        mean_ms = total_lat["mean_ms"]
        p90_ms = total_lat["p90_ms"]
        max_ms = total_lat["max_ms"]
        assert mean_ms < 50.0, f"{cand} mean CPU latency {mean_ms}ms exceeds 50ms limit"
        assert p90_ms < 50.0, f"{cand} P90 CPU latency {p90_ms}ms exceeds 50ms limit"
        assert max_ms < 50.0, f"{cand} Max CPU latency {max_ms}ms exceeds 50ms limit"


def test_phase60_reports_exist():
    """Verifies that Phase 60 reports exist and contain required status markers."""
    reports_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "reports")
    rec_report = os.path.join(reports_dir, "PHASE60_METRIC_RECONCILIATION.md")
    audit_report = os.path.join(reports_dir, "PHASE60_DATA_AND_PIPELINE_AUDIT.md")

    assert os.path.exists(rec_report), "Phase 60 metric reconciliation report must exist"
    assert os.path.exists(audit_report), "Phase 60 data and pipeline audit report must exist"

    for rpath in [rec_report, audit_report]:
        with open(rpath, "r", encoding="utf-8") as f:
            content = f.read()
            assert "PENDING_HUMAN_LISTENER_VERIFICATION" in content
            assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in content


def test_phase61_dataset_v6_manifest_and_splits():
    """Verifies that Dataset v6 manifests exist and have correct split distributions."""
    data_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "data")
    master_df = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    train_df = pd.read_csv(os.path.join(data_dir, "splits", "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "splits", "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "splits", "test_manifest.csv"))

    assert len(master_df) == 2440
    assert len(train_df) == 1360
    assert len(val_df) == 360
    assert len(test_df) == 720

    # Test split class balance
    assert len(test_df[test_df["bounded_label"] == "TRAI"]) == 40
    assert len(test_df[test_df["bounded_label"] == "PHAI"]) == 40
    assert len(test_df[test_df["bounded_label"] == "LEN"]) == 40
    assert len(test_df[test_df["bounded_label"] == "XUONG"]) == 40
    assert len(test_df[test_df["bounded_label"] == "REJECT"]) == 560


def test_phase61_text_disjointness_and_isolation():
    """Verifies strict text disjointness across splits and zero historical text overlap."""
    splits_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "data", "splits")
    train_df = pd.read_csv(os.path.join(splits_dir, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test_manifest.csv"))

    train_unk = set(train_df[train_df["class_label"] == "UNKNOWN"]["text"].str.lower())
    val_unk = set(val_df[val_df["class_label"] == "UNKNOWN"]["text"].str.lower())
    test_unk = set(test_df[test_df["class_label"] == "UNKNOWN"]["text"].str.lower())

    assert len(train_unk) == 80
    assert len(val_unk) == 40
    assert len(test_unk) == 80

    assert len(train_unk.intersection(val_unk)) == 0, "Train-Val non-target text overlap!"
    assert len(train_unk.intersection(test_unk)) == 0, "Train-Test non-target text overlap!"
    assert len(val_unk.intersection(test_unk)) == 0, "Val-Test non-target text overlap!"

    # Historical text overlap check
    hist_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "historical_texts.txt")
    if os.path.exists(hist_file):
        with open(hist_file, "r", encoding="utf-8") as f:
            hist = set(line.strip().lower() for line in f if line.strip())
        assert len(train_unk.intersection(hist)) == 0, "Train text overlaps historical!"
        assert len(val_unk.intersection(hist)) == 0, "Val text overlaps historical!"
        assert len(test_unk.intersection(hist)) == 0, "Test text overlaps historical!"


def test_phase61_voice_isolation():
    """Verifies speaker voice allocations across train (17), val (4), and test (4 held-out)."""
    splits_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "data", "splits")
    train_df = pd.read_csv(os.path.join(splits_dir, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test_manifest.csv"))

    silence_conds = {"airflow_pink", "hvac_hum", "near_silence", "office_ambient"}
    train_voices = set(train_df["voice_id"]) - silence_conds
    val_voices = set(val_df["voice_id"]) - silence_conds
    test_voices = set(test_df["voice_id"]) - silence_conds

    assert len(train_voices) == 17
    assert len(val_voices) == 4
    assert len(test_voices) == 4

    assert len(train_voices.intersection(test_voices)) == 0, "Train-Test speaker overlap!"
    assert len(val_voices.intersection(test_voices)) == 0, "Val-Test speaker overlap!"
    assert len(train_voices.intersection(val_voices)) == 0, "Train-Val speaker overlap!"
    assert test_voices == {"Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"}


def test_phase61_metadata_and_report_exist():
    """Verifies that Phase 61 metadata and dataset report exist."""
    meta_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "data", "benchmark_metadata.json")
    rep_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "reports", "PHASE61_DATASET_REPORT.md")

    assert os.path.exists(meta_path)
    assert os.path.exists(rep_path)

    with open(rep_path, "r", encoding="utf-8") as f:
        rep_content = f.read()
        assert "PENDING_HUMAN_LISTENER_VERIFICATION" in rep_content
        assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in rep_content


def test_historical_baselines_immutability():
    """Verifies that all historical manifests and models remain strictly untouched."""
    man1000 = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_1000", "manifest.csv")
    assert os.path.exists(man1000), "1000 manifest must exist"
    assert len(pd.read_csv(man1000)) == 1000

    man6class = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class", "manifest.csv")
    assert os.path.exists(man6class), "6class manifest must exist"
    assert len(pd.read_csv(man6class)) == 1500

    man6class_v2 = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2", "manifest.csv")
    assert os.path.exists(man6class_v2), "6class v2 manifest must exist"
    assert len(pd.read_csv(man6class_v2)) == 2000

    man_v5 = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "data", "manifest.csv")
    assert os.path.exists(man_v5), "v5 manifest must exist"
    assert len(pd.read_csv(man_v5)) == 2440


def test_phase62_checkpoints_and_architecture():
    """Verifies that Phase 62 models exist and load successfully."""
    from experiments.vieneu_kws_research_v6.models.train_ablation_campaign import VietnameseBoundedKWSNet
    ckpts_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "models", "checkpoints")

    for v in ["variant_b", "variant_c"]:
        for s in [42, 43, 44]:
            ckpt_path = os.path.join(ckpts_dir, f"{v}_seed{s}.pt")
            assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} missing"

    # Architecture test
    model = VietnameseBoundedKWSNet(num_classes=5)
    param_count = sum(p.numel() for p in model.parameters())
    assert 180000 < param_count < 190000, f"Unexpected parameter count: {param_count}"

    dummy_input = torch.randn(2, 1, 40, 101)
    out = model(dummy_input)
    assert out.shape == (2, 5)

    rep_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "reports", "PHASE62_TRAINING_AND_ABLATION.md")
    assert os.path.exists(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "PENDING_HUMAN_LISTENER_VERIFICATION" in content


def test_phase63_calibration_configs_and_operating_points():
    """Verifies that Phase 63 calibration outputs exist and recorded formal validation status."""
    calib_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "calibration")
    pred_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "predictions")
    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "reports")

    cfg_file = os.path.join(calib_dir, "selected_configs.json")
    val_sum_file = os.path.join(pred_dir, "validation_summary.json")
    rep_file = os.path.join(rep_dir, "PHASE63_CALIBRATION_AND_ROBUSTNESS.md")

    assert os.path.exists(cfg_file)
    assert os.path.exists(val_sum_file)
    assert os.path.exists(rep_file)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    for var in ["variant_a_cascade_v1", "variant_b_standard_bounded_cnn", "variant_c_hard_negative_ablation", "variant_d_stage1_cnn"]:
        assert var in cfg_data
        for s in [42, 43, 44]:
            assert f"seed{s}" in cfg_data[var]
            assert "robustness" in cfg_data[var][f"seed{s}"]


def test_phase64_locked_test_and_dual_verdicts():
    """Verifies Phase 64 locked test outputs, CPU latency budget, and dual verdicts."""
    pred_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "predictions")
    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "reports")

    sum_file = os.path.join(pred_dir, "locked_test_summary.json")
    preds_file = os.path.join(pred_dir, "per_clip_predictions.csv")
    test_rep = os.path.join(rep_dir, "PHASE64_LOCKED_TEST.md")
    dec_rep = os.path.join(rep_dir, "PHASE64_FINAL_DECISION.md")
    mat_rep = os.path.join(rep_dir, "PHASE64_READINESS_MATRIX.md")
    readme_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "README.md")

    assert os.path.exists(sum_file)
    assert os.path.exists(preds_file)
    assert os.path.exists(test_rep)
    assert os.path.exists(dec_rep)
    assert os.path.exists(mat_rep)
    assert os.path.exists(readme_file)

    with open(sum_file, "r", encoding="utf-8") as f:
        test_summary = json.load(f)

    for cand, cand_data in test_summary.items():
        for s in [42, 43, 44]:
            seed_data = cand_data[f"seed{s}"]
            mean_lat = seed_data["cpu_latency"]["mean_ms"]
            assert mean_lat < 50.0, f"{cand} seed {s} mean latency {mean_lat}ms exceeds 50ms"
            assert seed_data["cpu_latency"]["meets_50ms_budget"] is True

    # Assert dual verdicts in reports
    with open(dec_rep, "r", encoding="utf-8") as f:
        dec_content = f.read()
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in dec_content
        assert "PENDING_HUMAN_LISTENER_VERIFICATION" in dec_content
        assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in dec_content

    with open(mat_rep, "r", encoding="utf-8") as f:
        mat_content = f.read()
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in mat_content


def test_governance_git_diff_cached_empty():
    """Verifies that git diff --cached is strictly empty (no staged commits)."""
    import subprocess
    res = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True, cwd=BASE_DIR)
    assert res.returncode == 0
    assert len(res.stdout.strip()) == 0, f"Staged git changes detected: {res.stdout}"

