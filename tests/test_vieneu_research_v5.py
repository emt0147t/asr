"""
Unit and regression tests for Phases 55-59: Vietnamese KWS Research Campaign v5.
Covers:
- Phase 55: Metric recomputation bit-exactness and CPU latency profiling.
- Phase 56: Dataset v5 construction, strict text disjointness, and speaker isolation.
- Phase 57: Training ablation model architectures and checkpoint loading.
- Phase 58: Validation calibration, robustness testing, and formal decision.
- Phase 59: Locked test evaluation, dual-verdict certifications, and historical immutability.
"""

import os
import sys
import json
import pytest
import pandas as pd
import soundfile as sf
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from experiments.vieneu_kws_research_v5.models.train_ablation_campaign import (
    VietnameseBoundedKWSNet,
    LogMelFrontend
)


def test_phase55_baseline_audit_and_cpu_latency():
    """Verifies that Phase 55 audit recomputed all metrics bit-for-bit with CPU latency < 50ms."""
    audit_json = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "phase55_audit_findings.json")
    assert os.path.exists(audit_json), "Phase 55 audit findings JSON must exist"

    with open(audit_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metrics_reproduced"] is True
    assert len(data["mismatches"]) == 0

    cpu_bench = data["cpu_latency_benchmark"]
    for cand, stats in cpu_bench.items():
        assert stats["mean_ms"] < 50.0, f"Candidate {cand} mean latency {stats['mean_ms']}ms exceeds 50ms limit"
        assert stats["p90_ms"] < 50.0, f"Candidate {cand} P90 latency {stats['p90_ms']}ms exceeds 50ms limit"


def test_phase56_dataset_v5_isolation_and_disjointness():
    """Verifies Dataset v5 splits, strictly disjoint non-target texts, and zero voice leakage."""
    data_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "data", "splits")
    train_df = pd.read_csv(os.path.join(data_dir, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_manifest.csv"))

    assert len(train_df) == 1360
    assert len(val_df) == 360
    assert len(test_df) == 720

    # Non-target text disjointness
    train_unk_texts = set(train_df[train_df["class_label"] == "UNKNOWN"]["text"])
    val_unk_texts = set(val_df[val_df["class_label"] == "UNKNOWN"]["text"])
    test_unk_texts = set(test_df[test_df["class_label"] == "UNKNOWN"]["text"])

    assert len(train_unk_texts.intersection(val_unk_texts)) == 0, "Non-target text overlap between train and val"
    assert len(train_unk_texts.intersection(test_unk_texts)) == 0, "Non-target text overlap between train and test"
    assert len(val_unk_texts.intersection(test_unk_texts)) == 0, "Non-target text overlap between val and test"

    # Voice isolation
    silence_conds = {'airflow_pink', 'hvac_hum', 'near_silence', 'office_ambient'}
    train_voices = set(train_df["voice_id"]) - silence_conds
    val_voices = set(val_df["voice_id"]) - silence_conds
    test_voices = set(test_df["voice_id"]) - silence_conds

    assert len(train_voices.intersection(test_voices)) == 0, "Speaker overlap between train and test"
    assert len(val_voices.intersection(test_voices)) == 0, "Speaker overlap between val and test"
    assert len(train_voices.intersection(val_voices)) == 0, "Speaker overlap between train and val"
    assert len(test_voices) == 4
    assert len(val_voices) == 4
    assert len(train_voices) == 17


def test_phase57_ablation_checkpoint_compatibility():
    """Verifies that Candidate C and D checkpoints load successfully with expected parameters."""
    device = torch.device("cpu")
    models_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "models")

    for variant in ["candidate_c_standard", "candidate_d_ablation"]:
        ckpt_path = os.path.join(models_dir, variant, "checkpoints", "model_seed42.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint for {variant} must exist"

        ckpt = torch.load(ckpt_path, map_location=device)
        assert ckpt["num_classes"] == 5
        assert ckpt["model_architecture"] == "VietnameseBoundedKWSNet"

        model = VietnameseBoundedKWSNet(num_classes=5)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_spec = torch.randn(2, 1, 40, 101)
        out = model(dummy_spec)
        assert out.shape == (2, 5)


def test_phase58_calibration_and_validation_verdict():
    """Verifies that validation calibration recorded NO_FEASIBLE_VALIDATION_OPERATING_POINT."""
    val_sum_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "predictions", "validation_summary.json")
    calib_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "calibration", "selected_configs.json")

    assert os.path.exists(val_sum_path) and os.path.exists(calib_path)

    with open(val_sum_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    assert val_data["formal_validation_verdict"] == "NO_FEASIBLE_VALIDATION_OPERATING_POINT"

    with open(calib_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    assert "variant_a_cascade_v1" in calib_data
    assert "variant_b_stage1_cnn" in calib_data


def test_phase59_locked_test_and_dual_verdict():
    """Verifies locked test results, latency compliance, and dual verdicts in reports."""
    locked_sum_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "predictions", "locked_test_summary.json")
    assert os.path.exists(locked_sum_path)

    with open(locked_sum_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    v_a = test_data["variant_a_cascade_v1"]["mean_summary"]
    # Variant A achieved mean FAR <= 5%
    assert v_a["mean_false_acceptance_rate"] <= 0.05
    assert v_a["mean_cpu_latency_ms"] < 50.0

    # Verify report verdicts
    rep_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "reports")
    dec_file = os.path.join(rep_dir, "PHASE59_FINAL_DECISION.md")
    assert os.path.exists(dec_file)

    with open(dec_file, "r", encoding="utf-8") as f:
        dec_text = f.read()

    assert "BOUNDED_UTTERANCE" in dec_text.replace("\\", "")
    assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in dec_text.replace("\\", "")
    assert "PENDING_HUMAN_LISTENER_VERIFICATION" in dec_text
    assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in dec_text


def test_historical_baselines_immutability():
    """Verifies that all historical manifests remain strictly untouched."""
    man1000 = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_1000", "manifest.csv")
    assert os.path.exists(man1000)
    assert len(pd.read_csv(man1000)) == 1000

    man6class = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class", "manifest.csv")
    assert os.path.exists(man6class)
    assert len(pd.read_csv(man6class)) == 1500

    man6class_v2 = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2", "manifest.csv")
    assert os.path.exists(man6class_v2)
    assert len(pd.read_csv(man6class_v2)) == 2000
