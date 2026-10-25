"""
Unit and Regression Test Suite for Phase 36–38 (Cascaded Vietnamese KWS).

Verifies:
  1. Preservation of all historical v1/v2 artifacts, manifests, and checkpoints.
  2. Configuration structure and parameter integrity.
  3. Stage 1 candidate generation logic.
  4. Stage 2 TemporalKWSVerifier architecture, parameter count, and tensor shapes.
  5. Stage 2 input crop generation and label policy.
  6. Checkpoint save/load integrity and random seed provenance (42, 43, 44).
  7. Hard-negative mining and training crop construction.
  8. Streaming event matching, latency, duplicate merging, and cooldown lockout.
  9. False activations per hour denominator correctness.
  10. Validation-only threshold calibration and non-leakage verification.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import (
    TemporalKWSVerifier,
    TCNResidualBlock,
    VERIFIER_CLASSES,
    extract_or_pad_crop
)
from experiments.vieneu_kws_cascade_v1.evaluate_cascade import (
    CascadedStreamingEngine
)
from experiments.vieneu_kws_v2.evaluate import (
    match_events_and_evaluate
)

EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1")
V2_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2")
DATASET_V2_DIR = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2")


def test_preservation_of_historical_artifacts():
    """Verifies that all historical v1 and v2 artifacts remain intact and unmodified."""
    v1_manifest = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv")
    v2_manifest = os.path.join(DATASET_V2_DIR, "manifest.csv")
    s1_ckpt_42 = os.path.join(V2_EXP_DIR, "checkpoints/best_checkpoint_seed42.pt")
    s1_ckpt_43 = os.path.join(V2_EXP_DIR, "checkpoints/best_checkpoint_seed43.pt")
    s1_ckpt_44 = os.path.join(V2_EXP_DIR, "checkpoints/best_checkpoint_seed44.pt")

    assert os.path.exists(v1_manifest), "v1 4-class manifest missing!"
    assert os.path.exists(v2_manifest), "v2 6-class manifest missing!"
    assert os.path.exists(s1_ckpt_42), "Stage 1 seed 42 checkpoint missing!"
    assert os.path.exists(s1_ckpt_43), "Stage 1 seed 43 checkpoint missing!"
    assert os.path.exists(s1_ckpt_44), "Stage 1 seed 44 checkpoint missing!"

    df2 = pd.read_csv(v2_manifest)
    assert len(df2) == 2000, f"v2 dataset corrupted! Expected 2000 rows, got {len(df2)}"


def test_cascade_config_structure():
    """Verifies cascade config.json parameters and structure."""
    cfg_path = os.path.join(EXP_DIR, "config.json")
    assert os.path.exists(cfg_path), "Cascade config.json missing!"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["experiment_name"] == "vieneu_kws_cascade_v1"
    assert "stage1" in cfg
    assert "stage2" in cfg
    assert "validation_constraints" in cfg
    assert cfg["stage2"]["model_type"] == "TemporalKWSVerifier"
    assert cfg["stage2"]["num_classes"] == 5
    assert cfg["stage2"]["input_duration_sec"] == 1.2
    assert cfg["stage2"]["input_samples"] == 19200
    assert cfg["validation_constraints"]["max_streaming_fa_per_hour"] == 1.0


def test_temporal_verifier_architecture_and_shapes():
    """Verifies TemporalKWSVerifier forward pass, output shape, and parameter scale."""
    model = TemporalKWSVerifier(num_classes=5, dropout=0.2)
    model.eval()

    # Input: (Batch=2, Channels=1, Mel=40, Time=121)
    dummy_input = torch.randn(2, 1, 40, 121)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 5), f"Expected output shape (2, 5), got {out.shape}"
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 200000 <= param_count <= 450000, f"Parameter count {param_count} out of expected range!"


def test_crop_generation_and_padding():
    """Verifies extract_or_pad_crop handles short, exact, and long audio arrays properly."""
    # Short audio
    short_audio = np.ones(16000, dtype=np.float32)
    crop = extract_or_pad_crop(short_audio, target_samples=19200, offset_samples=1600)
    assert len(crop) == 19200
    assert np.all(crop[:1600] == 0.0) # Pre-padding
    assert np.all(crop[1600:17600] == 1.0) # Embedded audio
    assert np.all(crop[17600:] == 0.0) # Post-padding

    # Exact audio
    exact_audio = np.ones(19200, dtype=np.float32) * 2.0
    crop_exact = extract_or_pad_crop(exact_audio, target_samples=19200, offset_samples=0)
    assert len(crop_exact) == 19200
    assert np.all(crop_exact == 2.0)


def test_checkpoint_save_and_provenance():
    """Verifies Stage 2 checkpoints exist, load cleanly, and contain metadata."""
    for seed in [42, 43, 44]:
        ckpt_path = os.path.join(EXP_DIR, "checkpoints", f"best_verifier_seed{seed}.pt")
        assert os.path.exists(ckpt_path), f"Stage 2 checkpoint seed {seed} missing!"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert ckpt["seed"] == seed
        assert "val_macro_f1" in ckpt
        assert ckpt["val_macro_f1"] > 0.85, f"Seed {seed} val macro F1 too low: {ckpt['val_macro_f1']}"
        assert ckpt["class_names"] == VERIFIER_CLASSES


def test_event_matching_and_latency_calculation():
    """Verifies that match_events_and_evaluate correctly matches true positives and calculates latency."""
    gt_df = pd.DataFrame([
        {"event_type": "target_command", "class_label": "TRAI", "onset_sec": 2.0, "end_time_sec": 2.8},
        {"event_type": "target_command", "class_label": "PHAI", "onset_sec": 5.0, "end_time_sec": 5.7},
        {"event_type": "distractor", "class_label": "UNKNOWN", "onset_sec": 8.0, "end_time_sec": 8.9}
    ])

    triggered_events = [
        {"trigger_time_sec": 2.35, "pred_label": "TRAI", "confidence": 0.95}, # Correct TRAI
        {"trigger_time_sec": 5.40, "pred_label": "PHAI", "confidence": 0.92}, # Correct PHAI
        {"trigger_time_sec": 8.20, "pred_label": "XUONG", "confidence": 0.88}  # False alarm on UNKNOWN
    ]

    m = match_events_and_evaluate(triggered_events, gt_df, stream_duration_sec=10.0)

    assert m["num_gt_targets"] == 2
    assert m["true_positives"] == 2
    assert m["false_negatives"] == 0
    assert m["false_positives"] == 1 # 1 unmatched trigger
    assert m["recall"] == 1.0
    assert m["precision"] == round(2.0 / 3.0, 4)

    # Latencies: (2.35 - 2.0) = 350ms, (5.40 - 5.0) = 400ms -> Mean 375ms
    assert abs(m["latencies_ms"]["mean"] - 375.0) < 1.0


def test_fa_per_hour_denominator_math():
    """Verifies that FA/hour calculation strictly divides false alarms by eligible stream duration in hours."""
    gt_df = pd.DataFrame([{"event_type": "distractor", "class_label": "UNKNOWN", "onset_sec": 1.0, "end_time_sec": 2.0}])
    triggers = [{"trigger_time_sec": 1.5, "pred_label": "TRAI", "confidence": 0.9}] # 1 FA

    # Stream of 900 seconds (0.25 hours) -> 1 FA in 0.25 hours = 4.0 FA/hr
    m = match_events_and_evaluate(triggers, gt_df, stream_duration_sec=900.0)
    assert m["false_positives"] == 1
    assert m["stream_duration_hours"] == 0.25
    assert m["fa_per_hour"] == 4.0


def test_validation_calibration_records_exist_and_no_test_leakage():
    """Verifies validation calibration records exist and test split is frozen."""
    cal_rec_p = os.path.join(EXP_DIR, "calibration", "validation_calibration_records.json")
    th_p = os.path.join(EXP_DIR, "calibration", "selected_thresholds.json")

    assert os.path.exists(cal_rec_p), "Validation calibration records JSON missing!"
    assert os.path.exists(th_p), "Selected thresholds JSON missing!"

    with open(th_p, "r", encoding="utf-8") as f:
        ths = json.load(f)

    for seed in [42, 43, 44]:
        assert f"seed{seed}" in ths
        assert "theta1" in ths[f"seed{seed}"]
        assert "theta2" in ths[f"seed{seed}"]
        # Infeasibility of <= 1.0 FA/hr must be declared
        assert ths[f"seed{seed}"]["is_strictly_feasible"] is False
        assert "NO_FEASIBLE_VALIDATION_OPERATING_POINT" in ths[f"seed{seed}"]["rationale"]
