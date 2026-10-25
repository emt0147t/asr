"""
Unit and regression tests for Phase 40-44: Vietnamese KWS Research Campaign v2.
Covers:
- Phase 40: Event matching logic, duplicate suppression, boundary handling, exact FA/hour denominator, VAD gating behavior, schema consistency.
- Stage 1 and Stage 2 architectural integrity and parameter count verification.
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

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_v2.evaluate import match_events_and_evaluate
from experiments.vieneu_kws_vad_ablation_v1.vad_engine import SpeechBandFormantVAD, ShortTimeEnergyGate


def test_stage1_and_stage2_architectures_and_param_counts():
    """Verifies actual architecture classes and parameter counts."""
    m1 = VietnameseKWSNet(num_classes=6)
    m2 = TemporalKWSVerifier(num_classes=5)

    p1 = get_model_param_count(m1)
    p2 = get_model_param_count(m2)

    assert p1 == 138534, f"Stage 1 VietnameseKWSNet parameter count should be 138534, got {p1}"
    assert p2 == 340773, f"Stage 2 TemporalKWSVerifier parameter count should be 340773, got {p2}"
    assert hasattr(m2, "tcn"), "Stage 2 verifier must have TCN sequence backbone, not recurrent BiGRU"


def test_event_matching_and_duplicate_suppression():
    """Verifies that duplicate triggers for the same ground truth are suppressed and counted correctly."""
    gt_df = pd.DataFrame([
        {
            "event_type": "target_command",
            "class_label": "LEN",
            "onset_sec": 5.0,
            "end_time_sec": 5.8,
            "sample_id": "test_cmd_1"
        },
        {
            "event_type": "target_command",
            "class_label": "XUONG",
            "onset_sec": 10.0,
            "end_time_sec": 10.9,
            "sample_id": "test_cmd_2"
        }
    ])

    # Three triggers:
    # 1. Matches LEN (onset 5.0, trigger 5.4) -> TP
    # 2. Duplicate trigger for LEN (trigger 5.6) -> FP because gt_matched is True
    # 3. Mismatched class at 10.4 -> FP
    triggers = [
        {"trigger_time_sec": 5.4, "pred_label": "LEN", "confidence": 0.95},
        {"trigger_time_sec": 5.6, "pred_label": "LEN", "confidence": 0.92},
        {"trigger_time_sec": 10.4, "pred_label": "TRAI", "confidence": 0.88}
    ]

    res = match_events_and_evaluate(triggers, gt_df, stream_duration_sec=20.0)

    assert res["true_positives"] == 1, "Only first matching trigger should be TP"
    assert res["false_negatives"] == 1, "XUONG command was not matched (TRAI trigger)"
    assert res["false_positives"] == 2, "Duplicate LEN and mismatched TRAI must be FPs"
    assert res["recall"] == 0.5
    assert abs(res["latencies_ms"]["mean"] - 400.0) < 1.0


def test_exact_fa_per_hour_denominator():
    """Verifies that FA/hour strictly uses stream_duration_sec / 3600.0."""
    gt_df = pd.DataFrame(columns=["event_type", "class_label", "onset_sec", "end_time_sec", "sample_id"])
    triggers = [{"trigger_time_sec": 12.0, "pred_label": "PHAI", "confidence": 0.9}]

    # 1800 seconds = exactly 0.5 hours
    res = match_events_and_evaluate(triggers, gt_df, stream_duration_sec=1800.0)
    assert res["stream_duration_hours"] == 0.5
    assert res["fa_per_hour"] == 2.0  # 1 FP / 0.5 hours = 2.0 FA/hr


def test_stream_boundary_behavior():
    """Verifies matching window edges: onset - 0.20s and end + 0.60s."""
    gt_df = pd.DataFrame([
        {
            "event_type": "target_command",
            "class_label": "TRAI",
            "onset_sec": 2.0,
            "end_time_sec": 2.7,
            "sample_id": "test_cmd_edge"
        }
    ])

    # Window is [2.0 - 0.20, 2.7 + 0.60] = [1.80, 3.30]
    # Trigger at 1.79 should be unmatched (FP)
    trig_early = [{"trigger_time_sec": 1.79, "pred_label": "TRAI", "confidence": 0.9}]
    res_early = match_events_and_evaluate(trig_early, gt_df, stream_duration_sec=10.0)
    assert res_early["true_positives"] == 0
    assert res_early["false_positives"] == 1

    # Trigger at 1.80 should be matched (TP)
    trig_exact = [{"trigger_time_sec": 1.80, "pred_label": "TRAI", "confidence": 0.9}]
    res_exact = match_events_and_evaluate(trig_exact, gt_df, stream_duration_sec=10.0)
    assert res_exact["true_positives"] == 1

    # Trigger at 3.30 should be matched (TP)
    trig_late_edge = [{"trigger_time_sec": 3.30, "pred_label": "TRAI", "confidence": 0.9}]
    res_late_edge = match_events_and_evaluate(trig_late_edge, gt_df, stream_duration_sec=10.0)
    assert res_late_edge["true_positives"] == 1

    # Trigger at 3.31 should be unmatched (FP)
    trig_late = [{"trigger_time_sec": 3.31, "pred_label": "TRAI", "confidence": 0.9}]
    res_late = match_events_and_evaluate(trig_late, gt_df, stream_duration_sec=10.0)
    assert res_late["true_positives"] == 0
    assert res_late["false_positives"] == 1


def test_vad_speech_silence_boundary_and_gating():
    """Verifies VAD gate activation on pure silence vs speech-like audio."""
    vad = SpeechBandFormantVAD(sample_rate=16000, min_speech_band_ratio=0.35, min_energy_db=-48.0)

    # 1. Micro-dither floor: should be False
    silence = np.random.normal(0, 0.0005, 16000).astype(np.float32)
    is_sp_sil, active_ratio_sil, _ = vad.process_window(silence)
    assert not is_sp_sil, "Silence floor should not trigger VAD"
    assert active_ratio_sil < 0.10

    # 2. Strong formant energy at 800 Hz (speech-like): should be True
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    speech_sim = (0.2 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.sin(2 * np.pi * 1600 * t)).astype(np.float32)
    is_sp_vox, active_ratio_vox, _ = vad.process_window(speech_sim)
    assert is_sp_vox, "Speech formant signal must trigger VAD"
    assert active_ratio_vox > 0.50


def test_ctc_model_and_tokenizer():
    """Verifies VietnameseCTCKWSNet architecture, parameter count, and tokenizer."""
    from experiments.vieneu_kws_research_v2.models.ctc_kws.ctc_model import (
        VietnameseCharTokenizer,
        VietnameseCTCKWSNet,
        get_ctc_param_count
    )

    tokenizer = VietnameseCharTokenizer()
    assert tokenizer.vocab_size == 91, f"Expected vocab size 91, got {tokenizer.vocab_size}"

    # Verify encoding and decoding round-trip
    text = "trái phải lên xuống"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Tokenizer round-trip failed: '{decoded}' vs '{text}'"

    # Verify keyword matching
    for kw in ["TRAI", "PHAI", "LEN", "XUONG"]:
        sample_text = {"TRAI": "trái", "PHAI": "phải", "LEN": "lên", "XUONG": "xuống"}[kw]
        matched, exact = tokenizer.match_keyword(sample_text)
        assert matched == kw and exact, f"Keyword match failed for {kw}: got {matched}, exact={exact}"

    # Verify model forward pass and parameter count
    ctc_model = VietnameseCTCKWSNet(vocab_size=tokenizer.vocab_size)
    param_count = get_ctc_param_count(ctc_model)
    assert param_count == 229115, f"Expected 229115 parameters, got {param_count}"

    dummy_input = torch.randn(2, 40, 101)  # (Batch=2, Mels=40, Frames=101)
    output = ctc_model(dummy_input)
    assert output.shape == (101, 2, 91), f"Expected shape (101, 2, 91), got {output.shape}"


def test_research_campaign_artifacts_and_discrepancy_resolution():
    """Verifies presence and validity of all Phase 40-44 experiment artifacts."""
    exp_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v2")

    # 1. Manifests
    manifest_p = os.path.join(exp_dir, "data", "manifest.csv")
    assert os.path.exists(manifest_p), "Master manifest must exist"
    df = pd.read_csv(manifest_p)
    assert len(df) == 2000, f"Expected 2000 utterances, got {len(df)}"

    # 2. Checkpoints
    for s in [42, 43, 44]:
        s1 = os.path.join(exp_dir, "models", "hard_negative_cascade", "checkpoints", f"stage1_retrained_seed{s}.pt")
        s2 = os.path.join(exp_dir, "models", "hard_negative_cascade", "checkpoints", f"stage2_retrained_seed{s}.pt")
        ctc = os.path.join(exp_dir, "models", "ctc_kws", "checkpoints", f"best_ctc_seed{s}.pt")
        assert os.path.exists(s1), f"Missing {s1}"
        assert os.path.exists(s2), f"Missing {s2}"
        assert os.path.exists(ctc), f"Missing {ctc}"

    # 3. Reports
    for r_num in [40, 41, 42, 43, 44]:
        matches = [f for f in os.listdir(os.path.join(exp_dir, "reports")) if f"PHASE{r_num}" in f]
        assert len(matches) == 1, f"Missing Phase {r_num} report in reports/"

    # 4. Calibration file
    calib_p = os.path.join(exp_dir, "calibration", "selected_operating_points.json")
    assert os.path.exists(calib_p), "Selected operating points JSON must exist"
    with open(calib_p, "r", encoding="utf-8") as f:
        calib = json.load(f)
    for c in calib.values():
        assert not c["is_strictly_feasible"], "All operating points must be infeasible"
        assert "NO_FEASIBLE_VALIDATION_OPERATING_POINT" in c["rationale"]

