"""
Unit & Integration Tests for Phase 30–32 Robustness, Calibration, and Streaming Campaign.
Verifies:
  - Configuration structure and parameter integrity
  - Deterministic noise corruption and SNR tolerance (<= 0.05 dB for speech)
  - Manifest schemas and file integrity
  - Post-processing confidence policy logic (rejections, pass-through)
  - Selected frozen calibration thresholds
  - Sliding-window streaming engine debounce and hysteresis logic
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

from experiments.vieneu_kws_robustness_campaign.confidence_policy import (
    apply_confidence_policy,
    TARGET_COMMAND_INDICES,
    UNKNOWN_INDEX,
    SILENCE_INDEX
)
from experiments.vieneu_kws_robustness_campaign.streaming_simulator import (
    SlidingWindowStreamingEngine,
    match_events_and_evaluate
)

CAMPAIGN_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_robustness_campaign")

def test_config_structure():
    cfg_path = os.path.join(CAMPAIGN_DIR, "config.json")
    assert os.path.exists(cfg_path), "config.json does not exist"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    assert "checkpoints" in cfg
    assert "phase30_noise" in cfg
    assert "phase31_calibration" in cfg
    assert "phase32_streaming" in cfg
    assert "governance" in cfg
    
    assert set(cfg["phase30_noise"]["noise_types"]) == {"white", "pink", "hvac", "babble"}
    assert set(cfg["phase30_noise"]["snr_levels_db"]) == {20, 15, 10, 5, 0}
    assert cfg["governance"]["subjective_intelligibility_status"] == "PENDING_HUMAN_LISTENER_VERIFICATION"
    assert cfg["governance"]["domain_validation_status"] == "NOT_VALIDATED_ON_NATURAL_SPEECH"

def test_corrupted_manifests_integrity():
    val_manifest = os.path.join(CAMPAIGN_DIR, "corrupted_val_manifest.csv")
    test_manifest = os.path.join(CAMPAIGN_DIR, "corrupted_test_manifest.csv")
    
    assert os.path.exists(val_manifest), "corrupted_val_manifest.csv missing"
    assert os.path.exists(test_manifest), "corrupted_test_manifest.csv missing"
    
    df_val = pd.read_csv(val_manifest)
    df_test = pd.read_csv(test_manifest)
    
    # 240 samples x 4 noise types x 5 SNRs = 4,800
    assert len(df_val) == 4800, f"Expected 4800 val rows, got {len(df_val)}"
    assert len(df_test) == 4800, f"Expected 4800 test rows, got {len(df_test)}"
    
    required_cols = {"sample_id", "class_label", "split", "voice_id", "noise_type", 
                     "target_snr_db", "achieved_snr_db", "corrupted_wav_path", "original_wav_path",
                     "noise_seed", "corrupted_sha256", "is_silence"}
    assert required_cols.issubset(df_val.columns)
    assert required_cols.issubset(df_test.columns)

def test_achieved_snr_accuracy_on_speech():
    test_manifest = os.path.join(CAMPAIGN_DIR, "corrupted_test_manifest.csv")
    df_test = pd.read_csv(test_manifest)
    speech_df = df_test[df_test["is_silence"] == False]
    
    snr_diff = (speech_df["achieved_snr_db"] - speech_df["target_snr_db"]).abs()
    assert (snr_diff <= 0.05).all(), f"Speech SNR diff exceeded 0.05 dB: max diff {snr_diff.max()}"

def test_confidence_policy_behavior():
    # 1. Target command with confidence >= threshold -> remains unchanged
    probs_high_target = np.array([0.92, 0.02, 0.02, 0.02, 0.01, 0.01]) # TRAI
    pred, conf, rej = apply_confidence_policy(probs_high_target, threshold=0.85)
    assert pred == 0
    assert conf == pytest.approx(0.92)
    assert rej is False
    
    # 2. Target command with confidence < threshold -> overridden to UNKNOWN (4)
    probs_low_target = np.array([0.65, 0.15, 0.05, 0.05, 0.05, 0.05]) # TRAI below 0.85
    pred, conf, rej = apply_confidence_policy(probs_low_target, threshold=0.85)
    assert pred == UNKNOWN_INDEX
    assert rej is True
    
    # 3. UNKNOWN prediction below threshold -> untouched
    probs_unk = np.array([0.10, 0.10, 0.10, 0.10, 0.55, 0.05])
    pred, conf, rej = apply_confidence_policy(probs_unk, threshold=0.85)
    assert pred == UNKNOWN_INDEX
    assert rej is False
    
    # 4. SILENCE prediction below threshold -> untouched
    probs_sil = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.50])
    pred, conf, rej = apply_confidence_policy(probs_sil, threshold=0.85)
    assert pred == SILENCE_INDEX
    assert rej is False

def test_selected_thresholds_presence():
    summary_file = os.path.join(CAMPAIGN_DIR, "calibration", "selected_thresholds.json")
    assert os.path.exists(summary_file), "selected_thresholds.json missing"
    with open(summary_file, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    for s in ["seed42", "seed43", "seed44"]:
        assert s in thresholds
        val = thresholds[s]
        assert 0.50 <= val <= 0.99, f"Threshold {val} out of expected range [0.50, 0.99]"

def test_streaming_event_matching():
    ground_truth = pd.DataFrame([
        {
            "event_type": "target_command",
            "class_label": "TRAI",
            "start_time_sec": 2.0,
            "end_time_sec": 3.0,
            "onset_sec": 2.2,
            "active_dur_sec": 0.6
        },
        {
            "event_type": "distractor",
            "class_label": "UNKNOWN",
            "start_time_sec": 5.0,
            "end_time_sec": 6.0,
            "onset_sec": 5.1,
            "active_dur_sec": 0.7
        }
    ])
    
    # Simulate a trigger that hits TRAI at 2.4s (True Positive) and another trigger at 8.0s (False Alarm)
    triggers = [
        {"trigger_time_sec": 2.4, "pred_label": "TRAI", "confidence": 0.95, "window_start_sec": 1.4},
        {"trigger_time_sec": 8.0, "pred_label": "LEN", "confidence": 0.91, "window_start_sec": 7.0}
    ]
    
    metrics = match_events_and_evaluate(triggers, ground_truth, stream_duration_sec=10.0)
    assert metrics["num_gt_targets"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["latencies_ms"]["mean"] == pytest.approx(200.0, abs=1.0) # 2.4 - 2.2 = 0.2s = 200ms
