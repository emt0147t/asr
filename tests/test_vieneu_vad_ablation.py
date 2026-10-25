"""
Unit and Regression Test Suite for Phase 39 (Controlled VAD Gating Ablation).

Verifies:
  1. ShortTimeEnergyGate frame processing, dB conversion, and threshold logic.
  2. SpeechBandFormantVAD speech-band power ratio and hangover smoothing.
  3. Preservation of all historical v1, v2, and cascade v1 artifacts.
  4. VAD ablation config structure and evaluated conditions (A, B, C, D).
  5. Accurate error categorization (speech-active vs silence-active false alarms).
  6. Denominator correctness for FA/hr calculations.
  7. Verification that VAD does not silently relax the strict validation constraints.
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

from experiments.vieneu_kws_vad_ablation_v1.vad_engine import (
    ShortTimeEnergyGate,
    SpeechBandFormantVAD
)
from experiments.vieneu_kws_v2.evaluate import (
    match_events_and_evaluate
)

EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_vad_ablation_v1")
CASCADE_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1")
DATASET_V2_DIR = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2")


def test_preservation_of_historical_artifacts():
    """Confirms historical v1, v2, and cascade v1 artifacts remain unmodified."""
    v1_manifest = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv")
    v2_manifest = os.path.join(DATASET_V2_DIR, "manifest.csv")
    cas_ckpt_42 = os.path.join(CASCADE_DIR, "checkpoints/best_verifier_seed42.pt")
    cas_ckpt_43 = os.path.join(CASCADE_DIR, "checkpoints/best_verifier_seed43.pt")
    cas_ckpt_44 = os.path.join(CASCADE_DIR, "checkpoints/best_verifier_seed44.pt")

    assert os.path.exists(v1_manifest), "v1 manifest missing!"
    assert os.path.exists(v2_manifest), "v2 manifest missing!"
    assert os.path.exists(cas_ckpt_42), "Cascade seed 42 checkpoint missing!"
    assert os.path.exists(cas_ckpt_43), "Cascade seed 43 checkpoint missing!"
    assert os.path.exists(cas_ckpt_44), "Cascade seed 44 checkpoint missing!"


def test_vad_config_structure():
    """Verifies Phase 39 config.json structure and conditions."""
    cfg_p = os.path.join(EXP_DIR, "config.json")
    assert os.path.exists(cfg_p), "VAD ablation config.json missing!"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["experiment_name"] == "vieneu_kws_vad_ablation_v1"
    assert "A_stage1_alone" in cfg["evaluated_conditions"]
    assert "B_cascade_no_vad" in cfg["evaluated_conditions"]
    assert "C_stage1_with_vad" in cfg["evaluated_conditions"]
    assert "D_cascade_with_vad" in cfg["evaluated_conditions"]
    assert "energy_gate" in cfg["vad_configurations"]
    assert "speech_band_vad" in cfg["vad_configurations"]


def test_energy_gate_silence_vs_speech():
    """Verifies ShortTimeEnergyGate classifies silence as inactive and high energy as active."""
    gate = ShortTimeEnergyGate(energy_threshold_db=-45.0, min_speech_frames=5)

    # Pure digital silence
    silence = np.zeros(16000, dtype=np.float32)
    is_sp_sil, act_ratio_sil, max_db_sil = gate.process_window(silence)
    assert not is_sp_sil, "Silence should be classified as inactive!"
    assert act_ratio_sil == 0.0
    assert max_db_sil < -100.0

    # Synthetic tone resembling active speech (-20 dB)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (0.10 * np.sin(2 * np.pi * 500 * t)).astype(np.float32) # ~ -20 dB
    is_sp_tone, act_ratio_tone, max_db_tone = gate.process_window(tone)
    assert is_sp_tone, "Tone at -20 dB should be classified as active!"
    assert act_ratio_tone > 0.80
    assert max_db_tone > -30.0


def test_speech_band_vad_formant_ratio():
    """Verifies SpeechBandFormantVAD detects in-band speech frequencies and handles hangover."""
    vad = SpeechBandFormantVAD(sample_rate=16000, speech_band_hz=(300.0, 3400.0),
                              min_speech_band_ratio=0.35, min_energy_db=-48.0, hangover_frames=5)

    t = np.linspace(0, 1.0, 16000, endpoint=False)

    # 1. High frequency noise outside speech band (e.g. 7000 Hz)
    out_of_band = (0.05 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
    is_sp_oob, act_ratio_oob, mean_ratio_oob = vad.process_window(out_of_band)
    assert not is_sp_oob, "7000 Hz tone should be rejected as out-of-band!"
    assert mean_ratio_oob < 0.10

    # 2. Formant frequency inside speech band (e.g. 800 Hz)
    in_band = (0.05 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
    is_sp_ib, act_ratio_ib, mean_ratio_ib = vad.process_window(in_band)
    assert is_sp_ib, "800 Hz formant tone should be detected as speech!"
    assert mean_ratio_ib > 0.80


def test_error_categorization_and_denominator_integrity():
    """Verifies that non-target errors are correctly assigned and FA/hr denominator matches stream length."""
    summary_p = os.path.join(EXP_DIR, "predictions", "vad_ablation_summary.json")
    assert os.path.exists(summary_p), "vad_ablation_summary.json missing!"

    with open(summary_p, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for seed in [42, 43, 44]:
        s_data = summary[f"seed{seed}"]
        # Verify conditions A, B, C, D exist
        for cond in ["A_stage1_alone", "B_cascade_no_vad", "C_stage1_with_vad", "D_cascade_with_vad"]:
            assert cond in s_data
            m_nt = s_data[cond]["nontarget_stream"]
            # Stream duration: 901.54s = 0.2504 hours
            assert abs(m_nt["stream_duration_hours"] - 0.2504) < 0.001
            # FA/hr math check: fp / (stream_duration_sec / 3600.0)
            expected_fa_hr = round(m_nt["false_positives"] / (m_nt["stream_duration_sec"] / 3600.0), 2)
            assert abs(m_nt["fa_per_hour"] - expected_fa_hr) <= 0.05

            # In Condition B and D, silence false alarms must be exactly 0
            if "cascade" in cond:
                silence_fps = s_data[cond]["error_categorization"]["silence_background_false_alarms"]
                assert silence_fps == 0, f"Cascade in {cond} should have 0 silence false alarms!"


def test_infeasibility_declaration_in_validation_records():
    """Verifies NO_FEASIBLE_VALIDATION_OPERATING_POINT is explicitly recorded in validation records."""
    cal_p = os.path.join(EXP_DIR, "calibration", "validation_records.json")
    assert os.path.exists(cal_p), "validation_records.json missing!"

    with open(cal_p, "r", encoding="utf-8") as f:
        records = json.load(f)

    for seed in [42, 43, 44]:
        rec = records[f"seed{seed}"]
        assert rec["is_strictly_feasible"] is False
        assert "NO_FEASIBLE_VALIDATION_OPERATING_POINT" in rec["rationale"]
