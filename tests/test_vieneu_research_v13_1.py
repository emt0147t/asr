#!/usr/bin/env python3
"""
Unit and Regression Test Suite for Phase 13.1 Preprocessing Remediation.
Workspace: D:\trainidentificationaudio
Target: tests/test_vieneu_research_v13_1.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V13_1_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1"
BUNDLE_DIR = V13_1_DIR / "bundle"
CKPT_DIR = BUNDLE_DIR / "checkpoint"
FIXTURES_DIR = V13_1_DIR / "fixtures"

CKPT_FILE = CKPT_DIR / "variant_v11_hard_negative_weighted_seed42.pt"
EXPECTED_CKPT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

from experiments.vieneu_kws_research_v13_1.bundle.infer import (
    KWSInferenceEngine,
    preprocess_audio_file,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_hash_preservation():
    """Verify that the checkpoint file exists and its hash remains pristine and unmodified."""
    assert CKPT_FILE.exists(), f"Missing checkpoint file: {CKPT_FILE}"
    actual_hash = sha256_of_file(CKPT_FILE)
    assert actual_hash.lower() == EXPECTED_CKPT_HASH.lower(), (
        f"Checkpoint hash altered! Expected {EXPECTED_CKPT_HASH}, got {actual_hash}"
    )


def test_digital_silence_remains_exact_zeros():
    """Verify that preprocessing digital silence yields exact bitwise zeros."""
    silence_wav = FIXTURES_DIR / "dev_01_digital_silence.wav"
    assert silence_wav.exists()
    audio = preprocess_audio_file(silence_wav)
    assert audio.shape == (16000,), f"Unexpected shape: {audio.shape}"
    assert np.all(audio == 0.0), "Digital silence was corrupted or offset from zero!"


def test_quiet_signals_not_amplified():
    """Verify that quiet signals (ambient room tone) are not amplified upward."""
    ambient_wav = FIXTURES_DIR / "dev_02_quiet_ambient_room.wav"
    assert ambient_wav.exists()
    raw_audio, sr = sf.read(ambient_wav, dtype="float32")
    raw_peak = float(np.max(np.abs(raw_audio)))
    assert raw_peak < 0.02, f"Raw ambient test fixture has unexpected peak: {raw_peak}"

    processed = preprocess_audio_file(ambient_wav)
    processed_peak = float(np.max(np.abs(processed)))

    # Processed peak must equal raw peak (0 gain applied)
    assert np.isclose(processed_peak, raw_peak, atol=1e-5), (
        f"Quiet signal was amplified! Raw peak: {raw_peak}, Processed peak: {processed_peak}"
    )


def test_high_peak_clamping_behavior():
    """Verify that signals with peak > 1.0 are clamped by dividing by max(abs(x))."""
    high_trai_wav = FIXTURES_DIR / "dev_07_high_amplitude_trai_peak180.wav"
    high_xuong_wav = FIXTURES_DIR / "dev_08_high_amplitude_xuong_peak250.wav"

    for wav_path, expected_raw in [(high_trai_wav, 1.8), (high_xuong_wav, 2.5)]:
        assert wav_path.exists()
        raw_audio, sr = sf.read(wav_path, dtype="float32")
        raw_peak = float(np.max(np.abs(raw_audio)))
        assert np.isclose(raw_peak, expected_raw, atol=1e-2)

        processed = preprocess_audio_file(wav_path)
        processed_peak = float(np.max(np.abs(processed)))
        # Must be scaled down so max absolute value is 1.0
        assert np.isclose(processed_peak, 1.0, atol=1e-5), (
            f"Failed to clamp high peak to 1.0 on {wav_path.name}: got {processed_peak}"
        )


def test_five_class_mapping_and_threshold_contract():
    """Verify model configuration preserves 5 classes and theta_cmd = 0.20."""
    cfg_file = BUNDLE_DIR / "model_config.json"
    assert cfg_file.exists()
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["classes"] == ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
    assert cfg["calibrated_threshold"] == 0.20
    assert cfg["parameters"] == 138469
    assert cfg["normalization_policy"] == "clamped_only_if_max_abs_greater_than_1.0"


def test_remediation_eliminates_ambient_false_alarm():
    """Verify that remediated inference correctly rejects quiet ambient room tone."""
    engine = KWSInferenceEngine(checkpoint_path=CKPT_FILE, theta_cmd=0.20, device="cpu")
    ambient_wav = FIXTURES_DIR / "dev_02_quiet_ambient_room.wav"

    res = engine.predict(ambient_wav)
    assert res["predicted_class"] == "REJECT", (
        f"Quiet ambient noise was falsely accepted as {res['predicted_class']}!"
    )
    assert res["rejection_status"] == "REJECTED_LOW_CONFIDENCE"
    assert res["probabilities"]["REJECT"] > 0.90, (
        f"Expected high rejection probability on ambient noise, got: {res['probabilities']}"
    )


def test_remediation_matches_training_reference():
    """Verify that remediated preprocessing matches train_v11.py BoundedDatasetV11 reference."""
    comp_file = V13_1_DIR / "preprocessing_comparison_results.json"
    assert comp_file.exists(), f"Missing comparison report: {comp_file}"
    with open(comp_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for item in results:
        assert item["p13_matches_training_ref"] is True, (
            f"Remediated inference diverged from training reference on fixture: {item['fixture']}"
        )


def test_governance_string_invariants():
    """Ensure no forbidden strings exist in Phase 13.1 artifacts."""
    forbidden = "data/" + "raw/" + "field_recordings"
    for p in V13_1_DIR.rglob("*"):
        if p.is_file() and p.suffix in [".py", ".md", ".json"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert forbidden not in text, f"Forbidden path string detected in {p}"
