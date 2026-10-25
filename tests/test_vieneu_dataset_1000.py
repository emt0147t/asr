"""
Automated regression tests for the 1,000-Utterance Vietnamese Synthetic KWS Dataset.
Verifies manifest completeness, class balance, split isolation, audio formats,
SHA-256 bit-exactness, absence of clipping/truncation, and uniqueness of hashes.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import pytest

MANIFEST_PATH = "experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv"
SPLITS_DIR = "experiments/synthetic_kws_dataset_vieneu_1000/splits"

@pytest.fixture(scope="module")
def manifest_df():
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing: {MANIFEST_PATH}"
    df = pd.read_csv(MANIFEST_PATH)
    return df

def test_manifest_structure_and_counts(manifest_df):
    assert len(manifest_df) == 1000, f"Expected 1,000 rows, got {len(manifest_df)}"
    
    # Class balance
    counts = manifest_df["class_label"].value_counts().to_dict()
    assert counts == {"TRAI": 250, "PHAI": 250, "LEN": 250, "XUONG": 250}

    # Required columns
    required_cols = [
        "sample_id", "class_label", "text", "ipa_phonetic_target",
        "vietnamese_tone", "split", "engine", "engine_version",
        "backend", "upstream_repo", "model_checkpoint", "license",
        "voice_id", "voice_gender", "voice_region", "voice_style",
        "temperature", "top_k", "top_p", "repetition_penalty",
        "rng_method", "seed", "raw_wav_path", "kws_wav_path",
        "raw_sample_rate", "kws_sample_rate", "raw_duration_sec",
        "kws_duration_sec", "kws_active_speech_duration_sec",
        "kws_onset_sec", "kws_peak_amplitude", "kws_speech_rms_dbfs",
        "kws_mean_spectral_centroid_hz", "raw_sha256", "kws_sha256",
        "truncation_detected", "clipping_detected",
        "technical_validation_status", "rejection_history",
        "subjective_intelligibility_status"
    ]
    for col in required_cols:
        assert col in manifest_df.columns, f"Missing required column: {col}"

    # Diacritic text check
    expected_texts = {"TRAI": "Trái", "PHAI": "Phải", "LEN": "Lên", "XUONG": "Xuống"}
    for kw, exp_t in expected_texts.items():
        sub = manifest_df[manifest_df["class_label"] == kw]
        assert (sub["text"] == exp_t).all(), f"Text diacritics mismatch for {kw}"

    # Status check
    assert (manifest_df["subjective_intelligibility_status"] == "PENDING_HUMAN_LISTENER_VERIFICATION").all()
    assert (manifest_df["technical_validation_status"] == "VALID").all()

def test_splits_integrity_and_zero_leakage(manifest_df):
    for split_name, exp_count in [("train", 680), ("val", 160), ("test", 160)]:
        p = os.path.join(SPLITS_DIR, f"{split_name}_manifest.csv")
        assert os.path.exists(p), f"Missing split manifest: {p}"
        sdf = pd.read_csv(p)
        assert len(sdf) == exp_count, f"Expected {exp_count} in {split_name}, got {len(sdf)}"
        # Each split must have balanced classes
        s_counts = sdf["class_label"].value_counts().to_dict()
        assert len(set(s_counts.values())) == 1, f"Classes not balanced in {split_name}: {s_counts}"

    # Zero voice leakage
    train_voices = set(manifest_df[manifest_df["split"] == "train"]["voice_id"].unique())
    val_voices = set(manifest_df[manifest_df["split"] == "val"]["voice_id"].unique())
    test_voices = set(manifest_df[manifest_df["split"] == "test"]["voice_id"].unique())

    assert len(train_voices.intersection(val_voices)) == 0, "Voice leakage between train and val"
    assert len(train_voices.intersection(test_voices)) == 0, "Voice leakage between train and test"
    assert len(val_voices.intersection(test_voices)) == 0, "Voice leakage between val and test"

def test_no_duplicate_hashes(manifest_df):
    raw_hashes = manifest_df["raw_sha256"]
    kws_hashes = manifest_df["kws_sha256"]
    assert raw_hashes.nunique() == len(manifest_df), "Duplicate raw audio hashes detected"
    assert kws_hashes.nunique() == len(manifest_df), "Duplicate KWS audio hashes detected"

def test_audio_files_and_signal_health(manifest_df):
    # Sample 40 items across classes and splits for thorough verification
    sample_sub = manifest_df.sample(n=40, random_state=42)
    for _, row in sample_sub.iterrows():
        raw_p = row["raw_wav_path"]
        kws_p = row["kws_wav_path"]

        assert os.path.exists(raw_p), f"Raw WAV missing: {raw_p}"
        assert os.path.exists(kws_p), f"KWS WAV missing: {kws_p}"

        with open(raw_p, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == row["raw_sha256"]
        with open(kws_p, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == row["kws_sha256"]

        raw_a, raw_sr = sf.read(raw_p)
        assert raw_sr == 48000
        assert not np.isnan(raw_a).any()
        assert not np.isinf(raw_a).any()

        kws_a, kws_sr = sf.read(kws_p)
        assert kws_sr == 16000
        assert len(kws_a) == 16000
        assert not np.isnan(kws_a).any()
        assert not np.isinf(kws_a).any()

        peak = np.max(np.abs(kws_a))
        assert peak < 0.999, f"Clipping in {row['sample_id']}: peak={peak}"
        assert peak > 0.05, f"Signal weak in {row['sample_id']}: peak={peak}"

def test_no_phoneme_truncation(manifest_df):
    assert not manifest_df["truncation_detected"].any()
    for _, row in manifest_df.iterrows():
        onset = row["kws_onset_sec"]
        dur = row["kws_active_speech_duration_sec"]
        assert onset >= 0.049, f"Onset too early ({onset}s) in {row['sample_id']}"
        assert onset + dur <= 0.951, f"Speech exceeds safe margin ({onset + dur}s) in {row['sample_id']}"
