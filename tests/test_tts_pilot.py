"""
Automated regression tests for the Vietnamese Neural TTS Pilot.
Verifies manifest integrity, WAV decoding, SHA-256 bit-exactness,
headroom/clipping compliance, and absence of phoneme truncation.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import pytest

MANIFEST_PATH = "experiments/synthetic_kws_tts_pilot/tts_pilot_manifest.csv"

@pytest.fixture(scope="module")
def manifest_df():
    assert os.path.exists(MANIFEST_PATH), f"Manifest file missing: {MANIFEST_PATH}"
    df = pd.read_csv(MANIFEST_PATH)
    return df

def test_manifest_structure(manifest_df):
    assert len(manifest_df) == 32, f"Expected 32 rows, got {len(manifest_df)}"
    required_cols = [
        "clip_id", "target_keyword", "intended_text", "ipa_phonetic_target",
        "vietnamese_tone", "tts_engine", "voice_id", "voice_gender",
        "rate_parameter", "pitch_parameter", "natural_sample_rate",
        "natural_duration_sec", "natural_wav_relative_path", "natural_sha256",
        "kws_sample_rate", "kws_duration_sec", "kws_active_speech_duration_sec",
        "kws_onset_sec", "kws_peak_amplitude", "kws_speech_rms_dbfs",
        "kws_wav_relative_path", "kws_sha256", "truncation_detected",
        "clipping_detected", "objective_signal_check", "subjective_intelligibility_status"
    ]
    for col in required_cols:
        assert col in manifest_df.columns, f"Missing required column: {col}"
    
    # Check class balance (8 per target word)
    counts = manifest_df["target_keyword"].value_counts().to_dict()
    assert counts == {"TRAI": 8, "PHAI": 8, "LEN": 8, "XUONG": 8}
    
    # Check gender balance (16 female, 16 male)
    gender_counts = manifest_df["voice_gender"].value_counts().to_dict()
    assert gender_counts == {"female": 16, "male": 16}

def test_audio_files_exist_and_sha256(manifest_df):
    for _, row in manifest_df.iterrows():
        # Natural WAV
        nat_path = row["natural_wav_relative_path"]
        assert os.path.exists(nat_path), f"Natural WAV missing: {nat_path}"
        with open(nat_path, "rb") as f:
            nat_hash = hashlib.sha256(f.read()).hexdigest()
        assert nat_hash == row["natural_sha256"], f"SHA-256 mismatch for {nat_path}"

        # KWS WAV
        kws_path = row["kws_wav_relative_path"]
        assert os.path.exists(kws_path), f"KWS WAV missing: {kws_path}"
        with open(kws_path, "rb") as f:
            kws_hash = hashlib.sha256(f.read()).hexdigest()
        assert kws_hash == row["kws_sha256"], f"SHA-256 mismatch for {kws_path}"

def test_audio_properties_and_headroom(manifest_df):
    for _, row in manifest_df.iterrows():
        # Check Natural WAV
        nat_audio, nat_sr = sf.read(row["natural_wav_relative_path"])
        assert nat_sr == 24000
        assert not np.isnan(nat_audio).any()
        assert not np.isinf(nat_audio).any()
        assert len(nat_audio) > 0

        # Check KWS WAV
        kws_audio, kws_sr = sf.read(row["kws_wav_relative_path"])
        assert kws_sr == 16000
        assert len(kws_audio) == 16000, f"Expected 16000 samples, got {len(kws_audio)}"
        assert not np.isnan(kws_audio).any()
        assert not np.isinf(kws_audio).any()
        
        peak = np.max(np.abs(kws_audio))
        assert peak < 0.999, f"Clipping detected in {row['clip_id']}: peak={peak}"
        assert peak > 0.10, f"Signal suspiciously low in {row['clip_id']}: peak={peak}"

def test_no_phoneme_truncation(manifest_df):
    assert not manifest_df["truncation_detected"].any(), "Truncation was detected in pilot clips"
    for _, row in manifest_df.iterrows():
        onset = row["kws_onset_sec"]
        active_dur = row["kws_active_speech_duration_sec"]
        total_dur = row["kws_duration_sec"]
        assert onset >= 0.050, f"Onset too early ({onset}s) in {row['clip_id']}"
        assert onset + active_dur <= total_dur, f"Speech exceeds clip boundary ({onset + active_dur}s > {total_dur}s)"

def test_phonetic_and_tone_consistency(manifest_df):
    expected_tones = {
        "TRAI": ("Trái", "sắc (high-rising)"),
        "PHAI": ("Phải", "hỏi (mid-falling rising / dipping)"),
        "LEN": ("Lên", "ngang (mid-level)"),
        "XUONG": ("Xuống", "sắc (high-rising)")
    }
    for _, row in manifest_df.iterrows():
        kw = row["target_keyword"]
        exp_text, exp_tone = expected_tones[kw]
        assert row["intended_text"] == exp_text
        assert row["vietnamese_tone"] == exp_tone
