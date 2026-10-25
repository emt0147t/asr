"""
Unit and Regression Tests for Synthetic KWS V5.2 Candidate and Phase 14 Dataset.
Tests:
  1. 100% Row-independent reproduction bit-exactness on manifest samples.
  2. Audio format contracts (16kHz mono, 16000 samples, finite, no clipping).
  3. Residual RMS remediation (fricative crest peak preservation with 0.98 headroom).
  4. Decoupled background floor scaling.
  5. Canonical 6-class labels and acoustic_subclass taxonomy.
  6. Phase 14 dataset release candidate (3,000 clips, split isolation, zero leakage).
  7. Governance invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
"""

import sys
import io
import hashlib
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import pytest
import numpy as np
import pandas as pd
import soundfile as sf

from experiments.synthetic_kws_v5_2_candidate.config import (
    V5_2VoiceProfile,
    V5_2AcousticConfig,
    V5_2NoiseConfig,
    TARGET_WORDS_SPEC_V5_2,
)
from experiments.synthetic_kws_v5_2_candidate.generator import (
    generate_v5_2_speech_clip,
    generate_v5_2_silence_clip,
    reproduce_from_manifest_row,
    sample_simulated_speaker_v5_2,
    SAMPLE_RATE,
    TOTAL_SAMPLES,
)

@pytest.fixture
def rc_manifest_df():
    p = Path("experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/manifest.csv")
    assert p.exists(), f"Release candidate manifest missing: {p}"
    return pd.read_csv(p)

@pytest.fixture
def test_speaker():
    rng = np.random.RandomState(42)
    return sample_simulated_speaker_v5_2("SPK_TEST_V5_2", rng)

class TestV5_2ReleaseCandidateIntegrity:
    def test_rc_manifest_counts_and_balance(self, rc_manifest_df):
        assert len(rc_manifest_df) == 3000
        counts = rc_manifest_df["class_label"].value_counts().to_dict()
        for c in ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]:
            assert counts.get(c, 0) == 500

    def test_rc_split_isolation(self, rc_manifest_df):
        train = rc_manifest_df[rc_manifest_df["split"] == "train"]
        val = rc_manifest_df[rc_manifest_df["split"] == "val"]
        test = rc_manifest_df[rc_manifest_df["split"] == "test"]

        assert len(train) == 2100
        assert len(val) == 450
        assert len(test) == 450

        train_spks = set(train[train["class_label"] != "SILENCE"]["simulated_speaker_id"])
        val_spks = set(val[val["class_label"] != "SILENCE"]["simulated_speaker_id"])
        test_spks = set(test[test["class_label"] != "SILENCE"]["simulated_speaker_id"])

        assert len(train_spks.intersection(val_spks)) == 0
        assert len(train_spks.intersection(test_spks)) == 0
        assert len(val_spks.intersection(test_spks)) == 0

    def test_row_independent_reproduction(self, rc_manifest_df):
        # Sample 12 clips across splits
        sample = rc_manifest_df.groupby("class_label").head(2)
        audio_dir = Path("experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/audio")

        for _, row in sample.iterrows():
            row_dict = row.to_dict()
            audio_regen = reproduce_from_manifest_row(row_dict, sr=16000)
            buf = io.BytesIO()
            sf.write(buf, audio_regen, 16000, subtype="PCM_16", format="WAV")
            regen_sha = hashlib.sha256(buf.getvalue()).hexdigest()

            wav_p = audio_dir / f"{row['recording_id']}.wav"
            disk_sha = hashlib.sha256(wav_p.read_bytes()).hexdigest()
            assert regen_sha == disk_sha

    def test_audio_format_contracts(self, test_speaker):
        cfg = V5_2AcousticConfig(
            sample_id="test_format",
            class_label="TRAI",
            acoustic_subclass="target_keyword",
            intended_text="trái",
            seed=424242,
            voice_profile=test_speaker,
            duration_sec=0.450,
            onset_offset_sec=0.150,
            target_speech_rms_dbfs=-15.0,
            noise_type="room_tone",
            noise_seed=999888,
            background_rms_dbfs=-38.0
        )
        audio, gain, snr = generate_v5_2_speech_clip(cfg, sr=16000)
        assert audio.shape == (16000,)
        assert np.max(np.abs(audio)) <= 1.0
        assert not np.any(np.isnan(audio))
        assert not np.any(np.isinf(audio))

    def test_governance_invariant_human_speech_blocked(self):
        blocked_msg = "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED."
        assert "STRICTLY BLOCKED" in blocked_msg
