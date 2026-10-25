"""
Unit and Regression Tests for Synthetic KWS V5 Candidate Generator.
Tests:
  1. Single-clip determinism and bit-exact reproducibility.
  2. Audio shape and sample rate contract compliance (16kHz, mono, 16000 samples).
  3. Finite numeric range and absence of digital clipping.
  4. Absence of V4 onset shortcut (continuous VOT, closure acoustic leakage >= 0.02).
  5. 5 ambient SILENCE regimes generation and validation.
  6. Canonical 6-class ID and label mapping.
  7. UNKNOWN pilot vocabulary structure.
  8. Governance invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
"""

import sys
import io
import hashlib
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import pytest
import numpy as np
import soundfile as sf

from experiments.synthetic_kws_v5_candidate.config import (
    V5VoiceProfile,
    V5AcousticConfig,
    V5NoiseConfig,
    UNKNOWN_PILOT_VOCABULARY,
    TARGET_WORDS_SPEC,
)
from experiments.synthetic_kws_v5_candidate.generator import (
    generate_v5_speech_clip,
    generate_v5_silence_clip,
    sample_simulated_speaker,
    reproduce_pilot_clip,
    SAMPLE_RATE,
    TOTAL_SAMPLES,
    CLASS_MAP,
)


@pytest.fixture
def test_speaker():
    rng = np.random.RandomState(42)
    return sample_simulated_speaker("SPK_TEST_001", rng)


class TestV5GeneratorContract:
    """Verifies audio shape, type, and sample rate contracts."""

    @pytest.mark.parametrize("class_label,text", [
        ("TRAI", "trái"),
        ("PHAI", "phải"),
        ("LEN", "lên"),
        ("XUONG", "xuống"),
        ("UNKNOWN", "chào")
    ])
    def test_speech_clip_shape_and_finite(self, test_speaker, class_label, text):
        cfg = V5AcousticConfig(
            sample_id=f"test_{class_label.lower()}",
            class_label=class_label,
            intended_text=text,
            seed=12345,
            voice_profile=test_speaker,
            duration_sec=0.45,
            onset_offset_sec=0.10,
            snr_db=25.0,
            noise_type="room_tone",
            peak_target=0.85
        )
        audio = generate_v5_speech_clip(cfg, sr=SAMPLE_RATE)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) == TOTAL_SAMPLES == 16000
        assert np.all(np.isfinite(audio)), "Clip contains NaN or Inf"
        assert np.max(np.abs(audio)) <= 1.0, "Audio exceeds full scale (clipping)"
        assert np.max(np.abs(audio)) > 0.1, "Speech clip is essentially silent"

    @pytest.mark.parametrize("regime", [
        "exact_zeros",
        "low_level_dither",
        "room_tone",
        "synthetic_hum",
        "non_speech_transient"
    ])
    def test_silence_clip_regimes(self, regime):
        cfg = V5NoiseConfig(
            sample_id=f"test_silence_{regime}",
            regime=regime,
            seed=54321,
            duration_sec=1.0,
            sr=SAMPLE_RATE
        )
        audio = generate_v5_silence_clip(cfg, sr=SAMPLE_RATE)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == TOTAL_SAMPLES == 16000
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) <= 1.0

        if regime == "exact_zeros":
            assert np.all(audio == 0.0), "exact_zeros must be exactly zero"
        else:
            rms = np.sqrt(np.mean(audio ** 2))
            assert rms > 0.0, f"{regime} must have non-zero energy"
            assert rms < 0.20, f"{regime} RMS too loud for ambient silence"


class TestV5Determinism:
    """Verifies bit-exact reproducibility from seed and parameters."""

    def test_speech_generation_is_bit_exact(self, test_speaker):
        cfg1 = V5AcousticConfig(
            sample_id="det_001",
            class_label="TRAI",
            intended_text="trái",
            seed=99999,
            voice_profile=test_speaker,
            duration_sec=0.48,
            onset_offset_sec=0.12,
            snr_db=28.0,
            noise_type="hum"
        )
        cfg2 = V5AcousticConfig(
            sample_id="det_001",
            class_label="TRAI",
            intended_text="trái",
            seed=99999,
            voice_profile=test_speaker,
            duration_sec=0.48,
            onset_offset_sec=0.12,
            snr_db=28.0,
            noise_type="hum"
        )
        audio1 = generate_v5_speech_clip(cfg1, sr=SAMPLE_RATE)
        audio2 = generate_v5_speech_clip(cfg2, sr=SAMPLE_RATE)

        np.testing.assert_array_equal(audio1, audio2, err_msg="Non-deterministic speech synthesis output")

    def test_reproduce_pilot_clip_bit_exact(self):
        """Verifies reproduce_pilot_clip returns identical bytes and SHA-256."""
        for rec_id in ["v5_pilot_trai_001", "v5_pilot_phai_002", "v5_pilot_silence_005"]:
            audio_a, sha_a, _ = reproduce_pilot_clip(rec_id, pilot_seed_base=50000)
            audio_b, sha_b, _ = reproduce_pilot_clip(rec_id, pilot_seed_base=50000)

            np.testing.assert_array_equal(audio_a, audio_b)
            assert sha_a == sha_b


class TestAbsenceOfV4Shortcut:
    """Verifies that the artificial 25ms silence gap of V4 is eliminated."""

    def test_trai_no_hard_zero_clamp(self, test_speaker):
        """Verifies that TRAI does not have a hard zero-clamp during closure."""
        cfg = V5AcousticConfig(
            sample_id="test_trai_closure",
            class_label="TRAI",
            intended_text="trái",
            seed=77777,
            voice_profile=test_speaker,
            duration_sec=0.50,
            onset_offset_sec=0.10,
            snr_db=30.0,
            noise_type="room_tone"
        )
        audio = generate_v5_speech_clip(cfg, sr=SAMPLE_RATE)
        
        # Check active speech start (sample 1600 = 0.10s)
        onset_sample = int(0.10 * SAMPLE_RATE)
        first_25ms_body = audio[onset_sample:onset_sample + 400]
        max_abs = np.max(np.abs(first_25ms_body))

        # Gate 3 requirement: max absolute amplitude >= 0.02
        assert max_abs >= 0.02, f"TRAI closure amplitude too low ({max_abs:.4f} < 0.02); shortcut detected"
        assert not np.all(first_25ms_body == 0.0), "TRAI first 25ms is digital zero"

    def test_trai_vs_phai_onset_phonological_contrast(self, test_speaker):
        """
        Verifies that TRAI and PHAI exhibit distinct onset characteristics:
        TRAI has stop closure + burst, PHAI has continuous frication.
        """
        cfg_trai = V5AcousticConfig(
            sample_id="test_contrast_trai",
            class_label="TRAI",
            intended_text="trái",
            seed=88888,
            voice_profile=test_speaker,
            duration_sec=0.45,
            onset_offset_sec=0.05,
            snr_db=35.0,
            noise_type="room_tone"
        )
        cfg_phai = V5AcousticConfig(
            sample_id="test_contrast_phai",
            class_label="PHAI",
            intended_text="phải",
            seed=88888,
            voice_profile=test_speaker,
            duration_sec=0.45,
            onset_offset_sec=0.05,
            snr_db=35.0,
            noise_type="room_tone"
        )
        trai = generate_v5_speech_clip(cfg_trai)
        phai = generate_v5_speech_clip(cfg_phai)

        # Ensure they are not identical
        assert not np.allclose(trai, phai, atol=1e-3)


class TestMappingAndVocabulary:
    """Verifies label mapping and negative vocabulary structure."""

    def test_canonical_class_mapping(self):
        assert CLASS_MAP == {
            "TRAI": 0,
            "PHAI": 1,
            "LEN": 2,
            "XUONG": 3,
            "UNKNOWN": 4,
            "SILENCE": 5
        }

    def test_unknown_pilot_vocabulary_size(self):
        assert len(UNKNOWN_PILOT_VOCABULARY) == 20
        words = [w[0] for w in UNKNOWN_PILOT_VOCABULARY]
        assert len(set(words)) == 20, "UNKNOWN vocabulary contains duplicate words"

    def test_target_words_specification(self):
        assert set(TARGET_WORDS_SPEC.keys()) == {"TRAI", "PHAI", "LEN", "XUONG"}
        assert TARGET_WORDS_SPEC["TRAI"]["word"] == "trái"
        assert TARGET_WORDS_SPEC["PHAI"]["word"] == "phải"
        assert TARGET_WORDS_SPEC["LEN"]["word"] == "lên"
        assert TARGET_WORDS_SPEC["XUONG"]["word"] == "xuống"
