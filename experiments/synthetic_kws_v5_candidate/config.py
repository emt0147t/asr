"""
Configuration structures for Synthetic KWS V5 Candidate Generator.
Provides explicit configuration objects and deterministic parameter sampling.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class V5VoiceProfile:
    """
    Simulated vocal tract and glottal source parameters for a synthetic speaker.
    Parameters are tunable synthesis hypotheses, NOT calibrated human measurements.
    """
    voice_id: str
    vtl_scale: float           # Vocal tract length scale factor (0.85 to 1.15)
    f0_base: float             # Base fundamental frequency in Hz (95 to 250 Hz)
    open_quotient: float       # Glottal pulse open quotient (0.40 to 0.70)
    spectral_tilt_db: float    # Glottal spectral tilt attenuation per octave (-6 to -14 dB)
    jitter_pct: float          # Micro-intonation frequency perturbation (0.4 to 1.2%)


@dataclass
class V5AcousticConfig:
    """
    Full specification for generating a single 1.0s synthetic speech clip.
    """
    sample_id: str
    class_label: str           # TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE
    intended_text: str         # Lexical representation or descriptor
    seed: int                  # Deterministic per-sample seed
    voice_profile: V5VoiceProfile
    duration_sec: float        # Active utterance duration (0.28 to 0.70s)
    onset_offset_sec: float    # Temporal offset within 1.0s window (0.02 to 0.35s)
    snr_db: float              # Additive synthetic noise SNR (15 to 40 dB, or inf)
    noise_type: str            # "none", "room_tone", "dither", "hum"
    peak_target: float = 0.85  # Normalization peak ceiling
    provenance: str = "synthetic_parametric_candidate"
    generator_version: str = "v5.0-pilot"


@dataclass
class V5NoiseConfig:
    """
    Specification for a pure non-target / ambient SILENCE clip.
    """
    sample_id: str
    regime: str                # "exact_zeros", "low_level_dither", "room_tone", "synthetic_hum", "non_speech_transient"
    seed: int
    duration_sec: float = 1.0
    sr: int = 16000
    provenance: str = "synthetic_parametric_candidate"
    generator_version: str = "v5.0-pilot"


# Documented synthetic vocabulary for UNKNOWN non-target syllables
UNKNOWN_PILOT_VOCABULARY = [
    ("chào", "falling"),   ("bạn", "heavy"),    ("đi", "level"),      ("xe", "level"),
    ("nhà", "falling"),    ("số", "rising"),    ("ngày", "falling"),  ("học", "heavy"),
    ("hoa", "level"),      ("mây", "level"),    ("rừng", "falling"),  ("núi", "rising"),
    ("sông", "level"),     ("biển", "dipping"), ("trời", "falling"),  ("pha", "level"),
    ("mái", "rising"),     ("tai", "level"),    ("đèn", "falling"),   ("cuống", "rising")
]

# Standard target words specification
TARGET_WORDS_SPEC = {
    "TRAI": {"word": "trái", "tone": "rising", "class_id": 0},
    "PHAI": {"word": "phải", "tone": "dipping", "class_id": 1},
    "LEN": {"word": "lên", "tone": "level", "class_id": 2},
    "XUONG": {"word": "xuống", "tone": "rising", "class_id": 3}
}
