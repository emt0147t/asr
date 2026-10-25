"""
Configuration structures and vocabularies for Synthetic KWS V5.1 Candidate Generator.
Remediates Phase 13.1 findings:
  1. Decoupled speech gain policy (RMS-standardized speech level).
  2. Decoupled background floor scaling (independent absolute background level).
  3. Explicit two-tier semantic taxonomy (canonical label + acoustic_subclass).
  4. Complete parameter serialization for 100% row-independent reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class V5_1VoiceProfile:
    """
    Simulated vocal tract and glottal source parameters for a synthetic speaker.
    All parameters are explicitly stored in manifest columns for row-independent reproduction.
    """
    voice_id: str
    vtl_scale: float           # Vocal tract length scale factor (0.85 to 1.15)
    f0_base: float             # Base fundamental frequency in Hz (95 to 250 Hz)
    open_quotient: float       # Glottal pulse open quotient (0.40 to 0.70)
    spectral_tilt_db: float    # Glottal spectral tilt attenuation per octave (-6 to -14 dB)
    jitter_pct: float          # Micro-intonation frequency perturbation (0.4 to 1.2%)


@dataclass
class V5_1AcousticConfig:
    """
    Full specification for generating a single 1.0s synthetic speech clip in V5.1.
    Eliminates per-clip peak-normalization and SNR-relative background coupling.
    """
    sample_id: str
    class_label: str                   # TRAI, PHAI, LEN, XUONG, UNKNOWN
    acoustic_subclass: str             # target_keyword, confuser_initial_sharing, etc.
    intended_text: str                 # Lexical transcript or word
    seed: int                          # Synthesis seed
    voice_profile: V5_1VoiceProfile    # Explicit glottal and vocal tract profile
    duration_sec: float                # Utterance active duration (0.30 to 0.65s)
    onset_offset_sec: float            # Temporal onset offset in 1.0s window (0.025 to 0.320s)
    target_speech_rms_dbfs: float      # Standardized speech RMS target before mixing (-16 to -12 dBFS)
    noise_type: str                    # room_tone, dither, hum
    noise_seed: int                    # Decoupled independent noise seed
    background_rms_dbfs: float         # Decoupled independent background floor (-45 to -32 dBFS)
    provenance: str = "synthetic_parametric_v5_1"
    generator_version: str = "v5.1-pilot"


@dataclass
class V5_1NoiseConfig:
    """
    Specification for a non-target ambient audio clip in V5.1.
    """
    sample_id: str
    class_label: str = "SILENCE"       # Canonical model class label
    acoustic_subclass: str = "ambient_room_tone" # Specific acoustic regime
    seed: int = 50000
    duration_sec: float = 1.0
    sr: int = 16000
    provenance: str = "synthetic_parametric_v5_1"
    generator_version: str = "v5.1-pilot"


# Structured 20-word pilot vocabulary with explicit acoustic subclasses
UNKNOWN_PILOT_VOCABULARY_V5_1 = [
    # Initial-sharing confusers (6)
    ("trời", "falling", "confuser_initial_sharing"),
    ("pha", "level", "confuser_initial_sharing"),
    ("tròn", "falling", "confuser_initial_sharing"),
    ("phố", "rising", "confuser_initial_sharing"),
    ("lúa", "rising", "confuser_initial_sharing"),
    ("xinh", "level", "confuser_initial_sharing"),
    # Rhyme-sharing confusers (6)
    ("mái", "rising", "confuser_rhyme_sharing"),
    ("tai", "level", "confuser_rhyme_sharing"),
    ("đèn", "falling", "confuser_rhyme_sharing"),
    ("cuống", "rising", "confuser_rhyme_sharing"),
    ("sen", "level", "confuser_rhyme_sharing"),
    ("bài", "falling", "confuser_rhyme_sharing"),
    # Open vocabulary general words (8)
    ("chào", "falling", "open_vocabulary_general"),
    ("bạn", "heavy", "open_vocabulary_general"),
    ("đi", "level", "open_vocabulary_general"),
    ("xe", "level", "open_vocabulary_general"),
    ("nhà", "falling", "open_vocabulary_general"),
    ("số", "rising", "open_vocabulary_general"),
    ("ngày", "falling", "open_vocabulary_general"),
    ("học", "heavy", "open_vocabulary_general")
]

# Standard target words specification
TARGET_WORDS_SPEC_V5_1 = {
    "TRAI": {"word": "trái", "tone": "rising", "class_id": 0, "acoustic_subclass": "target_keyword"},
    "PHAI": {"word": "phải", "tone": "dipping", "class_id": 1, "acoustic_subclass": "target_keyword"},
    "LEN": {"word": "lên", "tone": "level", "class_id": 2, "acoustic_subclass": "target_keyword"},
    "XUONG": {"word": "xuống", "tone": "rising", "class_id": 3, "acoustic_subclass": "target_keyword"}
}

# 5 explicit silence subclasses
SILENCE_SUBCLASSES_V5_1 = [
    "ambient_digital_zero",
    "ambient_dither",
    "ambient_room_tone",
    "ambient_mains_hum",
    "transient_non_speech"
]
