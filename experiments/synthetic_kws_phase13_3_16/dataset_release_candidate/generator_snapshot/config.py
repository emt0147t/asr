"""
V5.2 Synthetic KWS Candidate Configuration
Remediates remaining residual RMS variance by:
1. Calibrating target speech RMS range to [-17.0, -14.0] dBFS with 0.98 peak headroom.
2. Eliminating premature soft-knee tanh compression on unvoiced fricatives (PHAI).
3. Maintaining independent absolute background noise floor.
4. Full parameter serialization for 100% row-independent reproducibility.
"""

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class V5_2VoiceProfile:
    """
    Explicit voice and articulatory tract parameters for V5.2.
    """
    voice_id: str
    vtl_scale: float           # Vocal tract length scale factor (0.85 to 1.15)
    f0_base: float             # Base fundamental frequency in Hz (95 to 250 Hz)
    open_quotient: float       # Glottal pulse open quotient (0.40 to 0.70)
    spectral_tilt_db: float    # Glottal spectral tilt attenuation per octave (-6 to -14 dB)
    jitter_pct: float          # Micro-intonation frequency perturbation (0.4 to 1.2%)


@dataclass
class V5_2AcousticConfig:
    """
    Full specification for generating a single 1.0s synthetic speech clip in V5.2.
    """
    sample_id: str
    class_label: str                   # TRAI, PHAI, LEN, XUONG, UNKNOWN
    acoustic_subclass: str             # target_keyword, confuser_initial_sharing, etc.
    intended_text: str                 # Lexical transcript or word
    seed: int                          # Synthesis seed
    voice_profile: V5_2VoiceProfile    # Explicit glottal and vocal tract profile
    duration_sec: float                # Utterance active duration (0.35 to 0.60s)
    onset_offset_sec: float            # Temporal onset offset in 1.0s window (0.050 to 0.250s)
    target_speech_rms_dbfs: float      # Standardized speech RMS target (-17.0 to -14.0 dBFS)
    noise_type: str                    # room_tone, dither, hum
    noise_seed: int                    # Decoupled independent noise seed
    background_rms_dbfs: float         # Decoupled independent background floor (-42.0 to -32.0 dBFS)
    provenance: str = "synthetic_parametric_v5_2"
    generator_version: str = "v5.2-candidate"


@dataclass
class V5_2NoiseConfig:
    """
    Specification for a non-target ambient audio clip in V5.2.
    """
    sample_id: str
    class_label: str = "SILENCE"       # Canonical model class label
    acoustic_subclass: str = "ambient_room_tone" # Specific acoustic regime
    seed: int = 50000
    duration_sec: float = 1.0
    sr: int = 16000
    provenance: str = "synthetic_parametric_v5_2"
    generator_version: str = "v5.2-candidate"


# Structured pilot vocabulary
UNKNOWN_PILOT_VOCABULARY_V5_2 = [
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

TARGET_WORDS_SPEC_V5_2: Dict[str, Dict[str, Any]] = {
    "TRAI": {
        "word": "trái",
        "class_id": 0,
        "tone": "rising",
        "acoustic_subclass": "target_keyword"
    },
    "PHAI": {
        "word": "phải",
        "class_id": 1,
        "tone": "falling_broken",
        "acoustic_subclass": "target_keyword"
    },
    "LEN": {
        "word": "lên",
        "class_id": 2,
        "tone": "level",
        "acoustic_subclass": "target_keyword"
    },
    "XUONG": {
        "word": "xuống",
        "class_id": 3,
        "tone": "rising",
        "acoustic_subclass": "target_keyword"
    }
}
