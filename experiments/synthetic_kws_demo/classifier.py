"""
Offline KWS Inference Engine for Synthetic Vietnamese Keyword Spotting.
Loads the frozen candidate checkpoint (V4 Arm B: arm_b_onset_no_resample.pt),
enforces strict audio input validation, and extracts Log-Mel spectrograms.
"""

import sys
import os
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import soundfile as sf
import torch

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
CLASS_LABELS = CLASS_NAMES_6
DEFAULT_CHECKPOINT = Path("experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt")

SYNTHETIC_DEMO_WARNING = (
    "SYNTHETIC-DEMO-ONLY: This inference engine uses a model trained exclusively on synthetic "
    "acoustic fixtures. Its high synthetic accuracy (99.33% on V4 native holdout) DOES NOT establish "
    "real-world speech accuracy, Vietnamese regional accent invariance, live microphone reliability, "
    "or mobile edge deployment performance. HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED."
)


# ============================================================================
# Audio Validation Exceptions
# ============================================================================

class AudioValidationError(Exception):
    """Base exception for all audio input validation failures."""
    pass

class AudioFileNotFoundError(AudioValidationError):
    """Raised when the specified WAV file does not exist on disk."""
    pass

class EmptyAudioError(AudioValidationError):
    """Raised when the WAV file is empty (0 bytes or 0 samples)."""
    pass

class InvalidSampleRateError(AudioValidationError):
    """Raised when the input sample rate differs from the required 16,000 Hz."""
    pass

class InvalidChannelCountError(AudioValidationError):
    """Raised when the input is multichannel (stereo) instead of mono (1 channel)."""
    pass

class InvalidDurationError(AudioValidationError):
    """Raised when the audio sample count differs from the required 16,000 samples (1.0 second)."""
    pass

class CorruptedAudioError(AudioValidationError):
    """Raised when the file format is malformed or unreadable by SoundFile."""
    pass

class CheckpointNotFoundError(Exception):
    """Raised when the specified model checkpoint cannot be located."""
    pass


# ============================================================================
# Audio Validator
# ============================================================================

class AudioValidator:
    """
    Enforces strict preprocessing and input policy:
      - File existence and non-zero size
      - Valid WAV/RIFF header readable as PCM
      - Exactly 16,000 Hz sample rate (no silent resampling)
      - Exactly 1 channel (mono)
      - Exactly 16,000 samples (1.0 second window)
      - Preserves full audio buffer for ambient background noise
    """
    REQUIRED_SAMPLE_RATE = 16000
    REQUIRED_CHANNELS = 1
    REQUIRED_SAMPLES = 16000

    @classmethod
    def validate_and_load(cls, wav_path: Path) -> Tuple[np.ndarray, int]:
        if not wav_path.exists():
            raise AudioFileNotFoundError(f"Audio file not found: {wav_path}")
            
        file_size = wav_path.stat().st_size
        if file_size == 0:
            raise EmptyAudioError(f"Audio file is empty (0 bytes): {wav_path}")
            
        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            raise CorruptedAudioError(f"Failed to parse audio header ({wav_path.name}): {str(e)}") from e
            
        if info.frames == 0:
            raise EmptyAudioError(f"Audio file contains 0 frames: {wav_path}")
            
        if info.samplerate != cls.REQUIRED_SAMPLE_RATE:
            raise InvalidSampleRateError(
                f"Unsupported sample rate: {info.samplerate} Hz. Expected exactly {cls.REQUIRED_SAMPLE_RATE} Hz. "
                "Silent resampling is prohibited by policy."
            )
            
        if info.channels != cls.REQUIRED_CHANNELS:
            raise InvalidChannelCountError(
                f"Unsupported channel count: {info.channels} channels. Expected exactly {cls.REQUIRED_CHANNELS} (mono)."
            )
            
        if info.frames != cls.REQUIRED_SAMPLES:
            raise InvalidDurationError(
                f"Unsupported duration: {info.frames} samples ({info.frames / info.samplerate:.3f}s). "
                f"Expected exactly {cls.REQUIRED_SAMPLES} samples (1.000s)."
            )
            
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            raise CorruptedAudioError(f"Failed to read audio samples from {wav_path}: {str(e)}") from e
            
        return audio, sr


# ============================================================================
# Inference Result Data Structure
# ============================================================================

@dataclass
class PredictionResult:
    is_valid: bool
    wav_path: str
    predicted_label: str
    predicted_class_id: int
    confidence: float
    class_probabilities: Dict[str, float]
    checkpoint_path: str
    checkpoint_sha256: str
    preprocessing_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    error_message: Optional[str] = None
    warning: str = SYNTHETIC_DEMO_WARNING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Offline KWS Classifier
# ============================================================================

class OfflineKWSClassifier:
    """
    Offline inference classifier wrapping VietnameseTCResNet and LogMelFeatureExtractor.
    Guarantees deterministic, frozen inference using V4 Arm B checkpoint.
    """
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise CheckpointNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
            
        self.device = torch.device(device)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest().upper()
        
        # Load model architecture
        self.model = VietnameseTCResNet(num_classes=6, dropout_p=0.0)
        ckpt_dict = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model state dict
        if "model_state_dict" in ckpt_dict:
            state_dict = ckpt_dict["model_state_dict"]
        else:
            state_dict = ckpt_dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Log-Mel feature extractor frontend (16k, 40 mels, 101 frames)
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
        self.extractor.to(self.device)
        self.extractor.eval()

    def predict(self, wav_path: Path | str) -> PredictionResult:
        """
        Executes end-to-end offline inference on a WAV file.
        Measures fine-grained latency and enforces strict validation.
        """
        wav_p = Path(wav_path)
        t_start = time.perf_counter()
        
        try:
            # 1. Validation & Loading
            t_load_start = time.perf_counter()
            audio, sr = AudioValidator.validate_and_load(wav_p)
            
            # 2. Preprocessing / Log-Mel extraction
            inp_audio = torch.from_numpy(audio).float().to(self.device)
            with torch.no_grad():
                log_mel = self.extractor(inp_audio)  # Shape: (1, 1, 40, 101)
            t_prep_end = time.perf_counter()
            prep_ms = (t_prep_end - t_load_start) * 1000.0
            
            # 3. Model forward pass
            t_inf_start = time.perf_counter()
            with torch.no_grad():
                logits = self.model(log_mel)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            t_inf_end = time.perf_counter()
            inf_ms = (t_inf_end - t_inf_start) * 1000.0
            
            pred_cid = int(np.argmax(probs))
            pred_label = CLASS_NAMES_6[pred_cid]
            conf = float(probs[pred_cid])
            
            prob_dict = {CLASS_NAMES_6[i]: float(probs[i]) for i in range(6)}
            total_ms = (time.perf_counter() - t_start) * 1000.0
            
            return PredictionResult(
                is_valid=True,
                wav_path=str(wav_p.resolve()),
                predicted_label=pred_label,
                predicted_class_id=pred_cid,
                confidence=conf,
                class_probabilities=prob_dict,
                checkpoint_path=str(self.checkpoint_path),
                checkpoint_sha256=self.checkpoint_sha256,
                preprocessing_latency_ms=prep_ms,
                inference_latency_ms=inf_ms,
                total_latency_ms=total_ms,
                error_message=None
            )
            
        except AudioValidationError as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return PredictionResult(
                is_valid=False,
                wav_path=str(wav_p),
                predicted_label="REJECTED",
                predicted_class_id=-1,
                confidence=0.0,
                class_probabilities={cl: 0.0 for cl in CLASS_NAMES_6},
                checkpoint_path=str(self.checkpoint_path),
                checkpoint_sha256=self.checkpoint_sha256,
                preprocessing_latency_ms=0.0,
                inference_latency_ms=0.0,
                total_latency_ms=total_ms,
                error_message=str(e)
            )
