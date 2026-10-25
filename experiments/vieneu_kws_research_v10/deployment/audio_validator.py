"""
Audio validation and loading module for Vietnamese KWS Phase 91 Deployment.
Enforces input audio contract:
- WAV file existence and header integrity
- Detection of empty, truncated, or corrupted files
- Support for both Strict validation mode (exact 16kHz, mono, 1.0s)
  and Permissive preprocessing mode (downmix, resample, zero-pad/truncate).
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


# ============================================================================
# Audio Validation Exceptions
# ============================================================================

class AudioValidationError(Exception):
    """Base exception for audio input errors."""
    exit_code = 1


class AudioFileNotFoundError(AudioValidationError):
    """Raised when WAV file does not exist on disk."""
    exit_code = 2


class EmptyAudioError(AudioValidationError):
    """Raised when file is 0 bytes or contains 0 samples."""
    exit_code = 3


class CorruptedAudioError(AudioValidationError):
    """Raised when audio header cannot be parsed or data is malformed."""
    exit_code = 4


class InvalidSampleRateError(AudioValidationError):
    """Raised in strict mode when sample rate differs from 16,000 Hz."""
    exit_code = 5


class InvalidChannelCountError(AudioValidationError):
    """Raised in strict mode when audio has multiple channels (not mono)."""
    exit_code = 6


class InvalidDurationError(AudioValidationError):
    """Raised in strict mode when sample count differs from 16,000 samples (1.0s)."""
    exit_code = 7


class CheckpointNotFoundError(Exception):
    """Raised when model checkpoint cannot be located."""
    exit_code = 8


# ============================================================================
# Audio Validator
# ============================================================================

class AudioValidator:
    REQUIRED_SAMPLE_RATE = 16000
    REQUIRED_CHANNELS = 1
    REQUIRED_SAMPLES = 16000

    @classmethod
    def inspect_file(cls, path: Path) -> sf.SoundFile:
        """Inspects file existence and header without loading full buffer."""
        if not path.exists():
            raise AudioFileNotFoundError(f"Audio file does not exist: {path}")
        if path.stat().st_size == 0:
            raise EmptyAudioError(f"Audio file is 0 bytes: {path}")
        try:
            info = sf.info(str(path))
        except Exception as e:
            raise CorruptedAudioError(f"Failed to read WAV header from {path.name}: {e}") from e

        if info.frames == 0:
            raise EmptyAudioError(f"Audio file contains 0 audio frames: {path}")
        return info

    @classmethod
    def load_and_validate(
        cls,
        wav_path: str | Path,
        strict: bool = False
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Loads and validates audio from wav_path.
        
        Args:
            wav_path: Path to target WAV file.
            strict: If True, raises exceptions for any deviation from 16kHz, mono, 1.0s.
                   If False, automatically downmixes, resamples, and pads/truncates to 16,000 samples.
                   
        Returns:
            wav_tensor: Tensor of shape [1, 16000], float32.
            orig_sr: Original sample rate in Hz.
            metadata: Dict of original file metadata.
        """
        path = Path(wav_path)
        info = cls.inspect_file(path)

        orig_sr = info.samplerate
        orig_channels = info.channels
        orig_frames = info.frames
        orig_duration_s = orig_frames / orig_sr

        metadata = {
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "original_sample_rate": orig_sr,
            "original_channels": orig_channels,
            "original_frames": orig_frames,
            "original_duration_seconds": orig_duration_s,
            "format": info.format,
            "subtype": info.subtype,
            "strict_validation": strict
        }

        if strict:
            if orig_sr != cls.REQUIRED_SAMPLE_RATE:
                raise InvalidSampleRateError(
                    f"Invalid sample rate: {orig_sr} Hz. Strict mode requires exactly {cls.REQUIRED_SAMPLE_RATE} Hz."
                )
            if orig_channels != cls.REQUIRED_CHANNELS:
                raise InvalidChannelCountError(
                    f"Invalid channel count: {orig_channels} channels. Strict mode requires mono (1 channel)."
                )
            if orig_frames != cls.REQUIRED_SAMPLES:
                raise InvalidDurationError(
                    f"Invalid duration: {orig_frames} samples ({orig_duration_s:.3f}s). "
                    f"Strict mode requires exactly {cls.REQUIRED_SAMPLES} samples (1.000s)."
                )

        # Read audio samples
        try:
            data, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            raise CorruptedAudioError(f"Failed to read samples from {path.name}: {e}") from e

        wav = torch.from_numpy(data)

        # Handle channel dimensions
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, N]
        elif wav.dim() == 2:
            # Soundfile returns [samples, channels]
            wav = wav.t()  # [channels, samples]
            if wav.size(0) > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)  # mono average [1, samples]

        # Resample to 16000 Hz if needed
        if sr != cls.REQUIRED_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cls.REQUIRED_SAMPLE_RATE)
            wav = resampler(wav)

        # Pad right or truncate to 16000 samples
        curr_len = wav.size(-1)
        if curr_len < cls.REQUIRED_SAMPLES:
            pad_right = cls.REQUIRED_SAMPLES - curr_len
            wav = F.pad(wav, (0, pad_right), mode="constant", value=0.0)
        elif curr_len > cls.REQUIRED_SAMPLES:
            wav = wav[:, :cls.REQUIRED_SAMPLES]

        # Peak normalization clamped only if amplitude > 1.0 (never amplify quiet noise floor)
        max_val = torch.max(torch.abs(wav))
        if max_val > 1.0:
            wav = wav / max_val

        return wav, orig_sr, metadata
