"""
Production Audio Normalizer for Vietnamese KWS.
Standardizes audio to 16 kHz mono 16-bit PCM, 1.0s duration (16,000 samples).
Implements clipping detection, silence detection, peak normalization, and centered padding.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
TARGET_SAMPLES = 16000  # 1.0 second at 16 kHz
PEAK_TARGET = 0.89125   # -1.0 dBFS


def load_and_resample(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads audio file and resamples to TARGET_SR (16 kHz) mono.
    Fast path uses soundfile directly when format is WAV/FLAC.
    Fallback uses librosa (with audioread/ffmpeg backend for M4A/AAC/MP3).
    """
    import warnings
    try:
        audio, sr = sf.read(str(file_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio.astype(np.float32), TARGET_SR
    except Exception:
        # Fallback to librosa.load for m4a/aac/mp3/opus via audioread/ffmpeg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
        return audio.astype(np.float32), TARGET_SR


def check_clipping(audio: np.ndarray, threshold: float = 0.999, max_consecutive: int = 3) -> bool:
    """
    Returns True if audio exhibits clipping (>= max_consecutive samples at threshold).
    """
    is_clipped = np.abs(audio) >= threshold
    if not np.any(is_clipped):
        return False
    # Check consecutive runs
    runs = 0
    for sample in is_clipped:
        if sample:
            runs += 1
            if runs >= max_consecutive:
                return True
        else:
            runs = 0
    return False


def compute_rms_db(audio: np.ndarray) -> float:
    """
    Computes Root-Mean-Square (RMS) energy in dBFS.
    """
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    return float(20.0 * np.log10(rms))


def normalize_to_target_window(audio: np.ndarray, 
                               target_samples: int = TARGET_SAMPLES,
                               jitter_samples: int = 0) -> np.ndarray:
    """
    Centers audio into a fixed target_samples (1.0s) window.
    Applies peak normalization to -1.0 dBFS.
    """
    n = len(audio)
    if n == target_samples:
        windowed = audio.copy()
    elif n < target_samples:
        # Center with optional jitter
        pad_total = target_samples - n
        pad_left = pad_total // 2
        if jitter_samples != 0:
            shift = int(np.clip(jitter_samples, -pad_left, pad_total - pad_left))
            pad_left += shift
        pad_right = target_samples - n - pad_left
        windowed = np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0.0)
    else:
        # Extract centered 1.0s chunk
        start = (n - target_samples) // 2
        windowed = audio[start:start + target_samples]

    # Peak normalization to -1.0 dBFS
    peak = np.max(np.abs(windowed))
    if peak > 1e-6:
        windowed = (windowed / peak) * PEAK_TARGET
        
    return windowed.astype(np.float32)


def process_audio_file(input_path: Path, output_path: Path, 
                       is_speech: bool = True) -> Dict[str, Any]:
    """
    Validates, resamples, windows, and exports a standardized 1.0s 16kHz mono WAV file.
    Returns status dict with metrics and exclusion reasons if any.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio, sr = load_and_resample(input_path)
    except Exception as e:
        return {
            "status": "REJECTED",
            "reason": f"Corrupt file / read error: {str(e)}",
            "output_path": None
        }

    raw_duration = len(audio) / sr
    is_clipped = check_clipping(audio)
    rms_db = compute_rms_db(audio)

    if is_clipped:
        return {
            "status": "REJECTED",
            "reason": "Digital clipping detected (>3 consecutive saturated samples)",
            "output_path": None,
            "rms_db": rms_db,
            "raw_duration": raw_duration
        }

    if is_speech and rms_db < -50.0:
        return {
            "status": "REJECTED",
            "reason": f"Abnormal silence for speech (RMS={rms_db:.1f} dBFS < -50 dBFS)",
            "output_path": None,
            "rms_db": rms_db,
            "raw_duration": raw_duration
        }

    # Standardize to 1.0s window
    standard_audio = normalize_to_target_window(audio, target_samples=TARGET_SAMPLES)
    
    # Save as 16-bit PCM WAV
    sf.write(str(output_path), standard_audio, TARGET_SR, subtype="PCM_16")

    return {
        "status": "APPROVED",
        "reason": "",
        "output_path": str(output_path),
        "raw_duration": raw_duration,
        "rms_db": rms_db,
        "peak": float(np.max(np.abs(standard_audio)))
    }
