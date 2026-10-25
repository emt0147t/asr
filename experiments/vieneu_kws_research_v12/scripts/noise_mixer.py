"""
Deterministic Noise Mixing Utility with Controlled SNR & Clipping Safeguards.
Dataset v12 Pipeline Component.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, List
from scipy import signal

TARGET_SR = 16000
SAMPLE_LEN = 16000  # 1.0s window


class NoiseAssetManager:
    """Manages noise assets from data/raw/noise/extracted/noise/mic-noise."""
    def __init__(self, noise_dir: Path):
        self.noise_dir = Path(noise_dir)
        self.noise_files = sorted(list(self.noise_dir.glob("*.mp3")) + list(self.noise_dir.glob("*.wav")))
        assert len(self.noise_files) > 0, f"No noise files found in {self.noise_dir}"
        self._noise_cache = {}

    def get_track_count(self) -> int:
        return len(self.noise_files)

    def get_track_path(self, track_index: int) -> Path:
        idx = track_index % len(self.noise_files)
        return self.noise_files[idx]

    def load_noise_track_16k(self, track_index: int) -> np.ndarray:
        """Loads and caches noise track resampled to 16 kHz mono float32."""
        idx = track_index % len(self.noise_files)
        if idx in self._noise_cache:
            return self._noise_cache[idx]

        fpath = self.noise_files[idx]
        audio, sr = sf.read(str(fpath), dtype="float32")
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16 kHz if necessary
        if sr != TARGET_SR:
            num_target = int(round(len(audio) * float(TARGET_SR) / sr))
            audio = signal.resample(audio, num_target).astype(np.float32)

        # Remove DC offset
        audio = audio - np.mean(audio)
        self._noise_cache[idx] = audio.astype(np.float32)
        return self._noise_cache[idx]

    def extract_noise_slice(self, track_index: int, seed: int, duration_samples: int = SAMPLE_LEN) -> np.ndarray:
        """Extracts a deterministic 1.0s slice of noise using seed."""
        track = self.load_noise_track_16k(track_index)
        n_samples = len(track)
        rng = np.random.RandomState(seed)

        if n_samples <= duration_samples:
            # Loop if shorter than 1s
            reps = int(np.ceil(duration_samples / max(1, n_samples)))
            tiled = np.tile(track, reps)
            return tiled[:duration_samples]

        max_start = n_samples - duration_samples
        start = rng.randint(0, max_start + 1)
        return track[start : start + duration_samples].copy()


def mix_noise_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
    max_peak_limit: float = 0.98,
    safe_headroom: float = 0.95
) -> Tuple[np.ndarray, float, float, float]:
    """
    Acoustically mixes noise into speech at a specified nominal SNR (dB).
    
    Returns:
        mixed_audio (np.ndarray): 16,000 float32 samples.
        observed_snr_db (float): Calculated SNR.
        peak_amplitude (float): Peak absolute amplitude.
        speech_rms_db (float): RMS energy in dBFS.
    """
    assert len(speech) == len(noise), f"Shape mismatch: {len(speech)} vs {len(noise)}"
    
    ps = float(np.mean(speech ** 2))
    pn = float(np.mean(noise ** 2))
    
    if pn < 1e-12 or ps < 1e-12:
        peak = float(np.max(np.abs(speech)))
        rms = float(np.sqrt(ps)) + 1e-7
        return speech.copy().astype(np.float32), float("inf"), peak, 20.0 * np.log10(rms)

    # Calculate scaling factor alpha
    alpha = np.sqrt(ps / (pn * (10.0 ** (target_snr_db / 10.0))))
    scaled_noise = alpha * noise
    mixed = speech + scaled_noise

    # Clipping safeguard: scale down if peak exceeds limit (never amplify quiet audio)
    peak = float(np.max(np.abs(mixed)))
    if peak > max_peak_limit:
        scaling_ratio = safe_headroom / peak
        mixed = mixed * scaling_ratio
        peak = float(np.max(np.abs(mixed)))

    # Compute observed SNR
    ps_actual = float(np.mean(speech ** 2))
    pn_actual = float(np.mean(scaled_noise ** 2))
    observed_snr = 10.0 * np.log10(ps_actual / (pn_actual + 1e-12))
    rms_val = float(np.sqrt(np.mean(mixed ** 2))) + 1e-7
    rms_db = 20.0 * np.log10(rms_val)

    return mixed.astype(np.float32), round(float(observed_snr), 2), round(peak, 4), round(rms_db, 2)
