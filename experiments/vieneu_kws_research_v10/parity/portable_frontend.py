"""
Portable, Dependency-Free Reference Audio Frontend for Vietnamese KWS (variant_w10).
Phase: 92 (Cross-Platform Audio Preprocessing Parity)
Workspace: D:\trainidentificationaudio

Provides bit-accurate, pure-NumPy reproduction of Torchaudio LogMelFrontend:
- Sample Rate: 16,000 Hz
- FFT Size (N_fft): 400 (25.0 ms)
- Window Length: 400 (Periodic Hann)
- Hop Length: 160 (10.0 ms)
- Mel Bands: 40 (Slaney scale, Slaney triangular area normalization)
- Frequency Range: 20.0 Hz to 8,000.0 Hz
- STFT Center / Pad: Reflect padding by 200 samples on both ends (101 frames)
- Log Compression: Natural log ln(mel_power + 1e-6)
- Output Shape: [1, 1, 40, 101] float32
"""

import math
from typing import Tuple, Optional
import numpy as np


def hz_to_mel_slaney(hz: float | np.ndarray) -> float | np.ndarray:
    """Converts frequency in Hz to Slaney Mel scale."""
    hz = np.asarray(hz, dtype=np.float64)
    min_log_hz = 1000.0
    min_log_mel = 1000.0 * 3.0 / 200.0
    log_step = 27.0 / np.log(6.4)

    return np.where(
        hz < min_log_hz,
        hz * 3.0 / 200.0,
        min_log_mel + np.log(hz / min_log_hz) * log_step
    )


def mel_to_hz_slaney(mel: float | np.ndarray) -> float | np.ndarray:
    """Converts Slaney Mel scale back to frequency in Hz."""
    mel = np.asarray(mel, dtype=np.float64)
    min_log_mel = 1000.0 * 3.0 / 200.0
    log_step = 27.0 / np.log(6.4)

    return np.where(
        mel < min_log_mel,
        mel * 200.0 / 3.0,
        1000.0 * np.exp((mel - min_log_mel) / log_step)
    )


def create_slaney_mel_filterbank(
    n_freqs: int = 201,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    n_mels: int = 40,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Constructs 40x201 triangular Slaney filterbank matrix with area normalization.
    Returns:
        filterbank: Array of shape [40, 201] (float32).
    """
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)

    mel_min = hz_to_mel_slaney(f_min)
    mel_max = hz_to_mel_slaney(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = mel_to_hz_slaney(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]

        enorm = 2.0 / (right - left)

        up_idx = np.logical_and(fft_freqs >= left, fft_freqs <= center)
        if center > left:
            fb[m, up_idx] = (fft_freqs[up_idx] - left) / (center - left) * enorm

        down_idx = np.logical_and(fft_freqs >= center, fft_freqs <= right)
        if right > center:
            fb[m, down_idx] = (right - fft_freqs[down_idx]) / (right - center) * enorm

    return fb


class PortableLogMelFrontend:
    """
    Pure NumPy implementation of Torchaudio LogMelFrontend.
    Zero external neural/audio dependencies.
    """
    SAMPLE_RATE = 16000
    N_FFT = 400
    WIN_LENGTH = 400
    HOP_LENGTH = 160
    N_MELS = 40
    N_FREQS = 201  # N_FFT // 2 + 1
    N_FRAMES = 101  # (16000 + 400 - 400) // 160 + 1
    EPS = 1e-6
    PAD_REFLECT = 200

    def __init__(self, filterbank_matrix: Optional[np.ndarray] = None):
        n = np.arange(self.WIN_LENGTH, dtype=np.float64)
        self.window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / float(self.WIN_LENGTH))).astype(np.float32)

        if filterbank_matrix is not None:
            assert filterbank_matrix.shape == (self.N_MELS, self.N_FREQS)
            self.filterbank = filterbank_matrix.astype(np.float32)
        else:
            self.filterbank = create_slaney_mel_filterbank(
                n_freqs=self.N_FREQS,
                f_min=20.0,
                f_max=8000.0,
                n_mels=self.N_MELS,
                sample_rate=self.SAMPLE_RATE
            )

    def wav_to_logmel(self, wav: np.ndarray) -> np.ndarray:
        """
        Pure STFT + Mel feature extraction without amplitude modification.
        Input wav must be shape (16000,) float32.
        """
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        assert wav.ndim == 1, f"Expected 1D audio buffer, got shape {wav.shape}"
        assert len(wav) == self.SAMPLE_RATE, f"Expected exactly {self.SAMPLE_RATE} samples, got {len(wav)}"

        # 1. Symmetric reflect padding by 200 samples on each side
        padded = np.pad(wav, (self.PAD_REFLECT, self.PAD_REFLECT), mode="reflect").astype(np.float32)

        # 2. Framing & 400-point FFT
        stft_power = np.zeros((self.N_FREQS, self.N_FRAMES), dtype=np.float32)
        for t in range(self.N_FRAMES):
            start = t * self.HOP_LENGTH
            frame = padded[start:start + self.WIN_LENGTH] * self.window
            rfft_out = np.fft.rfft(frame, n=self.N_FFT)
            stft_power[:, t] = np.abs(rfft_out) ** 2

        # 3. Mel Filterbank Dot Product
        mel_power = np.dot(self.filterbank, stft_power)  # [40, 101]

        # 4. Natural Logarithm with epsilon
        log_mel = np.log(mel_power + self.EPS).astype(np.float32)

        # 5. Output shape [1, 1, 40, 101]
        return np.expand_dims(log_mel, axis=(0, 1))

    def extract_features(self, wav: np.ndarray, apply_peak_clamp: bool = True) -> np.ndarray:
        """
        Full feature extraction pipeline including optional peak clamping.
        """
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        if apply_peak_clamp:
            max_val = float(np.max(np.abs(wav)))
            if max_val > 1.0:
                wav = wav / max_val
        return self.wav_to_logmel(wav)
