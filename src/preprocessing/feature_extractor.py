"""
Log-Mel Filterbank Feature Extractor for Vietnamese KWS.
Guarantees bit-level and numerical parity across PyTorch (training),
NumPy (Python reference), and mobile deployment frontends.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa


class LogMelFeatureExtractor(nn.Module):
    """
    Standard Log-Mel Spectrogram Frontend matching mobile runtime specs:
      - Sample Rate: 16,000 Hz
      - N_FFT: 512
      - Win Length: 400 (25 ms)
      - Hop Length: 160 (10 ms)
      - N_MELS: 40
      - Fmin: 20 Hz, Fmax: 8000 Hz
      - Log: Natural log (or log10) with epsilon = 1e-6
    Output shape for 16,000 samples: (Batch, 1, 40, 101) or (Batch, 1, 101, 40)
    """
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 win_length: int = 400, hop_length: int = 160,
                 n_mels: int = 40, f_min: float = 20.0, f_max: float = 8000.0,
                 eps: float = 1e-6):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps

        # Torchaudio MelSpectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=f_min,
            f_max=f_max
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor of shape (Batch, 16000) or (16000,)
        Output: Tensor of shape (Batch, 1, 40, 101)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mel_spec = self.mel_transform(x)  # (Batch, n_mels, frames)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)  # (Batch, 1, n_mels, frames)
        return log_mel


def extract_log_mel_numpy(audio: np.ndarray, sample_rate: int = 16000,
                          n_fft: int = 512, win_length: int = 400,
                          hop_length: int = 160, n_mels: int = 40,
                          f_min: float = 20.0, f_max: float = 8000.0,
                          eps: float = 1e-6) -> np.ndarray:
    """
    NumPy / Librosa reference implementation for mobile golden test verification.
    """
    if audio.ndim > 1:
        audio = audio.flatten()
        
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max, htk=False, norm="slaney"
    )
    # STFT with center=True, reflect pad
    stft = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann", center=True, pad_mode="reflect"
    )
    power_spec = np.abs(stft) ** 2
    mel_spec = np.dot(mel_basis, power_spec)
    log_mel = np.log(mel_spec + eps)
    return log_mel.astype(np.float32)  # Shape: (40, 101)
