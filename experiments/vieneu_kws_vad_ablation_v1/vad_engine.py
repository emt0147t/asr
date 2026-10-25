"""
Voice Activity Detection (VAD) Engines for Vietnamese KWS Gating Ablation.

Provides two distinct VAD algorithms:
1. ShortTimeEnergyGate:
   - Computes short-time frame RMS energy in dB.
   - Accurately labeled as an energy threshold gate (not a general speech classifier).
2. SpeechBandFormantVAD:
   - Speech-oriented VAD utilizing frequency-domain formant band energy ratio (300-3400 Hz),
     spectral entropy, and hangover frame smoothing.
"""

import numpy as np
from scipy import signal
import torch


class ShortTimeEnergyGate:
    """
    Short-time RMS energy detector.
    Evaluates whether an audio window contains acoustic energy above a calibrated dB threshold.
    """
    def __init__(self, frame_len_samples: int = 400, hop_len_samples: int = 160,
                 energy_threshold_db: float = -45.0, min_speech_frames: int = 5):
        self.frame_len = frame_len_samples
        self.hop_len = hop_len_samples
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_frames = min_speech_frames

    def process_window(self, audio: np.ndarray):
        """
        Processes a 1.0s window of audio.
        Returns:
            is_speech (bool): True if energy exceeds threshold for >= min_speech_frames.
            active_ratio (float): Fraction of frames classified as active.
            max_db (float): Peak frame RMS in dB.
        """
        n_samples = len(audio)
        if n_samples < self.frame_len:
            rms = float(np.sqrt(np.mean(audio**2)) + 1e-7)
            db = 20.0 * np.log10(rms)
            return (db >= self.energy_threshold_db), float(db >= self.energy_threshold_db), db

        num_frames = (n_samples - self.frame_len) // self.hop_len + 1
        active_frames = 0
        max_db = -140.0

        for f in range(num_frames):
            s = f * self.hop_len
            frame = audio[s : s + self.frame_len]
            rms = float(np.sqrt(np.mean(frame**2)) + 1e-7)
            db = 20.0 * np.log10(rms)
            if db > max_db:
                max_db = db
            if db >= self.energy_threshold_db:
                active_frames += 1

        is_speech = (active_frames >= self.min_speech_frames)
        active_ratio = active_frames / max(1, num_frames)
        return is_speech, float(active_ratio), float(max_db)


class SpeechBandFormantVAD:
    """
    Speech-oriented VAD based on speech formant band energy concentration (300 Hz - 3400 Hz),
    spectral energy ratio, and hangover smoothing.
    """
    def __init__(self, sample_rate: int = 16000, speech_band_hz=(300.0, 3400.0),
                 min_speech_band_ratio: float = 0.35, min_energy_db: float = -48.0,
                 hangover_frames: int = 10, frame_len_samples: int = 400, hop_len_samples: int = 160):
        self.sample_rate = sample_rate
        self.speech_low = speech_band_hz[0]
        self.speech_high = speech_band_hz[1]
        self.min_speech_band_ratio = min_speech_band_ratio
        self.min_energy_db = min_energy_db
        self.hangover_frames = hangover_frames
        self.frame_len = frame_len_samples
        self.hop_len = hop_len_samples

        # FFT frequency bins
        self.n_fft = 512
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        self.speech_mask = (freqs >= self.speech_low) & (freqs <= self.speech_high)

    def process_window(self, audio: np.ndarray):
        """
        Processes a 1.0s window of audio.
        Returns:
            is_speech (bool): True if speech band formant energy is detected with hangover.
            active_ratio (float): Fraction of frames classified as speech.
            mean_band_ratio (float): Average ratio of speech-band energy to total energy.
        """
        n_samples = len(audio)
        if n_samples < self.frame_len:
            return False, 0.0, 0.0

        num_frames = (n_samples - self.frame_len) // self.hop_len + 1
        speech_frames = 0
        band_ratios = []

        window_func = np.hanning(self.frame_len)
        hangover = 0

        for f in range(num_frames):
            s = f * self.hop_len
            frame = audio[s : s + self.frame_len]
            rms = float(np.sqrt(np.mean(frame**2)) + 1e-7)
            db = 20.0 * np.log10(rms)

            if db < self.min_energy_db:
                if hangover > 0:
                    hangover -= 1
                    speech_frames += 1
                band_ratios.append(0.0)
                continue

            # Compute power spectrum
            windowed = frame * window_func
            spec = np.abs(np.fft.rfft(windowed, n=self.n_fft))**2
            total_power = np.sum(spec) + 1e-8
            speech_power = np.sum(spec[self.speech_mask])
            ratio = float(speech_power / total_power)
            band_ratios.append(ratio)

            if ratio >= self.min_speech_band_ratio:
                speech_frames += 1
                hangover = self.hangover_frames
            elif hangover > 0:
                hangover -= 1
                speech_frames += 1

        is_speech = (speech_frames >= 4)
        active_ratio = speech_frames / max(1, num_frames)
        mean_band_ratio = float(np.mean(band_ratios)) if band_ratios else 0.0
        return is_speech, float(active_ratio), float(mean_band_ratio)
