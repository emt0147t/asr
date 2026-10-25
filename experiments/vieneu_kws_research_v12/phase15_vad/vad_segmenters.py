"""
VAD Segmenters for Vietnamese KWS Streaming Investigation (Phase 15).

Implements three distinct segmenter candidates:
1. BaselineFixedSilenceVAD:
   - Canonical reproduction of Phase 13.4 UtteranceSegmenter.
   - Fixed silence interval (300 ms), single energy threshold (-48 dB with dynamic floor).
2. DualThresholdAdaptiveVAD:
   - Asymmetric noise-floor tracking.
   - Dual-threshold hysteresis (Onset SNR +10 dB, Offset SNR +4 dB).
   - Adaptive hangover duration (160ms in clean/high-SNR to 220ms in low-SNR).
3. SpeechBandRatioVAD:
   - Exploratory speech-band energy-ratio heuristic.
   - Formant-band energy concentration (300-3400 Hz) via short-time FFT.
   - Dual energy + spectral-ratio gating.

All classes provide identical interfaces and serializable JSON configurations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BaseSegmenter:
    """Base class for utterance segmenters."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def reset(self):
        raise NotImplementedError

    def process_chunk(self, chunk: np.ndarray, chunk_time_sec: float) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError


class BaselineFixedSilenceVAD(BaseSegmenter):
    """
    Candidate A: Exact reproduction of Phase 13.4 UtteranceSegmenter.
    Fixed 300ms silence hangover.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_interval_ms: int = 300,
        preroll_ms: int = 150,
        energy_threshold_db: float = -48.0,
        min_speech_duration_ms: int = 120,
        max_segment_duration_ms: int = 1200
    ):
        super().__init__(sample_rate)
        self.silence_interval_ms = silence_interval_ms
        self.silence_threshold_samples = int((silence_interval_ms / 1000.0) * sample_rate)
        self.preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.max_segment_samples = int((max_segment_duration_ms / 1000.0) * sample_rate)

        self.preroll_buf = np.zeros(self.preroll_samples, dtype=np.float32)
        self.state = "WAITING"
        self.accumulated_chunks: List[np.ndarray] = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.noise_floor_db = energy_threshold_db
        self.segment_start_sample = 0
        self.current_sample_pos = 0

    def reset(self):
        self.state = "WAITING"
        self.accumulated_chunks = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.preroll_buf.fill(0.0)
        self.noise_floor_db = self.energy_threshold_db
        self.segment_start_sample = 0
        self.current_sample_pos = 0

    def process_chunk(self, chunk: np.ndarray, chunk_time_sec: float) -> Optional[Dict[str, Any]]:
        n_samples = len(chunk)
        if n_samples == 0:
            return None

        chunk_pos = self.current_sample_pos
        self.current_sample_pos += n_samples

        rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-7
        db = 20.0 * np.log10(rms)

        # Baseline dynamic floor adaptation
        effective_thresh = max(self.energy_threshold_db, min(self.noise_floor_db + 5.0, -32.0))
        is_speech_chunk = (db >= effective_thresh)

        if self.state == "WAITING" and not is_speech_chunk:
            self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db

        finalized_event = None

        if self.state == "WAITING":
            if n_samples >= self.preroll_samples:
                self.preroll_buf[:] = chunk[-self.preroll_samples:]
            else:
                self.preroll_buf[:-n_samples] = self.preroll_buf[n_samples:]
                self.preroll_buf[-n_samples:] = chunk

            if is_speech_chunk:
                self.state = "ACCUMULATING"
                self.accumulated_chunks = [self.preroll_buf.copy(), chunk.copy()]
                self.consecutive_silence_samples = 0
                self.active_speech_samples = n_samples
                self.segment_start_sample = max(0, chunk_pos - self.preroll_samples)

        elif self.state == "ACCUMULATING":
            self.accumulated_chunks.append(chunk.copy())

            if is_speech_chunk:
                self.consecutive_silence_samples = 0
                self.active_speech_samples += n_samples
            else:
                self.consecutive_silence_samples += n_samples

            total_samples = sum(len(c) for c in self.accumulated_chunks)
            silence_reached = (self.consecutive_silence_samples >= self.silence_threshold_samples)
            max_reached = (total_samples >= self.max_segment_samples)

            if silence_reached or max_reached:
                full_segment = np.concatenate(self.accumulated_chunks)

                if self.active_speech_samples >= self.min_speech_samples:
                    # Pad to 1.0s (16,000 samples)
                    audio_1s = np.zeros(self.sample_rate, dtype=np.float32)
                    copy_len = min(len(full_segment), self.sample_rate)
                    audio_1s[:copy_len] = full_segment[:copy_len]

                    # Segment end time: total accumulated minus trailing silence
                    trailing_silence = self.consecutive_silence_samples if silence_reached else 0
                    actual_end_sample = self.segment_start_sample + total_samples - trailing_silence
                    actual_dur_sec = (actual_end_sample - self.segment_start_sample) / self.sample_rate

                    finalized_event = {
                        "segment_audio_1s": audio_1s,
                        "raw_segment_audio": full_segment,
                        "start_sec": round(self.segment_start_sample / self.sample_rate, 4),
                        "end_sec": round(actual_end_sample / self.sample_rate, 4),
                        "duration_sec": round(actual_dur_sec, 4),
                        "active_speech_samples": self.active_speech_samples
                    }

                self.state = "WAITING"
                self.accumulated_chunks = []
                self.consecutive_silence_samples = 0
                self.active_speech_samples = 0
                self.preroll_buf.fill(0.0)

        return finalized_event

    def get_config(self) -> Dict[str, Any]:
        return {
            "algorithm": "BaselineFixedSilenceVAD",
            "sample_rate": self.sample_rate,
            "silence_interval_ms": self.silence_interval_ms,
            "preroll_ms": int((self.preroll_samples / self.sample_rate) * 1000),
            "energy_threshold_db": self.energy_threshold_db,
            "min_speech_duration_ms": int((self.min_speech_samples / self.sample_rate) * 1000),
            "max_segment_duration_ms": int((self.max_segment_samples / self.sample_rate) * 1000)
        }


class DualThresholdAdaptiveVAD(BaseSegmenter):
    """
    Candidate B: Dual-Threshold Adaptive Hangover VAD.
    - Asymmetric noise floor tracking.
    - Hysteresis thresholds (Onset vs Offset).
    - Adaptive hangover duration based on local SNR.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        onset_snr_offset_db: float = 10.0,
        offset_snr_offset_db: float = 4.0,
        min_onset_db: float = -45.0,
        max_onset_db: float = -32.0,
        min_offset_db: float = -52.0,
        max_offset_db: float = -40.0,
        hangover_clean_ms: int = 160,
        hangover_noisy_ms: int = 220,
        preroll_ms: int = 150,
        min_speech_duration_ms: int = 100,
        max_segment_duration_ms: int = 1200
    ):
        super().__init__(sample_rate)
        self.onset_snr_offset_db = onset_snr_offset_db
        self.offset_snr_offset_db = offset_snr_offset_db
        self.min_onset_db = min_onset_db
        self.max_onset_db = max_onset_db
        self.min_offset_db = min_offset_db
        self.max_offset_db = max_offset_db
        self.hangover_clean_ms = hangover_clean_ms
        self.hangover_noisy_ms = hangover_noisy_ms
        self.preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.max_segment_samples = int((max_segment_duration_ms / 1000.0) * sample_rate)

        self.preroll_buf = np.zeros(self.preroll_samples, dtype=np.float32)
        self.state = "WAITING"
        self.accumulated_chunks: List[np.ndarray] = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.noise_floor_db = -50.0
        self.segment_start_sample = 0
        self.current_sample_pos = 0

    def reset(self):
        self.state = "WAITING"
        self.accumulated_chunks = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.preroll_buf.fill(0.0)
        self.noise_floor_db = -50.0
        self.segment_start_sample = 0
        self.current_sample_pos = 0

    def process_chunk(self, chunk: np.ndarray, chunk_time_sec: float) -> Optional[Dict[str, Any]]:
        n_samples = len(chunk)
        if n_samples == 0:
            return None

        chunk_pos = self.current_sample_pos
        self.current_sample_pos += n_samples

        rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-7
        db = 20.0 * np.log10(rms)

        # Asymmetric noise tracking
        if self.state == "WAITING":
            if db < self.noise_floor_db:
                self.noise_floor_db = 0.85 * self.noise_floor_db + 0.15 * db
            else:
                self.noise_floor_db = 0.98 * self.noise_floor_db + 0.02 * db

        # Compute dual thresholds with clipping
        thresh_onset = np.clip(
            self.noise_floor_db + self.onset_snr_offset_db,
            self.min_onset_db,
            self.max_onset_db
        )
        thresh_offset = np.clip(
            self.noise_floor_db + self.offset_snr_offset_db,
            self.min_offset_db,
            self.max_offset_db
        )

        # Adaptive hangover threshold based on current SNR
        estimated_snr = db - self.noise_floor_db
        if estimated_snr >= 15.0:
            effective_hangover_ms = self.hangover_clean_ms
        elif estimated_snr <= 6.0:
            effective_hangover_ms = self.hangover_noisy_ms
        else:
            alpha = (estimated_snr - 6.0) / 9.0
            effective_hangover_ms = int(alpha * self.hangover_clean_ms + (1.0 - alpha) * self.hangover_noisy_ms)

        silence_threshold_samples = int((effective_hangover_ms / 1000.0) * self.sample_rate)

        finalized_event = None

        if self.state == "WAITING":
            if n_samples >= self.preroll_samples:
                self.preroll_buf[:] = chunk[-self.preroll_samples:]
            else:
                self.preroll_buf[:-n_samples] = self.preroll_buf[n_samples:]
                self.preroll_buf[-n_samples:] = chunk

            # Speech onset detection requires exceeding onset threshold
            if db >= thresh_onset:
                self.state = "ACCUMULATING"
                self.accumulated_chunks = [self.preroll_buf.copy(), chunk.copy()]
                self.consecutive_silence_samples = 0
                self.active_speech_samples = n_samples
                self.segment_start_sample = max(0, chunk_pos - self.preroll_samples)

        elif self.state == "ACCUMULATING":
            self.accumulated_chunks.append(chunk.copy())

            # Offset detection: falls below offset threshold
            if db >= thresh_offset:
                self.consecutive_silence_samples = 0
                self.active_speech_samples += n_samples
            else:
                self.consecutive_silence_samples += n_samples

            total_samples = sum(len(c) for c in self.accumulated_chunks)
            silence_reached = (self.consecutive_silence_samples >= silence_threshold_samples)
            max_reached = (total_samples >= self.max_segment_samples)

            if silence_reached or max_reached:
                full_segment = np.concatenate(self.accumulated_chunks)

                if self.active_speech_samples >= self.min_speech_samples:
                    audio_1s = np.zeros(self.sample_rate, dtype=np.float32)
                    copy_len = min(len(full_segment), self.sample_rate)
                    audio_1s[:copy_len] = full_segment[:copy_len]

                    trailing_silence = self.consecutive_silence_samples if silence_reached else 0
                    actual_end_sample = self.segment_start_sample + total_samples - trailing_silence
                    actual_dur_sec = (actual_end_sample - self.segment_start_sample) / self.sample_rate

                    finalized_event = {
                        "segment_audio_1s": audio_1s,
                        "raw_segment_audio": full_segment,
                        "start_sec": round(self.segment_start_sample / self.sample_rate, 4),
                        "end_sec": round(actual_end_sample / self.sample_rate, 4),
                        "duration_sec": round(actual_dur_sec, 4),
                        "active_speech_samples": self.active_speech_samples
                    }

                self.state = "WAITING"
                self.accumulated_chunks = []
                self.consecutive_silence_samples = 0
                self.active_speech_samples = 0
                self.preroll_buf.fill(0.0)

        return finalized_event

    def get_config(self) -> Dict[str, Any]:
        return {
            "algorithm": "DualThresholdAdaptiveVAD",
            "sample_rate": self.sample_rate,
            "onset_snr_offset_db": self.onset_snr_offset_db,
            "offset_snr_offset_db": self.offset_snr_offset_db,
            "min_onset_db": self.min_onset_db,
            "max_onset_db": self.max_onset_db,
            "min_offset_db": self.min_offset_db,
            "max_offset_db": self.max_offset_db,
            "hangover_clean_ms": self.hangover_clean_ms,
            "hangover_noisy_ms": self.hangover_noisy_ms,
            "preroll_ms": int((self.preroll_samples / self.sample_rate) * 1000),
            "min_speech_duration_ms": int((self.min_speech_samples / self.sample_rate) * 1000),
            "max_segment_duration_ms": int((self.max_segment_samples / self.sample_rate) * 1000)
        }


class SpeechBandRatioVAD(BaseSegmenter):
    """
    Candidate C: Exploratory Speech-Band Energy Ratio Heuristic.
    Monitors energy in speech band (300-3400 Hz) vs broadband power.
    Requires both acoustic energy and formant band ratio elevation.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        speech_band_hz: Tuple[float, float] = (300.0, 3400.0),
        min_speech_band_ratio: float = 0.32,
        energy_threshold_db: float = -46.0,
        hangover_ms: int = 160,
        preroll_ms: int = 150,
        min_speech_duration_ms: int = 100,
        max_segment_duration_ms: int = 1200
    ):
        super().__init__(sample_rate)
        self.speech_band_hz = speech_band_hz
        self.min_speech_band_ratio = min_speech_band_ratio
        self.energy_threshold_db = energy_threshold_db
        self.hangover_ms = hangover_ms
        self.hangover_samples = int((hangover_ms / 1000.0) * sample_rate)
        self.preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.max_segment_samples = int((max_segment_duration_ms / 1000.0) * sample_rate)

        # FFT parameters
        self.n_fft = 512
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        self.speech_mask = (freqs >= speech_band_hz[0]) & (freqs <= speech_band_hz[1])

        self.preroll_buf = np.zeros(self.preroll_samples, dtype=np.float32)
        self.state = "WAITING"
        self.accumulated_chunks: List[np.ndarray] = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.noise_floor_db = energy_threshold_db
        self.segment_start_sample = 0
        self.current_sample_pos = 0

    def reset(self):
        self.state = "WAITING"
        self.accumulated_chunks = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.preroll_buf.fill(0.0)
        self.noise_floor_db = self.energy_threshold_db
        self.segment_start_sample = 0
        self.current_sample_pos = 0

    def _compute_speech_band_ratio(self, chunk: np.ndarray) -> Tuple[float, float]:
        rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-7
        db = 20.0 * np.log10(rms)

        if len(chunk) < self.n_fft:
            padded = np.pad(chunk, (0, self.n_fft - len(chunk)))
        else:
            padded = chunk[:self.n_fft]

        windowed = padded * np.hanning(len(padded))
        spec = np.abs(np.fft.rfft(windowed, n=self.n_fft)) ** 2
        tot_pow = float(np.sum(spec)) + 1e-8
        speech_pow = float(np.sum(spec[self.speech_mask]))
        ratio = speech_pow / tot_pow
        return db, ratio

    def process_chunk(self, chunk: np.ndarray, chunk_time_sec: float) -> Optional[Dict[str, Any]]:
        n_samples = len(chunk)
        if n_samples == 0:
            return None

        chunk_pos = self.current_sample_pos
        self.current_sample_pos += n_samples

        db, ratio = self._compute_speech_band_ratio(chunk)

        effective_thresh = max(self.energy_threshold_db, min(self.noise_floor_db + 6.0, -32.0))
        is_speech_chunk = (db >= effective_thresh) and (ratio >= self.min_speech_band_ratio)

        if self.state == "WAITING" and not is_speech_chunk:
            self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db

        finalized_event = None

        if self.state == "WAITING":
            if n_samples >= self.preroll_samples:
                self.preroll_buf[:] = chunk[-self.preroll_samples:]
            else:
                self.preroll_buf[:-n_samples] = self.preroll_buf[n_samples:]
                self.preroll_buf[-n_samples:] = chunk

            if is_speech_chunk:
                self.state = "ACCUMULATING"
                self.accumulated_chunks = [self.preroll_buf.copy(), chunk.copy()]
                self.consecutive_silence_samples = 0
                self.active_speech_samples = n_samples
                self.segment_start_sample = max(0, chunk_pos - self.preroll_samples)

        elif self.state == "ACCUMULATING":
            self.accumulated_chunks.append(chunk.copy())

            # Continues accumulating if either energy holds or ratio remains speech-like
            if is_speech_chunk or (db >= effective_thresh - 4.0 and ratio >= 0.25):
                self.consecutive_silence_samples = 0
                self.active_speech_samples += n_samples
            else:
                self.consecutive_silence_samples += n_samples

            total_samples = sum(len(c) for c in self.accumulated_chunks)
            silence_reached = (self.consecutive_silence_samples >= self.hangover_samples)
            max_reached = (total_samples >= self.max_segment_samples)

            if silence_reached or max_reached:
                full_segment = np.concatenate(self.accumulated_chunks)

                if self.active_speech_samples >= self.min_speech_samples:
                    audio_1s = np.zeros(self.sample_rate, dtype=np.float32)
                    copy_len = min(len(full_segment), self.sample_rate)
                    audio_1s[:copy_len] = full_segment[:copy_len]

                    trailing_silence = self.consecutive_silence_samples if silence_reached else 0
                    actual_end_sample = self.segment_start_sample + total_samples - trailing_silence
                    actual_dur_sec = (actual_end_sample - self.segment_start_sample) / self.sample_rate

                    finalized_event = {
                        "segment_audio_1s": audio_1s,
                        "raw_segment_audio": full_segment,
                        "start_sec": round(self.segment_start_sample / self.sample_rate, 4),
                        "end_sec": round(actual_end_sample / self.sample_rate, 4),
                        "duration_sec": round(actual_dur_sec, 4),
                        "active_speech_samples": self.active_speech_samples
                    }

                self.state = "WAITING"
                self.accumulated_chunks = []
                self.consecutive_silence_samples = 0
                self.active_speech_samples = 0
                self.preroll_buf.fill(0.0)

        return finalized_event

    def get_config(self) -> Dict[str, Any]:
        return {
            "algorithm": "SpeechBandRatioVAD",
            "sample_rate": self.sample_rate,
            "speech_band_hz": list(self.speech_band_hz),
            "min_speech_band_ratio": self.min_speech_band_ratio,
            "energy_threshold_db": self.energy_threshold_db,
            "hangover_ms": self.hangover_ms,
            "preroll_ms": int((self.preroll_samples / self.sample_rate) * 1000),
            "min_speech_duration_ms": int((self.min_speech_samples / self.sample_rate) * 1000),
            "max_segment_duration_ms": int((self.max_segment_samples / self.sample_rate) * 1000)
        }
