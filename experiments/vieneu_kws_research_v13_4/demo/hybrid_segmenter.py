"""
Hybrid SpeechBand-Valley Segmenter and Game Mode Controller for Vietnamese KWS.
Phase 4: Hybrid Continuous KWS with False Activation Control.

Modules:
1. HybridSpeechBandValleySegmenter:
   - Fuses SpeechBandRatioVAD (formant energy concentration in 300-3400 Hz)
     with AcousticValleySegmenter (inter-syllable energy dip/valley detection).
   - Requires both acoustic energy elevation and speech-band ratio before onset.
   - Tracks energy dips (>= dip_threshold_db) during active speech and splits
     upon onset rebound confirmed by speech-band concentration.
   - Rejects non-speech transients, clicks, and broadband noise bursts.

2. GameModeSessionController:
   - Manages intentional command listening sessions for gaming and interactive control.
   - States: "IDLE" (gate closed: mic muted/suppressed), "ACTIVE_GAMEPLAY" (gate open).
   - In IDLE state: strictly suppresses open-mic conversational chatter (FAR = 0.0).
   - In ACTIVE_GAMEPLAY state: fully arms the high-performance valley/hybrid segmenter
     to detect rapid contiguous command bursts (e.g. 'LÊNLENPHAI') with minimal latency.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class HybridSpeechBandValleySegmenter:
    """
    Hybrid Segmenter combining Speech-Band Spectral Ratio Gating
    with Short-Time Acoustic Energy Valley Splitting.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        speech_band_hz: Tuple[float, float] = (100.0, 3400.0),
        min_speech_band_ratio: float = 0.32,
        energy_threshold_db: float = -46.0,
        silence_interval_ms: int = 140,
        preroll_ms: int = 120,
        min_speech_duration_ms: int = 100,
        min_command_dur_before_split_ms: int = 160,
        dip_threshold_db: float = 12.0,
        rebound_threshold_db: float = 6.0,
        max_segment_duration_ms: int = 1200
    ):
        self.sample_rate = sample_rate
        self.speech_band_hz = speech_band_hz
        self.min_speech_band_ratio = min_speech_band_ratio
        self.energy_threshold_db = energy_threshold_db
        self.silence_interval_ms = silence_interval_ms
        self.preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.min_command_samples = int((min_command_dur_before_split_ms / 1000.0) * sample_rate)
        self.dip_threshold_db = dip_threshold_db
        self.rebound_threshold_db = rebound_threshold_db
        self.max_segment_samples = int((max_segment_duration_ms / 1000.0) * sample_rate)

        # FFT Filterbank for Speech-Band Power Estimation
        self.n_fft = 512
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        self.speech_mask = (freqs >= speech_band_hz[0]) & (freqs <= speech_band_hz[1])

        # Buffers and State
        self.preroll_buf = np.zeros(self.preroll_samples, dtype=np.float32)
        self.state = "WAITING"  # "WAITING" or "ACCUMULATING"
        self.accumulated_chunks: List[np.ndarray] = []
        self.chunk_dbs: List[float] = []
        self.chunk_ratios: List[float] = []
        self.chunk_is_speech: List[bool] = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.peak_db = -120.0
        self.valley_db = 0.0
        self.in_valley = False
        self.valley_chunk_idx = -1
        self.noise_floor_db = energy_threshold_db
        self.current_db = -120.0
        self.current_ratio = 0.0
        self.last_segment_info: Optional[Dict[str, Any]] = None

    def reset(self):
        self.state = "WAITING"
        self.accumulated_chunks = []
        self.chunk_dbs = []
        self.chunk_ratios = []
        self.chunk_is_speech = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.preroll_buf.fill(0.0)
        self.peak_db = -120.0
        self.valley_db = 0.0
        self.in_valley = False
        self.valley_chunk_idx = -1
        self.last_segment_info = None

    def _compute_chunk_features(self, chunk: np.ndarray) -> Tuple[float, float]:
        """Calculates RMS energy in dB and spectral energy concentration in speech band."""
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

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Processes a single 16 kHz audio chunk.
        Returns (audio_1s, duration_ms) if a valid utterance segment is finalized.
        """
        n_samples = len(chunk)
        if n_samples == 0:
            return None

        db, ratio = self._compute_chunk_features(chunk)
        self.current_db = db
        self.current_ratio = ratio

        effective_thresh = max(self.energy_threshold_db, min(self.noise_floor_db + 5.0, -32.0))
        is_speech = (db >= effective_thresh) and (ratio >= self.min_speech_band_ratio)

        if self.state == "WAITING" and not is_speech:
            self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db

        finalized_segment = None

        if self.state == "WAITING":
            if is_speech:
                # Speech onset detected:
                # Capture preroll buffer containing audio strictly PRECEDING this onset chunk (zero duplication)
                preroll_audio = self.preroll_buf.copy()
                self.state = "ACCUMULATING"
                self.accumulated_chunks = [preroll_audio, chunk.copy()]
                self.chunk_dbs = [self.energy_threshold_db, db]
                self.chunk_ratios = [self.min_speech_band_ratio, ratio]
                self.chunk_is_speech = [False, True]  # Preroll chunk is silence; chunk is active speech
                self.consecutive_silence_samples = 0
                self.active_speech_samples = n_samples
                self.peak_db = db
                self.in_valley = False
                self.valley_chunk_idx = -1
            else:
                # Still waiting: advance circular preroll buffer with current non-speech chunk
                if n_samples >= self.preroll_samples:
                    self.preroll_buf[:] = chunk[-self.preroll_samples:]
                else:
                    self.preroll_buf[:-n_samples] = self.preroll_buf[n_samples:]
                    self.preroll_buf[-n_samples:] = chunk

        elif self.state == "ACCUMULATING":
            self.accumulated_chunks.append(chunk.copy())
            self.chunk_dbs.append(db)
            self.chunk_ratios.append(ratio)
            self.chunk_is_speech.append(is_speech)

            if is_speech:
                self.consecutive_silence_samples = 0
                self.active_speech_samples += n_samples
            else:
                self.consecutive_silence_samples += n_samples

            total_samples = sum(len(c) for c in self.accumulated_chunks)
            silence_samples_thresh = int((self.silence_interval_ms / 1000.0) * self.sample_rate)

            # Check 1: Normal silence offset termination
            if self.consecutive_silence_samples >= silence_samples_thresh:
                full_seg = np.concatenate(self.accumulated_chunks)
                actual_dur_ms = (len(full_seg) / self.sample_rate) * 1000.0
                if len(full_seg) < 16000:
                    audio_1s = np.pad(full_seg, (0, 16000 - len(full_seg)), mode="constant", constant_values=0.0)
                else:
                    audio_1s = full_seg[:16000]

                # Validate minimum active speech duration
                if self.active_speech_samples >= self.min_speech_samples:
                    finalized_segment = (audio_1s.astype(np.float32), actual_dur_ms)
                    info = {
                        "duration_sec": actual_dur_ms / 1000.0,
                        "split_reason": "silence_offset",
                        "active_samples": self.active_speech_samples,
                        "mean_ratio": float(np.mean(self.chunk_ratios)) if self.chunk_ratios else 0.0
                    }
                    self.reset()
                    self.last_segment_info = info
                    return finalized_segment

                self.reset()
                return finalized_segment

            # Check 2: Acoustic valley splitting during continuous speech
            if total_samples >= self.min_command_samples:
                if db > self.peak_db:
                    self.peak_db = db
                    self.in_valley = False
                elif not self.in_valley:
                    # Check for energy dip below vocalic peak
                    if (self.peak_db - db) >= self.dip_threshold_db and db < -22.0:
                        self.in_valley = True
                        self.valley_db = db
                        self.valley_chunk_idx = len(self.accumulated_chunks) - 1
                elif self.in_valley:
                    if db < self.valley_db:
                        self.valley_db = db
                        self.valley_chunk_idx = len(self.accumulated_chunks) - 1
                    # Rebound check: energy rebounds AND has speech-band spectral confirmation
                    elif (db - self.valley_db) >= self.rebound_threshold_db and is_speech:
                        split_idx = max(2, self.valley_chunk_idx)
                        if 2 <= split_idx < len(self.accumulated_chunks):
                            seg1_chunks = self.accumulated_chunks[:split_idx]
                            seg2_chunks = self.accumulated_chunks[split_idx:]

                            # Strict active speech accounting for seg1
                            seg1_active_speech = sum(
                                len(self.accumulated_chunks[j])
                                for j in range(split_idx)
                                if self.chunk_is_speech[j]
                            )

                            # Validate minimum active speech duration before finalizing seg1
                            if seg1_active_speech >= self.min_speech_samples:
                                full_seg = np.concatenate(seg1_chunks)
                                actual_dur_ms = (len(full_seg) / self.sample_rate) * 1000.0
                                if len(full_seg) < 16000:
                                    audio_1s = np.pad(full_seg, (0, 16000 - len(full_seg)), mode="constant", constant_values=0.0)
                                else:
                                    audio_1s = full_seg[:16000]

                                finalized_segment = (audio_1s.astype(np.float32), actual_dur_ms)
                                self.last_segment_info = {
                                    "duration_sec": actual_dur_ms / 1000.0,
                                    "split_reason": "acoustic_valley_hybrid",
                                    "valley_db": self.valley_db,
                                    "rebound_db": db,
                                    "rebound_ratio": ratio,
                                    "active_samples": seg1_active_speech
                                }

                                # Strict active speech accounting for re-armed seg2
                                seg2_active_speech = sum(
                                    len(self.accumulated_chunks[j])
                                    for j in range(split_idx, len(self.accumulated_chunks))
                                    if self.chunk_is_speech[j]
                                )

                                # Seamless handoff: re-arm with remaining chunks for next word
                                self.accumulated_chunks = seg2_chunks
                                self.chunk_dbs = self.chunk_dbs[split_idx:]
                                self.chunk_ratios = self.chunk_ratios[split_idx:]
                                self.chunk_is_speech = self.chunk_is_speech[split_idx:]
                                self.consecutive_silence_samples = 0 if self.chunk_is_speech[-1] else len(seg2_chunks[-1])
                                self.active_speech_samples = seg2_active_speech
                                self.peak_db = db
                                self.in_valley = False
                                self.valley_chunk_idx = -1
                                return finalized_segment

            # Check 3: Maximum segment duration safety cap
            if total_samples >= self.max_segment_samples:
                full_seg = np.concatenate(self.accumulated_chunks)
                actual_dur_ms = (len(full_seg) / self.sample_rate) * 1000.0
                if len(full_seg) < 16000:
                    audio_1s = np.pad(full_seg, (0, 16000 - len(full_seg)), mode="constant", constant_values=0.0)
                else:
                    audio_1s = full_seg[:16000]

                if self.active_speech_samples >= self.min_speech_samples:
                    finalized_segment = (audio_1s.astype(np.float32), actual_dur_ms)
                    info = {
                        "duration_sec": actual_dur_ms / 1000.0,
                        "split_reason": "max_duration_cap",
                        "active_samples": self.active_speech_samples
                    }
                    self.reset()
                    self.last_segment_info = info
                    return finalized_segment

                self.reset()
                return finalized_segment

        return finalized_segment

    def flush(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Finalizes any pending utterance at stream termination.
        Returns (audio_1s, duration_ms) if active speech >= min_speech_samples, else None.
        """
        if self.state == "ACCUMULATING" and len(self.accumulated_chunks) > 0:
            if self.active_speech_samples >= self.min_speech_samples:
                full_seg = np.concatenate(self.accumulated_chunks)
                actual_dur_ms = (len(full_seg) / self.sample_rate) * 1000.0
                if len(full_seg) < 16000:
                    audio_1s = np.pad(full_seg, (0, 16000 - len(full_seg)), mode="constant", constant_values=0.0)
                else:
                    audio_1s = full_seg[:16000]
                finalized = (audio_1s.astype(np.float32), actual_dur_ms)
                info = {
                    "duration_sec": actual_dur_ms / 1000.0,
                    "split_reason": "stream_flush",
                    "active_samples": self.active_speech_samples
                }
                self.reset()
                self.last_segment_info = info
                return finalized
        self.reset()
        return None


class GameModeSessionController:
    """
    Session-Gated Controller for Gaming / Interactive Voice Command Control.
    Implements Intentional Command Listening Sessions:
    - IDLE / Gate Closed: Audio is ignored, guaranteeing zero false activations during chatter.
    - ACTIVE / Gate Open: High-agility segmenter is armed to detect rapid contiguous bursts.
    """
    def __init__(
        self,
        base_segmenter: Any,
        session_timeout_sec: float = 6.0,
        extend_on_detection: bool = True
    ):
        self.segmenter = base_segmenter
        self.session_timeout_sec = session_timeout_sec
        self.extend_on_detection = extend_on_detection
        self.is_session_active = False
        self.session_expiry_time = 0.0

    def open_session(self, current_time: Optional[float] = None, duration_sec: Optional[float] = None):
        """Opens or extends the active command listening session."""
        if current_time is None:
            import time
            current_time = time.time()
        self.simulated_clock = current_time
        dur = duration_sec if duration_sec is not None else self.session_timeout_sec
        self.is_session_active = True
        self.session_expiry_time = current_time + dur
        self.segmenter.reset()

    @property
    def last_segment_info(self) -> Optional[Dict[str, Any]]:
        return getattr(self.segmenter, "last_segment_info", None)

    def close_session(self):
        """Immediately closes the listening session."""
        self.is_session_active = False
        self.session_expiry_time = 0.0
        self.simulated_clock = None
        self.segmenter.reset()

    def reset(self):
        """Resets utterance accumulation buffers without closing active session."""
        self.segmenter.reset()

    def process_chunk(self, chunk: np.ndarray, current_time: Optional[float] = None) -> Optional[Tuple[np.ndarray, float]]:
        """
        Processes audio chunk conditionally based on session gate.
        Returns segment only if session is active; otherwise returns None (gate closed).
        """
        if current_time is None:
            if hasattr(self, "simulated_clock") and self.simulated_clock is not None:
                self.simulated_clock += len(chunk) / 16000.0
                current_time = self.simulated_clock
            else:
                import time
                current_time = time.time()

        # Check session expiry
        if self.is_session_active and current_time > self.session_expiry_time:
            self.close_session()

        if not self.is_session_active:
            # Gate is closed: discard audio, zero false alarms
            return None

        res = self.segmenter.process_chunk(chunk)
        if res is not None:
            # Command detected during active session: extend session if configured
            if self.extend_on_detection:
                self.session_expiry_time = current_time + self.session_timeout_sec
        return res

    def flush(self) -> Optional[Tuple[np.ndarray, float]]:
        """Flushes underlying segmenter if session is active."""
        if not self.is_session_active:
            return None
        return self.segmenter.flush()
