"""
Acoustic Valley Segmenter for Continuous Vietnamese Keyword Spotting (KWS).
Phase 2 Prototype: Event Detection on Continuous / Connected Speech.

Solves the contiguous command challenge (e.g. 'LÊNLENPHAI' -> LEN, LEN, PHAI)
without requiring speakers to actively pause 250-300ms between words.

Mechanism:
1. Normal speech onset detection via energy threshold and pre-roll preservation.
2. Continuous accumulation while tracking short-time RMS energy trajectory.
3. Boundary / Cut detection on two criteria:
   - Silence termination: Short-time energy falls below silence threshold for silence_interval_ms.
   - Acoustic Valley / Dip detection: In continuous speech exceeding min_command_dur_before_split_ms,
     if acoustic energy dips by >= dip_threshold_db below the vocalic peak and subsequently rebounds
     by >= rebound_threshold_db, a syllable/word boundary is confirmed at the valley nadir.
4. Seamless handoff: Emits the finalized segment (right-padded to 1.0s) and re-arms immediately
   with the post-valley audio as the pre-roll of the subsequent command.
5. Preserves elongated single utterances (e.g. 'LÊÊÊN') without spurious over-splitting.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class AcousticValleySegmenter:
    """
    Syllable & Energy Valley Segmenter for Continuous KWS.
    Splits contiguous spoken commands at acoustic energy valleys without requiring silent pauses.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_interval_ms: int = 140,
        preroll_ms: int = 120,
        energy_threshold_db: float = -46.0,
        min_speech_duration_ms: int = 100,
        min_command_dur_before_split_ms: int = 160,
        dip_threshold_db: float = 12.0,
        rebound_threshold_db: float = 6.0,
        max_segment_duration_ms: int = 1200
    ):
        self.sample_rate = sample_rate
        self.silence_interval_ms = silence_interval_ms
        self.preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.min_command_samples = int((min_command_dur_before_split_ms / 1000.0) * sample_rate)
        self.dip_threshold_db = dip_threshold_db
        self.rebound_threshold_db = rebound_threshold_db
        self.max_segment_samples = int((max_segment_duration_ms / 1000.0) * sample_rate)

        # Pre-roll circular buffer
        self.preroll_buf = np.zeros(self.preroll_samples, dtype=np.float32)

        # State tracking
        self.state = "WAITING"  # "WAITING" or "ACCUMULATING"
        self.accumulated_chunks: List[np.ndarray] = []
        self.chunk_dbs: List[float] = []
        self.chunk_is_speech: List[bool] = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.peak_db = -120.0
        self.valley_db = 0.0
        self.in_valley = False
        self.valley_chunk_idx = -1
        self.current_db = -120.0
        self.noise_floor_db = energy_threshold_db
        self.is_speech_active = False
        self.last_segment_info: Optional[Dict[str, Any]] = None

    def reset(self):
        """Resets the segmenter state cleanly."""
        self.state = "WAITING"
        self.accumulated_chunks = []
        self.chunk_dbs = []
        self.chunk_is_speech = []
        self.consecutive_silence_samples = 0
        self.active_speech_samples = 0
        self.preroll_buf.fill(0.0)
        self.peak_db = -120.0
        self.valley_db = 0.0
        self.in_valley = False
        self.valley_chunk_idx = -1
        self.is_speech_active = False
        self.last_segment_info = None

    def process_chunk(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Processes an incoming 16 kHz audio chunk.
        Returns:
            (finalized_1s_audio, segment_duration_ms) if a valid utterance segment concluded,
            otherwise None.
        """
        n_samples = len(chunk)
        if n_samples == 0:
            return None

        # Compute chunk acoustic energy
        rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-7
        db = 20.0 * np.log10(rms)
        self.current_db = db

        effective_thresh = max(self.energy_threshold_db, min(self.noise_floor_db + 5.0, -32.0))
        is_speech_chunk = (db >= effective_thresh)

        if self.state == "WAITING" and not is_speech_chunk:
            self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db

        self.is_speech_active = is_speech_chunk
        finalized_segment = None

        if self.state == "WAITING":
            if is_speech_chunk:
                # Speech onset detected: capture preroll strictly preceding this chunk (zero duplication)
                preroll_audio = self.preroll_buf.copy()
                self.state = "ACCUMULATING"
                self.accumulated_chunks = [preroll_audio, chunk.copy()]
                self.chunk_dbs = [self.energy_threshold_db, db]
                self.chunk_is_speech = [False, True]
                self.consecutive_silence_samples = 0
                self.active_speech_samples = n_samples
                self.peak_db = db
                self.in_valley = False
                self.valley_chunk_idx = -1
            else:
                # Circular pre-roll update during silence
                if n_samples >= self.preroll_samples:
                    self.preroll_buf[:] = chunk[-self.preroll_samples:]
                else:
                    self.preroll_buf[:-n_samples] = self.preroll_buf[n_samples:]
                    self.preroll_buf[-n_samples:] = chunk

        elif self.state == "ACCUMULATING":
            self.accumulated_chunks.append(chunk.copy())
            self.chunk_dbs.append(db)
            self.chunk_is_speech.append(is_speech_chunk)

            if is_speech_chunk:
                self.consecutive_silence_samples = 0
                self.active_speech_samples += n_samples
            else:
                self.consecutive_silence_samples += n_samples

            total_samples = sum(len(c) for c in self.accumulated_chunks)
            silence_samples_thresh = int((self.silence_interval_ms / 1000.0) * self.sample_rate)

            # Check 1: Normal silence termination
            if self.consecutive_silence_samples >= silence_samples_thresh:
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
                        "split_reason": "silence",
                        "active_samples": self.active_speech_samples
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
                    # Check if energy has dropped significantly below recent vocalic peak
                    if (self.peak_db - db) >= self.dip_threshold_db and db < -22.0:
                        self.in_valley = True
                        self.valley_db = db
                        self.valley_chunk_idx = len(self.accumulated_chunks) - 1
                elif self.in_valley:
                    if db < self.valley_db:
                        self.valley_db = db
                        self.valley_chunk_idx = len(self.accumulated_chunks) - 1
                    # Check for energy rebound indicating onset of the next syllable/word
                    elif (db - self.valley_db) >= self.rebound_threshold_db and db >= (self.energy_threshold_db + 8.0):
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

                            # Validate minimum active speech duration
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
                                    "split_reason": "acoustic_valley",
                                    "valley_db": self.valley_db,
                                    "rebound_db": db,
                                    "active_samples": seg1_active_speech
                                }

                                # Strict active speech accounting for seg2
                                seg2_active_speech = sum(
                                    len(self.accumulated_chunks[j])
                                    for j in range(split_idx, len(self.accumulated_chunks))
                                    if self.chunk_is_speech[j]
                                )

                                # Seamless handoff: re-arm with remaining chunks for next word
                                self.accumulated_chunks = seg2_chunks
                                self.chunk_dbs = self.chunk_dbs[split_idx:]
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
