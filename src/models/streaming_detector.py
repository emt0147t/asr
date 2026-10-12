"""
Production Streaming Keyword Spotting Detector for Vietnamese Voice Commands.
Implements rolling-window buffer, STFT feature extraction, confidence thresholding,
temporal consecutive-window stability, debounce, and re-arming logic.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import soundfile as sf
import librosa

from src.models.kws_cnn import VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]


class StreamingKWSDetector:
    """
    Streaming Keyword Spotting Detector.
    Maintains a rolling audio circular buffer of 1.0 second (16,000 samples)
    and slides forward by hop_size (default 100 ms = 1,600 samples).
    """
    def __init__(self, model: torch.nn.Module,
                 sample_rate: int = 16000,
                 window_duration_sec: float = 1.0,
                 hop_duration_sec: float = 0.10,
                 threshold: float = 0.85,
                 consecutive_frames: int = 2,
                 debounce_sec: float = 0.60,
                 rearm_threshold: float = 0.30,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = sample_rate
        self.window_size = int(window_duration_sec * sample_rate)
        self.hop_size = int(hop_duration_sec * sample_rate)
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.debounce_sec = debounce_sec
        self.rearm_threshold = rearm_threshold

        self.extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=40)

        # State tracking
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.samples_processed = 0
        self.last_trigger_time = -999.0
        self.last_trigger_cmd = None
        self.consecutive_count = 0
        self.current_candidate = None
        self.is_armed = True

    def reset(self):
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.samples_processed = 0
        self.last_trigger_time = -999.0
        self.last_trigger_cmd = None
        self.consecutive_count = 0
        self.current_candidate = None
        self.is_armed = True

    def process_chunk(self, audio_chunk: np.ndarray) -> List[Dict[str, Any]]:
        """
        Feeds a chunk of audio (arbitrary length) into the streaming detector.
        Returns a list of detected command events.
        """
        events = []
        n_chunk = len(audio_chunk)
        if n_chunk == 0:
            return events

        for i in range(0, n_chunk, self.hop_size):
            slice_end = min(i + self.hop_size, n_chunk)
            hop_data = audio_chunk[i:slice_end]
            
            # Roll buffer
            hop_len = len(hop_data)
            self.buffer = np.roll(self.buffer, -hop_len)
            self.buffer[-hop_len:] = hop_data
            self.samples_processed += hop_len
            current_time_sec = self.samples_processed / self.sample_rate

            # Only infer after buffer is filled
            if self.samples_processed < self.window_size:
                continue

            event = self._infer_window(current_time_sec)
            if event:
                events.append(event)

        return events

    def _infer_window(self, current_time_sec: float) -> Optional[Dict[str, Any]]:
        # Peak normalization of current 1.0s window
        window_audio = self.buffer.copy()
        peak = np.max(np.abs(window_audio))
        if peak > 1e-5:
            window_audio = (window_audio / peak) * 0.89125
        else:
            # Low energy silence
            return None

        # Feature extraction
        with torch.no_grad():
            inp = torch.from_numpy(window_audio).float().to(self.device)
            log_mel = self.extractor(inp)  # (1, 1, 40, 101)
            logits = self.model(log_mel)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(np.argmax(probs))
        pred_prob = float(probs[pred_class])

        # Debounce and Re-arming logic
        time_since_last_trigger = current_time_sec - self.last_trigger_time
        if not self.is_armed:
            if time_since_last_trigger > self.debounce_sec:
                # Check if probability dropped below rearm threshold
                if pred_class >= 4 or pred_prob < self.rearm_threshold:
                    self.is_armed = True
            return None

        # Only classes 0..3 are triggerable commands
        if pred_class in [0, 1, 2, 3] and pred_prob >= self.threshold:
            cmd_name = CLASS_NAMES[pred_class]
            if cmd_name == self.current_candidate:
                self.consecutive_count += 1
            else:
                self.current_candidate = cmd_name
                self.consecutive_count = 1

            if self.consecutive_count >= self.consecutive_frames:
                # TRIGGER CONFIRMED
                event = {
                    "command": cmd_name,
                    "class_id": pred_class,
                    "timestamp_sec": round(current_time_sec, 3),
                    "confidence": round(pred_prob, 4),
                    "probabilities": {CLASS_NAMES[k]: round(float(probs[k]), 4) for k in range(6)}
                }
                self.last_trigger_time = current_time_sec
                self.last_trigger_cmd = cmd_name
                self.is_armed = False
                self.consecutive_count = 0
                self.current_candidate = None
                return event
        else:
            self.consecutive_count = 0
            self.current_candidate = None

        return None
