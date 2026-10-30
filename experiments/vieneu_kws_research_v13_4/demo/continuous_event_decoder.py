"""
Continuous Event Decoder for Vietnamese Keyword Spotting (KWS).
Phase 2 Prototype: Sliding-Window Probability Trajectory & NMS Decoding.

Evaluates Approach B:
Continuous sliding-window inference (e.g. 40ms or 80ms hop) over raw audio stream
with temporal peak detection, multi-frame confirmation, and Non-Maximum Suppression (NMS).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]


class ContinuousEventDecoder:
    """
    Decodes discrete keyword events from a continuous sliding-window probability trajectory.
    Implements multi-frame confirmation, hysteresis, and peak detection / NMS.
    """
    def __init__(
        self,
        theta_cmd: float = 0.20,
        min_distance_ms: int = 300,
        dip_delta: float = 0.15,
        confirmation_frames: int = 2,
        peak_hold_ms: int = 120
    ):
        self.theta_cmd = theta_cmd
        self.min_distance_sec = min_distance_ms / 1000.0
        self.dip_delta = dip_delta
        self.confirmation_frames = confirmation_frames
        self.peak_hold_sec = peak_hold_ms / 1000.0

        # State per target class
        self.last_event_time: Dict[str, float] = {c: -999.0 for c in TARGET_CLASSES}
        self.last_event_conf: Dict[str, float] = {c: 0.0 for c in TARGET_CLASSES}
        self.min_p_since_last: Dict[str, float] = {c: 1.0 for c in TARGET_CLASSES}
        self.consecutive_frames: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}

        # Trajectory buffer for local peak detection
        self.history: List[Tuple[float, str, float]] = []  # (t, top_class, top_prob)
        self.emitted_events: List[Dict[str, Any]] = []

    def reset(self):
        """Resets the decoder state."""
        self.last_event_time = {c: -999.0 for c in TARGET_CLASSES}
        self.last_event_conf = {c: 0.0 for c in TARGET_CLASSES}
        self.min_p_since_last = {c: 1.0 for c in TARGET_CLASSES}
        self.consecutive_frames = {c: 0 for c in TARGET_CLASSES}
        self.history = []
        self.emitted_events = []

    def process_frame(
        self,
        t_sec: float,
        probs: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Processes sliding window output probabilities at timestamp t_sec.
        Returns:
            Event dict if a confirmed, non-duplicate event fired at t_sec, else None.
        """
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_class = CLASSES[top_idx]

        event_emitted = None

        for idx, c in enumerate(TARGET_CLASSES):
            p = float(probs[idx])
            self.min_p_since_last[c] = min(self.min_p_since_last[c], p)

            if idx == top_idx and p >= self.theta_cmd:
                self.consecutive_frames[c] += 1
            else:
                self.consecutive_frames[c] = 0

            time_since = t_sec - self.last_event_time[c]
            dip_satisfied = (self.last_event_conf[c] - self.min_p_since_last[c]) >= self.dip_delta

            # Trigger condition:
            # 1. Class is argmax and confidence >= theta_cmd (canonical rule)
            # 2. Confirmed for >= consecutive_frames
            # 3. Time elapsed >= min_distance_sec
            # 4. If same class as last trigger, probability must have dipped by dip_delta
            #    (or elapsed beyond 1.2s to prevent permanent lockout)
            if (
                idx == top_idx
                and self.consecutive_frames[c] >= self.confirmation_frames
                and time_since >= self.min_distance_sec
            ):
                is_rearm = (time_since >= 1.20) or dip_satisfied or (self.last_event_time[c] < 0)
                if is_rearm:
                    event_dict = {
                        "timestamp": t_sec,
                        "command": c,
                        "confidence": top_prob,
                        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(5)}
                    }
                    self.emitted_events.append(event_dict)
                    self.last_event_time[c] = t_sec
                    self.last_event_conf[c] = top_prob
                    self.min_p_since_last[c] = top_prob
                    self.consecutive_frames[c] = 0
                    event_emitted = event_dict

        return event_emitted
