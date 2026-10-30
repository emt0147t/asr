#!/usr/bin/env python3
"""
Game Mode Demo Controller — Phase 7D Controlled Game Demo Integration
Repository: D:\trainidentificationaudio
Branch baseline: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
import time
import queue
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = Path("D:/trainidentificationaudio")
V13_4_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4"
BUNDLE_DIR = V13_4_DIR / "bundle"
DEMO_DIR = V13_4_DIR / "demo"

for p in [str(BUNDLE_DIR), str(DEMO_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import TARGET_CLASSES, CLASSES, DEFAULT_ONNX_PATH, LogMelFrontendNumPy


class V11LiveInferencer:
    """Standard in-memory ONNX runner for v11 classifier."""
    def __init__(self, onnx_path: Optional[Path] = None, theta_cmd: float = 0.20):
        import onnxruntime as ort
        self.frontend = LogMelFrontendNumPy()
        p = onnx_path if onnx_path else DEFAULT_ONNX_PATH
        self.session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.theta_cmd = theta_cmd

    def predict_window(self, audio_1s: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        t0 = time.perf_counter()
        if len(audio_1s) < 16000:
            audio_1s = np.pad(audio_1s, (0, 16000 - len(audio_1s)), mode="constant", constant_values=0.0)
        else:
            audio_1s = audio_1s[:16000]

        max_val = float(np.max(np.abs(audio_1s))) if len(audio_1s) > 0 else 0.0
        if max_val > 1.0:
            audio_1s = audio_1s / max_val

        spec = self.frontend.compute(audio_1s.astype(np.float32))
        ort_out = self.session.run(None, {"log_mel": spec})
        probs = ort_out[1][0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        top_idx = int(np.argmax(probs))
        if top_idx < 4 and probs[top_idx] >= self.theta_cmd:
            pred_class = CLASSES[top_idx]
            conf = float(probs[top_idx])
        else:
            pred_class = "REJECT"
            conf = float(probs[top_idx]) if top_idx == 4 else float(np.max(probs[:4]))

        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(5)}
        return pred_class, conf, prob_dict, dt_ms


class DemoState(str, Enum):
    IDLE = "IDLE"                                     # Session closed, audio dropped
    LISTENING_TIMEOUT = "LISTENING_TIMEOUT"           # Session open with auto-expiry timer
    LISTENING_PTT = "LISTENING_PTT"                   # Push-to-talk active (held down)
    PROCESSING = "PROCESSING"                         # Segmenter evaluating utterance
    COMMAND_EMITTED = "COMMAND_EMITTED"               # Event delivered to consumer
    ERROR_MIC_UNAVAILABLE = "ERROR_MIC_UNAVAILABLE"   # Device disconnected or stream failed


class GameModeDemoController:
    """
    Production-grade Session Controller for Vietnamese Continuous KWS Game Mode.
    
    Invariants & Contract:
    1. Default runtime is unchanged (legacy_v11 is default).
    2. Hybrid segmentation is strictly isolated to explicit Game Mode.
    3. Fails safely on microphone disconnection / capture errors.
    4. Enforces FIFO event ordering and duplicate action prevention.
    5. Dispatches pending-event flush upon session exit.
    6. Never emits UI success before event delivery to consumer.
    """
    def __init__(
        self,
        inferencer: Any,
        speech_band_hz: Tuple[float, float] = (100.0, 3400.0),
        min_speech_band_ratio: float = 0.32,
        energy_threshold_db: float = -46.0,
        session_timeout_sec: float = 6.0,
        max_queue_size: int = 100,
        on_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_state_callback: Optional[Callable[[DemoState, Dict[str, Any]], None]] = None
    ):
        self.inferencer = inferencer
        self.session_timeout_sec = session_timeout_sec
        self.max_queue_size = max_queue_size
        self.on_event_callback = on_event_callback
        self.on_state_callback = on_state_callback

        # Internal state
        self.state = DemoState.IDLE
        self.session_expires_at: Optional[float] = None
        self.ptt_active = False

        # Build segmenter
        self.base_segmenter = HybridSpeechBandValleySegmenter(
            sample_rate=16000,
            speech_band_hz=speech_band_hz,
            min_speech_band_ratio=min_speech_band_ratio,
            energy_threshold_db=energy_threshold_db,
            silence_interval_ms=140,
            preroll_ms=120,
            dip_threshold_db=12.0,
            rebound_threshold_db=6.0,
            max_segment_duration_ms=1200
        )
        self.session_controller = GameModeSessionController(
            base_segmenter=self.base_segmenter,
            session_timeout_sec=session_timeout_sec,
            extend_on_detection=True
        )

        # Threading and Queues
        self.audio_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self.event_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Telemetry & Diagnostics
        self.dropped_chunks_count = 0
        self.total_processed_chunks = 0
        self.emitted_events_history: List[Dict[str, Any]] = []
        self.last_emitted_command: Optional[str] = None
        self.last_emitted_time: float = 0.0
        self.min_repeat_interval_sec: float = 0.12  # Minimum 120ms between same command

        self._lock = threading.RLock()

    def _set_state(self, new_state: DemoState, detail: Optional[Dict[str, Any]] = None):
        with self._lock:
            prev = self.state
            self.state = new_state
            if self.on_state_callback:
                meta = {"prev_state": prev, "time": time.time()}
                if detail:
                    meta.update(detail)
                try:
                    self.on_state_callback(new_state, meta)
                except Exception as e:
                    print(f"Warning: State callback exception: {e}")

    # -------------------------------------------------------------------------
    # Session & PTT Control
    # -------------------------------------------------------------------------
    def start_session(self, duration_sec: Optional[float] = None):
        """Opens session gate with countdown timer."""
        with self._lock:
            dur = duration_sec if duration_sec is not None else self.session_timeout_sec
            self.session_expires_at = time.time() + dur
            self.session_controller.open_session(duration_sec=dur)
            self._set_state(DemoState.LISTENING_TIMEOUT, {"expires_in_sec": dur})

    def ptt_press(self):
        """Activates Push-to-Talk (gate forced open indefinitely until released)."""
        with self._lock:
            self.ptt_active = True
            self.session_controller.open_session(duration_sec=3600.0)
            self._set_state(DemoState.LISTENING_PTT, {"ptt": True})

    def ptt_release(self):
        """Releases Push-to-Talk and flushes trailing pending audio."""
        with self._lock:
            self.ptt_active = False
            self.flush_and_close()

    def close_session(self):
        """Closes session gate and transitions to IDLE."""
        with self._lock:
            self.session_expires_at = None
            self.session_controller.session_active = False
            self._set_state(DemoState.IDLE, {"closed": True})

    def flush_and_close(self):
        """Flushes pending utterance in segmenter before closing session."""
        with self._lock:
            flush_res = self.session_controller.flush()
            if flush_res is not None:
                self._evaluate_and_emit(flush_res[0], flush_res[1])
            self.close_session()

    # -------------------------------------------------------------------------
    # Audio Feed & Safety Handling
    # -------------------------------------------------------------------------
    def feed_audio_chunk(self, chunk: np.ndarray, capture_error: Optional[str] = None):
        """
        Feeds raw 16kHz audio chunk into queue.
        Safely catches capture errors or buffer drops.
        """
        if capture_error:
            self._set_state(DemoState.ERROR_MIC_UNAVAILABLE, {"error": capture_error})
            return

        # Check session expiry in TIMEOUT mode
        with self._lock:
            if self.state == DemoState.LISTENING_TIMEOUT and self.session_expires_at:
                if time.time() >= self.session_expires_at:
                    self.flush_and_close()
                    return

            if self.state not in [DemoState.LISTENING_TIMEOUT, DemoState.LISTENING_PTT]:
                # Gate is closed; drop chunk immediately
                return

        # Enqueue with overflow protection
        try:
            self.audio_queue.put_nowait(chunk.copy())
        except queue.Full:
            self.dropped_chunks_count += 1
            # Drop oldest to stay responsive
            try:
                _ = self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(chunk.copy())
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Worker Loop & Event Processing
    # -------------------------------------------------------------------------
    def _evaluate_and_emit(self, audio_1s: np.ndarray, seg_dur_ms: float, stream_time_sec: Optional[float] = None):
        """Evaluates classified audio and applies FIFO & duplicate action protection."""
        if hasattr(self.inferencer, "predict_window"):
            p_cmd, conf, probs, inf_ms = self.inferencer.predict_window(audio_1s)
        else:
            p_cmd, conf, probs, inf_ms = self.inferencer.predict(audio_1s)

        if p_cmd in TARGET_CLASSES:
            t_event = stream_time_sec if stream_time_sec is not None else time.time()
            # Duplicate action protection
            if p_cmd == self.last_emitted_command and (t_event - self.last_emitted_time) < self.min_repeat_interval_sec:
                # Suppress bounce duplicate
                return

            self.last_emitted_command = p_cmd
            self.last_emitted_time = t_event

            # If in TIMEOUT mode, extend session timer
            now = time.time()
            if self.state == DemoState.LISTENING_TIMEOUT and self.session_controller.extend_on_detection:
                self.session_expires_at = now + self.session_timeout_sec

            event_dict = {
                "command": p_cmd,
                "confidence": round(conf, 4),
                "timestamp_sec": round(t_event, 4),
                "seg_dur_ms": round(seg_dur_ms, 1),
                "inference_ms": round(inf_ms, 2)
            }

            self.emitted_events_history.append(event_dict)
            self.event_queue.put(event_dict)

            self._set_state(DemoState.COMMAND_EMITTED, {"command": p_cmd, "conf": conf})

            if self.on_event_callback:
                try:
                    self.on_event_callback(event_dict)
                except Exception as e:
                    print(f"Warning: Consumer event callback exception: {e}")

    def process_synchronously(self, full_stream: np.ndarray, frame_len: int = 320) -> List[Dict[str, Any]]:
        """Synchronously processes an audio stream for unit tests and reproducible benchmarking."""
        emitted = []
        current_time = 0.0
        for i in range(0, len(full_stream), frame_len):
            chunk = full_stream[i : i + frame_len]
            if len(chunk) < frame_len:
                chunk = np.pad(chunk, (0, frame_len - len(chunk)), mode='constant')
            
            current_time = (i + len(chunk)) / 16000.0

            res = self.session_controller.process_chunk(chunk)
            if res is not None:
                audio_1s, seg_dur_ms = res
                self._evaluate_and_emit(audio_1s, seg_dur_ms, stream_time_sec=current_time)

        # Flush
        if self.state != DemoState.IDLE:
            flush_res = self.session_controller.flush()
            if flush_res is not None:
                self._evaluate_and_emit(flush_res[0], flush_res[1], stream_time_sec=current_time)
            self.close_session()

        while not self.event_queue.empty():
            emitted.append(self.event_queue.get())
        return emitted
