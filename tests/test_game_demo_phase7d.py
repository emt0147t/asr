#!/usr/bin/env python3
"""
Test Suite for Phase 7D: Controlled Game Demo Integration
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
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path("D:/trainidentificationaudio")
V13_4_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4"
BUNDLE_DIR = V13_4_DIR / "bundle"
DEMO_DIR = V13_4_DIR / "demo"
SCRIPTS_DIR = V13_4_DIR / "scripts"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"

for p in [str(BUNDLE_DIR), str(DEMO_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from game_mode_demo_controller import GameModeDemoController, DemoState
from live_mic_engine import LiveKWSPipeline


class MockInferencer:
    """Mock inferencer for testing controller lifecycle and edge cases."""
    def __init__(self, fixed_command: str = "LEN", confidence: float = 0.95):
        self.fixed_command = fixed_command
        self.confidence = confidence

    def predict_window(self, audio_1s: np.ndarray):
        return self.fixed_command, self.confidence, {self.fixed_command: self.confidence}, 1.5


def get_real_clip(cmd: str, idx: int = 1) -> np.ndarray:
    path = AUDIO_DIR / f"v13_conf_cmd_{cmd}_{idx:03d}.wav"
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if sr != 16000:
        import soxr
        wav = soxr.resample(wav, sr, 16000)
    # Trim to speech
    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(wav[i : i + frame_len] ** 2)) + 1e-7) for i in range(0, len(wav), frame_len)]
    above = np.where(np.array(rms_db) > -42.0)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(wav), (above[-1] + 3) * frame_len)
        return wav[s:e]
    return wav[:6400]


def test_session_activation_and_ptt_transitions():
    """Verifies clean state transitions between IDLE, LISTENING_TIMEOUT, and LISTENING_PTT."""
    states_observed = []
    def state_cb(state, meta):
        states_observed.append(state)

    ctrl = GameModeDemoController(MockInferencer(), on_state_callback=state_cb)
    assert ctrl.state == DemoState.IDLE

    # Start session timeout
    ctrl.start_session(duration_sec=3.0)
    assert ctrl.state == DemoState.LISTENING_TIMEOUT
    assert DemoState.LISTENING_TIMEOUT in states_observed

    # PTT Press
    ctrl.ptt_press()
    assert ctrl.state == DemoState.LISTENING_PTT
    assert ctrl.ptt_active is True

    # PTT Release
    ctrl.ptt_release()
    assert ctrl.state == DemoState.IDLE
    assert ctrl.ptt_active is False


def test_session_timeout_expiry_and_auto_close():
    """Verifies that expired session timeout drops audio and returns to IDLE."""
    ctrl = GameModeDemoController(MockInferencer(), session_timeout_sec=0.1)
    ctrl.start_session(duration_sec=0.05)
    assert ctrl.state == DemoState.LISTENING_TIMEOUT

    # Sleep past expiration
    time.sleep(0.08)
    dummy_chunk = np.zeros(320, dtype=np.float32)
    ctrl.feed_audio_chunk(dummy_chunk)

    assert ctrl.state == DemoState.IDLE
    assert ctrl.audio_queue.empty()


def test_mic_unavailable_and_audio_callback_error_handling():
    """Verifies graceful handling of microphone failure or driver disconnection."""
    ctrl = GameModeDemoController(MockInferencer())
    ctrl.start_session(duration_sec=5.0)

    # Feed capture error
    ctrl.feed_audio_chunk(np.zeros(320, dtype=np.float32), capture_error="Device disconnected: Realtek(R) Audio")
    assert ctrl.state == DemoState.ERROR_MIC_UNAVAILABLE


def test_queue_overflow_and_dropped_chunks_resilience():
    """Verifies that bounded queue drops oldest chunk and updates telemetry without crashing."""
    ctrl = GameModeDemoController(MockInferencer(), max_queue_size=5)
    ctrl.start_session(duration_sec=10.0)

    # Feed 15 chunks (exceeding max_queue_size 5)
    for _ in range(15):
        ctrl.feed_audio_chunk(np.random.normal(0, 0.05, 320).astype(np.float32))

    assert ctrl.audio_queue.qsize() == 5
    assert ctrl.dropped_chunks_count > 0


def test_fifo_event_ordering_and_repeated_same_label_commands():
    """Verifies that continuous sequence LEN -> LEN -> PHAI emits distinct ordered events."""
    # Use real ONNX classifier
    from game_mode_demo_controller import V11LiveInferencer as V11ONNXInferencer
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
    inf = V11ONNXInferencer(onnx_path)

    ctrl = GameModeDemoController(inf, session_timeout_sec=10.0)
    ctrl.start_session()

    len1 = get_real_clip("LEN", 1)
    len2 = get_real_clip("LEN", 2)
    phai = get_real_clip("PHAI", 1)

    lead = np.zeros(4800, dtype=np.float32)
    gap0 = np.zeros(0, dtype=np.float32)
    tail = np.zeros(8000, dtype=np.float32)

    stream = np.concatenate([lead, len1, gap0, len2, gap0, phai, tail])
    events = ctrl.process_synchronously(stream)

    commands = [e["command"] for e in events]
    assert len(commands) == 3
    assert commands == ["LEN", "LEN", "PHAI"]


def test_duplicate_action_prevention():
    """Verifies that duplicate immediate event within <120ms is suppressed."""
    ctrl = GameModeDemoController(MockInferencer("TRAI"))
    ctrl.start_session()

    dummy_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)

    # First emission
    ctrl._evaluate_and_emit(dummy_audio, 500.0)
    assert len(ctrl.emitted_events_history) == 1

    # Immediate second emission (0ms interval) -> duplicate bounce
    ctrl._evaluate_and_emit(dummy_audio, 500.0)
    assert len(ctrl.emitted_events_history) == 1  # suppressed!


def test_pending_event_flush_on_session_close():
    """Verifies that pending audio in segmenter is flushed and emitted before session closes."""
    from game_mode_demo_controller import V11LiveInferencer as V11ONNXInferencer
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
    inf = V11ONNXInferencer(onnx_path)

    ctrl = GameModeDemoController(inf, session_timeout_sec=10.0)
    ctrl.start_session()

    len_clip = get_real_clip("LEN", 1)
    # Feed 90% of the clip without tail silence so it remains in ACCUMULATING
    partial_clip = len_clip[:int(len(len_clip) * 0.90)]
    frame_len = 320
    for i in range(0, len(partial_clip), frame_len):
        chunk = partial_clip[i : i + frame_len]
        if len(chunk) < frame_len:
            chunk = np.pad(chunk, (0, frame_len - len(chunk)), mode='constant')
        ctrl.session_controller.process_chunk(chunk)

    # State in segmenter is ACCUMULATING
    assert ctrl.base_segmenter.state == "ACCUMULATING"

    # Flush and close
    ctrl.flush_and_close()
    assert ctrl.state == DemoState.IDLE
    assert len(ctrl.emitted_events_history) == 1
    assert ctrl.emitted_events_history[0]["command"] == "LEN"


def test_production_default_isolation():
    """Verifies that LiveKWSPipeline default profile is legacy_v11 and Game Mode is opt-in."""
    pipe_default = LiveKWSPipeline()
    assert pipe_default.profile_name == "legacy_v11"
    assert pipe_default.vad_algorithm == "UtteranceSegmenter"

    # Explicit Game Mode opt-in
    pipe_game = LiveKWSPipeline(profile="game_mode")
    assert pipe_game.profile_name == "game_mode"
    assert pipe_game.vad_algorithm == "GameModeSessionController"
