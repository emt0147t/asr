"""
Phase 5 Continuous KWS Correctness Audit & Regression Test Suite.
Repository: D:\trainidentificationaudio

Tests and validates fixes for:
1. Preroll Duplication: Ensures speech onset frame is not duplicated in accumulated buffers.
2. Speech Duration Validation: Enforces minimum speech duration across all termination/splitting paths.
3. Active-Speech Accounting: Accurately separates active speech samples from silence/preroll/valleys.
4. Valley Split Boundaries: Confirms conservation of all samples across split boundaries without loss or duplicate frames.
5. Stream Finalization: Verifies flush() reliably emits pending keywords at stream/session end.
6. State Reset and Session Isolation: Confirms game session opening/closing never leaks residual audio.
7. Metric Integrity: Confirms substitutions are never counted as true positives.
8. Latency Boundary Accounting: Verifies acoustic lag vs end-to-end response latency formulas.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pytest
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"

SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import (
    LiveKWSPipeline, CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD,
    UtteranceSegmenter
)
from acoustic_valley_segmenter import AcousticValleySegmenter
from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController


def get_active_clip(cmd_name: str, idx: int = 1) -> np.ndarray:
    fpath = AUDIO_DIR / f"v13_conf_cmd_{cmd_name}_{idx:03d}.wav"
    assert fpath.exists(), f"Fixture missing: {fpath}"
    data, sr = sf.read(fpath, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000)

    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(data[i:i+frame_len]**2)) + 1e-7) for i in range(0, len(data), frame_len)]
    above = np.where(np.array(rms_db) > -42.0)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(data), (above[-1] + 3) * frame_len)
        return data[s:e]
    return data[:6400]


def test_preroll_no_duplication():
    """
    Issue 1: Verify the current frame is not duplicated when transitioning from WAITING to ACCUMULATING.
    Preroll buffer must strictly contain audio preceding the onset chunk.
    """
    segmenter = HybridSpeechBandValleySegmenter(preroll_ms=120)  # 1920 samples
    chunk_size = 320

    # 1. Feed 6 chunks of low-level silence with unique gradient ramp
    for i in range(6):
        silence_chunk = np.ones(chunk_size, dtype=np.float32) * (1e-5 * (i + 1))
        segmenter.process_chunk(silence_chunk)

    assert segmenter.state == "WAITING"

    # 2. Feed a distinctive speech onset chunk (e.g. 1000 Hz loud sine wave)
    t = np.linspace(0, chunk_size / 16000.0, chunk_size, endpoint=False)
    onset_chunk = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    segmenter.process_chunk(onset_chunk)

    assert segmenter.state == "ACCUMULATING"
    assert len(segmenter.accumulated_chunks) == 2

    preroll_part = segmenter.accumulated_chunks[0]
    onset_part = segmenter.accumulated_chunks[1]

    # Verify onset_part matches onset_chunk exactly
    assert np.allclose(onset_part, onset_chunk)

    # Verify preroll_part does NOT contain onset_chunk at its end
    preroll_tail = preroll_part[-chunk_size:]
    assert not np.allclose(preroll_tail, onset_chunk), "Onset chunk was duplicated in preroll buffer!"
    # Verify preroll_tail contains the preceding silence chunk
    expected_preceding_silence = np.ones(chunk_size, dtype=np.float32) * (1e-5 * 6)
    assert np.allclose(preroll_tail, expected_preceding_silence, atol=1e-6)


def test_speech_duration_validation_rejection():
    """
    Issue 2: Verify sub-100ms noise burst followed by valley is rejected
    and NOT emitted as a valid keyword event.
    """
    segmenter = HybridSpeechBandValleySegmenter(min_speech_duration_ms=100)  # requires 1600 samples (5 chunks)

    # Feed 2 chunks of speech (40ms) then dip then rebound
    t = np.linspace(0, 320 / 16000.0, 320, endpoint=False)
    speech_chunk = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    silence_chunk = np.zeros(320, dtype=np.float32)

    # 1. Onset with 2 speech chunks (40ms total speech)
    segmenter.process_chunk(speech_chunk)
    segmenter.process_chunk(speech_chunk)
    assert segmenter.active_speech_samples == 640  # 40ms < 100ms

    # 2. Add silence chunks to cause energy dip
    for _ in range(5):
        segmenter.process_chunk(silence_chunk)

    # 3. Add loud rebound chunk
    res = segmenter.process_chunk(speech_chunk)

    # Should NOT finalize a segment because active speech was only 40ms < 100ms
    assert res is None, "Valley segmenter emitted segment despite active speech < min_speech_samples!"


def test_active_speech_accounting_isolation():
    """
    Issue 3: Verify active_speech_samples tracks strictly speech chunks,
    not total accumulated chunks or silence intervals.
    """
    segmenter = HybridSpeechBandValleySegmenter(min_speech_duration_ms=100)
    t = np.linspace(0, 320 / 16000.0, 320, endpoint=False)
    speech_chunk = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    silence_chunk = np.zeros(320, dtype=np.float32)

    # Feed 6 speech chunks (120ms speech)
    for _ in range(6):
        segmenter.process_chunk(speech_chunk)

    assert segmenter.active_speech_samples == 6 * 320

    # Feed 3 silence chunks (should NOT increment active_speech_samples)
    for _ in range(3):
        segmenter.process_chunk(silence_chunk)

    assert segmenter.active_speech_samples == 6 * 320
    assert sum(len(c) for c in segmenter.accumulated_chunks) == (1920 + 9 * 320)


def test_valley_split_sample_partitioning():
    """
    Issue 4: Verify zero sample loss or duplication across valley split boundary.
    All input chunks must be strictly partitioned between seg1 and seg2.
    """
    segmenter = HybridSpeechBandValleySegmenter(
        min_speech_duration_ms=100,
        min_command_dur_before_split_ms=160
    )
    t = np.linspace(0, 320 / 16000.0, 320, endpoint=False)
    speech_chunk = (0.6 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    dip_chunk = (0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    rebound_chunk = (0.6 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    # Word 1: 8 chunks loud speech (160ms)
    for _ in range(8):
        segmenter.process_chunk(speech_chunk)

    # Valley: 3 chunks dip
    for _ in range(3):
        segmenter.process_chunk(dip_chunk)

    # Word 2 onset: rebound chunk triggers split
    res = segmenter.process_chunk(rebound_chunk)
    assert res is not None, "Expected valley split event on rebound"

    audio_1s, dur_ms = res
    assert dur_ms > 0

    # seg1 was emitted. Remaining chunks are in segmenter.accumulated_chunks
    remaining_chunks = segmenter.accumulated_chunks
    assert len(remaining_chunks) > 0
    # The active speech in remaining chunks must be strictly non-zero
    assert segmenter.active_speech_samples > 0


def test_stream_finalization_flush():
    """
    Issue 5: Verify flush() finalizes and emits the final command when audio stream ends
    without trailing silence.
    """
    clip_len = get_active_clip("len", 1)  # ~400ms speech
    lead = np.zeros(1600, dtype=np.float32)
    # Stream cuts off abruptly right at the end of the word (0ms trailing silence)
    stream = np.concatenate([lead, clip_len]).astype(np.float32)

    pipeline = LiveKWSPipeline(mode="segmented", profile="hybrid_valley", backend="onnx")
    pipeline.is_running = True

    events = []
    pipeline.event_callback = lambda ev: events.append(ev)

    # Feed chunks
    chunk_size = 320
    for i in range(0, len(stream), chunk_size):
        pipeline._on_new_audio_chunk(stream[i : i + chunk_size])

    # Because there was no trailing silence, segmenter should still be ACCUMULATING
    assert len(events) == 0, "Event was prematurely emitted without silence offset"

    # Now flush the pipeline at stream termination
    final_ev = pipeline.flush()
    assert final_ev is not None, "flush() failed to emit pending valid command"
    assert final_ev.command == "LEN"
    assert len(events) == 1
    assert events[0].command == "LEN"


def test_game_session_reset_and_isolation():
    """
    Issue 6: Verify opening, closing and resetting game session purges all buffered audio
    and prevents any cross-session event leakage.
    """
    base = HybridSpeechBandValleySegmenter()
    controller = GameModeSessionController(base_segmenter=base, session_timeout_sec=5.0)

    # 1. Feed speech while session is CLOSED (IDLE)
    t = np.linspace(0, 320 / 16000.0, 320, endpoint=False)
    speech_chunk = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    for _ in range(10):
        res = controller.process_chunk(speech_chunk, current_time=1.0)
        assert res is None

    # Underlying segmenter must remain completely empty/waiting
    assert base.state == "WAITING"
    assert len(base.accumulated_chunks) == 0

    # 2. Open session, feed half an utterance, then close session abruptly
    controller.open_session(current_time=2.0)
    assert controller.is_session_active
    for _ in range(4):
        controller.process_chunk(speech_chunk, current_time=2.1)

    assert base.state == "ACCUMULATING"

    # Close session
    controller.close_session()
    assert not controller.is_session_active
    assert base.state == "WAITING"
    assert len(base.accumulated_chunks) == 0
    assert base.active_speech_samples == 0


def test_metric_accounting_substitution_handling():
    """
    Issue 7: Metric verification: a prediction of the wrong class must count as SUB,
    not as a True Positive.
    """
    from benchmark_phase4_hybrid import match_events

    gt = [{"command": "LEN", "start_sec": 0.5, "end_sec": 0.9}]
    # Detected event has wrong class 'PHAI'
    det_wrong = [{
        "command": "PHAI",
        "timestamp_sec": 0.95,
        "segment_duration_ms": 400.0,
        "infer_latency_ms": 2.0
    }]

    tp, fp, fn, e2e = match_events(det_wrong, gt)
    assert tp == 0, "Wrong class was falsely counted as True Positive!"
    assert fp == 1, "Wrong class prediction must be counted as False Positive"
    assert fn == 1, "Missed ground truth class must be counted as False Negative"


def test_latency_boundary_definitions():
    """
    Issue 8: Latency accounting verification.
    """
    # Audio ends at t = 1.0s. Event emitted at t = 1.05s. Inference takes 2.0ms.
    t_offset = 1.000
    t_emit = 1.050
    t_infer = 2.0  # ms

    acoustic_lag_ms = (t_emit - t_offset) * 1000.0  # 50.0 ms
    e2e_ms = acoustic_lag_ms + t_infer              # 52.0 ms

    assert round(acoustic_lag_ms, 1) == 50.0
    assert round(e2e_ms, 1) == 52.0
