"""
Phase 4 Regression Test Suite: Hybrid Continuous KWS with False Activation Control.
Repository: D:\trainidentificationaudio

Validates:
1. Frozen Artifact Immutability: v11 Checkpoint and ONNX SHA-256 byte parity.
2. HybridSpeechBandValleySegmenter:
   - Contiguous Repeated Commands: 'LEN -> LEN -> PHAI' with 0ms, 50ms, 100ms gaps.
   - Triple Contiguous Identical Commands: 'LEN -> LEN -> LEN' (0ms gap).
   - Elongated Keyword Protection: 750ms sustained vocalic utterance produces strictly 1 event.
   - Out-of-band energy rejection (spectral concentration < 0.32).
3. Threshold Sweep Behavior:
   - Increasing theta_cmd from 0.20 to 0.50 suppresses false activations while retaining high-confidence keywords.
4. GameModeSessionController:
   - Idle State (Gate Closed): strictly suppresses conversational speech and ambient noise (FAR = 0.00).
   - Active Gameplay State (Gate Open): accurately detects contiguous command bursts ('LEN -> LEN -> PHAI').
   - Timeout Expiry: automatically transitions back to IDLE after timeout.
5. LiveKWSPipeline Integration:
   - Profile 'hybrid_valley' binds HybridSpeechBandValleySegmenter.
   - Profile 'game_mode' binds GameModeSessionController.
   - Profile 'legacy_v11' binds UtteranceSegmenter.
"""

import sys
import hashlib
import time
from pathlib import Path
from typing import List, Any
import pytest
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import (
    LiveKWSPipeline, CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD,
    UtteranceSegmenter
)
from acoustic_valley_segmenter import AcousticValleySegmenter
from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController

EXPECTED_CKPT_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_ONNX_SHA256 = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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


def feed_stream_to_pipeline(pipeline: LiveKWSPipeline, stream: np.ndarray, chunk_size: int = 320) -> List[Any]:
    events = []
    pipeline.event_callback = lambda ev: events.append(ev)
    pipeline.is_running = True
    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)
    return events


def test_frozen_artifact_immutability():
    """Verify v11 model checkpoint and ONNX artifact byte hashes are unaltered."""
    ckpt_path = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"

    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
    assert onnx_path.exists(), f"ONNX file missing: {onnx_path}"

    assert compute_sha256(ckpt_path) == EXPECTED_CKPT_SHA256
    assert compute_sha256(onnx_path) == EXPECTED_ONNX_SHA256


@pytest.mark.parametrize("gap_ms", [0, 50, 100])
def test_hybrid_repeated_command_len_len_phai(gap_ms):
    """
    Verify continuous 'LEN -> LEN -> PHAI' is correctly segmented and detected
    across small gaps (0ms, 50ms, 100ms) using HybridSpeechBandValleySegmenter.
    """
    len1 = get_active_clip("len", 1)
    len2 = get_active_clip("len", 2)
    phai1 = get_active_clip("phai", 1)

    gap = np.zeros(int((gap_ms / 1000.0) * 16000), dtype=np.float32)
    lead = np.zeros(8000, dtype=np.float32)   # 500ms
    tail = np.zeros(12000, dtype=np.float32)  # 750ms

    stream = np.concatenate([lead, len1, gap, len2, gap, phai1, tail]).astype(np.float32)

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="hybrid_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert cmds == ["LEN", "LEN", "PHAI"], f"Gap {gap_ms}ms expected ['LEN', 'LEN', 'PHAI'], got {cmds}"
    for ev in events:
        assert ev.confidence >= 0.20, f"Event {ev.command} confidence {ev.confidence} < 0.20"


def test_hybrid_three_contiguous_len():
    """Verify 'LEN -> LEN -> LEN' with 0ms gap produces exactly three separate LEN events in hybrid profile."""
    len1 = get_active_clip("len", 1)
    len2 = get_active_clip("len", 2)
    len3 = get_active_clip("len", 3)

    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, len1, len2, len3, tail]).astype(np.float32)

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="hybrid_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert cmds == ["LEN", "LEN", "LEN"], f"Expected ['LEN', 'LEN', 'LEN'], got {cmds}"


def test_hybrid_elongated_command_protection():
    """
    Verify a sustained 750ms elongated command ('LÊÊÊN') produces strictly 1 event
    without spurious over-splitting.
    """
    clip = get_active_clip("len", 1)
    frame_len = 160
    rms_vals = [np.sqrt(np.mean(clip[i:i+frame_len]**2)) for i in range(0, len(clip), frame_len)]
    peak_idx = int(np.argmax(rms_vals))
    vowel_slice = clip[peak_idx*frame_len : (peak_idx+4)*frame_len]
    num_loops = max(1, int((750 * 16 - len(clip)) / len(vowel_slice)))
    elongated = np.concatenate([clip[:peak_idx*frame_len]] + [vowel_slice] * num_loops + [clip[(peak_idx+4)*frame_len:]]).astype(np.float32)

    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, elongated, tail]).astype(np.float32)

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="hybrid_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert len(cmds) == 1, f"Expected exactly 1 event for elongated word, got {len(cmds)}: {cmds}"
    assert cmds[0] == "LEN", f"Expected 'LEN', got {cmds[0]}"


def test_hybrid_speechband_out_of_band_rejection():
    """Verify high-frequency sinusoidal tone outside 300-3400 Hz is rejected by ratio gating."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 6000 Hz pure tone (outside 300-3400 Hz speech band)
    high_freq_tone = (0.2 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    tail = np.zeros(8000, dtype=np.float32)
    stream = np.concatenate([high_freq_tone, tail]).astype(np.float32)

    segmenter = HybridSpeechBandValleySegmenter(
        speech_band_hz=(300.0, 3400.0),
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0
    )

    segments = []
    chunk_size = 320
    for i in range(0, len(stream), chunk_size):
        res = segmenter.process_chunk(stream[i : i + chunk_size])
        if res is not None:
            segments.append(res)

    assert len(segments) == 0, f"Expected 0 segments for 6000 Hz out-of-band tone, got {len(segments)}"


def test_threshold_sweep_rejection_behavior():
    """
    Verify increasing theta_cmd from 0.20 to 0.50 maintains high-confidence keywords
    while raising the confidence gate.
    """
    c_len = get_active_clip("len", 1)
    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, c_len, tail]).astype(np.float32)

    pipe_low = LiveKWSPipeline(mode="segmented", profile="hybrid_valley", theta_cmd=0.20)
    evs_low = feed_stream_to_pipeline(pipe_low, stream)
    assert len(evs_low) == 1
    assert evs_low[0].command == "LEN"
    conf = evs_low[0].confidence

    pipe_high = LiveKWSPipeline(mode="segmented", profile="hybrid_valley", theta_cmd=0.50)
    evs_high = feed_stream_to_pipeline(pipe_high, stream)
    if conf >= 0.50:
        assert len(evs_high) == 1
        assert evs_high[0].command == "LEN"
    else:
        assert len(evs_high) == 0


def test_game_mode_closed_gate_suppression():
    """Verify that when game mode session is closed, conversational speech is 100% suppressed."""
    conv_file = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))[0]
    audio, sr = sf.read(conv_file, dtype="float32")
    if audio.ndim > 1: audio = np.mean(audio, axis=-1)
    trail = np.zeros(8000, dtype=np.float32)
    stream = np.concatenate([audio, trail]).astype(np.float32)

    pipeline = LiveKWSPipeline(mode="segmented", profile="game_mode", backend="onnx")
    assert isinstance(pipeline.segmenter, GameModeSessionController)
    assert not pipeline.segmenter.is_session_active

    events = feed_stream_to_pipeline(pipeline, stream)
    assert len(events) == 0, f"Expected 0 events with closed game mode gate, got {len(events)}"


def test_game_mode_active_burst_detection():
    """Verify that opening the game mode session allows rapid contiguous command detection."""
    len1 = get_active_clip("len", 1)
    len2 = get_active_clip("len", 2)
    phai1 = get_active_clip("phai", 1)
    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, len1, len2, phai1, tail]).astype(np.float32)

    pipeline = LiveKWSPipeline(mode="segmented", profile="game_mode", backend="onnx")
    assert isinstance(pipeline.segmenter, GameModeSessionController)
    pipeline.segmenter.open_session(current_time=time.time(), duration_sec=10.0)
    assert pipeline.segmenter.is_session_active

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]
    assert cmds == ["LEN", "LEN", "PHAI"], f"Expected ['LEN', 'LEN', 'PHAI'], got {cmds}"


def test_game_mode_timeout_expiry():
    """Verify game mode session automatically expires after timeout."""
    base_seg = HybridSpeechBandValleySegmenter()
    controller = GameModeSessionController(base_segmenter=base_seg, session_timeout_sec=2.0)
    controller.open_session(current_time=100.0, duration_sec=2.0)
    assert controller.is_session_active

    chunk = np.zeros(320, dtype=np.float32)
    # Chunk arriving at t = 101.0 (active)
    controller.process_chunk(chunk, current_time=101.0)
    assert controller.is_session_active

    # Chunk arriving at t = 103.0 (past timeout 102.0)
    controller.process_chunk(chunk, current_time=103.0)
    assert not controller.is_session_active


def test_live_mic_engine_profile_bindings():
    """Verify LiveKWSPipeline correctly instantiates the expected segmenter for each profile."""
    pipe_legacy = LiveKWSPipeline(profile="legacy_v11")
    assert isinstance(pipe_legacy.segmenter, UtteranceSegmenter)
    assert pipe_legacy.vad_algorithm == "UtteranceSegmenter"

    pipe_valley = LiveKWSPipeline(profile="continuous_valley")
    assert isinstance(pipe_valley.segmenter, AcousticValleySegmenter)
    assert pipe_valley.vad_algorithm == "AcousticValleySegmenter"

    pipe_hybrid = LiveKWSPipeline(profile="hybrid_valley")
    assert isinstance(pipe_hybrid.segmenter, HybridSpeechBandValleySegmenter)
    assert pipe_hybrid.vad_algorithm == "HybridSpeechBandValleySegmenter"

    pipe_game = LiveKWSPipeline(profile="game_mode")
    assert isinstance(pipe_game.segmenter, GameModeSessionController)
    assert pipe_game.vad_algorithm == "GameModeSessionController"
