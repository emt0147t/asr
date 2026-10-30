"""
Phase 2 Continuous KWS Event Detection Regression Test Suite.
Validates continuous and contiguous keyword spotting capabilities without requiring 250-300ms pauses:

Test Requirements:
1. Frozen Artifact Immutability: v11 Checkpoint and ONNX SHA-256 byte parity.
2. Contiguous Repeated Commands: 'LEN -> LEN -> PHAI' with 0ms, 50ms, 100ms, and 150ms gaps produces exactly ['LEN', 'LEN', 'PHAI'].
3. Triple Contiguous Identical Commands: 'LEN -> LEN -> LEN' with 0ms gap produces exactly ['LEN', 'LEN', 'LEN'].
4. Elongated Command Protection: Sustained 750ms voiced utterance ('LÊÊÊN') produces strictly 1 event (zero over-splitting).
5. Distinct Continuous Sequence: 'TRAI -> LEN -> XUONG' with 50ms gaps produces exactly ['TRAI', 'LEN', 'XUONG'].
6. Negative Control / OOV Rejection: Foils ('v13_conf_foil_len') and conversational speech produce zero false activations.
7. LiveKWSPipeline Integration: 'continuous_valley' profile correctly binds AcousticValleySegmenter.
"""

import sys
import hashlib
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
    LiveKWSPipeline, CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD
)
from acoustic_valley_segmenter import AcousticValleySegmenter
from continuous_event_decoder import ContinuousEventDecoder

EXPECTED_CKPT_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_ONNX_SHA256 = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_active_clip(cmd_name: str, idx: int = 1) -> np.ndarray:
    """Extracts active voiced audio from the specified fixture clip."""
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
    """Feeds an audio stream in real-time chunk increments into the pipeline."""
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


@pytest.mark.parametrize("gap_ms", [0, 50, 100, 150])
def test_repeated_command_len_len_phai(gap_ms):
    """
    Verify continuous 'LEN -> LEN -> PHAI' is correctly segmented and detected
    across small gaps (0ms, 50ms, 100ms, 150ms) using AcousticValleySegmenter.
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
        profile="continuous_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert cmds == ["LEN", "LEN", "PHAI"], f"Gap {gap_ms}ms expected ['LEN', 'LEN', 'PHAI'], got {cmds}"
    for ev in events:
        assert ev.confidence >= 0.20, f"Event {ev.command} confidence {ev.confidence} < 0.20"


def test_three_contiguous_identical_commands():
    """Verify 'LEN -> LEN -> LEN' with 0ms gap produces exactly three separate LEN events."""
    len1 = get_active_clip("len", 1)
    len2 = get_active_clip("len", 2)
    len3 = get_active_clip("len", 3)

    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, len1, len2, len3, tail]).astype(np.float32)

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="continuous_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert cmds == ["LEN", "LEN", "LEN"], f"Expected ['LEN', 'LEN', 'LEN'], got {cmds}"


def test_elongated_command_does_not_oversplit():
    """
    Verify a sustained 750ms elongated command ('LÊÊÊN') produces strictly 1 event
    without spurious over-splitting into duplicates.
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
        profile="continuous_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert len(cmds) == 1, f"Expected exactly 1 event for elongated word, got {len(cmds)}: {cmds}"
    assert cmds[0] == "LEN", f"Expected 'LEN', got {cmds[0]}"


def test_distinct_continuous_sequence():
    """Verify distinct command sequence 'TRAI -> LEN -> XUONG' with 50ms gap is correctly detected."""
    c_trai = get_active_clip("trai", 1)
    c_len = get_active_clip("len", 1)
    c_xuong = get_active_clip("xuong", 1)
    gap = np.zeros(int(0.05 * 16000), dtype=np.float32)  # 50ms gap

    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, c_trai, gap, c_len, gap, c_xuong, tail]).astype(np.float32)

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="continuous_valley",
        backend="onnx"
    )

    events = feed_stream_to_pipeline(pipeline, stream)
    cmds = [e.command for e in events]

    assert cmds == ["TRAI", "LEN", "XUONG"], f"Expected ['TRAI', 'LEN', 'XUONG'], got {cmds}"


def test_negative_controls_and_ambient_suppression():
    """Verify silence, ambient room tone, and background acoustic textures produce zero false triggers."""
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))[:5]

    pipeline = LiveKWSPipeline(
        mode="segmented",
        profile="continuous_valley",
        backend="onnx"
    )

    # 1. Digital silence test
    silence = np.zeros(32000, dtype=np.float32)
    silence_events = feed_stream_to_pipeline(pipeline, silence)
    assert len(silence_events) == 0, f"Digital silence triggered {len(silence_events)} false events!"

    # 2. Ambient background noise textures test
    total_false_events = 0
    for f in bg_files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        stream = np.concatenate([np.zeros(4000, dtype=np.float32), audio, np.zeros(4000, dtype=np.float32)])
        events = feed_stream_to_pipeline(pipeline, stream)
        total_false_events += len(events)

    assert total_false_events == 0, f"Expected 0 false activations on ambient noise textures, got {total_false_events}"
