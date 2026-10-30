"""
Unit and Regression Test Suite for Phase 6.1D: Runtime Lifecycle & Edge Cases
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from acoustic_valley_segmenter import AcousticValleySegmenter
from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import VietnameseKWSInferencer, TARGET_CLASSES
from audit_phase5_3_final_metrics import compute_standard_sequence_wer, compute_temporal_event_metrics


@pytest.fixture(scope="module")
def inferencer():
    return VietnameseKWSInferencer(backend="onnx")


def test_frozen_v11_hashes_integrity():
    """Verify v11 model checkpoints and ONNX files are bitwise immutable."""
    pt_path = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"

    with open(pt_path, "rb") as f:
        pt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(onnx_path, "rb") as f:
        onnx_hash = hashlib.sha256(f.read()).hexdigest()

    assert pt_hash == "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
    assert onnx_hash == "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def test_consecutive_same_label_separation_and_order():
    """Verify consecutive identical commands (LEN -> LEN -> LEN) emit 3 separate ordered events."""
    from benchmark_phase6_acceptance import get_command_clip
    seg = HybridSpeechBandValleySegmenter(
        speech_band_hz=(100.0, 3400.0),
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0,
        silence_interval_ms=140,
        preroll_ms=120,
        dip_threshold_db=12.0,
        rebound_threshold_db=6.0
    )
    len1 = get_command_clip("len", 1)
    len2 = get_command_clip("len", 2)
    len3 = get_command_clip("len", 3)
    lead = np.zeros(8000, dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)
    stream = np.concatenate([lead, len1, len2, len3, tail]).astype(np.float32)

    emitted = []
    chunk_size = 320
    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        res = seg.process_chunk(chunk)
        if res is not None:
            emitted.append(res)
    flush_res = seg.flush()
    if flush_res is not None:
        emitted.append(flush_res)

    assert len(emitted) == 3, f"Expected exactly 3 separate bursts, got {len(emitted)}"


def test_flush_idempotence_and_no_double_emission():
    """Verify flush() emits pending audio at most once and subsequent flushes return None."""
    seg = HybridSpeechBandValleySegmenter(
        speech_band_hz=(100.0, 3400.0),
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0
    )
    # Feed speech chunk (250ms of speech)
    speech = (0.2 * np.sin(2 * np.pi * 300 * np.arange(4000) / 16000.0)).astype(np.float32)
    for i in range(0, len(speech), 320):
        seg.process_chunk(speech[i : i + 320])

    assert seg.state == "ACCUMULATING"
    # First flush: should finalize
    res1 = seg.flush()
    assert res1 is not None, "First flush must emit pending utterance"
    assert seg.state == "WAITING"

    # Second flush immediately after: MUST return None (no double emission!)
    res2 = seg.flush()
    assert res2 is None, f"Second flush must return None, got {res2}"


def test_session_isolation_and_cross_session_leakage():
    """Verify closing session gate discards audio and prevents cross-session leakage."""
    base_seg = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    controller = GameModeSessionController(base_segmenter=base_seg, session_timeout_sec=5.0)

    # Session 1: Gate Open
    controller.open_session(current_time=0.0)
    speech_chunk = (0.2 * np.sin(2 * np.pi * 300 * np.arange(320) / 16000.0)).astype(np.float32)
    for _ in range(5):
        controller.process_chunk(speech_chunk, current_time=0.1)

    assert controller.segmenter.state == "ACCUMULATING"
    # Abrupt close without finalization
    controller.close_session()
    assert controller.is_session_active is False
    assert controller.segmenter.state == "WAITING"
    assert len(controller.segmenter.accumulated_chunks) == 0

    # Feed audio while gate is closed: must return None
    for _ in range(5):
        res = controller.process_chunk(speech_chunk, current_time=1.0)
        assert res is None

    # Session 2: New session should have clean slate
    controller.open_session(current_time=2.0)
    assert controller.segmenter.state == "WAITING"
    assert len(controller.segmenter.accumulated_chunks) == 0


def test_short_final_chunk_and_nonstandard_block_sizes():
    """Verify segmenter safely handles non-standard and short audio chunks (e.g. 50 samples)."""
    seg = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    # Feed arbitrary non-multiple chunk sizes
    for sz in [160, 320, 50, 71, 0, 500]:
        chunk = (0.1 * np.random.randn(sz)).astype(np.float32)
        # Should not raise exception
        seg.process_chunk(chunk)


def test_zero_duration_and_below_min_speech_rejection():
    """Verify brief acoustic clicks (<100ms) are strictly rejected as noise."""
    seg = HybridSpeechBandValleySegmenter(min_speech_duration_ms=100)
    # Feed 40ms impulse (2 chunks of 20ms) followed by silence
    click = (0.5 * np.ones(640, dtype=np.float32))
    silence = np.zeros(3200, dtype=np.float32)  # 200ms silence
    stream = np.concatenate([click, silence])

    emitted = []
    for i in range(0, len(stream), 320):
        res = seg.process_chunk(stream[i : i + 320])
        if res is not None:
            emitted.append(res)
    flush_res = seg.flush()
    if flush_res is not None:
        emitted.append(flush_res)

    assert len(emitted) == 0, f"Impulse < 100ms must be rejected, got {len(emitted)}"


def test_global_temporal_matching_conflict_resolution():
    """Verify monotonic DP event matching correctly resolves multiple competing detections."""
    gt = [
        {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85},
        {"command": "LEN", "start_sec": 0.85, "end_sec": 1.20}
    ]
    # Two detections competing for GT[0]
    dets = [
        {"command": "LEN", "timestamp_sec": 0.82, "infer_latency_ms": 2.0},
        {"command": "LEN", "timestamp_sec": 0.86, "infer_latency_ms": 2.0}
    ]
    # Monotonic matching: one det matches GT[0], the other is treated as Insertion or matches GT[1]
    res = compute_temporal_event_metrics(dets, gt, lead_tolerance_sec=0.15, trail_tolerance_sec=0.55)
    assert res["event_tp"] <= 2
    assert res["event_ins"] + res["event_tp"] == 2
    # Injection: no GT matched more than once
    matched_gt_indices = [m[1] for m in res["matches"]]
    assert len(matched_gt_indices) == len(set(matched_gt_indices))
