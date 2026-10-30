"""
Unit and Regression Test Suite for Phase 6: Game Mode Acceptance & Live Event Validation
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
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
from infer import VietnameseKWSInferencer
from benchmark_phase6_acceptance import build_trial_stream, process_stream_through_pipeline
from benchmark_phase5_audit import build_elongated_clip
from audit_phase5_3_final_metrics import compute_standard_sequence_wer, compute_temporal_event_metrics


@pytest.fixture(scope="module")
def inferencer():
    return VietnameseKWSInferencer(backend="onnx")


def test_frozen_artifact_immutability():
    """Verify v11 checkpoint and ONNX model hashes remain strictly frozen."""
    import hashlib
    pt_path = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"

    with open(pt_path, "rb") as f:
        pt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(onnx_path, "rb") as f:
        onnx_hash = hashlib.sha256(f.read()).hexdigest()

    expected_pt = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
    expected_onnx = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"

    assert pt_hash == expected_pt, f"Checkpoint hash modified! Got {pt_hash}"
    assert onnx_hash == expected_onnx, f"ONNX model hash modified! Got {onnx_hash}"


@pytest.mark.parametrize("gap_ms", [0, 50])
def test_repeated_commands_len_len_phai_burst(inferencer, gap_ms):
    """
    Verify consecutive same-label commands (LEN -> LEN) with 0ms and 50ms intentional gap
    emit separate ordered events without merging.
    """
    seg = AcousticValleySegmenter(
        silence_interval_ms=140,
        preroll_ms=120,
        energy_threshold_db=-46.0,
        dip_threshold_db=12.0,
        rebound_threshold_db=6.0
    )
    # Test with trial 1
    stream, gt_events = build_trial_stream("len_len_phai", trial_idx=1, gap_ms=gap_ms)
    dets, _ = process_stream_through_pipeline(stream, seg, inferencer, theta_cmd=0.20)
    det_cmds = [d["command"] for d in dets]
    gt_cmds = [g["command"] for g in gt_events]

    assert det_cmds == gt_cmds, f"Failed at gap {gap_ms}ms: expected {gt_cmds}, got {det_cmds}"
    assert len(dets) == 3, f"Expected exactly 3 events, got {len(dets)}"


def test_elongated_command_does_not_oversplit(inferencer):
    """Verify that an elongated single command (750ms) is not over-split into multiple cuts."""
    seg = HybridSpeechBandValleySegmenter(
        speech_band_hz=(100.0, 3400.0),
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0,
        silence_interval_ms=140,
        preroll_ms=120,
        dip_threshold_db=12.0,
        rebound_threshold_db=6.0
    )
    stream, gt_events = build_trial_stream("elongated_len", trial_idx=1, gap_ms=0)
    dets, _ = process_stream_through_pipeline(stream, seg, inferencer, theta_cmd=0.20)

    assert len(dets) == 1, f"Expected 1 event (no over-splitting), got {len(dets)}: {dets}"
    assert dets[0]["command"] == "LEN"


def test_game_mode_closed_gate_drops_audio(inferencer):
    """Verify GameModeSessionController completely suppresses chatter when session gate is closed."""
    base_seg = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    controller = GameModeSessionController(base_segmenter=base_seg, session_timeout_sec=6.0)

    # Audio contains a valid command, but gate is closed:
    stream, _ = build_trial_stream("len_len_phai", trial_idx=1, gap_ms=50)
    dets, _ = process_stream_through_pipeline(
        stream, controller, inferencer,
        is_game_mode=True, gate_open_during_stream=False
    )

    assert len(dets) == 0, f"Expected 0 detections when gate is closed, got {len(dets)}"


def test_game_mode_open_session_and_expiry(inferencer):
    """Verify GameModeSessionController activates during session and expires after timeout."""
    base_seg = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))
    controller = GameModeSessionController(base_segmenter=base_seg, session_timeout_sec=1.0)

    # Open session at t=0.0
    controller.open_session(current_time=0.0, duration_sec=1.0)
    assert controller.is_session_active is True

    chunk_silent = np.zeros(320, dtype=np.float32)
    # Process at t=0.5 (within session)
    controller.process_chunk(chunk_silent, current_time=0.5)
    assert controller.is_session_active is True

    # Process at t=1.2 (expired)
    controller.process_chunk(chunk_silent, current_time=1.2)
    assert controller.is_session_active is False


def test_stream_finalization_flush(inferencer):
    """Verify flush() finalizes an ongoing utterance segment cut off at stream end."""
    seg = AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120)
    stream, gt_events = build_trial_stream("len_len", trial_idx=1, gap_ms=0)

    # Cut stream abruptly right after the second command without trailing silence
    c2_end = int(gt_events[1]["end_sec"] * 16000)
    truncated_stream = stream[:c2_end]

    # Process through pipeline: flush must finalize the second command
    dets, _ = process_stream_through_pipeline(truncated_stream, seg, inferencer, theta_cmd=0.20)
    det_cmds = [d["command"] for d in dets]
    assert det_cmds == ["LEN", "LEN"], f"Flush failed to recover second command: got {det_cmds}"


def test_synthetic_global_matching_conflict():
    """Verify monotonic DP event matching correctly resolves overlapping windows without greedy errors."""
    gt = [
        {"command": "LEN", "start_sec": 0.50, "end_sec": 0.85},
        {"command": "LEN", "start_sec": 0.85, "end_sec": 1.20},
        {"command": "PHAI", "start_sec": 1.20, "end_sec": 1.50}
    ]
    # Prediction: one detection near boundary (dist to gt0 is +0.10s, dist to gt1 is -0.25s)
    dets = [
        {"command": "LEN", "timestamp_sec": 0.95, "infer_latency_ms": 2.0},
        {"command": "LEN", "timestamp_sec": 1.28, "infer_latency_ms": 2.0},
        {"command": "PHAI", "timestamp_sec": 1.55, "infer_latency_ms": 2.0}
    ]
    ev_res = compute_temporal_event_metrics(dets, gt, lead_tolerance_sec=0.30, trail_tolerance_sec=0.30)
    assert ev_res["event_tp"] == 3
    assert ev_res["event_fp"] == 0
    assert ev_res["event_fn"] == 0
    assert len(ev_res["matches"]) == 3
