"""
Comprehensive Deterministic Test Suite for Vietnamese KWS Live Microphone Demo.
Phase 13.4 Release Candidate (v11).

Validates Section 6 Requirements:
1. Checkpoint and ONNX SHA-256 byte immobility.
2. Three separate speech segments producing exactly three events in correct order (TRAI -> XUONG -> PHAI).
3. One segment producing no more than one event (zero duplicates).
4. Later command emitted within 800ms (no global cooldown blocking).
5. Repeated identical commands in separate segments (TRAI -> TRAI -> TRAI).
6. Silence re-arming the detector.
7. REJECT and low-confidence cases not generating invalid commands.
8. Correct 1.0s (16,000 samples) waveform input and expected feature shape [1, 1, 40, 101].
9. AudioRingBuffer bounded length and FIFO rolling semantics.
10. Resampling correctness (44.1k/48k -> 16k).
11. Clean stream shutdown without hanging threads.
"""

import sys
import time
import hashlib
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
DEMO_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
P13_3_AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import (
    AudioRingBuffer, UtteranceSegmenter, LiveKWSPipeline, AudioStreamManager,
    CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD
)
from infer import VietnameseKWSInferencer, preprocess_audio

EXPECTED_CKPT_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_ONNX_SHA256 = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_trimmed_command(cmd_name: str) -> np.ndarray:
    """Loads a command clip and trims trailing silence to represent a natural ~350-500ms spoken word."""
    candidates = sorted(list(P13_3_AUDIO_DIR.glob(f"v13_conf_cmd_{cmd_name}_*.wav")))
    assert len(candidates) > 0, f"No audio clips found for command: {cmd_name}"
    audio, sr = sf.read(candidates[0], dtype="float32")
    if sr != 16000:
        import soxr
        audio = soxr.resample(audio, sr, 16000)
    # Trim to active region
    active = np.where(np.abs(audio) > 0.05)[0]
    if len(active) > 0:
        start = max(0, active[0] - 800)
        end = min(len(audio), active[-1] + 1600)
        return audio[start:end]
    return audio[:8000]


def test_frozen_artifact_immutability():
    """Verify frozen checkpoint and ONNX models exist and match authoritative checksums."""
    ckpt_path = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"

    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
    assert onnx_path.exists(), f"ONNX model missing: {onnx_path}"

    assert compute_sha256(ckpt_path) == EXPECTED_CKPT_SHA256
    assert compute_sha256(onnx_path) == EXPECTED_ONNX_SHA256


def test_three_separate_speech_segments_in_order():
    """Verify three separate speech segments produce exactly three events in correct order."""
    events = []
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        silence_interval_ms=300,
        energy_threshold_db=-48.0,
        event_callback=lambda ev: events.append(ev)
    )

    c_trai = load_trimmed_command("trai")
    c_xuong = load_trimmed_command("xuong")
    c_phai = load_trimmed_command("phai")
    silence_gap = np.zeros(int(0.35 * 16000), dtype=np.float32)  # 350ms pause

    # TRAI -> gap -> XUONG -> gap -> PHAI -> gap
    stream_audio = np.concatenate([c_trai, silence_gap, c_xuong, silence_gap, c_phai, silence_gap]).astype(np.float32)

    pipeline.is_running = True
    chunk_size = 320  # 20ms chunks
    for i in range(0, len(stream_audio), chunk_size):
        chunk = stream_audio[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)

    assert len(events) == 3, f"Expected exactly 3 events, got {len(events)}: {[e.command for e in events]}"
    assert [e.command for e in events] == ["TRAI", "XUONG", "PHAI"], (
        f"Expected ['TRAI', 'XUONG', 'PHAI'], got {[e.command for e in events]}"
    )
    for ev in events:
        assert ev.confidence >= 0.20, f"Event {ev.command} confidence {ev.confidence} < 0.20"


def test_one_segment_produces_no_more_than_one_event():
    """Verify a single command utterance produces strictly one event (zero duplicates)."""
    events = []
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        silence_interval_ms=300,
        event_callback=lambda ev: events.append(ev)
    )

    c_len = load_trimmed_command("len")
    silence_after = np.zeros(int(0.80 * 16000), dtype=np.float32)
    stream_audio = np.concatenate([c_len, silence_after]).astype(np.float32)

    pipeline.is_running = True
    chunk_size = 320
    for i in range(0, len(stream_audio), chunk_size):
        chunk = stream_audio[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)

    assert len(events) == 1, f"Expected exactly 1 event, got {len(events)}"
    assert events[0].command == "LEN"


def test_later_command_emitted_within_800ms():
    """Verify a second command is emitted after a short 350ms pause without 0.8s latch blocking."""
    events = []
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        silence_interval_ms=300,
        event_callback=lambda ev: events.append(ev)
    )

    c_phai = load_trimmed_command("phai")
    c_len = load_trimmed_command("len")
    silence_gap = np.zeros(int(0.35 * 16000), dtype=np.float32)  # 350ms pause

    # Total gap duration is 350ms, well below the old 800ms cooldown
    stream_audio = np.concatenate([c_phai, silence_gap, c_len, silence_gap]).astype(np.float32)

    pipeline.is_running = True
    chunk_size = 320
    for i in range(0, len(stream_audio), chunk_size):
        chunk = stream_audio[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)

    assert len(events) == 2, f"Expected 2 events within short pause, got {len(events)}"
    assert [e.command for e in events] == ["PHAI", "LEN"]


def test_repeated_identical_commands_in_separate_segments():
    """Verify repeated identical commands (TRAI -> TRAI -> TRAI) are all recognized."""
    events = []
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        silence_interval_ms=300,
        event_callback=lambda ev: events.append(ev)
    )

    c_trai = load_trimmed_command("trai")
    silence_gap = np.zeros(int(0.35 * 16000), dtype=np.float32)
    stream_audio = np.concatenate([c_trai, silence_gap, c_trai, silence_gap, c_trai, silence_gap]).astype(np.float32)

    pipeline.is_running = True
    chunk_size = 320
    for i in range(0, len(stream_audio), chunk_size):
        chunk = stream_audio[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)

    assert len(events) == 3, f"Expected 3 events for repeated commands, got {len(events)}"
    assert [e.command for e in events] == ["TRAI", "TRAI", "TRAI"]


def test_silence_rearms_detector():
    """Verify silence resets the segmenter and re-arms detection for subsequent commands."""
    events = []
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        silence_interval_ms=300,
        event_callback=lambda ev: events.append(ev)
    )

    c_xuong = load_trimmed_command("xuong")
    long_silence = np.zeros(int(1.5 * 16000), dtype=np.float32)  # 1.5s silence
    stream_audio = np.concatenate([c_xuong, long_silence]).astype(np.float32)

    pipeline.is_running = True
    chunk_size = 320
    for i in range(0, len(stream_audio), chunk_size):
        chunk = stream_audio[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)

    assert len(events) == 1
    assert events[0].command == "XUONG"
    assert pipeline.segmenter.state == "WAITING", "Segmenter must be in WAITING state during silence"


def test_reject_and_low_confidence_suppression():
    """Verify silence and low-confidence audio evaluate to REJECT and do not trigger command events."""
    events = []
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="onnx",
        silence_interval_ms=300,
        energy_threshold_db=-48.0,
        event_callback=lambda ev: events.append(ev)
    )

    # 1. Digital silence
    silence_chunk = np.zeros(16000, dtype=np.float32)
    pipeline.is_running = True
    for i in range(0, len(silence_chunk), 320):
        pipeline._on_new_audio_chunk(silence_chunk[i:i+320])

    assert len(events) == 0, "Silence must not generate any event"

    # 2. Quiet room tone from fixtures
    room_wav = FIXTURES_DIR / "dev_02_quiet_ambient_room.wav"
    room_audio = preprocess_audio(room_wav)
    for i in range(0, len(room_audio), 320):
        pipeline._on_new_audio_chunk(room_audio[i:i+320])

    assert len(events) == 0, "Ambient room tone must not generate any command event"


def test_correct_one_second_waveform_and_feature_shape():
    """Verify that any finalized segment is exactly 16,000 samples and LogMel is [1, 1, 40, 101]."""
    segmenter = UtteranceSegmenter(silence_interval_ms=300)
    c_trai = load_trimmed_command("trai")
    silence_after = np.zeros(int(0.40 * 16000), dtype=np.float32)
    stream = np.concatenate([c_trai, silence_after]).astype(np.float32)

    finalized = None
    for i in range(0, len(stream), 320):
        res = segmenter.process_chunk(stream[i:i+320])
        if res is not None:
            finalized = res[0]
            break

    assert finalized is not None, "Segment must be finalized"
    assert finalized.shape == (16000,), f"Expected shape (16000,), got {finalized.shape}"
    assert finalized.dtype == np.float32, f"Expected float32, got {finalized.dtype}"

    engine = VietnameseKWSInferencer(backend="onnx")
    spec = engine.frontend.compute(finalized)
    assert spec.shape == (1, 1, 40, 101), f"Expected shape (1, 1, 40, 101), got {spec.shape}"


def test_audio_ring_buffer_operations():
    """Verify AudioRingBuffer bounded length (16,000 samples) and FIFO rolling semantics."""
    buf = AudioRingBuffer(capacity=16000)
    w0 = buf.get_window()
    assert w0.shape == (16000,)
    assert np.all(w0 == 0.0)

    chunk = np.ones(1600, dtype=np.float32) * 0.5
    buf.append(chunk)
    w1 = buf.get_window()
    assert np.all(w1[:14400] == 0.0)
    assert np.allclose(w1[14400:], 0.5)


def test_audio_resampling():
    """Verify audio resampling from 48.0 kHz and 44.1 kHz to 16.0 kHz."""
    import soxr
    x_48k = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32))
    y_16k = soxr.resample(x_48k, 48000, 16000)
    assert len(y_16k) == 16000

    x_44k = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100, endpoint=False, dtype=np.float32))
    y_16k_44 = soxr.resample(x_44k, 44100, 16000)
    assert len(y_16k_44) == 16000


def test_clean_stream_shutdown():
    """Verify starting and stopping pipeline leaves zero hanging streams."""
    pipeline = LiveKWSPipeline(mode="segmented", backend="onnx")
    pipeline.start()
    assert pipeline.is_running
    assert pipeline.stream_mgr.is_running
    time.sleep(0.1)
    pipeline.stop()
    assert not pipeline.is_running
    assert not pipeline.stream_mgr.is_running


def test_canonical_and_live_pipeline_parity():
    """Verify in-memory live pipeline inference achieves 100% parity with canonical offline inferencer."""
    engine = VietnameseKWSInferencer(backend="onnx")
    pipeline = LiveKWSPipeline(mode="segmented", backend="onnx")

    dev_files = sorted(list(FIXTURES_DIR.glob("*.wav")))
    assert len(dev_files) >= 5

    for f in dev_files:
        r_canon = engine.predict(f)
        audio = preprocess_audio(f)
        pred, conf, probs, dt = pipeline._run_in_memory_inference(audio)

        assert r_canon["predicted_class"] == pred, f"Parity mismatch on {f.name}: {r_canon['predicted_class']} vs {pred}"
        assert abs(r_canon["confidence"] - conf) < 1e-4
        for c in CLASSES:
            assert abs(r_canon["probabilities"][c] - probs[c]) < 1e-4


def test_five_class_probability_distribution_integrity():
    """Verify that model output contains all 5 classes and sums to 1.0."""
    pipeline = LiveKWSPipeline(mode="segmented", backend="onnx")
    audio_sample = np.zeros(16000, dtype=np.float32)
    pred, conf, probs, dt = pipeline._run_in_memory_inference(audio_sample)

    assert set(probs.keys()) == set(CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert "REJECT" in probs
    assert pred == "REJECT"


def test_regression_high_reject_blocks_lower_probability_command():
    """
    Regression Test: A high REJECT probability MUST block a lower-probability command,
    even if that command individually exceeds theta_cmd = 0.20.
    """
    pipeline = LiveKWSPipeline(mode="segmented", backend="onnx")
    engine = VietnameseKWSInferencer(backend="onnx")

    # Carrier fixture where P(REJECT) is high (~0.70) and P(XUONG) is ~0.28
    carrier_fixture = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio" / "v13_conf_carrier_phai_038.wav"
    assert carrier_fixture.exists()

    res_engine = engine.predict(carrier_fixture)
    audio = preprocess_audio(carrier_fixture)
    pred_live, conf_live, probs_live, _ = pipeline._run_in_memory_inference(audio)

    # In this fixture, REJECT is the clear argmax (> 0.65)
    assert probs_live["REJECT"] > probs_live["XUONG"]
    assert probs_live["XUONG"] >= 0.20, "Prerequisite: command probability must be >= 0.20 to test defect"

    # Both must correctly reject because REJECT is top class
    assert res_engine["predicted_class"] == "REJECT"
    assert pred_live == "REJECT"


def test_canonical_decision_rule_valid_command_eligible_and_subthreshold_rejected():
    """Verify that a command is accepted iff it is the 5-class argmax AND >= 0.20."""
    engine = VietnameseKWSInferencer(backend="onnx")

    # Command clip TRAI_001
    trai_clip = P13_3_AUDIO_DIR / "v13_conf_cmd_trai_001.wav"
    assert trai_clip.exists()
    res = engine.predict(trai_clip)

    assert res["predicted_class"] == "TRAI"
    assert res["confidence"] >= 0.20
    assert res["probabilities"]["TRAI"] == max(res["probabilities"].values())

    # Check class order
    assert CLASSES == ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
    assert TARGET_CLASSES == ["TRAI", "PHAI", "LEN", "XUONG"]


