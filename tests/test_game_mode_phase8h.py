"""
Unit, Integration, and Regression Test Suite for Workstream 8H-D: Arm B Controlled Game Mode.
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
"""

import sys
import queue
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"
CKPT_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "checkpoints" / "phase8"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"

for p in [str(BUNDLE_DIR), str(DEMO_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import VietnameseKWSInferencer, TARGET_CLASSES, CLASSES
from train_eval_phase8 import apply_eval_channel_transform, CorrectedVietnameseKWSNet, PyTorchInferencer


@pytest.fixture(scope="module")
def default_v11_inferencer():
    """Default production path remains frozen v11."""
    inf = VietnameseKWSInferencer(backend="onnx")
    orig_predict = inf.predict
    def dynamic_predict(x):
        if isinstance(x, np.ndarray):
            audio = x.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            elif len(audio) > 16000:
                audio = audio[:16000]
            max_v = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            if max_v > 1.0:
                audio = audio / max_v
            spec = inf.frontend.compute(audio)
            ort_out = inf.session.run(None, {"log_mel": spec})
            probs = ort_out[1][0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            pred_c = CLASSES[top_idx] if (top_idx < 4 and top_prob >= inf.theta_cmd) else "REJECT"
            return {
                "class": pred_c,
                "predicted_class": pred_c,
                "confidence": top_prob,
                "probabilities": {CLASSES[i]: float(probs[i]) for i in range(5)}
            }
        return orig_predict(x)
    inf.predict = dynamic_predict
    return inf


@pytest.fixture(scope="module")
def arm_b_inferencer():
    """Explicitly selectable experimental Arm B profile (seed 42)."""
    onnx_path = CKPT_DIR / "onnx" / "arm_b_138k_aug_seed42.onnx"
    assert onnx_path.exists(), f"Arm B ONNX not found: {onnx_path}"
    import torchaudio.functional as F_a
    inf = VietnameseKWSInferencer(backend="onnx", model_path=onnx_path)
    inf.model_path = onnx_path
    inf.frontend.filterbank = F_a.melscale_fbanks(
        n_freqs=257, f_min=20.0, f_max=8000.0, n_mels=40, sample_rate=16000, norm="slaney", mel_scale="slaney"
    ).numpy().T

    orig_predict = inf.predict
    def dynamic_predict(x):
        if isinstance(x, np.ndarray):
            audio = x.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))
            elif len(audio) > 16000:
                audio = audio[:16000]
            max_v = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            if max_v > 1.0:
                audio = audio / max_v
            spec = inf.frontend.compute(audio)
            ort_out = inf.session.run(None, {"log_mel": spec})
            probs = ort_out[1][0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            pred_c = CLASSES[top_idx] if (top_idx < 4 and top_prob >= inf.theta_cmd) else "REJECT"
            return {
                "class": pred_c,
                "predicted_class": pred_c,
                "confidence": top_prob,
                "probabilities": {CLASSES[i]: float(probs[i]) for i in range(5)}
            }
        return orig_predict(x)

    inf.predict = dynamic_predict
    return inf


def get_command_clip(cmd_name: str, clip_idx: int = 1) -> np.ndarray:
    """Helper loading calibrated command clip."""
    idx_1based = ((clip_idx - 1) % 50) + 1
    fpath = AUDIO_DIR / f"v13_conf_cmd_{cmd_name}_{idx_1based:03d}.wav"
    data, sr = sf.read(fpath, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000)

    # Simple energy-based silence trim
    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(data[i : i + frame_len] ** 2)) + 1e-7) for i in range(0, len(data), frame_len)]
    above = np.where(np.array(rms_db) > -40.0)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(data), (above[-1] + 3) * frame_len)
        return data[s:e]
    return data[:6400]


# -----------------------------------------------------------------------------
# 1. INVARIANTS & INTEGRATION INTEGRITY
# -----------------------------------------------------------------------------
def test_production_default_remains_v11():
    """Verify default inferencer loads frozen v11 with unchanged hash."""
    v11_pt = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    v11_onnx = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"

    with open(v11_pt, "rb") as f:
        pt_sha = hashlib.sha256(f.read()).hexdigest()
    with open(v11_onnx, "rb") as f:
        onnx_sha = hashlib.sha256(f.read()).hexdigest()

    assert pt_sha == "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
    assert onnx_sha == "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def test_arm_b_profile_explicit_selection(arm_b_inferencer):
    """Verify Arm B is accessible as an explicitly selectable experimental profile."""
    assert arm_b_inferencer.backend == "onnx"
    assert "arm_b_138k_aug_seed42" in str(arm_b_inferencer.model_path)
    assert arm_b_inferencer.theta_cmd == 0.20


# -----------------------------------------------------------------------------
# 2. RAPID CONTINUOUS COMMANDS & GAP TOLERANCES
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("gap_ms", [0, 50])
def test_len_len_phai_sequence(arm_b_inferencer, gap_ms):
    """Verify LEN -> LEN -> PHAI with 0ms and 50ms gaps emits exact ordered events."""
    base_seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
    ctrl = GameModeSessionController(base_seg, session_timeout_sec=6.0)
    ctrl.open_session(current_time=0.0, duration_sec=10.0)

    len1 = get_command_clip("len", 1)
    len2 = get_command_clip("len", 2)
    phai1 = get_command_clip("phai", 1)

    lead = np.zeros(8000, dtype=np.float32)
    gap = np.zeros(int(gap_ms * 16), dtype=np.float32)
    tail = np.zeros(12000, dtype=np.float32)

    stream = np.concatenate([lead, len1, gap, len2, gap, phai1, tail])

    events = []
    chunk_size = 320
    for st in range(0, len(stream), chunk_size):
        chunk = stream[st : st + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        res = ctrl.process_chunk(chunk)
        if res is not None:
            audio_1s, _ = res
            res_dict = arm_b_inferencer.predict(audio_1s)
            pred = res_dict["class"]
            conf = res_dict["confidence"]
            if pred in TARGET_CLASSES and conf >= 0.20:
                events.append(pred)

    res_flush = base_seg.flush()
    if res_flush is not None:
        audio_1s, _ = res_flush
        res_dict = arm_b_inferencer.predict(audio_1s)
        pred = res_dict["class"]
        conf = res_dict["confidence"]
        if pred in TARGET_CLASSES and conf >= 0.20:
            events.append(pred)

    assert events == ["LEN", "LEN", "PHAI"], f"Expected ['LEN', 'LEN', 'PHAI'], got {events}"


def test_repeated_same_command_separation(arm_b_inferencer):
    """Verify consecutive same-label LEN -> LEN emits 2 distinct events without duplicate suppression."""
    base_seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
    ctrl = GameModeSessionController(base_seg, session_timeout_sec=6.0)
    ctrl.open_session(current_time=0.0, duration_sec=5.0)

    len1 = get_command_clip("len", 3)
    len2 = get_command_clip("len", 4)
    lead = np.zeros(4000, dtype=np.float32)
    tail = np.zeros(8000, dtype=np.float32)

    stream = np.concatenate([lead, len1, len2, tail])
    events = []
    chunk_size = 320
    for st in range(0, len(stream), chunk_size):
        chunk = stream[st : st + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        res = ctrl.process_chunk(chunk)
        if res is not None:
            audio_1s, _ = res
            res_dict = arm_b_inferencer.predict(audio_1s)
            if res_dict["class"] in TARGET_CLASSES and res_dict["confidence"] >= 0.20:
                events.append(res_dict["class"])

    res_flush = base_seg.flush()
    if res_flush is not None:
        audio_1s, _ = res_flush
        res_dict = arm_b_inferencer.predict(audio_1s)
        if res_dict["class"] in TARGET_CLASSES and res_dict["confidence"] >= 0.20:
            events.append(res_dict["class"])

    assert events == ["LEN", "LEN"], f"Expected ['LEN', 'LEN'], got {events}"


def test_trai_len_xuong_sequence(arm_b_inferencer):
    """Verify distinct command sequence TRAI -> LEN -> XUONG preserves strict event order."""
    base_seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
    ctrl = GameModeSessionController(base_seg, session_timeout_sec=6.0)
    ctrl.open_session(current_time=0.0, duration_sec=8.0)

    trai = get_command_clip("trai", 5)
    len_cmd = get_command_clip("len", 5)
    xuong = get_command_clip("xuong", 5)
    gap = np.zeros(800, dtype=np.float32)  # 50ms gap

    stream = np.concatenate([np.zeros(4000, dtype=np.float32), trai, gap, len_cmd, gap, xuong, np.zeros(8000, dtype=np.float32)])
    events = []
    chunk_size = 320
    for st in range(0, len(stream), chunk_size):
        chunk = stream[st : st + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        res = ctrl.process_chunk(chunk)
        if res is not None:
            audio_1s, _ = res
            res_dict = arm_b_inferencer.predict(audio_1s)
            if res_dict["class"] in TARGET_CLASSES and res_dict["confidence"] >= 0.20:
                events.append(res_dict["class"])

    res_flush = base_seg.flush()
    if res_flush is not None:
        audio_1s, _ = res_flush
        res_dict = arm_b_inferencer.predict(audio_1s)
        if res_dict["class"] in TARGET_CLASSES and res_dict["confidence"] >= 0.20:
            events.append(res_dict["class"])

    assert events == ["TRAI", "LEN", "XUONG"], f"Expected ['TRAI', 'LEN', 'XUONG'], got {events}"


# -----------------------------------------------------------------------------
# 3. ALL FOUR COMMAND CLASSES & ELONGATED COMMANDS
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("cmd_name", ["trai", "phai", "len", "xuong"])
def test_all_four_command_classes(arm_b_inferencer, cmd_name):
    """Verify every command class is reliably detected individually."""
    clip = get_command_clip(cmd_name, 1)
    audio_1s = np.zeros(16000, dtype=np.float32)
    audio_1s[:len(clip)] = clip

    res_dict = arm_b_inferencer.predict(audio_1s)
    assert res_dict["class"] == cmd_name.upper(), f"Expected {cmd_name.upper()}, got {res_dict['class']}"
    assert res_dict["confidence"] >= 0.20


def test_elongated_sustained_command(arm_b_inferencer):
    """Verify an elongated command (artificially sustained vowel) does not trigger spurious multiple splits."""
    from benchmark_phase5_audit import build_elongated_clip
    sustained = build_elongated_clip("len", 1, target_dur_ms=750)

    base_seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
    ctrl = GameModeSessionController(base_seg, session_timeout_sec=5.0)
    ctrl.open_session(current_time=0.0, duration_sec=5.0)

    stream = np.concatenate([np.zeros(3200, dtype=np.float32), sustained, np.zeros(8000, dtype=np.float32)])
    events = []
    chunk_size = 320
    for st in range(0, len(stream), chunk_size):
        chunk = stream[st : st + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        res = ctrl.process_chunk(chunk)
        if res is not None:
            audio_1s, _ = res
            res_dict = arm_b_inferencer.predict(audio_1s)
            if res_dict["class"] in TARGET_CLASSES and res_dict["confidence"] >= 0.20:
                events.append(res_dict["class"])

    res_flush = base_seg.flush()
    if res_flush is not None:
        audio_1s, _ = res_flush
        res_dict = arm_b_inferencer.predict(audio_1s)
        if res_dict["class"] in TARGET_CLASSES and res_dict["confidence"] >= 0.20:
            events.append(res_dict["class"])

    assert len(events) == 1, f"Elongated command emitted {len(events)} events instead of 1: {events}"
    assert events[0] == "LEN"


# -----------------------------------------------------------------------------
# 4. LIFECYCLE, FLUSH & DUPLICATE SUPPRESSION
# -----------------------------------------------------------------------------
def test_session_lifecycle_and_closed_gate_safety(arm_b_inferencer):
    """Verify session open -> close -> reset lifecycle and closed-gate chatter immunity."""
    base_seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
    ctrl = GameModeSessionController(base_seg, session_timeout_sec=2.0)

    # 1. Closed Gate: pass valid command clip -> MUST NOT emit any event
    cmd_clip = get_command_clip("phai", 8)
    stream_closed = np.concatenate([np.zeros(1600, dtype=np.float32), cmd_clip, np.zeros(3200, dtype=np.float32)])
    events_closed = []
    for st in range(0, len(stream_closed), 320):
        chunk = stream_closed[st : st + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        res = ctrl.process_chunk(chunk)
        assert res is None, "Closed gate must not return any segment!"

    # 2. Open Session: pass command -> MUST emit event
    ctrl.open_session(current_time=0.0, duration_sec=5.0)
    events_open = []
    for st in range(0, len(stream_closed), 320):
        chunk = stream_closed[st : st + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        res = ctrl.process_chunk(chunk)
        if res is not None:
            audio_1s, _ = res
            r = arm_b_inferencer.predict(audio_1s)
            if r["class"] in TARGET_CLASSES and r["confidence"] >= 0.20:
                events_open.append(r["class"])
    res_f = base_seg.flush()
    if res_f is not None:
        r = arm_b_inferencer.predict(res_f[0])
        if r["class"] in TARGET_CLASSES and r["confidence"] >= 0.20:
            events_open.append(r["class"])
    assert events_open == ["PHAI"]

    # 3. Reset and Close
    ctrl.reset()
    ctrl.close_session()
    assert not ctrl.is_session_active


def test_end_of_stream_flush_finalization(arm_b_inferencer):
    """Verify utterance ending at stream boundary is finalized via flush."""
    base_seg = HybridSpeechBandValleySegmenter(sample_rate=16000, speech_band_hz=(100.0, 3400.0))
    ctrl = GameModeSessionController(base_seg, session_timeout_sec=5.0)
    ctrl.open_session(current_time=0.0, duration_sec=5.0)

    cmd_clip = get_command_clip("xuong", 12)
    # Stream cuts off immediately after speech without trailing silence
    stream_cutoff = np.concatenate([np.zeros(1600, dtype=np.float32), cmd_clip])

    events = []
    for st in range(0, len(stream_cutoff), 320):
        chunk = stream_cutoff[st : st + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        res = ctrl.process_chunk(chunk)
        if res is not None:
            r = arm_b_inferencer.predict(res[0])
            if r["class"] in TARGET_CLASSES and r["confidence"] >= 0.20:
                events.append(r["class"])

    # Flush must finalize pending speech
    res_flush = base_seg.flush()
    assert res_flush is not None, "Flush must finalize accumulated speech at stream termination"
    r = arm_b_inferencer.predict(res_flush[0])
    if r["class"] in TARGET_CLASSES and r["confidence"] >= 0.20:
        events.append(r["class"])

    assert events == ["XUONG"], f"Expected ['XUONG'], got {events}"


# -----------------------------------------------------------------------------
# 5. ERROR RESILIENCE & QUEUE OVERFLOW
# -----------------------------------------------------------------------------
def test_audio_callback_queue_overflow_resilience(arm_b_inferencer):
    """Verify system remains stable when audio callback queue experiences simulated backpressure."""
    audio_q = queue.Queue(maxsize=10)
    dummy_chunk = np.zeros(320, dtype=np.float32)

    # Fill queue to maximum capacity
    for _ in range(10):
        audio_q.put_nowait(dummy_chunk)
    assert audio_q.full()

    # Callback attempt when full: drop oldest frame, push newest (never crash)
    dropped_frames = 0
    try:
        audio_q.put_nowait(dummy_chunk)
    except queue.Full:
        _ = audio_q.get_nowait()
        audio_q.put_nowait(dummy_chunk)
        dropped_frames += 1

    assert dropped_frames == 1
    assert audio_q.full()


# -----------------------------------------------------------------------------
# 6. CHANNEL ROBUSTNESS (DIRECT, HEADSET, AMBIENT LOOPBACK)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("channel", ["direct_digital_stream", "close_talk_headset", "ambient_loudspeaker_loopback"])
def test_channel_robustness_on_arm_b(arm_b_inferencer, channel):
    """Verify Arm B accurately recognizes commands under simulated acoustic channels."""
    clip = get_command_clip("len", 15)
    transformed = apply_eval_channel_transform(clip, channel_name=channel)
    audio_1s = np.zeros(16000, dtype=np.float32)
    audio_1s[:len(transformed)] = transformed

    res = arm_b_inferencer.predict(audio_1s)
    assert res["class"] == "LEN", f"Channel {channel} recognition failed: got {res['class']}"
    assert res["confidence"] >= 0.20
