"""
Controlled Smoke Test of Experiment E1 on Live-Microphone Pipeline.
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_e1_live_demo_smoke_test.py

Governance Boundaries:
- EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
- NOT_VALIDATED_ON_NATURAL_SPEECH
- NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
- NOT_VALIDATED_PENDING_DATA
- Zero modification to frozen v11 default demo configuration or logs
- Verification of E1 checkpoint SHA-256 prior to loading
- Zero raw audio recording/persistence
"""

import sys
import io
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import soundfile as sf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
FIXTURES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
P13_3_AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"
STAGING_OOV_DIR = V12_DIR / "audio" / "staging_oov"

E1_CKPT = V12_DIR / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt"
EXPECTED_E1_SHA256 = "f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878"

SMOKE_TEST_LOG = V12_DIR / "reports" / "e1_live_smoke_test_events.log"

for p in [str(DEMO_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import LiveKWSPipeline, DetectionEvent


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_clean_command(cmd_name: str) -> np.ndarray:
    candidates = sorted(list(P13_3_AUDIO_DIR.glob(f"v13_conf_cmd_{cmd_name}_*.wav")))
    assert len(candidates) > 0, f"No clips found for {cmd_name}"
    audio, sr = sf.read(candidates[0], dtype="float32")
    active = np.where(np.abs(audio) > 0.04)[0]
    if len(active) > 0:
        start = max(0, active[0] - 800)
        end = min(len(audio), active[-1] + 1600)
        return audio[start:end]
    return audio[:8000]


def run_smoke_test():
    print("=" * 80)
    print("  CONTROLLED E1 LIVE-MICROPHONE PIPELINE SMOKE TEST")
    print("  Governance: EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED")
    print("  Model: exp_v12_e1_baseline_scale_seed42.pt (PyTorch backend)")
    print("=" * 80)

    # 1. Verify Checkpoint SHA-256
    assert E1_CKPT.exists(), f"E1 Checkpoint not found: {E1_CKPT}"
    observed_sha = compute_sha256(E1_CKPT)
    print(f"\n[Security Check] E1 Checkpoint Hash: {observed_sha}")
    if observed_sha != EXPECTED_E1_SHA256:
        raise ValueError(f"E1 Checkpoint SHA-256 mismatch! Expected {EXPECTED_E1_SHA256}, got {observed_sha}")
    print("  >>> Cryptographic SHA-256 verification: PASSED")

    # Clear separate smoke test log
    SMOKE_TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    if SMOKE_TEST_LOG.exists():
        SMOKE_TEST_LOG.unlink()

    recorded_events: List[DetectionEvent] = []

    def on_smoke_event(ev: DetectionEvent):
        recorded_events.append(ev)
        log_entry = {
            "time_str": ev.time_str,
            "command": ev.command,
            "confidence": ev.confidence,
            "latency_ms": ev.latency_ms,
            "segment_duration_ms": ev.segment_duration_ms,
            "db_level": ev.db_level,
            "probabilities": ev.probabilities
        }
        with open(SMOKE_TEST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"  [EVENT] Detected: {ev.command:<6} | Conf: {ev.confidence:.3f} | Latency: {ev.latency_ms:.1f}ms | SegDur: {ev.segment_duration_ms:.0f}ms")

    # Initialize live pipeline with E1 checkpoint
    print("\nInitializing LiveKWSPipeline in segmented mode with E1 checkpoint...")
    pipeline = LiveKWSPipeline(
        mode="segmented",
        backend="pytorch",
        model_path=E1_CKPT,
        silence_interval_ms=300,
        energy_threshold_db=-48.0,
        event_callback=on_smoke_event
    )
    pipeline.is_running = True

    # Note: live_mic_engine.py line 602 calls `self.inferencer.frontend(x)` assuming a PyTorch nn.Module.
    # To avoid modifying production live_mic_engine.py, attach the PyTorch LogMelFrontend in this test harness:
    from eval_kws import LogMelFrontend
    pipeline.inferencer.frontend = LogMelFrontend().to(pipeline.inferencer.device)
    pipeline.inferencer.frontend.eval()

    test_results = {}

    def feed_audio_stream(audio_samples: np.ndarray, label_desc: str) -> List[DetectionEvent]:
        recorded_events.clear()
        print(f"\n--- Testing: {label_desc} ({len(audio_samples)/16000:.2f}s) ---")
        chunk_size = 320  # 20ms chunks
        for i in range(0, len(audio_samples), chunk_size):
            chunk = audio_samples[i : i + chunk_size]
            pipeline._on_new_audio_chunk(chunk)
        # Trailing flush
        silence_flush = np.zeros(int(0.40 * 16000), dtype=np.float32)
        for i in range(0, len(silence_flush), chunk_size):
            pipeline._on_new_audio_chunk(silence_flush[i : i + chunk_size])
        time.sleep(0.05)
        return list(recorded_events)

    # -------------------------------------------------------------
    # Test 1: Individual Commands
    # -------------------------------------------------------------
    silence_gap = np.zeros(int(0.35 * 16000), dtype=np.float32)
    for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
        clip = load_clean_command(cmd.lower())
        stream = np.concatenate([clip, silence_gap])
        evs = feed_audio_stream(stream, f"Single Command [{cmd}]")
        cmds = [e.command for e in evs]
        success = (len(evs) == 1 and cmds[0] == cmd)
        test_results[f"single_{cmd}"] = {
            "success": success,
            "detected": cmds,
            "confidence": evs[0].confidence if evs else 0.0
        }
        print(f"    Result: {'PASS' if success else 'FAIL'} (Detected: {cmds})")

    # -------------------------------------------------------------
    # Test 2: Multi-Command Sequence (TRAI -> XUONG -> PHAI)
    # -------------------------------------------------------------
    c_trai = load_clean_command("trai")
    c_xuong = load_clean_command("xuong")
    c_phai = load_clean_command("phai")
    stream_seq = np.concatenate([c_trai, silence_gap, c_xuong, silence_gap, c_phai, silence_gap])
    evs = feed_audio_stream(stream_seq, "Multi-Command Sequence [TRAI -> XUONG -> PHAI]")
    cmds = [e.command for e in evs]
    seq_success = (cmds == ["TRAI", "XUONG", "PHAI"])
    test_results["sequence_trai_xuong_phai"] = {
        "success": seq_success,
        "detected": cmds
    }
    print(f"    Result: {'PASS' if seq_success else 'FAIL'} (Detected: {cmds})")

    # -------------------------------------------------------------
    # Test 3: Repeated Commands (TRAI -> TRAI)
    # -------------------------------------------------------------
    stream_rep = np.concatenate([c_trai, silence_gap, c_trai, silence_gap])
    evs = feed_audio_stream(stream_rep, "Repeated Commands [TRAI -> TRAI]")
    cmds = [e.command for e in evs]
    rep_success = (cmds == ["TRAI", "TRAI"])
    test_results["repeated_trai_trai"] = {
        "success": rep_success,
        "detected": cmds
    }
    print(f"    Result: {'PASS' if rep_success else 'FAIL'} (Detected: {cmds})")

    # -------------------------------------------------------------
    # Test 4: OOV Negative Control — "Cửa" (The Bug Report Word)
    # -------------------------------------------------------------
    cua_clips = list(STAGING_OOV_DIR.glob("*c*a*.wav"))
    assert len(cua_clips) > 0, "No cua clips found"
    audio_cua, _ = sf.read(cua_clips[0], dtype="float32")
    stream_cua = np.concatenate([audio_cua, silence_gap])
    evs = feed_audio_stream(stream_cua, "OOV Negative Control ['cửa']")
    cmds = [e.command for e in evs]
    cua_success = (len(evs) == 0)  # Must be REJECTED (0 accepted commands)
    test_results["oov_control_cua"] = {
        "success": cua_success,
        "detected": cmds,
        "note": "Must produce zero accepted commands"
    }
    print(f"    Result: {'PASS (Correctly Rejected)' if cua_success else 'FAIL (False Activation)'} (Detected: {cmds})")

    # -------------------------------------------------------------
    # Test 5: OOV Negative Control — "Bàn"
    # -------------------------------------------------------------
    ban_clips = list(STAGING_OOV_DIR.glob("*b*n*.wav"))
    if ban_clips:
        audio_ban, _ = sf.read(ban_clips[0], dtype="float32")
        stream_ban = np.concatenate([audio_ban, silence_gap])
        evs = feed_audio_stream(stream_ban, "OOV Negative Control ['bàn']")
        cmds = [e.command for e in evs]
        ban_success = (len(evs) == 0)
        test_results["oov_control_ban"] = {
            "success": ban_success,
            "detected": cmds
        }
        print(f"    Result: {'PASS (Correctly Rejected)' if ban_success else 'FAIL (False Activation)'} (Detected: {cmds})")

    # -------------------------------------------------------------
    # Test 6: Silence & Ambient Noise
    # -------------------------------------------------------------
    ambient_audio, _ = sf.read(FIXTURES_DIR / "dev_02_quiet_ambient_room.wav", dtype="float32")
    evs = feed_audio_stream(ambient_audio, "Silence / Ambient Room Noise (3.0s)")
    cmds = [e.command for e in evs]
    amb_success = (len(evs) == 0)
    test_results["silence_ambient_room"] = {
        "success": amb_success,
        "detected": cmds
    }
    print(f"    Result: {'PASS (0 activations)' if amb_success else 'FAIL'} (Detected: {cmds})")

    # Save summary report
    summary_file = V12_DIR / "reports" / "e1_live_smoke_test_summary.json"
    summary_data = {
        "e1_checkpoint_path": str(E1_CKPT),
        "e1_checkpoint_sha256": observed_sha,
        "log_path": str(SMOKE_TEST_LOG),
        "test_results": test_results,
        "overall_status": "PASS" if all(r["success"] for r in test_results.values()) else "FAIL"
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  SMOKE TEST COMPLETED: {summary_data['overall_status']}")
    print(f"  Events Log: {SMOKE_TEST_LOG}")
    print(f"  Summary:    {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
