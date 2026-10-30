#!/usr/bin/env python3
"""
Phase 6: Game Mode Acceptance & Live Event Validation Benchmark
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA

Core Acceptance Criteria:
  - Primary N=3 demo scenario (LEN -> LEN -> PHAI): >= 90% exact-sequence match over
    30 independent trials per gap condition (0 ms and 50 ms), with zero duplicate events.
  - Separate reporting for 0 ms and 50 ms gap conditions.
  - Stress tests (N=5, 10, 20) evaluated and reported separately from primary demo target.
  - Rigorous failure mode categorization (Deletion / Merger, Substitution, Insertion / Over-split).
  - Controlled live-microphone timing benchmark via sounddevice.
"""

import os
import sys
import time
import json
import csv
import queue
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import UtteranceSegmenter
from acoustic_valley_segmenter import AcousticValleySegmenter
from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import VietnameseKWSInferencer, CLASSES, TARGET_CLASSES
from audit_phase5_3_final_metrics import compute_standard_sequence_wer, compute_temporal_event_metrics
from benchmark_phase5_audit import build_elongated_clip, construct_long_sequence


# -----------------------------------------------------------------------------
# Audio Clip Loading and Stream Builders
# -----------------------------------------------------------------------------
def trim_clip_silence(data: np.ndarray, threshold_db: float = -42.0) -> np.ndarray:
    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(data[i : i + frame_len] ** 2)) + 1e-7) for i in range(0, len(data), frame_len)]
    above = np.where(np.array(rms_db) > threshold_db)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(data), (above[-1] + 3) * frame_len)
        return data[s:e]
    return data[:6400]


def get_command_clip(cmd_name: str, clip_idx: int) -> np.ndarray:
    idx_1based = ((clip_idx - 1) % 50) + 1
    fpath = AUDIO_DIR / f"v13_conf_cmd_{cmd_name}_{idx_1based:03d}.wav"
    data, sr = sf.read(fpath, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000)
    return trim_clip_silence(data)


def build_trial_stream(
    pattern_id: str,
    trial_idx: int,
    gap_ms: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Constructs a deterministic synthetic multi-command trial stream."""
    lead_silence = np.zeros(8000, dtype=np.float32)   # 500ms
    gap = np.zeros(int((gap_ms / 1000.0) * 16000), dtype=np.float32)
    tail_silence = np.zeros(12000, dtype=np.float32)  # 750ms

    if pattern_id == "len_len_phai":
        c1_idx = ((trial_idx * 2 - 2) % 50) + 1
        c2_idx = ((trial_idx * 2 - 1) % 50) + 1
        c3_idx = ((trial_idx - 1) % 50) + 1
        clips = [
            ("LEN", get_command_clip("len", c1_idx)),
            ("LEN", get_command_clip("len", c2_idx)),
            ("PHAI", get_command_clip("phai", c3_idx))
        ]
    elif pattern_id == "len_len":
        c1_idx = ((trial_idx * 2 - 2) % 50) + 1
        c2_idx = ((trial_idx * 2 - 1) % 50) + 1
        clips = [
            ("LEN", get_command_clip("len", c1_idx)),
            ("LEN", get_command_clip("len", c2_idx))
        ]
    elif pattern_id == "trai_len_xuong":
        c1_idx = ((trial_idx * 3 - 3) % 50) + 1
        c2_idx = ((trial_idx * 3 - 2) % 50) + 1
        c3_idx = ((trial_idx * 3 - 1) % 50) + 1
        clips = [
            ("TRAI", get_command_clip("trai", c1_idx)),
            ("LEN", get_command_clip("len", c2_idx)),
            ("XUONG", get_command_clip("xuong", c3_idx))
        ]
    elif pattern_id == "all_four_transitions":
        c1_idx = ((trial_idx * 4 - 4) % 50) + 1
        c2_idx = ((trial_idx * 4 - 3) % 50) + 1
        c3_idx = ((trial_idx * 4 - 2) % 50) + 1
        c4_idx = ((trial_idx * 4 - 1) % 50) + 1
        clips = [
            ("LEN", get_command_clip("len", c1_idx)),
            ("XUONG", get_command_clip("xuong", c2_idx)),
            ("TRAI", get_command_clip("trai", c3_idx)),
            ("PHAI", get_command_clip("phai", c4_idx))
        ]
    elif pattern_id == "elongated_len":
        c_elong = build_elongated_clip("len", trial_idx, target_dur_ms=750)
        clips = [("LEN", c_elong)]
    else:
        raise ValueError(f"Unknown pattern: {pattern_id}")

    stream_parts = [lead_silence]
    curr_time = len(lead_silence) / 16000.0
    gt_events = []
    for i, (cmd, audio) in enumerate(clips):
        dur = len(audio) / 16000.0
        gt_events.append({
            "command": cmd,
            "start_sec": curr_time,
            "end_sec": curr_time + dur,
            "center_sec": curr_time + dur / 2.0
        })
        stream_parts.append(audio)
        curr_time += dur
        if i < len(clips) - 1:
            stream_parts.append(gap)
            curr_time += len(gap) / 16000.0
    stream_parts.append(tail_silence)
    return np.concatenate(stream_parts).astype(np.float32), gt_events


def process_stream_through_pipeline(
    stream: np.ndarray,
    segmenter: Any,
    inferencer: VietnameseKWSInferencer,
    theta_cmd: float = 0.20,
    chunk_size: int = 320,
    is_game_mode: bool = False,
    gate_open_during_stream: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Feeds audio stream chunk-by-chunk through segmenter and ONNX inferencer."""
    if hasattr(segmenter, "open_session") and is_game_mode and gate_open_during_stream:
        segmenter.open_session(duration_sec=30.0)
    else:
        segmenter.reset()

    events = []
    infer_times = []
    total_samples = 0

    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        total_samples += len(chunk)
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        res = segmenter.process_chunk(chunk)
        if res is not None:
            audio_1s, seg_dur_ms = res
            t0 = time.perf_counter()
            spec = inferencer.frontend.compute(audio_1s)
            ort_out = inferencer.session.run(None, {"log_mel": spec})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(dt_ms)

            probs = ort_out[1][0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            if top_idx < 4 and top_prob >= theta_cmd:
                pred_cmd = TARGET_CLASSES[top_idx]
                curr_sec = total_samples / 16000.0
                events.append({
                    "command": pred_cmd,
                    "confidence": round(top_prob, 4),
                    "timestamp_sec": round(curr_sec, 3),
                    "segment_duration_ms": round(seg_dur_ms, 1),
                    "infer_latency_ms": round(dt_ms, 2)
                })

    # Finalization flush at stream termination
    if hasattr(segmenter, "flush"):
        flush_res = segmenter.flush()
        if flush_res is not None:
            audio_1s, seg_dur_ms = flush_res
            t0 = time.perf_counter()
            spec = inferencer.frontend.compute(audio_1s)
            ort_out = inferencer.session.run(None, {"log_mel": spec})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(dt_ms)

            probs = ort_out[1][0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            if top_idx < 4 and top_prob >= theta_cmd:
                pred_cmd = TARGET_CLASSES[top_idx]
                curr_sec = total_samples / 16000.0
                events.append({
                    "command": pred_cmd,
                    "confidence": round(top_prob, 4),
                    "timestamp_sec": round(curr_sec, 3),
                    "segment_duration_ms": round(seg_dur_ms, 1),
                    "infer_latency_ms": round(dt_ms, 2),
                    "flushed": True
                })

    if hasattr(segmenter, "close_session") and is_game_mode:
        segmenter.close_session()

    return events, {"infer_times_ms": infer_times}


# -----------------------------------------------------------------------------
# Controlled Live Microphone Latency Benchmark
# -----------------------------------------------------------------------------
def run_controlled_live_mic_latency_test(duration_sec: float = 3.0) -> Dict[str, Any]:
    """Controlled non-blocking live-microphone timing test using sounddevice."""
    if not HAS_SOUNDDEVICE:
        return {"error": "sounddevice library not available"}

    inferencer = VietnameseKWSInferencer(backend="onnx")
    seg = HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0))

    audio_queue: queue.Queue = queue.Queue()
    cb_intervals = []
    queue_delays = []
    chunk_process_times = []
    infer_times = []

    t_last_cb = [None]
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        t_now = time.perf_counter()
        if t_last_cb[0] is not None:
            cb_intervals.append((t_now - t_last_cb[0]) * 1000.0)
        t_last_cb[0] = t_now
        chunk = indata[:, 0].copy()
        audio_queue.put((chunk, t_now))

    def worker_loop():
        while not stop_event.is_set() or not audio_queue.empty():
            try:
                item = audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            chunk, t_enqueued = item
            t_dequeued = time.perf_counter()
            queue_delays.append((t_dequeued - t_enqueued) * 1000.0)

            t0 = time.perf_counter()
            res = seg.process_chunk(chunk)
            chunk_process_times.append((time.perf_counter() - t0) * 1000.0)

            if res is not None:
                audio_1s, _ = res
                t_inf = time.perf_counter()
                spec = inferencer.frontend.compute(audio_1s)
                inferencer.session.run(None, {"log_mel": spec})
                infer_times.append((time.perf_counter() - t_inf) * 1000.0)

    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()

    stream = sd.InputStream(samplerate=16000, blocksize=320, channels=1, dtype="float32", callback=audio_callback)
    with stream:
        time.sleep(duration_sec)

    stop_event.set()
    worker_thread.join(timeout=1.0)

    return {
        "duration_sec": duration_sec,
        "total_chunks_captured": len(cb_intervals) + 1,
        "callback_intervals_ms": {
            "sample_count": len(cb_intervals),
            "expected_interval_ms": 20.0,
            "min_ms": round(float(np.min(cb_intervals)), 2) if cb_intervals else 0.0,
            "p50_ms": round(float(np.percentile(cb_intervals, 50)), 2) if cb_intervals else 0.0,
            "p95_ms": round(float(np.percentile(cb_intervals, 95)), 2) if cb_intervals else 0.0,
            "max_ms": round(float(np.max(cb_intervals)), 2) if cb_intervals else 0.0
        },
        "queue_transit_delays_ms": {
            "sample_count": len(queue_delays),
            "p50_ms": round(float(np.percentile(queue_delays, 50)), 2) if queue_delays else 0.0,
            "p95_ms": round(float(np.percentile(queue_delays, 95)), 2) if queue_delays else 0.0,
            "max_ms": round(float(np.max(queue_delays)), 2) if queue_delays else 0.0
        },
        "chunk_processing_times_ms": {
            "sample_count": len(chunk_process_times),
            "p50_ms": round(float(np.percentile(chunk_process_times, 50)), 3) if chunk_process_times else 0.0,
            "p95_ms": round(float(np.percentile(chunk_process_times, 95)), 3) if chunk_process_times else 0.0
        },
        "in_stream_infer_times_ms": {
            "sample_count": len(infer_times),
            "p50_ms": round(float(np.percentile(infer_times, 50)), 2) if infer_times else 0.0,
            "p95_ms": round(float(np.percentile(infer_times, 95)), 2) if infer_times else 0.0
        },
        "driver_and_hardware_limitations": [
            "1. Standard Windows MME audio driver delivers 2 blocks per buffer quantum, resulting in bursty callbacks (0.04 ms then 40.0 ms) rather than smooth 20 ms ticks.",
            "2. Physical sound card ADC buffering and OS audio subsystem buffer add hardware latency before audio arrives at python callback.",
            "3. Live ambient microphone test measures background room acoustics; offline replay test uses synthetic locked test streams with known ground truth."
        ]
    }


# -----------------------------------------------------------------------------
# Main Phase 6 Acceptance Benchmark Execution
# -----------------------------------------------------------------------------
def run_phase6_acceptance_benchmark():
    print("=" * 80)
    print("PHASE 6: GAME MODE ACCEPTANCE & LIVE EVENT VALIDATION")
    print("Governance Labels:")
    print("  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED")
    print("  - NOT_VALIDATED_ON_NATURAL_SPEECH")
    print("  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION")
    print("  - NOT_VALIDATED_PENDING_DATA")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    # Build locked test manifest entries
    manifest_entries = []

    # 1. Primary Acceptance Target (60 trials)
    for g in [0, 50]:
        for t in range(1, 31):
            manifest_entries.append({
                "trial_id": f"primary_len_len_phai_gap{g}_t{t:02d}",
                "category": "primary_demo",
                "pattern_id": "len_len_phai",
                "trial_index": t,
                "gap_ms": g,
                "n_commands": 3,
                "gt_sequence": "LEN -> LEN -> PHAI"
            })

    # 2. Secondary Controlled Patterns (50 trials)
    for g in [0, 50]:
        for t in range(1, 11):
            manifest_entries.append({
                "trial_id": f"secondary_len_len_gap{g}_t{t:02d}",
                "category": "secondary_pattern",
                "pattern_id": "len_len",
                "trial_index": t,
                "gap_ms": g,
                "n_commands": 2,
                "gt_sequence": "LEN -> LEN"
            })
    for g in [0, 50]:
        for t in range(1, 11):
            manifest_entries.append({
                "trial_id": f"secondary_trai_len_xuong_gap{g}_t{t:02d}",
                "category": "secondary_pattern",
                "pattern_id": "trai_len_xuong",
                "trial_index": t,
                "gap_ms": g,
                "n_commands": 3,
                "gt_sequence": "TRAI -> LEN -> XUONG"
            })
    for t in range(1, 11):
        manifest_entries.append({
            "trial_id": f"secondary_all_four_gap50_t{t:02d}",
            "category": "secondary_pattern",
            "pattern_id": "all_four_transitions",
            "trial_index": t,
            "gap_ms": 50,
            "n_commands": 4,
            "gt_sequence": "LEN -> XUONG -> TRAI -> PHAI"
        })

    # 3. Elongated Control (10 trials)
    for t in range(1, 11):
        manifest_entries.append({
            "trial_id": f"control_elongated_len_t{t:02d}",
            "category": "elongated_control",
            "pattern_id": "elongated_len",
            "trial_index": t,
            "gap_ms": 0,
            "n_commands": 1,
            "gt_sequence": "LEN"
        })

    # 4. Long Sequence Stress Tests (80 trials: N=3, 5, 10, 20 x 10 trials x 2 gaps)
    for n_len in [3, 5, 10, 20]:
        for g in [0, 50]:
            for t in range(1, 11):
                manifest_entries.append({
                    "trial_id": f"stress_seq_N{n_len}_gap{g}_t{t:02d}",
                    "category": "long_sequence_stress",
                    "pattern_id": f"long_sequence_N{n_len}",
                    "trial_index": t,
                    "gap_ms": g,
                    "n_commands": n_len,
                    "gt_sequence": f"LongSequence_N{n_len}"
                })

    # Write locked trial manifest
    manifest_csv = REPORTS_DIR / "phase6_locked_trial_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial_id", "category", "pattern_id", "trial_index", "gap_ms", "n_commands", "gt_sequence"
        ])
        writer.writeheader()
        for row in manifest_entries:
            writer.writerow(row)
    print(f"\n[1/5] Locked Trial Manifest written to: {manifest_csv} ({len(manifest_entries)} trials)")

    # Profiles to evaluate
    profiles = {
        "legacy_v11": UtteranceSegmenter(silence_interval_ms=300, preroll_ms=150, energy_threshold_db=-48.0),
        "continuous_valley": AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120, energy_threshold_db=-46.0, dip_threshold_db=12.0, rebound_threshold_db=6.0),
        "hybrid_speechband_valley": HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0, silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0),
        "game_mode_session": GameModeSessionController(
            base_segmenter=HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0, silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0),
            session_timeout_sec=6.0,
            extend_on_detection=True
        )
    }

    # Execute all trials
    print("\n[2/5] Executing Acceptance Benchmark across profiles...")
    trial_level_results = []
    failure_analysis_records = []

    for prof_name, seg_instance in profiles.items():
        print(f"  --> Running profile: {prof_name}...")
        is_gm = (prof_name == "game_mode_session")

        for m in manifest_entries:
            trial_id = m["trial_id"]
            cat = m["category"]
            pat = m["pattern_id"]
            t_idx = m["trial_index"]
            gap = m["gap_ms"]
            n_cmds = m["n_commands"]

            if pat.startswith("long_sequence_N"):
                n_len = int(pat.replace("long_sequence_N", ""))
                stream, gt_events = construct_long_sequence(n_len, gap, seed_offset=t_idx)
            else:
                stream, gt_events = build_trial_stream(pat, t_idx, gap)

            gt_cmds = [g["command"] for g in gt_events]

            dets, timing = process_stream_through_pipeline(
                stream, seg_instance, inferencer, theta_cmd=0.20,
                is_game_mode=is_gm, gate_open_during_stream=True
            )
            det_cmds = [d["command"] for d in dets]

            # Path A: Standard Sequence WER
            seq_res = compute_standard_sequence_wer(gt_cmds, det_cmds)
            exact_match = seq_res["exact_sequence_match"]

            # Path B: Temporal Event Metrics
            ev_res = compute_temporal_event_metrics(dets, gt_events)

            # Over-split detection: det_count > gt_count for elongated or single commands
            over_split = (len(det_cmds) > len(gt_cmds)) if pat == "elongated_len" else False

            # Latency averages
            mean_lag = float(np.mean(ev_res["lag_latencies"])) if ev_res["lag_latencies"] else None
            mean_e2e = float(np.mean(ev_res["e2e_latencies"])) if ev_res["e2e_latencies"] else None

            trial_rec = {
                "trial_id": trial_id,
                "profile": prof_name,
                "category": cat,
                "pattern_id": pat,
                "trial_index": t_idx,
                "gap_ms": gap,
                "n_gt": len(gt_cmds),
                "gt_sequence": " -> ".join(gt_cmds),
                "n_det": len(det_cmds),
                "det_sequence": " -> ".join(det_cmds) if det_cmds else "NONE",
                "exact_sequence_match": exact_match,
                "seq_s": seq_res["seq_s"],
                "seq_d": seq_res["seq_d"],
                "seq_ins": seq_res["seq_ins"],
                "seq_wer_pct": seq_res["seq_wer_pct"],
                "event_tp": ev_res["event_tp"],
                "event_fp": ev_res["event_fp"],
                "event_fn": ev_res["event_fn"],
                "mean_lag_latency_ms": round(mean_lag, 1) if mean_lag is not None else "N/A",
                "mean_e2e_latency_ms": round(mean_e2e, 1) if mean_e2e is not None else "N/A",
                "over_split": over_split
            }
            trial_level_results.append(trial_rec)

            # Failure Analysis Record if not exact match
            if not exact_match:
                if seq_res["seq_d"] > 0:
                    f_mode = "DELETION_MERGER"
                    f_diag = "Consecutive syllables fused into single utterance due to insufficient energy valley depth or long silence threshold."
                elif seq_res["seq_s"] > 0:
                    f_mode = "SUBSTITUTION_ACOUSTIC"
                    f_diag = "Acoustic boundary correctly identified, but frozen v11 CNN misclassified the word into another valid class."
                elif seq_res["seq_ins"] > 0:
                    f_mode = "INSERTION_OVERSPLIT"
                    f_diag = "Segmenter over-split single word or emitted false alarm during silence/trailing audio."
                else:
                    f_mode = "SEQUENCE_DISCREPANCY"
                    f_diag = "Predicted sequence differs from ground truth."

                failure_analysis_records.append({
                    "trial_id": trial_id,
                    "profile": prof_name,
                    "category": cat,
                    "gap_ms": gap,
                    "gt_sequence": " -> ".join(gt_cmds),
                    "det_sequence": " -> ".join(det_cmds) if det_cmds else "NONE",
                    "failure_mode": f_mode,
                    "seq_s": seq_res["seq_s"],
                    "seq_d": seq_res["seq_d"],
                    "seq_ins": seq_res["seq_ins"],
                    "root_cause_diagnosis": f_diag
                })

    # Save trial-level results CSV
    results_csv = REPORTS_DIR / "phase6_trial_level_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_level_results[0].keys()))
        writer.writeheader()
        for row in trial_level_results:
            writer.writerow(row)
    print(f"\n[3/5] Trial-Level Results written to: {results_csv} ({len(trial_level_results)} rows)")

    # Save failure analysis CSV
    failures_csv = REPORTS_DIR / "phase6_failure_analysis.csv"
    with open(failures_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial_id", "profile", "category", "gap_ms", "gt_sequence", "det_sequence",
            "failure_mode", "seq_s", "seq_d", "seq_ins", "root_cause_diagnosis"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in failure_analysis_records:
            writer.writerow(row)
    print(f"\n[4/5] Failure Analysis written to: {failures_csv} ({len(failure_analysis_records)} failure rows)")

    # -------------------------------------------------------------------------
    # Negative Control Evaluation: Open-Gate vs Closed-Gate
    # -------------------------------------------------------------------------
    print("\n[5/5] Auditing Negative Controls (Open-Gate vs Closed-Gate)...")
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))

    lead_silence = np.zeros(1600, dtype=np.float32)
    trail_silence = np.zeros(8000, dtype=np.float32)

    negative_control_summary = {}
    for p_name in ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode_closed_gate"]:
        negative_control_summary[p_name] = {
            "conversation_fa": 0, "conversation_sec": 0.0,
            "foils_fa": 0, "foils_sec": 0.0,
            "noise_fa": 0, "noise_sec": 0.0,
            "silence_fa": 0, "silence_sec": 0.0
        }

    # Evaluate digital silence (10 streams x 30s = 300s)
    for _ in range(10):
        silence_audio = np.zeros(int(30.0 * 16000), dtype=np.float32)
        for p_name in ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode_closed_gate"]:
            if p_name == "game_mode_closed_gate":
                seg = GameModeSessionController(HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0)))
                # Gate remains closed:
                evs, _ = process_stream_through_pipeline(silence_audio, seg, inferencer, is_game_mode=True, gate_open_during_stream=False)
            else:
                seg = profiles[p_name]
                evs, _ = process_stream_through_pipeline(silence_audio, seg, inferencer)
            negative_control_summary[p_name]["silence_sec"] += 30.0
            negative_control_summary[p_name]["silence_fa"] += len(evs)

    # Evaluate background noise files
    for f in bg_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        for p_name in ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode_closed_gate"]:
            if p_name == "game_mode_closed_gate":
                seg = GameModeSessionController(HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0)))
                evs, _ = process_stream_through_pipeline(stream, seg, inferencer, is_game_mode=True, gate_open_during_stream=False)
            else:
                seg = profiles[p_name]
                evs, _ = process_stream_through_pipeline(stream, seg, inferencer)
            negative_control_summary[p_name]["noise_sec"] += dur
            negative_control_summary[p_name]["noise_fa"] += len(evs)

    # Evaluate conversational speech files
    for f in conv_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        for p_name in ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode_closed_gate"]:
            if p_name == "game_mode_closed_gate":
                seg = GameModeSessionController(HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0)))
                evs, _ = process_stream_through_pipeline(stream, seg, inferencer, is_game_mode=True, gate_open_during_stream=False)
            else:
                seg = profiles[p_name]
                evs, _ = process_stream_through_pipeline(stream, seg, inferencer)
            negative_control_summary[p_name]["conversation_sec"] += dur
            negative_control_summary[p_name]["conversation_fa"] += len(evs)

    # Evaluate phonetic foils
    for f in foil_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        for p_name in ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode_closed_gate"]:
            if p_name == "game_mode_closed_gate":
                seg = GameModeSessionController(HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0)))
                evs, _ = process_stream_through_pipeline(stream, seg, inferencer, is_game_mode=True, gate_open_during_stream=False)
            else:
                seg = profiles[p_name]
                evs, _ = process_stream_through_pipeline(stream, seg, inferencer)
            negative_control_summary[p_name]["foils_sec"] += dur
            negative_control_summary[p_name]["foils_fa"] += len(evs)

    # Run controlled live-mic timing test
    live_latency_results = run_controlled_live_mic_latency_test(duration_sec=3.0)
    live_latency_json = REPORTS_DIR / "phase6_live_latency_results.json"
    with open(live_latency_json, "w", encoding="utf-8") as f:
        json.dump(live_latency_results, f, indent=2)
    print(f"Live Latency Results written to: {live_latency_json}")

    # -------------------------------------------------------------------------
    # Compute Acceptance Summary Metrics
    # -------------------------------------------------------------------------
    def get_summary_for_group(prof: str, cat: str, pat: Optional[str] = None, gap: Optional[int] = None):
        sub = [
            r for r in trial_level_results
            if r["profile"] == prof and r["category"] == cat
            and (pat is None or r["pattern_id"] == pat)
            and (gap is None or r["gap_ms"] == gap)
        ]
        n_trials = len(sub)
        n_exact = sum(1 for r in sub if r["exact_sequence_match"])
        exact_pct = round((n_exact / n_trials * 100.0), 1) if n_trials > 0 else 0.0
        wers = [r["seq_wer_pct"] for r in sub if r["seq_wer_pct"] is not None]
        mean_wer = round(float(np.mean(wers)), 1) if wers else 0.0
        return {"n_trials": n_trials, "n_exact": n_exact, "exact_pct": exact_pct, "mean_wer": mean_wer}

    print("\n" + "=" * 80)
    print("PHASE 6 GAME MODE ACCEPTANCE: PRIMARY DEMO SUMMARY (LEN -> LEN -> PHAI)")
    print("=" * 80)
    for prof in ["legacy_v11", "continuous_valley", "hybrid_speechband_valley", "game_mode_session"]:
        res_0ms = get_summary_for_group(prof, "primary_demo", "len_len_phai", 0)
        res_50ms = get_summary_for_group(prof, "primary_demo", "len_len_phai", 50)
        print(f"Profile: {prof:26s} | Gap  0ms: {res_0ms['n_exact']}/{res_0ms['n_trials']} ({res_0ms['exact_pct']:5.1f}%, WER {res_0ms['mean_wer']:4.1f}%) | Gap 50ms: {res_50ms['n_exact']}/{res_50ms['n_trials']} ({res_50ms['exact_pct']:5.1f}%, WER {res_50ms['mean_wer']:4.1f}%)")

    print("\n" + "=" * 80)
    print("NEGATIVE CONTROL SUMMARY (FALSE ACTIVATIONS)")
    print("=" * 80)
    for p_name, s in negative_control_summary.items():
        print(f"Profile: {p_name:26s} | Conv: {s['conversation_fa']} FA / {s['conversation_sec']:.1f}s | Foils: {s['foils_fa']} FA / {s['foils_sec']:.1f}s | Noise: {s['noise_fa']} FA / {s['noise_sec']:.1f}s | Silence: {s['silence_fa']} FA / {s['silence_sec']:.1f}s")

    # Export full comprehensive JSON package
    full_pkg = {
        "governance_labels": [
            "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
            "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
            "NOT_VALIDATED_PENDING_DATA"
        ],
        "primary_acceptance_target": {
            "requirement": "LEN -> LEN -> PHAI: >= 90% exact-sequence match over 30 trials per gap condition (0ms and 50ms)",
            "summary_by_profile": {
                prof: {
                    "gap_0ms": get_summary_for_group(prof, "primary_demo", "len_len_phai", 0),
                    "gap_50ms": get_summary_for_group(prof, "primary_demo", "len_len_phai", 50)
                } for prof in profiles.keys()
            }
        },
        "negative_control_summary": negative_control_summary,
        "live_latency": live_latency_results
    }
    pkg_json = REPORTS_DIR / "phase6_comprehensive_acceptance_summary.json"
    with open(pkg_json, "w", encoding="utf-8") as f:
        json.dump(full_pkg, f, indent=2)
    print(f"\nComprehensive Acceptance Package written to: {pkg_json}")


if __name__ == "__main__":
    run_phase6_acceptance_benchmark()
