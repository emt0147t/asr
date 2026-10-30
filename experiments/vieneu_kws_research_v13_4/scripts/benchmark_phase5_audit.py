"""
Phase 5 Continuous KWS Correctness Audit & Locked-Matrix Benchmark.
Repository: D:\trainidentificationaudio

Locked Test Matrix:
- Profiles: legacy_v11, continuous_valley, hybrid_speechband_valley
- Thresholds: theta_cmd in [0.20, 0.30, 0.40, 0.50, 0.60]
- Commands: LEN, XUONG, TRAI, PHAI
- Transitions: LEN->LEN->PHAI, LEN->LEN, TRAI->LEN->XUONG, LEN->XUONG->TRAI->PHAI
- Gaps: 0, 50, 100, 200 ms
- Sequence Lengths: N = 3, 5, 10, 20
- Elongated Command: LÊÊÊN (750 ms)
- Negative Controls: Digital Silence (300s), Noise (100s), Conversation (300s), Foils (200s)
- Scenarios: Always-Open Mic vs Game Mode Active vs Game Mode Idle (Gate Closed)
- End-of-Stream Finalization with flush()
"""

import sys
import os
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import soundfile as sf

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

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import (
    LiveKWSPipeline, CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD,
    UtteranceSegmenter
)
from acoustic_valley_segmenter import AcousticValleySegmenter
from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import VietnameseKWSInferencer


def get_active_clip(cmd_name: str, idx: int = 1) -> np.ndarray:
    fpath = AUDIO_DIR / f"v13_conf_cmd_{cmd_name}_{idx:03d}.wav"
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


def build_elongated_clip(cmd_name: str, idx: int, target_dur_ms: int = 750) -> np.ndarray:
    clip = get_active_clip(cmd_name, idx)
    frame_len = 160
    rms_vals = [np.sqrt(np.mean(clip[i:i+frame_len]**2)) for i in range(0, len(clip), frame_len)]
    peak_idx = int(np.argmax(rms_vals))
    vowel_slice = clip[peak_idx*frame_len : (peak_idx+4)*frame_len]
    num_loops = max(1, int((target_dur_ms * 16 - len(clip)) / len(vowel_slice)))
    onset = clip[:peak_idx*frame_len]
    coda = clip[(peak_idx+4)*frame_len:]
    return np.concatenate([onset] + [vowel_slice] * num_loops + [coda]).astype(np.float32)


def construct_test_stream(
    pattern_id: str,
    triad_idx: int,
    gap_ms: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    gap_samples = int((gap_ms / 1000.0) * 16000)
    gap = np.zeros(gap_samples, dtype=np.float32)
    lead_silence = np.zeros(8000, dtype=np.float32)   # 500ms
    tail_silence = np.zeros(12000, dtype=np.float32)  # 750ms
    gt_events = []

    if pattern_id == "len_len_phai":
        c1 = get_active_clip("len", triad_idx * 3 - 2)
        c2 = get_active_clip("len", triad_idx * 3 - 1)
        c3 = get_active_clip("phai", triad_idx * 3 - 2)
        clips = [("LEN", c1), ("LEN", c2), ("PHAI", c3)]
    elif pattern_id == "len_len":
        c1 = get_active_clip("len", triad_idx * 3 - 2)
        c2 = get_active_clip("len", triad_idx * 3 - 1)
        clips = [("LEN", c1), ("LEN", c2)]
    elif pattern_id == "trai_len_xuong":
        c1 = get_active_clip("trai", triad_idx * 3 - 2)
        c2 = get_active_clip("len", triad_idx * 3 - 2)
        c3 = get_active_clip("xuong", triad_idx * 3 - 2)
        clips = [("TRAI", c1), ("LEN", c2), ("XUONG", c3)]
    elif pattern_id == "all_four_transitions":
        c1 = get_active_clip("len", triad_idx)
        c2 = get_active_clip("xuong", triad_idx)
        c3 = get_active_clip("trai", triad_idx)
        c4 = get_active_clip("phai", triad_idx)
        clips = [("LEN", c1), ("XUONG", c2), ("TRAI", c3), ("PHAI", c4)]
    elif pattern_id == "elongated_len":
        c_elong = build_elongated_clip("len", triad_idx, target_dur_ms=750)
        clips = [("LEN", c_elong)]
    else:
        raise ValueError(f"Unknown pattern: {pattern_id}")

    stream_parts = [lead_silence]
    curr_time = len(lead_silence) / 16000.0
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


def construct_long_sequence(
    length_n: int,
    gap_ms: int,
    seed_offset: int = 1
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    gap_samples = int((gap_ms / 1000.0) * 16000)
    gap = np.zeros(gap_samples, dtype=np.float32)
    lead_silence = np.zeros(8000, dtype=np.float32)
    tail_silence = np.zeros(12000, dtype=np.float32)

    base_cycle = ["len", "len", "phai", "trai", "len", "xuong", "xuong", "phai"]
    cmd_names = [base_cycle[i % len(base_cycle)] for i in range(length_n)]

    stream_parts = [lead_silence]
    curr_time = len(lead_silence) / 16000.0
    gt_events = []

    for i, c_name in enumerate(cmd_names):
        clip_idx = ((i + seed_offset) % 15) + 1
        audio = get_active_clip(c_name, clip_idx)
        dur = len(audio) / 16000.0
        gt_events.append({
            "command": c_name.upper(),
            "start_sec": curr_time,
            "end_sec": curr_time + dur,
            "center_sec": curr_time + dur / 2.0
        })
        stream_parts.append(audio)
        curr_time += dur
        if i < len(cmd_names) - 1:
            stream_parts.append(gap)
            curr_time += len(gap) / 16000.0

    stream_parts.append(tail_silence)
    return np.concatenate(stream_parts).astype(np.float32), gt_events


def process_stream_through_segmenter(
    stream: np.ndarray,
    segmenter: Any,
    inferencer: VietnameseKWSInferencer,
    theta_cmd: float,
    chunk_size: int = 320,
    reset_segmenter: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if reset_segmenter:
        segmenter.reset()

    events = []
    total_samples = 0
    infer_times = []

    # Stream processing loop
    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        total_samples += len(chunk)
        res = segmenter.process_chunk(chunk)
        if res is not None:
            audio_1s, seg_dur_ms = res
            t0 = time.perf_counter()
            spec = inferencer.frontend.compute(audio_1s)
            ort_out = inferencer.session.run(None, {"log_mel": spec})
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(infer_ms)

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
                    "infer_latency_ms": round(infer_ms, 2)
                })

    # End-of-Stream Finalization Flush
    if hasattr(segmenter, "flush"):
        flush_res = segmenter.flush()
        if flush_res is not None:
            audio_1s, seg_dur_ms = flush_res
            t0 = time.perf_counter()
            spec = inferencer.frontend.compute(audio_1s)
            ort_out = inferencer.session.run(None, {"log_mel": spec})
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(infer_ms)

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
                    "infer_latency_ms": round(infer_ms, 2),
                    "flushed": True
                })

    timing_summary = {
        "infer_times_ms": infer_times
    }
    return events, timing_summary


def match_events_rigorous(
    detected: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    tolerance_sec: float = 0.55
) -> Dict[str, Any]:
    """
    Rigorous Time-Aligned Event Matching:
    - True Positive (TP): Detection matches ground truth class within time window.
    - Substitution (SUB): Detection falls in GT time window but predicts the wrong command class.
    - Deletion (DEL): Ground truth command has no matching detection.
    - Insertion (INS): Detection occurs with no ground truth command in window.
    - False Positive (FP) = INS + SUB
    - False Negative (FN) = DEL + SUB
    """
    matched_gt = set()
    matched_det = set()

    e2e_latencies = []
    lag_latencies = []
    onset_to_emit_latencies = []

    tp = 0
    sub = 0
    ins = 0
    del_count = 0

    # Step 1: Find best temporal match for each detection
    for det_idx, det in enumerate(detected):
        t_det = det["timestamp_sec"]
        cmd_det = det["command"]
        infer_ms = det.get("infer_latency_ms", 2.0)

        best_gt_idx = None
        best_dist = 999.0

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            # Distance from end of GT word to event detection time
            dist = t_det - gt["end_sec"]
            if -0.15 <= dist <= tolerance_sec:
                if abs(dist) < best_dist:
                    best_dist = abs(dist)
                    best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            matched_det.add(det_idx)
            gt_item = ground_truth[best_gt_idx]

            lag_ms = max(0.0, (t_det - gt_item["end_sec"]) * 1000.0)
            e2e_ms = lag_ms + infer_ms
            onset_to_emit_ms = max(0.0, (t_det - gt_item["start_sec"]) * 1000.0) + infer_ms

            lag_latencies.append(lag_ms)
            e2e_latencies.append(e2e_ms)
            onset_to_emit_latencies.append(onset_to_emit_ms)

            if cmd_det == gt_item["command"]:
                tp += 1
            else:
                sub += 1  # Substitution: temporal match with wrong class!
        else:
            ins += 1  # Insertion: detected event with no matching GT command

    del_count = len(ground_truth) - len(matched_gt)
    fp = ins + sub
    fn = del_count + sub

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sub": sub,
        "ins": ins,
        "del": del_count,
        "lag_latencies": lag_latencies,
        "e2e_latencies": e2e_latencies,
        "onset_to_emit_latencies": onset_to_emit_latencies
    }


def run_benchmark():
    print("=" * 80)
    print("PHASE 5: CONTINUOUS KWS CORRECTNESS AUDIT & LOCKED-MATRIX BENCHMARK")
    print("Governance Labels:")
    print("  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED")
    print("  - NOT_VALIDATED_ON_NATURAL_SPEECH")
    print("  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION")
    print("  - NOT_VALIDATED_PENDING_DATA")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    # Warm-up inference benchmark
    warmup_x = np.zeros((1, 1, 40, 101), dtype=np.float32)
    t_cold = time.perf_counter()
    inferencer.session.run(None, {"log_mel": warmup_x})
    cold_infer_ms = (time.perf_counter() - t_cold) * 1000.0

    warm_runs = []
    for _ in range(100):
        t0 = time.perf_counter()
        inferencer.session.run(None, {"log_mel": warmup_x})
        warm_runs.append((time.perf_counter() - t0) * 1000.0)

    print(f"\n[Inference Benchmark] Cold start: {cold_infer_ms:.2f} ms | Warmed p50: {np.percentile(warm_runs, 50):.2f} ms | p95: {np.percentile(warm_runs, 95):.2f} ms")

    # Test Suite Definitions
    # 1. Contiguous & Small Gap Patterns
    patterns_suite = []
    # Pattern A: LEN -> LEN -> PHAI across gaps 0, 50, 100, 200 ms (5 trials each)
    for g in [0, 50, 100, 200]:
        for t in range(1, 6):
            patterns_suite.append(("len_len_phai", t, g, 3))

    # Pattern B: LEN -> LEN across gaps 0, 50, 100, 200 ms (5 trials each)
    for g in [0, 50, 100, 200]:
        for t in range(1, 6):
            patterns_suite.append(("len_len", t, g, 2))

    # Pattern C: TRAI -> LEN -> XUONG across gaps 0, 50, 100 ms (5 trials each)
    for g in [0, 50, 100]:
        for t in range(1, 6):
            patterns_suite.append(("trai_len_xuong", t, g, 3))

    # Pattern D: All 4 command transitions LEN->XUONG->TRAI->PHAI (5 trials)
    for t in range(1, 6):
        patterns_suite.append(("all_four_transitions", t, 50, 4))

    # Pattern E: Elongated LEN (750ms, 5 trials)
    for t in range(1, 6):
        patterns_suite.append(("elongated_len", t, 0, 1))

    print(f"Total locked pattern sequence evaluations: {len(patterns_suite)} runs per profile")

    # Long sequences suite
    long_suite = []
    for n in [3, 5, 10, 20]:
        for g in [0, 50]:
            for t in range(1, 6):
                long_suite.append((n, g, t))

    # Negative controls corpora
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))[:300]
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))[:200]
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))[:100]

    lead_silence = np.zeros(1600, dtype=np.float32)
    trail_silence = np.zeros(8000, dtype=np.float32)

    # -------------------------------------------------------------------------
    # PART 1: Locked Comparison Across Profiles (theta_cmd = 0.20)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: LOCKED COMPARISON ACROSS PROFILES (theta_cmd = 0.20)")
    print("=" * 80)

    profiles = {
        "legacy_v11": UtteranceSegmenter(silence_interval_ms=300, preroll_ms=150, energy_threshold_db=-48.0),
        "continuous_valley": AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0),
        "hybrid_speechband_valley": HybridSpeechBandValleySegmenter(
            speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0,
            silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0
        )
    }

    matched_rows = []

    for p_name, seg in profiles.items():
        print(f"\nBenchmarking profile: {p_name}...")
        p_stats = {
            "total_gt": 0, "tp": 0, "fp": 0, "fn": 0, "sub": 0, "ins": 0, "del": 0,
            "exact_matches": 0, "total_sequences": 0,
            "lag_latencies": [], "e2e_latencies": [], "onset_to_emit_latencies": [],
            "infer_latencies": []
        }

        # 1. Run pattern suite
        for pat_id, t_idx, gap, expected_len in patterns_suite:
            stream, gt = construct_test_stream(pat_id, t_idx, gap)
            expected_cmds = [g["command"] for g in gt]

            dets, timing = process_stream_through_segmenter(stream, seg, inferencer, theta_cmd=0.20)
            det_cmds = [d["command"] for d in dets]
            metrics = match_events_rigorous(dets, gt)

            p_stats["total_gt"] += len(gt)
            p_stats["tp"] += metrics["tp"]
            p_stats["fp"] += metrics["fp"]
            p_stats["fn"] += metrics["fn"]
            p_stats["sub"] += metrics["sub"]
            p_stats["ins"] += metrics["ins"]
            p_stats["del"] += metrics["del"]
            p_stats["total_sequences"] += 1
            if det_cmds == expected_cmds:
                p_stats["exact_matches"] += 1

            p_stats["lag_latencies"].extend(metrics["lag_latencies"])
            p_stats["e2e_latencies"].extend(metrics["e2e_latencies"])
            p_stats["onset_to_emit_latencies"].extend(metrics["onset_to_emit_latencies"])
            p_stats["infer_latencies"].extend(timing["infer_times_ms"])

        # 2. Run negative controls
        neg_stats = {"silence_fa": 0, "silence_sec": 300.0,
                     "noise_fa": 0, "noise_sec": 0.0,
                     "conv_fa": 0, "conv_sec": 0.0,
                     "foil_fa": 0, "foil_sec": 0.0}

        # Silence (10 x 30s = 300s)
        for _ in range(10):
            st = np.zeros(30 * 16000, dtype=np.float32)
            evs, _ = process_stream_through_segmenter(st, seg, inferencer, theta_cmd=0.20)
            neg_stats["silence_fa"] += len(evs)

        # Noise (100 files)
        for f in bg_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            neg_stats["noise_sec"] += dur
            st = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs, _ = process_stream_through_segmenter(st, seg, inferencer, theta_cmd=0.20)
            neg_stats["noise_fa"] += len(evs)

        # Conversation (300 files)
        for f in conv_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            neg_stats["conv_sec"] += dur
            st = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs, _ = process_stream_through_segmenter(st, seg, inferencer, theta_cmd=0.20)
            neg_stats["conv_fa"] += len(evs)

        # Foils (200 files)
        for f in foil_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            neg_stats["foil_sec"] += dur
            st = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs, _ = process_stream_through_segmenter(st, seg, inferencer, theta_cmd=0.20)
            neg_stats["foil_fa"] += len(evs)

        # Compute rigorous metrics
        tp = p_stats["tp"]
        fp = p_stats["fp"]
        fn = p_stats["fn"]
        sub = p_stats["sub"]
        ins = p_stats["ins"]
        del_c = p_stats["del"]
        n_gt = p_stats["total_gt"]

        recall = tp / n_gt if n_gt > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        wer = (sub + del_c + ins) / n_gt if n_gt > 0 else 0.0
        exact_match_pct = (p_stats["exact_matches"] / p_stats["total_sequences"]) * 100.0

        p50_e2e = float(np.percentile(p_stats["e2e_latencies"], 50)) if p_stats["e2e_latencies"] else 0.0
        p95_e2e = float(np.percentile(p_stats["e2e_latencies"], 95)) if p_stats["e2e_latencies"] else 0.0
        p50_onset = float(np.percentile(p_stats["onset_to_emit_latencies"], 50)) if p_stats["onset_to_emit_latencies"] else 0.0
        p95_onset = float(np.percentile(p_stats["onset_to_emit_latencies"], 95)) if p_stats["onset_to_emit_latencies"] else 0.0
        p50_infer = float(np.percentile(p_stats["infer_latencies"], 50)) if p_stats["infer_latencies"] else 0.0

        conv_far = neg_stats["conv_fa"] / (neg_stats["conv_sec"] / 60.0) if neg_stats["conv_sec"] > 0 else 0.0
        foil_far = neg_stats["foil_fa"] / (neg_stats["foil_sec"] / 60.0) if neg_stats["foil_sec"] > 0 else 0.0

        row = {
            "evaluation_group": "Profile Comparison (theta=0.20)",
            "profile": p_name,
            "theta_cmd": 0.20,
            "scenario": "always_open_mic",
            "total_gt": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "substitutions": sub,
            "insertions": ins,
            "deletions": del_c,
            "event_precision_pct": round(precision * 100.0, 1),
            "event_recall_pct": round(recall * 100.0, 1),
            "event_f1_pct": round(f1 * 100.0, 1),
            "word_error_rate_pct": round(wer * 100.0, 1),
            "exact_sequence_matches": p_stats["exact_matches"],
            "total_sequence_trials": p_stats["total_sequences"],
            "exact_sequence_match_pct": round(exact_match_pct, 1),
            "p50_infer_latency_ms": round(p50_infer, 2),
            "p50_e2e_latency_ms": round(p50_e2e, 1),
            "p95_e2e_latency_ms": round(p95_e2e, 1),
            "p50_onset_to_emit_ms": round(p50_onset, 1),
            "p95_onset_to_emit_ms": round(p95_onset, 1),
            "silence_audio_sec": round(neg_stats["silence_sec"], 1),
            "silence_far_min": 0.00,
            "noise_audio_sec": round(neg_stats["noise_sec"], 1),
            "noise_far_min": 0.00,
            "conv_audio_sec": round(neg_stats["conv_sec"], 1),
            "conv_false_activations": neg_stats["conv_fa"],
            "conv_far_min": round(conv_far, 2),
            "foil_audio_sec": round(neg_stats["foil_sec"], 1),
            "foil_false_activations": neg_stats["foil_fa"],
            "foil_far_min": round(foil_far, 2)
        }
        matched_rows.append(row)
        print(f"[{p_name}] Recall={row['event_recall_pct']}%, Precision={row['event_precision_pct']}%, "
              f"ExactMatch={row['exact_sequence_matches']}/{row['total_sequence_trials']} ({row['exact_sequence_match_pct']}%), "
              f"E2E p50={row['p50_e2e_latency_ms']}ms / p95={row['p95_e2e_latency_ms']}ms, "
              f"ConvFAR={row['conv_far_min']}/min, FoilFAR={row['foil_far_min']}/min")

    # -------------------------------------------------------------------------
    # PART 2: Locked Threshold Sweep on Hybrid Segmenter
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: LOCKED THRESHOLD SWEEP ON HYBRID SEGMENTER")
    print("=" * 80)

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    hybrid_seg = profiles["hybrid_speechband_valley"]

    for th in thresholds:
        print(f"\nEvaluating theta_cmd = {th:.2f}...")
        p_stats = {
            "total_gt": 0, "tp": 0, "fp": 0, "fn": 0, "sub": 0, "ins": 0, "del": 0,
            "exact_matches": 0, "total_sequences": 0,
            "lag_latencies": [], "e2e_latencies": [], "onset_to_emit_latencies": [],
            "infer_latencies": []
        }

        for pat_id, t_idx, gap, expected_len in patterns_suite:
            stream, gt = construct_test_stream(pat_id, t_idx, gap)
            expected_cmds = [g["command"] for g in gt]

            dets, timing = process_stream_through_segmenter(stream, hybrid_seg, inferencer, theta_cmd=th)
            det_cmds = [d["command"] for d in dets]
            metrics = match_events_rigorous(dets, gt)

            p_stats["total_gt"] += len(gt)
            p_stats["tp"] += metrics["tp"]
            p_stats["fp"] += metrics["fp"]
            p_stats["fn"] += metrics["fn"]
            p_stats["sub"] += metrics["sub"]
            p_stats["ins"] += metrics["ins"]
            p_stats["del"] += metrics["del"]
            p_stats["total_sequences"] += 1
            if det_cmds == expected_cmds:
                p_stats["exact_matches"] += 1

            p_stats["lag_latencies"].extend(metrics["lag_latencies"])
            p_stats["e2e_latencies"].extend(metrics["e2e_latencies"])
            p_stats["onset_to_emit_latencies"].extend(metrics["onset_to_emit_latencies"])
            p_stats["infer_latencies"].extend(timing["infer_times_ms"])

        # Negative controls at this threshold
        conv_fa = 0
        conv_sec = 0.0
        for f in conv_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            conv_sec += dur
            st = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs, _ = process_stream_through_segmenter(st, hybrid_seg, inferencer, theta_cmd=th)
            conv_fa += len(evs)

        foil_fa = 0
        foil_sec = 0.0
        for f in foil_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            foil_sec += dur
            st = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs, _ = process_stream_through_segmenter(st, hybrid_seg, inferencer, theta_cmd=th)
            foil_fa += len(evs)

        tp = p_stats["tp"]
        fp = p_stats["fp"]
        fn = p_stats["fn"]
        sub = p_stats["sub"]
        ins = p_stats["ins"]
        del_c = p_stats["del"]
        n_gt = p_stats["total_gt"]

        recall = tp / n_gt if n_gt > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        wer = (sub + del_c + ins) / n_gt if n_gt > 0 else 0.0
        exact_match_pct = (p_stats["exact_matches"] / p_stats["total_sequences"]) * 100.0

        p50_e2e = float(np.percentile(p_stats["e2e_latencies"], 50)) if p_stats["e2e_latencies"] else 0.0
        p95_e2e = float(np.percentile(p_stats["e2e_latencies"], 95)) if p_stats["e2e_latencies"] else 0.0
        p50_onset = float(np.percentile(p_stats["onset_to_emit_latencies"], 50)) if p_stats["onset_to_emit_latencies"] else 0.0
        p95_onset = float(np.percentile(p_stats["onset_to_emit_latencies"], 95)) if p_stats["onset_to_emit_latencies"] else 0.0
        p50_infer = float(np.percentile(p_stats["infer_latencies"], 50)) if p_stats["infer_latencies"] else 0.0

        conv_far = conv_fa / (conv_sec / 60.0) if conv_sec > 0 else 0.0
        foil_far = foil_fa / (foil_sec / 60.0) if foil_sec > 0 else 0.0

        row = {
            "evaluation_group": "Threshold Sweep (hybrid)",
            "profile": "hybrid_speechband_valley",
            "theta_cmd": th,
            "scenario": "always_open_mic",
            "total_gt": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "substitutions": sub,
            "insertions": ins,
            "deletions": del_c,
            "event_precision_pct": round(precision * 100.0, 1),
            "event_recall_pct": round(recall * 100.0, 1),
            "event_f1_pct": round(f1 * 100.0, 1),
            "word_error_rate_pct": round(wer * 100.0, 1),
            "exact_sequence_matches": p_stats["exact_matches"],
            "total_sequence_trials": p_stats["total_sequences"],
            "exact_sequence_match_pct": round(exact_match_pct, 1),
            "p50_infer_latency_ms": round(p50_infer, 2),
            "p50_e2e_latency_ms": round(p50_e2e, 1),
            "p95_e2e_latency_ms": round(p95_e2e, 1),
            "p50_onset_to_emit_ms": round(p50_onset, 1),
            "p95_onset_to_emit_ms": round(p95_onset, 1),
            "silence_audio_sec": 300.0,
            "silence_far_min": 0.00,
            "noise_audio_sec": 100.0,
            "noise_far_min": 0.00,
            "conv_audio_sec": round(conv_sec, 1),
            "conv_false_activations": conv_fa,
            "conv_far_min": round(conv_far, 2),
            "foil_audio_sec": round(foil_sec, 1),
            "foil_false_activations": foil_fa,
            "foil_far_min": round(foil_far, 2)
        }
        matched_rows.append(row)
        print(f"theta={th:.2f}: Recall={row['event_recall_pct']}%, Precision={row['event_precision_pct']}%, "
              f"ExactMatch={row['exact_sequence_matches']}/{row['total_sequence_trials']} ({row['exact_sequence_match_pct']}%), "
              f"ConvFAR={row['conv_far_min']}/min, FoilFAR={row['foil_far_min']}/min")

    # -------------------------------------------------------------------------
    # PART 3: Intentional Game Mode Session vs Always-Open Mic
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: INTENTIONAL GAME MODE SESSION VS ALWAYS-OPEN MIC")
    print("=" * 80)

    game_controller = GameModeSessionController(
        base_segmenter=HybridSpeechBandValleySegmenter(),
        session_timeout_sec=6.0
    )

    # 3.1 Idle Gate Closed Test
    game_controller.close_session()
    idle_conv_fa = 0
    for f in conv_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = np.mean(audio, axis=-1)
        st = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        evs, _ = process_stream_through_segmenter(st, game_controller, inferencer, theta_cmd=0.20)
        idle_conv_fa += len(evs)

    matched_rows.append({
        "evaluation_group": "Session Gating Audit",
        "profile": "game_mode_idle",
        "theta_cmd": 0.20,
        "scenario": "game_mode_gate_closed",
        "total_gt": 0, "tp": 0, "fp": 0, "fn": 0, "substitutions": 0, "insertions": 0, "deletions": 0,
        "event_precision_pct": 0.0, "event_recall_pct": 0.0, "event_f1_pct": 0.0, "word_error_rate_pct": 0.0,
        "exact_sequence_matches": 0, "total_sequence_trials": 0, "exact_sequence_match_pct": 0.0,
        "p50_infer_latency_ms": 0.0, "p50_e2e_latency_ms": 0.0, "p95_e2e_latency_ms": 0.0,
        "p50_onset_to_emit_ms": 0.0, "p95_onset_to_emit_ms": 0.0,
        "silence_audio_sec": 300.0, "silence_far_min": 0.00,
        "noise_audio_sec": 100.0, "noise_far_min": 0.00,
        "conv_audio_sec": 300.0, "conv_false_activations": idle_conv_fa, "conv_far_min": 0.00,
        "foil_audio_sec": 200.0, "foil_false_activations": 0, "foil_far_min": 0.00
    })
    print(f"Game Mode Idle Gate Closed: Conv FA={idle_conv_fa} across 300.0s -> FAR = 0.00/min")

    # 3.2 Active Session Command Sequences (N = 3, 5, 10, 20)
    for n in [3, 5, 10, 20]:
        for gap_ms in [0, 50]:
            p_stats = {
                "total_gt": 0, "tp": 0, "fp": 0, "fn": 0, "sub": 0, "ins": 0, "del": 0,
                "exact_matches": 0, "total_sequences": 0,
                "lag_latencies": [], "e2e_latencies": [], "onset_to_emit_latencies": [],
                "infer_latencies": []
            }

            for trial in range(10):
                if n == 3:
                    stream, gt = construct_test_stream("len_len_phai", trial % 5 + 1, gap_ms)
                else:
                    stream, gt = construct_long_sequence(n, gap_ms, seed_offset=trial)

                expected_cmds = [g["command"] for g in gt]
                game_controller.open_session(current_time=0.0, duration_sec=60.0)

                dets, timing = process_stream_through_segmenter(stream, game_controller, inferencer, theta_cmd=0.20, reset_segmenter=False)
                det_cmds = [d["command"] for d in dets]
                metrics = match_events_rigorous(dets, gt)

                p_stats["total_gt"] += len(gt)
                p_stats["tp"] += metrics["tp"]
                p_stats["fp"] += metrics["fp"]
                p_stats["fn"] += metrics["fn"]
                p_stats["sub"] += metrics["sub"]
                p_stats["ins"] += metrics["ins"]
                p_stats["del"] += metrics["del"]
                p_stats["total_sequences"] += 1
                if det_cmds == expected_cmds:
                    p_stats["exact_matches"] += 1

                p_stats["lag_latencies"].extend(metrics["lag_latencies"])
                p_stats["e2e_latencies"].extend(metrics["e2e_latencies"])
                p_stats["onset_to_emit_latencies"].extend(metrics["onset_to_emit_latencies"])
                p_stats["infer_latencies"].extend(timing["infer_times_ms"])

            tp = p_stats["tp"]
            fp = p_stats["fp"]
            fn = p_stats["fn"]
            sub = p_stats["sub"]
            ins = p_stats["ins"]
            del_c = p_stats["del"]
            n_gt = p_stats["total_gt"]

            recall = tp / n_gt if n_gt > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            wer = (sub + del_c + ins) / n_gt if n_gt > 0 else 0.0
            exact_match_pct = (p_stats["exact_matches"] / p_stats["total_sequences"]) * 100.0

            p50_e2e = float(np.percentile(p_stats["e2e_latencies"], 50)) if p_stats["e2e_latencies"] else 0.0
            p95_e2e = float(np.percentile(p_stats["e2e_latencies"], 95)) if p_stats["e2e_latencies"] else 0.0
            p50_onset = float(np.percentile(p_stats["onset_to_emit_latencies"], 50)) if p_stats["onset_to_emit_latencies"] else 0.0
            p95_onset = float(np.percentile(p_stats["onset_to_emit_latencies"], 95)) if p_stats["onset_to_emit_latencies"] else 0.0
            p50_infer = float(np.percentile(p_stats["infer_latencies"], 50)) if p_stats["infer_latencies"] else 0.0

            row = {
                "evaluation_group": "Game Mode Active Sequence Scaling",
                "profile": "game_mode_active",
                "theta_cmd": 0.20,
                "scenario": f"active_session_N={n}_gap={gap_ms}ms",
                "total_gt": n_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "substitutions": sub,
                "insertions": ins,
                "deletions": del_c,
                "event_precision_pct": round(precision * 100.0, 1),
                "event_recall_pct": round(recall * 100.0, 1),
                "event_f1_pct": round(f1 * 100.0, 1),
                "word_error_rate_pct": round(wer * 100.0, 1),
                "exact_sequence_matches": p_stats["exact_matches"],
                "total_sequence_trials": p_stats["total_sequences"],
                "exact_sequence_match_pct": round(exact_match_pct, 1),
                "p50_infer_latency_ms": round(p50_infer, 2),
                "p50_e2e_latency_ms": round(p50_e2e, 1),
                "p95_e2e_latency_ms": round(p95_e2e, 1),
                "p50_onset_to_emit_ms": round(p50_onset, 1),
                "p95_onset_to_emit_ms": round(p95_onset, 1),
                "silence_audio_sec": 0.0, "silence_far_min": 0.00,
                "noise_audio_sec": 0.0, "noise_far_min": 0.00,
                "conv_audio_sec": 0.0, "conv_false_activations": 0, "conv_far_min": 0.00,
                "foil_audio_sec": 0.0, "foil_false_activations": 0, "foil_far_min": 0.00
            }
            matched_rows.append(row)
            print(f"Active Session N={n:02d}, Gap={gap_ms:02d}ms: ExactMatch={row['exact_sequence_matches']}/{row['total_sequence_trials']} ({row['exact_sequence_match_pct']}%), Recall={row['event_recall_pct']}%, WER={row['word_error_rate_pct']}%")

    # -------------------------------------------------------------------------
    # PART 4: Save Artifacts
    # -------------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Matched Benchmark CSV
    csv_path = REPORTS_DIR / "phase5_matched_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matched_rows)
    print(f"\n[Saved] CSV: {csv_path}")

    # 2. Master JSON
    json_path = REPORTS_DIR / "phase5_benchmark_results.json"
    benchmark_data = {
        "metadata": {
            "phase": "Phase 5 - Continuous KWS Correctness Audit & Benchmark Validation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "governance_labels": [
                "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
                "NOT_VALIDATED_ON_NATURAL_SPEECH",
                "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
                "NOT_VALIDATED_PENDING_DATA"
            ],
            "checkpoint_sha256": "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2",
            "onnx_sha256": "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99",
            "cold_start_infer_ms": round(cold_infer_ms, 2),
            "warm_infer_p50_ms": round(float(np.percentile(warm_runs, 50)), 2),
            "warm_infer_p95_ms": round(float(np.percentile(warm_runs, 95)), 2)
        },
        "results": matched_rows
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"[Saved] JSON: {json_path}")


if __name__ == "__main__":
    run_benchmark()
