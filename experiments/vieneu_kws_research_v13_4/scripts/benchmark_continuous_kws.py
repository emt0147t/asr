"""
Unified Benchmark Suite for Continuous Vietnamese Keyword Spotting (KWS).
Phase 2: Event Detection on Continuous / Connected Speech.

Evaluates 5 Configurations across 4 Speech Patterns, 6 Inter-word Gaps,
3 Speaker Triads, and Negative Controls (Foils, Conversational, Background Noise).

Evaluated Configurations:
1. baseline_segmented: UtteranceSegmenter (silence=300ms, preroll=150ms)
2. baseline_streaming: Legacy 160ms sliding window with 800ms cooldown latch
3. approach_a1_speechband: SpeechBandRatioSegmenterAdapter (Phase 15/18 Profile B, hangover=140ms)
4. approach_a2_acoustic_valley: AcousticValleySegmenter (Energy valley split, dip=12dB, rebound=6dB)
5. approach_b_continuous_decoder: ContinuousEventDecoder (Sliding window 40ms hop, min_dist=250ms, dip=0.15)
"""

import sys
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import soundfile as sf

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
    UtteranceSegmenter, SpeechBandRatioSegmenterAdapter
)
from acoustic_valley_segmenter import AcousticValleySegmenter
from continuous_event_decoder import ContinuousEventDecoder
from infer import VietnameseKWSInferencer


def get_exact_active(cmd_name: str, idx: int) -> np.ndarray:
    """Extracts the active speech segment of a command clip."""
    fpath = AUDIO_DIR / f"v13_conf_cmd_{cmd_name}_{idx:03d}.wav"
    assert fpath.exists(), f"Audio file missing: {fpath}"
    data, sr = sf.read(fpath, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000)

    # Frame RMS energy profiling
    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(data[i:i+frame_len]**2)) + 1e-7) for i in range(0, len(data), frame_len)]
    above = np.where(np.array(rms_db) > -42.0)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(data), (above[-1] + 3) * frame_len)
        return data[s:e]
    return data[:6400]


def build_elongated_clip(cmd_name: str, idx: int, target_dur_ms: int = 750) -> np.ndarray:
    """Builds a continuous elongated voiced command utterance (e.g. LÊÊÊN)."""
    clip = get_exact_active(cmd_name, idx)
    frame_len = 160
    rms_vals = [np.sqrt(np.mean(clip[i:i+frame_len]**2)) for i in range(0, len(clip), frame_len)]
    peak_idx = int(np.argmax(rms_vals))
    vowel_slice = clip[peak_idx*frame_len : (peak_idx+4)*frame_len]  # ~40ms vowel kernel
    num_loops = max(1, int((target_dur_ms * 16 - len(clip)) / len(vowel_slice)))
    onset = clip[:peak_idx*frame_len]
    coda = clip[(peak_idx+4)*frame_len:]
    elongated = np.concatenate([onset] + [vowel_slice] * num_loops + [coda]).astype(np.float32)
    return elongated


def construct_test_stream(
    pattern_id: str,
    triad_idx: int,
    gap_ms: int
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Constructs an audio stream and ground truth event ledger.
    Returns:
        (stream_audio, ground_truth_events)
    """
    gap_samples = int((gap_ms / 1000.0) * 16000)
    gap = np.zeros(gap_samples, dtype=np.float32)
    lead_silence = np.zeros(8000, dtype=np.float32)   # 500ms
    tail_silence = np.zeros(12000, dtype=np.float32)  # 750ms

    gt_events = []

    if pattern_id == "pattern_1_len_len_phai":
        c1 = get_exact_active("len", triad_idx * 3 - 2)
        c2 = get_exact_active("len", triad_idx * 3 - 1)
        c3 = get_exact_active("phai", triad_idx * 3 - 2)
        clips = [("LEN", c1), ("LEN", c2), ("PHAI", c3)]

    elif pattern_id == "pattern_2_trai_len_xuong":
        c1 = get_exact_active("trai", triad_idx * 3 - 2)
        c2 = get_exact_active("len", triad_idx * 3 - 2)
        c3 = get_exact_active("xuong", triad_idx * 3 - 2)
        clips = [("TRAI", c1), ("LEN", c2), ("XUONG", c3)]

    elif pattern_id == "pattern_3_len_len_len":
        c1 = get_exact_active("len", triad_idx * 3 - 2)
        c2 = get_exact_active("len", triad_idx * 3 - 1)
        c3 = get_exact_active("len", triad_idx * 3)
        clips = [("LEN", c1), ("LEN", c2), ("LEN", c3)]

    elif pattern_id == "pattern_4_elongated_len":
        c_elong = build_elongated_clip("len", triad_idx, target_dur_ms=750)
        clips = [("LEN", c_elong)]

    else:
        raise ValueError(f"Unknown pattern: {pattern_id}")

    # Assemble stream
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
    stream_audio = np.concatenate(stream_parts).astype(np.float32)
    return stream_audio, gt_events


def run_segmented_pipeline(
    stream: np.ndarray,
    segmenter_instance: Any,
    inferencer: VietnameseKWSInferencer,
    profile_name: str
) -> List[Dict[str, Any]]:
    """Runs audio through a segmented architecture using 20ms chunks."""
    segmenter_instance.reset()
    events = []
    chunk_size = 320  # 20ms at 16kHz
    total_samples = 0

    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        total_samples += len(chunk)
        res = segmenter_instance.process_chunk(chunk)
        if res is not None:
            audio_1s, seg_dur_ms = res
            t0 = time.perf_counter()
            spec = inferencer.frontend.compute(audio_1s)
            ort_out = inferencer.session.run(None, {"log_mel": spec})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            probs = ort_out[1][0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            if top_idx < 4 and top_prob >= inferencer.theta_cmd:
                pred_cmd = TARGET_CLASSES[top_idx]
                curr_sec = total_samples / 16000.0
                events.append({
                    "command": pred_cmd,
                    "confidence": round(top_prob, 4),
                    "timestamp_sec": round(curr_sec, 3),
                    "segment_duration_ms": round(seg_dur_ms, 1),
                    "latency_ms": round(dt_ms, 2)
                })

    return events


def run_streaming_pipeline(
    stream: np.ndarray,
    inferencer: VietnameseKWSInferencer,
    update_interval_ms: int = 160,
    cooldown_sec: float = 0.80,
    consecutive_frames: int = 2
) -> List[Dict[str, Any]]:
    """Simulates the legacy streaming sliding-window engine with 800ms cooldown."""
    hop_samples = int(16000 * (update_interval_ms / 1000.0))
    win_samples = 16000
    events = []

    last_trigger_time = -999.0
    consecutive_count = 0
    consecutive_candidate = None
    is_armed = True

    for pos in range(0, len(stream) - win_samples + 1, hop_samples):
        window = stream[pos : pos + win_samples]
        curr_time = (pos + win_samples) / 16000.0

        t0 = time.perf_counter()
        spec = inferencer.frontend.compute(window)
        ort_out = inferencer.session.run(None, {"log_mel": spec})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        probs = ort_out[1][0]
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        pred_class = TARGET_CLASSES[top_idx] if (top_idx < 4 and top_prob >= inferencer.theta_cmd) else "REJECT"

        # Re-arm check
        max_cmd_p = max(probs[:4])
        if not is_armed:
            if max_cmd_p < 0.15:
                if curr_time - last_trigger_time >= (cooldown_sec * 0.75):
                    is_armed = True
                    consecutive_count = 0
                    consecutive_candidate = None

        if pred_class in TARGET_CLASSES:
            if pred_class == consecutive_candidate:
                consecutive_count += 1
            else:
                consecutive_candidate = pred_class
                consecutive_count = 1

            time_since_last = curr_time - last_trigger_time
            if (
                consecutive_count >= consecutive_frames
                and time_since_last >= cooldown_sec
                and is_armed
            ):
                events.append({
                    "command": pred_class,
                    "confidence": round(top_prob, 4),
                    "timestamp_sec": round(curr_time, 3),
                    "latency_ms": round(dt_ms, 2)
                })
                last_trigger_time = curr_time
                is_armed = False
                consecutive_count = 0
        else:
            consecutive_candidate = None
            consecutive_count = 0

    return events


def run_continuous_event_decoder(
    stream: np.ndarray,
    inferencer: VietnameseKWSInferencer,
    hop_ms: int = 40,
    min_dist_ms: int = 250,
    dip_delta: float = 0.15
) -> List[Dict[str, Any]]:
    """Runs sliding window with ContinuousEventDecoder (Approach B)."""
    hop_samples = int(16000 * (hop_ms / 1000.0))
    win_samples = 16000
    decoder = ContinuousEventDecoder(
        theta_cmd=inferencer.theta_cmd,
        min_distance_ms=min_dist_ms,
        dip_delta=dip_delta
    )
    events = []

    for pos in range(0, len(stream) - win_samples + 1, hop_samples):
        window = stream[pos : pos + win_samples]
        curr_time = (pos + win_samples) / 16000.0

        t0 = time.perf_counter()
        spec = inferencer.frontend.compute(window)
        ort_out = inferencer.session.run(None, {"log_mel": spec})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        probs = ort_out[1][0]

        ev = decoder.process_frame(curr_time, probs)
        if ev is not None:
            events.append({
                "command": ev["command"],
                "confidence": round(ev["confidence"], 4),
                "timestamp_sec": round(curr_time, 3),
                "latency_ms": round(dt_ms, 2)
            })

    return events


def evaluate_detection(
    detected_events: List[Dict[str, Any]],
    gt_events: List[Dict[str, Any]],
    tolerance_sec: float = 0.65
) -> Dict[str, Any]:
    """
    Evaluates detected events against ground truth using bipartite temporal matching.
    Calculates TP, FP, FN, duplicates, exact sequence match, and latency.
    """
    gt_matched = [False] * len(gt_events)
    det_matched = [False] * len(detected_events)
    latencies = []

    tp = 0
    fp = 0
    fn = 0
    duplicates = 0

    for d_idx, det in enumerate(detected_events):
        d_cmd = det["command"]
        d_time = det["timestamp_sec"]

        best_g_idx = None
        best_dist = 999.0

        for g_idx, gt in enumerate(gt_events):
            if gt["command"] == d_cmd:
                dist = abs(d_time - gt["end_sec"])
                if dist <= tolerance_sec:
                    if not gt_matched[g_idx]:
                        if dist < best_dist:
                            best_dist = dist
                            best_g_idx = g_idx
                    else:
                        # Already matched: this is a duplicate event!
                        pass

        if best_g_idx is not None:
            gt_matched[best_g_idx] = True
            det_matched[d_idx] = True
            tp += 1
            lat = max(0.0, (d_time - gt_events[best_g_idx]["end_sec"]) * 1000.0)
            latencies.append(lat)
        else:
            # Check if this was a duplicate trigger of an already matched GT
            is_dup = False
            for g_idx, gt in enumerate(gt_events):
                if gt["command"] == d_cmd and gt_matched[g_idx]:
                    if abs(d_time - gt["end_sec"]) <= (tolerance_sec * 1.5):
                        is_dup = True
                        break
            if is_dup:
                duplicates += 1
            fp += 1

    fn = sum(1 for m in gt_matched if not m)

    # Exact Sequence Match
    det_cmds = [d["command"] for d in detected_events]
    gt_cmds = [g["command"] for g in gt_events]
    exact_match = (det_cmds == gt_cmds)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": duplicates,
        "exact_match": exact_match,
        "latencies": latencies,
        "detected_commands": det_cmds,
        "ground_truth_commands": gt_cmds
    }


def main():
    print("=" * 80)
    print("PHASE 2: UNIFIED CONTINUOUS KWS EVENT DETECTION BENCHMARK")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    # Configurations under test
    configs = [
        {
            "id": "baseline_segmented",
            "name": "Baseline Segmented (300ms silence)",
            "type": "segmented",
            "instance": UtteranceSegmenter(silence_interval_ms=300, preroll_ms=150)
        },
        {
            "id": "baseline_streaming",
            "name": "Baseline Streaming (160ms hop, 800ms cooldown)",
            "type": "streaming"
        },
        {
            "id": "approach_a1_speechband",
            "name": "Approach A1: SpeechBandRatio (hangover=140ms)",
            "type": "segmented",
            "instance": SpeechBandRatioSegmenterAdapter(hangover_ms=140, preroll_ms=150)
        },
        {
            "id": "approach_a2_acoustic_valley",
            "name": "Approach A2: Acoustic Valley Segmenter (dip=12dB)",
            "type": "segmented",
            "instance": AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0)
        },
        {
            "id": "approach_b_continuous_decoder",
            "name": "Approach B: Continuous Event Decoder (40ms hop, NMS min_dist=250ms)",
            "type": "continuous_decoder"
        }
    ]

    patterns = [
        "pattern_1_len_len_phai",
        "pattern_2_trai_len_xuong",
        "pattern_3_len_len_len",
        "pattern_4_elongated_len"
    ]

    gaps = [0, 50, 100, 150, 250, 400]
    triads = [1, 2, 3]

    raw_results = []
    summary_by_config = {cfg["id"]: {
        "name": cfg["name"],
        "tp": 0, "fp": 0, "fn": 0, "duplicates": 0,
        "total_trials": 0, "exact_matches": 0,
        "latencies": [],
        "by_gap_exact": {g: {"total": 0, "matches": 0} for g in gaps},
        "by_pattern_exact": {p: {"total": 0, "matches": 0} for p in patterns}
    } for cfg in configs}

    print(f"\n[1/2] Running Core Benchmark: 4 Patterns x 6 Gaps x 3 Speaker Triads = 72 streams per config...")

    for pat in patterns:
        for gap in gaps:
            for triad in triads:
                stream, gt = construct_test_stream(pat, triad, gap)

                for cfg in configs:
                    cfg_id = cfg["id"]
                    if cfg["type"] == "segmented":
                        dets = run_segmented_pipeline(stream, cfg["instance"], inferencer, cfg_id)
                    elif cfg["type"] == "streaming":
                        dets = run_streaming_pipeline(stream, inferencer, update_interval_ms=160, cooldown_sec=0.80)
                    elif cfg["type"] == "continuous_decoder":
                        dets = run_continuous_event_decoder(stream, inferencer, hop_ms=40, min_dist_ms=250)

                    eval_res = evaluate_detection(dets, gt)

                    # Accumulate summary
                    s = summary_by_config[cfg_id]
                    s["tp"] += eval_res["tp"]
                    s["fp"] += eval_res["fp"]
                    s["fn"] += eval_res["fn"]
                    s["duplicates"] += eval_res["duplicates"]
                    s["total_trials"] += 1
                    if eval_res["exact_match"]:
                        s["exact_matches"] += 1
                        s["by_gap_exact"][gap]["matches"] += 1
                        s["by_pattern_exact"][pat]["matches"] += 1
                    s["by_gap_exact"][gap]["total"] += 1
                    s["by_pattern_exact"][pat]["total"] += 1
                    s["latencies"].extend(eval_res["latencies"])

                    raw_results.append({
                        "config_id": cfg_id,
                        "pattern": pat,
                        "gap_ms": gap,
                        "triad_idx": triad,
                        "gt_commands": eval_res["ground_truth_commands"],
                        "det_commands": eval_res["detected_commands"],
                        "exact_match": eval_res["exact_match"],
                        "tp": eval_res["tp"],
                        "fp": eval_res["fp"],
                        "fn": eval_res["fn"],
                        "duplicates": eval_res["duplicates"]
                    })

    # [2/2] Negative Control Evaluation: Foils, Conversational, Background Noise
    print("\n[2/2] Running Negative Control Evaluation (Foils, Conversational, Noise)...")
    negative_files = (
        sorted(list(AUDIO_DIR.glob("v13_conf_foil_len_*.wav")))[:5] +
        sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))[:5] +
        sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))[:5]
    )

    neg_summary = {cfg["id"]: {"trials": 0, "false_activations": 0} for cfg in configs}

    for neg_f in negative_files:
        audio, sr = sf.read(neg_f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        neg_stream = np.concatenate([np.zeros(4000, dtype=np.float32), audio, np.zeros(4000, dtype=np.float32)])

        for cfg in configs:
            cfg_id = cfg["id"]
            if cfg["type"] == "segmented":
                dets = run_segmented_pipeline(neg_stream, cfg["instance"], inferencer, cfg_id)
            elif cfg["type"] == "streaming":
                dets = run_streaming_pipeline(neg_stream, inferencer)
            elif cfg["type"] == "continuous_decoder":
                dets = run_continuous_event_decoder(neg_stream, inferencer)

            neg_summary[cfg_id]["trials"] += 1
            if len(dets) > 0:
                neg_summary[cfg_id]["false_activations"] += len(dets)

    # Compute final metrics table
    final_metrics_table = []
    print("\n" + "=" * 95)
    print(f"{'Configuration':<38} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'F1 (%)':<9} | {'SeqMatch':<9} | {'DupRate':<8} | {'Lat p50':<8}")
    print("-" * 95)

    for cfg in configs:
        cfg_id = cfg["id"]
        s = summary_by_config[cfg_id]

        prec = (s["tp"] / (s["tp"] + s["fp"]) * 100.0) if (s["tp"] + s["fp"]) > 0 else 0.0
        rec = (s["tp"] / (s["tp"] + s["fn"]) * 100.0) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        seq_match = (s["exact_matches"] / s["total_trials"] * 100.0) if s["total_trials"] > 0 else 0.0
        total_dets = s["tp"] + s["fp"]
        dup_rate = (s["duplicates"] / total_dets * 100.0) if total_dets > 0 else 0.0
        p50 = float(np.median(s["latencies"])) if len(s["latencies"]) > 0 else 0.0
        p95 = float(np.percentile(s["latencies"], 95)) if len(s["latencies"]) > 0 else 0.0

        print(f"{cfg['name']:<38} | {prec:8.2f}% | {rec:8.2f}% | {f1:8.2f}% | {seq_match:8.2f}% | {dup_rate:7.2f}% | {p50:6.1f}ms")

        final_metrics_table.append({
            "config_id": cfg_id,
            "config_name": cfg["name"],
            "event_precision": round(prec, 2),
            "event_recall": round(rec, 2),
            "event_f1": round(f1, 2),
            "exact_sequence_match_pct": round(seq_match, 2),
            "duplicate_rate_pct": round(dup_rate, 2),
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "duplicates": s["duplicates"],
            "total_trials": s["total_trials"],
            "exact_matches": s["exact_matches"],
            "by_gap_exact_pct": {g: round(s["by_gap_exact"][g]["matches"] / s["by_gap_exact"][g]["total"] * 100.0, 1) for g in gaps},
            "by_pattern_exact_pct": {p: round(s["by_pattern_exact"][p]["matches"] / s["by_pattern_exact"][p]["total"] * 100.0, 1) for p in patterns},
            "neg_trials": neg_summary[cfg_id]["trials"],
            "neg_false_activations": neg_summary[cfg_id]["false_activations"]
        })

    print("=" * 95)

    # Save JSON and CSV reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "continuous_kws_benchmark_results.json"
    csv_path = REPORTS_DIR / "continuous_kws_benchmark_results.csv"

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_reference": "legacy_v11_hard_negative_weighted_seed42",
            "threshold_theta_cmd": DEFAULT_THRESHOLD,
            "num_patterns": len(patterns),
            "gaps_ms": gaps,
            "num_triads": len(triads),
            "total_streams_per_config": len(patterns) * len(gaps) * len(triads),
            "negative_control_files": len(negative_files)
        },
        "summary_table": final_metrics_table,
        "raw_trials": raw_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config_id", "config_name", "event_precision", "event_recall", "event_f1",
            "exact_sequence_match_pct", "duplicate_rate_pct", "latency_p50_ms", "latency_p95_ms",
            "tp", "fp", "fn", "duplicates", "total_trials", "exact_matches", "neg_false_activations"
        ])
        writer.writeheader()
        for row in final_metrics_table:
            row_copy = {k: v for k, v in row.items() if k in writer.fieldnames}
            writer.writerow(row_copy)

    print(f"\nSaved benchmark results to:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
