"""
Long-Sequence Benchmark for Continuous Vietnamese Keyword Spotting (KWS).
Phase 3: Systematic evaluation of extended command sequences (N = 3, 5, 10, 20 events).

Evaluates:
- Sequence Lengths: N = 3, 5, 10, 20 commands.
- Sequence Patterns:
  1. repetitive_identical: [LEN, LEN, LEN, ...]
  2. repeating_prefix_distinct: [LEN, LEN, PHAI, LEN, LEN, ...]
  3. alternating_distinct: [TRAI, LEN, XUONG, PHAI, ...]
  4. mixed_gameplay: realistic gaming command control streams.
- Inter-word Gaps: 0ms (contiguous), 50ms, 100ms, 200ms.
- Profiles: legacy_v11 vs continuous_valley.

Metrics:
- Event Precision, Event Recall, Event F1
- Exact Sequence Match %
- Duplicate Rate %
- Latency p50, p95 (ms)
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
    UtteranceSegmenter
)
from acoustic_valley_segmenter import AcousticValleySegmenter
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


def construct_long_sequence(
    length_n: int,
    pattern_type: str,
    gap_ms: int,
    seed_offset: int = 1
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Constructs an audio stream of length N commands and its ground truth ledger.
    """
    gap_samples = int((gap_ms / 1000.0) * 16000)
    gap = np.zeros(gap_samples, dtype=np.float32)
    lead_silence = np.zeros(8000, dtype=np.float32)   # 500ms
    tail_silence = np.zeros(12000, dtype=np.float32)  # 750ms

    # Define command list based on pattern type
    if pattern_type == "repetitive_identical":
        cmd_names = ["len"] * length_n
    elif pattern_type == "repeating_prefix_distinct":
        base_cycle = ["len", "len", "phai", "xuong"]
        cmd_names = [base_cycle[i % len(base_cycle)] for i in range(length_n)]
    elif pattern_type == "alternating_distinct":
        base_cycle = ["trai", "phai", "len", "xuong"]
        cmd_names = [base_cycle[i % len(base_cycle)] for i in range(length_n)]
    elif pattern_type == "mixed_gameplay":
        base_cycle = ["len", "len", "phai", "trai", "len", "xuong", "xuong", "phai"]
        cmd_names = [base_cycle[i % len(base_cycle)] for i in range(length_n)]
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    # Build stream
    stream_parts = [lead_silence]
    curr_time = len(lead_silence) / 16000.0
    gt_events = []

    for i, c_name in enumerate(cmd_names):
        clip_idx = ((i + seed_offset) % 15) + 1  # Rotate across available fixture indices 1..15
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
    stream_audio = np.concatenate(stream_parts).astype(np.float32)
    return stream_audio, gt_events


def run_pipeline(
    stream: np.ndarray,
    segmenter: Any,
    inferencer: VietnameseKWSInferencer,
    chunk_size: int = 320
) -> List[Dict[str, Any]]:
    segmenter.reset()
    events = []
    total_samples = 0

    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        total_samples += len(chunk)
        res = segmenter.process_chunk(chunk)
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


def evaluate_stream(
    detected: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    tolerance_sec: float = 0.65
) -> Dict[str, Any]:
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detected)
    latencies = []
    tp = 0
    fp = 0
    duplicates = 0

    for d_idx, det in enumerate(detected):
        d_cmd = det["command"]
        d_time = det["timestamp_sec"]
        best_g_idx = None
        best_dist = 999.0

        for g_idx, gt in enumerate(ground_truth):
            if gt["command"] == d_cmd:
                dist = abs(d_time - gt["end_sec"])
                if dist <= tolerance_sec:
                    if not gt_matched[g_idx] and dist < best_dist:
                        best_dist = dist
                        best_g_idx = g_idx

        if best_g_idx is not None:
            gt_matched[best_g_idx] = True
            det_matched[d_idx] = True
            tp += 1
            lat = max(0.0, (d_time - ground_truth[best_g_idx]["end_sec"]) * 1000.0)
            latencies.append(lat)
        else:
            # Check if duplicate trigger of already matched GT
            is_dup = False
            for g_idx, gt in enumerate(ground_truth):
                if gt["command"] == d_cmd and gt_matched[g_idx]:
                    if abs(d_time - gt["end_sec"]) <= (tolerance_sec * 1.5):
                        is_dup = True
                        break
            if is_dup:
                duplicates += 1
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    det_cmds = [d["command"] for d in detected]
    gt_cmds = [g["command"] for g in ground_truth]
    exact_match = (det_cmds == gt_cmds)

    return {
        "tp": tp, "fp": fp, "fn": fn, "duplicates": duplicates,
        "exact_match": exact_match, "latencies": latencies,
        "det_cmds": det_cmds, "gt_cmds": gt_cmds
    }


def main():
    print("=" * 80)
    print("PHASE 3 — EXTENDED LONG-SEQUENCE BENCHMARK (N = 3, 5, 10, 20)")
    print("Comparing: legacy_v11 vs continuous_valley")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    profiles = {
        "legacy_v11": UtteranceSegmenter(silence_interval_ms=300, preroll_ms=150, energy_threshold_db=-48.0),
        "continuous_valley": AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0)
    }

    lengths = [3, 5, 10, 20]
    patterns = [
        "repetitive_identical",
        "repeating_prefix_distinct",
        "alternating_distinct",
        "mixed_gameplay"
    ]
    gaps = [0, 50, 100, 200]

    raw_trials = []
    stats_by_length_profile = {
        l: {p_name: {
            "tp": 0, "fp": 0, "fn": 0, "duplicates": 0, "total_trials": 0, "exact_matches": 0,
            "latencies": []
        } for p_name in profiles}
        for l in lengths
    }

    total_runs = len(lengths) * len(patterns) * len(gaps)
    print(f"Running {total_runs} sequence trials per profile ({total_runs * 2} runs total)...")

    for l in lengths:
        print(f"\nEvaluating Sequence Length N = {l} commands...")
        for pat in patterns:
            for gap in gaps:
                stream, gt = construct_long_sequence(l, pat, gap, seed_offset=l + gap)

                for p_name, seg in profiles.items():
                    dets = run_pipeline(stream, seg, inferencer)
                    ev = evaluate_stream(dets, gt)

                    st = stats_by_length_profile[l][p_name]
                    st["tp"] += ev["tp"]
                    st["fp"] += ev["fp"]
                    st["fn"] += ev["fn"]
                    st["duplicates"] += ev["duplicates"]
                    st["total_trials"] += 1
                    if ev["exact_match"]:
                        st["exact_matches"] += 1
                    st["latencies"].extend(ev["latencies"])

                    raw_trials.append({
                        "profile": p_name, "length_n": l, "pattern": pat, "gap_ms": gap,
                        "exact_match": ev["exact_match"], "tp": ev["tp"], "fp": ev["fp"],
                        "fn": ev["fn"], "duplicates": ev["duplicates"],
                        "gt_count": len(gt), "det_count": len(dets),
                        "det_cmds": ev["det_cmds"], "gt_cmds": ev["gt_cmds"]
                    })

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'Length N':<10} | {'Profile':<18} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'F1 (%)':<9} | {'SeqMatch':<9} | {'DupRate':<8} | {'Lat p50':<8}")
    print("-" * 105)

    summary_rows = []
    for l in lengths:
        for p_name in ["legacy_v11", "continuous_valley"]:
            st = stats_by_length_profile[l][p_name]
            prec = (st["tp"] / (st["tp"] + st["fp"]) * 100.0) if (st["tp"] + st["fp"]) > 0 else 0.0
            rec = (st["tp"] / (st["tp"] + st["fn"]) * 100.0) if (st["tp"] + st["fn"]) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            seq_match = (st["exact_matches"] / st["total_trials"] * 100.0) if st["total_trials"] > 0 else 0.0
            tot_dets = st["tp"] + st["fp"]
            dup_rate = (st["duplicates"] / tot_dets * 100.0) if tot_dets > 0 else 0.0
            p50 = float(np.median(st["latencies"])) if len(st["latencies"]) > 0 else 0.0
            p95 = float(np.percentile(st["latencies"], 95)) if len(st["latencies"]) > 0 else 0.0

            print(f"{l:<10} | {p_name:<18} | {prec:8.2f}% | {rec:8.2f}% | {f1:8.2f}% | {seq_match:8.2f}% | {dup_rate:7.2f}% | {p50:6.1f}ms")

            summary_rows.append({
                "sequence_length_n": l,
                "profile": p_name,
                "event_precision": round(prec, 2),
                "event_recall": round(rec, 2),
                "event_f1": round(f1, 2),
                "exact_sequence_match_pct": round(seq_match, 2),
                "duplicate_rate_pct": round(dup_rate, 2),
                "latency_p50_ms": round(p50, 1),
                "latency_p95_ms": round(p95, 1),
                "tp": st["tp"], "fp": st["fp"], "fn": st["fn"], "duplicates": st["duplicates"],
                "total_trials": st["total_trials"], "exact_matches": st["exact_matches"]
            })

    print("=" * 105)

    # Save CSV and JSON
    csv_path = REPORTS_DIR / "long_sequence_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence_length_n", "profile", "event_precision", "event_recall", "event_f1",
            "exact_sequence_match_pct", "duplicate_rate_pct", "latency_p50_ms", "latency_p95_ms",
            "tp", "fp", "fn", "duplicates", "total_trials", "exact_matches"
        ])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    json_path = REPORTS_DIR / "long_sequence_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "lengths": lengths,
                "patterns": patterns,
                "gaps_ms": gaps,
                "trials_per_length": len(patterns) * len(gaps)
            },
            "summary": summary_rows,
            "raw_trials": raw_trials
        }, f, indent=2)

    print(f"\nArtifacts saved successfully:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
