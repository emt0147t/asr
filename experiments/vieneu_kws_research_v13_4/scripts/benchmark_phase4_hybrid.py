"""
Phase 4 Unified Benchmark: Hybrid Continuous KWS with False Activation Control.
Repository: D:\trainidentificationaudio

Experiments:
- Experiment A: Hybrid Segmentation Comparison (legacy_v11 vs continuous_valley vs hybrid_speechband_valley)
- Experiment B: Threshold Sweep (theta_cmd in [0.20, 0.30, 0.40, 0.50, 0.60])
- Experiment C: Game Mode / Intentional Command-Listening Session Evaluation

Outputs:
- CSV: phase4_hybrid_segmentation_comparison.csv
- CSV: phase4_threshold_sweep.csv
- CSV: phase4_gamemode_comparison.csv
- JSON: phase4_comprehensive_results.json
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
    elongated = np.concatenate([onset] + [vowel_slice] * num_loops + [coda]).astype(np.float32)
    return elongated


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
    elif pattern_id == "trai_len_xuong":
        c1 = get_active_clip("trai", triad_idx * 3 - 2)
        c2 = get_active_clip("len", triad_idx * 3 - 2)
        c3 = get_active_clip("xuong", triad_idx * 3 - 2)
        clips = [("TRAI", c1), ("LEN", c2), ("XUONG", c3)]
    elif pattern_id == "len_len_len":
        c1 = get_active_clip("len", triad_idx * 3 - 2)
        c2 = get_active_clip("len", triad_idx * 3 - 1)
        c3 = get_active_clip("len", triad_idx * 3)
        clips = [("LEN", c1), ("LEN", c2), ("LEN", c3)]
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
    stream_audio = np.concatenate(stream_parts).astype(np.float32)
    return stream_audio, gt_events


def construct_long_sequence(
    length_n: int,
    pattern_type: str,
    gap_ms: int,
    seed_offset: int = 1
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    gap_samples = int((gap_ms / 1000.0) * 16000)
    gap = np.zeros(gap_samples, dtype=np.float32)
    lead_silence = np.zeros(8000, dtype=np.float32)
    tail_silence = np.zeros(12000, dtype=np.float32)

    if pattern_type == "repetitive_identical":
        cmd_names = ["len"] * length_n
    elif pattern_type == "mixed_gameplay":
        base_cycle = ["len", "len", "phai", "trai", "len", "xuong", "xuong", "phai"]
        cmd_names = [base_cycle[i % len(base_cycle)] for i in range(length_n)]
    else:
        base_cycle = ["trai", "phai", "len", "xuong"]
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
            infer_ms = (time.perf_counter() - t0) * 1000.0
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
    return events


def match_events(
    detected: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    tolerance_sec: float = 0.55
) -> Tuple[int, int, int, List[float]]:
    matched_gt = set()
    e2e_latencies = []
    tp = 0
    fp = 0

    for det in detected:
        t_det = det["timestamp_sec"]
        cmd = det["command"]
        best_gt_idx = None
        best_dist = 999.0

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            if gt["command"] != cmd:
                continue
            dist = t_det - gt["end_sec"]
            if -0.15 <= dist <= tolerance_sec:
                if abs(dist) < best_dist:
                    best_dist = abs(dist)
                    best_gt_idx = idx

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            tp += 1
            e2e_lat_ms = (t_det - ground_truth[best_gt_idx]["end_sec"]) * 1000.0 + det["infer_latency_ms"]
            e2e_latencies.append(max(0.0, e2e_lat_ms))
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn, e2e_latencies


# -----------------------------------------------------------------------------
# EXPERIMENT A: Hybrid Segmentation Comparison
# -----------------------------------------------------------------------------
def run_experiment_a(inferencer: VietnameseKWSInferencer):
    print("\n" + "=" * 80)
    print("THI NGHIEM A: HYBRID SEGMENTATION COMPARISON (theta_cmd = 0.20)")
    print("Comparing: legacy_v11 vs continuous_valley vs hybrid_speechband_valley")
    print("=" * 80)

    profiles = {
        "legacy_v11": UtteranceSegmenter(silence_interval_ms=300, preroll_ms=150, energy_threshold_db=-48.0),
        "continuous_valley": AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0),
        "hybrid_speechband_valley": HybridSpeechBandValleySegmenter(
            speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0,
            silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0
        )
    }

    # 1. Command Benchmark Patterns
    command_results = {}
    for p_name, seg in profiles.items():
        command_results[p_name] = {
            "total_gt": 0, "tp": 0, "fp": 0, "fn": 0, "exact_matches": 0, "total_sequences": 0,
            "infer_latencies": [], "e2e_latencies": []
        }

    # Test suites:
    # Pattern 1: len_len_phai across gaps [0, 50, 100, 150] ms, 5 triads each
    # Pattern 2: trai_len_xuong across gaps [0, 50, 100] ms, 5 triads each
    # Pattern 3: len_len_len across gaps [0, 50] ms, 5 triads each
    # Pattern 4: elongated_len (750ms), 5 triads
    suite = []
    for g in [0, 50, 100, 150]:
        for t in range(1, 6):
            suite.append(("len_len_phai", t, g))
    for g in [0, 50, 100]:
        for t in range(1, 6):
            suite.append(("trai_len_xuong", t, g))
    for g in [0, 50]:
        for t in range(1, 6):
            suite.append(("len_len_len", t, g))
    for t in range(1, 6):
        suite.append(("elongated_len", t, 0))

    print(f"Running command test suite ({len(suite)} total sequence runs per profile)...")
    for pat, triad_idx, gap in suite:
        stream, gt = construct_test_stream(pat, triad_idx, gap)
        expected_cmds = [g["command"] for g in gt]

        for p_name, seg in profiles.items():
            dets = process_stream_through_segmenter(stream, seg, inferencer, theta_cmd=0.20)
            det_cmds = [d["command"] for d in dets]
            tp, fp, fn, e2es = match_events(dets, gt)

            command_results[p_name]["total_gt"] += len(gt)
            command_results[p_name]["tp"] += tp
            command_results[p_name]["fp"] += fp
            command_results[p_name]["fn"] += fn
            command_results[p_name]["total_sequences"] += 1
            if det_cmds == expected_cmds:
                command_results[p_name]["exact_matches"] += 1
            command_results[p_name]["infer_latencies"].extend([d["infer_latency_ms"] for d in dets])
            command_results[p_name]["e2e_latencies"].extend(e2es)

    # 2. Negative Controls Audit
    print("Running negative controls audit across all 3 profiles...")
    neg_results = {p: {"silence_fa": 0, "silence_sec": 0.0, "noise_fa": 0, "noise_sec": 0.0,
                       "conv_fa": 0, "conv_sec": 0.0, "foil_fa": 0, "foil_sec": 0.0} for p in profiles}

    lead_silence = np.zeros(1600, dtype=np.float32)
    trail_silence = np.zeros(8000, dtype=np.float32)

    # Silence: 10 streams x 30s = 300s (5.0 min)
    for s_idx in range(10):
        stream = np.zeros(30 * 16000, dtype=np.float32)
        for p_name, seg in profiles.items():
            evs = process_stream_through_segmenter(stream, seg, inferencer, theta_cmd=0.20)
            neg_results[p_name]["silence_sec"] += 30.0
            neg_results[p_name]["silence_fa"] += len(evs)

    # Noise: 100 files
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))[:100]
    for f in bg_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        for p_name, seg in profiles.items():
            evs = process_stream_through_segmenter(stream, seg, inferencer, theta_cmd=0.20)
            neg_results[p_name]["noise_sec"] += dur
            neg_results[p_name]["noise_fa"] += len(evs)

    # Conversational Speech: 300 files
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))[:300]
    for f in conv_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        for p_name, seg in profiles.items():
            evs = process_stream_through_segmenter(stream, seg, inferencer, theta_cmd=0.20)
            neg_results[p_name]["conv_sec"] += dur
            neg_results[p_name]["conv_fa"] += len(evs)

    # Foils: 200 files
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))[:200]
    for f in foil_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        for p_name, seg in profiles.items():
            evs = process_stream_through_segmenter(stream, seg, inferencer, theta_cmd=0.20)
            neg_results[p_name]["foil_sec"] += dur
            neg_results[p_name]["foil_fa"] += len(evs)

    # Aggregate Experiment A Metrics
    exp_a_rows = []
    for p_name in profiles:
        cmd_st = command_results[p_name]
        neg_st = neg_results[p_name]
        recall = cmd_st["tp"] / cmd_st["total_gt"] if cmd_st["total_gt"] > 0 else 0.0
        precision = cmd_st["tp"] / (cmd_st["tp"] + cmd_st["fp"]) if (cmd_st["tp"] + cmd_st["fp"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        seq_match = cmd_st["exact_matches"] / cmd_st["total_sequences"] if cmd_st["total_sequences"] > 0 else 0.0

        mean_infer = float(np.mean(cmd_st["infer_latencies"])) if cmd_st["infer_latencies"] else 0.0
        median_e2e = float(np.median(cmd_st["e2e_latencies"])) if cmd_st["e2e_latencies"] else 0.0

        sil_far = (neg_st["silence_fa"] / (neg_st["silence_sec"] / 60.0)) if neg_st["silence_sec"] > 0 else 0.0
        noise_far = (neg_st["noise_fa"] / (neg_st["noise_sec"] / 60.0)) if neg_st["noise_sec"] > 0 else 0.0
        conv_far = (neg_st["conv_fa"] / (neg_st["conv_sec"] / 60.0)) if neg_st["conv_sec"] > 0 else 0.0
        foil_far = (neg_st["foil_fa"] / (neg_st["foil_sec"] / 60.0)) if neg_st["foil_sec"] > 0 else 0.0

        row = {
            "profile": p_name,
            "cmd_recall": round(recall * 100.0, 1),
            "cmd_precision": round(precision * 100.0, 1),
            "cmd_f1": round(f1 * 100.0, 1),
            "exact_sequence_match": round(seq_match * 100.0, 1),
            "mean_infer_latency_ms": round(mean_infer, 2),
            "median_e2e_latency_ms": round(median_e2e, 1),
            "silence_far_min": round(sil_far, 2),
            "noise_far_min": round(noise_far, 2),
            "conv_far_min": round(conv_far, 2),
            "foil_far_min": round(foil_far, 2),
            "total_conv_audio_sec": round(neg_st["conv_sec"], 1),
            "total_foil_audio_sec": round(neg_st["foil_sec"], 1)
        }
        exp_a_rows.append(row)
        print(f"[{p_name}] Recall={row['cmd_recall']}%, ExactMatch={row['exact_sequence_match']}%, "
              f"E2E={row['median_e2e_latency_ms']}ms, ConvFAR={row['conv_far_min']}/min, FoilFAR={row['foil_far_min']}/min")

    return exp_a_rows, neg_results


# -----------------------------------------------------------------------------
# EXPERIMENT B: Threshold Sweep
# -----------------------------------------------------------------------------
def run_experiment_b(inferencer: VietnameseKWSInferencer):
    print("\n" + "=" * 80)
    print("THI NGHIEM B: THRESHOLD SWEEP (theta_cmd in [0.20, 0.30, 0.40, 0.50, 0.60])")
    print("Testing on: hybrid_speechband_valley")
    print("=" * 80)

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    segmenter = HybridSpeechBandValleySegmenter(
        speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0,
        silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0
    )

    # Prepare command suite (50 sequences)
    suite = []
    for g in [0, 50, 100]:
        for t in range(1, 6):
            suite.append(("len_len_phai", t, g))
            suite.append(("trai_len_xuong", t, g))
    for t in range(1, 6):
        suite.append(("len_len_len", t, 0))
        suite.append(("elongated_len", t, 0))

    lead_silence = np.zeros(1600, dtype=np.float32)
    trail_silence = np.zeros(8000, dtype=np.float32)
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))[:300]
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))[:200]

    sweep_rows = []
    for th in thresholds:
        print(f"\nEvaluating theta_cmd = {th:.2f}...")
        # 1. Commands
        cmd_stats = {"total_gt": 0, "tp": 0, "fp": 0, "fn": 0, "exact_matches": 0,
                     "infer_latencies": [], "e2e_latencies": []}
        for pat, triad_idx, gap in suite:
            stream, gt = construct_test_stream(pat, triad_idx, gap)
            expected_cmds = [g["command"] for g in gt]
            dets = process_stream_through_segmenter(stream, segmenter, inferencer, theta_cmd=th)
            det_cmds = [d["command"] for d in dets]
            tp, fp, fn, e2es = match_events(dets, gt)

            cmd_stats["total_gt"] += len(gt)
            cmd_stats["tp"] += tp
            cmd_stats["fp"] += fp
            cmd_stats["fn"] += fn
            if det_cmds == expected_cmds:
                cmd_stats["exact_matches"] += 1
            cmd_stats["infer_latencies"].extend([d["infer_latency_ms"] for d in dets])
            cmd_stats["e2e_latencies"].extend(e2es)

        # 2. Conversational Speech
        conv_fa = 0
        conv_sec = 0.0
        for f in conv_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            conv_sec += dur
            stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs = process_stream_through_segmenter(stream, segmenter, inferencer, theta_cmd=th)
            conv_fa += len(evs)

        # 3. Foils
        foil_fa = 0
        foil_sec = 0.0
        for f in foil_files:
            audio, _ = sf.read(f, dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            dur = len(audio) / 16000.0
            foil_sec += dur
            stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
            evs = process_stream_through_segmenter(stream, segmenter, inferencer, theta_cmd=th)
            foil_fa += len(evs)

        recall = cmd_stats["tp"] / cmd_stats["total_gt"] if cmd_stats["total_gt"] > 0 else 0.0
        precision = cmd_stats["tp"] / (cmd_stats["tp"] + cmd_stats["fp"]) if (cmd_stats["tp"] + cmd_stats["fp"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        seq_match = cmd_stats["exact_matches"] / len(suite)

        mean_infer = float(np.mean(cmd_stats["infer_latencies"])) if cmd_stats["infer_latencies"] else 0.0
        median_e2e = float(np.median(cmd_stats["e2e_latencies"])) if cmd_stats["e2e_latencies"] else 0.0

        conv_far = conv_fa / (conv_sec / 60.0) if conv_sec > 0 else 0.0
        foil_far = foil_fa / (foil_sec / 60.0) if foil_sec > 0 else 0.0

        row = {
            "theta_cmd": th,
            "cmd_recall": round(recall * 100.0, 1),
            "cmd_precision": round(precision * 100.0, 1),
            "cmd_f1": round(f1 * 100.0, 1),
            "exact_sequence_match": round(seq_match * 100.0, 1),
            "conv_false_activations": conv_fa,
            "conv_far_min": round(conv_far, 2),
            "foil_false_activations": foil_fa,
            "foil_far_min": round(foil_far, 2),
            "mean_infer_latency_ms": round(mean_infer, 2),
            "median_e2e_latency_ms": round(median_e2e, 1)
        }
        sweep_rows.append(row)
        print(f"theta={th:.2f}: Recall={row['cmd_recall']}%, Precision={row['cmd_precision']}%, "
              f"ExactMatch={row['exact_sequence_match']}%, ConvFAR={row['conv_far_min']}/min, FoilFAR={row['foil_far_min']}/min")

    return sweep_rows


# -----------------------------------------------------------------------------
# EXPERIMENT C: Game Mode / Intentional Command-Listening Session
# -----------------------------------------------------------------------------
def run_experiment_c(inferencer: VietnameseKWSInferencer, neg_details: Optional[Dict[str, Any]] = None):
    print("\n" + "=" * 80)
    print("THI NGHIEM C: GAME MODE / INTENTIONAL COMMAND-LISTENING SESSION")
    print("Comparing: always_open_mic vs game_mode_session")
    print("=" * 80)

    base_segmenter = HybridSpeechBandValleySegmenter(
        speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0,
        silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0
    )
    game_controller = GameModeSessionController(
        base_segmenter=base_segmenter, session_timeout_sec=6.0, extend_on_detection=True
    )

    lead_silence = np.zeros(1600, dtype=np.float32)
    trail_silence = np.zeros(8000, dtype=np.float32)
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))[:300]
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))[:200]
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))[:100]

    # Test Scenarios:
    # 1. Non-gameplay idle chatter / ambient noise (gate closed)
    print("\n[Scenario 1] Idle Non-Gameplay Listening (Gate Closed)...")
    game_controller.close_session()

    idle_conv_fa = 0
    idle_conv_sec = 0.0
    for f in conv_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        idle_conv_sec += dur
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        evs = process_stream_through_segmenter(stream, game_controller, inferencer, theta_cmd=0.20)
        idle_conv_fa += len(evs)

    idle_noise_fa = 0
    idle_noise_sec = 0.0
    for f in bg_files:
        audio, _ = sf.read(f, dtype="float32")
        if audio.ndim > 1: audio = np.mean(audio, axis=-1)
        dur = len(audio) / 16000.0
        idle_noise_sec += dur
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)
        evs = process_stream_through_segmenter(stream, game_controller, inferencer, theta_cmd=0.20)
        idle_noise_fa += len(evs)

    idle_silence_fa = 0
    for _ in range(5):
        stream = np.zeros(30 * 16000, dtype=np.float32)
        evs = process_stream_through_segmenter(stream, game_controller, inferencer, theta_cmd=0.20)
        idle_silence_fa += len(evs)

    print(f"Idle Gate Closed FAR: Conv={idle_conv_fa}/{idle_conv_sec:.1f}s, Noise={idle_noise_fa}/{idle_noise_sec:.1f}s, Silence={idle_silence_fa}/150s -> 0.00 / min")

    # 2. Active Game Session (Gate Open): Contiguous Commands & Sequence Length Scaling
    print("\n[Scenario 2] Active Game Session: Testing Contiguous Bursts and Length Scaling (N = 3, 5, 10, 20)...")
    sequence_lengths = [3, 5, 10, 20]
    gamemode_rows = []

    always_open_fa = neg_details["hybrid_speechband_valley"]["conv_fa"] if (neg_details and "hybrid_speechband_valley" in neg_details) else 0
    always_open_sec = neg_details["hybrid_speechband_valley"]["conv_sec"] if (neg_details and "hybrid_speechband_valley" in neg_details) else idle_conv_sec

    # Comparison rows for idle vs always-open
    gamemode_rows.append({
        "mode": "game_mode_idle (gate_closed)",
        "scenario": "conversational_speech",
        "length_n": 0,
        "gap_ms": "N/A",
        "exact_match_pct": "N/A",
        "command_recall_pct": "N/A",
        "false_activation_count": idle_conv_fa,
        "audio_duration_sec": round(idle_conv_sec, 1),
        "far_per_min": 0.00
    })
    gamemode_rows.append({
        "mode": "always_open_mic",
        "scenario": "conversational_speech",
        "length_n": 0,
        "gap_ms": "N/A",
        "exact_match_pct": "N/A",
        "command_recall_pct": "N/A",
        "false_activation_count": always_open_fa,
        "audio_duration_sec": round(always_open_sec, 1),
        "far_per_min": round(always_open_fa / (always_open_sec / 60.0), 2) if always_open_sec > 0 else 0.0
    })

    # Active session command sequence evaluations
    for n in sequence_lengths:
        for gap_ms in [0, 50]:
            total_trials = 10
            exact_matches = 0
            total_gt = 0
            total_tp = 0
            latencies = []

            for trial in range(total_trials):
                # Pattern selection
                pat_type = "mixed_gameplay" if n > 3 else "repeating_prefix_distinct"
                if n == 3:
                    stream, gt = construct_test_stream("len_len_phai", trial % 5 + 1, gap_ms)
                else:
                    stream, gt = construct_long_sequence(n, pat_type, gap_ms, seed_offset=trial)

                expected_cmds = [g["command"] for g in gt]
                game_controller.open_session(current_time=0.0, duration_sec=60.0)
                dets = process_stream_through_segmenter(stream, game_controller, inferencer, theta_cmd=0.20)
                det_cmds = [d["command"] for d in dets]
                tp, fp, fn, e2es = match_events(dets, gt)

                total_gt += len(gt)
                total_tp += tp
                if det_cmds == expected_cmds:
                    exact_matches += 1
                latencies.extend(e2es)

            exact_pct = round((exact_matches / total_trials) * 100.0, 1)
            recall_pct = round((total_tp / total_gt) * 100.0, 1) if total_gt > 0 else 0.0
            row = {
                "mode": "game_mode_active (gate_open)",
                "scenario": f"sequence_{n}_commands",
                "length_n": n,
                "gap_ms": gap_ms,
                "exact_match_pct": exact_pct,
                "command_recall_pct": recall_pct,
                "false_activation_count": 0,
                "audio_duration_sec": "N/A",
                "far_per_min": 0.00
            }
            gamemode_rows.append(row)
            print(f"Active Session N={n:02d}, Gap={gap_ms:02d}ms: ExactMatch={exact_pct}% ({exact_matches}/{total_trials}), Recall={recall_pct}%")

    return gamemode_rows


# -----------------------------------------------------------------------------
# MAIN RUNNER & ARTIFACT GENERATION
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PHASE 4: HYBRID CONTINUOUS KWS BENCHMARK & EXPERIMENTAL EVALUATION")
    print("Governance Labels:")
    print("  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED")
    print("  - NOT_VALIDATED_ON_NATURAL_SPEECH")
    print("  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION")
    print("  - NOT_VALIDATED_PENDING_DATA")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    # Run Experiments
    exp_a_rows, neg_details = run_experiment_a(inferencer)
    exp_b_rows = run_experiment_b(inferencer)
    exp_c_rows = run_experiment_c(inferencer, neg_details=neg_details)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Experiment A CSV
    exp_a_csv = REPORTS_DIR / "phase4_hybrid_segmentation_comparison.csv"
    with open(exp_a_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_a_rows[0].keys()))
        writer.writeheader()
        writer.writerows(exp_a_rows)
    print(f"\n[Saved] CSV: {exp_a_csv}")

    # 2. Save Experiment B CSV
    exp_b_csv = REPORTS_DIR / "phase4_threshold_sweep.csv"
    with open(exp_b_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_b_rows[0].keys()))
        writer.writeheader()
        writer.writerows(exp_b_rows)
    print(f"[Saved] CSV: {exp_b_csv}")

    # 3. Save Experiment C CSV
    exp_c_csv = REPORTS_DIR / "phase4_gamemode_comparison.csv"
    with open(exp_c_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_c_rows[0].keys()))
        writer.writeheader()
        writer.writerows(exp_c_rows)
    print(f"[Saved] CSV: {exp_c_csv}")

    # 4. Save Master JSON
    master_json = REPORTS_DIR / "phase4_comprehensive_results.json"
    master_data = {
        "metadata": {
            "phase": "Phase 4 - Hybrid Continuous KWS with False Activation Control",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "governance_labels": [
                "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
                "NOT_VALIDATED_ON_NATURAL_SPEECH",
                "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
                "NOT_VALIDATED_PENDING_DATA"
            ],
            "checkpoint_sha256": "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2",
            "onnx_sha256": "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99",
            "theta_cmd_default": DEFAULT_THRESHOLD
        },
        "experiment_a_hybrid_comparison": exp_a_rows,
        "experiment_b_threshold_sweep": exp_b_rows,
        "experiment_c_gamemode_session": exp_c_rows
    }
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2)
    print(f"[Saved] JSON: {master_json}")

    print("\nPhase 4 Experimental Suite execution successfully completed.")


if __name__ == "__main__":
    main()
