"""
Negative-Control Audit for Vietnamese Keyword Spotting (KWS).
Phase 3: Systematic comparative audit between legacy_v11 and continuous_valley.

Evaluates:
1. Digital Silence (5.0 minutes total across 10 streams)
2. Environmental Noise (100 files, v13_conf_bg_*.wav)
3. Vietnamese Conversational Speech (300 files, v13_conf_conv_*.wav)
4. Out-of-Vocabulary Foils / Confusable Syllables (200 files, v13_conf_foil_*.wav)
5. Elongated Target Keywords (20 streams across all 4 commands, 750-1000ms duration)

Ensures every clip is processed completely with required trailing silence for proper segment finalization.
Records every false activation, timestamp, class, and confidence into CSV and JSON.
Computes False Activation Rate per minute (FAR / min).
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


def build_elongated_clip(cmd_name: str, idx: int, target_dur_ms: int = 800) -> np.ndarray:
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


def process_audio_stream(
    stream: np.ndarray,
    segmenter: Any,
    inferencer: VietnameseKWSInferencer,
    chunk_size: int = 320
) -> List[Dict[str, Any]]:
    """Feeds an audio stream in 20ms chunks into segmenter and runs inference on finalized segments."""
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
                    "latency_ms": round(dt_ms, 2),
                    "probabilities": {CLASSES[c_i]: round(float(probs[c_i]), 4) for c_i in range(5)}
                })

    return events


def run_negative_audit():
    print("=" * 80)
    print("PHASE 3 — SYSTEMATIC NEGATIVE-CONTROL AUDIT")
    print("Comparing: legacy_v11 (Baseline Segmented) vs continuous_valley (Acoustic Valley)")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    # Segmenters under comparison
    profiles = {
        "legacy_v11": UtteranceSegmenter(silence_interval_ms=300, preroll_ms=150, energy_threshold_db=-48.0),
        "continuous_valley": AcousticValleySegmenter(silence_interval_ms=140, preroll_ms=120, dip_threshold_db=12.0, rebound_threshold_db=6.0)
    }

    # Data collection containers
    all_events_log: List[Dict[str, Any]] = []
    group_stats: Dict[str, Dict[str, Any]] = {}

    # Define test groups
    groups = [
        "digital_silence",
        "environmental_noise",
        "conversational_speech",
        "oov_phonetic_foils",
        "elongated_target_words"
    ]

    for g in groups:
        group_stats[g] = {
            "legacy_v11": {"audio_sec": 0.0, "clips": 0, "false_activations": 0, "by_class": {c: 0 for c in TARGET_CLASSES}},
            "continuous_valley": {"audio_sec": 0.0, "clips": 0, "false_activations": 0, "by_class": {c: 0 for c in TARGET_CLASSES}}
        }

    # Trailing silence padding to ensure complete segment processing for each clip
    # 500ms trailing silence is sufficient for both 140ms and 300ms silence segmenters
    trail_silence = np.zeros(8000, dtype=np.float32)
    lead_silence = np.zeros(1600, dtype=np.float32)  # 100ms lead

    # --- Group 1: Digital Silence ---
    print("\n[1/5] Auditing Digital Silence (10 streams x 30.0s = 300.0s / 5.0 min)...")
    for s_idx in range(10):
        dur_s = 30.0
        stream = np.zeros(int(dur_s * 16000), dtype=np.float32)

        for p_name, seg in profiles.items():
            evs = process_audio_stream(stream, seg, inferencer)
            group_stats["digital_silence"][p_name]["audio_sec"] += dur_s
            group_stats["digital_silence"][p_name]["clips"] += 1
            group_stats["digital_silence"][p_name]["false_activations"] += len(evs)
            for ev in evs:
                group_stats["digital_silence"][p_name]["by_class"][ev["command"]] += 1
                all_events_log.append({
                    "profile": p_name, "group": "digital_silence", "file": f"silence_{s_idx:02d}.raw",
                    "ground_truth": "NONE", "command": ev["command"], "confidence": ev["confidence"],
                    "timestamp_sec": ev["timestamp_sec"], "seg_dur_ms": ev["segment_duration_ms"],
                    "probabilities": ev["probabilities"]
                })

    # --- Group 2: Environmental Noise ---
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))
    print(f"\n[2/5] Auditing Environmental Noise ({len(bg_files)} files)...")
    for f in bg_files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        dur_s = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)

        for p_name, seg in profiles.items():
            evs = process_audio_stream(stream, seg, inferencer)
            group_stats["environmental_noise"][p_name]["audio_sec"] += dur_s
            group_stats["environmental_noise"][p_name]["clips"] += 1
            group_stats["environmental_noise"][p_name]["false_activations"] += len(evs)
            for ev in evs:
                group_stats["environmental_noise"][p_name]["by_class"][ev["command"]] += 1
                all_events_log.append({
                    "profile": p_name, "group": "environmental_noise", "file": f.name,
                    "ground_truth": "NONE", "command": ev["command"], "confidence": ev["confidence"],
                    "timestamp_sec": ev["timestamp_sec"], "seg_dur_ms": ev["segment_duration_ms"],
                    "probabilities": ev["probabilities"]
                })

    # --- Group 3: Vietnamese Conversational Speech ---
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))
    print(f"\n[3/5] Auditing Conversational Speech ({len(conv_files)} files)...")
    for f in conv_files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        dur_s = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)

        for p_name, seg in profiles.items():
            evs = process_audio_stream(stream, seg, inferencer)
            group_stats["conversational_speech"][p_name]["audio_sec"] += dur_s
            group_stats["conversational_speech"][p_name]["clips"] += 1
            group_stats["conversational_speech"][p_name]["false_activations"] += len(evs)
            for ev in evs:
                group_stats["conversational_speech"][p_name]["by_class"][ev["command"]] += 1
                all_events_log.append({
                    "profile": p_name, "group": "conversational_speech", "file": f.name,
                    "ground_truth": "NONE", "command": ev["command"], "confidence": ev["confidence"],
                    "timestamp_sec": ev["timestamp_sec"], "seg_dur_ms": ev["segment_duration_ms"],
                    "probabilities": ev["probabilities"]
                })

    # --- Group 4: Out-of-Vocabulary Foils ---
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))
    print(f"\n[4/5] Auditing OOV Foils & Confusable Syllables ({len(foil_files)} files)...")
    for f in foil_files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        dur_s = len(audio) / 16000.0
        stream = np.concatenate([lead_silence, audio, trail_silence]).astype(np.float32)

        for p_name, seg in profiles.items():
            evs = process_audio_stream(stream, seg, inferencer)
            group_stats["oov_phonetic_foils"][p_name]["audio_sec"] += dur_s
            group_stats["oov_phonetic_foils"][p_name]["clips"] += 1
            group_stats["oov_phonetic_foils"][p_name]["false_activations"] += len(evs)
            for ev in evs:
                group_stats["oov_phonetic_foils"][p_name]["by_class"][ev["command"]] += 1
                all_events_log.append({
                    "profile": p_name, "group": "oov_phonetic_foils", "file": f.name,
                    "ground_truth": "NONE", "command": ev["command"], "confidence": ev["confidence"],
                    "timestamp_sec": ev["timestamp_sec"], "seg_dur_ms": ev["segment_duration_ms"],
                    "probabilities": ev["probabilities"]
                })

    # --- Group 5: Elongated Target Words (Over-splitting Test) ---
    print("\n[5/5] Auditing Elongated Target Words (4 commands x 5 variations = 20 clips)...")
    cmd_names = ["len", "xuong", "trai", "phai"]
    for c_name in cmd_names:
        for idx in range(1, 6):
            target_dur = 750 + idx * 50  # 800ms - 1000ms
            elongated = build_elongated_clip(c_name, idx, target_dur_ms=target_dur)
            dur_s = len(elongated) / 16000.0
            stream = np.concatenate([lead_silence, elongated, trail_silence]).astype(np.float32)

            for p_name, seg in profiles.items():
                evs = process_audio_stream(stream, seg, inferencer)
                group_stats["elongated_target_words"][p_name]["audio_sec"] += dur_s
                group_stats["elongated_target_words"][p_name]["clips"] += 1
                # Here, 1 event of the correct command is expected (True Positive).
                # Any EXTRA event (> 1) is a duplicate false activation due to over-splitting.
                expected_cmd = c_name.upper()
                extra_events = max(0, len(evs) - 1)
                group_stats["elongated_target_words"][p_name]["false_activations"] += extra_events

                for ev in evs:
                    all_events_log.append({
                        "profile": p_name, "group": "elongated_target_words", "file": f"elongated_{c_name}_{idx}.raw",
                        "ground_truth": expected_cmd, "command": ev["command"], "confidence": ev["confidence"],
                        "timestamp_sec": ev["timestamp_sec"], "seg_dur_ms": ev["segment_duration_ms"],
                        "probabilities": ev["probabilities"]
                    })

    # Print Summary Tables
    print("\n" + "=" * 105)
    print(f"{'Test Group':<26} | {'Audio Dur':<12} | {'legacy_v11 FA':<14} | {'legacy FAR/min':<15} | {'valley FA':<11} | {'valley FAR/min':<15}")
    print("-" * 105)

    summary_rows = []
    tot_leg_fa = 0
    tot_val_fa = 0
    tot_audio_s = 0.0

    for g in groups:
        leg_st = group_stats[g]["legacy_v11"]
        val_st = group_stats[g]["continuous_valley"]

        dur_min = leg_st["audio_sec"] / 60.0
        leg_fa = leg_st["false_activations"]
        val_fa = val_st["false_activations"]
        leg_far = leg_fa / dur_min if dur_min > 0 else 0.0
        val_far = val_fa / dur_min if dur_min > 0 else 0.0

        tot_leg_fa += leg_fa
        tot_val_fa += val_fa
        tot_audio_s += leg_st["audio_sec"]

        dur_str = f"{leg_st['audio_sec']:.1f}s ({dur_min:.2f}m)"
        print(f"{g:<26} | {dur_str:<12} | {leg_fa:<14} | {leg_far:11.2f} /min | {val_fa:<11} | {val_far:11.2f} /min")

        summary_rows.append({
            "test_group": g,
            "clips_count": leg_st["clips"],
            "audio_duration_sec": round(leg_st["audio_sec"], 2),
            "audio_duration_min": round(dur_min, 2),
            "legacy_v11_false_activations": leg_fa,
            "legacy_v11_far_per_min": round(leg_far, 2),
            "continuous_valley_false_activations": val_fa,
            "continuous_valley_far_per_min": round(val_far, 2),
            "legacy_by_class": leg_st["by_class"],
            "valley_by_class": val_st["by_class"]
        })

    tot_dur_min = tot_audio_s / 60.0
    tot_leg_far = tot_leg_fa / tot_dur_min if tot_dur_min > 0 else 0.0
    tot_val_far = tot_val_fa / tot_dur_min if tot_dur_min > 0 else 0.0
    print("-" * 105)
    print(f"{'TOTAL AUDIT':<26} | {tot_audio_s:.1f}s ({tot_dur_min:.2f}m) | {tot_leg_fa:<14} | {tot_leg_far:11.2f} /min | {tot_val_fa:<11} | {tot_val_far:11.2f} /min")
    print("=" * 105)

    # Save CSV Summary
    summary_csv = REPORTS_DIR / "negative_control_audit_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test_group", "clips_count", "audio_duration_sec", "audio_duration_min",
            "legacy_v11_false_activations", "legacy_v11_far_per_min",
            "continuous_valley_false_activations", "continuous_valley_far_per_min"
        ])
        writer.writeheader()
        for r in summary_rows:
            r_copy = {k: v for k, v in r.items() if k in writer.fieldnames}
            writer.writerow(r_copy)

    # Save Events CSV
    events_csv = REPORTS_DIR / "negative_control_audit_events.csv"
    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "profile", "group", "file", "ground_truth", "command", "confidence",
            "timestamp_sec", "seg_dur_ms"
        ])
        writer.writeheader()
        for ev in all_events_log:
            ev_copy = {k: v for k, v in ev.items() if k in writer.fieldnames}
            writer.writerow(ev_copy)

    # Save Complete JSON
    results_json = REPORTS_DIR / "negative_control_audit_results.json"
    audit_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_reference": "legacy_v11_hard_negative_weighted_seed42",
            "threshold_theta_cmd": DEFAULT_THRESHOLD,
            "total_audio_sec": round(tot_audio_s, 2),
            "total_audio_min": round(tot_dur_min, 2)
        },
        "summary": summary_rows,
        "totals": {
            "total_audio_min": round(tot_dur_min, 2),
            "legacy_v11_total_fa": tot_leg_fa,
            "legacy_v11_overall_far_per_min": round(tot_leg_far, 2),
            "continuous_valley_total_fa": tot_val_fa,
            "continuous_valley_overall_far_per_min": round(tot_val_far, 2)
        },
        "events_log": all_events_log
    }
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\nArtifacts saved successfully:\n  - {summary_csv}\n  - {events_csv}\n  - {results_json}")


if __name__ == "__main__":
    run_negative_audit()
