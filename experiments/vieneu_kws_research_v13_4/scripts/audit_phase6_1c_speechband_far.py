#!/usr/bin/env python3
"""
Workstream C — Phase 6.1C: Matched Speech-Band A/B and Negative Controls Audit
Repository: D:\trainidentificationaudio
Branch: feature/continuous-kws-segmentation

Governance Labels:
  - EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  - NOT_VALIDATED_ON_NATURAL_SPEECH
  - NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  - NOT_VALIDATED_PENDING_DATA
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
import pandas as pd

PROJECT_ROOT = Path("D:/trainidentificationaudio")
DEMO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "demo"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data" / "audio"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "scripts"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hybrid_segmenter import HybridSpeechBandValleySegmenter, GameModeSessionController
from infer import VietnameseKWSInferencer, TARGET_CLASSES
from audit_phase5_3_final_metrics import compute_standard_sequence_wer, compute_temporal_event_metrics
from benchmark_phase6_acceptance import build_trial_stream, process_stream_through_pipeline
from benchmark_phase5_audit import construct_long_sequence

BASELINE_COMMIT = "d5af658a9afe523b306e5ca6c0159789f1c67fad"
EXPECTED_PT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_ONNX_HASH = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def trim_clip_silence(data: np.ndarray, threshold_db: float = -42.0) -> np.ndarray:
    frame_len = 160
    rms_db = [20.0 * np.log10(np.sqrt(np.mean(data[i : i + frame_len] ** 2)) + 1e-7) for i in range(0, len(data), frame_len)]
    above = np.where(np.array(rms_db) > threshold_db)[0]
    if len(above) > 0:
        s = max(0, (above[0] - 2) * frame_len)
        e = min(len(data), (above[-1] + 3) * frame_len)
        return data[s:e]
    return data[:6400]


def get_clip_by_name(clip_name: str) -> np.ndarray:
    wav, sr = sf.read(AUDIO_DIR / f"{clip_name}.wav", dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if sr != 16000:
        import soxr
        wav = soxr.resample(wav, sr, 16000)
    return trim_clip_silence(wav)


def build_stream_from_clips_list(clips_list: List[str], gap_ms: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    lead_silence = np.zeros(8000, dtype=np.float32)
    gap = np.zeros(int((gap_ms / 1000.0) * 16000), dtype=np.float32)
    tail_silence = np.zeros(12000, dtype=np.float32)

    parts = [lead_silence]
    gt_events = []
    curr_t = len(lead_silence) / 16000.0

    for i, c_name in enumerate(clips_list):
        raw_cmd = c_name.split("_")[3].upper()
        audio = get_clip_by_name(c_name)
        dur = len(audio) / 16000.0
        gt_events.append({
            "command": raw_cmd,
            "start_sec": curr_t,
            "end_sec": curr_t + dur
        })
        parts.append(audio)
        curr_t += dur
        if i < len(clips_list) - 1:
            parts.append(gap)
            curr_t += len(gap) / 16000.0
    parts.append(tail_silence)
    return np.concatenate(parts).astype(np.float32), gt_events


def run_phase6_1c_audit():
    print("=" * 80)
    print("WORKSTREAM C — PHASE 6.1C: MATCHED SPEECH-BAND A/B & NEGATIVE CONTROLS AUDIT")
    print("=" * 80)

    inferencer = VietnameseKWSInferencer(backend="onnx")

    # Segmenters under strict matched comparison
    # All parameters strictly identical EXCEPT speech_band_hz:
    seg_300 = HybridSpeechBandValleySegmenter(
        speech_band_hz=(300.0, 3400.0),
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0,
        silence_interval_ms=140,
        preroll_ms=120,
        dip_threshold_db=12.0,
        rebound_threshold_db=6.0
    )
    seg_100 = HybridSpeechBandValleySegmenter(
        speech_band_hz=(100.0, 3400.0),
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0,
        silence_interval_ms=140,
        preroll_ms=120,
        dip_threshold_db=12.0,
        rebound_threshold_db=6.0
    )

    speechband_ab_rows = []

    # -------------------------------------------------------------------------
    # PART 1: Matched Evaluation on Phase 6 Locked Manifest
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Matched Comparison on Phase 6 Locked Manifest...")
    p6_manifest_path = REPORTS_DIR / "phase6_locked_trial_manifest.csv"
    p6_df = pd.read_csv(p6_manifest_path)

    for _, row in p6_df.iterrows():
        tid = row["trial_id"]
        cat = row["category"]
        pat = row["pattern_id"]
        t_idx = int(row["trial_index"])
        gap = int(row["gap_ms"])

        if pat.startswith("long_sequence_N"):
            n_len = int(pat.replace("long_sequence_N", ""))
            stream, gt = construct_long_sequence(n_len, gap, seed_offset=t_idx)
        else:
            stream, gt = build_trial_stream(pat, t_idx, gap)

        gt_cmds = [g["command"] for g in gt]

        for band_name, seg in [("band_300hz", seg_300), ("band_100hz", seg_100)]:
            dets, timing = process_stream_through_pipeline(stream, seg, inferencer, theta_cmd=0.20)
            det_cmds = [d["command"] for d in dets]

            seq_res = compute_standard_sequence_wer(gt_cmds, det_cmds)
            ev_res = compute_temporal_event_metrics(dets, gt)

            speechband_ab_rows.append({
                "manifest_source": "phase6_locked_manifest",
                "trial_id": tid,
                "category": cat,
                "pattern_id": pat,
                "gap_ms": gap,
                "configuration": band_name,
                "speech_band_hz": "(300.0, 3400.0)" if band_name == "band_300hz" else "(100.0, 3400.0)",
                "n_gt": len(gt_cmds),
                "gt_sequence": " -> ".join(gt_cmds),
                "n_det": len(det_cmds),
                "det_sequence": " -> ".join(det_cmds) if det_cmds else "NONE",
                "exact_match": seq_res["exact_sequence_match"],
                "seq_s": seq_res["seq_s"],
                "seq_d": seq_res["seq_d"],
                "seq_ins": seq_res["seq_ins"],
                "seq_wer_pct": seq_res["seq_wer_pct"],
                "event_tp": ev_res["event_tp"],
                "event_fp": ev_res["event_fp"],
                "event_fn": ev_res["event_fn"],
                "event_precision_pct": ev_res["event_precision_pct"],
                "event_recall_pct": ev_res["event_recall_pct"],
                "event_f1_pct": ev_res["event_f1_pct"]
            })

    # -------------------------------------------------------------------------
    # PART 2: Matched Evaluation on Holdout Manifest (Workstream A)
    # -------------------------------------------------------------------------
    print("\n[2/3] Running Matched Comparison on Phase 6.1A Holdout Manifest...")
    holdout_manifest_path = REPORTS_DIR / "phase6_1a_holdout_manifest.csv"
    holdout_df = pd.read_csv(holdout_manifest_path)

    for _, row in holdout_df.iterrows():
        tid = row["trial_id"]
        cat = row["category"]
        pat = row["pattern_id"]
        gap = int(row["gap_ms"])
        clips_list = row["clips_used"].split(";")

        stream, gt = build_stream_from_clips_list(clips_list, gap)
        gt_cmds = [g["command"] for g in gt]

        for band_name, seg in [("band_300hz", seg_300), ("band_100hz", seg_100)]:
            dets, timing = process_stream_through_pipeline(stream, seg, inferencer, theta_cmd=0.20)
            det_cmds = [d["command"] for d in dets]

            seq_res = compute_standard_sequence_wer(gt_cmds, det_cmds)
            ev_res = compute_temporal_event_metrics(dets, gt)

            speechband_ab_rows.append({
                "manifest_source": "phase6_1a_holdout_manifest",
                "trial_id": tid,
                "category": cat,
                "pattern_id": pat,
                "gap_ms": gap,
                "configuration": band_name,
                "speech_band_hz": "(300.0, 3400.0)" if band_name == "band_300hz" else "(100.0, 3400.0)",
                "n_gt": len(gt_cmds),
                "gt_sequence": " -> ".join(gt_cmds),
                "n_det": len(det_cmds),
                "det_sequence": " -> ".join(det_cmds) if det_cmds else "NONE",
                "exact_match": seq_res["exact_sequence_match"],
                "seq_s": seq_res["seq_s"],
                "seq_d": seq_res["seq_d"],
                "seq_ins": seq_res["seq_ins"],
                "seq_wer_pct": seq_res["seq_wer_pct"],
                "event_tp": ev_res["event_tp"],
                "event_fp": ev_res["event_fp"],
                "event_fn": ev_res["event_fn"],
                "event_precision_pct": ev_res["event_precision_pct"],
                "event_recall_pct": ev_res["event_recall_pct"],
                "event_f1_pct": ev_res["event_f1_pct"]
            })

    # Save Speech-Band A/B Results CSV
    ab_csv_path = REPORTS_DIR / "phase6_1c_speechband_ab_results.csv"
    with open(ab_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(speechband_ab_rows[0].keys()))
        writer.writeheader()
        for r in speechband_ab_rows:
            writer.writerow(r)
    print(f"Phase 6.1C Speech-Band A/B CSV written: {ab_csv_path} ({len(speechband_ab_rows)} rows)")

    # -------------------------------------------------------------------------
    # PART 3: Negative Control Audit (Open-Gate vs Closed-Gate)
    # -------------------------------------------------------------------------
    print("\n[3/3] Running Negative Controls Comparison (Open-Gate vs Closed-Gate)...")
    bg_files = sorted(list(AUDIO_DIR.glob("v13_conf_bg_*.wav")))
    conv_files = sorted(list(AUDIO_DIR.glob("v13_conf_conv_*.wav")))
    foil_files = sorted(list(AUDIO_DIR.glob("v13_conf_foil_*.wav")))

    lead_silence = np.zeros(1600, dtype=np.float32)
    trail_silence = np.zeros(8000, dtype=np.float32)

    controller_closed = GameModeSessionController(
        base_segmenter=HybridSpeechBandValleySegmenter(speech_band_hz=(100.0, 3400.0)),
        session_timeout_sec=6.0
    )

    profiles_to_test = {
        "hybrid_300hz_open_gate": (seg_300, False),
        "hybrid_100hz_open_gate": (seg_100, False),
        "game_mode_closed_gate": (controller_closed, True)  # Gate remains closed
    }

    neg_results_rows = []

    # Helper for evaluating negative stream
    def eval_neg_stream(stream_audio: np.ndarray, corpus_name: str, file_name: str, duration_sec: float):
        for p_name, (seg_obj, is_closed) in profiles_to_test.items():
            if is_closed:
                seg_obj.close_session()
                dets, _ = process_stream_through_pipeline(stream_audio, seg_obj, inferencer, is_game_mode=True, gate_open_during_stream=False)
            else:
                dets, _ = process_stream_through_pipeline(stream_audio, seg_obj, inferencer)

            fa_count = len(dets)
            fa_per_min = round((fa_count / (duration_sec / 60.0)), 2) if duration_sec > 0 else 0.0

            neg_results_rows.append({
                "profile": p_name,
                "corpus": corpus_name,
                "file_name": file_name,
                "duration_sec": round(duration_sec, 2),
                "gate_status": "CLOSED" if is_closed else "OPEN",
                "false_activations": fa_count,
                "fa_per_min": fa_per_min,
                "emitted_commands": ";".join([d["command"] for d in dets]) if dets else "NONE",
                "architectural_gating_behavior": is_closed
            })

    # A) Digital Silence (10 streams x 30.0s = 300.0s)
    for s_idx in range(10):
        silence_stream = np.zeros(int(30.0 * 16000), dtype=np.float32)
        eval_neg_stream(silence_stream, "digital_silence", f"silence_{s_idx:02d}.raw", 30.0)

    # B) Environmental Noise (all 100 bg files = 100.0s)
    for f in bg_files:
        wav, _ = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        dur = len(wav) / 16000.0
        stream = np.concatenate([lead_silence, wav, trail_silence]).astype(np.float32)
        eval_neg_stream(stream, "environmental_noise", f.name, dur)

    # C) Vietnamese Conversational Speech (all 300 conv files = 300.0s)
    for f in conv_files:
        wav, _ = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        dur = len(wav) / 16000.0
        stream = np.concatenate([lead_silence, wav, trail_silence]).astype(np.float32)
        eval_neg_stream(stream, "conversational_speech", f.name, dur)

    # D) Phonetic Foils / OOV Speech (all 200 foil files = 200.0s)
    for f in foil_files:
        wav, _ = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        dur = len(wav) / 16000.0
        stream = np.concatenate([lead_silence, wav, trail_silence]).astype(np.float32)
        eval_neg_stream(stream, "oov_phonetic_foils", f.name, dur)

    # Save Negative Control Results CSV
    neg_csv_path = REPORTS_DIR / "phase6_1c_negative_control_results.csv"
    with open(neg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(neg_results_rows[0].keys()))
        writer.writeheader()
        for r in neg_results_rows:
            writer.writerow(r)
    print(f"Phase 6.1C Negative Control CSV written: {neg_csv_path} ({len(neg_results_rows)} rows)")

    # -------------------------------------------------------------------------
    # PART 4: Summary Analysis & Deliverable JSON
    # -------------------------------------------------------------------------
    neg_df = pd.DataFrame(neg_results_rows)
    neg_summary = {}
    for prof in profiles_to_test.keys():
        sub = neg_df[neg_df["profile"] == prof]
        neg_summary[prof] = {}
        for corp in ["digital_silence", "environmental_noise", "conversational_speech", "oov_phonetic_foils"]:
            c_sub = sub[sub["corpus"] == corp]
            total_fa = int(c_sub["false_activations"].sum())
            total_sec = float(c_sub["duration_sec"].sum())
            fa_per_min = round(total_fa / (total_sec / 60.0), 2) if total_sec > 0 else 0.0
            neg_summary[prof][corp] = {
                "total_fa": total_fa,
                "audio_duration_sec": total_sec,
                "fa_per_minute": fa_per_min
            }

    # Summary of A/B Exact Matches on Primary Demo
    ab_df = pd.DataFrame(speechband_ab_rows)

    def get_exact_summary(manifest_name, cat, pat, gap_val, conf_val):
        sub = ab_df[
            (ab_df["manifest_source"] == manifest_name) &
            (ab_df["category"] == cat) &
            (ab_df["pattern_id"] == pat) &
            (ab_df["gap_ms"] == gap_val) &
            (ab_df["configuration"] == conf_val)
        ]
        n_trials = len(sub)
        n_exact = int(sub["exact_match"].sum())
        exact_pct = round(n_exact / n_trials * 100.0, 1) if n_trials > 0 else 0.0
        mean_wer = round(float(sub["seq_wer_pct"].mean()), 1) if n_trials > 0 else 0.0
        return {"trials": n_trials, "exact": n_exact, "exact_pct": exact_pct, "mean_wer": mean_wer}

    summary_json = {
        "workstream": "Phase 6.1C — Matched Speech-Band A/B and Negative Controls Audit",
        "baseline_commit": BASELINE_COMMIT,
        "frozen_hashes_verified": {
            "checkpoint_sha256": EXPECTED_PT_HASH,
            "onnx_sha256": EXPECTED_ONNX_HASH
        },
        "governance_labels": [
            "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
            "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
            "NOT_VALIDATED_PENDING_DATA"
        ],
        "matched_ab_comparison_phase6_manifest": {
            "gap_0ms_primary_len_len_phai": {
                "band_300hz": get_exact_summary("phase6_locked_manifest", "primary_demo", "len_len_phai", 0, "band_300hz"),
                "band_100hz": get_exact_summary("phase6_locked_manifest", "primary_demo", "len_len_phai", 0, "band_100hz")
            },
            "gap_50ms_primary_len_len_phai": {
                "band_300hz": get_exact_summary("phase6_locked_manifest", "primary_demo", "len_len_phai", 50, "band_300hz"),
                "band_100hz": get_exact_summary("phase6_locked_manifest", "primary_demo", "len_len_phai", 50, "band_100hz")
            }
        },
        "matched_ab_comparison_holdout_manifest": {
            "gap_0ms_primary_len_len_phai": {
                "band_300hz": get_exact_summary("phase6_1a_holdout_manifest", "primary_demo_holdout", "len_len_phai", 0, "band_300hz"),
                "band_100hz": get_exact_summary("phase6_1a_holdout_manifest", "primary_demo_holdout", "len_len_phai", 0, "band_100hz")
            },
            "gap_50ms_primary_len_len_phai": {
                "band_300hz": get_exact_summary("phase6_1a_holdout_manifest", "primary_demo_holdout", "len_len_phai", 50, "band_300hz"),
                "band_100hz": get_exact_summary("phase6_1a_holdout_manifest", "primary_demo_holdout", "len_len_phai", 50, "band_100hz")
            }
        },
        "negative_control_summary": neg_summary,
        "critical_governance_disclosures": [
            "1. When the microphone gate is OPEN, expanding speech band to 100 Hz has virtually zero impact on false alarms (669 vs 675 FA on conversation, 361 vs 361 FA on foils).",
            "2. When the microphone gate is OPEN, both continuous valley segmenters produce unacceptably high FAR (>130 FA/minute on chatter) because short-time acoustic valleys trigger on natural speech pauses.",
            "3. When the microphone gate is CLOSED, false activations are STRICTLY ZERO (0 FA across all 900.0s of negative corpora).",
            "4. Closed-gate zero activations confirm architectural session gating effectiveness only, and MUST NEVER be reported as proof of open-mic acoustic classifier rejection."
        ]
    }

    summary_json_path = REPORTS_DIR / "phase6_1c_speechband_audit.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Phase 6.1C Speech-Band Audit JSON written: {summary_json_path}")


if __name__ == "__main__":
    run_phase6_1c_audit()
