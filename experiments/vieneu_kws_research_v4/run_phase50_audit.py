"""
Phase 50: Independent Benchmark and Metric Audit Script.
Workspace: D:\trainidentificationaudio
Target: Independent audit of Phase 49 outputs, 3-hour benchmark, non-overlapping intervals,
waveform hashes, repetition statistics, and reproduce per-class recall.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_v2.evaluate import match_events_and_evaluate

V3_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v3")
V3_DATA = os.path.join(V3_DIR, "data")
V3_STREAMS = os.path.join(V3_DATA, "continuous_streams")
V3_PREDS = os.path.join(V3_DIR, "predictions")
V3_MODELS = os.path.join(V3_DIR, "models", "hard_negative_cascade_v3")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_streams_and_metadata():
    print("--- 1. Auditing Streams, Audio Metadata, and Waveform Hashes ---")
    stream_files = {
        "val_commands": "val_stream_continuous_commands.wav",
        "val_nontarget": "val_stream_continuous_nontarget.wav",
        "test_commands": "test_stream_continuous_commands.wav",
        "test_nontarget_3h": "test_stream_continuous_nontarget_3h.wav"
    }

    info_dict = {}
    for name, fname in stream_files.items():
        wav_p = os.path.join(V3_STREAMS, fname)
        gt_p = os.path.join(V3_STREAMS, fname.replace(".wav", "_ground_truth.json"))
        meta_p = os.path.join(V3_STREAMS, fname.replace(".wav", "_metadata.json"))

        sinfo = sf.info(wav_p)
        with open(gt_p, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        meta_data = {}
        if os.path.exists(meta_p):
            with open(meta_p, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

        fhash = sha256_file(wav_p)
        info_dict[name] = {
            "wav_path": wav_p,
            "samplerate": sinfo.samplerate,
            "channels": sinfo.channels,
            "frames": sinfo.frames,
            "duration_sec": sinfo.duration,
            "duration_hours": sinfo.duration / 3600.0,
            "num_gt_events": len(gt_data),
            "sha256": fhash,
            "meta_reported_dur_sec": meta_data.get("duration_seconds"),
            "meta_reported_dur_hr": meta_data.get("duration_hours"),
            "meta_sha256_match": meta_data.get("sha256") == fhash
        }

    return info_dict


def audit_repetition_and_content_diversity():
    print("--- 2. Auditing 3-Hour Benchmark Repetition & Diversity ---")
    gt_p = os.path.join(V3_STREAMS, "test_stream_continuous_nontarget_3h_ground_truth.json")
    with open(gt_p, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_df = pd.DataFrame(gt_data)

    total_events = len(gt_df)
    unique_sample_ids = gt_df["sample_id"].nunique()
    unique_texts = gt_df["text"].nunique()
    unique_voices = gt_df["voice_id"].nunique()
    confusion_distribution = gt_df["confusion_family"].value_counts().to_dict()

    # Repetition count per sample_id
    repetition_counts = gt_df["sample_id"].value_counts()
    min_rep = int(repetition_counts.min())
    max_rep = int(repetition_counts.max())
    mean_rep = float(repetition_counts.mean())

    # Voice distribution
    voice_dist = gt_df["voice_id"].value_counts().to_dict()

    return {
        "total_speech_events": total_events,
        "unique_synthetic_utterances": unique_sample_ids,
        "unique_non_target_texts": unique_texts,
        "unique_test_voices": unique_voices,
        "repetition_stats": {
            "min_repetitions_per_utterance": min_rep,
            "max_repetitions_per_utterance": max_rep,
            "mean_repetitions_per_utterance": mean_rep,
            "num_passes": total_events // unique_sample_ids
        },
        "confusion_family_distribution": confusion_distribution,
        "voice_distribution": voice_dist
    }


def audit_false_alarm_interval_partitioning():
    print("--- 3. Auditing Non-Overlapping False Alarm Categorization ---")
    # Disentangle event-matching tolerance from error-source categorization
    # Non-overlapping intervals:
    # 1. Active Speech: [onset_sec, end_time_sec]
    # 2. Boundary Margin: [onset_sec - 0.20, onset_sec) OR (end_time_sec, end_time_sec + 0.60]
    # 3. Inter-Utterance Gap: strictly outside any [onset_sec - 0.20, end_time_sec + 0.60]
    gt_p = os.path.join(V3_STREAMS, "test_stream_continuous_nontarget_3h_ground_truth.json")
    with open(gt_p, "r", encoding="utf-8") as f:
        gt_df = pd.DataFrame(json.load(f))

    preds_p = os.path.join(V3_PREDS, "per_event_predictions.csv")
    preds_df = pd.read_csv(preds_p)
    nt_preds = preds_df[preds_df["stream"] == "test_nontarget_3h"]

    results = {}
    candidates = nt_preds["candidate"].unique()

    for cand in candidates:
        c_df = nt_preds[nt_preds["candidate"] == cand]
        results[cand] = {}

        for seed in [42, 43, 44]:
            s_df = c_df[c_df["seed"] == seed]
            total_fps = len(s_df)

            in_active_speech = 0
            in_boundary_margin = 0
            in_gap = 0

            trigger_details = []

            for _, row in s_df.iterrows():
                t = row["trigger_time_sec"]

                # Check strict active speech
                active_hits = gt_df[(gt_df["onset_sec"] <= t) & (t <= gt_df["end_time_sec"])]
                if len(active_hits) > 0:
                    in_active_speech += 1
                    loc = "active_speech"
                    ref = active_hits.iloc[0]
                else:
                    # Check boundary margin
                    margin_hits = gt_df[
                        ((gt_df["onset_sec"] - 0.20 <= t) & (t < gt_df["onset_sec"])) |
                        ((gt_df["end_time_sec"] < t) & (t <= gt_df["end_time_sec"] + 0.60))
                    ]
                    if len(margin_hits) > 0:
                        in_boundary_margin += 1
                        loc = "boundary_margin"
                        ref = margin_hits.iloc[0]
                    else:
                        in_gap += 1
                        loc = "inter_utterance_gap"
                        ref = None

                trigger_details.append({
                    "trigger_time_sec": t,
                    "pred_label": row["pred_label"],
                    "location_category": loc,
                    "matched_utterance_text": ref["text"] if ref is not None else None,
                    "matched_voice": ref["voice_id"] if ref is not None else None,
                    "confusion_family": ref["confusion_family"] if ref is not None else None
                })

            results[cand][f"seed{seed}"] = {
                "total_false_positives": total_fps,
                "in_active_speech": in_active_speech,
                "in_boundary_margin": in_boundary_margin,
                "in_inter_utterance_gap": in_gap,
                "active_speech_pct": round(in_active_speech / total_fps * 100, 2) if total_fps > 0 else 0,
                "boundary_margin_pct": round(in_boundary_margin / total_fps * 100, 2) if total_fps > 0 else 0,
                "gap_pct": round(in_gap / total_fps * 100, 2) if total_fps > 0 else 0
            }

    return results


def audit_reproduce_metrics():
    print("--- 4. Auditing Metric Reproduction from Predictions ---")
    test_summary_p = os.path.join(V3_PREDS, "locked_test_summary.json")
    with open(test_summary_p, "r", encoding="utf-8") as f:
        saved_summary = json.load(f)

    preds_p = os.path.join(V3_PREDS, "per_event_predictions.csv")
    preds_df = pd.read_csv(preds_p)

    cmd_gt_p = os.path.join(V3_STREAMS, "test_stream_continuous_commands_ground_truth.json")
    with open(cmd_gt_p, "r", encoding="utf-8") as f:
        cmd_gt_df = pd.DataFrame(json.load(f))

    nt_gt_p = os.path.join(V3_STREAMS, "test_stream_continuous_nontarget_3h_ground_truth.json")
    with open(nt_gt_p, "r", encoding="utf-8") as f:
        nt_gt_df = pd.DataFrame(json.load(f))

    cmd_info = sf.info(os.path.join(V3_STREAMS, "test_stream_continuous_commands.wav"))
    nt_info = sf.info(os.path.join(V3_STREAMS, "test_stream_continuous_nontarget_3h.wav"))

    recomputed = {}
    mismatches = []

    for cand in ["1_historical_cascade", "2_recall_preserving_stage1", "3_recall_preserving_cascade"]:
        recomputed[cand] = {}
        for seed in [42, 43, 44]:
            c_preds = preds_df[(preds_df["candidate"] == cand) & (preds_df["seed"] == seed) & (preds_df["stream"] == "test_commands")]
            cmd_triggers = c_preds[["trigger_time_sec", "pred_label", "confidence"]].to_dict(orient="records")
            cmd_eval = match_events_and_evaluate(cmd_triggers, cmd_gt_df, cmd_info.duration)

            n_preds = preds_df[(preds_df["candidate"] == cand) & (preds_df["seed"] == seed) & (preds_df["stream"] == "test_nontarget_3h")]
            nt_triggers = n_preds[["trigger_time_sec", "pred_label", "confidence"]].to_dict(orient="records")
            nt_eval = match_events_and_evaluate(nt_triggers, nt_gt_df, nt_info.duration)

            saved_c = saved_summary[cand][f"seed{seed}"]["command_stream"]
            saved_n = saved_summary[cand][f"seed{seed}"]["nontarget_stream"]

            # Check precision match
            if cmd_eval["recall"] != saved_c["recall"]:
                mismatches.append(f"{cand} seed{seed} recall: recomputed {cmd_eval['recall']} vs saved {saved_c['recall']}")
            if nt_eval["false_positives"] != saved_n["false_positives"]:
                mismatches.append(f"{cand} seed{seed} nt_fp: recomputed {nt_eval['false_positives']} vs saved {saved_n['false_positives']}")
            if abs(nt_eval["fa_per_hour"] - saved_n["fa_per_hour"]) > 1e-4:
                mismatches.append(f"{cand} seed{seed} fa_hr: recomputed {nt_eval['fa_per_hour']} vs saved {saved_n['fa_per_hour']}")

            # Per-keyword recall breakdown
            kw_rec = {}
            for kw in ["TRAI", "PHAI", "LEN", "XUONG"]:
                kw_gt = cmd_gt_df[cmd_gt_df["class_label"] == kw]
                hits = 0
                for _, r in kw_gt.iterrows():
                    m = c_preds[(c_preds["pred_label"] == kw) & (r["onset_sec"] - 0.20 <= c_preds["trigger_time_sec"]) & (c_preds["trigger_time_sec"] <= r["end_time_sec"] + 0.60)]
                    if len(m) > 0:
                        hits += 1
                kw_rec[kw] = {"hits": hits, "total": len(kw_gt), "recall": round(hits / len(kw_gt), 4)}

            recomputed[cand][f"seed{seed}"] = {
                "cmd_recall": cmd_eval["recall"],
                "nt_fp": nt_eval["false_positives"],
                "wall_fa_hr": nt_eval["fa_per_hour"],
                "stream_duration_hr": nt_eval["stream_duration_hours"],
                "per_keyword_recall": kw_rec
            }

    return {
        "mismatches": mismatches,
        "is_perfect_reproduction": len(mismatches) == 0,
        "recomputed_summary": recomputed
    }


def main():
    report = {
        "streams_and_audio_metadata": audit_streams_and_metadata(),
        "repetition_and_diversity": audit_repetition_and_content_diversity(),
        "non_overlapping_false_alarm_analysis": audit_false_alarm_interval_partitioning(),
        "metric_reproduction": audit_reproduce_metrics()
    }

    out_p = os.path.join(os.path.dirname(__file__), "audit_findings_phase50.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nPhase 50 Audit Complete. Findings saved to {out_p}")


if __name__ == "__main__":
    main()
