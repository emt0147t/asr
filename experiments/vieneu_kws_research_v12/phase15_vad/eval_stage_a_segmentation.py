"""
Stage A: Segmentation-Only Evaluation Harness (Phase 15).

Evaluates VAD segmenter boundaries against ground-truth reference events
completely independently of any CNN classifier predictions.

Matching Rule:
  1-to-1 Greedy Bipartite Matching based on Temporal IoU >= 0.50.
  One predicted segment can match at most one reference event.
  One reference event can match at most one predicted segment.

Metrics Computed:
  - Event Precision, Recall, F1
  - Merge Count & Rate (Predicted segment overlaps >= 2 reference events by >= 50ms)
  - Split Count & Rate (Reference event covered by >= 2 distinct predicted segments)
  - Missed Event Count & Rate
  - Premature Cut Count & Rate (Predicted segment ends > 80ms before reference end)
  - Segment Count Bias (N_pred - N_gt)
  - Onset & Offset Absolute Error (MAE): Median and P95 in milliseconds
  - Detailed breakdown by nominal gap and acoustic condition.
"""

import sys
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import soundfile as sf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def calculate_temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Calculates temporal Intersection-over-Union between two time intervals [s, e]."""
    inter_s = max(s1, s2)
    inter_e = min(e1, e2)
    intersection = max(0.0, inter_e - inter_s)
    union = max(e1, e2) - min(s1, s2)
    if union <= 1e-9:
        return 0.0
    return intersection / union


def match_events_one_to_one(
    ref_events: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Performs greedy 1-to-1 bipartite matching based on temporal IoU.
    """
    n_ref = len(ref_events)
    n_pred = len(pred_segments)

    # Compute pairwise IoU and overlap durations
    pair_scores = []
    for r_idx, ref in enumerate(ref_events):
        r_start, r_end = ref["start_sec"], ref["end_sec"]
        for p_idx, pred in enumerate(pred_segments):
            p_start, p_end = pred["start_sec"], pred["end_sec"]
            iou = calculate_temporal_iou(r_start, r_end, p_start, p_end)

            # Overlap in seconds
            inter_s = max(r_start, p_start)
            inter_e = min(r_end, p_end)
            overlap_sec = max(0.0, inter_e - inter_s)

            pair_scores.append((iou, overlap_sec, r_idx, p_idx))

    # Sort pairs by highest IoU descending
    pair_scores.sort(key=lambda x: x[0], reverse=True)

    matched_ref = set()
    matched_pred = set()
    matches = []

    for iou, overlap_sec, r_idx, p_idx in pair_scores:
        if iou >= iou_threshold:
            if r_idx not in matched_ref and p_idx not in matched_pred:
                matched_ref.add(r_idx)
                matched_pred.add(p_idx)
                matches.append({
                    "ref_idx": r_idx,
                    "pred_idx": p_idx,
                    "iou": iou,
                    "ref_event": ref_events[r_idx],
                    "pred_segment": pred_segments[p_idx]
                })

    # Detect Merges: A predicted segment overlapping >= 2 distinct reference events by >= 50ms each
    merges = []
    for p_idx, pred in enumerate(pred_segments):
        p_start, p_end = pred["start_sec"], pred["end_sec"]
        overlapped_refs = []
        for r_idx, ref in enumerate(ref_events):
            inter_s = max(ref["start_sec"], p_start)
            inter_e = min(ref["end_sec"], p_end)
            overlap = max(0.0, inter_e - inter_s)
            if overlap >= 0.050:  # >= 50ms overlap
                overlapped_refs.append(r_idx)
        if len(overlapped_refs) >= 2:
            merges.append({
                "pred_idx": p_idx,
                "pred_segment": pred,
                "merged_ref_indices": overlapped_refs,
                "count": len(overlapped_refs)
            })

    # Detect Splits: A reference event covered by >= 2 distinct predicted segments (>= 50ms each)
    splits = []
    for r_idx, ref in enumerate(ref_events):
        r_start, r_end = ref["start_sec"], ref["end_sec"]
        overlapping_preds = []
        for p_idx, pred in enumerate(pred_segments):
            inter_s = max(r_start, pred["start_sec"])
            inter_e = min(r_end, pred["end_sec"])
            overlap = max(0.0, inter_e - inter_s)
            if overlap >= 0.050:
                overlapping_preds.append(p_idx)
        if len(overlapping_preds) >= 2:
            splits.append({
                "ref_idx": r_idx,
                "ref_event": ref,
                "split_pred_indices": overlapping_preds,
                "count": len(overlapping_preds)
            })

    # Premature cuts: Matched pairs where pred ends > 80ms before ref ends
    premature_cuts = []
    onset_errors_ms = []
    offset_errors_ms = []

    for m in matches:
        ref = m["ref_event"]
        pred = m["pred_segment"]

        onset_err = abs(pred["start_sec"] - ref["start_sec"]) * 1000.0
        offset_err = abs(pred["end_sec"] - ref["end_sec"]) * 1000.0
        onset_errors_ms.append(onset_err)
        offset_errors_ms.append(offset_err)

        if pred["end_sec"] < ref["end_sec"] - 0.080:  # cut early by > 80ms
            premature_cuts.append({
                "ref_idx": m["ref_idx"],
                "pred_idx": m["pred_idx"],
                "truncation_ms": (ref["end_sec"] - pred["end_sec"]) * 1000.0,
                "ref_event": ref
            })

    # Spurious / False Alarms: Predicted segments that did not match any reference event
    spurious = [pred_segments[p_idx] for p_idx in range(n_pred) if p_idx not in matched_pred]

    return {
        "n_ref": n_ref,
        "n_pred": n_pred,
        "n_matched": len(matches),
        "n_missed": n_ref - len(matched_ref),
        "n_merges": len(merges),
        "merged_event_count": sum(m["count"] for m in merges),
        "n_splits": len(splits),
        "n_premature_cuts": len(premature_cuts),
        "n_spurious": len(spurious),
        "matches": matches,
        "merges": merges,
        "splits": splits,
        "premature_cuts": premature_cuts,
        "spurious": spurious,
        "onset_errors_ms": onset_errors_ms,
        "offset_errors_ms": offset_errors_ms
    }


def evaluate_segmenter_on_corpus(
    segmenter,
    manifest_csv: Path,
    iou_threshold: float = 0.50,
    chunk_samples: int = 320
) -> Dict[str, Any]:
    df = pd.read_csv(manifest_csv)
    cfg = segmenter.get_config()
    algo_name = cfg["algorithm"]
    split_name = df["corpus_split"].iloc[0]

    print(f"\n[Stage A Evaluation] Running {algo_name} on {split_name.upper()} Corpus ({len(df)} sequences)...")

    per_sequence_results = []
    total_ref_events = 0
    total_pred_segments = 0
    total_matched = 0
    total_missed = 0
    total_merges = 0
    total_merged_events = 0
    total_splits = 0
    total_premature_cuts = 0
    total_spurious = 0

    all_onset_errors = []
    all_offset_errors = []

    # Breakdowns
    gap_stats = {gap: {"ref": 0, "pred": 0, "matched": 0, "merges": 0, "cuts": 0} for gap in [100, 150, 200, 250, 300, 400]}
    cond_stats = {cond: {"ref": 0, "pred": 0, "matched": 0, "merges": 0, "cuts": 0} for cond in ["clean", "snr20", "snr15", "snr10", "snr05"]}

    for idx, row in df.iterrows():
        seq_id = row["sequence_id"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        ref_events = json.loads(row["events_json"])
        nominal_gap = int(row["nominal_gap_ms"])
        cond = row["condition"]

        audio, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == 16000

        segmenter.reset()
        pred_segments = []

        # Replay chunk-by-chunk
        for c_start in range(0, len(audio), chunk_samples):
            chunk = audio[c_start : c_start + chunk_samples]
            chunk_time = c_start / sr
            seg = segmenter.process_chunk(chunk, chunk_time)
            if seg is not None:
                pred_segments.append(seg)

        match_res = match_events_one_to_one(ref_events, pred_segments, iou_threshold=iou_threshold)

        total_ref_events += match_res["n_ref"]
        total_pred_segments += match_res["n_pred"]
        total_matched += match_res["n_matched"]
        total_missed += match_res["n_missed"]
        total_merges += match_res["n_merges"]
        total_merged_events += match_res["merged_event_count"]
        total_splits += match_res["n_splits"]
        total_premature_cuts += match_res["n_premature_cuts"]
        total_spurious += match_res["n_spurious"]

        all_onset_errors.extend(match_res["onset_errors_ms"])
        all_offset_errors.extend(match_res["offset_errors_ms"])

        gap_stats[nominal_gap]["ref"] += match_res["n_ref"]
        gap_stats[nominal_gap]["pred"] += match_res["n_pred"]
        gap_stats[nominal_gap]["matched"] += match_res["n_matched"]
        gap_stats[nominal_gap]["merges"] += match_res["n_merges"]
        gap_stats[nominal_gap]["cuts"] += match_res["n_premature_cuts"]

        cond_stats[cond]["ref"] += match_res["n_ref"]
        cond_stats[cond]["pred"] += match_res["n_pred"]
        cond_stats[cond]["matched"] += match_res["n_matched"]
        cond_stats[cond]["merges"] += match_res["n_merges"]
        cond_stats[cond]["cuts"] += match_res["n_premature_cuts"]

        per_sequence_results.append({
            "sequence_id": seq_id,
            "nominal_gap_ms": nominal_gap,
            "condition": cond,
            "n_ref": match_res["n_ref"],
            "n_pred": match_res["n_pred"],
            "n_matched": match_res["n_matched"],
            "n_missed": match_res["n_missed"],
            "n_merges": match_res["n_merges"],
            "n_premature_cuts": match_res["n_premature_cuts"],
            "pred_segments": [
                {"start_sec": p["start_sec"], "end_sec": p["end_sec"], "duration_sec": p["duration_sec"]}
                for p in pred_segments
            ]
        })

    # Summary metrics
    precision = total_matched / max(1, total_pred_segments)
    recall = total_matched / max(1, total_ref_events)
    f1 = (2.0 * precision * recall) / max(1e-9, precision + recall)

    merge_rate = total_merged_events / max(1, total_ref_events)
    split_rate = total_splits / max(1, total_ref_events)
    miss_rate = total_missed / max(1, total_ref_events)
    premature_cut_rate = total_premature_cuts / max(1, total_matched)

    median_onset_mae = float(np.median(all_onset_errors)) if all_onset_errors else 0.0
    p95_onset_mae = float(np.percentile(all_onset_errors, 95)) if all_onset_errors else 0.0

    median_offset_mae = float(np.median(all_offset_errors)) if all_offset_errors else 0.0
    p95_offset_mae = float(np.percentile(all_offset_errors, 95)) if all_offset_errors else 0.0

    summary = {
        "algorithm": algo_name,
        "config": cfg,
        "corpus_split": split_name,
        "manifest_path": str(manifest_csv),
        "total_sequences": len(df),
        "total_ref_events": total_ref_events,
        "total_pred_segments": total_pred_segments,
        "total_matched": total_matched,
        "total_missed": total_missed,
        "total_merges": total_merges,
        "total_merged_events": total_merged_events,
        "total_splits": total_splits,
        "total_premature_cuts": total_premature_cuts,
        "total_spurious": total_spurious,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "merge_rate": round(merge_rate, 4),
        "split_rate": round(split_rate, 4),
        "miss_rate": round(miss_rate, 4),
        "premature_cut_rate": round(premature_cut_rate, 4),
        "segment_count_bias": total_pred_segments - total_ref_events,
        "median_onset_mae_ms": round(median_onset_mae, 2),
        "p95_onset_mae_ms": round(p95_onset_mae, 2),
        "median_offset_mae_ms": round(median_offset_mae, 2),
        "p95_offset_mae_ms": round(p95_offset_mae, 2),
        "gap_breakdown": gap_stats,
        "condition_breakdown": cond_stats,
        "per_sequence_results": per_sequence_results
    }

    print("-" * 70)
    print(f"  {algo_name} Stage A Results:")
    print(f"    Event Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1: {f1:.4f}")
    print(f"    Merges: {total_merges} (Affected events: {total_merged_events}/{total_ref_events} = {merge_rate * 100:.2f}%)")
    print(f"    Missed Events: {total_missed}/{total_ref_events} ({miss_rate * 100:.2f}%)")
    print(f"    Premature Cuts: {total_premature_cuts}/{total_matched} ({premature_cut_rate * 100:.2f}%)")
    print(f"    Onset MAE: Median={median_onset_mae:.1f}ms, P95={p95_onset_mae:.1f}ms")
    print(f"    Offset MAE: Median={median_offset_mae:.1f}ms, P95={p95_offset_mae:.1f}ms")
    print("-" * 70)

    return summary


def main():
    from vad_segmenters import BaselineFixedSilenceVAD, DualThresholdAdaptiveVAD, SpeechBandRatioVAD

    manifest_dev = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests" / "vad_dev_corpus_manifest.csv"

    print("=" * 80)
    print("  PHASE 15: STAGE A — SEGMENTATION-ONLY EVALUATION (DEV CORPUS)")
    print("=" * 80)

    candidates = [
        BaselineFixedSilenceVAD(),
        DualThresholdAdaptiveVAD(),
        SpeechBandRatioVAD()
    ]

    all_stage_a_dev = {}
    for cand in candidates:
        cfg = cand.get_config()
        name = cfg["algorithm"]
        res = evaluate_segmenter_on_corpus(cand, manifest_dev)
        all_stage_a_dev[name] = res

    # Save summary JSON
    out_json = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports" / "stage_a_dev_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_stage_a_dev, f, indent=2, ensure_ascii=False)
    print(f"\nSaved Stage A Dev summary: {out_json}")


if __name__ == "__main__":
    main()
