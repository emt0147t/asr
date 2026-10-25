"""
Stage B: End-to-End Streaming KWS Evaluation Harness (Phase 15).

Connects VAD segmentation outputs directly to the canonical offline frozen inference path.
Evaluates both frozen v11 baseline and E1 models under the exact Phase 13.4 decision contract:
  - 5 classes evaluated simultaneously
  - REJECT participates fully in the argmax
  - Command threshold theta_cmd = 0.20
  - Preprocessing and class mapping unchanged

Metrics Reported:
  - Exact-Sequence Accuracy (100% correct sequence order and commands)
  - Command Precision, Recall, F1
  - Missed Commands & Incorrect Substitutions
  - False Command Activations on OOV speech
  - False Command Activations during Silence / Noise
  - Duplicate Triggers
  - Trigger Latency from Reference Onset (Median / P50 and P95 in ms)
  - False Activations per Minute of Eligible Negative Audio
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
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from eval_kws import KWSEvaluator, TARGET_CLASSES, CLASSES, DEFAULT_THRESHOLD


def calculate_temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter_s = max(s1, s2)
    inter_e = min(e1, e2)
    intersection = max(0.0, inter_e - inter_s)
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 1e-9 else 0.0


def evaluate_streaming_kws(
    segmenter,
    evaluator: KWSEvaluator,
    manifest_csv: Path,
    chunk_samples: int = 320,
    model_name: str = "v11"
) -> Dict[str, Any]:
    df = pd.read_csv(manifest_csv)
    cfg = segmenter.get_config()
    algo_name = cfg["algorithm"]
    split_name = df["corpus_split"].iloc[0]

    print(f"\n[Stage B Evaluation] Running {algo_name} + {model_name.upper()} on {split_name.upper()} Corpus ({len(df)} sequences)...")

    total_target_events = 0
    total_matched_commands = 0
    total_missed_commands = 0
    total_substitutions = 0
    total_oov_false_activations = 0
    total_silence_false_activations = 0
    total_duplicate_triggers = 0
    exact_correct_sequences = 0

    all_latencies_ms = []
    total_negative_seconds = 0.0

    gap_stats = {
        gap: {"target_events": 0, "matched": 0, "exact_seqs": 0, "total_seqs": 0, "fa_total": 0}
        for gap in [100, 150, 200, 250, 300, 400]
    }
    cond_stats = {
        cond: {"target_events": 0, "matched": 0, "exact_seqs": 0, "total_seqs": 0, "fa_total": 0}
        for cond in ["clean", "snr20", "snr15", "snr10", "snr05"]
    }

    per_sequence_records = []

    for idx, row in df.iterrows():
        seq_id = row["sequence_id"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        ref_events = json.loads(row["events_json"])
        nominal_gap = int(row["nominal_gap_ms"])
        cond = row["condition"]
        total_duration = float(row["duration_seconds"])

        # Target command list for this sequence
        target_ground_truth = [e["label"] for e in ref_events if e["is_target"]]
        n_targets_in_seq = len(target_ground_truth)
        total_target_events += n_targets_in_seq

        # Negative audio duration: total sequence duration minus target command durations
        target_duration_sum = sum(e["duration_sec"] for e in ref_events if e["is_target"])
        neg_duration = max(0.0, total_duration - target_duration_sum)
        total_negative_seconds += neg_duration

        audio, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == 16000

        segmenter.reset()
        emitted_predictions = []

        # Replay chunk-by-chunk
        for c_start in range(0, len(audio), chunk_samples):
            chunk = audio[c_start : c_start + chunk_samples]
            chunk_time = c_start / sr
            seg = segmenter.process_chunk(chunk, chunk_time)
            if seg is not None:
                audio_1s = seg["segment_audio_1s"]
                pred_class, conf, probs, inf_ms = evaluator.predict_clip(audio_1s)
                emitted_predictions.append({
                    "pred_class": pred_class,
                    "confidence": round(float(conf), 4),
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "trigger_time_sec": seg["end_sec"],
                    "duration_sec": seg["duration_sec"]
                })

        # Emitted command list (filtering out REJECT)
        emitted_commands = [p["pred_class"] for p in emitted_predictions if p["pred_class"] in TARGET_CLASSES]

        # Check Exact Sequence Accuracy
        is_exact = (emitted_commands == target_ground_truth)
        if is_exact:
            exact_correct_sequences += 1

        # Match predicted command events to reference events
        pred_cmd_indices = [i for i, p in enumerate(emitted_predictions) if p["pred_class"] in TARGET_CLASSES]
        matched_target_refs = set()
        matched_pred_cmds = set()
        seq_fa_oov = 0
        seq_fa_silence = 0
        seq_duplicates = 0

        # Attempt to match each predicted command to a reference event
        for p_idx in pred_cmd_indices:
            p_evt = emitted_predictions[p_idx]
            p_cls = p_evt["pred_class"]
            p_s, p_e = p_evt["start_sec"], p_evt["end_sec"]

            best_iou = 0.0
            best_r_idx = -1
            best_overlap = 0.0

            for r_idx, r_evt in enumerate(ref_events):
                r_s, r_e = r_evt["start_sec"], r_evt["end_sec"]
                iou = calculate_temporal_iou(p_s, p_e, r_s, r_e)
                inter_s = max(p_s, r_s)
                inter_e = min(p_e, r_e)
                overlap = max(0.0, inter_e - inter_s)
                if iou > best_iou:
                    best_iou = iou
                    best_r_idx = r_idx
                    best_overlap = overlap

            # If overlapping a reference event
            if best_r_idx >= 0 and (best_iou >= 0.20 or best_overlap >= 0.080):
                ref_match = ref_events[best_r_idx]

                if ref_match["is_target"]:
                    if best_r_idx not in matched_target_refs:
                        matched_target_refs.add(best_r_idx)
                        matched_pred_cmds.add(p_idx)
                        if p_cls == ref_match["label"]:
                            total_matched_commands += 1
                            lat_ms = (p_evt["trigger_time_sec"] - ref_match["start_sec"]) * 1000.0
                            all_latencies_ms.append(lat_ms)
                        else:
                            total_substitutions += 1
                    else:
                        # Duplicate trigger on same reference target event
                        total_duplicate_triggers += 1
                        seq_duplicates += 1
                else:
                    # Overlapping an OOV reference event -> False activation on OOV
                    total_oov_false_activations += 1
                    seq_fa_oov += 1
            else:
                # No reference event overlap -> False activation during silence/noise
                total_silence_false_activations += 1
                seq_fa_silence += 1

        seq_missed = n_targets_in_seq - len(matched_target_refs)
        total_missed_commands += seq_missed

        gap_stats[nominal_gap]["target_events"] += n_targets_in_seq
        gap_stats[nominal_gap]["matched"] += len([r for r in matched_target_refs if ref_events[r]["label"] in emitted_commands])
        gap_stats[nominal_gap]["exact_seqs"] += int(is_exact)
        gap_stats[nominal_gap]["total_seqs"] += 1
        gap_stats[nominal_gap]["fa_total"] += (seq_fa_oov + seq_fa_silence)

        cond_stats[cond]["target_events"] += n_targets_in_seq
        cond_stats[cond]["matched"] += len([r for r in matched_target_refs if ref_events[r]["label"] in emitted_commands])
        cond_stats[cond]["exact_seqs"] += int(is_exact)
        cond_stats[cond]["total_seqs"] += 1
        cond_stats[cond]["fa_total"] += (seq_fa_oov + seq_fa_silence)

        per_sequence_records.append({
            "sequence_id": seq_id,
            "nominal_gap_ms": nominal_gap,
            "condition": cond,
            "target_ground_truth": target_ground_truth,
            "emitted_commands": emitted_commands,
            "is_exact_match": is_exact,
            "n_targets": n_targets_in_seq,
            "n_emitted_cmds": len(emitted_commands),
            "fa_oov": seq_fa_oov,
            "fa_silence": seq_fa_silence,
            "duplicates": seq_duplicates,
            "raw_predictions": emitted_predictions
        })

    # Summary metrics
    n_total_emitted_cmds = total_matched_commands + total_substitutions + total_oov_false_activations + total_silence_false_activations + total_duplicate_triggers
    cmd_precision = total_matched_commands / max(1, n_total_emitted_cmds)
    cmd_recall = total_matched_commands / max(1, total_target_events)
    cmd_f1 = (2.0 * cmd_precision * cmd_recall) / max(1e-9, cmd_precision + cmd_recall)

    exact_accuracy = exact_correct_sequences / len(df)
    total_negative_minutes = total_negative_seconds / 60.0
    total_false_activations = total_oov_false_activations + total_silence_false_activations
    fa_per_minute = total_false_activations / max(1e-6, total_negative_minutes)

    median_latency_ms = float(np.median(all_latencies_ms)) if all_latencies_ms else 0.0
    p95_latency_ms = float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else 0.0

    summary = {
        "algorithm": algo_name,
        "model_name": model_name,
        "corpus_split": split_name,
        "manifest_path": str(manifest_csv),
        "total_sequences": len(df),
        "exact_correct_sequences": exact_correct_sequences,
        "exact_sequence_accuracy": round(exact_accuracy, 4),
        "total_target_events": total_target_events,
        "matched_commands": total_matched_commands,
        "missed_commands": total_missed_commands,
        "substitutions": total_substitutions,
        "oov_false_activations": total_oov_false_activations,
        "silence_false_activations": total_silence_false_activations,
        "duplicate_triggers": total_duplicate_triggers,
        "command_precision": round(cmd_precision, 4),
        "command_recall": round(cmd_recall, 4),
        "command_f1": round(cmd_f1, 4),
        "total_negative_minutes": round(total_negative_minutes, 2),
        "false_activations_per_minute": round(fa_per_minute, 2),
        "median_latency_ms": round(median_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "gap_breakdown": gap_stats,
        "condition_breakdown": cond_stats,
        "per_sequence_records": per_sequence_records
    }

    print("-" * 75)
    print(f"  {algo_name} + {model_name.upper()} Stage B Results:")
    print(f"    Exact-Sequence Accuracy: {exact_accuracy * 100:.2f}% ({exact_correct_sequences}/{len(df)} sequences)")
    print(f"    Command Precision: {cmd_precision * 100:.2f}% | Recall: {cmd_recall * 100:.2f}% | F1: {cmd_f1:.4f}")
    print(f"    Matched: {total_matched_commands}/{total_target_events} | Missed: {total_missed_commands} | Substitutions: {total_substitutions}")
    print(f"    False Activations: OOV={total_oov_false_activations}, Silence={total_silence_false_activations} ({fa_per_minute:.2f}/min)")
    print(f"    Duplicates: {total_duplicate_triggers}")
    print(f"    Latency: Median={median_latency_ms:.1f}ms, P95={p95_latency_ms:.1f}ms")
    print("-" * 75)

    return summary


def main():
    from vad_segmenters import BaselineFixedSilenceVAD, DualThresholdAdaptiveVAD, SpeechBandRatioVAD

    manifest_dev = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests" / "vad_dev_corpus_manifest.csv"
    v11_ckpt = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    e1_ckpt = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt"

    eval_v11 = KWSEvaluator(v11_ckpt)
    eval_e1 = KWSEvaluator(e1_ckpt)

    print("=" * 80)
    print("  PHASE 15: STAGE B — END-TO-END STREAMING EVALUATION (DEV CORPUS)")
    print("=" * 80)

    # 1. Baseline Fixed VAD with v11 and E1
    base_vad = BaselineFixedSilenceVAD()
    res_base_v11 = evaluate_streaming_kws(base_vad, eval_v11, manifest_dev, model_name="v11")
    res_base_e1 = evaluate_streaming_kws(base_vad, eval_e1, manifest_dev, model_name="e1")

    # 2. Dual-Threshold Adaptive VAD (Tuned) with v11 and E1
    dual_vad = DualThresholdAdaptiveVAD(
        hangover_clean_ms=60,
        hangover_noisy_ms=100,
        onset_snr_offset_db=10.0,
        offset_snr_offset_db=8.0
    )
    res_dual_v11 = evaluate_streaming_kws(dual_vad, eval_v11, manifest_dev, model_name="v11")
    res_dual_e1 = evaluate_streaming_kws(dual_vad, eval_e1, manifest_dev, model_name="e1")

    # 3. Speech-Band Ratio VAD (Tuned) with v11 and E1
    speech_vad = SpeechBandRatioVAD(
        hangover_ms=140,
        min_speech_band_ratio=0.32,
        energy_threshold_db=-46.0
    )
    res_speech_v11 = evaluate_streaming_kws(speech_vad, eval_v11, manifest_dev, model_name="v11")
    res_speech_e1 = evaluate_streaming_kws(speech_vad, eval_e1, manifest_dev, model_name="e1")

    # Aggregate and save Dev Stage B summary
    dev_stage_b = {
        "Baseline_v11": res_base_v11,
        "Baseline_e1": res_base_e1,
        "DualThresholdAdaptive_v11": res_dual_v11,
        "DualThresholdAdaptive_e1": res_dual_e1,
        "SpeechBandRatio_v11": res_speech_v11,
        "SpeechBandRatio_e1": res_speech_e1
    }

    out_json = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports" / "stage_b_dev_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dev_stage_b, f, indent=2, ensure_ascii=False)
    print(f"\nSaved Stage B Dev summary: {out_json}")


if __name__ == "__main__":
    main()
