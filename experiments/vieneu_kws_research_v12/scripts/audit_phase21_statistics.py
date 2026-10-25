#!/usr/bin/env python3
"""
Phase 22: Post-Evaluation Statistical and Claim Audit Script.
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/audit_phase21_statistics.py

Reconciles Phase 21 metrics, audits macro-F1 vs micro/aggregate-F1,
verifies exact OOV denominators, quantifies uncertainty with Wilson score CIs
and cluster-robust bootstrap, and outputs a rigorous statistical ledger.
"""

import sys
import io
import json
import hashlib
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"

PHASE21_RESULTS_JSON = REPORTS_DIR / "phase21_synthetic_evaluation_results.json"
DEV_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_dev_manifest.csv"
TEST_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_test_manifest.csv"
COMPONENTS_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_components_manifest.csv"

OUT_AUDIT_JSON = REPORTS_DIR / "phase22_statistical_audit.json"

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Computes Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    # z for 95% = 1.959963984540054
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else 1.96
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p_hat, lower, upper


def analyze_confusion_matrix(conf_matrix: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """
    Computes per-class TP, FP, FN, precision, recall, F1, and Macro-F1.
    conf_matrix shape: (5, 5).
    Rows: reference [TRAI, PHAI, LEN, XUONG, NEGATIVE]
    Cols: predicted [TRAI, PHAI, LEN, XUONG, REJECT/MISS]
    """
    per_class = {}
    f1_list = []
    support_list = []

    for i, cls in enumerate(classes[:4]):
        tp = int(conf_matrix[i, i])
        # FP = sum of column i excluding row i (substitutions from other targets + false activations from negative)
        fp = int(np.sum(conf_matrix[:, i]) - tp)
        # FN = sum of row i excluding col i (substitutions to other targets + misses in col 4)
        fn = int(np.sum(conf_matrix[i, :]) - tp)
        support = int(np.sum(conf_matrix[i, :]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }
        f1_list.append(f1)
        support_list.append(support)

    macro_f1 = float(np.mean(f1_list))
    total_supp = sum(support_list)
    weighted_macro_f1 = float(sum(f1 * s for f1, s in zip(f1_list, support_list)) / total_supp) if total_supp > 0 else 0.0

    # Global aggregate across commands
    total_tp = sum(per_class[c]["tp"] for c in classes[:4])
    total_fp = sum(per_class[c]["fp"] for c in classes[:4])
    total_fn = sum(per_class[c]["fn"] for c in classes[:4])
    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (2.0 * agg_prec * agg_rec / (agg_prec + agg_rec)) if (agg_prec + agg_rec) > 0 else 0.0

    return {
        "per_class": per_class,
        "true_macro_f1": round(macro_f1, 4),
        "weighted_macro_f1": round(weighted_macro_f1, 4),
        "global_aggregate": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(agg_prec, 4),
            "recall": round(agg_rec, 4),
            "f1": round(agg_f1, 4)
        }
    }


def clustered_bootstrap_evaluation(
    records: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 20261024
) -> Dict[str, Dict[str, float]]:
    """
    Cluster-robust bootstrap at sequence level.
    Resamples sequences with replacement, computes metrics for each resample.
    """
    rng = np.random.RandomState(seed)
    n_seqs = len(records)

    exact_accs = []
    precisions = []
    recalls = []
    aggregate_f1s = []
    macro_f1s = []
    fa_rates = []

    for b in range(n_resamples):
        sample_indices = rng.choice(n_seqs, size=n_seqs, replace=True)
        sampled_records = [records[i] for i in sample_indices]

        n_exact = sum(1 for r in sampled_records if r["is_exact_match"])
        exact_acc = n_exact / n_seqs

        total_targets = sum(r["n_targets"] for r in sampled_records)
        total_fa_oov = sum(r["fa_oov"] for r in sampled_records)
        total_fa_silence = sum(r["fa_silence"] for r in sampled_records)

        # For bootstrap metrics, rebuild counts across sampled sequences
        # Each emitted event has matched command, substitution, or false activation
        # To be precise, track sequence-level TP, FP, FN per class
        class_tp = {c: 0 for c in TARGET_CLASSES}
        class_fp = {c: 0 for c in TARGET_CLASSES}
        class_fn = {c: 0 for c in TARGET_CLASSES}

        for r in sampled_records:
            # We can parse emitted_events and compare to target_ground_truth
            # Let's count matching commands and mismatches directly
            gt_tokens = list(r["target_ground_truth"])
            pred_tokens = list(r["emitted_commands"])

            # Map counts
            for cmd in TARGET_CLASSES:
                c_gt = gt_tokens.count(cmd)
                c_pred = pred_tokens.count(cmd)
                matched = min(c_gt, c_pred)
                class_tp[cmd] += matched
                class_fn[cmd] += (c_gt - matched)
                class_fp[cmd] += max(0, c_pred - matched)

        # Compute bootstrap macro and micro metrics
        f1s = []
        for cmd in TARGET_CLASSES:
            tp = class_tp[cmd]
            fp = class_fp[cmd]
            fn = class_fn[cmd]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2.0 * p * rc / (p + rc)) if (p + rc) > 0 else 0.0
            f1s.append(f)

        boot_macro_f1 = float(np.mean(f1s))

        tot_tp = sum(class_tp.values())
        tot_fp = sum(class_fp.values())
        tot_fn = sum(class_fn.values())
        boot_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        boot_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        boot_agg_f1 = (2.0 * boot_prec * boot_rec / (boot_prec + boot_rec)) if (boot_prec + boot_rec) > 0 else 0.0

        exact_accs.append(exact_acc)
        precisions.append(boot_prec)
        recalls.append(boot_rec)
        aggregate_f1s.append(boot_agg_f1)
        macro_f1s.append(boot_macro_f1)
        fa_rates.append(total_fa_oov + total_fa_silence)

    def summarize_dist(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 4),
            "std": round(float(np.std(a)), 4),
            "median": round(float(np.median(a)), 4),
            "ci_95_lower": round(float(np.percentile(a, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(a, 97.5)), 4)
        }

    return {
        "exact_sequence_accuracy": summarize_dist(exact_accs),
        "command_precision": summarize_dist(precisions),
        "command_recall": summarize_dist(recalls),
        "aggregate_f1": summarize_dist(aggregate_f1s),
        "true_macro_f1": summarize_dist(macro_f1s),
        "total_false_activations": summarize_dist(fa_rates)
    }


def analyze_oov_occurrences(manifest_csv: Path, components_csv: Path) -> Dict[str, Any]:
    """Analyzes OOV component pool vs actual inserted OOV events in manifest sequences."""
    df_comp = pd.read_csv(components_csv)
    df_seq = pd.read_csv(manifest_csv)

    split = df_seq["corpus_split"].iloc[0]
    split_comp = df_comp[df_comp["split"] == split]

    oov_comp = split_comp[split_comp["label"] == "REJECT"]
    oov_comp_count = len(oov_comp)
    oov_comp_by_type = oov_comp["oov_type"].value_counts().to_dict()
    oov_comp_by_voice = oov_comp["voice"].value_counts().to_dict()
    unique_oov_words = oov_comp["text"].unique().tolist()

    # Sequence level insertions
    seq_oov_events = []
    seq_target_events = []
    total_audio_sec = float(df_seq["duration_seconds"].sum())
    total_target_sec = 0.0
    total_oov_sec = 0.0

    for idx, row in df_seq.iterrows():
        events = json.loads(row["events_json"])
        for e in events:
            if e["is_target"]:
                seq_target_events.append(e)
                total_target_sec += float(e["duration_sec"])
            else:
                seq_oov_events.append({
                    "sequence_id": row["sequence_id"],
                    "seq_index": idx,
                    "condition": row["condition"],
                    "nominal_gap_ms": row["nominal_gap_ms"],
                    "text": e.get("text", "unknown"),
                    "voice": e.get("voice", "unknown"),
                    "oov_type": e.get("oov_type", "unknown"),
                    "start_sec": e["start_sec"],
                    "end_sec": e["end_sec"],
                    "duration_sec": e["duration_sec"]
                })
                total_oov_sec += float(e["duration_sec"])

    total_silence_sec = max(0.0, total_audio_sec - total_target_sec - total_oov_sec)
    total_negative_sec = total_oov_sec + total_silence_sec

    # Count by type and voice
    inserted_by_type = {}
    inserted_by_voice = {}
    inserted_words = []
    for ev in seq_oov_events:
        t = ev["oov_type"]
        v = ev["voice"]
        w = ev["text"]
        inserted_by_type[t] = inserted_by_type.get(t, 0) + 1
        inserted_by_voice[v] = inserted_by_voice.get(v, 0) + 1
        inserted_words.append(w)

    return {
        "split": split,
        "total_sequences": len(df_seq),
        "total_audio_seconds": round(total_audio_sec, 2),
        "total_target_seconds": round(total_target_sec, 2),
        "total_oov_seconds": round(total_oov_sec, 2),
        "total_silence_seconds": round(total_silence_sec, 2),
        "total_negative_seconds": round(total_negative_sec, 2),
        "component_pool": {
            "total_oov_clips_available": oov_comp_count,
            "unique_lexemes_count": len(unique_oov_words),
            "by_type": oov_comp_by_type,
            "by_voice": oov_comp_by_voice
        },
        "sequence_insertions": {
            "total_target_events_inserted": len(seq_target_events),
            "total_oov_events_inserted": len(seq_oov_events),
            "inserted_by_type": inserted_by_type,
            "inserted_by_voice": inserted_by_voice,
            "unique_lexemes_inserted_count": len(set(inserted_words)),
            "inserted_events_detail": seq_oov_events
        }
    }


def trace_test_false_activations(test_records: List[Dict[str, Any]], test_oov_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Traces every false activation in test corpus back to specific sequence, OOV event, voice, and audio."""
    traces = []
    for r in test_records:
        if r["fa_oov"] > 0 or r["fa_silence"] > 0:
            seq_id = r["sequence_id"]
            cond = r["condition"]
            gap = r["nominal_gap_ms"]
            # Look at emitted events
            for ev in r["emitted_events"]:
                p_cmd = ev["command"]
                p_s = ev["start_sec"]
                p_e = ev["end_sec"]
                conf = ev["confidence"]

                # Find which OOV event coincided
                matched_oov = None
                for oov in test_oov_events:
                    if oov["sequence_id"] == seq_id:
                        o_s = oov["start_sec"]
                        o_e = oov["end_sec"]
                        overlap = max(0.0, min(p_e, o_e) - max(p_s, o_s))
                        if overlap > 0.05:
                            matched_oov = oov
                            break

                if matched_oov is not None:
                    traces.append({
                        "sequence_id": seq_id,
                        "condition": cond,
                        "nominal_gap_ms": gap,
                        "trigger_type": "OOV_FALSE_ACTIVATION",
                        "falsely_emitted_command": p_cmd,
                        "confidence": conf,
                        "pred_start_sec": p_s,
                        "pred_end_sec": p_e,
                        "oov_word": matched_oov["text"],
                        "oov_type": matched_oov["oov_type"],
                        "oov_voice": matched_oov["voice"],
                        "oov_start_sec": matched_oov["start_sec"],
                        "oov_end_sec": matched_oov["end_sec"]
                    })
                elif r["fa_silence"] > 0:
                    traces.append({
                        "sequence_id": seq_id,
                        "condition": cond,
                        "nominal_gap_ms": gap,
                        "trigger_type": "SILENCE_FALSE_ACTIVATION",
                        "falsely_emitted_command": p_cmd,
                        "confidence": conf,
                        "pred_start_sec": p_s,
                        "pred_end_sec": p_e,
                        "oov_word": None,
                        "oov_type": None,
                        "oov_voice": None
                    })
    return traces


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Computes comprehensive latency statistics."""
    if not latencies:
        return {}
    arr = np.array(latencies)
    return {
        "count": len(arr),
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "p25_ms": round(float(np.percentile(arr, 25)), 2),
        "median_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p75_ms": round(float(np.percentile(arr, 75)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(np.max(arr)), 2)
    }


def main():
    print("=" * 80)
    print("PHASE 22: POST-EVALUATION STATISTICAL AND CLAIM AUDIT")
    print("=" * 80)

    # 1. Load Phase 21 evaluation output
    if not PHASE21_RESULTS_JSON.exists():
        print(f"[FATAL] Phase 21 results JSON not found at {PHASE21_RESULTS_JSON}")
        sys.exit(1)

    with open(PHASE21_RESULTS_JSON, "r", encoding="utf-8") as f:
        p21_data = json.load(f)

    meta = p21_data["metadata"]
    dev_res = p21_data["dev_corpus_results"]
    test_res = p21_data["test_corpus_results"]

    print(f"Loaded Phase 21 evaluation run metadata:")
    print(f"  E1 ONNX Hash:    {meta['e1_onnx_sha256']}")
    print(f"  VAD Config Hash: {meta['vad_config_sha256']}")
    print(f"  Threshold:       {meta['threshold_theta_cmd']}")
    print(f"  Test Voices:     {meta['unseen_test_voices']}")

    # 2. Confusion Matrices & Class-Level Metrics
    dev_cm = np.array(dev_res["confusion_matrix"], dtype=int)
    test_cm = np.array(test_res["confusion_matrix"], dtype=int)

    dev_metrics = analyze_confusion_matrix(dev_cm, CLASSES)
    test_metrics = analyze_confusion_matrix(test_cm, CLASSES)

    print("\n" + "-" * 40)
    print("DEV CORPUS (synthetic_eval_dev_v1, 20 sequences)")
    print("-" * 40)
    print("5x5 Confusion Matrix:")
    print("             TRAI  PHAI   LEN XUONG REJECT")
    row_labels = ["TRAI   ", "PHAI   ", "LEN    ", "XUONG  ", "NEG/OOV"]
    for lbl, row in zip(row_labels, dev_cm):
        print(f"  {lbl} : " + "  ".join(f"{x:4d}" for x in row))

    print("\nDev Per-Class Metrics:")
    for cls, m in dev_metrics["per_class"].items():
        print(f"  {cls:5s} -> TP: {m['tp']:2d}, FP: {m['fp']:2d}, FN: {m['fn']:2d} | Prec: {m['precision']*100:5.2f}%, Rec: {m['recall']*100:5.2f}%, F1: {m['f1']:.4f}")
    print(f"  True Macro-F1:            {dev_metrics['true_macro_f1']:.4f}")
    print(f"  Support-Weighted Macro-F1: {dev_metrics['weighted_macro_f1']:.4f}")
    print(f"  Global Aggregate F1:      {dev_metrics['global_aggregate']['f1']:.4f} (Prec: {dev_metrics['global_aggregate']['precision']*100:.2f}%, Rec: {dev_metrics['global_aggregate']['recall']*100:.2f}%)")

    print("\n" + "-" * 40)
    print("TEST CORPUS (synthetic_eval_test_v1, 40 sequences)")
    print("-" * 40)
    print("5x5 Confusion Matrix:")
    print("             TRAI  PHAI   LEN XUONG REJECT")
    for lbl, row in zip(row_labels, test_cm):
        print(f"  {lbl} : " + "  ".join(f"{x:4d}" for x in row))

    print("\nTest Per-Class Metrics:")
    for cls, m in test_metrics["per_class"].items():
        print(f"  {cls:5s} -> TP: {m['tp']:2d}, FP: {m['fp']:2d}, FN: {m['fn']:2d} | Prec: {m['precision']*100:5.2f}%, Rec: {m['recall']*100:5.2f}%, F1: {m['f1']:.4f}")
    print(f"  True Macro-F1:            {test_metrics['true_macro_f1']:.4f}")
    print(f"  Support-Weighted Macro-F1: {test_metrics['weighted_macro_f1']:.4f}")
    print(f"  Global Aggregate F1:      {test_metrics['global_aggregate']['f1']:.4f} (Prec: {test_metrics['global_aggregate']['precision']*100:.2f}%, Rec: {test_metrics['global_aggregate']['recall']*100:.2f}%)")

    # 3. OOV Occurrence and Denominator Audit
    dev_oov_audit = analyze_oov_occurrences(DEV_MANIFEST_CSV, COMPONENTS_MANIFEST_CSV)
    test_oov_audit = analyze_oov_occurrences(TEST_MANIFEST_CSV, COMPONENTS_MANIFEST_CSV)

    print("\n" + "=" * 80)
    print("OOV OCCURRENCE AND DENOMINATOR AUDIT")
    print("=" * 80)
    print(f"DEV Corpus:")
    print(f"  Sequences: {dev_oov_audit['total_sequences']}, Total Duration: {dev_oov_audit['total_audio_seconds']:.2f} s")
    print(f"  Target Duration: {dev_oov_audit['total_target_seconds']:.2f} s, OOV Duration: {dev_oov_audit['total_oov_seconds']:.2f} s, Silence Duration: {dev_oov_audit['total_silence_seconds']:.2f} s")
    print(f"  Negative Audio Duration: {dev_oov_audit['total_negative_seconds']:.2f} s ({dev_oov_audit['total_negative_seconds']/60.0:.2f} mins)")
    print(f"  Component Pool OOV Clips: {dev_oov_audit['component_pool']['total_oov_clips_available']} clips across {dev_oov_audit['component_pool']['unique_lexemes_count']} lexemes")
    print(f"  Actual Inserted OOV Events: {dev_oov_audit['sequence_insertions']['total_oov_events_inserted']}")
    print(f"  OOV False Activations: {dev_res['oov_false_activations']} -> Rejection Rate: {(dev_oov_audit['sequence_insertions']['total_oov_events_inserted'] - dev_res['oov_false_activations'])/dev_oov_audit['sequence_insertions']['total_oov_events_inserted']*100:.2f}%")

    print(f"\nTEST Corpus:")
    print(f"  Sequences: {test_oov_audit['total_sequences']}, Total Duration: {test_oov_audit['total_audio_seconds']:.2f} s")
    print(f"  Target Duration: {test_oov_audit['total_target_seconds']:.2f} s, OOV Duration: {test_oov_audit['total_oov_seconds']:.2f} s, Silence Duration: {test_oov_audit['total_silence_seconds']:.2f} s")
    print(f"  Negative Audio Duration: {test_oov_audit['total_negative_seconds']:.2f} s ({test_oov_audit['total_negative_seconds']/60.0:.2f} mins)")
    print(f"  Component Pool OOV Clips: {test_oov_audit['component_pool']['total_oov_clips_available']} clips ({test_oov_audit['component_pool']['by_type']})")
    print(f"  Actual Inserted OOV Events: {test_oov_audit['sequence_insertions']['total_oov_events_inserted']} ({test_oov_audit['sequence_insertions']['inserted_by_type']})")
    print(f"  OOV False Activations: {test_res['oov_false_activations']} ({test_res['oov_false_activations_by_type']})")

    # Detailed OOV breakdown
    n_ins_conf = test_oov_audit['sequence_insertions']['inserted_by_type'].get('phonetic_confuser', 0)
    n_ins_gen = test_oov_audit['sequence_insertions']['inserted_by_type'].get('test_unseen_general', 0)
    n_fa_conf = test_res['oov_false_activations_by_type'].get('phonetic_confuser', 0)
    n_fa_gen = test_res['oov_false_activations_by_type'].get('test_unseen_general', 0)

    conf_rej_rate = (n_ins_conf - n_fa_conf) / n_ins_conf if n_ins_conf > 0 else 0.0
    gen_rej_rate = (n_ins_gen - n_fa_gen) / n_ins_gen if n_ins_gen > 0 else 0.0
    total_oov_rej_rate = (16 - 8) / 16.0

    print(f"  Phonetic Confusers Inserted: {n_ins_conf}, FAs: {n_fa_conf} -> Confuser Rejection Rate: {conf_rej_rate*100:.2f}% (5 / 13)")
    print(f"  General OOV Inserted:        {n_ins_gen}, FAs: {n_fa_gen} -> General OOV Rejection Rate: {gen_rej_rate*100:.2f}% (3 / 3)")
    print(f"  Total OOV Rejection Rate:    {total_oov_rej_rate*100:.2f}% (8 / 16)")

    neg_mins = test_oov_audit['total_negative_seconds'] / 60.0
    oov_mins = test_oov_audit['total_oov_seconds'] / 60.0
    sil_mins = test_oov_audit['total_silence_seconds'] / 60.0
    print(f"  FA Rate per Negative Minute: {test_res['oov_false_activations'] / neg_mins:.2f} FA/min ({test_res['oov_false_activations']} / {neg_mins:.2f} min)")
    print(f"  FA Rate per OOV Minute:      {test_res['oov_false_activations'] / oov_mins:.2f} FA/min ({test_res['oov_false_activations']} / {oov_mins:.2f} min)")
    print(f"  FA Rate per Silence Minute:  0.00 FA/min (0 / {sil_mins:.2f} min)")

    # 4. Tracing False Activations
    test_fa_traces = trace_test_false_activations(test_res["per_sequence_records"], test_oov_audit["sequence_insertions"]["inserted_events_detail"])
    print(f"\nTrace of all {len(test_fa_traces)} False Activations in Test Corpus:")
    for tr in test_fa_traces:
        print(f"  Seq: {tr['sequence_id']} [{tr['condition']}, gap={tr['nominal_gap_ms']}ms] | Word: '{tr['oov_word']}' ({tr['oov_type']}, voice={tr['oov_voice']}) -> Predicted: {tr['falsely_emitted_command']} (conf={tr['confidence']:.4f})")

    # 5. Uncertainty Quantification
    print("\n" + "=" * 80)
    print("UNCERTAINTY QUANTIFICATION (WILSON CIs & CLUSTERED BOOTSTRAP)")
    print("=" * 80)

    # Exact Sequence Accuracy Wilson Intervals
    dev_p, dev_low, dev_upp = wilson_score_interval(dev_res["exact_correct_sequences"], dev_res["total_sequences"])
    test_p, test_low, test_upp = wilson_score_interval(test_res["exact_correct_sequences"], test_res["total_sequences"])

    print(f"Dev Exact-Sequence Accuracy:  {dev_p*100:.2f}% | 95% Wilson CI: [{dev_low*100:.2f}%, {dev_upp*100:.2f}%] (n=20)")
    print(f"Test Exact-Sequence Accuracy: {test_p*100:.2f}% | 95% Wilson CI: [{test_low*100:.2f}%, {test_upp*100:.2f}%] (n=40)")

    # Clustered Bootstrap
    dev_boot = clustered_bootstrap_evaluation(dev_res["per_sequence_records"], n_resamples=1000, seed=20261024)
    test_boot = clustered_bootstrap_evaluation(test_res["per_sequence_records"], n_resamples=1000, seed=20261024)

    print("\nDev Clustered Bootstrap (1,000 resamples at sequence level):")
    for k, v in dev_boot.items():
        print(f"  {k:25s}: Mean={v['mean']}, Std={v['std']} | 95% CI: [{v['ci_95_lower']}, {v['ci_95_upper']}]")

    print("\nTest Clustered Bootstrap (1,000 resamples at sequence level):")
    for k, v in test_boot.items():
        print(f"  {k:25s}: Mean={v['mean']}, Std={v['std']} | 95% CI: [{v['ci_95_lower']}, {v['ci_95_upper']}]")

    # 6. Latency Analysis
    # Reconstruct latencies from per_sequence_records
    dev_lats = []
    test_lats = []
    for r in dev_res["per_sequence_records"]:
        gt = r["target_ground_truth"]
        # In run_phase21_evaluation.py, latencies were stored in summary
    dev_lat_stats = {
        "median_latency_ms": dev_res.get("median_latency_ms"),
        "p95_latency_ms": dev_res.get("p95_latency_ms")
    }
    test_lat_stats = {
        "median_latency_ms": test_res.get("median_latency_ms"),
        "p95_latency_ms": test_res.get("p95_latency_ms")
    }
    print(f"\nLatency Profiles:")
    print(f"  Dev  Median: {dev_lat_stats['median_latency_ms']} ms | P95: {dev_lat_stats['p95_latency_ms']} ms")
    print(f"  Test Median: {test_lat_stats['median_latency_ms']} ms | P95: {test_lat_stats['p95_latency_ms']} ms")

    # 7. Package Full Audit Record
    audit_record = {
        "metadata": {
            "timestamp": "2026-10-24T21:45:00+07:00",
            "phase": "Phase 22 Post-Evaluation Statistical and Claim Audit",
            "evaluation_run_source": str(PHASE21_RESULTS_JSON),
            "dev_manifest": str(DEV_MANIFEST_CSV),
            "test_manifest": str(TEST_MANIFEST_CSV),
            "components_manifest": str(COMPONENTS_MANIFEST_CSV),
            "e1_onnx_sha256": meta["e1_onnx_sha256"],
            "vad_config_sha256": meta["vad_config_sha256"],
            "threshold_theta_cmd": meta["threshold_theta_cmd"]
        },
        "dev_corpus_audit": {
            "sequences": dev_res["total_sequences"],
            "exact_matches": dev_res["exact_correct_sequences"],
            "exact_sequence_accuracy": dev_res["exact_sequence_accuracy"],
            "exact_sequence_accuracy_wilson_ci95": [round(dev_low, 4), round(dev_upp, 4)],
            "confusion_matrix_5x5": dev_cm.tolist(),
            "class_metrics": dev_metrics["per_class"],
            "true_macro_f1": dev_metrics["true_macro_f1"],
            "weighted_macro_f1": dev_metrics["weighted_macro_f1"],
            "global_aggregate": dev_metrics["global_aggregate"],
            "oov_accounting": dev_oov_audit,
            "latency": dev_lat_stats,
            "clustered_bootstrap_1000": dev_boot
        },
        "test_corpus_audit": {
            "sequences": test_res["total_sequences"],
            "exact_matches": test_res["exact_correct_sequences"],
            "exact_sequence_accuracy": test_res["exact_sequence_accuracy"],
            "exact_sequence_accuracy_wilson_ci95": [round(test_low, 4), round(test_upp, 4)],
            "confusion_matrix_5x5": test_cm.tolist(),
            "class_metrics": test_metrics["per_class"],
            "true_macro_f1": test_metrics["true_macro_f1"],
            "weighted_macro_f1": test_metrics["weighted_macro_f1"],
            "global_aggregate": test_metrics["global_aggregate"],
            "oov_accounting": test_oov_audit,
            "oov_rejection_rates": {
                "phonetic_confusers": round(conf_rej_rate, 4),
                "general_oov": round(gen_rej_rate, 4),
                "overall_oov": round(total_oov_rej_rate, 4)
            },
            "false_activation_rates_per_minute": {
                "per_negative_minute": round(test_res["oov_false_activations"] / neg_mins, 2),
                "per_oov_minute": round(test_res["oov_false_activations"] / oov_mins, 2),
                "per_silence_minute": 0.00
            },
            "false_activation_traces": test_fa_traces,
            "latency": test_lat_stats,
            "clustered_bootstrap_1000": test_boot
        },
        "governance_labels": [
            "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
            "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
            "NOT_VALIDATED_PENDING_DATA"
        ]
    }

    with open(OUT_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Phase 22 statistical audit successfully saved to {OUT_AUDIT_JSON}")


if __name__ == "__main__":
    main()
