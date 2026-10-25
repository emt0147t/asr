import os
import sys
import json
import csv
import time
import math
import platform
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v5.models.train_ablation_campaign import (
    VietnameseBoundedKWSNet,
    LogMelFrontend
)

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
ALL_5_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)

def reconcile_denominators_and_metrics():
    print("--- 1. Reconciling Test Denominators & Metrics from Raw Predictions ---")
    pred_csv_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "predictions", "per_clip_predictions.csv")
    test_manifest_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "data", "splits", "test_manifest.csv")
    v5_summary_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "predictions", "locked_test_summary.json")

    df_preds = pd.read_csv(pred_csv_path)
    df_manifest = pd.read_csv(test_manifest_path)

    with open(v5_summary_path, "r", encoding="utf-8") as f:
        v5_reported = json.load(f)

    # Manifest counts
    manifest_counts = {
        "total_clips": len(df_manifest),
        "target_clips_total": len(df_manifest[df_manifest["is_target_command"] == True]),
        "non_target_clips_total": len(df_manifest[df_manifest["is_target_command"] == False]),
        "per_keyword_counts": {kw: len(df_manifest[df_manifest["class_label"] == kw]) for kw in TARGET_CLASSES},
        "unknown_clips": len(df_manifest[df_manifest["class_label"] == "UNKNOWN"]),
        "silence_clips": len(df_manifest[df_manifest["class_label"] == "SILENCE"]),
        "voices": list(sorted(df_manifest["voice_id"].unique()))
    }
    print("Manifest Test Set Profile:", manifest_counts)

    candidates = ["variant_a_cascade_v1", "variant_b_stage1_cnn"]
    seeds = [42, 43, 44]

    reconciled_results = {}

    for cand in candidates:
        reconciled_results[cand] = {"per_seed": {}}
        seed_recalls = []
        seed_fars = []
        seed_accs = []
        seed_frrs = []

        for s in seeds:
            sub = df_preds[(df_preds["finalist"] == cand) & (df_preds["seed"] == s)]
            assert len(sub) == 720, f"Expected 720 rows for {cand} seed {s}, found {len(sub)}"

            # 5x5 confusion matrix: rows=True, cols=Pred
            cm = {true_c: {pred_c: 0 for pred_c in ALL_5_CLASSES} for true_c in ALL_5_CLASSES}
            for _, r in sub.iterrows():
                cm[r["true_label"]][r["pred_label"]] += 1

            # Verify total count
            cm_total = sum(sum(cm[tc].values()) for tc in ALL_5_CLASSES)
            assert cm_total == 720, f"Confusion matrix sum {cm_total} != 720"

            # Per-keyword counts
            kw_metrics = {}
            for kw in TARGET_CLASSES:
                hits = cm[kw][kw]
                total = sum(cm[kw].values())  # 40
                false_rej = cm[kw]["REJECT"]  # Misses to reject
                wrong_accepts = sum(cm[kw][other] for other in TARGET_CLASSES if other != kw)  # Cross-command
                total_pred_as_kw = sum(cm[tc][kw] for tc in ALL_5_CLASSES)

                rec = hits / total if total > 0 else 0.0
                prec = hits / total_pred_as_kw if total_pred_as_kw > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                kw_metrics[kw] = {
                    "total_true": total,
                    "correct_hits": hits,
                    "false_rejections": false_rej,
                    "wrong_accepted_cross_command": wrong_accepts,
                    "total_predicted_as_keyword": total_pred_as_kw,
                    "recall": round(rec, 4),
                    "precision": round(prec, 4),
                    "f1": round(f1, 4),
                    "recall_increment_step": "1/40 = 2.50%",
                    "recall_95ci": wilson_score_interval(hits, total)
                }

            # Command aggregates
            total_cmd_hits = sum(cm[kw][kw] for kw in TARGET_CLASSES)
            total_cmd_clips = sum(sum(cm[kw].values()) for kw in TARGET_CLASSES)  # 160
            total_cmd_false_rej = sum(cm[kw]["REJECT"] for kw in TARGET_CLASSES)
            total_cmd_wrong_cross = sum(sum(cm[kw][other] for other in TARGET_CLASSES if other != kw) for kw in TARGET_CLASSES)

            micro_recall = total_cmd_hits / total_cmd_clips
            macro_recall = np.mean([kw_metrics[kw]["recall"] for kw in TARGET_CLASSES])
            # Note on equivalence:
            macro_micro_diff = abs(macro_recall - micro_recall)
            assert macro_micro_diff < 1e-6, "Macro and micro recall must be identical on balanced classes"

            target_frr = total_cmd_false_rej / total_cmd_clips

            # Non-target metrics
            nt_total = sum(cm["REJECT"].values())  # 560
            nt_correct_rejects = cm["REJECT"]["REJECT"]
            nt_false_accepts = nt_total - nt_correct_rejects
            nt_far = nt_false_accepts / nt_total
            nt_tnr = nt_correct_rejects / nt_total

            # 5-class overall accuracy
            total_correct = total_cmd_hits + nt_correct_rejects
            overall_acc = total_correct / 720.0

            # Confidence intervals on individual seed observations
            ci_recall = wilson_score_interval(total_cmd_hits, total_cmd_clips)
            ci_far = wilson_score_interval(nt_false_accepts, nt_total)
            ci_acc = wilson_score_interval(total_correct, 720)

            seed_recalls.append(micro_recall)
            seed_fars.append(nt_far)
            seed_accs.append(overall_acc)
            seed_frrs.append(target_frr)

            reconciled_results[cand]["per_seed"][f"seed{s}"] = {
                "sample_denominators": {
                    "total_evaluated_clips": 720,
                    "target_clips_total": total_cmd_clips,
                    "target_clips_by_keyword": {kw: 40 for kw in TARGET_CLASSES},
                    "non_target_clips_total": nt_total,
                    "unknown_clips": 480,
                    "silence_clips": 80,
                    "correct_target_predictions": total_cmd_hits,
                    "target_false_rejections": total_cmd_false_rej,
                    "wrong_accepted_cross_command": total_cmd_wrong_cross,
                    "non_target_false_acceptances": nt_false_accepts,
                    "non_target_correct_rejections": nt_correct_rejects
                },
                "performance_metrics": {
                    "macro_command_recall": round(macro_recall, 4),
                    "micro_command_recall": round(micro_recall, 4),
                    "macro_equals_micro": True,
                    "recall_95ci": ci_recall,
                    "target_false_rejection_rate": round(target_frr, 4),
                    "frr_95ci": wilson_score_interval(total_cmd_false_rej, total_cmd_clips),
                    "non_target_false_acceptance_rate": round(nt_far, 4),
                    "far_95ci": ci_far,
                    "non_target_rejection_rate": round(nt_tnr, 4),
                    "overall_5class_accuracy": round(overall_acc, 4),
                    "accuracy_95ci": ci_acc,
                    "per_keyword": kw_metrics
                },
                "confusion_matrix": cm
            }

        # Cross-seed statistics (Mean and Sample Standard Deviation)
        reconciled_results[cand]["cross_seed_summary"] = {
            "evaluation_clips_per_seed": 720,
            "number_of_seeds": len(seeds),
            "macro_command_recall": {
                "mean": round(float(np.mean(seed_recalls)), 4),
                "std": round(float(np.std(seed_recalls, ddof=1)), 4),
                "values": [round(x, 4) for x in seed_recalls]
            },
            "non_target_false_acceptance_rate": {
                "mean": round(float(np.mean(seed_fars)), 4),
                "std": round(float(np.std(seed_fars, ddof=1)), 4),
                "values": [round(x, 4) for x in seed_fars]
            },
            "target_false_rejection_rate": {
                "mean": round(float(np.mean(seed_frrs)), 4),
                "std": round(float(np.std(seed_frrs, ddof=1)), 4),
                "values": [round(x, 4) for x in seed_frrs]
            },
            "overall_5class_accuracy": {
                "mean": round(float(np.mean(seed_accs)), 4),
                "std": round(float(np.std(seed_accs, ddof=1)), 4),
                "values": [round(x, 4) for x in seed_accs]
            },
            "statistical_aggregation_note": (
                "Seeds 42, 43, 44 evaluated the identical 720 test clips. They are reported "
                "with per-seed Wilson CIs and cross-seed sample standard deviation. Pooling "
                "3 seeds into 2,160 clips would violate independence."
            )
        }

    return reconciled_results, manifest_counts


def decompose_and_reconcile_latency():
    print("\n--- 2. Decomposing and Reconciling Single-Clip CPU Latency ---")
    device = torch.device("cpu")
    num_threads = 4
    torch.set_num_threads(num_threads)

    # CPU Information
    cpu_info = {
        "processor": platform.processor(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_num_threads": num_threads,
        "batch_size": 1,
        "input_samples": 16000,
        "input_duration_sec": 1.0,
        "warmup_trials": 30,
        "benchmark_trials": 200,
        "timer": "time.perf_counter()"
    }
    print(f"CPU Benchmarking Setup: {cpu_info['processor']} | Threads: {num_threads} | Batch Size: 1")

    frontend = LogMelFrontend().to(device)
    frontend.eval()

    # Load Candidate 1 (Stage 1 CNN alone)
    s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", "best_checkpoint_seed42.pt")
    m1 = VietnameseKWSNet(num_classes=6).to(device)
    m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
    m1.eval()

    # Load Candidate 2 (Cascade v1: Stage 1 + Stage 2 TCN)
    s2_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1", "checkpoints", "best_verifier_seed42.pt")
    m2 = TemporalKWSVerifier(num_classes=5).to(device)
    m2.load_state_dict(torch.load(s2_ckpt, map_location=device)["model_state_dict"])
    m2.eval()

    dummy_audio = torch.randn(1, 16000, device=device)
    dummy_crop = torch.randn(1, 19200, device=device)

    # Warmup
    for _ in range(cpu_info["warmup_trials"]):
        with torch.no_grad():
            s = frontend(dummy_audio)
            _ = m1(s)
            sc = frontend(dummy_crop)
            _ = m2(sc)

    # Component-level latency storage (in milliseconds)
    timings = {
        "candidate_1_stage1_cnn": {
            "feature_extraction_ms": [],
            "nn_forward_ms": [],
            "decision_rejection_ms": [],
            "total_post_capture_ms": []
        },
        "candidate_2_cascade_v1": {
            "stage1_feature_ms": [],
            "stage1_forward_ms": [],
            "stage2_feature_ms": [],
            "stage2_forward_ms": [],
            "total_nn_forward_ms": [],
            "decision_rejection_ms": [],
            "total_post_capture_ms": []
        }
    }

    trials = cpu_info["benchmark_trials"]

    for _ in range(trials):
        # --- Candidate 1: Stage 1 CNN Alone ---
        t_start = time.perf_counter()
        
        # 1. Feature extraction
        t0 = time.perf_counter()
        with torch.no_grad():
            spec = frontend(dummy_audio)
        t_feat = (time.perf_counter() - t0) * 1000.0

        # 2. NN forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = m1(spec)
        t_nn = (time.perf_counter() - t0) * 1000.0

        # 3. Decision & thresholding logic
        t0 = time.perf_counter()
        probs = F.softmax(logits, dim=1).numpy()[0]
        top_c = int(np.argmax(probs[:4]))
        val = float(probs[top_c])
        pred = TARGET_CLASSES[top_c] if (val >= 0.85 and val > probs[4] and val > probs[5]) else "REJECT"
        t_dec = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        timings["candidate_1_stage1_cnn"]["feature_extraction_ms"].append(t_feat)
        timings["candidate_1_stage1_cnn"]["nn_forward_ms"].append(t_nn)
        timings["candidate_1_stage1_cnn"]["decision_rejection_ms"].append(t_dec)
        timings["candidate_1_stage1_cnn"]["total_post_capture_ms"].append(t_total)

        # --- Candidate 2: Cascade v1 (Worst-Case Path: S1 fires, S2 executes) ---
        t_start = time.perf_counter()

        # 1. Stage 1 Feature
        t0 = time.perf_counter()
        with torch.no_grad():
            spec1 = frontend(dummy_audio)
        t_s1_feat = (time.perf_counter() - t0) * 1000.0

        # 2. Stage 1 Forward
        t0 = time.perf_counter()
        with torch.no_grad():
            logits1 = m1(spec1)
        t_s1_fwd = (time.perf_counter() - t0) * 1000.0

        # 3. Intermediate check
        p1 = F.softmax(logits1, dim=1).numpy()[0]
        top_c1 = int(np.argmax(p1[:4]))
        val1 = float(p1[top_c1])

        # Stage 2 path (worst-case: Stage 2 called)
        t0 = time.perf_counter()
        with torch.no_grad():
            spec2 = frontend(dummy_crop)
        t_s2_feat = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad():
            logits2 = m2(spec2)
        t_s2_fwd = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        p2 = F.softmax(logits2, dim=1).numpy()[0]
        val2 = float(p2[top_c1])
        rej2 = float(p2[4])
        pred2 = TARGET_CLASSES[top_c1] if (val2 >= 0.55 and val2 > rej2) else "REJECT"
        t_s2_dec = (time.perf_counter() - t0) * 1000.0

        t_total2 = (time.perf_counter() - t_start) * 1000.0

        timings["candidate_2_cascade_v1"]["stage1_feature_ms"].append(t_s1_feat)
        timings["candidate_2_cascade_v1"]["stage1_forward_ms"].append(t_s1_fwd)
        timings["candidate_2_cascade_v1"]["stage2_feature_ms"].append(t_s2_feat)
        timings["candidate_2_cascade_v1"]["stage2_forward_ms"].append(t_s2_fwd)
        timings["candidate_2_cascade_v1"]["total_nn_forward_ms"].append(t_s1_fwd + t_s2_fwd)
        timings["candidate_2_cascade_v1"]["decision_rejection_ms"].append(t_s2_dec)
        timings["candidate_2_cascade_v1"]["total_post_capture_ms"].append(t_total2)

    # Compute summary percentiles
    latency_summary = {"cpu_hardware_profile": cpu_info, "candidate_profiles": {}}

    for cand, comps in timings.items():
        latency_summary["candidate_profiles"][cand] = {}
        for comp_name, vals in comps.items():
            arr = np.array(vals)
            latency_summary["candidate_profiles"][cand][comp_name] = {
                "mean_ms": round(float(np.mean(arr)), 3),
                "median_ms": round(float(np.median(arr)), 3),
                "p90_ms": round(float(np.percentile(arr, 90)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "min_ms": round(float(np.min(arr)), 3),
                "max_ms": round(float(np.max(arr)), 3)
            }
        # Add end-to-end user perceived latency
        post_cap_mean = latency_summary["candidate_profiles"][cand]["total_post_capture_ms"]["mean_ms"]
        post_cap_p90 = latency_summary["candidate_profiles"][cand]["total_post_capture_ms"]["p90_ms"]
        latency_summary["candidate_profiles"][cand]["end_to_end_user_perceived"] = {
            "audio_capture_duration_ms": 1000.0,
            "post_capture_decision_mean_ms": post_cap_mean,
            "total_e2e_perceived_mean_ms": round(1000.0 + post_cap_mean, 3),
            "total_e2e_perceived_p90_ms": round(1000.0 + post_cap_p90, 3),
            "meets_50ms_post_capture_budget": bool(post_cap_mean <= 50.0)
        }

    return latency_summary

def run_phase60():
    print("=" * 80)
    print("Phase 60: Comprehensive Metric Reconciliation and Latency Audit")
    print("=" * 80)

    reconciled_results, manifest_counts = reconcile_denominators_and_metrics()
    latency_summary = decompose_and_reconcile_latency()

    out_json = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6", "predictions", "phase60_reproduced_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "test_manifest_structure": manifest_counts,
            "discrete_step_sizes": {
                "per_keyword_clip_step (1/40)": 0.0250,
                "command_clip_step (1/160)": 0.00625,
                "non_target_clip_step (1/560)": 0.0017857,
                "total_clip_step (1/720)": 0.0013889
            },
            "reconciled_candidates": reconciled_results,
            "latency_decomposition_audit": latency_summary
        }, f, indent=2)

    print(f"\nPhase 60 reproduced metrics and latency audit saved to {out_json}")

if __name__ == "__main__":
    run_phase60()
