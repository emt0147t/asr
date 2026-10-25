import os
import sys
import json
import csv
import time
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
from experiments.vieneu_kws_research_v4.models.bounded_utterance.train_bounded_model import (
    VietnameseBoundedKWSNet,
    LogMelFrontend
)

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
ALL_5_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]

def audit_metrics_from_csv():
    print("--- 1. Auditing Metrics Reproduction from Prediction CSV ---")
    pred_csv_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "predictions", "per_utterance_predictions.csv")
    test_summary_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "predictions", "locked_test_summary.json")

    with open(test_summary_path, "r", encoding="utf-8") as f:
        reported_summary = json.load(f)

    df_preds = pd.read_csv(pred_csv_path)

    recomputed = {}
    mismatches = []
    confusion_matrices = {}

    candidates = ["1_stage1_cnn_baseline", "2_cascade_v1_bounded", "3_bounded_kws_net"]
    seeds = [42, 43, 44]

    for cand in candidates:
        recomputed[cand] = {}
        confusion_matrices[cand] = {}
        for s in seeds:
            sub = df_preds[(df_preds["candidate"] == cand) & (df_preds["seed"] == s)]
            assert len(sub) == 720, f"Expected 720 clips, got {len(sub)}"

            # Compute confusion matrix
            cm = {true_c: {pred_c: 0 for pred_c in ALL_5_CLASSES} for true_c in ALL_5_CLASSES}
            for _, r in sub.iterrows():
                cm[r["true_label"]][r["pred_label"]] += 1
            confusion_matrices[cand][f"seed{s}"] = cm

            # Metrics
            # Target commands
            cmd_hits = sum(cm[c][c] for c in TARGET_CLASSES)
            cmd_total = sum(sum(cm[c].values()) for c in TARGET_CLASSES)
            cmd_recall = cmd_hits / cmd_total

            per_kw_recall = {c: cm[c][c] / sum(cm[c].values()) for c in TARGET_CLASSES}

            # Non-target
            nt_total = sum(cm["REJECT"].values())
            nt_rejects = cm["REJECT"]["REJECT"]
            nt_false_accepts = nt_total - nt_rejects
            far = nt_false_accepts / nt_total
            tnr = nt_rejects / nt_total

            overall_acc = (cmd_hits + nt_rejects) / 720.0

            recomputed[cand][f"seed{s}"] = {
                "cmd_recall": round(cmd_recall, 4),
                "far": round(far, 4),
                "tnr": round(tnr, 4),
                "accuracy": round(overall_acc, 4),
                "per_kw_recall": {k: round(v, 4) for k, v in per_kw_recall.items()}
            }

            rep = reported_summary[cand][f"seed{s}"]["metrics"]
            # Verify bit-exact or round match
            if abs(rep["overall_command_recall"] - round(cmd_recall, 4)) > 1e-4:
                mismatches.append(f"{cand} s{s} recall: rep {rep['overall_command_recall']} vs recomputed {round(cmd_recall, 4)}")
            if abs(rep["non_target_false_acceptance_rate"] - round(far, 4)) > 1e-4:
                mismatches.append(f"{cand} s{s} far: rep {rep['non_target_false_acceptance_rate']} vs recomputed {round(far, 4)}")
            if abs(rep["overall_5class_accuracy"] - round(overall_acc, 4)) > 1e-4:
                mismatches.append(f"{cand} s{s} acc: rep {rep['overall_5class_accuracy']} vs recomputed {round(overall_acc, 4)}")

    print(f"Metrics reproduction completed. Mismatches: {len(mismatches)}")
    if mismatches:
        for m in mismatches:
            print("  MISMATCH:", m)

    return recomputed, confusion_matrices, mismatches


def benchmark_cpu_latency():
    print("\n--- 2. Benchmarking True CPU Latency (Batch Size = 1) ---")
    device = torch.device("cpu")
    torch.set_num_threads(4)

    frontend = LogMelFrontend().to(device)
    frontend.eval()

    # Load one checkpoint for each candidate
    # Candidate 1: Stage 1 CNN
    c1_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", "best_checkpoint_seed42.pt")
    m1 = VietnameseKWSNet(num_classes=6).to(device)
    m1.load_state_dict(torch.load(c1_path, map_location=device)["model_state_dict"])
    m1.eval()

    # Candidate 2: Stage 1 CNN + Stage 2 TCN Verifier
    c2_s2_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1", "checkpoints", "best_verifier_seed42.pt")
    m2_s2 = TemporalKWSVerifier(num_classes=5).to(device)
    m2_s2.load_state_dict(torch.load(c2_s2_path, map_location=device)["model_state_dict"])
    m2_s2.eval()

    # Candidate 3: BoundedKWSNet
    c3_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "models", "bounded_utterance", "checkpoints", "bounded_kws_net_seed42.pt")
    m3 = VietnameseBoundedKWSNet(num_classes=5).to(device)
    m3.load_state_dict(torch.load(c3_path, map_location=device)["model_state_dict"])
    m3.eval()

    dummy_audio = torch.randn(1, 16000, device=device)
    dummy_crop = torch.randn(1, 19200, device=device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            s = frontend(dummy_audio)
            _ = m1(s)
            _ = m3(s)
            s_crop = frontend(dummy_crop)
            _ = m2_s2(s_crop)

    num_trials = 200
    timings = {"1_stage1_cnn": [], "2_cascade_v1": [], "3_bounded_kws_net": []}

    for _ in range(num_trials):
        # Candidate 1: Frontend + M1 + threshold logic
        t0 = time.perf_counter()
        with torch.no_grad():
            spec = frontend(dummy_audio)
            logits = m1(spec)
            p = F.softmax(logits, dim=1).numpy()[0]
            top_c = int(np.argmax(p[:4]))
            val = float(p[top_c])
            _ = (val >= 0.9 and val > p[4] and val > p[5])
        t1 = time.perf_counter()
        timings["1_stage1_cnn"].append((t1 - t0) * 1000.0)

        # Candidate 2: Frontend + M1 + (if candidate passes) Stage 2 Crop + M2 + S2 logic
        t0 = time.perf_counter()
        with torch.no_grad():
            spec = frontend(dummy_audio)
            logits = m1(spec)
            p1 = F.softmax(logits, dim=1).numpy()[0]
            top_c = int(np.argmax(p1[:4]))
            val1 = float(p1[top_c])
            if val1 >= 0.6 and val1 > p1[4] and val1 > p1[5]:
                spec2 = frontend(dummy_crop)
                p2 = F.softmax(m2_s2(spec2), dim=1).numpy()[0]
                _ = (p2[top_c] >= 0.3 and p2[top_c] > p2[4])
        t1 = time.perf_counter()
        timings["2_cascade_v1"].append((t1 - t0) * 1000.0)

        # Candidate 3: Frontend + M3 + rejection logic
        t0 = time.perf_counter()
        with torch.no_grad():
            spec = frontend(dummy_audio)
            logits = m3(spec)
            p = F.softmax(logits, dim=1).numpy()[0]
            top_c = int(np.argmax(p[:4]))
            val = float(p[top_c])
            _ = (val >= 0.4 and val > p[4])
        t1 = time.perf_counter()
        timings["3_bounded_kws_net"].append((t1 - t0) * 1000.0)

    cpu_stats = {}
    for k, v in timings.items():
        arr = np.array(v)
        cpu_stats[k] = {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2)
        }
        print(f"[{k}] CPU Latency (ms): Mean={cpu_stats[k]['mean_ms']}ms, Median={cpu_stats[k]['median_ms']}ms, P90={cpu_stats[k]['p90_ms']}ms, Max={cpu_stats[k]['max_ms']}ms")

    return cpu_stats


def run_phase55_audit():
    recomputed, confusion_matrices, mismatches = audit_metrics_from_csv()
    cpu_stats = benchmark_cpu_latency()

    out_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5", "phase55_audit_findings.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics_reproduced": len(mismatches) == 0,
            "mismatches": mismatches,
            "recomputed_metrics": recomputed,
            "confusion_matrices": confusion_matrices,
            "cpu_latency_benchmark": cpu_stats
        }, f, indent=2)

    print(f"\nPhase 55 audit findings saved to {out_file}")

if __name__ == "__main__":
    run_phase55_audit()
