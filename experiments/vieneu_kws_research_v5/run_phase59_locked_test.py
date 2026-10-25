import os
import sys
import json
import csv
import time
import math
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
sys.path.insert(0, BASE_DIR)

EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5")
DATA_DIR = os.path.join(EXP_DIR, "data", "splits")
CALIB_DIR = os.path.join(EXP_DIR, "calibration")
PRED_DIR = os.path.join(EXP_DIR, "predictions")

from src.models.kws_cnn import VietnameseKWSNet
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

def load_audio_clip(wav_path, target_length=16000):
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(BASE_DIR, wav_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}")
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    elif len(audio) > target_length:
        audio = audio[:target_length]
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio

def evaluate_test_finalists():
    print("=" * 80)
    print("Phase 59: Locked Test Evaluation on New Dataset v5 (720 clips)")
    print("=" * 80)

    test_manifest = os.path.join(DATA_DIR, "test_manifest.csv")
    test_rows = []
    with open(test_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_rows.append(r)

    print(f"Loaded {len(test_rows)} test clips.")
    # Check test clip counts
    cmd_counts = {c: sum(1 for r in test_rows if r["bounded_label"] == c) for c in ALL_5_CLASSES}
    print("Test Class Counts:", cmd_counts)

    # Load frozen calibration configs
    calib_file = os.path.join(CALIB_DIR, "selected_configs.json")
    with open(calib_file, "r", encoding="utf-8") as f:
        selected_configs = json.load(f)

    device = torch.device("cpu")  # CPU execution for true latency measurement
    torch.set_num_threads(4)
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    finalists = ["variant_a_cascade_v1", "variant_b_stage1_cnn"]
    seeds = [42, 43, 44]

    locked_test_summary = {}
    all_clip_preds = []

    for fin in finalists:
        locked_test_summary[fin] = {}
        for s in seeds:
            params = selected_configs[fin][f"seed{s}"]

            # Load models
            if fin == "variant_a_cascade_v1":
                s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{s}.pt")
                s2_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1", "checkpoints", f"best_verifier_seed{s}.pt")
                m1 = VietnameseKWSNet(num_classes=6).to(device)
                m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
                m1.eval()

                m2 = TemporalKWSVerifier(num_classes=5).to(device)
                m2.load_state_dict(torch.load(s2_ckpt, map_location=device)["model_state_dict"])
                m2.eval()

            elif fin == "variant_b_stage1_cnn":
                s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{s}.pt")
                m1 = VietnameseKWSNet(num_classes=6).to(device)
                m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
                m1.eval()
                m2 = None

            # Prediction loop & CPU latency timing
            latencies = []
            preds = []
            cm = {true_c: {pred_c: 0 for pred_c in ALL_5_CLASSES} for true_c in ALL_5_CLASSES}

            for r in test_rows:
                audio = load_audio_clip(r["kws_wav_path"])
                t_audio = torch.from_numpy(audio).unsqueeze(0).to(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    spec = frontend(t_audio)
                    p1 = F.softmax(m1(spec), dim=1).numpy()[0]
                    top_cmd_idx = int(np.argmax(p1[:4]))
                    s1_top = float(p1[:4].max())

                    pred_lbl = "REJECT"

                    if fin == "variant_a_cascade_v1":
                        th1 = params["theta_s1"]
                        th2 = params["theta_s2"]
                        if s1_top >= th1 and s1_top > p1[4] and s1_top > p1[5]:
                            crop = np.pad(audio, (0, 3200), mode="constant")
                            crop_t = torch.from_numpy(crop).unsqueeze(0).to(device)
                            crop_spec = frontend(crop_t)
                            p2 = F.softmax(m2(crop_spec), dim=1).numpy()[0]
                            if p2[top_cmd_idx] >= th2 and p2[top_cmd_idx] > p2[4]:
                                pred_lbl = TARGET_CLASSES[top_cmd_idx]

                    elif fin == "variant_b_stage1_cnn":
                        th = params["theta_cmd"]
                        if s1_top >= th and s1_top > p1[4] and s1_top > p1[5]:
                            pred_lbl = TARGET_CLASSES[top_cmd_idx]

                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                true_lbl = r["bounded_label"]
                cm[true_lbl][pred_lbl] += 1
                preds.append({
                    "finalist": fin,
                    "seed": s,
                    "sample_id": r["sample_id"],
                    "voice_id": r["voice_id"],
                    "true_label": true_lbl,
                    "pred_label": pred_lbl
                })

            all_clip_preds.extend(preds)

            # Compute detailed metrics
            cmd_hits = sum(cm[c][c] for c in TARGET_CLASSES)
            cmd_total = sum(sum(cm[c].values()) for c in TARGET_CLASSES)
            cmd_recall = cmd_hits / cmd_total

            per_kw_recall = {}
            per_kw_prec = {}
            per_kw_f1 = {}
            for c in TARGET_CLASSES:
                c_hits = cm[c][c]
                c_total = sum(cm[c].values())
                c_pred_total = sum(cm[other][c] for other in ALL_5_CLASSES)
                rec = c_hits / c_total if c_total > 0 else 0.0
                prec = c_hits / c_pred_total if c_pred_total > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                per_kw_recall[c] = round(rec, 4)
                per_kw_prec[c] = round(prec, 4)
                per_kw_f1[c] = round(f1, 4)

            nt_total = sum(cm["REJECT"].values())
            nt_rejects = cm["REJECT"]["REJECT"]
            nt_fa = nt_total - nt_rejects
            far = nt_fa / nt_total
            tnr = nt_rejects / nt_total

            overall_acc = (cmd_hits + nt_rejects) / 720.0

            ci_recall = wilson_score_interval(cmd_hits, cmd_total)
            ci_far = wilson_score_interval(nt_fa, nt_total)

            lat_arr = np.array(latencies)
            locked_test_summary[fin][f"seed{s}"] = {
                "calibrated_params": params,
                "metrics": {
                    "macro_command_recall": round(cmd_recall, 4),
                    "recall_95ci": ci_recall,
                    "false_rejection_rate": round(1.0 - cmd_recall, 4),
                    "non_target_false_acceptance_rate": round(far, 4),
                    "far_95ci": ci_far,
                    "non_target_rejection_rate": round(tnr, 4),
                    "overall_5class_accuracy": round(overall_acc, 4),
                    "per_keyword_recall": per_kw_recall,
                    "per_keyword_precision": per_kw_prec,
                    "per_keyword_f1": per_kw_f1,
                    "confusion_matrix": cm
                },
                "cpu_latency": {
                    "mean_ms": round(float(np.mean(lat_arr)), 2),
                    "median_ms": round(float(np.median(lat_arr)), 2),
                    "p90_ms": round(float(np.percentile(lat_arr, 90)), 2),
                    "min_ms": round(float(np.min(lat_arr)), 2),
                    "max_ms": round(float(np.max(lat_arr)), 2)
                }
            }
            print(f"[{fin}] Seed {s} TEST -> Recall: {cmd_recall*100:.2f}% (95% CI: {ci_recall}), FAR: {far*100:.2f}% (95% CI: {ci_far}), 5-Class Acc: {overall_acc*100:.2f}%, Latency: {np.mean(lat_arr):.2f}ms")

    # Finalist Averages across seeds
    for fin in finalists:
        recalls = [locked_test_summary[fin][f"seed{s}"]["metrics"]["macro_command_recall"] for s in seeds]
        fars = [locked_test_summary[fin][f"seed{s}"]["metrics"]["non_target_false_acceptance_rate"] for s in seeds]
        accs = [locked_test_summary[fin][f"seed{s}"]["metrics"]["overall_5class_accuracy"] for s in seeds]
        lats = [locked_test_summary[fin][f"seed{s}"]["cpu_latency"]["mean_ms"] for s in seeds]

        locked_test_summary[fin]["mean_summary"] = {
            "mean_command_recall": round(float(np.mean(recalls)), 4),
            "mean_false_acceptance_rate": round(float(np.mean(fars)), 4),
            "mean_rejection_accuracy": round(float(1.0 - np.mean(fars)), 4),
            "mean_5class_accuracy": round(float(np.mean(accs)), 4),
            "mean_cpu_latency_ms": round(float(np.mean(lats)), 2)
        }
        print(f"\n===> {fin} TEST MEAN:")
        print(f"     Command Recall: {np.mean(recalls)*100:.2f}%")
        print(f"     False Acceptance Rate (FAR): {np.mean(fars)*100:.2f}%")
        print(f"     Overall 5-Class Accuracy: {np.mean(accs)*100:.2f}%")
        print(f"     Mean CPU Latency: {np.mean(lats):.2f} ms")

    # Save locked test summary and per-clip predictions
    test_sum_file = os.path.join(PRED_DIR, "locked_test_summary.json")
    with open(test_sum_file, "w", encoding="utf-8") as f:
        json.dump(locked_test_summary, f, indent=2)

    pred_csv_file = os.path.join(PRED_DIR, "per_clip_predictions.csv")
    with open(pred_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["finalist", "seed", "sample_id", "voice_id", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(all_clip_preds)

    print(f"\nLocked test outputs saved to:")
    print(f"- {test_sum_file}")
    print(f"- {pred_csv_file}")

if __name__ == "__main__":
    evaluate_test_finalists()
