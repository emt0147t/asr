import os
import sys
import json
import csv
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v4.models.bounded_utterance.train_bounded_model import VietnameseBoundedKWSNet, LogMelFrontend

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
BOUNDED_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]

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
    return audio

def evaluate_dataset_scores(candidate_id, seed, manifest_path, device, frontend):
    """
    Extract raw predictions for each clip in the manifest without thresholding.
    """
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Load models
    if candidate_id == "1_stage1_cnn_baseline":
        # Candidate 1: Stage 1 CNN alone
        ckpt_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{seed}.pt")
        model = VietnameseKWSNet(num_classes=6).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        s2_model = None

    elif candidate_id == "2_cascade_v1_bounded":
        # Candidate 2: Stage 1 CNN + Stage 2 TCN Verifier
        s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{seed}.pt")
        s2_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1", "checkpoints", f"best_verifier_seed{seed}.pt")
        model = VietnameseKWSNet(num_classes=6).to(device)
        ckpt1 = torch.load(s1_ckpt, map_location=device)
        model.load_state_dict(ckpt1["model_state_dict"])
        model.eval()

        s2_model = TemporalKWSVerifier(num_classes=5).to(device)
        ckpt2 = torch.load(s2_ckpt, map_location=device)
        s2_model.load_state_dict(ckpt2["model_state_dict"])
        s2_model.eval()

    elif candidate_id == "3_bounded_kws_net":
        # Candidate 3: VietnameseBoundedKWSNet
        ckpt_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "models", "bounded_utterance", "checkpoints", f"bounded_kws_net_seed{seed}.pt")
        model = VietnameseBoundedKWSNet(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        s2_model = None
    else:
        raise ValueError(f"Unknown candidate {candidate_id}")

    results = []
    t_start = time.time()
    for row in rows:
        audio = load_audio_clip(row["kws_wav_path"])
        t_audio = torch.from_numpy(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            specs = frontend(t_audio)
            if candidate_id == "1_stage1_cnn_baseline":
                probs = F.softmax(model(specs), dim=1).cpu().numpy()[0]
                results.append({
                    "sample_id": row["sample_id"],
                    "true_label": row["bounded_label"],
                    "voice_id": row["voice_id"],
                    "probs": probs.tolist(),
                    "top_cmd_idx": int(np.argmax(probs[:4])),
                    "top_cmd_prob": float(probs[:4].max()),
                    "p_unk": float(probs[4]),
                    "p_sil": float(probs[5])
                })
            elif candidate_id == "2_cascade_v1_bounded":
                p1 = F.softmax(model(specs), dim=1).cpu().numpy()[0]
                top_cmd_idx = int(np.argmax(p1[:4]))
                top_cmd_prob = float(p1[:4].max())
                
                # S2 crop is 19200 samples
                crop = np.pad(audio, (0, 3200), mode="constant")
                crop_t = torch.from_numpy(crop).unsqueeze(0).to(device)
                crop_spec = frontend(crop_t)
                p2 = F.softmax(s2_model(crop_spec), dim=1).cpu().numpy()[0]
                results.append({
                    "sample_id": row["sample_id"],
                    "true_label": row["bounded_label"],
                    "voice_id": row["voice_id"],
                    "s1_probs": p1.tolist(),
                    "s2_probs": p2.tolist(),
                    "top_cmd_idx": top_cmd_idx,
                    "s1_top_prob": top_cmd_prob,
                    "s2_top_prob": float(p2[top_cmd_idx]),
                    "s2_rej_prob": float(p2[4])
                })
            elif candidate_id == "3_bounded_kws_net":
                probs = F.softmax(model(specs), dim=1).cpu().numpy()[0]
                results.append({
                    "sample_id": row["sample_id"],
                    "true_label": row["bounded_label"],
                    "voice_id": row["voice_id"],
                    "probs": probs.tolist(),
                    "top_cmd_idx": int(np.argmax(probs[:4])),
                    "top_cmd_prob": float(probs[:4].max()),
                    "rej_prob": float(probs[4])
                })

    elapsed_ms = (time.time() - t_start) * 1000.0 / len(rows)
    return results, elapsed_ms


def score_predictions(candidate_id, raw_results, params):
    """
    Applies calibrated thresholds to raw prediction outputs.
    """
    preds = []
    for r in raw_results:
        true_lbl = r["true_label"]
        pred_lbl = "REJECT"

        if candidate_id == "1_stage1_cnn_baseline":
            th = params["theta_cmd"]
            if r["top_cmd_prob"] >= th and r["top_cmd_prob"] > r["p_unk"] and r["top_cmd_prob"] > r["p_sil"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        elif candidate_id == "2_cascade_v1_bounded":
            th1 = params["theta_s1"]
            th2 = params["theta_s2"]
            if r["s1_top_prob"] >= th1 and r["s2_top_prob"] >= th2 and r["s2_top_prob"] > r["s2_rej_prob"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        elif candidate_id == "3_bounded_kws_net":
            th = params["theta_reject"]
            # Decision rule: command probability exceeds reject threshold and exceeds reject probability
            if r["top_cmd_prob"] >= th and r["top_cmd_prob"] > r["rej_prob"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        preds.append({
            "sample_id": r["sample_id"],
            "true_label": true_lbl,
            "pred_label": pred_lbl,
            "voice_id": r["voice_id"]
        })

    # Compute metrics
    # Target commands: TRAI, PHAI, LEN, XUONG (each 40 in val/test)
    cmd_hits = 0
    cmd_total = 0
    kw_hits = {kw: 0 for kw in TARGET_CLASSES}
    kw_total = {kw: 0 for kw in TARGET_CLASSES}

    # Non-target: REJECT (200 in val, 560 in test)
    nt_rejects = 0
    nt_false_accepts = 0
    nt_total = 0

    correct_overall = 0
    total_overall = len(preds)

    for p in preds:
        t = p["true_label"]
        pr = p["pred_label"]

        if t in TARGET_CLASSES:
            cmd_total += 1
            kw_total[t] += 1
            if pr == t:
                cmd_hits += 1
                kw_hits[t] += 1
                correct_overall += 1
        elif t == "REJECT":
            nt_total += 1
            if pr == "REJECT":
                nt_rejects += 1
                correct_overall += 1
            else:
                nt_false_accepts += 1

    overall_recall = cmd_hits / cmd_total if cmd_total > 0 else 0.0
    far = nt_false_accepts / nt_total if nt_total > 0 else 0.0
    tnr = nt_rejects / nt_total if nt_total > 0 else 0.0
    overall_acc = correct_overall / total_overall if total_overall > 0 else 0.0

    per_kw_recall = {kw: (kw_hits[kw] / kw_total[kw] if kw_total[kw] > 0 else 0.0) for kw in TARGET_CLASSES}

    return {
        "overall_command_recall": round(overall_recall, 4),
        "false_rejection_rate": round(1.0 - overall_recall, 4),
        "non_target_false_acceptance_rate": round(far, 4),
        "non_target_rejection_rate": round(tnr, 4),
        "overall_5class_accuracy": round(overall_acc, 4),
        "per_keyword_recall": per_kw_recall,
        "raw_counts": {
            "cmd_hits": cmd_hits,
            "cmd_total": cmd_total,
            "nt_false_accepts": nt_false_accepts,
            "nt_rejects": nt_rejects,
            "nt_total": nt_total
        },
        "preds": preds
    }


def run_campaign():
    print("=== Phase 53: Validation-Only Calibration & Locked Utterance Test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)

    data_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "data", "splits")
    val_manifest = os.path.join(data_dir, "val_manifest.csv")
    test_manifest = os.path.join(data_dir, "test_manifest.csv")

    candidates = ["1_stage1_cnn_baseline", "2_cascade_v1_bounded", "3_bounded_kws_net"]
    seeds = [42, 43, 44]

    selected_configs = {}
    val_summary = {}

    # 1. Validation Calibration
    print("\n--- 1. Calibrating on Validation Split (360 clips) ---")
    for cand in candidates:
        selected_configs[cand] = {}
        val_summary[cand] = {}
        for seed in seeds:
            raw_val, lat = evaluate_dataset_scores(cand, seed, val_manifest, device, frontend)

            best_params = None
            best_score = -1.0
            best_metrics = None

            if cand == "1_stage1_cnn_baseline":
                for th in np.arange(0.20, 0.95, 0.05):
                    params = {"theta_cmd": round(float(th), 2)}
                    m = score_predictions(cand, raw_val, params)
                    # Objective: High Command Recall + FAR <= 0.08
                    # Score = Recall - 2.0 * max(0, FAR - 0.05)
                    score = m["overall_command_recall"] - 2.5 * m["non_target_false_acceptance_rate"]
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = m

            elif cand == "2_cascade_v1_bounded":
                for th1 in np.arange(0.30, 0.85, 0.10):
                    for th2 in np.arange(0.30, 0.85, 0.10):
                        params = {"theta_s1": round(float(th1), 2), "theta_s2": round(float(th2), 2)}
                        m = score_predictions(cand, raw_val, params)
                        score = m["overall_command_recall"] - 2.5 * m["non_target_false_acceptance_rate"]
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_metrics = m

            elif cand == "3_bounded_kws_net":
                for th in np.arange(0.20, 0.95, 0.05):
                    params = {"theta_reject": round(float(th), 2)}
                    m = score_predictions(cand, raw_val, params)
                    score = m["overall_command_recall"] - 2.5 * m["non_target_false_acceptance_rate"]
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = m

            selected_configs[cand][f"seed{seed}"] = best_params
            val_summary[cand][f"seed{seed}"] = {
                "selected_params": best_params,
                "metrics": {
                    "overall_command_recall": best_metrics["overall_command_recall"],
                    "false_rejection_rate": best_metrics["false_rejection_rate"],
                    "non_target_false_acceptance_rate": best_metrics["non_target_false_acceptance_rate"],
                    "non_target_rejection_rate": best_metrics["non_target_rejection_rate"],
                    "overall_5class_accuracy": best_metrics["overall_5class_accuracy"],
                    "per_keyword_recall": best_metrics["per_keyword_recall"]
                }
            }
            print(f"[{cand}] Seed {seed}: Best Params {best_params} -> Val Recall: {best_metrics['overall_command_recall']*100:.2f}%, FAR: {best_metrics['non_target_false_acceptance_rate']*100:.2f}%, Acc: {best_metrics['overall_5class_accuracy']*100:.2f}%")

    calib_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "calibration", "selected_bounded_utterance_configs.json")
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(selected_configs, f, indent=2)

    val_sum_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "predictions", "validation_summary.json")
    with open(val_sum_file, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    # 2. Locked Evaluation on Held-Out Test Set
    print("\n--- 2. Evaluating on Locked Held-Out Test Set (720 clips) ---")
    locked_test_summary = {}
    all_test_preds = []

    for cand in candidates:
        locked_test_summary[cand] = {}
        for seed in seeds:
            params = selected_configs[cand][f"seed{seed}"]
            raw_test, lat_ms = evaluate_dataset_scores(cand, seed, test_manifest, device, frontend)
            m = score_predictions(cand, raw_test, params)

            locked_test_summary[cand][f"seed{seed}"] = {
                "calibrated_params": params,
                "latency_ms_per_clip": round(lat_ms, 2),
                "metrics": {
                    "overall_command_recall": m["overall_command_recall"],
                    "false_rejection_rate": m["false_rejection_rate"],
                    "non_target_false_acceptance_rate": m["non_target_false_acceptance_rate"],
                    "non_target_rejection_rate": m["non_target_rejection_rate"],
                    "overall_5class_accuracy": m["overall_5class_accuracy"],
                    "per_keyword_recall": m["per_keyword_recall"],
                    "raw_counts": m["raw_counts"]
                }
            }

            for p in m["preds"]:
                all_test_preds.append({
                    "candidate": cand,
                    "seed": seed,
                    "sample_id": p["sample_id"],
                    "voice_id": p["voice_id"],
                    "true_label": p["true_label"],
                    "pred_label": p["pred_label"]
                })

            print(f"[{cand}] Seed {seed} TEST -> Recall: {m['overall_command_recall']*100:.2f}%, FAR: {m['non_target_false_acceptance_rate']*100:.2f}%, Acc: {m['overall_5class_accuracy']*100:.2f}%, Latency: {lat_ms:.2f}ms")

    # Compute Candidate Averages on Test Set
    for cand in candidates:
        recalls = [locked_test_summary[cand][f"seed{s}"]["metrics"]["overall_command_recall"] for s in seeds]
        fars = [locked_test_summary[cand][f"seed{s}"]["metrics"]["non_target_false_acceptance_rate"] for s in seeds]
        accs = [locked_test_summary[cand][f"seed{s}"]["metrics"]["overall_5class_accuracy"] for s in seeds]
        lats = [locked_test_summary[cand][f"seed{s}"]["latency_ms_per_clip"] for s in seeds]
        locked_test_summary[cand]["mean_summary"] = {
            "mean_command_recall": round(float(np.mean(recalls)), 4),
            "mean_false_acceptance_rate": round(float(np.mean(fars)), 4),
            "mean_rejection_accuracy": round(float(1.0 - np.mean(fars)), 4),
            "mean_overall_accuracy": round(float(np.mean(accs)), 4),
            "mean_latency_ms": round(float(np.mean(lats)), 2)
        }
        print(f"\n===> {cand} MEAN TEST SUMMARY:")
        print(f"     Command Recall: {np.mean(recalls)*100:.2f}%")
        print(f"     False Acceptance Rate (FAR): {np.mean(fars)*100:.2f}%")
        print(f"     Overall 5-Class Accuracy: {np.mean(accs)*100:.2f}%")
        print(f"     Mean Latency: {np.mean(lats):.2f} ms")

    test_sum_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "predictions", "locked_test_summary.json")
    with open(test_sum_file, "w", encoding="utf-8") as f:
        json.dump(locked_test_summary, f, indent=2)

    pred_csv_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "predictions", "per_utterance_predictions.csv")
    with open(pred_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "seed", "sample_id", "voice_id", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(all_test_preds)

    print(f"\nAll outputs saved successfully:")
    print(f"- Calibration configs: {calib_file}")
    print(f"- Validation summary: {val_sum_file}")
    print(f"- Locked test summary: {test_sum_file}")
    print(f"- Per-utterance predictions: {pred_csv_file}")

if __name__ == "__main__":
    run_campaign()
