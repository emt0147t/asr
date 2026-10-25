"""
Phase 38: Cascaded KWS Evaluation and Validation-Only Calibration Pipeline
Evaluates Stage 1 (CNN v2) + Stage 2 (Temporal Verifier) in continuous streaming.

Executes:
1. Validation-Only Calibration & Constraint Checking:
   - Evaluates on validation continuous streams (commands and non-target).
   - Strict constraints: Target Recall >= 95%, Non-target FA/hr <= 1.0, Latency <= 600 ms.
   - Declares NO_FEASIBLE_VALIDATION_OPERATING_POINT if strict constraints are unmet.
2. Final Held-Out Test Evaluation:
   - Evaluates frozen held-out test streams.
   - Evaluates Stage 1 Alone vs Cascade (Stage 1 + Stage 2) on the identical stream protocol.
   - Controlled noise robustness suite across White, Pink, HVAC, Babble at 20, 15, 10, 5, 0 dB.
   - Benchmarks single-sample CPU latencies.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier, VERIFIER_CLASSES
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_v2.evaluate import (
    match_events_and_evaluate,
    generate_noisy_test_batch,
    apply_confidence_policy,
    TARGET_COMMAND_INDICES
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(EXP_DIR, "predictions")
CALIBRATION_DIR = os.path.join(EXP_DIR, "calibration")
CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")

STREAMING_DIR = os.path.join(BASE_DIR, "experiments/vieneu_kws_v2/streaming_simulation")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
LABEL_TO_ID_6 = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5}


class CascadedStreamingEngine:
    def __init__(self, stage1_model, stage2_model, extractor, device,
                 theta1: float = 0.50, theta2: float = 0.60,
                 consecutive_hits: int = 2, debounce_sec: float = 0.60,
                 crop_samples: int = 19200, crop_offset_samples: int = 4800):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.extractor = extractor
        self.device = device
        self.theta1 = theta1
        self.theta2 = theta2
        self.consecutive_hits = consecutive_hits
        self.debounce_sec = debounce_sec
        self.crop_samples = crop_samples
        self.crop_offset_samples = crop_offset_samples
        self.win_samples = 16000
        self.hop_samples = 1600
        self.sample_rate = 16000

    def process_stream(self, audio_data: np.ndarray, bypass_stage2: bool = False):
        self.stage1.eval()
        if self.stage2 is not None:
            self.stage2.eval()

        stream_len = len(audio_data)
        num_windows = (stream_len - self.win_samples) // self.hop_samples + 1
        chunk_size = 128

        confirmed_events = []
        all_stage1_proposals = []

        streak_class = None
        consecutive_count = 0
        lockout_sample = -1

        for w_start in range(0, num_windows, chunk_size):
            w_end = min(num_windows, w_start + chunk_size)
            batch_windows = []
            w_indices = []

            for w in range(w_start, w_end):
                s_idx = w * self.hop_samples
                e_idx = s_idx + self.win_samples
                batch_windows.append(audio_data[s_idx:e_idx])
                w_indices.append(w)

            batch_t = torch.from_numpy(np.array(batch_windows, dtype=np.float32)).float().to(self.device)
            with torch.no_grad():
                feats = self.extractor(batch_t)
                probs = F.softmax(self.stage1(feats), dim=1).cpu().numpy()

            for i, w in enumerate(w_indices):
                s_idx = w * self.hop_samples
                t_sec = s_idx / self.sample_rate
                p = probs[i]

                if s_idx < lockout_sample:
                    streak_class = None
                    consecutive_count = 0
                    continue

                tgt_p = p[:4]
                max_c = int(np.argmax(tgt_p))
                max_val = float(tgt_p[max_c])

                # Stage 1 proposal logic
                if max_val >= self.theta1 and max_val > p[4] and max_val > p[5]:
                    if max_c == streak_class:
                        consecutive_count += 1
                    else:
                        streak_class = max_c
                        consecutive_count = 1

                    if consecutive_count >= self.consecutive_hits:
                        t_trig = (s_idx + self.win_samples) / self.sample_rate
                        proposal = {
                            "w": w,
                            "s_idx": s_idx,
                            "class_idx": max_c,
                            "class_label": TARGET_CLASSES[max_c],
                            "stage1_conf": max_val,
                            "trigger_time_sec": t_trig,
                            "window_start_sec": t_sec
                        }
                        all_stage1_proposals.append(proposal)

                        if bypass_stage2 or self.stage2 is None:
                            # Stage 1 directly emits event
                            confirmed_events.append({
                                "trigger_time_sec": round(t_trig, 3),
                                "pred_label": TARGET_CLASSES[max_c],
                                "confidence": round(max_val, 4),
                                "window_start_sec": round(t_sec, 3)
                            })
                            lockout_sample = s_idx + int(self.debounce_sec * self.sample_rate)
                            streak_class = None
                            consecutive_count = 0
                        else:
                            # Pass candidate to Stage 2 Temporal Verifier
                            c_start = max(0, s_idx + self.crop_offset_samples)
                            c_end = c_start + self.crop_samples
                            if c_end <= stream_len:
                                crop = audio_data[c_start:c_end]
                            else:
                                crop = np.pad(audio_data[c_start:], (0, c_end - stream_len))

                            crop_t = torch.from_numpy(crop).unsqueeze(0).float().to(self.device)
                            with torch.no_grad():
                                feat2 = self.extractor(crop_t)
                                p2 = F.softmax(self.stage2(feat2), dim=1).cpu().numpy()[0]

                            s2_conf = float(p2[max_c])
                            s2_rej = float(p2[4])

                            # Stage 2 verification rule
                            if s2_conf >= self.theta2 and s2_conf > s2_rej:
                                confirmed_events.append({
                                    "trigger_time_sec": round(t_trig, 3),
                                    "pred_label": TARGET_CLASSES[max_c],
                                    "confidence": round(s2_conf, 4),
                                    "stage1_confidence": round(max_val, 4),
                                    "window_start_sec": round(t_sec, 3)
                                })
                                lockout_sample = s_idx + int(self.debounce_sec * self.sample_rate)
                                streak_class = None
                                consecutive_count = 0
                            else:
                                # Rejected by Stage 2
                                streak_class = None
                                consecutive_count = 0
                else:
                    streak_class = None
                    consecutive_count = 0

        return confirmed_events, all_stage1_proposals


def run_cascade_campaign():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    cfg_p = os.path.join(EXP_DIR, "config.json")
    with open(cfg_p, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cascade Evaluation Engine Running on Device: {device}")

    extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
    ).to(device)

    # Load audio streams
    val_cmd_audio_p = os.path.join(STREAMING_DIR, "val_stream_continuous_commands.wav")
    val_cmd_gt_p = os.path.join(STREAMING_DIR, "val_stream_commands_ground_truth.csv")
    val_nt_audio_p = os.path.join(STREAMING_DIR, "val_stream_continuous_nontarget.wav")
    val_nt_gt_p = os.path.join(STREAMING_DIR, "val_stream_nontarget_ground_truth.csv")

    test_cmd_audio_p = os.path.join(STREAMING_DIR, "test_stream_continuous_commands.wav")
    test_cmd_gt_p = os.path.join(STREAMING_DIR, "test_stream_commands_ground_truth.csv")
    test_nt_audio_p = os.path.join(STREAMING_DIR, "test_stream_continuous_nontarget.wav")
    test_nt_gt_p = os.path.join(STREAMING_DIR, "test_stream_nontarget_ground_truth.csv")

    val_cmd_audio, _ = sf.read(val_cmd_audio_p, dtype="float32")
    val_cmd_gt = pd.read_csv(val_cmd_gt_p)
    val_nt_audio, _ = sf.read(val_nt_audio_p, dtype="float32")
    val_nt_gt = pd.read_csv(val_nt_gt_p)

    test_cmd_audio, _ = sf.read(test_cmd_audio_p, dtype="float32")
    test_cmd_gt = pd.read_csv(test_cmd_gt_p)
    test_nt_audio, _ = sf.read(test_nt_audio_p, dtype="float32")
    test_nt_gt = pd.read_csv(test_nt_gt_p)

    df_test_manifest = pd.read_csv(os.path.join(BASE_DIR, config["dataset"]["test_manifest"]))
    test_targets_6 = np.array([LABEL_TO_ID_6[l] for l in df_test_manifest["class_label"]])

    seeds = config["stage2"]["training"]["seeds"]
    campaign_summary = {}
    selected_thresholds = {}
    validation_calibration_records = {}

    min_tgt_rec = config["validation_constraints"]["min_target_command_recall"]
    max_fa_hr = config["validation_constraints"]["max_streaming_fa_per_hour"]
    max_lat = config["validation_constraints"]["max_detection_latency_ms"]

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"Evaluating Cascaded KWS Pipeline (Seed {seed})")
        print(f"=======================================================")

        # Load Stage 1
        s1_ckpt_p = os.path.join(BASE_DIR, config["stage1"]["checkpoints"][f"seed{seed}"])
        s1_ckpt = torch.load(s1_ckpt_p, map_location="cpu", weights_only=False)
        stage1 = VietnameseKWSNet(num_classes=6)
        stage1.load_state_dict(s1_ckpt["model_state_dict"])
        stage1.to(device)
        stage1.eval()

        # Load Stage 2
        s2_ckpt_p = os.path.join(CHECKPOINTS_DIR, f"best_verifier_seed{seed}.pt")
        s2_ckpt = torch.load(s2_ckpt_p, map_location="cpu", weights_only=False)
        stage2 = TemporalKWSVerifier(num_classes=5)
        stage2.load_state_dict(s2_ckpt["model_state_dict"])
        stage2.to(device)
        stage2.eval()

        # -------------------------------------------------------------
        # Part 1: Validation Calibration & Constraint Sweep
        # -------------------------------------------------------------
        print("\n--- Running Validation-Only Calibration Sweep ---")
        val_eval_candidates = []
        feasible_configs = []

        candidate_th1 = [0.45, 0.50, 0.55]
        candidate_th2 = [0.40, 0.50, 0.60, 0.70, 0.80]

        for th1 in candidate_th1:
            for th2 in candidate_th2:
                engine = CascadedStreamingEngine(
                    stage1, stage2, extractor, device,
                    theta1=th1, theta2=th2,
                    crop_samples=19200, crop_offset_samples=4800
                )
                cmd_trigs, s1_props = engine.process_stream(val_cmd_audio, bypass_stage2=False)
                m_cmd = match_events_and_evaluate(cmd_trigs, val_cmd_gt, len(val_cmd_audio) / 16000.0)

                nt_trigs, _ = engine.process_stream(val_nt_audio, bypass_stage2=False)
                m_nt = match_events_and_evaluate(nt_trigs, val_nt_gt, len(val_nt_audio) / 16000.0)

                is_strictly_ok = (
                    m_cmd["recall"] >= min_tgt_rec and
                    m_nt["fa_per_hour"] <= max_fa_hr and
                    (m_cmd["latencies_ms"]["mean"] is not None and m_cmd["latencies_ms"]["mean"] <= max_lat)
                )

                rec_entry = {
                    "theta1": th1,
                    "theta2": th2,
                    "val_cmd_recall": m_cmd["recall"],
                    "val_cmd_precision": m_cmd["precision"],
                    "val_cmd_mean_lat_ms": m_cmd["latencies_ms"]["mean"],
                    "val_nt_fa_per_hour": m_nt["fa_per_hour"],
                    "val_nt_false_alarms": m_nt["false_positives"],
                    "strictly_feasible": is_strictly_ok
                }
                val_eval_candidates.append(rec_entry)
                if is_strictly_ok:
                    feasible_configs.append(rec_entry)

        if feasible_configs:
            best_cfg = max(feasible_configs, key=lambda x: (x["val_cmd_recall"], -x["val_nt_fa_per_hour"]))
            is_feasible = True
            selected_th1 = best_cfg["theta1"]
            selected_th2 = best_cfg["theta2"]
            rationale = "Strict validation constraints satisfied."
        else:
            is_feasible = False
            # Select best empirical Pareto compromise: Target recall >= 90%, minimize FA/hr
            compromises = [c for c in val_eval_candidates if c["val_cmd_recall"] >= 0.90]
            if compromises:
                best_cfg = min(compromises, key=lambda x: (x["val_nt_fa_per_hour"], -x["val_cmd_recall"]))
            else:
                best_cfg = min(val_eval_candidates, key=lambda x: x["val_nt_fa_per_hour"])
            selected_th1 = best_cfg["theta1"]
            selected_th2 = best_cfg["theta2"]
            rationale = (f"NO_FEASIBLE_VALIDATION_OPERATING_POINT: FA/hr <= 1.0 unmet on unconstrained speech. "
                         f"Selected Pareto compromise (theta1={selected_th1:.2f}, theta2={selected_th2:.2f}) for test evaluation.")

        print(f"Validation Result Seed {seed}: Feasible = {is_feasible}")
        print(f"  Selected Thresholds: theta1 = {selected_th1:.2f}, theta2 = {selected_th2:.2f}")
        print(f"  Rationale: {rationale}")

        selected_thresholds[f"seed{seed}"] = {
            "theta1": selected_th1,
            "theta2": selected_th2,
            "is_strictly_feasible": is_feasible,
            "rationale": rationale
        }
        validation_calibration_records[f"seed{seed}"] = {
            "selected": selected_thresholds[f"seed{seed}"],
            "sweep": val_eval_candidates
        }

        # -------------------------------------------------------------
        # Part 2: Final Held-Out Test Evaluation
        # -------------------------------------------------------------
        print("\n--- Evaluating Held-Out Test Streams ---")

        # 1. Stage 1 Alone Baseline on Test Streams
        s1_engine = CascadedStreamingEngine(
            stage1, None, extractor, device,
            theta1=selected_th1, theta2=0.0,
            crop_samples=19200, crop_offset_samples=4800
        )
        s1_test_cmd_trigs, s1_all_props = s1_engine.process_stream(test_cmd_audio, bypass_stage2=True)
        s1_test_cmd_m = match_events_and_evaluate(s1_test_cmd_trigs, test_cmd_gt, len(test_cmd_audio) / 16000.0)

        s1_test_nt_trigs, _ = s1_engine.process_stream(test_nt_audio, bypass_stage2=True)
        s1_test_nt_m = match_events_and_evaluate(s1_test_nt_trigs, test_nt_gt, len(test_nt_audio) / 16000.0)

        # 2. Cascade (Stage 1 + Stage 2) on Test Streams
        cascade_engine = CascadedStreamingEngine(
            stage1, stage2, extractor, device,
            theta1=selected_th1, theta2=selected_th2,
            crop_samples=19200, crop_offset_samples=4800
        )
        cas_test_cmd_trigs, cas_proposals = cascade_engine.process_stream(test_cmd_audio, bypass_stage2=False)
        cas_test_cmd_m = match_events_and_evaluate(cas_test_cmd_trigs, test_cmd_gt, len(test_cmd_audio) / 16000.0)

        cas_test_nt_trigs, _ = cascade_engine.process_stream(test_nt_audio, bypass_stage2=False)
        cas_test_nt_m = match_events_and_evaluate(cas_test_nt_trigs, test_nt_gt, len(test_nt_audio) / 16000.0)

        print(f"Stage 1 Alone on Test Command Stream: Recall={s1_test_cmd_m['recall']*100:.1f}%, Prec={s1_test_cmd_m['precision']*100:.1f}%, FPs={s1_test_cmd_m['false_positives']}, Lat={s1_test_cmd_m['latencies_ms']['mean']}ms")
        print(f"Stage 1 Alone on Test Non-Target Stream: Duration={s1_test_nt_m['stream_duration_sec']/60:.1f}m, FPs={s1_test_nt_m['false_positives']}, FA/hr={s1_test_nt_m['fa_per_hour']:.1f}")
        print(f"Cascade on Test Command Stream:       Recall={cas_test_cmd_m['recall']*100:.1f}%, Prec={cas_test_cmd_m['precision']*100:.1f}%, FPs={cas_test_cmd_m['false_positives']}, Lat={cas_test_cmd_m['latencies_ms']['mean']}ms")
        print(f"Cascade on Test Non-Target Stream:    Duration={cas_test_nt_m['stream_duration_sec']/60:.1f}m, FPs={cas_test_nt_m['false_positives']}, FA/hr={cas_test_nt_m['fa_per_hour']:.1f}")

        # Save per-event predictions CSV
        pred_records = []
        for tr in cas_test_cmd_trigs:
            pred_records.append({
                "stream": "test_commands",
                "trigger_time_sec": tr["trigger_time_sec"],
                "pred_label": tr["pred_label"],
                "confidence": tr["confidence"],
                "stage1_confidence": tr.get("stage1_confidence", 0.0),
                "window_start_sec": tr["window_start_sec"]
            })
        for tr in cas_test_nt_trigs:
            pred_records.append({
                "stream": "test_nontarget",
                "trigger_time_sec": tr["trigger_time_sec"],
                "pred_label": tr["pred_label"],
                "confidence": tr["confidence"],
                "stage1_confidence": tr.get("stage1_confidence", 0.0),
                "window_start_sec": tr["window_start_sec"]
            })
        pd.DataFrame(pred_records).to_csv(os.path.join(PREDICTIONS_DIR, f"test_cascade_predictions_seed{seed}.csv"), index=False)

        # 3. Controlled Noise Robustness Benchmark on Test Split Audio
        print("Running Controlled Noise Robustness on Test Split...")
        noise_results = {}
        for n_type in ["white", "pink", "hvac", "babble"]:
            noise_results[n_type] = {}
            for snr in [20, 15, 10, 5, 0]:
                correct_count = 0
                total_target_count = 0
                for _, r in df_test_manifest.iterrows():
                    if r["class_label"] not in TARGET_CLASSES:
                        continue
                    total_target_count += 1
                    full_p = os.path.join(BASE_DIR, r["kws_wav_path"])
                    audio, _ = sf.read(full_p, dtype="float32")
                    noisy_audio = generate_noisy_test_batch(audio, n_type, snr, seed=seed)

                    # Test Stage 1 + Stage 2 crop
                    s1_t = torch.from_numpy(noisy_audio[:16000]).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        s1_probs = F.softmax(stage1(extractor(s1_t)), dim=1).cpu().numpy()[0]

                    tgt_p = s1_probs[:4]
                    c_cand = int(np.argmax(tgt_p))

                    # If Stage 1 detects target candidate, test Stage 2
                    if tgt_p[c_cand] >= selected_th1:
                        crop_19200 = np.pad(noisy_audio, (0, max(0, 19200 - len(noisy_audio))))[:19200]
                        s2_t = torch.from_numpy(crop_19200).unsqueeze(0).float().to(device)
                        with torch.no_grad():
                            s2_probs = F.softmax(stage2(extractor(s2_t)), dim=1).cpu().numpy()[0]
                        if s2_probs[c_cand] >= selected_th2 and s2_probs[c_cand] > s2_probs[4]:
                            if TARGET_CLASSES[c_cand] == r["class_label"]:
                                correct_count += 1

                acc = correct_count / total_target_count if total_target_count > 0 else 0.0
                noise_results[n_type][f"{snr}db"] = {
                    "accuracy": acc,
                    "correct": correct_count,
                    "total": total_target_count
                }
                print(f"  Seed {seed} | {n_type:6s} at {snr:2d} dB | Acc: {acc*100:.1f}% ({correct_count}/{total_target_count})")

        # 4. CPU Latency Benchmark (Single Window)
        stage1_cpu = VietnameseKWSNet(num_classes=6)
        stage1_cpu.load_state_dict(s1_ckpt["model_state_dict"])
        stage1_cpu.eval()

        stage2_cpu = TemporalKWSVerifier(num_classes=5)
        stage2_cpu.load_state_dict(s2_ckpt["model_state_dict"])
        stage2_cpu.eval()

        dummy_feat_s1 = torch.randn(1, 1, 40, 101)
        dummy_feat_s2 = torch.randn(1, 1, 40, 121)

        t_s1 = []
        t_s2 = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                _ = stage1_cpu(dummy_feat_s1)
                t_s1.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                _ = stage2_cpu(dummy_feat_s2)
                t_s2.append((time.perf_counter() - t0) * 1000.0)

        cpu_s1_ms = float(np.mean(t_s1))
        cpu_s2_ms = float(np.mean(t_s2))
        cpu_cascade_ms = cpu_s1_ms + cpu_s2_ms
        print(f"CPU Latency Benchmark: Stage 1 = {cpu_s1_ms:.2f} ms | Stage 2 = {cpu_s2_ms:.2f} ms | Total = {cpu_cascade_ms:.2f} ms")

        campaign_summary[f"seed{seed}"] = {
            "selected_thresholds": selected_thresholds[f"seed{seed}"],
            "stage1_alone": {
                "command_stream": s1_test_cmd_m,
                "nontarget_stream": s1_test_nt_m
            },
            "cascade": {
                "command_stream": cas_test_cmd_m,
                "nontarget_stream": cas_test_nt_m
            },
            "noise_robustness": noise_results,
            "cpu_latency_ms": {
                "stage1": cpu_s1_ms,
                "stage2": cpu_s2_ms,
                "cascade_total": cpu_cascade_ms
            }
        }

    # Save calibration outputs
    with open(os.path.join(CALIBRATION_DIR, "selected_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(selected_thresholds, f, indent=2)

    with open(os.path.join(CALIBRATION_DIR, "validation_calibration_records.json"), "w", encoding="utf-8") as f:
        json.dump(validation_calibration_records, f, indent=2)

    # Save consolidated evaluation summary
    with open(os.path.join(PREDICTIONS_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(campaign_summary, f, indent=2)

    print(f"\nCascade Evaluation Campaign Completed! Summary saved to: {os.path.join(PREDICTIONS_DIR, 'evaluation_summary.json')}")


if __name__ == "__main__":
    run_cascade_campaign()
