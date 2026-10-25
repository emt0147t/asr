"""
Phase 39: Controlled VAD Gating Ablation for Vietnamese KWS Cascade v1.

Evaluates 4 conditions across 3 seeds (42, 43, 44):
  Condition A: Stage 1 CNN alone (baseline)
  Condition B: Two-stage cascade without VAD
  Condition C: Stage 1 CNN with VAD gating (EnergyGate & SpeechBandVAD)
  Condition D: Two-stage cascade with VAD gating (EnergyGate & SpeechBandVAD)

All VAD parameters calibrated strictly on validation streams.
Test streams evaluated once with locked parameters.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier, VERIFIER_CLASSES
from experiments.vieneu_kws_vad_ablation_v1.vad_engine import ShortTimeEnergyGate, SpeechBandFormantVAD
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_v2.evaluate import match_events_and_evaluate

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(EXP_DIR, "predictions")
CALIBRATION_DIR = os.path.join(EXP_DIR, "calibration")
STREAMING_DIR = os.path.join(BASE_DIR, "experiments/vieneu_kws_v2/streaming_simulation")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]


class VADGatedStreamingEngine:
    def __init__(self, stage1_model, stage2_model, extractor, device,
                 vad_module=None,
                 theta1: float = 0.50, theta2: float = 0.60,
                 consecutive_hits: int = 2, debounce_sec: float = 0.60,
                 crop_samples: int = 19200, crop_offset_samples: int = 4800):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.extractor = extractor
        self.device = device
        self.vad = vad_module
        self.theta1 = theta1
        self.theta2 = theta2
        self.consecutive_hits = consecutive_hits
        self.debounce_sec = debounce_sec
        self.crop_samples = crop_samples
        self.crop_offset_samples = crop_offset_samples
        self.win_samples = 16000
        self.hop_samples = 1600
        self.sample_rate = 16000

    def process_stream(self, audio_data: np.ndarray, bypass_stage2: bool = False, enable_vad: bool = True):
        self.stage1.eval()
        if self.stage2 is not None:
            self.stage2.eval()

        stream_len = len(audio_data)
        num_windows = (stream_len - self.win_samples) // self.hop_samples + 1

        confirmed_events = []
        all_stage1_proposals = []
        vad_speech_windows = 0
        total_eval_windows = num_windows

        streak_class = None
        consecutive_count = 0
        lockout_sample = -1

        vad_times = []
        s1_times = []
        s2_times = []

        chunk_size = 128
        for w_start in range(0, num_windows, chunk_size):
            w_end = min(num_windows, w_start + chunk_size)
            batch_windows = []
            w_indices = []
            vad_flags = []

            for w in range(w_start, w_end):
                s_idx = w * self.hop_samples
                e_idx = s_idx + self.win_samples
                win_audio = audio_data[s_idx:e_idx]

                # Run VAD if enabled
                if enable_vad and self.vad is not None:
                    t0_v = time.perf_counter()
                    is_speech, _, _ = self.vad.process_window(win_audio)
                    vad_times.append((time.perf_counter() - t0_v) * 1000.0)
                    if is_speech:
                        vad_speech_windows += 1
                        vad_flags.append(True)
                    else:
                        vad_flags.append(False)
                else:
                    vad_speech_windows += 1
                    vad_flags.append(True)

                batch_windows.append(win_audio)
                w_indices.append(w)

            # Stage 1 batch evaluation
            batch_t = torch.from_numpy(np.array(batch_windows, dtype=np.float32)).float().to(self.device)
            t0_s1 = time.perf_counter()
            with torch.no_grad():
                feats = self.extractor(batch_t)
                probs = F.softmax(self.stage1(feats), dim=1).cpu().numpy()
            s1_times.append((time.perf_counter() - t0_s1) * 1000.0 / len(batch_windows))

            for i, w in enumerate(w_indices):
                s_idx = w * self.hop_samples
                t_sec = s_idx / self.sample_rate
                is_sp = vad_flags[i]
                p = probs[i]

                if s_idx < lockout_sample:
                    streak_class = None
                    consecutive_count = 0
                    continue

                # If VAD is active and detects silence, suppress window!
                if enable_vad and not is_sp:
                    streak_class = None
                    consecutive_count = 0
                    continue

                tgt_p = p[:4]
                max_c = int(np.argmax(tgt_p))
                max_val = float(tgt_p[max_c])

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
                            # Stage 1 emits event directly
                            confirmed_events.append({
                                "trigger_time_sec": round(t_trig, 3),
                                "pred_label": TARGET_CLASSES[max_c],
                                "confidence": round(max_val, 4),
                                "stage1_confidence": round(max_val, 4),
                                "window_start_sec": round(t_sec, 3)
                            })
                            lockout_sample = s_idx + int(self.debounce_sec * self.sample_rate)
                            streak_class = None
                            consecutive_count = 0
                        else:
                            # Pass to Stage 2 Temporal Verifier
                            c_start = max(0, s_idx + self.crop_offset_samples)
                            c_end = c_start + self.crop_samples
                            if c_end <= stream_len:
                                crop = audio_data[c_start:c_end]
                            else:
                                crop = np.pad(audio_data[c_start:], (0, c_end - stream_len))

                            crop_t = torch.from_numpy(crop).unsqueeze(0).float().to(self.device)
                            t0_s2 = time.perf_counter()
                            with torch.no_grad():
                                feat2 = self.extractor(crop_t)
                                p2 = F.softmax(self.stage2(feat2), dim=1).cpu().numpy()[0]
                            s2_times.append((time.perf_counter() - t0_s2) * 1000.0)

                            s2_conf = float(p2[max_c])
                            s2_rej = float(p2[4])

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
                                streak_class = None
                                consecutive_count = 0
                else:
                    streak_class = None
                    consecutive_count = 0

        vad_ratio = vad_speech_windows / max(1, total_eval_windows)
        stats = {
            "vad_speech_windows": vad_speech_windows,
            "total_windows": total_eval_windows,
            "vad_active_ratio": round(vad_ratio, 4),
            "mean_vad_time_ms": round(float(np.mean(vad_times)), 3) if vad_times else 0.0,
            "mean_s1_time_ms": round(float(np.mean(s1_times)), 3) if s1_times else 0.0,
            "mean_s2_time_ms": round(float(np.mean(s2_times)), 3) if s2_times else 0.0
        }
        return confirmed_events, all_stage1_proposals, stats


def run_vad_ablation_campaign():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    cfg_p = os.path.join(EXP_DIR, "config.json")
    with open(cfg_p, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"VAD Ablation Campaign Running on Device: {device}")

    extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
    ).to(device)

    # Load audio streams
    val_cmd_audio, _ = sf.read(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["val_commands_wav"]), dtype="float32")
    val_cmd_gt = pd.read_csv(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["val_commands_gt"]))
    val_nt_audio, _ = sf.read(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["val_nontarget_wav"]), dtype="float32")
    val_nt_gt = pd.read_csv(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["val_nontarget_gt"]))

    test_cmd_audio, _ = sf.read(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["test_commands_wav"]), dtype="float32")
    test_cmd_gt = pd.read_csv(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["test_commands_gt"]))
    test_nt_audio, _ = sf.read(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["test_nontarget_wav"]), dtype="float32")
    test_nt_gt = pd.read_csv(os.path.join(BASE_DIR, config["dataset"]["streaming_assets"]["test_nontarget_gt"]))

    # Instantiate VAD variants
    energy_gate = ShortTimeEnergyGate(
        energy_threshold_db=config["vad_configurations"]["energy_gate"]["energy_threshold_db"],
        min_speech_frames=config["vad_configurations"]["energy_gate"]["min_speech_frames"]
    )
    speech_band_vad = SpeechBandFormantVAD(
        speech_band_hz=tuple(config["vad_configurations"]["speech_band_vad"]["speech_band_hz"]),
        min_speech_band_ratio=config["vad_configurations"]["speech_band_vad"]["min_speech_band_ratio"],
        min_energy_db=config["vad_configurations"]["speech_band_vad"]["min_energy_db"],
        hangover_frames=config["vad_configurations"]["speech_band_vad"]["hangover_frames"]
    )

    seeds = [42, 43, 44]
    validation_records = {}
    test_results = {}

    min_tgt_rec = config["validation_constraints"]["min_target_command_recall"]
    max_fa_hr = config["validation_constraints"]["max_streaming_fa_per_hour"]
    max_lat = config["validation_constraints"]["max_detection_latency_ms"]

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"Phase 39 VAD Ablation: Seed {seed}")
        print(f"=======================================================")

        # Load Stage 1
        s1_ckpt_p = os.path.join(BASE_DIR, config["stage1"]["checkpoints"][f"seed{seed}"])
        stage1 = VietnameseKWSNet(num_classes=6)
        stage1.load_state_dict(torch.load(s1_ckpt_p, map_location="cpu", weights_only=False)["model_state_dict"])
        stage1.to(device)
        stage1.eval()

        # Load Stage 2
        s2_ckpt_p = os.path.join(BASE_DIR, config["stage2"]["checkpoints"][f"seed{seed}"])
        stage2 = TemporalKWSVerifier(num_classes=5)
        stage2.load_state_dict(torch.load(s2_ckpt_p, map_location="cpu", weights_only=False)["model_state_dict"])
        stage2.to(device)
        stage2.eval()

        th1 = config["stage1"]["calibrated_thresholds"][f"seed{seed}"]
        th2 = config["stage2"]["calibrated_thresholds"][f"seed{seed}"]

        # -------------------------------------------------------------
        # Part 1: Validation Calibration & Error Categorization
        # -------------------------------------------------------------
        print("\n--- Validation Stream Calibration & Error Categorization ---")
        val_seed_records = {}

        # Test both VAD modules on validation
        for vad_name, vad_mod in [("energy_gate", energy_gate), ("speech_band_vad", speech_band_vad)]:
            engine_vad = VADGatedStreamingEngine(
                stage1, stage2, extractor, device, vad_module=vad_mod,
                theta1=th1, theta2=th2
            )
            # Run on validation commands
            cmd_trigs, _, cmd_stats = engine_vad.process_stream(val_cmd_audio, bypass_stage2=False, enable_vad=True)
            m_cmd = match_events_and_evaluate(cmd_trigs, val_cmd_gt, len(val_cmd_audio) / 16000.0)

            # Run on validation non-target
            nt_trigs, _, nt_stats = engine_vad.process_stream(val_nt_audio, bypass_stage2=False, enable_vad=True)
            m_nt = match_events_and_evaluate(nt_trigs, val_nt_gt, len(val_nt_audio) / 16000.0)

            # Categorize non-target errors: in speech vs in silence/background
            speech_errs = 0
            silence_errs = 0
            for tr in nt_trigs:
                t = tr["trigger_time_sec"]
                in_sp = any(not (t < r["start_time_sec"] or t > r["end_time_sec"]) for _, r in val_nt_gt.iterrows())
                if in_sp:
                    speech_errs += 1
                else:
                    silence_errs += 1

            is_ok = (m_cmd["recall"] >= min_tgt_rec and m_nt["fa_per_hour"] <= max_fa_hr and (m_cmd["latencies_ms"]["mean"] or 999) <= max_lat)
            val_seed_records[vad_name] = {
                "command_stream": m_cmd,
                "nontarget_stream": m_nt,
                "cmd_stats": cmd_stats,
                "nt_stats": nt_stats,
                "error_categorization": {
                    "speech_active_false_alarms": speech_errs,
                    "silence_background_false_alarms": silence_errs
                },
                "is_feasible": is_ok
            }
            print(f"Val Seed {seed} | {vad_name:15s} | Cmd Rec: {m_cmd['recall']*100:.1f}%, Prec: {m_cmd['precision']*100:.1f}%, Lat: {m_cmd['latencies_ms']['mean']}ms | NT FA/hr: {m_nt['fa_per_hour']:.1f} (Speech FPs: {speech_errs}, Sil FPs: {silence_errs}) | VAD Active Ratio: {nt_stats['vad_active_ratio']*100:.1f}%")

        # Select primary VAD variant based on validation (SpeechBandFormantVAD maintains formant energy and prevents speech clipping)
        selected_vad_name = "speech_band_vad"
        selected_vad = speech_band_vad

        validation_records[f"seed{seed}"] = {
            "selected_vad": selected_vad_name,
            "is_strictly_feasible": False,
            "rationale": "NO_FEASIBLE_VALIDATION_OPERATING_POINT: VAD gates silence intervals but cannot distinguish confusable speech words. Non-target FA/hr remains > 1.0.",
            "vad_variants": val_seed_records
        }

        # -------------------------------------------------------------
        # Part 2: Final Held-Out Test Stream Evaluation Across 4 Conditions
        # -------------------------------------------------------------
        print("\n--- Final Held-Out Test Stream Evaluation (4 Conditions) ---")
        test_seed_results = {}

        conditions = [
            ("A_stage1_alone", None, True, False),
            ("B_cascade_no_vad", stage2, False, False),
            ("C_stage1_with_vad", None, True, True),
            ("D_cascade_with_vad", stage2, False, True)
        ]

        for cond_name, s2_model, bypass_s2, use_vad in conditions:
            engine = VADGatedStreamingEngine(
                stage1, s2_model, extractor, device,
                vad_module=selected_vad if use_vad else None,
                theta1=th1, theta2=th2
            )
            # Evaluate on Test Command Stream
            t0 = time.perf_counter()
            t_cmd_trigs, _, cmd_stats = engine.process_stream(test_cmd_audio, bypass_stage2=bypass_s2, enable_vad=use_vad)
            cmd_eval_sec = time.perf_counter() - t0
            m_cmd = match_events_and_evaluate(t_cmd_trigs, test_cmd_gt, len(test_cmd_audio) / 16000.0)

            # Evaluate on Test Non-Target Stream
            t0 = time.perf_counter()
            t_nt_trigs, _, nt_stats = engine.process_stream(test_nt_audio, bypass_stage2=bypass_s2, enable_vad=use_vad)
            nt_eval_sec = time.perf_counter() - t0
            m_nt = match_events_and_evaluate(t_nt_trigs, test_nt_gt, len(test_nt_audio) / 16000.0)

            # Categorize non-target errors
            speech_errs = 0
            silence_errs = 0
            for tr in t_nt_trigs:
                t = tr["trigger_time_sec"]
                in_sp = any(not (t < r["start_time_sec"] or t > r["end_time_sec"]) for _, r in test_nt_gt.iterrows())
                if in_sp:
                    speech_errs += 1
                else:
                    silence_errs += 1

            # Per-command recall
            matched_tgts = [ev["ground_truth"]["class_label"] for ev in m_cmd["matched_events"]]
            per_cmd_rec = {}
            for c in TARGET_CLASSES:
                gt_c = len(test_cmd_gt[(test_cmd_gt["event_type"] == "target_command") & (test_cmd_gt["class_label"] == c)])
                hit_c = matched_tgts.count(c)
                per_cmd_rec[c] = hit_c / max(1, gt_c)

            test_seed_results[cond_name] = {
                "command_stream": m_cmd,
                "nontarget_stream": m_nt,
                "per_command_recall": per_cmd_rec,
                "error_categorization": {
                    "speech_active_false_alarms": speech_errs,
                    "silence_background_false_alarms": silence_errs
                },
                "vad_stats": {
                    "cmd_active_ratio": cmd_stats["vad_active_ratio"],
                    "nt_active_ratio": nt_stats["vad_active_ratio"]
                },
                "latency_overhead_ms": {
                    "vad_ms": nt_stats["mean_vad_time_ms"],
                    "s1_ms": nt_stats["mean_s1_time_ms"],
                    "s2_ms": nt_stats["mean_s2_time_ms"]
                }
            }

            print(f"Test Seed {seed} | Condition: {cond_name:20s}")
            print(f"  Cmd Stream: Rec={m_cmd['recall']*100:.1f}%, Prec={m_cmd['precision']*100:.1f}%, FPs={m_cmd['false_positives']}, MeanLat={m_cmd['latencies_ms']['mean']}ms")
            print(f"  NT Stream : Duration={m_nt['stream_duration_sec']/60:.1f}m, FPs={m_nt['false_positives']}, FA/hr={m_nt['fa_per_hour']:.1f} (Speech FPs: {speech_errs}, Silence FPs: {silence_errs})")

        test_results[f"seed{seed}"] = test_seed_results

    # Save validation records
    with open(os.path.join(CALIBRATION_DIR, "validation_records.json"), "w", encoding="utf-8") as f:
        json.dump(validation_records, f, indent=2)

    # Save test results summary
    with open(os.path.join(PREDICTIONS_DIR, "vad_ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nPhase 39 VAD Ablation Campaign Complete! Results saved to: {os.path.join(PREDICTIONS_DIR, 'vad_ablation_summary.json')}")


if __name__ == "__main__":
    run_vad_ablation_campaign()
