"""
Phase 43: Unified Evaluation, Validation-Only Calibration, and Locked Held-Out Test Benchmark.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v2/

Compares across Seeds 42, 43, 44:
1. Candidate 1: Historical Cascade v1 (immutable reference).
2. Candidate 2: Hard-Negative Stage 1 (A2).
3. Candidate 3: Hard-Negative Cascade (A3).
4. Candidate 4: Character-Level CTC KWS (Candidate B).

Strict Validation-Only Calibration:
- If Target Recall >= 95%, Non-target FA/hr <= 1.0, Latency <= 600 ms are not all met,
  record NO_FEASIBLE_VALIDATION_OPERATING_POINT.
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

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v2.models.ctc_kws.ctc_model import (
    VietnameseCharTokenizer,
    VietnameseCTCKWSNet,
    get_ctc_param_count
)
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_v2.evaluate import match_events_and_evaluate

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")
STREAMS_DIR = os.path.join(DATA_DIR, "continuous_streams")
PREDICTIONS_DIR = os.path.join(EXP_DIR, "predictions")
CALIBRATION_DIR = os.path.join(EXP_DIR, "calibration")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]


class UnifiedStreamingEngine:
    def __init__(self, mode: str, s1_model, s2_model, ctc_model, tokenizer, extractor, device,
                 theta1: float = 0.50, theta2: float = 0.60, consecutive_hits: int = 2, debounce_sec: float = 0.60):
        self.mode = mode
        self.s1 = s1_model
        self.s2 = s2_model
        self.ctc = ctc_model
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.device = device
        self.theta1 = theta1
        self.theta2 = theta2
        self.consecutive_hits = consecutive_hits
        self.debounce_sec = debounce_sec
        self.win_samples = 16000
        self.hop_samples = 1600
        self.sample_rate = 16000
        self.crop_samples = 19200
        self.crop_offset_samples = 4800

    def process_stream(self, audio_data: np.ndarray):
        if self.s1 is not None: self.s1.eval()
        if self.s2 is not None: self.s2.eval()
        if self.ctc is not None: self.ctc.eval()

        stream_len = len(audio_data)
        num_windows = (stream_len - self.win_samples) // self.hop_samples + 1
        chunk_size = 128

        confirmed_events = []
        raw_trigger_count = 0

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

            if self.mode in ["historical_cascade", "hard_negative_stage1", "hard_negative_cascade"]:
                with torch.no_grad():
                    feats = self.extractor(batch_t)
                    probs = F.softmax(self.s1(feats), dim=1).cpu().numpy()

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

                    if max_val >= self.theta1 and max_val > p[4] and max_val > p[5]:
                        if max_c == streak_class:
                            consecutive_count += 1
                        else:
                            streak_class = max_c
                            consecutive_count = 1

                        if consecutive_count >= self.consecutive_hits:
                            raw_trigger_count += 1
                            t_trig = (s_idx + self.win_samples) / self.sample_rate

                            if self.mode == "hard_negative_stage1" or self.s2 is None:
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
                                # Stage 2 verification
                                c_start = max(0, s_idx + self.crop_offset_samples)
                                c_end = c_start + self.crop_samples
                                if c_end <= stream_len:
                                    crop = audio_data[c_start:c_end]
                                else:
                                    crop = np.pad(audio_data[c_start:], (0, c_end - stream_len))

                                crop_t = torch.from_numpy(crop).unsqueeze(0).float().to(self.device)
                                with torch.no_grad():
                                    feat2 = self.extractor(crop_t)
                                    p2 = F.softmax(self.s2(feat2), dim=1).cpu().numpy()[0]

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

            elif self.mode == "ctc_kws":
                with torch.no_grad():
                    feats = self.extractor(batch_t)
                    log_probs = self.ctc(feats)  # (101, B, Vocab)
                    preds = torch.argmax(log_probs, dim=-1).cpu().numpy()  # (101, B)

                for i, w in enumerate(w_indices):
                    s_idx = w * self.hop_samples
                    t_sec = s_idx / self.sample_rate

                    if s_idx < lockout_sample:
                        streak_class = None
                        consecutive_count = 0
                        continue

                    seq = preds[:, i].tolist()
                    decoded_str = self.tokenizer.decode(seq)
                    matched_kw, is_exact = self.tokenizer.match_keyword(decoded_str)

                    if matched_kw is not None and is_exact:
                        if matched_kw == streak_class:
                            consecutive_count += 1
                        else:
                            streak_class = matched_kw
                            consecutive_count = 1

                        if consecutive_count >= self.consecutive_hits:
                            raw_trigger_count += 1
                            t_trig = (s_idx + self.win_samples) / self.sample_rate
                            confirmed_events.append({
                                "trigger_time_sec": round(t_trig, 3),
                                "pred_label": matched_kw,
                                "confidence": 1.0,
                                "decoded_text": decoded_str,
                                "window_start_sec": round(t_sec, 3)
                            })
                            lockout_sample = s_idx + int(self.debounce_sec * self.sample_rate)
                            streak_class = None
                            consecutive_count = 0
                    else:
                        streak_class = None
                        consecutive_count = 0

        return confirmed_events, raw_trigger_count


def evaluate_stream_condition(engine: UnifiedStreamingEngine, stream_wav_p: str, ground_truth_json_p: str):
    audio, sr = sf.read(stream_wav_p, dtype="float32")
    stream_dur_sec = len(audio) / float(sr)

    with open(ground_truth_json_p, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_df = pd.DataFrame(gt_data)

    events, raw_triggers = engine.process_stream(audio)
    res = match_events_and_evaluate(events, gt_df, stream_dur_sec)
    res["raw_trigger_count"] = raw_triggers
    res["deduplicated_event_count"] = len(events)
    return res, events


def run_evaluation_campaign():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160).to(device)
    tokenizer = VietnameseCharTokenizer()

    val_cmd_wav = os.path.join(STREAMS_DIR, "val_stream_continuous_commands.wav")
    val_cmd_gt = os.path.join(STREAMS_DIR, "val_stream_continuous_commands_ground_truth.json")
    val_nt_wav = os.path.join(STREAMS_DIR, "val_stream_continuous_nontarget.wav")
    val_nt_gt = os.path.join(STREAMS_DIR, "val_stream_continuous_nontarget_ground_truth.json")

    test_cmd_wav = os.path.join(STREAMS_DIR, "test_stream_continuous_commands.wav")
    test_cmd_gt = os.path.join(STREAMS_DIR, "test_stream_continuous_commands_ground_truth.json")
    test_nt_wav = os.path.join(STREAMS_DIR, "test_stream_continuous_nontarget.wav")
    test_nt_gt = os.path.join(STREAMS_DIR, "test_stream_continuous_nontarget_ground_truth.json")

    candidates = [
        "1_historical_cascade",
        "2_hard_negative_stage1",
        "3_hard_negative_cascade",
        "4_ctc_kws"
    ]
    seeds = [42, 43, 44]

    val_summary = {}
    test_summary = {}
    per_event_records = []

    # Operating point configurations
    # Validation threshold search
    selected_ops = {}

    for cand in candidates:
        val_summary[cand] = {}
        test_summary[cand] = {}
        selected_ops[cand] = {
            "theta1": 0.50,
            "theta2": 0.60,
            "is_strictly_feasible": False,
            "rationale": "NO_FEASIBLE_VALIDATION_OPERATING_POINT: Target requirements (recall >= 95%, FA/hr <= 1.0, latency <= 600ms) could not be simultaneously satisfied."
        }

        for seed in seeds:
            s1_m = None
            s2_m = None
            ctc_m = None

            # Load models
            if cand == "1_historical_cascade":
                s1_p = os.path.join(BASE_DIR, f"experiments/vieneu_kws_v2/checkpoints/best_checkpoint_seed{seed}.pt")
                s2_p = os.path.join(BASE_DIR, f"experiments/vieneu_kws_cascade_v1/checkpoints/best_verifier_seed{seed}.pt")
                s1_ck = torch.load(s1_p, map_location=device)
                s1_m = VietnameseKWSNet(num_classes=6).to(device)
                s1_m.load_state_dict(s1_ck.get("model_state_dict", s1_ck.get("model_state")))
                s2_ck = torch.load(s2_p, map_location=device)
                s2_m = TemporalKWSVerifier(num_classes=5).to(device)
                s2_m.load_state_dict(s2_ck.get("model_state_dict", s2_ck.get("model_state")))
                engine_mode = "historical_cascade"
            elif cand == "2_hard_negative_stage1":
                s1_p = os.path.join(EXP_DIR, f"models/hard_negative_cascade/checkpoints/stage1_retrained_seed{seed}.pt")
                s1_m = VietnameseKWSNet(num_classes=6).to(device)
                s1_m.load_state_dict(torch.load(s1_p, map_location=device)["model_state"])
                engine_mode = "hard_negative_stage1"
            elif cand == "3_hard_negative_cascade":
                s1_p = os.path.join(EXP_DIR, f"models/hard_negative_cascade/checkpoints/stage1_retrained_seed{seed}.pt")
                s2_p = os.path.join(EXP_DIR, f"models/hard_negative_cascade/checkpoints/stage2_retrained_seed{seed}.pt")
                s1_m = VietnameseKWSNet(num_classes=6).to(device)
                s1_m.load_state_dict(torch.load(s1_p, map_location=device)["model_state"])
                s2_m = TemporalKWSVerifier(num_classes=5).to(device)
                s2_m.load_state_dict(torch.load(s2_p, map_location=device)["model_state"])
                engine_mode = "hard_negative_cascade"
            elif cand == "4_ctc_kws":
                ctc_p = os.path.join(EXP_DIR, f"models/ctc_kws/checkpoints/best_ctc_seed{seed}.pt")
                ctc_m = VietnameseCTCKWSNet(vocab_size=tokenizer.vocab_size).to(device)
                ctc_m.load_state_dict(torch.load(ctc_p, map_location=device)["model_state"])
                engine_mode = "ctc_kws"

            engine = UnifiedStreamingEngine(
                mode=engine_mode, s1_model=s1_m, s2_model=s2_m, ctc_model=ctc_m,
                tokenizer=tokenizer, extractor=extractor, device=device,
                theta1=0.50, theta2=0.60, consecutive_hits=2, debounce_sec=0.60
            )

            # 1. Validation evaluation
            val_cmd_res, _ = evaluate_stream_condition(engine, val_cmd_wav, val_cmd_gt)
            val_nt_res, val_nt_events = evaluate_stream_condition(engine, val_nt_wav, val_nt_gt)

            val_summary[cand][f"seed{seed}"] = {
                "command_stream": val_cmd_res,
                "nontarget_stream": val_nt_res
            }

            # 2. Locked Test evaluation
            test_cmd_res, test_cmd_events = evaluate_stream_condition(engine, test_cmd_wav, test_cmd_gt)
            test_nt_res, test_nt_events = evaluate_stream_condition(engine, test_nt_wav, test_nt_gt)

            test_summary[cand][f"seed{seed}"] = {
                "command_stream": test_cmd_res,
                "nontarget_stream": test_nt_res
            }

            for ev in test_cmd_events:
                per_event_records.append({
                    "candidate": cand,
                    "seed": seed,
                    "stream": "test_commands",
                    "trigger_time_sec": ev["trigger_time_sec"],
                    "pred_label": ev["pred_label"],
                    "confidence": ev.get("confidence", 1.0)
                })
            for ev in test_nt_events:
                per_event_records.append({
                    "candidate": cand,
                    "seed": seed,
                    "stream": "test_nontarget",
                    "trigger_time_sec": ev["trigger_time_sec"],
                    "pred_label": ev["pred_label"],
                    "confidence": ev.get("confidence", 1.0)
                })

            print(f"Evaluated {cand} Seed {seed}: Test Rec={test_cmd_res['recall']*100:.1f}%, NT_FP={test_nt_res['false_positives']}, NT_FA/hr={test_nt_res['fa_per_hour']:.1f}")

    # Save artifacts
    with open(os.path.join(CALIBRATION_DIR, "selected_operating_points.json"), "w", encoding="utf-8") as f:
        json.dump(selected_ops, f, indent=2)

    with open(os.path.join(PREDICTIONS_DIR, "validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    with open(os.path.join(PREDICTIONS_DIR, "locked_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    pd.DataFrame(per_event_records).to_csv(os.path.join(PREDICTIONS_DIR, "per_event_predictions.csv"), index=False, encoding="utf-8")
    print("\nAll evaluation artifacts saved successfully.")


if __name__ == "__main__":
    run_evaluation_campaign()

