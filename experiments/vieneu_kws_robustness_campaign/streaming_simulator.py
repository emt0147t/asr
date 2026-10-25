"""
Phase 32: Continuous-Stream Simulation and False-Trigger/Latency Analysis.
Simulates a real-time sliding-window KWS engine operating on continuous audio streams.

Components:
  1. Stream Builder:
     - Constructs reproducible continuous audio streams:
       * Command Stream: Target commands, UNKNOWN distractors, and SILENCE intervals interspersed.
       * Non-Target Stream: Pure UNKNOWN phrases, multi-talker babble, and silence intervals (for FA/hr measurement).
  2. Sliding-Window Detector:
     - Window size: 1.0 s (16,000 samples)
     - Stride / Hop: 100 ms (1,600 samples)
     - Confidence thresholding: Phase 31 calibrated threshold theta (or uncalibrated baseline)
     - Consecutive hit integration (N=2 consecutive frames >= theta)
     - Debounce / Cooldown lockout (0.60 s) to prevent multi-triggering
  3. Ground-Truth Event Matcher & Metric Calculator:
     - True Positive (TP), False Positive (FP), False Negative (FN)
     - False Alarms per Hour (FA/hr)
     - Trigger Latency (ms from command onset to trigger event)
     - Command Recall, Precision, and F1
"""

import os
import sys
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
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_robustness_campaign.confidence_policy import (
    apply_confidence_policy,
    TARGET_COMMAND_INDICES,
    UNKNOWN_INDEX,
    SILENCE_INDEX
)

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(CAMPAIGN_DIR, "predictions")
STREAMING_DIR = os.path.join(CAMPAIGN_DIR, "streaming_simulation")

LABEL_TO_ID = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]

def build_continuous_streams(test_manifest_path: str, seed: int = 999001):
    """
    Constructs two continuous audio streams with exact ground truth event logs:
      1. Command Stream: ~10 minutes, alternating target commands, distractors, and silence.
      2. Non-Target Stream: ~20 minutes, pure non-target phrases, babble, and silence (0 target commands).
    """
    os.makedirs(STREAMING_DIR, exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    df = pd.read_csv(test_manifest_path)
    
    target_rows = df[df["class_label"].isin(["TRAI", "PHAI", "LEN", "XUONG"])].to_dict("records")
    unknown_rows = df[df["class_label"] == "UNKNOWN"].to_dict("records")
    silence_rows = df[df["class_label"] == "SILENCE"].to_dict("records")
    
    # -------------------------------------------------------------
    # Stream 1: Command Stream (~10 minutes, ~120 target events + distractors)
    # -------------------------------------------------------------
    print("Building Continuous Command Stream (~10 minutes)...", flush=True)
    cmd_audio_blocks = []
    cmd_events = []
    curr_time_sec = 1.0  # Start with 1.0s initial silence
    
    # 1.0s lead-in silence
    lead_silence = np.zeros(16000, dtype=np.float32)
    cmd_audio_blocks.append(lead_silence)
    
    # Shuffle targets
    shuffled_targets = list(target_rows)
    rng.shuffle(shuffled_targets)
    
    # Insert target commands with random gap (1.0s to 3.5s) and occasional UNKNOWN distractors
    for i, tgt in enumerate(shuffled_targets):
        # 30% chance to insert an UNKNOWN distractor before this target
        if rng.random() < 0.35 and unknown_rows:
            unk = rng.choice(unknown_rows)
            full_p = os.path.join(BASE_DIR, unk["kws_wav_path"])
            unk_data, _ = sf.read(full_p, dtype="float32")
            unk_dur = len(unk_data) / 16000.0
            
            cmd_events.append({
                "stream": "command_stream",
                "event_type": "distractor",
                "class_label": "UNKNOWN",
                "start_time_sec": round(curr_time_sec, 3),
                "end_time_sec": round(curr_time_sec + unk_dur, 3),
                "onset_sec": round(curr_time_sec + float(unk.get("kws_onset_sec", 0.0)), 3),
                "active_dur_sec": round(float(unk.get("kws_active_speech_duration_sec", 0.8)), 3),
                "sample_id": unk["sample_id"]
            })
            cmd_audio_blocks.append(unk_data)
            curr_time_sec += unk_dur
            
            # Short gap after distractor (0.5s - 1.5s)
            gap_len = int(np_rng.uniform(0.5, 1.5) * 16000)
            cmd_audio_blocks.append(np.zeros(gap_len, dtype=np.float32))
            curr_time_sec += gap_len / 16000.0
            
        # Insert target command
        full_p = os.path.join(BASE_DIR, tgt["kws_wav_path"])
        tgt_data, _ = sf.read(full_p, dtype="float32")
        tgt_dur = len(tgt_data) / 16000.0
        
        onset_in_clip = float(tgt.get("kws_onset_sec", 0.15))
        active_dur = float(tgt.get("kws_active_speech_duration_sec", 0.65))
        
        cmd_events.append({
            "stream": "command_stream",
            "event_type": "target_command",
            "class_label": tgt["class_label"],
            "start_time_sec": round(curr_time_sec, 3),
            "end_time_sec": round(curr_time_sec + tgt_dur, 3),
            "onset_sec": round(curr_time_sec + onset_in_clip, 3),
            "active_dur_sec": round(active_dur, 3),
            "sample_id": tgt["sample_id"]
        })
        cmd_audio_blocks.append(tgt_data)
        curr_time_sec += tgt_dur
        
        # Inter-command silence gap (1.5s - 3.5s) with subtle ambient floor
        gap_sec = np_rng.uniform(1.5, 3.5)
        gap_len = int(gap_sec * 16000)
        ambient_noise = np_rng.normal(0, 0.001, gap_len).astype(np.float32)
        cmd_audio_blocks.append(ambient_noise)
        curr_time_sec += gap_sec
        
    cmd_audio = np.concatenate(cmd_audio_blocks)
    cmd_audio_path = os.path.join(STREAMING_DIR, "continuous_command_stream.wav")
    sf.write(cmd_audio_path, cmd_audio, 16000, subtype="PCM_16")
    cmd_events_df = pd.DataFrame(cmd_events)
    cmd_events_path = os.path.join(STREAMING_DIR, "command_stream_ground_truth.csv")
    cmd_events_df.to_csv(cmd_events_path, index=False)
    print(f"Command stream created: {len(cmd_audio)/16000:.1f}s ({len(cmd_audio)/16000/60:.2f} min), {len(cmd_events)} events saved.", flush=True)

    # -------------------------------------------------------------
    # Stream 2: Non-Target Background Stream (~20 minutes, 0 targets)
    # -------------------------------------------------------------
    print("Building Continuous Non-Target Stream (~20 minutes)...", flush=True)
    nontgt_audio_blocks = []
    nontgt_events = []
    curr_time_sec = 0.5
    nontgt_audio_blocks.append(np.zeros(8000, dtype=np.float32))
    
    # Loop non-target elements until duration >= 1200 seconds (20 min)
    target_duration_sec = 1200.0
    while curr_time_sec < target_duration_sec:
        action = rng.choice(["unknown_clip", "babble_burst", "silence_pause"])
        if action == "unknown_clip" and unknown_rows:
            unk = rng.choice(unknown_rows)
            full_p = os.path.join(BASE_DIR, unk["kws_wav_path"])
            unk_data, _ = sf.read(full_p, dtype="float32")
            unk_dur = len(unk_data) / 16000.0
            
            nontgt_events.append({
                "stream": "nontarget_stream",
                "event_type": "distractor",
                "class_label": "UNKNOWN",
                "start_time_sec": round(curr_time_sec, 3),
                "end_time_sec": round(curr_time_sec + unk_dur, 3),
                "onset_sec": round(curr_time_sec + float(unk.get("kws_onset_sec", 0.0)), 3),
                "active_dur_sec": round(float(unk.get("kws_active_speech_duration_sec", 0.8)), 3),
                "sample_id": unk["sample_id"]
            })
            nontgt_audio_blocks.append(unk_data)
            curr_time_sec += unk_dur
        elif action == "silence_pause":
            pause_sec = np_rng.uniform(1.0, 4.0)
            pause_samples = int(pause_sec * 16000)
            noise_floor = np_rng.normal(0, 0.0008, pause_samples).astype(np.float32)
            nontgt_audio_blocks.append(noise_floor)
            curr_time_sec += pause_sec
        else: # babble_burst
            burst_sec = np_rng.uniform(1.5, 3.5)
            burst_samples = int(burst_sec * 16000)
            babble_p = os.path.join(CAMPAIGN_DIR, "babble_sources", "synthetic_vietnamese_babble_30s.wav")
            if os.path.exists(babble_p):
                babble_audio, _ = sf.read(babble_p, dtype="float32")
                start_idx = rng.randint(0, max(0, len(babble_audio) - burst_samples - 1))
                burst = babble_audio[start_idx:start_idx + burst_samples] * 0.4
            else:
                burst = np_rng.normal(0, 0.01, burst_samples).astype(np.float32)
            nontgt_audio_blocks.append(burst)
            curr_time_sec += burst_sec
            
    nontgt_audio = np.concatenate(nontgt_audio_blocks)
    nontgt_audio_path = os.path.join(STREAMING_DIR, "continuous_nontarget_stream.wav")
    sf.write(nontgt_audio_path, nontgt_audio, 16000, subtype="PCM_16")
    nontgt_events_df = pd.DataFrame(nontgt_events)
    nontgt_events_path = os.path.join(STREAMING_DIR, "nontarget_stream_ground_truth.csv")
    nontgt_events_df.to_csv(nontgt_events_path, index=False)
    print(f"Non-target stream created: {len(nontgt_audio)/16000:.1f}s ({len(nontgt_audio)/16000/60:.2f} min), {len(nontgt_events)} distractor events saved.", flush=True)

    return cmd_audio_path, cmd_events_path, nontgt_audio_path, nontgt_events_path

class SlidingWindowStreamingEngine:
    def __init__(self, model, extractor, device, threshold: float = 0.85,
                 window_samples: int = 16000, hop_samples: int = 1600,
                 consecutive_hits: int = 2, debounce_sec: float = 0.60):
        self.model = model
        self.extractor = extractor
        self.device = device
        self.threshold = threshold
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.consecutive_hits = consecutive_hits
        self.debounce_sec = debounce_sec
        self.sample_rate = 16000
        
    def process_stream(self, audio_data: np.ndarray):
        """
        Runs sliding-window inference over audio array.
        Returns list of triggered events:
          [{'trigger_time_sec': float, 'pred_label': str, 'confidence': float, 'window_start_sec': float}]
        """
        self.model.eval()
        total_samples = len(audio_data)
        triggered_events = []
        
        # State tracking
        current_streak_class = None
        consecutive_count = 0
        lockout_until_sec = -1.0
        
        # Sliding window buffer
        # For efficiency, batch extract windows
        num_windows = (total_samples - self.window_samples) // self.hop_samples + 1
        
        # Process in chunks of windows
        chunk_size = 128
        
        for w_start_idx in range(0, num_windows, chunk_size):
            w_end_idx = min(num_windows, w_start_idx + chunk_size)
            batch_windows = []
            window_times = []
            
            for w in range(w_start_idx, w_end_idx):
                start_sample = w * self.hop_samples
                end_sample = start_sample + self.window_samples
                batch_windows.append(audio_data[start_sample:end_sample])
                window_times.append(start_sample / self.sample_rate)
                
            batch_windows_t = torch.from_numpy(np.array(batch_windows, dtype=np.float32)).float()
            with torch.no_grad():
                feats = self.extractor(batch_windows_t).to(self.device)
                logits = self.model(feats)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                
            # Apply confidence policy
            cal_preds, confs, rejs = apply_confidence_policy(probs, self.threshold)
            
            # Step through individual window predictions sequentially
            for i in range(len(batch_windows)):
                t_sec = window_times[i]
                pred_c = cal_preds[i]
                conf = confs[i]
                
                # Check if in debounce lockout
                if t_sec < lockout_until_sec:
                    consecutive_count = 0
                    current_streak_class = None
                    continue
                    
                # If target command detected
                if pred_c in TARGET_COMMAND_INDICES:
                    if pred_c == current_streak_class:
                        consecutive_count += 1
                    else:
                        current_streak_class = pred_c
                        consecutive_count = 1
                        
                    if consecutive_count >= self.consecutive_hits:
                        # FIRE TRIGGER
                        trigger_time = t_sec + (self.window_samples / self.sample_rate)
                        triggered_events.append({
                            "trigger_time_sec": round(trigger_time, 3),
                            "pred_label": CLASS_NAMES[pred_c],
                            "confidence": round(float(conf), 4),
                            "window_start_sec": round(t_sec, 3)
                        })
                        # Enter lockout
                        lockout_until_sec = trigger_time + self.debounce_sec
                        current_streak_class = None
                        consecutive_count = 0
                else:
                    consecutive_count = 0
                    current_streak_class = None
                    
        return triggered_events

def match_events_and_evaluate(triggered_events, ground_truth_df, stream_duration_sec,
                              match_window_left=0.20, match_window_right=0.60):
    """
    Matches triggered detections against ground truth target commands.
    """
    gt_targets = ground_truth_df[ground_truth_df["event_type"] == "target_command"].copy()
    num_gt_targets = len(gt_targets)
    
    gt_matched = [False] * num_gt_targets
    latencies = []
    matched_events = []
    unmatched_triggers = []
    
    for trig in triggered_events:
        t_trig = trig["trigger_time_sec"]
        lbl = trig["pred_label"]
        
        match_found = False
        for idx, (_, gt_row) in enumerate(gt_targets.iterrows()):
            if gt_matched[idx]:
                continue
                
            t_onset = gt_row["onset_sec"]
            t_end = gt_row["end_time_sec"]
            
            # Valid detection window: from (onset - left_margin) to (end + right_margin)
            if (t_onset - match_window_left) <= t_trig <= (t_end + match_window_right):
                if lbl == gt_row["class_label"]:
                    match_found = True
                    gt_matched[idx] = True
                    latency_ms = (t_trig - t_onset) * 1000.0
                    latencies.append(latency_ms)
                    matched_events.append({
                        "trigger": trig,
                        "ground_truth": gt_row.to_dict(),
                        "latency_ms": round(latency_ms, 1)
                    })
                    break
                    
        if not match_found:
            unmatched_triggers.append(trig)
            
    tp = sum(gt_matched)
    fn = num_gt_targets - tp
    fp = len(unmatched_triggers)
    
    stream_hours = stream_duration_sec / 3600.0
    fa_per_hour = fp / stream_hours if stream_hours > 0 else 0.0
    
    recall = tp / num_gt_targets if num_gt_targets > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "num_gt_targets": num_gt_targets,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "stream_duration_sec": round(stream_duration_sec, 2),
        "stream_duration_hours": round(stream_hours, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "fa_per_hour": round(fa_per_hour, 2),
        "latencies_ms": {
            "mean": round(float(np.mean(latencies)), 1) if latencies else None,
            "median": round(float(np.median(latencies)), 1) if latencies else None,
            "min": round(float(np.min(latencies)), 1) if latencies else None,
            "max": round(float(np.max(latencies)), 1) if latencies else None,
            "p90": round(float(np.percentile(latencies, 90)), 1) if latencies else None
        },
        "matched_events": matched_events,
        "false_alarms": unmatched_triggers
    }

def run_streaming_campaign():
    os.makedirs(STREAMING_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    cfg_path = os.path.join(CAMPAIGN_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    test_manifest_path = os.path.join(BASE_DIR, config["dataset"]["test_manifest"])
    
    # 1. Build continuous streams (reproducible)
    cmd_audio_p, cmd_gt_p, nontgt_audio_p, nontgt_gt_p = build_continuous_streams(test_manifest_path)
    
    cmd_audio, sr = sf.read(cmd_audio_p, dtype="float32")
    nontgt_audio, sr = sf.read(nontgt_audio_p, dtype="float32")
    cmd_gt_df = pd.read_csv(cmd_gt_p)
    nontgt_gt_df = pd.read_csv(nontgt_gt_p)
    
    # Load frozen calibrated thresholds
    thresh_file = os.path.join(CAMPAIGN_DIR, "calibration", "selected_thresholds.json")
    if os.path.exists(thresh_file):
        with open(thresh_file, "r", encoding="utf-8") as f:
            frozen_thetas = json.load(f)
    else:
        frozen_thetas = {"seed42": 0.60, "seed43": 0.75, "seed44": 0.85}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
    )
    
    streaming_cfg = config["phase32_streaming"]
    win_samples = streaming_cfg["window_samples"]
    hop_samples = streaming_cfg["hop_samples"]
    consec_hits = streaming_cfg["consecutive_positive_windows"]
    debounce_sec = streaming_cfg["debounce_duration_sec"]
    
    seeds = [42, 43, 44]
    campaign_streaming_results = {}
    
    for seed in seeds:
        print(f"\n=================================================================")
        print(f"Running Streaming Simulation for Seed {seed}")
        print(f"=================================================================")
        
        ckpt_path = os.path.join(BASE_DIR, config["checkpoints"][f"seed{seed}"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = VietnameseKWSNet(num_classes=6)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        
        theta_cal = frozen_thetas[f"seed{seed}"]
        
        # Test 2 modes: Uncalibrated (theta=0.0) vs Calibrated (theta=theta_cal)
        seed_eval = {}
        for mode_name, th in [("uncalibrated", 0.0), ("calibrated", theta_cal)]:
            print(f"\n--- Mode: {mode_name.upper()} (theta = {th:.2f}) ---")
            engine = SlidingWindowStreamingEngine(
                model=model, extractor=extractor, device=device, threshold=th,
                window_samples=win_samples, hop_samples=hop_samples,
                consecutive_hits=consec_hits, debounce_sec=debounce_sec
            )
            
            # Evaluate Command Stream
            print("  Processing Command Stream...", flush=True)
            cmd_triggers = engine.process_stream(cmd_audio)
            cmd_metrics = match_events_and_evaluate(
                cmd_triggers, cmd_gt_df, len(cmd_audio)/16000.0,
                match_window_left=streaming_cfg["match_window_left_sec"],
                match_window_right=streaming_cfg["match_window_right_sec"]
            )
            
            # Evaluate Non-Target Stream
            print("  Processing Non-Target Background Stream...", flush=True)
            nontgt_triggers = engine.process_stream(nontgt_audio)
            nontgt_metrics = match_events_and_evaluate(
                nontgt_triggers, nontgt_gt_df, len(nontgt_audio)/16000.0,
                match_window_left=streaming_cfg["match_window_left_sec"],
                match_window_right=streaming_cfg["match_window_right_sec"]
            )
            
            print(f"  Command Stream: Recall {cmd_metrics['recall']*100:.1f}%, Prec {cmd_metrics['precision']*100:.1f}%, Mean Latency: {cmd_metrics['latencies_ms']['mean']} ms, False Alarms: {cmd_metrics['false_positives']}")
            print(f"  Non-Target Stream: Duration {nontgt_metrics['stream_duration_sec']/60:.1f} min | False Alarms: {nontgt_metrics['false_positives']} | FA/hr: {nontgt_metrics['fa_per_hour']}")
            
            seed_eval[mode_name] = {
                "threshold": th,
                "command_stream": cmd_metrics,
                "nontarget_stream": nontgt_metrics
            }
            
        campaign_streaming_results[f"seed{seed}"] = seed_eval
        
    # Save full streaming evaluation results
    streaming_results_path = os.path.join(PREDICTIONS_DIR, "streaming_results.json")
    with open(streaming_results_path, "w", encoding="utf-8") as f:
        json.dump(campaign_streaming_results, f, indent=2)
    print(f"\nSaved all streaming simulation results to: {streaming_results_path}")
    
    # Print Multi-seed Comparison Table
    print("\n" + "=" * 95)
    print("PHASE 32: CONTINUOUS-STREAM SIMULATION SUMMARY (3 Seeds, Before vs After Calibration)")
    print("=" * 95)
    print(f"{'Seed':<8} | {'Mode':<14} | {'Theta':<6} | {'Recall (%)':<12} | {'Precision (%)':<15} | {'Mean Latency':<14} | {'FA/hr (Non-Tgt)':<16}")
    print("-" * 95)
    for s in seeds:
        k = f"seed{s}"
        for mode in ["uncalibrated", "calibrated"]:
            m_cmd = campaign_streaming_results[k][mode]["command_stream"]
            m_nt = campaign_streaming_results[k][mode]["nontarget_stream"]
            th = campaign_streaming_results[k][mode]["threshold"]
            lat_str = f"{m_cmd['latencies_ms']['mean']} ms" if m_cmd['latencies_ms']['mean'] else "N/A"
            print(f"{k:<8} | {mode:<14} | {th:<6.2f} | {m_cmd['recall']*100:<12.1f} | {m_cmd['precision']*100:<15.1f} | {lat_str:<14} | {m_nt['fa_per_hour']:<16.2f}")
        print("-" * 95)

if __name__ == "__main__":
    run_streaming_campaign()
