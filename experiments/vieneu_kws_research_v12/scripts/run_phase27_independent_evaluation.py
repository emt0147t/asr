#!/usr/bin/env python3
"""
Phase 27: Independent Synthetic Evaluation Harness
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_phase27_independent_evaluation.py

Evaluates the frozen Phase 26 candidate alongside historical v11 and E1 models
across the independent Edge-TTS isolated and streaming evaluation corpora.
"""

import sys
import io
import time
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
CANDIDATE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase25_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX

ISOLATED_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_isolated_test_manifest.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_streaming_manifest.csv"
STREAMING_EVENTS = MANIFESTS_DIR / "edge_tts_eval_streaming_events.json"

MODELS_CONFIG = {
    "Phase26_Candidate_D0_H1_seed44": {
        "path": CANDIDATE_DIR / "candidate_checkpoint.pt",
        "description": "Phase 26 winning condition candidate (D0_H1_seed44)"
    },
    "Baseline_E1_seed42": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt",
        "description": "Phase 18 E1 baseline scale model"
    },
    "Baseline_v11_seed42": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt",
        "description": "Phase 13.4 v11 release model"
    }
}


def load_model(ckpt_path: Path, device: torch.device) -> CorrectedVietnameseKWSNet:
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_isolated(model: torch.nn.Module, frontend: torch.nn.Module, df_iso: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    preds = []
    targets = []

    for idx, row in df_iso.iterrows():
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]

        wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            spec = frontend(wav_t)
            logits = model(spec)
            probs = F.softmax(logits, dim=-1)[0]

            target_probs = probs[:4]
            max_p, max_idx = torch.max(target_probs, dim=-1)
            pred = int(max_idx.item()) if max_p.item() >= 0.20 else 4

        preds.append(pred)
        targets.append(CLASS_TO_IDX[row["label"]])

    y_true = np.array(targets)
    y_pred = np.array(preds)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)

    macro_f1 = float(np.mean(f1[:4]))
    acc = float(np.mean(y_true == y_pred))

    # OOV-specific analysis
    oov_mask = (y_true == 4)
    oov_fa_count = int(np.sum(y_pred[oov_mask] != 4))
    oov_total = int(np.sum(oov_mask))
    oov_fa_rate = oov_fa_count / oov_total if oov_total > 0 else 0.0

    # Phonetic vs General OOV
    phonetic_mask = (df_iso["subgroup"] == "test_oov_phonetic").values
    phonetic_fa = int(np.sum(y_pred[phonetic_mask] != 4))
    phonetic_total = int(np.sum(phonetic_mask))

    general_mask = (df_iso["subgroup"] == "test_oov_general").values
    general_fa = int(np.sum(y_pred[general_mask] != 4))
    general_total = int(np.sum(general_mask))

    # SNR breakdown
    snr_breakdown = {}
    for snr in ["clean", 20.0, 15.0, 10.0]:
        snr_m = (df_iso["target_snr_db"] == snr).values
        if np.sum(snr_m) > 0:
            yt_s = y_true[snr_m]
            yp_s = y_pred[snr_m]
            p_s, r_s, f1_s, _ = precision_recall_fscore_support(yt_s, yp_s, labels=[0, 1, 2, 3], average=None, zero_division=0)
            snr_breakdown[str(snr)] = {
                "macro_f1": float(np.mean(f1_s)),
                "accuracy": float(np.mean(yt_s == yp_s)),
                "oov_fa_rate": float(np.mean(yp_s[yt_s == 4] != 4)) if np.sum(yt_s == 4) > 0 else 0.0
            }

    # Voice breakdown
    voice_breakdown = {}
    for v in df_iso["voice"].unique():
        v_m = (df_iso["voice"] == v).values
        yt_v = y_true[v_m]
        yp_v = y_pred[v_m]
        p_v, r_v, f1_v, _ = precision_recall_fscore_support(yt_v, yp_v, labels=[0, 1, 2, 3], average=None, zero_division=0)
        voice_breakdown[v] = {
            "macro_f1": float(np.mean(f1_v)),
            "accuracy": float(np.mean(yt_v == yp_v)),
            "oov_fa_rate": float(np.mean(yp_v[yt_v == 4] != 4)) if np.sum(yt_v == 4) > 0 else 0.0
        }

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASSES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i])
            }
            for i in range(5)
        },
        "oov_total": oov_total,
        "oov_fa_count": oov_fa_count,
        "oov_fa_rate": oov_fa_rate,
        "phonetic_oov_fa_count": phonetic_fa,
        "phonetic_oov_total": phonetic_total,
        "phonetic_oov_fa_rate": phonetic_fa / phonetic_total if phonetic_total > 0 else 0.0,
        "general_oov_fa_count": general_fa,
        "general_oov_total": general_total,
        "general_oov_fa_rate": general_fa / general_total if general_total > 0 else 0.0,
        "snr_breakdown": snr_breakdown,
        "voice_breakdown": voice_breakdown
    }


def evaluate_streaming(model: torch.nn.Module, frontend: torch.nn.Module, df_stream: pd.DataFrame, events_list: List[Dict], device: torch.device) -> Dict[str, Any]:
    events_by_seq = {}
    for ev in events_list:
        events_by_seq.setdefault(ev["sequence_id"], []).append(ev)

    exact_matches = 0
    total_seqs = len(df_stream)

    total_gt_cmds = 0
    total_detected_cmds = 0
    true_positive_cmds = 0
    oov_false_activations = 0
    silence_noise_false_activations = 0

    latencies = []
    total_speech_duration_min = 0.0
    total_silence_duration_min = 0.0

    step_samples = 1600  # 100ms
    win_samples = 16000  # 1.0s

    for idx, row in df_stream.iterrows():
        seq_id = row["sequence_id"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, _ = sf.read(str(wav_path), dtype="float32")
        seq_len = len(audio)

        gt_events = events_by_seq.get(seq_id, [])
        gt_cmds = [ev for ev in gt_events if ev["label"] != "REJECT"]
        gt_oovs = [ev for ev in gt_events if ev["label"] == "REJECT"]

        total_gt_cmds += len(gt_cmds)

        # Compute durations for FA/min denominators
        oov_dur_s = sum(ev["duration"] for ev in gt_oovs)
        total_speech_duration_min += (oov_dur_s / 60.0)
        silence_dur_s = max(0.0, (seq_len / 16000.0) - sum(ev["duration"] for ev in gt_events))
        total_silence_duration_min += (silence_dur_s / 60.0)

        # Sliding window streaming inference
        detected_events = []
        last_trigger_sample = -16000  # debounce 400ms = 6400 samples

        for s_idx in range(0, seq_len - win_samples + 1, step_samples):
            chunk = audio[s_idx : s_idx + win_samples]

            # Fast energy VAD pre-filter
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.005:
                continue

            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(device)
            with torch.no_grad():
                spec = frontend(chunk_t)
                logits = model(spec)
                probs = F.softmax(logits, dim=-1)[0]
                target_probs = probs[:4]
                max_p, max_idx = torch.max(target_probs, dim=-1)

                if max_p.item() >= 0.20 and (s_idx - last_trigger_sample) >= 6400:
                    trigger_t = (s_idx + win_samples) / 16000.0
                    detected_events.append({
                        "label": CLASSES[max_idx.item()],
                        "time": trigger_t,
                        "sample": s_idx
                    })
                    last_trigger_sample = s_idx

        # Evaluate detected events against ground truth
        detected_seq = [d["label"] for d in detected_events]
        expected_seq = eval(row["expected_command_sequence"]) if isinstance(row["expected_command_sequence"], str) else row["expected_command_sequence"]

        if detected_seq == expected_seq:
            exact_matches += 1

        total_detected_cmds += len(detected_events)

        # Match detected events to GT commands within tolerance [start - 0.2, end + 0.5]
        matched_gt = set()
        for d in detected_events:
            dt = d["time"]
            dlbl = d["label"]
            match_found = False

            # Check GT commands
            for i, gt in enumerate(gt_cmds):
                if i not in matched_gt and gt["label"] == dlbl:
                    if (gt["start_time"] - 0.20) <= dt <= (gt["end_time"] + 0.60):
                        match_found = True
                        matched_gt.add(i)
                        true_positive_cmds += 1
                        latencies.append(dt - gt["start_time"])
                        break

            if not match_found:
                # Was it in an OOV interval?
                in_oov = any(gt["start_time"] - 0.10 <= dt <= gt["end_time"] + 0.30 for gt in gt_oovs)
                if in_oov:
                    oov_false_activations += 1
                else:
                    silence_noise_false_activations += 1

    # Aggregate streaming metrics
    seq_acc = exact_matches / total_seqs
    event_recall = true_positive_cmds / total_gt_cmds if total_gt_cmds > 0 else 0.0
    event_precision = true_positive_cmds / total_detected_cmds if total_detected_cmds > 0 else 0.0
    event_f1 = (2 * event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0.0

    fa_per_min_oov = oov_false_activations / total_speech_duration_min if total_speech_duration_min > 0 else 0.0
    fa_per_min_silence = silence_noise_false_activations / total_silence_duration_min if total_silence_duration_min > 0 else 0.0

    p50_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "exact_sequence_accuracy": seq_acc,
        "exact_sequence_matches": exact_matches,
        "total_sequences": total_seqs,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "total_gt_commands": total_gt_cmds,
        "total_detected_commands": total_detected_cmds,
        "true_positive_commands": true_positive_cmds,
        "oov_false_activations": oov_false_activations,
        "silence_noise_false_activations": silence_noise_false_activations,
        "fa_per_minute_oov": fa_per_min_oov,
        "fa_per_minute_silence": fa_per_min_silence,
        "latency_p50_s": p50_lat,
        "latency_p95_s": p95_lat
    }


def main():
    print("=" * 80)
    print("PHASE 27: RUNNING INDEPENDENT SYNTHETIC EVALUATION ON EDGE-TTS CORPUS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    df_iso = pd.read_csv(ISOLATED_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)
    with open(STREAMING_EVENTS, "r", encoding="utf-8") as f:
        events_list = json.load(f)

    evaluation_report = {}

    for model_name, cfg in MODELS_CONFIG.items():
        print(f"\n>>> Evaluating {model_name} ({cfg['description']}) <<<")
        model = load_model(cfg["path"], device)

        iso_res = evaluate_isolated(model, frontend, df_iso, device)
        print(f"  [Isolated] Macro-F1: {iso_res['macro_f1']*100:.2f}% | Acc: {iso_res['accuracy']*100:.2f}% | OOV FA Rate: {iso_res['oov_fa_rate']*100:.2f}%")
        print(f"             Phonetic OOV FA: {iso_res['phonetic_oov_fa_rate']*100:.2f}% | General OOV FA: {iso_res['general_oov_fa_rate']*100:.2f}%")

        stream_res = evaluate_streaming(model, frontend, df_stream, events_list, device)
        print(f"  [Streaming] Seq Acc: {stream_res['exact_sequence_accuracy']*100:.2f}% ({stream_res['exact_sequence_matches']}/100)")
        print(f"              Event F1: {stream_res['event_f1']*100:.2f}% (P={stream_res['event_precision']*100:.2f}%, R={stream_res['event_recall']*100:.2f}%)")
        print(f"              OOV FA/min: {stream_res['fa_per_minute_oov']:.2f} | Silence FA/min: {stream_res['fa_per_minute_silence']:.2f}")
        print(f"              P50 Latency: {stream_res['latency_p50_s']:.3f}s | P95 Latency: {stream_res['latency_p95_s']:.3f}s")

        evaluation_report[model_name] = {
            "checkpoint_path": str(cfg["path"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "isolated_metrics": iso_res,
            "streaming_metrics": stream_res
        }

    out_json = REPORTS_DIR / "phase27_independent_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n[OK] Evaluation completed. Saved results to {out_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
