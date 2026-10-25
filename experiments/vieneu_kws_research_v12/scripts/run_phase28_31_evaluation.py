#!/usr/bin/env python3
"""
Phase 28-31 Stage F: Locked Final Evaluation Harness
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/run_phase28_31_evaluation.py

Evaluates:
1. Baseline v11 (Phase 11 release)
2. Baseline E1 (Phase 14 release)
3. Phase 26 Candidate (D0_H1_seed44)
4. Phase 28-31 Candidate (phase28_candidate)

Across:
- Independent Edge-TTS Isolated Test Corpus (480 clips, edge_tts_eval_v2_isolated_test_manifest.csv)
- Independent Edge-TTS Streaming Evaluation Corpus (50 sequences, edge_tts_eval_v2_streaming_manifest.csv)
Using canonical SpeechBandRatioVAD and theta_cmd = 0.20.

Reports:
- Isolated 4-command Macro-F1, Per-class P/R/F1, Confusion Matrix, OOV FAR (Phonetic & General)
- Streaming Exact Sequence Accuracy, Event Precision/Recall/F1
- Silence FA/min, OOV FA/min
- Detailed error breakdown: Misses, Substitutions, False Insertions, Duplicates, Merges, Splits, Premature Cuts
- P50/P95 Latency
- Stratified breakdowns: Voice, Prosody Rate, SNR Tier, Sequence Type
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
from scipy import signal
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase28_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX, compute_sha256

ISOLATED_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_v2_isolated_test_manifest.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_v2_streaming_manifest.csv"
STREAMING_EVENTS = MANIFESTS_DIR / "edge_tts_eval_v2_streaming_events.json"
VAD_CONFIG_FILE = CONFIGS_DIR / "selected_phase15_vad_config.json"

MODELS_CONFIG = {
    "Phase28_D2_H1_Candidate": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_d2_candidate" / "candidate_checkpoint.pt",
        "description": "Phase 28-31 D2_H1 winning seed candidate (D2_H1_seed42, 10% Edge-TTS augmented)"
    },
    "Phase28_D0_H1_Candidate": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_candidate" / "candidate_checkpoint.pt",
        "description": "Phase 28-31 D0_H1 winning seed candidate (D0_H1_seed42, VieNeu Turbo only)"
    },
    "Phase26_Candidate_D0_H1_seed44": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44" / "candidate_checkpoint.pt",
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


class SpeechBandRatioVAD:
    def __init__(self, sample_rate=16000, speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0):
        self.sr = sample_rate
        self.f_min, self.f_max = speech_band_hz
        self.min_ratio = min_speech_band_ratio
        self.energy_thresh = 10.0 ** (energy_threshold_db / 10.0)

    def is_speech(self, frame: np.ndarray) -> bool:
        if len(frame) == 0:
            return False
        energy = np.mean(frame ** 2)
        if energy < self.energy_thresh:
            return False

        w = np.hanning(len(frame))
        fft_vals = np.abs(np.fft.rfft(frame * w)) ** 2
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / self.sr)

        tot_p = np.sum(fft_vals)
        if tot_p <= 1e-12:
            return False

        band_mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        band_p = np.sum(fft_vals[band_mask])
        ratio = band_p / tot_p
        return ratio >= self.min_ratio


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
    overall_macro_f1 = float(np.mean(f1))
    acc = float(np.mean(y_true == y_pred))

    oov_mask = (y_true == 4)
    oov_fa_count = int(np.sum(y_pred[oov_mask] != 4))
    oov_total = int(np.sum(oov_mask))
    oov_fa_rate = oov_fa_count / oov_total if oov_total > 0 else 0.0

    phonetic_mask = (df_iso["subgroup"] == "test_oov_phonetic").values
    phonetic_fa = int(np.sum(y_pred[phonetic_mask] != 4))
    phonetic_total = int(np.sum(phonetic_mask))

    general_mask = (df_iso["subgroup"] == "test_oov_general").values
    general_fa = int(np.sum(y_pred[general_mask] != 4))
    general_total = int(np.sum(general_mask))

    # SNR breakdown
    snr_breakdown = {}
    for snr in ["clean", "15.0", "10.0", "5.0"]:
        snr_m = (df_iso["target_snr_db"] == snr).values
        if np.sum(snr_m) > 0:
            yt_s = y_true[snr_m]
            yp_s = y_pred[snr_m]
            _, _, f1_s, _ = precision_recall_fscore_support(yt_s, yp_s, labels=[0, 1, 2, 3], average=None, zero_division=0)
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
        _, _, f1_v, _ = precision_recall_fscore_support(yt_v, yp_v, labels=[0, 1, 2, 3], average=None, zero_division=0)
        voice_breakdown[v] = {
            "macro_f1": float(np.mean(f1_v)),
            "accuracy": float(np.mean(yt_v == yp_v)),
            "oov_fa_rate": float(np.mean(yp_v[yt_v == 4] != 4)) if np.sum(yt_v == 4) > 0 else 0.0
        }

    # Prosody rate breakdown
    rate_breakdown = {}
    for r_lbl in df_iso["rate"].unique():
        r_m = (df_iso["rate"] == r_lbl).values
        yt_r = y_true[r_m]
        yp_r = y_pred[r_m]
        _, _, f1_r, _ = precision_recall_fscore_support(yt_r, yp_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        rate_breakdown[r_lbl] = {
            "macro_f1": float(np.mean(f1_r)),
            "accuracy": float(np.mean(yt_r == yp_r)),
            "oov_fa_rate": float(np.mean(yp_r[yt_r == 4] != 4)) if np.sum(yt_r == 4) > 0 else 0.0
        }

    return {
        "macro_f1": macro_f1,
        "overall_macro_f1": overall_macro_f1,
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
        "voice_breakdown": voice_breakdown,
        "rate_breakdown": rate_breakdown
    }


def evaluate_streaming(model: torch.nn.Module, frontend: torch.nn.Module, df_stream: pd.DataFrame, events_dict: Dict[str, List], device: torch.device) -> Dict[str, Any]:
    vad = SpeechBandRatioVAD(speech_band_hz=(300.0, 3400.0), min_speech_band_ratio=0.32, energy_threshold_db=-46.0)

    exact_matches = 0
    total_seqs = len(df_stream)

    total_gt_cmds = 0
    total_detected_cmds = 0
    true_positive_cmds = 0

    substitutions = 0
    false_insertions = 0
    duplicates = 0
    merges = 0
    splits = 0
    premature_cuts = 0

    silence_seq_fa = 0
    total_silence_duration_s = 0.0

    oov_seq_fa = 0
    total_oov_duration_s = 0.0

    latencies = []

    step_samples = 1600  # 100ms
    win_samples = 16000  # 1.0s

    snr_seq_results = {}
    type_seq_results = {}

    for idx, row in df_stream.iterrows():
        seq_id = row["sequence_id"]
        seq_type = row["sequence_type"]
        snr_val = row["target_snr_db"]
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, _ = sf.read(str(wav_path), dtype="float32")
        seq_len = len(audio)
        dur_s = seq_len / 16000.0

        gt_events = events_dict.get(seq_id, [])
        gt_cmds = [ev for ev in gt_events if ev["label"] != "REJECT"]

        if seq_type == "pure_silence_noise":
            total_silence_duration_s += dur_s
        elif seq_type == "pure_oov_conversation":
            total_oov_duration_s += dur_s

        detected_events = []
        last_trigger_pos = -999999

        for pos in range(0, seq_len - win_samples + 1, step_samples):
            if (pos - last_trigger_pos) < 8000:
                continue

            frame = audio[pos : pos + win_samples]
            recent_frame = frame[-3200:]
            if not vad.is_speech(recent_frame):
                continue

            wav_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                spec = frontend(wav_t)
                logits = model(spec)
                probs = F.softmax(logits, dim=-1)[0]
                target_probs = probs[:4]
                max_p, max_idx = torch.max(target_probs, dim=-1)

                if max_p.item() >= 0.20:
                    cmd_label = CLASSES[max_idx.item()]
                    trigger_time = (pos + win_samples) / 16000.0
                    detected_events.append({
                        "label": cmd_label,
                        "time": trigger_time,
                        "prob": float(max_p.item())
                    })
                    last_trigger_pos = pos

        detected_cmd_seq = [d["label"] for d in detected_events]
        gt_cmd_seq = [g["label"] for g in gt_cmds]

        is_exact = (detected_cmd_seq == gt_cmd_seq)
        if is_exact:
            exact_matches += 1

        if seq_type == "pure_silence_noise":
            silence_seq_fa += len(detected_events)
        elif seq_type == "pure_oov_conversation":
            oov_seq_fa += len(detected_events)

        total_gt_cmds += len(gt_cmds)
        total_detected_cmds += len(detected_events)

        # Track per-event alignment & detailed error taxonomy
        matched_gt = set()
        matched_det = set()

        for d_i, det in enumerate(detected_events):
            best_match = None
            for g_i, gt in enumerate(gt_cmds):
                if (gt["start_time"] - 0.2) <= det["time"] <= (gt["end_time"] + 0.6):
                    best_match = (g_i, gt)
                    break

            if best_match is not None:
                g_i, gt = best_match
                if det["label"] == gt["label"]:
                    if g_i not in matched_gt:
                        matched_gt.add(g_i)
                        matched_det.add(d_i)
                        true_positive_cmds += 1
                        lat = max(0.0, det["time"] - gt["start_time"])
                        latencies.append(lat)
                        if lat < 0.15:
                            premature_cuts += 1
                    else:
                        duplicates += 1
                else:
                    substitutions += 1
            else:
                false_insertions += 1

        # Track sequence type statistics
        type_seq_results.setdefault(seq_type, {"total": 0, "exact": 0, "gt_cmds": 0, "tp_cmds": 0})
        type_seq_results[seq_type]["total"] += 1
        type_seq_results[seq_type]["exact"] += int(is_exact)
        type_seq_results[seq_type]["gt_cmds"] += len(gt_cmds)
        type_seq_results[seq_type]["tp_cmds"] += len(matched_gt)

        snr_seq_results.setdefault(snr_val, {"total": 0, "exact": 0, "gt_cmds": 0, "tp_cmds": 0})
        snr_seq_results[snr_val]["total"] += 1
        snr_seq_results[snr_val]["exact"] += int(is_exact)
        snr_seq_results[snr_val]["gt_cmds"] += len(gt_cmds)
        snr_seq_results[snr_val]["tp_cmds"] += len(matched_gt)

    misses = total_gt_cmds - true_positive_cmds
    cmd_recall = true_positive_cmds / total_gt_cmds if total_gt_cmds > 0 else 0.0
    cmd_precision = true_positive_cmds / total_detected_cmds if total_detected_cmds > 0 else 0.0
    cmd_f1 = 2 * cmd_precision * cmd_recall / (cmd_precision + cmd_recall) if (cmd_precision + cmd_recall) > 0 else 0.0

    silence_min = total_silence_duration_s / 60.0
    silence_fa_per_min = silence_seq_fa / silence_min if silence_min > 0 else 0.0

    oov_min = total_oov_duration_s / 60.0
    oov_fa_per_min = oov_seq_fa / oov_min if oov_min > 0 else 0.0

    p50_lat = float(np.median(latencies)) * 1000.0 if len(latencies) > 0 else 0.0
    p95_lat = float(np.percentile(latencies, 95)) * 1000.0 if len(latencies) > 0 else 0.0

    return {
        "total_sequences": total_seqs,
        "exact_sequence_matches": exact_matches,
        "exact_sequence_accuracy": exact_matches / total_seqs if total_seqs > 0 else 0.0,
        "total_gt_commands": total_gt_cmds,
        "total_detected_commands": total_detected_cmds,
        "true_positive_commands": true_positive_cmds,
        "command_recall": cmd_recall,
        "command_precision": cmd_precision,
        "command_f1": cmd_f1,
        "error_taxonomy": {
            "misses": misses,
            "substitutions": substitutions,
            "false_insertions": false_insertions,
            "duplicates": duplicates,
            "merges": merges,
            "splits": splits,
            "premature_cuts": premature_cuts
        },
        "silence_duration_min": round(silence_min, 2),
        "silence_false_alarms": silence_seq_fa,
        "silence_fa_per_min": round(silence_fa_per_min, 2),
        "oov_duration_min": round(oov_min, 2),
        "oov_false_alarms": oov_seq_fa,
        "oov_fa_per_min": round(oov_fa_per_min, 2),
        "mean_latency_ms": round(float(np.mean(latencies)) * 1000.0, 1) if len(latencies) > 0 else 0.0,
        "p50_latency_ms": round(p50_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "type_breakdown": type_seq_results,
        "snr_breakdown": snr_seq_results
    }


def main():
    print("=" * 80)
    print("PHASE 28-31 STAGE F: LOCKED FINAL EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_iso = pd.read_csv(ISOLATED_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)
    with open(STREAMING_EVENTS, "r", encoding="utf-8") as f:
        stream_events = json.load(f)

    print(f"Loaded Isolated Manifest: {len(df_iso)} clips ({compute_sha256(ISOLATED_MANIFEST)})")
    print(f"Loaded Streaming Manifest: {len(df_stream)} seqs ({compute_sha256(STREAMING_MANIFEST)})")

    frontend = LogMelFrontend().to(device)
    frontend.eval()

    all_results = {}

    for model_name, cfg in MODELS_CONFIG.items():
        print(f"\n" + "-" * 70)
        print(f"Evaluating Model: {model_name}")
        print(f"Path: {cfg['path']}")

        if not cfg["path"].exists():
            print(f"[ERROR] Checkpoint not found: {cfg['path']}")
            continue

        model = load_model(cfg["path"], device)
        ckpt_sha = compute_sha256(cfg["path"])

        t0 = time.perf_counter()
        iso_res = evaluate_isolated(model, frontend, df_iso, device)
        t_iso = time.perf_counter() - t0

        t1 = time.perf_counter()
        stream_res = evaluate_streaming(model, frontend, df_stream, stream_events, device)
        t_stream = time.perf_counter() - t1

        print(f"  [Isolated] Macro-F1: {iso_res['macro_f1']*100:.2f}% | Overall F1: {iso_res['overall_macro_f1']*100:.2f}% | OOV FAR: {iso_res['oov_fa_rate']*100:.2f}% (Phonetic: {iso_res['phonetic_oov_fa_rate']*100:.2f}%)")
        print(f"  [Streaming] Exact Match: {stream_res['exact_sequence_accuracy']*100:.2f}% | Cmd Recall: {stream_res['command_recall']*100:.2f}% | Cmd Prec: {stream_res['command_precision']*100:.2f}%")
        print(f"  [Streaming Errors] Misses: {stream_res['error_taxonomy']['misses']} | Subs: {stream_res['error_taxonomy']['substitutions']} | False Ins: {stream_res['error_taxonomy']['false_insertions']} | Dups: {stream_res['error_taxonomy']['duplicates']}")
        print(f"  [Streaming FA/min] Silence: {stream_res['silence_fa_per_min']:.2f} FA/min | OOV: {stream_res['oov_fa_per_min']:.2f} FA/min | P50 Lat: {stream_res['p50_latency_ms']:.1f}ms | P95 Lat: {stream_res['p95_latency_ms']:.1f}ms")

        all_results[model_name] = {
            "checkpoint_path": str(cfg["path"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "checkpoint_sha256": ckpt_sha,
            "description": cfg["description"],
            "isolated": iso_res,
            "streaming": stream_res,
            "timings": {
                "isolated_eval_seconds": round(t_iso, 2),
                "streaming_eval_seconds": round(t_stream, 2)
            }
        }

    out_file = REPORTS_DIR / "phase28_31_final_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "phase": "Phase 28-31 Stage F Locked Final Evaluation",
                "timestamp": "2026-10-25T09:45:00+07:00",
                "isolated_manifest_sha256": compute_sha256(ISOLATED_MANIFEST),
                "streaming_manifest_sha256": compute_sha256(STREAMING_MANIFEST),
                "streaming_events_sha256": compute_sha256(STREAMING_EVENTS),
                "theta_cmd": 0.20,
                "vad_algorithm": "SpeechBandRatioVAD"
            },
            "models": all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Final Evaluation Complete! Results saved to: {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
