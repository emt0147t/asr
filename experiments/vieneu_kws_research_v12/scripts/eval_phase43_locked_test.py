#!/usr/bin/env python3
"""
Phase 43: Locked Inter-Generator Evaluation Harness
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/eval_phase43_locked_test.py

Evaluates:
- Baseline v11 (default model)
- Baseline E1
- Phase 28 D2_H1 Candidate (D2_H1_seed42)
- Phase 28 D0_H1 Candidate (D0_H1_seed42)
- Phase 26 Candidate (D0_H1_seed44)

Across:
- Independent MMS-TTS + ESC-50 Isolated Manifest (480 clips)
- MMS-TTS Diagnostic Tone Probes (96 clips)
- MMS-TTS + ESC-50 Streaming Evaluation Corpus (50 sequences)

Strict Protocol Invariants:
- Fixed frontend: LogMelFrontend (40 mels, win=400, hop=160, n_fft=512)
- Fixed VAD: SpeechBandRatioVAD (300-3400 Hz, ratio >= 0.32, energy >= -46 dBFS)
- Fixed decision threshold: theta_cmd = 0.20
- Fixed parameter count: 138,469 parameters
"""

import sys
import io
import time
import json
import hashlib
from datetime import datetime, timezone
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

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase28_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX, compute_sha256

ISOLATED_MANIFEST = MANIFESTS_DIR / "PHASE43_LOCKED_MANIFEST.csv"
TONE_PROBE_MANIFEST = MANIFESTS_DIR / "PHASE43_TONE_PROBE_MANIFEST.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "PHASE43_STREAMING_MANIFEST.csv"
EVENT_LEDGER_CSV = MANIFESTS_DIR / "PHASE43_EVENT_LEDGER.csv"
EVENT_LEDGER_JSON = MANIFESTS_DIR / "PHASE43_EVENT_LEDGER.json"

MODELS_CONFIG = {
    "Baseline_v11_seed42": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt",
        "description": "Canonical v11 default model (Phase 13.4)",
        "expected_hash": "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
    },
    "Baseline_E1_seed42": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt",
        "description": "Phase 18 E1 baseline scale model",
        "expected_hash": "f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878"
    },
    "Phase28_D2_H1_Candidate": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_d2_candidate" / "candidate_checkpoint.pt",
        "description": "Phase 28-31 D2_H1 winning candidate (D2_H1_seed42)",
        "expected_hash": "c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8"
    },
    "Phase28_D0_H1_Candidate": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_candidate" / "candidate_checkpoint.pt",
        "description": "Phase 28-31 D0_H1 winning candidate (D0_H1_seed42)",
        "expected_hash": "84a3457f105058bb7ca843b830013d00cb6d2e9ce737cb389267a84fcf51a6af"
    },
    "Phase26_Candidate_D0_H1_seed44": {
        "path": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44" / "candidate_checkpoint.pt",
        "description": "Phase 26 D0_H1 candidate (seed44)",
        "expected_hash": "00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3"
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
    raw_probs = []

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
        raw_probs.append(probs.cpu().numpy().tolist())

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

    # Speaking rate breakdown
    rate_breakdown = {}
    for r_lbl in df_iso["rate_label"].unique():
        r_m = (df_iso["rate_label"] == r_lbl).values
        yt_r = y_true[r_m]
        yp_r = y_pred[r_m]
        _, _, f1_r, _ = precision_recall_fscore_support(yt_r, yp_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        rate_breakdown[r_lbl] = {
            "macro_f1": float(np.mean(f1_r)),
            "accuracy": float(np.mean(yt_r == yp_r)),
            "oov_fa_rate": float(np.mean(yp_r[yt_r == 4] != 4)) if np.sum(yt_r == 4) > 0 else 0.0
        }

    # Rule of Three UCB95 bounds
    ucb95_overall_oov = (1.0 - (1.0 - 0.95) ** (1.0 / oov_total)) if oov_fa_count == 0 else (oov_fa_rate + 1.96 * np.sqrt(oov_fa_rate * (1 - oov_fa_rate) / oov_total))
    ucb95_phonetic_oov = (1.0 - (1.0 - 0.95) ** (1.0 / phonetic_total)) if phonetic_fa == 0 else (phonetic_fa / phonetic_total + 1.96 * np.sqrt((phonetic_fa / phonetic_total) * (1 - phonetic_fa / phonetic_total) / phonetic_total))
    ucb95_general_oov = (1.0 - (1.0 - 0.95) ** (1.0 / general_total)) if general_fa == 0 else (general_fa / general_total + 1.96 * np.sqrt((general_fa / general_total) * (1 - general_fa / general_total) / general_total))

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
        "ucb95_overall_oov_far": float(ucb95_overall_oov),
        "phonetic_oov_total": phonetic_total,
        "phonetic_oov_fa_count": phonetic_fa,
        "phonetic_oov_fa_rate": phonetic_fa / phonetic_total if phonetic_total > 0 else 0.0,
        "ucb95_phonetic_oov_far": float(ucb95_phonetic_oov),
        "general_oov_total": general_total,
        "general_oov_fa_count": general_fa,
        "general_oov_fa_rate": general_fa / general_total if general_total > 0 else 0.0,
        "ucb95_general_oov_far": float(ucb95_general_oov),
        "snr_breakdown": snr_breakdown,
        "rate_breakdown": rate_breakdown,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }


def evaluate_tone_probes(model: torch.nn.Module, frontend: torch.nn.Module, df_probe: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    preds = []
    targets = []
    for idx, row in df_probe.iterrows():
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, _ = sf.read(str(wav_path), dtype="float32")
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
        targets.append(4)  # All probes are REJECT

    y_pred = np.array(preds)
    fa_count = int(np.sum(y_pred != 4))
    total = len(y_pred)
    fa_rate = fa_count / total if total > 0 else 0.0
    ucb95 = (1.0 - (1.0 - 0.95) ** (1.0 / total)) if fa_count == 0 else (fa_rate + 1.96 * np.sqrt(fa_rate * (1 - fa_rate) / total))

    return {
        "probe_total": total,
        "probe_fa_count": fa_count,
        "probe_fa_rate": fa_rate,
        "probe_ucb95_far": float(ucb95),
        "predictions": y_pred.tolist()
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
    silence_ucb95 = (3.0 / silence_min) if (silence_seq_fa == 0 and silence_min > 0) else (silence_fa_per_min + 1.96 * np.sqrt(silence_fa_per_min / silence_min))

    oov_min = total_oov_duration_s / 60.0
    oov_fa_per_min = oov_seq_fa / oov_min if oov_min > 0 else 0.0
    oov_ucb95 = (3.0 / oov_min) if (oov_seq_fa == 0 and oov_min > 0) else (oov_fa_per_min + 1.96 * np.sqrt(oov_fa_per_min / oov_min))

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
        "silence_duration_min": round(silence_min, 3),
        "silence_false_alarms": silence_seq_fa,
        "silence_fa_per_min": round(silence_fa_per_min, 2),
        "silence_ucb95_fa_per_min": round(silence_ucb95, 2),
        "oov_duration_min": round(oov_min, 3),
        "oov_false_alarms": oov_seq_fa,
        "oov_fa_per_min": round(oov_fa_per_min, 2),
        "oov_ucb95_fa_per_min": round(oov_ucb95, 2),
        "mean_latency_ms": round(float(np.mean(latencies)) * 1000.0, 1) if len(latencies) > 0 else 0.0,
        "p50_latency_ms": round(p50_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "type_breakdown": type_seq_results,
        "snr_breakdown": snr_seq_results
    }


def compute_bootstrap_estimands(
    df_iso: pd.DataFrame,
    preds_d2: List[int],
    preds_baseline: List[int],
    y_true: List[int],
    B: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    N = len(df_iso)
    y_t = np.array(y_true)
    y_d2 = np.array(preds_d2)
    y_b = np.array(preds_baseline)

    # Estimand A: Observed
    _, _, f1_d2_obs, _ = precision_recall_fscore_support(y_t, y_d2, labels=[0, 1, 2, 3], average=None, zero_division=0)
    _, _, f1_b_obs, _ = precision_recall_fscore_support(y_t, y_b, labels=[0, 1, 2, 3], average=None, zero_division=0)
    obs_d2 = float(np.mean(f1_d2_obs))
    obs_b = float(np.mean(f1_b_obs))
    obs_delta = obs_d2 - obs_b

    # Define clusters
    # Base text clusters: 30 unique words
    unique_texts = df_iso["text"].unique()
    text_to_indices = {txt: np.where(df_iso["text"] == txt)[0] for txt in unique_texts}
    K_text = len(unique_texts)

    # Synthesis instance clusters: 120 unique (text + config_id)
    df_iso_copy = df_iso.copy()
    df_iso_copy["instance_id"] = df_iso_copy["text"] + "_" + df_iso_copy["config_id"]
    unique_instances = df_iso_copy["instance_id"].unique()
    instance_to_indices = {inst: np.where(df_iso_copy["instance_id"] == inst)[0] for inst in unique_instances}
    K_inst = len(unique_instances)

    # Estimand B: Text Cluster Bootstrap
    deltas_b = []
    for _ in range(B):
        resampled_texts = rng.choice(unique_texts, size=K_text, replace=True)
        idx_list = []
        for t in resampled_texts:
            idx_list.extend(text_to_indices[t])
        idx_arr = np.array(idx_list)
        yt_r = y_t[idx_arr]
        yd2_r = y_d2[idx_arr]
        yb_r = y_b[idx_arr]
        _, _, f1_d2, _ = precision_recall_fscore_support(yt_r, yd2_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        _, _, f1_b, _ = precision_recall_fscore_support(yt_r, yb_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        deltas_b.append(float(np.mean(f1_d2) - np.mean(f1_b)))

    deltas_b = np.array(deltas_b)
    ci_b = np.percentile(deltas_b, [2.5, 97.5])
    p_b = float(np.mean(deltas_b <= 0.0))

    # Estimand C: Synthesis Instance Cluster Bootstrap
    deltas_c = []
    for _ in range(B):
        resampled_insts = rng.choice(unique_instances, size=K_inst, replace=True)
        idx_list = []
        for inst in resampled_insts:
            idx_list.extend(instance_to_indices[inst])
        idx_arr = np.array(idx_list)
        yt_r = y_t[idx_arr]
        yd2_r = y_d2[idx_arr]
        yb_r = y_b[idx_arr]
        _, _, f1_d2, _ = precision_recall_fscore_support(yt_r, yd2_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        _, _, f1_b, _ = precision_recall_fscore_support(yt_r, yb_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        deltas_c.append(float(np.mean(f1_d2) - np.mean(f1_b)))

    deltas_c = np.array(deltas_c)
    ci_c = np.percentile(deltas_c, [2.5, 97.5])
    p_c = float(np.mean(deltas_c <= 0.0))

    # Estimand D: Clip i.i.d. Bootstrap
    deltas_d = []
    for _ in range(B):
        idx_arr = rng.choice(N, size=N, replace=True)
        yt_r = y_t[idx_arr]
        yd2_r = y_d2[idx_arr]
        yb_r = y_b[idx_arr]
        _, _, f1_d2, _ = precision_recall_fscore_support(yt_r, yd2_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        _, _, f1_b, _ = precision_recall_fscore_support(yt_r, yb_r, labels=[0, 1, 2, 3], average=None, zero_division=0)
        deltas_d.append(float(np.mean(f1_d2) - np.mean(f1_b)))

    deltas_d = np.array(deltas_d)
    ci_d = np.percentile(deltas_d, [2.5, 97.5])
    p_d = float(np.mean(deltas_d <= 0.0))

    return {
        "estimand_A_observed_delta": float(obs_delta),
        "d2_observed_macro_f1": float(obs_d2),
        "baseline_observed_macro_f1": float(obs_b),
        "estimand_B_text_cluster": {
            "mean_delta": float(np.mean(deltas_b)),
            "ci_95": [float(ci_b[0]), float(ci_b[1])],
            "p_value": p_b
        },
        "estimand_C_instance_cluster": {
            "mean_delta": float(np.mean(deltas_c)),
            "ci_95": [float(ci_c[0]), float(ci_c[1])],
            "p_value": p_c
        },
        "estimand_D_clip_iid": {
            "mean_delta": float(np.mean(deltas_d)),
            "ci_95": [float(ci_d[0]), float(ci_d[1])],
            "p_value": p_d
        }
    }


def main():
    print("=" * 80)
    print("PHASE 43: LOCKED INTER-GENERATOR EVALUATION HARNESS")
    print("=" * 80)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load manifests
    print("\n--- 1. Loading Locked Manifests & Event Ledger ---")
    df_iso = pd.read_csv(ISOLATED_MANIFEST)
    df_probe = pd.read_csv(TONE_PROBE_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)
    with open(EVENT_LEDGER_JSON, "r", encoding="utf-8") as f:
        events_dict = json.load(f)

    print(f"Loaded Isolated Manifest: {len(df_iso)} clips (SHA-256: {compute_sha256(ISOLATED_MANIFEST)})")
    print(f"Loaded Tone Probe Manifest: {len(df_probe)} clips (SHA-256: {compute_sha256(TONE_PROBE_MANIFEST)})")
    print(f"Loaded Streaming Manifest: {len(df_stream)} sequences (SHA-256: {compute_sha256(STREAMING_MANIFEST)})")

    # 2. Checkpoint Verification
    print("\n--- 2. Verifying Frozen Checkpoints & Loading Models ---")
    models = {}
    checkpoint_hashes = {}
    frontend = LogMelFrontend(sample_rate=16000, n_mels=40, win_length=400, hop_length=160, n_fft=512).to(device)
    frontend.eval()

    for m_name, m_cfg in MODELS_CONFIG.items():
        ckpt_path = m_cfg["path"]
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        actual_hash = compute_sha256(ckpt_path)
        checkpoint_hashes[m_name] = actual_hash
        assert actual_hash == m_cfg["expected_hash"], f"Hash mismatch for {m_name}: {actual_hash} != {m_cfg['expected_hash']}"
        print(f"  [OK] {m_name}: Verified SHA-256: {actual_hash}")
        models[m_name] = load_model(ckpt_path, device)

    # 3. Isolated Evaluation
    print("\n--- 3. Running Isolated Evaluation (480 Clips) ---")
    isolated_results = {}
    for m_name, model in models.items():
        print(f"  Evaluating {m_name}...")
        res = evaluate_isolated(model, frontend, df_iso, device)
        isolated_results[m_name] = res
        print(f"    Macro-F1 (4-cmd): {res['macro_f1']*100:.2f}% | 5-class F1: {res['overall_macro_f1']*100:.2f}% | Phonetic FAR: {res['phonetic_oov_fa_rate']*100:.2f}% | General FAR: {res['general_oov_fa_rate']*100:.2f}%")

    # 4. Tone Probe Evaluation
    print("\n--- 4. Running Tone Probe Diagnostic Evaluation (96 Clips) ---")
    probe_results = {}
    for m_name, model in models.items():
        p_res = evaluate_tone_probes(model, frontend, df_probe, device)
        probe_results[m_name] = p_res
        print(f"    {m_name}: Probe FAR: {p_res['probe_fa_rate']*100:.2f}% ({p_res['probe_fa_count']}/{p_res['probe_total']}) | UCB95: {p_res['probe_ucb95_far']*100:.2f}%")

    # 5. Streaming Evaluation
    print("\n--- 5. Running Streaming Evaluation (50 Sequences) ---")
    streaming_results = {}
    for m_name, model in models.items():
        print(f"  Streaming {m_name}...")
        s_res = evaluate_streaming(model, frontend, df_stream, events_dict, device)
        streaming_results[m_name] = s_res
        print(f"    Exact Seq Acc: {s_res['exact_sequence_accuracy']*100:.2f}% ({s_res['exact_sequence_matches']}/50) | Event F1: {s_res['command_f1']*100:.2f}% | Silence FA/min: {s_res['silence_fa_per_min']} | OOV FA/min: {s_res['oov_fa_per_min']}")

    # 6. Bootstrap Resampling Estimands
    print("\n--- 6. Computing Bootstrap Resampling Estimands (B=10,000) ---")
    y_true_iso = isolated_results["Phase28_D2_H1_Candidate"]["y_true"]
    preds_d2 = isolated_results["Phase28_D2_H1_Candidate"]["y_pred"]

    bootstrap_vs_e1 = compute_bootstrap_estimands(
        df_iso=df_iso,
        preds_d2=preds_d2,
        preds_baseline=isolated_results["Baseline_E1_seed42"]["y_pred"],
        y_true=y_true_iso,
        B=10000,
        seed=42
    )
    print(f"  D2_H1 vs E1:")
    print(f"    Observed Delta (A): {bootstrap_vs_e1['estimand_A_observed_delta']*100:+.2f} pp")
    print(f"    Text Cluster (B): Mean = {bootstrap_vs_e1['estimand_B_text_cluster']['mean_delta']*100:+.2f} pp, 95% CI: [{bootstrap_vs_e1['estimand_B_text_cluster']['ci_95'][0]*100:+.2f}, {bootstrap_vs_e1['estimand_B_text_cluster']['ci_95'][1]*100:+.2f}], p={bootstrap_vs_e1['estimand_B_text_cluster']['p_value']:.4f}")
    print(f"    Instance Cluster (C): Mean = {bootstrap_vs_e1['estimand_C_instance_cluster']['mean_delta']*100:+.2f} pp, 95% CI: [{bootstrap_vs_e1['estimand_C_instance_cluster']['ci_95'][0]*100:+.2f}, {bootstrap_vs_e1['estimand_C_instance_cluster']['ci_95'][1]*100:+.2f}], p={bootstrap_vs_e1['estimand_C_instance_cluster']['p_value']:.4f}")
    print(f"    Clip IID (D): Mean = {bootstrap_vs_e1['estimand_D_clip_iid']['mean_delta']*100:+.2f} pp, 95% CI: [{bootstrap_vs_e1['estimand_D_clip_iid']['ci_95'][0]*100:+.2f}, {bootstrap_vs_e1['estimand_D_clip_iid']['ci_95'][1]*100:+.2f}], p={bootstrap_vs_e1['estimand_D_clip_iid']['p_value']:.4f}")

    bootstrap_vs_v11 = compute_bootstrap_estimands(
        df_iso=df_iso,
        preds_d2=preds_d2,
        preds_baseline=isolated_results["Baseline_v11_seed42"]["y_pred"],
        y_true=y_true_iso,
        B=10000,
        seed=42
    )
    print(f"  D2_H1 vs v11:")
    print(f"    Observed Delta (A): {bootstrap_vs_v11['estimand_A_observed_delta']*100:+.2f} pp")
    print(f"    Text Cluster (B): Mean = {bootstrap_vs_v11['estimand_B_text_cluster']['mean_delta']*100:+.2f} pp, 95% CI: [{bootstrap_vs_v11['estimand_B_text_cluster']['ci_95'][0]*100:+.2f}, {bootstrap_vs_v11['estimand_B_text_cluster']['ci_95'][1]*100:+.2f}], p={bootstrap_vs_v11['estimand_B_text_cluster']['p_value']:.4f}")
    print(f"    Instance Cluster (C): Mean = {bootstrap_vs_v11['estimand_C_instance_cluster']['mean_delta']*100:+.2f} pp, 95% CI: [{bootstrap_vs_v11['estimand_C_instance_cluster']['ci_95'][0]*100:+.2f}, {bootstrap_vs_v11['estimand_C_instance_cluster']['ci_95'][1]*100:+.2f}], p={bootstrap_vs_v11['estimand_C_instance_cluster']['p_value']:.4f}")

    # Remove raw array dumps before serializing JSON
    iso_clean = {}
    for m_name, res in isolated_results.items():
        res_copy = dict(res)
        res_copy.pop("y_true", None)
        res_copy.pop("y_pred", None)
        iso_clean[m_name] = res_copy

    out_master_dict = {
        "metadata": {
            "evaluation_phase": "Phase 43 (Locked Inter-Generator Evaluation)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "speech_generator": "facebook/mms-tts-vie",
            "noise_background": "Karol Piczak ESC-50 (Fold 1 Indoor)",
            "isolated_manifest_sha256": compute_sha256(ISOLATED_MANIFEST),
            "streaming_manifest_sha256": compute_sha256(STREAMING_MANIFEST),
            "event_ledger_sha256": compute_sha256(EVENT_LEDGER_CSV),
            "checkpoint_hashes": checkpoint_hashes,
            "governance_labels": [
                "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
                "NOT_VALIDATED_ON_NATURAL_SPEECH",
                "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
                "NOT_VALIDATED_PENDING_DATA"
            ]
        },
        "isolated_results": iso_clean,
        "tone_probe_results": probe_results,
        "streaming_results": streaming_results,
        "bootstrap_inference": {
            "d2_vs_e1": bootstrap_vs_e1,
            "d2_vs_v11": bootstrap_vs_v11
        }
    }

    out_json = REPORTS_DIR / "PHASE43_EVALUATION_RESULTS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_master_dict, f, indent=2, ensure_ascii=False)
    json_hash = compute_sha256(out_json)
    print(f"\n[OK] Saved PHASE43_EVALUATION_RESULTS.json (SHA-256: {json_hash})")

    elapsed = time.time() - t0
    print(f"[DONE] Locked evaluation completed in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
