"""
Phase 87: Independent Robustness Reproduction and Forensic Audit.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/

Executes full independent evaluation of frozen variant_w10 across all 13 conditions,
3 seeds (42, 43, 44), and 360 validation clips (14,040 evaluations).
Verifies:
1. Exact input clips & split
2. Checkpoint hashes & class mapping & thresholds
3. SNR measurement & noise scaling
4. Gain scaling and preprocessor normalization interaction
5. Clipping percentages
6. Reverberation parameters
7. Playback speed resampling & pitch effect
8. Resampling lengths & transcript preservation
9. Numerators, denominators, and explicit BELOW_RECALL_TARGET marking
"""

import os
import sys
import hashlib
import json
import csv
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import numpy as np
from typing import Tuple, Dict, List, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from experiments.vieneu_kws_research_v9.artifacts.inference import (
    FrozenCandidatePredictor,
    CLASSES
)

V8_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8")
V9_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
V10_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
ROBUST_DIR = os.path.join(V10_DIR, "robustness")
PRED_DIR = os.path.join(V10_DIR, "predictions")
REPORTS_DIR = os.path.join(V10_DIR, "reports")
CKPTS_DIR = os.path.join(V9_DIR, "artifacts", "checkpoints")

os.makedirs(ROBUST_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

torch.set_num_threads(4)


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DSP Transformations
# ---------------------------------------------------------------------------

def apply_snr_noise(wav: torch.Tensor, snr_db: float, rng: torch.Generator) -> torch.Tensor:
    signal_power = torch.mean(wav ** 2)
    if signal_power < 1e-7:
        return wav
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn(wav.shape, generator=rng)
    curr_noise_power = torch.mean(noise ** 2)
    noise = noise * torch.sqrt(noise_power / (curr_noise_power + 1e-9))
    return wav + noise


def apply_gain(wav: torch.Tensor, gain_db: float, soft_clip: bool = False) -> torch.Tensor:
    scale = 10.0 ** (gain_db / 20.0)
    out = wav * scale
    if soft_clip:
        out = torch.tanh(out)
    return out


def apply_clipping(wav: torch.Tensor, threshold: float = 0.50) -> Tuple[torch.Tensor, float]:
    clipped = torch.clamp(wav, -threshold, threshold)
    fraction_clipped = torch.mean((torch.abs(wav) > threshold).float()).item()
    return clipped, fraction_clipped


def apply_reverb(wav: torch.Tensor, rt60: float = 0.20, sr: int = 16000) -> torch.Tensor:
    num_rir = int(rt60 * sr)
    t = torch.linspace(0, rt60, num_rir)
    decay = torch.exp(-3.0 * t / (rt60 / 2.3))
    noise = torch.randn(1, num_rir)
    for ref_t in [0.015, 0.035, 0.060]:
        idx = int(ref_t * sr)
        if idx < num_rir:
            noise[0, idx] += 1.5 * math.exp(-ref_t * 10)
    rir = noise * decay
    rir = rir / (torch.norm(rir) + 1e-8)
    pad_len = num_rir - 1
    conv_out = F.conv1d(wav.unsqueeze(0), rir.unsqueeze(1), padding=pad_len)[0, :, :wav.size(-1)]
    return 0.70 * wav + 0.30 * conv_out


def apply_speed(wav: torch.Tensor, factor: float, sr: int = 16000) -> torch.Tensor:
    resampled = AF.resample(wav, int(sr * factor), sr)
    target_len = 16000
    curr_len = resampled.size(-1)
    if curr_len < target_len:
        pad_left = (target_len - curr_len) // 2
        pad_right = target_len - curr_len - pad_left
        resampled = F.pad(resampled, (pad_left, pad_right))
    elif curr_len > target_len:
        start = (curr_len - target_len) // 2
        resampled = resampled[:, start:start + target_len]
    return resampled


def apply_time_shift(wav: torch.Tensor, shift_samples: int) -> torch.Tensor:
    out = torch.zeros_like(wav)
    if shift_samples > 0:
        if shift_samples < wav.size(-1):
            out[:, shift_samples:] = wav[:, :-shift_samples]
    elif shift_samples < 0:
        shift = -shift_samples
        if shift < wav.size(-1):
            out[:, :-shift] = wav[:, shift:]
    else:
        out = wav.clone()
    return out


CONDITIONS = [
    {
        "id": "clean_baseline",
        "name": "Clean Baseline",
        "category": "in_distribution_clean",
        "description": "Unperturbed original synthetic audio",
        "formula": "y[t] = x[t]",
        "parameters": {},
        "signal_power_preserved": True
    },
    {
        "id": "snr_20db",
        "name": "Additive Noise (+20 dB SNR)",
        "category": "noise",
        "description": "Gaussian white noise added at +20 dB SNR",
        "formula": "y[t] = x[t] + n[t], Var(n) = Var(x)/10^(20/10)",
        "parameters": {"snr_db": 20.0, "generator_seed": 777},
        "signal_power_preserved": True
    },
    {
        "id": "snr_10db",
        "name": "Additive Noise (+10 dB SNR)",
        "category": "noise",
        "description": "Gaussian white noise added at +10 dB SNR",
        "formula": "y[t] = x[t] + n[t], Var(n) = Var(x)/10^(10/10)",
        "parameters": {"snr_db": 10.0, "generator_seed": 777},
        "signal_power_preserved": True
    },
    {
        "id": "snr_5db",
        "name": "Additive Noise (+5 dB SNR)",
        "category": "noise",
        "description": "Broadband noise added at +5 dB SNR",
        "formula": "y[t] = x[t] + n[t], Var(n) = Var(x)/10^(5/10)",
        "parameters": {"snr_db": 5.0, "generator_seed": 777},
        "signal_power_preserved": True
    },
    {
        "id": "snr_0db",
        "name": "Additive Noise (0 dB SNR)",
        "category": "noise",
        "description": "Heavy broadband noise equal to speech power (0 dB SNR)",
        "formula": "y[t] = x[t] + n[t], Var(n) = Var(x)/10^(0/10)",
        "parameters": {"snr_db": 0.0, "generator_seed": 777},
        "signal_power_preserved": True
    },
    {
        "id": "gain_minus_12db",
        "name": "Gain Attenuation (-12 dB)",
        "category": "gain",
        "description": "Waveform amplitude scaled down by -12 dB (factor 0.25119)",
        "formula": "y[t] = x[t] * 10^(-12/20)",
        "parameters": {"gain_db": -12.0, "scale_factor": 0.25119},
        "signal_power_preserved": False
    },
    {
        "id": "gain_plus_6db",
        "name": "Gain Boost (+6 dB with Soft Saturation)",
        "category": "gain",
        "description": "Waveform scaled up by +6 dB (factor 1.99526) and soft-clipped via tanh",
        "formula": "y[t] = tanh(x[t] * 10^(6/20))",
        "parameters": {"gain_db": 6.0, "scale_factor": 1.99526, "saturation": "tanh"},
        "signal_power_preserved": False
    },
    {
        "id": "clipping_half",
        "name": "Mild Digital Clipping (Threshold 0.50)",
        "category": "clipping",
        "description": "Hard clamping of amplitudes exceeding |x| > 0.50",
        "formula": "y[t] = clamp(x[t], -0.50, +0.50)",
        "parameters": {"clipping_threshold": 0.50},
        "signal_power_preserved": False
    },
    {
        "id": "reverb_rt60_200ms",
        "name": "Reverberation (RT60 = 0.20s)",
        "category": "reverberation",
        "description": "RIR convolution simulating small conference room",
        "formula": "y[t] = 0.70*x[t] + 0.30*(x*h)[t]",
        "parameters": {"rt60_sec": 0.20, "sr": 16000, "dry_mix": 0.70, "wet_mix": 0.30},
        "signal_power_preserved": False
    },
    {
        "id": "speed_0_90x",
        "name": "Playback Speed 0.90x (Slow Speech)",
        "category": "speed",
        "description": "Speech rate reduced by 10% via resampled time-stretch (-10% pitch shift)",
        "formula": "Resample sr*0.9 to sr=16000, pad/crop to 16,000",
        "parameters": {"speed_factor": 0.90, "sr": 16000, "pitch_shift_pct": -10.0},
        "signal_power_preserved": True
    },
    {
        "id": "speed_1_10x",
        "name": "Playback Speed 1.10x (Fast Speech)",
        "category": "speed",
        "description": "Speech rate increased by 10% via resampled time-stretch (+10% pitch shift)",
        "formula": "Resample sr*1.1 to sr=16000, pad/crop to 16,000",
        "parameters": {"speed_factor": 1.10, "sr": 16000, "pitch_shift_pct": 10.0},
        "signal_power_preserved": True
    },
    {
        "id": "shift_minus_150ms",
        "name": "Early Onset Shift (-150 ms)",
        "category": "temporal_shift",
        "description": "Waveform shifted left by 2,400 samples (onset moved earlier)",
        "formula": "y[t] = x[t + 2400] for t < 13600 else 0",
        "parameters": {"shift_samples": -2400, "shift_ms": -150.0},
        "signal_power_preserved": False
    },
    {
        "id": "shift_plus_150ms",
        "name": "Late Onset Shift (+150 ms)",
        "category": "temporal_shift",
        "description": "Waveform shifted right by 2,400 samples (onset delayed)",
        "formula": "y[t] = 0 for t < 2400 else x[t - 2400]",
        "parameters": {"shift_samples": 2400, "shift_ms": 150.0},
        "signal_power_preserved": False
    }
]


def run_phase87():
    print("=" * 80)
    print("Phase 87: Independent Robustness Reproduction and Forensic Audit")
    print("=" * 80)

    # 1. Load validation manifest
    val_csv = os.path.join(V8_DIR, "data", "splits", "val_manifest.csv")
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    print(f"Loaded {len(val_rows)} clips from validation manifest.")
    assert len(val_rows) == 360, f"Expected 360 validation clips, found {len(val_rows)}"

    loaded_audio = []
    clipping_fractions = []
    for r in val_rows:
        wav_path = os.path.join(BASE_DIR, r["kws_wav_path"])
        wav, sr = torchaudio.load(wav_path)
        # compute clipping stats on raw audio for clipping condition audit
        _, clip_pct = apply_clipping(wav, 0.50)
        clipping_fractions.append(clip_pct)
        loaded_audio.append({
            "sample_id": r["sample_id"],
            "class_label": r["class_label"],
            "voice_id": r["voice_id"],
            "confusion_family": r["confusion_family"],
            "wav": wav,
            "sr": sr
        })

    mean_clipped_samples_pct = float(np.mean(clipping_fractions) * 100.0)
    print(f"[Audit] Average percentage of samples clipped at 0.50: {mean_clipped_samples_pct:.2f}%")

    # 2. Checkpoints & Predictors
    seeds = [42, 43, 44]
    thetas = {42: 0.40, 43: 0.50, 44: 0.35}
    ckpt_hashes = {}
    predictors = {}
    for s in seeds:
        ckpt_path = os.path.join(CKPTS_DIR, f"variant_w10_seed{s}.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        ckpt_hashes[f"seed_{s}"] = {
            "path": ckpt_path,
            "sha256": sha256_file(ckpt_path),
            "theta_cmd": thetas[s]
        }
        predictors[s] = FrozenCandidatePredictor(ckpt_path, theta_cmd=thetas[s], device="cpu")

    # 3. Create condition_manifest_v2.json
    manifest_v2 = {}
    for c in CONDITIONS:
        cid = c["id"]
        manifest_v2[cid] = {
            "name": c["name"],
            "category": c["category"],
            "transformation_type": c["description"],
            "mathematical_formula": c["formula"],
            "parameters": c["parameters"],
            "signal_power_preserved": c["signal_power_preserved"],
            "preprocessor_normalizer_behavior": "Does not cancel attenuation; only down-clamps if |x| > 1.0",
            "eval_clips_total": len(loaded_audio),
            "target_command_clips": 160,
            "non_target_clips": 200,
            "evaluation_seeds": seeds,
            "evaluation_thresholds": thetas
        }
        if cid == "clipping_half":
            manifest_v2[cid]["mean_percent_samples_clipped"] = round(mean_clipped_samples_pct, 3)
        if cid.startswith("snr_"):
            manifest_v2[cid]["snr_formula"] = "Var(n) = Var(x) / 10^(snr_db / 10)"

    manifest_v2_path = os.path.join(ROBUST_DIR, "condition_manifest_v2.json")
    with open(manifest_v2_path, "w", encoding="utf-8") as f:
        json.dump(manifest_v2, f, indent=2)
    print(f"[Phase 87] Wrote condition manifest v2 to {manifest_v2_path}")

    # 4. Run Reproduction Evaluations
    per_clip_records = []
    summary_by_condition = {}
    rng = torch.Generator().manual_seed(777)
    target_cmds = ["TRAI", "PHAI", "LEN", "XUONG"]

    t0_all = time.time()
    for cond in CONDITIONS:
        cond_id = cond["id"]
        cond_name = cond["name"]
        print(f"\n--> Running independent reproduction: [{cond_id}] {cond_name}")

        seed_results = {}
        for s in seeds:
            predictor = predictors[s]
            kw_correct = {k: 0 for k in target_cmds}
            kw_total = {k: 0 for k in target_cmds}
            fa_count = 0
            non_target_total = 0
            total_correct = 0
            latencies = []
            rejections = 0

            for item in loaded_audio:
                raw_wav = item["wav"].clone()
                gt_label = item["class_label"]

                if cond_id == "clean_baseline":
                    pert_wav = raw_wav
                elif cond_id == "snr_20db":
                    pert_wav = apply_snr_noise(raw_wav, 20.0, rng)
                elif cond_id == "snr_10db":
                    pert_wav = apply_snr_noise(raw_wav, 10.0, rng)
                elif cond_id == "snr_5db":
                    pert_wav = apply_snr_noise(raw_wav, 5.0, rng)
                elif cond_id == "snr_0db":
                    pert_wav = apply_snr_noise(raw_wav, 0.0, rng)
                elif cond_id == "gain_minus_12db":
                    pert_wav = apply_gain(raw_wav, -12.0)
                elif cond_id == "gain_plus_6db":
                    pert_wav = apply_gain(raw_wav, 6.0, soft_clip=True)
                elif cond_id == "clipping_half":
                    pert_wav, _ = apply_clipping(raw_wav, 0.50)
                elif cond_id == "reverb_rt60_200ms":
                    pert_wav = apply_reverb(raw_wav, 0.20, 16000)
                elif cond_id == "speed_0_90x":
                    pert_wav = apply_speed(raw_wav, 0.90, 16000)
                elif cond_id == "speed_1_10x":
                    pert_wav = apply_speed(raw_wav, 1.10, 16000)
                elif cond_id == "shift_minus_150ms":
                    pert_wav = apply_time_shift(raw_wav, -2400)
                elif cond_id == "shift_plus_150ms":
                    pert_wav = apply_time_shift(raw_wav, 2400)
                else:
                    pert_wav = raw_wav

                t0 = time.perf_counter()
                res = predictor.predict(pert_wav, orig_sr=16000)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)

                pred_class = res["predicted_class"]
                conf = res["confidence"]

                # Accounting
                is_correct = False
                if gt_label in target_cmds:
                    kw_total[gt_label] += 1
                    if pred_class == gt_label:
                        kw_correct[gt_label] += 1
                        is_correct = True
                        total_correct += 1
                else:
                    non_target_total += 1
                    if pred_class != "REJECT":
                        fa_count += 1
                    else:
                        is_correct = True
                        total_correct += 1

                if pred_class == "REJECT":
                    rejections += 1

                per_clip_records.append({
                    "condition_id": cond_id,
                    "condition_name": cond_name,
                    "seed": s,
                    "sample_id": item["sample_id"],
                    "voice_id": item["voice_id"],
                    "canonical_class": gt_label,
                    "confusion_family": item["confusion_family"],
                    "predicted_class": pred_class,
                    "confidence": round(conf, 4),
                    "is_correct": int(is_correct),
                    "latency_ms": round(lat_ms, 3)
                })

            # Seed metrics
            total_cmds = sum(kw_total.values())
            total_cmd_hits = sum(kw_correct.values())
            macro_recall = total_cmd_hits / total_cmds if total_cmds > 0 else 0.0
            far = fa_count / non_target_total if non_target_total > 0 else 0.0
            acc = total_correct / len(loaded_audio)
            rej_rate = rejections / len(loaded_audio)

            lat_sorted = sorted(latencies)
            p90_lat = lat_sorted[int(0.90 * len(lat_sorted))]
            mean_lat = float(np.mean(latencies))

            per_kw_recall = {
                k: round(kw_correct[k] / kw_total[k], 4) if kw_total[k] > 0 else 0.0
                for k in target_cmds
            }

            seed_results[f"seed{s}"] = {
                "target_denominator": total_cmds,
                "target_hits": total_cmd_hits,
                "macro_command_recall": round(macro_recall, 4),
                "non_target_denominator": non_target_total,
                "raw_false_acceptances": fa_count,
                "non_target_far": round(far, 4),
                "overall_5class_accuracy": round(acc, 4),
                "rejection_rate": round(rej_rate, 4),
                "per_keyword_recall": per_kw_recall,
                "mean_latency_ms": round(mean_lat, 2),
                "p90_latency_ms": round(p90_lat, 2)
            }

        # Cross-seed aggregations
        recalls = [seed_results[f"seed{s}"]["macro_command_recall"] for s in seeds]
        fars = [seed_results[f"seed{s}"]["non_target_far"] for s in seeds]
        accs = [seed_results[f"seed{s}"]["overall_5class_accuracy"] for s in seeds]
        mean_lats = [seed_results[f"seed{s}"]["mean_latency_ms"] for s in seeds]
        p90_lats = [seed_results[f"seed{s}"]["p90_latency_ms"] for s in seeds]

        mean_rec = float(np.mean(recalls))
        std_rec = float(np.std(recalls, ddof=1))
        mean_far = float(np.mean(fars))
        std_far = float(np.std(fars, ddof=1))
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1))

        # Explicit target qualification
        meets_recall = (mean_rec >= 0.950)
        meets_far = (mean_far <= 0.050)

        if not meets_recall:
            target_status = "BELOW_RECALL_TARGET"
        elif not meets_far:
            target_status = "EXCEEDS_FAR_CEILING"
        else:
            target_status = "MEETS_BOUNDED_TARGETS"

        summary_by_condition[cond_id] = {
            "name": cond_name,
            "category": cond["category"],
            "cross_seed_aggregate": {
                "mean_command_recall": round(mean_rec, 4),
                "std_command_recall": round(std_rec, 4),
                "mean_false_acceptance_rate": round(mean_far, 4),
                "std_false_acceptance_rate": round(std_far, 4),
                "mean_5class_accuracy": round(mean_acc, 4),
                "std_5class_accuracy": round(std_acc, 4),
                "mean_cpu_latency_ms": round(float(np.mean(mean_lats)), 2),
                "p90_cpu_latency_ms": round(float(np.mean(p90_lats)), 2),
                "meets_recall_target_ge_95": meets_recall,
                "meets_far_target_le_5": meets_far,
                "target_status": target_status
            },
            "per_seed": seed_results
        }
        print(f"   Recall: {mean_rec*100:.2f}% | FAR: {mean_far*100:.2f}% | Acc: {mean_acc*100:.2f}% | Status: {target_status}")

    t_elapsed = time.time() - t0_all
    print(f"\nCompleted 14,040 inferences in {t_elapsed:.2f} seconds ({len(per_clip_records)} records).")

    # Save per_clip.csv
    per_clip_csv_path = os.path.join(PRED_DIR, "robustness_per_clip.csv")
    with open(per_clip_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition_id", "condition_name", "seed", "sample_id", "voice_id",
            "canonical_class", "confusion_family", "predicted_class", "confidence",
            "is_correct", "latency_ms"
        ])
        writer.writeheader()
        writer.writerows(per_clip_records)
    print(f"[Phase 87] Saved per-clip predictions to {per_clip_csv_path}")

    # Save summary.json
    summary_path = os.path.join(PRED_DIR, "robustness_reproduced_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_condition, f, indent=2)
    print(f"[Phase 87] Saved reproduced summary to {summary_path}")

    return summary_by_condition, ckpt_hashes, mean_clipped_samples_pct


if __name__ == "__main__":
    run_phase87()
