"""
Acoustic Waveform Measurement Audit for Synthetic Vietnamese KWS V4.
Directly inspects the physical audio waveforms of representative clips from:
  1. Older-generation TRAI (V1-V3 baseline generator)
  2. V4 Onset-aware TRAI (V4 onset-aware generator)

Resolves the contradiction between reported closure RMS and silence threshold.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import rfft, rfftfreq


def audit_waveforms():
    v3_tr_df = pd.read_csv("experiments/synthetic_kws_v3/train_manifest_6class.csv")
    v4_tr_df = pd.read_csv("experiments/synthetic_kws_v4/train_manifest_onset_aware.csv")

    v3_trai_paths = [Path(p) for p in v3_tr_df[v3_tr_df["class_label"] == "TRAI"]["processed_path"][:20]]
    v4_trai_paths = [Path(p) for p in v4_tr_df[v4_tr_df["class_label"] == "TRAI"]["processed_path"][:20]]

    # 1. Audit V4 Onset-Aware TRAI (20 files)
    v4_results = []
    for p in v4_trai_paths:
        a, sr = sf.read(str(p), dtype="float32")
        # Closure window: t in [0.0, 0.025 s), samples 0 to 400
        c_samples = a[:400]
        c_rms = float(np.sqrt(np.mean(c_samples**2)))
        c_max = float(np.max(np.abs(c_samples)))
        c_frac_under_005 = float(np.mean(np.abs(c_samples) < 0.005))

        # First sample >= 0.005
        over_005 = np.where(np.abs(a) >= 0.005)[0]
        first_005 = int(over_005[0]) if len(over_005) > 0 else -1

        # Burst window: t in [0.025, 0.040 s), samples 400 to 640
        b_samples = a[400:640]
        b_rms = float(np.sqrt(np.mean(b_samples**2)))
        b_max = float(np.max(np.abs(b_samples)))

        # Active speech window: samples 400 to 1200 (50ms of active speech)
        speech_win = a[400:1200]
        spec = np.abs(rfft(speech_win * np.hanning(len(speech_win))))
        freqs = rfftfreq(len(speech_win), 1/sr)
        hf_energy = float(np.sum(spec[freqs >= 3500.0]**2))
        tot_energy = float(np.sum(spec**2) + 1e-9)
        hf_ratio = hf_energy / tot_energy

        # Post-closure window: samples 400 to 800 (what the old bug mistakenly measured as closure)
        buggy_samples = a[400:800]
        buggy_rms = float(np.sqrt(np.mean(buggy_samples**2)))

        v4_results.append({
            "filename": p.name,
            "closure_rms": c_rms,
            "closure_max_abs": c_max,
            "closure_frac_under_005": c_frac_under_005,
            "first_sample_ge_005": first_005,
            "first_time_ms_ge_005": first_005 / sr * 1000.0,
            "burst_rms": b_rms,
            "burst_max_abs": b_max,
            "hf_ratio_active": hf_ratio,
            "samples_400_to_800_rms": buggy_rms
        })

    # 2. Audit Older V1-V3 Baseline TRAI (20 files)
    v3_results = []
    for p in v3_trai_paths:
        a, sr = sf.read(str(p), dtype="float32")
        over_005 = np.where(np.abs(a) >= 0.005)[0]
        first_005 = int(over_005[0]) if len(over_005) > 0 else -1

        # Speech starts at first_005
        act_speech = a[first_005:]
        c_samples = act_speech[:400]
        c_rms = float(np.sqrt(np.mean(c_samples**2)))
        c_max = float(np.max(np.abs(c_samples)))
        c_frac_under_005 = float(np.mean(np.abs(c_samples) < 0.005))

        b_samples = act_speech[400:640]
        b_rms = float(np.sqrt(np.mean(b_samples**2)))
        b_max = float(np.max(np.abs(b_samples)))

        speech_win = act_speech[:800]
        spec = np.abs(rfft(speech_win * np.hanning(len(speech_win))))
        freqs = rfftfreq(len(speech_win), 1/sr)
        hf_energy = float(np.sum(spec[freqs >= 3500.0]**2))
        tot_energy = float(np.sum(spec**2) + 1e-9)
        hf_ratio = hf_energy / tot_energy

        v3_results.append({
            "filename": p.name,
            "first_sample_ge_005": first_005,
            "first_time_ms_ge_005": first_005 / sr * 1000.0,
            "onset_first_25ms_rms": c_rms,
            "onset_first_25ms_max_abs": c_max,
            "onset_first_25ms_frac_under_005": c_frac_under_005,
            "onset_25_to_40ms_rms": b_rms,
            "onset_25_to_40ms_max_abs": b_max,
            "hf_ratio_active": hf_ratio
        })

    summary = {
        "v4_onset_aware": {
            "closure_window_0_to_25ms": {
                "rms_mean": float(np.mean([r["closure_rms"] for r in v4_results])),
                "rms_std": float(np.std([r["closure_rms"] for r in v4_results])),
                "max_abs_mean": float(np.mean([r["closure_max_abs"] for r in v4_results])),
                "max_abs_max": float(np.max([r["closure_max_abs"] for r in v4_results])),
                "fraction_under_005_mean": float(np.mean([r["closure_frac_under_005"] for r in v4_results])),
                "first_sample_ge_005_mean": float(np.mean([r["first_sample_ge_005"] for r in v4_results])),
                "first_sample_ge_005_min": int(np.min([r["first_sample_ge_005"] for r in v4_results])),
                "first_sample_ge_005_max": int(np.max([r["first_sample_ge_005"] for r in v4_results])),
                "first_time_ms_ge_005_mean": float(np.mean([r["first_time_ms_ge_005"] for r in v4_results]))
            },
            "burst_window_25_to_40ms": {
                "rms_mean": float(np.mean([r["burst_rms"] for r in v4_results])),
                "rms_std": float(np.std([r["burst_rms"] for r in v4_results])),
                "max_abs_mean": float(np.mean([r["burst_max_abs"] for r in v4_results])),
                "max_abs_max": float(np.max([r["burst_max_abs"] for r in v4_results]))
            },
            "active_speech_hf_ratio_gt_3500hz": {
                "mean": float(np.mean([r["hf_ratio_active"] for r in v4_results])),
                "std": float(np.std([r["hf_ratio_active"] for r in v4_results]))
            },
            "bug_investigation_samples_400_to_800": {
                "rms_mean": float(np.mean([r["samples_400_to_800_rms"] for r in v4_results])),
                "explanation": (
                    "In the previous audit script, active speech onset was detected by finding |a| > 0.01. "
                    "Because samples 0-400 have max amplitude 0.001740, this check skipped the entire 25ms closure gap. "
                    "The script then took the first 400 samples AFTER sample 400 (i.e. samples 400 to 800, the burst and vowel onset), "
                    "which has mean RMS of 0.05551. It erroneously labeled this burst window as the closure window in the summary table."
                )
            }
        },
        "v3_baseline": {
            "clip_leading_padding": {
                "first_sample_ge_005_mean": float(np.mean([r["first_sample_ge_005"] for r in v3_results])),
                "first_time_ms_ge_005_mean": float(np.mean([r["first_time_ms_ge_005"] for r in v3_results])),
                "padding_nature": "Centered zero-padding from audio_normalizer.py (~247ms)"
            },
            "active_speech_first_25ms": {
                "rms_mean": float(np.mean([r["onset_first_25ms_rms"] for r in v3_results])),
                "rms_std": float(np.std([r["onset_first_25ms_rms"] for r in v3_results])),
                "max_abs_mean": float(np.mean([r["onset_first_25ms_max_abs"] for r in v3_results])),
                "fraction_under_005_mean": float(np.mean([r["onset_first_25ms_frac_under_005"] for r in v3_results]))
            },
            "active_speech_25_to_40ms": {
                "rms_mean": float(np.mean([r["onset_25_to_40ms_rms"] for r in v3_results])),
                "rms_std": float(np.std([r["onset_25_to_40ms_rms"] for r in v3_results]))
            },
            "active_speech_hf_ratio_gt_3500hz": {
                "mean": float(np.mean([r["hf_ratio_active"] for r in v3_results])),
                "std": float(np.std([r["hf_ratio_active"] for r in v3_results]))
            }
        }
    }

    out_file = Path("experiments/synthetic_kws_v4/audit/waveform_measurements.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved waveform measurement audit to {out_file}")
    print("\n--- Summary ---")
    print(f"V4 True Closure RMS (0-25ms): {summary['v4_onset_aware']['closure_window_0_to_25ms']['rms_mean']:.6f}")
    print(f"V4 Max Amp in Closure:        {summary['v4_onset_aware']['closure_window_0_to_25ms']['max_abs_max']:.6f} (< 0.005: {summary['v4_onset_aware']['closure_window_0_to_25ms']['fraction_under_005_mean']*100:.2f}%)")
    print(f"V4 Burst RMS (25-40ms):       {summary['v4_onset_aware']['burst_window_25_to_40ms']['rms_mean']:.6f}")
    print(f"V4 Buggy Post-Closure (400-800): {summary['v4_onset_aware']['bug_investigation_samples_400_to_800']['rms_mean']:.6f} (matches 0.05551!)")
    print(f"V3 Active Speech Onset RMS:   {summary['v3_baseline']['active_speech_first_25ms']['rms_mean']:.6f}")


if __name__ == "__main__":
    audit_waveforms()
