"""
Detailed acoustic and phonetic comparison between V5.2 parametric audio
and the new Neural TTS pilot clips for Vietnamese KWS target words.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import librosa

sys.stdout.reconfigure(encoding="utf-8")

V5_2_MANIFEST = "experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/manifest.csv"
TTS_MANIFEST = "experiments/synthetic_kws_tts_pilot/tts_pilot_manifest.csv"

def extract_detailed_features(wav_path: str, sr_target: int = 16000):
    audio, sr = sf.read(wav_path)
    if sr != sr_target:
        audio = signal.resample_poly(audio, sr_target, sr).astype(np.float32)
        sr = sr_target
    
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
    
    # Active speech region
    thresh = peak * 0.03
    active_idx = np.where(np.abs(audio) > thresh)[0]
    if len(active_idx) > 0:
        active_dur = (active_idx[-1] - active_idx[0]) / sr
        speech_seg = audio[active_idx[0]:active_idx[-1]+1]
        speech_rms = float(np.sqrt(np.mean(speech_seg ** 2)))
        speech_rms_dbfs = 20.0 * np.log10(max(speech_rms, 1e-9))
    else:
        active_dur, speech_rms_dbfs = 0.0, -100.0

    # Spectral features
    freqs, times, spec = signal.spectrogram(audio, fs=sr, nperseg=400, noverlap=240)
    power = np.abs(spec) ** 2
    spec_sum = np.sum(power, axis=0) + 1e-9
    centroid = np.sum(freqs[:, None] * power, axis=0) / spec_sum
    mean_centroid = float(np.mean(centroid))
    hf_energy = float(np.sum(power[freqs >= 3000, :]) / np.sum(power))

    # Pitch tracking using pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=75, fmax=400, sr=sr)
        valid_f0 = f0[voiced_flag]
        if len(valid_f0) >= 3:
            f0_start = float(valid_f0[0])
            f0_mid = float(valid_f0[len(valid_f0)//2])
            f0_end = float(valid_f0[-1])
            f0_min = float(np.min(valid_f0))
            f0_max = float(np.max(valid_f0))
            f0_range = f0_max - f0_min
            f0_slope = (f0_end - f0_start) / max(0.01, len(valid_f0) * (160 / sr))
        else:
            f0_start, f0_mid, f0_end, f0_min, f0_max, f0_range, f0_slope = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception:
        f0_start, f0_mid, f0_end, f0_min, f0_max, f0_range, f0_slope = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "duration_sec": len(audio) / sr,
        "peak_amplitude": round(peak, 4),
        "full_clip_rms_dbfs": round(rms_dbfs, 2),
        "speech_rms_dbfs": round(speech_rms_dbfs, 2),
        "active_duration_sec": round(active_dur, 4),
        "spectral_centroid_hz": round(mean_centroid, 1),
        "hf_energy_ratio_above_3k": round(hf_energy, 4),
        "f0_start_hz": round(f0_start, 1),
        "f0_mid_hz": round(f0_mid, 1),
        "f0_end_hz": round(f0_end, 1),
        "f0_min_hz": round(f0_min, 1),
        "f0_max_hz": round(f0_max, 1),
        "f0_range_hz": round(f0_range, 1),
        "f0_slope_hz_per_sec": round(f0_slope, 1)
    }

def main():
    df_v5 = pd.read_csv(V5_2_MANIFEST)
    df_tts = pd.read_csv(TTS_MANIFEST)

    words = ["TRAI", "PHAI", "LEN", "XUONG"]
    comparison = {}

    for w in words:
        # V5.2 representative test sample
        v5_sample = df_v5[(df_v5["class_label"] == w) & (df_v5["split"] == "test")].iloc[0]
        v5_feats = extract_detailed_features(v5_sample["processed_path"])

        # TTS representative female and male samples
        tts_female = df_tts[(df_tts["target_keyword"] == w) & (df_tts["voice_gender"] == "female") & (df_tts["rate_parameter"] == "+0%")].iloc[0]
        tts_female_feats = extract_detailed_features(tts_female["kws_wav_relative_path"])

        tts_male = df_tts[(df_tts["target_keyword"] == w) & (df_tts["voice_gender"] == "male") & (df_tts["rate_parameter"] == "+0%")].iloc[0]
        tts_male_feats = extract_detailed_features(tts_male["kws_wav_relative_path"])

        comparison[w] = {
            "v5_2_parametric_sample": {
                "recording_id": v5_sample["recording_id"],
                "speaker": v5_sample["simulated_speaker_id"],
                "features": v5_feats
            },
            "tts_neural_female_sample": {
                "clip_id": tts_female["clip_id"],
                "voice": tts_female["voice_id"],
                "features": tts_female_feats
            },
            "tts_neural_male_sample": {
                "clip_id": tts_male["clip_id"],
                "voice": tts_male["voice_id"],
                "features": tts_male_feats
            }
        }

    out_json = "experiments/synthetic_kws_tts_pilot/acoustic_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Comparison data saved to {out_json}")

if __name__ == "__main__":
    main()
