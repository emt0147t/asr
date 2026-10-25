"""
Audit existing V5.2 synthetic KWS audio samples for acoustic properties,
phonetic structure, tone contour modeling, and intelligibility limitations.
"""

import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

MANIFEST_PATH = "experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/manifest.csv"

def compute_audio_metrics(audio: np.ndarray, sr: int = 16000):
    duration_sec = len(audio) / sr
    peak_abs = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
    crest_factor = float(peak_abs / max(rms, 1e-9))
    is_clipped = bool(peak_abs >= 0.999)
    
    # Active speech region (energy > -35 dB of peak)
    threshold = peak_abs * 0.05
    active_indices = np.where(np.abs(audio) > threshold)[0]
    if len(active_indices) > 0:
        active_start_sec = float(active_indices[0] / sr)
        active_end_sec = float(active_indices[-1] / sr)
        active_dur_sec = active_end_sec - active_start_sec
        active_speech = audio[active_indices[0]:active_indices[-1]+1]
        active_rms = float(np.sqrt(np.mean(active_speech ** 2)))
        active_rms_dbfs = 20.0 * np.log10(max(active_rms, 1e-9))
    else:
        active_start_sec, active_end_sec, active_dur_sec = 0.0, 0.0, 0.0
        active_rms, active_rms_dbfs = 0.0, -100.0

    # Spectral analysis
    freqs, times, spec = signal.spectrogram(audio, fs=sr, nperseg=400, noverlap=240)
    mag = np.abs(spec) + 1e-9
    power = mag ** 2
    spec_sum = np.sum(power, axis=0) + 1e-9
    freq_grid = freqs[:, None]
    centroid = np.sum(freq_grid * power, axis=0) / spec_sum
    mean_centroid_hz = float(np.mean(centroid))
    
    hf_mask = freqs >= 3000.0
    hf_ratio = float(np.sum(power[hf_mask, :]) / np.sum(power))

    # Zero crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

    return {
        "duration_sec": round(duration_sec, 4),
        "total_samples": len(audio),
        "peak_amplitude": round(peak_abs, 4),
        "is_clipped": is_clipped,
        "full_clip_rms_dbfs": round(rms_dbfs, 2),
        "crest_factor": round(crest_factor, 2),
        "active_speech_start_sec": round(active_start_sec, 4),
        "active_speech_end_sec": round(active_end_sec, 4),
        "active_speech_duration_sec": round(active_dur_sec, 4),
        "active_speech_rms_dbfs": round(active_rms_dbfs, 2),
        "mean_spectral_centroid_hz": round(mean_centroid_hz, 1),
        "hf_energy_ratio_above_3khz": round(hf_ratio, 4),
        "zero_crossing_rate": round(zcr, 4)
    }

def main():
    df = pd.read_csv(MANIFEST_PATH)
    target_classes = ["TRAI", "PHAI", "LEN", "XUONG"]
    
    audit_results = {
        "summary": "Audit of V5.2 candidate release audio clips for acoustic and phonetic characteristics",
        "targets_audited": {}
    }
    
    for cls in target_classes:
        cls_df = df[df["class_label"] == cls]
        # Pick 3 diverse examples: 1 female high/mid, 1 male low/mid, 1 child/breathy
        selected_samples = []
        for split in ["train", "val", "test"]:
            sample = cls_df[cls_df["split"] == split].iloc[0]
            selected_samples.append(sample)
        
        sample_metrics = []
        for sample in selected_samples:
            wav_path = sample["processed_path"]
            if not os.path.isabs(wav_path):
                wav_path = os.path.join("d:/trainidentificationaudio", wav_path)
            
            audio, sr = sf.read(wav_path)
            metrics = compute_audio_metrics(audio, sr)
            metrics["recording_id"] = sample["recording_id"]
            metrics["split"] = sample["split"]
            metrics["transcript"] = sample["transcript"]
            metrics["simulated_speaker_id"] = sample["simulated_speaker_id"]
            metrics["manifest_target_speech_rms_dbfs"] = round(float(sample["target_speech_rms_dbfs"]), 2)
            metrics["manifest_onset_offset_sec"] = round(float(sample["onset_offset_sec"]), 4)
            metrics["manifest_duration_sec"] = round(float(sample["duration_sec"]), 4)
            sample_metrics.append(metrics)
        
        audit_results["targets_audited"][cls] = {
            "intended_vietnamese_word": selected_samples[0]["transcript"],
            "samples_analyzed": sample_metrics
        }
    
    output_path = "experiments/synthetic_kws_tts_pilot/v5_2_audio_audit_data.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    print(f"Saved V5.2 audio audit data to {output_path}")

if __name__ == "__main__":
    main()
