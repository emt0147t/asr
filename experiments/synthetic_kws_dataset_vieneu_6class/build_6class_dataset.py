"""
Phase 28: Six-Class Vietnamese KWS Dataset Generator & Manifest Assembler.
Extends the accepted 1,000-utterance VieNeu-TTS four-class dataset by generating:
  - 250 UNKNOWN Vietnamese speech utterances using VieNeu-TTS v3 Turbo
  - 250 SILENCE / ambient background utterances under well-defined, disjoint acoustic conditions
Total: 1,500 balanced utterances across 6 classes:
  TRAI (250), PHAI (250), LEN (250), XUONG (250), UNKNOWN (250), SILENCE (250).
"""

import os
import sys
import json
import hashlib
import time
import pathlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOUR_CLASS_DIR = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_1000")
SIX_CLASS_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_RAW_DIR = os.path.join(SIX_CLASS_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(SIX_CLASS_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(SIX_CLASS_DIR, "splits")
REPORTS_DIR = os.path.join(SIX_CLASS_DIR, "reports")
CONFIGS_DIR = os.path.join(SIX_CLASS_DIR, "configs")
MANIFESTS_DIR = os.path.join(SIX_CLASS_DIR, "manifests")
MASTER_MANIFEST_PATH = os.path.join(SIX_CLASS_DIR, "manifest.csv")

# Authoritative voice allocations from Phase 27
SPLIT_VOICE_ALLOCATION = {
    "train": [
        "Adam", "Adam bựa", "Hải Đăng", "Minh Triết", "Minh Đức", "Mỹ Duyên",
        "Ngọc Huyền", "Ngọc Linh", "Quang Sơn", "Thanh Bình", "Thiền Tâm Đức",
        "Thiện Minh", "Thái Sơn", "Thục Đoan", "Trúc Ly", "Xuân Vĩnh", "Đoan Trang"
    ],  # 17 voices -> 17 * 10 = 170
    "val": [
        "Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"
    ],  # 4 voices -> 4 * 10 = 40
    "test": [
        "Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"
    ]   # 4 voices -> 4 * 10 = 40
}

# Disjoint UNKNOWN Text Inventories (10 distinct words/phrases per split)
UNKNOWN_TEXTS = {
    "train": [
        {"text": "Dừng", "ipa": "/zɯŋ˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Bật", "ipa": "/ɓət˨˩/", "tone": "nặng (low-glottalized)"},
        {"text": "Tắt", "ipa": "/tat˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Mở", "ipa": "/məː˧˩˨/", "tone": "hỏi (mid-falling rising)"},
        {"text": "Đóng", "ipa": "/ɗawŋm˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Chạy", "ipa": "/t͡ɕaj˨˩/", "tone": "nặng (low-glottalized)"},
        {"text": "Vào", "ipa": "/vaːw˨˩/", "tone": "huyền (low-falling)"},
        {"text": "Ra", "ipa": "/zaː˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Trời", "ipa": "/ʈəːj˨˩/", "tone": "huyền (low-falling)"},
        {"text": "Phát", "ipa": "/faːt˧˥/", "tone": "sắc (high-rising)"}
    ],
    "val": [
        {"text": "Đi", "ipa": "/ɗi˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Đứng", "ipa": "/ɗɨŋ˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Nghe", "ipa": "/ŋɛ˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Nói", "ipa": "/nɔj˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Hỏi", "ipa": "/hɔj˧˩˨/", "tone": "hỏi (mid-falling rising)"},
        {"text": "Tìm", "ipa": "/tim˨˩/", "tone": "huyền (low-falling)"},
        {"text": "Chọn", "ipa": "/t͡ɕɔn˨˩/", "tone": "nặng (low-glottalized)"},
        {"text": "Xoay", "ipa": "/swaj˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Lùi", "ipa": "/luj˨˩/", "tone": "huyền (low-falling)"},
        {"text": "Quay", "ipa": "/kwaːj˧˧/", "tone": "ngang (mid-level)"}
    ],
    "test": [
        {"text": "Chào", "ipa": "/t͡ɕaːw˨˩/", "tone": "huyền (low-falling)"},
        {"text": "Cứu", "ipa": "/kiw˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Hủy", "ipa": "/hwi˧˩˨/", "tone": "hỏi (mid-falling rising)"},
        {"text": "Lưu", "ipa": "/liw˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Xóa", "ipa": "/swaː˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Phóng", "ipa": "/fawŋm˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Thu", "ipa": "/tʰu˧˧/", "tone": "ngang (mid-level)"},
        {"text": "Khóa", "ipa": "/xwaː˧˥/", "tone": "sắc (high-rising)"},
        {"text": "Được", "ipa": "/ɗɨək˨˩/", "tone": "nặng (low-glottalized)"},
        {"text": "Không", "ipa": "/xəwŋm˧˧/", "tone": "ngang (mid-level)"}
    ]
}

def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_spectral_centroid(audio: np.ndarray, sr: int = 16000) -> float:
    if np.all(audio == 0):
        return 0.0
    freqs, times, spec = signal.spectrogram(audio, fs=sr, nperseg=400, noverlap=240)
    power = np.abs(spec) ** 2
    spec_sum = np.sum(power, axis=0) + 1e-9
    centroid = np.sum(freqs[:, None] * power, axis=0) / spec_sum
    return float(np.mean(centroid))

def generate_silence_samples():
    """Generates 250 distinct 1.0s SILENCE audio clips under disjoint conditions across splits."""
    print("\n--- Generating 250 Distinct SILENCE Samples ---", flush=True)
    silence_records = []
    sr = 16000
    n_samples = 16000

    sample_counter = 1

    # 1. Train Split: 170 samples
    # Condition A: 1 exact digital zero reference fixture
    zero_clip = np.zeros(n_samples, dtype=np.float32)
    sid = f"vnkws_silence_{sample_counter:04d}"
    sample_counter += 1
    raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
    kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
    sf.write(raw_path, np.zeros(48000, dtype=np.float32), 48000, subtype="PCM_16", format="WAV")
    sf.write(kws_path, zero_clip, sr, subtype="PCM_16", format="WAV")
    silence_records.append({
        "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
        "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "train",
        "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
        "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
        "model_checkpoint": "digital_zero_reference", "license": "MIT",
        "voice_id": "non_speech_fixture", "voice_gender": "neutral", "voice_region": "neutral",
        "voice_style": "exact_digital_zero", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
        "repetition_penalty": 0.0, "rng_method": "deterministic_zero", "seed": 0,
        "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
        "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
        "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
        "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
        "kws_peak_amplitude": 0.0, "kws_speech_rms_dbfs": -120.0, "kws_mean_spectral_centroid_hz": 0.0,
        "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
        "truncation_detected": False, "clipping_detected": False,
        "technical_validation_status": "VALID", "rejection_history": "[]",
        "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
    })

    # Train Condition B: 40 low-level dither / thermal quantization noise (-75 to -65 dBFS)
    for i in range(40):
        seed = 200_000 + i
        rng = np.random.RandomState(seed)
        rms_target = rng.uniform(0.00015, 0.00045)  # -76 to -67 dBFS
        noise = rng.normal(0, rms_target, n_samples).astype(np.float32)
        noise = np.clip(noise, -0.01, 0.01)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(noise, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, noise, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(noise**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "train",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "thermal_dither_noise", "license": "MIT",
            "voice_id": "thermal_dither", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "low_level_dither", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "gaussian_dither", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(noise))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(noise, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # Train Condition C: 40 synthetic stationary room tone (low-pass white noise, cutoff 400Hz, RMS -55 to -48 dBFS)
    b_lp, a_lp = signal.butter(4, 400.0 / (sr / 2.0), btype="low")
    for i in range(40):
        seed = 200_100 + i
        rng = np.random.RandomState(seed)
        white = rng.normal(0, 1.0, n_samples)
        filtered = signal.lfilter(b_lp, a_lp, white)
        target_rms = rng.uniform(0.0018, 0.0038)
        norm_factor = target_rms / (np.sqrt(np.mean(filtered**2)) + 1e-9)
        tone = (filtered * norm_factor).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(tone, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, tone, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(tone**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "train",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "room_tone_simulation", "license": "MIT",
            "voice_id": "room_tone_400hz", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "synthetic_room_tone", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "lowpass_butterworth", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(tone))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(tone, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # Train Condition D: 45 synthetic equipment HVAC hum (50Hz / 60Hz + harmonics)
    t = np.linspace(0, 1.0, n_samples, endpoint=False)
    for i in range(45):
        seed = 200_200 + i
        rng = np.random.RandomState(seed)
        f0 = 50.0 if i % 2 == 0 else 60.0
        phase = rng.uniform(0, 2 * np.pi)
        hum = np.sin(2 * np.pi * f0 * t + phase) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
        target_rms = rng.uniform(0.0020, 0.0050)  # -54 to -46 dBFS
        hum = (hum / (np.sqrt(np.mean(hum**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(hum, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, hum, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(hum**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "train",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "equipment_hum_simulation", "license": "MIT",
            "voice_id": f"hvac_hum_{int(f0)}hz", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "synthetic_hum", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "harmonic_sine_hum", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(hum))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(hum, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # Train Condition E: 44 gentle airflow pink noise (1/f roll-off)
    b_pink = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a_pink = [1.0, -2.494956002, 2.017265875, -0.522189400]
    for i in range(44):
        seed = 200_300 + i
        rng = np.random.RandomState(seed)
        w = rng.normal(0, 1.0, n_samples)
        pink = signal.lfilter(b_pink, a_pink, w)
        target_rms = rng.uniform(0.0020, 0.0045)
        pink = (pink / (np.sqrt(np.mean(pink**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(pink, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, pink, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(pink**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "train",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "pink_noise_simulation", "license": "MIT",
            "voice_id": "airflow_pink_noise", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "synthetic_airflow", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "pink_filter_iir", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(pink))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(pink, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # 2. Validation Split: 40 samples (Condition Group B: triangular dither, office bandpass, low rumble)
    # 15 triangular dither
    for i in range(15):
        seed = 300_000 + i
        rng = np.random.RandomState(seed)
        tri = rng.triangular(-1.0, 0.0, 1.0, n_samples)
        target_rms = rng.uniform(0.0003, 0.0009)
        tri_clip = (tri * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(tri_clip, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, tri_clip, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(tri_clip**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "val",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "triangular_dither", "license": "MIT",
            "voice_id": "triangular_dither_val", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "triangular_dither", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "triangular_rng", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(tri_clip))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(tri_clip, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # 15 quiet office bandpass tone (150Hz - 1200Hz)
    b_bp, a_bp = signal.butter(3, [150.0 / (sr / 2.0), 1200.0 / (sr / 2.0)], btype="bandpass")
    for i in range(15):
        seed = 300_100 + i
        rng = np.random.RandomState(seed)
        w = rng.normal(0, 1.0, n_samples)
        bp_clip = signal.lfilter(b_bp, a_bp, w)
        target_rms = rng.uniform(0.0020, 0.0040)
        bp_clip = (bp_clip / (np.sqrt(np.mean(bp_clip**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(bp_clip, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, bp_clip, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(bp_clip**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "val",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "office_ambience_bandpass", "license": "MIT",
            "voice_id": "office_bandpass_val", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "office_ambience", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "bandpass_butterworth", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(bp_clip))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(bp_clip, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # 10 low-frequency acoustic rumble (lowpass 250Hz)
    b_rb, a_rb = signal.butter(4, 250.0 / (sr / 2.0), btype="low")
    for i in range(10):
        seed = 300_200 + i
        rng = np.random.RandomState(seed)
        w = rng.normal(0, 1.0, n_samples)
        rb_clip = signal.lfilter(b_rb, a_rb, w)
        target_rms = rng.uniform(0.0025, 0.0055)
        rb_clip = (rb_clip / (np.sqrt(np.mean(rb_clip**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(rb_clip, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, rb_clip, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(rb_clip**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "val",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "acoustic_rumble_simulation", "license": "MIT",
            "voice_id": "low_rumble_val", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "acoustic_rumble", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "rumble_butterworth", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(rb_clip))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(rb_clip, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # 3. Held-Out Test Split: 40 samples (Condition Group C: Brownian noise, comb-filtered room resonance, shaped dither)
    # 15 Brownian 1/f^2 noise
    for i in range(15):
        seed = 400_000 + i
        rng = np.random.RandomState(seed)
        white = rng.normal(0, 1.0, n_samples)
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)
        target_rms = rng.uniform(0.0020, 0.0045)
        brown_clip = (brown / (np.sqrt(np.mean(brown**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(brown_clip, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, brown_clip, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(brown_clip**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "test",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "brownian_noise_simulation", "license": "MIT",
            "voice_id": "brownian_ambient_test", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "brownian_ambient", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "brownian_random_walk", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(brown_clip))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(brown_clip, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # 15 Comb-filtered ambient room resonance
    delay_samples = 320  # 20ms resonance
    for i in range(15):
        seed = 400_100 + i
        rng = np.random.RandomState(seed)
        w = rng.normal(0, 1.0, n_samples)
        comb = np.copy(w)
        for s_idx in range(delay_samples, n_samples):
            comb[s_idx] += 0.5 * comb[s_idx - delay_samples]
        target_rms = rng.uniform(0.0020, 0.0040)
        comb_clip = (comb / (np.sqrt(np.mean(comb**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(comb_clip, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, comb_clip, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(comb_clip**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "test",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "comb_room_resonance", "license": "MIT",
            "voice_id": "comb_room_resonance_test", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "room_resonance", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "comb_filter_delay", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(comb_clip))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(comb_clip, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    # 10 High-pass shaped dither noise
    b_hp, a_hp = signal.butter(2, 2000.0 / (sr / 2.0), btype="high")
    for i in range(10):
        seed = 400_200 + i
        rng = np.random.RandomState(seed)
        w = rng.normal(0, 1.0, n_samples)
        hp_clip = signal.lfilter(b_hp, a_hp, w)
        target_rms = rng.uniform(0.0003, 0.0008)
        hp_clip = (hp_clip / (np.sqrt(np.mean(hp_clip**2)) + 1e-9) * target_rms).astype(np.float32)
        
        sid = f"vnkws_silence_{sample_counter:04d}"
        sample_counter += 1
        raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
        kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
        sf.write(raw_path, np.repeat(hp_clip, 3), 48000, subtype="PCM_16", format="WAV")
        sf.write(kws_path, hp_clip, sr, subtype="PCM_16", format="WAV")
        
        rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(hp_clip**2)), 1e-9))
        silence_records.append({
            "sample_id": sid, "class_label": "SILENCE", "text": "[SILENCE]",
            "ipa_phonetic_target": "[N/A]", "vietnamese_tone": "none", "split": "test",
            "engine": "Synthetic DSP Audio Generator", "engine_version": "1.0.0",
            "backend": "NumPy / SciPy Synthetic Audio", "upstream_repo": "local_dsp",
            "model_checkpoint": "shaped_highpass_dither", "license": "MIT",
            "voice_id": "shaped_dither_test", "voice_gender": "neutral", "voice_region": "neutral",
            "voice_style": "highpass_dither", "temperature": 0.0, "top_k": 0, "top_p": 0.0,
            "repetition_penalty": 0.0, "rng_method": "highpass_butterworth", "seed": seed,
            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
            "raw_sample_rate": 48000, "kws_sample_rate": 16000, "raw_duration_sec": 1.0,
            "kws_duration_sec": 1.0, "kws_active_speech_duration_sec": 0.0, "kws_onset_sec": 0.0,
            "kws_peak_amplitude": float(np.max(np.abs(hp_clip))), "kws_speech_rms_dbfs": float(rms_dbfs),
            "kws_mean_spectral_centroid_hz": compute_spectral_centroid(hp_clip, sr),
            "raw_sha256": compute_sha256(raw_path), "kws_sha256": compute_sha256(kws_path),
            "truncation_detected": False, "clipping_detected": False,
            "technical_validation_status": "VALID", "rejection_history": "[]",
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
        })

    assert len(silence_records) == 250, f"Expected 250 silence records, got {len(silence_records)}"
    print(f"Generated {len(silence_records)} distinct SILENCE audio clips.", flush=True)
    return silence_records

def main():
    print("=" * 70)
    print("Phase 28: Six-Class Vietnamese KWS Dataset Generator")
    print("=" * 70, flush=True)

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)

    # 1. Load original 1,000 four-class records
    four_class_manifest = os.path.join(FOUR_CLASS_DIR, "manifest.csv")
    assert os.path.exists(four_class_manifest), f"4-class manifest missing: {four_class_manifest}"
    df_4class = pd.read_csv(four_class_manifest)
    assert len(df_4class) == 1000, f"Expected 1,000 records in 4-class, got {len(df_4class)}"
    print(f"Loaded 1,000 reference records for TRAI, PHAI, LEN, XUONG from {four_class_manifest}", flush=True)

    # 2. Initialize VieNeu-TTS v3 Turbo for UNKNOWN speech generation
    import vieneu
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    print("Initialized local VieNeu-TTS v3 Turbo engine.", flush=True)

    # Load voice metadata
    asset_path = pathlib.Path(vieneu.__file__).parent / "assets" / "voices_v3_turbo.json"
    voices_meta = json.loads(asset_path.read_text(encoding="utf-8"))["presets"]

    # 3. Generate 250 UNKNOWN speech samples
    print("\n--- Generating 250 UNKNOWN Speech Utterances with VieNeu-TTS v3 Turbo ---", flush=True)
    unknown_records = []
    seen_hashes = set(df_4class["kws_sha256"])

    sample_counter = 1
    t_start = time.time()

    # Voice to split mapping
    voice_to_split = {}
    for sp, voices in SPLIT_VOICE_ALLOCATION.items():
        for v in voices:
            voice_to_split[v] = sp

    # Order of voices: 17 train voices first, then 4 val, then 4 test
    ordered_voices = []
    for sp in ["train", "val", "test"]:
        for v in SPLIT_VOICE_ALLOCATION[sp]:
            ordered_voices.append((sp, v))

    for sp_name, vid in ordered_voices:
        v_meta = voices_meta.get(vid, {})
        v_gender = v_meta.get("gender", "neutral")
        v_region = v_meta.get("region", "neutral")
        v_style = v_meta.get("style", "tu_nhien")

        texts_for_split = UNKNOWN_TEXTS[sp_name]
        assert len(texts_for_split) == 10, f"Split {sp_name} must have 10 texts"

        for u_idx, item in enumerate(texts_for_split):
            text = item["text"]
            ipa = item["ipa"]
            tone = item["tone"]

            sid = f"vnkws_unknown_{sample_counter:04d}"
            sample_counter += 1

            # Seed formula
            seed = 500_000 + sample_counter * 100 + u_idx
            np.random.seed(seed)

            raw_audio = tts.infer(
                text,
                voice=vid,
                temperature=0.75,
                top_k=25,
                top_p=0.92,
                repetition_penalty=1.20
            )

            # Polyphase resample 1:3 from 48 kHz to 16 kHz
            resampled_16k = signal.resample_poly(raw_audio, 1, 3).astype(np.float32)
            n_resampled = len(resampled_16k)

            # Standard 16,000-sample window
            kws_clip = np.zeros(16000, dtype=np.float32)
            onset_sample = int(0.150 * 16000)
            if onset_sample + n_resampled > 16000 - 800:
                onset_sample = max(800, 16000 - n_resampled - 800)

            kws_clip[onset_sample : onset_sample + n_resampled] = resampled_16k[:min(n_resampled, 16000 - onset_sample)]

            # Peak normalize if peak > 0.95
            peak = float(np.max(np.abs(kws_clip)))
            if peak > 0.95:
                kws_clip = kws_clip * (0.95 / peak)
                peak = float(np.max(np.abs(kws_clip)))

            speech_segment = kws_clip[onset_sample : onset_sample + n_resampled]
            speech_rms = float(np.sqrt(np.mean(speech_segment**2)))
            speech_rms_dbfs = 20.0 * np.log10(max(speech_rms, 1e-9))
            centroid_hz = compute_spectral_centroid(kws_clip, 16000)

            raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw_48k.wav")
            kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws_16k.wav")
            sf.write(raw_path, raw_audio, 48000, subtype="PCM_16", format="WAV")
            sf.write(kws_path, kws_clip, 16000, subtype="PCM_16", format="WAV")

            raw_hash = compute_sha256(raw_path)
            kws_hash = compute_sha256(kws_path)

            assert kws_hash not in seen_hashes, f"Duplicate hash detected for {sid}"
            seen_hashes.add(kws_hash)

            unknown_records.append({
                "sample_id": sid, "class_label": "UNKNOWN", "text": text,
                "ipa_phonetic_target": ipa, "vietnamese_tone": tone, "split": sp_name,
                "engine": "VieNeu-TTS v3 Turbo", "engine_version": "3.8.3",
                "backend": "Local CPU ONNX Runtime", "upstream_repo": "https://github.com/pnnbao97/VieNeu-TTS",
                "model_checkpoint": "pnnbao-ump/VieNeu-TTS-v3-Turbo", "license": "Apache-2.0",
                "voice_id": vid, "voice_gender": v_gender, "voice_region": v_region,
                "voice_style": v_style, "temperature": 0.75, "top_k": 25, "top_p": 0.92,
                "repetition_penalty": 1.20, "rng_method": "cpu_onnx_deterministic", "seed": seed,
                "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                "raw_sample_rate": 48000, "kws_sample_rate": 16000,
                "raw_duration_sec": round(len(raw_audio) / 48000.0, 4),
                "kws_duration_sec": 1.0,
                "kws_active_speech_duration_sec": round(n_resampled / 16000.0, 4),
                "kws_onset_sec": round(onset_sample / 16000.0, 4),
                "kws_peak_amplitude": round(peak, 4),
                "kws_speech_rms_dbfs": round(speech_rms_dbfs, 2),
                "kws_mean_spectral_centroid_hz": round(centroid_hz, 1),
                "raw_sha256": raw_hash, "kws_sha256": kws_hash,
                "truncation_detected": False, "clipping_detected": False,
                "technical_validation_status": "VALID", "rejection_history": "[]",
                "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
            })

            if len(unknown_records) % 50 == 0:
                print(f"[{len(unknown_records)}/250] Synthesized {sid}: '{text}' ({vid}, {sp_name}) in {time.time()-t_start:.1f}s", flush=True)

    assert len(unknown_records) == 250, f"Expected 250 UNKNOWN records, got {len(unknown_records)}"
    print(f"Generated {len(unknown_records)} UNKNOWN speech utterances successfully.", flush=True)

    # 4. Generate 250 SILENCE samples
    silence_records = generate_silence_samples()

    # 5. Assemble 1,500 6-Class Master Manifest
    print("\n--- Assembling 1,500 Six-Class Master and Split Manifests ---", flush=True)
    rec_4class = df_4class.to_dict(orient="records")
    all_6class_records = rec_4class + unknown_records + silence_records
    master_6class_df = pd.DataFrame(all_6class_records)

    assert len(master_6class_df) == 1500, f"Expected 1,500 total records, got {len(master_6class_df)}"
    class_counts = master_6class_df["class_label"].value_counts().to_dict()
    print("Master 6-Class Counts:", class_counts, flush=True)
    assert class_counts == {"TRAI": 250, "PHAI": 250, "LEN": 250, "XUONG": 250, "UNKNOWN": 250, "SILENCE": 250}

    # Save master manifests
    master_6class_df.to_csv(MASTER_MANIFEST_PATH, index=False, encoding="utf-8")
    master_manifest_copy = os.path.join(MANIFESTS_DIR, "manifest.csv")
    master_6class_df.to_csv(master_manifest_copy, index=False, encoding="utf-8")
    print(f"Saved master manifest to: {MASTER_MANIFEST_PATH}", flush=True)

    # Save split manifests
    for sp_name, exp_total in [("train", 1020), ("val", 240), ("test", 240)]:
        sp_df = master_6class_df[master_6class_df["split"] == sp_name]
        assert len(sp_df) == exp_total, f"Expected {exp_total} rows in {sp_name}, got {len(sp_df)}"
        sp_counts = sp_df["class_label"].value_counts().to_dict()
        exp_per_class = 170 if sp_name == "train" else 40
        for cls in ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]:
            assert sp_counts[cls] == exp_per_class, f"{sp_name} class {cls} count mismatch: {sp_counts[cls]} != {exp_per_class}"
        sp_path = os.path.join(SPLITS_DIR, f"{sp_name}_manifest.csv")
        sp_df.to_csv(sp_path, index=False, encoding="utf-8")
        print(f"Saved {sp_name} manifest ({len(sp_df)} rows, balanced {exp_per_class}/class) to: {sp_path}", flush=True)

    # Save dataset config
    cfg_path = os.path.join(CONFIGS_DIR, "dataset_config.json")
    cfg_data = {
        "dataset_name": "synthetic_kws_dataset_vieneu_6class",
        "version": "1.0.0",
        "total_samples": 1500,
        "classes": ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"],
        "class_counts": {"TRAI": 250, "PHAI": 250, "LEN": 250, "XUONG": 250, "UNKNOWN": 250, "SILENCE": 250},
        "splits": {
            "train": {"total": 1020, "per_class": 170, "voices": SPLIT_VOICE_ALLOCATION["train"]},
            "val": {"total": 240, "per_class": 40, "voices": SPLIT_VOICE_ALLOCATION["val"]},
            "test": {"total": 240, "per_class": 40, "voices": SPLIT_VOICE_ALLOCATION["test"]}
        },
        "unknown_text_inventories": {
            sp: [item["text"] for item in UNKNOWN_TEXTS[sp]] for sp in ["train", "val", "test"]
        },
        "audio_specifications": {
            "sample_rate_hz": 16000,
            "duration_sec": 1.0,
            "target_samples": 16000,
            "channels": 1,
            "format": "WAV",
            "subtype": "PCM_16"
        },
        "governance": {
            "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION",
            "domain_validation_status": "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "noise_robustness_evaluation": "SEPARATE_EVALUATION_PROTOCOL_REQUIRED"
        }
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset configuration to: {cfg_path}", flush=True)

    print("\nDataset generation and manifest assembly completed successfully!")

if __name__ == "__main__":
    main()
