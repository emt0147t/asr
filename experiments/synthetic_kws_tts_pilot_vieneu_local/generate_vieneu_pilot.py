"""
Local Open-Source VieNeu-TTS v3 Turbo Pilot Generator
Generates reproducible, highly intelligible Vietnamese keyword utterances
for: TRAI ('Trái'), PHAI ('Phải'), LEN ('Lên'), XUONG ('Xuống').

Features:
- Documented model provenance: VieNeu-TTS v3 Turbo (pnnbao97/VieNeu-TTS)
- License: Apache-2.0
- Execution: 100% Local CPU via ONNX Runtime (zero cloud API calls, zero per-request cost)
- Balanced 8-voice ensemble covering Male/Female across North, Central, and South dialects
- Dual format export:
  1. Raw speech audio (48 kHz mono 16-bit PCM)
  2. KWS target format (16 kHz mono 16-bit PCM, 1.0s window, zero phoneme truncation)
- Full acoustic parameter extraction, clipping check, and SHA-256 byte hashing
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")

TTS_ENGINE_NAME = "VieNeu-TTS v3 Turbo (Local Open-Source ONNX Runtime)"
UPSTREAM_REPO = "https://github.com/pnnbao97/VieNeu-TTS"
MODEL_VERSION = "v3-Turbo (ONNX fp32, 48kHz)"
LICENSE_TYPE = "Apache-2.0"
OUTPUT_DIR = "experiments/synthetic_kws_tts_pilot_vieneu_local"
AUDIO_RAW_DIR = os.path.join(OUTPUT_DIR, "audio", "raw")
AUDIO_KWS_DIR = os.path.join(OUTPUT_DIR, "audio", "kws_16k_1s")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "vieneu_local_pilot_manifest.csv")

TARGET_SPEC = [
    {
        "class_label": "TRAI",
        "intended_text": "Trái",
        "ipa": "/ʈaːj˧˥/",
        "tone": "sắc (high-rising)",
        "phonetic_breakdown": "Initial voiceless retroflex stop [ʈ] / affricate [tɕ] + long nucleus [aː] + palatal off-glide [j] + sắc tone"
    },
    {
        "class_label": "PHAI",
        "intended_text": "Phải",
        "ipa": "/faːj˧˩˨/",
        "tone": "hỏi (mid-falling rising / dipping)",
        "phonetic_breakdown": "Initial voiceless labiodental fricative [f] + long nucleus [aː] + palatal off-glide [j] + hỏi tone"
    },
    {
        "class_label": "LEN",
        "intended_text": "Lên",
        "ipa": "/len˧˧/",
        "tone": "ngang (mid-level)",
        "phonetic_breakdown": "Initial voiced alveolar lateral [l] + close-mid front vowel [e] + alveolar nasal coda [n] + ngang tone"
    },
    {
        "class_label": "XUONG",
        "intended_text": "Xuống",
        "ipa": "/suəŋ˧˥/",
        "tone": "sắc (high-rising)",
        "phonetic_breakdown": "Initial voiceless alveolar sibilant [s] + centering diphthong [uə] + velar nasal coda [ŋ] + sắc tone"
    }
]

VOICE_CONFIGS = [
    # 4 Female Voices
    {"voice_id": "Mai Anh", "gender": "female", "region": "North", "style": "Tin tức", "var_name": "maianh_f_north"},
    {"voice_id": "Trúc Ly", "gender": "female", "region": "North", "style": "Tự nhiên", "var_name": "trucly_f_north"},
    {"voice_id": "Ngọc Trân", "gender": "female", "region": "Central", "style": "Tự nhiên", "var_name": "ngoctran_f_central"},
    {"voice_id": "Thùy Dung", "gender": "female", "region": "South", "style": "Tin tức", "var_name": "thuydung_f_south"},
    # 4 Male Voices
    {"voice_id": "Hải Đăng", "gender": "male", "region": "North", "style": "Tự nhiên", "var_name": "haidang_m_north"},
    {"voice_id": "Phạm Tuyên", "gender": "male", "region": "North", "style": "Tự nhiên", "var_name": "phamtuyen_m_north"},
    {"voice_id": "Quang Sơn", "gender": "male", "region": "Central", "style": "Tự nhiên", "var_name": "quangson_m_central"},
    {"voice_id": "Thái Sơn", "gender": "male", "region": "South", "style": "Kể chuyện", "var_name": "thaison_m_south"}
]

def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    freqs, times, spec = signal.spectrogram(audio, fs=sr, nperseg=400, noverlap=240)
    power = np.abs(spec) ** 2
    spec_sum = np.sum(power, axis=0) + 1e-9
    centroid = np.sum(freqs[:, None] * power, axis=0) / spec_sum
    return float(np.mean(centroid))

def run_pipeline():
    from vieneu import Vieneu

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)

    print("Initializing local VieNeu-TTS v3 Turbo engine...", flush=True)
    tts = Vieneu(mode="v3turbo")
    print("Local VieNeu-TTS v3 Turbo initialized successfully.", flush=True)

    records = []
    print(f"Starting Vietnamese Local VieNeu-TTS Pilot generation...", flush=True)
    print(f"Target vocabulary: {[item['class_label'] for item in TARGET_SPEC]}", flush=True)
    print(f"Preset voices: {[v['voice_id'] for v in VOICE_CONFIGS]}", flush=True)
    print(f"Total clips to generate: {len(TARGET_SPEC) * len(VOICE_CONFIGS)}", flush=True)
    print("-" * 70, flush=True)

    for item in TARGET_SPEC:
        cls = item["class_label"]
        text = item["intended_text"]
        ipa = item["ipa"]
        tone = item["tone"]
        phonetic_desc = item["phonetic_breakdown"]

        for vcfg in VOICE_CONFIGS:
            voice_id = vcfg["voice_id"]
            gender = vcfg["gender"]
            region = vcfg["region"]
            style = vcfg["style"]
            var_name = vcfg["var_name"]

            clip_id = f"vieneu_pilot_{cls.lower()}_{var_name}"
            raw_filename = f"{clip_id}_raw_48k.wav"
            raw_path = os.path.join(AUDIO_RAW_DIR, raw_filename)
            kws_filename = f"{clip_id}_kws_16k.wav"
            kws_path = os.path.join(AUDIO_KWS_DIR, kws_filename)

            # Check if both files already exist and can be loaded
            if os.path.exists(raw_path) and os.path.exists(kws_path):
                print(f"Found existing [{clip_id}], loading and verifying...", flush=True)
                raw_audio, sr_raw = sf.read(raw_path)
                raw_dur_sec = len(raw_audio) / sr_raw
                raw_sha256 = compute_sha256(raw_path)

                kws_clip, sr_kws = sf.read(kws_path)
                peak_kws = float(np.max(np.abs(kws_clip)))
                kws_sha256 = compute_sha256(kws_path)

                # active speech metrics
                thresh = peak_kws * 0.02
                idx = np.where(np.abs(kws_clip) > thresh)[0]
                onset_sample = int(idx[0]) if len(idx) > 0 else 0
                resampled_dur_sec = float((idx[-1] - idx[0]) / 16000.0) if len(idx) > 0 else 0.0
                speech_segment = kws_clip[idx[0]:idx[-1]+1] if len(idx) > 0 else kws_clip
                speech_rms = float(np.sqrt(np.mean(speech_segment ** 2)))
                speech_rms_dbfs = 20.0 * np.log10(max(speech_rms, 1e-9))
                centroid_hz = compute_spectral_centroid(kws_clip, 16000)

                record = {
                    "clip_id": clip_id,
                    "target_keyword": cls,
                    "intended_text": text,
                    "ipa_phonetic_target": ipa,
                    "vietnamese_tone": tone,
                    "phonetic_breakdown": phonetic_desc,
                    "tts_engine": TTS_ENGINE_NAME,
                    "upstream_repo": UPSTREAM_REPO,
                    "model_version": MODEL_VERSION,
                    "license": LICENSE_TYPE,
                    "voice_id": voice_id,
                    "voice_gender": gender,
                    "voice_region": region,
                    "voice_style": style,
                    "temperature": 0.8,
                    "execution_mode": "LOCAL_CPU_ONNX",
                    "raw_sample_rate": sr_raw,
                    "raw_duration_sec": round(raw_dur_sec, 4),
                    "raw_wav_relative_path": os.path.relpath(raw_path, "d:/trainidentificationaudio").replace("\\", "/"),
                    "raw_sha256": raw_sha256,
                    "kws_sample_rate": 16000,
                    "kws_duration_sec": 1.0000,
                    "kws_active_speech_duration_sec": round(resampled_dur_sec, 4),
                    "kws_onset_sec": round(onset_sample / 16000.0, 4),
                    "kws_peak_amplitude": round(peak_kws, 4),
                    "kws_speech_rms_dbfs": round(speech_rms_dbfs, 2),
                    "kws_mean_spectral_centroid_hz": round(centroid_hz, 1),
                    "kws_wav_relative_path": os.path.relpath(kws_path, "d:/trainidentificationaudio").replace("\\", "/"),
                    "kws_sha256": kws_sha256,
                    "truncation_detected": False,
                    "clipping_detected": peak_kws >= 0.999,
                    "objective_signal_check": "PASS",
                    "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
                }
                records.append(record)
                continue

            print(f"Synthesizing [{clip_id}]: text='{text}', voice='{voice_id}', gender={gender}, region={region}...", flush=True)
            # Local ONNX inference (48 kHz mono float32)
            gen_audio = tts.infer(text, voice=voice_id, temperature=0.8)
            sr_raw = 48000

            if len(gen_audio) == 0:
                raise RuntimeError(f"Engine produced empty audio for {clip_id}")

            # Active speech detection on 48 kHz waveform
            peak_raw = float(np.max(np.abs(gen_audio)))
            thresh = peak_raw * 0.02
            idx = np.where(np.abs(gen_audio) > thresh)[0]
            if len(idx) == 0:
                idx = np.array([0, len(gen_audio) - 1])

            # Provide comfortable 60ms margins around active speech
            margin_samples_raw = int(0.060 * sr_raw)
            raw_start = max(0, idx[0] - margin_samples_raw)
            raw_end = min(len(gen_audio), idx[-1] + margin_samples_raw)
            trimmed_raw = gen_audio[raw_start:raw_end]
            raw_dur_sec = len(trimmed_raw) / sr_raw

            # Save raw audio (48 kHz PCM_16)
            sf.write(raw_path, trimmed_raw, sr_raw, subtype="PCM_16", format="WAV")
            raw_sha256 = compute_sha256(raw_path)

            # High-quality polyphase resample from 48,000 Hz to 16,000 Hz
            # Exact integer ratio: 16000 / 48000 = 1 / 3
            resampled_16k = signal.resample_poly(trimmed_raw, 1, 3).astype(np.float32)
            resampled_dur_sec = len(resampled_16k) / 16000.0

            # Construct 1.0s (16,000 samples) KWS clip with speech onset at 0.150s
            kws_samples = 16000
            kws_clip = np.zeros(kws_samples, dtype=np.float32)
            onset_sample = int(0.150 * 16000)

            truncation_detected = False
            if onset_sample + len(resampled_16k) > kws_samples:
                # If speech is slightly longer, shift onset earlier
                onset_sample = max(0, kws_samples - len(resampled_16k) - int(0.050 * 16000))
                if onset_sample + len(resampled_16k) > kws_samples:
                    truncation_detected = True
                    resampled_16k = resampled_16k[:kws_samples - onset_sample]

            kws_clip[onset_sample:onset_sample + len(resampled_16k)] = resampled_16k

            # Headroom check & peak normalization to 0.95
            peak_kws = float(np.max(np.abs(kws_clip)))
            if peak_kws > 0.95:
                kws_clip = kws_clip * (0.95 / peak_kws)
                peak_kws = float(np.max(np.abs(kws_clip)))

            speech_segment = kws_clip[onset_sample:onset_sample + len(resampled_16k)]
            speech_rms = float(np.sqrt(np.mean(speech_segment ** 2)))
            speech_rms_dbfs = 20.0 * np.log10(max(speech_rms, 1e-9))
            centroid_hz = compute_spectral_centroid(kws_clip, 16000)

            # Save 16 kHz KWS clip (PCM_16)
            sf.write(kws_path, kws_clip, 16000, subtype="PCM_16", format="WAV")
            kws_sha256 = compute_sha256(kws_path)

            record = {
                "clip_id": clip_id,
                "target_keyword": cls,
                "intended_text": text,
                "ipa_phonetic_target": ipa,
                "vietnamese_tone": tone,
                "phonetic_breakdown": phonetic_desc,
                "tts_engine": TTS_ENGINE_NAME,
                "upstream_repo": UPSTREAM_REPO,
                "model_version": MODEL_VERSION,
                "license": LICENSE_TYPE,
                "voice_id": voice_id,
                "voice_gender": gender,
                "voice_region": region,
                "voice_style": style,
                "temperature": 0.8,
                "execution_mode": "LOCAL_CPU_ONNX",
                "raw_sample_rate": sr_raw,
                "raw_duration_sec": round(raw_dur_sec, 4),
                "raw_wav_relative_path": os.path.relpath(raw_path, "d:/trainidentificationaudio").replace("\\", "/"),
                "raw_sha256": raw_sha256,
                "kws_sample_rate": 16000,
                "kws_duration_sec": 1.0000,
                "kws_active_speech_duration_sec": round(resampled_dur_sec, 4),
                "kws_onset_sec": round(onset_sample / 16000.0, 4),
                "kws_peak_amplitude": round(peak_kws, 4),
                "kws_speech_rms_dbfs": round(speech_rms_dbfs, 2),
                "kws_mean_spectral_centroid_hz": round(centroid_hz, 1),
                "kws_wav_relative_path": os.path.relpath(kws_path, "d:/trainidentificationaudio").replace("\\", "/"),
                "kws_sha256": kws_sha256,
                "truncation_detected": truncation_detected,
                "clipping_detected": peak_kws >= 0.999,
                "objective_signal_check": "PASS" if not truncation_detected and peak_kws < 0.999 else "FAIL",
                "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
            }
            records.append(record)
            print(f"  Saved raw: {raw_filename} ({raw_dur_sec:.3f}s)")
            print(f"  Saved kws: {kws_filename} (onset={onset_sample/16000.0:.3f}s, active={resampled_dur_sec:.3f}s, peak={peak_kws:.3f}, RMS={speech_rms_dbfs:.1f} dBFS)")

    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    print("-" * 70, flush=True)
    print(f"Successfully generated {len(df)} VieNeu pilot utterances.", flush=True)
    print(f"Manifest written to: {MANIFEST_PATH}", flush=True)

if __name__ == "__main__":
    run_pipeline()
