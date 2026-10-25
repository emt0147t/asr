"""
Vietnamese Synthetic KWS — 1,000-Utterance Dataset Generator
Generates 1,000 distinct, balanced, reproducible Vietnamese command utterances
(250 TRAI, 250 PHAI, 250 LEN, 250 XUONG) using local open-source VieNeu-TTS v3 Turbo.
"""

import os
import sys
import json
import hashlib
import time
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import pathlib

# Ensure UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_RAW_DIR = os.path.join(DATASET_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(DATASET_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
REPORTS_DIR = os.path.join(DATASET_DIR, "reports")
CONFIGS_DIR = os.path.join(DATASET_DIR, "configs")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")
REJECTION_LOG_PATH = os.path.join(REPORTS_DIR, "rejection_log.json")

# Classes and target texts
CLASSES = [
    {
        "class_label": "TRAI",
        "text": "Trái",
        "ipa": "/ʈaːj˧˥/",
        "tone": "sắc (high-rising)",
        "id_prefix": "vnkws_trai"
    },
    {
        "class_label": "PHAI",
        "text": "Phải",
        "ipa": "/faːj˧˩˨/",
        "tone": "hỏi (mid-falling rising / dipping)",
        "id_prefix": "vnkws_phai"
    },
    {
        "class_label": "LEN",
        "text": "Lên",
        "ipa": "/len˧˧/",
        "tone": "ngang (mid-level)",
        "id_prefix": "vnkws_len"
    },
    {
        "class_label": "XUONG",
        "text": "Xuống",
        "ipa": "/suəŋ˧˥/",
        "tone": "sắc (high-rising)",
        "id_prefix": "vnkws_xuong"
    }
]

# Voice Split Partition (Disjoint Voice Groups for Train, Val, Test)
SPLIT_VOICE_ALLOCATION = {
    "train": [
        "Adam bựa", "Thiện Minh", "Hải Đăng", "Thiền Tâm Đức", "Minh Đức",
        "Xuân Vĩnh", "Thanh Bình", "Trúc Ly", "Ngọc Huyền", "Ngọc Linh",
        "Đoan Trang", "Quang Sơn", "Thái Sơn", "Minh Triết", "Adam",
        "Thục Đoan", "Mỹ Duyên"
    ],  # 17 voices -> 17 * 10 = 170 per class -> 680 total (68%)
    "val": [
        "Mai Anh", "Phạm Tuyên", "Ngọc Trân", "Thùy Dung"
    ],  # 4 voices -> 4 * 10 = 40 per class -> 160 total (16%)
    "test": [
        "Quỳnh Anh", "Quốc Tuấn", "Đức Trí", "Kim Thanh"
    ]   # 4 voices -> 4 * 10 = 40 per class -> 160 total (16%)
}

def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    freqs, times, spec = signal.spectrogram(audio, fs=sr, nperseg=400, noverlap=240)
    power = np.abs(spec) ** 2
    spec_sum = np.sum(power, axis=0) + 1e-9
    centroid = np.sum(freqs[:, None] * power, axis=0) / spec_sum
    return float(np.mean(centroid))

def run_dataset_generation():
    import vieneu
    from vieneu import Vieneu

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("Vietnamese Synthetic KWS — 1,000 Utterance Dataset Generator", flush=True)
    print("Engine: VieNeu-TTS v3 Turbo (Local CPU/ONNX Runtime)", flush=True)
    print("=" * 70, flush=True)

    # Initialize local TTS engine
    tts = Vieneu(mode="v3turbo")
    voices_list = tts.list_preset_voices()
    print(f"Loaded {len(voices_list)} preset voices from installed VieNeu.", flush=True)

    # Load raw voices metadata JSON for ground-truth gender & region
    asset_path = pathlib.Path(vieneu.__file__).parent / "assets" / "voices_v3_turbo.json"
    voices_meta = json.loads(asset_path.read_text(encoding="utf-8"))["presets"]

    # Verify all allocated voices exist in installed presets
    all_allocated_voices = []
    for split_name, voice_ids in SPLIT_VOICE_ALLOCATION.items():
        for vid in voice_ids:
            assert vid in voices_meta, f"Voice {vid} not in upstream voices_v3_turbo.json"
            all_allocated_voices.append(vid)
    assert len(all_allocated_voices) == 25, f"Expected 25 allocated voices, got {len(all_allocated_voices)}"
    print("Voice split verified: 17 train, 4 val, 4 test (25 total voices, zero overlap).", flush=True)

    # Invert mapping: voice_id -> split
    voice_to_split = {}
    for split_name, voice_ids in SPLIT_VOICE_ALLOCATION.items():
        for vid in voice_ids:
            voice_to_split[vid] = split_name

    # Check for existing manifest to enable resume capability
    existing_records = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            old_df = pd.read_csv(MANIFEST_PATH)
            for _, r in old_df.iterrows():
                sid = r["sample_id"]
                if os.path.exists(r["raw_wav_path"]) and os.path.exists(r["kws_wav_path"]):
                    existing_records[sid] = r.to_dict()
            print(f"Found {len(existing_records)} existing valid records on disk. Resuming...", flush=True)
        except Exception as e:
            print(f"Could not read existing manifest ({e}), starting fresh.", flush=True)

    accepted_records = []
    rejections = []
    seen_raw_hashes = set()

    # Pre-register seen hashes from existing records
    for sid, rec in existing_records.items():
        seen_raw_hashes.add(rec["raw_sha256"])

    # Generation parameter schedules (conservative, tone-preserving)
    # 10 configurations per voice per class
    PARAM_SCHEDULE = [
        {"temp": 0.70, "top_p": 0.95, "rep_pen": 1.20},
        {"temp": 0.75, "top_p": 0.95, "rep_pen": 1.20},
        {"temp": 0.80, "top_p": 0.95, "rep_pen": 1.20},
        {"temp": 0.70, "top_p": 0.90, "rep_pen": 1.15},
        {"temp": 0.75, "top_p": 0.90, "rep_pen": 1.15},
        {"temp": 0.80, "top_p": 0.90, "rep_pen": 1.25},
        {"temp": 0.65, "top_p": 0.95, "rep_pen": 1.20},
        {"temp": 0.85, "top_p": 0.95, "rep_pen": 1.20},
        {"temp": 0.72, "top_p": 0.92, "rep_pen": 1.18},
        {"temp": 0.78, "top_p": 0.94, "rep_pen": 1.22},
    ]

    total_target = 1000
    generated_count = 0
    t_start = time.time()

    for cls_idx, cls_info in enumerate(CLASSES):
        cls_label = cls_info["class_label"]
        text = cls_info["text"]
        id_prefix = cls_info["id_prefix"]
        ipa = cls_info["ipa"]
        tone = cls_info["tone"]

        print(f"\n--- Generating Class: {cls_label} ('{text}') [Target: 250 utterances] ---", flush=True)

        class_sample_num = 1
        for voice_idx, vid in enumerate(all_allocated_voices):
            v_meta = voices_meta[vid]
            v_gender = v_meta.get("gender", "unknown")
            v_region = v_meta.get("region", "unknown")
            v_style = v_meta.get("style", "tu_nhien")
            v_split = voice_to_split[vid]

            # Generate 10 utterances per voice
            for u_idx in range(10):
                sample_id = f"{id_prefix}_{class_sample_num:04d}"
                class_sample_num += 1

                # If already generated and valid on disk, keep and continue
                if sample_id in existing_records:
                    rec = existing_records[sample_id]
                    accepted_records.append(rec)
                    generated_count += 1
                    if generated_count % 50 == 0:
                        print(f"[{generated_count}/{total_target}] Verified existing: {sample_id} ({rec['class_label']}, {rec['voice_id']})", flush=True)
                    continue

                param = PARAM_SCHEDULE[u_idx]
                temperature = param["temp"]
                top_p = param["top_p"]
                rep_pen = param["rep_pen"]
                top_k = 25

                # Deterministic seed per sample
                base_seed = 100_000 + cls_idx * 10_000 + voice_idx * 100 + u_idx

                # Retry loop (up to 3 retries)
                accepted = False
                retry_count = 0
                while not accepted and retry_count < 3:
                    current_seed = base_seed + retry_count * 50_000
                    np.random.seed(current_seed)

                    try:
                        raw_audio = tts.infer(
                            text,
                            voice=vid,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=rep_pen
                        )
                    except Exception as e:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": f"Synthesis exception: {str(e)}",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    sr_raw = 48000
                    if len(raw_audio) == 0:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": "Empty audio output",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    # Validate finite values
                    if np.isnan(raw_audio).any() or np.isinf(raw_audio).any():
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": "NaN/Inf detected in raw audio",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    # Validate peak level and duration
                    raw_peak = float(np.max(np.abs(raw_audio)))
                    raw_dur_sec = len(raw_audio) / sr_raw

                    if raw_peak < 0.05:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": f"Signal too weak: peak={raw_peak:.4f}",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    if raw_peak >= 0.999:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": f"Clipping in raw output: peak={raw_peak:.4f}",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    # Check natural speech duration bounds (monosyllable should be 0.20s - 0.85s)
                    if raw_dur_sec < 0.20 or raw_dur_sec > 0.85:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": f"Duration out of bounds: {raw_dur_sec:.3f}s",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    # Convert to 16 kHz mono via polyphase filter (exact integer 1:3 downsampling)
                    # Note: We preserve the natural onset and coda intact with NO heuristic threshold cutting
                    resampled_16k = signal.resample_poly(raw_audio, 1, 3).astype(np.float32)
                    n_resampled = len(resampled_16k)
                    resampled_dur_sec = n_resampled / 16000.0

                    if n_resampled > 16000:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": f"Resampled speech exceeds 1.0s window ({n_resampled} samples)",
                            "seed": current_seed,
                            "voice": vid
                        })
                        retry_count += 1
                        continue

                    # Align into standard 1.0s (16,000 samples) KWS window
                    # Target onset at 0.150s (sample 2400)
                    kws_samples = 16000
                    kws_clip = np.zeros(kws_samples, dtype=np.float32)
                    onset_sample = int(0.150 * 16000)

                    # Ensure leading and trailing margins >= 50ms (800 samples)
                    if onset_sample + n_resampled > kws_samples - 800:
                        onset_sample = max(800, kws_samples - n_resampled - 800)

                    # Place resampled speech cleanly
                    kws_clip[onset_sample : onset_sample + n_resampled] = resampled_16k

                    # Headroom check & peak normalization to 0.95 if needed
                    kws_peak = float(np.max(np.abs(kws_clip)))
                    if kws_peak > 0.95:
                        kws_clip = kws_clip * (0.95 / kws_peak)
                        kws_peak = float(np.max(np.abs(kws_clip)))

                    speech_segment = kws_clip[onset_sample : onset_sample + n_resampled]
                    speech_rms = float(np.sqrt(np.mean(speech_segment ** 2)))
                    speech_rms_dbfs = 20.0 * np.log10(max(speech_rms, 1e-9))
                    centroid_hz = compute_spectral_centroid(kws_clip, 16000)

                    # Save raw 48 kHz WAV
                    raw_filename = f"{sample_id}_raw_48k.wav"
                    raw_path = os.path.join(AUDIO_RAW_DIR, raw_filename)
                    sf.write(raw_path, raw_audio, sr_raw, subtype="PCM_16", format="WAV")
                    raw_hash = compute_sha256(raw_path)

                    # Check for exact duplicate raw hash
                    if raw_hash in seen_raw_hashes:
                        rejections.append({
                            "sample_id": sample_id,
                            "reason": f"Duplicate raw audio hash: {raw_hash}",
                            "seed": current_seed,
                            "voice": vid
                        })
                        if os.path.exists(raw_path):
                            os.remove(raw_path)
                        retry_count += 1
                        continue

                    seen_raw_hashes.add(raw_hash)

                    # Save 16 kHz KWS WAV
                    kws_filename = f"{sample_id}_kws_16k.wav"
                    kws_path = os.path.join(AUDIO_KWS_DIR, kws_filename)
                    sf.write(kws_path, kws_clip, 16000, subtype="PCM_16", format="WAV")
                    kws_hash = compute_sha256(kws_path)

                    rec = {
                        "sample_id": sample_id,
                        "class_label": cls_label,
                        "text": text,
                        "ipa_phonetic_target": ipa,
                        "vietnamese_tone": tone,
                        "split": v_split,
                        "engine": "VieNeu-TTS v3 Turbo",
                        "engine_version": "3.8.3",
                        "backend": "Local CPU ONNX Runtime",
                        "upstream_repo": "https://github.com/pnnbao97/VieNeu-TTS",
                        "model_checkpoint": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
                        "license": "Apache-2.0",
                        "voice_id": vid,
                        "voice_gender": v_gender,
                        "voice_region": v_region,
                        "voice_style": v_style,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": rep_pen,
                        "rng_method": "numpy.random.seed",
                        "seed": current_seed,
                        "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                        "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                        "raw_sample_rate": sr_raw,
                        "kws_sample_rate": 16000,
                        "raw_duration_sec": round(raw_dur_sec, 4),
                        "kws_duration_sec": 1.0000,
                        "kws_active_speech_duration_sec": round(resampled_dur_sec, 4),
                        "kws_onset_sec": round(onset_sample / 16000.0, 4),
                        "kws_peak_amplitude": round(kws_peak, 4),
                        "kws_speech_rms_dbfs": round(speech_rms_dbfs, 2),
                        "kws_mean_spectral_centroid_hz": round(centroid_hz, 1),
                        "raw_sha256": raw_hash,
                        "kws_sha256": kws_hash,
                        "truncation_detected": False,
                        "clipping_detected": False,
                        "technical_validation_status": "VALID",
                        "rejection_history": f"{retry_count}_RETRIES",
                        "subjective_intelligibility_status": "PENDING_HUMAN_LISTENER_VERIFICATION"
                    }
                    accepted_records.append(rec)
                    accepted = True
                    generated_count += 1

                    if generated_count % 50 == 0 or generated_count == total_target:
                        elapsed = time.time() - t_start
                        rate = generated_count / max(1, elapsed)
                        print(f"[{generated_count}/{total_target}] Generated: {sample_id} ({cls_label}, {vid}) | {rate:.1f} samples/s | Remaining: {(total_target - generated_count)/max(0.01, rate):.0f}s", flush=True)

                if not accepted:
                    print(f"CRITICAL ERROR: Failed to generate acceptable audio for {sample_id} after 3 retries!", flush=True)

    # Save rejections log
    with open(REJECTION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(rejections, f, indent=2, ensure_ascii=False)
    print(f"Rejections log saved to: {REJECTION_LOG_PATH} ({len(rejections)} rejection events)", flush=True)

    # Save full manifest
    df_all = pd.DataFrame(accepted_records)
    # Sort deterministically
    df_all = df_all.sort_values(by="sample_id").reset_index(drop=True)
    df_all.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    print(f"Full manifest saved to: {MANIFEST_PATH} ({len(df_all)} rows)", flush=True)

    # Save split manifests
    for split_name in ["train", "val", "test"]:
        df_split = df_all[df_all["split"] == split_name].reset_index(drop=True)
        split_path = os.path.join(SPLITS_DIR, f"{split_name}_manifest.csv")
        df_split.to_csv(split_path, index=False, encoding="utf-8")
        print(f"Saved {split_name} manifest: {split_path} ({len(df_split)} rows)", flush=True)

    total_time = time.time() - t_start
    print("=" * 70, flush=True)
    print(f"Dataset generation complete in {total_time:.1f}s ({len(df_all)} samples accepted).", flush=True)
    print(f"Class counts:\n{df_all['class_label'].value_counts()}", flush=True)
    print(f"Split counts:\n{df_all['split'].value_counts()}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_dataset_generation()
