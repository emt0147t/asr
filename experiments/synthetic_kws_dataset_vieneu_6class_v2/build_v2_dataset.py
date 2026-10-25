"""
Phase 33-35: Six-Class Vietnamese KWS Dataset Generator v2 with Idempotent Resume.
Builds the 2,000-utterance six-class dataset:
  - 1,000 Target Commands (250 TRAI, 250 PHAI, 250 LEN, 250 XUONG)
  - 500 UNKNOWN non-target speech utterances (120 distinct texts, disjoint splits)
  - 500 SILENCE ambient background utterances (4 distinct condition families)

Split distribution (68% / 16% / 16%):
  - Train: 1,360 utterances (680 targets, 340 UNKNOWN, 340 SILENCE)
  - Val:     320 utterances (160 targets,  80 UNKNOWN,  80 SILENCE)
  - Test:    320 utterances (160 targets,  80 UNKNOWN,  80 SILENCE)
Total: 2,000 utterances.
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

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_V2_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_RAW_DIR = os.path.join(DATASET_V2_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(DATASET_V2_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(DATASET_V2_DIR, "splits")
REPORTS_DIR = os.path.join(DATASET_V2_DIR, "reports")
CONFIGS_DIR = os.path.join(DATASET_V2_DIR, "configs")
MASTER_MANIFEST = os.path.join(DATASET_V2_DIR, "manifest.csv")

# 25 Verified VieNeu Preset Voices & Strict Split Allocation
SPLIT_VOICE_ALLOCATION = {
    "train": [
        "Adam", "Adam bựa", "Hải Đăng", "Minh Triết", "Minh Đức", "Mỹ Duyên",
        "Ngọc Huyền", "Ngọc Linh", "Quang Sơn", "Thanh Bình", "Thiền Tâm Đức",
        "Thiện Minh", "Thái Sơn", "Thục Đoan", "Trúc Ly", "Xuân Vĩnh", "Đoan Trang"
    ],  # 17 voices
    "val": [
        "Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"
    ],  # 4 voices
    "test": [
        "Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"
    ]   # 4 voices
}

TARGET_CLASSES = [
    {"class_label": "TRAI", "text": "Trái", "ipa": "/ʈaːj˧˥/", "tone": "sắc"},
    {"class_label": "PHAI", "text": "Phải", "ipa": "/faːj˧˩˨/", "tone": "hỏi"},
    {"class_label": "LEN", "text": "Lên", "ipa": "/len˧˧/", "tone": "ngang"},
    {"class_label": "XUONG", "text": "Xuống", "ipa": "/suəŋ˧˥/", "tone": "sắc"}
]

# 120 Disjoint Non-Target Vietnamese Texts
UNKNOWN_TEXTS = {
    "train": [
        "Dừng", "Bật", "Tắt", "Mở", "Đóng", "Chạy", "Vào", "Ra", "Trời", "Phát",
        "Trại", "Trạm", "Tròn", "Trước", "Trắng", "Phòng", "Phút", "Phần", "Phím", "Phố",
        "Làm", "Lấy", "Lúc", "Luôn", "Lại", "Lời", "Xuồng", "Xem", "Xong", "Xanh",
        "Xa", "Xuất", "Gặp", "Hẹn", "Nhớ", "Quên", "Biết", "Hiểu", "Đọc", "Viết",
        "Hát", "Đàn", "Mua", "Bán", "Ăn", "Uống", "Ngủ", "Thức", "Xin chào", "Cảm ơn",
        "Tạm biệt", "Đồng ý", "Từ chối", "Bắt đầu", "Kết thúc", "Tiếp tục", "Trợ giúp", "Cài đặt", "Thông báo", "Hệ thống"
    ],  # 60 texts
    "val": [
        "Đi", "Đứng", "Nghe", "Nói", "Hỏi", "Tìm", "Chọn", "Xoay", "Lùi", "Quay",
        "Trông", "Trẻ", "Trường", "Phim", "Phục", "Phấn", "Lược", "Lớp", "Lửa", "Xốp",
        "Xúc", "Xinh", "Đếm", "Cân", "Đo", "Xem lại", "Làm lại", "Quay lại", "Bỏ qua", "Xác nhận"
    ],  # 30 texts
    "test": [
        "Chào", "Cứu", "Hủy", "Lưu", "Xóa", "Phóng", "Thu", "Khóa", "Được", "Không",
        "Trưa", "Trà", "Trách", "Phở", "Pháo", "Phép", "Lá", "Làng", "Lạc", "Xôi",
        "Xương", "Xích", "Nhập", "Xuất bản", "Gửi đi", "Nhận tin", "Tìm kiếm", "Chia sẻ", "Tải về", "Đăng nhập"
    ]   # 30 texts
}

def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def detect_speech_bounds(audio: np.ndarray, sr: int = 16000, frame_len_ms: float = 25.0,
                         hop_len_ms: float = 10.0, threshold_ratio: float = 0.05):
    """Detects onset and active speech duration using short-time RMS energy."""
    frame_len = int(frame_len_ms * sr / 1000.0)
    hop_len = int(hop_len_ms * sr / 1000.0)
    
    num_frames = (len(audio) - frame_len) // hop_len + 1
    rms = np.zeros(num_frames)
    for i in range(num_frames):
        seg = audio[i * hop_len : i * hop_len + frame_len]
        rms[i] = np.sqrt(np.mean(seg**2))
        
    peak_rms = np.max(rms) if len(rms) > 0 else 0.0
    thresh = max(peak_rms * threshold_ratio, 1e-4)
    
    active_idx = np.where(rms >= thresh)[0]
    if len(active_idx) == 0:
        return 0.15, max(0.2, len(audio) / sr - 0.3)
        
    start_sample = max(0, active_idx[0] * hop_len - int(0.05 * sr))
    end_sample = min(len(audio), active_idx[-1] * hop_len + frame_len + int(0.05 * sr))
    
    onset_sec = round(start_sample / sr, 3)
    dur_sec = round((end_sample - start_sample) / sr, 3)
    return onset_sec, dur_sec

def process_raw_to_kws_1s(raw_audio: np.ndarray, sr_raw: int = 48000, target_sr: int = 16000):
    """Resamples raw 48k audio to 16k and formats into 1.0s window (16,000 samples)."""
    num_target_samples = int(round(len(raw_audio) * float(target_sr) / sr_raw))
    resampled = signal.resample(raw_audio, num_target_samples).astype(np.float32)
    
    peak = np.max(np.abs(resampled))
    if peak > 0.95:
        resampled = resampled * (0.95 / peak)
    elif peak < 0.20 and peak > 0.01:
        resampled = resampled * (0.75 / peak)
        
    target_len = 16000
    if len(resampled) < target_len:
        lead_pad = min(2400, (target_len - len(resampled)) // 2)
        trail_pad = target_len - len(resampled) - lead_pad
        kws_audio = np.pad(resampled, (lead_pad, trail_pad))
    else:
        kws_audio = resampled[:target_len]
        
    kws_audio = np.clip(kws_audio, -0.98, 0.98).astype(np.float32)
    onset_sec, dur_sec = detect_speech_bounds(kws_audio, target_sr)
    return kws_audio, onset_sec, dur_sec

def generate_silence_sample(family: str, seed: int, duration_samples: int = 16000):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1.0, duration_samples, endpoint=False)
    
    if family == "near_silence":
        noise = rng.normal(0, 0.001, duration_samples)
        b, a = signal.butter(2, 50.0 / 8000.0, btype="highpass")
        audio = signal.lfilter(b, a, noise)
    elif family == "hvac_hum":
        fund = 50.0 if (seed % 2 == 0) else 60.0
        hum = (
            0.015 * np.sin(2 * np.pi * fund * t + rng.uniform(0, 2*np.pi)) +
            0.008 * np.sin(2 * np.pi * 2 * fund * t + rng.uniform(0, 2*np.pi)) +
            0.004 * np.sin(2 * np.pi * 3 * fund * t + rng.uniform(0, 2*np.pi))
        )
        rumble = rng.normal(0, 0.008, duration_samples)
        b, a = signal.butter(3, 300.0 / 8000.0, btype="lowpass")
        rumble_filt = signal.lfilter(b, a, rumble)
        audio = hum + rumble_filt
    elif family == "office_ambient":
        raw = rng.normal(0, 0.012, duration_samples)
        b, a = signal.butter(2, [100.0 / 8000.0, 1200.0 / 8000.0], btype="bandpass")
        audio = signal.lfilter(b, a, raw)
    elif family == "airflow_pink":
        white = rng.normal(0, 0.015, duration_samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        audio = signal.lfilter(b, a, white)
    else:
        audio = rng.normal(0, 0.002, duration_samples)
        
    peak = np.max(np.abs(audio))
    if peak > 0.08:
        audio = audio * (0.08 / peak)
    return audio.astype(np.float32)

def build_v2_dataset():
    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    
    from vieneu import Vieneu
    print("=" * 75, flush=True)
    print("Initializing Local VieNeu-TTS v3 Turbo for Dataset v2 Build (with Resume)", flush=True)
    print("=" * 75, flush=True)
    tts = Vieneu(mode="v3turbo")
    
    records = []
    seen_hashes = set()
    
    t_start = time.time()
    
    # -------------------------------------------------------------
    # 1. Target Commands: 1,000 utterances (250 per class)
    # -------------------------------------------------------------
    print("\n--- Processing 1,000 Target Command Utterances ---", flush=True)
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
    
    for cls_idx, cinfo in enumerate(TARGET_CLASSES):
        clbl = cinfo["class_label"]
        txt = cinfo["text"]
        id_prefix = f"vnkws_v2_{clbl.lower()}"
        
        sample_num = 1
        for sp_name, voice_list in SPLIT_VOICE_ALLOCATION.items():
            for v_idx, vid in enumerate(voice_list):
                for u_idx in range(10):
                    sid = f"{id_prefix}_{sample_num:04d}"
                    sample_num += 1
                    
                    param = PARAM_SCHEDULE[u_idx]
                    base_seed = 200_000 + cls_idx * 10_000 + v_idx * 100 + u_idx
                    
                    raw_fn = f"{sid}_raw48k.wav"
                    kws_fn = f"{sid}_kws16k.wav"
                    raw_path = os.path.join(AUDIO_RAW_DIR, raw_fn)
                    kws_path = os.path.join(AUDIO_KWS_DIR, kws_fn)
                    
                    # Idempotent resume check
                    if os.path.exists(kws_path) and os.path.exists(raw_path):
                        kws_audio, _ = sf.read(kws_path, dtype="float32")
                        onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)
                        kws_hash = compute_sha256(kws_path)
                        seen_hashes.add(kws_hash)
                        records.append({
                            "sample_id": sid,
                            "dataset_version": "v2.0",
                            "class_label": clbl,
                            "class_id": cls_idx,
                            "text": txt,
                            "voice_id": vid,
                            "split": sp_name,
                            "seed": base_seed,
                            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                            "kws_onset_sec": onset_s,
                            "kws_active_speech_duration_sec": dur_s,
                            "kws_sha256": kws_hash,
                            "sample_rate": 16000,
                            "num_channels": 1,
                            "num_samples": 16000,
                            "is_target_command": True
                        })
                        continue
                        
                    # Synthesis fallback
                    accepted = False
                    retry = 0
                    while not accepted and retry < 3:
                        cur_seed = base_seed + retry * 70_000
                        np.random.seed(cur_seed)
                        try:
                            raw = tts.infer(txt, voice=vid, temperature=param["temp"],
                                            top_k=25, top_p=param["top_p"], repetition_penalty=param["rep_pen"])
                        except Exception as e:
                            retry += 1
                            continue
                            
                        if len(raw) == 0 or np.isnan(raw).any() or np.isinf(raw).any():
                            retry += 1
                            continue
                            
                        kws_audio, onset_s, dur_s = process_raw_to_kws_1s(raw, 48000, 16000)
                        sf.write(raw_path, raw, 48000, subtype="PCM_16")
                        sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                        kws_hash = compute_sha256(kws_path)
                        seen_hashes.add(kws_hash)
                        
                        records.append({
                            "sample_id": sid,
                            "dataset_version": "v2.0",
                            "class_label": clbl,
                            "class_id": cls_idx,
                            "text": txt,
                            "voice_id": vid,
                            "split": sp_name,
                            "seed": cur_seed,
                            "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                            "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                            "kws_onset_sec": onset_s,
                            "kws_active_speech_duration_sec": dur_s,
                            "kws_sha256": kws_hash,
                            "sample_rate": 16000,
                            "num_channels": 1,
                            "num_samples": 16000,
                            "is_target_command": True
                        })
                        accepted = True

    print(f"Target commands processed: {len(records)} records registered.", flush=True)

    # -------------------------------------------------------------
    # 2. UNKNOWN Class: 500 utterances (340 train, 80 val, 80 test)
    # -------------------------------------------------------------
    print("\n--- Processing 500 UNKNOWN Non-Target Utterances (120 Distinct Texts) ---", flush=True)
    unknown_splits_config = [
        {"split": "train", "target_count": 340, "voices": SPLIT_VOICE_ALLOCATION["train"], "texts": UNKNOWN_TEXTS["train"]},
        {"split": "val",   "target_count": 80,  "voices": SPLIT_VOICE_ALLOCATION["val"],   "texts": UNKNOWN_TEXTS["val"]},
        {"split": "test",  "target_count": 80,  "voices": SPLIT_VOICE_ALLOCATION["test"],  "texts": UNKNOWN_TEXTS["test"]}
    ]
    
    unk_sample_num = 1
    for u_cfg in unknown_splits_config:
        sp_name = u_cfg["split"]
        v_list = u_cfg["voices"]
        t_list = u_cfg["texts"]
        n_per_voice = u_cfg["target_count"] // len(v_list)
        
        for v_idx, vid in enumerate(v_list):
            for i in range(n_per_voice):
                sid = f"vnkws_v2_unknown_{unk_sample_num:04d}"
                unk_sample_num += 1
                
                txt = t_list[(v_idx * n_per_voice + i) % len(t_list)]
                param = PARAM_SCHEDULE[i % len(PARAM_SCHEDULE)]
                base_seed = 300_000 + (0 if sp_name=="train" else (50_000 if sp_name=="val" else 100_000)) + v_idx * 100 + i
                
                raw_fn = f"{sid}_raw48k.wav"
                kws_fn = f"{sid}_kws16k.wav"
                raw_path = os.path.join(AUDIO_RAW_DIR, raw_fn)
                kws_path = os.path.join(AUDIO_KWS_DIR, kws_fn)
                
                # Resume check
                if os.path.exists(kws_path) and os.path.exists(raw_path):
                    kws_audio, _ = sf.read(kws_path, dtype="float32")
                    onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)
                    kws_hash = compute_sha256(kws_path)
                    seen_hashes.add(kws_hash)
                    records.append({
                        "sample_id": sid,
                        "dataset_version": "v2.0",
                        "class_label": "UNKNOWN",
                        "class_id": 4,
                        "text": txt,
                        "voice_id": vid,
                        "split": sp_name,
                        "seed": base_seed,
                        "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                        "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                        "kws_onset_sec": onset_s,
                        "kws_active_speech_duration_sec": dur_s,
                        "kws_sha256": kws_hash,
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "num_samples": 16000,
                        "is_target_command": False
                    })
                    continue
                    
                accepted = False
                retry = 0
                while not accepted and retry < 3:
                    cur_seed = base_seed + retry * 50_000
                    np.random.seed(cur_seed)
                    try:
                        raw = tts.infer(txt, voice=vid, temperature=param["temp"],
                                        top_k=25, top_p=param["top_p"], repetition_penalty=param["rep_pen"])
                    except Exception as e:
                        retry += 1
                        continue
                        
                    if len(raw) == 0 or np.isnan(raw).any() or np.isinf(raw).any():
                        retry += 1
                        continue
                        
                    kws_audio, onset_s, dur_s = process_raw_to_kws_1s(raw, 48000, 16000)
                    sf.write(raw_path, raw, 48000, subtype="PCM_16")
                    sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                    kws_hash = compute_sha256(kws_path)
                    seen_hashes.add(kws_hash)
                    
                    records.append({
                        "sample_id": sid,
                        "dataset_version": "v2.0",
                        "class_label": "UNKNOWN",
                        "class_id": 4,
                        "text": txt,
                        "voice_id": vid,
                        "split": sp_name,
                        "seed": cur_seed,
                        "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                        "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                        "kws_onset_sec": onset_s,
                        "kws_active_speech_duration_sec": dur_s,
                        "kws_sha256": kws_hash,
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "num_samples": 16000,
                        "is_target_command": False
                    })
                    accepted = True
                    
                if len(records) % 50 == 0:
                    print(f"  Processed {len(records)} / 2000 total utterances...", flush=True)

    print(f"UNKNOWN processed: {len(records)} records registered.", flush=True)

    # -------------------------------------------------------------
    # 3. SILENCE Class: 500 utterances (340 train, 80 val, 80 test)
    # -------------------------------------------------------------
    print("\n--- Generating 500 SILENCE Ambient Background Utterances ---", flush=True)
    sil_families = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]
    silence_splits_config = [
        {"split": "train", "count": 340},
        {"split": "val",   "count": 80},
        {"split": "test",  "count": 80}
    ]
    
    sil_sample_num = 1
    for s_cfg in silence_splits_config:
        sp_name = s_cfg["split"]
        total_sp = s_cfg["count"]
        
        for i in range(total_sp):
            sid = f"vnkws_v2_silence_{sil_sample_num:04d}"
            sil_sample_num += 1
            
            fam = sil_families[i % len(sil_families)]
            sil_seed = 400_000 + (0 if sp_name=="train" else (50_000 if sp_name=="val" else 100_000)) + i
            
            kws_fn = f"{sid}_kws16k.wav"
            kws_path = os.path.join(AUDIO_KWS_DIR, kws_fn)
            
            if os.path.exists(kws_path):
                kws_hash = compute_sha256(kws_path)
            else:
                sil_audio = generate_silence_sample(fam, sil_seed, 16000)
                sf.write(kws_path, sil_audio, 16000, subtype="PCM_16")
                kws_hash = compute_sha256(kws_path)
                
            seen_hashes.add(kws_hash)
            records.append({
                "sample_id": sid,
                "dataset_version": "v2.0",
                "class_label": "SILENCE",
                "class_id": 5,
                "text": f"[{fam}]",
                "voice_id": f"ambient_{fam}_{sp_name}",
                "split": sp_name,
                "seed": sil_seed,
                "raw_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                "kws_onset_sec": 0.0,
                "kws_active_speech_duration_sec": 1.0,
                "kws_sha256": kws_hash,
                "sample_rate": 16000,
                "num_channels": 1,
                "num_samples": 16000,
                "is_target_command": False
            })

    # Save Master Manifest
    df_all = pd.DataFrame(records)
    df_all.to_csv(MASTER_MANIFEST, index=False, encoding="utf-8")
    print(f"\n=======================================================", flush=True)
    print(f"Master v2 manifest saved: {MASTER_MANIFEST} ({len(df_all)} records)", flush=True)
    print(f"=======================================================", flush=True)
    
    # Save Split Manifests
    train_df = df_all[df_all["split"] == "train"]
    val_df = df_all[df_all["split"] == "val"]
    test_df = df_all[df_all["split"] == "test"]
    
    train_path = os.path.join(SPLITS_DIR, "train_manifest.csv")
    val_path = os.path.join(SPLITS_DIR, "val_manifest.csv")
    test_path = os.path.join(SPLITS_DIR, "test_manifest.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"Train split saved: {train_path} ({len(train_df)} records)", flush=True)
    print(f"Val split saved:   {val_path} ({len(val_df)} records)", flush=True)
    print(f"Test split saved:  {test_path} ({len(test_df)} records)", flush=True)
    
    return df_all

if __name__ == "__main__":
    build_v2_dataset()
