"""
Phase 41: Build Hard-Negative Corpus and Locked Challenge Benchmark.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v2/

Executes:
1. Synthesizes 100 disjoint hard-negative texts (60 train, 20 val, 20 test) across 5 linguistic confusion families.
2. Formats all synthetic audio to 16 kHz 1.0s mono PCM with energy-based onset detection.
3. Generates 4 distinct SILENCE background families (near_silence, hvac_hum, office_ambient, airflow_pink).
4. Reuses the 1,000 verified target command audio files from synthetic_kws_dataset_vieneu_6class_v2.
5. Emits data/manifest.csv, data/splits/{train,val,test}_manifest.csv, and text_inventory.csv.
6. Builds continuous challenge streams (commands and non-target) with ground-truth manifests.
"""

import os
import sys
import json
import time
import hashlib
import shutil
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")
AUDIO_RAW_DIR = os.path.join(DATA_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(DATA_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
STREAMS_DIR = os.path.join(DATA_DIR, "continuous_streams")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

HISTORICAL_V2_DATA = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2")

# 25 Voices strictly allocated
SPLIT_VOICE_ALLOCATION = {
    "train": [
        "Adam", "Adam bựa", "Hải Đăng", "Minh Triết", "Minh Đức", "Mỹ Duyên",
        "Ngọc Huyền", "Ngọc Linh", "Quang Sơn", "Thanh Bình", "Thiền Tâm Đức",
        "Thiện Minh", "Thái Sơn", "Thục Đoan", "Trúc Ly", "Xuân Vĩnh", "Đoan Trang"
    ],
    "val": ["Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"],
    "test": ["Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"]
}

# 100 Strictly Disjoint Non-Target Vietnamese Texts Across 5 Confusion Families
HARD_NEGATIVE_TEXTS = {
    "train": [
        # TRAI confusions (12)
        {"text": "trai", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Same segment /ʈaːj/, tone ngang vs sắc"},
        {"text": "trài", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Same segment /ʈaːj/, tone huyền vs sắc"},
        {"text": "trải", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Same segment /ʈaːj/, tone hỏi vs sắc"},
        {"text": "trại", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Same segment /ʈaːj/, tone nặng vs sắc"},
        {"text": "chái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial affricate /tɕ/ vs retroflex /ʈ/"},
        {"text": "sái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial sibilant /s/ vs retroflex /ʈ/"},
        {"text": "trát", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Final stop /t/ vs glide /j/"},
        {"text": "tráo", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Final back glide /w/ vs front /j/"},
        {"text": "trách", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Palatal stop /c/ coda"},
        {"text": "trang", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Velar nasal /ŋ/ coda"},
        {"text": "trái cây", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Compound phrase embedding target"},
        {"text": "bên trái", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Preceding unstressed syllable"},

        # PHAI confusions (12)
        {"text": "phai", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Same segment /faːj/, tone ngang vs hỏi"},
        {"text": "phài", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Same segment /faːj/, tone huyền vs hỏi"},
        {"text": "phái", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Same segment /faːj/, tone sắc vs hỏi"},
        {"text": "phại", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Same segment /faːj/, tone nặng vs hỏi"},
        {"text": "thải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial aspirated /tʰ/ vs labiodental /f/"},
        {"text": "thái", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial aspirated /tʰ/ + sắc"},
        {"text": "khải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial velar fricative /x/ vs /f/"},
        {"text": "phát", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Final coronal stop /t/ vs glide /j/"},
        {"text": "phạt", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Final stop /t/ + nặng"},
        {"text": "pháp", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Final bilabial stop /p/ vs glide /j/"},
        {"text": "phải chăng", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Phrase context embedding keyword"},
        {"text": "bên phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Preceding context word"},

        # LEN confusions (12)
        {"text": "lèn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Same segment /len/, tone huyền vs ngang"},
        {"text": "lén", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Same segment /len/, tone sắc vs ngang"},
        {"text": "lẻn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Same segment /len/, tone hỏi vs ngang"},
        {"text": "lẹn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Same segment /len/, tone nặng vs ngang"},
        {"text": "nén", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial nasal /n/ vs lateral /l/"},
        {"text": "khen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial velar /x/ vs lateral /l/"},
        {"text": "chen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial affricate /tɕ/ vs lateral /l/"},
        {"text": "đen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial implosive /ɗ/ vs /l/"},
        {"text": "lớp", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Vowel /ə/ + stop /p/"},
        {"text": "lực", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Vowel /ɨ/ + stop /k/"},
        {"text": "đi lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Phrase embedding command"},
        {"text": "lên đường", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Command as initial word in phrase"},

        # XUONG confusions (12)
        {"text": "xuồng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Same segment /suəŋ/, tone huyền vs sắc"},
        {"text": "xuổng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Same segment /suəŋ/, tone hỏi vs sắc"},
        {"text": "xướng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Vowel /ɨə/ vs /uə/, tone sắc"},
        {"text": "sương", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial sibilant + diphthong /ɨəŋ/"},
        {"text": "sườn", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Diphthong /ɨə/ + alveolar nasal /n/"},
        {"text": "hướng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial glottal /h/ vs dental /s/"},
        {"text": "buông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial bilabial /ɓ/ vs /s/"},
        {"text": "chuông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial affricate /tɕ/ vs /s/"},
        {"text": "xuất", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Short vowel /ə/ + stop /t/"},
        {"text": "xong", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Short back rounded vowel /awŋ/"},
        {"text": "đi xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Phrase embedding command"},
        {"text": "xuống dốc", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Command followed by dốc"},

        # Neutral distractors (12)
        {"text": "bật", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday control keyword"},
        {"text": "tắt", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday control keyword"},
        {"text": "mở", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday control keyword"},
        {"text": "đóng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday control keyword"},
        {"text": "chạy", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday action verb"},
        {"text": "dừng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday action verb"},
        {"text": "nghe", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "nói", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "hát", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "đọc", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "viết", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "xin chào", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common greeting phrase"}
    ],
    "val": [
        # Disjoint Val hard negatives (20)
        {"text": "trãi", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Same segment /ʈaːj/, tone ngã vs sắc"},
        {"text": "mái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial nasal /m/ vs /ʈ/"},
        {"text": "tròn", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Back vowel /ɔ/ + alveolar nasal /n/"},
        {"text": "trại giam", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Compound phrase starting with trại"},

        {"text": "phãi", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Same segment /faːj/, tone ngã vs hỏi"},
        {"text": "mải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial nasal /m/ vs /f/"},
        {"text": "phay", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Short vowel /ăj/ vs long /aːj/"},
        {"text": "không phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Negative prefix + phải"},

        {"text": "lẽn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Same segment /len/, tone ngã vs ngang"},
        {"text": "quên", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Labial-velar /kw/ + /en/"},
        {"text": "lưng", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Unrounded high vowel /ɨ/ + /ŋ/"},
        {"text": "tiến lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Compound phrase ending in lên"},

        {"text": "xuỗng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Same segment /suəŋ/, tone ngã vs sắc"},
        {"text": "hưởng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial /h/ + hỏi vs /s/ + sắc"},
        {"text": "xanh", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Short front vowel /aɲ/"},
        {"text": "quê hương", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Phrase ending in rhyming hương"},

        {"text": "xem", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday verb"},
        {"text": "tìm", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday verb"},
        {"text": "chọn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday verb"},
        {"text": "bắt đầu", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday command"}
    ],
    "test": [
        # Disjoint Test hard negatives (20)
        {"text": "lái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial /l/ vs /ʈ/ with sắc"},
        {"text": "trước", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Diphthong /ɨə/ + stop /k/"},
        {"text": "trắng", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Short vowel /ă/ + /ŋ/"},
        {"text": "con trai", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Phrase ending in trai"},

        {"text": "bải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial implosive /ɓ/ vs /f/"},
        {"text": "pháo", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Vowel /aː/ + glide /w/"},
        {"text": "phần", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Short central vowel /ə/ + /n/"},
        {"text": "phần mềm", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Compound software term"},

        {"text": "nem", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /n/ + bilabial nasal /m/"},
        {"text": "leng", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Velar nasal /ŋ/ vs alveolar /n/"},
        {"text": "lòng", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Back vowel /awŋ/"},
        {"text": "bước lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Phrase context step up"},

        {"text": "xưởng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Same base with tone hỏi"},
        {"text": "xốp", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Vowel /o/ + stop /p/"},
        {"text": "xúc", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "High vowel /u/ + stop /k/"},
        {"text": "nhà xưởng", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Factory compound phrase"},

        {"text": "hỏi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Everyday speech word"},
        {"text": "lưu", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Save command word"},
        {"text": "xóa", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Delete command word"},
        {"text": "cảm ơn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Thank you phrase"}
    ]
}


def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def detect_speech_bounds(audio: np.ndarray, sr: int = 16000, frame_len_ms: float = 25.0,
                         hop_len_ms: float = 10.0, threshold_ratio: float = 0.05):
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


def format_to_1s(raw_audio: np.ndarray, sr_raw: int = 48000, target_sr: int = 16000):
    num_target_samples = int(round(len(raw_audio) * float(target_sr) / sr_raw))
    resampled = signal.resample(raw_audio, num_target_samples).astype(np.float32)
    peak = np.max(np.abs(resampled))
    if peak > 0.95:
        resampled = resampled * (0.95 / peak)
    elif 0.01 < peak < 0.20:
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
            0.015 * np.sin(2 * np.pi * fund * t + rng.uniform(0, 2 * np.pi)) +
            0.008 * np.sin(2 * np.pi * 2 * fund * t + rng.uniform(0, 2 * np.pi)) +
            0.004 * np.sin(2 * np.pi * 3 * fund * t + rng.uniform(0, 2 * np.pi))
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


def main():
    print("=" * 80)
    print("Phase 41: Building Vietnamese KWS Hard-Negative Dataset & Benchmark v2")
    print("=" * 80)

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(STREAMS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Initialize Vieneu TTS engine
    print("\n--- Initializing VieNeu TTS Engine ---")
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")

    records = []
    text_inventory_records = []
    text_counter = 1

    # -----------------------------------------------------------------
    # Part A: Target Commands (1,000 utterances from historical v2)
    # -----------------------------------------------------------------
    print("\n--- Incorporating 1,000 Approved Target Commands ---")
    hist_manifest = pd.read_csv(os.path.join(HISTORICAL_V2_DATA, "manifest.csv"))
    target_hist = hist_manifest[hist_manifest["is_target_command"] == True].copy()
    assert len(target_hist) == 1000, f"Expected 1,000 target commands, found {len(target_hist)}"

    for _, row in target_hist.iterrows():
        src_kws = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        src_raw = os.path.join(BASE_DIR, row["raw_wav_path"]) if not os.path.isabs(row["raw_wav_path"]) else row["raw_wav_path"]

        sid = row["sample_id"]
        dst_kws = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")
        dst_raw = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")

        if not os.path.exists(dst_kws):
            shutil.copy2(src_kws, dst_kws)
        if not os.path.exists(dst_raw) and os.path.exists(src_raw):
            shutil.copy2(src_raw, dst_raw)

        rec = row.to_dict()
        rec["dataset_version"] = "research_v2.0"
        rec["kws_wav_path"] = os.path.relpath(dst_kws, BASE_DIR).replace("\\", "/")
        rec["raw_wav_path"] = os.path.relpath(dst_raw, BASE_DIR).replace("\\", "/")
        rec["confusion_family"] = "target_command"
        records.append(rec)

    # -----------------------------------------------------------------
    # Part B: Synthesize Hard-Negative UNKNOWN Corpus
    # -----------------------------------------------------------------
    print("\n--- Synthesizing Hard-Negative Corpus Across Splits ---")
    # Voice-Text schedules:
    # Train: 60 texts. 17 voices -> 60 texts spoken across voices (approx 340 utterances, 20 per voice)
    # Val: 20 texts. 4 voices -> each voice speaks all 20 texts = 80 utterances
    # Test: 20 texts. 4 voices -> each voice speaks all 20 texts = 80 utterances

    for split in ["train", "val", "test"]:
        voice_list = SPLIT_VOICE_ALLOCATION[split]
        text_list = HARD_NEGATIVE_TEXTS[split]
        print(f"\nProcessing {split.upper()} split: {len(text_list)} texts across {len(voice_list)} voices...")

        if split == "train":
            # 60 texts allocated across 17 voices (~20 per voice, total 340)
            target_count = 340
            t_idx = 0
            for i in range(target_count):
                vid = voice_list[i % len(voice_list)]
                t_info = text_list[t_idx % len(text_list)]
                t_idx += 1

                sid = f"hn_unk_{split}_{i+1:04d}"
                raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                if not os.path.exists(kws_path):
                    raw_audio = tts.infer(t_info["text"], voice=vid, temperature=0.75, top_p=0.92, repetition_penalty=1.20)
                    kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                    sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                    sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                else:
                    kws_audio, _ = sf.read(kws_path, dtype="float32")
                    onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

                kws_h = compute_sha256(kws_path)
                rec = {
                    "sample_id": sid,
                    "dataset_version": "research_v2.0",
                    "class_label": "UNKNOWN",
                    "class_id": 4,
                    "text": t_info["text"],
                    "voice_id": vid,
                    "split": split,
                    "seed": 400_000 + i,
                    "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                    "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                    "kws_onset_sec": onset_s,
                    "kws_active_speech_duration_sec": dur_s,
                    "kws_sha256": kws_h,
                    "sample_rate": 16000,
                    "num_channels": 1,
                    "num_samples": 16000,
                    "is_target_command": False,
                    "confusion_family": t_info["confusion_family"],
                    "target_keyword": t_info["target"],
                    "linguistic_rationale": t_info["rationale"]
                }
                records.append(rec)
                text_inventory_records.append({
                    "text_id": f"txt_{split}_{text_counter:04d}",
                    "text": t_info["text"],
                    "split": split,
                    "intended_label": "UNKNOWN",
                    "confusion_family": t_info["confusion_family"],
                    "source": "VieNeu-TTS v3turbo",
                    "tts_voice_id": vid
                })
                text_counter += 1
                if (i + 1) % 50 == 0:
                    print(f"[{i+1}/{target_count}] Generated train hard negatives...")
        else:
            # Val / Test: 4 voices x 20 texts = 80 utterances per split
            i = 0
            for vid in voice_list:
                for t_info in text_list:
                    i += 1
                    sid = f"hn_unk_{split}_{i:04d}"
                    raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                    kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                    if not os.path.exists(kws_path):
                        raw_audio = tts.infer(t_info["text"], voice=vid, temperature=0.75, top_p=0.92, repetition_penalty=1.20)
                        kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                        sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                        sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                    else:
                        kws_audio, _ = sf.read(kws_path, dtype="float32")
                        onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

                    kws_h = compute_sha256(kws_path)
                    rec = {
                        "sample_id": sid,
                        "dataset_version": "research_v2.0",
                        "class_label": "UNKNOWN",
                        "class_id": 4,
                        "text": t_info["text"],
                        "voice_id": vid,
                        "split": split,
                        "seed": 500_000 + i if split == "val" else 600_000 + i,
                        "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                        "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                        "kws_onset_sec": onset_s,
                        "kws_active_speech_duration_sec": dur_s,
                        "kws_sha256": kws_h,
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "num_samples": 16000,
                        "is_target_command": False,
                        "confusion_family": t_info["confusion_family"],
                        "target_keyword": t_info["target"],
                        "linguistic_rationale": t_info["rationale"]
                    }
                    records.append(rec)
                    text_inventory_records.append({
                        "text_id": f"txt_{split}_{text_counter:04d}",
                        "text": t_info["text"],
                        "split": split,
                        "intended_label": "UNKNOWN",
                        "confusion_family": t_info["confusion_family"],
                        "source": "VieNeu-TTS v3turbo",
                        "tts_voice_id": vid
                    })
                    text_counter += 1
            print(f"Completed {split.upper()} split: 80 hard negatives.")

    # -----------------------------------------------------------------
    # Part C: Generate SILENCE Background Utterances
    # -----------------------------------------------------------------
    print("\n--- Generating Fresh SILENCE Background Utterances ---")
    # Balanced with target/unknown: 340 train, 80 val, 80 test
    silence_counts = {"train": 340, "val": 80, "test": 80}
    families = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]

    for split, count in silence_counts.items():
        for i in range(count):
            fam = families[i % len(families)]
            sid = f"hn_sil_{split}_{i+1:04d}"
            raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
            kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

            base_seed = 700_000 + i if split == "train" else (800_000 + i if split == "val" else 900_000 + i)
            if not os.path.exists(kws_path):
                sil_audio = generate_silence_sample(fam, base_seed, 16000)
                sf.write(kws_path, sil_audio, 16000, subtype="PCM_16")
                sf.write(raw_path, np.repeat(sil_audio, 3), 48000, subtype="PCM_16")
            else:
                sil_audio, _ = sf.read(kws_path, dtype="float32")

            kws_h = compute_sha256(kws_path)
            rec = {
                "sample_id": sid,
                "dataset_version": "research_v2.0",
                "class_label": "SILENCE",
                "class_id": 5,
                "text": "[SILENCE]",
                "voice_id": fam,
                "split": split,
                "seed": base_seed,
                "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                "kws_onset_sec": 0.0,
                "kws_active_speech_duration_sec": 0.0,
                "kws_sha256": kws_h,
                "sample_rate": 16000,
                "num_channels": 1,
                "num_samples": 16000,
                "is_target_command": False,
                "confusion_family": fam,
                "target_keyword": "NONE",
                "linguistic_rationale": "Acoustic background simulation"
            }
            records.append(rec)

    # -----------------------------------------------------------------
    # Part D: Master Manifest and Split Generation
    # -----------------------------------------------------------------
    master_df = pd.DataFrame(records)
    master_csv = os.path.join(DATA_DIR, "manifest.csv")
    master_df.to_csv(master_csv, index=False, encoding="utf-8")
    print(f"\nMaster manifest saved to {master_csv} (Total: {len(master_df)} rows).")

    # Text Inventory
    text_inv_df = pd.DataFrame(text_inventory_records)
    text_inv_csv = os.path.join(DATA_DIR, "text_inventory.csv")
    text_inv_df.to_csv(text_inv_csv, index=False, encoding="utf-8")
    print(f"Text inventory saved to {text_inv_csv} (Total: {len(text_inv_df)} rows).")

    # Splits
    for sp in ["train", "val", "test"]:
        sp_df = master_df[master_df["split"] == sp].copy()
        sp_path = os.path.join(SPLITS_DIR, f"{sp}_manifest.csv")
        sp_df.to_csv(sp_path, index=False, encoding="utf-8")
        print(f"Split '{sp}': {len(sp_df)} rows ({sp_path})")

    # -----------------------------------------------------------------
    # Part E: Build Continuous Challenge Streams
    # -----------------------------------------------------------------
    print("\n--- Constructing Continuous Challenge Streams ---")
    # For validation and test:
    # 1. Command stream: 160 targets + distractors
    # 2. Non-target challenge stream: 80 hard-negative utterances + varied background intervals

    for sp in ["val", "test"]:
        sp_df = master_df[master_df["split"] == sp].copy()
        tgt_items = sp_df[sp_df["is_target_command"] == True].sample(frac=1.0, random_state=42).to_dict("records")
        hn_items = sp_df[sp_df["class_label"] == "UNKNOWN"].sample(frac=1.0, random_state=42).to_dict("records")
        sil_items = sp_df[sp_df["class_label"] == "SILENCE"].sample(frac=1.0, random_state=42).to_dict("records")

        # --- Continuous Command Stream ---
        # Stitches 160 targets with random gaps of 1.0 - 3.5s (average ~4.2s per event -> ~670s = ~11 minutes)
        rng = np.random.RandomState(100 if sp == "val" else 200)
        cmd_audio_list = []
        cmd_gt_records = []
        cur_sample = 0

        # Lead in silence (2s)
        lead_sil = rng.normal(0, 0.001, 32000).astype(np.float32)
        cmd_audio_list.append(lead_sil)
        cur_sample += len(lead_sil)

        for idx, item in enumerate(tgt_items):
            audio, sr = sf.read(os.path.join(BASE_DIR, item["kws_wav_path"]), dtype="float32")
            onset_samp = cur_sample + int(item["kws_onset_sec"] * 16000)
            end_samp = cur_sample + 16000

            cmd_audio_list.append(audio)
            cmd_gt_records.append({
                "stream": f"{sp}_stream_commands",
                "event_type": "target_command",
                "class_label": item["class_label"],
                "start_time_sec": round(cur_sample / 16000.0, 3),
                "end_time_sec": round(end_samp / 16000.0, 3),
                "onset_sec": round(onset_samp / 16000.0, 3),
                "sample_id": item["sample_id"],
                "text": item["text"],
                "voice_id": item["voice_id"]
            })
            cur_sample += len(audio)

            # Insert gap (1.5 - 4.0s) occasionally with distractor or background
            gap_dur = rng.uniform(1.5, 4.0)
            gap_samples = int(gap_dur * 16000)
            if idx % 3 == 0 and len(hn_items) > 0:
                dist = hn_items[idx % len(hn_items)]
                dist_audio, _ = sf.read(os.path.join(BASE_DIR, dist["kws_wav_path"]), dtype="float32")
                # Pad to gap
                if len(dist_audio) < gap_samples:
                    gap_pad = np.pad(dist_audio, (0, gap_samples - len(dist_audio)))
                else:
                    gap_pad = dist_audio[:gap_samples]
                cmd_audio_list.append(gap_pad)
            else:
                gap_noise = rng.normal(0, 0.002, gap_samples).astype(np.float32)
                cmd_audio_list.append(gap_noise)
            cur_sample += gap_samples

        full_cmd_wav = np.concatenate(cmd_audio_list).astype(np.float32)
        cmd_wav_path = os.path.join(STREAMS_DIR, f"{sp}_stream_continuous_commands.wav")
        cmd_gt_path = os.path.join(STREAMS_DIR, f"{sp}_stream_continuous_commands_ground_truth.json")
        sf.write(cmd_wav_path, full_cmd_wav, 16000, subtype="PCM_16")
        with open(cmd_gt_path, "w", encoding="utf-8") as f:
            json.dump(cmd_gt_records, f, indent=2, ensure_ascii=False)
        print(f"Created {sp} command stream: {len(full_cmd_wav)/16000:.1f}s ({len(full_cmd_wav)/16000/3600:.4f}h), {len(cmd_gt_records)} targets.")

        # --- Continuous Non-Target Challenge Stream ---
        # Stitches 80 hard-negative utterances with varied realistic background intervals
        # Gaps of 4.0 - 12.0s to ensure thorough background and phonetic exposure
        nt_audio_list = []
        nt_gt_records = []
        cur_sample = 0

        # Lead-in (3s)
        lead_nt = rng.normal(0, 0.001, 48000).astype(np.float32)
        nt_audio_list.append(lead_nt)
        cur_sample += len(lead_nt)

        for idx, item in enumerate(hn_items):
            audio, sr = sf.read(os.path.join(BASE_DIR, item["kws_wav_path"]), dtype="float32")
            onset_samp = cur_sample + int(item["kws_onset_sec"] * 16000)
            end_samp = cur_sample + 16000

            nt_audio_list.append(audio)
            nt_gt_records.append({
                "stream": f"{sp}_stream_nontarget",
                "event_type": "distractor_speech",
                "class_label": "UNKNOWN",
                "start_time_sec": round(cur_sample / 16000.0, 3),
                "end_time_sec": round(end_samp / 16000.0, 3),
                "onset_sec": round(onset_samp / 16000.0, 3),
                "sample_id": item["sample_id"],
                "text": item["text"],
                "voice_id": item["voice_id"],
                "confusion_family": item.get("confusion_family", "unknown")
            })
            cur_sample += len(audio)

            # Insert varied background interval (4.0s - 12.0s)
            gap_dur = rng.uniform(4.0, 12.0)
            gap_samples = int(gap_dur * 16000)
            fam = families[idx % len(families)]
            gap_bg = generate_silence_sample(fam, 1000 + idx, gap_samples)
            nt_audio_list.append(gap_bg)
            cur_sample += gap_samples

        full_nt_wav = np.concatenate(nt_audio_list).astype(np.float32)
        nt_wav_path = os.path.join(STREAMS_DIR, f"{sp}_stream_continuous_nontarget.wav")
        nt_gt_path = os.path.join(STREAMS_DIR, f"{sp}_stream_continuous_nontarget_ground_truth.json")
        sf.write(nt_wav_path, full_nt_wav, 16000, subtype="PCM_16")
        with open(nt_gt_path, "w", encoding="utf-8") as f:
            json.dump(nt_gt_records, f, indent=2, ensure_ascii=False)
        print(f"Created {sp} non-target stream: {len(full_nt_wav)/16000:.1f}s ({len(full_nt_wav)/16000/3600:.4f}h), {len(nt_gt_records)} distractor events.")

    print("\nDataset generation and stream synthesis completed successfully.")


if __name__ == "__main__":
    main()
