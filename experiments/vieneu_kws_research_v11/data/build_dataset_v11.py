#!/usr/bin/env python3
"""
Stage B: Synthesize and Assemble Bounded Dataset v11.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v11/data/

Synthesizes:
1. Locked Test Split (640 clips: 160 commands + 480 non-targets).
2. Validation Split (400 clips: 160 commands + 240 non-targets).
3. Training Split (2000 clips: 800 commands + 1200 non-targets).
"""

import os
import sys
import json
import csv
import time
import hashlib
import random
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V11_DATA_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11" / "data"
AUDIO_DIR = V11_DATA_DIR / "audio" / "kws_16k"
SPLITS_DIR = V11_DATA_DIR / "splits"

# Historical sources
V6_SPLITS = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v6" / "data" / "splits"
V8_KWS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v8" / "data" / "audio" / "kws_16k"

from prepare_v11_lexicon import (
    TEST_NON_TARGET_TEXTS,
    VAL_NON_TARGET_TEXTS,
    TRAIN_HARD_NEGATIVE_TEXTS
)

TEST_VOICES = ["Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"]
VAL_VOICES = ["Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"]
TRAIN_VOICES = [
    "Adam", "Minh Triết", "Mỹ Duyên", "Ngọc Huyền",
    "Quang Sơn", "Thanh Bình", "Thái Sơn", "Thục Đoan"
]

TARGET_COMMANDS = {
    "TRAI": ("trái", 0),
    "PHAI": ("phải", 1),
    "LEN": ("lên", 2),
    "XUONG": ("xuống", 3)
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def detect_speech_bounds(audio_16k: np.ndarray, sr: int = 16000):
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = (len(audio_16k) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return 0.0, len(audio_16k) / sr

    frames = np.array([audio_16k[i * hop_len : i * hop_len + frame_len] for i in range(num_frames)])
    energy = np.sum(frames ** 2, axis=1)
    max_e = np.max(energy) if len(energy) > 0 else 0
    if max_e < 1e-7:
        return 0.0, len(audio_16k) / sr

    thresh = max_e * 0.01
    active = np.where(energy > thresh)[0]
    if len(active) == 0:
        return 0.0, len(audio_16k) / sr

    onset_s = round(float(active[0] * hop_len) / sr, 3)
    offset_s = round(float((active[-1] * hop_len) + frame_len) / sr, 3)
    return onset_s, round(offset_s - onset_s, 3)


def format_to_1s(audio: np.ndarray, orig_sr: int, target_sr: int = 16000):
    if orig_sr != target_sr:
        num_target_samples = int(round(len(audio) * float(target_sr) / orig_sr))
        audio = signal.resample(audio, num_target_samples)

    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 10.0 ** (-1.0 / 20.0)
        audio = audio * (target_peak / peak)

    target_samples = 16000
    if len(audio) < target_samples:
        pad_total = target_samples - len(audio)
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        audio = np.pad(audio, (pad_front, pad_back), mode="constant", constant_values=0.0)
    elif len(audio) > target_samples:
        frame_len = int(0.025 * target_sr)
        hop_len = int(0.010 * target_sr)
        num_frames = (len(audio) - frame_len) // hop_len + 1
        energies = [np.sum(audio[i * hop_len : i * hop_len + frame_len] ** 2) for i in range(num_frames)]
        center_frame = int(np.argmax(energies))
        center_sample = center_frame * hop_len + frame_len // 2
        start = max(0, center_sample - target_samples // 2)
        if start + target_samples > len(audio):
            start = len(audio) - target_samples
        audio = audio[start : start + target_samples]

    onset_s, dur_s = detect_speech_bounds(audio, target_sr)
    return audio.astype(np.float32), onset_s, dur_s


def generate_synthetic_ambient(family: str, seed: int, duration_samples: int = 16000) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if family == "near_silence":
        audio = rng.normal(0, 0.0008, duration_samples)
    elif family == "hvac_hum":
        t = np.arange(duration_samples) / 16000.0
        audio = (
            0.025 * np.sin(2 * np.pi * 50 * t) +
            0.015 * np.sin(2 * np.pi * 100 * t) +
            0.008 * np.sin(2 * np.pi * 150 * t) +
            rng.normal(0, 0.004, duration_samples)
        )
    elif family == "office_ambient":
        audio = rng.normal(0, 0.012, duration_samples)
        b, a = signal.butter(4, [200.0 / 8000.0, 3500.0 / 8000.0], btype="band")
        audio = signal.lfilter(b, a, audio)
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
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE B: SYNTHESIZING & ASSEMBLING BOUNDED KWS DATASET V11")
    print("=" * 80)

    # Initialize Vieneu
    print("Initializing Vieneu TTS Engine (mode='v3turbo')...")
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    print("[OK] TTS Engine ready.")

    # ---------------------------------------------------------
    # PART 1: LOCKED TEST SPLIT (640 CLIPS)
    # ---------------------------------------------------------
    print("\n--- Generating Locked Test Split (640 clips) ---")
    test_records = []

    # 1.1: 160 Target Commands (40 per keyword across 4 held-out test voices)
    # Verified reuse from locked v8 test audio
    for kw, (text, cid) in TARGET_COMMANDS.items():
        for v_idx, vid in enumerate(TEST_VOICES):
            for r in range(10):
                sid = f"v11_cmd_test_{kw.lower()}_{v_idx:02d}_{r:02d}"
                # Check if v8 audio exists to preserve exact verified acoustic signal
                v8_src = V8_KWS_DIR / f"v8_cmd_test_{kw.lower()}_{v_idx:02d}_{r:02d}_kws16k.wav"
                dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"

                if v8_src.exists() and not dst_wav.exists():
                    audio_data, _ = sf.read(v8_src, dtype="float32")
                    sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
                elif not dst_wav.exists():
                    raw_audio = tts.infer(text, voice=vid, temperature=0.72, top_p=0.92)
                    audio_data, _, _ = format_to_1s(raw_audio, 48000, 16000)
                    sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")

                audio_data, _ = sf.read(dst_wav, dtype="float32")
                onset_s, dur_s = detect_speech_bounds(audio_data, 16000)
                sha = compute_sha256(dst_wav)

                test_records.append({
                    "sample_id": sid,
                    "dataset_version": "vieneu_kws_v11.0_bounded",
                    "split": "test",
                    "voice_id": vid,
                    "canonical_class": kw,
                    "class_id": cid,
                    "bounded_label": kw,
                    "bounded_class_id": cid,
                    "confusion_family": "target_command",
                    "transcription": text,
                    "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "duration_samples": 16000,
                    "sample_rate": 16000,
                    "speech_onset_sec": onset_s,
                    "speech_duration_sec": dur_s,
                    "sha256": sha,
                    "is_historical_reuse": v8_src.exists(),
                    "historical_source_id": v8_src.name if v8_src.exists() else ""
                })
    print(f"Test commands assembled: {len(test_records)}/160")

    # 1.2: 400 Fresh Speech Non-Target Clips (100 texts x 4 held-out test voices)
    for idx, item in enumerate(TEST_NON_TARGET_TEXTS):
        text = item["text"]
        fam = item["confusion_family"]
        tgt = item["target"]
        for v_idx, vid in enumerate(TEST_VOICES):
            sid = f"v11_unk_test_{fam[:4]}_{idx:03d}_{v_idx:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"

            if not dst_wav.exists():
                raw_audio = tts.infer(text, voice=vid, temperature=0.75, top_p=0.90)
                audio_data, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            else:
                audio_data, _ = sf.read(dst_wav, dtype="float32")
                onset_s, dur_s = detect_speech_bounds(audio_data, 16000)

            sha = compute_sha256(dst_wav)
            test_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "test",
                "voice_id": vid,
                "canonical_class": "UNKNOWN",
                "class_id": 4,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": text,
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": onset_s,
                "speech_duration_sec": dur_s,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
        if (idx + 1) % 25 == 0:
            print(f"Synthesized {idx + 1}/100 test speech texts across 4 voices...")

    # 1.3: 80 Ambient / Silence Clips (20 per family)
    ambient_families = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]
    for fam in ambient_families:
        for r in range(20):
            sid = f"v11_sil_test_{fam}_{r:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
            if not dst_wav.exists():
                audio_data = generate_synthetic_ambient(fam, seed=11000 + len(test_records))
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            sha = compute_sha256(dst_wav)
            test_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "test",
                "voice_id": fam,
                "canonical_class": "SILENCE",
                "class_id": 5,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": "[silence]",
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": 0.0,
                "speech_duration_sec": 1.0,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
    print(f"Total locked test records: {len(test_records)} (Expected 640)")

    # ---------------------------------------------------------
    # PART 2: VALIDATION SPLIT (400 CLIPS)
    # ---------------------------------------------------------
    print("\n--- Generating Validation Split (400 clips) ---")
    val_records = []

    # 2.1: 160 Target Commands (40 per keyword across 4 val voices)
    for kw, (text, cid) in TARGET_COMMANDS.items():
        for v_idx, vid in enumerate(VAL_VOICES):
            for r in range(10):
                sid = f"v11_cmd_val_{kw.lower()}_{v_idx:02d}_{r:02d}"
                dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
                if not dst_wav.exists():
                    raw_audio = tts.infer(text, voice=vid, temperature=0.72, top_p=0.92)
                    audio_data, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                    sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
                else:
                    audio_data, _ = sf.read(dst_wav, dtype="float32")
                    onset_s, dur_s = detect_speech_bounds(audio_data, 16000)
                sha = compute_sha256(dst_wav)
                val_records.append({
                    "sample_id": sid,
                    "dataset_version": "vieneu_kws_v11.0_bounded",
                    "split": "val",
                    "voice_id": vid,
                    "canonical_class": kw,
                    "class_id": cid,
                    "bounded_label": kw,
                    "bounded_class_id": cid,
                    "confusion_family": "target_command",
                    "transcription": text,
                    "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "duration_samples": 16000,
                    "sample_rate": 16000,
                    "speech_onset_sec": onset_s,
                    "speech_duration_sec": dur_s,
                    "sha256": sha,
                    "is_historical_reuse": False,
                    "historical_source_id": ""
                })

    # 2.2: 200 Fresh Speech Non-Target Clips (50 texts x 4 val voices)
    for idx, item in enumerate(VAL_NON_TARGET_TEXTS):
        text = item["text"]
        fam = item["confusion_family"]
        tgt = item["target"]
        for v_idx, vid in enumerate(VAL_VOICES):
            sid = f"v11_unk_val_{fam[:4]}_{idx:03d}_{v_idx:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
            if not dst_wav.exists():
                raw_audio = tts.infer(text, voice=vid, temperature=0.75, top_p=0.90)
                audio_data, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            else:
                audio_data, _ = sf.read(dst_wav, dtype="float32")
                onset_s, dur_s = detect_speech_bounds(audio_data, 16000)
            sha = compute_sha256(dst_wav)
            val_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "val",
                "voice_id": vid,
                "canonical_class": "UNKNOWN",
                "class_id": 4,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": text,
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": onset_s,
                "speech_duration_sec": dur_s,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
        if (idx + 1) % 25 == 0:
            print(f"Synthesized {idx + 1}/50 val speech texts across 4 val voices...")

    # 2.3: 40 Ambient / Silence Clips (10 per family)
    for fam in ambient_families:
        for r in range(10):
            sid = f"v11_sil_val_{fam}_{r:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
            if not dst_wav.exists():
                audio_data = generate_synthetic_ambient(fam, seed=21000 + len(val_records))
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            sha = compute_sha256(dst_wav)
            val_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "val",
                "voice_id": fam,
                "canonical_class": "SILENCE",
                "class_id": 5,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": "[silence]",
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": 0.0,
                "speech_duration_sec": 1.0,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
    print(f"Total validation records: {len(val_records)} (Expected 400)")

    # ---------------------------------------------------------
    # PART 3: TRAINING SPLIT (2000 CLIPS)
    # ---------------------------------------------------------
    print("\n--- Generating Training Split (2000 clips) ---")
    train_records = []

    # 3.1: 800 Target Commands (200 per keyword)
    # Ingest historical v6 train commands (170 per keyword = 680 clips)
    v6_train_csv = V6_SPLITS / "train_manifest.csv"
    with open(v6_train_csv, "r", encoding="utf-8") as f:
        v6_train_rows = list(csv.DictReader(f))

    v6_cmds_by_kw = {kw: [r for r in v6_train_rows if r["class_label"] == kw] for kw in TARGET_COMMANDS}
    for kw, (text, cid) in TARGET_COMMANDS.items():
        # First 170 from v6
        for r in v6_cmds_by_kw[kw][:170]:
            train_records.append({
                "sample_id": f"v11_cmd_tr_{r['sample_id']}",
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "train",
                "voice_id": r["voice_id"],
                "canonical_class": kw,
                "class_id": cid,
                "bounded_label": kw,
                "bounded_class_id": cid,
                "confusion_family": "target_command",
                "transcription": text,
                "kws_wav_path": r["kws_wav_path"],
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": float(r.get("kws_onset_sec", 0.0)),
                "speech_duration_sec": float(r.get("kws_active_speech_duration_sec", 1.0)),
                "sha256": r.get("kws_sha256", ""),
                "is_historical_reuse": True,
                "historical_source_id": r["sample_id"]
            })
        # Plus 30 additional per keyword synthesized with train voices
        for add_idx in range(30):
            vid = TRAIN_VOICES[add_idx % len(TRAIN_VOICES)]
            sid = f"v11_cmd_tr_add_{kw.lower()}_{add_idx:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
            if not dst_wav.exists():
                raw_audio = tts.infer(text, voice=vid, temperature=0.70 + 0.02 * (add_idx % 5), top_p=0.90)
                audio_data, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            else:
                audio_data, _ = sf.read(dst_wav, dtype="float32")
                onset_s, dur_s = detect_speech_bounds(audio_data, 16000)
            sha = compute_sha256(dst_wav)
            train_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "train",
                "voice_id": vid,
                "canonical_class": kw,
                "class_id": cid,
                "bounded_label": kw,
                "bounded_class_id": cid,
                "confusion_family": "target_command",
                "transcription": text,
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": onset_s,
                "speech_duration_sec": dur_s,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
    print(f"Train commands assembled: {len(train_records)}/800")

    # 3.2: 480 Fresh Hard-Negative Speech Clips (60 texts x 8 train voices)
    for idx, item in enumerate(TRAIN_HARD_NEGATIVE_TEXTS[:60]):
        text = item["text"]
        fam = item["confusion_family"]
        tgt = item["target"]
        for v_idx, vid in enumerate(TRAIN_VOICES):
            sid = f"v11_unk_tr_{fam[:4]}_{idx:03d}_{v_idx:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
            if not dst_wav.exists():
                raw_audio = tts.infer(text, voice=vid, temperature=0.74, top_p=0.90)
                audio_data, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            else:
                audio_data, _ = sf.read(dst_wav, dtype="float32")
                onset_s, dur_s = detect_speech_bounds(audio_data, 16000)
            sha = compute_sha256(dst_wav)
            train_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "train",
                "voice_id": vid,
                "canonical_class": "UNKNOWN",
                "class_id": 4,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": text,
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": onset_s,
                "speech_duration_sec": dur_s,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
        if (idx + 1) % 20 == 0:
            print(f"Synthesized {idx + 1}/60 train hard-negative texts across 8 train voices...")

    # 3.3: 560 Clean Historical Non-Targets from v6 train (minimal pairs, neutral speech)
    v6_non_target_candidates = [
        r for r in v6_train_rows
        if r["bounded_label"] == "REJECT" and r["confusion_family"] != "target_command"
    ]
    # Filter to ensure zero overlap with test/val texts
    test_texts_set = {x["text"].strip().lower() for x in TEST_NON_TARGET_TEXTS}
    val_texts_set = {x["text"].strip().lower() for x in VAL_NON_TARGET_TEXTS}
    filtered_v6_nt = [
        r for r in v6_non_target_candidates
        if r["text"].strip().lower() not in test_texts_set and r["text"].strip().lower() not in val_texts_set
    ]
    random.seed(42)
    selected_v6_nt = random.sample(filtered_v6_nt, min(560, len(filtered_v6_nt)))
    for r in selected_v6_nt:
        train_records.append({
            "sample_id": f"v11_unk_tr_{r['sample_id']}",
            "dataset_version": "vieneu_kws_v11.0_bounded",
            "split": "train",
            "voice_id": r["voice_id"],
            "canonical_class": "UNKNOWN" if r["class_label"] != "SILENCE" else "SILENCE",
            "class_id": int(r["class_id"]),
            "bounded_label": "REJECT",
            "bounded_class_id": 4,
            "confusion_family": r["confusion_family"],
            "transcription": r["text"],
            "kws_wav_path": r["kws_wav_path"],
            "duration_samples": 16000,
            "sample_rate": 16000,
            "speech_onset_sec": float(r.get("kws_onset_sec", 0.0)),
            "speech_duration_sec": float(r.get("kws_active_speech_duration_sec", 1.0)),
            "sha256": r.get("kws_sha256", ""),
            "is_historical_reuse": True,
            "historical_source_id": r["sample_id"]
        })
    print(f"Train non-targets after historical ingestion: {len(train_records) - 800}/1040")

    # 3.4: 160 Ambient / Silence Clips (40 per family)
    for fam in ambient_families:
        for r in range(40):
            sid = f"v11_sil_tr_{fam}_{r:02d}"
            dst_wav = AUDIO_DIR / f"{sid}_kws16k.wav"
            if not dst_wav.exists():
                audio_data = generate_synthetic_ambient(fam, seed=31000 + len(train_records))
                sf.write(dst_wav, audio_data, 16000, subtype="PCM_16")
            sha = compute_sha256(dst_wav)
            train_records.append({
                "sample_id": sid,
                "dataset_version": "vieneu_kws_v11.0_bounded",
                "split": "train",
                "voice_id": fam,
                "canonical_class": "SILENCE",
                "class_id": 5,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "confusion_family": fam,
                "transcription": "[silence]",
                "kws_wav_path": str(dst_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "duration_samples": 16000,
                "sample_rate": 16000,
                "speech_onset_sec": 0.0,
                "speech_duration_sec": 1.0,
                "sha256": sha,
                "is_historical_reuse": False,
                "historical_source_id": ""
            })
    print(f"Total training records: {len(train_records)} (Expected 2000)")

    # ---------------------------------------------------------
    # PART 4: SAVE MANIFESTS AND METADATA REGISTRIES
    # ---------------------------------------------------------
    fieldnames = [
        "sample_id", "dataset_version", "split", "voice_id", "canonical_class",
        "class_id", "bounded_label", "bounded_class_id", "confusion_family",
        "transcription", "kws_wav_path", "duration_samples", "sample_rate",
        "speech_onset_sec", "speech_duration_sec", "sha256",
        "is_historical_reuse", "historical_source_id"
    ]

    for split_name, recs in [("test", test_records), ("val", val_records), ("train", train_records)]:
        out_csv = SPLITS_DIR / f"{split_name}_manifest.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)
        h = compute_sha256(out_csv)
        print(f"[OK] Saved {split_name}_manifest.csv: {len(recs)} rows (SHA256: {h[:16]}...)")

    summary = {
        "dataset_name": "vieneu_kws_research_v11",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": {
            "test": {
                "total_clips": len(test_records),
                "command_clips": sum(1 for r in test_records if r["bounded_label"] != "REJECT"),
                "non_target_clips": sum(1 for r in test_records if r["bounded_label"] == "REJECT"),
                "manifest_sha256": compute_sha256(SPLITS_DIR / "test_manifest.csv")
            },
            "val": {
                "total_clips": len(val_records),
                "command_clips": sum(1 for r in val_records if r["bounded_label"] != "REJECT"),
                "non_target_clips": sum(1 for r in val_records if r["bounded_label"] == "REJECT"),
                "manifest_sha256": compute_sha256(SPLITS_DIR / "val_manifest.csv")
            },
            "train": {
                "total_clips": len(train_records),
                "command_clips": sum(1 for r in train_records if r["bounded_label"] != "REJECT"),
                "non_target_clips": sum(1 for r in train_records if r["bounded_label"] == "REJECT"),
                "manifest_sha256": compute_sha256(SPLITS_DIR / "train_manifest.csv")
            }
        }
    }
    with open(V11_DATA_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved dataset summary JSON.")
    print("=" * 80)


if __name__ == "__main__":
    main()
