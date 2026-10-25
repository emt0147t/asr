"""
Generates a brand new fresh synthetic-seed holdout (SPK_CALIB_026..030) for V3 evaluation.
Total: 300 clips (50 TRAI, 50 PHAI, 50 LEN, 50 XUONG, 50 UNKNOWN, 50 SILENCE).
All audio is created with independent random seeds after training is frozen.
Clearly labeled as a fresh synthetic-seed holdout, NOT an independent real-world benchmark.
"""

import sys
import os
import random
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal

from src.acquisition.collect_field_data import generate_vietnamese_acoustic_utterance, TARGET_COMMANDS
from src.acquisition.campaign_validator import compute_file_sha256

NON_TARGET_WORDS = [
    ("chào", "falling"),
    ("bạn", "heavy"),
    ("đi", "level"),
    ("xe", "level"),
    ("nhà", "falling"),
    ("số", "rising"),
    ("ngày", "falling"),
    ("học", "heavy")
]


def generate_synthetic_silence_clip(sr: int = 16000, duration_sec: float = 1.0, mode: str = "dither") -> np.ndarray:
    n_samples = int(sr * duration_sec)
    if mode == "pure_silence":
        audio = np.zeros(n_samples, dtype=np.float32)
    elif mode == "dither":
        audio = np.random.randn(n_samples).astype(np.float32) * 1e-3
    elif mode == "room_tone":
        raw_noise = np.random.randn(n_samples).astype(np.float32)
        b, a = signal.butter(4, 1000.0 / (sr / 2.0), btype='low')
        filtered = signal.lfilter(b, a, raw_noise)
        rms_target = 10.0 ** (-50.0 / 20.0)
        current_rms = np.sqrt(np.mean(filtered ** 2)) + 1e-9
        audio = (filtered / current_rms) * rms_target
    else:
        audio = np.zeros(n_samples, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def generate_v3_holdout(base_dir: Path = Path("experiments/synthetic_kws_v3"), seed: int = 20261016):
    np.random.seed(seed)
    random.seed(seed)
    
    fresh_dir = base_dir / "data" / "fresh_holdout"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    
    fresh_speakers = [
        {"speaker_id": "SPK_CALIB_026", "gender": "MALE", "f0_base": 115.0, "dialect": "Central"},
        {"speaker_id": "SPK_CALIB_027", "gender": "FEMALE", "f0_base": 205.0, "dialect": "Northern"},
        {"speaker_id": "SPK_CALIB_028", "gender": "MALE", "f0_base": 135.0, "dialect": "Southern"},
        {"speaker_id": "SPK_CALIB_029", "gender": "FEMALE", "f0_base": 225.0, "dialect": "Central"},
        {"speaker_id": "SPK_CALIB_030", "gender": "MALE", "f0_base": 145.0, "dialect": "Northern"}
    ]
    
    fresh_records = []
    
    # 1. Target Commands (10 takes each per speaker = 50 per class = 200 clips)
    for spk in fresh_speakers:
        for cid, cmd_info in TARGET_COMMANDS.items():
            word = cmd_info["word"]
            label = cmd_info["label"]
            for take in range(10):
                dur = np.random.uniform(0.35, 0.65)
                f0_take = spk["f0_base"] * np.random.uniform(0.95, 1.05)
                raw = generate_vietnamese_acoustic_utterance(
                    word=word, sr=16000, f0_base=f0_take, duration_sec=dur, dialect=spk["dialect"]
                )
                if len(raw) < 16000:
                    raw = np.pad(raw, (0, 16000 - len(raw)))
                else:
                    raw = raw[:16000]
                    
                rec_id = f"fresh_v3_cmd_{label.lower()}_{spk['speaker_id']}_take{take:03d}"
                fpath = fresh_dir / f"{rec_id}.wav"
                sf.write(str(fpath), raw, 16000, subtype="PCM_16")
                
                fresh_records.append({
                    "recording_id": rec_id,
                    "source": "calibration_acoustic_generator_v1",
                    "source_version": "v3.0-fresh-holdout",
                    "license": "Internal-Research",
                    "speaker_id": spk["speaker_id"],
                    "accent": spk["dialect"],
                    "raw_duration_sec": round(dur, 3),
                    "transcript": word,
                    "class_id": cid,
                    "class_label": label,
                    "label_provenance": "acoustic_reference_utterance",
                    "split": "fresh_holdout_v3",
                    "sha256": compute_file_sha256(fpath),
                    "processed_path": str(fpath.resolve()),
                    "status": "APPROVED",
                    "exclusion_reason": None
                })
                
        # 2. UNKNOWN class (10 clips per speaker = 50 clips)
        for take in range(10):
            word, _ = NON_TARGET_WORDS[take % len(NON_TARGET_WORDS)]
            dur = np.random.uniform(0.4, 0.6)
            f0_take = spk["f0_base"] * np.random.uniform(0.95, 1.05)
            raw = generate_vietnamese_acoustic_utterance(
                word=word, sr=16000, f0_base=f0_take, duration_sec=dur, dialect=spk["dialect"]
            )
            if len(raw) < 16000:
                raw = np.pad(raw, (0, 16000 - len(raw)))
            else:
                raw = raw[:16000]
            rec_id = f"fresh_v3_unknown_{spk['speaker_id']}_{word}_take{take:03d}"
            fpath = fresh_dir / f"{rec_id}.wav"
            sf.write(str(fpath), raw, 16000, subtype="PCM_16")
            fresh_records.append({
                "recording_id": rec_id,
                "source": "synthetic_negative_test_fixture",
                "source_version": "v3.0-fresh-holdout",
                "license": "Internal-Research",
                "speaker_id": spk["speaker_id"],
                "accent": spk["dialect"],
                "raw_duration_sec": round(dur, 3),
                "transcript": word,
                "class_id": 4,
                "class_label": "UNKNOWN",
                "label_provenance": "confusing_token_hard_negative",
                "split": "fresh_holdout_v3",
                "sha256": compute_file_sha256(fpath),
                "processed_path": str(fpath.resolve()),
                "status": "APPROVED",
                "exclusion_reason": None
            })
            
        # 3. SILENCE class (10 clips per speaker = 50 clips)
        for take in range(10):
            mode = "room_tone" if take % 2 == 0 else "dither"
            audio = generate_synthetic_silence_clip(mode=mode)
            rec_id = f"fresh_v3_silence_{spk['speaker_id']}_take{take:03d}"
            fpath = fresh_dir / f"{rec_id}.wav"
            sf.write(str(fpath), audio, 16000, subtype="PCM_16")
            fresh_records.append({
                "recording_id": rec_id,
                "source": "synthetic_ambient_test_fixture",
                "source_version": "v3.0-fresh-holdout",
                "license": "Internal-Research",
                "speaker_id": spk["speaker_id"],
                "accent": "Unspecified-Native",
                "raw_duration_sec": 1.0,
                "transcript": "[SILENCE]",
                "class_id": 5,
                "class_label": "SILENCE",
                "label_provenance": "synthetic_ambient_test_fixture",
                "split": "fresh_holdout_v3",
                "sha256": compute_file_sha256(fpath),
                "processed_path": str(fpath.resolve()),
                "status": "APPROVED",
                "exclusion_reason": None
            })

    df_fresh = pd.DataFrame(fresh_records)
    out_csv = base_dir / "fresh_holdout_manifest_6class.csv"
    df_fresh.to_csv(out_csv, index=False)
    print(f"Generated {len(df_fresh)} fresh synthetic holdout clips -> {out_csv}")
    print(f"Breakdown per class:\n{df_fresh['class_label'].value_counts()}")
    return out_csv


if __name__ == "__main__":
    generate_v3_holdout()
