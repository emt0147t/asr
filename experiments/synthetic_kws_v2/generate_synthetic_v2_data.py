"""
Generates synthetic negative fixtures (SILENCE, UNKNOWN) and fresh synthetic holdout speakers
for the 6-class Vietnamese KWS V2 experimental pipeline.

All audio files generated here are explicitly tagged as synthetic test fixtures
(SYNTHETIC_TEST_FIXTURE / synthetic_ambient_test_fixture).
Zero unverified external corpora or human recordings are used.
"""

import sys
import os
import json
import random
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal

from src.acquisition.collect_field_data import generate_vietnamese_acoustic_utterance, TARGET_COMMANDS
from src.acquisition.campaign_validator import compute_file_sha256, validate_split_manifests_preflight

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
    """
    Generates synthetic ambient noise or digital dither for pipeline testing only.
    Clearly classified as SYNTHETIC_TEST_FIXTURE.
    """
    n_samples = int(sr * duration_sec)
    if mode == "pure_silence":
        audio = np.zeros(n_samples, dtype=np.float32)
    elif mode == "dither":
        # RMS ~ -60 dBFS
        audio = np.random.randn(n_samples).astype(np.float32) * 1e-3
    elif mode == "room_tone":
        # Low-pass filtered noise to simulate acoustic room floor (RMS ~ -50 dBFS)
        raw_noise = np.random.randn(n_samples).astype(np.float32)
        b, a = signal.butter(4, 1000.0 / (sr / 2.0), btype='low')
        filtered = signal.lfilter(b, a, raw_noise)
        rms_target = 10.0 ** (-50.0 / 20.0)  # ~0.003
        current_rms = np.sqrt(np.mean(filtered ** 2)) + 1e-9
        audio = (filtered / current_rms) * rms_target
    else:
        audio = np.zeros(n_samples, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def build_v2_manifests_and_data(base_dir: Path = Path("experiments/synthetic_kws_v2")):
    data_dir = base_dir / "data"
    silence_dir = data_dir / "synthetic_silence"
    unknown_dir = data_dir / "synthetic_unknown"
    fresh_dir = data_dir / "fresh_holdout"
    
    silence_dir.mkdir(parents=True, exist_ok=True)
    unknown_dir.mkdir(parents=True, exist_ok=True)
    fresh_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load existing approved calibration data from source_registry_v2.csv
    registry = pd.read_csv("data/manifests/source_registry_v2.csv")
    calib_targets = registry[registry["source"] == "calibration_acoustic_generator_v1"].copy()
    confusing_words = registry[registry["source"] == "calibration_confusing_words_v1"].copy()
    
    print(f"Loaded {len(calib_targets)} approved calibration target clips (250/cmd).")
    print(f"Loaded {len(confusing_words)} approved confusing words (UNKNOWN).")
    
    # 2. Generate Synthetic SILENCE clips (Train: 120, Val: 36, Test: 36)
    np.random.seed(42)
    random.seed(42)
    
    silence_records = []
    splits_spec = [
        ("train", 120, [f"SPK_CALIB_{i:03d}" for i in range(1, 15)]),
        ("validation", 36, [f"SPK_CALIB_{i:03d}" for i in range(15, 18)]),
        ("test", 36, [f"SPK_CALIB_{i:03d}" for i in range(18, 21)])
    ]
    
    for split_name, count, spk_pool in splits_spec:
        for idx in range(count):
            spk_id = spk_pool[idx % len(spk_pool)]
            mode = "room_tone" if idx % 2 == 0 else "dither"
            audio = generate_synthetic_silence_clip(mode=mode)
            rec_id = f"synth_silence_{split_name}_{spk_id}_{idx:03d}"
            fpath = silence_dir / f"{rec_id}.wav"
            sf.write(str(fpath), audio, 16000, subtype="PCM_16")
            
            silence_records.append({
                "recording_id": rec_id,
                "source": "synthetic_ambient_test_fixture",
                "source_version": "v2.0-test",
                "license": "Internal-Research",
                "speaker_id": spk_id,
                "accent": "Unspecified-Native",
                "raw_duration_sec": 1.0,
                "transcript": "[SILENCE]",
                "class_id": 5,
                "class_label": "SILENCE",
                "label_provenance": "synthetic_ambient_test_fixture",
                "split": split_name,
                "sha256": compute_file_sha256(fpath),
                "processed_path": str(fpath.resolve()),
                "status": "APPROVED",
                "exclusion_reason": None
            })
            
    df_silence = pd.DataFrame(silence_records)
    print(f"Generated {len(df_silence)} synthetic SILENCE clips across train/val/test.")
    
    # 3. UNKNOWN samples for Validation (36) and Test (36)
    # Train already has 120 confusing words in confusing_words. We add synthetic negative words for val and test speakers:
    unknown_val_test = []
    dialects = ["Northern", "Central", "Southern"]
    
    for split_name, count, spk_pool in [
        ("validation", 36, [f"SPK_CALIB_{i:03d}" for i in range(15, 18)]),
        ("test", 36, [f"SPK_CALIB_{i:03d}" for i in range(18, 21)])
    ]:
        for idx in range(count):
            spk_id = spk_pool[idx % len(spk_pool)]
            s_num = int(spk_id.split("_")[-1])
            is_fem = (s_num > 10)
            f0 = np.random.uniform(180, 240) if is_fem else np.random.uniform(100, 150)
            dialect = dialects[s_num % 3]
            word, tone = NON_TARGET_WORDS[idx % len(NON_TARGET_WORDS)]
            dur = np.random.uniform(0.4, 0.6)
            
            raw_audio = generate_vietnamese_acoustic_utterance(word=word, sr=16000, f0_base=f0, duration_sec=dur, dialect=dialect)
            # Pad to 16000
            if len(raw_audio) < 16000:
                raw_audio = np.pad(raw_audio, (0, 16000 - len(raw_audio)))
            else:
                raw_audio = raw_audio[:16000]
                
            rec_id = f"synth_unknown_{split_name}_{spk_id}_{word}_{idx:03d}"
            fpath = unknown_dir / f"{rec_id}.wav"
            sf.write(str(fpath), raw_audio, 16000, subtype="PCM_16")
            
            unknown_val_test.append({
                "recording_id": rec_id,
                "source": "synthetic_negative_test_fixture",
                "source_version": "v2.0-test",
                "license": "Internal-Research",
                "speaker_id": spk_id,
                "accent": dialect,
                "raw_duration_sec": round(dur, 3),
                "transcript": word,
                "class_id": 4,
                "class_label": "UNKNOWN",
                "label_provenance": "confusing_token_hard_negative",
                "split": split_name,
                "sha256": compute_file_sha256(fpath),
                "processed_path": str(fpath.resolve()),
                "status": "APPROVED",
                "exclusion_reason": None
            })
            
    df_unknown_val_test = pd.DataFrame(unknown_val_test)
    print(f"Generated {len(df_unknown_val_test)} synthetic UNKNOWN clips for val/test.")
    
    # 4. Construct 6-class splits:
    # Train: 712 target + 120 confusing words + 120 silence = 952 rows
    # Val: 144 target + 36 unknown + 36 silence = 216 rows
    # Test (Historical Holdout): 144 target + 36 unknown + 36 silence = 216 rows
    df_tr_targets = calib_targets[calib_targets["split"] == "train"]
    df_va_targets = calib_targets[calib_targets["split"] == "validation"]
    df_te_targets = calib_targets[calib_targets["split"] == "test"]
    
    df_train_6c = pd.concat([
        df_tr_targets,
        confusing_words,
        df_silence[df_silence["split"] == "train"]
    ], ignore_index=True)
    
    df_val_6c = pd.concat([
        df_va_targets,
        df_unknown_val_test[df_unknown_val_test["split"] == "validation"],
        df_silence[df_silence["split"] == "validation"]
    ], ignore_index=True)
    
    df_test_6c = pd.concat([
        df_te_targets,
        df_unknown_val_test[df_unknown_val_test["split"] == "test"],
        df_silence[df_silence["split"] == "test"]
    ], ignore_index=True)
    
    # 5. Preflight validation of 6-class splits
    is_valid, errs, diag = validate_split_manifests_preflight(
        df_train_6c, df_val_6c, df_test_6c, validation_profile="benchmark"
    )
    if not is_valid:
        raise ValueError(f"Split preflight validation failed for 6-class manifests: {errs}")
    print(f"[OK] 6-class split preflight passed: {diag['speaker_status']} (Train: {diag['train_speaker_count']}, Val: {diag['val_speaker_count']}, Test: {diag['test_speaker_count']} speakers)")
    
    train_out = base_dir / "train_manifest_6class.csv"
    val_out = base_dir / "val_manifest_6class.csv"
    test_out = base_dir / "test_manifest_6class.csv"
    
    df_train_6c.to_csv(train_out, index=False)
    df_val_6c.to_csv(val_out, index=False)
    df_test_6c.to_csv(test_out, index=False)
    
    print(f"Saved 6-class manifests:")
    print(f"  Train ({len(df_train_6c)}): {train_out}")
    print(f"  Val   ({len(df_val_6c)}): {val_out}")
    print(f"  Test  ({len(df_test_6c)}): {test_out}")
    
    # 6. Generate FRESH SYNTHETIC HOLDOUT (5 unseen speakers: SPK_CALIB_021 to 025)
    # Distinct seeds, F0 base, dialects:
    fresh_speakers = [
        {"speaker_id": "SPK_CALIB_021", "gender": "MALE", "f0_base": 125.0, "dialect": "Northern"},
        {"speaker_id": "SPK_CALIB_022", "gender": "FEMALE", "f0_base": 215.0, "dialect": "Central"},
        {"speaker_id": "SPK_CALIB_023", "gender": "FEMALE", "f0_base": 195.0, "dialect": "Southern"},
        {"speaker_id": "SPK_CALIB_024", "gender": "MALE", "f0_base": 140.0, "dialect": "Central"},
        {"speaker_id": "SPK_CALIB_025", "gender": "FEMALE", "f0_base": 230.0, "dialect": "Northern"}
    ]
    
    fresh_records = []
    # 10 takes per command for each speaker = 5 * 10 * 4 = 200 target clips
    for spk in fresh_speakers:
        for cid, cmd_info in TARGET_COMMANDS.items():
            word = cmd_info["word"]
            label = cmd_info["label"]
            for take in range(10):
                dur = np.random.uniform(0.35, 0.65)
                f0_take = spk["f0_base"] * np.random.uniform(0.95, 1.05)
                raw = generate_vietnamese_acoustic_utterance(word=word, sr=16000, f0_base=f0_take, duration_sec=dur, dialect=spk["dialect"])
                if len(raw) < 16000:
                    raw = np.pad(raw, (0, 16000 - len(raw)))
                else:
                    raw = raw[:16000]
                    
                rec_id = f"fresh_cmd_{label.lower()}_{spk['speaker_id']}_take{take:03d}"
                fpath = fresh_dir / f"{rec_id}.wav"
                sf.write(str(fpath), raw, 16000, subtype="PCM_16")
                
                fresh_records.append({
                    "recording_id": rec_id,
                    "source": "calibration_acoustic_generator_v1",
                    "source_version": "v2.0-fresh-holdout",
                    "license": "Internal-Research",
                    "speaker_id": spk["speaker_id"],
                    "accent": spk["dialect"],
                    "raw_duration_sec": round(dur, 3),
                    "transcript": word,
                    "class_id": cid,
                    "class_label": label,
                    "label_provenance": "acoustic_reference_utterance",
                    "split": "fresh_holdout",
                    "sha256": compute_file_sha256(fpath),
                    "processed_path": str(fpath.resolve()),
                    "status": "APPROVED",
                    "exclusion_reason": None
                })
                
        # 10 UNKNOWN clips per speaker = 50 clips
        for take in range(10):
            word, _ = NON_TARGET_WORDS[take % len(NON_TARGET_WORDS)]
            dur = np.random.uniform(0.4, 0.6)
            raw = generate_vietnamese_acoustic_utterance(word=word, sr=16000, f0_base=spk["f0_base"], duration_sec=dur, dialect=spk["dialect"])
            if len(raw) < 16000:
                raw = np.pad(raw, (0, 16000 - len(raw)))
            else:
                raw = raw[:16000]
            rec_id = f"fresh_unknown_{spk['speaker_id']}_{word}_take{take:03d}"
            fpath = fresh_dir / f"{rec_id}.wav"
            sf.write(str(fpath), raw, 16000, subtype="PCM_16")
            fresh_records.append({
                "recording_id": rec_id,
                "source": "synthetic_negative_test_fixture",
                "source_version": "v2.0-fresh-holdout",
                "license": "Internal-Research",
                "speaker_id": spk["speaker_id"],
                "accent": spk["dialect"],
                "raw_duration_sec": round(dur, 3),
                "transcript": word,
                "class_id": 4,
                "class_label": "UNKNOWN",
                "label_provenance": "confusing_token_hard_negative",
                "split": "fresh_holdout",
                "sha256": compute_file_sha256(fpath),
                "processed_path": str(fpath.resolve()),
                "status": "APPROVED",
                "exclusion_reason": None
            })
            
        # 10 SILENCE clips per speaker = 50 clips
        for take in range(10):
            mode = "room_tone" if take % 2 == 0 else "dither"
            audio = generate_synthetic_silence_clip(mode=mode)
            rec_id = f"fresh_silence_{spk['speaker_id']}_take{take:03d}"
            fpath = fresh_dir / f"{rec_id}.wav"
            sf.write(str(fpath), audio, 16000, subtype="PCM_16")
            fresh_records.append({
                "recording_id": rec_id,
                "source": "synthetic_ambient_test_fixture",
                "source_version": "v2.0-fresh-holdout",
                "license": "Internal-Research",
                "speaker_id": spk["speaker_id"],
                "accent": "Unspecified-Native",
                "raw_duration_sec": 1.0,
                "transcript": "[SILENCE]",
                "class_id": 5,
                "class_label": "SILENCE",
                "label_provenance": "synthetic_ambient_test_fixture",
                "split": "fresh_holdout",
                "sha256": compute_file_sha256(fpath),
                "processed_path": str(fpath.resolve()),
                "status": "APPROVED",
                "exclusion_reason": None
            })

    df_fresh = pd.DataFrame(fresh_records)
    fresh_out = base_dir / "fresh_holdout_manifest_6class.csv"
    df_fresh.to_csv(fresh_out, index=False)
    print(f"Generated {len(df_fresh)} fresh synthetic holdout clips (5 unseen speakers SPK_CALIB_021-025) -> {fresh_out}")


if __name__ == "__main__":
    build_v2_manifests_and_data()
