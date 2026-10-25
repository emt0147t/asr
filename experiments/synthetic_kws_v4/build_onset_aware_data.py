"""
Builds training and validation datasets using the Onset-Aware Synthetic Generator.
Maintains exact parity with V3 calibration speakers:
  - Train: SPK_CALIB_001..014 (14 speakers, 178 target/cmd + 120 UNKNOWN + 120 SILENCE = 952 clips)
  - Val:   SPK_CALIB_015..017 (3 speakers, 36 target/cmd + 36 UNKNOWN + 36 SILENCE = 216 clips)
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

from experiments.synthetic_kws_v4.onset_aware_generator import (
    generate_vietnamese_onset_aware_utterance,
    generate_synthetic_silence_clip,
    NON_TARGET_WORDS
)
from src.acquisition.collect_field_data import TARGET_COMMANDS
from src.acquisition.campaign_validator import compute_file_sha256, validate_split_manifests_preflight


def build_data():
    base_dir = Path("experiments/synthetic_kws_v4")
    data_dir = base_dir / "data" / "onset_aware"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load speaker metadata from V3 manifests
    df_v3_tr = pd.read_csv("experiments/synthetic_kws_v3/train_manifest_6class.csv")
    df_v3_va = pd.read_csv("experiments/synthetic_kws_v3/val_manifest_6class.csv")
    
    # 20 original calibration speakers metadata
    dialects = ["Northern", "Central", "Southern"]
    np.random.seed(42)
    random.seed(42)
    
    speakers = {}
    for s_idx in range(1, 21):
        spk_id = f"SPK_CALIB_{s_idx:03d}"
        is_female = (s_idx > 10)
        gender = "FEMALE" if is_female else "MALE"
        f0 = np.random.uniform(180, 240) if is_female else np.random.uniform(100, 150)
        dialect = dialects[s_idx % len(dialects)]
        speakers[spk_id] = {
            "speaker_id": spk_id,
            "gender": gender,
            "f0_base": f0,
            "dialect": dialect
        }
        
    print("Generating Onset-Aware Data for Train (952) and Val (216)...")
    
    splits_config = [
        ("train", [f"SPK_CALIB_{i:03d}" for i in range(1, 15)], 178, 120, 120),
        ("validation", [f"SPK_CALIB_{i:03d}" for i in range(15, 18)], 36, 36, 36)
    ]
    
    manifest_records = {"train": [], "validation": []}
    
    for split_name, spk_pool, target_per_cmd, unknown_count, silence_count in splits_config:
        split_dir = data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Target Commands
        for cid, cmd_info in TARGET_COMMANDS.items():
            word = cmd_info["word"]
            label = cmd_info["label"]
            for take in range(target_per_cmd):
                spk_id = spk_pool[take % len(spk_pool)]
                spk = speakers[spk_id]
                dur = np.random.uniform(0.35, 0.65)
                f0_take = spk["f0_base"] * np.random.uniform(0.95, 1.05)
                
                raw = generate_vietnamese_onset_aware_utterance(
                    word=word, sr=16000, f0_base=f0_take, duration_sec=dur, dialect=spk["dialect"]
                )
                if len(raw) < 16000:
                    raw = np.pad(raw, (0, 16000 - len(raw)))
                else:
                    raw = raw[:16000]
                    
                rec_id = f"onset_v4_{split_name}_{label.lower()}_{spk_id}_take{take:03d}"
                fpath = split_dir / f"{rec_id}.wav"
                sf.write(str(fpath), raw, 16000, subtype="PCM_16")
                
                manifest_records[split_name].append({
                    "recording_id": rec_id,
                    "source": "calibration_onset_aware_generator_v4",
                    "source_version": "v4.0",
                    "license": "Internal-Research",
                    "speaker_id": spk_id,
                    "accent": spk["dialect"],
                    "raw_duration_sec": round(dur, 3),
                    "transcript": word,
                    "class_id": cid,
                    "class_label": label,
                    "label_provenance": "onset_acoustic_reference_utterance",
                    "split": split_name,
                    "sha256": compute_file_sha256(fpath),
                    "processed_path": str(fpath.resolve()),
                    "status": "APPROVED",
                    "exclusion_reason": None
                })
                
        # 2. UNKNOWN class
        for take in range(unknown_count):
            spk_id = spk_pool[take % len(spk_pool)]
            spk = speakers[spk_id]
            word, _ = NON_TARGET_WORDS[take % len(NON_TARGET_WORDS)]
            dur = np.random.uniform(0.4, 0.6)
            f0_take = spk["f0_base"] * np.random.uniform(0.95, 1.05)
            
            raw = generate_vietnamese_onset_aware_utterance(
                word=word, sr=16000, f0_base=f0_take, duration_sec=dur, dialect=spk["dialect"]
            )
            if len(raw) < 16000:
                raw = np.pad(raw, (0, 16000 - len(raw)))
            else:
                raw = raw[:16000]
                
            rec_id = f"onset_v4_{split_name}_unknown_{spk_id}_{word}_take{take:03d}"
            fpath = split_dir / f"{rec_id}.wav"
            sf.write(str(fpath), raw, 16000, subtype="PCM_16")
            
            manifest_records[split_name].append({
                "recording_id": rec_id,
                "source": "synthetic_negative_test_fixture",
                "source_version": "v4.0",
                "license": "Internal-Research",
                "speaker_id": spk_id,
                "accent": spk["dialect"],
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
            
        # 3. SILENCE class
        for take in range(silence_count):
            spk_id = spk_pool[take % len(spk_pool)]
            mode = "room_tone" if take % 2 == 0 else "dither"
            audio = generate_synthetic_silence_clip(mode=mode)
            rec_id = f"onset_v4_{split_name}_silence_{spk_id}_take{take:03d}"
            fpath = split_dir / f"{rec_id}.wav"
            sf.write(str(fpath), audio, 16000, subtype="PCM_16")
            
            manifest_records[split_name].append({
                "recording_id": rec_id,
                "source": "synthetic_ambient_test_fixture",
                "source_version": "v4.0",
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

    df_tr = pd.DataFrame(manifest_records["train"])
    df_va = pd.DataFrame(manifest_records["validation"])
    
    tr_out = base_dir / "train_manifest_onset_aware.csv"
    va_out = base_dir / "val_manifest_onset_aware.csv"
    
    df_tr.to_csv(tr_out, index=False)
    df_va.to_csv(va_out, index=False)
    
    print(f"Generated Onset-Aware Data:")
    print(f"  Train ({len(df_tr)}): {tr_out}")
    print(f"  Val   ({len(df_va)}): {va_out}")
    return tr_out, va_out


if __name__ == "__main__":
    build_data()
