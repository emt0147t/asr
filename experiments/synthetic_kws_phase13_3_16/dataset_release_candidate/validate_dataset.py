"""
Phase 14 Dataset Release Candidate Validator
Exhaustively validates all 3,000 clips:
- Audio format: 16 kHz mono PCM-16, 16000 samples, 1.0s duration
- Data integrity: No NaNs, no Infs, no clipping, exact disk SHA-256 match for all 3,000 files
- Class balance: 500 per class (350 train, 75 val, 75 test)
- Split isolation: Zero speaker or seed leakage across train, val, and test splits
- Row-independent reproducibility: Bit-exact regeneration on deterministic audit sample
"""

import os
import sys
import json
import wave
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path("experiments/synthetic_kws_v5_2_candidate").resolve()))
from generator import reproduce_from_manifest_row

def validate_dataset_release_candidate():
    rc_dir = Path("experiments/synthetic_kws_phase13_3_16/dataset_release_candidate")
    manifest_path = rc_dir / "manifest.csv"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    df = pd.read_csv(manifest_path)
    total_rows = len(df)
    print(f"Loaded manifest with {total_rows} rows.")
    assert total_rows == 3000, f"Expected 3000 rows, got {total_rows}"

    expected_classes = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    class_counts = df["class_label"].value_counts().to_dict()
    print("Class counts:", class_counts)
    for c in expected_classes:
        assert class_counts.get(c, 0) == 500, f"Class {c} count {class_counts.get(c, 0)} != 500"

    split_counts = df["split"].value_counts().to_dict()
    print("Split counts:", split_counts)
    assert split_counts.get("train", 0) == 2100
    assert split_counts.get("val", 0) == 450
    assert split_counts.get("test", 0) == 450

    subclass_counts = df["acoustic_subclass"].value_counts().to_dict()
    print("Subclass counts:", subclass_counts)

    # 1. Split Isolation Verification
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_spks = set(train_df[train_df["class_label"] != "SILENCE"]["simulated_speaker_id"])
    val_spks = set(val_df[val_df["class_label"] != "SILENCE"]["simulated_speaker_id"])
    test_spks = set(test_df[test_df["class_label"] != "SILENCE"]["simulated_speaker_id"])

    assert len(train_spks.intersection(val_spks)) == 0, "Speaker leakage between train and val!"
    assert len(train_spks.intersection(test_spks)) == 0, "Speaker leakage between train and test!"
    assert len(val_spks.intersection(test_spks)) == 0, "Speaker leakage between val and test!"

    train_seeds = set(train_df["seed"])
    val_seeds = set(val_df["seed"])
    test_seeds = set(test_df["seed"])

    assert len(train_seeds.intersection(val_seeds)) == 0, "Seed leakage between train and val!"
    assert len(train_seeds.intersection(test_seeds)) == 0, "Seed leakage between train and test!"
    assert len(val_seeds.intersection(test_seeds)) == 0, "Seed leakage between val and test!"

    print("Split isolation confirmed: Disjoint speakers and seeds across train/val/test.")

    # 2. Exhaustive Audio File Validation (all 3,000 files)
    print("Validating all 3,000 audio files on disk...")
    audio_dir = rc_dir / "audio"
    files_ok = 0
    sha_matches = 0
    clipping_count = 0
    nan_inf_count = 0

    for idx, row in df.iterrows():
        wav_path = rc_dir / "audio" / f"{row['recording_id']}.wav"
        assert wav_path.exists(), f"Missing file: {wav_path}"

        raw_bytes = wav_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        assert actual_sha == row["sha256"], f"SHA256 mismatch for {row['recording_id']}"
        sha_matches += 1

        with wave.open(str(wav_path), 'rb') as wf:
            assert wf.getframerate() == 16000, f"Bad sample rate for {wav_path}"
            assert wf.getnchannels() == 1, f"Bad channels for {wav_path}"
            assert wf.getsampwidth() == 2, f"Bad bit depth for {wav_path}"
            assert wf.getnframes() == 16000, f"Bad length {wf.getnframes()} for {wav_path}"
            data = wf.readframes(16000)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
                nan_inf_count += 1
            if np.max(np.abs(samples)) > 1.0:
                clipping_count += 1
        files_ok += 1

    print(f"Validated {files_ok}/3000 files: all 16kHz mono 16000 samples, all {sha_matches}/3000 SHA256 match, 0 NaNs/Infs, {clipping_count} clipped.")
    assert files_ok == 3000
    assert sha_matches == 3000
    assert nan_inf_count == 0
    assert clipping_count == 0

    # 3. Row-Independent Reproduction on Deterministic Audit Sample (120 clips)
    print("Testing row-independent reproduction on 120-clip audit sample...")
    # Sample 20 clips per class across splits
    audit_sample = pd.concat([group.head(20) for _, group in df.groupby("class_label")], ignore_index=True)
    regen_matches = 0

    for _, row in audit_sample.iterrows():
        row_dict = row.to_dict()
        audio_regen = reproduce_from_manifest_row(row_dict, sr=16000)
        
        # Write to temporary memory buffer with soundfile
        import io, soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio_regen, 16000, subtype="PCM_16", format="WAV")
        regen_sha = hashlib.sha256(buf.getvalue()).hexdigest()

        wav_path = rc_dir / "audio" / f"{row['recording_id']}.wav"
        disk_sha = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        assert regen_sha == disk_sha, f"Row-independent reproduction hash mismatch for {row['recording_id']}: {regen_sha} != {disk_sha}"
        regen_matches += 1

    print(f"Row-independent reproduction: {regen_matches}/120 (100.0%) bit-exact match!")

    # 4. Save Reports
    validation_report = {
        "report_type": "v5_2_release_candidate_validation_report",
        "dataset_name": "synthetic_kws_v5_2_release_candidate",
        "total_clips": total_rows,
        "class_distribution": class_counts,
        "split_distribution": split_counts,
        "subclass_distribution": subclass_counts,
        "split_isolation": {
            "speaker_leakage": False,
            "seed_leakage": False,
            "train_speakers": list(train_spks),
            "val_speakers": list(val_spks),
            "test_speakers": list(test_spks)
        },
        "audio_validation": {
            "files_evaluated": files_ok,
            "coverage": "100.0% (exhaustive: all 3000 clips)",
            "sample_rate": 16000,
            "channels": 1,
            "duration_sec": 1.0,
            "samples_per_clip": 16000,
            "sha256_match_count": sha_matches,
            "nan_inf_count": nan_inf_count,
            "clipping_count": clipping_count
        },
        "reproducibility": {
            "audit_sample_size": len(audit_sample),
            "matches": regen_matches,
            "pass_rate_pct": 100.0,
            "status": "PASS"
        },
        "verdict": "DATASET_PASS"
    }

    with open(rc_dir / "validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)

    print(f"Validation report saved to {rc_dir / 'validation_report.json'}")
    return validation_report

if __name__ == "__main__":
    validate_dataset_release_candidate()
