#!/usr/bin/env python3
"""
Phase 24: Dataset Assembly and Factor Isolation Script.
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/assemble_phase24_manifests.py

Assembles immutable, versioned training manifests:
  - D0_train_v1.csv (5,440 samples, baseline v12 Turbo distribution)
  - D1_matched_train_v1.csv (5,440 samples, multi-generator Turbo+Nano distribution with exact class & SNR match)
  - validation_v1.csv (1,280 samples, frozen validation split)
  - source_eligibility_and_quarantine_ledger.csv (complete provenance and quarantine ledger)
"""

import sys
import io
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import soundfile as sf

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
NANO_STANDARDIZED_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "d1_nano_train"

TRAIN_MANIFEST_SRC = MANIFESTS_DIR / "train_manifest.csv"
VAL_MANIFEST_SRC = MANIFESTS_DIR / "val_manifest.csv"
COMPONENTS_MANIFEST_SRC = MANIFESTS_DIR / "synthetic_eval_components_manifest.csv"

OUT_D0_MANIFEST = MANIFESTS_DIR / "D0_train_v1.csv"
OUT_D1_MANIFEST = MANIFESTS_DIR / "D1_matched_train_v1.csv"
OUT_VAL_MANIFEST = MANIFESTS_DIR / "validation_v1.csv"
OUT_LEDGER_CSV = MANIFESTS_DIR / "source_eligibility_and_quarantine_ledger.csv"
OUT_LEDGER_JSON = MANIFESTS_DIR / "source_eligibility_and_quarantine_ledger.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def standardize_clip_to_1s(raw_path: Path, out_path: Path, target_samples: int = 16000) -> str:
    """Loads raw audio and standardizes to exactly 16,000 samples (1.0s) at 16 kHz."""
    audio, sr = sf.read(str(raw_path), dtype="float32")
    assert sr == 16000, f"Expected 16 kHz, got {sr}"
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Pad or center to 16,000 samples
    if len(audio) < target_samples:
        pad_total = target_samples - len(audio)
        pad_front = min(int(0.150 * 16000), pad_total // 2)
        pad_back = pad_total - pad_front
        std_audio = np.pad(audio, (pad_front, pad_back), mode="constant")
    elif len(audio) > target_samples:
        std_audio = audio[:target_samples]
    else:
        std_audio = audio

    # Ensure peak <= 0.85
    peak = np.max(np.abs(std_audio))
    if peak > 0.85:
        std_audio = std_audio * (0.85 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), std_audio.astype(np.float32), 16000)
    return compute_sha256(out_path)


def main():
    print("=" * 80)
    print("PHASE 24: TRAINING DATASET ASSEMBLY AND FACTOR ISOLATION")
    print("=" * 80)

    # 1. Load Baseline D0
    df_d0 = pd.read_csv(TRAIN_MANIFEST_SRC)
    print(f"\n1. Loaded baseline D0 training manifest: {len(df_d0)} samples from {TRAIN_MANIFEST_SRC.name}")
    assert len(df_d0) == 5440, f"Expected 5,440 samples in D0, got {len(df_d0)}"

    # Save immutable D0_train_v1.csv
    df_d0.to_csv(OUT_D0_MANIFEST, index=False, encoding="utf-8")
    d0_hash = compute_sha256(OUT_D0_MANIFEST)
    print(f"   [OK] Created D0_train_v1.csv (SHA-256: {d0_hash})")

    # 2. Load Validation Set
    df_val = pd.read_csv(VAL_MANIFEST_SRC)
    print(f"\n2. Loaded baseline validation manifest: {len(df_val)} samples from {VAL_MANIFEST_SRC.name}")
    assert len(df_val) == 1280, f"Expected 1,280 samples in val, got {len(df_val)}"

    # Save immutable validation_v1.csv
    df_val.to_csv(OUT_VAL_MANIFEST, index=False, encoding="utf-8")
    val_hash = compute_sha256(OUT_VAL_MANIFEST)
    print(f"   [OK] Created validation_v1.csv (SHA-256: {val_hash})")

    # 3. Assemble D1 Matched Training Set
    # Load available approved non-test Nano components
    df_comp = pd.read_csv(COMPONENTS_MANIFEST_SRC)
    df_nano_dev = df_comp[df_comp["split"] == "dev"].copy()
    print(f"\n3. Loaded approved non-test Nano dev components: {len(df_nano_dev)} clips from {COMPONENTS_MANIFEST_SRC.name}")
    print(f"   Breakdown: {df_nano_dev['label'].value_counts().to_dict()}")
    print(f"   Voices:    {df_nano_dev['voice'].value_counts().to_dict()}")

    # Standardize all 48 Nano clips to exact 16,000 samples (1.0s)
    NANO_STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    nano_std_info = {}
    for _, n_row in df_nano_dev.iterrows():
        cid = n_row["component_id"]
        raw_p = PROJECT_ROOT / n_row["audio_path"]
        std_p = NANO_STANDARDIZED_DIR / f"{cid}_1s.wav"
        std_sha = standardize_clip_to_1s(raw_p, std_p, target_samples=16000)
        rel_p = str(std_p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        nano_std_info[cid] = (rel_p, std_sha)

    # Build D1 by stratified substitution:
    df_d1 = df_d0.copy()
    substitutions_made = 0

    # Map target commands
    for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
        nano_cmd_clips = df_nano_dev[df_nano_dev["label"] == cmd]
        d0_clean_indices = df_d1[(df_d1["label"] == cmd) & (df_d1["source_type"] == "tts_synthetic_clean")].index.tolist()
        assert len(d0_clean_indices) >= len(nano_cmd_clips), f"Not enough clean clips for {cmd}"

        for idx_nano, (_, n_row) in enumerate(nano_cmd_clips.iterrows()):
            replace_idx = d0_clean_indices[idx_nano]
            cid = n_row["component_id"]
            rel_p, std_sha = nano_std_info[cid]

            # Update row in df_d1
            df_d1.loc[replace_idx, "source_type"] = "tts_synthetic_clean_v3nano"
            df_d1.loc[replace_idx, "source_utterance_id"] = cid
            df_d1.loc[replace_idx, "voice_or_speaker_id"] = n_row["voice"]
            df_d1.loc[replace_idx, "audio_path"] = rel_p
            df_d1.loc[replace_idx, "num_samples"] = 16000
            df_d1.loc[replace_idx, "duration_seconds"] = 1.0000
            df_d1.loc[replace_idx, "sha256"] = std_sha
            df_d1.loc[replace_idx, "generation_config_id"] = "vieneu_v3nano_dev_component_v1"
            df_d1.loc[replace_idx, "source_license_or_permission"] = "Proprietary Generated (VieNeu-TTS v3 Nano)"
            substitutions_made += 1

    # Map OOV REJECT clips (40 clips)
    nano_oov_clips = df_nano_dev[df_nano_dev["label"] == "REJECT"]
    d0_clean_oov_indices = df_d1[(df_d1["label"] == "REJECT") & (df_d1["source_type"] == "tts_synthetic_oov_clean")].index.tolist()
    assert len(d0_clean_oov_indices) >= len(nano_oov_clips), f"Not enough clean OOV clips in D0"

    for idx_nano, (_, n_row) in enumerate(nano_oov_clips.iterrows()):
        replace_idx = d0_clean_oov_indices[idx_nano]
        cid = n_row["component_id"]
        rel_p, std_sha = nano_std_info[cid]

        df_d1.loc[replace_idx, "source_type"] = "tts_synthetic_oov_clean_v3nano"
        df_d1.loc[replace_idx, "source_utterance_id"] = cid
        df_d1.loc[replace_idx, "voice_or_speaker_id"] = n_row["voice"]
        df_d1.loc[replace_idx, "text"] = n_row["text"]
        df_d1.loc[replace_idx, "oov_lexeme"] = n_row["text"]
        df_d1.loc[replace_idx, "audio_path"] = rel_p
        df_d1.loc[replace_idx, "num_samples"] = 16000
        df_d1.loc[replace_idx, "duration_seconds"] = 1.0000
        df_d1.loc[replace_idx, "sha256"] = std_sha
        df_d1.loc[replace_idx, "generation_config_id"] = "vieneu_v3nano_dev_component_v1"
        df_d1.loc[replace_idx, "source_license_or_permission"] = "Proprietary Generated (VieNeu-TTS v3 Nano)"
        substitutions_made += 1

    print(f"\n   Total stratified substitutions into D1: {substitutions_made} clips (8 targets, 40 OOV)")
    assert len(df_d1) == 5440, f"D1 length mismatch: {len(df_d1)}"

    # Verify identical class counts
    assert df_d1["label"].value_counts().to_dict() == df_d0["label"].value_counts().to_dict(), "Class count mismatch between D0 and D1!"

    # Verify identical SNR counts
    d0_snr = df_d0["target_snr_db"].fillna(-999.0).value_counts().to_dict()
    d1_snr = df_d1["target_snr_db"].fillna(-999.0).value_counts().to_dict()
    assert d0_snr == d1_snr, f"SNR count mismatch between D0 and D1: {d0_snr} vs {d1_snr}"

    # Save immutable D1_matched_train_v1.csv
    df_d1.to_csv(OUT_D1_MANIFEST, index=False, encoding="utf-8")
    d1_hash = compute_sha256(OUT_D1_MANIFEST)
    print(f"   [OK] Created D1_matched_train_v1.csv (SHA-256: {d1_hash})")

    # 4. Generate Source Eligibility and Quarantine Ledger
    print("\n4. Generating Comprehensive Source Eligibility and Quarantine Ledger...")
    ledger_entries = []

    # A. All D0 sources
    for _, row in df_d0.iterrows():
        ledger_entries.append({
            "source_id": row["sample_id"],
            "sha256": row["sha256"],
            "label": row["label"],
            "reject_subgroup": str(row.get("reject_subgroup", "none")),
            "text": str(row.get("text", "")),
            "engine": "VieNeu-TTS v3 Turbo" if "tts_synthetic" in str(row["source_type"]) else ("FLEURS" if "fleurs" in str(row["source_type"]) else "Environmental/Acoustic"),
            "voice": str(row.get("voice_or_speaker_id", "none")),
            "target_snr_db": "clean" if pd.isna(row["target_snr_db"]) else str(row["target_snr_db"]),
            "historical_exposure": "dataset_v12_train",
            "eligibility_status": "APPROVED_TRAINING_BASELINE",
            "audio_path": row["audio_path"]
        })

    # B. Nano Dev Components (Transferred to Training Material)
    for _, row in df_nano_dev.iterrows():
        cid = row["component_id"]
        rel_p, std_sha = nano_std_info[cid]
        ledger_entries.append({
            "source_id": cid,
            "sha256": std_sha,
            "label": row["label"],
            "reject_subgroup": "oov_word" if row["label"] == "REJECT" else "none",
            "text": str(row["text"]),
            "engine": "VieNeu-TTS v3 Nano",
            "voice": str(row["voice"]),
            "target_snr_db": "clean",
            "historical_exposure": "phase21_dev_component",
            "eligibility_status": "APPROVED_TRAINING_MATERIAL (EXPOSED_DEV_VOICE)",
            "audio_path": rel_p
        })

    # C. Quarantined Phase 21 Test Components
    df_nano_test = df_comp[df_comp["split"] == "test"].copy()
    for _, row in df_nano_test.iterrows():
        wav_p = PROJECT_ROOT / row["audio_path"]
        ledger_entries.append({
            "source_id": row["component_id"],
            "sha256": compute_sha256(wav_p),
            "label": row["label"],
            "reject_subgroup": "oov_word" if row["label"] == "REJECT" else "none",
            "text": str(row["text"]),
            "engine": "VieNeu-TTS v3 Nano",
            "voice": str(row["voice"]),
            "target_snr_db": "clean",
            "historical_exposure": "phase21_locked_test",
            "eligibility_status": "QUARANTINED_TEST_DATA",
            "audio_path": row["audio_path"]
        })

    df_ledger = pd.DataFrame(ledger_entries)
    df_ledger = df_ledger.drop_duplicates(subset=["source_id"])
    df_ledger.to_csv(OUT_LEDGER_CSV, index=False, encoding="utf-8")
    ledger_csv_hash = compute_sha256(OUT_LEDGER_CSV)

    with open(OUT_LEDGER_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": "2026-10-24T22:45:00+07:00",
                "phase": "Phase 24 Dataset Assembly",
                "total_ledger_records": len(df_ledger),
                "quarantined_records_count": len(df_ledger[df_ledger["eligibility_status"] == "QUARANTINED_TEST_DATA"]),
                "approved_training_records_count": len(df_ledger[df_ledger["eligibility_status"].str.startswith("APPROVED_TRAINING")]),
                "ledger_csv_sha256": ledger_csv_hash
            },
            "status_summary": df_ledger["eligibility_status"].value_counts().to_dict(),
            "engine_summary": df_ledger["engine"].value_counts().to_dict()
        }, f, indent=2, ensure_ascii=False)

    print(f"   [OK] Created source_eligibility_and_quarantine_ledger.csv ({len(df_ledger)} entries, SHA-256: {ledger_csv_hash})")
    print(f"   [OK] Created source_eligibility_and_quarantine_ledger.json")

    print("\n" + "=" * 80)
    print("SUMMARY OF MANIFESTS GENERATED IN PHASE 24:")
    print("=" * 80)
    print(f"  D0_train_v1.csv:           {d0_hash} ({len(df_d0)} samples)")
    print(f"  D1_matched_train_v1.csv:   {d1_hash} ({len(df_d1)} samples)")
    print(f"  validation_v1.csv:         {val_hash} ({len(df_val)} samples)")
    print(f"  Quarantine Ledger:         {ledger_csv_hash} ({len(df_ledger)} audited sources)")


if __name__ == "__main__":
    main()
