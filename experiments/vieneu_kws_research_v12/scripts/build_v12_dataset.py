"""
Master Vietnamese KWS Dataset v12 Generator.
Builds the 8,000-clip provenance-controlled dataset:
  - 4,000 Positive Keywords (1,000 TRAI, 1,000 PHAI, 1,000 LEN, 1,000 XUONG)
  - 4,000 REJECT clips:
      * 1,600 OOV monosyllabic / hard-negative words
      * 1,000 ordinary conversational phrases (FLEURS)
      * 800 environmental noise clips (Husein mic-noise)
      * 600 silence / non-speech ambience clips (synthetic acoustic models)

Split Distribution (Strict 68% / 16% / 16%):
  - Train: 5,440 clips (2,720 Positive, 2,720 REJECT)
  - Val:   1,280 clips (640 Positive, 640 REJECT)
  - Test:  1,280 clips (640 Positive, 640 REJECT)
Total: 8,000 clips.

Strict Leakage Prevention:
  - Voice-disjoint across splits (17 Train / 4 Val / 4 Test)
  - OOV lexeme-disjoint across splits (Train / Val / Test unseen)
  - Noise track-disjoint across splits (28 Train / 6 Val / 6 Test)
  - FLEURS recording-disjoint across splits
  - Multi-augmented variants never cross split boundaries.
"""

import os
import sys
import io
import re
import json
import hashlib
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
AUDIO_OUT_DIR = V12_DIR / "audio" / "kws_16k"
MANIFESTS_DIR = V12_DIR / "manifests"
CONFIGS_DIR = V12_DIR / "configs"

# External sources
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"
FLEURS_DIR = PROJECT_ROOT / "data" / "raw" / "fleurs_vi"
DATASET_1000_MANIFEST = PROJECT_ROOT / "experiments" / "synthetic_kws_dataset_vieneu_1000" / "manifest.csv"
DATASET_6CLASS_MANIFEST = PROJECT_ROOT / "experiments" / "synthetic_kws_dataset_vieneu_6class_v2" / "manifest.csv"
STAGING_OOV_DIR = V12_DIR / "audio" / "staging_oov"

sys.path.insert(0, str(V12_DIR / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

CONFIG_ID = "vietnamese_kws_dataset_v12_recipe_v1"
SEED_BASE = 42000


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_silence(family: str, seed: int, duration_samples: int = 16000) -> np.ndarray:
    """Generates synthetic ambient noise/silence using calibrated acoustic models."""
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


def build_v12_dataset(is_pilot: bool = False):
    print("=" * 80)
    print(f"  BUILDING VIETNAMESE KWS DATASET v12 {'(PILOT MODE: 80 CLIPS)' if is_pilot else '(FULL SCALE: 8,000 CLIPS)'}")
    print("  Candidate Architecture Target: E1 (Baseline Scale) -> E2 (419k Scale)")
    print("  Inference Contract: Mono 16 kHz, 16,000 samples, Log-Mel [1, 1, 40, 101], theta_cmd = 0.20")
    print("=" * 80)

    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load split policy
    split_policy_file = CONFIGS_DIR / "split_policy.json"
    with open(split_policy_file, "r", encoding="utf-8") as f:
        split_policy = json.load(f)

    noise_mgr = NoiseAssetManager(NOISE_DIR)
    noise_alloc = split_policy["noise_asset_allocation"]
    snr_levels = split_policy["target_snr_db_levels"]  # [20.0, 15.0, 10.0, 5.0, 0.0]

    # Target allocations
    if is_pilot:
        target_counts = {
            "TRAI": 10, "PHAI": 10, "LEN": 10, "XUONG": 10,
            "oov_word": 16, "conversation": 10, "environmental_noise": 8, "silence_non_speech": 6
        }
    else:
        target_counts = {
            "TRAI": 1000, "PHAI": 1000, "LEN": 1000, "XUONG": 1000,
            "oov_word": 1600, "conversation": 1000, "environmental_noise": 800, "silence_non_speech": 600
        }

    records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # PART 1: POSITIVE KEYWORDS (TRAI, PHAI, LEN, XUONG)
    # -------------------------------------------------------------------------
    print("\n[Part 1] Processing Positive Keywords...")
    # Load source clean utterances from dataset_1000 and dataset_6class_v2
    df_src1 = pd.read_csv(DATASET_1000_MANIFEST)
    df_src2 = pd.read_csv(DATASET_6CLASS_MANIFEST)
    df_src2_targets = df_src2[df_src2["class_label"].isin(["TRAI", "PHAI", "LEN", "XUONG"])].copy()

    for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
        total_cmd_target = target_counts[cmd]
        train_target = int(round(total_cmd_target * 0.68))
        val_target = int(round(total_cmd_target * 0.16))
        test_target = total_cmd_target - train_target - val_target

        split_targets = {"train": train_target, "val": val_target, "test": test_target}

        for split_name in ["train", "val", "test"]:
            needed = split_targets[split_name]
            clean_needed = max(1, int(round(needed * 0.30)))
            noisy_needed = needed - clean_needed

            # Filter source utterances for this class and split
            sub1 = df_src1[(df_src1["class_label"] == cmd) & (df_src1["split"] == split_name)]
            sub2 = df_src2_targets[(df_src2_targets["class_label"] == cmd) & (df_src2_targets["split"] == split_name)]
            
            src_rows = []
            for _, r in sub1.iterrows():
                src_rows.append({
                    "src_id": r["sample_id"],
                    "wav_path": PROJECT_ROOT / r["kws_wav_path"],
                    "voice_id": r["voice_id"],
                    "text": r["text"],
                    "license": r["license"]
                })
            for _, r in sub2.iterrows():
                p = PROJECT_ROOT / r["kws_wav_path"]
                src_rows.append({
                    "src_id": r["sample_id"],
                    "wav_path": p,
                    "voice_id": r["voice_id"],
                    "text": r["text"],
                    "license": r.get("license", "Apache-2.0")
                })

            assert len(src_rows) >= clean_needed, f"Not enough source rows for {cmd} {split_name}: {len(src_rows)} < {clean_needed}"

            # 1. Clean subset
            for i in range(clean_needed):
                src = src_rows[i % len(src_rows)]
                sample_id = f"v12_{cmd.lower()}_{split_name}_clean_{i+1:04d}"
                out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"
                
                audio, sr = sf.read(str(src["wav_path"]), dtype="float32")
                assert len(audio) == 16000 and sr == 16000
                sf.write(str(out_path), audio, 16000)

                sha = compute_sha256(out_path)
                records.append({
                    "sample_id": sample_id,
                    "label": cmd,
                    "reject_subgroup": None,
                    "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "source_type": "tts_synthetic_clean",
                    "source_utterance_id": src["src_id"],
                    "voice_or_speaker_id": src["voice_id"],
                    "text": src["text"],
                    "oov_lexeme": None,
                    "noise_asset_id": None,
                    "target_snr_db": None,
                    "observed_snr_db": None,
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "num_samples": 16000,
                    "duration_seconds": 1.0,
                    "transformations": "none",
                    "split": split_name,
                    "source_license_or_permission": src["license"],
                    "sha256": sha,
                    "generation_config_id": CONFIG_ID
                })

            # 2. Noisy subset
            noise_indices = noise_alloc[f"{split_name}_track_indices"]
            for i in range(noisy_needed):
                src = src_rows[i % len(src_rows)]
                snr_target = snr_levels[i % len(snr_levels)]
                noise_track_idx = noise_indices[i % len(noise_indices)]
                noise_track_name = noise_mgr.get_track_path(noise_track_idx).name
                seed = SEED_BASE + hash(f"{cmd}_{split_name}_{i}") % 100000

                clean_audio, _ = sf.read(str(src["wav_path"]), dtype="float32")
                noise_slice = noise_mgr.extract_noise_slice(noise_track_idx, seed=seed)
                mixed, obs_snr, peak, rms_db = mix_noise_at_snr(clean_audio, noise_slice, snr_target)

                sample_id = f"v12_{cmd.lower()}_{split_name}_snr{int(snr_target):02d}_{i+1:04d}"
                out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"
                sf.write(str(out_path), mixed, 16000)

                sha = compute_sha256(out_path)
                records.append({
                    "sample_id": sample_id,
                    "label": cmd,
                    "reject_subgroup": None,
                    "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "source_type": "noise_augmented",
                    "source_utterance_id": src["src_id"],
                    "voice_or_speaker_id": src["voice_id"],
                    "text": src["text"],
                    "oov_lexeme": None,
                    "noise_asset_id": noise_track_name,
                    "target_snr_db": snr_target,
                    "observed_snr_db": obs_snr,
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "num_samples": 16000,
                    "duration_seconds": 1.0,
                    "transformations": f"noise_mix_snr_{snr_target}db",
                    "split": split_name,
                    "source_license_or_permission": f"{src['license']} + CC-BY-4.0 (noise)",
                    "sha256": sha,
                    "generation_config_id": CONFIG_ID
                })

        print(f"  Generated {total_cmd_target} clips for {cmd}.")

    # -------------------------------------------------------------------------
    # PART 2: REJECT SUBGROUP 1: OOV MONOSYLLABLES (1,600 CLIPS)
    # -------------------------------------------------------------------------
    print("\n[Part 2] Processing OOV Monosyllabic Confusers...")
    total_oov_target = target_counts["oov_word"]
    oov_split_targets = {
        "train": int(round(total_oov_target * 0.68)),
        "val": int(round(total_oov_target * 0.16)),
        "test": total_oov_target - int(round(total_oov_target * 0.68)) - int(round(total_oov_target * 0.16))
    }

    # Gather available OOV base sources:
    # 1. 6class_v2 UNKNOWN clips
    df_unk = df_src2[df_src2["class_label"] == "UNKNOWN"].copy()
    # 2. staging_oov clips (cửa, bàn, nhà, ghế)
    staging_files = sorted(list(STAGING_OOV_DIR.glob("*.wav")))

    for split_name in ["train", "val", "test"]:
        needed = oov_split_targets[split_name]
        clean_needed = max(1, int(round(needed * 0.30)))
        noisy_needed = needed - clean_needed

        sub_unk = df_unk[df_unk["split"] == split_name]
        split_staging = [f for f in staging_files if f"_{split_name}_" in f.name]

        oov_sources = []
        # Add staging confusers first
        for sf_path in split_staging:
            parts = sf_path.stem.split("_")
            w = parts[2]
            v = parts[3]
            oov_sources.append({
                "src_id": sf_path.stem,
                "wav_path": sf_path,
                "voice_id": v,
                "text": w,
                "oov_lexeme": w.lower(),
                "license": "Apache-2.0"
            })
        # Add 6class_v2 UNKNOWN sources
        for _, r in sub_unk.iterrows():
            p = PROJECT_ROOT / r["kws_wav_path"]
            oov_sources.append({
                "src_id": r["sample_id"],
                "wav_path": p,
                "voice_id": r["voice_id"],
                "text": r["text"],
                "oov_lexeme": str(r["text"]).strip().lower(),
                "license": "Apache-2.0"
            })

        assert len(oov_sources) > 0, f"No OOV sources for {split_name}"

        # Clean OOV clips
        for i in range(clean_needed):
            src = oov_sources[i % len(oov_sources)]
            sample_id = f"v12_reject_oov_{split_name}_clean_{i+1:04d}"
            out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"

            audio, sr = sf.read(str(src["wav_path"]), dtype="float32")
            assert len(audio) == 16000 and sr == 16000
            sf.write(str(out_path), audio, 16000)

            sha = compute_sha256(out_path)
            records.append({
                "sample_id": sample_id,
                "label": "REJECT",
                "reject_subgroup": "oov_word",
                "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_type": "tts_synthetic_oov_clean",
                "source_utterance_id": src["src_id"],
                "voice_or_speaker_id": src["voice_id"],
                "text": src["text"],
                "oov_lexeme": src["oov_lexeme"],
                "noise_asset_id": None,
                "target_snr_db": None,
                "observed_snr_db": None,
                "sample_rate_hz": 16000,
                "channels": 1,
                "num_samples": 16000,
                "duration_seconds": 1.0,
                "transformations": "none",
                "split": split_name,
                "source_license_or_permission": src["license"],
                "sha256": sha,
                "generation_config_id": CONFIG_ID
            })

        # Noisy OOV clips
        noise_indices = noise_alloc[f"{split_name}_track_indices"]
        for i in range(noisy_needed):
            src = oov_sources[i % len(oov_sources)]
            snr_target = snr_levels[i % len(snr_levels)]
            noise_track_idx = noise_indices[i % len(noise_indices)]
            noise_track_name = noise_mgr.get_track_path(noise_track_idx).name
            seed = SEED_BASE + hash(f"oov_{split_name}_{i}") % 100000

            clean_audio, _ = sf.read(str(src["wav_path"]), dtype="float32")
            noise_slice = noise_mgr.extract_noise_slice(noise_track_idx, seed=seed)
            mixed, obs_snr, peak, rms_db = mix_noise_at_snr(clean_audio, noise_slice, snr_target)

            sample_id = f"v12_reject_oov_{split_name}_snr{int(snr_target):02d}_{i+1:04d}"
            out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"
            sf.write(str(out_path), mixed, 16000)

            sha = compute_sha256(out_path)
            records.append({
                "sample_id": sample_id,
                "label": "REJECT",
                "reject_subgroup": "oov_word",
                "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_type": "noise_augmented_oov",
                "source_utterance_id": src["src_id"],
                "voice_or_speaker_id": src["voice_id"],
                "text": src["text"],
                "oov_lexeme": src["oov_lexeme"],
                "noise_asset_id": noise_track_name,
                "target_snr_db": snr_target,
                "observed_snr_db": obs_snr,
                "sample_rate_hz": 16000,
                "channels": 1,
                "num_samples": 16000,
                "duration_seconds": 1.0,
                "transformations": f"noise_mix_snr_{snr_target}db",
                "split": split_name,
                "source_license_or_permission": f"{src['license']} + CC-BY-4.0 (noise)",
                "sha256": sha,
                "generation_config_id": CONFIG_ID
            })

    print(f"  Generated {total_oov_target} OOV clips.")

    # -------------------------------------------------------------------------
    # PART 3: REJECT SUBGROUP 2: CONVERSATIONAL SPEECH (FLEURS) (1,000 CLIPS)
    # -------------------------------------------------------------------------
    print("\n[Part 3] Processing Conversational Speech (Google FLEURS)...")
    total_conv_target = target_counts["conversation"]
    conv_split_targets = {
        "train": int(round(total_conv_target * 0.68)),
        "val": int(round(total_conv_target * 0.16)),
        "test": total_conv_target - int(round(total_conv_target * 0.68)) - int(round(total_conv_target * 0.16))
    }

    # Load FLEURS transcripts to filter out any target command occurrences ('trái', 'phải', 'lên', 'xuống')
    fleurs_cols = ['id', 'wav_name', 'raw_text', 'norm_text', 'tokens', 'num_samples', 'gender']
    df_dev_fl = pd.read_csv(FLEURS_DIR / 'dev.tsv', sep='\t', header=None, names=fleurs_cols)
    df_test_fl = pd.read_csv(FLEURS_DIR / 'test.tsv', sep='\t', header=None, names=fleurs_cols)
    df_fleurs_all = pd.concat([df_dev_fl, df_test_fl], ignore_index=True)
    fleurs_transcripts = {}
    for _, r in df_fleurs_all.iterrows():
        fleurs_transcripts[Path(r['wav_name']).name] = (str(r['raw_text']), str(r['norm_text']))

    target_pattern = re.compile(r'\b(trái|trai|phải|phai|lên|len|xuống|xuong)\b', re.IGNORECASE)

    raw_fleurs_files = sorted(list(FLEURS_DIR.rglob("*.wav")))
    clean_fleurs_entries = []
    excluded_count = 0
    for fpath in raw_fleurs_files:
        if fpath.name in fleurs_transcripts:
            raw_t, norm_t = fleurs_transcripts[fpath.name]
            if target_pattern.search(f"{raw_t} {norm_t}"):
                excluded_count += 1
            else:
                clean_fleurs_entries.append((fpath, raw_t))

    print(f"  FLEURS audit: {len(raw_fleurs_files)} total files, {excluded_count} excluded for target word matches, {len(clean_fleurs_entries)} verified clean files.")
    assert len(clean_fleurs_entries) >= total_conv_target, f"Not enough clean FLEURS files: {len(clean_fleurs_entries)} < {total_conv_target}"

    # Partition verified clean FLEURS files disjointly
    train_fleurs = clean_fleurs_entries[:conv_split_targets["train"]]
    val_fleurs = clean_fleurs_entries[conv_split_targets["train"] : conv_split_targets["train"] + conv_split_targets["val"]]
    test_fleurs = clean_fleurs_entries[conv_split_targets["train"] + conv_split_targets["val"] : total_conv_target]

    fleurs_partitions = {"train": train_fleurs, "val": val_fleurs, "test": test_fleurs}

    for split_name in ["train", "val", "test"]:
        split_entries = fleurs_partitions[split_name]
        for i, (fpath, transcript_text) in enumerate(split_entries):
            sample_id = f"v12_reject_conv_{split_name}_{i+1:04d}"
            out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"

            raw_audio, sr = sf.read(str(fpath), dtype="float32")
            assert sr == 16000

            # Extract an active 1.0s window from middle
            if len(raw_audio) >= 16000:
                mid = len(raw_audio) // 2
                start_sample = mid - 8000
                end_sample = mid + 8000
                clip_1s = raw_audio[start_sample:end_sample]
            else:
                start_sample = 0
                end_sample = len(raw_audio)
                clip_1s = np.pad(raw_audio, (0, 16000 - len(raw_audio)))

            # Normalization safeguard
            peak = float(np.max(np.abs(clip_1s)))
            if peak > 0.95:
                clip_1s = clip_1s * (0.95 / peak)

            sf.write(str(out_path), clip_1s.astype(np.float32), 16000)
            sha = compute_sha256(out_path)

            records.append({
                "sample_id": sample_id,
                "label": "REJECT",
                "reject_subgroup": "conversation",
                "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_type": "conversation_fleurs",
                "source_utterance_id": fpath.stem,
                "voice_or_speaker_id": fpath.stem.split("_")[0],
                "text": transcript_text,
                "oov_lexeme": None,
                "noise_asset_id": None,
                "target_snr_db": None,
                "observed_snr_db": None,
                "sample_rate_hz": 16000,
                "channels": 1,
                "num_samples": 16000,
                "duration_seconds": 1.0,
                "transformations": f"verified_clean_center_slice_1s_samples_{start_sample}_{end_sample}",
                "split": split_name,
                "source_license_or_permission": "CC-BY-4.0",
                "sha256": sha,
                "generation_config_id": CONFIG_ID
            })

    print(f"  Generated {total_conv_target} verified clean Conversational Speech clips.")

    # -------------------------------------------------------------------------
    # PART 4: REJECT SUBGROUP 3: ENVIRONMENTAL NOISE (800 CLIPS)
    # -------------------------------------------------------------------------
    print("\n[Part 4] Processing Environmental Noise (Husein Mic-Noise)...")
    total_env_target = target_counts["environmental_noise"]
    env_split_targets = {
        "train": int(round(total_env_target * 0.68)),
        "val": int(round(total_env_target * 0.16)),
        "test": total_env_target - int(round(total_env_target * 0.68)) - int(round(total_env_target * 0.16))
    }

    for split_name in ["train", "val", "test"]:
        needed = env_split_targets[split_name]
        noise_indices = noise_alloc[f"{split_name}_track_indices"]
        split_offset = {"train": 0, "val": 10000, "test": 20000}[split_name]

        for i in range(needed):
            track_idx = noise_indices[i % len(noise_indices)]
            track_name = noise_mgr.get_track_path(track_idx).name
            seed = SEED_BASE + 7000 + split_offset + i * 31

            noise_slice = noise_mgr.extract_noise_slice(track_idx, seed=seed)
            peak = float(np.max(np.abs(noise_slice)))
            if peak > 0.95:
                noise_slice = noise_slice * (0.95 / peak)

            sample_id = f"v12_reject_envnoise_{split_name}_{i+1:04d}"
            out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"
            sf.write(str(out_path), noise_slice.astype(np.float32), 16000)

            sha = compute_sha256(out_path)
            records.append({
                "sample_id": sample_id,
                "label": "REJECT",
                "reject_subgroup": "environmental_noise",
                "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_type": "environmental_noise",
                "source_utterance_id": f"{track_name}_slice_{i+1:04d}",
                "voice_or_speaker_id": None,
                "text": None,
                "oov_lexeme": None,
                "noise_asset_id": track_name,
                "target_snr_db": None,
                "observed_snr_db": None,
                "sample_rate_hz": 16000,
                "channels": 1,
                "num_samples": 16000,
                "duration_seconds": 1.0,
                "transformations": "deterministic_slice_1s",
                "split": split_name,
                "source_license_or_permission": "CC-BY-4.0",
                "sha256": sha,
                "generation_config_id": CONFIG_ID
            })

    print(f"  Generated {total_env_target} Environmental Noise clips.")

    # -------------------------------------------------------------------------
    # PART 5: REJECT SUBGROUP 4: SILENCE / NON-SPEECH AMBIENCE (600 CLIPS)
    # -------------------------------------------------------------------------
    print("\n[Part 5] Processing Silence / Non-Speech Ambience...")
    total_sil_target = target_counts["silence_non_speech"]
    sil_split_targets = {
        "train": int(round(total_sil_target * 0.68)),
        "val": int(round(total_sil_target * 0.16)),
        "test": total_sil_target - int(round(total_sil_target * 0.68)) - int(round(total_sil_target * 0.16))
    }

    families = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]

    for split_name in ["train", "val", "test"]:
        needed = sil_split_targets[split_name]
        split_seed_offset = {"train": 0, "val": 10000, "test": 20000}[split_name]

        for i in range(needed):
            fam = families[i % len(families)]
            seed = SEED_BASE + 9000 + split_seed_offset + i * 17
            sil_audio = generate_synthetic_silence(fam, seed=seed)

            sample_id = f"v12_reject_silence_{split_name}_{fam}_{i+1:04d}"
            out_path = AUDIO_OUT_DIR / f"{sample_id}.wav"
            sf.write(str(out_path), sil_audio.astype(np.float32), 16000)

            sha = compute_sha256(out_path)
            records.append({
                "sample_id": sample_id,
                "label": "REJECT",
                "reject_subgroup": "silence_non_speech",
                "audio_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_type": "ambient_silence",
                "source_utterance_id": f"synthetic_{fam}_{split_name}_{i+1:04d}",
                "voice_or_speaker_id": None,
                "text": None,
                "oov_lexeme": None,
                "noise_asset_id": f"synthetic_{fam}_{split_name}",
                "target_snr_db": None,
                "observed_snr_db": None,
                "sample_rate_hz": 16000,
                "channels": 1,
                "num_samples": 16000,
                "duration_seconds": 1.0,
                "transformations": f"synthetic_acoustic_{fam}",
                "split": split_name,
                "source_license_or_permission": "Apache-2.0 / Internal-Research",
                "sha256": sha,
                "generation_config_id": CONFIG_ID
            })

    print(f"  Generated {total_sil_target} Silence / Ambience clips.")

    # -------------------------------------------------------------------------
    # MANIFEST OUTPUT & SPLITS
    # -------------------------------------------------------------------------
    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("  GENERATION COMPLETE — SUMMARY TABLE")
    print("=" * 80)
    print(f"Total Rows Generated: {len(df)}")
    print("\nCounts by Label:")
    print(df["label"].value_counts().to_string())
    print("\nCounts by Reject Subgroup:")
    print(df["reject_subgroup"].value_counts(dropna=False).to_string())
    print("\nCounts by Split:")
    print(df["split"].value_counts().to_string())

    manifest_csv = MANIFESTS_DIR / "dataset_v12_manifest.csv"
    manifest_jsonl = MANIFESTS_DIR / "dataset_v12_manifest.jsonl"
    df.to_csv(manifest_csv, index=False, encoding="utf-8")
    
    with open(manifest_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Split manifests
    for sp in ["train", "val", "test"]:
        sp_df = df[df["split"] == sp]
        sp_csv = MANIFESTS_DIR / f"{sp}_manifest.csv"
        sp_df.to_csv(sp_csv, index=False, encoding="utf-8")
        print(f"Saved: {sp_csv} ({len(sp_df)} rows)")

    print(f"\nMaster Manifests written to:")
    print(f"  - CSV:   {manifest_csv}")
    print(f"  - JSONL: {manifest_jsonl}")
    print("=" * 80)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 80-clip pilot generation only")
    args = parser.parse_args()

    build_v12_dataset(is_pilot=args.pilot)
