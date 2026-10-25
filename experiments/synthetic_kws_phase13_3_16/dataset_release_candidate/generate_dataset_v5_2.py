"""
Phase 14: Synthetic KWS V5.2 Release Candidate Dataset Generator
Generates a 3,000-clip balanced corpus:
- 500 TRAI, 500 PHAI, 500 LEN, 500 XUONG, 500 UNKNOWN, 500 SILENCE
- Splits: 2,100 Train (70%), 450 Val (15%), 450 Test (15%)
- Strict split isolation: disjoint simulated speaker profiles and non-overlapping seed domains.
"""

import os
import sys
import json
import wave
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path("experiments/synthetic_kws_v5_2_candidate").resolve()))
from config import V5_2VoiceProfile, V5_2AcousticConfig, V5_2NoiseConfig, TARGET_WORDS_SPEC_V5_2
from generator import (
    generate_v5_2_speech_clip,
    generate_v5_2_silence_clip,
    reproduce_from_manifest_row,
    SAMPLE_RATE,
    TOTAL_SAMPLES
)

# 12 Disjoint Speaker Profiles across splits
SPEAKERS_BY_SPLIT = {
    "train": [
        V5_2VoiceProfile("SPK_TR_01_FEMALE_HIGH", 0.90, 225.0, 0.50, -10.0, 0.006),
        V5_2VoiceProfile("SPK_TR_02_FEMALE_MID", 0.94, 195.0, 0.55, -11.0, 0.007),
        V5_2VoiceProfile("SPK_TR_03_MALE_LOW", 1.10, 115.0, 0.62, -13.0, 0.008),
        V5_2VoiceProfile("SPK_TR_04_MALE_MID", 1.04, 140.0, 0.58, -12.0, 0.007),
        V5_2VoiceProfile("SPK_TR_05_CHILD_NEUTRAL", 0.86, 250.0, 0.48, -9.0, 0.005),
        V5_2VoiceProfile("SPK_TR_06_ADULT_DEEP", 1.14, 102.0, 0.65, -14.0, 0.009),
        V5_2VoiceProfile("SPK_TR_07_FEMALE_BREATHY", 0.92, 210.0, 0.68, -15.0, 0.010),
        V5_2VoiceProfile("SPK_TR_08_MALE_TIGHT", 1.02, 148.0, 0.45, -8.0, 0.005),
    ],
    "val": [
        V5_2VoiceProfile("SPK_VAL_01_FEMALE", 0.91, 215.0, 0.52, -10.5, 0.006),
        V5_2VoiceProfile("SPK_VAL_02_MALE", 1.08, 125.0, 0.60, -12.5, 0.008),
    ],
    "test": [
        V5_2VoiceProfile("SPK_TEST_01_FEMALE", 0.88, 230.0, 0.49, -9.5, 0.005),
        V5_2VoiceProfile("SPK_TEST_02_MALE", 1.12, 110.0, 0.63, -13.5, 0.009),
    ]
}

SPLIT_QUOTAS = {
    "train": {"per_class": 350, "total": 2100, "seed_base": 100000, "noise_seed_base": 1000000},
    "val":   {"per_class": 75,  "total": 450,  "seed_base": 200000, "noise_seed_base": 2000000},
    "test":  {"per_class": 75,  "total": 450,  "seed_base": 300000, "noise_seed_base": 3000000}
}

UNKNOWN_VOCABULARY = [
    # Initial confusers
    ("trời", "confuser_initial_sharing"),
    ("pha", "confuser_initial_sharing"),
    ("tròn", "confuser_initial_sharing"),
    ("phố", "confuser_initial_sharing"),
    ("lúa", "confuser_initial_sharing"),
    ("xinh", "confuser_initial_sharing"),
    # Rhyme confusers
    ("mái", "confuser_rhyme_sharing"),
    ("tai", "confuser_rhyme_sharing"),
    ("đèn", "confuser_rhyme_sharing"),
    ("cuống", "confuser_rhyme_sharing"),
    ("sen", "confuser_rhyme_sharing"),
    ("bài", "confuser_rhyme_sharing"),
    # General words
    ("chào", "open_vocabulary_general"),
    ("bạn", "open_vocabulary_general"),
    ("đi", "open_vocabulary_general"),
    ("xe", "open_vocabulary_general"),
    ("nhà", "open_vocabulary_general"),
    ("số", "open_vocabulary_general"),
    ("ngày", "open_vocabulary_general"),
    ("học", "open_vocabulary_general"),
    ("cơm", "open_vocabulary_general"),
    ("nước", "open_vocabulary_general"),
    ("sách", "open_vocabulary_general"),
    ("bút", "open_vocabulary_general")
]

SILENCE_SUBCLASSES = [
    ("ambient_room_tone", 0.60),
    ("ambient_low_freq_hum", 0.25),
    ("ambient_digital_zero", 0.15)
]

def generate_v5_2_release_candidate_dataset():
    out_dir = Path("experiments/synthetic_kws_phase13_3_16/dataset_release_candidate")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    target_labels = ["TRAI", "PHAI", "LEN", "XUONG"]
    all_classes = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    class_id_map = {c: i for i, c in enumerate(all_classes)}

    manifest_rows = []
    print("=== Generating 3,000-clip V5.2 Release Candidate Dataset ===")

    for split_name, q_info in SPLIT_QUOTAS.items():
        n_per_class = q_info["per_class"]
        seed_base = q_info["seed_base"]
        noise_seed_base = q_info["noise_seed_base"]
        spk_list = SPEAKERS_BY_SPLIT[split_name]
        split_rng = np.random.RandomState(seed_base)

        print(f"\n--- Generating {split_name.upper()} split ({n_per_class} per class, {len(all_classes)*n_per_class} total) ---")

        for c_idx, c_label in enumerate(all_classes):
            for i in range(n_per_class):
                rec_idx = c_idx * n_per_class + i
                cur_seed = seed_base + rec_idx
                cur_noise_seed = noise_seed_base + rec_idx
                rec_id = f"v5_2_rc_{split_name}_{c_label.lower()}_{i+1:04d}"
                wav_filename = f"{rec_id}.wav"
                wav_path = audio_dir / wav_filename

                item_rng = np.random.RandomState(cur_seed)

                # Select speaker profile
                base_spk = spk_list[item_rng.choice(len(spk_list))]
                spk = V5_2VoiceProfile(
                    voice_id=base_spk.voice_id,
                    vtl_scale=float(base_spk.vtl_scale * item_rng.uniform(0.98, 1.02)),
                    f0_base=float(base_spk.f0_base * item_rng.uniform(0.96, 1.04)),
                    open_quotient=float(np.clip(base_spk.open_quotient * item_rng.uniform(0.95, 1.05), 0.35, 0.75)),
                    spectral_tilt_db=float(base_spk.spectral_tilt_db + item_rng.uniform(-1.0, 1.0)),
                    jitter_pct=float(np.clip(base_spk.jitter_pct * item_rng.uniform(0.90, 1.10), 0.002, 0.015))
                )

                duration_sec = float(item_rng.uniform(0.38, 0.58))
                onset_offset_sec = float(item_rng.uniform(0.060, 0.220))
                target_speech_rms_dbfs = float(item_rng.uniform(-16.5, -14.0))
                background_rms_dbfs = float(item_rng.uniform(-42.0, -32.0))
                noise_type = item_rng.choice(["room_tone", "dither", "hum"], p=[0.70, 0.15, 0.15])

                if c_label in target_labels:
                    word_spec = TARGET_WORDS_SPEC_V5_2[c_label]
                    transcript = word_spec["word"]
                    acoustic_subclass = "target_keyword"
                    cfg = V5_2AcousticConfig(
                        sample_id=rec_id,
                        class_label=c_label,
                        acoustic_subclass=acoustic_subclass,
                        intended_text=transcript,
                        seed=cur_seed,
                        voice_profile=spk,
                        duration_sec=duration_sec,
                        onset_offset_sec=onset_offset_sec,
                        target_speech_rms_dbfs=target_speech_rms_dbfs,
                        noise_type=noise_type,
                        noise_seed=cur_noise_seed,
                        background_rms_dbfs=background_rms_dbfs
                    )
                    audio, gain_factor, effective_snr_db = generate_v5_2_speech_clip(cfg, sr=SAMPLE_RATE)

                elif c_label == "UNKNOWN":
                    vocab_item = UNKNOWN_VOCABULARY[item_rng.choice(len(UNKNOWN_VOCABULARY))]
                    transcript = vocab_item[0]
                    acoustic_subclass = vocab_item[1]
                    cfg = V5_2AcousticConfig(
                        sample_id=rec_id,
                        class_label=c_label,
                        acoustic_subclass=acoustic_subclass,
                        intended_text=transcript,
                        seed=cur_seed,
                        voice_profile=spk,
                        duration_sec=duration_sec,
                        onset_offset_sec=onset_offset_sec,
                        target_speech_rms_dbfs=target_speech_rms_dbfs,
                        noise_type=noise_type,
                        noise_seed=cur_noise_seed,
                        background_rms_dbfs=background_rms_dbfs
                    )
                    audio, gain_factor, effective_snr_db = generate_v5_2_speech_clip(cfg, sr=SAMPLE_RATE)

                else: # SILENCE
                    sub_choices = [s[0] for s in SILENCE_SUBCLASSES]
                    sub_probs = [s[1] for s in SILENCE_SUBCLASSES]
                    acoustic_subclass = item_rng.choice(sub_choices, p=sub_probs)
                    transcript = "<silence>"
                    cfg_noise = V5_2NoiseConfig(
                        sample_id=rec_id,
                        class_label=c_label,
                        acoustic_subclass=acoustic_subclass,
                        seed=cur_seed,
                        duration_sec=1.0,
                        sr=SAMPLE_RATE
                    )
                    audio = generate_v5_2_silence_clip(cfg_noise, sr=SAMPLE_RATE)
                    gain_factor = 1.0
                    effective_snr_db = -99.0
                    target_speech_rms_dbfs = -99.0

                # Write audio file
                sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
                sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()

                row_dict = {
                    "recording_id": rec_id,
                    "split": split_name,
                    "class_id": class_id_map[c_label],
                    "class_label": c_label,
                    "acoustic_subclass": acoustic_subclass,
                    "transcript": transcript,
                    "generator_version": "v5.2-release-candidate",
                    "generator_recipe": "articulatory_glottal_filter_standardized_rms",
                    "seed": cur_seed,
                    "simulated_speaker_id": spk.voice_id if c_label != "SILENCE" else "N/A",
                    "vtl_scale": spk.vtl_scale if c_label != "SILENCE" else 1.0,
                    "f0_base": spk.f0_base if c_label != "SILENCE" else 0.0,
                    "open_quotient": spk.open_quotient if c_label != "SILENCE" else 0.0,
                    "spectral_tilt_db": spk.spectral_tilt_db if c_label != "SILENCE" else 0.0,
                    "jitter_pct": spk.jitter_pct if c_label != "SILENCE" else 0.0,
                    "duration_sec": duration_sec if c_label != "SILENCE" else 1.0,
                    "onset_offset_sec": onset_offset_sec if c_label != "SILENCE" else 0.0,
                    "target_speech_rms_dbfs": target_speech_rms_dbfs,
                    "gain_factor": gain_factor,
                    "noise_type": noise_type if c_label != "SILENCE" else acoustic_subclass,
                    "noise_seed": cur_noise_seed,
                    "background_rms_dbfs": background_rms_dbfs if c_label != "SILENCE" else float(20.0 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)),
                    "effective_snr_db": effective_snr_db,
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "n_samples": TOTAL_SAMPLES,
                    "pcm_subtype": "PCM_16",
                    "sha256": sha256,
                    "processed_path": f"experiments/synthetic_kws_phase13_3_16/dataset_release_candidate/audio/{wav_filename}",
                    "status": "APPROVED_SYNTHETIC"
                }
                manifest_rows.append(row_dict)

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved main manifest ({len(df_manifest)} rows) to {manifest_path}")

    # Save split manifests
    for split_name in ["train", "val", "test"]:
        df_split = df_manifest[df_manifest["split"] == split_name]
        split_file = splits_dir / f"{split_name}.csv"
        df_split.to_csv(split_file, index=False)
        print(f"Saved {split_name} split manifest ({len(df_split)} rows) to {split_file}")

    print("\nDataset generation complete!")

if __name__ == "__main__":
    generate_v5_2_release_candidate_dataset()
