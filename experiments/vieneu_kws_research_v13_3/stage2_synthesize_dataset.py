#!/usr/bin/env python3
"""
Phase 13.3 - Stage 2: Confirmatory Audio Synthesis and Dataset Freeze.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v13_3/stage2_synthesize_dataset.py

Synthesizes the complete 1,000-clip confirmatory evaluation dataset:
- 200 Target Commands (50 TRAI, 50 PHAI, 50 LEN, 50 XUONG)
- 200 Phonetic Minimal Pairs (50 per keyword direction)
- 200 Syntactic Carrier Collisions (50 per keyword direction)
- 300 General Conversational Non-Targets
- 100 Acoustic Background and Silence Clips

STRICT GOVERNANCE:
- Generates audio strictly according to the frozen catalogue and assigned 8 novel Northern voices.
- Zero model predictions are executed during synthesis.
- Enforces 16 kHz mono, 1.0-second (16,000 samples) PCM_16 WAV format.
- Creates immutable manifest and integrity validation report.
"""

import os
import sys
import json
import csv
import time
import hashlib
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v13_2"
    / "lexicon"
    / "confirmatory_lexicon_catalogue.json"
)
OUTPUT_DATA_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_3" / "data"
AUDIO_DIR = OUTPUT_DATA_DIR / "audio"
MANIFEST_CSV = OUTPUT_DATA_DIR / "confirmatory_evaluation_manifest.csv"
MANIFEST_JSON = OUTPUT_DATA_DIR / "confirmatory_evaluation_manifest.json"
INTEGRITY_REPORT_PATH = OUTPUT_DATA_DIR / "dataset_integrity_report.json"

TARGET_COMMAND_TEXTS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def format_to_1s_kws(audio_raw: np.ndarray, orig_sr: int = 48000, target_sr: int = 16000) -> np.ndarray:
    """Polyphase resample to 16 kHz, center/pad/crop to exactly 16,000 samples (1.0s), normalize peak to -1 dBFS."""
    if orig_sr != target_sr:
        # Exact 1:3 downsampling from 48 kHz to 16 kHz
        audio_16k = signal.resample_poly(audio_raw, target_sr // 16000, orig_sr // 16000).astype(np.float32)
    else:
        audio_16k = audio_raw.astype(np.float32)

    target_samples = target_sr  # 16,000 samples

    if len(audio_16k) < target_samples:
        pad_total = target_samples - len(audio_16k)
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        audio_16k = np.pad(audio_16k, (pad_front, pad_back), mode="constant", constant_values=0.0)
    elif len(audio_16k) > target_samples:
        # Find active energy center to avoid cutting keyword
        frame_len = int(0.025 * target_sr)
        hop_len = int(0.010 * target_sr)
        num_frames = (len(audio_16k) - frame_len) // hop_len + 1
        if num_frames > 0:
            energies = [np.sum(audio_16k[i * hop_len : i * hop_len + frame_len] ** 2) for i in range(num_frames)]
            center_frame = int(np.argmax(energies))
            center_sample = center_frame * hop_len + frame_len // 2
            start = max(0, center_sample - target_samples // 2)
            if start + target_samples > len(audio_16k):
                start = len(audio_16k) - target_samples
            audio_16k = audio_16k[start : start + target_samples]
        else:
            audio_16k = audio_16k[:target_samples]

    # Peak normalization to -1.0 dBFS (0.89125) to provide headroom and prevent clipping
    peak = float(np.max(np.abs(audio_16k)))
    if peak > 0.0:
        target_peak = float(10.0 ** (-1.0 / 20.0))  # ~0.89125
        audio_16k = audio_16k * (target_peak / peak)

    return audio_16k.astype(np.float32)


def generate_ambient_clip(family: str, seed: int, duration_samples: int = 16000) -> np.ndarray:
    """Generates acoustic ambient and background textures matching catalogue specifications."""
    rng = np.random.RandomState(seed)
    if family == "digital_silence":
        return np.zeros(duration_samples, dtype=np.float32)
    elif family == "office_ambient_fan":
        t = np.arange(duration_samples) / 16000.0
        # Low frequency fan harmonics (50 Hz, 100 Hz, 150 Hz) + pinkish noise
        hum = (
            0.004 * np.sin(2 * np.pi * 50 * t)
            + 0.002 * np.sin(2 * np.pi * 100 * t)
            + 0.001 * np.sin(2 * np.pi * 150 * t)
            + rng.normal(0, 0.001, duration_samples)
        )
        peak = np.max(np.abs(hum))
        if peak > 0.009:
            hum = hum * (0.009 / peak)
        return hum.astype(np.float32)
    elif family == "room_tone_gaussian":
        noise = rng.normal(0, 0.001, duration_samples)
        # Lowpass filter at 2 kHz
        b, a = signal.butter(2, 2000.0 / 8000.0, btype="low")
        noise = signal.lfilter(b, a, noise)
        peak = np.max(np.abs(noise))
        if peak > 0.0045:
            noise = noise * (0.0045 / peak)
        return noise.astype(np.float32)
    elif family == "acoustic_impulse_texture":
        # Subtle non-speech transients: paper rustle / keyboard tap simulation
        base = rng.normal(0, 0.0008, duration_samples)
        num_impulses = rng.randint(2, 6)
        for _ in range(num_impulses):
            pos = rng.randint(1000, duration_samples - 1000)
            imp_len = rng.randint(80, 240)
            env = np.hanning(imp_len)
            base[pos : pos + imp_len] += rng.normal(0, 0.02, imp_len) * env
        peak = np.max(np.abs(base))
        if peak > 0.045:
            base = base * (0.045 / peak)
        return base.astype(np.float32)
    else:
        return np.zeros(duration_samples, dtype=np.float32)


def run_stage2_synthesis():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 13.3 - STAGE 2: AUDIO SYNTHESIS & MANIFEST LOCK")
    print("=" * 80)

    # 1. Load Lexicon Catalogue
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        catalogue = json.load(f)

    speakers = catalogue["speakers"]
    spk_map = {s["speaker_id"]: s for s in speakers}
    categories = catalogue["categories"]

    # 2. Initialize Vieneu TTS Engine
    print("Initializing Vieneu TTS Engine (v3turbo)...")
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    print("[OK] Vieneu TTS Engine ready.")

    manifest_records = []
    t_start = time.time()

    # -------------------------------------------------------------------------
    # PART 1: 200 TARGET COMMANDS (50 TRAI, 50 PHAI, 50 LEN, 50 XUONG)
    # -------------------------------------------------------------------------
    print("\n[1/5] Synthesizing 200 Target Commands...")
    # Balanced speaker allocation across 8 speakers:
    # 50 per keyword: spk 0..1 get 7 clips each (14), spk 2..7 get 6 clips each (36) -> 50 total
    cmd_spk_alloc = [
        ("spk_north_01", 7), ("spk_north_02", 7),
        ("spk_north_03", 6), ("spk_north_04", 6),
        ("spk_north_05", 6), ("spk_north_06", 6),
        ("spk_north_07", 6), ("spk_north_08", 6)
    ]

    for kw in ["TRAI", "PHAI", "LEN", "XUONG"]:
        text = TARGET_COMMAND_TEXTS[kw]
        clip_num = 0
        for spk_id, count in cmd_spk_alloc:
            spk_info = spk_map[spk_id]
            for rep in range(count):
                clip_num += 1
                clip_id = f"v13_conf_cmd_{kw.lower()}_{clip_num:03d}"
                wav_path = AUDIO_DIR / f"{clip_id}.wav"

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    raw_audio = tts.infer(text, voice=spk_info["voice_name"], temperature=0.8)
                    kws_audio = format_to_1s_kws(raw_audio, orig_sr=48000, target_sr=16000)
                    sf.write(str(wav_path), kws_audio, 16000, subtype="PCM_16")

                wav_sha = compute_sha256(wav_path)
                data, sr = sf.read(str(wav_path))
                peak_amp = float(np.max(np.abs(data)))

                manifest_records.append({
                    "clip_id": clip_id,
                    "rel_path": f"experiments/vieneu_kws_research_v13_3/data/audio/{clip_id}.wav",
                    "target_class": kw,
                    "ground_truth": kw,
                    "category": "target_commands",
                    "subcategory": kw,
                    "speaker_id": spk_id,
                    "voice_name": spk_info["voice_name"],
                    "voice_gender": spk_info["gender"],
                    "voice_accent": spk_info["accent"],
                    "intended_text": text,
                    "sample_rate": sr,
                    "num_samples": len(data),
                    "duration_sec": round(len(data) / sr, 4),
                    "peak_amplitude": round(peak_amp, 5),
                    "sha256": wav_sha
                })
        print(f"  - Completed {kw}: 50 clips generated.")

    # -------------------------------------------------------------------------
    # PART 2: 200 PHONETIC MINIMAL PAIRS (50 per keyword direction)
    # -------------------------------------------------------------------------
    print("\n[2/5] Synthesizing 200 Phonetic Minimal Pairs...")
    foil_texts = categories["phonetic_minimal_pairs"]["texts"]
    for kw in ["TRAI", "PHAI", "LEN", "XUONG"]:
        texts = foil_texts[kw]
        for idx, text in enumerate(texts):
            clip_num = idx + 1
            clip_id = f"v13_conf_foil_{kw.lower()}_{clip_num:03d}"
            wav_path = AUDIO_DIR / f"{clip_id}.wav"

            spk_info = speakers[idx % len(speakers)]
            spk_id = spk_info["speaker_id"]

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                raw_audio = tts.infer(text, voice=spk_info["voice_name"], temperature=0.8)
                kws_audio = format_to_1s_kws(raw_audio, orig_sr=48000, target_sr=16000)
                sf.write(str(wav_path), kws_audio, 16000, subtype="PCM_16")

            wav_sha = compute_sha256(wav_path)
            data, sr = sf.read(str(wav_path))
            peak_amp = float(np.max(np.abs(data)))

            manifest_records.append({
                "clip_id": clip_id,
                "rel_path": f"experiments/vieneu_kws_research_v13_3/data/audio/{clip_id}.wav",
                "target_class": "REJECT",
                "ground_truth": "REJECT",
                "category": "phonetic_minimal_pairs",
                "subcategory": f"{kw}_foil",
                "speaker_id": spk_id,
                "voice_name": spk_info["voice_name"],
                "voice_gender": spk_info["gender"],
                "voice_accent": spk_info["accent"],
                "intended_text": text,
                "sample_rate": sr,
                "num_samples": len(data),
                "duration_sec": round(len(data) / sr, 4),
                "peak_amplitude": round(peak_amp, 5),
                "sha256": wav_sha
            })
        print(f"  - Completed {kw} foils: 50 clips generated.")

    # -------------------------------------------------------------------------
    # PART 3: 200 SYNTACTIC CARRIER COLLISIONS (50 per keyword direction)
    # -------------------------------------------------------------------------
    print("\n[3/5] Synthesizing 200 Syntactic Carrier Collisions...")
    carrier_texts = categories["syntactic_carrier_collisions"]["texts"]
    for kw in ["TRAI", "PHAI", "LEN", "XUONG"]:
        texts = carrier_texts[kw]
        for idx, text in enumerate(texts):
            clip_num = idx + 1
            clip_id = f"v13_conf_carrier_{kw.lower()}_{clip_num:03d}"
            wav_path = AUDIO_DIR / f"{clip_id}.wav"

            spk_info = speakers[idx % len(speakers)]
            spk_id = spk_info["speaker_id"]

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                raw_audio = tts.infer(text, voice=spk_info["voice_name"], temperature=0.8)
                kws_audio = format_to_1s_kws(raw_audio, orig_sr=48000, target_sr=16000)
                sf.write(str(wav_path), kws_audio, 16000, subtype="PCM_16")

            wav_sha = compute_sha256(wav_path)
            data, sr = sf.read(str(wav_path))
            peak_amp = float(np.max(np.abs(data)))

            manifest_records.append({
                "clip_id": clip_id,
                "rel_path": f"experiments/vieneu_kws_research_v13_3/data/audio/{clip_id}.wav",
                "target_class": "REJECT",
                "ground_truth": "REJECT",
                "category": "syntactic_carrier_collisions",
                "subcategory": f"{kw}_carrier",
                "speaker_id": spk_id,
                "voice_name": spk_info["voice_name"],
                "voice_gender": spk_info["gender"],
                "voice_accent": spk_info["accent"],
                "intended_text": text,
                "sample_rate": sr,
                "num_samples": len(data),
                "duration_sec": round(len(data) / sr, 4),
                "peak_amplitude": round(peak_amp, 5),
                "sha256": wav_sha
            })
        print(f"  - Completed {kw} carriers: 50 clips generated.")

    # -------------------------------------------------------------------------
    # PART 4: 300 GENERAL CONVERSATIONAL NON-TARGETS
    # -------------------------------------------------------------------------
    print("\n[4/5] Synthesizing 300 General Conversational Non-Targets...")
    conv_texts = categories["general_conversational_non_targets"]["texts"]
    for idx, text in enumerate(conv_texts):
        clip_num = idx + 1
        clip_id = f"v13_conf_conv_{clip_num:03d}"
        wav_path = AUDIO_DIR / f"{clip_id}.wav"

        spk_info = speakers[idx % len(speakers)]
        spk_id = spk_info["speaker_id"]

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raw_audio = tts.infer(text, voice=spk_info["voice_name"], temperature=0.8)
            kws_audio = format_to_1s_kws(raw_audio, orig_sr=48000, target_sr=16000)
            sf.write(str(wav_path), kws_audio, 16000, subtype="PCM_16")

        wav_sha = compute_sha256(wav_path)
        data, sr = sf.read(str(wav_path))
        peak_amp = float(np.max(np.abs(data)))

        manifest_records.append({
            "clip_id": clip_id,
            "rel_path": f"experiments/vieneu_kws_research_v13_3/data/audio/{clip_id}.wav",
            "target_class": "REJECT",
            "ground_truth": "REJECT",
            "category": "general_conversational_non_targets",
            "subcategory": "conversational",
            "speaker_id": spk_id,
            "voice_name": spk_info["voice_name"],
            "voice_gender": spk_info["gender"],
            "voice_accent": spk_info["accent"],
            "intended_text": text,
            "sample_rate": sr,
            "num_samples": len(data),
            "duration_sec": round(len(data) / sr, 4),
            "peak_amplitude": round(peak_amp, 5),
            "sha256": wav_sha
        })
        if (idx + 1) % 50 == 0:
            print(f"  - Conversational: {idx + 1}/300 clips completed.")

    # -------------------------------------------------------------------------
    # PART 5: 100 ACOUSTIC BACKGROUND & SILENCE
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating 100 Acoustic Background and Silence clips...")
    bg_specs = [
        ("digital_silence", 25, 10000),
        ("office_ambient_fan", 25, 20000),
        ("room_tone_gaussian", 25, 30000),
        ("acoustic_impulse_texture", 25, 40000)
    ]

    for family, count, base_seed in bg_specs:
        for i in range(count):
            clip_num = i + 1
            clip_id = f"v13_conf_bg_{family}_{clip_num:02d}"
            wav_path = AUDIO_DIR / f"{clip_id}.wav"

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                audio_bg = generate_ambient_clip(family, seed=base_seed + i, duration_samples=16000)
                sf.write(str(wav_path), audio_bg, 16000, subtype="PCM_16")

            wav_sha = compute_sha256(wav_path)
            data, sr = sf.read(str(wav_path))
            peak_amp = float(np.max(np.abs(data)))

            manifest_records.append({
                "clip_id": clip_id,
                "rel_path": f"experiments/vieneu_kws_research_v13_3/data/audio/{clip_id}.wav",
                "target_class": "REJECT",
                "ground_truth": "REJECT",
                "category": "acoustic_background_and_silence",
                "subcategory": family,
                "speaker_id": "NONE",
                "voice_name": "NONE",
                "voice_gender": "NONE",
                "voice_accent": "NONE",
                "intended_text": f"Acoustic Background: {family}",
                "sample_rate": sr,
                "num_samples": len(data),
                "duration_sec": round(len(data) / sr, 4),
                "peak_amplitude": round(peak_amp, 5),
                "sha256": wav_sha
            })
        print(f"  - Completed {family}: {count} clips generated.")

    t_elapsed = time.time() - t_start
    print(f"\n[OK] All 1,000 clips generated in {t_elapsed:.1f}s.")
    assert len(manifest_records) == 1000, f"Expected 1000 records, got {len(manifest_records)}"

    # 3. Write Manifest Files (CSV and JSON)
    keys = list(manifest_records[0].keys())
    with open(MANIFEST_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest_records)
    print(f"[OK] Manifest CSV written: {MANIFEST_CSV.name}")

    manifest_sha = compute_sha256(MANIFEST_CSV)

    manifest_meta = {
        "dataset_name": "Vietnamese_Confirmatory_KWS_Dataset_v13_3",
        "protocol_version": "13.2.0_amended",
        "amendment": "Amendment 13.1-A",
        "total_clips": len(manifest_records),
        "manifest_sha256": manifest_sha,
        "sample_rate": 16000,
        "format": "PCM_16",
        "channels": 1,
        "duration_sec": 1.0,
        "num_samples": 16000,
        "categories_breakdown": {
            "target_commands": 200,
            "phonetic_minimal_pairs": 200,
            "syntactic_carrier_collisions": 200,
            "general_conversational_non_targets": 300,
            "acoustic_background_and_silence": 100
        },
        "target_class_breakdown": {
            "TRAI": 50,
            "PHAI": 50,
            "LEN": 50,
            "XUONG": 50,
            "REJECT": 800
        },
        "records": manifest_records
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Manifest JSON written: {MANIFEST_JSON.name}")

    # 4. Perform Rigorous Dataset Integrity Audit
    print("\n--- Performing Immutable Dataset Integrity Audit ---")
    clip_ids = set()
    sha_set = set()
    category_counts = {}
    class_counts = {}
    speaker_counts = {}

    for r in manifest_records:
        cid = r["clip_id"]
        assert cid not in clip_ids, f"Duplicate clip_id detected: {cid}"
        clip_ids.add(cid)

        # Check file exists and properties
        fpath = PROJECT_ROOT / r["rel_path"]
        assert fpath.exists(), f"Audio file missing: {fpath}"
        data, sr = sf.read(str(fpath))
        assert sr == 16000, f"Sample rate mismatch for {cid}: {sr}"
        assert len(data) == 16000, f"Sample length mismatch for {cid}: {len(data)}"
        assert len(data.shape) == 1 or data.shape[1] == 1, f"Audio is not mono for {cid}"
        assert np.max(np.abs(data)) <= 1.0, f"Audio clipping detected for {cid}: {np.max(np.abs(data))}"

        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        cls = r["target_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

        spk = r["speaker_id"]
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    integrity_report = {
        "status": "DATASET_INTEGRITY_VERIFIED_AND_LOCKED",
        "total_clips": len(manifest_records),
        "manifest_csv_sha256": manifest_sha,
        "sample_rate_verified": 16000,
        "sample_length_verified": 16000,
        "channels_verified": "mono",
        "clipping_detected": False,
        "category_counts": category_counts,
        "class_counts": class_counts,
        "speaker_counts": speaker_counts,
        "dataset_locked": True,
        "lock_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(INTEGRITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Integrity report written: {INTEGRITY_REPORT_PATH.name}")
    print("\nSTAGE 2 GATE PASSED: Dataset is formally locked and frozen for one-shot evaluation.")


if __name__ == "__main__":
    run_stage2_synthesis()
