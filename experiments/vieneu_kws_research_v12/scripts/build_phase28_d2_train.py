#!/usr/bin/env python3
"""
Phase 28-31 Stage B: Build D2 Edge-TTS-Augmented Training Distribution
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_phase28_d2_train.py

Replaces exactly 544 Turbo synthetic samples (10.0% of the 5,440 dataset)
with newly synthesized Edge-TTS audio across non-test voices (vi-VN-HoaiMyNeural, vi-VN-NamMinhNeural)
using strictly non-test training lexemes, preserving exact class counts (680 each, 2720 REJECT)
and exact 6-tier SNR matching down to 0 dB.
"""

import sys
import io
import time
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import edge_tts

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
AUDIO_OUT_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "d2_edge_tts_train"
RAW_CLEAN_DIR = AUDIO_OUT_DIR / "raw_clean_base"
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"

D0_TRAIN_SRC = MANIFESTS_DIR / "D0_train_v1.csv"
OUT_D2_MANIFEST = MANIFESTS_DIR / "D2_matched_train_v1.csv"
OUT_LEDGER_CSV = MANIFESTS_DIR / "source_eligibility_and_quarantine_ledger.csv"
OUT_LEDGER_JSON = MANIFESTS_DIR / "source_eligibility_and_quarantine_ledger.json"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

VOICES = [
    {"id": "vi-VN-HoaiMyNeural", "safe_id": "hoaimy_f"},
    {"id": "vi-VN-NamMinhNeural", "safe_id": "namminh_m"}
]

COMMANDS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống"
}

# Strictly excluded test lexemes from Phase 21 & Phase 27
EXCLUDED_TEST_LEXEMES = {
    "trời", "trên", "tròn", "trả", "pháo", "phố", "phòng", "phần", "lớn", "lấy", "lưng",
    "xương", "xuân", "xong", "cửa", "bàn", "ghế", "nhà", "xe", "tiền", "nước", "sách", "bút", "máy",
    "lăn", "len", "lân", "lền", "phái", "phay", "phai", "trại", "trát", "trai", "xuồng", "xưởng", "xướng"
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def synthesize_word(text: str, voice: str) -> np.ndarray:
    comm = edge_tts.Communicate(text, voice)
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]

    f = io.BytesIO(mp3_data)
    audio, sr = sf.read(f, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if sr != 16000:
        num_16k = int(round(len(audio) * 16000.0 / float(sr)))
        audio = signal.resample(audio, num_16k).astype(np.float32)

    # Trim energy < -45 dBFS
    energy = audio ** 2
    max_e = np.max(energy) if len(energy) > 0 else 0.0
    if max_e > 1e-6:
        thresh = max_e * 10.0 ** (-45.0 / 10.0)
        idx_valid = np.where(energy >= thresh)[0]
        if len(idx_valid) > 0:
            start_idx = max(0, idx_valid[0] - int(0.05 * 16000))
            end_idx = min(len(audio), idx_valid[-1] + int(0.05 * 16000))
            audio = audio[start_idx:end_idx]

    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0.0:
        audio = audio * (0.85 / peak)

    return audio.astype(np.float32)


def make_1s_clip(raw_audio: np.ndarray, target_samples: int = 16000) -> np.ndarray:
    if len(raw_audio) < target_samples:
        pad_total = target_samples - len(raw_audio)
        pad_front = min(int(0.15 * 16000), pad_total // 2)
        pad_back = pad_total - pad_front
        return np.pad(raw_audio, (pad_front, pad_back), mode="constant")
    elif len(raw_audio) > target_samples:
        return raw_audio[:target_samples]
    return raw_audio


async def main_async():
    print("=" * 80)
    print("PHASE 28-31 STAGE B: BUILD D2 EDGE-TTS-AUGMENTED TRAINING DISTRIBUTION")
    print("=" * 80)

    RAW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_d0 = pd.read_csv(D0_TRAIN_SRC)
    print(f"Loaded D0 baseline: {len(df_d0)} samples")

    # Extract non-test training OOV lexemes
    all_d0_oovs = set(df_d0[df_d0["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna().unique())
    clean_training_oovs = sorted(list(all_d0_oovs - EXCLUDED_TEST_LEXEMES))
    print(f"Extracted {len(clean_training_oovs)} strictly non-test training OOV lexemes.")

    # 1. Synthesize Clean Base Utterances
    clean_base_cache = {}
    print("\n--- Synthesizing Base Clean Training Audio via Edge-TTS ---")

    # Commands
    for cmd_lbl, cmd_text in COMMANDS.items():
        for v in VOICES:
            key = f"cmd_{cmd_lbl}_{v['safe_id']}"
            raw_path = RAW_CLEAN_DIR / f"{key}.wav"
            if not raw_path.exists():
                audio = await synthesize_word(cmd_text, v["id"])
                sf.write(str(raw_path), audio, 16000)
                await asyncio.sleep(0.05)
            data, _ = sf.read(str(raw_path), dtype="float32")
            clean_base_cache[key] = make_1s_clip(data)

    # OOVs
    for w in clean_training_oovs:
        safe_w = w.replace(" ", "_")
        for v in VOICES:
            key = f"oov_{safe_w}_{v['safe_id']}"
            raw_path = RAW_CLEAN_DIR / f"{key}.wav"
            if not raw_path.exists():
                audio = await synthesize_word(w, v["id"])
                sf.write(str(raw_path), audio, 16000)
                await asyncio.sleep(0.05)
            data, _ = sf.read(str(raw_path), dtype="float32")
            clean_base_cache[key] = make_1s_clip(data)

    print(f"[OK] Total unique base clean utterances ready: {len(clean_base_cache)}")

    # 2. Select 544 Stratified Samples in D0 for Replacement (10.0% Dose)
    print("\n--- Selecting 544 Stratified Samples for Substitution ---")
    rng = np.random.RandomState(42028)
    sub_indices = []

    # 68 per command class (272 total commands)
    for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
        cmd_synth_idxs = df_d0[(df_d0["label"] == cmd) & (df_d0["source_type"].isin(["tts_synthetic_clean", "noise_augmented"]))].index.to_numpy()
        chosen = []
        for snr in [np.nan, 20.0, 15.0, 10.0, 5.0, 0.0]:
            if pd.isna(snr):
                pool = [i for i in cmd_synth_idxs if pd.isna(df_d0.loc[i, "target_snr_db"])]
                n_take = 35
            else:
                pool = [i for i in cmd_synth_idxs if df_d0.loc[i, "target_snr_db"] == snr]
                n_take = 7
            chosen.extend(rng.choice(pool, size=min(n_take, len(pool)), replace=False).tolist())
        if len(chosen) > 68:
            chosen = chosen[:68]
        elif len(chosen) < 68:
            rem = [i for i in cmd_synth_idxs if i not in chosen]
            chosen.extend(rng.choice(rem, size=68 - len(chosen), replace=False).tolist())
        sub_indices.extend(chosen)

    # 272 OOV samples
    oov_synth_idxs = df_d0[(df_d0["label"] == "REJECT") & (df_d0["reject_subgroup"] == "oov_word")].index.to_numpy()
    chosen_oov = []
    for snr in [np.nan, 20.0, 15.0, 10.0, 5.0, 0.0]:
        if pd.isna(snr):
            pool = [i for i in oov_synth_idxs if pd.isna(df_d0.loc[i, "target_snr_db"])]
            n_take = 139
        else:
            pool = [i for i in oov_synth_idxs if df_d0.loc[i, "target_snr_db"] == snr]
            n_take = 27
        chosen_oov.extend(rng.choice(pool, size=min(n_take, len(pool)), replace=False).tolist())
    if len(chosen_oov) > 272:
        chosen_oov = chosen_oov[:272]
    elif len(chosen_oov) < 272:
        rem = [i for i in oov_synth_idxs if i not in chosen_oov]
        chosen_oov.extend(rng.choice(rem, size=272 - len(chosen_oov), replace=False).tolist())
    sub_indices.extend(chosen_oov)

    assert len(sub_indices) == 544, f"Expected 544 substituted indices, got {len(sub_indices)}"
    print(f"Selected exactly 544 samples in D0 for Edge-TTS substitution (272 Target, 272 OOV).")

    # 3. Generate Matched Noise Audio & Update D2 DataFrame
    print("\n--- Generating 544 Matched Audio Clips for D2 ---")
    noise_mgr = NoiseAssetManager(NOISE_DIR)
    df_d2 = df_d0.copy()

    voice_turn = 0
    for sample_count, idx in enumerate(sub_indices):
        orig_row = df_d0.loc[idx]
        lbl = orig_row["label"]
        snr_target = orig_row["target_snr_db"]

        v = VOICES[voice_turn % len(VOICES)]
        voice_turn += 1

        if lbl in COMMANDS:
            cmd_text = COMMANDS[lbl]
            base_key = f"cmd_{lbl}_{v['safe_id']}"
            clean_audio = clean_base_cache[base_key].copy()
            cid = f"d2_edge_{lbl.lower()}_{v['safe_id']}_{sample_count:04d}"
            lexeme = cmd_text
            stype = "edge_tts_clean" if pd.isna(snr_target) else "edge_tts_noise_augmented"
        else:
            word = orig_row["oov_lexeme"]
            if pd.isna(word) or word in EXCLUDED_TEST_LEXEMES:
                word = clean_training_oovs[sample_count % len(clean_training_oovs)]
            safe_w = word.replace(" ", "_")
            base_key = f"oov_{safe_w}_{v['safe_id']}"
            if base_key not in clean_base_cache:
                base_key = f"oov_{clean_training_oovs[0]}_{v['safe_id']}"
            clean_audio = clean_base_cache[base_key].copy()
            cid = f"d2_edge_oov_{safe_w}_{v['safe_id']}_{sample_count:04d}"
            lexeme = word
            stype = "edge_tts_oov_clean" if pd.isna(snr_target) else "edge_tts_oov_noise_augmented"

        # Matched noise mixing
        if pd.isna(snr_target) or snr_target == "clean":
            final_audio = clean_audio
            actual_noise_id = np.nan
        else:
            snr_val = float(snr_target)
            track_idx = sample_count % 28  # training pool tracks 0-27
            noise_slice = noise_mgr.extract_noise_slice(track_index=track_idx, seed=42900 + sample_count, duration_samples=16000)
            final_audio, obs_snr, _, _ = mix_noise_at_snr(
                speech=clean_audio,
                noise=noise_slice,
                target_snr_db=snr_val
            )
            actual_noise_id = f"mic_noise_track_{track_idx}.mp3"

        out_wav = AUDIO_OUT_DIR / f"{cid}.wav"
        sf.write(str(out_wav), final_audio.astype(np.float32), 16000)
        actual_sha = compute_sha256(out_wav)
        rel_path = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

        df_d2.loc[idx, "source_type"] = stype
        df_d2.loc[idx, "source_utterance_id"] = cid
        df_d2.loc[idx, "voice_or_speaker_id"] = v["id"]
        df_d2.loc[idx, "text"] = lexeme
        df_d2.loc[idx, "oov_lexeme"] = lexeme if lbl == "REJECT" else np.nan
        df_d2.loc[idx, "audio_path"] = rel_path
        df_d2.loc[idx, "num_samples"] = 16000
        df_d2.loc[idx, "duration_seconds"] = 1.0000
        df_d2.loc[idx, "sha256"] = actual_sha
        df_d2.loc[idx, "generation_config_id"] = "edge_tts_7.3.1_d2_10pct_recipe"
        df_d2.loc[idx, "source_license_or_permission"] = "Microsoft Azure Speech Service (Edge-TTS)"

    # 4. Invariant Verification
    assert len(df_d2) == 5440, f"D2 count {len(df_d2)} != 5440"
    assert df_d2["label"].value_counts().to_dict() == df_d0["label"].value_counts().to_dict(), "Class mismatch"
    assert df_d2["target_snr_db"].fillna(-999.0).value_counts().to_dict() == df_d0["target_snr_db"].fillna(-999.0).value_counts().to_dict(), "SNR mismatch"

    df_d2.to_csv(OUT_D2_MANIFEST, index=False, encoding="utf-8")
    d2_hash = compute_sha256(OUT_D2_MANIFEST)
    print(f"\n[OK] Successfully created D2_matched_train_v1.csv (SHA-256: {d2_hash})")
    print(f"     Total samples: {len(df_d2)}")
    print(f"     Edge-TTS samples: 544 (10.0% of dataset, 14.3% of synthetic speech)")
    print(f"     Voices: 272 HoaiMy (Female), 272 NamMinh (Male)")

    # 5. Update Quarantine Ledger
    print("\n--- Updating Quarantine Ledger ---")
    df_ledger = pd.read_csv(OUT_LEDGER_CSV)
    new_rows = []
    for idx in sub_indices:
        r = df_d2.loc[idx]
        new_rows.append({
            "source_id": r["source_utterance_id"],
            "sha256": r["sha256"],
            "label": r["label"],
            "reject_subgroup": str(r.get("reject_subgroup", "none")),
            "text": str(r.get("text", "")),
            "engine": "Microsoft Edge-TTS 7.3.1",
            "voice": str(r.get("voice_or_speaker_id", "none")),
            "target_snr_db": "clean" if pd.isna(r["target_snr_db"]) else str(r["target_snr_db"]),
            "historical_exposure": "phase28_d2_material",
            "eligibility_status": "APPROVED_TRAINING_MATERIAL (EDGE_TTS_AUGMENTED)",
            "audio_path": r["audio_path"]
        })

    df_new_ledger = pd.concat([df_ledger, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["source_id"])
    df_new_ledger.to_csv(OUT_LEDGER_CSV, index=False, encoding="utf-8")
    ledger_hash = compute_sha256(OUT_LEDGER_CSV)

    with open(OUT_LEDGER_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": "2026-10-25T01:20:00+07:00",
                "phase": "Phase 28-31 Stage B (D2 Assembly)",
                "total_ledger_records": len(df_new_ledger),
                "ledger_csv_sha256": ledger_hash
            },
            "status_summary": df_new_ledger["eligibility_status"].value_counts().to_dict(),
            "engine_summary": df_new_ledger["engine"].value_counts().to_dict()
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] Updated Quarantine Ledger ({len(df_new_ledger)} entries, SHA-256: {ledger_hash})")


if __name__ == "__main__":
    asyncio.run(main_async())
