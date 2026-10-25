#!/usr/bin/env python3
"""
Phase 21 Step 4: Synthetic Evaluation Corpus Generator.
Repository: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_phase21_synthetic_corpora.py

Generates:
  1. synthetic_eval_dev_v1 (20 sequences, Dev voices: Anh Khôi, Minh Quân)
  2. synthetic_eval_test_v1 (40 sequences, Test unseen voices: Ái Hân, Hữu Quân, Xuân Tiên, Mạnh Dũng)

Enforces:
  - Unseen Generator: VieNeu-TTS v3 Nano (24 kHz -> resampled to 16 kHz)
  - Strict Voice Disjointness (Dev voices != Test voices)
  - Strict OOV Lexeme Disjointness (Dev OOV != Test OOV)
  - Strict Noise Asset Disjointness (Tracks 28-33 for Dev, Tracks 34-39 for Test)
  - Auditable acoustic boundaries and exact gap measurement.
"""

import os
import sys
import io
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIENEU_PKG_DIR = PROJECT_ROOT / ".venv_vieneu" / "Lib" / "site-packages"
if str(VIENEU_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(VIENEU_PKG_DIR))

from vieneu import Vieneu

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

TARGET_SR = 16000
SYNTH_EVAL_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "synthetic_eval"
COMPONENTS_DIR = SYNTH_EVAL_DIR / "components"
AUDIO_DEV_DIR = SYNTH_EVAL_DIR / "audio" / "dev"
AUDIO_TEST_DIR = SYNTH_EVAL_DIR / "audio" / "test"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"

DEV_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_dev_manifest.csv"
TEST_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_test_manifest.csv"
COMPONENTS_MANIFEST_CSV = MANIFESTS_DIR / "synthetic_eval_components_manifest.csv"

# Pre-registered Lexicons
COMMANDS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống"
}

DEV_OOV = [
    "bật", "tắt", "dừng", "chạy", "mở", "đóng", "quay", "lùi", "tìm", "chọn",
    "xoay", "trông", "trẻ", "trường", "phim", "phục", "phấn", "lược", "lớp", "lửa"
]

TEST_OOV = [
    # Phonetic confusers
    "lăn", "len", "lân", "lền", "phái", "phay", "phai", "trại", "trát", "trai", "xuồng", "xưởng", "xướng",
    # General vocabulary
    "ghế", "bàn", "nhà", "đèn", "sách", "bút", "xe", "nước", "cơm", "đồng", "hồ"
]

DEV_VOICES = ["Anh Khôi", "Minh Quân"]
TEST_VOICES = ["Ái Hân", "Hữu Quân", "Xuân Tiên", "Mạnh Dũng"]


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def trim_to_speech_envelope(audio: np.ndarray, sr: int = 16000, threshold: float = 0.015, pad_samples: int = 480) -> Tuple[np.ndarray, int, int]:
    """Finds active speech boundaries, trimming excessive silence while preserving natural transitions."""
    above = np.where(np.abs(audio) >= threshold)[0]
    if len(above) == 0:
        return audio, 0, len(audio)

    start_idx = max(0, int(above[0]) - pad_samples)
    end_idx = min(len(audio), int(above[-1]) + pad_samples)
    trimmed = audio[start_idx:end_idx]
    return trimmed, start_idx, end_idx


def synthesize_all_components(tts: Vieneu) -> pd.DataFrame:
    """Synthesizes all individual speech clips needed for dev and test corpora."""
    print("\n--- 1. Synthesizing Component Clips using VieNeu v3 Nano ---")
    COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # 1. Dev Components (Commands + Dev OOV across Dev Voices)
    for voice in DEV_VOICES:
        safe_voice = voice.replace(" ", "_")
        # Target commands
        for cmd_label, cmd_text in COMMANDS.items():
            cid = f"synth_dev_v3nano_{cmd_label.lower()}_{safe_voice}"
            out_p = COMPONENTS_DIR / f"{cid}.wav"
            if not out_p.exists():
                raw_24k = tts.infer(cmd_text, voice=voice)
                num_16k = int(round(len(raw_24k) * 16000.0 / 24000.0))
                audio_16k = signal.resample(raw_24k, num_16k).astype(np.float32)
                peak = np.max(np.abs(audio_16k))
                if peak > 0.0:
                    audio_16k = audio_16k * (0.85 / max(peak, 0.85))
                sf.write(str(out_p), audio_16k, TARGET_SR)

            audio, sr = sf.read(str(out_p), dtype="float32")
            records.append({
                "component_id": cid,
                "split": "dev",
                "label": cmd_label,
                "is_target": True,
                "text": cmd_text,
                "oov_type": "none",
                "voice": voice,
                "engine": "VieNeu-TTS v3 Nano",
                "sample_rate": sr,
                "duration_seconds": round(len(audio) / sr, 4),
                "audio_path": str(out_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(out_p)
            })

        # Dev OOV
        for oov_word in DEV_OOV:
            cid = f"synth_dev_v3nano_oov_{oov_word}_{safe_voice}"
            out_p = COMPONENTS_DIR / f"{cid}.wav"
            if not out_p.exists():
                raw_24k = tts.infer(oov_word, voice=voice)
                num_16k = int(round(len(raw_24k) * 16000.0 / 24000.0))
                audio_16k = signal.resample(raw_24k, num_16k).astype(np.float32)
                peak = np.max(np.abs(audio_16k))
                if peak > 0.0:
                    audio_16k = audio_16k * (0.85 / max(peak, 0.85))
                sf.write(str(out_p), audio_16k, TARGET_SR)

            audio, sr = sf.read(str(out_p), dtype="float32")
            records.append({
                "component_id": cid,
                "split": "dev",
                "label": "REJECT",
                "is_target": False,
                "text": oov_word,
                "oov_type": "dev_general_oov",
                "voice": voice,
                "engine": "VieNeu-TTS v3 Nano",
                "sample_rate": sr,
                "duration_seconds": round(len(audio) / sr, 4),
                "audio_path": str(out_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(out_p)
            })

    # 2. Test Components (Commands + Test OOV across Unseen Test Voices)
    for voice in TEST_VOICES:
        safe_voice = voice.replace(" ", "_")
        for cmd_label, cmd_text in COMMANDS.items():
            cid = f"synth_test_v3nano_{cmd_label.lower()}_{safe_voice}"
            out_p = COMPONENTS_DIR / f"{cid}.wav"
            if not out_p.exists():
                raw_24k = tts.infer(cmd_text, voice=voice)
                num_16k = int(round(len(raw_24k) * 16000.0 / 24000.0))
                audio_16k = signal.resample(raw_24k, num_16k).astype(np.float32)
                peak = np.max(np.abs(audio_16k))
                if peak > 0.0:
                    audio_16k = audio_16k * (0.85 / max(peak, 0.85))
                sf.write(str(out_p), audio_16k, TARGET_SR)

            audio, sr = sf.read(str(out_p), dtype="float32")
            records.append({
                "component_id": cid,
                "split": "test",
                "label": cmd_label,
                "is_target": True,
                "text": cmd_text,
                "oov_type": "none",
                "voice": voice,
                "engine": "VieNeu-TTS v3 Nano",
                "sample_rate": sr,
                "duration_seconds": round(len(audio) / sr, 4),
                "audio_path": str(out_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(out_p)
            })

        for oov_word in TEST_OOV:
            is_confuser = oov_word in ["lăn", "len", "lân", "lền", "phái", "phay", "phai", "trại", "trát", "trai", "xuồng", "xưởng", "xướng"]
            oov_type = "phonetic_confuser" if is_confuser else "test_unseen_general"
            cid = f"synth_test_v3nano_oov_{oov_word}_{safe_voice}"
            out_p = COMPONENTS_DIR / f"{cid}.wav"
            if not out_p.exists():
                raw_24k = tts.infer(oov_word, voice=voice)
                num_16k = int(round(len(raw_24k) * 16000.0 / 24000.0))
                audio_16k = signal.resample(raw_24k, num_16k).astype(np.float32)
                peak = np.max(np.abs(audio_16k))
                if peak > 0.0:
                    audio_16k = audio_16k * (0.85 / max(peak, 0.85))
                sf.write(str(out_p), audio_16k, TARGET_SR)

            audio, sr = sf.read(str(out_p), dtype="float32")
            records.append({
                "component_id": cid,
                "split": "test",
                "label": "REJECT",
                "is_target": False,
                "text": oov_word,
                "oov_type": oov_type,
                "voice": voice,
                "engine": "VieNeu-TTS v3 Nano",
                "sample_rate": sr,
                "duration_seconds": round(len(audio) / sr, 4),
                "audio_path": str(out_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(out_p)
            })

    df_comp = pd.DataFrame(records)
    COMPONENTS_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(COMPONENTS_MANIFEST_CSV, index=False, encoding="utf-8")
    print(f"[OK] Total component clips synthesized: {len(df_comp)} -> saved to {COMPONENTS_MANIFEST_CSV}")
    return df_comp


def assemble_streaming_sequences(
    corpus_split: str,
    df_components: pd.DataFrame,
    num_sequences: int,
    noise_tracks: List[str],
    noise_mgr: NoiseAssetManager,
    out_dir: Path,
    out_manifest: Path,
    seed_base: int
) -> pd.DataFrame:
    """Assembles multi-command streaming sequences with exact acoustic gap alignment."""
    print(f"\n--- 2. Assembling {num_sequences} Streaming Sequences for {corpus_split.upper()} ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_components = df_components[df_components["split"] == corpus_split].copy()
    targets_pool = {
        cmd: split_components[split_components["label"] == cmd].to_dict("records")
        for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]
    }
    oov_pool = split_components[split_components["label"] == "REJECT"].to_dict("records")

    templates = [
        {"type": "permutation", "tokens": ["TRAI", "PHAI", "LEN", "XUONG"]},
        {"type": "permutation", "tokens": ["LEN", "XUONG", "TRAI", "PHAI"]},
        {"type": "permutation", "tokens": ["PHAI", "TRAI", "XUONG", "LEN"]},
        {"type": "permutation", "tokens": ["XUONG", "LEN", "PHAI", "TRAI"]},
        {"type": "repetition", "tokens": ["TRAI", "TRAI", "PHAI"]},
        {"type": "repetition", "tokens": ["LEN", "LEN", "XUONG", "XUONG"]},
        {"type": "confuser_interspersed", "tokens": ["TRAI", "OOV", "PHAI"]},
        {"type": "confuser_interspersed", "tokens": ["OOV", "LEN", "XUONG"]},
        {"type": "confuser_interspersed", "tokens": ["LEN", "OOV", "XUONG", "OOV"]},
        {"type": "command_burst", "tokens": ["XUONG", "TRAI", "LEN"]}
    ]

    gaps = [100, 150, 200, 250, 300, 400]
    conditions = [
        ("clean", None),
        ("snr20", 20.0),
        ("snr15", 15.0),
        ("snr10", 10.0),
        ("snr05", 5.0)
    ]

    lead_in_samples = int(0.200 * TARGET_SR)  # 200ms initial silence
    seq_records = []

    cmd_ptrs = {cmd: 0 for cmd in targets_pool}
    oov_ptr = 0

    for seq_idx in range(num_sequences):
        seq_num = seq_idx + 1
        seq_id = f"synthetic_{corpus_split}_seq_{seq_num:04d}"
        tmpl = templates[seq_idx % len(templates)]
        nominal_gap_ms = gaps[seq_idx % len(gaps)]
        cond_name, snr_target = conditions[seq_idx % len(conditions)]

        rng = np.random.RandomState(seed_base + seq_num * 31)

        # Build clean audio stream
        audio_stream = [np.zeros(lead_in_samples, dtype=np.float32)]
        curr_pos = lead_in_samples

        event_records = []
        comp_records = []
        tokens = tmpl["tokens"]

        for token_idx, tok in enumerate(tokens):
            if tok in targets_pool:
                pool = targets_pool[tok]
                comp = pool[cmd_ptrs[tok] % len(pool)]
                cmd_ptrs[tok] += 1
                is_target = True
                label = tok
            else:
                comp = oov_pool[oov_ptr % len(oov_pool)]
                oov_ptr += 1
                is_target = False
                label = "REJECT"

            raw_audio, sr = sf.read(str(PROJECT_ROOT / comp["audio_path"]), dtype="float32")
            trimmed_audio, t_start, t_end = trim_to_speech_envelope(raw_audio, TARGET_SR, threshold=0.015, pad_samples=240)
            t_len = len(trimmed_audio)

            evt_start_sec = round(curr_pos / TARGET_SR, 4)
            evt_end_sec = round((curr_pos + t_len) / TARGET_SR, 4)

            event_records.append({
                "event_index": token_idx,
                "label": label,
                "is_target": is_target,
                "text": comp["text"],
                "oov_type": comp["oov_type"],
                "start_sec": evt_start_sec,
                "end_sec": evt_end_sec,
                "duration_sec": round(t_len / TARGET_SR, 4),
                "voice": comp["voice"],
                "source_component_id": comp["component_id"]
            })

            comp_records.append({
                "component_id": comp["component_id"],
                "audio_path": comp["audio_path"],
                "sha256": comp["sha256"],
                "trim_start_sample": int(t_start),
                "trim_end_sample": int(t_end)
            })

            audio_stream.append(trimmed_audio)
            curr_pos += t_len

            # Insert inter-command gap if not last token
            if token_idx < len(tokens) - 1:
                gap_samples = int((nominal_gap_ms / 1000.0) * TARGET_SR)
                audio_stream.append(np.zeros(gap_samples, dtype=np.float32))
                curr_pos += gap_samples

        # 300ms trailing silence
        audio_stream.append(np.zeros(int(0.300 * TARGET_SR), dtype=np.float32))
        clean_audio = np.concatenate(audio_stream).astype(np.float32)

        # Add noise if condition != clean
        noise_track_name = "none_clean"
        obs_snr = float("inf")
        if cond_name != "clean" and snr_target is not None:
            track_idx = noise_tracks[seq_idx % len(noise_tracks)]
            noise_track_name = noise_mgr.noise_files[track_idx].name
            noise_clip = noise_mgr.extract_noise_slice(track_idx, seed=seed_base + seq_num * 101, duration_samples=len(clean_audio))
            final_audio, obs_snr, _, _ = mix_noise_at_snr(clean_audio, noise_clip, snr_target)
        else:
            final_audio = clean_audio

        # Peak clamping check
        peak = np.max(np.abs(final_audio))
        if peak > 1.0:
            final_audio = final_audio / peak

        out_wav_path = out_dir / f"{seq_id}.wav"
        sf.write(str(out_wav_path), final_audio, TARGET_SR)
        wav_sha = compute_file_sha256(out_wav_path)

        seq_records.append({
            "sequence_id": seq_id,
            "corpus_split": corpus_split,
            "nominal_gap_ms": nominal_gap_ms,
            "condition": cond_name,
            "target_snr_db": "clean" if cond_name == "clean" else str(snr_target),
            "observed_snr_db": round(obs_snr, 2) if obs_snr != float("inf") else "inf",
            "noise_asset_id": noise_track_name,
            "tokens": "->".join(tmpl["tokens"]),
            "sequence_type": tmpl["type"],
            "num_components": len(tokens),
            "num_target_events": sum(1 for e in event_records if e["is_target"]),
            "duration_seconds": round(len(final_audio) / TARGET_SR, 4),
            "audio_path": str(out_wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": wav_sha,
            "events_json": json.dumps(event_records, ensure_ascii=False),
            "components_json": json.dumps(comp_records, ensure_ascii=False)
        })

    df_seq = pd.DataFrame(seq_records)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df_seq.to_csv(out_manifest, index=False, encoding="utf-8")
    print(f"[OK] Assembled {len(df_seq)} sequences -> saved to {out_manifest}")
    return df_seq


def main():
    print("=" * 80)
    print("PHASE 21: BUILDING SYNTHETIC EVALUATION CORPORA")
    print("=" * 80)

    # 1. Initialize Vieneu v3 Nano
    print("[1/4] Initializing VieNeu-TTS v3 Nano...")
    tts = Vieneu(mode="v3nano")

    # 2. Synthesize all components
    df_components = synthesize_all_components(tts)

    # 3. Initialize noise asset manager
    noise_mgr = NoiseAssetManager(NOISE_DIR)
    num_tracks = noise_mgr.get_track_count()
    print(f"[2/4] Loaded {num_tracks} noise assets from {NOISE_DIR.name}")

    # Enforce strictly disjoint noise track allocation:
    # Dev: tracks 28-33 (6 tracks)
    # Test: tracks 34-39 (6 tracks)
    dev_noise_tracks = [28, 29, 30, 31, 32, 33]
    test_noise_tracks = [34, 35, 36, 37, 38, 39]
    print(f"Dev noise tracks:  {dev_noise_tracks}")
    print(f"Test noise tracks: {test_noise_tracks}")
    assert set(dev_noise_tracks).isdisjoint(set(test_noise_tracks)), "Noise tracks must be disjoint!"

    # 4. Assemble Dev Corpus (20 sequences)
    print("[3/4] Assembling synthetic_eval_dev_v1 (20 sequences)...")
    assemble_streaming_sequences(
        corpus_split="dev",
        df_components=df_components,
        num_sequences=20,
        noise_tracks=dev_noise_tracks,
        noise_mgr=noise_mgr,
        out_dir=AUDIO_DEV_DIR,
        out_manifest=DEV_MANIFEST_CSV,
        seed_base=42100
    )

    # 5. Assemble Test Corpus (40 sequences)
    print("[4/4] Assembling synthetic_eval_test_v1 (40 sequences)...")
    assemble_streaming_sequences(
        corpus_split="test",
        df_components=df_components,
        num_sequences=40,
        noise_tracks=test_noise_tracks,
        noise_mgr=noise_mgr,
        out_dir=AUDIO_TEST_DIR,
        out_manifest=TEST_MANIFEST_CSV,
        seed_base=42500
    )

    print("\n" + "=" * 80)
    print("PHASE 21 CORPUS GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
