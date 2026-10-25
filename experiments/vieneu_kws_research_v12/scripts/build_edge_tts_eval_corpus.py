#!/usr/bin/env python3
"""
Phase 27: Build Independent Synthetic Evaluation Corpus using Edge-TTS
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_edge_tts_eval_corpus.py
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
AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "edge_tts_eval"
STREAMING_DIR = AUDIO_DIR / "streaming_sequences"
ISOLATED_DIR = AUDIO_DIR / "isolated_clips"
RAW_CLEAN_DIR = AUDIO_DIR / "raw_clean_base"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

VOICES = [
    {"id": "vi-VN-HoaiMyNeural", "gender": "female", "region": "North", "safe_id": "hoaimy_f"},
    {"id": "vi-VN-NamMinhNeural", "gender": "male", "region": "North", "safe_id": "namminh_m"}
]

COMMANDS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống"
}

DEV_OOV = [
    "bật", "tắt", "dừng", "chạy", "mở", "đóng", "quay", "lùi", "tìm", "chọn"
]

TEST_OOV_PHONETIC = [
    "trời", "trên", "tròn", "trả", "pháo", "phố", "phòng", "phần", "lớn", "lấy", "lưng", "xương", "xuân", "xong"
]

TEST_OOV_GENERAL = [
    "cửa", "bàn", "ghế", "nhà", "xe", "tiền", "nước", "sách", "bút", "máy"
]

TEST_OOV = TEST_OOV_PHONETIC + TEST_OOV_GENERAL


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def synthesize_word(text: str, voice: str) -> np.ndarray:
    """Synthesizes text via edge-tts, decodes MP3 stream, and resamples to 16 kHz float32."""
    comm = edge_tts.Communicate(text, voice)
    mp3_data = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]

    f = io.BytesIO(mp3_data)
    audio, sr = sf.read(f, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Resample 24k -> 16k
    if sr != 16000:
        num_16k = int(round(len(audio) * 16000.0 / float(sr)))
        audio = signal.resample(audio, num_16k).astype(np.float32)

    # Trim extreme silence padding (< -45 dBFS)
    energy = audio ** 2
    max_e = np.max(energy) if len(energy) > 0 else 0.0
    if max_e > 1e-6:
        thresh = max_e * 10.0 ** (-45.0 / 10.0)
        idx_valid = np.where(energy >= thresh)[0]
        if len(idx_valid) > 0:
            start_idx = max(0, idx_valid[0] - int(0.05 * 16000))
            end_idx = min(len(audio), idx_valid[-1] + int(0.05 * 16000))
            audio = audio[start_idx:end_idx]

    # Peak normalize to 0.85
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0.0:
        audio = audio * (0.85 / peak)

    return audio.astype(np.float32)


def make_1s_clip(raw_audio: np.ndarray, target_samples: int = 16000) -> np.ndarray:
    """Centers / pads raw audio into a 1.0s window."""
    if len(raw_audio) < target_samples:
        pad_total = target_samples - len(raw_audio)
        pad_front = min(int(0.15 * 16000), pad_total // 2)
        pad_back = pad_total - pad_front
        return np.pad(raw_audio, (pad_front, pad_back), mode="constant")
    elif len(raw_audio) > target_samples:
        return raw_audio[:target_samples]
    return raw_audio


async def generate_all_base_clean():
    RAW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_DIR.mkdir(parents=True, exist_ok=True)
    STREAMING_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Step 1: Synthesizing Base Clean Utterances via Edge-TTS ---")
    clean_cache = {}

    all_words = {}
    for c_id, text in COMMANDS.items():
        all_words[f"cmd_{c_id}"] = (text, c_id, "command")
    for w in DEV_OOV:
        all_words[f"oov_dev_{w}"] = (w, "REJECT", "dev_oov")
    for w in TEST_OOV_PHONETIC:
        all_words[f"oov_test_phonetic_{w}"] = (w, "REJECT", "test_oov_phonetic")
    for w in TEST_OOV_GENERAL:
        all_words[f"oov_test_general_{w}"] = (w, "REJECT", "test_oov_general")

    total_syn = len(all_words) * len(VOICES)
    done_count = 0

    for w_key, (text, label, group) in all_words.items():
        for v in VOICES:
            safe_v = v["safe_id"]
            cache_key = f"{w_key}_{safe_v}"
            raw_file = RAW_CLEAN_DIR / f"{cache_key}.wav"

            if not raw_file.exists():
                audio_16k = await synthesize_word(text, v["id"])
                sf.write(str(raw_file), audio_16k, 16000)
                await asyncio.sleep(0.05)  # small throttle

            audio_data, _ = sf.read(str(raw_file), dtype="float32")
            clean_cache[cache_key] = {
                "audio": audio_data,
                "text": text,
                "label": label,
                "group": group,
                "voice_id": v["id"],
                "voice_safe": safe_v,
                "gender": v["gender"],
                "duration": len(audio_data) / 16000.0
            }
            done_count += 1
            if done_count % 10 == 0 or done_count == total_syn:
                print(f"  Synthesized {done_count}/{total_syn} utterances...")

    print(f"[OK] Total unique base clean utterances generated: {len(clean_cache)}")
    return clean_cache


def build_isolated_test_set(clean_cache: Dict[str, Any], noise_mgr: NoiseAssetManager):
    print("\n--- Step 2: Assembling Isolated 1.0s Component Test Manifest ---")
    rows = []
    clip_count = 0

    # We evaluate test words: 4 commands + 24 test OOVs across 2 voices = 28 words x 2 voices = 56 clean clips
    test_keys = [k for k, v in clean_cache.items() if v["group"] in ["command", "test_oov_phonetic", "test_oov_general"]]

    # Tiers: clean, 20 dB, 15 dB, 10 dB
    snr_tiers = [None, 20.0, 15.0, 10.0]

    for k in test_keys:
        info = clean_cache[k]
        raw_audio = info["audio"]
        std_1s = make_1s_clip(raw_audio)

        for snr in snr_tiers:
            clip_count += 1
            snr_tag = "clean" if snr is None else f"snr{int(snr):02d}"
            cid = f"edge_test_{k}_{snr_tag}"
            out_wav = ISOLATED_DIR / f"{cid}.wav"

            if snr is None:
                final_audio = std_1s
                noise_file = np.nan
            else:
                track_idx = (clip_count % 6) + 34  # held-out mic-noise tracks 34-39
                noise_slice = noise_mgr.extract_noise_slice(track_index=track_idx, seed=42700 + clip_count, duration_samples=len(std_1s))
                final_audio, obs_snr, _, _ = mix_noise_at_snr(
                    speech=std_1s,
                    noise=noise_slice,
                    target_snr_db=float(snr)
                )
                noise_file = f"mic_noise_track_{track_idx}.mp3"

            sf.write(str(out_wav), final_audio.astype(np.float32), 16000)
            sha = compute_sha256(out_wav)
            rel_p = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

            rows.append({
                "clip_id": cid,
                "label": info["label"],
                "text": info["text"],
                "subgroup": info["group"],
                "voice": info["voice_id"],
                "gender": info["gender"],
                "target_snr_db": "clean" if snr is None else snr,
                "audio_path": rel_p,
                "duration_seconds": 1.000,
                "num_samples": 16000,
                "sha256": sha
            })

    df = pd.DataFrame(rows)
    out_manifest = MANIFESTS_DIR / "edge_tts_eval_isolated_test_manifest.csv"
    df.to_csv(out_manifest, index=False, encoding="utf-8")
    print(f"[OK] Isolated test manifest written: {out_manifest.name} ({len(df)} clips, SHA-256: {compute_sha256(out_manifest)})")
    return df


def build_streaming_sequences(clean_cache: Dict[str, Any], noise_mgr: NoiseAssetManager):
    print("\n--- Step 3: Assembling 100 Multi-Command Streaming Sequences ---")
    rng = np.random.RandomState(42027)

    cmd_keys = [k for k, v in clean_cache.items() if v["group"] == "command"]
    test_oov_keys = [k for k, v in clean_cache.items() if v["group"].startswith("test_oov")]

    seq_records = []
    events_all = []

    # Nominal gap choices in seconds
    GAPS = [0.40, 0.60, 0.80, 1.00, 1.20, 1.50]
    SNRS = [None, 20.0, 15.0, 10.0]  # 25 of each for 100 total

    for seq_idx in range(1, 101):
        snr = SNRS[(seq_idx - 1) // 25]
        snr_str = "clean" if snr is None else f"snr{int(snr):02d}"
        seq_id = f"edge_stream_seq_{seq_idx:03d}_{snr_str}"

        # Choose 5 to 8 speech events per sequence: 3-5 commands, 2-3 OOVs
        n_cmds = rng.randint(3, 6)
        n_oovs = rng.randint(2, 4)

        chosen_cmds = rng.choice(cmd_keys, size=n_cmds, replace=True).tolist()
        chosen_oovs = rng.choice(test_oov_keys, size=n_oovs, replace=True).tolist()

        events_order = chosen_cmds + chosen_oovs
        rng.shuffle(events_order)

        # Assemble audio with silence gaps
        seq_audio_chunks = []
        seq_events_meta = []
        curr_sample = 0

        # Initial silence (0.3s to 0.6s)
        init_silence_len = int(rng.uniform(0.3, 0.6) * 16000)
        seq_audio_chunks.append(np.zeros(init_silence_len, dtype=np.float32))
        curr_sample += init_silence_len

        for ev_key in events_order:
            info = clean_cache[ev_key]
            ev_audio = info["audio"]
            ev_len = len(ev_audio)

            start_t = curr_sample / 16000.0
            end_t = (curr_sample + ev_len) / 16000.0

            seq_events_meta.append({
                "sequence_id": seq_id,
                "label": info["label"],
                "text": info["text"],
                "voice": info["voice_id"],
                "start_time": round(start_t, 4),
                "end_time": round(end_t, 4),
                "duration": round(end_t - start_t, 4)
            })

            seq_audio_chunks.append(ev_audio)
            curr_sample += ev_len

            # Inter-command gap
            gap_dur = rng.choice(GAPS)
            gap_len = int(gap_dur * 16000)
            seq_audio_chunks.append(np.zeros(gap_len, dtype=np.float32))
            curr_sample += gap_len

        clean_seq_audio = np.concatenate(seq_audio_chunks)

        # Mix noise if SNR is not None
        if snr is None:
            final_seq_audio = clean_seq_audio
        else:
            track_idx = (seq_idx % 6) + 34
            noise_slice = noise_mgr.extract_noise_slice(track_index=track_idx, seed=42800 + seq_idx, duration_samples=len(clean_seq_audio))
            final_seq_audio, obs_snr, _, _ = mix_noise_at_snr(
                speech=clean_seq_audio,
                noise=noise_slice,
                target_snr_db=float(snr)
            )

        out_wav = STREAMING_DIR / f"{seq_id}.wav"
        sf.write(str(out_wav), final_seq_audio.astype(np.float32), 16000)
        sha = compute_sha256(out_wav)
        rel_p = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

        seq_records.append({
            "sequence_id": seq_id,
            "snr_tier": snr_str,
            "target_snr_db": "clean" if snr is None else snr,
            "duration_seconds": round(len(final_seq_audio) / 16000.0, 4),
            "num_samples": len(final_seq_audio),
            "num_events": len(events_order),
            "num_commands": n_cmds,
            "num_oovs": n_oovs,
            "expected_command_sequence": [clean_cache[k]["label"] for k in events_order if clean_cache[k]["label"] != "REJECT"],
            "audio_path": rel_p,
            "sha256": sha
        })
        events_all.extend(seq_events_meta)

    df_seq = pd.DataFrame(seq_records)
    out_manifest = MANIFESTS_DIR / "edge_tts_eval_streaming_manifest.csv"
    df_seq.to_csv(out_manifest, index=False, encoding="utf-8")

    out_events = MANIFESTS_DIR / "edge_tts_eval_streaming_events.json"
    with open(out_events, "w", encoding="utf-8") as f:
        json.dump(events_all, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated exactly {len(df_seq)} streaming sequences.")
    print(f"     Manifest: {out_manifest.name} (SHA-256: {compute_sha256(out_manifest)})")
    print(f"     Events JSON: {out_events.name} ({len(events_all)} total reference events)")
    return df_seq, events_all


async def main_async():
    print("=" * 80)
    print("PHASE 27: BUILDING INDEPENDENT SYNTHETIC EVALUATION CORPUS (EDGE-TTS)")
    print("=" * 80)

    t0 = time.perf_counter()
    clean_cache = await generate_all_base_clean()

    noise_mgr = NoiseAssetManager(NOISE_DIR)

    df_isolated = build_isolated_test_set(clean_cache, noise_mgr)
    df_seq, events_all = build_streaming_sequences(clean_cache, noise_mgr)

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"CORPUS GENERATION COMPLETED IN {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main_async())
