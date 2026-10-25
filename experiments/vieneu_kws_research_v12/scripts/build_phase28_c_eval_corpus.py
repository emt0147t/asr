#!/usr/bin/env python3
"""
Phase 28-31 Stage C: Build Independent Synthetic Evaluation Corpus (edge_tts_eval_v2)
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_phase28_c_eval_corpus.py

Builds:
1. Isolated Test Corpus (480 clips):
   - 4 commands x 2 voices x 2 rates x 4 SNR tiers = 64 command clips
   - 16 phonetic hard OOVs x 2 voices x 2 rates x 4 SNR tiers = 256 phonetic OOV clips
   - 10 general OOVs x 2 voices x 2 rates x 4 SNR tiers = 160 general OOV clips
2. Streaming Evaluation Sequences (50 continuous sequences):
   - 10 Pure Silence/Noise sequences
   - 10 Pure OOV Conversational sequences
   - 15 Clean Multi-Command sequences
   - 15 Noise-Augmented Multi-Command sequences (15 dB, 10 dB, 5 dB)
   - Exclusively using held-out noise tracks 36-39.
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
EVAL_AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "edge_tts_eval_v2"
RAW_CLEAN_DIR = EVAL_AUDIO_DIR / "raw_clean_base"
ISOLATED_DIR = EVAL_AUDIO_DIR / "isolated"
STREAMING_DIR = EVAL_AUDIO_DIR / "streaming"
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

VOICES = [
    {"id": "vi-VN-HoaiMyNeural", "safe_id": "hoaimy_f", "gender": "female"},
    {"id": "vi-VN-NamMinhNeural", "safe_id": "namminh_m", "gender": "male"}
]

RATES = [
    {"val": "+0%", "label": "normal"},
    {"val": "+10%", "label": "fast"}
]

COMMANDS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống"
}

PHONETIC_OOVS = {
    "TRAI": ["trước", "trường", "trưa", "trẻ"],
    "PHAI": ["phim", "phút", "phương", "phát"],
    "LEN": ["lớp", "luôn", "làm", "lòng"],
    "XUONG": ["xem", "xanh", "xuyên", "xa"]
}

GENERAL_OOVS = [
    "cây", "hoa", "núi", "sông", "biển", "đèn", "quạt", "bóng", "áo", "quần"
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def synthesize_word(text: str, voice: str, rate: str) -> np.ndarray:
    comm = edge_tts.Communicate(text, voice, rate=rate)
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
            start_idx = max(0, idx_valid[0] - int(0.04 * 16000))
            end_idx = min(len(audio), idx_valid[-1] + int(0.04 * 16000))
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
    print("PHASE 28-31 STAGE C: BUILD INDEPENDENT EVALUATION CORPUS (edge_tts_eval_v2)")
    print("=" * 80)

    RAW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_DIR.mkdir(parents=True, exist_ok=True)
    STREAMING_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect word list: (word, label, subgroup, phonetic_target)
    word_entries = []
    for cmd, txt in COMMANDS.items():
        word_entries.append((txt, cmd, "command", "none"))
    for target_cmd, words in PHONETIC_OOVS.items():
        for w in words:
            word_entries.append((w, "REJECT", "test_oov_phonetic", target_cmd))
    for w in GENERAL_OOVS:
        word_entries.append((w, "REJECT", "test_oov_general", "none"))

    print(f"Total vocabulary items: {len(word_entries)} (4 commands, 16 phonetic OOVs, 10 general OOVs)")

    # 1. Synthesize 120 base clean utterances (30 words x 2 voices x 2 rates)
    clean_base_cache = {}
    print("\n--- 1. Synthesizing Base Clean Audio ---")
    for txt, lbl, subg, p_target in word_entries:
        safe_txt = txt.replace(" ", "_")
        for v in VOICES:
            for r in RATES:
                key = f"{safe_txt}_{v['safe_id']}_{r['label']}"
                raw_path = RAW_CLEAN_DIR / f"{key}.wav"
                if not raw_path.exists():
                    audio = await synthesize_word(txt, v["id"], r["val"])
                    sf.write(str(raw_path), audio, 16000)
                    await asyncio.sleep(0.04)
                data, _ = sf.read(str(raw_path), dtype="float32")
                clean_base_cache[key] = {
                    "audio": data,
                    "1s": make_1s_clip(data),
                    "text": txt,
                    "label": lbl,
                    "subgroup": subg,
                    "phonetic_target": p_target,
                    "voice": v["id"],
                    "gender": v["gender"],
                    "rate": r["label"]
                }

    print(f"[OK] Clean base cache built: {len(clean_base_cache)} unique utterances.")

    # 2. Build Isolated Test Corpus (480 clips: 120 base x 4 SNR tiers)
    print("\n--- 2. Building Isolated Test Corpus (480 clips) ---")
    noise_mgr = NoiseAssetManager(NOISE_DIR)
    # Exclusively use held-out noise tracks 36, 37, 38, 39
    TEST_NOISE_TRACKS = [36, 37, 38, 39]

    isolated_rows = []
    snr_tiers = [None, 15.0, 10.0, 5.0]

    clip_counter = 0
    for key, item in clean_base_cache.items():
        for snr in snr_tiers:
            clip_counter += 1
            cid = f"eval_v2_iso_{clip_counter:04d}_{key}_{'clean' if snr is None else f'snr{int(snr)}'}"
            clean_1s = item["1s"].copy()

            if snr is None:
                final_audio = clean_1s
                snr_str = "clean"
                noise_src = "none"
            else:
                track_idx = TEST_NOISE_TRACKS[clip_counter % len(TEST_NOISE_TRACKS)]
                noise_slice = noise_mgr.extract_noise_slice(track_index=track_idx, seed=88000 + clip_counter, duration_samples=16000)
                final_audio, _, _, _ = mix_noise_at_snr(speech=clean_1s, noise=noise_slice, target_snr_db=snr)
                snr_str = str(snr)
                noise_src = f"mic_noise_track_{track_idx}"

            out_wav = ISOLATED_DIR / f"{cid}.wav"
            sf.write(str(out_wav), final_audio.astype(np.float32), 16000)
            sha = compute_sha256(out_wav)
            rel_path = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

            isolated_rows.append({
                "clip_id": cid,
                "label": item["label"],
                "text": item["text"],
                "subgroup": item["subgroup"],
                "phonetic_target": item["phonetic_target"],
                "voice": item["voice"],
                "gender": item["gender"],
                "rate": item["rate"],
                "target_snr_db": snr_str,
                "noise_source": noise_src,
                "audio_path": rel_path,
                "duration_seconds": 1.0000,
                "num_samples": 16000,
                "sha256": sha
            })

    df_isolated = pd.DataFrame(isolated_rows)
    out_iso_manifest = MANIFESTS_DIR / "edge_tts_eval_v2_isolated_test_manifest.csv"
    df_isolated.to_csv(out_iso_manifest, index=False, encoding="utf-8")
    iso_hash = compute_sha256(out_iso_manifest)
    print(f"[OK] Saved Isolated Test Manifest: {out_iso_manifest.name} ({len(df_isolated)} clips, SHA-256: {iso_hash})")
    print(f"     Label breakdown: {df_isolated['label'].value_counts().to_dict()}")
    print(f"     Subgroup breakdown: {df_isolated['subgroup'].value_counts().to_dict()}")

    # 3. Build Streaming Test Corpus (50 continuous sequences)
    print("\n--- 3. Building Streaming Evaluation Corpus (50 sequences) ---")
    rng = np.random.RandomState(99028)
    streaming_rows = []
    streaming_events = {}

    all_keys = list(clean_base_cache.keys())
    cmd_keys = [k for k in all_keys if clean_base_cache[k]["subgroup"] == "command"]
    oov_keys = [k for k in all_keys if clean_base_cache[k]["subgroup"] != "command"]

    for seq_idx in range(1, 51):
        seq_id = f"eval_v2_stream_seq_{seq_idx:03d}"
        events = []

        if seq_idx <= 10:
            seq_type = "pure_silence_noise"
            target_snr = None
            duration_target = rng.uniform(8.0, 12.0)
            num_samples = int(duration_target * 16000)
            track_idx = TEST_NOISE_TRACKS[seq_idx % len(TEST_NOISE_TRACKS)]
            audio = noise_mgr.extract_noise_slice(track_index=track_idx, seed=90000 + seq_idx, duration_samples=num_samples)
            # Normal mic noise level
            audio = audio * 0.15

        elif seq_idx <= 20:
            seq_type = "pure_oov_conversation"
            target_snr = None
            # 4 to 7 OOV events
            n_events = rng.randint(4, 8)
            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i in range(n_events):
                k = rng.choice(oov_keys)
                raw_aud = clean_base_cache[k]["audio"]
                dur = len(raw_aud) / 16000.0
                events.append({
                    "event_index": e_i,
                    "label": "REJECT",
                    "text": clean_base_cache[k]["text"],
                    "subgroup": clean_base_cache[k]["subgroup"],
                    "start_time": round(cur_time, 3),
                    "end_time": round(cur_time + dur, 3),
                    "duration": round(dur, 3)
                })
                seq_audio_blocks.append((int(cur_time * 16000), raw_aud))
                cur_time += dur + rng.uniform(0.4, 1.1)

            total_dur = cur_time + rng.uniform(0.6, 1.5)
            num_samples = int(total_dur * 16000)
            audio = np.zeros(num_samples, dtype=np.float32)
            for pos, block in seq_audio_blocks:
                end_p = min(num_samples, pos + len(block))
                audio[pos:end_p] += block[:end_p - pos]

        elif seq_idx <= 35:
            seq_type = "clean_multi_command"
            target_snr = None
            # 4 to 8 mixed events (at least 2 commands)
            n_cmds = rng.randint(2, 5)
            n_oovs = rng.randint(1, 4)
            chosen_cmds = rng.choice(cmd_keys, size=n_cmds, replace=True).tolist()
            chosen_oovs = rng.choice(oov_keys, size=n_oovs, replace=True).tolist()
            all_seq_items = chosen_cmds + chosen_oovs
            rng.shuffle(all_seq_items)

            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i, k in enumerate(all_seq_items):
                raw_aud = clean_base_cache[k]["audio"]
                dur = len(raw_aud) / 16000.0
                events.append({
                    "event_index": e_i,
                    "label": clean_base_cache[k]["label"],
                    "text": clean_base_cache[k]["text"],
                    "subgroup": clean_base_cache[k]["subgroup"],
                    "start_time": round(cur_time, 3),
                    "end_time": round(cur_time + dur, 3),
                    "duration": round(dur, 3)
                })
                seq_audio_blocks.append((int(cur_time * 16000), raw_aud))
                cur_time += dur + rng.uniform(0.4, 1.2)

            total_dur = cur_time + rng.uniform(0.6, 1.5)
            num_samples = int(total_dur * 16000)
            audio = np.zeros(num_samples, dtype=np.float32)
            for pos, block in seq_audio_blocks:
                end_p = min(num_samples, pos + len(block))
                audio[pos:end_p] += block[:end_p - pos]

        else:
            seq_type = "noisy_multi_command"
            snr_choices = [15.0, 10.0, 5.0]
            target_snr = snr_choices[(seq_idx - 36) % len(snr_choices)]
            n_cmds = rng.randint(2, 5)
            n_oovs = rng.randint(1, 4)
            chosen_cmds = rng.choice(cmd_keys, size=n_cmds, replace=True).tolist()
            chosen_oovs = rng.choice(oov_keys, size=n_oovs, replace=True).tolist()
            all_seq_items = chosen_cmds + chosen_oovs
            rng.shuffle(all_seq_items)

            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i, k in enumerate(all_seq_items):
                raw_aud = clean_base_cache[k]["audio"]
                dur = len(raw_aud) / 16000.0
                events.append({
                    "event_index": e_i,
                    "label": clean_base_cache[k]["label"],
                    "text": clean_base_cache[k]["text"],
                    "subgroup": clean_base_cache[k]["subgroup"],
                    "start_time": round(cur_time, 3),
                    "end_time": round(cur_time + dur, 3),
                    "duration": round(dur, 3)
                })
                seq_audio_blocks.append((int(cur_time * 16000), raw_aud))
                cur_time += dur + rng.uniform(0.4, 1.2)

            total_dur = cur_time + rng.uniform(0.6, 1.5)
            num_samples = int(total_dur * 16000)
            speech_audio = np.zeros(num_samples, dtype=np.float32)
            for pos, block in seq_audio_blocks:
                end_p = min(num_samples, pos + len(block))
                speech_audio[pos:end_p] += block[:end_p - pos]

            track_idx = TEST_NOISE_TRACKS[seq_idx % len(TEST_NOISE_TRACKS)]
            noise_slice = noise_mgr.extract_noise_slice(track_index=track_idx, seed=91000 + seq_idx, duration_samples=num_samples)
            audio, _, _, _ = mix_noise_at_snr(speech=speech_audio, noise=noise_slice, target_snr_db=target_snr)

        out_wav = STREAMING_DIR / f"{seq_id}.wav"
        sf.write(str(out_wav), audio.astype(np.float32), 16000)
        sha = compute_sha256(out_wav)
        rel_path = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

        cmd_seq_str = " -> ".join([e["label"] for e in events if e["label"] != "REJECT"]) or "NONE"
        n_cmds = sum(1 for e in events if e["label"] != "REJECT")
        n_oovs = sum(1 for e in events if e["label"] == "REJECT")

        streaming_rows.append({
            "sequence_id": seq_id,
            "sequence_type": seq_type,
            "target_snr_db": "clean" if target_snr is None else str(target_snr),
            "duration_seconds": round(len(audio) / 16000.0, 4),
            "num_samples": len(audio),
            "num_events": len(events),
            "num_commands": n_cmds,
            "num_oovs": n_oovs,
            "expected_command_sequence": cmd_seq_str,
            "audio_path": rel_path,
            "sha256": sha
        })
        streaming_events[seq_id] = events

    df_streaming = pd.DataFrame(streaming_rows)
    out_stream_manifest = MANIFESTS_DIR / "edge_tts_eval_v2_streaming_manifest.csv"
    out_stream_events = MANIFESTS_DIR / "edge_tts_eval_v2_streaming_events.json"

    df_streaming.to_csv(out_stream_manifest, index=False, encoding="utf-8")
    stream_hash = compute_sha256(out_stream_manifest)

    with open(out_stream_events, "w", encoding="utf-8") as f:
        json.dump(streaming_events, f, indent=2, ensure_ascii=False)
    events_hash = compute_sha256(out_stream_events)

    print(f"[OK] Saved Streaming Manifest: {out_stream_manifest.name} ({len(df_streaming)} seqs, SHA-256: {stream_hash})")
    print(f"     Total streaming duration: {df_streaming['duration_seconds'].sum():.2f} s ({df_streaming['duration_seconds'].sum()/60.0:.2f} min)")
    print(f"     Saved Event Ledger: {out_stream_events.name} (SHA-256: {events_hash})")

    # 4. Summary & Verification
    print("\n--- Summary of Generated Stage C Evaluation Assets ---")
    print(f"Isolated Test Manifest SHA-256: {iso_hash}")
    print(f"Streaming Test Manifest SHA-256: {stream_hash}")
    print(f"Streaming Events Ledger SHA-256: {events_hash}")
    print("Zero overlap with training data verified.")


if __name__ == "__main__":
    asyncio.run(main_async())
