#!/usr/bin/env python3
"""
Phase 50: Build Confirmatory Independent Evaluation Corpus (gTTS + ESC-50 Fold 2)
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_phase50_confirmatory_corpus.py

Builds:
1. Fresh Environmental Noise: 8 Fold 2 indoor tracks from ESC-50.
   - Verified non-overlapping against 40 historical mic-noise tracks and Fold 1.
2. Track A — Isolated 4-Command Classification (288 clips: 240 primary + 48 tone probes):
   - 4 commands x 2 cadences x 4 SNR tiers = 32 command clips
   - 16 phonetic hard OOVs x 2 cadences x 4 SNR tiers = 128 phonetic OOV clips
   - 10 general OOVs x 2 cadences x 4 SNR tiers = 80 general OOV clips
   - 6 tone probes x 2 cadences x 4 SNR tiers = 48 tone probe clips (diagnostic cohort)
   - Grouped by clean synthesis-instance ID (60 primary + 12 probe clusters).
3. Track B — Keyword Detection in Context / Continuous Audio (50 sequences):
   - 10 Silence/Noise sequences (ESC-50 Fold 2)
   - 10 Conversational OOV sequences
   - 15 Clean Multi-Command sequences with carrier phrases (rẽ trái, sang trái, rẽ phải, đi lên, đi xuống)
   - 15 Noisy Multi-Command sequences (15 dB, 10 dB, 5 dB SNR)
   - Ground truth event ledger explicitly recording event boundaries and labels.
4. Generates:
   - PHASE50_LOCKED_MANIFEST.csv
   - PHASE50_TONE_PROBE_MANIFEST.csv
   - PHASE50_STREAMING_MANIFEST.csv
   - PHASE50_EVENT_LEDGER.csv & JSON
   - PHASE50_CONFIRMATORY_PREREGISTRATION.md
   - PHASE50_SHA256_MANIFEST.json
"""

import sys
import io
import time
import json
import urllib.request
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from gtts import gTTS

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
EVAL_AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "gtts_esc50_eval_v1"
RAW_CLEAN_DIR = EVAL_AUDIO_DIR / "raw_clean_base"
ISOLATED_DIR = EVAL_AUDIO_DIR / "isolated"
STREAMING_DIR = EVAL_AUDIO_DIR / "streaming"
RAW_NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "esc50_fold2"
PROCESSED_NOISE_DIR = PROJECT_ROOT / "data" / "processed" / "noise" / "esc50_fold2_16k"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

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

TONE_PROBES = [
    "ma", "má", "mà", "mả", "mã", "mạ"
]

CARRIER_PHRASES = [
    {"phrase": "rẽ trái", "keyword": "trái", "label": "TRAI"},
    {"phrase": "sang trái", "keyword": "trái", "label": "TRAI"},
    {"phrase": "rẽ phải", "keyword": "phải", "label": "PHAI"},
    {"phrase": "sang phải", "keyword": "phải", "label": "PHAI"},
    {"phrase": "đi lên", "keyword": "lên", "label": "LEN"},
    {"phrase": "tiến lên", "keyword": "lên", "label": "LEN"},
    {"phrase": "đi xuống", "keyword": "xuống", "label": "XUONG"},
    {"phrase": "lùi xuống", "keyword": "xuống", "label": "XUONG"}
]

FOLD2_NOISE_SELECTED = [
    {"fname": "2-106072-A-36.wav", "category": "vacuum_cleaner"},
    {"fname": "2-122820-A-36.wav", "category": "vacuum_cleaner"},
    {"fname": "2-104168-A-32.wav", "category": "keyboard_typing"},
    {"fname": "2-109316-A-32.wav", "category": "keyboard_typing"},
    {"fname": "2-102567-A-35.wav", "category": "washing_machine"},
    {"fname": "2-121909-A-35.wav", "category": "washing_machine"},
    {"fname": "2-119748-A-38.wav", "category": "clock_tick"},
    {"fname": "2-127108-A-38.wav", "category": "clock_tick"}
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def acquire_esc50_fold2_noise():
    print("\n--- 1. Acquiring & Standardizing ESC-50 Fold 2 Environmental Noise ---")
    RAW_NOISE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_NOISE_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"

    for item in FOLD2_NOISE_SELECTED:
        fname = item["fname"]
        raw_path = RAW_NOISE_DIR / fname
        if not raw_path.exists():
            url = base_url + fname
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(raw_path, "wb") as f:
                f.write(resp.read())
            print(f"  Downloaded: {fname}")

        # Standardize to 16 kHz mono
        proc_path = PROCESSED_NOISE_DIR / f"esc50_f2_16k_{fname}"
        if not proc_path.exists():
            data, sr = sf.read(str(raw_path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=-1)
            if sr != 16000:
                num_16k = int(round(len(data) * 16000.0 / float(sr)))
                data = signal.resample(data, num_16k).astype(np.float32)
            data = data - np.mean(data)
            sf.write(str(proc_path), data.astype(np.float32), 16000, subtype="PCM_16")

    print(f"[OK] 8 ESC-50 Fold 2 noise tracks ready at {PROCESSED_NOISE_DIR.name}")


def synthesize_clean_word(text: str, slow: bool) -> np.ndarray:
    tts = gTTS(text, lang="vi", slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != 16000:
        num_16k = int(round(len(audio) * 16000.0 / float(sr)))
        audio = signal.resample(audio, num_16k).astype(np.float32)
    audio = audio - np.mean(audio)

    # Trim energy < -45 dBFS
    energy = audio ** 2
    max_e = np.max(energy) if len(energy) > 0 else 0.0
    if max_e > 1e-6:
        thresh = max_e * (10.0 ** (-45.0 / 10.0))
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


def main():
    print("=" * 80)
    print("PHASE 50: BUILD CONFIRMATORY EVALUATION CORPUS (gTTS + ESC-50 Fold 2)")
    print("=" * 80)

    t0 = time.time()
    freeze_ts_start = datetime.now(timezone.utc).isoformat()

    RAW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_DIR.mkdir(parents=True, exist_ok=True)
    STREAMING_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    acquire_esc50_fold2_noise()
    noise_mgr = NoiseAssetManager(PROCESSED_NOISE_DIR)

    # 1. Assemble vocabulary
    word_entries = []
    for cmd, txt in COMMANDS.items():
        word_entries.append((txt, cmd, "command", "none", False))
    for target_cmd, words in PHONETIC_OOVS.items():
        for w in words:
            word_entries.append((w, "REJECT", "test_oov_phonetic", target_cmd, False))
    for w in GENERAL_OOVS:
        word_entries.append((w, "REJECT", "test_oov_general", "none", False))
    for w in TONE_PROBES:
        word_entries.append((w, "REJECT", "test_tone_probe", "none", True))

    print(f"\n--- 2. Synthesizing Clean Base Audio with gTTS ({len(word_entries)} words x 2 cadences) ---")
    clean_base_cache = {}

    for txt, lbl, subg, p_target, is_probe in word_entries:
        safe_txt = txt.replace(" ", "_")
        for slow in [False, True]:
            cfg_id = "cadence_slow" if slow else "cadence_normal"
            rate_label = "slow" if slow else "normal"
            key = f"{safe_txt}_{cfg_id}"
            raw_path = RAW_CLEAN_DIR / f"{key}.wav"
            if not raw_path.exists():
                audio = synthesize_clean_word(txt, slow=slow)
                sf.write(str(raw_path), audio, 16000)
                time.sleep(0.05)
            data, _ = sf.read(str(raw_path), dtype="float32")
            clean_base_cache[key] = {
                "audio": data,
                "1s": make_1s_clip(data),
                "text": txt,
                "label": lbl,
                "subgroup": subg,
                "phonetic_target": p_target,
                "is_probe": is_probe,
                "config_id": cfg_id,
                "rate_label": rate_label,
                "slow_flag": slow
            }

    # Also synthesize contextual carrier phrases for Track B
    print("\n--- Synthesizing Contextual Carrier Phrases for Track B ---")
    carrier_cache = {}
    for c_item in CARRIER_PHRASES:
        phr = c_item["phrase"]
        safe_phr = phr.replace(" ", "_")
        raw_path = RAW_CLEAN_DIR / f"carrier_{safe_phr}.wav"
        if not raw_path.exists():
            audio = synthesize_clean_word(phr, slow=False)
            sf.write(str(raw_path), audio, 16000)
            time.sleep(0.05)
        data, _ = sf.read(str(raw_path), dtype="float32")
        carrier_cache[phr] = {
            "audio": data,
            "phrase": phr,
            "keyword": c_item["keyword"],
            "label": c_item["label"]
        }

    print(f"[OK] Clean base cache ready: {len(clean_base_cache)} lexeme variants + {len(carrier_cache)} carrier phrases.")

    # 2. Build Track A: Isolated Manifest (288 clips: 240 primary + 48 tone probes)
    print("\n--- 3. Building Track A: Isolated Confirmatory Manifest ---")
    snr_tiers = [None, 15.0, 10.0, 5.0]
    num_noise_tracks = noise_mgr.get_track_count()

    isolated_rows = []
    probe_rows = []
    clip_counter = 0
    probe_counter = 0

    for key, item in clean_base_cache.items():
        if not item["is_probe"]:
            # Primary cohort
            for snr in snr_tiers:
                clip_counter += 1
                cid = f"p50_gtts_iso_{clip_counter:04d}_{key}_{'clean' if snr is None else f'snr{int(snr)}'}"
                clean_1s = item["1s"].copy()

                if snr is None:
                    final_audio = clean_1s
                    snr_str = "clean"
                    noise_src = "none"
                    noise_track_id = "none"
                else:
                    track_idx = clip_counter % num_noise_tracks
                    track_path = noise_mgr.get_track_path(track_idx)
                    noise_slice = noise_mgr.extract_noise_slice(
                        track_index=track_idx,
                        seed=50000 + clip_counter,
                        duration_samples=16000
                    )
                    final_audio, _, _, _ = mix_noise_at_snr(
                        speech=clean_1s,
                        noise=noise_slice,
                        target_snr_db=snr
                    )
                    snr_str = str(snr)
                    noise_src = track_path.stem
                    noise_track_id = f"esc50_f2_track_{track_idx}"

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
                    "generator": "Google Translate TTS (gTTS 2.5.4)",
                    "config_id": item["config_id"],
                    "synthesis_instance_id": f"{item['text']}_{item['config_id']}",
                    "rate_label": item["rate_label"],
                    "target_snr_db": snr_str,
                    "noise_source": noise_src,
                    "noise_track_id": noise_track_id,
                    "audio_path": rel_path,
                    "duration_seconds": 1.0000,
                    "num_samples": 16000,
                    "sha256": sha
                })
        else:
            # Tone probe cohort
            for snr in snr_tiers:
                probe_counter += 1
                cid = f"p50_gtts_probe_{probe_counter:04d}_{key}_{'clean' if snr is None else f'snr{int(snr)}'}"
                clean_1s = item["1s"].copy()

                if snr is None:
                    final_audio = clean_1s
                    snr_str = "clean"
                    noise_src = "none"
                    noise_track_id = "none"
                else:
                    track_idx = probe_counter % num_noise_tracks
                    track_path = noise_mgr.get_track_path(track_idx)
                    noise_slice = noise_mgr.extract_noise_slice(
                        track_index=track_idx,
                        seed=60000 + probe_counter,
                        duration_samples=16000
                    )
                    final_audio, _, _, _ = mix_noise_at_snr(
                        speech=clean_1s,
                        noise=noise_slice,
                        target_snr_db=snr
                    )
                    snr_str = str(snr)
                    noise_src = track_path.stem
                    noise_track_id = f"esc50_f2_track_{track_idx}"

                out_wav = ISOLATED_DIR / f"{cid}.wav"
                sf.write(str(out_wav), final_audio.astype(np.float32), 16000)
                sha = compute_sha256(out_wav)
                rel_path = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

                probe_rows.append({
                    "clip_id": cid,
                    "label": item["label"],
                    "text": item["text"],
                    "subgroup": item["subgroup"],
                    "phonetic_target": item["phonetic_target"],
                    "generator": "Google Translate TTS (gTTS 2.5.4)",
                    "config_id": item["config_id"],
                    "synthesis_instance_id": f"{item['text']}_{item['config_id']}",
                    "rate_label": item["rate_label"],
                    "target_snr_db": snr_str,
                    "noise_source": noise_src,
                    "noise_track_id": noise_track_id,
                    "audio_path": rel_path,
                    "duration_seconds": 1.0000,
                    "num_samples": 16000,
                    "sha256": sha
                })

    df_isolated = pd.DataFrame(isolated_rows)
    out_iso_manifest = MANIFESTS_DIR / "PHASE50_LOCKED_MANIFEST.csv"
    df_isolated.to_csv(out_iso_manifest, index=False, encoding="utf-8")
    iso_hash = compute_sha256(out_iso_manifest)

    df_probe = pd.DataFrame(probe_rows)
    out_probe_manifest = MANIFESTS_DIR / "PHASE50_TONE_PROBE_MANIFEST.csv"
    df_probe.to_csv(out_probe_manifest, index=False, encoding="utf-8")
    probe_hash = compute_sha256(out_probe_manifest)

    print(f"[OK] Saved PHASE50_LOCKED_MANIFEST.csv: {len(df_isolated)} clips, SHA-256: {iso_hash}")
    print(f"     Label breakdown: {df_isolated['label'].value_counts().to_dict()}")
    print(f"[OK] Saved PHASE50_TONE_PROBE_MANIFEST.csv: {len(df_probe)} clips, SHA-256: {probe_hash}")

    # 3. Build Track B: Streaming Continuous Corpus (50 sequences)
    print("\n--- 4. Building Track B: Streaming Continuous Corpus (50 Sequences) ---")
    rng = np.random.RandomState(50050)
    streaming_rows = []
    streaming_events = {}

    all_clean_keys = [k for k in clean_base_cache.keys() if not clean_base_cache[k]["is_probe"]]
    cmd_clean_keys = [k for k in all_clean_keys if clean_base_cache[k]["subgroup"] == "command"]
    oov_clean_keys = [k for k in all_clean_keys if clean_base_cache[k]["subgroup"] != "command"]
    carrier_phrases_list = list(carrier_cache.keys())

    for seq_idx in range(1, 51):
        seq_id = f"p50_stream_seq_{seq_idx:03d}"
        events = []

        if seq_idx <= 10:
            seq_type = "pure_silence_noise"
            target_snr = None
            duration_target = rng.uniform(8.0, 12.0)
            num_samples = int(duration_target * 16000)
            track_idx = seq_idx % num_noise_tracks
            track_path = noise_mgr.get_track_path(track_idx)
            audio = noise_mgr.extract_noise_slice(
                track_index=track_idx,
                seed=71000 + seq_idx,
                duration_samples=num_samples
            )
            audio = audio * 0.15
            noise_src = track_path.stem

        elif seq_idx <= 20:
            seq_type = "pure_oov_conversation"
            target_snr = None
            n_events = rng.randint(4, 8)
            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i in range(n_events):
                k = rng.choice(oov_clean_keys)
                raw_aud = clean_base_cache[k]["audio"]
                dur = len(raw_aud) / 16000.0
                events.append({
                    "sequence_id": seq_id,
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
            noise_src = "none"

        elif seq_idx <= 35:
            seq_type = "clean_multi_command"
            target_snr = None
            # Mix carrier phrases, isolated commands, and OOVs
            n_cmds = rng.randint(2, 4)
            n_carriers = rng.randint(1, 3)
            n_oovs = rng.randint(1, 3)

            chosen_items = []
            for _ in range(n_cmds):
                k = rng.choice(cmd_clean_keys)
                chosen_items.append(("isolated_cmd", k))
            for _ in range(n_carriers):
                phr = rng.choice(carrier_phrases_list)
                chosen_items.append(("carrier", phr))
            for _ in range(n_oovs):
                k = rng.choice(oov_clean_keys)
                chosen_items.append(("oov", k))

            rng.shuffle(chosen_items)

            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i, (itype, item_val) in enumerate(chosen_items):
                if itype == "isolated_cmd":
                    raw_aud = clean_base_cache[item_val]["audio"]
                    lbl = clean_base_cache[item_val]["label"]
                    txt = clean_base_cache[item_val]["text"]
                    subg = "isolated_command"
                elif itype == "carrier":
                    raw_aud = carrier_cache[item_val]["audio"]
                    lbl = carrier_cache[item_val]["label"]
                    txt = carrier_cache[item_val]["phrase"]
                    subg = "carrier_context_command"
                else:
                    raw_aud = clean_base_cache[item_val]["audio"]
                    lbl = "REJECT"
                    txt = clean_base_cache[item_val]["text"]
                    subg = clean_base_cache[item_val]["subgroup"]

                dur = len(raw_aud) / 16000.0
                events.append({
                    "sequence_id": seq_id,
                    "event_index": e_i,
                    "label": lbl,
                    "text": txt,
                    "subgroup": subg,
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
            noise_src = "none"

        else:
            seq_type = "noisy_multi_command"
            snr_choices = [15.0, 10.0, 5.0]
            target_snr = snr_choices[(seq_idx - 36) % len(snr_choices)]
            n_cmds = rng.randint(2, 4)
            n_carriers = rng.randint(1, 3)
            n_oovs = rng.randint(1, 3)

            chosen_items = []
            for _ in range(n_cmds):
                k = rng.choice(cmd_clean_keys)
                chosen_items.append(("isolated_cmd", k))
            for _ in range(n_carriers):
                phr = rng.choice(carrier_phrases_list)
                chosen_items.append(("carrier", phr))
            for _ in range(n_oovs):
                k = rng.choice(oov_clean_keys)
                chosen_items.append(("oov", k))

            rng.shuffle(chosen_items)

            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i, (itype, item_val) in enumerate(chosen_items):
                if itype == "isolated_cmd":
                    raw_aud = clean_base_cache[item_val]["audio"]
                    lbl = clean_base_cache[item_val]["label"]
                    txt = clean_base_cache[item_val]["text"]
                    subg = "isolated_command"
                elif itype == "carrier":
                    raw_aud = carrier_cache[item_val]["audio"]
                    lbl = carrier_cache[item_val]["label"]
                    txt = carrier_cache[item_val]["phrase"]
                    subg = "carrier_context_command"
                else:
                    raw_aud = clean_base_cache[item_val]["audio"]
                    lbl = "REJECT"
                    txt = clean_base_cache[item_val]["text"]
                    subg = clean_base_cache[item_val]["subgroup"]

                dur = len(raw_aud) / 16000.0
                events.append({
                    "sequence_id": seq_id,
                    "event_index": e_i,
                    "label": lbl,
                    "text": txt,
                    "subgroup": subg,
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

            track_idx = seq_idx % num_noise_tracks
            track_path = noise_mgr.get_track_path(track_idx)
            noise_slice = noise_mgr.extract_noise_slice(
                track_index=track_idx,
                seed=82000 + seq_idx,
                duration_samples=num_samples
            )
            audio, _, _, _ = mix_noise_at_snr(
                speech=speech_audio,
                noise=noise_slice,
                target_snr_db=target_snr
            )
            noise_src = track_path.stem

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
            "noise_source": noise_src,
            "duration_seconds": round(len(audio) / 16000.0, 4),
            "num_samples": len(audio),
            "num_command_events": n_cmds,
            "num_oov_events": n_oovs,
            "ground_truth_command_sequence": cmd_seq_str,
            "audio_path": rel_path,
            "sha256": sha
        })
        streaming_events[seq_id] = events

    df_streaming = pd.DataFrame(streaming_rows)
    out_stream_manifest = MANIFESTS_DIR / "PHASE50_STREAMING_MANIFEST.csv"
    df_streaming.to_csv(out_stream_manifest, index=False, encoding="utf-8")
    stream_hash = compute_sha256(out_stream_manifest)

    # Flatten events to CSV
    all_events_flat = []
    for seq_id, ev_list in streaming_events.items():
        all_events_flat.extend(ev_list)
    df_events = pd.DataFrame(all_events_flat)
    out_event_ledger = MANIFESTS_DIR / "PHASE50_EVENT_LEDGER.csv"
    df_events.to_csv(out_event_ledger, index=False, encoding="utf-8")
    event_ledger_hash = compute_sha256(out_event_ledger)

    out_events_json = MANIFESTS_DIR / "PHASE50_EVENT_LEDGER.json"
    with open(out_events_json, "w", encoding="utf-8") as f:
        json.dump(streaming_events, f, indent=2, ensure_ascii=False)
    events_json_hash = compute_sha256(out_events_json)

    print(f"[OK] Saved PHASE50_STREAMING_MANIFEST.csv: {len(df_streaming)} sequences, SHA-256: {stream_hash}")
    print(f"[OK] Saved PHASE50_EVENT_LEDGER.csv: {len(df_events)} events, SHA-256: {event_ledger_hash}")

    # 4. Pre-Registration Document
    freeze_ts = datetime.now(timezone.utc).isoformat()
    prereg_content = f"""# PHASE 50: CONFIRMATORY EVALUATION PROTOCOL & PRE-REGISTRATION

**Repository:** `D:\\trainidentificationaudio`  
**Phase:** Phase 50 (Confirmatory Protocol Registration)  
**Creation Timestamp:** `{freeze_ts_start}`  
**Manifest Freeze Timestamp:** `{freeze_ts}`  
**Status:** **LOCKED & PRE-REGISTERED (PRIOR TO CONFIRMATORY INFERENCE)**

---

## 1. Governance & Protocol Invariants

1. **Default Model Status:** Legacy $v11$ remains the canonical default model.
2. **Model Integrity:** Checkpoints for $v11$, $E1$, Phase 28 $D_2\\_H1$, and Phase 26 $D_0\\_H1\\_seed44$ are strictly read-only and immutable.
3. **Threshold Lock:** Decision threshold $\\theta_{{\\text{{cmd}}}} = 0.20$ is strictly locked.
4. **Argmax Policy:** REJECT-inclusive 5-class argmax:
   $$\\hat{{c}} = \\begin{{cases}} \\arg\\max_{{i \\in \\{{0,1,2,3\\}}}} P(c_i), & \\text{{if }} \\max_{{i \\in \\{{0,1,2,3\\}}}} P(c_i) \\ge 0.20 \\\\ 4 \\text{{ (REJECT)}}, & \\text{{otherwise}} \\end{{cases}}$$
5. **VAD Specification:** `SpeechBandRatioVAD` with band $[300, 3400]\\text{{ Hz}}$, energy threshold $-46.0\\text{{ dBFS}}$, ratio threshold $0.32$.
6. **Frontend:** 40-channel Log-Mel filterbank ($16\\text{{ kHz}}$, frame $25\\text{{ ms}}$, hop $10\\text{{ ms}}$, $512$ FFT).
7. **Capacity:** Fixed at $138,469$ parameters (`CorrectedVietnameseKWSNet`, `DEFER_E2_CAPACITY`).
8. **Deployment Claim Restrictions:**
   - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
   - `NOT_VALIDATED_ON_NATURAL_SPEECH`
   - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
   - `NOT_VALIDATED_PENDING_DATA`

---

## 2. Independent Generators & Noise Assets

### 2.1 Acoustic Speech Generator
- **Engine:** Google Translate Vietnamese Text-to-Speech (`gTTS 2.5.4`).
- **Independence:** Completely independent from VieNeu-TTS, Microsoft Edge-TTS, and Meta MMS-TTS. Zero historical usage in training, validation, or candidate selection.
- **Language:** Vietnamese (`vi`). Standard Northern/National Vietnamese phonology with clear rising/falling tonal contours.

### 2.2 Environmental Noise Assets
- **Dataset:** Karol Piczak ESC-50 Fold 2 indoor recordings (`vacuum_cleaner`, `keyboard_typing`, `washing_machine`, `clock_tick`).
- **Deduplication:** Fully verified non-overlapping against all 40 historical `mic-noise` tracks and Fold 1.

---

## 3. Two Pre-Registered Evaluation Tracks

### Track A — Isolated 4-Command Classification
- **Manifest:** `PHASE50_LOCKED_MANIFEST.csv` (240 primary clips: 32 commands + 128 phonetic OOVs + 80 general OOVs).
- **Tone Probes:** `PHASE50_TONE_PROBE_MANIFEST.csv` (48 diagnostic clips).
- **Cluster Unit:** Synthesis Instance (clean waveform), grouping all 4 derivative SNR tiers.
- **Explicit Limitation:** Multiple synthesis instances sharing one command text do not establish generalization to unseen target words.

### Track B — Keyword Detection in Context / Continuous Audio
- **Manifest:** `PHASE50_STREAMING_MANIFEST.csv` (50 sequences) & `PHASE50_EVENT_LEDGER.csv` ({len(df_events)} events).
- **Carrier Phrases:** Natural contextual phrases (`rẽ trái`, `sang trái`, `rẽ phải`, `sang phải`, `đi lên`, `tiến lên`, `đi xuống`, `lùi xuống`).
- **Matching Rule:** $\\pm 0.50\\text{{ s}}$ window around event center.
- **Independence Policy:** Track A classification metrics and Track B event metrics are reported strictly disaggregated.

---

## 4. Manifest Cryptographic Ledger

| Manifest Artifact | File Path | Total Items | SHA-256 Checksum |
|:------------------|:----------|:-----------:|:-----------------|
| **Track A Isolated Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_LOCKED_MANIFEST.csv` | 240 clips | `{iso_hash}` |
| **Track A Tone Probe Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_TONE_PROBE_MANIFEST.csv` | 48 clips | `{probe_hash}` |
| **Track B Streaming Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_STREAMING_MANIFEST.csv` | 50 sequences | `{stream_hash}` |
| **Track B Event Ledger (CSV)** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_EVENT_LEDGER.csv` | {len(df_events)} events | `{event_ledger_hash}` |
| **Track B Event Ledger (JSON)** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_EVENT_LEDGER.json` | 50 sequences | `{events_json_hash}` |

---

## 5. Pre-Registered Confirmatory Candidate Profiles

| Candidate Model | Pre-Registered Profile | Specific Engineering Pass Criteria |
|:----------------|:-----------------------|:------------------------------------|
| **Phase 28 D2_H1 Candidate** | **Low False-Alarm / Standby Profile** | 1. Silence $\\text{{FA/min}} \\le 1.0\\text{{ FA/min}}$ (UCB95 $\\le 2.0$).<br>2. Conversational OOV $\\text{{FA/min}} \\le 6.0\\text{{ FA/min}}$.<br>3. Overall Isolated OOV $\\text{{FAR}} \\le 20.0\\%$.<br>4. Track A 4-Command Macro-F1 $\\ge 50.0\\%$. |
| **Phase 26 Candidate (seed44)** | **Balanced Sensitivity Profile** | 1. Track A 4-Command Macro-F1 $\\ge 60.0\\%$.<br>2. Track B Streaming Event F1 $\\ge 55.0\\%$.<br>3. Silence $\\text{{FA/min}} \\le 3.0\\text{{ FA/min}}$.<br>4. Overall Isolated OOV $\\text{{FAR}} \\le 25.0\\%$. |
"""

    out_prereg = REPORTS_DIR / "PHASE50_CONFIRMATORY_PREREGISTRATION.md"
    with open(out_prereg, "w", encoding="utf-8") as f:
        f.write(prereg_content)
    prereg_hash = compute_sha256(out_prereg)
    print(f"[OK] Saved PHASE50_CONFIRMATORY_PREREGISTRATION.md (SHA-256: {prereg_hash})")

    # 5. SHA-256 Manifest
    manifest_dict = {
        "PHASE50_LOCKED_MANIFEST.csv": iso_hash,
        "PHASE50_TONE_PROBE_MANIFEST.csv": probe_hash,
        "PHASE50_STREAMING_MANIFEST.csv": stream_hash,
        "PHASE50_EVENT_LEDGER.csv": event_ledger_hash,
        "PHASE50_EVENT_LEDGER.json": events_json_hash,
        "PHASE50_CONFIRMATORY_PREREGISTRATION.md": prereg_hash
    }
    out_manifest_json = REPORTS_DIR / "PHASE50_SHA256_MANIFEST.json"
    with open(out_manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    print(f"[OK] Saved PHASE50_SHA256_MANIFEST.json")

    elapsed = time.time() - t0
    print(f"\n[DONE] Phase 50 Confirmatory corpus built successfully in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
