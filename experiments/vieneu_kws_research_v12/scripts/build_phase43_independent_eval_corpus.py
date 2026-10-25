#!/usr/bin/env python3
"""
Phase 43: Build Genuinely Independent Evaluation Corpus (MMS-TTS-vie + ESC-50)
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_phase43_independent_eval_corpus.py

Independent Components:
1. Speech Generator: facebook/mms-tts-vie (Meta AI VITS architecture)
   - Independent of VieNeu-TTS and Edge-TTS.
   - Fully verified with 0 UNKs, CC-BY-NC 4.0 license.
2. Noise Background: Karol Piczak ESC-50 indoor categories (Fold 1)
   - washing_machine, vacuum_cleaner, keyboard_typing, clock_tick.
   - Max cross-correlation < 0.12 with all 40 historical mic-noise tracks.
3. Multi-Instance Command & OOV Structure:
   - 4 instances per lexeme (varied seeds & duration scales) solving N_text=1 limitation.
   - 4 commands x 4 instances x 4 SNR tiers = 64 command clips
   - 16 phonetic hard OOVs x 4 instances x 4 SNR tiers = 256 phonetic OOV clips
   - 10 general OOVs x 4 instances x 4 SNR tiers = 160 general OOV clips
   - Total Isolated Clips = 480 clips.
   - 6 tone probes x 4 instances x 4 SNR tiers = 96 tone probe clips (diagnostic cohort).
4. Continuous Streaming Sequences:
   - 50 continuous sequences with exact event ledger.
"""

import sys
import io
import time
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
from scipy import signal

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
EVAL_AUDIO_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "audio" / "mms_esc50_eval_v1"
RAW_CLEAN_DIR = EVAL_AUDIO_DIR / "raw_clean_base"
ISOLATED_DIR = EVAL_AUDIO_DIR / "isolated"
STREAMING_DIR = EVAL_AUDIO_DIR / "streaming"
NOISE_DIR = PROJECT_ROOT / "data" / "processed" / "noise" / "esc50_16k"

MODEL_CACHE = Path("C:/Users/DELL/.cache/huggingface/hub/models--facebook--mms-tts-vie/snapshots/b58928d033932a49aa8e3d6cf11625b25fe928d2")

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from noise_mixer import NoiseAssetManager, mix_noise_at_snr

# 4 Synthesis Configurations per Lexeme (varying seed & speaking_rate to induce natural acoustic diversity)
SYNTH_CONFIGS = [
    {"cfg_id": "cfg_norm_s1", "seed": 1001, "rate": 1.00, "rate_label": "normal"},
    {"cfg_id": "cfg_fast_s2", "seed": 1002, "rate": 1.15, "rate_label": "fast"},
    {"cfg_id": "cfg_slow_s3", "seed": 1003, "rate": 0.85, "rate_label": "slow"},
    {"cfg_id": "cfg_cadn_s4", "seed": 1004, "rate": 1.05, "rate_label": "cadenced"}
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

TONE_PROBES = [
    "ma", "má", "mà", "mả", "mã", "mạ"
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def synthesize_word_mms(text: str, tokenizer: Any, model: Any, seed: int, speaking_rate: float) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt")
    torch.manual_seed(seed)
    with torch.no_grad():
        waveform = model(**inputs, speaking_rate=speaking_rate).waveform[0].cpu().numpy()

    # Model outputs 16 kHz mono float32
    audio = waveform.astype(np.float32)

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
    print("PHASE 43: BUILD GENUINELY INDEPENDENT EVALUATION CORPUS (MMS-TTS + ESC-50)")
    print("=" * 80)

    t_start = time.time()
    iso_creation_timestamp = datetime.now(timezone.utc).isoformat()

    RAW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_DIR.mkdir(parents=True, exist_ok=True)
    STREAMING_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MMS-TTS-vie model from {MODEL_CACHE}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CACHE))
    model = VitsModel.from_pretrained(str(MODEL_CACHE))
    model.eval()

    noise_mgr = NoiseAssetManager(NOISE_DIR)
    num_noise_tracks = noise_mgr.get_track_count()
    print(f"Loaded {num_noise_tracks} verified ESC-50 noise tracks from {NOISE_DIR}.")

    # 1. Assemble vocabulary
    # (word, label, subgroup, phonetic_target, is_probe)
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

    print(f"Total vocabulary items: {len(word_entries)} (4 commands, 16 phonetic OOVs, 10 general OOVs, 6 tone probes)")

    # 2. Synthesize base clean audio
    print("\n--- 1. Synthesizing Base Clean Audio ---")
    clean_base_cache = {}
    for txt, lbl, subg, p_target, is_probe in word_entries:
        safe_txt = txt.replace(" ", "_")
        for cfg in SYNTH_CONFIGS:
            key = f"{safe_txt}_{cfg['cfg_id']}"
            raw_path = RAW_CLEAN_DIR / f"{key}.wav"
            if not raw_path.exists():
                audio = synthesize_word_mms(
                    text=txt,
                    tokenizer=tokenizer,
                    model=model,
                    seed=cfg["seed"],
                    speaking_rate=cfg["rate"]
                )
                sf.write(str(raw_path), audio, 16000)
            data, _ = sf.read(str(raw_path), dtype="float32")
            clean_base_cache[key] = {
                "audio": data,
                "1s": make_1s_clip(data),
                "text": txt,
                "label": lbl,
                "subgroup": subg,
                "phonetic_target": p_target,
                "is_probe": is_probe,
                "config_id": cfg["cfg_id"],
                "synth_seed": cfg["seed"],
                "speaking_rate": cfg["rate"],
                "rate_label": cfg["rate_label"]
            }

    print(f"[OK] Clean base cache built: {len(clean_base_cache)} unique utterances.")

    # 3. Build Isolated Evaluation Manifests
    # Primary: 480 clips (4 commands x 16 + 16 phonetic x 16 + 10 general x 16)
    # Diagnostic: 96 tone probe clips
    print("\n--- 2. Building Isolated Test Corpus (480 Primary Clips + 96 Diagnostic Probes) ---")
    snr_tiers = [None, 15.0, 10.0, 5.0]
    isolated_rows = []
    diagnostic_rows = []

    clip_counter = 0
    probe_counter = 0

    for key, item in clean_base_cache.items():
        if not item["is_probe"]:
            # Primary cohort
            for snr in snr_tiers:
                clip_counter += 1
                cid = f"p43_mms_iso_{clip_counter:04d}_{key}_{'clean' if snr is None else f'snr{int(snr)}'}"
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
                        seed=77000 + clip_counter,
                        duration_samples=16000
                    )
                    final_audio, _, _, _ = mix_noise_at_snr(
                        speech=clean_1s,
                        noise=noise_slice,
                        target_snr_db=snr
                    )
                    snr_str = str(snr)
                    noise_src = track_path.stem
                    noise_track_id = f"esc50_track_{track_idx}"

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
                    "generator": "facebook/mms-tts-vie",
                    "config_id": item["config_id"],
                    "synth_seed": item["synth_seed"],
                    "speaking_rate": item["speaking_rate"],
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
                cid = f"p43_mms_probe_{probe_counter:04d}_{key}_{'clean' if snr is None else f'snr{int(snr)}'}"
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
                        seed=88000 + probe_counter,
                        duration_samples=16000
                    )
                    final_audio, _, _, _ = mix_noise_at_snr(
                        speech=clean_1s,
                        noise=noise_slice,
                        target_snr_db=snr
                    )
                    snr_str = str(snr)
                    noise_src = track_path.stem
                    noise_track_id = f"esc50_track_{track_idx}"

                out_wav = ISOLATED_DIR / f"{cid}.wav"
                sf.write(str(out_wav), final_audio.astype(np.float32), 16000)
                sha = compute_sha256(out_wav)
                rel_path = str(out_wav.relative_to(PROJECT_ROOT)).replace("\\", "/")

                diagnostic_rows.append({
                    "clip_id": cid,
                    "label": item["label"],
                    "text": item["text"],
                    "subgroup": item["subgroup"],
                    "phonetic_target": item["phonetic_target"],
                    "generator": "facebook/mms-tts-vie",
                    "config_id": item["config_id"],
                    "synth_seed": item["synth_seed"],
                    "speaking_rate": item["speaking_rate"],
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
    out_iso_manifest = MANIFESTS_DIR / "PHASE43_LOCKED_MANIFEST.csv"
    df_isolated.to_csv(out_iso_manifest, index=False, encoding="utf-8")
    iso_hash = compute_sha256(out_iso_manifest)

    df_diag = pd.DataFrame(diagnostic_rows)
    out_diag_manifest = MANIFESTS_DIR / "PHASE43_TONE_PROBE_MANIFEST.csv"
    df_diag.to_csv(out_diag_manifest, index=False, encoding="utf-8")
    diag_hash = compute_sha256(out_diag_manifest)

    print(f"[OK] Saved PHASE43_LOCKED_MANIFEST.csv: {len(df_isolated)} clips, SHA-256: {iso_hash}")
    print(f"     Label breakdown: {df_isolated['label'].value_counts().to_dict()}")
    print(f"     Subgroup breakdown: {df_isolated['subgroup'].value_counts().to_dict()}")
    print(f"[OK] Saved PHASE43_TONE_PROBE_MANIFEST.csv: {len(df_diag)} clips, SHA-256: {diag_hash}")

    # 4. Build Continuous Streaming Evaluation Corpus (50 Sequences)
    print("\n--- 3. Building Streaming Evaluation Corpus (50 sequences) ---")
    rng = np.random.RandomState(43043)
    streaming_rows = []
    streaming_events = {}

    all_keys = [k for k in clean_base_cache.keys() if not clean_base_cache[k]["is_probe"]]
    cmd_keys = [k for k in all_keys if clean_base_cache[k]["subgroup"] == "command"]
    oov_keys = [k for k in all_keys if clean_base_cache[k]["subgroup"] != "command"]

    for seq_idx in range(1, 51):
        seq_id = f"p43_stream_seq_{seq_idx:03d}"
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
                seed=93000 + seq_idx,
                duration_samples=num_samples
            )
            # Normal ambient indoor level
            audio = audio * 0.15
            noise_src = track_path.stem

        elif seq_idx <= 20:
            seq_type = "pure_oov_conversation"
            target_snr = None
            n_events = rng.randint(4, 8)
            seq_audio_blocks = []
            cur_time = rng.uniform(0.5, 1.2)
            for e_i in range(n_events):
                k = rng.choice(oov_keys)
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
                    "sequence_id": seq_id,
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
            noise_src = "none"

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
                    "sequence_id": seq_id,
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

            track_idx = seq_idx % num_noise_tracks
            track_path = noise_mgr.get_track_path(track_idx)
            noise_slice = noise_mgr.extract_noise_slice(
                track_index=track_idx,
                seed=94000 + seq_idx,
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
    out_stream_manifest = MANIFESTS_DIR / "PHASE43_STREAMING_MANIFEST.csv"
    df_streaming.to_csv(out_stream_manifest, index=False, encoding="utf-8")
    stream_hash = compute_sha256(out_stream_manifest)

    # Flatten events to CSV and JSON
    all_events_flat = []
    for seq_id, ev_list in streaming_events.items():
        all_events_flat.extend(ev_list)
    df_events = pd.DataFrame(all_events_flat)
    out_event_ledger = MANIFESTS_DIR / "PHASE43_EVENT_LEDGER.csv"
    df_events.to_csv(out_event_ledger, index=False, encoding="utf-8")
    event_ledger_hash = compute_sha256(out_event_ledger)

    out_events_json = MANIFESTS_DIR / "PHASE43_EVENT_LEDGER.json"
    with open(out_events_json, "w", encoding="utf-8") as f:
        json.dump(streaming_events, f, indent=2, ensure_ascii=False)
    events_json_hash = compute_sha256(out_events_json)

    print(f"[OK] Saved PHASE43_STREAMING_MANIFEST.csv: {len(df_streaming)} sequences, SHA-256: {stream_hash}")
    print(f"[OK] Saved PHASE43_EVENT_LEDGER.csv: {len(df_events)} events, SHA-256: {event_ledger_hash}")
    print(f"[OK] Saved PHASE43_EVENT_LEDGER.json, SHA-256: {events_json_hash}")

    # 5. Pre-Registration Document
    freeze_timestamp = datetime.now(timezone.utc).isoformat()
    prereg_content = f"""# PHASE 43: LOCKED INDEPENDENT EVALUATION PROTOCOL & PRE-REGISTRATION

**Repository:** `D:\\trainidentificationaudio`  
**Phase:** Phase 43 (Locked Inter-Generator Evaluation Protocol)  
**Creation Timestamp:** `{iso_creation_timestamp}`  
**Manifest Freeze Timestamp:** `{freeze_timestamp}`  
**Status:** **LOCKED & PRE-REGISTERED (PRIOR TO EVALUATION INFERENCE)**

---

## 1. Governance & Protocol Invariants

1. **Default Model Status:** Legacy $v11$ remains the canonical default model.
2. **Model Integrity:** Checkpoints for $v11$, $E1$, and Phase 28 $D_2\\_H1\\_seed42$ are strictly read-only and immutable.
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

## 2. Independent Corpus Provenance & Hashes

### 2.1 Acoustic Generators & Noise Assets
- **Acoustic Speech Generator:** `facebook/mms-tts-vie` (Meta AI VITS architecture, HuggingFace commit `b58928d0...`).
- **Environmental Noise:** Karol Piczak ESC-50 Fold 1 indoor categories (`washing_machine`, `vacuum_cleaner`, `keyboard_typing`, `clock_tick`). Fully verified non-overlapping with historical noise ($r_{{\\max}} < 0.12$).
- **Multi-Instance Strategy:** 4 distinct synthesis instances per lexeme (varying seed and speaking rate $\\in [0.85, 1.15]$), solving the $N_{{\\text{{text}}}}=1$ cluster limitation identified in Phase 40.

### 2.2 Manifest SHA-256 Ledger

| Manifest Artifact | File Path | Total Items | SHA-256 Checksum |
|:------------------|:----------|:-----------:|:-----------------|
| **Isolated Test Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_LOCKED_MANIFEST.csv` | 480 clips | `{iso_hash}` |
| **Tone Probe Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_TONE_PROBE_MANIFEST.csv` | 96 clips | `{diag_hash}` |
| **Streaming Test Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_STREAMING_MANIFEST.csv` | 50 sequences | `{stream_hash}` |
| **Streaming Event Ledger (CSV)** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_EVENT_LEDGER.csv` | {len(df_events)} events | `{event_ledger_hash}` |
| **Streaming Event Ledger (JSON)** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_EVENT_LEDGER.json` | 50 sequences | `{events_json_hash}` |

---

## 3. Pre-Registered Metrics & Statistical Protocol

### 3.1 Primary Isolated Metrics
1. **4-Command Macro-F1:** Unweighted mean of $F_1$ across `TRAI`, `PHAI`, `LEN`, `XUONG`.
2. **5-Class Macro-F1:** Unweighted mean of $F_1$ across all 5 classes (including `REJECT`).
3. **Phonetic OOV False Alarm Rate:** FAR on phonetic hard negative cohort ($N = 256$).
4. **General OOV False Alarm Rate:** FAR on general everyday OOV cohort ($N = 160$).
5. **Stratified Slices:** SNR tiers (`clean`, `15 dB`, `10 dB`, `5 dB`), Speaking Rate (`slow`, `normal`, `cadenced`, `fast`), and Synthesis Instance.

### 3.2 Primary Streaming Metrics
1. **Exact Sequence Accuracy:** Fraction of 50 sequences where predicted command sequence matches ground truth exactly.
2. **Event Metrics:** Precision, Recall, $F_1$ with temporal matching window $\\pm 0.50\\text{{ s}}$ around event midpoint.
3. **Continuous Exposure Denominators:**
   - Pure Silence/Noise FA/min (10 sequences, $\\sim 100\\text{{ s}}$ total exposure).
   - Pure OOV Conversation FA/min (10 sequences, $\\sim 100\\text{{ s}}$ total exposure).
4. **Zero-Numerator Upper 95% Confidence Bounds (Rule of Three):**
   - When 0 false alarms are observed over $N$ opportunities:
     $$\\text{{UCB95}} = 1 - (1 - 0.95)^{{1/N}} \\approx \\frac{{3}}{{N}}$$
   - When 0 false alarms are observed over $T_{{\\text{{min}}}}$ minutes of continuous audio:
     $$\\text{{UCB95}} = \\frac{{3}}{{T_{{\\text{{min}}}}}} \\text{{ FA/min}}$$

### 3.3 Bootstrap Resampling Hierarchy
- Estimand A: Observed point estimate difference.
- Estimand B: Lexeme Cluster Bootstrap (30 base text clusters).
- Estimand C: Synthesis Instance Cluster Bootstrap (120 groups: 30 texts $\\times$ 4 synthesis instances).
- Estimand D: Clip-level i.i.d. bootstrap ($N = 480$).
"""
    prereg_file = REPORTS_DIR / "PHASE43_FINAL_PREREGISTRATION.md"
    with open(prereg_file, "w", encoding="utf-8") as f:
        f.write(prereg_content)
    prereg_hash = compute_sha256(prereg_file)
    print(f"[OK] Saved PHASE43_FINAL_PREREGISTRATION.md (SHA-256: {prereg_hash})")

    elapsed = time.time() - t_start
    print(f"\n[DONE] Phase 43 independent corpus built successfully in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
