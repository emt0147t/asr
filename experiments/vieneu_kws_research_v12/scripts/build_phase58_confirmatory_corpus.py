#!/usr/bin/env python3
"""
Phase 58: Confirmatory Corpus Builder (Piper VAIS-1000 + ESC-50 Fold 3)
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/build_phase58_confirmatory_corpus.py

Builds:
1. Track A Isolated Manifest: 300 primary clips (40 commands + 160 phonetic OOVs + 100 general OOVs)
   derived from 75 clean waveforms x 4 SNR tiers (clean, 15 dB, 10 dB, 5 dB).
2. Track A Diagnostic Tone Probe Manifest: 60 clips derived from 15 clean waveforms x 4 SNR tiers.
3. Track B Continuous Streaming Manifest: 50 sequences (silence, OOV conversation, commands in carrier context)
   along with exact annotated event ledger (PHASE58_EVENT_LEDGER.csv).
All synthesized with Piper VAIS-1000 (deterministic noise_scale=0.0) and ESC-50 Fold 3 indoor noise.
"""

import sys
import io
import wave
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from piper.voice import PiperVoice
from piper.config import SynthesisConfig

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIO_OUT_DIR = PROJECT_ROOT / "data" / "candidate_tts" / "phase58_confirmatory"
TRACKA_DIR = AUDIO_OUT_DIR / "trackA_isolated"
TRACKB_DIR = AUDIO_OUT_DIR / "trackB_streaming"
CLEAN_BASE_DIR = AUDIO_OUT_DIR / "clean_base"
NOISE_DIR = PROJECT_ROOT / "data" / "processed" / "noise" / "esc50_fold3_16k"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"

SNR_TIERS = ["clean", 15.0, 10.0, 5.0]

COMMANDS = {
    "TRAI": "trái",
    "PHAI": "phải",
    "LEN": "lên",
    "XUONG": "xuống"
}

PHONETIC_OOVS = [
    {"text": "trước", "phonetic_target": "TRAI"},
    {"text": "trường", "phonetic_target": "TRAI"},
    {"text": "trưa", "phonetic_target": "TRAI"},
    {"text": "trẻ", "phonetic_target": "TRAI"},
    {"text": "phim", "phonetic_target": "PHAI"},
    {"text": "phút", "phonetic_target": "PHAI"},
    {"text": "phương", "phonetic_target": "PHAI"},
    {"text": "phát", "phonetic_target": "PHAI"},
    {"text": "lớp", "phonetic_target": "LEN"},
    {"text": "luôn", "phonetic_target": "LEN"},
    {"text": "làm", "phonetic_target": "LEN"},
    {"text": "lòng", "phonetic_target": "LEN"},
    {"text": "xem", "phonetic_target": "XUONG"},
    {"text": "xanh", "phonetic_target": "XUONG"},
    {"text": "xuyên", "phonetic_target": "XUONG"},
    {"text": "xa", "phonetic_target": "XUONG"}
]

GENERAL_OOVS = [
    "cửa", "bàn", "ghế", "nhà", "xe", "tiền", "nước", "sách", "bút", "máy"
]

TONE_PROBES = ["ma", "má", "mà", "mả", "mã", "mạ"]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resample_22k_to_16k(audio: np.ndarray, orig_sr: int = 22050, target_sr: int = 16000) -> np.ndarray:
    return signal.resample_poly(audio, 320, 441).astype(np.float32)


def mix_speech_with_noise(speech: np.ndarray, noise: np.ndarray, target_snr_db: Any) -> np.ndarray:
    if target_snr_db == "clean" or target_snr_db is None:
        return speech.copy()

    snr_val = float(target_snr_db)
    speech_energy = np.mean(speech ** 2)
    if speech_energy < 1e-8:
        return speech.copy()

    # Match length
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)[:len(speech)]
    elif len(noise) > len(speech):
        noise = noise[:len(speech)]

    noise_energy = np.mean(noise ** 2)
    if noise_energy < 1e-8:
        return speech.copy()

    desired_noise_energy = speech_energy / (10.0 ** (snr_val / 10.0))
    scale = np.sqrt(desired_noise_energy / noise_energy)
    mixed = speech + noise * scale

    # Peak normalize if clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    return mixed.astype(np.float32)


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
    print("PHASE 58: CONFIRMATORY CORPUS GENERATION & VALIDATION")
    print("=" * 80)

    for d in [AUDIO_OUT_DIR, TRACKA_DIR, TRACKB_DIR, CLEAN_BASE_DIR, MANIFESTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Piper VAIS-1000
    onnx_path = PROJECT_ROOT / "data" / "candidate_tts" / "piper_vais1000" / "vi_VN-vais1000-medium.onnx"
    json_path = PROJECT_ROOT / "data" / "candidate_tts" / "piper_vais1000" / "vi_VN-vais1000-medium.onnx.json"
    voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
    print(f"Loaded Piper VAIS-1000: Native SR = {voice.config.sample_rate}")

    # 2. Load ESC-50 Fold 3 noise tracks
    noise_tracks = {}
    for p in NOISE_DIR.glob("*.wav"):
        data, _ = sf.read(str(p), dtype="float32")
        noise_tracks[p.name] = data
    print(f"Loaded {len(noise_tracks)} ESC-50 Fold 3 noise tracks into RAM.")
    noise_names = sorted(list(noise_tracks.keys()))

    # -------------------------------------------------------------------------
    # STAGE 1: SYNTHESIZE CLEAN INSTANCES FOR TRACK A
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Synthesizing Clean Waveforms for Track A ---")
    clean_manifest = []

    # A. Target Commands: 4 commands x 10 length_scales = 40 clean instances
    cmd_length_scales = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    for cmd_lbl, cmd_txt in COMMANDS.items():
        for idx, ls in enumerate(cmd_length_scales):
            inst_id = f"cmd_{cmd_lbl}_ls{int(ls*100)}_{idx+1}"
            cfg = SynthesisConfig(length_scale=ls, noise_scale=0.0, noise_w_scale=0.0)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w_out:
                voice.synthesize_wav(cmd_txt, w_out, syn_config=cfg)
            buf.seek(0)
            raw_a, sr = sf.read(buf, dtype="float32")
            a_16k = resample_22k_to_16k(raw_a, sr, 16000)

            # normalize and pad to 1.0s
            peak = np.max(np.abs(a_16k))
            if peak > 0:
                a_16k = a_16k * (0.85 / peak)
            a_1s = make_1s_clip(a_16k, 16000)

            raw_path = CLEAN_BASE_DIR / f"{inst_id}.wav"
            sf.write(str(raw_path), a_1s, 16000, subtype="PCM_16")

            clean_manifest.append({
                "instance_id": inst_id,
                "text": cmd_txt,
                "label": cmd_lbl,
                "subgroup": "command",
                "phonetic_target": cmd_lbl,
                "length_scale": ls,
                "clean_audio_path": str(raw_path.relative_to(PROJECT_ROOT))
            })

    # B. Phonetic Hard Negatives: 16 words x 2.5 configs = 40 clean instances
    # We assign 2 configs to 8 words, 3 configs to 8 words (total 40)
    for w_i, item in enumerate(PHONETIC_OOVS):
        w_txt = item["text"]
        w_target = item["phonetic_target"]
        n_configs = 3 if w_i < 8 else 2
        scales = [0.95, 1.05, 1.20] if n_configs == 3 else [0.95, 1.15]

        for idx, ls in enumerate(scales):
            inst_id = f"oov_phon_{w_txt}_ls{int(ls*100)}_{idx+1}"
            cfg = SynthesisConfig(length_scale=ls, noise_scale=0.0, noise_w_scale=0.0)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w_out:
                voice.synthesize_wav(w_txt, w_out, syn_config=cfg)
            buf.seek(0)
            raw_a, sr = sf.read(buf, dtype="float32")
            a_16k = resample_22k_to_16k(raw_a, sr, 16000)

            peak = np.max(np.abs(a_16k))
            if peak > 0:
                a_16k = a_16k * (0.85 / peak)
            a_1s = make_1s_clip(a_16k, 16000)

            raw_path = CLEAN_BASE_DIR / f"{inst_id}.wav"
            sf.write(str(raw_path), a_1s, 16000, subtype="PCM_16")

            clean_manifest.append({
                "instance_id": inst_id,
                "text": w_txt,
                "label": "REJECT",
                "subgroup": "test_oov_phonetic",
                "phonetic_target": w_target,
                "length_scale": ls,
                "clean_audio_path": str(raw_path.relative_to(PROJECT_ROOT))
            })

    # C. General Conversational OOVs: 10 words x 2.5 configs = 25 clean instances
    for w_i, w_txt in enumerate(GENERAL_OOVS):
        n_configs = 3 if w_i < 5 else 2
        scales = [0.95, 1.05, 1.20] if n_configs == 3 else [0.95, 1.15]

        for idx, ls in enumerate(scales):
            inst_id = f"oov_gen_{w_txt}_ls{int(ls*100)}_{idx+1}"
            cfg = SynthesisConfig(length_scale=ls, noise_scale=0.0, noise_w_scale=0.0)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w_out:
                voice.synthesize_wav(w_txt, w_out, syn_config=cfg)
            buf.seek(0)
            raw_a, sr = sf.read(buf, dtype="float32")
            a_16k = resample_22k_to_16k(raw_a, sr, 16000)

            peak = np.max(np.abs(a_16k))
            if peak > 0:
                a_16k = a_16k * (0.85 / peak)
            a_1s = make_1s_clip(a_16k, 16000)

            raw_path = CLEAN_BASE_DIR / f"{inst_id}.wav"
            sf.write(str(raw_path), a_1s, 16000, subtype="PCM_16")

            clean_manifest.append({
                "instance_id": inst_id,
                "text": w_txt,
                "label": "REJECT",
                "subgroup": "test_oov_general",
                "phonetic_target": "none",
                "length_scale": ls,
                "clean_audio_path": str(raw_path.relative_to(PROJECT_ROOT))
            })

    # D. Tone Probes: 6 words x 2.5 configs = 15 clean instances
    tone_manifest = []
    for w_i, w_txt in enumerate(TONE_PROBES):
        n_configs = 3 if w_i < 3 else 2
        scales = [0.95, 1.05, 1.20] if n_configs == 3 else [0.95, 1.15]

        for idx, ls in enumerate(scales):
            inst_id = f"probe_tone_{w_txt}_ls{int(ls*100)}_{idx+1}"
            cfg = SynthesisConfig(length_scale=ls, noise_scale=0.0, noise_w_scale=0.0)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w_out:
                voice.synthesize_wav(w_txt, w_out, syn_config=cfg)
            buf.seek(0)
            raw_a, sr = sf.read(buf, dtype="float32")
            a_16k = resample_22k_to_16k(raw_a, sr, 16000)

            peak = np.max(np.abs(a_16k))
            if peak > 0:
                a_16k = a_16k * (0.85 / peak)
            a_1s = make_1s_clip(a_16k, 16000)

            raw_path = CLEAN_BASE_DIR / f"{inst_id}.wav"
            sf.write(str(raw_path), a_1s, 16000, subtype="PCM_16")

            tone_manifest.append({
                "instance_id": inst_id,
                "text": w_txt,
                "label": "REJECT",
                "subgroup": "tone_probe",
                "phonetic_target": "none",
                "length_scale": ls,
                "clean_audio_path": str(raw_path.relative_to(PROJECT_ROOT))
            })

    print(f"Generated clean instances: {len(clean_manifest)} primary (40 cmd, 40 phon, 25 gen) + {len(tone_manifest)} tone probes.")
    assert len(clean_manifest) == 105, f"Expected 105 primary instances, got {len(clean_manifest)}"
    assert len(tone_manifest) == 15, f"Expected 15 tone probe instances, got {len(tone_manifest)}"

    # -------------------------------------------------------------------------
    # STAGE 2: BUILD TRACK A EVALUATION MANIFEST (4 SNR TIERS)
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Mixing Track A Clips with ESC-50 Fold 3 Noise ---")
    rng = np.random.RandomState(58042)

    trackA_rows = []
    for inst in clean_manifest:
        inst_id = inst["instance_id"]
        clean_path = PROJECT_ROOT / inst["clean_audio_path"]
        speech_1s, _ = sf.read(str(clean_path), dtype="float32")

        for snr in SNR_TIERS:
            snr_str = str(snr)
            if snr == "clean":
                noise_name = "none"
                mixed = speech_1s.copy()
            else:
                noise_name = rng.choice(noise_names)
                noise_full = noise_tracks[noise_name]
                start_n = rng.randint(0, len(noise_full) - 16000)
                noise_slice = noise_full[start_n : start_n + 16000]
                mixed = mix_speech_with_noise(speech_1s, noise_slice, snr)

            clip_id = f"p58_trackA_{inst_id}_{snr_str}"
            out_file = TRACKA_DIR / f"{clip_id}.wav"
            sf.write(str(out_file), mixed, 16000, subtype="PCM_16")

            trackA_rows.append({
                "clip_id": clip_id,
                "label": inst["label"],
                "text": inst["text"],
                "subgroup": inst["subgroup"],
                "phonetic_target": inst["phonetic_target"],
                "generator": "Piper_VAIS1000",
                "synthesis_instance_id": inst_id,
                "length_scale": inst["length_scale"],
                "target_snr_db": snr_str,
                "noise_source": "ESC50_Fold3" if snr != "clean" else "none",
                "noise_track_id": noise_name,
                "audio_path": str(out_file.relative_to(PROJECT_ROOT)),
                "duration_seconds": 1.0,
                "num_samples": 16000,
                "sha256": compute_sha256(out_file)
            })

    df_trackA = pd.DataFrame(trackA_rows)
    trackA_manifest_path = MANIFESTS_DIR / "PHASE58_CONFIRMATORY_MANIFEST.csv"
    df_trackA.to_csv(trackA_manifest_path, index=False, encoding="utf-8")
    print(f"Track A Primary Manifest: {len(df_trackA)} clips saved to {trackA_manifest_path}")
    print("Class counts in Track A:")
    print(df_trackA["label"].value_counts())
    print("Subgroup counts in Track A:")
    print(df_trackA["subgroup"].value_counts())

    # Tone Probe Manifest (15 instances x 4 SNRs = 60 clips)
    probe_rows = []
    for inst in tone_manifest:
        inst_id = inst["instance_id"]
        clean_path = PROJECT_ROOT / inst["clean_audio_path"]
        speech_1s, _ = sf.read(str(clean_path), dtype="float32")

        for snr in SNR_TIERS:
            snr_str = str(snr)
            if snr == "clean":
                noise_name = "none"
                mixed = speech_1s.copy()
            else:
                noise_name = rng.choice(noise_names)
                noise_full = noise_tracks[noise_name]
                start_n = rng.randint(0, len(noise_full) - 16000)
                noise_slice = noise_full[start_n : start_n + 16000]
                mixed = mix_speech_with_noise(speech_1s, noise_slice, snr)

            clip_id = f"p58_probe_{inst_id}_{snr_str}"
            out_file = TRACKA_DIR / f"{clip_id}.wav"
            sf.write(str(out_file), mixed, 16000, subtype="PCM_16")

            probe_rows.append({
                "clip_id": clip_id,
                "label": "REJECT",
                "text": inst["text"],
                "subgroup": "tone_probe",
                "phonetic_target": "none",
                "generator": "Piper_VAIS1000",
                "synthesis_instance_id": inst_id,
                "length_scale": inst["length_scale"],
                "target_snr_db": snr_str,
                "noise_source": "ESC50_Fold3" if snr != "clean" else "none",
                "noise_track_id": noise_name,
                "audio_path": str(out_file.relative_to(PROJECT_ROOT)),
                "duration_seconds": 1.0,
                "num_samples": 16000,
                "sha256": compute_sha256(out_file)
            })

    df_probe = pd.DataFrame(probe_rows)
    probe_manifest_path = MANIFESTS_DIR / "PHASE58_TONE_PROBE_MANIFEST.csv"
    df_probe.to_csv(probe_manifest_path, index=False, encoding="utf-8")
    print(f"Track A Tone Probe Manifest: {len(df_probe)} clips saved to {probe_manifest_path}")

    # -------------------------------------------------------------------------
    # STAGE 3: BUILD TRACK B CONTINUOUS STREAMING AUDIO (50 SEQUENCES)
    # -------------------------------------------------------------------------
    print("\n--- Stage 3: Synthesizing Track B Continuous Streaming Sequences ---")
    CARRIER_PHRASES = {
        "TRAI": ["rẽ trái", "sang trái", "cua trái", "bẻ lái trái"],
        "PHAI": ["rẽ phải", "sang phải", "cua phải", "bẻ lái phải"],
        "LEN": ["đi lên", "tiến lên", "bước lên", "chạy lên"],
        "XUONG": ["đi xuống", "lùi xuống", "bước xuống", "chạy xuống"]
    }

    CONVERSATIONAL_PHRASES = [
        "hôm nay thời tiết đẹp quá", "tôi đang đọc một cuốn sách", "chúng ta sẽ gặp nhau ngày mai",
        "xin vui lòng đóng cửa lại", "bạn có khỏe không", "chiếc xe này chạy rất nhanh",
        "buổi sáng thức dậy lúc sáu giờ", "trà này uống rất ngon", "hãy ngồi xuống chiếc ghế đó"
    ]

    streaming_manifest_rows = []
    event_ledger_rows = []

    # 50 Sequences total:
    # - 10 pure silence/noise sequences (4.0s - 6.0s each)
    # - 10 pure conversational OOV sequences (4.0s - 6.0s each)
    # - 30 command sequences with natural carrier phrases and background noise
    seq_rng = np.random.RandomState(58050)

    # 1. Pure Silence / Indoor Noise Sequences (10)
    for i in range(1, 11):
        seq_id = f"p58_stream_silence_{i:02d}"
        dur_s = seq_rng.uniform(4.5, 6.0)
        num_s = int(dur_s * 16000)

        noise_name = seq_rng.choice(noise_names)
        noise_full = noise_tracks[noise_name]
        start_n = seq_rng.randint(0, max(1, len(noise_full) - num_s))
        noise_slice = noise_full[start_n : start_n + num_s]
        if len(noise_slice) < num_s:
            noise_slice = np.pad(noise_slice, (0, num_s - len(noise_slice)), mode="wrap")

        # Scale noise to -24 dBFS
        peak = np.max(np.abs(noise_slice))
        if peak > 0:
            noise_slice = noise_slice * (0.063 / peak)

        out_path = TRACKB_DIR / f"{seq_id}.wav"
        sf.write(str(out_path), noise_slice, 16000, subtype="PCM_16")

        streaming_manifest_rows.append({
            "sequence_id": seq_id,
            "sequence_type": "pure_silence_noise",
            "duration_seconds": round(dur_s, 3),
            "num_samples": num_s,
            "noise_source": "ESC50_Fold3",
            "noise_track_id": noise_name,
            "audio_path": str(out_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(out_path)
        })

    # 2. Pure Conversational OOV Sequences (10)
    for i in range(1, 11):
        seq_id = f"p58_stream_oov_{i:02d}"
        phrase = seq_rng.choice(CONVERSATIONAL_PHRASES)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w_out:
            voice.synthesize_wav(phrase, w_out, syn_config=SynthesisConfig(length_scale=1.0, noise_scale=0.0, noise_w_scale=0.0))
        buf.seek(0)
        raw_a, sr = sf.read(buf, dtype="float32")
        a_16k = resample_22k_to_16k(raw_a, sr, 16000)

        # Pad with 1.0s silence on both sides
        pad_front = np.zeros(int(16000 * seq_rng.uniform(0.8, 1.2)), dtype=np.float32)
        pad_back = np.zeros(int(16000 * seq_rng.uniform(0.8, 1.2)), dtype=np.float32)
        full_a = np.concatenate([pad_front, a_16k, pad_back])

        # Mix with low background noise
        noise_name = seq_rng.choice(noise_names)
        noise_slice = noise_tracks[noise_name][:len(full_a)]
        if len(noise_slice) < len(full_a):
            noise_slice = np.pad(noise_slice, (0, len(full_a) - len(noise_slice)), mode="wrap")

        mixed = mix_speech_with_noise(full_a, noise_slice, 15.0)
        dur_s = len(mixed) / 16000.0

        out_path = TRACKB_DIR / f"{seq_id}.wav"
        sf.write(str(out_path), mixed, 16000, subtype="PCM_16")

        streaming_manifest_rows.append({
            "sequence_id": seq_id,
            "sequence_type": "pure_oov_conversation",
            "duration_seconds": round(dur_s, 3),
            "num_samples": len(mixed),
            "noise_source": "ESC50_Fold3",
            "noise_track_id": noise_name,
            "audio_path": str(out_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(out_path)
        })

    # 3. Contextual Command Sequences (30)
    for i in range(1, 31):
        seq_id = f"p58_stream_cmd_{i:02d}"
        n_cmds = seq_rng.choice([1, 2, 3])
        cmds_chosen = seq_rng.choice(["TRAI", "PHAI", "LEN", "XUONG"], size=n_cmds, replace=True)

        audio_chunks = []
        cur_pos_s = 0.0

        # Initial silence
        init_sil = seq_rng.uniform(0.6, 1.2)
        audio_chunks.append(np.zeros(int(init_sil * 16000), dtype=np.float32))
        cur_pos_s += init_sil

        for c_lbl in cmds_chosen:
            carrier_text = seq_rng.choice(CARRIER_PHRASES[c_lbl])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w_out:
                voice.synthesize_wav(carrier_text, w_out, syn_config=SynthesisConfig(length_scale=1.0, noise_scale=0.0, noise_w_scale=0.0))
            buf.seek(0)
            raw_a, sr = sf.read(buf, dtype="float32")
            c_audio = resample_22k_to_16k(raw_a, sr, 16000)

            # Record event
            event_dur = len(c_audio) / 16000.0
            start_t = cur_pos_s
            end_t = start_t + event_dur

            event_ledger_rows.append({
                "sequence_id": seq_id,
                "label": c_lbl,
                "text": carrier_text,
                "start_time": round(start_t, 3),
                "end_time": round(end_t, 3),
                "duration_seconds": round(event_dur, 3)
            })

            audio_chunks.append(c_audio)
            cur_pos_s += event_dur

            # Inter-command pause
            pause_s = seq_rng.uniform(1.0, 1.8)
            audio_chunks.append(np.zeros(int(pause_s * 16000), dtype=np.float32))
            cur_pos_s += pause_s

        full_speech = np.concatenate(audio_chunks)
        dur_s = len(full_speech) / 16000.0

        # Mix with noise at 15 dB or 10 dB
        noise_name = seq_rng.choice(noise_names)
        noise_slice = noise_tracks[noise_name][:len(full_speech)]
        if len(noise_slice) < len(full_speech):
            noise_slice = np.pad(noise_slice, (0, len(full_speech) - len(noise_slice)), mode="wrap")

        snr_target = seq_rng.choice([15.0, 10.0])
        mixed = mix_speech_with_noise(full_speech, noise_slice, snr_target)

        out_path = TRACKB_DIR / f"{seq_id}.wav"
        sf.write(str(out_path), mixed, 16000, subtype="PCM_16")

        streaming_manifest_rows.append({
            "sequence_id": seq_id,
            "sequence_type": "contextual_commands",
            "duration_seconds": round(dur_s, 3),
            "num_samples": len(mixed),
            "noise_source": "ESC50_Fold3",
            "noise_track_id": noise_name,
            "audio_path": str(out_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(out_path)
        })

    df_stream = pd.DataFrame(streaming_manifest_rows)
    stream_manifest_path = MANIFESTS_DIR / "PHASE58_STREAMING_MANIFEST.csv"
    df_stream.to_csv(stream_manifest_path, index=False, encoding="utf-8")
    print(f"Track B Streaming Manifest: {len(df_stream)} sequences saved to {stream_manifest_path}")

    df_ledger = pd.DataFrame(event_ledger_rows)
    ledger_csv_path = MANIFESTS_DIR / "PHASE58_EVENT_LEDGER.csv"
    df_ledger.to_csv(ledger_csv_path, index=False, encoding="utf-8")

    # Group events by sequence_id for JSON ledger
    events_by_seq = {}
    for seq_id, group in df_ledger.groupby("sequence_id"):
        events_by_seq[seq_id] = group.to_dict(orient="records")

    ledger_json_path = MANIFESTS_DIR / "PHASE58_EVENT_LEDGER.json"
    ledger_json_path.write_text(json.dumps(events_by_seq, indent=2), encoding="utf-8")
    print(f"Track B Event Ledger: {len(df_ledger)} events across {len(events_by_seq)} sequences saved to {ledger_csv_path} and {ledger_json_path}")

    # Generate Phase 58 SHA-256 Manifest
    manifest_ledger = {
        "PHASE58_CONFIRMATORY_MANIFEST.csv": {
            "path": str(trackA_manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(trackA_manifest_path),
            "total_clips": len(df_trackA)
        },
        "PHASE58_TONE_PROBE_MANIFEST.csv": {
            "path": str(probe_manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(probe_manifest_path),
            "total_clips": len(df_probe)
        },
        "PHASE58_STREAMING_MANIFEST.csv": {
            "path": str(stream_manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(stream_manifest_path),
            "total_sequences": len(df_stream)
        },
        "PHASE58_EVENT_LEDGER.csv": {
            "path": str(ledger_csv_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(ledger_csv_path),
            "total_events": len(df_ledger)
        },
        "PHASE58_EVENT_LEDGER.json": {
            "path": str(ledger_json_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(ledger_json_path),
            "total_sequences_with_events": len(events_by_seq)
        }
    }

    manifest_ledger_path = REPORTS_DIR / "PHASE58_SHA256_MANIFEST.json"
    manifest_ledger_path.write_text(json.dumps(manifest_ledger, indent=2), encoding="utf-8")
    print(f"Saved Phase 58 Manifest Ledger to {manifest_ledger_path}")
    print("\nPhase 58 Confirmatory Corpus generation completed successfully!")


if __name__ == "__main__":
    main()
