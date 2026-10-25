"""
Controlled VAD Evaluation Corpus Generator (Phase 15).
Builds two rigorously isolated sequential speech corpora:
  1. vad_dev_corpus (60 sequences) from Val split (voices: Mai Anh, Ngọc Trân, Phạm Tuyên, Thùy Dung; Val OOV; Noise 28..33)
  2. vad_test_corpus (60 sequences) from Test split (voices: Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí; Test OOV; Noise 34..39)

Calibrated Inter-Utterance Gaps:
  - 100 ms, 150 ms, 200 ms, 250 ms, 300 ms, 400 ms (10 sequences per gap per corpus)

Acoustic Environments per Gap (10 sequences):
  - 2 Clean, 2 SNR 20 dB, 2 SNR 15 dB, 2 SNR 10 dB, 2 SNR 5 dB

Preserves full provenance, component utterance IDs, source hashes, and exact reference event boundaries.
"""

import sys
import io
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
VAD_PHASE_DIR = V12_DIR / "phase15_vad"
SCRIPTS_DIR = V12_DIR / "scripts"
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"
MANIFESTS_DIR = V12_DIR / "manifests"

sys.path.insert(0, str(SCRIPTS_DIR))
from noise_mixer import NoiseAssetManager

TARGET_SR = 16000
SEED_BASE_DEV = 42100
SEED_BASE_TEST = 42200

NOMINAL_GAPS = [100, 150, 200, 250, 300, 400]
CONDITIONS = [
    ("clean", None),
    ("clean", None),
    ("snr20", 20.0),
    ("snr20", 20.0),
    ("snr15", 15.0),
    ("snr15", 15.0),
    ("snr10", 10.0),
    ("snr10", 10.0),
    ("snr05", 5.0),
    ("snr05", 5.0),
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trim_to_speech_envelope(
    audio: np.ndarray,
    sr: int = 16000,
    threshold: float = 0.015,
    pad_samples: int = 160
) -> Tuple[np.ndarray, int, int]:
    """
    Auditable trimming procedure:
    Finds acoustic active region where abs(sample) > threshold.
    Returns trimmed audio and original sample start/end offsets.
    """
    energy = np.abs(audio)
    active = np.where(energy > threshold)[0]
    if len(active) == 0:
        return audio, 0, len(audio)
    start = max(0, active[0] - pad_samples)
    end = min(len(audio), active[-1] + pad_samples)
    return audio[start:end], start, end


def mix_noise_stream(
    clean_stream: np.ndarray,
    noise_mgr: NoiseAssetManager,
    track_idx: int,
    target_snr_db: float,
    seed: int
) -> Tuple[np.ndarray, float]:
    """Mixes noise slice into clean sequential stream at target SNR."""
    noise_full = noise_mgr.load_noise_track_16k(track_idx)
    needed_len = len(clean_stream)
    rng = np.random.RandomState(seed)

    if len(noise_full) > needed_len:
        start_idx = rng.randint(0, len(noise_full) - needed_len)
        noise_slice = noise_full[start_idx : start_idx + needed_len]
    else:
        repeats = int(np.ceil(needed_len / len(noise_full)))
        tiled = np.tile(noise_full, repeats)
        noise_slice = tiled[:needed_len]

    s_rms = np.sqrt(np.mean(clean_stream ** 2) + 1e-12)
    n_rms = np.sqrt(np.mean(noise_slice ** 2) + 1e-12)
    desired_n_rms = s_rms / (10.0 ** (target_snr_db / 20.0))
    scale = desired_n_rms / (n_rms + 1e-12)
    scaled_noise = noise_slice * scale

    mixed = clean_stream + scaled_noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    # Calculate actual observed SNR
    actual_s_pow = np.mean(clean_stream ** 2) + 1e-12
    actual_n_pow = np.mean((scaled_noise * (0.95 / peak if peak > 0.95 else 1.0)) ** 2) + 1e-12
    observed_snr = 10.0 * np.log10(actual_s_pow / actual_n_pow)

    return mixed.astype(np.float32), float(observed_snr)


def generate_single_corpus(
    corpus_split: str,
    manifest_csv: Path,
    out_audio_dir: Path,
    out_manifest_csv: Path,
    seed_base: int,
    noise_track_indices: List[int],
    noise_mgr: NoiseAssetManager
) -> pd.DataFrame:
    print(f"\n--- Generating {corpus_split.upper()} VAD Corpus (60 sequences) ---")
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    df_source = pd.read_csv(manifest_csv)
    # Filter clean speech clips
    clean_speech = df_source[df_source["source_type"].str.contains("clean", na=False)].copy()

    # Organize clean clips by label
    command_clips: Dict[str, List[Dict[str, Any]]] = {
        "TRAI": [], "PHAI": [], "LEN": [], "XUONG": []
    }
    oov_clips: List[Dict[str, Any]] = []

    for _, row in clean_speech.iterrows():
        item = {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "audio_path": PROJECT_ROOT / row["audio_path"],
            "voice_or_speaker_id": row.get("voice_or_speaker_id", "unknown"),
            "text": row.get("text", row["label"]),
            "sha256": row["sha256"]
        }
        if row["label"] in command_clips:
            command_clips[row["label"]].append(item)
        elif row["label"] == "REJECT":
            oov_clips.append(item)

    print(f"Loaded source pool for {corpus_split}:")
    for cmd in command_clips:
        print(f"  {cmd}: {len(command_clips[cmd])} clean clips")
    print(f"  OOV: {len(oov_clips)} clean clips")

    # Sequence templates (10 distinct templates repeated for each gap)
    # Templates cover: 3 permutations, 2 repetitions, 3 confuser-interspersed, 2 short-command bursts
    templates = [
        {"type": "permutation", "tokens": ["TRAI", "PHAI", "LEN", "XUONG"]},
        {"type": "permutation", "tokens": ["LEN", "XUONG", "TRAI", "PHAI"]},
        {"type": "permutation", "tokens": ["PHAI", "TRAI", "XUONG", "LEN"]},
        {"type": "repetition", "tokens": ["TRAI", "TRAI", "PHAI"]},
        {"type": "repetition", "tokens": ["LEN", "LEN", "XUONG", "XUONG"]},
        {"type": "confuser_interspersed", "tokens": ["TRAI", "OOV", "PHAI"]},
        {"type": "confuser_interspersed", "tokens": ["OOV", "LEN", "XUONG"]},
        {"type": "confuser_interspersed", "tokens": ["LEN", "OOV", "XUONG", "OOV"]},
        {"type": "command_burst", "tokens": ["XUONG", "TRAI", "LEN"]},
        {"type": "command_burst", "tokens": ["PHAI", "LEN", "XUONG", "TRAI"]}
    ]

    sequences_records = []
    seq_counter = 1

    # Indices for round-robin clip selection
    cmd_indices = {cmd: 0 for cmd in command_clips}
    oov_idx = 0

    lead_in_samples = int(0.500 * TARGET_SR)   # 500ms initial silence
    lead_out_samples = int(0.500 * TARGET_SR)  # 500ms trailing silence

    for gap_ms in NOMINAL_GAPS:
        for cond_idx, (cond_name, snr_db) in enumerate(CONDITIONS):
            seq_id = f"vad_{corpus_split}_seq_{seq_counter:04d}"
            tmpl = templates[cond_idx]
            seq_type = tmpl["type"]
            tokens = tmpl["tokens"]

            seq_seed = seed_base + seq_counter * 17
            rng = np.random.RandomState(seq_seed)

            # Build sequence audio
            audio_segments: List[np.ndarray] = [np.zeros(lead_in_samples, dtype=np.float32)]
            event_records: List[Dict[str, Any]] = []
            component_records: List[Dict[str, Any]] = []

            current_sample_pos = lead_in_samples
            nominal_gap_samples = int((gap_ms / 1000.0) * TARGET_SR)
            effective_gaps_ms = []

            last_speech_end = lead_in_samples

            for token_pos, token in enumerate(tokens):
                # Pick source clip
                if token in command_clips:
                    pool = command_clips[token]
                    c_idx = cmd_indices[token] % len(pool)
                    src_item = pool[c_idx]
                    cmd_indices[token] += 1
                    is_target = True
                    label = token
                else:  # OOV
                    src_item = oov_clips[oov_idx % len(oov_clips)]
                    oov_idx += 1
                    is_target = False
                    label = "REJECT"

                # Load raw clip
                raw_audio, sr = sf.read(str(src_item["audio_path"]), dtype="float32")
                assert sr == TARGET_SR

                # Auditable envelope trimming
                trimmed_audio, trim_start, trim_end = trim_to_speech_envelope(raw_audio, sr, threshold=0.015, pad_samples=160)
                trimmed_len = len(trimmed_audio)

                # Record effective gap if not first token
                if token_pos > 0:
                    eff_gap = ((current_sample_pos - last_speech_end) / TARGET_SR) * 1000.0
                    effective_gaps_ms.append(eff_gap)

                event_start_sec = current_sample_pos / TARGET_SR
                event_end_sec = (current_sample_pos + trimmed_len) / TARGET_SR
                event_dur_sec = trimmed_len / TARGET_SR

                event_records.append({
                    "event_index": token_pos,
                    "label": label,
                    "is_target": is_target,
                    "text": src_item["text"],
                    "start_sec": round(event_start_sec, 4),
                    "end_sec": round(event_end_sec, 4),
                    "duration_sec": round(event_dur_sec, 4),
                    "voice": src_item["voice_or_speaker_id"],
                    "source_sample_id": src_item["sample_id"]
                })

                component_records.append({
                    "sample_id": src_item["sample_id"],
                    "audio_path": str(src_item["audio_path"]),
                    "sha256": src_item["sha256"],
                    "trim_start_sample": int(trim_start),
                    "trim_end_sample": int(trim_end)
                })

                audio_segments.append(trimmed_audio)
                current_sample_pos += trimmed_len
                last_speech_end = current_sample_pos

                # Insert gap if not last token
                if token_pos < len(tokens) - 1:
                    gap_silence = np.zeros(nominal_gap_samples, dtype=np.float32)
                    audio_segments.append(gap_silence)
                    current_sample_pos += nominal_gap_samples

            # Trailing lead-out silence
            audio_segments.append(np.zeros(lead_out_samples, dtype=np.float32))
            current_sample_pos += lead_out_samples

            clean_stream = np.concatenate(audio_segments)
            total_duration_sec = len(clean_stream) / TARGET_SR

            # Noise mixing
            noise_track_id = None
            observed_snr = None
            if snr_db is not None:
                track_choice_idx = noise_track_indices[(seq_counter + cond_idx) % len(noise_track_indices)]
                noise_path = noise_mgr.get_track_path(track_choice_idx)
                noise_track_id = noise_path.name
                mixed_stream, observed_snr = mix_noise_stream(
                    clean_stream, noise_mgr, track_choice_idx, snr_db, seed=seq_seed
                )
                final_audio = mixed_stream
            else:
                final_audio = clean_stream.copy()
                observed_snr = float("inf")
                noise_track_id = "none_clean"

            # Save WAV
            out_wav_path = out_audio_dir / f"{seq_id}_{cond_name}_gap{gap_ms}ms.wav"
            sf.write(str(out_wav_path), final_audio, TARGET_SR, subtype="PCM_16")
            seq_sha256 = compute_sha256(out_wav_path)

            mean_eff_gap = float(np.mean(effective_gaps_ms)) if effective_gaps_ms else float(gap_ms)

            sequences_records.append({
                "sequence_id": seq_id,
                "corpus_split": corpus_split,
                "nominal_gap_ms": gap_ms,
                "effective_acoustic_gap_ms": round(mean_eff_gap, 2),
                "condition": cond_name,
                "target_snr_db": snr_db if snr_db is not None else "clean",
                "observed_snr_db": round(observed_snr, 2) if observed_snr != float("inf") else "inf",
                "noise_asset_id": noise_track_id,
                "sequence_type": seq_type,
                "num_components": len(tokens),
                "num_target_events": sum(1 for e in event_records if e["is_target"]),
                "tokens": "->".join(tokens),
                "duration_seconds": round(total_duration_sec, 4),
                "audio_path": str(out_wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": seq_sha256,
                "generation_seed": seq_seed,
                "events_json": json.dumps(event_records, ensure_ascii=False),
                "components_json": json.dumps(component_records, ensure_ascii=False)
            })

            seq_counter += 1

    df_out = pd.DataFrame(sequences_records)
    df_out.to_csv(out_manifest_csv, index=False, encoding="utf-8")
    print(f"Generated {len(df_out)} sequences for {corpus_split}.")
    print(f"Saved manifest: {out_manifest_csv}")
    return df_out


def main():
    print("=" * 80)
    print("  PHASE 15: GENERATING CONTROLLED VAD EVALUATION CORPORA")
    print("  60 Dev Sequences + 60 Locked Test Sequences")
    print("=" * 80)

    noise_mgr = NoiseAssetManager(NOISE_DIR)

    # Split policy allocation
    val_noise_tracks = [28, 29, 30, 31, 32, 33]
    test_noise_tracks = [34, 35, 36, 37, 38, 39]

    dev_manifest_src = MANIFESTS_DIR / "val_manifest.csv"
    test_manifest_src = MANIFESTS_DIR / "test_manifest.csv"

    dev_out_dir = V12_DIR / "audio" / "vad_dev_corpus"
    test_out_dir = V12_DIR / "audio" / "vad_test_corpus"

    dev_manifest_out = MANIFESTS_DIR / "vad_dev_corpus_manifest.csv"
    test_manifest_out = MANIFESTS_DIR / "vad_test_corpus_manifest.csv"

    df_dev = generate_single_corpus(
        "dev", dev_manifest_src, dev_out_dir, dev_manifest_out, SEED_BASE_DEV, val_noise_tracks, noise_mgr
    )

    df_test = generate_single_corpus(
        "test", test_manifest_src, test_out_dir, test_manifest_out, SEED_BASE_TEST, test_noise_tracks, noise_mgr
    )

    # Manifest verification and SHA-256 digest
    dev_hash = compute_sha256(dev_manifest_out)
    test_hash = compute_sha256(test_manifest_out)

    print("\n" + "=" * 80)
    print("  CORPUS GENERATION COMPLETE & LOCKED")
    print(f"  vad_dev_corpus:  {len(df_dev)} sequences | Manifest SHA-256: {dev_hash}")
    print(f"  vad_test_corpus: {len(df_test)} sequences | Manifest SHA-256: {test_hash}")
    print("=" * 80)


if __name__ == "__main__":
    main()
