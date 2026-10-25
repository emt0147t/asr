"""
Streaming Multi-Command Evaluation Corpus Generator (Section I).
Constructs realistic sequential speech evaluation streams with:
- Target command permutations (TRAI, PHAI, LEN, XUONG)
- Repeated command pairs (TRAI->TRAI, LEN->LEN, etc.)
- Interspersed OOV confusers ("ghế", test unknown words)
- Calibrated inter-command gaps: standard (350-500ms), stress short (150-250ms), long (800-1200ms)
- Multi-condition acoustic environments: Clean, SNR 15 dB, 10 dB, 5 dB using test-only noise tracks (34..39)
- Precise event timestamp annotations for evaluating streaming/sliding-window KWS pipelines.
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
from noise_mixer import NoiseAssetManager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
EVAL_OUT_DIR = V12_DIR / "audio" / "streaming_eval"
MANIFEST_FILE = V12_DIR / "manifests" / "streaming_eval_corpus_manifest.csv"
NOISE_DIR = PROJECT_ROOT / "data" / "raw" / "noise" / "extracted" / "noise" / "mic-noise"
SPLIT_POLICY_FILE = V12_DIR / "configs" / "split_policy.json"
STAGING_OOV_DIR = V12_DIR / "audio" / "staging_oov"

# Reference base datasets
DATASET_1000_MANIFEST = PROJECT_ROOT / "experiments" / "synthetic_kws_dataset_vieneu_1000" / "manifest.csv"
DATASET_6CLASS_MANIFEST = PROJECT_ROOT / "experiments" / "synthetic_kws_dataset_vieneu_6class_v2" / "manifest.csv"

SEED_BASE = 42000


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def trim_silence(audio: np.ndarray, threshold: float = 0.02, pad_samples: int = 800) -> np.ndarray:
    """Trims leading/trailing silence from 1.0s clip leaving core utterance with small padding."""
    energy = np.abs(audio)
    active = np.where(energy > threshold)[0]
    if len(active) == 0:
        return audio
    start = max(0, active[0] - pad_samples)
    end = min(len(audio), active[-1] + pad_samples)
    return audio[start:end]


def mix_noise_stream(clean_stream: np.ndarray, noise_mgr: NoiseAssetManager, track_idx: int, target_snr_db: float, seed: int) -> np.ndarray:
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

    s_rms = np.sqrt(np.mean(clean_stream**2) + 1e-12)
    n_rms = np.sqrt(np.mean(noise_slice**2) + 1e-12)
    desired_n_rms = s_rms / (10.0 ** (target_snr_db / 20.0))
    scale = desired_n_rms / (n_rms + 1e-12)
    scaled_noise = noise_slice * scale

    mixed = clean_stream + scaled_noise
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
    return mixed.astype(np.float32)


def generate_streaming_eval_corpus():
    print("=" * 80)
    print("  GENERATING VIETNAMESE KWS STREAMING MULTI-COMMAND EVALUATION CORPUS")
    print("  Evaluating sequential live detection under varied gap and noise conditions")
    print("=" * 80)

    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_POLICY_FILE, "r", encoding="utf-8") as f:
        split_policy = json.load(f)

    noise_mgr = NoiseAssetManager(NOISE_DIR)
    test_noise_indices = split_policy["noise_asset_allocation"]["test_track_indices"]

    # Gather test-only base audio utterances
    df1 = pd.read_csv(DATASET_1000_MANIFEST)
    df2 = pd.read_csv(DATASET_6CLASS_MANIFEST)

    df1_test = df1[df1["split"] == "test"]
    df2_test = df2[df2["split"] == "test"]

    test_cmd_clips: Dict[str, List[Path]] = {
        "TRAI": [], "PHAI": [], "LEN": [], "XUONG": []
    }
    for cmd in test_cmd_clips:
        sub1 = df1_test[df1_test["class_label"] == cmd]
        for _, r in sub1.iterrows():
            test_cmd_clips[cmd].append(PROJECT_ROOT / r["kws_wav_path"])
        sub2 = df2_test[df2_test["class_label"] == cmd]
        for _, r in sub2.iterrows():
            test_cmd_clips[cmd].append(PROJECT_ROOT / r["kws_wav_path"])

    # Test confuser clips ("ghe" staging and 6class_v2 test UNKNOWN)
    test_confusers: List[Tuple[str, Path]] = []
    ghe_files = list(STAGING_OOV_DIR.glob("*_test_ghe_*.wav"))
    for gf in ghe_files:
        test_confusers.append(("ghe", gf))

    sub2_unk = df2_test[df2_test["class_label"] == "UNKNOWN"]
    for _, r in sub2_unk.iterrows():
        test_confusers.append((str(r["text"]).strip().lower(), PROJECT_ROOT / r["kws_wav_path"]))

    print(f"Loaded test resources:")
    for cmd, clips in test_cmd_clips.items():
        print(f"  {cmd}: {len(clips)} test clips")
    print(f"  Test confusers: {len(test_confusers)} clips")
    print(f"  Test noise tracks: {len(test_noise_indices)} tracks")

    # Sequence definitions: 40 distinct evaluation streams
    # 10 Permutations, 8 Repeated commands, 12 Confuser-interspersed, 10 Stress gap sequences
    sequences_spec = []

    # 1. 10 Permutations (4 distinct commands in each)
    perm_orders = [
        ["LEN", "XUONG", "TRAI", "PHAI"],
        ["TRAI", "PHAI", "LEN", "XUONG"],
        ["PHAI", "TRAI", "XUONG", "LEN"],
        ["XUONG", "LEN", "PHAI", "TRAI"],
        ["LEN", "TRAI", "PHAI", "XUONG"],
        ["PHAI", "XUONG", "LEN", "TRAI"],
        ["TRAI", "LEN", "XUONG", "PHAI"],
        ["XUONG", "PHAI", "TRAI", "LEN"],
        ["LEN", "PHAI", "TRAI", "XUONG"],
        ["TRAI", "XUONG", "PHAI", "LEN"]
    ]
    for order in perm_orders:
        sequences_spec.append({
            "type": "permutation",
            "tokens": order,
            "gap_ms": 400
        })

    # 2. 8 Repeated commands
    rep_orders = [
        ["TRAI", "TRAI", "PHAI", "PHAI"],
        ["LEN", "LEN", "XUONG", "XUONG"],
        ["PHAI", "PHAI", "PHAI"],
        ["LEN", "LEN", "LEN"],
        ["XUONG", "XUONG", "TRAI", "TRAI"],
        ["TRAI", "TRAI", "TRAI"],
        ["XUONG", "XUONG", "XUONG"],
        ["PHAI", "PHAI", "LEN", "LEN"]
    ]
    for order in rep_orders:
        sequences_spec.append({
            "type": "repetition",
            "tokens": order,
            "gap_ms": 450
        })

    # 3. 12 Confuser-interspersed sequences
    conf_orders = [
        ["TRAI", "ghe", "PHAI"],
        ["ghe", "LEN", "XUONG"],
        ["LEN", "bep", "XUONG", "ghe"],
        ["PHAI", "cua", "TRAI"],
        ["may", "TRAI", "PHAI", "den"],
        ["LEN", "quat", "XUONG"],
        ["ghe", "TRAI", "ghe", "PHAI"],
        ["bat", "LEN", "tat", "XUONG"],
        ["TRAI", "PHAI", "ghe"],
        ["ghe", "LEN", "TRAI", "XUONG"],
        ["nhac", "LEN", "dung", "XUONG"],
        ["PHAI", "ghe", "TRAI", "ghe"]
    ]
    for order in conf_orders:
        sequences_spec.append({
            "type": "confuser_interspersed",
            "tokens": order,
            "gap_ms": 400
        })

    # 4. 10 Stress gap sequences (short 180ms or long 900ms)
    stress_specs = [
        {"tokens": ["TRAI", "PHAI", "LEN"], "gap_ms": 180, "type": "stress_short_gap"},
        {"tokens": ["LEN", "XUONG", "PHAI"], "gap_ms": 200, "type": "stress_short_gap"},
        {"tokens": ["XUONG", "TRAI", "PHAI"], "gap_ms": 160, "type": "stress_short_gap"},
        {"tokens": ["PHAI", "LEN", "XUONG"], "gap_ms": 220, "type": "stress_short_gap"},
        {"tokens": ["TRAI", "TRAI", "LEN"], "gap_ms": 180, "type": "stress_short_gap"},
        {"tokens": ["LEN", "XUONG", "TRAI"], "gap_ms": 900, "type": "stress_long_gap"},
        {"tokens": ["TRAI", "PHAI", "XUONG"], "gap_ms": 1100, "type": "stress_long_gap"},
        {"tokens": ["PHAI", "LEN", "TRAI"], "gap_ms": 850, "type": "stress_long_gap"},
        {"tokens": ["XUONG", "PHAI", "LEN"], "gap_ms": 1000, "type": "stress_long_gap"},
        {"tokens": ["TRAI", "LEN", "PHAI", "XUONG"], "gap_ms": 950, "type": "stress_long_gap"}
    ]
    sequences_spec.extend(stress_specs)

    assert len(sequences_spec) == 40

    # Distribute acoustic conditions across the 40 sequences:
    # 10 clean, 10 snr15, 10 snr10, 10 snr05
    conditions = ["clean"] * 10 + ["snr15"] * 10 + ["snr10"] * 10 + ["snr05"] * 10
    # Shuffle or deterministically interleave conditions
    np.random.RandomState(SEED_BASE).shuffle(conditions)

    manifest_rows = []

    print(f"\nBuilding 40 multi-command evaluation streams...")

    for idx, (spec, cond) in enumerate(zip(sequences_spec, conditions)):
        seq_id = f"seq_{idx+1:04d}"
        rng = np.random.RandomState(SEED_BASE + idx * 100)
        
        # Build stream components: leading silence + (token + gap) + trailing silence
        stream_parts = []
        events = []
        current_sample_idx = 0

        # Leading silence (300-500ms)
        lead_samples = rng.randint(4800, 8000)
        stream_parts.append(np.zeros(lead_samples, dtype=np.float32))
        current_sample_idx += lead_samples

        for token in spec["tokens"]:
            is_target = token in ["TRAI", "PHAI", "LEN", "XUONG"]
            if is_target:
                clip_candidates = test_cmd_clips[token]
                clip_path = clip_candidates[rng.randint(0, len(clip_candidates))]
            else:
                # Find matching or fallback confuser
                matching = [c for c in test_confusers if c[0] == token]
                if matching:
                    clip_path = matching[rng.randint(0, len(matching))][1]
                else:
                    clip_path = test_confusers[rng.randint(0, len(test_confusers))][1]

            audio, sr = sf.read(str(clip_path), dtype="float32")
            assert sr == 16000
            # Trim silence to retain natural core duration (~400-700ms)
            core_audio = trim_silence(audio, threshold=0.015, pad_samples=600)
            
            start_sec = round(current_sample_idx / 16000.0, 3)
            end_sec = round((current_sample_idx + len(core_audio)) / 16000.0, 3)

            events.append({
                "label": token,
                "is_target": is_target,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(end_sec - start_sec, 3)
            })

            stream_parts.append(core_audio)
            current_sample_idx += len(core_audio)

            # Inter-command gap
            gap_jitter = rng.randint(-30, 30)
            actual_gap_ms = max(100, spec["gap_ms"] + gap_jitter)
            gap_samples = int(actual_gap_ms * 16)
            stream_parts.append(np.zeros(gap_samples, dtype=np.float32))
            current_sample_idx += gap_samples

        # Trailing silence (500ms)
        trail_samples = 8000
        stream_parts.append(np.zeros(trail_samples, dtype=np.float32))
        current_sample_idx += trail_samples

        # Concatenate clean stream
        clean_stream = np.concatenate(stream_parts)

        # Apply acoustic condition
        noise_track_id = None
        if cond == "clean":
            final_audio = clean_stream
        else:
            snr_val = float(cond.replace("snr", ""))
            track_idx = test_noise_indices[idx % len(test_noise_indices)]
            noise_track_id = noise_mgr.get_track_path(track_idx).name
            final_audio = mix_noise_stream(clean_stream, noise_mgr, track_idx, snr_val, seed=SEED_BASE + idx * 7)

        # Write WAV file
        out_wav_path = EVAL_OUT_DIR / f"{seq_id}_{cond}.wav"
        sf.write(str(out_wav_path), final_audio, 16000)

        sha = compute_sha256(out_wav_path)
        total_dur_sec = round(len(final_audio) / 16000.0, 3)
        num_targets = sum(1 for e in events if e["is_target"])

        manifest_rows.append({
            "sequence_id": seq_id,
            "condition": cond,
            "sequence_type": spec["type"],
            "noise_asset_id": noise_track_id,
            "duration_seconds": total_dur_sec,
            "num_total_events": len(events),
            "num_target_events": num_targets,
            "events_json": json.dumps(events, ensure_ascii=False),
            "audio_path": str(out_wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha
        })

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(MANIFEST_FILE, index=False)
    print(f"\nSaved {len(manifest_rows)} evaluation sequences to: {EVAL_OUT_DIR}")
    print(f"Manifest written to: {MANIFEST_FILE}")
    print(f"Summary by condition:\n{df_manifest['condition'].value_counts()}")
    print(f"Summary by type:\n{df_manifest['sequence_type'].value_counts()}")
    print("=" * 80)


if __name__ == "__main__":
    generate_streaming_eval_corpus()
