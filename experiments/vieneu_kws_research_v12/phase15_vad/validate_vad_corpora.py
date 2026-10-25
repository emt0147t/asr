"""
Corpus Validation and Integrity Audit (Phase 15).
Verifies:
  1. Sequence counts and gap balance (10 sequences per gap across 6 gaps = 60).
  2. Acoustic condition distribution (Clean, SNR 20, 15, 10, 5 dB).
  3. Physical audio file existence, sample rate (16 kHz), mono, non-zero duration.
  4. Complete split isolation:
     - Voice-disjoint between Dev and Test.
     - OOV-disjoint between Dev and Test.
     - Noise asset-disjoint between Dev and Test.
  5. Manifest cryptographic hashes.
"""

import sys
import io
import json
import hashlib
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
MANIFESTS_DIR = V12_DIR / "manifests"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_corpus(split_name: str, manifest_path: Path):
    print(f"\nAuditing {split_name.upper()} Corpus: {manifest_path.name}")
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
    df = pd.read_csv(manifest_path)

    # 1. Sequence count
    assert len(df) == 60, f"Expected 60 sequences, got {len(df)}"
    print(f"  [PASS] Sequence count: 60/60")

    # 2. Gap balance
    gap_counts = df["nominal_gap_ms"].value_counts().to_dict()
    for gap in [100, 150, 200, 250, 300, 400]:
        assert gap_counts.get(gap, 0) == 10, f"Gap {gap}ms has {gap_counts.get(gap, 0)} sequences (expected 10)"
    print(f"  [PASS] Gap balance: 10 sequences per gap across [100, 150, 200, 250, 300, 400] ms")

    # 3. Condition balance
    cond_counts = df["condition"].value_counts().to_dict()
    for cond in ["clean", "snr20", "snr15", "snr10", "snr05"]:
        assert cond_counts.get(cond, 0) == 12, f"Condition {cond} has {cond_counts.get(cond, 0)} (expected 12)"
    print(f"  [PASS] Condition balance: 12 sequences per condition across [clean, snr20, snr15, snr10, snr05]")

    # 4. Check physical files on disk
    voices_found = set()
    oov_found = set()
    noise_found = set()
    total_events = 0
    total_targets = 0

    for idx, row in df.iterrows():
        p = PROJECT_ROOT / row["audio_path"]
        assert p.exists(), f"Audio file missing: {p}"
        audio, sr = sf.read(str(p), dtype="float32")
        assert sr == 16000, f"Sample rate mismatch: {sr}"
        assert audio.ndim == 1, f"Audio not mono: {audio.ndim}"
        assert len(audio) > 0, "Empty audio file"

        # Verify hash
        actual_hash = compute_sha256(p)
        assert actual_hash == row["sha256"], f"Hash mismatch for {p.name}"

        # Parse events
        events = json.loads(row["events_json"])
        total_events += len(events)
        for ev in events:
            if ev["is_target"]:
                total_targets += 1
            else:
                oov_found.add(ev["text"])
            voices_found.add(ev["voice"])

        if row["noise_asset_id"] != "none_clean":
            noise_found.add(row["noise_asset_id"])

    print(f"  [PASS] Physical audio files: 60/60 verified with SHA-256 byte-for-byte")
    print(f"  [PASS] Total events: {total_events} (Target commands: {total_targets}, OOV: {total_events - total_targets})")
    print(f"  [INFO] Unique voices: {len(voices_found)}: {sorted(list(voices_found))}")
    print(f"  [INFO] Unique OOV words: {len(oov_found)}: {sorted(list(oov_found))}")
    print(f"  [INFO] Unique noise tracks: {len(noise_found)}")

    return {
        "df": df,
        "voices": voices_found,
        "oov": oov_found,
        "noise": noise_found,
        "total_events": total_events,
        "total_targets": total_targets
    }


def main():
    print("=" * 80)
    print("  PHASE 15: VAD CORPORA INTEGRITY & LEAKAGE AUDIT")
    print("=" * 80)

    dev_manifest = MANIFESTS_DIR / "vad_dev_corpus_manifest.csv"
    test_manifest = MANIFESTS_DIR / "vad_test_corpus_manifest.csv"

    dev_res = audit_corpus("dev", dev_manifest)
    test_res = audit_corpus("test", test_manifest)

    # Cross-corpus leakage audit
    print("\n--- Cross-Split Independence Audit ---")
    voice_leakage = dev_res["voices"].intersection(test_res["voices"])
    assert len(voice_leakage) == 0, f"Voice leakage detected: {voice_leakage}"
    print(f"  [PASS] Voice isolation: Dev voices {dev_res['voices']} vs Test voices {test_res['voices']} (Intersection = 0)")

    oov_leakage = dev_res["oov"].intersection(test_res["oov"])
    assert len(oov_leakage) == 0, f"OOV leakage detected: {oov_leakage}"
    print(f"  [PASS] OOV isolation: Dev OOV {dev_res['oov']} vs Test OOV {test_res['oov']} (Intersection = 0)")

    noise_leakage = dev_res["noise"].intersection(test_res["noise"])
    assert len(noise_leakage) == 0, f"Noise track leakage detected: {noise_leakage}"
    print(f"  [PASS] Noise track isolation: Dev noise {len(dev_res['noise'])} vs Test noise {len(test_res['noise'])} (Intersection = 0)")

    print("\n" + "=" * 80)
    print("  CORPUS AUDIT: 100% PASSED — COMPLETE SPLIT DISJOINTNESS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
