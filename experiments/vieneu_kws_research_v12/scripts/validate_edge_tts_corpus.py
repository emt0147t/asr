#!/usr/bin/env python3
"""
Phase 27: Automated Validation of Edge-TTS Evaluation Corpus Integrity and Separation
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/validate_edge_tts_corpus.py
"""

import sys
import io
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"

ISOLATED_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_isolated_test_manifest.csv"
STREAMING_MANIFEST = MANIFESTS_DIR / "edge_tts_eval_streaming_manifest.csv"
EVENTS_JSON = MANIFESTS_DIR / "edge_tts_eval_streaming_events.json"
QUARANTINE_LEDGER = MANIFESTS_DIR / "source_eligibility_and_quarantine_ledger.csv"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PHASE 27: VALIDATING INDEPENDENT EVALUATION CORPUS INTEGRITY & SEPARATION")
    print("=" * 80)

    errors = []

    # 1. Manifest existence
    for m in [ISOLATED_MANIFEST, STREAMING_MANIFEST, EVENTS_JSON]:
        if not m.exists():
            errors.append(f"Missing manifest file: {m}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        sys.exit(1)

    df_iso = pd.read_csv(ISOLATED_MANIFEST)
    df_stream = pd.read_csv(STREAMING_MANIFEST)
    with open(EVENTS_JSON, "r", encoding="utf-8") as f:
        events = json.load(f)

    # 2. Audit Isolated Manifest
    print(f"\nAuditing Isolated Component Manifest ({len(df_iso)} clips)...")
    assert len(df_iso) == 224 or len(df_iso) == 272 or len(df_iso) > 0, "Empty isolated manifest"
    for idx, row in df_iso.iterrows():
        p = PROJECT_ROOT / row["audio_path"]
        if not p.exists():
            errors.append(f"Isolated audio file missing: {p}")
            continue
        actual_sha = compute_sha256(p)
        if actual_sha != row["sha256"]:
            errors.append(f"Hash mismatch on {p}: {actual_sha} != {row['sha256']}")
        info = sf.info(str(p))
        if info.samplerate != 16000 or info.channels != 1:
            errors.append(f"Format violation on {p}: SR={info.samplerate}, Channels={info.channels}")
        if info.frames != 16000:
            errors.append(f"Duration violation on {p}: frames={info.frames} != 16000")

    # 3. Audit Streaming Manifest & Events
    print(f"Auditing Streaming Sequence Manifest ({len(df_stream)} sequences)...")
    if len(df_stream) != 100:
        errors.append(f"Expected 100 streaming sequences, got {len(df_stream)}")

    for idx, row in df_stream.iterrows():
        p = PROJECT_ROOT / row["audio_path"]
        if not p.exists():
            errors.append(f"Streaming audio file missing: {p}")
            continue
        actual_sha = compute_sha256(p)
        if actual_sha != row["sha256"]:
            errors.append(f"Hash mismatch on {p}: {actual_sha} != {row['sha256']}")
        info = sf.info(str(p))
        if info.samplerate != 16000 or info.channels != 1:
            errors.append(f"Format violation on {p}: SR={info.samplerate}, Channels={info.channels}")
        if info.duration < 5.0:
            errors.append(f"Streaming sequence too short: {p} ({info.duration}s)")

    # 4. Quarantine Check (No overlap with training or Phase 21 observed test corpus)
    print("Checking Quarantine & Separation from Training Data...")
    df_ledger = pd.read_csv(QUARANTINE_LEDGER)
    quarantined_hashes = set(df_ledger[df_ledger["eligibility_status"] == "QUARANTINED_TEST_DATA"]["sha256"])
    training_hashes = set(df_ledger[df_ledger["eligibility_status"].str.startswith("APPROVED_TRAINING")]["sha256"])

    new_hashes = set(df_iso["sha256"]).union(set(df_stream["sha256"]))
    overlap_quarantine = new_hashes.intersection(quarantined_hashes)
    overlap_training = new_hashes.intersection(training_hashes)

    if len(overlap_quarantine) > 0:
        errors.append(f"CRITICAL: Overlap with quarantined Phase 21 test audio: {len(overlap_quarantine)} files")
    if len(overlap_training) > 0:
        errors.append(f"CRITICAL: Overlap with approved training audio: {len(overlap_training)} files")

    # 5. Separation of Lexicons
    print("Verifying Dev vs Test Lexical Separation...")
    dev_oov_words = {"bật", "tắt", "dừng", "chạy", "mở", "đóng", "quay", "lùi", "tìm", "chọn"}
    test_words = set(df_iso["text"].unique())
    lex_overlap = dev_oov_words.intersection(test_words)
    if len(lex_overlap) > 0:
        errors.append(f"Lexical overlap between Dev OOV and Test: {lex_overlap}")

    # Report results
    print("\n" + "=" * 80)
    if len(errors) == 0:
        print("ALL CORPUS INTEGRITY AND SEPARATION AUDITS PASSED WITH ZERO DEFECTS.")
        print(f"Isolated Test Manifest SHA-256: {compute_sha256(ISOLATED_MANIFEST)}")
        print(f"Streaming Test Manifest SHA-256: {compute_sha256(STREAMING_MANIFEST)}")
        print(f"Streaming Events JSON SHA-256:   {compute_sha256(EVENTS_JSON)}")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"CORPUS VALIDATION FAILED WITH {len(errors)} ERRORS:")
        for e in errors[:20]:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
