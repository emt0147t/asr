#!/usr/bin/env python3
"""
Phase 26 Candidate Freeze Script
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/freeze_phase26_candidate.py

Freezes the selected Phase 26 candidate checkpoint and metadata into an immutable
candidate directory without overwriting v11 or E1 artifacts.
"""

import sys
import io
import shutil
import json
import hashlib
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase25"
CANDIDATES_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates"
TARGET_CANDIDATE_DIR = CANDIDATES_DIR / "phase26_candidate_d0_h1_seed44"

SRC_CKPT = TRAIN_DIR / "D0_H1_seed44" / "best_model.pt"
SRC_SUMMARY = TRAIN_DIR / "D0_H1_seed44" / "run_summary.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PHASE 26: FREEZING SELECTED CANDIDATE CHECKPOINT")
    print("=" * 80)

    TARGET_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify source
    assert SRC_CKPT.exists(), f"Source checkpoint not found: {SRC_CKPT}"
    src_sha = compute_sha256(SRC_CKPT)
    print(f"Source Checkpoint: {SRC_CKPT.relative_to(PROJECT_ROOT)}")
    print(f"Source SHA-256:    {src_sha}")

    # 2. Copy checkpoint
    dst_ckpt = TARGET_CANDIDATE_DIR / "candidate_checkpoint.pt"
    shutil.copy2(SRC_CKPT, dst_ckpt)
    dst_sha = compute_sha256(dst_ckpt)
    assert src_sha == dst_sha, "Copy hash mismatch!"
    print(f"Frozen Checkpoint: {dst_ckpt.relative_to(PROJECT_ROOT)}")
    print(f"Frozen SHA-256:    {dst_sha}")

    # 3. Create frozen metadata
    with open(SRC_SUMMARY, "r", encoding="utf-8") as f:
        src_meta = json.load(f)

    meta = {
        "candidate_id": "phase26_candidate_d0_h1_seed44",
        "phase": "Phase 26 — Factorial Results Audit and Candidate Freeze",
        "date_frozen": "2026-10-25T01:00:00+07:00",
        "condition": "D0_H1",
        "seed": 44,
        "selection_type": "post_hoc_on_validation",
        "selection_rationale": "Leading checkpoint from winning condition D0_H1 (highest mean macro-F1: 93.02% ± 0.17%, lowest inter-seed variance σ=0.0017, superior OOV false alarm control: 26.95%, high keyword recall: 97.5%, and single-run macro-F1: 93.22%).",
        "architecture": "CorrectedVietnameseKWSNet",
        "trainable_parameters": 138469,
        "decision_threshold_theta_cmd": 0.20,
        "validation_metrics": {
            "val_macro_f1": src_meta["best_val_macro_f1"],
            "val_accuracy": src_meta["best_val_accuracy"],
            "val_loss": src_meta["best_val_loss"],
            "best_epoch": src_meta["best_epoch"],
            "total_epochs": src_meta["total_epochs"],
            "per_class": src_meta["best_per_class"]
        },
        "provenance": {
            "training_manifest": "experiments/vieneu_kws_research_v12/manifests/D0_train_v1.csv",
            "training_manifest_sha256": src_meta["train_manifest_sha256"],
            "validation_manifest": "experiments/vieneu_kws_research_v12/manifests/validation_v1.csv",
            "validation_manifest_sha256": src_meta["val_manifest_sha256"],
            "original_checkpoint_path": str(SRC_CKPT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "checkpoint_sha256": dst_sha
        },
        "governance_labels": [
            "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED",
            "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION",
            "NOT_VALIDATED_PENDING_DATA"
        ],
        "status": "FROZEN_CANDIDATE_FOR_INDEPENDENT_EVALUATION"
    }

    dst_meta = TARGET_CANDIDATE_DIR / "candidate_metadata.json"
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 4. Checksums file
    checksum_file = TARGET_CANDIDATE_DIR / "checksums.sha256"
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write(f"{dst_sha}  candidate_checkpoint.pt\n")
        meta_sha = compute_sha256(dst_meta)
        f.write(f"{meta_sha}  candidate_metadata.json\n")

    print(f"\n[OK] Candidate frozen successfully in: {TARGET_CANDIDATE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
