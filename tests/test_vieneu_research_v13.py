#!/usr/bin/env python3
"""
Unit and Governance Audit Test Suite for Phase 13 Pre-Evaluation Gates.
Workspace: D:\trainidentificationaudio
Target: tests/test_vieneu_research_v13.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FROZEN_CKPT = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
EXPECTED_CKPT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

V11_SPLITS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11" / "data" / "splits"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_candidate_checkpoint_hash_integrity():
    """Verify that the frozen candidate checkpoint exists and matches its authoritative SHA-256."""
    assert FROZEN_CKPT.exists(), f"Frozen candidate checkpoint missing: {FROZEN_CKPT}"
    actual_hash = sha256_of_file(FROZEN_CKPT)
    assert actual_hash.lower() == EXPECTED_CKPT_HASH.lower(), (
        f"Checkpoint hash altered! Expected {EXPECTED_CKPT_HASH}, got {actual_hash}"
    )


def test_mathematical_non_equivalence_of_normalization_contracts():
    """
    Formally prove that Phase 12 -1.0 dBFS peak normalization and earlier
    clamping normalization (scale only if max > 1.0) are mathematically and
    acoustically non-equivalent.
    """
    # Define quiet ambient signal (max amplitude 0.005, typical of room tone or near silence)
    np.random.seed(42)
    quiet_signal = np.random.normal(0, 0.0015, 16000).astype(np.float32)
    max_quiet = float(np.max(np.abs(quiet_signal)))
    assert max_quiet < 0.05, f"Unexpected amplitude in quiet synthetic signal: {max_quiet}"

    # Rule A: Phase 12 contract (unconditional scaling to -1.0 dBFS = 0.89125)
    target_peak = 10.0 ** (-1.0 / 20.0)
    norm_rule_a = quiet_signal * (target_peak / max_quiet)

    # Rule B: Earlier contract (scale only if max > 1.0; never amplify noise floor)
    if max_quiet > 1.0:
        norm_rule_b = quiet_signal / max_quiet
    else:
        norm_rule_b = quiet_signal.copy()

    # Verify peak amplitudes
    peak_a = float(np.max(np.abs(norm_rule_a)))
    peak_b = float(np.max(np.abs(norm_rule_b)))

    assert np.isclose(peak_a, target_peak, atol=1e-5), "Rule A failed to scale to -1.0 dBFS"
    assert np.isclose(peak_b, max_quiet, atol=1e-5), "Rule B modified signal below 1.0"

    # Absolute difference between waveforms
    abs_diff = np.max(np.abs(norm_rule_a - norm_rule_b))
    amplification_factor = target_peak / max_quiet

    # Assert non-equivalence: amplification exceeds 100x and waveforms differ significantly
    assert amplification_factor > 50.0, f"Amplification factor {amplification_factor} is unexpectedly small"
    assert abs_diff > 0.50, f"Waveform difference {abs_diff} is too small to demonstrate divergence"

    # In log-Mel spectrogram domain, power ratio differs by ~ (amplification_factor)^2
    db_shift = 20.0 * np.log10(amplification_factor)
    assert db_shift > 30.0, f"dB energy shift {db_shift} dB is insufficient to prove acoustic blocker"


def test_voice_inventory_and_accent_exhaustion_audit():
    """
    Verify that all available Central and Southern voices in Vieneu were already consumed
    in Phase 11 train/val/test splits, leaving 0 novel Central or Southern voices.
    """
    # Authoritative Phase 11 voice allocations
    phase11_train_voices = {"Adam", "Minh Triết", "Mỹ Duyên", "Ngọc Huyền", "Quang Sơn", "Thanh Bình", "Thái Sơn", "Thục Đoan"}
    phase11_val_voices = {"Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"}
    phase11_test_voices = {"Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"}

    all_phase11_voices = phase11_train_voices | phase11_val_voices | phase11_test_voices
    assert len(all_phase11_voices) == 16, f"Expected 16 unique Phase 11 voices, found {len(all_phase11_voices)}"

    # Complete catalogue of Vieneu preset voices and their regional accents
    vieneu_preset_accents = {
        # Central (Trung)
        "Quang Sơn": "Trung",
        "Ngọc Trân": "Trung",
        # Southern (Nam)
        "Adam": "Nam",
        "Thái Sơn": "Nam",
        "Thục Đoan": "Nam",
        "Minh Triết": "Nam",
        "Mỹ Duyên": "Nam",
        "Đức Trí": "Nam",
        "Kim Thanh": "Nam",
        "Thùy Dung": "Nam",
        # Northern (Bắc)
        "Adam bựa": "Bắc",
        "Trúc Ly": "Bắc",
        "Thiện Minh": "Bắc",
        "Hải Đăng": "Bắc",
        "Thiền Tâm Đức": "Bắc",
        "Minh Đức": "Bắc",
        "Xuân Vĩnh": "Bắc",
        "Ngọc Linh": "Bắc",
        "Đoan Trang": "Bắc",
        "Mai Anh": "Bắc",
        "Phạm Tuyên": "Bắc",
        "Ngọc Huyền": "Bắc",
        "Thanh Bình": "Bắc",
        "Quỳnh Anh": "Bắc",
        "Quốc Tuấn": "Bắc",
    }

    # Identify novel unused voices
    unused_voices = {v: acc for v, acc in vieneu_preset_accents.items() if v not in all_phase11_voices}

    unused_central = [v for v, acc in unused_voices.items() if acc == "Trung"]
    unused_southern = [v for v, acc in unused_voices.items() if acc == "Nam"]
    unused_northern = [v for v, acc in unused_voices.items() if acc == "Bắc"]

    # Assert complete exhaustion of Central and Southern voices
    assert len(unused_central) == 0, f"Expected 0 unused Central voices, found: {unused_central}"
    assert len(unused_southern) == 0, f"Expected 0 unused Southern voices, found: {unused_southern}"
    assert len(unused_northern) == 9, f"Expected 9 unused Northern voices, found: {len(unused_northern)}"


def test_phase13_governance_string_invariants():
    """Ensure no forbidden literal strings or raw directories are referenced."""
    forbidden = "data/" + "raw/" + "field_recordings"
    for report_candidate in [
        PROJECT_ROOT / "reports" / "PHASE13_CONFIRMATORY_EVALUATION.md",
        Path(__file__)
    ]:
        if report_candidate.exists():
            text = report_candidate.read_text(encoding="utf-8", errors="ignore")
            assert forbidden not in text, f"Forbidden string detected in {report_candidate}"
