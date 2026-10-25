#!/usr/bin/env python3
"""
Unit and Regression Test Suite for Phase 13.2 Confirmatory Preparation.
Workspace: D:\trainidentificationaudio
Target: tests/test_vieneu_research_v13_2.py
"""

import os
import sys
import json
import csv
import glob
import hashlib
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V13_2_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_2"
LEXICON_DIR = V13_2_DIR / "lexicon"
CATALOGUE_PATH = LEXICON_DIR / "confirmatory_lexicon_catalogue.json"
AUDIT_PATH = LEXICON_DIR / "lexical_independence_audit.json"

FROZEN_CKPT = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
EXPECTED_CKPT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_candidate_checkpoint_hash_integrity():
    """Verify candidate checkpoint hash is unmodified and pristine."""
    assert FROZEN_CKPT.exists(), f"Missing frozen checkpoint: {FROZEN_CKPT}"
    h = sha256_of_file(FROZEN_CKPT)
    assert h.lower() == EXPECTED_CKPT_HASH.lower(), f"Checkpoint hash mismatch: {h}"


def test_confirmatory_catalogue_structure_and_counts():
    """Verify confirmatory catalogue contains exactly 1,000 clips with required breakdown."""
    assert CATALOGUE_PATH.exists(), f"Missing catalogue: {CATALOGUE_PATH}"
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        cat = json.load(f)

    assert cat["total_clips"] == 1000
    assert cat["amendment"] == "Amendment 13.1-A"
    assert len(cat["speakers"]) == 8

    cats = cat["categories"]
    assert cats["target_commands"]["count"] == 200
    assert cats["target_commands"]["breakdown"] == {"TRAI": 50, "PHAI": 50, "LEN": 50, "XUONG": 50}

    assert cats["phonetic_minimal_pairs"]["count"] == 200
    assert len(cats["phonetic_minimal_pairs"]["texts"]["TRAI"]) == 50
    assert len(cats["phonetic_minimal_pairs"]["texts"]["PHAI"]) == 50
    assert len(cats["phonetic_minimal_pairs"]["texts"]["LEN"]) == 50
    assert len(cats["phonetic_minimal_pairs"]["texts"]["XUONG"]) == 50

    assert cats["syntactic_carrier_collisions"]["count"] == 200
    assert len(cats["syntactic_carrier_collisions"]["texts"]["TRAI"]) == 50
    assert len(cats["syntactic_carrier_collisions"]["texts"]["PHAI"]) == 50
    assert len(cats["syntactic_carrier_collisions"]["texts"]["LEN"]) == 50
    assert len(cats["syntactic_carrier_collisions"]["texts"]["XUONG"]) == 50

    assert cats["general_conversational_non_targets"]["count"] == 300
    assert len(cats["general_conversational_non_targets"]["texts"]) == 300

    assert cats["acoustic_background_and_silence"]["count"] == 100


def test_speaker_novelty_and_accent_isolation():
    """Verify that all 8 confirmatory speakers are Northern and have 0% overlap with Phase 11."""
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        cat = json.load(f)

    confirmatory_voice_names = {s["voice_name"] for s in cat["speakers"]}
    assert len(confirmatory_voice_names) == 8

    # All must have Northern accent
    for s in cat["speakers"]:
        assert s["accent"] == "Bắc", f"Speaker {s['voice_name']} is not Northern (Bắc)!"

    # Must be 100% disjoint from Phase 11 voices
    phase11_train_voices = {"Adam", "Minh Triết", "Mỹ Duyên", "Ngọc Huyền", "Quang Sơn", "Thanh Bình", "Thái Sơn", "Thục Đoan"}
    phase11_val_voices = {"Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"}
    phase11_test_voices = {"Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"}
    all_phase11_voices = phase11_train_voices | phase11_val_voices | phase11_test_voices

    overlap = confirmatory_voice_names.intersection(all_phase11_voices)
    assert len(overlap) == 0, f"Speaker leakage detected: {overlap}"


def test_lexical_independence_zero_overlap():
    """Verify 0% lexical overlap between confirmatory texts and all historical datasets."""
    assert AUDIT_PATH.exists()
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "LEXICAL_INDEPENDENCE_CONFIRMED"
    assert audit["lexical_overlap_count"] == 0
    assert audit["lexical_overlap_ratio"] == 0.0

    # Live verification: re-check all texts from catalogue against historical files
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        cat = json.load(f)

    new_texts = set()
    for kw in ["TRAI", "PHAI", "LEN", "XUONG"]:
        new_texts.update(cat["categories"]["phonetic_minimal_pairs"]["texts"][kw])
        new_texts.update(cat["categories"]["syntactic_carrier_collisions"]["texts"][kw])
    new_texts.update(cat["categories"]["general_conversational_non_targets"]["texts"])

    assert len(new_texts) == 700

    hist_texts = set()
    patterns = [
        str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v11" / "data" / "splits" / "*.csv"),
        str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review" / "review_manifest.csv")
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            with open(p, "r", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    t = r.get("transcription", r.get("text", "")).strip().lower()
                    if t and t not in {"[silence]", "trái", "phải", "lên", "xuống"}:
                        hist_texts.add(t)

    overlap = new_texts.intersection(hist_texts)
    assert len(overlap) == 0, f"Live lexical leakage detected: {overlap}"


def test_remediated_preprocessing_contract_frozen():
    """Verify that the preprocessing contract matches the remediated Phase 13.1 specification."""
    contract_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "bundle" / "preprocessing_contract.json"
    assert contract_path.exists()
    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    assert contract["audio_input"]["peak_normalization"]["policy"] == "clamped_only_if_max_abs_greater_than_1.0"
    assert contract["audio_input"]["peak_normalization"]["floor_amplification_allowed"] is False
    assert contract["decision_logic"]["calibrated_theta_cmd"] == 0.20


def test_phase13_2_governance_string_invariants():
    """Ensure no forbidden strings exist in Phase 13.2 artifacts."""
    forbidden = "data/" + "raw/" + "field_recordings"
    for p in V13_2_DIR.rglob("*"):
        if p.is_file() and p.suffix in [".py", ".md", ".json"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert forbidden not in text, f"Forbidden path string detected in {p}"
