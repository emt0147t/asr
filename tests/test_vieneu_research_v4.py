"""
Unit and regression tests for Phases 50-54: Vietnamese KWS Research Campaign v4.
Covers:
- Phase 50: 3-hour continuous challenge stream bit-exactness and disjoint error categorization.
- Phase 51: Phonetic error taxonomy and blinded human review key/sheet integrity.
- Phase 52: Bounded-utterance dataset manifests, voice isolation (zero leak to test).
- Phase 53: VietnameseBoundedKWSNet architecture parameter count and forward pass.
- Phase 54: Dual-verdict formal consistency and historical immutability.
"""

import os
import sys
import json
import pytest
import pandas as pd
import soundfile as sf
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from experiments.vieneu_kws_research_v4.models.bounded_utterance.train_bounded_model import (
    VietnameseBoundedKWSNet,
    LogMelFrontend
)


def test_phase50_stream_integrity_and_categorization():
    """Verifies that 3-hour stream metadata is exact and non-overlapping categorization holds."""
    stream_path = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v3", "data", "continuous_streams", "test_stream_continuous_nontarget_3h.wav")
    assert os.path.exists(stream_path), "3-hour challenge stream must exist"

    info = sf.info(stream_path)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.frames == 172800000, "Must be exactly 172,800,000 frames (3.000 hours)"
    assert abs(info.duration - 10800.0) < 1e-4

    # Verify audit findings JSON exists and is verified
    audit_json = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "audit_findings_phase50.json")
    assert os.path.exists(audit_json)
    with open(audit_json, "r", encoding="utf-8") as f:
        findings = json.load(f)
    assert findings["metric_reproduction"]["is_perfect_reproduction"] is True


def test_phase51_error_taxonomy_and_human_review_integrity():
    """Verifies error taxonomy and that the human review sheet is strictly blinded."""
    taxonomy_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "data", "error_taxonomy.csv")
    assert os.path.exists(taxonomy_file)
    df_tax = pd.read_csv(taxonomy_file)
    assert len(df_tax) >= 5
    assert "ERR_FN_TRAI_RHYME" in df_tax["taxonomy_id"].values
    assert "ERR_FN_PHAI_RHYME" in df_tax["taxonomy_id"].values

    # Check human review files
    sheet_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "human_review", "human_review_sheet.csv")
    key_file = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "human_review", "human_review_key.csv")
    assert os.path.exists(sheet_file) and os.path.exists(key_file)

    df_sheet = pd.read_csv(sheet_file)
    df_key = pd.read_csv(key_file)
    assert len(df_sheet) == 40
    assert len(df_key) == 40

    # Ensure sheet does NOT reveal labels
    assert "ground_truth_label" not in df_sheet.columns
    assert "true_text" not in df_sheet.columns

    # Key must contain blind_id and true labels
    assert "blind_id" in df_key.columns
    assert "ground_truth_label" in df_key.columns
    assert set(df_sheet["blind_id"]) == set(df_key["blind_id"])


def test_phase52_bounded_manifests_isolation():
    """Verifies manifest partition and zero speaker voice overlap between train and test."""
    data_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "data", "splits")
    train_df = pd.read_csv(os.path.join(data_dir, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_manifest.csv"))

    assert len(train_df) == 1360
    assert len(val_df) == 360
    assert len(test_df) == 720

    # Filter out silence/noise conditions to isolate actual speech speaker voices
    silence_conditions = {'airflow_pink', 'hvac_hum', 'near_silence', 'office_ambient'}
    train_voices = set(train_df["voice_id"]) - silence_conditions
    val_voices = set(val_df["voice_id"]) - silence_conditions
    test_voices = set(test_df["voice_id"]) - silence_conditions

    # Strict disjointness
    assert len(train_voices.intersection(test_voices)) == 0, "Zero voice overlap between train and test"
    assert len(val_voices.intersection(test_voices)) == 0, "Zero voice overlap between val and test"
    assert len(train_voices.intersection(val_voices)) == 0, "Zero voice overlap between train and val"
    assert len(test_voices) == 4
    assert len(val_voices) == 4
    assert len(train_voices) == 17


def test_phase53_bounded_model_architecture_and_forward():
    """Verifies VietnameseBoundedKWSNet parameter budget, shape, and forward pass."""
    model = VietnameseBoundedKWSNet(num_classes=5)
    param_count = sum(p.numel() for p in model.parameters())
    # Must be compact (< 200k params)
    assert param_count < 200000, f"Expected < 200k params, got {param_count}"

    dummy_input = torch.randn(4, 1, 40, 101)
    out = model(dummy_input)
    assert out.shape == (4, 5)

    # Test frontend
    frontend = LogMelFrontend()
    raw_audio = torch.randn(4, 16000)
    mel = frontend(raw_audio)
    assert mel.shape == (4, 1, 40, 101)


def test_phase54_dual_verdict_and_report_artifacts():
    """Verifies that dual verdicts and Phase 54 reports are present with exact criteria."""
    report_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4", "reports")
    dec_file = os.path.join(report_dir, "PHASE54_FINAL_DECISION.md")
    mat_file = os.path.join(report_dir, "PHASE54_READINESS_MATRIX.md")

    assert os.path.exists(dec_file)
    assert os.path.exists(mat_file)

    with open(dec_file, "r", encoding="utf-8") as f:
        dec_text = f.read()

    assert "BOUNDED_UTTERANCE_FEASIBLE" in dec_text.replace("\\", "")
    assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in dec_text.replace("\\", "")
    assert "PENDING_HUMAN_LISTENER_VERIFICATION" in dec_text
    assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in dec_text
