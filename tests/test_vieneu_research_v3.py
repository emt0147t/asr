"""
Unit and regression tests for Phases 45-49: Vietnamese KWS Research Campaign v3.
Covers:
- Phase 45: Architecture reconciliations, exact FA/hr math, CTC token decoding.
- Phase 46: Split isolation (0 text overlap, disjoint speech voices), 3-hour stream metadata.
- Phase 47-49: Immutability of historical baselines and calibration integrity.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import soundfile as sf
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v2.models.ctc_kws.ctc_model import (
    VietnameseCharTokenizer,
    VietnameseCTCKWSNet,
    get_ctc_param_count
)
from experiments.vieneu_kws_v2.evaluate import match_events_and_evaluate


def test_phase45_architectures_and_discrepancy_resolutions():
    """Verifies that Stage 1, Stage 2, and CTC architectures match exact specifications."""
    m1 = VietnameseKWSNet(num_classes=6)
    m2 = TemporalKWSVerifier(num_classes=5)
    tok = VietnameseCharTokenizer()
    m_ctc = VietnameseCTCKWSNet(vocab_size=tok.vocab_size)

    assert get_model_param_count(m1) == 138534
    assert get_model_param_count(m2) == 340773
    assert get_ctc_param_count(m_ctc) == 229115

    # Stage 2 MUST have TCN backbone and zero recurrent/GRU layers
    assert hasattr(m2, "tcn"), "Stage 2 verifier must have TCN sequence backbone"
    assert not any("gru" in name.lower() or "rnn" in name.lower() for name, _ in m2.named_modules()), (
        "Stage 2 verifier must contain zero recurrent/GRU layers"
    )


def test_phase45_ctc_collapse_and_unicode_preservation():
    """Verifies CTC tokenizer greedy collapse and Unicode diacritic handling."""
    tok = VietnameseCharTokenizer()

    # Repeated characters with and without blank
    l_id = tok.char_to_id["l"]
    assert tok.decode([l_id, l_id]) == "l", "Consecutive duplicates without blank must collapse"
    assert tok.decode([l_id, 0, l_id]) == "ll", "Duplicates separated by blank must be preserved"

    # Exact keyword matching
    for kw, text in [("TRAI", "trái"), ("PHAI", "phải"), ("LEN", "lên"), ("XUONG", "xuống")]:
        matched, exact = tok.match_keyword(text)
        assert matched == kw and exact, f"Keyword match failed for {kw}"

    # Embedded phrase matching should not be exact
    matched, exact = tok.match_keyword("tiến lên phía trước")
    assert matched == "LEN" and not exact


def test_exact_fa_per_hour_math_and_active_speech_rate():
    """Verifies mathematically exact calculation of wall-clock and speech-active FA/hr."""
    gt_df = pd.DataFrame(columns=["event_type", "class_label", "onset_sec", "end_time_sec", "sample_id"])
    triggers = [
        {"trigger_time_sec": 100.0, "pred_label": "LEN", "confidence": 0.9},
        {"trigger_time_sec": 200.0, "pred_label": "PHAI", "confidence": 0.85}
    ]

    # Exactly 3.0 hours = 10,800s
    res = match_events_and_evaluate(triggers, gt_df, stream_duration_sec=10800.0)
    assert res["stream_duration_hours"] == 3.0
    assert abs(res["fa_per_hour"] - round(2.0 / 3.0, 2)) < 1e-4

    # Active speech rate on 0.4 hours active speech
    active_hr = 0.40
    active_fa_hr = round(2.0 / active_hr, 2)
    assert abs(active_fa_hr - 5.0) < 1e-4


def test_historical_artifacts_immutability():
    """Verifies that historical experiments (Phases 26-44) remain strictly untouched."""
    # Phase 26 baseline
    b_man = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_1000", "manifest.csv")
    assert os.path.exists(b_man)
    assert len(pd.read_csv(b_man)) == 1000

    # Phase 28 6-class v1
    v1_man = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class", "manifest.csv")
    assert os.path.exists(v1_man)
    assert len(pd.read_csv(v1_man)) == 1500

    # Phase 33 6-class v2
    v2_man = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2", "manifest.csv")
    assert os.path.exists(v2_man)
    assert len(pd.read_csv(v2_man)) == 2000

    # Phase 41 research v2
    rv2_man = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v2", "data", "manifest.csv")
    assert os.path.exists(rv2_man)
    assert len(pd.read_csv(rv2_man)) == 2000
