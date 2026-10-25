"""
Automated validation tests for Phase 25: 80-Clip Blinded Listening Workflow.
Verifies clip uniqueness, audio decodability, strict metadata privacy (zero key leaks),
persistence behavior, and accurate accounting of completed vs unanswered reviews.
"""

import os
import sys
import tempfile
import csv
import json
import pytest
import numpy as np
import soundfile as sf
import pandas as pd

sys.path.insert(0, "experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening")
from analyze_blinded_review import analyze_blinded_review

BLINDED_DIR = "experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening"
AUDIO_DIR = os.path.join(BLINDED_DIR, "audio")
REVIEW_SHEET = os.path.join(BLINDED_DIR, "blinded_listening_review_sheet.csv")
KEY_CSV = os.path.join(BLINDED_DIR, "blinded_listening_key.csv")
KEY_JSON = os.path.join(BLINDED_DIR, "blinded_listening_key.json")
RUN_SCRIPT = os.path.join(BLINDED_DIR, "run_blinded_review.py")

def test_all_80_clips_unique_and_complete():
    assert os.path.exists(REVIEW_SHEET), "Review sheet missing"
    assert os.path.exists(KEY_CSV), "Quarantined key missing"

    rdf = pd.read_csv(REVIEW_SHEET)
    kdf = pd.read_csv(KEY_CSV)

    assert len(rdf) == 80, f"Expected 80 rows in review sheet, got {len(rdf)}"
    assert len(kdf) == 80, f"Expected 80 rows in key, got {len(kdf)}"

    expected_ids = [f"BLIND_EVAL_{i:02d}" for i in range(1, 81)]
    assert sorted(rdf["blind_clip_id"].tolist()) == expected_ids
    assert sorted(kdf["blind_clip_id"].tolist()) == expected_ids

    # Uniqueness
    assert rdf["blind_clip_id"].nunique() == 80
    assert kdf["blind_clip_id"].nunique() == 80

def test_all_audio_files_exist_and_playable():
    for i in range(1, 81):
        filename = f"blind_eval_{i:02d}.wav"
        p = os.path.join(AUDIO_DIR, filename)
        assert os.path.exists(p), f"Audio clip missing: {p}"

        data, sr = sf.read(p)
        assert sr == 16000, f"Sample rate mismatch for {filename}: {sr}"
        assert len(data) == 16000, f"Length mismatch for {filename}: {len(data)}"
        assert not np.isnan(data).any(), f"NaN in {filename}"
        assert not np.isinf(data).any(), f"Inf in {filename}"
        assert np.max(np.abs(data)) > 0.05, f"Near-silent audio in {filename}"

def test_strict_metadata_isolation_and_privacy():
    """Verify that run_blinded_review.py does NOT contain or reference the key."""
    assert os.path.exists(RUN_SCRIPT)
    with open(RUN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    # The server script must NEVER reference or import the answer key
    assert "blinded_listening_key.csv" not in content, "Leak: Key CSV referenced in server script"
    assert "blinded_listening_key.json" not in content, "Leak: Key JSON referenced in server script"

    # The HTML template in the script must NOT contain original sample IDs or voice identities
    assert "vnkws_" not in content, "Leak: Original sample ID pattern found in review script"
    assert "Hải Đăng" not in content, "Leak: Voice name found in review script"
    assert "pnnbao-ump" not in content, "Leak: HuggingFace model repo found in review script"

def test_empty_review_sheet_reports_pending_without_fabrication():
    """Verify that the empty review sheet produces 0 completed, 80 missing, and no fabricated metrics."""
    res = analyze_blinded_review(responses_path=REVIEW_SHEET, key_path=KEY_CSV)
    assert res["completed"] == 0
    assert res["missing"] == 80
    assert res["status"] == "PENDING_HUMAN_LISTENER_VERIFICATION"
    assert "accuracy" not in res, "Accuracy must not be computed when 0 reviews exist"

def test_simulated_isolated_review_persistence_and_reloading():
    """Test using an isolated temporary CSV to confirm incremental saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = os.path.join(tmpdir, "isolated_test_responses.csv")

        # Copy original review sheet structure
        rdf = pd.read_csv(REVIEW_SHEET)
        rdf.to_csv(test_csv, index=False)

        # Fill in exactly 3 test items
        test_df = pd.read_csv(test_csv, dtype=str)
        col_cmd = [c for c in test_df.columns if "perceived_command" in c][0]
        col_intel = [c for c in test_df.columns if "intelligibility" in c][0]
        col_nat = [c for c in test_df.columns if "naturalness" in c][0]
        col_tone = [c for c in test_df.columns if "perceived_tone" in c][0]

        test_df.loc[0, col_cmd] = "TRAI"
        test_df.loc[0, col_intel] = "5"
        test_df.loc[0, col_nat] = "4"
        test_df.loc[0, col_tone] = "SAC"

        test_df.loc[1, col_cmd] = "PHAI"
        test_df.loc[1, col_intel] = "4"
        test_df.loc[1, col_nat] = "4"
        test_df.loc[1, col_tone] = "HOI"

        test_df.loc[2, col_cmd] = "UNCERTAIN"
        test_df.loc[2, col_intel] = "2"
        test_df.loc[2, col_nat] = "2"
        test_df.loc[2, col_tone] = "UNCERTAIN"

        test_df.to_csv(test_csv, index=False)

        # Run analysis on the 3-item test file
        res = analyze_blinded_review(responses_path=test_csv, key_path=KEY_CSV)
        assert res["completed"] == 3
        assert res["missing"] == 77
        assert res["status"] == "REVIEW_COMPLETED"
        assert res["relisten_count"] >= 1, "UNCERTAIN clip must be flagged for re-listening"
