"""
Automated Integrity and Regression Tests for Phases 85-89.
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v10/

Verifies:
1. Phase 85: Review interface blinding, zero answer key leakage, localhost binding, response validation.
2. Phase 86: Honest accounting of human responses (0 fabricated, BLOCKED_NO_HUMAN_RESPONSES).
3. Phase 87: Robustness reproduction integrity, 14,040 evaluations, SNR +20 dB recall < 95.0% marking.
4. Phase 88: Physical loopback hardware audit (NOT_RUN status, zero fake audio).
5. Governance: Checkpoint hashes preserved, git cached status empty, field recordings untouched.
"""

import os
import sys
import json
import csv
import hashlib
import pytest
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V10_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
HUMAN_DIR = os.path.join(V10_DIR, "human_review")
ROBUST_DIR = os.path.join(V10_DIR, "robustness")
PRED_DIR = os.path.join(V10_DIR, "predictions")
LOOPBACK_DIR = os.path.join(V10_DIR, "acoustic_loopback")
CKPTS_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9", "artifacts", "checkpoints")

sys.path.insert(0, HUMAN_DIR)
from response_validation import validate_response_file, EXPECTED_CLIP_IDS


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Phase 85 Tests: Blinding & Security
# ---------------------------------------------------------------------------

def test_phase85_blinding_and_zero_key_leakage():
    app_py = os.path.join(HUMAN_DIR, "review_app.py")
    assert os.path.exists(app_py), "review_app.py missing"
    with open(app_py, "r", encoding="utf-8") as f:
        content = f.read()

    # Must never import or open review_manifest.csv
    assert "review_manifest.csv" not in content, "Leak: review_manifest.csv found in server code"
    assert "TRAI" not in content, "Leak: Keyword label found in server code"
    assert "PHAI" not in content, "Leak: Keyword label found in server code"
    assert "XUONG" not in content, "Leak: Keyword label found in server code"
    assert "Ngọc Trân" not in content, "Leak: Voice name found in server code"
    assert "Mai Anh" not in content, "Leak: Voice name found in server code"

    # Must bind strictly to 127.0.0.1
    assert '("127.0.0.1", port)' in content, "Server does not bind exclusively to 127.0.0.1"


def test_phase85_audio_assets_parity():
    manifest_csv = os.path.join(HUMAN_DIR, "review_manifest.csv")
    assert os.path.exists(manifest_csv), "review_manifest.csv missing"
    with open(manifest_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 120, f"Expected 120 clips in manifest, got {len(rows)}"
    for r in rows:
        audio_path = os.path.join(HUMAN_DIR, "audio", r["audio_file"])
        assert os.path.exists(audio_path), f"Audio file missing: {audio_path}"
        assert sha256_file(audio_path) == r["sha256"], f"SHA256 mismatch for {r['audio_file']}"


def test_phase85_response_validation_logic():
    template_csv = os.path.join(HUMAN_DIR, "response_template.csv")
    res = validate_response_file(template_csv)
    # Blank template must NOT be accepted as complete
    assert not res["is_complete"], "Blank template erroneously marked complete"
    assert res["valid_rows_count"] == 0
    assert res["invalid_rows_count"] == 120


# ---------------------------------------------------------------------------
# Phase 86 Tests: Anti-Fabrication & Honest Review Accounting
# ---------------------------------------------------------------------------

def test_phase86_honest_human_review_accounting():
    summary_json = os.path.join(HUMAN_DIR, "review_quality_summary.json")
    assert os.path.exists(summary_json), "review_quality_summary.json missing"
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "BLOCKED_NO_HUMAN_RESPONSES"
    assert data["governance_banner"] == "PENDING_HUMAN_LISTENER_VERIFICATION"
    assert data["completed_reviewers_count"] == 0
    assert data["validated_clips_count"] == 0

    validated_csv = os.path.join(HUMAN_DIR, "validated_responses.csv")
    assert os.path.exists(validated_csv), "validated_responses.csv missing"
    with open(validated_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 0, f"Expected 0 data rows in validated responses, got {len(reader)}"


# ---------------------------------------------------------------------------
# Phase 87 Tests: Independent Robustness Reproduction
# ---------------------------------------------------------------------------

def test_phase87_reproduction_metrics_and_snr20db_marking():
    summary_path = os.path.join(PRED_DIR, "robustness_reproduced_summary.json")
    per_clip_path = os.path.join(PRED_DIR, "robustness_per_clip.csv")
    manifest_v2_path = os.path.join(ROBUST_DIR, "condition_manifest_v2.json")

    assert os.path.exists(summary_path), "robustness_reproduced_summary.json missing"
    assert os.path.exists(per_clip_path), "robustness_per_clip.csv missing"
    assert os.path.exists(manifest_v2_path), "condition_manifest_v2.json missing"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(summary) == 13, f"Expected 13 conditions, got {len(summary)}"

    # Clean baseline verification
    clean = summary["clean_baseline"]["cross_seed_aggregate"]
    assert clean["mean_command_recall"] == 0.9688
    assert clean["mean_false_acceptance_rate"] == 0.0583

    # SNR +20 dB verification: MUST be below 95.0% recall
    snr20 = summary["snr_20db"]["cross_seed_aggregate"]
    assert snr20["mean_command_recall"] == 0.9437
    assert snr20["mean_command_recall"] < 0.950, "SNR +20 dB recall must be below 95%"
    assert not snr20["meets_recall_target_ge_95"], "SNR +20 dB must not meet recall target"
    assert snr20["target_status"] in ["BELOW_RECALL_TARGET", "BELOW_RECALL_AND_EXCEEDS_FAR"]

    # Check per-clip inferences count: 14,040
    with open(per_clip_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 14040, f"Expected 14,040 rows in per_clip.csv, got {len(rows)}"


# ---------------------------------------------------------------------------
# Phase 88 Tests: Acoustic Loopback Feasibility
# ---------------------------------------------------------------------------

def test_phase88_acoustic_loopback_not_run_and_zero_fake_audio():
    loopback_script = os.path.join(LOOPBACK_DIR, "run_loopback_capture.py")
    assert os.path.exists(loopback_script), "run_loopback_capture.py missing"

    captures_dir = os.path.join(LOOPBACK_DIR, "captures")
    if os.path.exists(captures_dir):
        files = os.listdir(captures_dir)
        assert len(files) == 0, f"Captures directory must be empty when NOT_RUN, found {files}"

    report_path = os.path.join(V10_DIR, "reports", "PHASE88_ACOUSTIC_LOOPBACK.md")
    assert os.path.exists(report_path), "PHASE88_ACOUSTIC_LOOPBACK.md missing"
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert "NOT_RUN" in text
    assert "UNRESOLVED" in text


# ---------------------------------------------------------------------------
# Governance & Model Hash Invariance Tests
# ---------------------------------------------------------------------------

def test_governance_and_model_hash_invariance():
    # Verify frozen checkpoints are intact
    expected_hashes = {
        "variant_w10_seed42.pt": "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed",
        "variant_w10_seed43.pt": "77f468c68b261de506a3fcb022c8d7cc68f8b0c2ad650852b7c300f274dda251",
        "variant_w10_seed44.pt": "bebab580e4caa6934c98a62b095611e4eef9221a7f4282d36f9a4b6928abde64"
    }

    for fname, exp_hash in expected_hashes.items():
        p = os.path.join(CKPTS_DIR, fname)
        assert os.path.exists(p), f"Checkpoint missing: {p}"
        assert sha256_file(p) == exp_hash, f"Checkpoint modified: {fname}"

    # Verify git cached diff is strictly empty
    res = subprocess.run(["git", "diff", "--cached"], cwd=BASE_DIR, capture_output=True, text=True)
    assert res.returncode == 0
    assert res.stdout.strip() == "", f"Staged git changes detected: {res.stdout}"
