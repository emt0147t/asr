"""
Unit and regression tests for Phases 80-84: Vietnamese KWS Research Campaign v10.
Covers:
- Phase 80: Metric reconciliation between Phase 74 (test) and Phase 77 (val), configuration audit.
- Phase 81: 120-clip blinded review package, template, review app, zero ground-truth exposure.
- Phase 82: Robustness audit, 13 conditions, strict target marking (20dB SNR recall < 95%).
- Phase 83: Acoustic feasibility NOT_RUN documentation, blocker disclosure, zero simulation.
- Phase 84: Quad-partite operational verdicts, readiness matrix, next bottleneck decision.
- Historical immutability and protected field recording isolation.
"""

import os
import sys
import json
import csv
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)


def test_phase80_metric_reconciliation_outputs():
    """Verifies that Phase 80 reproduced metrics JSONs exist and reconcile P74 vs P77 correctly."""
    v10_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
    p74_json = os.path.join(v10_dir, "predictions", "reproduced_phase74_metrics.json")
    p77_json = os.path.join(v10_dir, "predictions", "reproduced_phase77_metrics.json")
    r1 = os.path.join(v10_dir, "reports", "PHASE80_METRIC_RECONCILIATION.md")
    r2 = os.path.join(v10_dir, "reports", "PHASE80_CONFIGURATION_AUDIT.md")

    for p in [p74_json, p77_json, r1, r2]:
        assert os.path.exists(p), f"{p} must exist"

    with open(p74_json, "r", encoding="utf-8") as f:
        data74 = json.load(f)
    assert data74["total_clips"] == 720
    assert data74["non_target_clips"] == 560
    assert data74["cross_seed_aggregate"]["mean_command_recall"] >= 0.99
    assert data74["cross_seed_aggregate"]["mean_false_acceptance_rate"] <= 0.03

    with open(p77_json, "r", encoding="utf-8") as f:
        data77 = json.load(f)
    assert data77["total_clips"] == 360
    assert data77["non_target_clips"] == 200
    assert data77["cross_seed_aggregate"]["mean_command_recall"] >= 0.96
    assert data77["cross_seed_aggregate"]["mean_false_acceptance_rate"] <= 0.065


def test_phase81_human_review_package_and_app():
    """Verifies Phase 81 review package, 120 audio files, template, and review app."""
    v10_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
    review_dir = os.path.join(v10_dir, "human_review")

    template_csv = os.path.join(review_dir, "response_template.csv")
    manifest_csv = os.path.join(review_dir, "review_manifest.csv")
    instructions_md = os.path.join(review_dir, "reviewer_instructions.md")
    app_py = os.path.join(review_dir, "review_app.py")
    report_md = os.path.join(v10_dir, "reports", "PHASE81_LISTENING_REVIEW_PROTOCOL.md")

    for p in [template_csv, manifest_csv, instructions_md, app_py, report_md]:
        assert os.path.exists(p), f"{p} must exist"

    with open(template_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120, f"Expected 120 template rows, got {len(rows)}"

    # Ensure template has blank reviewer entries and zero ground truth columns
    forbidden_cols = ["canonical_class", "class_label", "ground_truth", "transcription", "confusion_family", "pred_label"]
    with open(template_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            for f_col in forbidden_cols:
                assert f_col != col.lower(), f"Reviewer template must NOT contain ground-truth column: {col}"

    for r in rows:
        assert r["perceived_text"] == ""
        assert r["intelligibility_1to5"] == ""
        assert r["naturalness_1to5"] == ""
        assert r["pronunciation_correct_yn"] == ""
        assert r["confidence_1to5"] == ""

    # Check 120 audio files in audio/
    audio_dir = os.path.join(review_dir, "audio")
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    assert len(audio_files) == 120, f"Expected 120 audio files, got {len(audio_files)}"


def test_phase82_robustness_audit_outputs():
    """Verifies Phase 82 13 conditions, manifest, and strict target threshold marking."""
    v10_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
    manifest_path = os.path.join(v10_dir, "robustness", "condition_manifest.json")
    summary_path = os.path.join(v10_dir, "predictions", "robustness_reproduced_summary.json")
    report_path = os.path.join(v10_dir, "reports", "PHASE82_ROBUSTNESS_AUDIT.md")

    for p in [manifest_path, summary_path, report_path]:
        assert os.path.exists(p), f"{p} must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        conditions = json.load(f)
    assert len(conditions) == 13, f"Expected 13 conditions, got {len(conditions)}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Verify +20 dB SNR condition is explicitly marked as failing recall research target (< 95%)
    snr20 = summary["snr_20db"]["cross_seed_aggregate"]
    assert snr20["mean_command_recall"] < 0.95, f"Expected < 0.95, got {snr20['mean_command_recall']}"
    assert snr20["meets_recall_target_ge_95"] is False
    assert snr20["target_status"] in ["BELOW_RECALL_TARGET", "FAILS_BOTH_TARGETS"]


def test_phase83_acoustic_protocol_and_not_run():
    """Verifies Phase 83 protocol and execution script properly document NOT_RUN status."""
    v10_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
    report_path = os.path.join(v10_dir, "reports", "PHASE83_ACOUSTIC_PROTOCOL.md")
    script_path = os.path.join(v10_dir, "acoustic_loopback", "run_loopback_capture.py")
    template_path = os.path.join(v10_dir, "acoustic_loopback", "recording_manifest_template.csv")

    for p in [report_path, script_path, template_path]:
        assert os.path.exists(p), f"{p} must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        c = f.read()
        assert "NOT_RUN" in c
        assert "Hardware" in c or "hardware" in c

    with open(template_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "source_clip_id" in header
        assert "captured_sha256" in header


def test_phase84_final_decision_and_readiness_matrix():
    """Verifies Phase 84 quad-partite verdicts, status banners, and bottleneck recommendations."""
    v10_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
    final_dec = os.path.join(v10_dir, "reports", "PHASE84_FINAL_DECISION.md")
    readiness = os.path.join(v10_dir, "reports", "PHASE84_READINESS_MATRIX.md")
    readme = os.path.join(v10_dir, "README.md")

    for doc in [final_dec, readiness, readme]:
        assert os.path.exists(doc), f"{doc} must exist"
        with open(doc, "r", encoding="utf-8") as f:
            c = f.read()
            assert "PENDING_HUMAN_LISTENER_VERIFICATION" in c
            assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c

    with open(final_dec, "r", encoding="utf-8") as f:
        text = f.read()
        assert "FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED" in text
        assert "NOT_RUN" in text
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in text
        assert "Further model retraining" in text or "not justified" in text.lower()


def test_historical_immutability_and_field_recording_protection():
    """Verifies that no field recordings were accessed and historical campaigns remain intact."""
    v10_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v10")
    for root, _, files in os.walk(v10_dir):
        for f in files:
            if f.endswith((".py", ".json", ".md", ".csv")):
                fpath = os.path.join(root, f)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                    assert "data/raw/field_recordings" not in content, f"{fpath} must not reference data/raw/field_recordings"

    historical_dirs = [
        os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_1000"),
        os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class"),
        os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_vad_ablation_v1"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v2"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v3"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v4"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8"),
        os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9"),
    ]

    for hdir in historical_dirs:
        assert os.path.exists(hdir), f"Historical directory {hdir} must exist and be preserved"
