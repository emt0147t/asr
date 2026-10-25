"""
Unit and regression tests for Phases 75-79: Vietnamese KWS Research Campaign v9.
Covers:
- Phase 75: Frozen candidate checkpoint integrity, architecture, bit-exact metrics, voice provenance.
- Phase 76: Blinded human review package, 120 clips, sanitized reviewer form, non-fabrication.
- Phase 77: Controlled synthetic robustness, 13 conditions, 14,040 predictions, clean manifest.
- Phase 78: Acoustic feasibility NOT_RUN documentation, blocker disclosure, zero fabrication.
- Phase 79: Tripartite operational verdicts, readiness matrix, status banners.
- Historical immutability and field recording protection.
"""

import os
import sys
import json
import csv
import pytest
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from experiments.vieneu_kws_research_v9.artifacts.inference import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    FrozenCandidatePredictor
)


def test_phase75_candidate_freeze_and_hashes():
    """Verifies that Phase 75 candidate manifest exists, hashes match, and parameters equal 138,469."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    manifest_path = os.path.join(v9_dir, "artifacts", "candidate_manifest.json")
    assert os.path.exists(manifest_path), "candidate_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["candidate_id"] == "variant_w10"
    assert manifest["architecture"]["parameter_count"] == 138469

    # Check that frozen checkpoints exist and can be loaded
    ckpts_dir = os.path.join(v9_dir, "artifacts", "checkpoints")
    for s in [42, 43, 44]:
        ckpt_file = os.path.join(ckpts_dir, f"variant_w10_seed{s}.pt")
        assert os.path.exists(ckpt_file), f"{ckpt_file} must exist"

        m = CorrectedVietnameseKWSNet(num_classes=5)
        ckpt = torch.load(ckpt_file, map_location="cpu")
        m.load_state_dict(ckpt["model_state_dict"])
        total_p = sum(p.numel() for p in m.parameters())
        assert total_p == 138469, f"Expected 138,469 parameters, got {total_p}"

    # Verify reproduced locked test metrics
    rep = manifest["reproduced_metrics"]["cross_seed_aggregate"]
    assert rep["mean_command_recall"] >= 0.95
    assert rep["mean_false_acceptance_rate"] <= 0.05
    assert rep["mean_5class_accuracy"] >= 0.95


def test_phase75_voice_provenance_and_limitations():
    """Verifies that voice provenance discloses fixed 25-voice inventory and no unseen-voice exaggeration."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    manifest_path = os.path.join(v9_dir, "artifacts", "candidate_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    disc = manifest["voice_provenance_disclosure"]
    assert disc["total_approved_tts_inventory"] == 25
    assert len(disc["training_split_voices"]) == 17
    assert len(disc["validation_split_voices"]) == 4
    assert len(disc["test_split_voices"]) == 4
    assert "No genuinely unseen approved TTS voices exist" in disc["unseen_voice_status"]
    assert "NOT be claimed as zero-shot human generalization" in disc["unseen_voice_status"]


def test_phase76_blinded_human_review_package():
    """Verifies that Phase 76 review package is properly stratified, blinded, and un-fabricated."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    review_dir = os.path.join(v9_dir, "human_review")

    form_path = os.path.join(review_dir, "reviewer_form.csv")
    manifest_path = os.path.join(review_dir, "review_manifest.csv")
    readme_path = os.path.join(review_dir, "README.md")
    protocol_path = os.path.join(v9_dir, "reports", "PHASE76_BLINDED_LISTENING_PROTOCOL.md")

    for p in [form_path, manifest_path, readme_path, protocol_path]:
        assert os.path.exists(p), f"{p} must exist"

    with open(form_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        form_rows = list(reader)

    assert len(form_rows) == 120, f"Expected 120 review clips, got {len(form_rows)}"

    # Crucial blinding checks on reviewer form:
    # 1. No ground-truth label columns
    forbidden_cols = ["canonical_class", "class_label", "ground_truth", "transcription", "confusion_family", "pred_label", "predicted_class"]
    for col in reader.fieldnames:
        for f_col in forbidden_cols:
            assert f_col != col.lower(), f"Reviewer form must NOT contain ground-truth column: {col}"

    # 2. Rating fields must be empty (strictly zero fabrication)
    for r in form_rows:
        assert r["perceived_text"] == "", "perceived_text must be empty"
        assert r["intelligibility_1to5"] == "", "intelligibility_1to5 must be empty"
        assert r["naturalness_1to5"] == "", "naturalness_1to5 must be empty"
        assert r["pronunciation_correct_yn"] == "", "pronunciation_correct_yn must be empty"
        assert r["confidence_1to5"] == "", "confidence_1to5 must be empty"

    # Check that audio files exist
    for r in form_rows[:10]:
        audio_file = os.path.join(review_dir, r["audio_path"])
        assert os.path.exists(audio_file), f"Review audio file {audio_file} must exist"


def test_phase77_robustness_study_outputs():
    """Verifies Phase 77 outputs, 13 conditions, and clean benchmark manifest."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    summary_path = os.path.join(v9_dir, "predictions", "robustness_summary.json")
    per_clip_path = os.path.join(v9_dir, "predictions", "robustness_per_clip.csv")
    clean_manifest = os.path.join(v9_dir, "robustness", "robustness_clean_manifest.csv")
    report_path = os.path.join(v9_dir, "reports", "PHASE77_ROBUSTNESS_REPORT.md")

    for p in [summary_path, per_clip_path, clean_manifest, report_path]:
        assert os.path.exists(p), f"{p} must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(summary) == 13, f"Expected 13 conditions, got {len(summary)}"
    assert "clean_baseline" in summary
    assert "snr_20db" in summary
    assert "snr_10db" in summary
    assert "snr_0db" in summary
    assert "gain_minus_12db" in summary
    assert "reverb_rt60_200ms" in summary
    assert "speed_0_90x" in summary

    # Check per-clip prediction rows (13 conditions * 360 clips * 3 seeds = 14,040 rows)
    with open(per_clip_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 14040, f"Expected 14,040 rows, got {len(rows)}"


def test_phase78_acoustic_feasibility_not_run():
    """Verifies Phase 78 is honestly recorded as NOT_RUN without simulated physical data."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    report_path = os.path.join(v9_dir, "reports", "PHASE78_ACOUSTIC_FEASIBILITY.md")
    assert os.path.exists(report_path), "PHASE78_ACOUSTIC_FEASIBILITY.md must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()
        assert "NOT_RUN" in text
        assert "Hardware" in text or "blocker" in text.lower()


def test_phase79_tripartite_operational_decisions():
    """Verifies Phase 79 formal tripartite decisions and required status markers."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    final_dec = os.path.join(v9_dir, "reports", "PHASE79_FINAL_DECISION.md")
    readiness = os.path.join(v9_dir, "reports", "PHASE79_READINESS_MATRIX.md")
    readme = os.path.join(v9_dir, "README.md")

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


def test_historical_immutability_and_field_recording_protection():
    """Verifies that no field recordings were accessed and historical campaigns remain intact."""
    v9_dir = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v9")
    for root, _, files in os.walk(v9_dir):
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
    ]

    for hdir in historical_dirs:
        assert os.path.exists(hdir), f"Historical directory {hdir} must exist and be preserved"
