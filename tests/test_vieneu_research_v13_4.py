"""
Tests for Phase 13.4: Final Vietnamese Bounded-Utterance CNN Release Candidate.

Validates:
1. Candidate checkpoint SHA-256 byte immobility in release bundle.
2. ONNX model export integrity and verified SHA-256 hash.
3. Complete bundle integrity against checksums.json.
4. PyTorch vs ONNX Runtime inference parity and latency SLA on development fixtures.
5. Governance invariants: labels FROZEN_CNN_RELEASE_CANDIDATE, NOT_VALIDATED_ON_NATURAL_SPEECH,
   and NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"
CHECKPOINT_PATH = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
ONNX_PATH = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
CHECKSUMS_JSON = BUNDLE_DIR / "checksums.json"
MODEL_CONFIG_JSON = BUNDLE_DIR / "model_config.json"
MODEL_CARD_MD = BUNDLE_DIR / "MODEL_CARD.md"
ROOT_REPORT = PROJECT_ROOT / "reports" / "PHASE13_4_FINAL_CNN_RELEASE.md"
EXP_REPORT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "reports" / "PHASE13_4_FINAL_CNN_RELEASE.md"
DEV_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"

EXPECTED_CHECKPOINT_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_ONNX_SHA256 = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_immutability():
    """Verify release bundle checkpoint exists and matches authoritative SHA-256."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    actual_sha = compute_sha256(CHECKPOINT_PATH)
    assert actual_sha == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_sha}"
    )


def test_onnx_model_integrity():
    """Verify exported ONNX model exists and matches recorded SHA-256."""
    assert ONNX_PATH.exists(), f"ONNX file missing: {ONNX_PATH}"
    actual_sha = compute_sha256(ONNX_PATH)
    assert actual_sha == EXPECTED_ONNX_SHA256, (
        f"ONNX SHA-256 mismatch!\nExpected: {EXPECTED_ONNX_SHA256}\nActual:   {actual_sha}"
    )


def test_bundle_checksums_all_files():
    """Verify all files indexed in checksums.json exist and match recorded hashes."""
    assert CHECKSUMS_JSON.exists(), "checksums.json missing"
    with open(CHECKSUMS_JSON, "r", encoding="utf-8") as f:
        chk = json.load(f)

    assert len(chk) >= 10, f"Expected at least 10 checksum entries, got {len(chk)}"
    for rel_path, expected_hash in chk.items():
        fpath = BUNDLE_DIR / rel_path
        assert fpath.exists(), f"Bundle file missing: {rel_path}"
        actual_hash = compute_sha256(fpath)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}:\nExpected: {expected_hash}\nActual:   {actual_hash}"


def test_infer_cli_parity_and_latency():
    """Verify PyTorch and ONNX inference parity and < 50ms latency on development fixtures."""
    sys.path.insert(0, str(BUNDLE_DIR))
    from infer import VietnameseKWSInferencer

    pt_engine = VietnameseKWSInferencer(backend="pytorch", model_path=CHECKPOINT_PATH)
    onnx_engine = VietnameseKWSInferencer(backend="onnx", model_path=ONNX_PATH)

    dev_files = sorted(list(DEV_DIR.glob("*.wav")))
    assert len(dev_files) >= 5, f"Expected >= 5 fixtures, got {len(dev_files)}"

    for f in dev_files:
        pt_res = pt_engine.predict(f)
        onnx_res = onnx_engine.predict(f)

        assert pt_res["predicted_class"] == onnx_res["predicted_class"], (
            f"Prediction mismatch on {f.name}: PyTorch={pt_res['predicted_class']} vs ONNX={onnx_res['predicted_class']}"
        )
        assert pt_res["latency_ms"] <= 50.0, f"PyTorch latency {pt_res['latency_ms']} ms exceeded 50ms SLA"
        assert onnx_res["latency_ms"] <= 50.0, f"ONNX latency {onnx_res['latency_ms']} ms exceeded 50ms SLA"


def test_governance_invariants():
    """Verify authoritative governance labels in release reports and model configs."""
    assert ROOT_REPORT.exists()
    assert EXP_REPORT.exists()
    assert MODEL_CONFIG_JSON.exists()
    assert MODEL_CARD_MD.exists()

    content_root = ROOT_REPORT.read_text(encoding="utf-8")
    content_exp = EXP_REPORT.read_text(encoding="utf-8")
    content_card = MODEL_CARD_MD.read_text(encoding="utf-8")

    with open(MODEL_CONFIG_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Required positive labels
    for text in [content_root, content_exp, content_card]:
        assert "FROZEN_CNN_RELEASE_CANDIDATE" in text
        assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in text
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in text
        assert EXPECTED_CHECKPOINT_SHA256 in text
        assert EXPECTED_ONNX_SHA256 in text

        # Prohibited overstatement labels
        assert "NATURAL_SPEECH_VALIDATED" not in text
        assert "OPEN_MIC_DEPLOYABLE" not in text

    assert "FROZEN_CNN_RELEASE_CANDIDATE" in cfg["governance_tags"]
    assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in cfg["governance_tags"]
    assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in cfg["governance_tags"]
