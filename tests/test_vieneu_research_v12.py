#!/usr/bin/env python3
"""
Unit and Regression Test Suite for Vietnamese KWS Phase 12 Candidate Finalization.
Workspace: D:\trainidentificationaudio
Target: tests/test_vieneu_research_v12.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
BUNDLE_DIR = V12_DIR / "bundle"
CKPT_DIR = BUNDLE_DIR / "checkpoint"
ONNX_DIR = V12_DIR / "onnx"
FIXTURES_DIR = V12_DIR / "fixtures"

V11_ORIGINAL_CKPT = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v11"
    / "checkpoints"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
BUNDLE_CKPT = CKPT_DIR / "variant_v11_hard_negative_weighted_seed42.pt"
EXPECTED_CKPT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

V9_BASELINE_CKPT = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v9"
    / "artifacts"
    / "checkpoints"
    / "variant_w10_seed42.pt"
)
EXPECTED_V9_HASH = "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed"

from experiments.vieneu_kws_research_v12.bundle.infer import (
    KWSInferenceEngine,
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    preprocess_audio_file,
    CLASSES,
    TARGET_CLASSES,
    DEFAULT_THRESHOLD
)
from experiments.vieneu_kws_research_v12.onnx.export_onnx_v12 import ExportableVietnameseKWSNet


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_integrity_and_hash_resolution():
    """Verify original v11 and bundled checkpoints exist and have identical verified SHA-256."""
    assert V11_ORIGINAL_CKPT.exists(), f"Missing original v11 checkpoint: {V11_ORIGINAL_CKPT}"
    h_orig = sha256_of_file(V11_ORIGINAL_CKPT)
    assert h_orig.lower() == EXPECTED_CKPT_HASH.lower(), (
        f"Original v11 hash mismatch: expected {EXPECTED_CKPT_HASH}, got {h_orig}"
    )

    assert BUNDLE_CKPT.exists(), f"Missing bundled checkpoint: {BUNDLE_CKPT}"
    h_bundle = sha256_of_file(BUNDLE_CKPT)
    assert h_bundle.lower() == EXPECTED_CKPT_HASH.lower(), (
        f"Bundled checkpoint hash mismatch: expected {EXPECTED_CKPT_HASH}, got {h_bundle}"
    )


def test_historical_v9_checkpoint_immutability():
    """Ensure baseline v9 checkpoint was not modified during finalization."""
    assert V9_BASELINE_CKPT.exists(), f"Missing v9 baseline checkpoint: {V9_BASELINE_CKPT}"
    h_v9 = sha256_of_file(V9_BASELINE_CKPT)
    assert h_v9.lower() == EXPECTED_V9_HASH.lower(), f"v9 baseline checkpoint modified: {h_v9}"


def test_bundle_artifacts_completeness():
    """Verify that all required candidate bundle artifacts exist and are well-formed."""
    checksum_file = CKPT_DIR / "checksums.sha256"
    assert checksum_file.exists(), f"Missing checksum file: {checksum_file}"
    checksum_text = checksum_file.read_text(encoding="utf-8")
    assert EXPECTED_CKPT_HASH in checksum_text

    model_config_file = BUNDLE_DIR / "model_config.json"
    assert model_config_file.exists(), f"Missing model_config.json: {model_config_file}"
    with open(model_config_file, "r", encoding="utf-8") as f:
        mcfg = json.load(f)
    assert mcfg["model_name"] == "CorrectedVietnameseKWSNet"
    assert mcfg["classes"] == ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
    assert mcfg["calibrated_threshold"] == 0.20
    assert mcfg["parameters"] == 138469

    contract_file = BUNDLE_DIR / "preprocessing_contract.json"
    assert contract_file.exists(), f"Missing preprocessing_contract.json: {contract_file}"
    with open(contract_file, "r", encoding="utf-8") as f:
        pcfg = json.load(f)
    assert pcfg["audio_input"]["sample_rate_hz"] == 16000
    assert pcfg["audio_input"]["num_samples"] == 16000
    assert pcfg["log_mel_spectrogram"]["n_mels"] == 40

    assert (BUNDLE_DIR / "requirements.txt").exists()
    assert (BUNDLE_DIR / "model_card.md").exists()
    assert (BUNDLE_DIR / "benchmark_latency.py").exists()
    assert (BUNDLE_DIR / "infer.py").exists()


def test_deterministic_inference_reproducibility():
    """Verify that running inference twice on the same WAV file produces bitwise-identical results."""
    test_wav = FIXTURES_DIR / "parity_dev_01_trai.wav"
    assert test_wav.exists(), f"Missing dev fixture: {test_wav}"

    engine = KWSInferenceEngine(checkpoint_path=BUNDLE_CKPT, theta_cmd=0.20, device="cpu")
    pred1 = engine.predict(test_wav)
    pred2 = engine.predict(test_wav)

    assert pred1["predicted_class"] == pred2["predicted_class"]
    assert pred1["confidence"] == pred2["confidence"]
    assert pred1["rejection_status"] == pred2["rejection_status"]
    assert pred1["probabilities"] == pred2["probabilities"]


def test_rejection_logic():
    """Verify that commands below threshold theta_cmd are classified as REJECT."""
    engine = KWSInferenceEngine(checkpoint_path=BUNDLE_CKPT, theta_cmd=0.20, device="cpu")

    # Fixture 8 (transient impulse) and Fixture 10 (short padded) have max command prob < 0.20
    impulse_wav = FIXTURES_DIR / "parity_dev_08_transient_impulse.wav"
    assert impulse_wav.exists()
    res = engine.predict(impulse_wav)
    assert res["predicted_class"] == "REJECT"
    assert res["rejection_status"] == "REJECTED_LOW_CONFIDENCE"
    assert res["confidence"] < 0.20


def test_onnx_export_and_numerical_parity():
    """Verify ONNX FP32 export matches PyTorch reference with max absolute prob diff <= 1e-4."""
    onnx_fp32_file = ONNX_DIR / "variant_v11_kws_model_fp32.onnx"
    assert onnx_fp32_file.exists(), f"Missing ONNX model: {onnx_fp32_file}"

    # Verify model structure with ONNX checker
    model_proto = onnx.load(str(onnx_fp32_file))
    onnx.checker.check_model(model_proto)

    manifest_file = ONNX_DIR / "onnx_export_manifest.json"
    assert manifest_file.exists(), f"Missing ONNX export manifest: {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "PARITY_VERIFIED"
    assert manifest["models"]["onnx_fp32"]["parity_passed"] is True
    assert manifest["models"]["onnx_fp32"]["label_mismatch_count"] == 0
    assert manifest["models"]["onnx_fp32"]["max_prob_deviation"] <= 1e-4

    # Run live parity check on fixtures
    ort_session = ort.InferenceSession(str(onnx_fp32_file), providers=["CPUExecutionProvider"])
    frontend = LogMelFrontend().eval()

    base_model = CorrectedVietnameseKWSNet(num_classes=5)
    ckpt = torch.load(BUNDLE_CKPT, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base_model.load_state_dict(state_dict)
    base_model.eval()
    export_model = ExportableVietnameseKWSNet(base_model).eval()

    fixtures = sorted(FIXTURES_DIR.glob("*.wav"))
    assert len(fixtures) == 10

    max_prob_diff = 0.0
    for wav_path in fixtures:
        audio = preprocess_audio_file(wav_path)
        x = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            log_mel = frontend(x)
            py_logits, py_probs = export_model(log_mel)

        py_probs_np = py_probs.numpy()[0]
        onnx_outs = ort_session.run(None, {"log_mel": log_mel.numpy()})
        onnx_probs_np = onnx_outs[1][0]

        diff = float(np.max(np.abs(py_probs_np - onnx_probs_np)))
        max_prob_diff = max(max_prob_diff, diff)

        py_cmd_max = float(py_probs_np[:4].max())
        onnx_cmd_max = float(onnx_probs_np[:4].max())
        py_label = TARGET_CLASSES[py_probs_np[:4].argmax()] if py_cmd_max >= DEFAULT_THRESHOLD else "REJECT"
        onnx_label = TARGET_CLASSES[onnx_probs_np[:4].argmax()] if onnx_cmd_max >= DEFAULT_THRESHOLD else "REJECT"
        assert py_label == onnx_label, f"Label mismatch on {wav_path.name}: PyTorch={py_label}, ONNX={onnx_label}"

    assert max_prob_diff <= 1e-4, f"Max probability deviation ({max_prob_diff}) exceeds tolerance 1e-4"


def test_latency_benchmark_budget():
    """Verify single-clip latency meets budget (< 10 ms)."""
    bench_file = BUNDLE_DIR / "latency_benchmark.json"
    assert bench_file.exists(), f"Missing latency benchmark file: {bench_file}"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench = json.load(f)

    assert bench["mean_ms"] < 10.0, f"Latency {bench['mean_ms']} ms exceeds 10 ms limit"
    assert bench["p95_ms"] < 15.0, f"p95 Latency {bench['p95_ms']} ms exceeds 15 ms limit"


def test_phase12_governance_invariants():
    """Ensure no forbidden strings exist in Phase 12 artifacts."""
    forbidden = "data/" + "raw/" + "field_recordings"
    for p in V12_DIR.rglob("*"):
        if p.is_file() and p.suffix in [".py", ".md", ".json"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert forbidden not in text, f"Forbidden path string detected in {p}"
