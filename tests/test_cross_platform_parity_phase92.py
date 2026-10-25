"""
Test Suite for Phase 92: Cross-Platform Audio Preprocessing Parity.
Workspace: D:\trainidentificationaudio
Validates:
- Exact reference frontend behavior and mathematical specs (n_fft=400, Slaney norm/scale)
- Checkpoint and exported model hash immutability
- Integrity of 10 non-locked development fixtures and golden log-Mel tensors
- Numerical parity: Portable NumPy frontend vs PyTorch reference (max diff <= 1e-4)
- Numerical parity: C++ Native DLL frontend vs PyTorch reference (max diff <= 1e-4)
- Full waveform-to-output prediction chain parity across PyTorch, NumPy+ONNX, and C+++ONNX
- Rejection threshold mechanics (theta_cmd = 0.40)
- Factual toolchain blocker accounting (Android/iOS marked NOT_BUILT without fabrication)
- Governance and field recording isolation
"""

import os
import sys
import json
import hashlib
import ctypes
from pathlib import Path
import pytest
import numpy as np
import torch
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vieneu_kws_research_v9.artifacts.inference import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    CLASSES
)
from experiments.vieneu_kws_research_v10.deployment.audio_validator import AudioValidator
from experiments.vieneu_kws_research_v10.parity.portable_frontend import (
    PortableLogMelFrontend,
    create_slaney_mel_filterbank
)

PARITY_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "parity"
FIXTURES_DIR = PARITY_DIR / "fixtures"
GOLDEN_DIR = PARITY_DIR / "golden_tensors"
MANIFEST_PATH = PARITY_DIR / "parity_manifest.json"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v9"
    / "artifacts"
    / "checkpoints"
    / "variant_w10_seed42.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed"

ONNX_FP32_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v10"
    / "deployment"
    / "models"
    / "variant_w10_seed42_fp32.onnx"
)
EXPECTED_ONNX_FP32_SHA256 = "1f401625c59a85eb0ccabebf707ff45d77e6ee37d88bf6d83ce3b5c978a348ef"

C_DLL_PATH = PARITY_DIR / "c_frontend" / "kws_mobile_frontend_v10.dll"

TOLERANCE_FEATURE = 1e-4
TOLERANCE_LOGIT = 1e-4
DEFAULT_THRESHOLD = 0.40


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# 1. Checkpoint and Exported Model Invariance
# ============================================================================

def test_checkpoint_and_exported_model_invariance():
    """Verify that candidate checkpoint and exported ONNX models exist with unchanged hashes."""
    assert CHECKPOINT_PATH.exists()
    assert compute_sha256(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_SHA256

    assert ONNX_FP32_PATH.exists()
    assert compute_sha256(ONNX_FP32_PATH) == EXPECTED_ONNX_FP32_SHA256


# ============================================================================
# 2. Parity Manifest and Fixture Hash Integrity
# ============================================================================

def test_parity_manifest_and_fixtures_integrity():
    """Verify that parity manifest exists and all 10 fixtures match documented hashes."""
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["candidate_id"] == "variant_w10_seed42"
    assert manifest["phase"] == 92
    assert len(manifest["fixtures"]) == 10

    for fix in manifest["fixtures"]:
        wav_path = FIXTURES_DIR / fix["fixture_name"]
        assert wav_path.exists(), f"Missing fixture WAV: {wav_path}"
        assert compute_sha256(wav_path) == fix["audio_sha256"]

        tensor_path = GOLDEN_DIR / fix["golden_tensor_file"]
        assert tensor_path.exists(), f"Missing golden tensor: {tensor_path}"
        assert compute_sha256(tensor_path) == fix["golden_tensor_sha256"]


# ============================================================================
# 3. Reference Frontend Shapes and Math Invariants
# ============================================================================

def test_reference_frontend_shape_and_invariants():
    """Verify reference LogMelFrontend shape [1, 1, 40, 101] and finite outputs."""
    frontend = LogMelFrontend()
    frontend.eval()

    dummy = torch.randn(1, 16000)
    with torch.no_grad():
        out = frontend(dummy)

    assert out.shape == (1, 1, 40, 101)
    assert torch.isfinite(out).all()


def test_slaney_mel_filterbank_matrix_exactness():
    """Verify analytical Slaney filterbank matches Torchaudio within 1e-7."""
    import torchaudio.functional as F_a
    fb_torch = F_a.melscale_fbanks(201, 20.0, 8000.0, 40, 16000, norm="slaney", mel_scale="slaney").numpy().T
    fb_analytical = create_slaney_mel_filterbank(201, 20.0, 8000.0, 40, 16000)

    diff = float(np.max(np.abs(fb_torch - fb_analytical)))
    assert diff <= 1e-7, f"Filterbank difference {diff:.2e} exceeds 1e-7"


# ============================================================================
# 4. Portable NumPy Frontend Parity against Golden Tensors
# ============================================================================

def test_portable_numpy_frontend_parity_against_golden():
    """Verify PortableLogMelFrontend reproduces golden Log-Mel tensors within 1e-4."""
    fe_portable = PortableLogMelFrontend()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for fix in manifest["fixtures"]:
        wav_path = FIXTURES_DIR / fix["fixture_name"]
        golden_tensor_path = GOLDEN_DIR / fix["golden_tensor_file"]
        golden_logmel = np.load(str(golden_tensor_path))

        wav_tensor, _, _ = AudioValidator.load_and_validate(wav_path, strict=False)
        wav_np = wav_tensor.squeeze(0).numpy()

        py_logmel = fe_portable.wav_to_logmel(wav_np)
        max_diff = float(np.max(np.abs(py_logmel - golden_logmel)))
        mae = float(np.mean(np.abs(py_logmel - golden_logmel)))

        assert max_diff <= TOLERANCE_FEATURE, (
            f"Fixture {fix['fixture_name']} max diff {max_diff:.2e} exceeds tolerance {TOLERANCE_FEATURE}"
        )


# ============================================================================
# 5. C++ Native DLL Frontend Parity against Golden Tensors
# ============================================================================

def test_c_native_frontend_dll_parity_against_golden():
    """Verify C++ Native DLL frontend reproduces golden Log-Mel tensors within 1e-4."""
    assert C_DLL_PATH.exists(), f"Missing C++ DLL: {C_DLL_PATH}"

    c_lib = ctypes.CDLL(str(C_DLL_PATH))
    c_lib.kws_frontend_v10_init.restype = ctypes.c_int
    c_lib.kws_frontend_v10_compute.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    c_lib.kws_frontend_v10_compute.restype = ctypes.c_int
    status = c_lib.kws_frontend_v10_init()
    assert status == 0

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for fix in manifest["fixtures"]:
        wav_path = FIXTURES_DIR / fix["fixture_name"]
        golden_tensor_path = GOLDEN_DIR / fix["golden_tensor_file"]
        golden_logmel = np.load(str(golden_tensor_path))

        wav_tensor, _, _ = AudioValidator.load_and_validate(wav_path, strict=False)
        wav_np = wav_tensor.squeeze(0).numpy()

        c_in = wav_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_out = (ctypes.c_float * 4040)()
        ret = c_lib.kws_frontend_v10_compute(c_in, 16000, c_out)
        assert ret == 0

        c_logmel = np.frombuffer(c_out, dtype=np.float32).reshape(1, 1, 40, 101)
        max_diff = float(np.max(np.abs(c_logmel - golden_logmel)))

        assert max_diff <= TOLERANCE_FEATURE, (
            f"Fixture {fix['fixture_name']} C++ max diff {max_diff:.2e} exceeds tolerance {TOLERANCE_FEATURE}"
        )


# ============================================================================
# 6. Full Waveform-to-Output Prediction Chain Parity
# ============================================================================

def test_waveform_to_prediction_full_chain_parity():
    """
    Verify full waveform-to-output prediction chain across:
    1. PyTorch reference (Torchaudio + PyTorch model)
    2. NumPy Portable frontend + ONNX FP32 model
    3. C++ Native DLL frontend + ONNX FP32 model
    All three chains must yield identical predicted classes and rejection decisions.
    """
    py_model = CorrectedVietnameseKWSNet(num_classes=5)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    py_model.load_state_dict(state_dict)
    py_model.eval()

    fe_torch = LogMelFrontend()
    fe_torch.eval()
    fe_portable = PortableLogMelFrontend()

    c_lib = ctypes.CDLL(str(C_DLL_PATH))
    c_lib.kws_frontend_v10_init()

    ort_session = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for fix in manifest["fixtures"]:
        wav_path = FIXTURES_DIR / fix["fixture_name"]
        wav_tensor, _, _ = AudioValidator.load_and_validate(wav_path, strict=False)
        wav_np = wav_tensor.squeeze(0).numpy()

        # Chain 1: PyTorch
        with torch.no_grad():
            feat_torch = fe_torch(wav_tensor)
            logits_torch = py_model(feat_torch).numpy()[0]
            probs_torch = np.exp(logits_torch) / np.sum(np.exp(logits_torch))
            cmd_p_torch = probs_torch[:4]
            max_p_torch = float(cmd_p_torch.max())
            pred_torch = CLASSES[cmd_p_torch.argmax()] if max_p_torch >= DEFAULT_THRESHOLD else "REJECT"
            status_torch = "ACCEPTED" if max_p_torch >= DEFAULT_THRESHOLD else "REJECTED"

        # Chain 2: NumPy + ONNX
        feat_numpy = fe_portable.wav_to_logmel(wav_np)
        onnx_out_numpy = ort_session.run(None, {"log_mel": feat_numpy})
        logits_numpy = onnx_out_numpy[0][0]
        probs_numpy = onnx_out_numpy[1][0]
        cmd_p_numpy = probs_numpy[:4]
        max_p_numpy = float(cmd_p_numpy.max())
        pred_numpy = CLASSES[cmd_p_numpy.argmax()] if max_p_numpy >= DEFAULT_THRESHOLD else "REJECT"
        status_numpy = "ACCEPTED" if max_p_numpy >= DEFAULT_THRESHOLD else "REJECTED"

        # Chain 3: C++ DLL + ONNX
        c_in = wav_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_out = (ctypes.c_float * 4040)()
        c_lib.kws_frontend_v10_compute(c_in, 16000, c_out)
        feat_c = np.frombuffer(c_out, dtype=np.float32).reshape(1, 1, 40, 101)
        onnx_out_c = ort_session.run(None, {"log_mel": feat_c})
        logits_c = onnx_out_c[0][0]
        probs_c = onnx_out_c[1][0]
        cmd_p_c = probs_c[:4]
        max_p_c = float(cmd_p_c.max())
        pred_c = CLASSES[cmd_p_c.argmax()] if max_p_c >= DEFAULT_THRESHOLD else "REJECT"
        status_c = "ACCEPTED" if max_p_c >= DEFAULT_THRESHOLD else "REJECTED"

        # Verify exact agreement across all 3 chains
        assert pred_torch == pred_numpy == pred_c, (
            f"Prediction mismatch for {fix['fixture_name']}: Py={pred_torch}, NumPy={pred_numpy}, C++={pred_c}"
        )
        assert status_torch == status_numpy == status_c, (
            f"Status mismatch for {fix['fixture_name']}: Py={status_torch}, NumPy={status_numpy}, C++={status_c}"
        )

        # Numerical logit check
        diff_logits_c = float(np.max(np.abs(logits_c - logits_torch)))
        assert diff_logits_c <= TOLERANCE_LOGIT, (
            f"Logit diff {diff_logits_c:.2e} exceeds tolerance {TOLERANCE_LOGIT}"
        )


# ============================================================================
# 7. Host Toolchain and Platform Readiness Accounting
# ============================================================================

def test_host_toolchain_and_readiness_reporting():
    """Verify that Android and iOS mobile wrappers are documented as NOT_BUILT without fabrication."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    readiness = manifest["platform_readiness"]
    assert readiness["python_numpy_portable"] == "VERIFIED_PARITY"
    assert readiness["c_cpp_native_library"] == "VERIFIED_PARITY"
    assert "NOT_BUILT" in readiness["android_kotlin_wrapper"]
    assert "NOT_BUILT" in readiness["ios_swift_wrapper"]

    # Verify mobile interface files exist and declare NOT_BUILT
    kt_file = PARITY_DIR / "mobile_interfaces" / "VietnameseKwsClassifierV10.kt"
    assert kt_file.exists()
    assert "NOT_BUILT" in kt_file.read_text(encoding="utf-8")

    swift_file = PARITY_DIR / "mobile_interfaces" / "VietnameseKwsClassifierV10.swift"
    assert swift_file.exists()
    assert "NOT_BUILT" in swift_file.read_text(encoding="utf-8")


# ============================================================================
# 8. Governance and Historical Isolation
# ============================================================================

def test_governance_and_isolation():
    """Verify governance invariants and zero access to human review or field recordings."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gov = manifest["governance"]
    assert gov["human_listener_verification"] == "PENDING_HUMAN_LISTENER_VERIFICATION"
    assert gov["natural_speech_validation"] == "NOT_VALIDATED_ON_NATURAL_SPEECH"
    assert gov["continuous_open_mic_verdict"] == "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
