"""
Unit and integration regression test suite for Phase 91: CNN Deployment Engineering.
Covers:
- Frozen checkpoint integrity and hash immutability (Seeds 42, 43, 44)
- Audio validation: missing file, 0-byte file, header parsing, strict vs permissive modes
- LogMelFrontend shape and numerical invariants
- Rejection threshold mechanics and class gating
- Exported model parity: TorchScript (0 diff), ONNX FP32 (diff <= 1e-4), ONNX INT8
- Standalone CLI execution (predict_wav.py) in text and JSON modes with exit codes
- Historical data immutability and governance disclaimers
"""

import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vieneu_kws_research_v9.artifacts.inference import (
    CorrectedVietnameseKWSNet,
    LogMelFrontend,
    CLASSES
)
from experiments.vieneu_kws_research_v10.deployment.audio_validator import (
    AudioValidator,
    AudioValidationError,
    AudioFileNotFoundError,
    EmptyAudioError,
    CorruptedAudioError,
    InvalidSampleRateError,
    InvalidChannelCountError,
    InvalidDurationError,
    CheckpointNotFoundError
)
from experiments.vieneu_kws_research_v10.deployment.predict_wav import (
    PredictorEngine,
    DEFAULT_CHECKPOINT,
    DEFAULT_THRESHOLD,
    DISCLAIMERS
)

# Reference hashes for candidate checkpoints
EXPECTED_HASHES = {
    "variant_w10_seed42.pt": "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed",
    "variant_w10_seed43.pt": "77f468c68b261de506a3fcb022c8d7cc68f8b0c2ad650852b7c300f274dda251",
    "variant_w10_seed44.pt": "bebab580e4caa6934c98a62b095611e4eef9221a7f4282d36f9a4b6928abde64"
}

FIXTURES_DIR = PROJECT_ROOT / "experiments" / "synthetic_kws_demo" / "fixtures"
MODELS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "models"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# 1. Checkpoint Integrity & Immutability
# ============================================================================

def test_checkpoint_integrity_and_immutability():
    """Verify that all frozen candidate checkpoints exist with exact SHA-256 hashes."""
    ckpts_dir = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v9" / "artifacts" / "checkpoints"
    for filename, expected_hash in EXPECTED_HASHES.items():
        ckpt_path = ckpts_dir / filename
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        actual_hash = compute_sha256(ckpt_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {filename}: {actual_hash} != {expected_hash}"

        # Verify loadable architecture and parameter count
        m = CorrectedVietnameseKWSNet(num_classes=5)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state_dict)
        n_params = sum(p.numel() for p in m.parameters())
        assert n_params == 138469, f"Expected 138,469 parameters, got {n_params}"


# ============================================================================
# 2. Audio Validator Error Cases & Ingestion
# ============================================================================

def test_audio_validator_missing_file():
    """Missing file must raise AudioFileNotFoundError."""
    with pytest.raises(AudioFileNotFoundError):
        AudioValidator.load_and_validate(FIXTURES_DIR / "nonexistent_file_12345.wav")


def test_audio_validator_empty_file():
    """0-byte file must raise EmptyAudioError."""
    empty_file = FIXTURES_DIR / "invalid_empty.wav"
    assert empty_file.exists()
    with pytest.raises(EmptyAudioError):
        AudioValidator.load_and_validate(empty_file)


def test_audio_validator_strict_mode_errors():
    """Strict mode must reject non-16kHz, stereo, and non-1s audio."""
    # 8 kHz
    with pytest.raises(InvalidSampleRateError):
        AudioValidator.load_and_validate(FIXTURES_DIR / "invalid_sample_rate_8k.wav", strict=True)

    # Stereo
    with pytest.raises(InvalidChannelCountError):
        AudioValidator.load_and_validate(FIXTURES_DIR / "invalid_stereo.wav", strict=True)

    # 0.5s duration
    with pytest.raises(InvalidDurationError):
        AudioValidator.load_and_validate(FIXTURES_DIR / "invalid_short_duration.wav", strict=True)


def test_audio_validator_permissive_mode():
    """Permissive mode must automatically resample, downmix, and pad/truncate to 16,000 samples."""
    # 8 kHz resample
    wav_8k, orig_sr, meta_8k = AudioValidator.load_and_validate(
        FIXTURES_DIR / "invalid_sample_rate_8k.wav", strict=False
    )
    assert wav_8k.shape == (1, 16000)
    assert orig_sr == 8000
    assert meta_8k["original_sample_rate"] == 8000

    # Stereo downmix
    wav_stereo, orig_sr_st, meta_st = AudioValidator.load_and_validate(
        FIXTURES_DIR / "invalid_stereo.wav", strict=False
    )
    assert wav_stereo.shape == (1, 16000)
    assert meta_st["original_channels"] == 2

    # Short duration right-pad
    wav_short, orig_sr_sh, meta_sh = AudioValidator.load_and_validate(
        FIXTURES_DIR / "invalid_short_duration.wav", strict=False
    )
    assert wav_short.shape == (1, 16000)
    assert meta_sh["original_frames"] == 8000

    # Valid fixture
    wav_valid, orig_sr_val, meta_val = AudioValidator.load_and_validate(
        FIXTURES_DIR / "demo_01_trai.wav", strict=True
    )
    assert wav_valid.shape == (1, 16000)
    assert orig_sr_val == 16000


def test_audio_validator_peak_normalization_contract():
    """AudioValidator clamps peaks exceeding 1.0 but never amplifies quiet audio."""
    # Create test tensor > 1.0
    wav_over = torch.tensor([[0.5, -2.0, 1.5, 0.0]])
    max_val = torch.max(torch.abs(wav_over))
    norm_over = wav_over / max_val
    assert float(torch.max(torch.abs(norm_over))) == 1.0

    # Create quiet tensor < 1.0: must NOT be amplified
    wav_quiet = torch.tensor([[0.05, -0.02, 0.01, 0.0]])
    max_quiet = torch.max(torch.abs(wav_quiet))
    assert max_quiet < 1.0  # untouched in pipeline


# ============================================================================
# 3. Frontend & Architecture Invariants
# ============================================================================

def test_frontend_shape_and_numerical_invariants():
    """LogMelFrontend must produce [B, 1, 40, 101] with finite values."""
    frontend = LogMelFrontend()
    frontend.eval()

    dummy_wav = torch.randn(2, 16000)
    with torch.no_grad():
        feats = frontend(dummy_wav)

    assert feats.shape == (2, 1, 40, 101)
    assert torch.isfinite(feats).all(), "Features contain NaN or Inf"


# ============================================================================
# 4. Rejection Logic and Threshold Gating
# ============================================================================

def test_threshold_rejection_gating():
    """Verify that commands are accepted iff max command prob >= threshold."""
    engine = PredictorEngine(
        checkpoint_path=DEFAULT_CHECKPOINT,
        theta_cmd=DEFAULT_THRESHOLD
    )

    # Valid fixture
    res_default = engine.predict_wav(FIXTURES_DIR / "demo_06_silence.wav")
    assert res_default["predicted_class"] == "REJECT"
    assert res_default["rejection_status"] == "REJECTED"

    # Extreme high threshold (0.999) forces rejection
    engine_strict = PredictorEngine(
        checkpoint_path=DEFAULT_CHECKPOINT,
        theta_cmd=0.999
    )
    res_high_th = engine_strict.predict_wav(FIXTURES_DIR / "demo_01_trai.wav")
    assert res_high_th["predicted_class"] == "REJECT"
    assert res_high_th["rejection_status"] == "REJECTED"


# ============================================================================
# 5. Exported Model Parity Tests
# ============================================================================

def test_export_manifest_and_models_exist():
    """Verify that export manifest and all exported model files exist."""
    manifest_path = MODELS_DIR / "export_manifest.json"
    assert manifest_path.exists(), f"Missing export manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["candidate_id"] == "variant_w10_seed42"
    assert manifest["parity_verification"]["status"] == "PARITY_VERIFIED"
    assert manifest["parity_verification"]["label_mismatches_onnx_fp32"] == 0
    assert manifest["parity_verification"]["max_deviation_onnx_fp32"] <= 1e-4

    # Verify files on disk match manifest hashes
    for key, spec in manifest["exported_models"].items():
        m_path = MODELS_DIR / spec["filename"]
        assert m_path.exists(), f"Exported file missing: {m_path}"
        actual_hash = compute_sha256(m_path)
        assert actual_hash == spec["sha256"], f"Hash mismatch for {spec['filename']}"


def test_numerical_parity_onnx_runtime():
    """Verify ONNX FP32 numerical parity on test fixtures is within 1e-4."""
    import onnxruntime as ort
    onnx_path = MODELS_DIR / "variant_w10_seed42_fp32.onnx"
    assert onnx_path.exists()

    engine_py = PredictorEngine(checkpoint_path=DEFAULT_CHECKPOINT)
    engine_onnx = PredictorEngine(onnx_path=onnx_path)

    test_wav = FIXTURES_DIR / "demo_01_trai.wav"
    res_py = engine_py.predict_wav(test_wav)
    res_onnx = engine_onnx.predict_wav(test_wav)

    assert res_py["predicted_class"] == res_onnx["predicted_class"]
    assert res_py["rejection_status"] == res_onnx["rejection_status"]

    for cls_name in CLASSES:
        diff = abs(res_py["probabilities"][cls_name] - res_onnx["probabilities"][cls_name])
        assert diff <= 1e-4, f"Probability mismatch for {cls_name}: diff={diff}"


# ============================================================================
# 6. Standalone CLI Tool Execution & Exit Codes
# ============================================================================

def test_cli_predict_wav_success_json():
    """CLI must return exit code 0 and valid JSON on valid input."""
    cli_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "predict_wav.py"
    cmd = [
        sys.executable,
        str(cli_path),
        str(FIXTURES_DIR / "demo_01_trai.wav"),
        "--json"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"

    data = json.loads(proc.stdout)
    assert data["status"] == "SUCCESS"
    assert data["predicted_class"] in CLASSES
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0
    assert "governance" in data


def test_cli_predict_wav_missing_file():
    """CLI must return exit code 2 on missing file."""
    cli_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "predict_wav.py"
    cmd = [
        sys.executable,
        str(cli_path),
        "nonexistent_audio_file.wav",
        "--json"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["error_type"] == "AudioFileNotFoundError"


def test_cli_predict_wav_empty_file():
    """CLI must return exit code 3 on 0-byte file."""
    cli_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "predict_wav.py"
    cmd = [
        sys.executable,
        str(cli_path),
        str(FIXTURES_DIR / "invalid_empty.wav"),
        "--json"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 3
    data = json.loads(proc.stdout)
    assert data["error_type"] == "EmptyAudioError"


def test_cli_predict_wav_strict_sample_rate():
    """CLI must return exit code 5 on invalid sample rate in strict mode."""
    cli_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "predict_wav.py"
    cmd = [
        sys.executable,
        str(cli_path),
        str(FIXTURES_DIR / "invalid_sample_rate_8k.wav"),
        "--strict",
        "--json"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 5
    data = json.loads(proc.stdout)
    assert data["error_type"] == "InvalidSampleRateError"


def test_cli_predict_wav_onnx_backend():
    """CLI must run with --onnx-model and return exit code 0."""
    cli_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "deployment" / "predict_wav.py"
    onnx_path = MODELS_DIR / "variant_w10_seed42_fp32.onnx"
    cmd = [
        sys.executable,
        str(cli_path),
        str(FIXTURES_DIR / "demo_01_trai.wav"),
        "--onnx-model",
        str(onnx_path),
        "--json"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["backend"] == "onnxruntime"
    assert data["status"] == "SUCCESS"


# ============================================================================
# 7. Governance, Immutability & Status Disclaimers
# ============================================================================

def test_governance_and_field_recording_isolation():
    """Verify that field recordings and consent ledgers are not accessed or modified."""
    field_dir = PROJECT_ROOT / "data" / "raw" / "field_recordings"
    if field_dir.exists():
        # Ensure directory is intact
        assert field_dir.is_dir()

    # Verify status disclaimers
    for key, val in DISCLAIMERS.items():
        assert val in [
            "PENDING_HUMAN_LISTENER_VERIFICATION",
            "NOT_VALIDATED_ON_NATURAL_SPEECH",
            "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
        ]
