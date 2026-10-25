"""
Regression Test Suite for KWS Model Exportability and Runtime Parity.
Complies with Phase 6 requirements:
  - Checkpoint integrity verification
  - ONNX graph validity and opset verification
  - Input/Output shape contracts and class ordering
  - Numerical parity across PyTorch and ONNX Runtime (max diff <= 1e-4)
  - Governance invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
"""

import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from experiments.synthetic_kws_demo.export.export_and_benchmark import (
    ExportableTCResNet,
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    ONNX_PATH,
    CONFIG_PATH,
    REPORT_PATH,
)
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.synthetic_kws_demo.classifier import AudioValidator, CLASS_LABELS


@pytest.fixture(scope="module")
def loaded_pytorch_model():
    """Returns the frozen PyTorch model in eval mode."""
    base = VietnameseTCResNet(num_classes=6, dropout_p=0.0)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base.load_state_dict(state_dict)
    base.eval()
    model = ExportableTCResNet(base)
    model.eval()
    return model


@pytest.fixture(scope="module")
def onnx_session():
    """Returns an active ONNX Runtime inference session."""
    assert ONNX_PATH.exists(), f"ONNX model missing: {ONNX_PATH}"
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(ONNX_PATH), opts, providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def feature_extractor():
    """Returns the canonical LogMelFeatureExtractor."""
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    extractor.eval()
    return extractor


# ============================================================================
# 1. Checkpoint & Artifact Integrity
# ============================================================================

def test_checkpoint_hash_integrity():
    """Verify source checkpoint exists and matches frozen SHA-256 exactly."""
    assert CHECKPOINT_PATH.exists(), f"Source checkpoint missing: {CHECKPOINT_PATH}"
    calc_sha = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest().upper()
    assert calc_sha == EXPECTED_CHECKPOINT_SHA256


def test_onnx_model_file_and_validity():
    """Verify ONNX graph exists, is non-empty, and passes onnx.checker."""
    assert ONNX_PATH.exists(), f"Exported ONNX file missing: {ONNX_PATH}"
    assert ONNX_PATH.stat().st_size > 500000, "ONNX model unexpectedly small"
    model_proto = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_proto)
    assert model_proto.opset_import[0].version == 17


def test_model_config_metadata_integrity():
    """Verify model_config.json contains required metadata and contracts."""
    assert CONFIG_PATH.exists(), f"Configuration record missing: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["source_checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert cfg["onnx_export"]["opset_version"] == 17
    assert len(cfg["onnx_export"]["classes"]) == 6
    assert cfg["mobile_preprocessing_contract"]["required_sample_rate"] == 16000
    assert cfg["mobile_preprocessing_contract"]["n_mels"] == 40
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in cfg["governance"]


def test_parity_report_verdict():
    """Verify parity_report.json exists and confirms PARITY_AUDIT_PASS."""
    assert REPORT_PATH.exists(), f"Parity report missing: {REPORT_PATH}"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["summary"]["status"] == "PARITY_AUDIT_PASS"
    assert rep["summary"]["parity_threshold_met"] is True
    assert rep["summary"]["global_max_abs_probability_diff"] <= 1e-4
    assert rep["summary"]["overall_agreement_rate"] == 1.0


# ============================================================================
# 2. ONNX Graph Input/Output Specifications & Label Order
# ============================================================================

def test_onnx_input_output_contract(onnx_session):
    """Verify input name 'log_mel' and output names 'logits' and 'probabilities'."""
    inputs = onnx_session.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "log_mel"
    assert inputs[0].type == "tensor(float)"
    # Shape should be [batch_size, 1, 40, 101]
    assert len(inputs[0].shape) == 4
    assert inputs[0].shape[1] == 1
    assert inputs[0].shape[2] == 40
    assert inputs[0].shape[3] == 101

    outputs = onnx_session.get_outputs()
    assert len(outputs) == 2
    output_names = [o.name for o in outputs]
    assert "logits" in output_names
    assert "probabilities" in output_names

    for o in outputs:
        assert o.type == "tensor(float)"
        assert len(o.shape) == 2
        assert o.shape[1] == 6


def test_label_ordering_contract():
    """Verify canonical 6-class index order is preserved."""
    expected_classes = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    assert CLASS_LABELS == expected_classes
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    classes_dict = cfg["onnx_export"]["classes"]
    for i, clabel in enumerate(expected_classes):
        assert classes_dict[str(i)] == clabel


# ============================================================================
# 3. Numerical Runtime Parity on Audio Fixtures & Synthetic Tensors
# ============================================================================

def test_parity_on_six_demo_fixtures(loaded_pytorch_model, onnx_session, feature_extractor):
    """Verify bit-level numerical parity (max diff <= 1e-4) on all 6 demo audio fixtures."""
    fixtures_dir = Path("experiments/synthetic_kws_demo/fixtures")
    demo_files = [
        ("demo_01_trai.wav", "TRAI"),
        ("demo_02_phai.wav", "PHAI"),
        ("demo_03_len.wav", "LEN"),
        ("demo_04_xuong.wav", "XUONG"),
        ("demo_05_unknown.wav", "UNKNOWN"),
        ("demo_06_silence.wav", "SILENCE"),
    ]

    for fname, expected_label in demo_files:
        wav_p = fixtures_dir / fname
        audio, sr = AudioValidator.validate_and_load(wav_p)
        audio_t = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel_t = feature_extractor(audio_t)
            pt_logits, pt_probs = loaded_pytorch_model(log_mel_t)

        pt_probs_np = pt_probs.cpu().numpy()[0]
        pt_pred_cid = int(np.argmax(pt_probs_np))

        ort_logits, ort_probs = onnx_session.run(None, {"log_mel": log_mel_t.numpy()})
        ort_probs_vec = ort_probs[0]
        ort_pred_cid = int(np.argmax(ort_probs_vec))

        # 1. Exact class agreement
        assert pt_pred_cid == ort_pred_cid
        assert CLASS_LABELS[pt_pred_cid] == expected_label

        # 2. Maximum absolute difference well within 1e-4
        max_diff = float(np.max(np.abs(pt_probs_np - ort_probs_vec)))
        assert max_diff <= 1e-4, f"Parity diff {max_diff:.2e} exceeded 1e-4 on {fname}"

        # 3. Probability normalization
        assert pytest.approx(float(np.sum(ort_probs_vec)), abs=1e-5) == 1.0
        assert (ort_probs_vec >= 0.0).all()


def test_parity_on_deterministic_synthetic_tensors(loaded_pytorch_model, onnx_session):
    """Verify parity across varied statistical regimes using fixed random seed."""
    rng = np.random.RandomState(20261016)
    
    for i in range(25):
        regime = i % 5
        if regime == 0:
            feat = rng.randn(1, 1, 40, 101).astype(np.float32)
        elif regime == 1:
            feat = rng.uniform(-10.0, 5.0, (1, 1, 40, 101)).astype(np.float32)
        elif regime == 2:
            feat = (rng.randn(1, 1, 40, 101) * 0.1 - 12.0).astype(np.float32)
        elif regime == 3:
            feat = np.zeros((1, 1, 40, 101), dtype=np.float32)
        else:
            feat = (rng.randn(1, 1, 40, 101) * 5.0).astype(np.float32)

        feat_t = torch.from_numpy(feat)
        with torch.no_grad():
            _, pt_probs = loaded_pytorch_model(feat_t)

        pt_probs_np = pt_probs.cpu().numpy()[0]
        _, ort_probs = onnx_session.run(None, {"log_mel": feat})
        ort_probs_vec = ort_probs[0]

        # Argmax equality
        assert np.argmax(pt_probs_np) == np.argmax(ort_probs_vec)

        # Numerical difference
        max_diff = float(np.max(np.abs(pt_probs_np - ort_probs_vec)))
        assert max_diff <= 1e-4, f"Parity diff {max_diff:.2e} exceeded 1e-4 on test tensor {i}"


def test_batch_dimension_support(onnx_session):
    """Verify exported ONNX model supports dynamic batch sizes (e.g. batch size 4)."""
    batch_feat = np.random.randn(4, 1, 40, 101).astype(np.float32)
    logits, probs = onnx_session.run(None, {"log_mel": batch_feat})
    assert logits.shape == (4, 6)
    assert probs.shape == (4, 6)
    for b in range(4):
        assert pytest.approx(float(np.sum(probs[b])), abs=1e-5) == 1.0


# ============================================================================
# 4. Phase 7 Golden Vectors & Frontend Contract Tests
# ============================================================================

def test_golden_manifest_and_discrepancy_log_exist():
    """Verify Phase 7 golden manifest, contract, discrepancy log, and parity report exist."""
    golden_dir = Path("experiments/synthetic_kws_demo/frontend_golden")
    assert golden_dir.exists()
    
    manifest_p = golden_dir / "golden_manifest.json"
    contract_p = golden_dir / "frontend_contract.md"
    disc_p = golden_dir / "discrepancy_log.json"
    rep_p = golden_dir / "feature_parity_report.json"
    
    assert manifest_p.exists(), "golden_manifest.json missing"
    assert contract_p.exists(), "frontend_contract.md missing"
    assert disc_p.exists(), "discrepancy_log.json missing"
    assert rep_p.exists(), "feature_parity_report.json missing"

    with open(manifest_p, "r", encoding="utf-8") as f:
        man = json.load(f)
    assert man["total_cases"] == 12
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in man["governance"]

    with open(disc_p, "r", encoding="utf-8") as f:
        disc = json.load(f)
    assert disc["discrepancies"][0]["id"] == "DISC-09"
    assert "Periodic Hann" in disc["discrepancies"][0]["resolution"]


def test_periodic_hann_window_mathematical_identity(feature_extractor):
    """Verify feature extractor window is strictly bit-exact to Periodic Hann Window."""
    import torchaudio.transforms as T
    mel_tf = feature_extractor.mel_transform
    w = mel_tf.spectrogram.window
    
    # Mathematical proof of periodic Hann window identity
    w_hann_periodic = torch.hann_window(400, periodic=True)
    w_hann_symmetric = torch.hann_window(400, periodic=False)
    w_hamming = torch.hamming_window(400)
    
    assert torch.max(torch.abs(w - w_hann_periodic)).item() == 0.0, "Window is not periodic Hann"
    assert torch.max(torch.abs(w - w_hann_symmetric)).item() > 0.005, "Window should not be symmetric Hann"
    assert torch.max(torch.abs(w - w_hamming)).item() > 0.07, "Window should not be Hamming"


def test_golden_vectors_exact_reproducibility(feature_extractor):
    """Verify all 12 golden vectors recompute with bit-exact parity against saved .npy tensors."""
    import soundfile as sf
    manifest_p = Path("experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        man = json.load(f)

    for case in man["cases"]:
        wav_p = Path(case["wav_artifact"]["path"])
        tensor_p = Path(case["tensor_artifact"]["path"])
        assert wav_p.exists(), f"Missing WAV: {wav_p}"
        assert tensor_p.exists(), f"Missing tensor: {tensor_p}"

        saved_tensor = np.load(str(tensor_p))
        assert saved_tensor.shape == (1, 1, 40, 101)
        assert not np.isnan(saved_tensor).any()
        assert not np.isinf(saved_tensor).any()

        audio, sr = sf.read(str(wav_p), dtype="float32")
        assert len(audio) == 16000 and sr == 16000
        with torch.no_grad():
            recomputed = feature_extractor(torch.from_numpy(audio)).numpy()

        # Recomputed tensor must match saved golden tensor to machine precision (<= 1e-6)
        max_diff = float(np.max(np.abs(recomputed - saved_tensor)))
        assert max_diff <= 1e-6, f"Feature diff {max_diff:.2e} exceeded 1e-6 on {case['case_id']}"


def test_golden_vectors_pytorch_onnx_parity(loaded_pytorch_model, onnx_session):
    """Verify PyTorch and ONNX Runtime produce identical argmax and prob diff <= 1e-4 on golden set."""
    manifest_p = Path("experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        man = json.load(f)

    for case in man["cases"]:
        tensor_p = Path(case["tensor_artifact"]["path"])
        tensor_np = np.load(str(tensor_p))

        with torch.no_grad():
            _, pt_probs = loaded_pytorch_model(torch.from_numpy(tensor_np))
        pt_probs_vec = pt_probs.numpy()[0]

        _, ort_probs = onnx_session.run(None, {"log_mel": tensor_np})
        ort_probs_vec = ort_probs[0]

        # 1. Argmax agreement
        assert np.argmax(pt_probs_vec) == np.argmax(ort_probs_vec)

        # 2. Probability difference <= 1e-4
        max_diff = float(np.max(np.abs(pt_probs_vec - ort_probs_vec)))
        assert max_diff <= 1e-4, f"Parity diff {max_diff:.2e} exceeded 1e-4 on {case['case_id']}"

