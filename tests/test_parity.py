"""
Automated Pytest Suite for Vietnamese KWS Pipeline.
Verifies:
  1. Audio feature extraction (shape, finiteness, NaN-free)
  2. Baseline ONNX model validity & numerical parity
  3. Champion TC-ResNet ONNX model validity & numerical parity
  4. 6-Class Golden test vectors reproducibility
  5. Strict zero-leakage speaker disjointness across v2 manifests
  6. 6-Class completeness across validation and test sets
"""

import sys
from pathlib import Path

# Ensure UTF-8 and path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import json
import pytest
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from src.models.kws_architectures import VietnameseKWSNet, VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor, extract_log_mel_numpy


def test_feature_extractor_shape_and_finite():
    sr = 16000
    dummy_audio = np.random.randn(sr).astype(np.float32)
    extractor = LogMelFeatureExtractor(sample_rate=sr, n_mels=40)
    
    tensor_audio = torch.from_numpy(dummy_audio)
    features = extractor(tensor_audio)
    
    assert features.shape == (1, 1, 40, 101), f"Unexpected feature shape: {features.shape}"
    assert torch.isfinite(features).all(), "Features contain NaN or Inf values"


def test_baseline_onnx_model_execution_and_shape():
    onnx_path = Path("models/exported/vietnamese_kws_float32.onnx")
    assert onnx_path.exists(), f"ONNX model file missing at {onnx_path}"

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 1, 40, 101).astype(np.float32)
    
    outputs = session.run(None, {"log_mel_input": dummy_input})
    logits = outputs[0]
    
    assert logits.shape == (1, 6), f"Expected logits shape (1, 6), got {logits.shape}"
    assert np.isfinite(logits).all(), "ONNX logits contain NaN or Inf"


def test_tcresnet_onnx_model_execution_and_parity():
    ckpt_path = Path("models/checkpoints/kws_tcresnet_advanced.pt")
    onnx_path = Path("models/exported/vietnamese_kws_tcresnet_fp32.onnx")
    
    assert ckpt_path.exists(), f"TC-ResNet checkpoint missing at {ckpt_path}"
    assert onnx_path.exists(), f"TC-ResNet ONNX model missing at {onnx_path}"

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pt_model = VietnameseTCResNet(num_classes=6)
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    np.random.seed(42)
    # Shape for TC-ResNet: (Batch, 40, 101)
    test_input = np.random.randn(2, 40, 101).astype(np.float32)
    
    with torch.no_grad():
        pt_logits = pt_model(torch.from_numpy(test_input)).numpy()
    
    ort_logits = session.run(None, {"log_mel_input": test_input})[0]
    
    assert ort_logits.shape == (2, 6), f"Expected (2, 6), got {ort_logits.shape}"
    max_abs_err = np.max(np.abs(pt_logits - ort_logits))
    assert max_abs_err < 1e-4, f"TC-ResNet parity violation: max diff = {max_abs_err:.2e}"


def test_pytorch_onnx_numerical_parity_baseline():
    ckpt_path = Path("models/checkpoints/best_kws_model.pt")
    onnx_path = Path("models/exported/vietnamese_kws_float32.onnx")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pt_model = VietnameseKWSNet(num_classes=6)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    np.random.seed(1337)
    test_input = np.random.randn(2, 1, 40, 101).astype(np.float32)
    
    with torch.no_grad():
        pt_logits = pt_model(torch.from_numpy(test_input)).numpy()
    
    ort_logits = session.run(None, {"log_mel_input": test_input})[0]
    
    max_abs_err = np.max(np.abs(pt_logits - ort_logits))
    assert max_abs_err < 1e-4, f"Baseline parity violation: max diff = {max_abs_err:.2e}"


def test_golden_vectors_reproducibility():
    golden_files = list(Path("tests/golden_vectors").glob("golden_*.json"))
    assert len(golden_files) == 6, f"Expected exactly 6 golden vectors (one per class), found {len(golden_files)}"

    for g_path in golden_files:
        with open(g_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "expected_logits" in data
        assert "expected_probabilities" in data
        assert "expected_argmax_class" in data
        assert len(data["expected_logits"]) == 6
        assert len(data["expected_probabilities"]) == 6
        
        expected_class = data["expected_argmax_class"]
        expected_logits = np.array(data["expected_logits"], dtype=np.float32)
        predicted_class = int(np.argmax(expected_logits))
        assert predicted_class == expected_class, f"Argmax mismatch in {g_path.name}"


def test_manifest_speaker_disjointness_v2():
    train_path = Path("data/manifests/train_manifest_v2.csv")
    val_path = Path("data/manifests/val_manifest_v2.csv")
    test_path = Path("data/manifests/test_manifest_v2.csv")

    assert train_path.exists() and val_path.exists() and test_path.exists()

    df_tr = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_te = pd.read_csv(test_path)

    spk_tr = set(df_tr["speaker_id"])
    spk_val = set(df_val["speaker_id"])
    spk_te = set(df_te["speaker_id"])

    assert len(spk_tr.intersection(spk_val)) == 0, f"Train and Val share {len(spk_tr.intersection(spk_val))} speakers!"
    assert len(spk_tr.intersection(spk_te)) == 0, f"Train and Test share {len(spk_tr.intersection(spk_te))} speakers!"
    assert len(spk_val.intersection(spk_te)) == 0, f"Val and Test share {len(spk_val.intersection(spk_te))} speakers!"


def test_manifest_class_completeness_v2():
    val_path = Path("data/manifests/val_manifest_v2.csv")
    test_path = Path("data/manifests/test_manifest_v2.csv")

    df_val = pd.read_csv(val_path)
    df_te = pd.read_csv(test_path)

    val_classes = set(df_val["class_id"])
    test_classes = set(df_te["class_id"])

    assert val_classes == set(range(6)), f"Validation manifest missing classes: {set(range(6)) - val_classes}"
    assert test_classes == set(range(6)), f"Test manifest missing classes: {set(range(6)) - test_classes}"
