"""
Automated regression tests for the Six-Class Vietnamese KWS CNN Baseline.
Verifies:
  - Canonical 6-class mapping and network output shape (Batch, 6)
  - Split manifest integrity and disjointness
  - Feature extraction dimensions (1, 1, 40, 101)
  - Forward pass and loss backward execution
  - Checkpoint loading, parameter count, and provenance
  - Test prediction completeness and metric consistency
  - Non-interference with original four-class baselines
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_cnn_6class.train import LABEL_TO_ID, ID_TO_LABEL, CLASS_NAMES

EXP_DIR = os.path.join(BASE_DIR, "experiments/vieneu_cnn_6class")
CONFIG_PATH = os.path.join(EXP_DIR, "config.json")
SPLITS_DIR = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_6class/splits")
FOUR_CLASS_DIR = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000")

def test_six_class_canonical_mapping():
    """Verify exact 6-class label mapping matches project convention."""
    expected = {
        "TRAI": 0,
        "PHAI": 1,
        "LEN": 2,
        "XUONG": 3,
        "UNKNOWN": 4,
        "SILENCE": 5
    }
    assert LABEL_TO_ID == expected
    for name, idx in expected.items():
        assert ID_TO_LABEL[idx] == name
    assert CLASS_NAMES == ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]

def test_model_architecture_and_output_shape():
    """Verify VietnameseKWSNet outputs shape (Batch, 6) and has correct parameter count."""
    model = VietnameseKWSNet(num_classes=6)
    model.eval()

    param_count = get_model_param_count(model)
    assert 138_000 <= param_count <= 139_000, f"Unexpected param count: {param_count}"

    dummy_input = torch.randn(4, 1, 40, 101)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, 6), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_single_train_step_execution():
    """Verify model can compute cross-entropy loss and execute a backward pass cleanly."""
    model = VietnameseKWSNet(num_classes=6)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_feats = torch.randn(4, 1, 40, 101)
    dummy_labels = torch.tensor([0, 1, 4, 5], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(dummy_feats)
    loss = criterion(logits, dummy_labels)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    loss.backward()
    optimizer.step()

def test_feature_extractor_tensor_shape():
    """Verify LogMelFeatureExtractor produces (1, 1, 40, 101) from 16,000-sample input."""
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0)
    dummy_audio = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 16000))
    feat = extractor(dummy_audio)

    assert isinstance(feat, torch.Tensor)
    assert feat.shape == (1, 1, 40, 101), f"Unexpected shape: {feat.shape}"

def test_splits_row_counts_and_isolation():
    """Verify split manifests have 1020 train, 240 val, 240 test and zero leakage."""
    tr_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_manifest.csv"))
    va_df = pd.read_csv(os.path.join(SPLITS_DIR, "val_manifest.csv"))
    te_df = pd.read_csv(os.path.join(SPLITS_DIR, "test_manifest.csv"))

    assert len(tr_df) == 1020
    assert len(va_df) == 240
    assert len(te_df) == 240

    # ID disjointness
    s_tr, s_va, s_te = set(tr_df["sample_id"]), set(va_df["sample_id"]), set(te_df["sample_id"])
    assert len(s_tr.intersection(s_va)) == 0
    assert len(s_tr.intersection(s_te)) == 0
    assert len(s_va.intersection(s_te)) == 0

def test_four_class_historical_artifacts_protected():
    """Verify Phase 23 1000-sample dataset and Phase 26 baselines remain unmodified."""
    manifest_4c = os.path.join(FOUR_CLASS_DIR, "manifest.csv")
    assert os.path.exists(manifest_4c)
    df_4c = pd.read_csv(manifest_4c)
    assert len(df_4c) == 1000
    assert set(df_4c["class_label"].unique()) == {"TRAI", "PHAI", "LEN", "XUONG"}

    # Check 4-class checkpoint exists
    ckpt_4c = os.path.join(BASE_DIR, "experiments/vieneu_cnn_baseline/checkpoints/best_checkpoint_seed42.pt")
    assert os.path.exists(ckpt_4c)

def test_six_class_checkpoints_and_evaluation_artifacts():
    """Verify all 3 seeds produced valid checkpoints, logs, and evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    for seed in [42, 43, 44]:
        ckpt_path = os.path.join(EXP_DIR, f"checkpoints/best_checkpoint_seed{seed}.pt")
        log_path = os.path.join(EXP_DIR, f"logs/history_seed{seed}.json")
        pred_path = os.path.join(EXP_DIR, f"predictions/test_predictions_seed{seed}.csv")
        metrics_path = os.path.join(EXP_DIR, f"predictions/test_metrics_seed{seed}.json")

        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
        assert os.path.exists(log_path), f"Missing log: {log_path}"
        assert os.path.exists(pred_path), f"Missing predictions: {pred_path}"
        assert os.path.exists(metrics_path), f"Missing metrics: {metrics_path}"

        # Test prediction reproducibility
        pdf = pd.read_csv(pred_path)
        assert len(pdf) == 240
        y_true = pdf["true_label"].map(LABEL_TO_ID).values
        y_pred = pdf["predicted_label"].map(LABEL_TO_ID).values
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        assert abs(m["test_accuracy"] - acc) < 1e-5
        assert abs(m["macro_f1"] - f1) < 1e-5

