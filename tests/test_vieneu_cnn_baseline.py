"""
Automated regression tests for the VieNeu-TTS 4-Class CNN Keyword Spotting Baseline.
Verifies config integrity, model forward pass, feature extractor output dimensions,
label mappings, split isolation, checkpoint structure, and evaluation metrics.
"""

import os
import sys
import json
import pytest
import torch
import numpy as np
import pandas as pd

# Add repo root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet, get_model_param_count
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_cnn_baseline.train import LABEL_TO_ID, ID_TO_LABEL, CLASS_NAMES

CONFIG_PATH = os.path.join(BASE_DIR, "experiments/vieneu_cnn_baseline/config.json")
SPLITS_DIR = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000/splits")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "experiments/vieneu_cnn_baseline/checkpoints")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "experiments/vieneu_cnn_baseline/predictions")


def test_label_mapping_consistency():
    """Verify 4-class keyword label mapping conforms strictly to project specification."""
    expected = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3}
    assert LABEL_TO_ID == expected
    for name, idx in expected.items():
        assert ID_TO_LABEL[idx] == name
    assert len(LABEL_TO_ID) == 4
    assert CLASS_NAMES == ["TRAI", "PHAI", "LEN", "XUONG"]


def test_config_structure():
    """Verify config.json contains all required hyperparameters and correct settings."""
    assert os.path.exists(CONFIG_PATH), f"Config missing: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["audio"]["clip_duration_sec"] == 1.0
    assert cfg["feature_extractor"]["n_mels"] == 40
    assert cfg["feature_extractor"]["n_fft"] == 512
    assert cfg["feature_extractor"]["win_length"] == 400
    assert cfg["feature_extractor"]["hop_length"] == 160
    assert cfg["feature_extractor"]["f_min"] == 20.0
    assert cfg["model"]["classes"] == ["TRAI", "PHAI", "LEN", "XUONG"]
    assert cfg["model"]["label_to_id"] == {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3}


def test_feature_extractor_output_shape():
    """Verify log-mel feature extractor outputs correct tensor shape (1, 1, 40, 101) for 1-second audio."""
    extractor = LogMelFeatureExtractor(
        sample_rate=16000,
        n_mels=40,
        win_length=400,
        hop_length=160,
        n_fft=512,
        f_min=20.0,
        f_max=8000.0
    )
    
    # 1.0s dummy audio at 16kHz as torch.Tensor
    dummy_wav = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 16000))
    feat = extractor(dummy_wav)

    assert isinstance(feat, torch.Tensor)
    assert feat.shape == (1, 1, 40, 101), f"Unexpected feature shape: {feat.shape}"
    assert not torch.isnan(feat).any()
    assert not torch.isinf(feat).any()


def test_kws_net_architecture_and_forward_pass():
    """Verify model forward pass, parameter count, and logit outputs."""
    model = VietnameseKWSNet(num_classes=4)
    model.eval()

    total_params = get_model_param_count(model)
    assert 100_000 <= total_params <= 200_000, f"Unexpected parameter count: {total_params}"

    # Forward pass with batch of 4
    dummy_input = torch.randn(4, 1, 40, 101)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, 4), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_split_manifests_isolation():
    """Verify split manifests exist, are balanced, and have zero voice leakage."""
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test_manifest.csv"))

    assert len(train_df) == 680
    assert len(val_df) == 160
    assert len(test_df) == 160

    train_voices = set(train_df["voice_id"].unique())
    val_voices = set(val_df["voice_id"].unique())
    test_voices = set(test_df["voice_id"].unique())

    assert len(train_voices.intersection(val_voices)) == 0
    assert len(train_voices.intersection(test_voices)) == 0
    assert len(val_voices.intersection(test_voices)) == 0


def test_checkpoint_compatibility():
    """Verify all saved multi-seed checkpoints exist and can be loaded cleanly."""
    for seed in [42, 43, 44]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_checkpoint_seed{seed}.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        assert "model_state_dict" in ckpt
        assert "val_macro_f1" in ckpt
        assert "epoch" in ckpt
        assert ckpt["seed"] == seed

        model = VietnameseKWSNet(num_classes=4)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(2, 1, 40, 101)
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 4)


def test_evaluation_predictions_artifacts():
    """Verify evaluation outputs (predictions CSV and metrics JSON) exist for all seeds."""
    for seed in [42, 43, 44]:
        csv_path = os.path.join(PREDICTIONS_DIR, f"test_predictions_seed{seed}.csv")
        json_path = os.path.join(PREDICTIONS_DIR, f"test_metrics_seed{seed}.json")

        assert os.path.exists(csv_path), f"Missing predictions CSV: {csv_path}"
        assert os.path.exists(json_path), f"Missing metrics JSON: {json_path}"

        df = pd.read_csv(csv_path)
        assert len(df) == 160
        assert "correct" in df.columns
        assert "confidence" in df.columns

        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["total_test_samples"] == 160
        assert metrics["test_accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
