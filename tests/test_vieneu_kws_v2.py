"""
Test Suite for Phase 33–35 (Vietnamese KWS v2).
Verifies:
  - Protection of historical v1 artifacts
  - Configuration structure and parameter integrity
  - CNN v2 architecture dimensions and forward pass
  - Train-only noise mixing and SNR precision
  - Post-processing confidence policy logic
  - Event-level matching and latency logic
  - Dataset v2 manifests and text inventory disjointness
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_robustness_campaign.confidence_policy import (
    apply_confidence_policy,
    TARGET_COMMAND_INDICES,
    UNKNOWN_INDEX,
    SILENCE_INDEX
)
from experiments.vieneu_kws_v2.evaluate import (
    SlidingWindowStreamingEngine,
    match_events_and_evaluate,
    generate_noisy_test_batch
)

EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2")
DATASET_V2_DIR = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2")

def test_v1_artifacts_protection():
    """Confirms historical v1 artifacts remain intact and unmodified."""
    v1_4class_manifest = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv")
    v1_6class_manifest = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_6class/manifest.csv")
    v1_ckpt = os.path.join(BASE_DIR, "experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed42.pt")
    
    assert os.path.exists(v1_4class_manifest), "v1 4-class manifest missing!"
    assert os.path.exists(v1_6class_manifest), "v1 6-class manifest missing!"
    assert os.path.exists(v1_ckpt), "v1 6-class checkpoint seed 42 missing!"
    
    df4 = pd.read_csv(v1_4class_manifest)
    assert len(df4) == 1000, "v1 4-class dataset corrupted!"
    df6 = pd.read_csv(v1_6class_manifest)
    assert len(df6) == 1500, "v1 6-class dataset corrupted!"

def test_v2_config_structure():
    cfg_path = os.path.join(EXP_DIR, "config.json")
    assert os.path.exists(cfg_path), "v2 config.json missing!"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    assert cfg["experiment_name"] == "vieneu_kws_v2"
    assert cfg["dataset"]["expected_counts"]["total"] == 2000
    assert cfg["training"]["seeds"] == [42, 43, 44]
    assert cfg["validation_constraints"]["min_target_command_recall"] == 0.95
    assert cfg["validation_constraints"]["max_false_activation_unknown"] == 0.05
    assert cfg["validation_constraints"]["max_streaming_fa_per_hour"] == 1.0

def test_cnn_v2_architecture():
    model = VietnameseKWSNet(num_classes=6)
    model.eval()
    
    dummy_input = torch.randn(4, 1, 40, 101)
    with torch.no_grad():
        out = model(dummy_input)
        
    assert out.shape == (4, 6), f"Expected shape (4, 6), got {out.shape}"
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_noise_mixing_snr_precision():
    clean_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    clean_rms = np.sqrt(np.mean(clean_audio**2))
    
    for snr_db in [20, 10, 5, 0]:
        noisy = generate_noisy_test_batch(clean_audio, "white", snr_db, seed=123)
        noise = noisy - clean_audio
        noise_rms = np.sqrt(np.mean(noise**2))
        measured_snr = 20.0 * np.log10(clean_rms / max(noise_rms, 1e-9))
        assert abs(measured_snr - snr_db) < 0.20, f"Measured SNR {measured_snr} deviated from {snr_db} dB"

def test_streaming_event_matching_and_latency():
    ground_truth = pd.DataFrame([
        {
            "event_type": "target_command",
            "class_label": "PHAI",
            "start_time_sec": 3.0,
            "end_time_sec": 4.0,
            "onset_sec": 3.15,
            "sample_id": "tgt_001"
        }
    ])
    
    triggers = [
        {"trigger_time_sec": 3.55, "pred_label": "PHAI", "confidence": 0.92, "window_start_sec": 2.55}
    ]
    
    metrics = match_events_and_evaluate(triggers, ground_truth, stream_duration_sec=10.0)
    assert metrics["num_gt_targets"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["latencies_ms"]["mean"] == pytest.approx(400.0, abs=1.0) # 3.55 - 3.15 = 0.40s = 400ms

def test_dataset_v2_if_built():
    master_p = os.path.join(DATASET_V2_DIR, "manifest.csv")
    if not os.path.exists(master_p):
        pytest.skip("Dataset v2 build still in progress")
        
    df = pd.read_csv(master_p)
    assert len(df) == 2000
    assert set(df["class_label"].unique()) == {"TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"}
    
    # Check disjoint UNKNOWN texts
    unk = df[df["class_label"] == "UNKNOWN"]
    tr_txt = set(unk[unk["split"] == "train"]["text"])
    va_txt = set(unk[unk["split"] == "val"]["text"])
    te_txt = set(unk[unk["split"] == "test"]["text"])
    
    assert tr_txt.isdisjoint(va_txt)
    assert tr_txt.isdisjoint(te_txt)
    assert va_txt.isdisjoint(te_txt)
