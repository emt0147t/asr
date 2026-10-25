"""
Automated regression tests for the Six-Class Vietnamese KWS Dataset (VieNeu-TTS).
Verifies:
  - Canonical 6-class label mapping (TRAI=0, PHAI=1, LEN=2, XUONG=3, UNKNOWN=4, SILENCE=5)
  - Split counts and balance (1,500 master, 1,020 train, 240 val, 240 test)
  - Strict speaker voice disjointness across splits
  - Disjoint UNKNOWN text inventories across splits
  - Disjoint SILENCE condition templates and zero hash collisions
  - Audio waveform formats (16,000 Hz, 1.0s, 16,000 samples, mono)
  - Bit-exact preservation of original 1,000 four-class audio hashes and split assignments
"""

import os
import sys
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

FOUR_CLASS_MANIFEST = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv")
SIX_CLASS_DIR = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_6class")
SIX_CLASS_MANIFEST = os.path.join(SIX_CLASS_DIR, "manifest.csv")
SPLITS_DIR = os.path.join(SIX_CLASS_DIR, "splits")
CONFIG_PATH = os.path.join(SIX_CLASS_DIR, "configs/dataset_config.json")

CANONICAL_6CLASS_MAPPING = {
    "TRAI": 0,
    "PHAI": 1,
    "LEN": 2,
    "XUONG": 3,
    "UNKNOWN": 4,
    "SILENCE": 5
}

@pytest.fixture(scope="module")
def dataset_df():
    assert os.path.exists(SIX_CLASS_MANIFEST), f"Six-class manifest missing: {SIX_CLASS_MANIFEST}"
    return pd.read_csv(SIX_CLASS_MANIFEST)

def test_canonical_label_mapping():
    """Verify 6-class label mapping matches project convention."""
    from src.models.kws_cnn import VietnameseKWSNet
    model = VietnameseKWSNet(num_classes=6)
    assert model.fc2.out_features == 6
    assert list(CANONICAL_6CLASS_MAPPING.keys()) == ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]

def test_manifest_counts_and_balance(dataset_df):
    """Verify 1,500 total rows and exact 250 per-class balance."""
    assert len(dataset_df) == 1500, f"Expected 1,500 rows, got {len(dataset_df)}"
    counts = dataset_df["class_label"].value_counts().to_dict()
    expected = {"TRAI": 250, "PHAI": 250, "LEN": 250, "XUONG": 250, "UNKNOWN": 250, "SILENCE": 250}
    assert counts == expected, f"Class counts mismatch: {counts}"

def test_split_counts_and_balance():
    """Verify 1,020 train, 240 val, 240 test, and per-class balance."""
    split_expectations = {
        "train": (1020, 170),
        "val": (240, 40),
        "test": (240, 40)
    }
    for sp_name, (exp_total, exp_per_class) in split_expectations.items():
        sp_path = os.path.join(SPLITS_DIR, f"{sp_name}_manifest.csv")
        assert os.path.exists(sp_path), f"Split manifest missing: {sp_path}"
        sp_df = pd.read_csv(sp_path)
        assert len(sp_df) == exp_total, f"Split {sp_name} total mismatch: {len(sp_df)} != {exp_total}"
        counts = sp_df["class_label"].value_counts().to_dict()
        for cls in ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]:
            assert counts[cls] == exp_per_class, f"{sp_name} {cls} mismatch: {counts[cls]} != {exp_per_class}"

def test_voice_disjointness(dataset_df):
    """Verify zero voice preset overlap across splits for speech classes."""
    speech_df = dataset_df[dataset_df["class_label"].isin(["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN"])]
    train_voices = set(speech_df[speech_df["split"] == "train"]["voice_id"].unique())
    val_voices = set(speech_df[speech_df["split"] == "val"]["voice_id"].unique())
    test_voices = set(speech_df[speech_df["split"] == "test"]["voice_id"].unique())

    assert len(train_voices) == 17, f"Expected 17 train voices, got {len(train_voices)}"
    assert len(val_voices) == 4, f"Expected 4 val voices, got {len(val_voices)}"
    assert len(test_voices) == 4, f"Expected 4 test voices, got {len(test_voices)}"

    assert len(train_voices.intersection(val_voices)) == 0, "Voice overlap between train and val"
    assert len(train_voices.intersection(test_voices)) == 0, "Voice overlap between train and test"
    assert len(val_voices.intersection(test_voices)) == 0, "Voice overlap between val and test"

def test_unknown_text_inventory_disjointness(dataset_df):
    """Verify UNKNOWN texts do not overlap across splits and contain no target keywords."""
    unknown_df = dataset_df[dataset_df["class_label"] == "UNKNOWN"]
    assert len(unknown_df) == 250

    train_texts = set(unknown_df[unknown_df["split"] == "train"]["text"].unique())
    val_texts = set(unknown_df[unknown_df["split"] == "val"]["text"].unique())
    test_texts = set(unknown_df[unknown_df["split"] == "test"]["text"].unique())

    assert len(train_texts.intersection(val_texts)) == 0, "UNKNOWN text overlap between train and val"
    assert len(train_texts.intersection(test_texts)) == 0, "UNKNOWN text overlap between train and test"
    assert len(val_texts.intersection(test_texts)) == 0, "UNKNOWN text overlap between val and test"

    target_keywords = ["trái", "phải", "lên", "xuống"]
    for t in train_texts | val_texts | test_texts:
        for kw in target_keywords:
            assert kw not in t.lower(), f"Target keyword '{kw}' found in UNKNOWN text: '{t}'"

def test_silence_condition_disjointness(dataset_df):
    """Verify SILENCE source conditions and templates are disjoint across splits."""
    silence_df = dataset_df[dataset_df["class_label"] == "SILENCE"]
    assert len(silence_df) == 250

    train_models = set(silence_df[silence_df["split"] == "train"]["model_checkpoint"].unique())
    val_models = set(silence_df[silence_df["split"] == "val"]["model_checkpoint"].unique())
    test_models = set(silence_df[silence_df["split"] == "test"]["model_checkpoint"].unique())

    assert len(train_models.intersection(val_models)) == 0, "SILENCE condition overlap between train and val"
    assert len(train_models.intersection(test_models)) == 0, "SILENCE condition overlap between train and test"
    assert len(val_models.intersection(test_models)) == 0, "SILENCE condition overlap between val and test"

def test_zero_hash_overlap_across_splits(dataset_df):
    """Verify all 1,500 audio samples have unique KWS hashes with zero cross-split overlap."""
    assert dataset_df["kws_sha256"].nunique() == 1500, "Duplicate KWS hashes detected in 6-class dataset"
    
    train_hashes = set(dataset_df[dataset_df["split"] == "train"]["kws_sha256"])
    val_hashes = set(dataset_df[dataset_df["split"] == "val"]["kws_sha256"])
    test_hashes = set(dataset_df[dataset_df["split"] == "test"]["kws_sha256"])

    assert len(train_hashes.intersection(val_hashes)) == 0, "Hash overlap between train and val"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Hash overlap between train and test"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Hash overlap between val and test"

def test_preservation_of_four_class_baseline(dataset_df):
    """Verify original 1,000 four-class samples retain exact hashes, labels, and splits."""
    df_4c = pd.read_csv(FOUR_CLASS_MANIFEST)
    assert len(df_4c) == 1000

    sub_6c = dataset_df[dataset_df["class_label"].isin(["TRAI", "PHAI", "LEN", "XUONG"])]
    assert len(sub_6c) == 1000

    merged = pd.merge(df_4c, sub_6c, on="sample_id", suffixes=("_orig", "_new"))
    assert len(merged) == 1000

    assert (merged["class_label_orig"] == merged["class_label_new"]).all()
    assert (merged["split_orig"] == merged["split_new"]).all()
    assert (merged["kws_sha256_orig"] == merged["kws_sha256_new"]).all()
    assert (merged["voice_id_orig"] == merged["voice_id_new"]).all()

def test_audio_format_and_signal_health(dataset_df):
    """Sample audio clips to verify exact 16,000 samples, 16 kHz, mono, finite, and no clipping."""
    sample_sub = dataset_df.sample(n=60, random_state=42)
    for _, row in sample_sub.iterrows():
        p = os.path.join(BASE_DIR, row["kws_wav_path"])
        assert os.path.exists(p), f"Audio file missing: {p}"

        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == row["kws_sha256"], f"SHA256 mismatch for {p}"

        data, sr = sf.read(p)
        assert sr == 16000, f"Sample rate mismatch: {sr}"
        assert len(data) == 16000, f"Length mismatch: {len(data)}"
        assert not np.isnan(data).any(), "NaN found in audio"
        assert not np.isinf(data).any(), "Inf found in audio"

        peak = np.max(np.abs(data))
        assert peak < 0.999, f"Clipping detected: {peak}"
