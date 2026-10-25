"""
Automated Pytest Suite for Vietnamese KWS Dataset v12 & Evaluation Artifacts.
Performs verification of:
1. Manifest schemas and data completeness
2. Split isolation and zero-leakage constraints
3. Audio physical properties (16 kHz, mono, 1.0s, non-clipping)
4. Class balance and subgroup distributions
5. Streaming evaluation corpus integrity
6. Immutability of canonical v11 governance artifacts
"""

import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
MANIFEST_CSV = V12_DIR / "manifests" / "dataset_v12_manifest.csv"
STREAMING_MANIFEST_CSV = V12_DIR / "manifests" / "streaming_eval_corpus_manifest.csv"

FROZEN_PT = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
FROZEN_PT_SHA = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

FROZEN_ONNX = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "onnx" / "vietnamese_kws_v11.onnx"
FROZEN_ONNX_SHA = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest_df():
    assert MANIFEST_CSV.exists(), f"Manifest file missing: {MANIFEST_CSV}"
    df = pd.read_csv(MANIFEST_CSV)
    return df


@pytest.fixture(scope="module")
def streaming_df():
    assert STREAMING_MANIFEST_CSV.exists(), f"Streaming manifest missing: {STREAMING_MANIFEST_CSV}"
    return pd.read_csv(STREAMING_MANIFEST_CSV)


def test_frozen_artifacts_immutability():
    """Verify v11 checkpoint and ONNX files have not been altered or replaced."""
    assert FROZEN_PT.exists(), f"Frozen checkpoint missing: {FROZEN_PT}"
    assert compute_sha256(FROZEN_PT) == FROZEN_PT_SHA, "Checkpoint SHA-256 corrupted!"

    assert FROZEN_ONNX.exists(), f"Frozen ONNX missing: {FROZEN_ONNX}"
    assert compute_sha256(FROZEN_ONNX) == FROZEN_ONNX_SHA, "ONNX SHA-256 corrupted!"


def test_manifest_schema_and_columns(manifest_df):
    """Verify all 21 mandatory columns are present with non-empty sample_ids."""
    expected_cols = [
        "sample_id", "label", "reject_subgroup", "audio_path", "source_type",
        "source_utterance_id", "voice_or_speaker_id", "text", "oov_lexeme",
        "noise_asset_id", "target_snr_db", "observed_snr_db", "sample_rate_hz",
        "channels", "num_samples", "duration_seconds", "transformations",
        "split", "source_license_or_permission", "sha256", "generation_config_id"
    ]
    for col in expected_cols:
        assert col in manifest_df.columns, f"Column {col} missing from manifest schema"
    assert len(manifest_df) == 8000, f"Expected 8000 rows, got {len(manifest_df)}"
    assert manifest_df["sample_id"].nunique() == 8000, "sample_id is not globally unique"


def test_class_balance_and_subgroups(manifest_df):
    """Verify 4 target commands have exactly 1,000 clips each, and REJECT has 4,000 clips."""
    for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
        assert len(manifest_df[manifest_df["label"] == cmd]) == 1000, f"Class {cmd} not 1000 clips"

    reject_df = manifest_df[manifest_df["label"] == "REJECT"]
    assert len(reject_df) == 4000, "REJECT class count != 4000"

    subgroups = reject_df["reject_subgroup"].value_counts()
    assert subgroups["oov_word"] == 1600, "oov_word subgroup != 1600"
    assert subgroups["conversation"] == 1000, "conversation subgroup != 1000"
    assert subgroups["environmental_noise"] == 800, "environmental_noise subgroup != 800"
    assert subgroups["silence_non_speech"] == 600, "silence_non_speech subgroup != 600"


def test_split_proportions(manifest_df):
    """Verify train/val/test splits match ~68% / 16% / 16% ratio."""
    counts = manifest_df["split"].value_counts()
    assert counts["train"] == 5440, f"Train split != 5440, got {counts['train']}"
    assert counts["val"] == 1280, f"Val split != 1280, got {counts['val']}"
    assert counts["test"] == 1280, f"Test split != 1280, got {counts['test']}"


def test_split_isolation_and_zero_leakage(manifest_df):
    """Verify voice, utterance, OOV unseen, and noise track disjointness across splits."""
    train_df = manifest_df[manifest_df["split"] == "train"]
    val_df = manifest_df[manifest_df["split"] == "val"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    # 1. Voice disjointness
    train_voices = set(train_df["voice_or_speaker_id"].dropna())
    val_voices = set(val_df["voice_or_speaker_id"].dropna())
    test_voices = set(test_df["voice_or_speaker_id"].dropna())
    assert len(train_voices.intersection(val_voices)) == 0, "Voice leak between train and val"
    assert len(train_voices.intersection(test_voices)) == 0, "Voice leak between train and test"
    assert len(val_voices.intersection(test_voices)) == 0, "Voice leak between val and test"

    # 2. Source Utterance disjointness
    train_utts = set(train_df["source_utterance_id"].dropna())
    val_utts = set(val_df["source_utterance_id"].dropna())
    test_utts = set(test_df["source_utterance_id"].dropna())
    assert len(train_utts.intersection(val_utts)) == 0, "Source utterance leak train/val"
    assert len(train_utts.intersection(test_utts)) == 0, "Source utterance leak train/test"
    assert len(val_utts.intersection(test_utts)) == 0, "Source utterance leak val/test"

    # 3. OOV Unseen test lexeme disjointness
    test_oov_lexemes = set(test_df[test_df["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna())
    train_oov_lexemes = set(train_df[train_df["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna())
    val_oov_lexemes = set(val_df[val_df["reject_subgroup"] == "oov_word"]["oov_lexeme"].dropna())
    assert len(test_oov_lexemes.intersection(train_oov_lexemes)) == 0, "Test OOV lexeme leaked to train"
    assert len(test_oov_lexemes.intersection(val_oov_lexemes)) == 0, "Test OOV lexeme leaked to val"

    # 4. Noise asset disjointness
    train_noise = set(train_df["noise_asset_id"].dropna())
    val_noise = set(val_df["noise_asset_id"].dropna())
    test_noise = set(test_df["noise_asset_id"].dropna())
    assert len(train_noise.intersection(val_noise)) == 0, "Noise asset leak train/val"
    assert len(train_noise.intersection(test_noise)) == 0, "Noise asset leak train/test"
    assert len(val_noise.intersection(test_noise)) == 0, "Noise asset leak val/test"


def test_audio_properties_sample(manifest_df):
    """Spot check 100 randomly sampled audio files for length, sample rate, channels, peak amplitude."""
    sample = manifest_df.sample(n=100, random_state=42)
    for _, r in sample.iterrows():
        p = PROJECT_ROOT / r["audio_path"]
        assert p.exists(), f"Audio file not found: {p}"
        audio, sr = sf.read(str(p), dtype="float32")
        assert sr == 16000, f"Sample rate != 16000: {p}"
        assert len(audio) == 16000, f"Audio length != 16000 samples: {p}"
        assert audio.ndim == 1, f"Audio is not mono: {p}"
        peak = float(np.max(np.abs(audio)))
        assert peak <= 0.99, f"Audio clipped (peak {peak} > 0.99): {p}"


def test_streaming_eval_corpus_integrity(streaming_df):
    """Verify streaming evaluation corpus has 40 sequences, valid JSON events, and all files exist."""
    assert len(streaming_df) == 40, f"Expected 40 sequences, got {len(streaming_df)}"
    for _, r in streaming_df.iterrows():
        p = PROJECT_ROOT / r["audio_path"]
        assert p.exists(), f"Streaming audio file missing: {p}"
        events = json.loads(r["events_json"])
        assert len(events) == r["num_total_events"], "Event count mismatch"
        for ev in events:
            assert "label" in ev and "start_sec" in ev and "end_sec" in ev
            assert ev["start_sec"] < ev["end_sec"]
