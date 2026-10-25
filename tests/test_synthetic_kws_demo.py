"""
Integration and Unit Tests for Phase 5 Synthetic KWS Inference & Game-Control Prototype.
Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
Tests run offline using synthetic fixtures only.
"""

import sys
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from experiments.synthetic_kws_demo.classifier import (
    OfflineKWSClassifier,
    AudioValidator,
    AudioValidationError,
    AudioFileNotFoundError,
    InvalidSampleRateError,
    InvalidChannelCountError,
    InvalidDurationError,
    EmptyAudioError,
    CheckpointNotFoundError,
    SYNTHETIC_DEMO_WARNING,
    CLASS_LABELS,
)
from experiments.synthetic_kws_demo.game_adapter import (
    GameControlAdapter,
    GameAction,
    AdaptedAction,
)
from experiments.synthetic_kws_demo.game_simulator import GridGameSimulator


EXPECTED_ARM_B_SHA256 = "6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96"
FIXTURES_DIR = Path("experiments/synthetic_kws_demo/fixtures")
CHECKPOINT_PATH = Path("experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt")


@pytest.fixture(scope="module")
def classifier():
    return OfflineKWSClassifier(checkpoint_path=CHECKPOINT_PATH, device="cpu")


@pytest.fixture
def game_adapter():
    return GameControlAdapter(confidence_threshold=0.50, debounce_cooldown_sec=0.0)


# ============================================================================
# 1. Checkpoint Integrity & Architecture Contracts
# ============================================================================

def test_arm_b_checkpoint_hash_and_existence():
    """Verify primary experimental candidate checkpoint exists and matches frozen SHA-256."""
    assert CHECKPOINT_PATH.exists(), f"Candidate checkpoint missing: {CHECKPOINT_PATH}"
    file_bytes = CHECKPOINT_PATH.read_bytes()
    computed_sha = hashlib.sha256(file_bytes).hexdigest().upper()
    assert computed_sha == EXPECTED_ARM_B_SHA256, (
        f"Arm B checkpoint SHA-256 mismatch! Expected {EXPECTED_ARM_B_SHA256}, got {computed_sha}"
    )


def test_classifier_initialization_and_shapes(classifier):
    """Verify model architecture, feature extractor dimensions, and eval mode."""
    assert classifier.model.training is False, "Classifier model must be in eval mode."
    assert classifier.extractor.training is False, "Feature extractor must be in eval mode."
    
    # Dummy input forward pass
    dummy_feat = torch.zeros((1, 1, 40, 101), dtype=torch.float32)
    with torch.no_grad():
        logits = classifier.model(dummy_feat)
    assert logits.shape == (1, 6), f"Expected logits shape (1, 6), got {logits.shape}"
    assert len(CLASS_LABELS) == 6


def test_missing_checkpoint_raises_error():
    """Verify missing checkpoint path raises CheckpointNotFoundError."""
    with pytest.raises(CheckpointNotFoundError):
        OfflineKWSClassifier(checkpoint_path=Path("non_existent_path.pt"))


# ============================================================================
# 2. Strict Input Validation & Negative Conditions
# ============================================================================

def test_validation_rejects_stereo():
    """Stereo audio must be rejected with InvalidChannelCountError."""
    stereo_path = FIXTURES_DIR / "invalid_stereo.wav"
    assert stereo_path.exists()
    with pytest.raises(InvalidChannelCountError) as exc_info:
        AudioValidator.validate_and_load(stereo_path)
    assert "2 channels" in str(exc_info.value)


def test_validation_rejects_wrong_sample_rate():
    """8,000 Hz audio must be rejected with InvalidSampleRateError without silent resampling."""
    sr8k_path = FIXTURES_DIR / "invalid_sample_rate_8k.wav"
    assert sr8k_path.exists()
    with pytest.raises(InvalidSampleRateError) as exc_info:
        AudioValidator.validate_and_load(sr8k_path)
    assert "8000 Hz" in str(exc_info.value)
    assert "resampling is prohibited" in str(exc_info.value)


def test_validation_rejects_truncated_duration():
    """Audio shorter than 16,000 samples must be rejected with InvalidDurationError."""
    short_path = FIXTURES_DIR / "invalid_short_duration.wav"
    assert short_path.exists()
    with pytest.raises(InvalidDurationError) as exc_info:
        AudioValidator.validate_and_load(short_path)
    assert "8000 samples" in str(exc_info.value)


def test_validation_rejects_empty_audio():
    """Empty 0-byte file must be rejected with EmptyAudioError."""
    empty_path = FIXTURES_DIR / "invalid_empty.wav"
    assert empty_path.exists()
    with pytest.raises(EmptyAudioError):
        AudioValidator.validate_and_load(empty_path)


def test_validation_rejects_nonexistent_file():
    """Nonexistent path must raise AudioFileNotFoundError."""
    nonexistent = FIXTURES_DIR / "ghost_file.wav"
    with pytest.raises(AudioFileNotFoundError):
        AudioValidator.validate_and_load(nonexistent)


def test_classifier_handles_invalid_audio_gracefully(classifier):
    """Classifier predict() must return structured is_valid=False without crashing on invalid audio."""
    stereo_path = FIXTURES_DIR / "invalid_stereo.wav"
    res = classifier.predict(stereo_path)
    assert res.is_valid is False
    assert res.predicted_label == "REJECTED"
    assert res.predicted_class_id == -1
    assert "Unsupported channel count" in res.error_message
    assert res.total_latency_ms > 0.0


# ============================================================================
# 3. Determinism, Probability Norms, and Warning Text
# ============================================================================

def test_deterministic_repeated_inference(classifier):
    """Inference must produce identical class probabilities on repeated runs."""
    wav_path = FIXTURES_DIR / "demo_01_trai.wav"
    res1 = classifier.predict(wav_path)
    res2 = classifier.predict(wav_path)
    
    assert res1.is_valid and res2.is_valid
    assert res1.predicted_label == res2.predicted_label
    assert pytest.approx(res1.confidence, abs=1e-6) == res2.confidence
    
    for cls in CLASS_LABELS:
        assert pytest.approx(res1.class_probabilities[cls], abs=1e-6) == res2.class_probabilities[cls]


def test_probability_distribution_sums_to_one(classifier):
    """Class probabilities must sum to 1.0."""
    wav_path = FIXTURES_DIR / "demo_02_phai.wav"
    res = classifier.predict(wav_path)
    total_prob = sum(res.class_probabilities.values())
    assert pytest.approx(total_prob, abs=1e-5) == 1.0


def test_warning_notice_and_governance_invariants(classifier):
    """Prediction result must contain the prominent synthetic demo caveat and blocked notice."""
    wav_path = FIXTURES_DIR / "demo_03_len.wav"
    res = classifier.predict(wav_path)
    assert SYNTHETIC_DEMO_WARNING in res.warning
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in res.warning


# ============================================================================
# 4. Game Action Mapping & Safety Suppression
# ============================================================================

def test_target_command_to_game_action_mapping(game_adapter, classifier):
    """Verify target command mapping: TRAI->MOVE_LEFT, PHAI->MOVE_RIGHT, LEN->JUMP, XUONG->CROUCH."""
    test_cases = [
        ("demo_01_trai.wav", "TRAI", GameAction.MOVE_LEFT, -1, 0),
        ("demo_02_phai.wav", "PHAI", GameAction.MOVE_RIGHT, 1, 0),
        ("demo_03_len.wav", "LEN", GameAction.JUMP, 0, -1),
        ("demo_04_xuong.wav", "XUONG", GameAction.CROUCH, 0, 1),
    ]
    
    for fname, expected_label, expected_action, exp_dx, exp_dy in test_cases:
        p = FIXTURES_DIR / fname
        pred = classifier.predict(p)
        assert pred.predicted_label == expected_label, (
            f"Fixture {fname} expected {expected_label}, got {pred.predicted_label}"
        )
        action = game_adapter.adapt(pred)
        assert action.action == expected_action
        assert action.delta_x == exp_dx
        assert action.delta_y == exp_dy
        assert action.is_active is True


def test_non_target_safety_suppression(game_adapter, classifier):
    """Verify UNKNOWN and SILENCE classes are suppressed to NO_ACTION."""
    non_targets = [
        ("demo_05_unknown.wav", "UNKNOWN"),
        ("demo_06_silence.wav", "SILENCE"),
    ]
    for fname, expected_label in non_targets:
        p = FIXTURES_DIR / fname
        pred = classifier.predict(p)
        assert pred.predicted_label == expected_label
        action = game_adapter.adapt(pred)
        assert action.action == GameAction.NO_ACTION
        assert action.delta_x == 0
        assert action.delta_y == 0
        assert action.is_active is False
        assert "safely suppressed" in action.notes


def test_low_confidence_suppression(game_adapter, classifier):
    """Predictions below confidence threshold must be suppressed to NO_ACTION."""
    high_threshold_adapter = GameControlAdapter(confidence_threshold=0.9999, debounce_cooldown_sec=0.0)
    p = FIXTURES_DIR / "demo_01_trai.wav"
    pred = classifier.predict(p)
    action = high_threshold_adapter.adapt(pred)
    assert action.action == GameAction.NO_ACTION
    assert action.is_active is False
    assert "below" in action.notes.lower() and "threshold" in action.notes.lower()


# ============================================================================
# 5. Grid World Simulator Safety
# ============================================================================

def test_grid_simulator_state_invariant_on_no_action(game_adapter, classifier):
    """Simulator player position must remain unchanged when non-target audio is received."""
    sim = GridGameSimulator(width=7, height=7, initial_pos=(3, 3))
    init_pos = (sim.player_x, sim.player_y)
    
    # UNKNOWN fixture
    pred_unk = classifier.predict(FIXTURES_DIR / "demo_05_unknown.wav")
    act_unk = game_adapter.adapt(pred_unk)
    rec_unk = sim.step(act_unk)
    assert (sim.player_x, sim.player_y) == init_pos
    assert "NO_ACTION" in rec_unk.event
    
    # SILENCE fixture
    pred_sil = classifier.predict(FIXTURES_DIR / "demo_06_silence.wav")
    act_sil = game_adapter.adapt(pred_sil)
    rec_sil = sim.step(act_sil)
    assert (sim.player_x, sim.player_y) == init_pos
    assert "NO_ACTION" in rec_sil.event


def test_grid_simulator_movement_and_collision(game_adapter, classifier):
    """Simulator moves player on active commands and detects obstacles."""
    sim = GridGameSimulator(width=7, height=7, initial_pos=(3, 3))
    
    # LEN moves player up from (3, 3) to (3, 2)
    pred_len = classifier.predict(FIXTURES_DIR / "demo_03_len.wav")
    act_len = game_adapter.adapt(pred_len)
    rec_len = sim.step(act_len)
    assert "MOVED_JUMP" in rec_len.event
    assert (sim.player_x, sim.player_y) == (3, 2)
    
    # There is an obstacle at (4, 2); PHAI tries to move to (4, 2) and gets blocked
    pred_phai = classifier.predict(FIXTURES_DIR / "demo_02_phai.wav")
    act_phai = game_adapter.adapt(pred_phai)
    rec_phai = sim.step(act_phai)
    assert "BLOCKED_BY_OBSTACLE" in rec_phai.event
    assert (sim.player_x, sim.player_y) == (3, 2)


# ============================================================================
# 6. Ambient Audio Buffer Preservation & Manifest Integrity
# ============================================================================

def test_silence_buffer_preservation():
    """Verify ambient silence fixture is exactly 16,000 samples and unshifted."""
    sil_path = FIXTURES_DIR / "demo_06_silence.wav"
    audio, sr = AudioValidator.validate_and_load(sil_path)
    assert sr == 16000
    assert len(audio) == 16000
    # Ambient dither noise must have non-zero RMS but low energy (< 0.05 max abs)
    assert np.max(np.abs(audio)) < 0.05
    assert np.std(audio) > 1e-5


def test_fixture_manifest_hashes():
    """Verify fixture_manifest.csv accurately lists all files and matches on-disk SHA-256."""
    manifest_p = FIXTURES_DIR / "fixture_manifest.csv"
    assert manifest_p.exists()
    df = pd.read_csv(manifest_p)
    assert len(df) == 10
    
    for _, row in df.iterrows():
        p = FIXTURES_DIR / row["filename"]
        assert p.exists()
        calc_sha = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        assert calc_sha == row["sha256"], f"Hash mismatch for {row['filename']}"
