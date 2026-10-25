"""
Tests for Phase 13.2.1: Pre-Synthesis Frontend Compatibility Audit.

Validates:
1. Candidate checkpoint SHA-256 byte immutability.
2. Executable source evidence confirming training frontend is n_fft=512, win_length=400, hop_length=160.
3. Bitwise exact identity between training LogMelFrontend and remediated Phase 13.1 infer.py LogMelFrontend.
4. Numerical parity of portable pure-NumPy 512 frontend against PyTorch (diff <= 5e-5).
5. Verification that N_fft=400 produces measurable divergence on active speech, proving it is non-authoritative for v11.
6. Audit gate status PRE_SYNTHESIS_GATE_PASSED and governance invariants.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vieneu_kws_research_v12"
    / "bundle"
    / "checkpoint"
    / "variant_v11_hard_negative_weighted_seed42.pt"
)
EXPECTED_SHA256 = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"

DEV_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_1" / "fixtures"
AUDIT_REPORT_ROOT = PROJECT_ROOT / "reports" / "PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md"
AUDIT_REPORT_EXP = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_2" / "reports" / "PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md"
AUDIT_JSON = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_2" / "reports" / "frontend_audit_results.json"


def test_checkpoint_sha256_integrity():
    """Verify frozen candidate checkpoint exists and matches authoritative SHA-256."""
    assert CHECKPOINT_PATH.exists(), f"Candidate checkpoint missing at {CHECKPOINT_PATH}"
    with open(CHECKPOINT_PATH, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha256 == EXPECTED_SHA256, (
        f"Checkpoint SHA-256 mismatch!\nExpected: {EXPECTED_SHA256}\nActual:   {actual_sha256}"
    )


def test_training_frontend_source_evidence():
    """Verify that train_v11.py strictly specifies n_fft=512, win_length=400, hop_length=160."""
    from experiments.vieneu_kws_research_v11.train_v11 import LogMelFrontend as TrainFrontend
    
    tf = TrainFrontend()
    assert tf.mel.n_fft == 512, f"Expected n_fft=512, got {tf.mel.n_fft}"
    assert tf.mel.win_length == 400, f"Expected win_length=400, got {tf.mel.win_length}"
    assert tf.mel.hop_length == 160, f"Expected hop_length=160, got {tf.mel.hop_length}"
    assert tf.mel.n_mels == 40, f"Expected n_mels=40, got {tf.mel.n_mels}"
    assert tf.mel.mel_scale.norm == "slaney", f"Expected norm='slaney', got {tf.mel.mel_scale.norm}"
    assert tf.mel.mel_scale.mel_scale == "slaney", f"Expected mel_scale='slaney', got {tf.mel.mel_scale.mel_scale}"
    assert tf.mel.power == 2.0, f"Expected power=2.0, got {tf.mel.power}"
    assert tf.mel.mel_scale.f_min == 20.0, f"Expected f_min=20.0, got {tf.mel.mel_scale.f_min}"
    assert tf.mel.mel_scale.f_max == 8000.0, f"Expected f_max=8000.0, got {tf.mel.mel_scale.f_max}"


def test_remediated_frontend_bitwise_identity_with_training():
    """Verify bitwise identity (max_diff == 0.0) between training and remediated 13.1 frontends."""
    from experiments.vieneu_kws_research_v11.train_v11 import LogMelFrontend as TrainFrontend
    from experiments.vieneu_kws_research_v13_1.bundle.infer import LogMelFrontend as InferFrontend

    tf = TrainFrontend()
    inf = InferFrontend()

    torch.manual_seed(42)
    test_wave = torch.randn(1, 16000)

    train_out = tf(test_wave)
    infer_out = inf(test_wave)

    max_diff = torch.max(torch.abs(train_out - infer_out)).item()
    assert max_diff == 0.0, f"Bitwise mismatch between training and remediated infer frontends: {max_diff}"


def test_portable_numpy_512_numerical_parity():
    """Verify that pure-NumPy PortableLogMelFrontend512 matches PyTorch within 5e-5."""
    from experiments.vieneu_kws_research_v13_2.audit_frontend_compatibility import (
        PortableLogMelFrontend512,
        TrainingLogMelFrontend,
    )

    tr_frontend = TrainingLogMelFrontend()
    np_frontend = PortableLogMelFrontend512()

    rng = np.random.default_rng(12345)
    wav_np = rng.standard_normal(16000).astype(np.float32) * 0.5
    wav_t = torch.from_numpy(wav_np).unsqueeze(0)

    with torch.no_grad():
        tr_mel = tr_frontend(wav_t).numpy()
    np_mel = np_frontend.compute(wav_np)

    max_diff = np.max(np.abs(tr_mel - np_mel))
    assert max_diff <= 5e-5, f"NumPy vs PyTorch log-mel max diff {max_diff} exceeded 5e-5 tolerance"


def test_phase92_400_divergence_on_active_speech():
    """Verify that Phase 92 (N_fft=400) diverges significantly from Training (N_fft=512) on active speech."""
    from experiments.vieneu_kws_research_v13_2.audit_frontend_compatibility import (
        TrainingLogMelFrontend,
        Phase92LogMelFrontend,
    )

    tr_frontend = TrainingLogMelFrontend()
    p92_frontend = Phase92LogMelFrontend()

    # Active speech dev fixture
    wav_path = DEV_DIR / "dev_05_normal_len_peak089.wav"
    assert wav_path.exists(), f"Dev fixture missing: {wav_path}"

    import soundfile as sf
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)

    wav_t = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        tr_mel = tr_frontend(wav_t).numpy()
        p92_mel = p92_frontend(wav_t).numpy()

    max_diff = np.max(np.abs(tr_mel - p92_mel))
    # N_fft=400 vs 512 MUST show substantial divergence (> 0.5) on active speech
    assert max_diff > 0.5, f"Expected substantial divergence (>0.5) for N_fft=400, got {max_diff}"


def test_audit_reports_and_json_gate_passed():
    """Verify that audit reports and JSON exist, contain PRE_SYNTHESIS_GATE_PASSED, and respect governance."""
    assert AUDIT_REPORT_ROOT.exists(), "Root audit report missing"
    assert AUDIT_REPORT_EXP.exists(), "Experiments audit report missing"
    assert AUDIT_JSON.exists(), "Audit JSON missing"

    content_root = AUDIT_REPORT_ROOT.read_text(encoding="utf-8")
    content_exp = AUDIT_REPORT_EXP.read_text(encoding="utf-8")

    for c in [content_root, content_exp]:
        assert "PRE_SYNTHESIS_GATE_PASSED" in c
        assert "FROZEN_CNN_CANDIDATE" in c
        assert "NOT_VALIDATED_ON_NATURAL_SPEECH" in c
        assert "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION" in c
        assert EXPECTED_SHA256 in c
        assert "512" in c
        assert "400" in c

    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 5, f"Expected at least 5 audit fixture entries, got {len(data)}"
    for entry in data:
        assert entry["remediated_frontend_13_1"]["spec_max_diff_vs_tr"] == 0.0
        assert entry["remediated_frontend_13_1"]["logits_max_diff_vs_tr"] == 0.0
        assert entry["remediated_frontend_13_1"]["matches_training_decision"] is True
        assert entry["portable_numpy_512"]["matches_training_decision"] is True
