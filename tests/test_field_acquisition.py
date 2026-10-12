"""
Automated Pytest Suite for Vietnamese KWS Field Recording Acquisition Pipeline.
Verifies:
  1. Speaker registration & explicit consent enforcement
  2. Quality gates: clipping detection, abnormal silence, duration bounds
  3. Duplicate SHA256 detection
  4. Raw original audio preservation & 16 kHz standardization
  5. Mock recording session execution
  6. Audit report generation
"""

import sys
from pathlib import Path

# Ensure UTF-8 and sys.path
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
import soundfile as sf
import pandas as pd

from src.acquisition.speaker_consent import (
    register_speaker, verify_speaker_consent, get_speaker_metadata, load_consent_ledger
)
from src.acquisition.ingest_field_recordings import (
    ingest_recording, validate_field_audio, compute_sha256,
    RAW_AUDIO_DIR, STANDARDIZED_AUDIO_DIR, QUARANTINE_LOG_PATH, MANIFEST_PATH
)
from src.acquisition.record_command_cli import run_recording_session, generate_mock_human_audio
from src.acquisition.audit_field_campaign import audit_field_campaign


@pytest.fixture(scope="module")
def setup_test_speaker():
    spk_id = "SPK_FIELD_N001_TESTSUITE"
    register_speaker(
        dialect="Northern",
        gender="FEMALE",
        age_bracket="18-30",
        consent_given=True,
        device_model="Test Harness Device",
        acoustic_environment="quiet_room",
        custom_speaker_id=spk_id
    )
    yield spk_id


def test_consent_enforcement(setup_test_speaker):
    spk_id = setup_test_speaker
    assert verify_speaker_consent(spk_id) is True
    assert verify_speaker_consent("SPK_NON_EXISTENT") is False

    # Attempting to register without consent must fail
    with pytest.raises(ValueError, match="consent"):
        register_speaker(
            dialect="Northern",
            gender="MALE",
            consent_given=False,
            custom_speaker_id="SPK_NO_CONSENT"
        )


def test_quality_gate_clipping_rejection(tmp_path, setup_test_speaker):
    spk_id = setup_test_speaker
    sr = 16000
    # Create clipped audio (saturated to 1.0)
    audio = np.ones(sr, dtype=np.float32)
    test_wav = tmp_path / "clipped_sample.wav"
    sf.write(test_wav, audio, sr, subtype="PCM_16")

    res = ingest_recording(
        input_audio_path=test_wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_TEST_CLIP"
    )
    assert res["status"] == "QUARANTINED"
    assert "clipping" in res["reason"]


def test_quality_gate_silence_rejection(tmp_path, setup_test_speaker):
    spk_id = setup_test_speaker
    sr = 16000
    # Create very low energy audio (RMS < -60 dBFS)
    audio = np.random.randn(sr).astype(np.float32) * 1e-4
    test_wav = tmp_path / "silent_sample.wav"
    sf.write(test_wav, audio, sr, subtype="PCM_16")

    res = ingest_recording(
        input_audio_path=test_wav,
        speaker_id=spk_id,
        command_label="PHAI",
        session_id="SES_TEST_SILENCE"
    )
    assert res["status"] == "QUARANTINED"
    assert "abnormal_silence" in res["reason"]


def test_quality_gate_duration_bounds(tmp_path, setup_test_speaker):
    spk_id = setup_test_speaker
    sr = 16000

    # Truncated (< 0.25s)
    short_audio = generate_mock_human_audio("lên", duration_sec=0.15)
    short_wav = tmp_path / "short_sample.wav"
    sf.write(short_wav, short_audio, sr, subtype="PCM_16")
    res_short = ingest_recording(short_wav, spk_id, "LEN", "SES_TEST_DUR")
    assert res_short["status"] == "QUARANTINED"
    assert "truncated" in res_short["reason"]

    # Excessive (> 2.5s)
    long_audio = np.tile(generate_mock_human_audio("xuống", duration_sec=0.8), 4)  # 3.2s
    long_wav = tmp_path / "long_sample.wav"
    sf.write(long_wav, long_audio, sr, subtype="PCM_16")
    res_long = ingest_recording(long_wav, spk_id, "XUONG", "SES_TEST_DUR")
    assert res_long["status"] == "QUARANTINED"
    assert "excessive" in res_long["reason"]


def test_raw_preservation_and_standardization(tmp_path, setup_test_speaker):
    spk_id = setup_test_speaker
    sr = 44100  # non-standard mobile sample rate
    dur_sec = 0.65
    audio = generate_mock_human_audio("trái", duration_sec=dur_sec)
    # Resample to 44.1k for test
    import librosa
    audio_44k = librosa.resample(audio, orig_sr=16000, target_sr=sr)
    
    test_wav = tmp_path / "genuine_take_44k.wav"
    sf.write(test_wav, audio_44k, sr, subtype="PCM_16")
    raw_bytes_original = test_wav.read_bytes()

    res = ingest_recording(
        input_audio_path=test_wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_PRESERVE_TEST",
        take_index=1
    )
    assert res["status"] == "APPROVED"
    assert res["recording_id"] is not None

    # Verify untouched raw preservation
    raw_saved = Path(res["raw_filepath"])
    assert raw_saved.exists()
    assert raw_saved.read_bytes() == raw_bytes_original

    # Verify standardized copy
    std_saved = Path(res["processed_path"])
    assert std_saved.exists()
    info = sf.info(str(std_saved))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.frames == 16000
    assert info.subtype == "PCM_16"


def test_duplicate_sha256_detection(tmp_path, setup_test_speaker):
    spk_id = setup_test_speaker
    sr = 16000
    audio = generate_mock_human_audio("phải", duration_sec=0.7)
    test_wav = tmp_path / "dup_sample.wav"
    sf.write(test_wav, audio, sr, subtype="PCM_16")

    # Ingest first take
    res1 = ingest_recording(test_wav, spk_id, "PHAI", "SES_DUP_TEST", take_index=1)
    assert res1["status"] == "APPROVED"

    # Attempt to ingest exact same file again
    res2 = ingest_recording(test_wav, spk_id, "PHAI", "SES_DUP_TEST", take_index=2)
    assert res2["status"] == "QUARANTINED"
    assert "duplicate" in res2["reason"]


def test_mock_cli_recording_session(setup_test_speaker):
    import uuid
    spk_id = setup_test_speaker
    session_id = f"SES_MOCK_{uuid.uuid4().hex[:8]}"
    # Runs the recording CLI in mock mode
    run_recording_session(
        speaker_id=spk_id,
        commands=["TRAI", "LEN"],
        takes_per_command=2,
        session_id=session_id,
        mock_mode=True
    )
    # Verify manifest updated
    assert MANIFEST_PATH.exists()
    df = pd.read_csv(MANIFEST_PATH)
    test_records = df[df["session_id"] == session_id]
    assert len(test_records) == 4  # 2 commands * 2 takes


def test_audit_field_campaign_generation():
    report_file = audit_field_campaign()
    assert Path(report_file).exists()
    content = Path(report_file).read_text(encoding="utf-8")
    assert "Human Field Recording Campaign Audit Report" in content
    assert "TRAI" in content
    assert "Quality Gate Validation" in content
