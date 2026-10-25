"""
Automated Pytest Suite for Vietnamese KWS Field Recording Acquisition Pipeline.
Verifies:
  1. Speaker registration & explicit consent enforcement
  2. Quality gates: clipping detection, abnormal silence, duration bounds
  3. Duplicate SHA256 detection
  4. Raw original audio preservation & 16 kHz standardization
  5. Mock recording session execution
  6. Audit report generation
  7. Security: path traversal rejection
  8. Duplicate recording ID rejection
  9. Compressed .m4a audio ingestion & decoding
  10. Windows hardware recorder configuration
  11. Complete test isolation: zero production data pollution regression
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
import hashlib
import pytest
import numpy as np
import soundfile as sf
import pandas as pd

from src.acquisition.speaker_consent import (
    register_speaker, verify_speaker_consent, get_speaker_metadata, load_consent_ledger
)
from src.acquisition.ingest_field_recordings import (
    ingest_recording, validate_field_audio, compute_sha256, batch_ingest_directory,
    RAW_AUDIO_DIR, STANDARDIZED_AUDIO_DIR, QUARANTINE_LOG_PATH, MANIFEST_PATH
)
from src.acquisition.record_command_cli import run_recording_session, generate_mock_human_audio, AudioRecorder
from src.acquisition.audit_field_campaign import audit_field_campaign


@pytest.fixture(autouse=True)
def isolate_field_environment(tmp_path, monkeypatch):
    """
    Autouse fixture that isolates all field acquisition write operations
    to a temporary sandbox directory for each test.
    Guarantees that production manifests, ledgers, audio files, and audit reports
    are NEVER touched or polluted during automated test execution.
    """
    sandbox_dir = tmp_path / "field_sandbox"
    raw_dir = sandbox_dir / "raw"
    std_dir = sandbox_dir / "processed"
    raw_audio = raw_dir / "audio_raw"
    std_audio = std_dir / "audio_16k_standardized"
    quarantine_dir = raw_dir / "quarantined"
    quarantine_log = quarantine_dir / "quarantine_log.csv"
    manifest = raw_dir / "field_collection_manifest.csv"
    ledger = sandbox_dir / "consent_ledger.json"
    report_path = sandbox_dir / "field_recording_campaign_audit.md"

    raw_audio.mkdir(parents=True, exist_ok=True)
    std_audio.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Patch constants in src.acquisition.speaker_consent
    monkeypatch.setattr("src.acquisition.speaker_consent.LEDGER_PATH", ledger)

    # Patch constants in src.acquisition.ingest_field_recordings
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.FIELD_BASE_DIR", sandbox_dir)
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.LEDGER_PATH", ledger)
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.RAW_AUDIO_DIR", raw_audio)
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.STANDARDIZED_AUDIO_DIR", std_audio)
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.QUARANTINE_DIR", quarantine_dir)
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.QUARANTINE_LOG_PATH", quarantine_log)
    monkeypatch.setattr("src.acquisition.ingest_field_recordings.MANIFEST_PATH", manifest)

    # Patch constants in src.acquisition.audit_field_campaign
    monkeypatch.setattr("src.acquisition.audit_field_campaign.LEDGER_PATH", ledger)
    monkeypatch.setattr("src.acquisition.audit_field_campaign.MANIFEST_PATH", manifest)
    monkeypatch.setattr("src.acquisition.audit_field_campaign.QUARANTINE_LOG_PATH", quarantine_log)
    monkeypatch.setattr("src.acquisition.audit_field_campaign.REPORT_PATH", report_path)

    # Patch constants in this test module
    curr_mod = sys.modules[__name__]
    monkeypatch.setattr(curr_mod, "RAW_AUDIO_DIR", raw_audio)
    monkeypatch.setattr(curr_mod, "STANDARDIZED_AUDIO_DIR", std_audio)
    monkeypatch.setattr(curr_mod, "QUARANTINE_LOG_PATH", quarantine_log)
    monkeypatch.setattr(curr_mod, "MANIFEST_PATH", manifest)

    return {
        "sandbox_dir": sandbox_dir,
        "raw_dir": raw_dir,
        "std_dir": std_dir,
        "raw_audio": raw_audio,
        "std_audio": std_audio,
        "quarantine_log": quarantine_log,
        "manifest": manifest,
        "ledger": ledger,
        "report_path": report_path
    }


@pytest.fixture
def setup_test_speaker(isolate_field_environment):
    """
    Registers a standard test speaker inside the isolated sandbox ledger.
    """
    spk_id = "SPK_FIELD_N001_TESTSUITE"
    register_speaker(
        dialect="Northern",
        gender="FEMALE",
        age_bracket="18-30",
        consent_given=True,
        device_model="Test Harness Device",
        acoustic_environment="quiet_room",
        custom_speaker_id=spk_id,
        ledger_path=isolate_field_environment["ledger"]
    )
    return spk_id


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
    import uuid
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

    session_id = f"SES_PRESERVE_{uuid.uuid4().hex[:8]}"
    res = ingest_recording(
        input_audio_path=test_wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id=session_id,
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
    import uuid
    spk_id = setup_test_speaker
    sr = 16000
    audio = generate_mock_human_audio("phải", duration_sec=0.7)
    test_wav = tmp_path / "dup_sample.wav"
    sf.write(test_wav, audio, sr, subtype="PCM_16")

    session_id = f"SES_DUP_{uuid.uuid4().hex[:8]}"
    # Ingest first take
    res1 = ingest_recording(test_wav, spk_id, "PHAI", session_id, take_index=1)
    assert res1["status"] == "APPROVED"

    # Attempt to ingest exact same file again
    res2 = ingest_recording(test_wav, spk_id, "PHAI", session_id, take_index=2)
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
    # Verify manifest updated in sandbox
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
    assert "File Format & Audio Manifest Integrity Audit" in content


def test_security_path_traversal_rejection(tmp_path, setup_test_speaker):
    """
    Regression test: Verifies that malicious speaker_id or session_id with directory traversal
    is blocked immediately and raises ValueError.
    """
    test_wav = tmp_path / "valid.wav"
    audio = generate_mock_human_audio("trái")
    sf.write(test_wav, audio, 16000, subtype="PCM_16")

    # Malicious speaker_id with traversal
    with pytest.raises(ValueError, match="Invalid characters in speaker_id"):
        ingest_recording(test_wav, "../evil_speaker", "TRAI", "SES_001")

    # Malicious session_id with traversal
    with pytest.raises(ValueError, match="Invalid characters in session_id"):
        ingest_recording(test_wav, setup_test_speaker, "TRAI", "../../etc_passwd")


def test_duplicate_recording_id_rejection(tmp_path, setup_test_speaker):
    """
    Regression test: Verifies that ingesting different audio files with identical recording_id
    (same speaker, session, command, take) is quarantined as duplicate recording ID.
    """
    import uuid
    spk_id = setup_test_speaker
    sr = 16000
    session_id = f"SES_DUP_RECID_{uuid.uuid4().hex[:8]}"

    # Take 1
    wav1 = tmp_path / "take1.wav"
    sf.write(wav1, generate_mock_human_audio("trái", duration_sec=0.7), sr, subtype="PCM_16")
    res1 = ingest_recording(wav1, spk_id, "TRAI", session_id=session_id, take_index=1)
    assert res1["status"] == "APPROVED"

    # Take 2 with different audio content but same take_index -> same recording_id
    wav2 = tmp_path / "take2_diff_audio.wav"
    sf.write(wav2, generate_mock_human_audio("phải", duration_sec=0.75), sr, subtype="PCM_16")
    res2 = ingest_recording(wav2, spk_id, "TRAI", session_id=session_id, take_index=1)
    assert res2["status"] == "QUARANTINED"
    assert "duplicate_recording_id" in res2["reason"]


def test_m4a_audio_ingestion_and_decoding(tmp_path, setup_test_speaker):
    """
    Regression test: Verifies that compressed .m4a audio (e.g. from iPhone Voice Memos)
    is decoded properly, standardized to exact 16kHz mono 16-bit PCM (16,000 samples),
    and raw .m4a original is preserved untouched.
    """
    import subprocess
    import uuid
    spk_id = setup_test_speaker
    session_id = f"SES_M4A_{uuid.uuid4().hex[:8]}"

    # Create temporary WAV then convert to m4a via ffmpeg
    wav_temp = tmp_path / "source.wav"
    m4a_temp = tmp_path / "voice_memo.m4a"
    audio = generate_mock_human_audio("xuống", duration_sec=0.85)
    sf.write(wav_temp, audio, 16000, subtype="PCM_16")

    # Encode to AAC/m4a
    cmd = ["ffmpeg", "-y", "-i", str(wav_temp), "-c:a", "aac", str(m4a_temp)]
    res_ffmpeg = subprocess.run(cmd, capture_output=True)
    if res_ffmpeg.returncode != 0:
        pytest.skip("ffmpeg not available for m4a generation in test environment")

    # Ingest .m4a
    res = ingest_recording(m4a_temp, spk_id, "XUONG", session_id=session_id, take_index=1)
    assert res["status"] == "APPROVED"

    # Verify standardized audio properties
    std_path = Path(res["processed_path"])
    assert std_path.exists()
    info = sf.info(str(std_path))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.frames == 16000  # Exactly 16,000 samples (1.00s)
    assert info.subtype == "PCM_16"

    # Verify raw preservation preserves .m4a extension and bytes
    raw_path = Path(res["raw_filepath"])
    assert raw_path.exists()
    assert raw_path.suffix == ".m4a"
    assert raw_path.read_bytes() == m4a_temp.read_bytes()


def test_windows_hardware_recorder_configuration():
    """
    Regression test: Verifies AudioRecorder initializes with 44.1 kHz stereo configuration
    to prevent Windows Realtek waveIn driver failure (MCI Error 328/322).
    """
    recorder = AudioRecorder()
    assert recorder.sample_rate == 44100
    assert recorder.channels == 2
    assert recorder.bits_per_sample == 16


def test_no_production_pollution_regression(isolate_field_environment):
    """
    Regression test: Verifies that executing speaker registration, recording ingestion,
    CLI sessions, and campaign audit reports in tests strictly preserves all production
    manifests, consent ledgers, quarantine logs, audit reports, and audio directories
    without altering a single byte or creating any untracked production files.
    """
    prod_manifest = Path("data/raw/field_recordings/field_collection_manifest.csv")
    prod_ledger = Path("data/raw/field_recordings/consent_ledger.json")
    prod_quarantine = Path("data/raw/field_recordings/quarantined/quarantine_log.csv")
    prod_raw_dir = Path("data/raw/field_recordings/audio_raw")
    prod_std_dir = Path("data/raw/field_recordings/audio_16k_standardized")
    prod_report = Path("reports/field_recording_campaign_audit.md")
    prod_test_v2 = Path("data/manifests/test_manifest_v2.csv")

    def file_hash_and_size(path: Path):
        if not path.exists():
            return None, 0
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest(), len(data)

    def list_files_recursive(dir_path: Path):
        if not dir_path.exists():
            return set()
        return set(p.relative_to(dir_path) for p in dir_path.rglob("*.*") if p.is_file())

    # Snapshot state of production files before test actions
    manifest_hash_before, manifest_size_before = file_hash_and_size(prod_manifest)
    ledger_hash_before, ledger_size_before = file_hash_and_size(prod_ledger)
    quarantine_hash_before, quarantine_size_before = file_hash_and_size(prod_quarantine)
    report_hash_before, report_size_before = file_hash_and_size(prod_report)
    test_v2_hash_before, _ = file_hash_and_size(prod_test_v2)
    raw_files_before = list_files_recursive(prod_raw_dir)
    std_files_before = list_files_recursive(prod_std_dir)

    # Invariant: test_manifest_v2.csv must match exact frozen SHA-256
    assert test_v2_hash_before.upper() == "9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13"

    # --- Execute comprehensive acquisition actions in sandbox ---
    spk_id = "SPK_REGRESSION_SANDBOX"
    register_speaker(
        dialect="Southern",
        gender="MALE",
        age_bracket="18-30",
        consent_given=True,
        device_model="Regression Test Rig",
        acoustic_environment="isolated_box",
        custom_speaker_id=spk_id
    )

    sr = 16000
    tmp_dir = isolate_field_environment["sandbox_dir"] / "test_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Action 1: Ingest approved recording
    sample_wav = tmp_dir / "sample_appr.wav"
    sf.write(sample_wav, generate_mock_human_audio("trái", duration_sec=0.7), sr, subtype="PCM_16")
    res_appr = ingest_recording(
        input_audio_path=sample_wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_REGRESSION_001",
        take_index=1
    )
    assert res_appr["status"] == "APPROVED"

    # Action 2: Ingest quarantined recording
    sample_clip = tmp_dir / "sample_clip.wav"
    sf.write(sample_clip, np.ones(sr, dtype=np.float32), sr, subtype="PCM_16")
    res_clip = ingest_recording(
        input_audio_path=sample_clip,
        speaker_id=spk_id,
        command_label="PHAI",
        session_id="SES_REGRESSION_001",
        take_index=2
    )
    assert res_clip["status"] == "QUARANTINED"

    # Action 3: Execute mock recording session CLI
    run_recording_session(
        speaker_id=spk_id,
        commands=["LEN", "XUONG"],
        takes_per_command=1,
        session_id="SES_REGRESSION_CLI",
        mock_mode=True
    )

    # Action 4: Run campaign auditor
    audit_field_campaign()

    # Action 5: Execute batch directory ingestion
    batch_in_dir = tmp_dir / "batch_input"
    batch_in_dir.mkdir(parents=True, exist_ok=True)
    batch_wav = batch_in_dir / "len_take1.wav"
    sf.write(batch_wav, generate_mock_human_audio("lên", duration_sec=0.6), sr, subtype="PCM_16")
    res_batch = batch_ingest_directory(
        input_dir=batch_in_dir,
        speaker_id=spk_id,
        session_id="SES_REGRESSION_BATCH",
        command_label="LEN"
    )
    assert len(res_batch) == 1
    assert res_batch[0]["status"] == "APPROVED"

    # Verify that test data was indeed created in sandbox
    assert isolate_field_environment["manifest"].exists(), "Sandbox manifest was not created"
    assert isolate_field_environment["ledger"].exists(), "Sandbox ledger was not created"
    assert isolate_field_environment["quarantine_log"].exists(), "Sandbox quarantine was not created"
    assert isolate_field_environment["report_path"].exists(), "Sandbox report was not created"

    # Snapshot state of production files after test actions
    manifest_hash_after, manifest_size_after = file_hash_and_size(prod_manifest)
    ledger_hash_after, ledger_size_after = file_hash_and_size(prod_ledger)
    quarantine_hash_after, quarantine_size_after = file_hash_and_size(prod_quarantine)
    report_hash_after, report_size_after = file_hash_and_size(prod_report)
    test_v2_hash_after, _ = file_hash_and_size(prod_test_v2)
    raw_files_after = list_files_recursive(prod_raw_dir)
    std_files_after = list_files_recursive(prod_std_dir)

    # Assert 100% preservation of all production artifacts
    assert manifest_hash_after == manifest_hash_before, "Production manifest SHA-256 changed during test run!"
    assert manifest_size_after == manifest_size_before, "Production manifest size changed during test run!"

    assert ledger_hash_after == ledger_hash_before, "Production consent ledger SHA-256 changed during test run!"
    assert ledger_size_after == ledger_size_before, "Production consent ledger size changed during test run!"

    assert quarantine_hash_after == quarantine_hash_before, "Production quarantine log SHA-256 changed during test run!"
    assert quarantine_size_after == quarantine_size_before, "Production quarantine log size changed during test run!"

    assert report_hash_after == report_hash_before, "Production campaign audit report SHA-256 changed during test run!"
    assert report_size_after == report_size_before, "Production campaign audit report size changed during test run!"

    assert test_v2_hash_after == test_v2_hash_before, "Frozen test_manifest_v2.csv SHA-256 changed during test run!"

    assert raw_files_after == raw_files_before, (
        f"Production audio_raw directory was polluted! New files: {raw_files_after - raw_files_before}"
    )
    assert std_files_after == std_files_before, (
        f"Production audio_16k_standardized directory was polluted! New files: {std_files_after - std_files_before}"
    )
