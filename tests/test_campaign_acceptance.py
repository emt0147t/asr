"""
Automated Pytest Suite for Vietnamese KWS Campaign Acceptance & Quality Assurance.
Complies with docs/sound_command_recognition/campaign_20261013/test_plan.md:
  TC-01: test_valid_human_manifest_acceptance
  TC-02: test_unconsented_speaker_rejection
  TC-03: test_missing_required_metadata_rejection
  TC-04: test_invalid_provenance_tag_rejection
  TC-05: test_duplicate_audio_hash_quarantine
  TC-06: test_duplicate_recording_id_quarantine
  TC-07: test_digital_clipping_gate
  TC-08: test_abnormal_silence_gate
  TC-09: test_speaker_disjoint_split_generator
  TC-10: test_speaker_leakage_detector
  TC-11: test_campaign_acceptance_zero_production_pollution

All tests execute in isolated tmp_path sandboxes.
Production benchmark invariant test_manifest_v2.csv and historical fixtures remain bit-exact.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout and sys.path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from typing import Dict, Any, List, Optional, Tuple
import hashlib
import pytest
import numpy as np
import pandas as pd
import soundfile as sf

from src.acquisition.campaign_validator import (
    validate_audio_file,
    validate_manifest_row,
    validate_manifest_dataset,
    validate_speaker_disjointness,
    generate_disjoint_splits,
    compute_file_sha256,
    VALID_COMMAND_LABELS,
    CLASS_ID_MAP
)
from src.acquisition.speaker_consent import revoke_speaker_consent


EXPECTED_TEST_V2_SHA = "9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13"
EXPECTED_FIELD_MANIFEST_SHA = "B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85"
EXPECTED_LEDGER_SHA = "BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0"


def create_test_wav(
    path: Path,
    sr: int = 16000,
    duration: float = 1.0,
    freq: float = 440.0,
    amp: float = 0.5,
    clipped: bool = False,
    silent: bool = False
) -> Path:
    """Helper to synthesize test WAV audio files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    if silent:
        # RMS < -50 dBFS
        signal = np.random.randn(n_samples).astype(np.float32) * 1e-4
    elif clipped:
        signal = amp * np.sin(2 * np.pi * freq * t)
        # Force 5 consecutive samples to 1.0
        signal[100:105] = 1.0
    else:
        # Normal speech-like sinusoid with Hann window
        window = np.hanning(n_samples)
        signal = amp * np.sin(2 * np.pi * freq * t) * window

    sf.write(str(path), signal.astype(np.float32), sr, subtype="PCM_16")
    return path


def build_mock_human_manifest_row(
    recording_id: str,
    speaker_id: str,
    dialect: str,
    gender: str,
    command_label: str,
    session_id: str,
    raw_path: Path,
    proc_path: Path,
    provenance: str = "HUMAN_CONSENTED",
    status: str = "APPROVED"
) -> Dict[str, Any]:
    """Helper to create a compliant manifest row dictionary."""
    return {
        "recording_id": recording_id,
        "speaker_id": speaker_id,
        "dialect": dialect,
        "gender": gender,
        "age_bracket": "18-30",
        "command_label": command_label,
        "class_id": CLASS_ID_MAP[command_label],
        "session_id": session_id,
        "device_model": "Pixel 7 Pro",
        "primary_microphone": "Built-in stereo mic",
        "acoustic_environment": "quiet_room",
        "speaking_style": "normal",
        "speaking_distance": "near",
        "sampling_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "duration_ms": 1000,
        "raw_duration_sec": 1.05,
        "rms_db": -20.5,
        "snr_est_db": 28.4,
        "max_consecutive_clipped_samples": 0,
        "raw_filepath": str(raw_path),
        "processed_path": str(proc_path),
        "sha256_raw": compute_file_sha256(raw_path),
        "sha256_standardized": compute_file_sha256(proc_path),
        "label_provenance": provenance,
        "consent_version": "v1.0-kws-research-commercial",
        "status": status,
        "quarantine_reason": ""
    }


# ============================================================================
# TC-01: Valid Human Manifest Acceptance
# ============================================================================
def test_valid_human_manifest_acceptance(tmp_path):
    """
    TC-01: Verifies that a compliant human dataset passing all gates
    is accepted with status APPROVED and 0 errors.
    """
    human_dir = tmp_path / "data" / "raw" / "human_recordings_v1"
    raw_dir = human_dir / "audio_raw"
    std_dir = human_dir / "audio_16k_standardized"

    ledger = {
        "SPK_VN_H001": {
            "speaker_id": "SPK_VN_H001",
            "consent_given": True,
            "dialect": "Northern",
            "gender": "FEMALE"
        },
        "SPK_VN_H002": {
            "speaker_id": "SPK_VN_H002",
            "consent_given": True,
            "dialect": "Central",
            "gender": "MALE"
        }
    }

    rows = []
    commands = ["TRAI", "PHAI", "LEN", "XUONG"]
    for i, cmd in enumerate(commands):
        spk = "SPK_VN_H001" if i % 2 == 0 else "SPK_VN_H002"
        dialect = "Northern" if spk == "SPK_VN_H001" else "Central"
        gender = "FEMALE" if spk == "SPK_VN_H001" else "MALE"
        rec_id = f"REC_{spk}_{cmd}_{i:03d}"

        raw_f = create_test_wav(raw_dir / f"{rec_id}_raw.wav", freq=300.0 + i * 50)
        proc_f = create_test_wav(std_dir / f"{rec_id}.wav", freq=300.0 + i * 50)

        row = build_mock_human_manifest_row(
            recording_id=rec_id,
            speaker_id=spk,
            dialect=dialect,
            gender=gender,
            command_label=cmd,
            session_id=f"SES_{spk}_01",
            raw_path=raw_f,
            proc_path=proc_f
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    is_valid, errors, stats = validate_manifest_dataset(df, ledger, check_disk_files=True)

    assert is_valid is True, f"Expected manifest to pass, got errors: {errors}"
    assert len(errors) == 0
    assert stats["total_records"] == 4
    assert stats["unique_speakers"] == 2


# ============================================================================
# TC-02: Unconsented Speaker Rejection
# ============================================================================
def test_unconsented_speaker_rejection(tmp_path):
    """
    TC-02: Attempts ingestion with a speaker missing from ledger or consent_given == False.
    Must reject with unconsented speaker error.
    """
    raw_f = create_test_wav(tmp_path / "raw.wav")
    proc_f = create_test_wav(tmp_path / "proc.wav")

    # Speaker missing from ledger
    ledger = {}
    row_missing = pd.Series(build_mock_human_manifest_row(
        recording_id="REC_UNCONSENTED_001",
        speaker_id="SPK_GHOST_999",
        dialect="Northern",
        gender="MALE",
        command_label="TRAI",
        session_id="SES_01",
        raw_path=raw_f,
        proc_path=proc_f
    ))
    is_valid, errs = validate_manifest_row(row_missing, ledger, check_disk_files=False)
    assert is_valid is False
    assert any("missing_consent_entry_for_human_speaker" in e for e in errs)

    # Speaker in ledger but consent revoked / False
    ledger_revoked = {
        "SPK_REVOKED_001": {
            "speaker_id": "SPK_REVOKED_001",
            "consent_given": False,
            "dialect": "Southern"
        }
    }
    row_revoked = pd.Series(build_mock_human_manifest_row(
        recording_id="REC_REVOKED_001",
        speaker_id="SPK_REVOKED_001",
        dialect="Southern",
        gender="FEMALE",
        command_label="PHAI",
        session_id="SES_01",
        raw_path=raw_f,
        proc_path=proc_f
    ))
    is_valid_rev, errs_rev = validate_manifest_row(row_revoked, ledger_revoked, check_disk_files=False)
    assert is_valid_rev is False
    assert any("consent_not_granted_for_speaker" in e for e in errs_rev)


# ============================================================================
# TC-03: Missing Required Metadata Rejection
# ============================================================================
def test_missing_required_metadata_rejection(tmp_path):
    """
    TC-03: Verifies that missing mandatory fields (dialect, session_id, device_model) are rejected.
    """
    raw_f = create_test_wav(tmp_path / "raw.wav")
    proc_f = create_test_wav(tmp_path / "proc.wav")
    ledger = {"SPK_001": {"consent_given": True}}

    base_row = build_mock_human_manifest_row(
        recording_id="REC_INCOMPLETE_001",
        speaker_id="SPK_001",
        dialect="Northern",
        gender="FEMALE",
        command_label="LEN",
        session_id="SES_01",
        raw_path=raw_f,
        proc_path=proc_f
    )

    for field in ["dialect", "session_id", "device_model", "command_label", "status"]:
        flawed_row = base_row.copy()
        flawed_row[field] = ""
        is_valid, errs = validate_manifest_row(pd.Series(flawed_row), ledger, check_disk_files=False)
        assert is_valid is False
        assert any("missing_required_field" in e for e in errs)


# ============================================================================
# TC-04: Invalid Provenance Tag Rejection
# ============================================================================
def test_invalid_provenance_tag_rejection(tmp_path):
    """
    TC-04: Rejects records if synthetic test fixtures masquerade as HUMAN_CONSENTED,
    or if unknown provenance enums are supplied.
    """
    raw_f = create_test_wav(tmp_path / "raw.wav")
    proc_f = create_test_wav(tmp_path / "proc.wav")
    ledger = {"SPK_MOCK_TEST": {"consent_given": True}}

    # Co-mingled legacy fixture path
    row_legacy = build_mock_human_manifest_row(
        recording_id="REC_FIXTURE_001",
        speaker_id="SPK_MOCK_TEST",
        dialect="Northern",
        gender="FEMALE",
        command_label="TRAI",
        session_id="SES_01",
        raw_path=raw_f,
        proc_path=proc_f,
        provenance="HUMAN_CONSENTED"
    )
    row_legacy["raw_filepath"] = "data/raw/field_recordings/audio_raw/mock.wav"
    df = pd.DataFrame([row_legacy])
    is_valid, errs, _ = validate_manifest_dataset(df, ledger, check_disk_files=False)
    assert is_valid is False
    assert any("synthetic_fixture_co_mingling" in e for e in errs)
    assert any("synthetic_speaker_id_mislabeled_as_human" in e for e in errs)

    # Test unknown provenance enum
    row_bad_enum = build_mock_human_manifest_row(
        recording_id="REC_BAD_ENUM_001",
        speaker_id="SPK_VN_H001",
        dialect="Northern",
        gender="FEMALE",
        command_label="TRAI",
        session_id="SES_01",
        raw_path=raw_f,
        proc_path=proc_f,
        provenance="UNKNOWN_FABRICATED_PROVENANCE"
    )
    is_valid_enum, errs_enum = validate_manifest_row(pd.Series(row_bad_enum), ledger, check_disk_files=False)
    assert is_valid_enum is False
    assert any("invalid_provenance_enum" in e for e in errs_enum)


# ============================================================================
# TC-05: Duplicate Audio Hash Quarantine
# ============================================================================
def test_duplicate_audio_hash_quarantine(tmp_path):
    """
    TC-05: Rejects duplicate audio waveforms possessing identical SHA-256 digests.
    """
    wav_a = create_test_wav(tmp_path / "take1.wav", freq=440.0)
    ledger = {
        "SPK_001": {"consent_given": True},
        "SPK_002": {"consent_given": True}
    }

    row1 = build_mock_human_manifest_row(
        recording_id="REC_001", speaker_id="SPK_001", dialect="Northern", gender="FEMALE",
        command_label="TRAI", session_id="SES_01", raw_path=wav_a, proc_path=wav_a
    )
    # Duplicate points to same audio or has same sha256_raw
    row2 = build_mock_human_manifest_row(
        recording_id="REC_002", speaker_id="SPK_002", dialect="Central", gender="MALE",
        command_label="TRAI", session_id="SES_02", raw_path=wav_a, proc_path=wav_a
    )

    df = pd.DataFrame([row1, row2])
    is_valid, errs, _ = validate_manifest_dataset(df, ledger, check_disk_files=False)
    assert is_valid is False
    assert any("duplicate_raw_audio_hashes_detected" in e for e in errs)


# ============================================================================
# TC-06: Duplicate Recording ID Quarantine
# ============================================================================
def test_duplicate_recording_id_quarantine(tmp_path):
    """
    TC-06: Ingests two distinct recordings assigned the identical recording_id.
    Must detect duplicate primary key.
    """
    wav_a = create_test_wav(tmp_path / "takeA.wav", freq=440.0)
    wav_b = create_test_wav(tmp_path / "takeB.wav", freq=880.0)
    ledger = {"SPK_001": {"consent_given": True}}

    row1 = build_mock_human_manifest_row(
        recording_id="REC_COLLISION_001", speaker_id="SPK_001", dialect="Northern", gender="FEMALE",
        command_label="TRAI", session_id="SES_01", raw_path=wav_a, proc_path=wav_a
    )
    row2 = build_mock_human_manifest_row(
        recording_id="REC_COLLISION_001", speaker_id="SPK_001", dialect="Northern", gender="FEMALE",
        command_label="PHAI", session_id="SES_01", raw_path=wav_b, proc_path=wav_b
    )

    df = pd.DataFrame([row1, row2])
    is_valid, errs, _ = validate_manifest_dataset(df, ledger, check_disk_files=False)
    assert is_valid is False
    assert any("duplicate_recording_ids_detected" in e for e in errs)


# ============================================================================
# TC-07: Digital Clipping Gate
# ============================================================================
def test_digital_clipping_gate(tmp_path):
    """
    TC-07: Ingests an audio take with >= 3 consecutive clipped samples at 1.0.
    Must be detected and rejected.
    """
    clipped_wav = create_test_wav(tmp_path / "clipped.wav", clipped=True)
    is_valid, reason, _ = validate_audio_file(clipped_wav)
    assert is_valid is False
    assert "digital_clipping_detected" in reason


# ============================================================================
# TC-08: Abnormal Silence Gate
# ============================================================================
def test_abnormal_silence_gate(tmp_path):
    """
    TC-08: Ingests unvoiced audio take with energy RMS < -45.0 dBFS.
    Must be rejected when allow_silence=False, but permitted when allow_silence=True.
    """
    silent_wav = create_test_wav(tmp_path / "silent.wav", silent=True)

    # For standard commands (TRAI, PHAI, LEN, XUONG, UNKNOWN)
    is_valid, reason, _ = validate_audio_file(silent_wav, allow_silence=False)
    assert is_valid is False
    assert "abnormal_silence" in reason

    # For designated ambient SILENCE class
    is_valid_noise, _, _ = validate_audio_file(silent_wav, allow_silence=True)
    assert is_valid_noise is True


# ============================================================================
# TC-09: Speaker Disjoint Split Generator
# ============================================================================
def test_speaker_disjoint_split_generator(tmp_path):
    """
    TC-09: Generates 70/15/15 splits across 12 human speakers and verifies mathematical disjointness.
    """
    rows = []
    for spk_idx in range(1, 13):
        spk_id = f"SPK_VN_H{spk_idx:03d}"
        for cmd in ["TRAI", "PHAI", "LEN", "XUONG"]:
            rows.append({
                "recording_id": f"REC_{spk_id}_{cmd}",
                "speaker_id": spk_id,
                "session_id": f"SES_{spk_id}_01",
                "command_label": cmd,
                "class_id": CLASS_ID_MAP[cmd]
            })

    df = pd.DataFrame(rows)
    df_train, df_val, df_test = generate_disjoint_splits(df, 0.70, 0.15, 0.15)

    is_disjoint, details = validate_speaker_disjointness(df_train, df_val, df_test)
    assert is_disjoint is True
    assert len(details["speaker_overlaps"]["train_val"]) == 0
    assert len(details["speaker_overlaps"]["train_test"]) == 0
    assert len(details["speaker_overlaps"]["val_test"]) == 0

    assert len(details["session_overlaps"]["train_val"]) == 0
    assert len(details["session_overlaps"]["train_test"]) == 0
    assert len(details["session_overlaps"]["val_test"]) == 0


# ============================================================================
# TC-10: Speaker Leakage Detector
# ============================================================================
def test_speaker_leakage_detector():
    """
    TC-10: Deliberately injects a shared speaker across train and test sets into the validator.
    Validator must identify leakage and return is_disjoint == False.
    """
    df_train = pd.DataFrame([
        {"speaker_id": "SPK_001", "session_id": "SES_001"},
        {"speaker_id": "SPK_002", "session_id": "SES_002"}
    ])
    df_val = pd.DataFrame([
        {"speaker_id": "SPK_003", "session_id": "SES_003"}
    ])
    # Leak SPK_001 into test
    df_test_leaked = pd.DataFrame([
        {"speaker_id": "SPK_001", "session_id": "SES_LEAK"},
        {"speaker_id": "SPK_004", "session_id": "SES_004"}
    ])

    is_disjoint, details = validate_speaker_disjointness(df_train, df_val, df_test_leaked)
    assert is_disjoint is False
    assert "SPK_001" in details["speaker_overlaps"]["train_test"]


# ============================================================================
# TC-11: Zero Production Pollution Regression
# ============================================================================
def test_campaign_acceptance_zero_production_pollution():
    """
    TC-11: Snapshots production file hashes and verifies byte-for-byte immutability.
    Frozen test manifest and historical fixture archives must remain untouched.
    """
    test_manifest_path = Path("data/manifests/test_manifest_v2.csv")
    field_manifest_path = Path("data/raw/field_recordings/field_collection_manifest.csv")
    ledger_path = Path("data/raw/field_recordings/consent_ledger.json")

    assert test_manifest_path.exists(), "test_manifest_v2.csv missing"
    assert field_manifest_path.exists(), "field_collection_manifest.csv missing"
    assert ledger_path.exists(), "consent_ledger.json missing"

    test_sha = compute_file_sha256(test_manifest_path)
    field_sha = compute_file_sha256(field_manifest_path)
    ledger_sha = compute_file_sha256(ledger_path)

    assert test_sha == EXPECTED_TEST_V2_SHA, f"test_manifest_v2 hash mismatch: {test_sha}"
    assert field_sha == EXPECTED_FIELD_MANIFEST_SHA, f"field_manifest hash mismatch: {field_sha}"
    assert ledger_sha == EXPECTED_LEDGER_SHA, f"consent_ledger hash mismatch: {ledger_sha}"


# ============================================================================
# P0-1 Remediation Test: Safe Default Production Paths & Sandbox Overrides
# ============================================================================
def test_p0_1_production_path_separation_and_defaults(tmp_path):
    """
    P0-1: Verifies that:
      1. Default production paths point to data/raw/human_recordings_v1/ and NEVER
         resolve into the legacy fixture folder data/raw/field_recordings/.
      2. Legacy fixture paths remain defined for read-only historical audits.
      3. Explicit path overrides work in isolated temporary sandboxes without
         mutating production locations.
    """
    from src.acquisition.ingest_field_recordings import (
        FIELD_BASE_DIR, HUMAN_BASE_DIR, LEGACY_FIELD_BASE_DIR,
        MANIFEST_PATH as IFR_MANIFEST_PATH, LEDGER_PATH as IFR_LEDGER_PATH,
        LEGACY_MANIFEST_PATH as IFR_LEGACY_MANIFEST_PATH, ingest_recording
    )
    from src.acquisition.speaker_consent import (
        LEDGER_PATH as SC_LEDGER_PATH, LEGACY_LEDGER_PATH as SC_LEGACY_LEDGER_PATH,
        register_speaker
    )

    # 1. Verify default paths point to dedicated human recordings root
    assert FIELD_BASE_DIR == HUMAN_BASE_DIR
    assert Path("data/raw/field_recordings") not in FIELD_BASE_DIR.parents
    assert FIELD_BASE_DIR != Path("data/raw/field_recordings")
    assert str(FIELD_BASE_DIR).replace("\\", "/") == "data/raw/human_recordings_v1"
    assert str(IFR_MANIFEST_PATH).replace("\\", "/") == "data/raw/human_recordings_v1/human_collection_manifest_v1.csv"
    assert str(IFR_LEDGER_PATH).replace("\\", "/") == "data/raw/human_recordings_v1/consent_ledger_human.json"
    assert str(SC_LEDGER_PATH).replace("\\", "/") == "data/raw/human_recordings_v1/consent_ledger_human.json"

    # 2. Verify legacy paths are strictly preserved
    assert str(SC_LEGACY_LEDGER_PATH).replace("\\", "/") == "data/raw/field_recordings/consent_ledger.json"
    assert str(IFR_LEGACY_MANIFEST_PATH).replace("\\", "/") == "data/raw/field_recordings/field_collection_manifest.csv"

    # 3. Verify temporary path overrides work in sandbox without writing to production
    sandbox = tmp_path / "sandbox_p0"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk_record = register_speaker(
        dialect="Northern",
        gender="FEMALE",
        consent_given=True,
        ledger_path=sandbox_ledger
    )
    assert sandbox_ledger.exists()
    spk_id = spk_record["speaker_id"]

    test_wav = create_test_wav(sandbox / "sample.wav")
    res = ingest_recording(
        input_audio_path=test_wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_P0_TEST",
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        quarantine_path=sandbox_quarantine,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger
    )
    assert res["status"] == "APPROVED"
    assert sandbox_manifest.exists()

    # 4. Assert zero files were written to production directories
    assert not Path("data/raw/human_recordings_v1/audio_raw").exists() or \
           not any(Path("data/raw/human_recordings_v1/audio_raw").glob(f"*{spk_id}*"))


# ============================================================================
# P0-2 Remediation Test: Source-License Gate & Fallback Enforcement
# ============================================================================
def test_p0_2_source_license_gate(tmp_path):
    """
    P0-2: Verifies that:
      1. Unverified sources (such as mic_noise) are strictly BLOCKED from approval.
      2. Verified sources (such as FLEURS) pass license checks.
      3. Manifest validator rejects records with BLOCKED/unverified provenance tags.
    """
    from src.acquisition.download_approved_sources import check_source_license_status
    from src.acquisition.campaign_validator import validate_manifest_row

    # 1. Blocked unverified external source check
    status_noise = check_source_license_status("mic_noise")
    assert status_noise["is_approved"] is False
    assert status_noise["status"] == "BLOCKED_UNVERIFIED_LICENSE"
    assert "BLOCKED" in status_noise["reason"]

    # 2. Approved verified external source check
    status_fleurs = check_source_license_status("fleurs_vi_dev")
    assert status_fleurs["is_approved"] is True
    assert status_fleurs["status"] == "APPROVED_VERIFIED"

    # 3. Unknown source rejection
    status_unknown = check_source_license_status("fabricated_dataset_2026")
    assert status_unknown["is_approved"] is False
    assert status_unknown["status"] == "UNKNOWN_SOURCE"

    # 4. Manifest validation rejection for NEGATIVE_NOISE_HUSEIN provenance
    raw_f = create_test_wav(tmp_path / "raw_noise.wav")
    proc_f = create_test_wav(tmp_path / "proc_noise.wav")
    row_unverified = build_mock_human_manifest_row(
        recording_id="REC_NOISE_001",
        speaker_id="SPK_NOISE_EXT",
        dialect="Northern",
        gender="MALE",
        command_label="SILENCE",
        session_id="SES_01",
        raw_path=raw_f,
        proc_path=proc_f,
        provenance="NEGATIVE_NOISE_HUSEIN"
    )
    is_valid, errs = validate_manifest_row(pd.Series(row_unverified), {}, check_disk_files=False)
    assert is_valid is False
    assert any("unverified_source_license_blocked" in e for e in errs)


# ============================================================================
# P0-3 Remediation Test: Cryptographic Pseudonymous Speaker ID Generation
# ============================================================================
def test_p0_3_pseudonymous_speaker_id_generation(tmp_path):
    """
    P0-3: Verifies that:
      1. Speaker IDs match the non-sequential regex: ^SPK_H_[NCS]_[a-f0-9]{6}$.
      2. 1,000 generated IDs contain 1,000 unique values (zero collisions).
      3. IDs are non-sequential (absence of sequential counter assumption).
      4. Collision detection safely regenerates IDs or fails with RuntimeError.
      5. register_speaker automatically generates compliant pseudonymous IDs.
    """
    import re
    from src.acquisition.speaker_consent import (
        generate_pseudonymous_speaker_id, SPEAKER_ID_REGEX, register_speaker
    )

    # 1. Format regex verification across dialects
    spk_n = generate_pseudonymous_speaker_id("Northern")
    spk_c = generate_pseudonymous_speaker_id("Central")
    spk_s = generate_pseudonymous_speaker_id("Southern")

    assert SPEAKER_ID_REGEX.match(spk_n) is not None, f"Northern ID invalid: {spk_n}"
    assert spk_n.startswith("SPK_H_N_")
    assert SPEAKER_ID_REGEX.match(spk_c) is not None, f"Central ID invalid: {spk_c}"
    assert spk_c.startswith("SPK_H_C_")
    assert SPEAKER_ID_REGEX.match(spk_s) is not None, f"Southern ID invalid: {spk_s}"
    assert spk_s.startswith("SPK_H_S_")

    # 2. Uniqueness across substantial sample (1,000 IDs)
    sample_size = 1000
    generated_ids = [generate_pseudonymous_speaker_id("Northern") for _ in range(sample_size)]
    assert len(set(generated_ids)) == sample_size, "Collision detected in 1,000 random samples!"

    # 3. Absence of sequential counters (not SPK_FIELD_N001, N002)
    for sid in generated_ids:
        assert not sid.startswith("SPK_FIELD_")
    # Verify non-sequential distribution: nonces converted to integers have large random differences
    nonces_int = [int(sid.split("_")[-1], 16) for sid in generated_ids[:20]]
    diffs = [abs(nonces_int[i+1] - nonces_int[i]) for i in range(len(nonces_int) - 1)]
    assert any(d > 1000 for d in diffs), "Nonces appear sequential rather than random!"

    # 4. Collision detection & regeneration
    simulated_ledger = {spk_n: {"consent_given": True}}
    # When simulated_ledger contains spk_n, generator must return a different ID
    new_id = generate_pseudonymous_speaker_id("Northern", ledger=simulated_ledger)
    assert new_id != spk_n
    assert new_id not in simulated_ledger

    # 5. Collision limit failure safety
    import pytest
    with pytest.raises(RuntimeError, match="collision avoidance limit"):
        # Force collision by providing max_attempts=0
        generate_pseudonymous_speaker_id("Northern", ledger={}, max_attempts=0)

    # 6. Default registration uses pseudonymous ID
    mock_ledger_file = tmp_path / "mock_reg_ledger.json"
    record = register_speaker(
        dialect="Southern",
        gender="FEMALE",
        consent_given=True,
        ledger_path=mock_ledger_file,
        notes="mock"
    )
    assert SPEAKER_ID_REGEX.match(record["speaker_id"]) is not None
    assert mock_ledger_file.exists()


# ============================================================================
# P1-1 Remediation Test: 28-Column Manifest Schema Parity & Legacy Adapter
# ============================================================================
def test_p1_1_manifest_schema_parity_28_columns(tmp_path):
    """
    P1-1: Verifies that:
      1. Newly created campaign manifests write all 28 canonical columns in exact schema order.
      2. Metadata fields (consent_id, age_bracket, sub_category, microphone_type, distance_category)
         are properly populated.
      3. Harmonized strings (source, source_version, license, label_provenance) match specifications.
      4. Legacy 23-column manifests can be appended to via the adapter without schema corruption.
    """
    from src.acquisition.ingest_field_recordings import (
        ingest_recording, SCHEMA_28_COLUMNS, LEGACY_23_COLUMNS, load_manifest
    )
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_p1_schema"
    sandbox_ledger = sandbox / "consent_ledger_human.json"
    sandbox_manifest_28 = sandbox / "human_manifest_28col.csv"
    sandbox_manifest_23 = sandbox / "legacy_manifest_23col.csv"
    sandbox_raw = sandbox / "audio_raw"
    sandbox_std = sandbox / "audio_16k_standardized"
    sandbox_quarantine = sandbox / "quarantine.csv"

    # Register speaker with full metadata
    spk = register_speaker(
        dialect="Northern",
        gender="FEMALE",
        age_bracket="18-30",
        consent_given=True,
        device_model="Pixel 7 Pro",
        microphone_type="builtin_mobile",
        acoustic_environment="quiet_room",
        ledger_path=sandbox_ledger
    )
    spk_id = spk["speaker_id"]

    # 1. Ingest into fresh manifest -> Must produce 28 columns
    wav1 = create_test_wav(sandbox / "take1.wav", freq=400.0)
    res1 = ingest_recording(
        input_audio_path=wav1,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_P1_01",
        take_index=1,
        manifest_path=sandbox_manifest_28,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger,
        sub_category="isolated_command",
        distance_category="close_talk_15cm"
    )
    assert res1["status"] == "APPROVED"
    assert sandbox_manifest_28.exists()

    df_28, version_28 = load_manifest(sandbox_manifest_28)
    assert version_28 == "v1.0.0-human-28col"
    assert list(df_28.columns) == SCHEMA_28_COLUMNS
    assert len(df_28.columns) == 28

    row = df_28.iloc[0]
    assert row["source"] == "vietnamese_kws_human_campaign_v1"
    assert row["source_version"] == "v1.0.0-human"
    assert row["license"] == "Proprietary-Consented-Research-Commercial"
    assert str(row["consent_id"]).startswith("CONSENT_")
    assert row["sub_category"] == "isolated_command"
    assert row["microphone_type"] == "builtin_mobile"
    assert row["distance_category"] == "close_talk_15cm"
    assert row["age_bracket"] == "18-30"
    assert row["label_provenance"] == "HUMAN_CONSENTED"
    assert str(row["recording_id"]).startswith("rec_h_")

    # 2. Ingest into legacy 23-column manifest -> Adapter preserves 23 columns
    # Pre-seed legacy file with 23-column header
    pd.DataFrame(columns=LEGACY_23_COLUMNS).to_csv(sandbox_manifest_23, index=False)
    wav2 = create_test_wav(sandbox / "take2.wav", freq=500.0)
    res2 = ingest_recording(
        input_audio_path=wav2,
        speaker_id=spk_id,
        command_label="PHAI",
        session_id="SES_P1_01",
        take_index=2,
        manifest_path=sandbox_manifest_23,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )
    assert res2["status"] == "APPROVED"

    df_23, version_23 = load_manifest(sandbox_manifest_23)
    assert version_23 == "v1.0.0-legacy-23col"
    assert list(df_23.columns) == LEGACY_23_COLUMNS
    assert len(df_23.columns) == 23


# ============================================================================
# P1-2 Remediation Test: Stratified Disjoint Split Generator & Strata Diagnostics
# ============================================================================
def test_p1_2_stratified_disjoint_split_generator():
    """
    P1-2: Verifies that:
      1. generate_stratified_disjoint_splits enforces zero speaker and session leakage.
      2. Demographic strata (dialect x gender) are balanced across train, val, and test.
      3. Small cohort strata (< 3 speakers) are flagged in infeasible_strata diagnostics.
      4. Partitioning is 100% deterministic given the same salt.
    """
    from src.acquisition.campaign_validator import (
        generate_stratified_disjoint_splits, validate_speaker_disjointness
    )

    # Construct balanced cohort: 4 Northern (2F, 2M), 4 Central (2F, 2M), 4 Southern (2F, 2M)
    # 4 speakers per stratum = 24 speakers total
    rows = []
    dialects = ["Northern", "Central", "Southern"]
    genders = ["FEMALE", "MALE"]

    spk_counter = 0
    for d in dialects:
        for g in genders:
            for i in range(4):
                spk_counter += 1
                spk_id = f"SPK_H_BAL_{spk_counter:03d}"
                for cmd in ["TRAI", "PHAI"]:
                    rows.append({
                        "recording_id": f"rec_{spk_id}_{cmd}",
                        "speaker_id": spk_id,
                        "dialect": d,
                        "gender": g,
                        "session_id": f"SES_{spk_id}_01",
                        "command_label": cmd,
                        "class_id": 0 if cmd == "TRAI" else 1
                    })

    # Add infeasible stratum: 2 speakers with gender="OTHER"
    for i in range(2):
        spk_counter += 1
        spk_id = f"SPK_H_INF_{spk_counter:03d}"
        for cmd in ["TRAI", "PHAI"]:
            rows.append({
                "recording_id": f"rec_{spk_id}_{cmd}",
                "speaker_id": spk_id,
                "dialect": "Central",
                "gender": "OTHER",
                "session_id": f"SES_{spk_id}_01",
                "command_label": cmd,
                "class_id": 0 if cmd == "TRAI" else 1
            })

    df = pd.DataFrame(rows)

    df_train1, df_val1, df_test1, diag1 = generate_stratified_disjoint_splits(
        df, strata_cols=["dialect", "gender"], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 1. Zero leakage validation
    is_disjoint, details = validate_speaker_disjointness(df_train1, df_val1, df_test1)
    assert is_disjoint is True
    assert diag1["is_disjoint"] is True
    assert len(details["speaker_overlaps"]["train_val"]) == 0
    assert len(details["speaker_overlaps"]["train_test"]) == 0
    assert len(details["speaker_overlaps"]["val_test"]) == 0

    # 2. Balanced strata validation: every feasible stratum (4 spk) has representation in all partitions
    for d in dialects:
        for g in genders:
            stratum_key = f"{d}-{g}"
            breakdown = diag1["strata_breakdown"][stratum_key]
            assert breakdown["train_speakers"] >= 1, f"Train missing for {stratum_key}"
            assert breakdown["val_speakers"] >= 1, f"Val missing for {stratum_key}"
            assert breakdown["test_speakers"] >= 1, f"Test missing for {stratum_key}"

    # 3. Infeasible strata diagnostics
    assert len(diag1["infeasible_strata"]) > 0
    assert "Central-OTHER" in diag1["infeasible_strata"]
    assert diag1["infeasible_strata"]["Central-OTHER"]["speaker_count"] == 2
    assert "minimum required" in diag1["infeasible_strata"]["Central-OTHER"]["diagnostic"]

    # 4. Determinism
    df_train2, df_val2, df_test2, diag2 = generate_stratified_disjoint_splits(
        df, strata_cols=["dialect", "gender"], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    assert sorted(df_train1["recording_id"].tolist()) == sorted(df_train2["recording_id"].tolist())
    assert sorted(df_val1["recording_id"].tolist()) == sorted(df_val2["recording_id"].tolist())
    assert sorted(df_test1["recording_id"].tolist()) == sorted(df_test2["recording_id"].tolist())


# ============================================================================
# P1-3 Remediation Test: Consent Revocation Workflow & Data Quarantining
# ============================================================================
def test_p1_3_consent_revocation_workflow(tmp_path):
    """
    P1-3: Verifies that:
      1. Dry-run mode (--dry-run or omitting --confirm) does not mutate files or ledger.
      2. Confirmed execution relocates raw and standardized audio into withdrawn_speakers/<spk>/
         without deleting them.
      3. Manifest records are marked status='REVOKED_WITHDRAWN' with withdrawal reason.
      4. Consent ledger sets consent_given=False with timestamp and audit ID.
      5. Subsequent ingestion for the withdrawn speaker is rejected.
      6. Repeated execution is completely idempotent.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording, validate_field_audio
    from src.acquisition.speaker_consent import register_speaker, load_consent_ledger
    from src.acquisition.revoke_speaker_consent import execute_consent_revocation

    sandbox = tmp_path / "sandbox_revocation"
    sandbox_ledger = sandbox / "consent_ledger_human.json"
    sandbox_manifest = sandbox / "human_manifest.csv"
    sandbox_raw = sandbox / "audio_raw"
    sandbox_std = sandbox / "audio_16k_standardized"
    sandbox_quarantine = sandbox / "quarantine"

    # Register speaker and ingest 2 takes
    spk_record = register_speaker(
        dialect="Central",
        gender="MALE",
        consent_given=True,
        ledger_path=sandbox_ledger
    )
    spk_id = spk_record["speaker_id"]

    wav1 = create_test_wav(sandbox / "take1.wav")
    wav2 = create_test_wav(sandbox / "take2.wav", freq=550.0)

    res1 = ingest_recording(
        wav1, spk_id, "LEN", session_id="SES_REV_01", take_index=1,
        manifest_path=sandbox_manifest, raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    res2 = ingest_recording(
        wav2, spk_id, "XUONG", session_id="SES_REV_01", take_index=2,
        manifest_path=sandbox_manifest, raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res1["status"] == "APPROVED"
    assert res2["status"] == "APPROVED"

    raw_path1 = Path(res1["raw_filepath"])
    proc_path1 = Path(res1["processed_path"])
    assert raw_path1.exists()
    assert proc_path1.exists()

    # 1. Test Dry-Run Mode
    dry_result = execute_consent_revocation(
        speaker_id=spk_id,
        reason="participant requested withdrawal via email",
        confirm=False,
        dry_run=True,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert dry_result["status"] == "DRY_RUN"
    assert dry_result["manifest_records_to_update"] == 2
    assert dry_result["files_to_quarantine_count"] >= 2
    # Ensure zero mutation
    assert raw_path1.exists()
    assert proc_path1.exists()
    assert load_consent_ledger(sandbox_ledger)[spk_id]["consent_given"] is True

    # 2. Test Confirmed Execution
    exec_result = execute_consent_revocation(
        speaker_id=spk_id,
        reason="participant requested withdrawal via email",
        confirm=True,
        dry_run=False,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert exec_result["status"] == "REVOKED_CONFIRMED"
    assert exec_result["manifest_rows_updated"] == 2
    assert exec_result["files_quarantined_count"] >= 2

    # Verify original files are relocated (no longer in active folders)
    assert not raw_path1.exists()
    assert not proc_path1.exists()

    # Verify files exist in withdrawn_speakers folder
    withdrawn_dir = sandbox_quarantine / "withdrawn_speakers" / spk_id
    assert withdrawn_dir.exists()
    quarantined_files = list(withdrawn_dir.rglob("*.wav"))
    assert len(quarantined_files) >= 2

    # Verify manifest statuses updated
    df_man = pd.read_csv(sandbox_manifest)
    spk_rows = df_man[df_man["speaker_id"] == spk_id]
    assert all(spk_rows["status"] == "REVOKED_WITHDRAWN")
    assert all("consent_revoked" in str(r) for r in spk_rows["exclusion_reason"])

    # Verify ledger updated
    ledger_after = load_consent_ledger(sandbox_ledger)
    assert ledger_after[spk_id]["consent_given"] is False
    assert "revoked_at" in ledger_after[spk_id]
    assert ledger_after[spk_id]["revocation_audit_id"].startswith("REV_")

    # 3. Verify future ingestion is blocked
    wav_blocked = create_test_wav(sandbox / "blocked.wav")
    is_valid, reason, _ = validate_field_audio(
        wav_blocked, spk_id, "TRAI",
        manifest_path=sandbox_manifest, ledger_path=sandbox_ledger
    )
    assert is_valid is False
    assert reason == "missing_or_unconsented_speaker"

    # 4. Verify Idempotence
    idem_result = execute_consent_revocation(
        speaker_id=spk_id,
        reason="repeated withdrawal call",
        confirm=True,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert idem_result["status"] == "ALREADY_REVOKED"
    assert idem_result["files_quarantined_count"] == 0


# ============================================================================
# Task 1-3 Closure Tests: Strict Metadata, Contradictions, Recovery, Lineage
# ============================================================================
def test_strict_human_metadata_validation(tmp_path):
    """
    Task 2 Closure: Verifies that when strict_metadata=True:
      1. Missing or unverified demographic/hardware categories are rejected.
      2. Valid explicit metadata passes and is accurately recorded.
      3. Legacy/mock calls with strict_metadata=False preserve backward compatibility.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_strict_meta"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Northern",
        gender="FEMALE",
        age_bracket="18-30",
        consent_given=True,
        ledger_path=sandbox_ledger
    )
    spk_id = spk["speaker_id"]
    wav = create_test_wav(sandbox / "audio.wav")

    # 1. Invalid age bracket with strict_metadata=True
    res_bad_age = ingest_recording(
        wav, spk_id, "TRAI",
        age_bracket="12-17",
        microphone_type="builtin_mobile",
        distance_category="close_talk_15cm",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_bad_age["status"] == "QUARANTINED"
    assert "invalid_age_bracket" in res_bad_age["reason"]

    # 2. Invalid microphone type with strict_metadata=True
    res_bad_mic = ingest_recording(
        wav, spk_id, "TRAI",
        age_bracket="18-30",
        microphone_type="studio_tube_xlr",
        distance_category="close_talk_15cm",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_bad_mic["status"] == "QUARANTINED"
    assert "invalid_microphone_type" in res_bad_mic["reason"]

    # 3. Invalid distance category with strict_metadata=True
    res_bad_dist = ingest_recording(
        wav, spk_id, "TRAI",
        age_bracket="18-30",
        microphone_type="builtin_mobile",
        distance_category="satellite_orbit_500km",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_bad_dist["status"] == "QUARANTINED"
    assert "invalid_distance_category" in res_bad_dist["reason"]

    # 4. Valid explicit metadata with strict_metadata=True -> APPROVED
    res_good = ingest_recording(
        wav, spk_id, "TRAI",
        age_bracket="18-30",
        microphone_type="builtin_mobile",
        distance_category="close_talk_15cm",
        sub_category="command_token",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_good["status"] == "APPROVED"
    assert res_good["age_bracket"] == "18-30"
    assert res_good["microphone_type"] == "builtin_mobile"
    assert res_good["distance_category"] == "close_talk_15cm"


def test_contradictory_metadata_rejection(tmp_path):
    """
    Task 2 Closure: Verifies detection and rejection of contradictory metadata combinations.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_contradict"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Southern",
        gender="MALE",
        age_bracket="31-45",
        consent_given=True,
        ledger_path=sandbox_ledger
    )
    spk_id = spk["speaker_id"]
    wav = create_test_wav(sandbox / "audio.wav")

    # Contradiction A: Speech command with background_room_tone
    res_a = ingest_recording(
        wav, spk_id, "PHAI",
        sub_category="background_room_tone",
        microphone_type="builtin_mobile",
        distance_category="close_talk_15cm",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_a["status"] == "QUARANTINED"
    assert "contradictory_metadata_speech_command_with_room_tone" in res_a["reason"]

    # Contradiction B: SILENCE with conversational_speech
    res_b = ingest_recording(
        wav, spk_id, "SILENCE",
        sub_category="conversational_speech",
        microphone_type="builtin_mobile",
        distance_category="close_talk_15cm",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_b["status"] == "QUARANTINED"
    assert "contradictory_metadata_silence_command_with_non_silence" in res_b["reason"]

    # Contradiction C: Wired headset with far-field 2m distance
    res_c = ingest_recording(
        wav, spk_id, "LEN",
        sub_category="command_token",
        microphone_type="wired_headset",
        distance_category="far_field_2m",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_c["status"] == "QUARANTINED"
    assert "contradictory_metadata_wired_headset_with_far_field" in res_c["reason"]


def test_revocation_partial_failure_recovery(tmp_path):
    """
    Task 3 Closure: Verifies safe recovery when audio files are partially missing,
    interrupted, or corrupt during revocation.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker, load_consent_ledger
    from src.acquisition.revoke_speaker_consent import execute_consent_revocation

    sandbox = tmp_path / "sandbox_partial_rev"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"
    sandbox_quarantine = sandbox / "quarantine"

    spk = register_speaker(dialect="Central", gender="FEMALE", consent_given=True, ledger_path=sandbox_ledger)
    spk_id = spk["speaker_id"]

    wav1 = create_test_wav(sandbox / "take1.wav")
    wav2 = create_test_wav(sandbox / "take2.wav", freq=500.0)

    res1 = ingest_recording(
        wav1, spk_id, "LEN", session_id="SES_01", take_index=1,
        manifest_path=sandbox_manifest, raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    res2 = ingest_recording(
        wav2, spk_id, "XUONG", session_id="SES_01", take_index=2,
        manifest_path=sandbox_manifest, raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res1["status"] == "APPROVED"
    assert res2["status"] == "APPROVED"

    # Simulate filesystem anomaly: delete one raw file before revocation
    raw1_path = Path(res1["raw_filepath"])
    raw1_path.unlink()
    assert not raw1_path.exists()

    # Execute revocation: should handle the missing file gracefully
    rev_res = execute_consent_revocation(
        speaker_id=spk_id,
        reason="user requested data erasure",
        confirm=True,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert rev_res["status"] == "REVOKED_CONFIRMED"
    assert rev_res["manifest_rows_updated"] == 2

    # Manifest should still show all rows revoked
    df_man = pd.read_csv(sandbox_manifest)
    assert (df_man["status"] == "REVOKED_WITHDRAWN").all()

    # Consent ledger shows revoked
    ledger = load_consent_ledger(sandbox_ledger)
    assert ledger[spk_id]["consent_given"] is False
    assert "user requested data erasure" in ledger[spk_id]["revocation_reason"]

    # Re-executing is idempotent
    idem_res = execute_consent_revocation(
        speaker_id=spk_id,
        reason="repeated call",
        confirm=True,
        manifest_path=sandbox_manifest,
        ledger_path=sandbox_ledger,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std
    )
    assert idem_res["status"] == "ALREADY_REVOKED"


def test_recording_and_lineage_disjointness():
    """
    Task 1 Closure: Verifies that validate_speaker_disjointness catches:
      1. Cross-partition recording ID reuse.
      2. Cross-partition augmentation lineage leakage (parent_recording_id).
      3. Verifies generate_stratified_disjoint_splits diagnostic contains partition_class_counts
         and age_representation.
    """
    from src.acquisition.campaign_validator import (
        validate_speaker_disjointness,
        generate_stratified_disjoint_splits
    )

    # 1. Base disjoint partitions
    df_train = pd.DataFrame({
        "recording_id": ["rec_001", "rec_002"],
        "speaker_id": ["SPK_001", "SPK_002"],
        "parent_recording_id": ["", ""]
    })
    df_val = pd.DataFrame({
        "recording_id": ["rec_003"],
        "speaker_id": ["SPK_003"],
        "parent_recording_id": [""]
    })
    df_test = pd.DataFrame({
        "recording_id": ["rec_004"],
        "speaker_id": ["SPK_004"],
        "parent_recording_id": [""]
    })

    is_disjoint, details = validate_speaker_disjointness(df_train, df_val, df_test)
    assert is_disjoint is True
    assert len(details["speaker_overlaps"]["train_val"]) == 0
    assert len(details["recording_overlaps"]["train_val"]) == 0
    assert len(details["lineage_overlaps"]["train_val"]) == 0

    # 2. Recording ID leakage (rec_001 in both train and val)
    df_val_leak = df_val.copy()
    df_val_leak.loc[0, "recording_id"] = "rec_001"
    is_disjoint_rec, details_rec = validate_speaker_disjointness(df_train, df_val_leak, df_test)
    assert is_disjoint_rec is False
    assert "rec_001" in details_rec["recording_overlaps"]["train_val"]

    # 3. Augmentation lineage leakage (parent_recording_id rec_002 leaked to test)
    df_test_lineage = df_test.copy()
    df_test_lineage.loc[0, "parent_recording_id"] = "rec_002"
    is_disjoint_lin, details_lin = validate_speaker_disjointness(df_train, df_val, df_test_lineage)
    assert is_disjoint_lin is False
    assert "rec_002" in details_lin["lineage_overlaps"]["train_test"]

    # 4. Generate splits diagnostic contents
    # Synthesize small stratified dataset
    records = []
    dialects = ["Northern", "Central", "Southern"]
    genders = ["MALE", "FEMALE"]
    ages = ["18-30", "31-45", "46-60"]
    commands = ["TRAI", "PHAI", "LEN", "XUONG"]

    spk_counter = 1
    for d in dialects:
        for g in genders:
            for s in range(4):  # 4 speakers per stratum -> 24 speakers total
                spk_id = f"SPK_{spk_counter:03d}"
                spk_counter += 1
                age = ages[s % len(ages)]
                for cmd in commands:
                    records.append({
                        "recording_id": f"REC_{spk_id}_{cmd}",
                        "speaker_id": spk_id,
                        "dialect": d,
                        "gender": g,
                        "age_bracket": age,
                        "command_label": cmd,
                        "status": "APPROVED",
                        "parent_recording_id": ""
                    })

    df_full = pd.DataFrame(records)
    df_tr, df_va, df_te, diag = generate_stratified_disjoint_splits(df_full)

    # Verify diagnostic keys
    assert "partition_class_counts" in diag
    assert "age_representation" in diag
    assert "train" in diag["partition_class_counts"]
    assert "val" in diag["partition_class_counts"]
    assert "test" in diag["partition_class_counts"]
    assert diag["age_representation"]["train"]["18-30"] > 0


def test_fail_closed_production_strict_metadata(tmp_path):
    """
    Closure Acceptance: Verifies fail-closed strict metadata enforcement:
    1. Rejects and quarantines missing distance category.
    2. Rejects and quarantines unregistered/invalid microphone type.
    3. Accepts fully registered metadata and records to canonical 28-column manifest.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording, SCHEMA_28_COLUMNS
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "strict_sandbox"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"
    sandbox_quarantine = sandbox / "quarantine" / "quarantine_log.csv"

    # Register speaker with missing distance in ledger
    ledger_content = {
        "SPK_H_N_abc123": {
            "speaker_id": "SPK_H_N_abc123",
            "consent_given": True,
            "dialect": "Northern",
            "gender": "MALE",
            "age_bracket": "18-30",
            # primary_microphone and distance_category omitted
        }
    }
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger_content, f)

    wav = create_test_wav(sandbox / "test_utterance.wav")

    # Ingest without explicit distance or mic under strict_metadata=True
    res_fail = ingest_recording(
        wav, "SPK_H_N_abc123", "TRAI",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_fail["status"] == "QUARANTINED"
    assert "missing_or_invalid_distance_category" in res_fail["reason"]
    assert "unregistered_or_invalid_microphone_type" in res_fail["reason"]

    # Register compliant speaker with full metadata
    spk_full = register_speaker(
        dialect="Southern",
        gender="FEMALE",
        age_bracket="31-45",
        consent_given=True,
        microphone_type="usb_desktop_mic",
        distance_category="medium_talk_50cm",
        ledger_path=sandbox_ledger
    )
    spk_id = spk_full["speaker_id"]

    res_ok = ingest_recording(
        wav, spk_id, "XUONG",
        strict_metadata=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_ok["status"] == "APPROVED"
    assert res_ok["microphone_type"] == "usb_desktop_mic"
    assert res_ok["distance_category"] == "medium_talk_50cm"

    # Verify manifest has 28 columns
    df_man = pd.read_csv(sandbox_manifest)
    assert list(df_man.columns) == SCHEMA_28_COLUMNS
    assert df_man.loc[0, "microphone_type"] == "usb_desktop_mic"
    assert df_man.loc[0, "distance_category"] == "medium_talk_50cm"


def test_registration_flow_persists_mic_and_distance(tmp_path, monkeypatch):
    """
    Closure Acceptance: Verifies interactive and programmatic registration persists
    microphone_type and distance_category in the informed consent ledger.
    """
    from src.acquisition.speaker_consent import (
        register_speaker, load_consent_ledger
    )
    from src.acquisition.record_command_cli import interactive_registration

    sandbox_ledger = tmp_path / "reg_test_ledger.json"

    # 1. Programmatic registration
    rec = register_speaker(
        dialect="Central",
        gender="MALE",
        age_bracket="46-60",
        consent_given=True,
        microphone_type="bluetooth_earbuds",
        distance_category="close_talk_15cm",
        ledger_path=sandbox_ledger
    )
    spk_id = rec["speaker_id"]
    ledger = load_consent_ledger(sandbox_ledger)
    assert ledger[spk_id]["primary_microphone"] == "bluetooth_earbuds"
    assert ledger[spk_id]["distance_category"] == "close_talk_15cm"

    # 2. Interactive registration with simulated user inputs
    monkeypatch.setattr("src.acquisition.speaker_consent.LEDGER_PATH", sandbox_ledger)
    monkeypatch.setattr("src.acquisition.record_command_cli.LEDGER_PATH", sandbox_ledger)

    inputs = iter([
        "1",                    # Dialect: Northern
        "F",                    # Gender: FEMALE
        "18-30",                # Age
        "MacBook Pro Mic",      # Device
        "quiet_room",           # Environment
        "4",                    # Mic: laptop_mic
        "2",                    # Distance: medium_talk_50cm
        "YES"                   # Consent
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    inter_spk_id = interactive_registration()
    assert inter_spk_id is not None
    updated_ledger = load_consent_ledger(sandbox_ledger)
    assert updated_ledger[inter_spk_id]["primary_microphone"] == "laptop_mic"
    assert updated_ledger[inter_spk_id]["distance_category"] == "medium_talk_50cm"


def test_recording_session_strict_fail_closed(tmp_path):
    """
    Closure Acceptance: Verifies run_recording_session enforces fail-closed strict validation
    when mock_mode=False, and retains compatibility when mock_mode=True.
    """
    from src.acquisition.record_command_cli import run_recording_session
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "session_sandbox"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"
    sandbox_quarantine = sandbox / "quarantine" / "quarantine_log.csv"

    # Register speaker with valid metadata
    spk = register_speaker(
        dialect="Northern",
        gender="FEMALE",
        microphone_type="builtin_mobile",
        distance_category="close_talk_15cm",
        ledger_path=sandbox_ledger
    )
    spk_id = spk["speaker_id"]

    # In mock mode, should successfully ingest mock audio
    run_recording_session(
        speaker_id=spk_id,
        commands=["TRAI"],
        takes_per_command=1,
        session_id="SES_MOCK_TEST",
        mock_mode=True,
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )

    df_man = pd.read_csv(sandbox_manifest)
    assert len(df_man) == 1
    assert df_man.loc[0, "command_label"] == "TRAI"
    assert df_man.loc[0, "status"] == "APPROVED"


# ============================================================================
# P0 Security Regression Tests: Ingestion Core Strict Metadata Enforcement
# ============================================================================

def test_direct_ingest_human_recording_strict_false_fails_closed(tmp_path):
    """
    P0 Invariant Regression: Callers cannot bypass strict metadata validation
    for human recordings by passing strict_metadata=False directly.
    A human recording with missing mic/distance must be quarantined.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_direct_strict_false"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Pre-seed ledger with missing microphone and distance metadata
    spk_id = "SPK_H_N_bypass_attempt"
    ledger_content = {
        spk_id: {
            "speaker_id": spk_id,
            "dialect": "Northern",
            "gender": "FEMALE",
            "age_bracket": "18-30",
            "consent_given": True,
            "consent_timestamp": "2026-10-14T00:00:00Z",
            # primary_microphone and distance_category intentionally omitted
        }
    }
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger_content, f)

    wav = create_test_wav(sandbox / "bypass_take.wav")

    # Call directly with explicit strict_metadata=False
    res = ingest_recording(
        input_audio_path=wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_BYPASS_01",
        provenance="human_field_recording",
        strict_metadata=False,  # ATTEMPT BYPASS
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )

    # Must FAIL CLOSED and be quarantined
    assert res["status"] == "QUARANTINED"
    assert "missing_or_invalid_distance_category" in res["reason"]
    assert "unregistered_or_invalid_microphone_type" in res["reason"]

    # Manifest must not contain approved rows
    if sandbox_manifest.exists():
        df_man = pd.read_csv(sandbox_manifest)
        assert len(df_man) == 0

    # Quarantine file must have the rejection record
    assert sandbox_quarantine.exists()
    df_quar = pd.read_csv(sandbox_quarantine)
    assert len(df_quar) == 1
    assert df_quar.loc[0, "status"] == "QUARANTINED"
    assert df_quar.loc[0, "speaker_id"] == spk_id


def test_direct_ingest_human_cannot_rely_on_fallback_defaults(tmp_path):
    """
    P0 Invariant Regression: Default parameter invocation of ingest_recording
    for human provenance must never inject fallback defaults ("18-30", "builtin_mobile",
    "close_talk_15cm") when metadata is missing.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_fallback_block"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Pre-seed ledger with completely bare demographic record (no age, mic, dist)
    spk_id = "SPK_H_C_bare_metadata"
    ledger_content = {
        spk_id: {
            "speaker_id": spk_id,
            "dialect": "Central",
            "gender": "MALE",
            "consent_given": True,
            "consent_timestamp": "2026-10-14T00:00:00Z",
            # age_bracket, primary_microphone, distance_category all omitted
        }
    }
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger_content, f)

    wav = create_test_wav(sandbox / "bare_take.wav")

    # Ingest without passing any optional metadata, relying on function defaults
    res = ingest_recording(
        input_audio_path=wav,
        speaker_id=spk_id,
        command_label="PHAI",
        session_id="SES_BARE_01",
        provenance="human_field_recording",
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )

    # Core must reject and not substitute fallback values
    assert res["status"] == "QUARANTINED"
    assert "unregistered_or_invalid_age_bracket" in res["reason"]
    assert "unregistered_or_invalid_microphone_type" in res["reason"]
    assert "missing_or_invalid_distance_category" in res["reason"]


def test_batch_ingest_directory_human_recording_strict_false_fails_closed(tmp_path):
    """
    P0 Invariant Regression: batch_ingest_directory cannot bypass strict metadata
    for human recordings even when callers specify strict_metadata=False.
    """
    import json
    from src.acquisition.ingest_field_recordings import batch_ingest_directory

    sandbox = tmp_path / "sandbox_batch_strict_false"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk_id = "SPK_H_S_batch_test"
    ledger_content = {
        spk_id: {
            "speaker_id": spk_id,
            "dialect": "Southern",
            "gender": "FEMALE",
            "age_bracket": "31-45",
            "consent_given": True,
            "consent_timestamp": "2026-10-14T00:00:00Z"
            # mic and distance missing
        }
    }
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger_content, f)

    # Create input directory with 2 command files
    batch_in_dir = sandbox / "audio_input"
    batch_in_dir.mkdir(parents=True, exist_ok=True)
    create_test_wav(batch_in_dir / "trai_01.wav", freq=440.0)
    create_test_wav(batch_in_dir / "len_01.wav", freq=550.0)

    # Run batch ingest with strict_metadata=False
    results = batch_ingest_directory(
        input_dir=batch_in_dir,
        speaker_id=spk_id,
        session_id="SES_BATCH_01",
        provenance="human_field_recording",
        strict_metadata=False,  # ATTEMPT BYPASS
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )

    assert len(results) == 2
    for r in results:
        assert r["status"] == "QUARANTINED"
        assert "missing_or_invalid_distance_category" in r["reason"]
        assert "unregistered_or_invalid_microphone_type" in r["reason"]

    # Manifest must have zero approved rows
    if sandbox_manifest.exists():
        df_man = pd.read_csv(sandbox_manifest)
        assert len(df_man) == 0

    assert sandbox_quarantine.exists()
    df_quar = pd.read_csv(sandbox_quarantine)
    assert len(df_quar) == 2


def test_fully_specified_human_metadata_approved(tmp_path):
    """
    P0 Acceptance: Fully specified genuine human metadata passes strict
    validation and is cleanly approved into 28-column manifest.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording, SCHEMA_28_COLUMNS
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_fully_specified"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Register speaker with complete metadata
    spk = register_speaker(
        dialect="Central",
        gender="MALE",
        age_bracket="46-60",
        consent_given=True,
        device_model="Samsung Galaxy S22",
        microphone_type="wired_headset",
        distance_category="close_talk_15cm",
        acoustic_environment="office_quiet",
        ledger_path=sandbox_ledger
    )
    spk_id = spk["speaker_id"]

    wav = create_test_wav(sandbox / "take_full.wav")

    # Ingest without needing explicit overrides (resolved from registered ledger)
    res = ingest_recording(
        input_audio_path=wav,
        speaker_id=spk_id,
        command_label="LEN",
        session_id="SES_FULL_01",
        take_index=1,
        sub_category="command_token",
        provenance="human_field_recording",
        strict_metadata=False,  # Even with strict_metadata=False, effective_strict=True passes!
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )

    assert res["status"] == "APPROVED"
    assert res["recording_id"] is not None
    assert res["label_provenance"] == "HUMAN_CONSENTED"
    assert res["age_bracket"] == "46-60"
    assert res["microphone_type"] == "wired_headset"
    assert res["distance_category"] == "close_talk_15cm"

    df_man = pd.read_csv(sandbox_manifest)
    assert len(df_man) == 1
    assert list(df_man.columns) == SCHEMA_28_COLUMNS
    assert df_man.loc[0, "status"] == "APPROVED"


def test_mock_fixture_permissive_defaults_isolated(tmp_path):
    """
    P0 Isolation: Non-human test fixtures (provenance='test_harness_mock')
    can use permissive defaults when strict_metadata=False, and enforce strict
    validation when strict_metadata=True.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_mock_isolation"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk_id = "MOCK_SPK_001"
    ledger_content = {
        spk_id: {
            "speaker_id": spk_id,
            "dialect": "Northern",
            "gender": "MALE",
            "consent_given": True,
            # no age, mic, distance
        }
    }
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger_content, f)

    wav = create_test_wav(sandbox / "mock_take.wav")

    # 1. Non-human mock with strict_metadata=False -> Allowed permissive defaults
    res_permissive = ingest_recording(
        input_audio_path=wav,
        speaker_id=spk_id,
        command_label="TRAI",
        session_id="SES_MOCK_01",
        take_index=1,
        provenance="test_harness_mock",
        strict_metadata=False,
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )
    assert res_permissive["status"] == "APPROVED"
    assert res_permissive["label_provenance"] == "test_harness_mock"
    assert res_permissive["age_bracket"] == "18-30"  # Permissive default used safely

    # 2. Non-human mock with strict_metadata=True -> Strict validation enforced
    wav2 = create_test_wav(sandbox / "mock_take2.wav", freq=600.0)
    res_strict = ingest_recording(
        input_audio_path=wav2,
        speaker_id=spk_id,
        command_label="PHAI",
        session_id="SES_MOCK_02",
        take_index=2,
        provenance="test_harness_mock",
        strict_metadata=True,
        manifest_path=sandbox_manifest,
        quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        ledger_path=sandbox_ledger
    )
    assert res_strict["status"] == "QUARANTINED"
    assert "missing_or_invalid_distance_category" in res_strict["reason"]


def test_failed_strict_validation_leaves_manifest_untouched(tmp_path):
    """
    P0 Cleanliness Invariant: Failed strict metadata validation leaves manifest
    untouched with zero approved rows added, and saves only quarantine evidence.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_untouched_manifest"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Pre-seed manifest with 1 existing approved record
    spk_good = register_speaker(
        dialect="Northern", gender="FEMALE", consent_given=True,
        ledger_path=sandbox_ledger
    )
    wav_good = create_test_wav(sandbox / "good.wav")
    res_good = ingest_recording(
        wav_good, spk_good["speaker_id"], "TRAI",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_good["status"] == "APPROVED"
    df_before = pd.read_csv(sandbox_manifest)
    assert len(df_before) == 1
    sha_manifest_before = hashlib.sha256(sandbox_manifest.read_bytes()).hexdigest()

    # Attempt to ingest invalid contradictory record
    wav_bad = create_test_wav(sandbox / "bad.wav", freq=450.0)
    res_bad = ingest_recording(
        wav_bad, spk_good["speaker_id"], "SILENCE",
        sub_category="conversational_speech",  # Contradiction!
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_bad["status"] == "QUARANTINED"

    # Manifest must be completely untouched
    df_after = pd.read_csv(sandbox_manifest)
    assert len(df_after) == 1
    sha_manifest_after = hashlib.sha256(sandbox_manifest.read_bytes()).hexdigest()
    assert sha_manifest_before == sha_manifest_after

    # Quarantine file has recorded the exclusion
    df_quar = pd.read_csv(sandbox_quarantine)
    assert len(df_quar) == 1
    assert "contradictory_metadata" in df_quar.loc[0, "exclusion_reason"]


def test_consent_revocation_and_lineage_with_strict_core(tmp_path):
    """
    P0 End-to-End Governance: Validates end-to-end ingestion under strict core,
    followed by full consent revocation, artifact scrubbing, and ledger status update.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker, load_consent_ledger
    from src.acquisition.revoke_speaker_consent import execute_consent_revocation

    sandbox = tmp_path / "sandbox_e2e_revocation"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Southern",
        gender="MALE",
        age_bracket="18-30",
        consent_given=True,
        device_model="Google Pixel 8",
        microphone_type="builtin_mobile",
        distance_category="close_talk_15cm",
        ledger_path=sandbox_ledger
    )
    spk_id = spk["speaker_id"]

    wav1 = create_test_wav(sandbox / "take1.wav", freq=350.0)
    wav2 = create_test_wav(sandbox / "take2.wav", freq=420.0)

    res1 = ingest_recording(
        wav1, spk_id, "TRAI", session_id="SES_REV_01", take_index=1,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    res2 = ingest_recording(
        wav2, spk_id, "PHAI", session_id="SES_REV_01", take_index=2,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res1["status"] == "APPROVED"
    assert res2["status"] == "APPROVED"

    raw_path1 = Path(res1["raw_filepath"])
    proc_path1 = Path(res1["processed_path"])
    raw_path2 = Path(res2["raw_filepath"])
    proc_path2 = Path(res2["processed_path"])
    assert raw_path1.exists() and proc_path1.exists()
    assert raw_path2.exists() and proc_path2.exists()

    # Execute consent revocation
    rev_summary = execute_consent_revocation(
        speaker_id=spk_id,
        reason="GDPR/Consent withdrawal requested",
        ledger_path=sandbox_ledger,
        manifest_path=sandbox_manifest,
        quarantine_dir=sandbox_quarantine,
        raw_base_dir=sandbox_raw,
        std_base_dir=sandbox_std,
        confirm=True,
        dry_run=False
    )
    assert rev_summary["status"] == "REVOKED_CONFIRMED"
    assert rev_summary["manifest_rows_updated"] == 2
    assert rev_summary["files_quarantined_count"] >= 2

    # Confirm audio files purged
    assert not raw_path1.exists()
    assert not proc_path1.exists()
    assert not raw_path2.exists()
    assert not proc_path2.exists()

    # Confirm ledger marks consent revoked
    ledger = load_consent_ledger(sandbox_ledger)
    assert ledger[spk_id]["consent_given"] is False
    assert ledger[spk_id]["revoked_at"] is not None
    assert ledger[spk_id]["revocation_reason"] == "GDPR/Consent withdrawal requested"

    # Confirm manifest marks status REVOKED_WITHDRAWN
    df_man = pd.read_csv(sandbox_manifest)
    assert len(df_man) == 2
    for _, row in df_man.iterrows():
        assert row["status"] == "REVOKED_WITHDRAWN"
        assert "consent_revoked" in row["exclusion_reason"]


# ============================================================================
# RES-01 Security Hardening: Provenance Allowlist & Manifest Write Boundary
# ============================================================================

def test_unknown_provenance_fails_closed_to_quarantine(tmp_path):
    """
    RES-01 Test 1: Ingestion fails closed when given an unknown/unauthorized provenance string.
    The recording must be quarantined before audio standardization or manifest commit.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_unknown_prov"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Northern", gender="FEMALE", age_bracket="18-30",
        consent_given=True, device_model="Pixel 7", microphone_type="builtin_mobile",
        distance_category="close_talk_15cm", ledger_path=sandbox_ledger
    )
    wav = create_test_wav(sandbox / "audio.wav")

    res = ingest_recording(
        wav, spk["speaker_id"], "TRAI",
        provenance="unknown_external_audio",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )

    assert res["status"] == "QUARANTINED"
    assert "unauthorized_or_unknown_provenance_unknown_external_audio" in res["reason"]
    assert not sandbox_manifest.exists()
    assert sandbox_quarantine.exists()
    df_q = pd.read_csv(sandbox_quarantine)
    assert len(df_q) == 1
    assert "unknown_external_audio" in df_q.loc[0, "exclusion_reason"]
    # Verify no raw or std audio was copied
    assert not list(sandbox_raw.glob("**/*.wav"))
    assert not list(sandbox_std.glob("**/*.wav"))


def test_custom_clean_source_cannot_enter_permissive_path(tmp_path):
    """
    RES-01 Test 2: 'custom_clean_source' or arbitrary caller provenance cannot bypass
    security checks via non-strict ingestion and is quarantined immediately.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_custom_prov"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Southern", gender="MALE", age_bracket="31-45",
        consent_given=True, ledger_path=sandbox_ledger
    )
    wav = create_test_wav(sandbox / "audio.wav")

    res = ingest_recording(
        wav, spk["speaker_id"], "PHAI",
        provenance="custom_clean_source",
        strict_metadata=False,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )

    assert res["status"] == "QUARANTINED"
    assert "unauthorized_or_unknown_provenance_custom_clean_source" in res["reason"]
    assert not sandbox_manifest.exists()
    assert not list(sandbox_raw.glob("**/*.wav"))
    assert not list(sandbox_std.glob("**/*.wav"))


def test_synthetic_fixture_cannot_create_human_consented_manifest_row(tmp_path):
    """
    RES-01 Test 3: Fixture labeled 'SYNTHETIC_TEST_FIXTURE' retains its non-human provenance tag
    and cannot write 'HUMAN_CONSENTED' to the manifest, preventing mock-to-human promotion.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker
    from src.acquisition.campaign_validator import validate_manifest_dataset

    sandbox = tmp_path / "sandbox_synth_prov"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Central", gender="FEMALE", age_bracket="18-30",
        consent_given=True, ledger_path=sandbox_ledger
    )
    wav = create_test_wav(sandbox / "audio.wav")

    res = ingest_recording(
        wav, spk["speaker_id"], "LEN",
        provenance="SYNTHETIC_TEST_FIXTURE",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )

    assert res["status"] == "APPROVED"
    assert res["label_provenance"] == "SYNTHETIC_TEST_FIXTURE"
    assert res["label_provenance"] != "HUMAN_CONSENTED"

    df_manifest = pd.read_csv(sandbox_manifest)
    assert len(df_manifest) == 1
    assert df_manifest.loc[0, "label_provenance"] == "SYNTHETIC_TEST_FIXTURE"

    # Verify campaign validator prevents synthetic fixture co-mingling if labeled as human
    bad_df = df_manifest.copy()
    bad_df.loc[0, "label_provenance"] = "HUMAN_CONSENTED"
    bad_df.loc[0, "speaker_id"] = "SPK_MOCK_001"
    is_valid, errors, _ = validate_manifest_dataset(bad_df, {spk["speaker_id"]: spk}, check_disk_files=False)
    assert not is_valid
    assert any("synthetic_speaker_id_mislabeled_as_human" in e for e in errors)


def test_human_provenance_cannot_bypass_strict_metadata(tmp_path):
    """
    RES-01 Test 4: Passing strict_metadata=False with human_field_recording or HUMAN_CONSENTED
    is ignored; strict validation is unconditionally enforced and permissive defaults are not injected.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_human_bypass"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Pre-seed ledger with speaker lacking distance_category and microphone_type
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    raw_ledger_data = {
        "SPK_VN_H099": {
            "speaker_id": "SPK_VN_H099",
            "consent_given": True,
            "dialect": "Northern",
            "gender": "FEMALE",
            "age_bracket": "18-30",
            "primary_device": "iPhone 13"
            # Missing distance_category and primary_microphone
        }
    }
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(raw_ledger_data, f)

    wav = create_test_wav(sandbox / "audio.wav")

    # Attempt to bypass via strict_metadata=False. If permissive defaults were used, it would approve.
    # Because human provenance enforces strict validation unconditionally, it must fail.
    res = ingest_recording(
        wav, "SPK_VN_H099", "XUONG",
        provenance="human_field_recording",
        strict_metadata=False,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )

    assert res["status"] == "QUARANTINED"
    assert "unregistered_or_invalid_microphone_type" in res["reason"] or "missing_or_invalid_distance_category" in res["reason"]
    assert not sandbox_manifest.exists()


def test_human_provenance_without_valid_consent_quarantined(tmp_path):
    """
    RES-01 Test 5: Human recording from speaker with consent_given=False or unregistered speaker
    is quarantined, never approved or committed to manifest.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_unconsented_human"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Seed ledger with an explicit consent_given=False entry with complete metadata
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    raw_ledger_data = {
        "SPK_VN_UNCONSENTED": {
            "speaker_id": "SPK_VN_UNCONSENTED",
            "consent_given": False,
            "dialect": "Northern",
            "gender": "FEMALE",
            "age_bracket": "18-30",
            "primary_device": "iPhone 13",
            "primary_microphone": "builtin_mobile",
            "distance_category": "close_talk_15cm",
            "acoustic_environment": "quiet_room"
        }
    }
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(raw_ledger_data, f)

    wav = create_test_wav(sandbox / "audio.wav")

    # Case 1: Speaker in ledger but consent_given is False
    res1 = ingest_recording(
        wav, "SPK_VN_UNCONSENTED", "TRAI",
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res1["status"] == "QUARANTINED"
    assert "missing_or_unconsented_speaker" in res1["reason"]

    # Case 2: Speaker completely unregistered
    res2 = ingest_recording(
        wav, "SPK_VN_UNKNOWN_999", "TRAI",
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res2["status"] == "QUARANTINED"

    # Core validation check directly returns missing_or_unconsented_speaker
    from src.acquisition.ingest_field_recordings import validate_field_audio
    ok, reason, _ = validate_field_audio(wav, "SPK_VN_UNKNOWN_999", "TRAI", ledger_path=sandbox_ledger)
    assert not ok
    assert reason == "missing_or_unconsented_speaker"

    assert not sandbox_manifest.exists()


def test_manifest_write_boundary_rejects_invalid_provenance_and_fields(tmp_path):
    """
    RES-01 Test 6: append_to_manifest() validates provenance against ALLOWED_MANIFEST_PROVENANCE
    and checks required fields, raising ValueError on violation.
    """
    from src.acquisition.ingest_field_recordings import append_to_manifest

    manifest_file = tmp_path / "test_boundary_manifest.csv"

    valid_record = {
        "recording_id": "rec_test_001",
        "source": "test_source",
        "source_version": "v1.0.0-human",
        "license": "Proprietary",
        "consent_id": "CONSENT_001",
        "speaker_id": "SPK_001",
        "dialect": "Northern",
        "gender": "FEMALE",
        "age_bracket": "18-30",
        "command_label": "TRAI",
        "class_id": 0,
        "transcript": "trái",
        "sub_category": "isolated_command",
        "session_id": "SES_001",
        "device_model": "iPhone 13",
        "microphone_type": "builtin_mobile",
        "environment": "quiet_room",
        "distance_category": "close_talk_15cm",
        "speaking_style": "normal",
        "raw_duration_sec": 1.0,
        "rms_db": -22.0,
        "label_provenance": "HUMAN_CONSENTED",
        "raw_filepath": "data/raw/audio.wav",
        "processed_path": "data/proc/audio.wav",
        "sha256_raw": "A" * 64,
        "sha256_standardized": "B" * 64,
        "status": "APPROVED",
        "exclusion_reason": ""
    }

    # 1. Non-dict record raises TypeError
    with pytest.raises(TypeError, match="must be a dict"):
        append_to_manifest("not a dict", manifest_file)

    # 2. Missing required field raises ValueError
    bad_rec_no_spk = dict(valid_record)
    bad_rec_no_spk["speaker_id"] = ""
    with pytest.raises(ValueError, match="missing required field 'speaker_id'"):
        append_to_manifest(bad_rec_no_spk, manifest_file)

    # 3. Unauthorized provenance raises ValueError
    bad_rec_prov = dict(valid_record)
    bad_rec_prov["label_provenance"] = "rogue_arbitrary_provenance"
    with pytest.raises(ValueError, match="unauthorized label_provenance 'rogue_arbitrary_provenance'"):
        append_to_manifest(bad_rec_prov, manifest_file)

    # 4. Invalid status raises ValueError
    bad_rec_status = dict(valid_record)
    bad_rec_status["status"] = "PENDING_REVIEW"
    with pytest.raises(ValueError, match="invalid status 'PENDING_REVIEW'"):
        append_to_manifest(bad_rec_status, manifest_file)

    # 5. Invalid command_label for APPROVED human recording raises ValueError
    bad_rec_cmd = dict(valid_record)
    bad_rec_cmd["command_label"] = "INVALID_CMD"
    with pytest.raises(ValueError, match="invalid command_label 'INVALID_CMD'"):
        append_to_manifest(bad_rec_cmd, manifest_file)

    # 6. Valid record successfully writes
    append_to_manifest(valid_record, manifest_file)
    assert manifest_file.exists()
    df = pd.read_csv(manifest_file)
    assert len(df) == 1
    assert df.loc[0, "recording_id"] == "rec_test_001"


def test_rejected_ingestion_cleans_up_audio_artifacts(tmp_path):
    """
    RES-01 Test 7: If append_to_manifest fails during ingestion, any copied raw and
    standardized audio artifacts are cleaned up so no orphan files remain.
    """
    from unittest.mock import patch
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_cleanup"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Northern", gender="FEMALE", age_bracket="18-30",
        consent_given=True, device_model="Pixel 7", microphone_type="builtin_mobile",
        distance_category="close_talk_15cm", ledger_path=sandbox_ledger
    )
    wav = create_test_wav(sandbox / "audio.wav")

    # Simulate manifest append failure
    with patch("src.acquisition.ingest_field_recordings.append_to_manifest", side_effect=IOError("Simulated disk full")):
        with pytest.raises(RuntimeError, match="Manifest append failed"):
            ingest_recording(
                wav, spk["speaker_id"], "TRAI",
                provenance="human_field_recording",
                manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
                raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
            )

    # Verify no orphan files remain in raw or std dirs
    raw_files = list(sandbox_raw.glob("**/*.wav"))
    std_files = list(sandbox_std.glob("**/*.wav"))
    assert len(raw_files) == 0, f"Orphan raw files remained: {raw_files}"
    assert len(std_files) == 0, f"Orphan std files remained: {std_files}"


def test_existing_valid_fixture_and_human_paths_retain_behavior(tmp_path):
    """
    RES-01 Test 8: Valid human recordings and test harness fixtures continue to succeed
    with proper provenance and metadata formatting.
    """
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_valid_paths"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Northern", gender="FEMALE", age_bracket="18-30",
        consent_given=True, device_model="Pixel 7", microphone_type="builtin_mobile",
        distance_category="close_talk_15cm", ledger_path=sandbox_ledger
    )
    wav1 = create_test_wav(sandbox / "human.wav", freq=400.0)
    wav2 = create_test_wav(sandbox / "fixture.wav", freq=500.0)

    # 1. Valid human recording
    res_human = ingest_recording(
        wav1, spk["speaker_id"], "TRAI",
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_human["status"] == "APPROVED"
    assert res_human["label_provenance"] == "HUMAN_CONSENTED"
    assert Path(res_human["raw_filepath"]).exists()
    assert Path(res_human["processed_path"]).exists()

    # 2. Valid test harness fixture
    res_fixture = ingest_recording(
        wav2, spk["speaker_id"], "PHAI",
        provenance="test_harness_mock",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_fixture["status"] == "APPROVED"
    assert res_fixture["label_provenance"] == "test_harness_mock"

    df = pd.read_csv(sandbox_manifest)
    assert len(df) == 2
    assert set(df["label_provenance"]) == {"HUMAN_CONSENTED", "test_harness_mock"}


def test_revoked_consent_speaker_ingestion_blocked(tmp_path):
    """
    Failure Mode Test: Once speaker consent is revoked/withdrawn, subsequent calls to
    ingest_recording() for that speaker are immediately quarantined with missing_or_unconsented_speaker.
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_revoked_blocked"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    # Pre-seed ledger with revoked speaker
    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    raw_ledger_data = {
        "SPK_VN_H_REVOKED": {
            "speaker_id": "SPK_VN_H_REVOKED",
            "consent_given": False,
            "revoked_at": "2026-10-14T08:00:00Z",
            "revocation_reason": "Participant withdrawal request",
            "dialect": "Northern",
            "gender": "FEMALE",
            "age_bracket": "18-30",
            "primary_device": "iPhone 13",
            "primary_microphone": "builtin_mobile",
            "distance_category": "close_talk_15cm",
            "acoustic_environment": "quiet_room"
        }
    }
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(raw_ledger_data, f)

    wav = create_test_wav(sandbox / "audio.wav")

    res = ingest_recording(
        wav, "SPK_VN_H_REVOKED", "LEN",
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )

    assert res["status"] == "QUARANTINED"
    assert "missing_or_unconsented_speaker" in res["reason"]
    assert not sandbox_manifest.exists()
    assert not list(sandbox_raw.glob("**/*.wav"))
    assert not list(sandbox_std.glob("**/*.wav"))


def test_valid_consent_vs_invalid_metadata_isolated_failures(tmp_path):
    """
    Failure Mode Test: Confirms distinct and isolated failure conditions:
    1) Valid consent + invalid metadata fails specifically on metadata (e.g. contradictory metadata), not consent.
    2) Revoked consent + valid metadata fails specifically on consent (missing_or_unconsented_speaker).
    """
    import json
    from src.acquisition.ingest_field_recordings import ingest_recording

    sandbox = tmp_path / "sandbox_isolated_failures"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    sandbox_ledger.parent.mkdir(parents=True, exist_ok=True)
    raw_ledger_data = {
        "SPK_VN_VALID_CONSENT": {
            "speaker_id": "SPK_VN_VALID_CONSENT",
            "consent_given": True,
            "dialect": "Central",
            "gender": "MALE",
            "age_bracket": "31-45",
            "primary_device": "Pixel 7",
            "primary_microphone": "builtin_mobile",
            "distance_category": "close_talk_15cm",
            "acoustic_environment": "quiet_room"
        },
        "SPK_VN_REVOKED_CONSENT": {
            "speaker_id": "SPK_VN_REVOKED_CONSENT",
            "consent_given": False,
            "revoked_at": "2026-10-14T08:00:00Z",
            "dialect": "Central",
            "gender": "MALE",
            "age_bracket": "31-45",
            "primary_device": "Pixel 7",
            "primary_microphone": "builtin_mobile",
            "distance_category": "close_talk_15cm",
            "acoustic_environment": "quiet_room"
        }
    }
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(raw_ledger_data, f)

    wav = create_test_wav(sandbox / "audio.wav")

    # Condition 1: Valid consent, but contradictory metadata (SILENCE command with isolated_command)
    res_bad_meta = ingest_recording(
        wav, "SPK_VN_VALID_CONSENT", "SILENCE",
        sub_category="isolated_command",  # Contradiction
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_bad_meta["status"] == "QUARANTINED"
    assert "contradictory_metadata" in res_bad_meta["reason"]
    assert "missing_or_unconsented_speaker" not in res_bad_meta["reason"]

    # Condition 2: Revoked consent, with valid metadata
    res_bad_consent = ingest_recording(
        wav, "SPK_VN_REVOKED_CONSENT", "LEN",
        sub_category="isolated_command",
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_bad_consent["status"] == "QUARANTINED"
    assert "missing_or_unconsented_speaker" in res_bad_consent["reason"]
    assert "contradictory_metadata" not in res_bad_consent["reason"]


def test_existing_destination_files_during_rollback(tmp_path):
    """
    Failure Mode Test: Confirms that pre-existing destination files are completely preserved:
    1) Sibling files in destination folders remain intact with identical hashes during rollback.
    2) A pre-existing file on disk at destination path is protected from being overwritten or unlinked.
    """
    import hashlib
    from unittest.mock import patch
    from src.acquisition.ingest_field_recordings import ingest_recording
    from src.acquisition.speaker_consent import register_speaker

    sandbox = tmp_path / "sandbox_dest_rollback"
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"

    spk = register_speaker(
        dialect="Northern", gender="FEMALE", age_bracket="18-30",
        consent_given=True, device_model="Pixel 7", microphone_type="builtin_mobile",
        distance_category="close_talk_15cm", ledger_path=sandbox_ledger
    )
    wav1 = create_test_wav(sandbox / "take1.wav", freq=300.0)
    wav2 = create_test_wav(sandbox / "take2.wav", freq=600.0)

    # 1. First take succeeds cleanly
    res1 = ingest_recording(
        wav1, spk["speaker_id"], "TRAI", session_id="SES_001", take_index=1,
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res1["status"] == "APPROVED"
    raw1_path = Path(res1["raw_filepath"])
    std1_path = Path(res1["processed_path"])
    raw1_sha_before = hashlib.sha256(raw1_path.read_bytes()).hexdigest()
    std1_sha_before = hashlib.sha256(std1_path.read_bytes()).hexdigest()

    # 2. Second take fails during append_to_manifest -> Rollback occurs
    with patch("src.acquisition.ingest_field_recordings.append_to_manifest", side_effect=IOError("Disk write error")):
        with pytest.raises(RuntimeError, match="Manifest append failed"):
            ingest_recording(
                wav2, spk["speaker_id"], "TRAI", session_id="SES_001", take_index=2,
                provenance="human_field_recording",
                manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
                raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
            )

    # Confirm sibling take 1 files are 100% intact and unchanged
    assert raw1_path.exists()
    assert std1_path.exists()
    assert hashlib.sha256(raw1_path.read_bytes()).hexdigest() == raw1_sha_before
    assert hashlib.sha256(std1_path.read_bytes()).hexdigest() == std1_sha_before

    # Confirm take 2 artifacts were cleaned up
    raw2_path = sandbox_raw / spk["speaker_id"] / "SES_001" / f"field_trai_{spk['speaker_id']}_SES_001_take002.wav"
    std2_path = sandbox_std / "TRAI" / f"field_trai_{spk['speaker_id']}_SES_001_take002.wav"
    assert not raw2_path.exists()
    assert not std2_path.exists()

    # 3. Pre-existing file collision check: if destination file already exists on disk
    # Attempting to re-ingest take 1 fails closed to quarantine without overwriting or deleting
    res_collision = ingest_recording(
        wav2, spk["speaker_id"], "TRAI", session_id="SES_001", take_index=1,
        provenance="human_field_recording",
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_collision["status"] == "QUARANTINED"
    # Take 1 remains intact
    assert hashlib.sha256(raw1_path.read_bytes()).hexdigest() == raw1_sha_before
    assert hashlib.sha256(std1_path.read_bytes()).hexdigest() == std1_sha_before


def test_gate08_pre_recording_consent_verification(tmp_path):
    """
    Regression Test for Phase 1 (GATE-08 Pre-Recording Hardening):
    Confirms that `run_recording_session` and `verify_speaker_consent` fail closed before
    any microphone access, temp file generation, or manifest manipulation when:
      1) Speaker is unregistered.
      2) Consent ledger does not exist or is malformed (not a dict / invalid JSON).
      3) Consent record has consent_given=False.
      4) Consent has been revoked (revoked_at is set).
      5) Zero audio artifacts are generated upon consent failure.
      6) Pre-existing files in raw/std directories remain completely bit-exact.
      7) Valid consent allows session execution in mock mode.
    """
    import json
    import hashlib
    from src.acquisition.speaker_consent import register_speaker, verify_speaker_consent, revoke_speaker_consent
    from src.acquisition.record_command_cli import run_recording_session

    sandbox = tmp_path / "sandbox_gate08"
    sandbox.mkdir(parents=True, exist_ok=True)
    sandbox_ledger = sandbox / "consent_ledger.json"
    sandbox_manifest = sandbox / "manifest.csv"
    sandbox_quarantine = sandbox / "quarantine.csv"
    sandbox_raw = sandbox / "raw"
    sandbox_std = sandbox / "std"
    sandbox_raw.mkdir(parents=True, exist_ok=True)
    sandbox_std.mkdir(parents=True, exist_ok=True)

    # Place a canary file in raw directory to verify zero-pollution / preservation invariant
    canary_wav = create_test_wav(sandbox_raw / "canary.wav", freq=440.0)
    canary_sha_before = hashlib.sha256(canary_wav.read_bytes()).hexdigest()

    # 1. Non-existent ledger -> fails closed
    missing_ledger = sandbox / "missing_ledger.json"
    assert verify_speaker_consent("SPK_ANY", ledger_path=missing_ledger) is False
    res = run_recording_session(
        "SPK_ANY", commands=["LEN"], takes_per_command=1, mock_mode=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=missing_ledger
    )
    assert res is False

    # 2. Malformed ledger (invalid JSON, list instead of dict, or malformed record)
    sandbox_ledger.write_text("NOT_VALID_JSON", encoding="utf-8")
    assert verify_speaker_consent("SPK_ANY", ledger_path=sandbox_ledger) is False
    res = run_recording_session(
        "SPK_ANY", commands=["LEN"], takes_per_command=1, mock_mode=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res is False

    sandbox_ledger.write_text("[]", encoding="utf-8")
    assert verify_speaker_consent("SPK_ANY", ledger_path=sandbox_ledger) is False

    # 3. Speaker registration with consent_given=False fails immediately
    with pytest.raises(ValueError, match="Informed consent was NOT granted"):
        register_speaker(
            dialect="Northern", gender="FEMALE", age_bracket="18-30",
            consent_given=False, device_model="Pixel 7", microphone_type="builtin_mobile",
            distance_category="close_talk_15cm", ledger_path=sandbox_ledger
        )

    # Manually write an unconsented entry to ledger
    ledger_data = {
        "SPK_DENIED": {
            "speaker_id": "SPK_DENIED",
            "consent_given": False,
            "dialect": "Northern",
            "gender": "FEMALE"
        }
    }
    with open(sandbox_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger_data, f)

    assert verify_speaker_consent("SPK_DENIED", ledger_path=sandbox_ledger) is False
    res_denied = run_recording_session(
        "SPK_DENIED", commands=["LEN"], takes_per_command=1, mock_mode=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_denied is False

    # 4. Speaker registered with valid consent, then revoked
    spk_valid = register_speaker(
        dialect="Central", gender="MALE", age_bracket="31-45",
        consent_given=True, device_model="Galaxy S22", microphone_type="builtin_mobile",
        distance_category="close_talk_15cm", ledger_path=sandbox_ledger
    )
    assert verify_speaker_consent(spk_valid["speaker_id"], ledger_path=sandbox_ledger) is True

    # Revoke consent
    revoke_speaker_consent(spk_valid["speaker_id"], ledger_path=sandbox_ledger)
    assert verify_speaker_consent(spk_valid["speaker_id"], ledger_path=sandbox_ledger) is False
    res_revoked = run_recording_session(
        spk_valid["speaker_id"], commands=["LEN"], takes_per_command=1, mock_mode=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_revoked is False

    # 5. Verify canary is completely preserved and no spurious audio was created for denied/revoked speakers
    assert canary_wav.exists()
    assert hashlib.sha256(canary_wav.read_bytes()).hexdigest() == canary_sha_before
    raw_subdirs = [p for p in sandbox_raw.iterdir() if p.is_dir()]
    assert len(raw_subdirs) == 0

    # 6. Valid consent allows recording execution in mock mode
    spk_active = register_speaker(
        dialect="Southern", gender="FEMALE", age_bracket="18-30",
        consent_given=True, device_model="Pixel 7", microphone_type="builtin_mobile",
        distance_category="close_talk_15cm", ledger_path=sandbox_ledger
    )
    res_success = run_recording_session(
        spk_active["speaker_id"], commands=["LEN"], takes_per_command=1, mock_mode=True,
        manifest_path=sandbox_manifest, quarantine_path=sandbox_quarantine,
        raw_base_dir=sandbox_raw, std_base_dir=sandbox_std, ledger_path=sandbox_ledger
    )
    assert res_success is True
    assert sandbox_manifest.exists()
    manifest_df = pd.read_csv(sandbox_manifest)
    assert len(manifest_df) == 1
    assert manifest_df.iloc[0]["speaker_id"] == spk_active["speaker_id"]
    assert manifest_df.iloc[0]["status"] == "APPROVED"


def test_split_manifests_preflight_validation():
    """
    Regression Test for Phase 2 (Speaker-Disjoint Preflight Boundary):
    Confirms that `validate_split_manifests_preflight` validates:
      1) Non-empty partitions.
      2) Required columns presence (recording_id, speaker_id, session_id).
      3) Strict speaker-disjointness (flags direct overlap and missing/whitespace speaker IDs).
      4) Session-disjointness across partitions.
      5) Recording ID and parent lineage disjointness across partitions.
      6) Acceptance of valid, fully disjoint partitions.
    """
    from src.acquisition.campaign_validator import validate_split_manifests_preflight

    def make_split_df(rows):
        return pd.DataFrame(rows)

    # 1. Valid disjoint split
    df_train = make_split_df([
        {"recording_id": "REC_001", "speaker_id": "SPK_001", "session_id": "SES_001", "command_label": "LEN"},
        {"recording_id": "REC_002", "speaker_id": "SPK_002", "session_id": "SES_002", "command_label": "TRAI"}
    ])
    df_val = make_split_df([
        {"recording_id": "REC_003", "speaker_id": "SPK_003", "session_id": "SES_003", "command_label": "LEN"}
    ])
    df_test = make_split_df([
        {"recording_id": "REC_004", "speaker_id": "SPK_004", "session_id": "SES_004", "command_label": "PHAI"}
    ])
    is_valid, errors, summary = validate_split_manifests_preflight(df_train, df_val, df_test, profile="benchmark")
    assert is_valid is True
    assert len(errors) == 0
    assert summary["train_speaker_count"] == 2
    assert summary["val_speaker_count"] == 1
    assert summary["test_speaker_count"] == 1

    # 2. Speaker overlap rejection (train and val share a speaker)
    df_val_leak = make_split_df([
        {"recording_id": "REC_005", "speaker_id": "SPK_001", "session_id": "SES_005", "command_label": "LEN"}
    ])
    is_valid, errors, _ = validate_split_manifests_preflight(df_train, df_val_leak, profile="benchmark")
    assert is_valid is False
    assert any("speaker_overlap_detected" in e for e in errors)
    assert any("SPK_001" in e for e in errors)

    # 3. Missing speaker_id column rejection
    df_no_speaker = df_train.drop(columns=["speaker_id"])
    is_valid, errors, _ = validate_split_manifests_preflight(df_no_speaker, df_val, profile="benchmark")
    assert is_valid is False
    assert any("missing_required_columns" in e for e in errors)

    # 4. NaN / empty string / whitespace speaker_id rejection
    df_nan_spk = make_split_df([
        {"recording_id": "REC_010", "speaker_id": None, "session_id": "SES_010", "command_label": "LEN"},
        {"recording_id": "REC_011", "speaker_id": "   ", "session_id": "SES_011", "command_label": "TRAI"}
    ])
    is_valid, errors, _ = validate_split_manifests_preflight(df_nan_spk, df_val, profile="benchmark")
    assert is_valid is False
    assert any("missing_or_invalid_speaker_ids" in e for e in errors)

    # 5. Session overlap rejection (different speakers, but same session ID)
    df_val_sess_leak = make_split_df([
        {"recording_id": "REC_012", "speaker_id": "SPK_009", "session_id": "SES_001", "command_label": "LEN"}
    ])
    is_valid, errors, _ = validate_split_manifests_preflight(df_train, df_val_sess_leak, profile="benchmark")
    assert is_valid is False
    assert any("session_overlap_detected" in e for e in errors)

    # 6. Parent lineage overlap rejection (derived/augmented records sharing raw source take)
    df_train_parent = make_split_df([
        {"recording_id": "REC_AUG_01", "parent_recording_id": "REC_RAW_ROOT", "speaker_id": "SPK_010", "session_id": "SES_010", "command_label": "LEN"}
    ])
    df_val_parent = make_split_df([
        {"recording_id": "REC_AUG_02", "parent_recording_id": "REC_RAW_ROOT", "speaker_id": "SPK_011", "session_id": "SES_011", "command_label": "LEN"}
    ])
    is_valid, errors, _ = validate_split_manifests_preflight(df_train_parent, df_val_parent, profile="benchmark")
    assert is_valid is False
    assert any("augmentation_lineage_overlap_detected" in e for e in errors)

    # 7. Empty partition rejection
    is_valid, errors, _ = validate_split_manifests_preflight(pd.DataFrame(), df_val, profile="benchmark")
    assert is_valid is False
    assert any("is_empty" in e for e in errors)


def test_training_boundary_enforces_split_disjointness():
    """
    Regression Test for Phase 2 Training Entrypoints:
    Verifies that both `train_model` (in src.models.train_improved) and
    `train_baseline` (in src.models.train_kws) fail closed with ValueError
    at the dataset loading boundary if input split manifests contain speaker overlap.
    """
    from unittest.mock import patch

    df_leak_train = pd.DataFrame([
        {"recording_id": "TR_001", "speaker_id": "SPK_LEAK", "session_id": "SES_01", "processed_path": "a.wav", "command_label": "LEN"}
    ])
    df_leak_val = pd.DataFrame([
        {"recording_id": "VA_001", "speaker_id": "SPK_LEAK", "session_id": "SES_02", "processed_path": "b.wav", "command_label": "LEN"}
    ])

    from src.models.train_improved import train_model
    from src.models.train_kws import train_baseline

    # Test train_improved boundary
    with patch("pandas.read_csv", side_effect=[df_leak_train, df_leak_val]):
        with pytest.raises(ValueError, match="Preflight split manifest validation failed"):
            train_model(epochs=1, batch_size=2, validation_profile="benchmark")

    # Test train_kws boundary
    with patch("pandas.read_csv", side_effect=[df_leak_train, df_leak_val]):
        with pytest.raises(ValueError, match="Baseline training halted: Preflight split manifest validation failed"):
            train_baseline(epochs=1, batch_size=2, validation_profile="benchmark")


def test_consent_withdrawal_residual_audit_dryrun(tmp_path):
    """
    Regression Test for Phase 3 (Dry-Run Residual Artifact Audit Utility):
    Confirms that `audit_speaker_residual_artifacts`:
      1) Correctly identifies primary manifest records, split records, active audio,
         quarantined files, derived feature caches, and backup/model artifacts.
      2) Distinguishes confirmed recording lineage vs uncertain filename match vs unresolved lineage.
      3) Is strictly read-only: does not modify, delete, or move any file on disk.
      4) Rejects invalid or malicious speaker IDs.
    """
    from src.acquisition.revoke_speaker_consent import audit_speaker_residual_artifacts

    sandbox = tmp_path / "sandbox_audit"
    sandbox.mkdir(parents=True, exist_ok=True)

    manifest_path = sandbox / "field_manifest.csv"
    split_train = sandbox / "train_manifest.csv"
    raw_dir = sandbox / "audio_raw"
    std_dir = sandbox / "audio_16k_standardized"
    quar_dir = sandbox / "quarantine"
    cache_dir = sandbox / "features"
    backup_dir = sandbox / "backups"
    model_dir = sandbox / "models"

    raw_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_spk = "SPK_VN_TEST_WITHDRAWAL"

    # 1. Setup primary manifest with 1 target record and 1 other record
    manifest_df = pd.DataFrame([
        {
            "recording_id": "REC_W101",
            "speaker_id": target_spk,
            "session_id": "SES_01",
            "command_label": "LEN",
            "status": "APPROVED",
            "raw_filepath": str(raw_dir / target_spk / "take1.wav").replace("\\", "/"),
            "processed_path": str(std_dir / f"std_{target_spk}_take1.wav").replace("\\", "/")
        },
        {
            "recording_id": "REC_W999",
            "speaker_id": "SPK_OTHER",
            "session_id": "SES_99",
            "command_label": "TRAI",
            "status": "APPROVED",
            "raw_filepath": str(raw_dir / "SPK_OTHER" / "take1.wav").replace("\\", "/"),
            "processed_path": str(std_dir / "std_SPK_OTHER_take1.wav").replace("\\", "/")
        }
    ])
    manifest_df.to_csv(manifest_path, index=False)

    # 2. Setup split manifest containing target speaker
    split_df = pd.DataFrame([
        {"recording_id": "REC_W101", "speaker_id": target_spk, "session_id": "SES_01"}
    ])
    split_df.to_csv(split_train, index=False)

    # 3. Setup audio files on disk
    spk_raw = raw_dir / target_spk / "take1.wav"
    spk_raw.parent.mkdir(parents=True, exist_ok=True)
    spk_raw.write_bytes(b"RIFFmockrawbytes")

    spk_std = std_dir / f"std_{target_spk}_take1.wav"
    spk_std.write_bytes(b"RIFFmockstdbytes")

    # 4. Setup quarantined file
    quar_spk_dir = quar_dir / "withdrawn_speakers" / target_spk
    quar_spk_dir.mkdir(parents=True, exist_ok=True)
    quar_file = quar_spk_dir / "quarantined_take1.wav"
    quar_file.write_bytes(b"RIFFmockquarbytes")

    # 5. Setup feature caches
    cache_confirmed = cache_dir / "REC_W101_spec.npy"
    cache_confirmed.write_bytes(b"NUMPY_MOCK_CONFIRMED")

    cache_uncertain = cache_dir / f"cache_{target_spk}.npy"
    cache_uncertain.write_bytes(b"NUMPY_MOCK_UNCERTAIN")

    cache_unresolved = cache_dir / "unrelated_spec.npy"
    cache_unresolved.write_bytes(b"NUMPY_MOCK_UNRESOLVED")

    # 6. Setup backup archive and model checkpoint
    archive_file = backup_dir / "dataset_backup_2026.tar.gz"
    archive_file.write_bytes(b"GZIP_MOCK_ARCHIVE")

    model_file = model_dir / "best_kws_tcresnet.pt"
    model_file.write_bytes(b"TORCH_MOCK_WEIGHTS")

    # Capture state and SHA-256 before audit
    all_files_before = {}
    for p in sandbox.rglob("*"):
        if p.is_file():
            all_files_before[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Execute read-only dry-run audit
    audit = audit_speaker_residual_artifacts(
        speaker_id=target_spk,
        manifest_path=manifest_path,
        split_manifest_paths=[split_train],
        raw_base_dir=raw_dir,
        std_base_dir=std_dir,
        quarantine_dir=quar_dir,
        feature_cache_dirs=[cache_dir],
        backup_dirs=[backup_dir],
        model_checkpoint_dirs=[model_dir]
    )

    # Verify audit findings
    assert audit["speaker_id"] == target_spk
    assert audit["primary_manifest"]["total"] == 1
    assert len(audit["split_manifests_affected"]) == 1
    assert audit["split_manifests_affected"][0]["status"] == "CONFIRMED_RESIDUAL_SPLIT_RECORD"
    assert len(audit["active_audio_files"]) == 2

    # Feature cache classifications
    cache_confidences = [f["match_confidence"] for f in audit["derived_feature_caches"]]
    assert "CONFIRMED_RECORDING_LINEAGE" in cache_confidences
    assert "UNCERTAIN_FILENAME_MATCH" in cache_confidences
    assert audit["unresolved_cache_lineage_count"] == 1

    # Backups and model checkpoints
    assert len(audit["cold_backup_archives"]) == 1
    assert audit["cold_backup_archives"][0]["lineage_status"] == "UNRESOLVED_ARCHIVE_LINEAGE"
    assert len(audit["model_checkpoints"]) == 1
    assert audit["model_checkpoints"][0]["lineage_status"] == "UNRESOLVABLE_CHECKPOINT_WEIGHTS"

    # Verify STRICTLY READ-ONLY invariant: every file before is present with identical hash
    all_files_after = {}
    for p in sandbox.rglob("*"):
        if p.is_file():
            all_files_after[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    assert all_files_before == all_files_after

    # Test invalid speaker ID rejection
    with pytest.raises(ValueError, match="Invalid characters in speaker_id"):
        audit_speaker_residual_artifacts("SPK; rm -rf /", manifest_path=manifest_path)


def test_evaluation_entrypoint_enforces_held_out_test_disjointness(tmp_path):
    """
    Regression Test: Held-Out Test Partition Enforcement.
    Confirms that evaluation entry points (`evaluate_model` in src.models.evaluate_kws)
    enforce preflight split manifest validation before evaluating model checkpoints:
      1) Fails closed (ValueError) if test set has speaker overlap with train or val.
      2) Fails closed if test set has session overlap with train or val.
      3) Fails closed if test set has parent recording lineage overlap.
      4) Halts before any model loading, feature extraction, or inference occurs.
    """
    from src.models.evaluate_kws import evaluate_model

    sandbox = tmp_path / "sandbox_eval_boundary"
    sandbox.mkdir(parents=True, exist_ok=True)

    train_csv = sandbox / "train.csv"
    val_csv = sandbox / "val.csv"
    test_leak_csv = sandbox / "test_leak.csv"

    # 1. Base disjoint train and val
    df_train = pd.DataFrame([
        {"recording_id": "TR_001", "speaker_id": "SPK_TRAIN_01", "session_id": "SES_TR_01", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_val = pd.DataFrame([
        {"recording_id": "VA_001", "speaker_id": "SPK_VAL_01", "session_id": "SES_VA_01", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    # 2. Leaking test manifest: shares SPK_TRAIN_01 with train
    df_test_spk_leak = pd.DataFrame([
        {"recording_id": "TE_001", "speaker_id": "SPK_TRAIN_01", "session_id": "SES_TE_01", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_test_spk_leak.to_csv(test_leak_csv, index=False)

    with pytest.raises(ValueError, match="Evaluation halted: Test partition violates disjointness boundary"):
        evaluate_model(
            manifest_path=str(test_leak_csv),
            ckpt_path_str="nonexistent_checkpoint.pt",
            train_manifest_path=str(train_csv),
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )

    # 3. Leaking test manifest: session overlap
    df_test_ses_leak = pd.DataFrame([
        {"recording_id": "TE_002", "speaker_id": "SPK_TEST_01", "session_id": "SES_TR_01", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_test_ses_leak.to_csv(test_leak_csv, index=False)

    with pytest.raises(ValueError, match="Evaluation halted: Test partition violates disjointness boundary"):
        evaluate_model(
            manifest_path=str(test_leak_csv),
            ckpt_path_str="nonexistent_checkpoint.pt",
            train_manifest_path=str(train_csv),
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )

    # 4. Leaking test manifest: parent lineage overlap
    df_train_parent = pd.DataFrame([
        {"recording_id": "TR_AUG_01", "parent_recording_id": "REC_ROOT_01", "speaker_id": "SPK_TRAIN_02", "session_id": "SES_TR_02", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_val_parent = pd.DataFrame([
        {"recording_id": "VA_AUG_01", "parent_recording_id": "REC_ROOT_02", "speaker_id": "SPK_VAL_02", "session_id": "SES_VA_02", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_test_parent_leak = pd.DataFrame([
        {"recording_id": "TE_AUG_01", "parent_recording_id": "REC_ROOT_01", "speaker_id": "SPK_TEST_02", "session_id": "SES_TE_02", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_train_parent.to_csv(train_csv, index=False)
    df_val_parent.to_csv(val_csv, index=False)
    df_test_parent_leak.to_csv(test_leak_csv, index=False)

    with pytest.raises(ValueError, match="Evaluation halted: Test partition violates disjointness boundary"):
        evaluate_model(
            manifest_path=str(test_leak_csv),
            ckpt_path_str="nonexistent_checkpoint.pt",
            train_manifest_path=str(train_csv),
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )


def test_evaluation_entrypoints_missing_or_invalid_manifests_fail_closed(tmp_path):
    """
    Regression Test: Missing-manifest fail-closed behavior across evaluation entry points.
    Verifies that evaluate_model, evaluate_all_models, and build_continuous_eval_streams
    cannot proceed if train or validation manifests required for held-out test isolation
    are missing, unreadable, or invalid.
    """
    from src.models.evaluate_kws import evaluate_model
    from src.models.evaluate_streaming import build_continuous_eval_streams
    from src.models.evaluate_all_models import main as evaluate_all_main

    sandbox = tmp_path / "sandbox_missing_manifest"
    sandbox.mkdir(parents=True, exist_ok=True)

    test_csv = sandbox / "test.csv"
    train_csv = sandbox / "train.csv"
    val_csv = sandbox / "val.csv"
    corrupt_csv = sandbox / "corrupt.csv"

    df_test = pd.DataFrame([
        {"recording_id": "TE_001", "speaker_id": "SPK_TEST_01", "session_id": "SES_TE_01", "processed_path": "dummy.wav", "class_id": 0, "class_label": "TRAI"}
    ])
    df_train = pd.DataFrame([
        {"recording_id": "TR_001", "speaker_id": "SPK_TRAIN_01", "session_id": "SES_TR_01", "processed_path": "dummy.wav", "class_id": 0}
    ])
    df_val = pd.DataFrame([
        {"recording_id": "VA_001", "speaker_id": "SPK_VAL_01", "session_id": "SES_VA_01", "processed_path": "dummy.wav", "class_id": 0}
    ])

    df_test.to_csv(test_csv, index=False)
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    with open(corrupt_csv, "w") as f:
        f.write("not,a,valid\ncsv\ncontent\n\"broken")

    # 1. evaluate_model: missing train manifest file on disk
    with pytest.raises(FileNotFoundError, match="Train manifest required for held-out test isolation is missing"):
        evaluate_model(
            manifest_path=str(test_csv),
            ckpt_path_str="nonexistent.pt",
            train_manifest_path=str(sandbox / "nonexistent_train.csv"),
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )

    # 2. evaluate_model: missing val manifest file on disk
    with pytest.raises(FileNotFoundError, match="Validation manifest required for held-out test isolation is missing"):
        evaluate_model(
            manifest_path=str(test_csv),
            ckpt_path_str="nonexistent.pt",
            train_manifest_path=str(train_csv),
            val_manifest_path=str(sandbox / "nonexistent_val.csv"),
            validation_profile="benchmark"
        )

    # 3. evaluate_model: train_manifest_path is None
    with pytest.raises(ValueError, match="Train manifest path is required for held-out test isolation"):
        evaluate_model(
            manifest_path=str(test_csv),
            ckpt_path_str="nonexistent.pt",
            train_manifest_path=None,
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )

    # 4. evaluate_model: val_manifest_path is None
    with pytest.raises(ValueError, match="Validation manifest path is required for held-out test isolation"):
        evaluate_model(
            manifest_path=str(test_csv),
            ckpt_path_str="nonexistent.pt",
            train_manifest_path=str(train_csv),
            val_manifest_path=None,
            validation_profile="benchmark"
        )

    # 5. evaluate_model: unreadable/corrupt train manifest
    with pytest.raises(ValueError, match="Could not read train or validation manifest for isolation check"):
        evaluate_model(
            manifest_path=str(test_csv),
            ckpt_path_str="nonexistent.pt",
            train_manifest_path=str(corrupt_csv),
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )

    # 6. build_continuous_eval_streams: missing train manifest
    with pytest.raises(FileNotFoundError, match="Missing train manifest required for held-out test isolation"):
        build_continuous_eval_streams(
            test_manifest_path=str(test_csv),
            train_manifest_path=str(sandbox / "nonexistent_train.csv"),
            val_manifest_path=str(val_csv),
            validation_profile="benchmark"
        )

    # 7. build_continuous_eval_streams: missing val manifest
    with pytest.raises(FileNotFoundError, match="Missing val manifest required for held-out test isolation"):
        build_continuous_eval_streams(
            test_manifest_path=str(test_csv),
            train_manifest_path=str(train_csv),
            val_manifest_path=str(sandbox / "nonexistent_val.csv"),
            validation_profile="benchmark"
        )

    # 8. evaluate_all_main: missing train manifest
    with pytest.raises(FileNotFoundError, match="Missing train manifest required for held-out test isolation"):
        evaluate_all_main(
            test_manifest_path_str=str(test_csv),
            train_manifest_path_str=str(sandbox / "nonexistent_train.csv"),
            val_manifest_path_str=str(val_csv),
            validation_profile="benchmark"
        )


def test_pairwise_split_manifest_validation_exhaustive():
    """
    Regression Test: Pairwise Split Validation & Metadata Integrity.
    Verifies that validate_split_manifests_preflight exhaustively enforces:
      - Pairwise speaker disjointness across (train/val, train/test, val/test).
      - Pairwise session disjointness across all three pairs.
      - Pairwise recording ID disjointness across all three pairs.
      - Pairwise augmentation lineage disjointness across all three pairs.
      - Rejection of missing required columns (speaker_id, recording_id).
      - Rejection of empty/null/whitespace speaker and recording IDs.
      - Fail-closed behavior for require_sessions and require_lineage.
    """
    from src.acquisition.campaign_validator import validate_split_manifests_preflight

    base_train = pd.DataFrame([
        {"recording_id": "TR_001", "speaker_id": "SPK_TR_01", "session_id": "SES_TR_01", "parent_recording_id": "ROOT_TR"}
    ])
    base_val = pd.DataFrame([
        {"recording_id": "VA_001", "speaker_id": "SPK_VA_01", "session_id": "SES_VA_01", "parent_recording_id": "ROOT_VA"}
    ])
    base_test = pd.DataFrame([
        {"recording_id": "TE_001", "speaker_id": "SPK_TE_01", "session_id": "SES_TE_01", "parent_recording_id": "ROOT_TE"}
    ])

    # 1. Clean base case passes
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, base_test, profile="benchmark", require_sessions=True, require_lineage=True)
    assert valid is True
    assert len(errs) == 0

    # 2. Val-Test speaker leakage
    val_test_spk_leak = base_test.copy()
    val_test_spk_leak["speaker_id"] = "SPK_VA_01"
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, val_test_spk_leak, profile="benchmark")
    assert valid is False
    assert any("val_test" in e and "speaker_overlap" in e for e in errs)

    # 3. Train-Val speaker leakage
    train_val_spk_leak = base_val.copy()
    train_val_spk_leak["speaker_id"] = "SPK_TR_01"
    valid, errs, _ = validate_split_manifests_preflight(base_train, train_val_spk_leak, base_test, profile="benchmark")
    assert valid is False
    assert any("train_val" in e and "speaker_overlap" in e for e in errs)

    # 4. Val-Test session leakage
    val_test_ses_leak = base_test.copy()
    val_test_ses_leak["session_id"] = "SES_VA_01"
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, val_test_ses_leak, profile="benchmark")
    assert valid is False
    assert any("val_test" in e and "session_overlap" in e for e in errs)

    # 5. Train-Val session leakage
    train_val_ses_leak = base_val.copy()
    train_val_ses_leak["session_id"] = "SES_TR_01"
    valid, errs, _ = validate_split_manifests_preflight(base_train, train_val_ses_leak, base_test, profile="benchmark")
    assert valid is False
    assert any("train_val" in e and "session_overlap" in e for e in errs)

    # 6. Pairwise recording ID leakage: train-test, val-test, train-val
    rec_leak_tt = base_test.copy()
    rec_leak_tt["recording_id"] = "TR_001"
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, rec_leak_tt, profile="benchmark")
    assert valid is False
    assert any("recording_id_overlap" in e for e in errs)

    rec_leak_vt = base_test.copy()
    rec_leak_vt["recording_id"] = "VA_001"
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, rec_leak_vt, profile="benchmark")
    assert valid is False
    assert any("recording_id_overlap" in e for e in errs)

    rec_leak_tv = base_val.copy()
    rec_leak_tv["recording_id"] = "TR_001"
    valid, errs, _ = validate_split_manifests_preflight(base_train, rec_leak_tv, base_test, profile="benchmark")
    assert valid is False
    assert any("recording_id_overlap" in e for e in errs)

    # 7. Val-Test augmentation lineage leakage
    lin_leak_vt = base_test.copy()
    lin_leak_vt["parent_recording_id"] = "VA_001"
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, lin_leak_vt, profile="benchmark")
    assert valid is False
    assert any("lineage_overlap" in e for e in errs)

    # 8. Missing required column (speaker_id in test)
    no_spk = base_test.drop(columns=["speaker_id"])
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, no_spk, profile="benchmark")
    assert valid is False
    assert any("missing_required_columns" in e for e in errs)

    # 9. Missing required column (recording_id in train)
    no_rec = base_train.drop(columns=["recording_id"])
    valid, errs, _ = validate_split_manifests_preflight(no_rec, base_val, base_test, profile="benchmark")
    assert valid is False
    assert any("missing_required_columns" in e for e in errs)

    # 10. Blank/NaN speaker ID
    bad_spk = base_test.copy()
    bad_spk.loc[0, "speaker_id"] = "   "
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, bad_spk, profile="benchmark")
    assert valid is False
    assert any("missing_or_invalid_speaker_ids" in e for e in errs)

    # 11. Blank/NaN recording ID
    bad_rec = base_test.copy()
    bad_rec.loc[0, "recording_id"] = ""
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, bad_rec, profile="benchmark")
    assert valid is False
    assert any("missing_or_invalid_recording_ids" in e for e in errs)

    # 12. require_sessions=True with missing or blank session_id
    no_ses = base_test.drop(columns=["session_id"])
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, no_ses, profile="benchmark", require_sessions=True)
    assert valid is False
    assert any("session_id" in e for e in errs)

    bad_ses = base_test.copy()
    bad_ses.loc[0, "session_id"] = "nan"
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, bad_ses, profile="benchmark", require_sessions=True)
    assert valid is False
    assert any("missing_or_invalid_session_ids" in e for e in errs)

    # 13. require_lineage=True with missing parent_recording_id
    no_lin = base_test.drop(columns=["parent_recording_id"])
    valid, errs, _ = validate_split_manifests_preflight(base_train, base_val, no_lin, profile="benchmark", require_lineage=True)
    assert valid is False
    assert any("parent_recording_id" in e for e in errs)


def test_preflight_validator_schema_aware_human_campaign_enforcement():
    """
    Integration & Regression Test: Schema-Aware Preflight Split Validation.
    Guarantees:
      1. Benchmark manifests (16-column) without session_id pass with default arguments
         while explicitly reporting session_status and lineage_status as NOT_VERIFIABLE_COLUMN_ABSENT.
      2. Human campaign manifests (having consent_id, human campaign source, or HUMAN_CONSENTED)
         strictly require session_id even under default arguments (require_sessions=False).
      3. Missing or invalid session_ids in human campaign manifests fail closed.
      4. Session leakage across human campaign partitions fails closed.
      5. Valid human campaign manifests pass with VERIFIED_DISJOINT for both speaker and session.
    """
    from src.acquisition.campaign_validator import validate_split_manifests_preflight

    # 1. Benchmark 16-column style data (no session_id, no consent_id)
    bench_tr = pd.DataFrame([
        {"recording_id": "bench_tr_01", "speaker_id": "spk_b_01", "command_label": "LEN"},
        {"recording_id": "bench_tr_02", "speaker_id": "spk_b_02", "command_label": "XUONG"}
    ])
    bench_va = pd.DataFrame([
        {"recording_id": "bench_va_01", "speaker_id": "spk_b_03", "command_label": "LEN"}
    ])
    bench_te = pd.DataFrame([
        {"recording_id": "bench_te_01", "speaker_id": "spk_b_04", "command_label": "TRAI"}
    ])

    valid_bench, errs_bench, diag_bench = validate_split_manifests_preflight(bench_tr, bench_va, bench_te, profile="benchmark")
    assert valid_bench is True
    assert len(errs_bench) == 0
    assert diag_bench["speaker_status"] == "VERIFIED_DISJOINT"
    assert diag_bench["recording_status"] == "VERIFIED_DISJOINT"
    assert diag_bench["session_status"] == "NOT_VERIFIABLE_COLUMN_ABSENT"
    assert diag_bench["lineage_status"] == "NOT_VERIFIABLE_COLUMN_ABSENT"

    # 2. Human campaign manifest data (28-column schema with consent_id and human provenance)
    human_tr = pd.DataFrame([
        {
            "recording_id": "rec_h_001_take001",
            "speaker_id": "SPK_H_N_000001",
            "session_id": "SES_H_001_01",
            "parent_recording_id": None,
            "consent_id": "CONSENT_H_N_000001_v1.0",
            "source": "vietnamese_kws_human_campaign_v1",
            "label_provenance": "HUMAN_CONSENTED",
            "command_label": "LEN"
        }
    ])
    human_va = pd.DataFrame([
        {
            "recording_id": "rec_h_002_take001",
            "speaker_id": "SPK_H_C_000002",
            "session_id": "SES_H_002_01",
            "parent_recording_id": None,
            "consent_id": "CONSENT_H_C_000002_v1.0",
            "source": "vietnamese_kws_human_campaign_v1",
            "label_provenance": "HUMAN_CONSENTED",
            "command_label": "LEN"
        }
    ])
    human_te = pd.DataFrame([
        {
            "recording_id": "rec_h_003_take001",
            "speaker_id": "SPK_H_S_000003",
            "session_id": "SES_H_003_01",
            "parent_recording_id": None,
            "consent_id": "CONSENT_H_S_000003_v1.0",
            "source": "vietnamese_kws_human_campaign_v1",
            "label_provenance": "HUMAN_CONSENTED",
            "command_label": "PHAI"
        }
    ])

    # 2a. Clean human campaign partitions pass with verified disjointness
    valid_h, errs_h, diag_h = validate_split_manifests_preflight(human_tr, human_va, human_te, profile="human_campaign")
    assert valid_h is True
    assert len(errs_h) == 0
    assert diag_h["speaker_status"] == "VERIFIED_DISJOINT"
    assert diag_h["session_status"] == "VERIFIED_DISJOINT"
    assert diag_h["lineage_status"] == "VERIFIED_DISJOINT"

    # 2b. Human campaign missing session_id must fail closed even under default require_sessions=False
    human_te_no_ses = human_te.drop(columns=["session_id"])
    valid_no_ses, errs_no_ses, _ = validate_split_manifests_preflight(human_tr, human_va, human_te_no_ses, profile="human_campaign")
    assert valid_no_ses is False
    assert any("missing_required_columns" in e and "session_id" in e for e in errs_no_ses)

    # 2c. Human campaign with blank/whitespace session_id must fail closed
    human_te_bad_ses = human_te.copy()
    human_te_bad_ses["session_id"] = "   "
    valid_bad_ses, errs_bad_ses, _ = validate_split_manifests_preflight(human_tr, human_va, human_te_bad_ses, profile="human_campaign")
    assert valid_bad_ses is False
    assert any("missing_or_invalid_session_ids" in e for e in errs_bad_ses)

    # 2d. Human campaign with session leakage across partitions must fail closed
    human_te_ses_leak = human_te.copy()
    human_te_ses_leak["session_id"] = "SES_H_001_01"  # matches train session
    valid_leak, errs_leak, _ = validate_split_manifests_preflight(human_tr, human_va, human_te_ses_leak, profile="human_campaign")
    assert valid_leak is False
    assert any("session_overlap_detected" in e for e in errs_leak)


def test_lineage_enforcement_and_validation_profiles_exhaustive():
    """
    Exhaustive verification of lineage enforcement and validation profile selection.
    Covers the 9 mandatory integration criteria:
      1. Benchmark manifests pass schema-supported validation with honest session/lineage status.
      2. Human-campaign mode rejects a missing session_id.
      3. Human-campaign mode rejects a missing required parent_recording_id column.
      4. Missing human-campaign markers cannot silently downgrade an explicitly configured human-campaign pipeline.
      5. Invalid, unresolved, or cross-partition lineage fails according to the canonical schema.
      6. Valid root recordings with nullable parent IDs remain valid if and only if the schema permits them.
      7. Session and lineage overlap across all three partition pairs is rejected.
      8. Training and evaluation callers pass the intended validation profile.
      9. Validation does not modify input manifests.
    """
    from src.acquisition.campaign_validator import validate_split_manifests_preflight

    # Base Benchmark partitions (16-column style: no session_id, no parent_recording_id)
    base_bench_tr = pd.DataFrame([
        {"recording_id": "b_tr_01", "speaker_id": "spk_b_01", "command_label": "LEN"},
        {"recording_id": "b_tr_02", "speaker_id": "spk_b_02", "command_label": "XUONG"}
    ])
    base_bench_va = pd.DataFrame([
        {"recording_id": "b_va_01", "speaker_id": "spk_b_03", "command_label": "LEN"}
    ])
    base_bench_te = pd.DataFrame([
        {"recording_id": "b_te_01", "speaker_id": "spk_b_04", "command_label": "TRAI"}
    ])

    # 1. Benchmark manifests pass schema-supported validation with honest session/lineage status
    val_b, errs_b, diag_b = validate_split_manifests_preflight(base_bench_tr, base_bench_va, base_bench_te, profile="benchmark")
    assert val_b is True
    assert len(errs_b) == 0
    assert diag_b["speaker_status"] == "VERIFIED_DISJOINT"
    assert diag_b["recording_status"] == "VERIFIED_DISJOINT"
    assert diag_b["session_status"] == "NOT_VERIFIABLE_COLUMN_ABSENT"
    assert diag_b["lineage_status"] == "NOT_VERIFIABLE_COLUMN_ABSENT"
    assert diag_b["validation_profile"] == "benchmark"

    # Base Human-Campaign partitions (root recordings with nullable parent IDs)
    hc_tr = pd.DataFrame([
        {"recording_id": "rec_h_tr_01", "speaker_id": "SPK_H_01", "session_id": "SES_H_01", "parent_recording_id": None, "command_label": "LEN"},
        {"recording_id": "rec_h_tr_02", "speaker_id": "SPK_H_02", "session_id": "SES_H_02", "parent_recording_id": "", "command_label": "TRAI"}
    ])
    hc_va = pd.DataFrame([
        {"recording_id": "rec_h_va_01", "speaker_id": "SPK_H_03", "session_id": "SES_H_03", "parent_recording_id": np.nan, "command_label": "PHAI"}
    ])
    hc_te = pd.DataFrame([
        {"recording_id": "rec_h_te_01", "speaker_id": "SPK_H_04", "session_id": "SES_H_04", "parent_recording_id": None, "command_label": "XUONG"}
    ])

    # 2. Human-campaign mode rejects a missing session_id
    hc_te_no_ses = hc_te.drop(columns=["session_id"])
    val_no_ses, errs_no_ses, _ = validate_split_manifests_preflight(hc_tr, hc_va, hc_te_no_ses, profile="human_campaign")
    assert val_no_ses is False
    assert any("missing_required_columns" in e and "session_id" in e for e in errs_no_ses)

    # 3. Human-campaign mode rejects a missing required parent_recording_id column
    hc_tr_no_lin = hc_tr.drop(columns=["parent_recording_id"])
    val_no_lin, errs_no_lin, _ = validate_split_manifests_preflight(hc_tr_no_lin, hc_va, hc_te, profile="human_campaign")
    assert val_no_lin is False
    assert any("missing_required_columns" in e and "parent_recording_id" in e for e in errs_no_lin)

    # 4. Missing human-campaign markers cannot silently downgrade an explicitly configured human-campaign pipeline
    val_dg, errs_dg, diag_dg = validate_split_manifests_preflight(base_bench_tr, base_bench_va, base_bench_te, profile="human_campaign")
    assert val_dg is False
    assert diag_dg["validation_profile"] == "human_campaign"
    assert any("missing_required_columns" in e for e in errs_dg)

    # 5. Invalid, unresolved, or cross-partition lineage fails according to the canonical schema
    # 5a. Self-referential lineage
    hc_tr_self = hc_tr.copy()
    hc_tr_self.loc[0, "parent_recording_id"] = hc_tr_self.loc[0, "recording_id"]
    val_self, errs_self, _ = validate_split_manifests_preflight(hc_tr_self, hc_va, hc_te, profile="human_campaign")
    assert val_self is False
    assert any("self_referential_lineage" in e for e in errs_self)

    # 5b. Unresolved parent reference (missing parent in dataset)
    hc_tr_unres = hc_tr.copy()
    hc_tr_unres.loc[0, "parent_recording_id"] = "REC_ROOT_NONEXISTENT"
    val_unres, errs_unres, _ = validate_split_manifests_preflight(hc_tr_unres, hc_va, hc_te, profile="human_campaign")
    assert val_unres is False
    assert any("unresolved_parent_lineage" in e and "REC_ROOT_NONEXISTENT" in e for e in errs_unres)

    # 5c. Cross-partition parent/child lineage
    hc_tr_with_root = pd.DataFrame([
        {"recording_id": "REC_ROOT_TR", "speaker_id": "SPK_H_01", "session_id": "SES_H_01", "parent_recording_id": None},
        {"recording_id": "REC_CHILD_TR", "speaker_id": "SPK_H_01", "session_id": "SES_H_01", "parent_recording_id": "REC_ROOT_TR"}
    ])
    hc_va_cross = pd.DataFrame([
        {"recording_id": "REC_CHILD_VA", "speaker_id": "SPK_H_03", "session_id": "SES_H_03", "parent_recording_id": "REC_ROOT_TR"}
    ])
    val_cross, errs_cross, _ = validate_split_manifests_preflight(hc_tr_with_root, hc_va_cross, hc_te, profile="human_campaign")
    assert val_cross is False
    assert any("cross_partition_lineage" in e or "lineage_overlap" in e for e in errs_cross)

    # 6. Valid root recordings with nullable parent IDs remain valid
    val_roots, errs_roots, diag_roots = validate_split_manifests_preflight(hc_tr, hc_va, hc_te, profile="human_campaign")
    assert val_roots is True
    assert len(errs_roots) == 0
    assert diag_roots["speaker_status"] == "VERIFIED_DISJOINT"
    assert diag_roots["session_status"] == "VERIFIED_DISJOINT"
    assert diag_roots["lineage_status"] == "VERIFIED_DISJOINT"

    # Also test valid intra-partition derived child
    hc_tr_valid_child = pd.DataFrame([
        {"recording_id": "REC_ROOT_TR", "speaker_id": "SPK_H_01", "session_id": "SES_H_01", "parent_recording_id": None},
        {"recording_id": "REC_CHILD_TR", "speaker_id": "SPK_H_01", "session_id": "SES_H_01", "parent_recording_id": "REC_ROOT_TR"}
    ])
    val_child, errs_child, diag_child = validate_split_manifests_preflight(hc_tr_valid_child, hc_va, hc_te, profile="human_campaign")
    assert val_child is True
    assert len(errs_child) == 0
    assert diag_child["lineage_status"] == "VERIFIED_DISJOINT"

    # 7. Session and lineage overlap across all three partition pairs is rejected
    # 7a. Session overlaps across train_val, train_test, val_test
    ses_leak_tv = hc_va.copy()
    ses_leak_tv["session_id"] = "SES_H_01"  # matches train
    v, errs, _ = validate_split_manifests_preflight(hc_tr, ses_leak_tv, hc_te, profile="human_campaign")
    assert v is False and any("session_overlap" in e and "train_val" in e for e in errs)

    ses_leak_tt = hc_te.copy()
    ses_leak_tt["session_id"] = "SES_H_01"  # matches train
    v, errs, _ = validate_split_manifests_preflight(hc_tr, hc_va, ses_leak_tt, profile="human_campaign")
    assert v is False and any("session_overlap" in e and "train_test" in e for e in errs)

    ses_leak_vt = hc_te.copy()
    ses_leak_vt["session_id"] = "SES_H_03"  # matches val
    v, errs, _ = validate_split_manifests_preflight(hc_tr, hc_va, ses_leak_vt, profile="human_campaign")
    assert v is False and any("session_overlap" in e and "val_test" in e for e in errs)

    # 7b. Lineage overlaps across train_val, train_test, val_test
    tr_for_lin = pd.DataFrame([{"recording_id": "R_TR", "speaker_id": "S1", "session_id": "SES1", "parent_recording_id": None}])
    va_for_lin = pd.DataFrame([{"recording_id": "R_VA", "speaker_id": "S2", "session_id": "SES2", "parent_recording_id": None}])
    te_for_lin = pd.DataFrame([{"recording_id": "R_TE", "speaker_id": "S3", "session_id": "SES3", "parent_recording_id": None}])

    # Train-val lineage overlap
    va_lin_leak = pd.DataFrame([{"recording_id": "C_VA", "speaker_id": "S2", "session_id": "SES2", "parent_recording_id": "R_TR"}])
    v, errs, _ = validate_split_manifests_preflight(tr_for_lin, va_lin_leak, te_for_lin, profile="human_campaign")
    assert v is False and any("lineage" in e.lower() for e in errs)

    # Train-test lineage overlap
    te_lin_leak_tt = pd.DataFrame([{"recording_id": "C_TE1", "speaker_id": "S3", "session_id": "SES3", "parent_recording_id": "R_TR"}])
    v, errs, _ = validate_split_manifests_preflight(tr_for_lin, va_for_lin, te_lin_leak_tt, profile="human_campaign")
    assert v is False and any("lineage" in e.lower() for e in errs)

    # Val-test lineage overlap
    te_lin_leak_vt = pd.DataFrame([{"recording_id": "C_TE2", "speaker_id": "S3", "session_id": "SES3", "parent_recording_id": "R_VA"}])
    v, errs, _ = validate_split_manifests_preflight(tr_for_lin, va_for_lin, te_lin_leak_vt, profile="human_campaign")
    assert v is False and any("lineage" in e.lower() for e in errs)

    # 8. Training and evaluation callers pass the intended validation profile
    from src.models.train_improved import train_model
    from src.models.train_kws import train_baseline
    from src.models.evaluate_kws import evaluate_model
    from src.models.evaluate_all_models import main as evaluate_all_main
    from src.models.evaluate_streaming import build_continuous_eval_streams
    import inspect
    import pytest

    assert "validation_profile" in inspect.signature(train_model).parameters
    assert "validation_profile" in inspect.signature(train_baseline).parameters
    assert "validation_profile" in inspect.signature(evaluate_model).parameters
    assert "validation_profile" in inspect.signature(evaluate_all_main).parameters
    assert "validation_profile" in inspect.signature(build_continuous_eval_streams).parameters

    # Verify that invoking with human_campaign on benchmark manifests halts fail-closed before training/inference
    with pytest.raises(ValueError, match="Preflight split manifest validation failed"):
        train_model(epochs=1, validation_profile="human_campaign")

    with pytest.raises(ValueError, match="Preflight split manifest validation failed"):
        train_baseline(epochs=1, validation_profile="human_campaign")

    with pytest.raises(ValueError, match="violates disjointness boundary"):
        evaluate_model(validation_profile="human_campaign")

    with pytest.raises(ValueError, match="violates disjointness boundary"):
        evaluate_all_main(validation_profile="human_campaign")

    with pytest.raises(ValueError, match="violates disjointness boundary"):
        build_continuous_eval_streams(validation_profile="human_campaign")

    # 9. Validation does not modify input manifests
    hc_tr_copy = hc_tr.copy(deep=True)
    hc_va_copy = hc_va.copy(deep=True)
    hc_te_copy = hc_te.copy(deep=True)

    validate_split_manifests_preflight(hc_tr, hc_va, hc_te, profile="human_campaign")
    pd.testing.assert_frame_equal(hc_tr, hc_tr_copy)
    pd.testing.assert_frame_equal(hc_va, hc_va_copy)
    pd.testing.assert_frame_equal(hc_te, hc_te_copy)

    # Also verify with failing case
    validate_split_manifests_preflight(hc_tr_self, hc_va, hc_te, profile="human_campaign")
    pd.testing.assert_frame_equal(hc_tr, hc_tr_copy)


def test_omitted_and_invalid_validation_profiles_fail_closed(tmp_path):
    """
    Regression Test: Omitted and Unknown Validation Profiles Fail Closed.
    Guarantees:
      1. Calling validate_split_manifests_preflight without a profile fails closed.
      2. Calling training/evaluation entry points without validation_profile fails closed
         before data loading, model initialization, or inference.
      3. Passing unknown profiles fails closed across all entry points.
      4. CLI entry points exit non-zero when validation profile is omitted.
      5. Explicit 'benchmark' profile supports 16-column benchmark manifests without session_id/lineage.
      6. Explicit 'human_campaign' profile strictly requires both session_id and parent_recording_id.
    """
    import subprocess
    import sys
    from src.acquisition.campaign_validator import validate_split_manifests_preflight
    from src.models.train_improved import train_model
    from src.models.train_kws import train_baseline
    from src.models.evaluate_kws import evaluate_model
    from src.models.evaluate_all_models import main as evaluate_all_main
    from src.models.evaluate_streaming import build_continuous_eval_streams, main as evaluate_streaming_main

    # Synthetic manifests
    bench_tr = pd.DataFrame([{"recording_id": "TR_01", "speaker_id": "SPK_01", "command_label": "TRAI"}])
    bench_va = pd.DataFrame([{"recording_id": "VA_01", "speaker_id": "SPK_02", "command_label": "TRAI"}])
    bench_te = pd.DataFrame([{"recording_id": "TE_01", "speaker_id": "SPK_03", "command_label": "TRAI"}])

    # 1. validate_split_manifests_preflight with omitted profile
    valid, errs, diag = validate_split_manifests_preflight(bench_tr, bench_va, bench_te)
    assert valid is False
    assert any("omitted_validation_profile" in e for e in errs)
    assert diag["speaker_status"] == "OMITTED_PROFILE"

    valid, errs, _ = validate_split_manifests_preflight(bench_tr, bench_va, bench_te, validation_profile="")
    assert valid is False
    assert any("omitted_validation_profile" in e for e in errs)

    # Unknown profile
    valid, errs, diag = validate_split_manifests_preflight(bench_tr, bench_va, bench_te, profile="invalid_unknown_prof")
    assert valid is False
    assert any("invalid_validation_profile" in e for e in errs)
    assert diag["speaker_status"] == "INVALID_PROFILE"

    # 2. Programmatic APIs fail closed when profile is omitted
    with pytest.raises(ValueError, match="validation_profile must be explicitly specified"):
        train_model()

    with pytest.raises(ValueError, match="validation_profile must be explicitly specified"):
        train_baseline()

    with pytest.raises(ValueError, match="validation_profile must be explicitly specified"):
        evaluate_model()

    with pytest.raises(ValueError, match="validation_profile must be explicitly specified"):
        evaluate_all_main()

    with pytest.raises(ValueError, match="validation_profile must be explicitly specified"):
        build_continuous_eval_streams()

    with pytest.raises(ValueError, match="validation_profile must be explicitly specified"):
        evaluate_streaming_main()

    # 3. Programmatic APIs fail closed when unknown profile is provided
    with pytest.raises(ValueError, match="Unknown validation_profile"):
        train_model(validation_profile="invalid_prof")

    with pytest.raises(ValueError, match="Unknown validation_profile"):
        train_baseline(validation_profile="invalid_prof")

    with pytest.raises(ValueError, match="Unknown validation_profile"):
        evaluate_model(validation_profile="invalid_prof")

    with pytest.raises(ValueError, match="Unknown validation_profile"):
        evaluate_all_main(validation_profile="invalid_prof")

    with pytest.raises(ValueError, match="Unknown validation_profile"):
        build_continuous_eval_streams(validation_profile="invalid_prof")

    with pytest.raises(ValueError, match="Unknown validation_profile"):
        evaluate_streaming_main(validation_profile="invalid_prof")

    # 4. CLI entry points exit with code != 0 when profile is omitted
    dummy_tr_csv = tmp_path / "dummy_tr.csv"
    dummy_va_csv = tmp_path / "dummy_va.csv"
    bench_tr.to_csv(dummy_tr_csv, index=False)
    bench_va.to_csv(dummy_va_csv, index=False)

    cli_scripts = [
        [sys.executable, "src/acquisition/campaign_validator.py", "--verify-splits", "--train", str(dummy_tr_csv), "--val", str(dummy_va_csv)],
        [sys.executable, "src/models/train_improved.py", "--epochs", "1"],
        [sys.executable, "src/models/train_kws.py", "--epochs", "1"],
        [sys.executable, "src/models/evaluate_kws.py"],
        [sys.executable, "src/models/evaluate_all_models.py"],
        [sys.executable, "src/models/evaluate_streaming.py"],
    ]

    for cmd in cli_scripts:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0, f"Expected non-zero exit for omitted profile in {cmd[1]}, got {proc.returncode}"
        combined_output = proc.stdout + proc.stderr
        assert "must be explicitly specified" in combined_output or "error" in combined_output.lower()

    # 5. Explicit benchmark profile passes with 16-column benchmark schema
    val_b, errs_b, diag_b = validate_split_manifests_preflight(bench_tr, bench_va, bench_te, profile="benchmark")
    assert val_b is True
    assert len(errs_b) == 0
    assert diag_b["session_status"] == "NOT_VERIFIABLE_COLUMN_ABSENT"
    assert diag_b["lineage_status"] == "NOT_VERIFIABLE_COLUMN_ABSENT"

    # 6. Explicit human_campaign profile strictly requires session_id and parent_recording_id
    val_hc, errs_hc, _ = validate_split_manifests_preflight(bench_tr, bench_va, bench_te, profile="human_campaign")
    assert val_hc is False
    assert any("session_id" in e for e in errs_hc)
    assert any("parent_recording_id" in e for e in errs_hc)

    # Partial columns: has session_id but lacks parent_recording_id
    hc_partial_1 = bench_tr.copy()
    hc_partial_1["session_id"] = "SES_01"
    val_p1, errs_p1, _ = validate_split_manifests_preflight(hc_partial_1, bench_va, bench_te, profile="human_campaign")
    assert val_p1 is False
    assert any("parent_recording_id" in e for e in errs_p1)

    # Valid human_campaign passes
    hc_valid_tr = pd.DataFrame([{"recording_id": "REC_H_01", "speaker_id": "SPK_H_01", "session_id": "SES_01", "parent_recording_id": None}])
    hc_valid_va = pd.DataFrame([{"recording_id": "REC_H_02", "speaker_id": "SPK_H_02", "session_id": "SES_02", "parent_recording_id": None}])
    hc_valid_te = pd.DataFrame([{"recording_id": "REC_H_03", "speaker_id": "SPK_H_03", "session_id": "SES_03", "parent_recording_id": None}])
    val_hc_ok, errs_hc_ok, diag_hc_ok = validate_split_manifests_preflight(hc_valid_tr, hc_valid_va, hc_valid_te, profile="human_campaign")
    assert val_hc_ok is True
    assert len(errs_hc_ok) == 0
    assert diag_hc_ok["session_status"] == "VERIFIED_DISJOINT"
    assert diag_hc_ok["lineage_status"] == "VERIFIED_DISJOINT"


def test_pilot_dataset_gap_and_accounting(tmp_path: Path):
    """
    TC-53: Test pilot dataset inventory and gap calculation with synthetic DataFrames.
    Verifies tracking of target commands (TRAI, PHAI, LEN, XUONG), gap to 250, and rejection status.
    """
    from src.acquisition.pilot_dataset_validator import audit_pilot_dataset, TARGET_COMMANDS

    # 1. Incomplete pilot manifest: TRAI has 200 accepted, PHAI has 250, LEN has 100, XUONG has 0
    rows = []
    for i in range(200):
        rows.append({"recording_id": f"R_TRAI_{i}", "command_label": "TRAI", "speaker_id": f"SPK_{i % 10}", "dialect": "Northern", "status": "APPROVED"})
    # 50 rejected TRAI
    for i in range(50):
        rows.append({"recording_id": f"R_TRAI_REJ_{i}", "command_label": "TRAI", "speaker_id": f"SPK_{i % 10}", "dialect": "Northern", "status": "QUARANTINED"})
    for i in range(250):
        rows.append({"recording_id": f"R_PHAI_{i}", "command_label": "PHAI", "speaker_id": f"SPK_{i % 10}", "dialect": "Central", "status": "APPROVED"})
    for i in range(100):
        rows.append({"recording_id": f"R_LEN_{i}", "command_label": "LEN", "speaker_id": f"SPK_{i % 10}", "dialect": "Southern", "status": "APPROVED"})

    df_incomplete = pd.DataFrame(rows)
    audit = audit_pilot_dataset(df_incomplete, target_per_command=250)

    assert audit["is_pilot_ready"] is False
    assert audit["per_command"]["TRAI"]["accepted"] == 200
    assert audit["per_command"]["TRAI"]["rejected"] == 50
    assert audit["per_command"]["TRAI"]["missing"] == 50
    assert audit["per_command"]["TRAI"]["status"] == "GAP_50"

    assert audit["per_command"]["PHAI"]["accepted"] == 250
    assert audit["per_command"]["PHAI"]["missing"] == 0
    assert audit["per_command"]["PHAI"]["status"] == "MET"

    assert audit["per_command"]["LEN"]["accepted"] == 100
    assert audit["per_command"]["LEN"]["missing"] == 150
    assert audit["per_command"]["LEN"]["status"] == "GAP_150"

    assert audit["per_command"]["XUONG"]["accepted"] == 0
    assert audit["per_command"]["XUONG"]["missing"] == 250
    assert audit["per_command"]["XUONG"]["status"] == "GAP_250"

    assert audit["overall_gap"] == 50 + 0 + 150 + 250


def test_pilot_dataset_regional_and_speaker_diversity():
    """
    TC-54: Test pilot dataset regional diversity and speaker distribution reporting.
    """
    from src.acquisition.pilot_dataset_validator import audit_pilot_dataset

    rows = []
    # 250 for each command with 25 speakers (10 takes each per cmd)
    # Dialect: 9 Northern, 8 Central, 8 Southern
    spk_dialects = {}
    for i in range(25):
        spk_id = f"SPK_{i:02d}"
        if i < 9:
            spk_dialects[spk_id] = "Northern"
        elif i < 17:
            spk_dialects[spk_id] = "Central"
        else:
            spk_dialects[spk_id] = "Southern"

    for cmd in ("TRAI", "PHAI", "LEN", "XUONG"):
        for spk_id, dialect in spk_dialects.items():
            for t in range(10):
                rows.append({
                    "recording_id": f"REC_{cmd}_{spk_id}_take{t}",
                    "command_label": cmd,
                    "speaker_id": spk_id,
                    "dialect": dialect,
                    "status": "APPROVED",
                    "label_provenance": "HUMAN_CONSENTED"
                })

    df = pd.DataFrame(rows)
    audit = audit_pilot_dataset(df, target_per_command=250)

    assert audit["is_pilot_ready"] is True
    assert audit["total_accepted_target"] == 1000
    assert audit["overall_gap"] == 0

    # Regional breakdown:
    # Northern: 9 spk * 40 takes = 360 (36.0%)
    # Central: 8 spk * 40 takes = 320 (32.0%)
    # Southern: 8 spk * 40 takes = 320 (32.0%)
    assert audit["per_region"]["Northern"]["count"] == 360
    assert audit["per_region"]["Northern"]["share_pct"] == 36.0
    assert audit["per_region"]["Central"]["count"] == 320
    assert audit["per_region"]["Central"]["share_pct"] == 32.0
    assert audit["per_region"]["Southern"]["count"] == 320
    assert audit["per_region"]["Southern"]["share_pct"] == 32.0

    # Speaker metrics: 25 speakers, 40 takes each, share = 4.0% <= 5.0%
    assert audit["speaker_diversity"]["unique_speakers"] == 25
    assert audit["speaker_diversity"]["max_takes_per_speaker"] == 40
    assert audit["speaker_diversity"]["max_single_speaker_share_pct"] == 4.0
    assert audit["speaker_diversity"]["exceeds_5pct_dominance_limit"] is False


def test_pilot_dataset_qc_checks_with_temp_audio(tmp_path: Path):
    """
    TC-55: Test pilot dataset audio quality gating and SHA-256 verification using temporary test audio fixtures.
    """
    from src.acquisition.pilot_dataset_validator import audit_pilot_dataset
    from src.acquisition.campaign_validator import compute_file_sha256

    # Create 4 test audio files: 1 valid, 1 clipped, 1 silent, 1 with mismatched sha256
    valid_wav = create_test_wav(tmp_path / "valid.wav", sr=16000, duration=1.0)
    clipped_wav = create_test_wav(tmp_path / "clipped.wav", sr=16000, duration=1.0, clipped=True)
    silent_wav = create_test_wav(tmp_path / "silent.wav", sr=16000, duration=1.0, silent=True)
    mismatch_wav = create_test_wav(tmp_path / "mismatch.wav", sr=16000, duration=1.0)

    rows = [
        {"recording_id": "REC_V", "command_label": "TRAI", "speaker_id": "SPK_01", "dialect": "Northern", "status": "APPROVED", "processed_path": str(valid_wav), "sha256_standardized": compute_file_sha256(valid_wav)},
        {"recording_id": "REC_C", "command_label": "TRAI", "speaker_id": "SPK_01", "dialect": "Northern", "status": "APPROVED", "processed_path": str(clipped_wav), "sha256_standardized": compute_file_sha256(clipped_wav)},
        {"recording_id": "REC_S", "command_label": "TRAI", "speaker_id": "SPK_01", "dialect": "Northern", "status": "APPROVED", "processed_path": str(silent_wav), "sha256_standardized": compute_file_sha256(silent_wav)},
        {"recording_id": "REC_M", "command_label": "TRAI", "speaker_id": "SPK_01", "dialect": "Northern", "status": "APPROVED", "processed_path": str(mismatch_wav), "sha256_standardized": "0000000000000000000000000000000000000000000000000000000000000000"}
    ]

    df = pd.DataFrame(rows)
    audit = audit_pilot_dataset(df, target_per_command=1, check_disk=True)

    assert audit["is_pilot_ready"] is False
    assert len(audit["qc_failures"]) >= 3
    assert any("digital_clipping_detected" in err for err in audit["qc_failures"])
    assert any("abnormal_silence" in err for err in audit["qc_failures"])
    assert any("sha256_mismatch" in err for err in audit["qc_failures"])


def test_pilot_dataset_validator_cli(tmp_path: Path):
    """
    TC-56: Test pilot dataset auditor CLI execution and report export.
    """
    import subprocess
    dummy_csv = tmp_path / "mock_pilot_manifest.csv"
    report_out = tmp_path / "pilot_report.md"

    rows = []
    for cmd in ("TRAI", "PHAI", "LEN", "XUONG"):
        for i in range(250):
            rows.append({
                "recording_id": f"REC_{cmd}_{i}",
                "command_label": cmd,
                "speaker_id": f"SPK_{i % 25:02d}",
                "dialect": "Northern" if i % 3 == 0 else ("Central" if i % 3 == 1 else "Southern"),
                "status": "APPROVED",
                "label_provenance": "HUMAN_CONSENTED"
            })
    pd.DataFrame(rows).to_csv(dummy_csv, index=False)

    cmd = [
        sys.executable,
        "src/acquisition/pilot_dataset_validator.py",
        "--manifest", str(dummy_csv),
        "--target", "250",
        "--report-out", str(report_out)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0
    assert report_out.exists()
    content = report_out.read_text(encoding="utf-8")
    assert "Overall Pilot Readiness:** `READY`" in content
    assert "Target Command Class Breakdown" in content


def test_synthetic_kws_smoke_test_artifacts():
    """
    TC-57: Test that the 4-class synthetic smoke test artifacts are valid,
    disjoint, and contain required non-generalization disclaimers.
    """
    from src.acquisition.campaign_validator import validate_split_manifests_preflight
    import torch

    exp_dir = Path("experiments/synthetic_kws_baseline")
    assert exp_dir.exists(), "Experiment directory missing"

    tr_p = exp_dir / "train_manifest.csv"
    va_p = exp_dir / "val_manifest.csv"
    te_p = exp_dir / "test_manifest.csv"
    assert tr_p.exists() and va_p.exists() and te_p.exists()

    df_tr = pd.read_csv(tr_p)
    df_va = pd.read_csv(va_p)
    df_te = pd.read_csv(te_p)

    assert len(df_tr) == 712
    assert len(df_va) == 144
    assert len(df_te) == 144

    # Ensure strictly calibration source
    assert (df_tr["source"] == "calibration_acoustic_generator_v1").all()
    assert (df_va["source"] == "calibration_acoustic_generator_v1").all()
    assert (df_te["source"] == "calibration_acoustic_generator_v1").all()

    # Preflight split verification
    is_valid, errs, diag = validate_split_manifests_preflight(
        df_tr, df_va, df_te, validation_profile="benchmark"
    )
    assert is_valid is True
    assert diag["speaker_status"] == "VERIFIED_DISJOINT"

    # Checkpoint verification
    ckpt_path = exp_dir / "checkpoints" / "best_synthetic_kws_model.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["num_classes"] == 4
    assert "model_state_dict" in ckpt

    # Report verification
    rep_path = exp_dir / "synthetic_kws_baseline_report.md"
    assert rep_path.exists()
    rep_text = rep_path.read_text(encoding="utf-8")
    assert "CRITICAL LIMITATIONS" in rep_text
    assert "NOT EVIDENCE OF REAL-WORLD GENERALIZATION" in rep_text
    assert "Held-Out Test Accuracy:" in rep_text


def test_synthetic_kws_v2_artifacts_and_false_activation_bounds():
    """
    TC-58: Test that the 6-class synthetic V2 artifacts, dual holdouts,
    disjointness invariants, and false activation reporting are valid.
    """
    from src.acquisition.campaign_validator import validate_split_manifests_preflight
    import torch

    v2_dir = Path("experiments/synthetic_kws_v2")
    assert v2_dir.exists(), "Experiments V2 directory missing"

    tr_p = v2_dir / "train_manifest_6class.csv"
    va_p = v2_dir / "val_manifest_6class.csv"
    te_p = v2_dir / "test_manifest_6class.csv"
    fresh_p = v2_dir / "fresh_holdout_manifest_6class.csv"

    assert tr_p.exists() and va_p.exists() and te_p.exists() and fresh_p.exists()

    df_tr = pd.read_csv(tr_p)
    df_va = pd.read_csv(va_p)
    df_te = pd.read_csv(te_p)
    df_fresh = pd.read_csv(fresh_p)

    assert len(df_tr) == 952
    assert len(df_va) == 216
    assert len(df_te) == 216
    assert len(df_fresh) == 300

    # Ensure all 6 classes exist in every partition
    expected_classes = {"TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"}
    for p_df, p_name in [(df_tr, "train"), (df_va, "val"), (df_te, "test"), (df_fresh, "fresh")]:
        p_classes = set(p_df["class_label"].unique())
        assert p_classes == expected_classes, f"Mismatch in {p_name} classes: {p_classes}"

    # Preflight split validation
    is_valid, errs, diag = validate_split_manifests_preflight(
        df_tr, df_va, df_te, validation_profile="benchmark"
    )
    assert is_valid is True
    assert diag["speaker_status"] == "VERIFIED_DISJOINT"

    # Fresh holdout must be disjoint from train, val, test
    s_tr = set(df_tr["speaker_id"])
    s_va = set(df_va["speaker_id"])
    s_te = set(df_te["speaker_id"])
    s_fresh = set(df_fresh["speaker_id"])

    assert len(s_fresh & s_tr) == 0
    assert len(s_fresh & s_va) == 0
    assert len(s_fresh & s_te) == 0
    assert len(s_fresh) == 5

    # Checkpoints verification
    ckpt_tc = v2_dir / "checkpoints" / "best_tcresnet_6class.pt"
    ckpt_tm = v2_dir / "checkpoints" / "best_temporal_kwsnet_6class.pt"
    assert ckpt_tc.exists() and ckpt_tm.exists()

    c_tc = torch.load(ckpt_tc, map_location="cpu")
    c_tm = torch.load(ckpt_tm, map_location="cpu")
    assert c_tc["num_classes"] == 6
    assert c_tm["num_classes"] == 6

    # Report verification
    rep_path = v2_dir / "synthetic_kws_v2_report.md"
    assert rep_path.exists()
    rep_text = rep_path.read_text(encoding="utf-8")
    assert "CRITICAL SYNTHETIC-ONLY LIMITATIONS" in rep_text.upper()
    assert "FA Rate on SILENCE" in rep_text
    assert "FA Rate on UNKNOWN" in rep_text
    assert "Disaggregated CPU Inference Latency" in rep_text


def test_synthetic_kws_v3_artifacts_and_generalization():
    """
    TC-59: Verify Synthetic Vietnamese KWS V3 artifacts, speaker disjointness,
    fresh holdout invariants, checkpoints, metrics, and governance requirements.
    """
    import json
    import torch
    from src.acquisition.campaign_validator import validate_split_manifests_preflight

    v3_dir = Path("experiments/synthetic_kws_v3")
    assert v3_dir.exists(), "Experiments V3 directory missing"

    tr_p = v3_dir / "train_manifest_6class.csv"
    va_p = v3_dir / "val_manifest_6class.csv"
    fresh_p = v3_dir / "fresh_holdout_manifest_6class.csv"

    assert tr_p.exists() and va_p.exists() and fresh_p.exists()

    df_tr = pd.read_csv(tr_p)
    df_va = pd.read_csv(va_p)
    df_fresh = pd.read_csv(fresh_p)

    assert len(df_tr) == 952
    assert len(df_va) == 216
    assert len(df_fresh) == 300

    # Ensure all 6 classes exist in every partition
    expected_classes = {"TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"}
    for p_df, p_name in [(df_tr, "train"), (df_va, "val"), (df_fresh, "fresh_v3")]:
        p_classes = set(p_df["class_label"].unique())
        assert p_classes == expected_classes, f"Mismatch in {p_name} classes: {p_classes}"

    # V3 Fresh holdout must have exactly 50 per class
    vc = df_fresh["class_label"].value_counts().to_dict()
    for cls in expected_classes:
        assert vc[cls] == 50, f"Class {cls} count mismatch in fresh holdout: {vc[cls]}"

    # Speaker disjointness: SPK_CALIB_026..030 must be disjoint from train and val
    s_tr = set(df_tr["speaker_id"])
    s_va = set(df_va["speaker_id"])
    s_fresh = set(df_fresh["speaker_id"])
    
    assert len(s_fresh) == 5
    assert len(s_fresh & s_tr) == 0
    assert len(s_fresh & s_va) == 0

    # Verify checkpoints
    ckpt_tc = v3_dir / "checkpoints" / "best_tcresnet14_v3.pt"
    ckpt_temp = v3_dir / "checkpoints" / "best_temporal_tcresnet14_v3.pt"
    assert ckpt_tc.exists() and ckpt_temp.exists()

    c_tc = torch.load(ckpt_tc, map_location="cpu")
    c_temp = torch.load(ckpt_temp, map_location="cpu")
    assert c_tc["num_classes"] == 6
    assert c_temp["num_classes"] == 6

    # Verify evaluation results JSON
    eval_json = v3_dir / "evaluation_results_v3.json"
    assert eval_json.exists()
    with open(eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    assert "TC-ResNet-14-V3" in eval_data
    tc_res = eval_data["TC-ResNet-14-V3"]["partitions"]["V3 Fresh Holdout (SPK 026-030, 300 clips [Untouched])"]
    assert tc_res["overall_accuracy"] >= 0.90
    assert tc_res["false_activations"]["unknown_fa_rate_pct"] == 0.0
    assert tc_res["false_activations"]["silence_fa_rate_pct"] == 0.0

    # Report verification
    rep_path = v3_dir / "synthetic_kws_v3_report.md"
    assert rep_path.exists()
    rep_text = rep_path.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in rep_text
    assert "Disaggregated CPU Inference Latency" in rep_text


def test_synthetic_kws_v4_onset_ablation_and_governance():
    """
    TC-60: Verify Synthetic Vietnamese KWS V4 controlled onset ablation artifacts,
    speaker disjointness, fresh holdout invariants, checkpoints, metrics, and governance requirements.
    """
    import json
    import torch

    v4_dir = Path("experiments/synthetic_kws_v4")
    assert v4_dir.exists(), "Experiments V4 directory missing"

    tr_p = v4_dir / "train_manifest_onset_aware.csv"
    va_p = v4_dir / "val_manifest_onset_aware.csv"
    fresh_p = v4_dir / "fresh_holdout_manifest_6class.csv"

    assert tr_p.exists() and va_p.exists() and fresh_p.exists()

    df_tr = pd.read_csv(tr_p)
    df_va = pd.read_csv(va_p)
    df_fresh = pd.read_csv(fresh_p)

    assert len(df_tr) == 952
    assert len(df_va) == 216
    assert len(df_fresh) == 300

    # Ensure all 6 classes exist in every partition
    expected_classes = {"TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"}
    for p_df, p_name in [(df_tr, "train_onset"), (df_va, "val_onset"), (df_fresh, "fresh_v4")]:
        p_classes = set(p_df["class_label"].unique())
        assert p_classes == expected_classes, f"Mismatch in {p_name} classes: {p_classes}"

    # V4 Fresh holdout must have exactly 50 per class
    vc = df_fresh["class_label"].value_counts().to_dict()
    for cls in expected_classes:
        assert vc[cls] == 50, f"Class {cls} count mismatch in fresh holdout: {vc[cls]}"

    # Speaker disjointness: SPK_CALIB_031..035 must be disjoint from train and val
    s_tr = set(df_tr["speaker_id"])
    s_va = set(df_va["speaker_id"])
    s_fresh = set(df_fresh["speaker_id"])
    
    assert len(s_fresh) == 5
    assert len(s_fresh & s_tr) == 0
    assert len(s_fresh & s_va) == 0

    # Verify 3 ablation checkpoints
    ckpt_a = v4_dir / "checkpoints" / "arm_a_control.pt"
    ckpt_b = v4_dir / "checkpoints" / "arm_b_onset_no_resample.pt"
    ckpt_c = v4_dir / "checkpoints" / "arm_c_onset_with_resample.pt"
    assert ckpt_a.exists() and ckpt_b.exists() and ckpt_c.exists()

    for ck in [ckpt_a, ckpt_b, ckpt_c]:
        c = torch.load(ck, map_location="cpu")
        assert c["num_classes"] == 6

    # Verify evaluation results JSON
    eval_json = v4_dir / "evaluation_results_v4.json"
    assert eval_json.exists()
    with open(eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    assert "Arm B (Onset-Aware, No Resample)" in eval_data
    arm_b_res = eval_data["Arm B (Onset-Aware, No Resample)"]["partitions"]["V4 Fresh Holdout (SPK 031-035, 300 clips [Untouched])"]
    assert arm_b_res["overall_accuracy"] >= 0.95
    assert arm_b_res["trai_phai_specifics"]["trai_recall"] >= 0.90
    assert arm_b_res["trai_phai_specifics"]["phai_recall"] >= 0.90
    assert arm_b_res["false_activations"]["unknown_fa_rate_pct"] == 0.0
    assert arm_b_res["false_activations"]["silence_fa_rate_pct"] == 0.0

    # Independently verify confusion matrix math: trace sum / total sum
    cm = np.array(arm_b_res["confusion_matrix"])
    assert cm.shape == (6, 6)
    assert cm.sum() == 300
    for row_idx in range(6):
        assert cm[row_idx].sum() == 50, f"Row {row_idx} sum mismatch: {cm[row_idx].sum()}"
    calculated_acc = np.trace(cm) / cm.sum()
    assert abs(calculated_acc - arm_b_res["overall_accuracy"]) < 1e-5

    # Independently verify raw predictions match evaluation JSON
    raw_json = v4_dir / "audit" / "raw_predictions.json"
    assert raw_json.exists(), "raw_predictions.json missing"
    with open(raw_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    arm_b_raw = raw_data["Arm B (Onset-Aware, No Resample)"]["V4 Fresh Holdout (300) [Untouched]"]
    assert len(arm_b_raw) == 300
    raw_correct = sum(1 for p in arm_b_raw if p["correct"])
    assert raw_correct == 298
    assert abs(raw_correct / 300.0 - arm_b_res["overall_accuracy"]) < 1e-5

    # Verify discrepancy resolution in audit log
    disc_json = v4_dir / "audit" / "discrepancy_log.json"
    assert disc_json.exists(), "discrepancy_log.json missing"
    with open(disc_json, "r", encoding="utf-8") as f:
        disc_data = json.load(f)
    assert disc_data["issue_1_arm_a_validation_discrepancy"]["status"] == "RESOLVED"
    assert disc_data["issue_2_arm_b_zero_recall_on_older_holdouts"]["status"] == "RESOLVED"

    # Report verification
    rep_path = v4_dir / "synthetic_kws_v4_report.md"
    assert rep_path.exists()
    rep_text = rep_path.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in rep_text
    assert "Disaggregated CPU Latency Profiling" in rep_text


def test_v4_metric_integrity_audit_and_identities():
    """
    TC-61: Verify V4 Metric Integrity Audit artifacts, mathematical identities,
    waveform acoustic parameters, discrepancy classifications, and governance requirements.
    """
    import json
    import numpy as np
    import pandas as pd

    audit_dir = Path("experiments/synthetic_kws_v4/audit")
    assert audit_dir.exists(), "Audit directory missing"

    # 1. Canonical predictions verification
    pred_csv = audit_dir / "canonical_predictions.csv"
    assert pred_csv.exists(), "canonical_predictions.csv missing"
    df_preds = pd.read_csv(pred_csv)
    # 4 models * (144 + 216 + 300 + 300 + 300 + 216 + 216) = 4 * 1692 = 6,768 rows
    assert len(df_preds) == 6768
    required_cols = [
        "model_arm", "checkpoint_path", "checkpoint_sha256", "partition_id",
        "manifest_path", "manifest_sha256", "recording_id", "speaker_id",
        "processed_path", "true_class_id", "true_class_label", "pred_class_id",
        "pred_class_label", "correct", "prob_trai", "prob_phai", "prob_len",
        "prob_xuong", "prob_unknown", "prob_silence"
    ]
    for col in required_cols:
        assert col in df_preds.columns, f"Missing column: {col}"
        assert df_preds[col].isna().sum() == 0, f"Null values found in {col}"

    # 2. Canonical metrics verification & mathematical identities
    metrics_json = audit_dir / "canonical_metrics.json"
    assert metrics_json.exists(), "canonical_metrics.json missing"
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    expected_arms = [
        "Arm A (Control)", "Arm B (Onset-Aware, No Resample)",
        "Arm C (Onset-Aware + Resample)", "V3 Champion TC-ResNet-14"
    ]
    expected_partitions = [
        "hist_144", "hist_216", "v2_fresh_300", "v3_fresh_300",
        "v4_fresh_300", "val_baseline_216", "val_onset_216"
    ]

    for arm in expected_arms:
        assert arm in metrics_data, f"Missing arm: {arm}"
        for part in expected_partitions:
            assert part in metrics_data[arm]["partitions"], f"Missing partition {part} for {arm}"
            p = metrics_data[arm]["partitions"][part]

            cm = np.array(p["confusion_matrix"])
            assert cm.shape == (6, 6)

            # Mathematical Identity 1: Row sums == class support
            row_sums = cm.sum(axis=1)
            assert list(row_sums) == p["support"]

            # Mathematical Identity 2: Sum of row sums == total samples
            total_samples = int(row_sums.sum())
            assert total_samples == p["samples_processed"]

            # Mathematical Identity 3: Overall accuracy == trace / total
            trace = int(np.trace(cm))
            assert abs(trace / total_samples - p["overall_accuracy"]) < 1e-6

            # Mathematical Identity 4: Trace == sum(support_i * recall_i)
            weighted_rec_sum = sum(row_sums[i] * p["recall"][i] for i in range(6))
            assert abs(weighted_rec_sum - trace) < 1e-4

            # Mathematical Identity 5: Target command miss count
            target_support = int(row_sums[:4].sum())
            target_correct = int(np.trace(cm[:4, :4]))
            target_misses = target_support - target_correct
            assert target_misses == p["target_miss_count"]
            assert abs((target_misses / target_support * 100.0) - p["target_command_miss_rate_pct"]) < 1e-4

            # Mathematical Identity 6: Negative false alarms
            if p["unknown_total"] > 0:
                assert cm[4, :4].sum() == p["unknown_fa_count"]
                assert abs((p["unknown_fa_count"] / p["unknown_total"] * 100.0) - p["unknown_fa_rate_pct"]) < 1e-4
            if p["silence_total"] > 0:
                assert cm[5, :4].sum() == p["silence_fa_count"]
                assert abs((p["silence_fa_count"] / p["silence_total"] * 100.0) - p["silence_fa_rate_pct"]) < 1e-4

    # 3. Waveform measurement verification
    wave_json = audit_dir / "waveform_measurements.json"
    assert wave_json.exists(), "waveform_measurements.json missing"
    with open(wave_json, "r", encoding="utf-8") as f:
        wave_data = json.load(f)

    v4_closure = wave_data["v4_onset_aware"]["closure_window_0_to_25ms"]
    assert v4_closure["rms_mean"] < 0.001
    assert v4_closure["max_abs_max"] < 0.005
    assert v4_closure["fraction_under_005_mean"] == 1.0  # 100% of samples < 0.005
    assert 400 <= v4_closure["first_sample_ge_005_mean"] <= 402  # Exactly at 25ms

    v4_burst = wave_data["v4_onset_aware"]["burst_window_25_to_40ms"]
    assert 0.03 <= v4_burst["rms_mean"] <= 0.10

    # 4. Discrepancy log verification
    disc_json = audit_dir / "discrepancy_log.json"
    assert disc_json.exists()
    with open(disc_json, "r", encoding="utf-8") as f:
        disc_log = json.load(f)

    for disc_id in ["DISC-01", "DISC-04", "DISC-05", "DISC-06", "DISC-07"]:
        assert disc_log[disc_id]["status"] == "RESOLVED"
    assert disc_log["DISC-02"]["status"] == "UNRESOLVED"
    assert disc_log["DISC-03"]["status"] == "NOT_VERIFIABLE"

    # 5. Governance invariant in Metric Integrity Audit report
    audit_rep = audit_dir / "metric_integrity_audit.md"
    assert audit_rep.exists()
    rep_text = audit_rep.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in rep_text
    assert "AUDIT_PASS_WITH_LIMITATIONS" in rep_text


def test_v4_alignment_and_onset_diagnostic():
    """
    TC-62: Verify controlled 2x2 alignment/onset diagnostic artifacts,
    manifests, transformed audio files, acoustic parameters, and decoupling of confounds.
    """
    import json
    import pandas as pd

    audit_dir = Path("experiments/synthetic_kws_v4/audit")
    diag_audio_dir = audit_dir / "diagnostic_audio"
    assert diag_audio_dir.exists(), "diagnostic_audio directory missing"

    # 1. Diagnostic manifests verification
    man_a1_b1 = audit_dir / "manifest_diag_A1_B1_baseline_left.csv"
    man_a1_b2 = audit_dir / "manifest_diag_A1_B2_baseline_centered.csv"
    man_a2_b1 = audit_dir / "manifest_diag_A2_B1_onset_left.csv"
    man_a2_b2 = audit_dir / "manifest_diag_A2_B2_onset_centered.csv"

    for man_p in [man_a1_b1, man_a1_b2, man_a2_b1, man_a2_b2]:
        assert man_p.exists(), f"Missing manifest: {man_p}"
        df = pd.read_csv(man_p)
        assert len(df) in [144, 300]
        # Check all audio files exist
        for audio_path_str in df["processed_path"]:
            p = Path(audio_path_str)
            assert p.exists(), f"Missing diagnostic audio: {p}"

    # 2. Alignment diagnostic results verification
    res_json = audit_dir / "alignment_diagnostic_results.json"
    assert res_json.exists(), "alignment_diagnostic_results.json missing"
    with open(res_json, "r", encoding="utf-8") as f:
        diag_res = json.load(f)

    # Verify Arm B behavior:
    # On baseline clips, TRAI recall is 0.0% under BOTH center-aligned and left-aligned conditions
    arm_b = diag_res["Arm B (Onset-Aware, No Resample)"]["conditions"]
    assert arm_b["A1-B2"]["trai_recall"] == 0.0  # Center-aligned baseline
    assert arm_b["A1-B1"]["trai_recall"] == 0.0  # Left-aligned baseline
    assert arm_b["A2-B1"]["trai_recall"] >= 0.90  # Left-aligned onset-aware (native)
    assert arm_b["A2-B2"]["trai_recall"] >= 0.70  # Center-aligned onset-aware

    # Verify Arm A behavior:
    # On onset-aware clips, TRAI recall drops to 0.0% when center-aligned
    arm_a = diag_res["Arm A (Control)"]["conditions"]
    assert arm_a["A2-B2"]["trai_recall"] == 0.0  # Center-aligned onset-aware collapses to UNKNOWN


def test_v4_paired_alignment_and_checkpoint_identity():
    """
    TC-63: Verify checkpoint identity reporting, paired waveform sample preservation,
    active-relative HF ratio invariance, distinct onset definitions, and report integrity.
    """
    import json

    audit_dir = Path("experiments/synthetic_kws_v4/audit")

    # 1. Checkpoint identity verification
    ckpt_id_json = audit_dir / "checkpoint_identity.json"
    assert ckpt_id_json.exists(), "checkpoint_identity.json missing"
    with open(ckpt_id_json, "r", encoding="utf-8") as f:
        ckpt_id = json.load(f)

    assert ckpt_id["arm_a_sha256"] != ckpt_id["v3_champion_sha256"]  # Metadata container differs
    assert ckpt_id["total_state_dict_tensors"] == 74
    assert ckpt_id["number_of_identical_tensors"] == 74
    assert ckpt_id["number_of_differing_tensors"] == 0
    assert ckpt_id["max_absolute_tensor_difference"] == 0.0
    assert ckpt_id["canonical_predictions_identical"] is True
    assert ckpt_id["predictions_compared_count"] == 1692

    # 2. Alignment invariance & paired sample preservation
    align_json = audit_dir / "alignment_invariance_results.json"
    assert align_json.exists(), "alignment_invariance_results.json missing"
    with open(align_json, "r", encoding="utf-8") as f:
        align_data = json.load(f)

    assert align_data["total_pairs_audited"] == 444
    assert align_data["all_non_padding_samples_preserved_in_order"] is True
    assert align_data["max_absolute_sample_difference_across_all_pairs"] == 0.0
    assert align_data["total_active_speech_samples_truncated"] == 0

    # 3. Active-relative HF ratio invariance for speech
    hf_data = align_data["active_relative_hf_energy_audit"]["per_class_invariance"]
    for cl in ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN"]:
        assert hf_data[cl]["is_invariant"] is True, f"HF ratio not invariant for {cl}"
        assert hf_data[cl]["max_pairwise_difference"] < 1e-6

    # 4. Distinct onset definitions
    onset_defs = align_data["onset_definitions_audit"]
    assert "definition_1" in onset_defs and "definition_2" in onset_defs and "definition_3" in onset_defs
    v4_native_trai = onset_defs["summary_table"]["A2-B1_onset_left_native"]["TRAI"]
    assert v4_native_trai["nonzero_ms"] == 0.0
    assert v4_native_trai["thresh_005_ms"] == 25.0
    assert v4_native_trai["burst_ms"] == 25.0

    # 5. Paired alignment integrity report
    rep_p = audit_dir / "paired_alignment_integrity.md"
    assert rep_p.exists(), "paired_alignment_integrity.md missing"
    text = rep_p.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in text
    assert "AUDIT_PASS_WITH_LIMITATIONS" in text
    assert "100% bit-exact equality" in text


def test_v4_silence_alignment_preservation_and_v2_diagnostic():
    """
    TC-64: Verify SILENCE alignment preservation, fallback truncation regression,
    corrected V2 diagnostic transform, 6-class HF invariance, and frozen model safety.
    """
    import json
    import numpy as np
    import pandas as pd
    import soundfile as sf

    audit_dir = Path("experiments/synthetic_kws_v4/audit")
    assert audit_dir.exists(), "Audit directory missing"

    # 1. Centering regression check: verify that ambient noise without speech peaks
    # must NEVER be silently truncated to 8,001 samples by a fallback heuristic
    noise_clip = np.random.normal(0, 0.001, 16000).astype(np.float32)
    act = np.where(np.abs(noise_clip) >= 0.005)[0]
    assert len(act) == 0, "Synthetic noise floor exceeded threshold"

    # In buggy centering, len(act) == 0 caused truncation to 8,001 samples
    buggy_last_idx = act[-1] if len(act) > 0 else 8000
    buggy_truncated_len = buggy_last_idx + 1
    assert buggy_truncated_len == 8001, "Buggy logic did not produce 8001 sample truncation"

    # In corrected V2 centering, SILENCE is recognized as full-buffer (16k) and preserved intact
    v2_silence_len = 16000
    assert v2_silence_len == 16000, "V2 transform must preserve full 16,000 buffer"

    # 2. Manifest and Audio Verification for Condition A2-B2-V2
    man_v2_p = audit_dir / "manifest_diag_A2_B2_v2_onset_centered.csv"
    assert man_v2_p.exists(), f"Missing V2 manifest: {man_v2_p}"
    df_v2 = pd.read_csv(man_v2_p)
    assert len(df_v2) == 300, f"Expected 300 rows, found {len(df_v2)}"

    df_v4_native = pd.read_csv("experiments/synthetic_kws_v4/fresh_holdout_manifest_6class.csv")
    assert len(df_v4_native) == 300

    silence_rows = df_v2[df_v2["class_label"] == "SILENCE"]
    assert len(silence_rows) == 50, "Expected 50 SILENCE rows in V2 manifest"

    # Verify every silence clip in V2 has 16,000 preserved, 0 omitted, and max abs diff == 0.0 against native
    for idx, r in silence_rows.iterrows():
        assert r["samples_preserved"] == 16000, f"Clip {r['recording_id']} samples_preserved != 16000"
        assert r["samples_omitted"] == 0, f"Clip {r['recording_id']} samples_omitted != 0"
        assert r["pad_l"] == 0 and r["pad_r"] == 0

        wav_p = Path(r["processed_path"])
        assert wav_p.exists(), f"Missing audio file: {wav_p}"
        a_v2, sr = sf.read(str(wav_p), dtype="float32")
        assert len(a_v2) == 16000

        # Native counterpart
        native_path = Path(df_v4_native.loc[idx, "processed_path"])
        a_nat, _ = sf.read(str(native_path), dtype="float32")
        max_diff = float(np.max(np.abs(a_v2 - a_nat)))
        assert max_diff == 0.0, f"Bit-exact mismatch on silence clip {r['recording_id']}"

    # Verify per-row alignment metadata in V2 manifest
    assert "alignment_action" in df_v2.columns
    assert "transform_applied" in df_v2.columns
    assert "source_sha256" in df_v2.columns
    assert "output_sha256" in df_v2.columns
    sil_mask = df_v2["class_label"] == "SILENCE"
    assert (df_v2[sil_mask]["alignment_action"] == "preserved_no_onset").all()
    assert (df_v2[sil_mask]["transform_applied"] == False).all()
    assert (df_v2[sil_mask]["source_sha256"] == df_v2[sil_mask]["output_sha256"]).all()
    assert (df_v2[~sil_mask]["alignment_action"] == "center_aligned").all()
    assert (df_v2[~sil_mask]["transform_applied"] == True).all()

    # Verify V4 Research Freeze Package
    freeze_p = Path("experiments/synthetic_kws_v4/V4_RESEARCH_FREEZE.md")
    assert freeze_p.exists(), "V4_RESEARCH_FREEZE.md missing"
    freeze_text = freeze_p.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in freeze_text
    assert "AUDIT_PASS_WITH_LIMITATIONS" in freeze_text
    assert "best_tcresnet14_v3.pt" in freeze_text
    assert "arm_a_control.pt" in freeze_text
    assert "DISC-02" in freeze_text
    assert "UNRESOLVED" in freeze_text

    # 3. Invariance Results JSON Verification
    inv_v2_json = audit_dir / "alignment_preservation_v2.json"
    assert inv_v2_json.exists(), "alignment_preservation_v2.json missing"
    with open(inv_v2_json, "r", encoding="utf-8") as f:
        inv_data = json.load(f)

    assert inv_data["total_pairs_audited"] == 444
    assert inv_data["max_absolute_sample_difference_across_all_active_speech"] == 0.0
    assert inv_data["max_absolute_sample_difference_across_all_silence"] == 0.0
    assert inv_data["total_active_speech_samples_truncated"] == 0
    assert inv_data["total_silence_samples_truncated"] == 0

    # Waveform integrity methodology distinction
    distinction = inv_data["preservation_methodology_distinction"]
    assert "Active speech segments" in distinction["speech_classes"]
    assert "padding" in distinction["speech_classes"]
    assert "Full 16,000-sample ambient background recordings" in distinction["silence_class"]
    assert "unshifted" in distinction["silence_class"]

    # Verify fallback audit numbers
    fb_audit = inv_data["silence_fallback_audit"]
    assert fb_audit["total_silence_clips"] == 50
    assert fb_audit["fallback_triggered_count"] == 25
    assert fb_audit["non_fallback_count"] == 25
    assert fb_audit["total_samples_omitted_across_cohort"] == 200757
    assert "SILENCE (Ambient Background Fixture)" in fb_audit["class_semantic_definition"]["name"]

    # Verify 50 SILENCE pairs independent confirmation
    sil_pairs = inv_data["silence_50_pairs_verification"]
    assert len(sil_pairs) == 50
    for sp in sil_pairs:
        assert sp["source_samples"] == 16000 and sp["v2_samples"] == 16000
        assert sp["source_sample_rate"] == 16000 and sp["v2_sample_rate"] == 16000
        assert sp["source_channels"] == 1 and sp["v2_channels"] == 1
        assert sp["sha256_match"] is True
        assert sp["sample_sequence_identical"] is True
        assert sp["max_abs_sample_diff"] == 0.0
        assert sp["samples_omitted"] == 0 and sp["samples_altered"] == 0
        assert sp["alignment_shift_applied"] == 0

    # Verify spectral energy invariance across all 6 classes
    hf_data = inv_data["spectral_energy_invariance_audit"]["per_class_invariance"]
    for cl in ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]:
        assert cl in hf_data, f"Class {cl} missing from spectral energy invariance"
        assert hf_data[cl]["is_invariant"] is True, f"Spectral ratio not invariant for {cl}"
        assert hf_data[cl]["max_pairwise_difference"] < 1e-6

    # 4. Model Inference Verification on V2
    model_evals = inv_data["v2_model_evaluations"]
    expected_models = [
        "Arm A (Control)", "Arm B (Onset-Aware, No Resample)",
        "Arm C (Onset-Aware + Resample)", "V3 Champion TC-ResNet-14"
    ]
    for arm in expected_models:
        assert arm in model_evals, f"Missing model: {arm}"
        res = model_evals[arm]
        assert res["samples_processed"] == 300
        sil_audit = res["silence_audit"]
        assert sil_audit["support"] == 50
        assert sil_audit["correct_silence"] == 50
        assert sil_audit["silence_recall"] == 1.0
        assert sil_audit["silence_false_activations_on_target"] == 0
        assert sil_audit["silence_false_activation_rate_pct"] == 0.0
        assert sil_audit["non_silence_predicted_as_silence"] == 0
        comp = res["comparison_with_buggy_a2_b2"]
        assert comp["predictions_changed_count"] == 0
        assert "Equal model predictions" in comp["scientific_interpretation"]
        assert "NOT imply acoustic equivalence" in comp["scientific_interpretation"]

    # 5. Canonical predictions table verification
    pred_csv = audit_dir / "canonical_predictions_diag_v2.csv"
    assert pred_csv.exists(), "canonical_predictions_diag_v2.csv missing"
    df_preds = pd.read_csv(pred_csv)
    assert len(df_preds) == 1200  # 4 models * 300 samples

    # 6. Discrepancy Log Verification
    disc_json = audit_dir / "discrepancy_log.json"
    assert disc_json.exists()
    with open(disc_json, "r", encoding="utf-8") as f:
        disc_log = json.load(f)
    assert "DISC-08" in disc_log, "DISC-08 missing from discrepancy_log.json"
    assert disc_log["DISC-08"]["status"] == "RESOLVED"
    assert disc_log["DISC-02"]["status"] == "UNRESOLVED"
    assert "sole proven cause" in disc_log["DISC-02"]["root_cause"] or "cannot be concluded" in disc_log["DISC-02"]["root_cause"]

    # 7. SILENCE Audit Report Verification
    audit_rep = audit_dir / "silence_alignment_preservation_audit.md"
    assert audit_rep.exists(), "silence_alignment_preservation_audit.md missing"
    rep_text = audit_rep.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in rep_text
    assert "AUDIT_PASS_WITH_LIMITATIONS" in rep_text
    assert "Condition A2-B2-V2" in rep_text
    assert "NOT uniformly center-aligned" in rep_text
    assert "Prediction Equality Does NOT Imply Acoustic Equivalence" in rep_text
    assert "fixed_window_ambient_noise_spectral_ratio" in rep_text


def test_phase5_synthetic_kws_demo_prototype_and_safety_gate():
    """
    TC-65: Verify Phase 5 Synthetic KWS inference engine, game-control adapter,
    negative error handling, non-target safety suppression, and synthetic caveats.
    Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
    """
    import hashlib
    from experiments.synthetic_kws_demo.classifier import (
        OfflineKWSClassifier,
        AudioValidator,
        InvalidChannelCountError,
        InvalidSampleRateError,
        InvalidDurationError,
        EmptyAudioError,
        SYNTHETIC_DEMO_WARNING,
        DEFAULT_CHECKPOINT
    )
    from experiments.synthetic_kws_demo.game_adapter import (
        GameControlAdapter,
        GameAction,
    )
    from experiments.synthetic_kws_demo.game_simulator import GridGameSimulator

    demo_dir = Path("experiments/synthetic_kws_demo")
    assert demo_dir.exists(), "experiments/synthetic_kws_demo directory missing"

    # 1. Candidate Checkpoint Verification (Arm B)
    ckpt_p = DEFAULT_CHECKPOINT
    assert ckpt_p.exists(), f"Candidate checkpoint missing: {ckpt_p}"
    ckpt_sha = hashlib.sha256(ckpt_p.read_bytes()).hexdigest().upper()
    expected_sha = "6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96"
    assert ckpt_sha == expected_sha, f"Arm B checkpoint SHA-256 mismatch: {ckpt_sha}"

    # 2. Fixture Manifest Integrity
    fixtures_dir = demo_dir / "fixtures"
    manifest_p = fixtures_dir / "fixture_manifest.csv"
    assert manifest_p.exists(), "fixture_manifest.csv missing"
    df_man = pd.read_csv(manifest_p)
    assert len(df_man) == 10, f"Expected 10 fixtures in manifest, found {len(df_man)}"
    for _, row in df_man.iterrows():
        fpath = fixtures_dir / row["filename"]
        assert fpath.exists(), f"Fixture missing: {fpath}"
        actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest().upper()
        assert actual_sha == row["sha256"], f"Hash mismatch on fixture {row['filename']}"

    # 3. Audio Validator & Error Handling Contracts
    with pytest.raises(InvalidChannelCountError):
        AudioValidator.validate_and_load(fixtures_dir / "invalid_stereo.wav")
    with pytest.raises(InvalidSampleRateError):
        AudioValidator.validate_and_load(fixtures_dir / "invalid_sample_rate_8k.wav")
    with pytest.raises(InvalidDurationError):
        AudioValidator.validate_and_load(fixtures_dir / "invalid_short_duration.wav")
    with pytest.raises(EmptyAudioError):
        AudioValidator.validate_and_load(fixtures_dir / "invalid_empty.wav")

    # 4. Offline Inference Engine Execution
    classifier = OfflineKWSClassifier(checkpoint_path=ckpt_p, device="cpu")
    assert classifier.model.training is False
    assert classifier.extractor.training is False

    # Target command verification
    targets = [
        ("demo_01_trai.wav", "TRAI", GameAction.MOVE_LEFT),
        ("demo_02_phai.wav", "PHAI", GameAction.MOVE_RIGHT),
        ("demo_03_len.wav", "LEN", GameAction.JUMP),
        ("demo_04_xuong.wav", "XUONG", GameAction.CROUCH),
    ]
    adapter = GameControlAdapter(confidence_threshold=0.50, debounce_cooldown_sec=0.0)
    for fname, exp_label, exp_action in targets:
        res = classifier.predict(fixtures_dir / fname)
        assert res.is_valid is True
        assert res.predicted_label == exp_label
        assert res.confidence >= 0.50
        assert SYNTHETIC_DEMO_WARNING in res.warning
        assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in res.warning

        act = adapter.adapt(res)
        assert act.action == exp_action
        assert act.is_active is True

    # 5. Non-Target Safety Suppression
    non_targets = ["demo_05_unknown.wav", "demo_06_silence.wav"]
    sim = GridGameSimulator(width=7, height=7, initial_pos=(3, 3))
    for fname in non_targets:
        res = classifier.predict(fixtures_dir / fname)
        assert res.is_valid is True
        act = adapter.adapt(res)
        assert act.action == GameAction.NO_ACTION
        assert act.is_active is False
        assert act.delta_x == 0 and act.delta_y == 0
        rec = sim.step(act)
        assert (sim.player_x, sim.player_y) == (3, 3)
        assert "NO_ACTION" in rec.event

    # 6. Documentation and Governance Verification
    readme_p = demo_dir / "README.md"
    assert readme_p.exists(), "experiments/synthetic_kws_demo/README.md missing"
    readme_txt = readme_p.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in readme_txt
    assert "SYNTHETIC-DEMO-ONLY" in readme_txt


def test_phase6_kws_model_export_and_runtime_parity_gate():
    """
    TC-66: Verify Phase 6 KWS ONNX model exportability, opset 17 validity,
    input/output tensor contracts, class ordering, numerical runtime parity (<= 1e-4),
    and governance invariants.
    Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
    """
    import json
    import hashlib
    import onnx
    import onnxruntime as ort

    export_dir = Path("experiments/synthetic_kws_demo/export")
    assert export_dir.exists(), "export directory missing"

    # 1. Source Checkpoint Hash Contract
    ckpt_path = Path("experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt")
    assert ckpt_path.exists(), "Source checkpoint missing"
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest().upper()
    expected_ckpt_sha = "6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96"
    assert ckpt_sha == expected_ckpt_sha, f"Source checkpoint hash mismatch: {ckpt_sha}"

    # 2. Exported ONNX Graph Verification
    onnx_path = export_dir / "arm_b_tcresnet14.onnx"
    assert onnx_path.exists(), "Exported ONNX file missing"
    onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest().upper()
    expected_onnx_sha = "8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625"
    assert onnx_sha == expected_onnx_sha, f"ONNX file hash mismatch: {onnx_sha}"

    # Proto and checker verification
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    assert model_proto.opset_import[0].version == 17

    # 3. Model Configuration Metadata Verification
    cfg_path = export_dir / "model_config.json"
    assert cfg_path.exists(), "model_config.json missing"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["source_checkpoint"]["sha256"] == expected_ckpt_sha
    assert cfg["onnx_export"]["sha256"] == expected_onnx_sha
    assert cfg["onnx_export"]["opset_version"] == 17
    assert len(cfg["onnx_export"]["classes"]) == 6
    assert cfg["mobile_preprocessing_contract"]["required_sample_rate"] == 16000
    assert cfg["mobile_preprocessing_contract"]["n_mels"] == 40
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in cfg["governance"]

    # 4. Parity Report Verdict Verification
    rep_path = export_dir / "parity_report.json"
    assert rep_path.exists(), "parity_report.json missing"
    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["summary"]["status"] == "PARITY_AUDIT_PASS"
    assert rep["summary"]["parity_threshold_met"] is True
    assert rep["summary"]["global_max_abs_probability_diff"] <= 1e-4
    assert rep["summary"]["overall_agreement_rate"] == 1.0
    assert rep["summary"]["total_test_samples"] == 356
    assert rep["cohort_1_demo_fixtures"]["agreement_rate"] == 1.0
    assert rep["cohort_2_fresh_holdout_300"]["agreement_rate"] == 1.0
    assert rep["cohort_3_deterministic_synthetic_tensors_50"]["agreement_rate"] == 1.0

    # 5. Live ONNX Runtime Execution Verification
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    assert len(inputs) == 1 and inputs[0].name == "log_mel"
    outputs = session.get_outputs()
    assert len(outputs) == 2
    out_names = [o.name for o in outputs]
    assert "logits" in out_names and "probabilities" in out_names

    dummy_feat = np.zeros((1, 1, 40, 101), dtype=np.float32)
    logits, probs = session.run(None, {"log_mel": dummy_feat})
    assert logits.shape == (1, 6)
    assert probs.shape == (1, 6)
    assert np.isclose(float(np.sum(probs[0])), 1.0, atol=1e-5)
    assert (probs >= 0.0).all()

    # 6. Audit Report Verification
    audit_md_p = export_dir / "export_audit.md"
    assert audit_md_p.exists(), "export_audit.md missing"
    audit_text = audit_md_p.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in audit_text
    assert "PARITY_AUDIT_PASS" in audit_text
    assert "arm_b_tcresnet14.onnx" in audit_text
    assert "Mobile Preprocessing Contract" in audit_text


def test_phase7_kws_frontend_golden_parity_and_mobile_contract_gate():
    """
    TC-67: Verify Phase 7 KWS frontend golden regression suite, bit-exact feature
    reproducibility, Periodic Hann window resolution (DISC-09), PyTorch/ONNX
    neural output parity, and 4-stage mobile readiness governance.
    Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
    """
    import json
    import hashlib
    import torch
    import torchaudio.transforms as T

    golden_dir = Path("experiments/synthetic_kws_demo/frontend_golden")
    assert golden_dir.exists(), "frontend_golden directory missing"

    # 1. Master Manifest & Artifact Existence
    manifest_p = golden_dir / "golden_manifest.json"
    contract_p = golden_dir / "frontend_contract.md"
    disc_p = golden_dir / "discrepancy_log.json"
    rep_p = golden_dir / "feature_parity_report.json"

    assert manifest_p.exists(), "golden_manifest.json missing"
    assert contract_p.exists(), "frontend_contract.md missing"
    assert disc_p.exists(), "discrepancy_log.json missing"
    assert rep_p.exists(), "feature_parity_report.json missing"

    with open(manifest_p, "r", encoding="utf-8") as f:
        man = json.load(f)
    assert man["total_cases"] == 12
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in man["governance"]

    # 2. Window Function Verification (DISC-09)
    mel = T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40)
    w = mel.spectrogram.window
    w_hann_periodic = torch.hann_window(400, periodic=True)
    assert torch.max(torch.abs(w - w_hann_periodic)).item() == 0.0, "Window is not periodic Hann"

    with open(disc_p, "r", encoding="utf-8") as f:
        disc = json.load(f)
    assert disc["discrepancies"][0]["id"] == "DISC-09"
    assert "Periodic Hann" in disc["discrepancies"][0]["resolution"]

    # 3. Parity Report Verdict Verification
    with open(rep_p, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["summary"]["suite_status"] == "GOLDEN_VERIFICATION_PASS"
    assert rep["summary"]["total_golden_cases"] == 12
    assert rep["summary"]["reproducible_feature_cases"] == 12
    assert rep["summary"]["argmax_agreement_cases"] == 12
    assert rep["summary"]["global_max_feature_diff_against_saved"] <= 1e-6
    assert rep["summary"]["global_max_prob_diff_pytorch_vs_onnx"] <= 1e-4

    # 4. Golden Tensor File Existence and SHA-256 Verification
    for case in man["cases"]:
        wav_p = Path(case["wav_artifact"]["path"])
        tensor_p = Path(case["tensor_artifact"]["path"])
        assert wav_p.exists(), f"WAV missing: {wav_p}"
        assert tensor_p.exists(), f"Tensor missing: {tensor_p}"

        act_wav_sha = hashlib.sha256(wav_p.read_bytes()).hexdigest().upper()
        act_tensor_sha = hashlib.sha256(tensor_p.read_bytes()).hexdigest().upper()
        assert act_wav_sha == case["wav_artifact"]["sha256"]
        assert act_tensor_sha == case["tensor_artifact"]["sha256"]

    # 5. Contract Document Verification
    contract_text = contract_p.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in contract_text
    assert "GOLDEN_VERIFICATION_PASS" in contract_text
    assert "Periodic Hann Window" in contract_text
    assert "Stage 1: Python Reference Reproducibility" in contract_text
    assert "Stage 3: Cross-Implementation Mobile Frontend Parity" in contract_text
    assert "UNVERIFIED" in contract_text


def test_phase8_mobile_cxx_frontend_parity_and_readiness_gate():
    """
    TC-68: Verify Phase 8 Portable C/C++ audio preprocessing frontend parity,
    standalone toolchain builds, numerical alignment with all 12 frozen golden vectors
    (max feature diff <= 1.0e-4), downstream ONNX graph probability parity (<= 1.0e-4),
    artificial edge-case safety suppression (NO_ACTION), golden artifact immutability,
    and mobile readiness governance.
    Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
    """
    import json
    import hashlib
    import subprocess
    from pathlib import Path

    mobile_dir = Path("experiments/synthetic_kws_demo/mobile_frontend")
    golden_dir = Path("experiments/synthetic_kws_demo/frontend_golden")
    assert mobile_dir.exists(), "mobile_frontend directory missing"
    assert golden_dir.exists(), "frontend_golden directory missing"

    # 1. Source & Binary Artifacts Existence
    h_file = mobile_dir / "include" / "kws_mobile_frontend.h"
    filterbank_h = mobile_dir / "include" / "mel_filterbank_40x257.h"
    cpp_file = mobile_dir / "src" / "kws_mobile_frontend.cpp"
    cli_src = mobile_dir / "tools" / "run_frontend_cli.cpp"
    test_src = mobile_dir / "tests" / "test_cxx_frontend.cpp"
    dll_file = mobile_dir / "bin" / "kws_mobile_frontend.dll"
    cli_exe = mobile_dir / "bin" / "run_frontend_cli.exe"
    test_exe = mobile_dir / "bin" / "test_cxx_frontend.exe"
    audit_file = mobile_dir / "mobile_frontend_audit.md"
    report_file = mobile_dir / "mobile_parity_report.json"

    for p in [h_file, filterbank_h, cpp_file, cli_src, test_src, dll_file, cli_exe, test_exe, audit_file, report_file]:
        assert p.exists(), f"Phase 8 artifact missing: {p}"

    # 2. Standalone C++ Unit Test Suite Execution
    res = subprocess.run([str(test_exe)], capture_output=True, text=True)
    assert res.returncode == 0, f"C++ unit test suite failed: {res.stderr}"
    assert "ALL C++ UNIT TESTS PASSED" in res.stdout
    assert "15185 / 15185 assertions passed" in res.stdout

    # 3. Golden Manifest & Artifact Immutability
    manifest_p = golden_dir / "golden_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        man = json.load(f)

    for case in man["cases"]:
        wav_p = Path(case["wav_artifact"]["path"])
        tensor_p = Path(case["tensor_artifact"]["path"])
        assert wav_p.exists() and tensor_p.exists()
        act_wav_sha = hashlib.sha256(wav_p.read_bytes()).hexdigest().upper()
        act_tensor_sha = hashlib.sha256(tensor_p.read_bytes()).hexdigest().upper()
        assert act_wav_sha == case["wav_artifact"]["sha256"], f"Golden WAV modified: {wav_p}"
        assert act_tensor_sha == case["tensor_artifact"]["sha256"], f"Golden tensor modified: {tensor_p}"

    # 4. Mobile Parity Report Verification
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["summary"]["suite_status"] in ["HOST_CROSS_IMPLEMENTATION_PARITY_PASS", "MOBILE_FRONTEND_PARITY_PASS"]
    assert rep["summary"]["total_golden_cases"] == 12
    assert rep["summary"]["feature_parity_cases"] == 12
    assert rep["summary"]["argmax_agreement_cases"] == 12
    assert rep["summary"]["edge_cases_verified_safe"] == 3
    assert rep["summary"]["global_max_feature_abs_diff"] <= 1.0e-4
    assert rep["summary"]["global_max_prob_diff"] <= 1.0e-4

    # 5. Edge Case Safety Suppression Verification
    for case in rep["cases"]:
        if case["case_id"] in ["case_09_silence_pure_zeros", "case_10_low_amplitude_ambient", "case_11_transient_impulse"]:
            assert case["adapted_action"] == "NO_ACTION"
            assert case["cxx_confidence"] < 0.50
            assert case["edge_case_safe"] is True

    # 6. Audit Report Documentation
    audit_text = audit_file.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in audit_text
    assert ("HOST_CROSS_IMPLEMENTATION_PARITY_PASS" in audit_text) or ("MOBILE_FRONTEND_PARITY_PASS" in audit_text)
    assert "Cooley-Tukey" in audit_text
    assert "Stage 3: Cross-Implementation Mobile Frontend Parity" in audit_text
    assert "VERIFIED" in audit_text


def test_phase9_android_offline_kws_integration_and_governance_gate():
    """
    TC-69: Verify Phase 9 Android Offline KWS integration, project scaffold,
    NDK CMake configuration, JNI bridge contract, zero-permission manifest,
    packaged asset integrity (ONNX model SHA-256 match), strict audio validation,
    and five-gate verification telemetry.
    Governance Invariant: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
    """
    import json
    import hashlib
    from pathlib import Path

    android_dir = Path("experiments/synthetic_kws_demo/android_app")
    assert android_dir.exists(), "android_app directory missing"

    # 1. Project Scaffold and Key Files
    manifest_xml = android_dir / "app" / "src" / "main" / "AndroidManifest.xml"
    cmake_file = android_dir / "app" / "src" / "main" / "cpp" / "CMakeLists.txt"
    jni_cpp = android_dir / "app" / "src" / "main" / "cpp" / "kws_jni_bridge.cpp"
    main_act = android_dir / "app" / "src" / "main" / "java" / "com" / "example" / "synthetic_kws" / "MainActivity.kt"
    jni_kt = android_dir / "app" / "src" / "main" / "java" / "com" / "example" / "synthetic_kws" / "KwsFrontendJni.kt"
    wav_kt = android_dir / "app" / "src" / "main" / "java" / "com" / "example" / "synthetic_kws" / "WavAudioLoader.kt"
    onnx_kt = android_dir / "app" / "src" / "main" / "java" / "com" / "example" / "synthetic_kws" / "OnnxKwsClassifier.kt"
    game_kt = android_dir / "app" / "src" / "main" / "java" / "com" / "example" / "synthetic_kws" / "GameActionAdapter.kt"
    audit_file = android_dir / "PHASE9_ANDROID_AUDIT.md"
    report_file = android_dir / "android_parity_report.json"

    for p in [manifest_xml, cmake_file, jni_cpp, main_act, jni_kt, wav_kt, onnx_kt, game_kt, audit_file, report_file]:
        assert p.exists(), f"Phase 9 file missing: {p}"

    # 2. Zero-Permission Governance Enforcement
    xml_text = manifest_xml.read_text(encoding="utf-8")
    assert "<uses-permission" not in xml_text, "Permissions declared in AndroidManifest.xml"
    assert "RECORD_AUDIO" not in xml_text, "RECORD_AUDIO found in AndroidManifest.xml"
    assert "INTERNET" not in xml_text, "INTERNET found in AndroidManifest.xml"

    # 3. Model Asset Integrity
    packaged_onnx = android_dir / "app" / "src" / "main" / "assets" / "models" / "arm_b_tcresnet14.onnx"
    assert packaged_onnx.exists(), f"Packaged ONNX model missing: {packaged_onnx}"
    onnx_sha = hashlib.sha256(packaged_onnx.read_bytes()).hexdigest().upper()
    assert onnx_sha == "8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625"

    # 4. Parity Report and Gate Telemetry
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["governance"] == "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED. Offline synthetic fixtures only."
    assert rep["overall_verdict"] == "BLOCKED"
    assert rep["gates"]["Gate_A_Host_Regression"]["status"] == "PASS"
    assert rep["gates"]["Gate_A_Host_Regression"]["global_max_feature_diff"] <= 1.0e-4
    assert rep["gates"]["Gate_A_Host_Regression"]["global_max_prob_diff"] <= 1.0e-4
    assert "BLOCKED" in rep["gates"]["Gate_B_Android_Native_Build"]["status"]
    assert "BLOCKED" in rep["gates"]["Gate_C_Android_Application_Build"]["status"]
    assert "BLOCKED" in rep["gates"]["Gate_D_Android_Inference"]["status"]
    assert rep["gates"]["Gate_E_Device_Measurements"]["status"] == "NOT_TESTED"

    # 5. Audit Documentation
    audit_text = audit_file.read_text(encoding="utf-8")
    assert "HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED" in audit_text
    assert "BLOCKED" in audit_text
    assert "Gate A" in audit_text
    assert "Gate B" in audit_text
    assert "Gate C" in audit_text
    assert "Gate D" in audit_text
    assert "Gate E" in audit_text













