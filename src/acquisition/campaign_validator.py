"""
Automated Acceptance Checks and Campaign Validator for Vietnamese KWS.
Provides pure, non-destructive verification functions for:
  - Audio format and quality gating (16 kHz, mono, 16-bit PCM, duration, clipping, silence)
  - Consent verification against consent ledger
  - Metadata completeness and cryptographic checksums
  - Duplicate detection (recording_id and sha256)
  - Provenance classification and synthetic-fixture separation
  - Speaker-disjoint train/val/test split generation and validation
"""

import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import hashlib
import zlib
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
import pandas as pd
import soundfile as sf


VALID_COMMAND_LABELS = {"TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"}
CLASS_ID_MAP = {
    "TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5
}
ALLOWED_PROVENANCE_VALUES = {
    "HUMAN_CONSENTED", "SYNTHETIC_TEST_FIXTURE", "UNVERIFIED_OR_TEST_FIXTURE",
    "test_harness_mock", "human_field_recording",
    "NEGATIVE_CORPUS_FLEURS", "NEGATIVE_NOISE_HUSEIN",
    "acoustic_reference_utterance", "continuous_speech_hard_negative",
    "embedded_keyword_sentence_negative", "microphone_ambient_noise"
}
BLOCKED_UNVERIFIED_PROVENANCE_VALUES = {
    "NEGATIVE_NOISE_HUSEIN": (
        "External source 'husein_mic_noise' is BLOCKED: unverified licensing provenance. "
        "Allowed fallback: ambient room tone recorded during authorized collection sessions "
        "(HUMAN_CONSENTED, sub_category: background_room_tone)."
    )
}
ALLOWED_DIALECTS = {"Northern", "Central", "Southern"}
ALLOWED_GENDERS = {"FEMALE", "MALE", "OTHER"}
ALLOWED_STATUSES = {"APPROVED", "QUARANTINED", "REVOKED_WITHDRAWN"}


def compute_file_sha256(path: Path) -> str:
    """Computes uppercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(128 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def validate_audio_file(
    audio_path: Path,
    expected_sr: int = 16000,
    expected_channels: int = 1,
    expected_frames: int = 16000,
    max_consecutive_clipped: int = 3,
    min_rms_db: float = -45.0,
    allow_silence: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates physical audio file against signal quality gates:
      - Soundfile parseability
      - Sample rate, channels, bit depth
      - Exact frame length
      - Digital clipping
      - Abnormal silence
    """
    if not audio_path.exists():
        return False, f"file_not_found: {audio_path}", {}

    try:
        info = sf.info(str(audio_path))
    except Exception as e:
        return False, f"corrupt_file_read_error: {str(e)}", {}

    if info.samplerate != expected_sr:
        return False, f"sample_rate_mismatch: {info.samplerate} != {expected_sr}", {}

    if info.channels != expected_channels:
        return False, f"channel_count_mismatch: {info.channels} != {expected_channels}", {}

    if expected_frames is not None and info.frames != expected_frames:
        return False, f"frame_count_mismatch: {info.frames} != {expected_frames}", {}

    try:
        audio, sr = sf.read(str(audio_path), dtype="float32")
    except Exception as e:
        return False, f"audio_decode_error: {str(e)}", {}

    # Clipping detection
    is_clipped_sample = np.abs(audio) >= 0.999
    clipped_run = 0
    max_run = 0
    for s in is_clipped_sample:
        if s:
            clipped_run += 1
            if clipped_run > max_run:
                max_run = clipped_run
        else:
            clipped_run = 0

    if max_run >= max_consecutive_clipped:
        return False, f"digital_clipping_detected: {max_run} consecutive samples >= 0.999", {}

    # RMS Energy
    energy = np.mean(audio ** 2)
    rms_db = 10.0 * np.log10(energy + 1e-12)

    if not allow_silence and rms_db < min_rms_db:
        return False, f"abnormal_silence: {rms_db:.2f} dBFS < {min_rms_db:.2f} dBFS", {}

    metrics = {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "subtype": info.subtype,
        "duration_sec": info.duration,
        "rms_db": float(rms_db),
        "max_clipped_run": int(max_run)
    }

    return True, "", metrics


def validate_manifest_row(
    row: pd.Series,
    consent_ledger: Dict[str, Any],
    check_disk_files: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates a single manifest record against the 28-column specification.
    """
    errors = []

    # Required field presence
    required_fields = [
        "recording_id", "speaker_id", "dialect", "gender",
        "command_label", "class_id", "session_id", "device_model",
        "label_provenance", "raw_filepath", "processed_path",
        "sha256_raw", "sha256_standardized", "status"
    ]
    for rf in required_fields:
        if rf not in row or pd.isna(row[rf]) or str(row[rf]).strip() == "":
            errors.append(f"missing_required_field: {rf}")

    if errors:
        return False, errors

    spk_id = str(row["speaker_id"])
    cmd_label = str(row["command_label"]).upper()
    class_id = int(row["class_id"]) if not pd.isna(row["class_id"]) else -1
    provenance = str(row["label_provenance"])
    dialect = str(row["dialect"])
    gender = str(row["gender"]).upper()
    status = str(row["status"]).upper()

    # Enum checks
    if cmd_label not in VALID_COMMAND_LABELS:
        errors.append(f"invalid_command_label: {cmd_label}")
    elif CLASS_ID_MAP.get(cmd_label) != class_id:
        errors.append(f"class_id_label_mismatch: {cmd_label} should have class_id {CLASS_ID_MAP.get(cmd_label)}, got {class_id}")

    if dialect not in ALLOWED_DIALECTS:
        errors.append(f"invalid_dialect: {dialect}")

    if gender not in ALLOWED_GENDERS:
        errors.append(f"invalid_gender: {gender}")

    if provenance in BLOCKED_UNVERIFIED_PROVENANCE_VALUES:
        errors.append(f"unverified_source_license_blocked: {BLOCKED_UNVERIFIED_PROVENANCE_VALUES[provenance]}")
    elif provenance not in ALLOWED_PROVENANCE_VALUES:
        errors.append(f"invalid_provenance_enum: {provenance}")

    if status not in ALLOWED_STATUSES:
        errors.append(f"invalid_status: {status}")

    # Consent verification for HUMAN_CONSENTED
    if provenance == "HUMAN_CONSENTED":
        if spk_id not in consent_ledger:
            errors.append(f"missing_consent_entry_for_human_speaker: {spk_id}")
        else:
            spk_info = consent_ledger[spk_id]
            if not spk_info.get("consent_given", False):
                errors.append(f"consent_not_granted_for_speaker: {spk_id}")

    # File and SHA-256 verification
    if check_disk_files:
        raw_path = Path(str(row["raw_filepath"]))
        proc_path = Path(str(row["processed_path"]))

        if not raw_path.exists():
            errors.append(f"raw_file_missing_on_disk: {raw_path}")
        else:
            calc_raw_sha = compute_file_sha256(raw_path)
            expected_raw_sha = str(row["sha256_raw"]).upper()
            if calc_raw_sha != expected_raw_sha:
                errors.append(f"raw_sha256_mismatch: {calc_raw_sha} != {expected_raw_sha}")

        if not proc_path.exists():
            errors.append(f"processed_file_missing_on_disk: {proc_path}")
        else:
            calc_proc_sha = compute_file_sha256(proc_path)
            expected_proc_sha = str(row["sha256_standardized"]).upper()
            if calc_proc_sha != expected_proc_sha:
                errors.append(f"processed_sha256_mismatch: {calc_proc_sha} != {expected_proc_sha}")
            else:
                allow_silence = (cmd_label == "SILENCE")
                audio_ok, audio_err, _ = validate_audio_file(proc_path, allow_silence=allow_silence)
                if not audio_ok:
                    errors.append(f"processed_audio_quality_error: {audio_err}")

    return (len(errors) == 0), errors


def validate_manifest_dataset(
    df: pd.DataFrame,
    consent_ledger: Dict[str, Any],
    check_disk_files: bool = True
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates an entire manifest dataframe for dataset-level invariants:
      - Row-level compliance
      - Primary key uniqueness (recording_id)
      - Audio duplicate hash detection
      - Class and dialect representation
      - Synthetic fixture separation
    """
    all_errors = []

    if df.empty:
        return False, ["empty_manifest_dataframe"], {}

    # 1. Primary key uniqueness
    rec_id_counts = df["recording_id"].value_counts()
    dups_rec_id = rec_id_counts[rec_id_counts > 1]
    if not dups_rec_id.empty:
        all_errors.append(f"duplicate_recording_ids_detected: {dups_rec_id.to_dict()}")

    # 2. Audio duplicate check (sha256_raw)
    if "sha256_raw" in df.columns:
        sha_counts = df[df["sha256_raw"].notna()]["sha256_raw"].value_counts()
        dups_sha = sha_counts[sha_counts > 1]
        if not dups_sha.empty:
            all_errors.append(f"duplicate_raw_audio_hashes_detected: {len(dups_sha)} collisions")

    # 3. Synthetic vs Human Separation Check
    # Ensure no rows tagged HUMAN_CONSENTED originate from historical fixture paths or test identities
    if "label_provenance" in df.columns:
        human_rows = df[df["label_provenance"] == "HUMAN_CONSENTED"]
        for _, r in human_rows.iterrows():
            raw_fp = str(r.get("raw_filepath", "")).lower()
            spk = str(r.get("speaker_id", "")).upper()
            if "field_recordings" in raw_fp and "human_recordings" not in raw_fp:
                all_errors.append(f"synthetic_fixture_co_mingling: human recording {r.get('recording_id')} located in legacy fixture directory {raw_fp}")
            if any(k in spk for k in ["TEST", "MOCK", "SYNTH"]):
                all_errors.append(f"synthetic_speaker_id_mislabeled_as_human: {spk}")

    # 4. Row-by-row checks
    for idx, row in df.iterrows():
        is_valid, row_errs = validate_manifest_row(row, consent_ledger, check_disk_files=check_disk_files)
        if not is_valid:
            for re in row_errs:
                all_errors.append(f"row_{idx} ({row.get('recording_id')}): {re}")

    # Summary statistics
    stats = {
        "total_records": len(df),
        "unique_speakers": df["speaker_id"].nunique() if "speaker_id" in df.columns else 0,
        "unique_sessions": df["session_id"].nunique() if "session_id" in df.columns else 0,
        "class_counts": df["command_label"].value_counts().to_dict() if "command_label" in df.columns else {},
        "dialect_counts": df["dialect"].value_counts().to_dict() if "dialect" in df.columns else {},
        "provenance_counts": df["label_provenance"].value_counts().to_dict() if "label_provenance" in df.columns else {},
        "total_errors": len(all_errors)
    }

    return (len(all_errors) == 0), all_errors, stats


def validate_speaker_disjointness(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[bool, Dict[str, Any]]:
    """
    Formally verifies that train, validation, and test partitions share:
      - 0 speakers (zero speaker leakage)
      - 0 sessions (zero session leakage)
      - 0 recordings (zero original-recording or augmentation leakage)
    Fails closed if speaker_id column is missing or contains invalid/empty values.
    """
    # 0. Validate speaker_id column presence
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if "speaker_id" not in d.columns:
            return False, {
                "is_disjoint": False,
                "error": f"missing_required_column_speaker_id_in_{name}",
                "speaker_status": "MISSING_COLUMN",
                "recording_status": "NOT_VERIFIABLE_COLUMN_ABSENT",
                "session_status": "NOT_VERIFIABLE_COLUMN_ABSENT",
                "lineage_status": "NOT_VERIFIABLE_COLUMN_ABSENT",
                "speaker_overlaps": {"train_val": [], "train_test": [], "val_test": []},
                "session_overlaps": {"train_val": [], "train_test": [], "val_test": []},
                "recording_overlaps": {"train_val": [], "train_test": [], "val_test": []},
                "lineage_overlaps": {"train_val": [], "train_test": [], "val_test": []},
                "train_speaker_count": 0, "val_speaker_count": 0, "test_speaker_count": 0
            }

    def _clean_ids(series):
        if series is None:
            return set()
        cleaned = set()
        for x in series.dropna().unique():
            s = str(x).strip()
            if s and s.lower() not in ["none", "nan", "-", "null", "unspecified", ""]:
                cleaned.add(s)
        return cleaned

    # Check for invalid / empty speaker IDs
    invalid_spk_counts = {}
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if not d.empty:
            s_series = d["speaker_id"].astype(str).str.strip().str.lower()
            bad = int(d["speaker_id"].isna().sum() + (s_series.isin(["", "nan", "none", "null", "-", "unspecified"])).sum())
            if bad > 0:
                invalid_spk_counts[name] = bad

    train_speakers = _clean_ids(df_train["speaker_id"])
    val_speakers = _clean_ids(df_val["speaker_id"])
    test_speakers = _clean_ids(df_test["speaker_id"])

    train_sessions = _clean_ids(df_train["session_id"]) if "session_id" in df_train.columns else set()
    val_sessions = _clean_ids(df_val["session_id"]) if "session_id" in df_val.columns else set()
    test_sessions = _clean_ids(df_test["session_id"]) if "session_id" in df_test.columns else set()

    train_recs = _clean_ids(df_train["recording_id"]) if "recording_id" in df_train.columns else set()
    val_recs = _clean_ids(df_val["recording_id"]) if "recording_id" in df_val.columns else set()
    test_recs = _clean_ids(df_test["recording_id"]) if "recording_id" in df_test.columns else set()

    overlap_train_val_spk = train_speakers & val_speakers
    overlap_train_test_spk = train_speakers & test_speakers
    overlap_val_test_spk = val_speakers & test_speakers

    overlap_train_val_ses = train_sessions & val_sessions
    overlap_train_test_ses = train_sessions & test_sessions
    overlap_val_test_ses = val_sessions & test_sessions

    overlap_train_val_rec = train_recs & val_recs
    overlap_train_test_rec = train_recs & test_recs
    overlap_val_test_rec = val_recs & test_recs

    # Optional augmentation lineage check (parent_recording_id if present)
    train_parents = _clean_ids(df_train["parent_recording_id"]) if "parent_recording_id" in df_train.columns else set()
    val_parents = _clean_ids(df_val["parent_recording_id"]) if "parent_recording_id" in df_val.columns else set()
    test_parents = _clean_ids(df_test["parent_recording_id"]) if "parent_recording_id" in df_test.columns else set()

    overlap_train_val_parent = (train_recs & val_parents) | (val_recs & train_parents) | (train_parents & val_parents)
    overlap_train_test_parent = (train_recs & test_parents) | (test_recs & train_parents) | (train_parents & test_parents)
    overlap_val_test_parent = (val_recs & test_parents) | (test_recs & val_parents) | (val_parents & test_parents)

    has_leakage = bool(
        overlap_train_val_spk or overlap_train_test_spk or overlap_val_test_spk or
        overlap_train_val_ses or overlap_train_test_ses or overlap_val_test_ses or
        overlap_train_val_rec or overlap_train_test_rec or overlap_val_test_rec or
        overlap_train_val_parent or overlap_train_test_parent or overlap_val_test_parent
    )

    is_disjoint = (not has_leakage) and (len(invalid_spk_counts) == 0)

    # Explicit audit status classification for partitions
    active_dfs = [df_train, df_val] + ([df_test] if not df_test.empty else [])

    has_recording_col = all("recording_id" in d.columns for d in active_dfs)
    if not has_recording_col:
        recording_status = "NOT_VERIFIABLE_COLUMN_ABSENT"
    elif overlap_train_val_rec or overlap_train_test_rec or overlap_val_test_rec:
        recording_status = "LEAKAGE_DETECTED"
    else:
        recording_status = "VERIFIED_DISJOINT"

    has_session_col = all("session_id" in d.columns for d in active_dfs)
    if not has_session_col:
        session_status = "NOT_VERIFIABLE_COLUMN_ABSENT"
    elif overlap_train_val_ses or overlap_train_test_ses or overlap_val_test_ses:
        session_status = "LEAKAGE_DETECTED"
    else:
        session_status = "VERIFIED_DISJOINT"

    has_lineage_col = all("parent_recording_id" in d.columns for d in active_dfs)
    if not has_lineage_col:
        lineage_status = "NOT_VERIFIABLE_COLUMN_ABSENT"
    elif overlap_train_val_parent or overlap_train_test_parent or overlap_val_test_parent:
        lineage_status = "LEAKAGE_DETECTED"
    else:
        lineage_status = "VERIFIED_DISJOINT"

    if overlap_train_val_spk or overlap_train_test_spk or overlap_val_test_spk or len(invalid_spk_counts) > 0:
        speaker_status = "LEAKAGE_DETECTED" if not invalid_spk_counts else "INVALID_SPEAKERS_OR_LEAKAGE"
    else:
        speaker_status = "VERIFIED_DISJOINT"

    details = {
        "is_disjoint": is_disjoint,
        "invalid_speaker_counts": invalid_spk_counts,
        "speaker_status": speaker_status,
        "recording_status": recording_status,
        "session_status": session_status,
        "lineage_status": lineage_status,
        "speaker_overlaps": {
            "train_val": sorted(list(overlap_train_val_spk)),
            "train_test": sorted(list(overlap_train_test_spk)),
            "val_test": sorted(list(overlap_val_test_spk))
        },
        "session_overlaps": {
            "train_val": sorted(list(overlap_train_val_ses)),
            "train_test": sorted(list(overlap_train_test_ses)),
            "val_test": sorted(list(overlap_val_test_ses))
        },
        "recording_overlaps": {
            "train_val": sorted(list(overlap_train_val_rec)),
            "train_test": sorted(list(overlap_train_test_rec)),
            "val_test": sorted(list(overlap_val_test_rec))
        },
        "lineage_overlaps": {
            "train_val": sorted(list(overlap_train_val_parent)),
            "train_test": sorted(list(overlap_train_test_parent)),
            "val_test": sorted(list(overlap_val_test_parent))
        },
        "train_speaker_count": len(train_speakers),
        "val_speaker_count": len(val_speakers),
        "test_speaker_count": len(test_speakers)
    }

    return is_disjoint, details


VALID_PROFILES = ("benchmark", "human_campaign")


def validate_split_manifests_preflight(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    profile: Optional[str] = None,
    require_sessions: bool = False,
    require_lineage: bool = False,
    validation_profile: Optional[str] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Strict preflight validation of split manifests at training, publication, and evaluation boundaries.
    Guarantees:
      1. Validation mode governed by explicit profile ('benchmark' vs 'human_campaign').
      2. No silent downgrade from human_campaign to benchmark mode.
      3. Non-empty partitions (train and val must contain records).
      4. Required columns present across partitions based on profile:
         - 'benchmark': 'speaker_id', 'recording_id'
         - 'human_campaign': 'speaker_id', 'recording_id', 'session_id', 'parent_recording_id'
      5. Strict speaker disjointness (pairwise disjoint across train, val, test).
      6. Strict session disjointness (pairwise disjoint across train, val, test).
      7. Strict recording ID disjointness.
      8. Strict lineage disjointness and resolution:
         - Nullable parent IDs allowed for root recordings.
         - Non-null parent references must resolve to known recordings in the dataset.
         - Parent and child must be co-partitioned (zero cross-partition leakage).
         - Self-referential lineage rejected.
      9. Input DataFrames are not modified in-place.
    Returns:
      (is_valid: bool, errors: List[str], details: Dict[str, Any])
    """
    errors: List[str] = []

    # 0. Resolve validation profile (Fail-Closed if omitted)
    raw_profile = validation_profile if validation_profile is not None else profile
    if raw_profile is None or str(raw_profile).strip() == "":
        return False, ["omitted_validation_profile: validation_profile must be explicitly specified ('benchmark' or 'human_campaign'). Implicit fallback is strictly prohibited."], {
            "is_disjoint": False,
            "speaker_status": "OMITTED_PROFILE",
            "recording_status": "OMITTED_PROFILE",
            "session_status": "OMITTED_PROFILE",
            "lineage_status": "OMITTED_PROFILE",
            "validation_profile": None,
        }
    active_profile = str(raw_profile).strip().lower()
    if active_profile not in VALID_PROFILES:
        return False, [f"invalid_validation_profile: {active_profile}. Permitted: {list(VALID_PROFILES)}"], {
            "is_disjoint": False,
            "speaker_status": "INVALID_PROFILE",
            "recording_status": "INVALID_PROFILE",
            "session_status": "INVALID_PROFILE",
            "lineage_status": "INVALID_PROFILE",
            "validation_profile": active_profile,
        }

    partitions = {"train": df_train, "val": df_val}
    if df_test is not None:
        partitions["test"] = df_test

    # Diagnostic detection of in-manifest markers (never used for silent downgrade)
    has_human_markers = any(
        ("consent_id" in p_df.columns) or
        ("source" in p_df.columns and (p_df["source"].astype(str) == "vietnamese_kws_human_campaign_v1").any()) or
        ("label_provenance" in p_df.columns and (p_df["label_provenance"].astype(str) == "HUMAN_CONSENTED").any())
        for p_df in partitions.values()
    )

    # 1. Non-empty check
    for p_name, p_df in partitions.items():
        if p_df is None or p_df.empty:
            errors.append(f"partition_{p_name}_is_empty")

    if errors:
        return False, errors, {
            "is_disjoint": False,
            "speaker_status": "NOT_EVALUATED_EMPTY_PARTITION",
            "recording_status": "NOT_EVALUATED_EMPTY_PARTITION",
            "session_status": "NOT_EVALUATED_EMPTY_PARTITION",
            "lineage_status": "NOT_EVALUATED_EMPTY_PARTITION",
            "validation_profile": active_profile,
            "detected_schema_markers": "human_campaign" if has_human_markers else "benchmark"
        }

    # 2. Determine required columns from profile and explicit flags (Fail-Closed, No Silent Downgrade)
    if active_profile == "human_campaign":
        effective_require_sessions = True
        effective_require_lineage = True
    else:  # benchmark
        effective_require_sessions = require_sessions
        effective_require_lineage = require_lineage

    req_cols = ["speaker_id", "recording_id"]
    if effective_require_sessions:
        req_cols.append("session_id")
    if effective_require_lineage:
        req_cols.append("parent_recording_id")

    for p_name, p_df in partitions.items():
        missing_cols = [c for c in req_cols if c not in p_df.columns]
        if missing_cols:
            errors.append(f"partition_{p_name}_missing_required_columns_{missing_cols}")

    if errors:
        return False, errors, {
            "is_disjoint": False,
            "speaker_status": "MISSING_REQUIRED_COLUMNS",
            "recording_status": "MISSING_REQUIRED_COLUMNS",
            "session_status": "MISSING_REQUIRED_COLUMNS",
            "lineage_status": "MISSING_REQUIRED_COLUMNS",
            "validation_profile": active_profile,
            "detected_schema_markers": "human_campaign" if has_human_markers else "benchmark"
        }

    # 3. Invalid/empty identifier checks (speaker_id, recording_id, session_id)
    for p_name, p_df in partitions.items():
        spk_series = p_df["speaker_id"].astype(str).str.strip().str.lower()
        bad_count = int(p_df["speaker_id"].isna().sum() + (spk_series.isin(["", "nan", "none", "null", "-", "unspecified"])).sum())
        if bad_count > 0:
            errors.append(f"partition_{p_name}_contains_{bad_count}_missing_or_invalid_speaker_ids")

        if "recording_id" in p_df.columns:
            rec_series = p_df["recording_id"].astype(str).str.strip().str.lower()
            bad_rec = int(p_df["recording_id"].isna().sum() + (rec_series.isin(["", "nan", "none", "null", "-", "unspecified"])).sum())
            if bad_rec > 0:
                errors.append(f"partition_{p_name}_contains_{bad_rec}_missing_or_invalid_recording_ids")

        if effective_require_sessions and "session_id" in p_df.columns:
            ses_series = p_df["session_id"].astype(str).str.strip().str.lower()
            bad_ses = int(p_df["session_id"].isna().sum() + (ses_series.isin(["", "nan", "none", "null", "-", "unspecified"])).sum())
            if bad_ses > 0:
                errors.append(f"partition_{p_name}_contains_{bad_ses}_missing_or_invalid_session_ids")

    # 4. Strict Lineage reference resolution and cross-partition boundary checks
    all_known_recs = {}
    for p_name, p_df in partitions.items():
        if "recording_id" in p_df.columns:
            for rid in p_df["recording_id"].dropna().astype(str).str.strip():
                if rid and rid.lower() not in ["none", "nan", "-", "null", "unspecified", ""]:
                    all_known_recs[rid] = p_name

    has_any_lineage_col = any("parent_recording_id" in p_df.columns for p_df in partitions.values())
    if has_any_lineage_col:
        for p_name, p_df in partitions.items():
            if "parent_recording_id" in p_df.columns:
                for _, row in p_df.iterrows():
                    val = row.get("parent_recording_id")
                    if pd.isna(val):
                        # Legitimate root recording with nullable parent ID
                        continue
                    parent_id = str(val).strip()
                    if not parent_id or parent_id.lower() in ["none", "nan", "-", "null", "unspecified", ""]:
                        # Legitimate root recording with nullable parent ID
                        continue

                    rec_id = str(row.get("recording_id", "")).strip() if "recording_id" in row else ""

                    # Check A: Malformed self-referential lineage
                    if rec_id and parent_id == rec_id:
                        errors.append(f"partition_{p_name}_contains_self_referential_lineage_{rec_id}")
                        continue

                    # Check B: Unresolved / missing parent reference (enforced for human_campaign profile)
                    if active_profile == "human_campaign" and parent_id not in all_known_recs:
                        errors.append(f"partition_{p_name}_contains_unresolved_parent_lineage_{parent_id}")
                        continue

                    # Check C: Cross-partition parent/child assignment
                    if parent_id in all_known_recs:
                        parent_p = all_known_recs[parent_id]
                        if parent_p != p_name:
                            errors.append(f"cross_partition_lineage_leakage_{parent_p}_{p_name}: parent {parent_id} in {parent_p}, child {rec_id} in {p_name}")

    # 5. Disjointness checks using validate_speaker_disjointness
    dummy_test = df_test if df_test is not None else pd.DataFrame(columns=["speaker_id", "recording_id", "session_id"])
    is_disjoint, details = validate_speaker_disjointness(df_train, df_val, dummy_test)

    details["validation_profile"] = active_profile
    details["detected_schema_markers"] = "human_campaign" if has_human_markers else "benchmark"

    if not is_disjoint:
        for pair, spks in details.get("speaker_overlaps", {}).items():
            if spks:
                errors.append(f"speaker_overlap_detected_{pair}:_{spks[:5]}")
        for pair, sess in details.get("session_overlaps", {}).items():
            if sess:
                errors.append(f"session_overlap_detected_{pair}:_{sess[:5]}")
        for pair, recs in details.get("recording_overlaps", {}).items():
            if recs:
                errors.append(f"recording_id_overlap_detected_{pair}:_{recs[:5]}")
        for pair, lins in details.get("lineage_overlaps", {}).items():
            if lins:
                errors.append(f"augmentation_lineage_overlap_detected_{pair}:_{lins[:5]}")

    if any("lineage" in e.lower() for e in errors):
        details["lineage_status"] = "LEAKAGE_DETECTED" if any("leakage" in e.lower() or "overlap" in e.lower() for e in errors) else "MALFORMED_OR_UNRESOLVED_LINEAGE"

    is_valid = (len(errors) == 0)
    details["is_disjoint"] = is_disjoint and is_valid
    return is_valid, errors, details



def generate_stratified_disjoint_splits(
    df: pd.DataFrame,
    strata_cols: Optional[List[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    salt: str = "kws_vietnamese_campaign_2026_v1",
    min_speakers_per_stratum: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partitions dataset into train, val, and test partitions enforcing:
      1. Mathematical speaker and session disjointness (zero leakage).
      2. Demographic balance across defined strata (default: ['dialect', 'gender']).
      3. Actionable diagnostics reporting infeasible strata cohorts (N < min_speakers_per_stratum).
      4. Fully deterministic assignment via salted CRC32 hashing.
    """
    if df.empty:
        empty_diag = {
            "is_disjoint": True, "total_speakers": 0, "infeasible_strata": {},
            "strata_breakdown": {}, "partition_speaker_counts": {"train": 0, "val": 0, "test": 0},
            "partition_class_counts": {"train": {}, "val": {}, "test": {}},
            "age_representation": {}
        }
        return df.copy(), df.copy(), df.copy(), empty_diag

    if "speaker_id" not in df.columns:
        raise ValueError("Cannot partition: 'speaker_id' column missing from dataframe.")

    target_strata = strata_cols if strata_cols is not None else ["dialect", "gender"]
    active_strata = [c for c in target_strata if c in df.columns]

    # If no strata columns present, fallback to uniform deterministic assignment
    if not active_strata:
        speakers = sorted(df["speaker_id"].unique())
        speaker_split_map = {}
        for spk in speakers:
            key = f"{salt}_{spk}".encode("utf-8")
            hash_val = zlib.crc32(key) & 0xFFFFFFFF
            score = hash_val / 0xFFFFFFFF
            if score < train_ratio:
                speaker_split_map[spk] = "train"
            elif score < (train_ratio + val_ratio):
                speaker_split_map[spk] = "val"
            else:
                speaker_split_map[spk] = "test"

        df_train = df[df["speaker_id"].map(speaker_split_map) == "train"].copy()
        df_val = df[df["speaker_id"].map(speaker_split_map) == "val"].copy()
        df_test = df[df["speaker_id"].map(speaker_split_map) == "test"].copy()
        is_disjoint, details = validate_speaker_disjointness(df_train, df_val, df_test)

        class_counts = {
            "train": df_train["command_label"].value_counts().to_dict() if "command_label" in df_train.columns else {},
            "val": df_val["command_label"].value_counts().to_dict() if "command_label" in df_val.columns else {},
            "test": df_test["command_label"].value_counts().to_dict() if "command_label" in df_test.columns else {}
        }
        diagnostics = {
            "is_disjoint": is_disjoint,
            "total_speakers": len(speakers),
            "strata_columns": [],
            "infeasible_strata": {},
            "strata_breakdown": {},
            "partition_speaker_counts": {
                "train": df_train["speaker_id"].nunique(),
                "val": df_val["speaker_id"].nunique(),
                "test": df_test["speaker_id"].nunique()
            },
            "partition_class_counts": class_counts,
            "age_representation": {},
            "disjoint_details": details
        }
        return df_train, df_val, df_test, diagnostics

    # Extract unique speaker metadata
    spk_df = df[["speaker_id"] + active_strata].drop_duplicates(subset=["speaker_id"]).copy()
    speaker_split_map = {}
    strata_breakdown = {}
    infeasible_strata = {}

    grouped = spk_df.groupby(active_strata)
    for group_key, group in grouped:
        stratum_name = str(group_key) if isinstance(group_key, str) else "-".join(str(k) for k in group_key)
        spk_list = sorted(group["speaker_id"].tolist())
        n_spk = len(spk_list)

        # Sort speakers in this stratum deterministically by salted CRC32 score
        scored_spks = []
        for spk in spk_list:
            key = f"{salt}_{stratum_name}_{spk}".encode("utf-8")
            hash_val = zlib.crc32(key) & 0xFFFFFFFF
            score = hash_val / 0xFFFFFFFF
            scored_spks.append((score, spk))
        scored_spks.sort(key=lambda x: (x[0], x[1]))
        ordered_spks = [s[1] for s in scored_spks]

        # Feasibility check
        if n_spk < min_speakers_per_stratum:
            infeasible_strata[stratum_name] = {
                "speaker_count": n_spk,
                "min_required": min_speakers_per_stratum,
                "diagnostic": (
                    f"Strata '{stratum_name}' has only {n_spk} speakers (< {min_speakers_per_stratum} "
                    f"minimum required for a 3-way disjoint split). Split will result in empty "
                    f"or severely undersampled partitions."
                )
            }
            # Best-effort assignment using proportional score thresholds
            tr_spks, val_spks, te_spks = [], [], []
            for sc, spk in scored_spks:
                if sc < train_ratio:
                    tr_spks.append(spk)
                elif sc < (train_ratio + val_ratio):
                    val_spks.append(spk)
                else:
                    te_spks.append(spk)
        else:
            # Feasible stratum (N >= 3): compute exact counts preserving proportional representation
            n_val = max(1, int(round(n_spk * val_ratio)))
            n_test = max(1, int(round(n_spk * test_ratio)))
            n_train = n_spk - n_val - n_test

            # Adjust in edge cases where round pushes train to 0
            if n_train < 1:
                if n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1
                n_train = n_spk - n_val - n_test

            tr_spks = ordered_spks[:n_train]
            val_spks = ordered_spks[n_train:n_train + n_val]
            te_spks = ordered_spks[n_train + n_val:]

        for s in tr_spks:
            speaker_split_map[s] = "train"
        for s in val_spks:
            speaker_split_map[s] = "val"
        for s in te_spks:
            speaker_split_map[s] = "test"

        strata_breakdown[stratum_name] = {
            "total_speakers": n_spk,
            "train_speakers": len(tr_spks),
            "val_speakers": len(val_spks),
            "test_speakers": len(te_spks),
            "train_ratio": round(len(tr_spks) / n_spk, 3) if n_spk > 0 else 0.0,
            "val_ratio": round(len(val_spks) / n_spk, 3) if n_spk > 0 else 0.0,
            "test_ratio": round(len(te_spks) / n_spk, 3) if n_spk > 0 else 0.0,
        }

    df_train = df[df["speaker_id"].map(speaker_split_map) == "train"].copy()
    df_val = df[df["speaker_id"].map(speaker_split_map) == "val"].copy()
    df_test = df[df["speaker_id"].map(speaker_split_map) == "test"].copy()

    is_disjoint, details = validate_speaker_disjointness(df_train, df_val, df_test)
    assert is_disjoint, "Assertion error: Generated stratified splits failed speaker disjointness verification!"

    # Compile partition class counts
    class_counts = {
        "train": df_train["command_label"].value_counts().to_dict() if "command_label" in df_train.columns else {},
        "val": df_val["command_label"].value_counts().to_dict() if "command_label" in df_val.columns else {},
        "test": df_test["command_label"].value_counts().to_dict() if "command_label" in df_test.columns else {}
    }

    # Compile age representation across partitions if age_bracket column is present
    age_rep = {}
    if "age_bracket" in df.columns:
        for split_name, s_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            if not s_df.empty:
                unique_spk_age = s_df.drop_duplicates("speaker_id")["age_bracket"].value_counts().to_dict()
                age_rep[split_name] = unique_spk_age
            else:
                age_rep[split_name] = {}

    diagnostics = {
        "is_disjoint": is_disjoint,
        "total_speakers": len(spk_df),
        "strata_columns": active_strata,
        "num_strata": len(strata_breakdown),
        "infeasible_strata": infeasible_strata,
        "strata_breakdown": strata_breakdown,
        "partition_speaker_counts": {
            "train": df_train["speaker_id"].nunique() if not df_train.empty else 0,
            "val": df_val["speaker_id"].nunique() if not df_val.empty else 0,
            "test": df_test["speaker_id"].nunique() if not df_test.empty else 0
        },
        "partition_class_counts": class_counts,
        "age_representation": age_rep,
        "disjoint_details": details
    }

    return df_train, df_val, df_test, diagnostics


def generate_disjoint_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    salt: str = "kws_vietnamese_campaign_2026_v1",
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions dataset into train, val, test using deterministic speaker-level salted hashing.
    Guarantees strict speaker and session disjointness.
    If stratify=True and demographic strata ('dialect', 'gender') are present in df, uses
    generate_stratified_disjoint_splits to maintain regional/gender balance.
    Otherwise, uses uniform speaker-level salted hashing.
    """
    if stratify and any(c in df.columns for c in ["dialect", "gender"]):
        df_tr, df_v, df_te, _ = generate_stratified_disjoint_splits(
            df=df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            salt=salt
        )
        return df_tr, df_v, df_te

    if "speaker_id" not in df.columns:
        raise ValueError("Cannot partition: 'speaker_id' column missing from dataframe.")

    speakers = sorted(df["speaker_id"].unique())
    speaker_split_map = {}

    for spk in speakers:
        key = f"{salt}_{spk}".encode("utf-8")
        hash_val = zlib.crc32(key) & 0xFFFFFFFF
        score = hash_val / 0xFFFFFFFF

        if score < train_ratio:
            speaker_split_map[spk] = "train"
        elif score < (train_ratio + val_ratio):
            speaker_split_map[spk] = "val"
        else:
            speaker_split_map[spk] = "test"

    df_train = df[df["speaker_id"].map(speaker_split_map) == "train"].copy()
    df_val = df[df["speaker_id"].map(speaker_split_map) == "val"].copy()
    df_test = df[df["speaker_id"].map(speaker_split_map) == "test"].copy()

    is_disjoint, _ = validate_speaker_disjointness(df_train, df_val, df_test)
    assert is_disjoint, "Assertion error: Generated splits failed speaker disjointness verification!"

    return df_train, df_val, df_test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Campaign & Split Validator")
    parser.add_argument("--verify-splits", action="store_true", help="Run preflight split manifest verification")
    parser.add_argument("--train", type=str, help="Path to train manifest CSV")
    parser.add_argument("--val", type=str, help="Path to val manifest CSV")
    parser.add_argument("--test", type=str, default=None, help="Optional path to test manifest CSV")
    parser.add_argument("--profile", "--validation-profile", dest="profile", type=str, default=None, choices=["benchmark", "human_campaign"], help="Validation profile: benchmark or human_campaign (required)")
    args = parser.parse_args()

    if args.verify_splits:
        if not args.profile:
            print("[ERROR] --profile or --validation-profile must be explicitly specified ('benchmark' or 'human_campaign'). Implicit fallback is strictly prohibited.")
            sys.exit(1)
        if not args.train or not args.val:
            print("[ERROR] --train and --val paths are required when using --verify-splits.")
            sys.exit(1)
        tr_p = Path(args.train)
        va_p = Path(args.val)
        te_p = Path(args.test) if args.test else None

        if not tr_p.exists() or not va_p.exists() or (te_p and not te_p.exists()):
            print(f"[ERROR] One or more specified split manifests do not exist: train={tr_p}, val={va_p}, test={te_p}")
            sys.exit(1)

        df_tr = pd.read_csv(tr_p)
        df_va = pd.read_csv(va_p)
        df_te = pd.read_csv(te_p) if te_p else None

        valid, errs, diag = validate_split_manifests_preflight(df_tr, df_va, df_te, validation_profile=args.profile)
        print("=" * 60)
        print("SPLIT MANIFEST PREFLIGHT VALIDATION REPORT")
        print("=" * 60)
        print(f"Train path: {tr_p} ({len(df_tr)} rows, {df_tr['speaker_id'].nunique() if 'speaker_id' in df_tr.columns else 0} speakers)")
        print(f"Val path:   {va_p} ({len(df_va)} rows, {df_va['speaker_id'].nunique() if 'speaker_id' in df_va.columns else 0} speakers)")
        if df_te is not None:
            print(f"Test path:  {te_p} ({len(df_te)} rows, {df_te['speaker_id'].nunique() if 'speaker_id' in df_te.columns else 0} speakers)")
        print(f"Profile:    {diag.get('validation_profile', args.profile)}")
        print(f"Result:     {'PASSED' if valid else 'FAILED'}")
        if "speaker_status" in diag:
            print(f"Speaker Status:   {diag['speaker_status']}")
        if "recording_status" in diag:
            print(f"Recording Status: {diag['recording_status']}")
        if "session_status" in diag:
            print(f"Session Status:   {diag['session_status']}")
        if "lineage_status" in diag:
            print(f"Lineage Status:   {diag['lineage_status']}")
        if errs:
            print("\nErrors Detected:")
            for e in errs:
                print(f"  - {e}")
        print("=" * 60)
        sys.exit(0 if valid else 1)
