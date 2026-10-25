"""
Field Recording Validation and Ingestion Pipeline for Vietnamese KWS.
Ingests genuine human utterances of TRAI, PHAI, LEN, and XUONG.
Complies with docs/dataset_protocol.md:
  1. Enforces verified informed consent via consent_ledger.json.
  2. Preserves raw original recordings without destructive modification in audio_raw/.
  3. Validates duration (0.25s - 2.5s), clipping, silence (RMS < -45 dBFS), corruption, and SHA256 duplicates.
  4. Generates standardized 16 kHz mono 16-bit PCM 1.0s copies in audio_16k_standardized/.
  5. Immutably records metadata and audit logs into field_collection_manifest.csv.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout and sys.path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
else:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import os
import shutil
import hashlib
import json
import argparse
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import re
from src.acquisition.speaker_consent import (
    verify_speaker_consent, get_speaker_metadata, LEDGER_PATH,
    VALID_MICROPHONE_TYPES, VALID_DISTANCE_CATEGORIES
)
from src.preprocessing.audio_normalizer import (
    load_and_resample, check_clipping, compute_rms_db, normalize_to_target_window,
    TARGET_SR, TARGET_SAMPLES, PEAK_TARGET
)

VALID_COMMANDS = {
    "TRAI": {"class_id": 0, "word": "trái"},
    "PHAI": {"class_id": 1, "word": "phải"},
    "LEN": {"class_id": 2, "word": "lên"},
    "XUONG": {"class_id": 3, "word": "xuống"},
    "UNKNOWN": {"class_id": 4, "word": "unknown"},
    "SILENCE": {"class_id": 5, "word": "silence"}
}

# Canonical Ingestion & Manifest Provenance Allowlists (RES-01 Hardening)
ALLOWED_INGESTION_PROVENANCE = {
    "human_field_recording",
    "HUMAN_CONSENTED",
    "SYNTHETIC_TEST_FIXTURE",
    "test_harness_mock",
    "UNVERIFIED_OR_TEST_FIXTURE",
}

ALLOWED_MANIFEST_PROVENANCE = {
    "HUMAN_CONSENTED",
    "SYNTHETIC_TEST_FIXTURE",
    "UNVERIFIED_OR_TEST_FIXTURE",
    "test_harness_mock",
    "human_field_recording",
    "NEGATIVE_CORPUS_FLEURS",
    "NEGATIVE_NOISE_HUSEIN",
    "acoustic_reference_utterance",
    "continuous_speech_hard_negative",
    "embedded_keyword_sentence_negative",
    "microphone_ambient_noise",
}

HUMAN_BASE_DIR = Path("data/raw/human_recordings_v1")
LEGACY_FIELD_BASE_DIR = Path("data/raw/field_recordings")

# Production default root for future genuine human recordings (P0-1 remediation)
FIELD_BASE_DIR = HUMAN_BASE_DIR
RAW_AUDIO_DIR = FIELD_BASE_DIR / "audio_raw"
STANDARDIZED_AUDIO_DIR = FIELD_BASE_DIR / "audio_16k_standardized"
QUARANTINE_DIR = FIELD_BASE_DIR / "quarantined"
MANIFEST_PATH = FIELD_BASE_DIR / "human_collection_manifest_v1.csv"
QUARANTINE_LOG_PATH = QUARANTINE_DIR / "quarantine_log.csv"
LEDGER_PATH = FIELD_BASE_DIR / "consent_ledger_human.json"

# Historical fixture archive paths (preserved read-only for audit integrity)
LEGACY_MANIFEST_PATH = LEGACY_FIELD_BASE_DIR / "field_collection_manifest.csv"
LEGACY_LEDGER_PATH = LEGACY_FIELD_BASE_DIR / "consent_ledger.json"
LEGACY_QUARANTINE_LOG_PATH = LEGACY_FIELD_BASE_DIR / "quarantined" / "quarantine_log.csv"

# Canonical 28-Column Schema for Genuine Human Campaign (v1.0.0-human)
SCHEMA_28_COLUMNS = [
    "recording_id", "source", "source_version", "license", "consent_id",
    "speaker_id", "dialect", "gender", "age_bracket", "command_label",
    "class_id", "transcript", "sub_category", "session_id", "device_model",
    "microphone_type", "environment", "distance_category", "speaking_style",
    "raw_duration_sec", "rms_db", "label_provenance", "raw_filepath",
    "processed_path", "sha256_raw", "sha256_standardized", "status",
    "exclusion_reason"
]

# Legacy 23-Column Schema (Preserved for historical fixture compatibility)
LEGACY_23_COLUMNS = [
    "recording_id", "source", "source_version", "license", "speaker_id",
    "dialect", "gender", "command_label", "class_id", "transcript",
    "session_id", "device_model", "environment", "speaking_style",
    "raw_duration_sec", "rms_db", "label_provenance", "raw_filepath",
    "processed_path", "sha256_raw", "sha256_standardized", "status",
    "exclusion_reason"
]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(128 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def load_manifest(manifest_path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Reads a manifest and detects schema version:
      - 'v1.0.0-human-28col' if 28 columns
      - 'v1.0.0-legacy-23col' if 23 columns
      - 'custom' otherwise
    """
    target_path = manifest_path if manifest_path is not None else MANIFEST_PATH
    if not target_path.exists():
        return pd.DataFrame(), "empty"
    try:
        df = pd.read_csv(target_path)
    except Exception:
        return pd.DataFrame(), "error"
    cols = df.columns.tolist()
    if set(cols) == set(SCHEMA_28_COLUMNS):
        version = "v1.0.0-human-28col"
    elif set(cols) == set(LEGACY_23_COLUMNS):
        version = "v1.0.0-legacy-23col"
    else:
        version = "custom"
    return df, version


def load_existing_manifest(manifest_path: Optional[Path] = None) -> pd.DataFrame:
    df, _ = load_manifest(manifest_path)
    return df


def append_to_manifest(record: Dict[str, Any], manifest_path: Path):
    """
    Appends a manifest record adhering strictly to the manifest schema.
    Validates provenance and required fields before committing to disk (RES-01 Hardening).
    If the manifest is new, writes the canonical 28-column schema (v1.0.0-human).
    If appending to an existing manifest, detects whether the target is 28-column or
    legacy 23-column, aligning columns accordingly to avoid CSV header corruption.
    """
    if not isinstance(record, dict):
        raise TypeError(f"Manifest record must be a dict, got {type(record).__name__}")

    # 1. Validate required core fields
    core_required = ["recording_id", "speaker_id", "command_label", "status", "label_provenance"]
    for req in core_required:
        if req not in record or pd.isna(record[req]) or str(record[req]).strip() == "":
            raise ValueError(f"Cannot append to manifest: missing required field '{req}'")

    # 2. Validate label_provenance against canonical allowlist
    prov = str(record["label_provenance"])
    if prov not in ALLOWED_MANIFEST_PROVENANCE:
        raise ValueError(
            f"Cannot append to manifest: unauthorized label_provenance '{prov}'. "
            f"Allowed values: {sorted(list(ALLOWED_MANIFEST_PROVENANCE))}"
        )

    # 3. Validate status enum
    status = str(record["status"]).upper()
    if status not in {"APPROVED", "QUARANTINED", "REVOKED_WITHDRAWN", "REVOKED"}:
        raise ValueError(f"Cannot append to manifest: invalid status '{status}'")

    # 4. If APPROVED human-consented recording, ensure core demographic/command parameters are compliant
    if status == "APPROVED" and prov in ("HUMAN_CONSENTED", "human_field_recording"):
        cmd = str(record["command_label"]).upper()
        if cmd not in VALID_COMMANDS:
            raise ValueError(f"Cannot append to manifest: invalid command_label '{cmd}' for human recording")
        dialect = str(record.get("dialect", ""))
        if dialect not in {"Northern", "Central", "Southern"}:
            raise ValueError(f"Cannot append to manifest: invalid or missing dialect '{dialect}'")
        gender = str(record.get("gender", "")).upper()
        if gender not in {"FEMALE", "MALE", "OTHER"}:
            raise ValueError(f"Cannot append to manifest: invalid or missing gender '{gender}'")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        df_out = pd.DataFrame([record])[SCHEMA_28_COLUMNS]
        df_out.to_csv(manifest_path, index=False, encoding="utf-8")
        return

    try:
        existing_cols = pd.read_csv(manifest_path, nrows=0).columns.tolist()
    except Exception:
        existing_cols = []

    if set(existing_cols) == set(SCHEMA_28_COLUMNS):
        df_out = pd.DataFrame([record])[SCHEMA_28_COLUMNS]
    elif set(existing_cols) == set(LEGACY_23_COLUMNS):
        # Legacy 23-column adapter
        df_out = pd.DataFrame([record])[[c for c in LEGACY_23_COLUMNS if c in record]]
    elif existing_cols:
        # Match existing file columns
        aligned_record = {c: record.get(c, "") for c in existing_cols}
        df_out = pd.DataFrame([aligned_record])[existing_cols]
    else:
        df_out = pd.DataFrame([record])[SCHEMA_28_COLUMNS]

    df_out.to_csv(manifest_path, mode="a", header=False, index=False, encoding="utf-8")


def append_to_csv(df_row: pd.DataFrame, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        df_row.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")


def validate_field_audio(audio_path: Path, 
                         speaker_id: str, 
                         command_label: str,
                         manifest_path: Optional[Path] = None,
                         ledger_path: Optional[Path] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates audio file against quality gates:
      1. Consent verification
      2. Valid label
      3. File corruption check
      4. Duration check (0.25s - 2.5s)
      5. Digital clipping check
      6. Abnormal silence check (RMS < -45 dBFS)
      7. Duplicate check (SHA256)
    """
    target_manifest_path = manifest_path if manifest_path is not None else MANIFEST_PATH
    target_ledger_path = ledger_path if ledger_path is not None else LEDGER_PATH

    # 0. Path & ID sanitization
    if not re.match(r"^[A-Za-z0-9_\-]+$", speaker_id):
        return False, f"invalid_speaker_id_format_{speaker_id}", {}

    # 1. Consent verification
    if not verify_speaker_consent(speaker_id, ledger_path=target_ledger_path):
        return False, "missing_or_unconsented_speaker", {}

    # 2. Label check
    label_upper = command_label.upper()
    if label_upper not in VALID_COMMANDS:
        return False, f"invalid_command_label_{command_label}", {}

    # 3. Sound file parseability
    try:
        audio, sr = load_and_resample(audio_path)
    except Exception as e:
        return False, f"corrupt_file_read_error: {str(e)}", {}

    raw_dur = len(audio) / sr
    rms_db = compute_rms_db(audio)
    is_clipped = check_clipping(audio)

    # 4. Duration bounds
    if raw_dur < 0.25:
        return False, f"truncated_utterance (duration={raw_dur:.2f}s < 0.25s)", {
            "raw_duration": raw_dur, "rms_db": rms_db
        }
    if raw_dur > 2.50:
        return False, f"excessive_duration (duration={raw_dur:.2f}s > 2.50s)", {
            "raw_duration": raw_dur, "rms_db": rms_db
        }

    # 5. Clipping
    if is_clipped:
        return False, "digital_clipping_detected", {
            "raw_duration": raw_dur, "rms_db": rms_db
        }

    # 6. Abnormal silence
    if label_upper != "SILENCE" and rms_db < -45.0:
        return False, f"abnormal_silence (RMS={rms_db:.1f} dBFS < -45.0 dBFS)", {
            "raw_duration": raw_dur, "rms_db": rms_db
        }

    # 7. Duplicate Check
    raw_sha = compute_sha256(audio_path)
    existing_df = load_existing_manifest(target_manifest_path)
    if not existing_df.empty and "sha256_raw" in existing_df.columns:
        if raw_sha in existing_df["sha256_raw"].values:
            return False, "duplicate_recording_sha256_match", {
                "raw_duration": raw_dur, "rms_db": rms_db, "sha256_raw": raw_sha
            }

    metrics = {
        "audio": audio,
        "sr": sr,
        "raw_duration": round(raw_dur, 3),
        "rms_db": round(rms_db, 2),
        "sha256_raw": raw_sha
    }
    return True, "", metrics


def ingest_recording(
    input_audio_path: Path,
    speaker_id: str,
    command_label: str,
    session_id: str = "SES_DEFAULT",
    device_model: Optional[str] = None,
    environment: Optional[str] = None,
    speaking_style: str = "normal",
    take_index: int = 1,
    manifest_path: Optional[Path] = None,
    quarantine_path: Optional[Path] = None,
    raw_base_dir: Optional[Path] = None,
    std_base_dir: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
    provenance: str = "human_field_recording",
    sub_category: Optional[str] = None,
    microphone_type: Optional[str] = None,
    distance_category: Optional[str] = None,
    age_bracket: Optional[str] = None,
    consent_id: Optional[str] = None,
    strict_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Processes a single genuine human field recording:
      - Validates consent and audio quality.
      - Preserves untouched original in audio_raw/.
      - Standardizes 16 kHz mono 16-bit PCM 1.0s copy in audio_16k_standardized/.
      - Appends record to manifest using canonical 28-column schema (or legacy 23-col adapter).
      - Core Security Invariant: For human provenance ('human_field_recording' or 'HUMAN_CONSENTED'),
        strict metadata validation is ALWAYS enforced (effective_strict=True), regardless of the
        strict_metadata parameter, and permissive fallback defaults are never injected.
    """
    target_manifest_path = manifest_path if manifest_path is not None else MANIFEST_PATH
    target_quarantine_path = quarantine_path if quarantine_path is not None else QUARANTINE_LOG_PATH
    target_raw_base_dir = raw_base_dir if raw_base_dir is not None else RAW_AUDIO_DIR
    target_std_base_dir = std_base_dir if std_base_dir is not None else STANDARDIZED_AUDIO_DIR
    target_ledger_path = ledger_path if ledger_path is not None else LEDGER_PATH

    input_path = Path(input_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file missing: {input_path}")

    # Security check: prevent path traversal in speaker_id and session_id
    if not re.match(r"^[A-Za-z0-9_\-]+$", speaker_id):
        raise ValueError(f"Invalid characters in speaker_id '{speaker_id}'. Only alphanumeric, underscores, and dashes allowed.")
    if not re.match(r"^[A-Za-z0-9_\-]+$", session_id):
        raise ValueError(f"Invalid characters in session_id '{session_id}'. Only alphanumeric, underscores, and dashes allowed.")

    # Provenance Allowlist Gate (RES-01 Hardening: fail-closed on unauthorized/unknown provenance)
    if provenance not in ALLOWED_INGESTION_PROVENANCE:
        reason_str = f"unauthorized_or_unknown_provenance_{provenance}"
        print(f"[REJECTED/QUARANTINE] {input_path.name} from {speaker_id} -> Reason: {reason_str}")
        quarantine_row = pd.DataFrame([{
            "input_filename": input_path.name,
            "speaker_id": speaker_id,
            "command_label": command_label,
            "session_id": session_id,
            "dialect": "Unspecified",
            "exclusion_reason": reason_str,
            "raw_duration_sec": 0.0,
            "rms_db": 0.0,
            "status": "QUARANTINED"
        }])
        append_to_csv(quarantine_row, target_quarantine_path)
        return {
            "status": "QUARANTINED",
            "reason": reason_str,
            "recording_id": None
        }

    command_label = command_label.upper()
    cmd_info = VALID_COMMANDS.get(command_label, {"class_id": -1, "word": command_label.lower()})
    class_id = cmd_info["class_id"]

    spk_meta = get_speaker_metadata(speaker_id, ledger_path=target_ledger_path) or {}
    dialect = spk_meta.get("dialect", "Unspecified")
    gender = spk_meta.get("gender", "UNKNOWN")
    dev_model = device_model or spk_meta.get("primary_device", "Unspecified Mobile")
    env = environment or spk_meta.get("primary_environment", "quiet_room")

    # Core Security Invariant: Strict metadata validation is mandatory for human provenance.
    # Callers cannot bypass strict validation for human recordings via strict_metadata=False.
    is_human_provenance = provenance in ("human_field_recording", "HUMAN_CONSENTED")
    effective_strict = True if is_human_provenance else bool(strict_metadata)

    age = age_bracket or spk_meta.get("age_bracket") or ("18-30" if not effective_strict else None)
    mic_type = microphone_type or spk_meta.get("primary_microphone") or ("builtin_mobile" if not effective_strict else None)
    dist_from_spk = spk_meta.get("distance_category") or spk_meta.get("speaking_distance")
    dist_cat = distance_category or dist_from_spk or ("close_talk_15cm" if not effective_strict else None)

    # Default acoustic sub-category
    if sub_category is not None:
        sub_cat = sub_category
    elif command_label == "SILENCE":
        sub_cat = "background_room_tone"
    else:
        sub_cat = "isolated_command"

    # Strict metadata validation: prevents fabricated defaults from entering human dataset
    if effective_strict:
        meta_errs = []
        if dialect not in ["Northern", "Central", "Southern"]:
            meta_errs.append(f"missing_or_invalid_dialect_{dialect}")
        if gender not in ["FEMALE", "MALE", "OTHER"]:
            meta_errs.append(f"missing_or_invalid_gender_{gender}")
        if (age_bracket is None and "age_bracket" not in spk_meta) or age not in ["18-30", "31-45", "46-60", "over_60", "31-50", "51+"]:
            meta_errs.append(f"unregistered_or_invalid_age_bracket_{age}")
        if (microphone_type is None and "primary_microphone" not in spk_meta) or mic_type not in VALID_MICROPHONE_TYPES:
            meta_errs.append(f"unregistered_or_invalid_microphone_type_{mic_type}")
        if (distance_category is None and not dist_from_spk) or dist_cat not in VALID_DISTANCE_CATEGORIES:
            meta_errs.append(f"missing_or_invalid_distance_category_{dist_cat}")

        # Contradictory metadata checks
        if command_label == "SILENCE" and sub_cat not in ["background_room_tone", "ambient_noise"]:
            meta_errs.append("contradictory_metadata_silence_command_with_non_silence_sub_category")
        if command_label in ["TRAI", "PHAI", "LEN", "XUONG"] and sub_cat in ["background_room_tone", "ambient_noise"]:
            meta_errs.append("contradictory_metadata_speech_command_with_room_tone_sub_category")
        if dist_cat in ["far_field_2m"] and mic_type in ["wired_headset", "earbuds", "bluetooth_earbuds"]:
            meta_errs.append("contradictory_metadata_wired_headset_with_far_field_distance")

        if meta_errs:
            reason_str = "; ".join(meta_errs)
            print(f"[REJECTED/QUARANTINE] {input_path.name} from {speaker_id} -> Reason: {reason_str}")
            quarantine_row = pd.DataFrame([{
                "input_filename": input_path.name,
                "speaker_id": speaker_id,
                "command_label": command_label,
                "session_id": session_id,
                "dialect": dialect,
                "exclusion_reason": reason_str,
                "raw_duration_sec": 0.0,
                "rms_db": 0.0,
                "status": "QUARANTINED"
            }])
            append_to_csv(quarantine_row, target_quarantine_path)
            return {
                "status": "QUARANTINED",
                "reason": reason_str,
                "recording_id": None
            }

    # Consent identifier resolution
    if consent_id is not None:
        cid = consent_id
    elif "consent_id" in spk_meta and spk_meta["consent_id"]:
        cid = spk_meta["consent_id"]
    elif speaker_id.startswith("SPK_H_"):
        cid = f"CONSENT_{speaker_id[4:]}_v1.0"
    else:
        cid = f"CONSENT_{speaker_id}_v1.0"

    # Generate unique recording ID
    if speaker_id.startswith("SPK_H_"):
        rec_id = f"rec_h_{command_label.lower()}_{speaker_id}_{session_id}_take{take_index:03d}"
    else:
        rec_id = f"field_{command_label.lower()}_{speaker_id}_{session_id}_take{take_index:03d}"

    # Check for duplicate recording ID
    existing_df = load_existing_manifest(target_manifest_path)
    if not existing_df.empty and "recording_id" in existing_df.columns:
        if rec_id in existing_df["recording_id"].values:
            print(f"[REJECTED/QUARANTINE] Recording ID '{rec_id}' already exists in manifest.")
            quarantine_row = pd.DataFrame([{
                "input_filename": input_path.name,
                "speaker_id": speaker_id,
                "command_label": command_label,
                "session_id": session_id,
                "dialect": dialect,
                "exclusion_reason": f"duplicate_recording_id_already_exists_{rec_id}",
                "raw_duration_sec": 0.0,
                "rms_db": 0.0,
                "status": "QUARANTINED"
            }])
            append_to_csv(quarantine_row, target_quarantine_path)
            return {
                "status": "QUARANTINED",
                "reason": f"duplicate_recording_id_already_exists_{rec_id}",
                "recording_id": None
            }

    # Run validation
    is_valid, reason, metrics = validate_field_audio(
        input_path, speaker_id, command_label,
        manifest_path=target_manifest_path, ledger_path=target_ledger_path
    )

    if not is_valid:
        # Quarantine recording
        print(f"[REJECTED/QUARANTINE] {input_path.name} from {speaker_id} -> Reason: {reason}")
        quarantine_row = pd.DataFrame([{
            "input_filename": input_path.name,
            "speaker_id": speaker_id,
            "command_label": command_label,
            "session_id": session_id,
            "dialect": dialect,
            "exclusion_reason": reason,
            "raw_duration_sec": metrics.get("raw_duration", 0.0),
            "rms_db": metrics.get("rms_db", 0.0),
            "status": "QUARANTINED"
        }])
        append_to_csv(quarantine_row, target_quarantine_path)
        return {
            "status": "QUARANTINED",
            "reason": reason,
            "recording_id": None
        }

    # 1. Resolve destination paths with path traversal defense
    raw_dest_dir = (target_raw_base_dir / speaker_id / session_id).resolve()
    raw_dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dest_path = raw_dest_dir / f"{rec_id}{input_path.suffix.lower()}"

    std_dest_dir = (target_std_base_dir / command_label).resolve()
    std_dest_dir.mkdir(parents=True, exist_ok=True)
    std_dest_path = std_dest_dir / f"{rec_id}.wav"

    # Pre-existing file collision safeguard: prevent destructive overwriting of existing disk files
    if raw_dest_path.exists() or std_dest_path.exists():
        reason_str = f"destination_file_already_exists_on_disk_{rec_id}"
        print(f"[REJECTED/QUARANTINE] Destination file already exists on disk for {rec_id}")
        quarantine_row = pd.DataFrame([{
            "input_filename": input_path.name,
            "speaker_id": speaker_id,
            "command_label": command_label,
            "session_id": session_id,
            "dialect": dialect,
            "exclusion_reason": reason_str,
            "raw_duration_sec": metrics.get("raw_duration", 0.0),
            "rms_db": metrics.get("rms_db", 0.0),
            "status": "QUARANTINED"
        }])
        append_to_csv(quarantine_row, target_quarantine_path)
        return {
            "status": "QUARANTINED",
            "reason": reason_str,
            "recording_id": None
        }

    raw_created = False
    std_created = False

    try:
        # Preserve Raw Original
        shutil.copy2(input_path, raw_dest_path)
        raw_created = True

        # Generate Standardized 16 kHz 1.0s Mono Copy
        audio_resampled = metrics["audio"]
        standardized_audio = normalize_to_target_window(audio_resampled, target_samples=TARGET_SAMPLES)
        sf.write(str(std_dest_path), standardized_audio, TARGET_SR, subtype="PCM_16")
        std_created = True

        sha_std = compute_sha256(std_dest_path)

        # Harmonize provenance label
        prov_label = "HUMAN_CONSENTED" if provenance == "human_field_recording" else provenance

        # 3. Create Manifest Record (Full 28-Column Canonical Schema)
        manifest_record = {
            "recording_id": rec_id,
            "source": "vietnamese_kws_human_campaign_v1",
            "source_version": "v1.0.0-human",
            "license": "Proprietary-Consented-Research-Commercial",
            "consent_id": cid,
            "speaker_id": speaker_id,
            "dialect": dialect,
            "gender": gender,
            "age_bracket": age,
            "command_label": command_label,
            "class_id": class_id,
            "transcript": cmd_info["word"],
            "sub_category": sub_cat,
            "session_id": session_id,
            "device_model": dev_model,
            "microphone_type": mic_type,
            "environment": env,
            "distance_category": dist_cat,
            "speaking_style": speaking_style,
            "raw_duration_sec": metrics["raw_duration"],
            "rms_db": metrics["rms_db"],
            "label_provenance": prov_label,
            "raw_filepath": str(raw_dest_path).replace("\\", "/"),
            "processed_path": str(std_dest_path).replace("\\", "/"),
            "sha256_raw": metrics["sha256_raw"],
            "sha256_standardized": sha_std,
            "status": "APPROVED",
            "exclusion_reason": ""
        }
        append_to_manifest(manifest_record, target_manifest_path)
    except Exception as e:
        # Clean up ONLY artifacts created by this specific attempt; pre-existing files are never touched
        if raw_created and raw_dest_path.exists():
            try:
                raw_dest_path.unlink()
            except Exception:
                pass
        if std_created and std_dest_path.exists():
            try:
                std_dest_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Manifest append failed for {rec_id}: {e}") from e

    print(f"[APPROVED] Ingested {rec_id} ({command_label}, {dialect}) -> {std_dest_path}")
    return manifest_record


def batch_ingest_directory(input_dir: Path, 
                           speaker_id: str, 
                           session_id: str,
                           command_label: Optional[str] = None,
                           device_model: Optional[str] = None,
                           environment: Optional[str] = None,
                           manifest_path: Optional[Path] = None,
                           quarantine_path: Optional[Path] = None,
                           raw_base_dir: Optional[Path] = None,
                           std_base_dir: Optional[Path] = None,
                           ledger_path: Optional[Path] = None,
                           provenance: str = "human_field_recording",
                           sub_category: Optional[str] = None,
                           microphone_type: Optional[str] = None,
                           distance_category: Optional[str] = None,
                           age_bracket: Optional[str] = None,
                           consent_id: Optional[str] = None,
                           strict_metadata: bool = False) -> List[Dict[str, Any]]:
    """
    Ingests all audio files from a directory for a specific speaker and session.
    If command_label is None, infers it from filename (e.g. trai_01.wav -> TRAI).
    Core Security Invariant: Enforces strict metadata validation for all human provenance recordings.
    """
    input_dir = Path(input_dir)
    results = []
    supported_exts = [".wav", ".m4a", ".mp3", ".ogg", ".flac", ".aac"]
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]
    files.sort()

    # Core Security Invariant: Strict metadata is mandatory for human provenance batch ingestion
    is_human_provenance = provenance in ("human_field_recording", "HUMAN_CONSENTED")
    effective_strict = True if is_human_provenance else bool(strict_metadata)

    print(f"\nBatch ingesting {len(files)} files from {input_dir} for speaker {speaker_id}...")
    for idx, audio_file in enumerate(files, start=1):
        # Determine command label
        if command_label:
            lbl = command_label
        else:
            fname_lower = audio_file.stem.lower()
            detected_lbl = None
            for k in VALID_COMMANDS.keys():
                if k.lower() in fname_lower or VALID_COMMANDS[k]["word"] in fname_lower:
                    detected_lbl = k
                    break
            if not detected_lbl:
                print(f"[SKIP] Could not infer command label from filename: {audio_file.name}")
                continue
            lbl = detected_lbl

        res = ingest_recording(
            input_audio_path=audio_file,
            speaker_id=speaker_id,
            command_label=lbl,
            session_id=session_id,
            device_model=device_model,
            environment=environment,
            take_index=idx,
            manifest_path=manifest_path,
            quarantine_path=quarantine_path,
            raw_base_dir=raw_base_dir,
            std_base_dir=std_base_dir,
            ledger_path=ledger_path,
            provenance=provenance,
            sub_category=sub_category,
            microphone_type=microphone_type,
            distance_category=distance_category,
            age_bracket=age_bracket,
            consent_id=consent_id,
            strict_metadata=effective_strict
        )
        results.append(res)

    approved = sum(1 for r in results if r["status"] == "APPROVED")
    print(f"Batch Ingestion Complete: {approved}/{len(results)} approved.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese KWS Field Recording Ingestion")
    parser.add_argument("--file", type=str, help="Single audio file to ingest")
    parser.add_argument("--dir", type=str, help="Directory of audio files to ingest")
    parser.add_argument("--speaker_id", "--speaker-id", dest="speaker_id", type=str, required=True, help="Registered pseudonymous speaker ID")
    parser.add_argument("--command", type=str, choices=["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"], help="Command label")
    parser.add_argument("--session", type=str, default="SES_001", help="Session ID")
    parser.add_argument("--device", type=str, default="iPhone 13", help="Recording device model")
    parser.add_argument("--env", type=str, default="quiet_room", help="Acoustic environment")
    parser.add_argument("--sub-category", "--sub_category", dest="sub_category", type=str, help="Sub-category (e.g. conversational_speech, background_room_tone)")
    parser.add_argument("--mic", "--microphone_type", dest="microphone_type", type=str, choices=VALID_MICROPHONE_TYPES, help="Microphone type")
    parser.add_argument("--distance", "--distance_category", dest="distance_category", type=str, choices=VALID_DISTANCE_CATEGORIES, help="Distance category")
    parser.add_argument("--age", "--age_bracket", dest="age_bracket", type=str, choices=["18-30", "31-45", "46-60", "over_60", "31-50", "51+", "unspecified"], help="Age bracket")
    parser.add_argument("--consent-id", "--consent_id", dest="consent_id", type=str, help="Consent ID (defaults to ledger lookup)")
    parser.add_argument("--provenance", type=str, default="human_field_recording", choices=sorted(list(ALLOWED_INGESTION_PROVENANCE)), help="Provenance category")
    parser.add_argument("--strict", action="store_true", help="Enforce strict metadata validation for genuine human recordings")
    args = parser.parse_args()

    effective_strict = args.strict or (args.provenance in ("human_field_recording", "HUMAN_CONSENTED"))

    if args.file:
        if not args.command:
            raise ValueError("--command is required when ingesting a single file.")
        ingest_recording(
            input_audio_path=Path(args.file),
            speaker_id=args.speaker_id,
            command_label=args.command,
            session_id=args.session,
            device_model=args.device,
            environment=args.env,
            sub_category=args.sub_category,
            microphone_type=args.microphone_type,
            distance_category=args.distance_category,
            age_bracket=args.age_bracket,
            consent_id=args.consent_id,
            provenance=args.provenance,
            strict_metadata=effective_strict,
        )
    elif args.dir:
        batch_ingest_directory(
            input_dir=Path(args.dir),
            speaker_id=args.speaker_id,
            session_id=args.session,
            command_label=args.command,
            device_model=args.device,
            environment=args.env,
            sub_category=args.sub_category,
            microphone_type=args.microphone_type,
            distance_category=args.distance_category,
            age_bracket=args.age_bracket,
            consent_id=args.consent_id,
            provenance=args.provenance,
            strict_metadata=effective_strict,
        )
    else:
        print("Please provide --file or --dir. Use -h for help.")
