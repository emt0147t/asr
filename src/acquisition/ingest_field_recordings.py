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

from src.acquisition.speaker_consent import verify_speaker_consent, get_speaker_metadata
from src.preprocessing.audio_normalizer import (
    load_and_resample, check_clipping, compute_rms_db, normalize_to_target_window,
    TARGET_SR, TARGET_SAMPLES, PEAK_TARGET
)

VALID_COMMANDS = {
    "TRAI": {"class_id": 0, "word": "trái"},
    "PHAI": {"class_id": 1, "word": "phải"},
    "LEN": {"class_id": 2, "word": "lên"},
    "XUONG": {"class_id": 3, "word": "xuống"}
}

FIELD_BASE_DIR = Path("data/raw/field_recordings")
RAW_AUDIO_DIR = FIELD_BASE_DIR / "audio_raw"
STANDARDIZED_AUDIO_DIR = FIELD_BASE_DIR / "audio_16k_standardized"
QUARANTINE_DIR = FIELD_BASE_DIR / "quarantined"
MANIFEST_PATH = FIELD_BASE_DIR / "field_collection_manifest.csv"
QUARANTINE_LOG_PATH = QUARANTINE_DIR / "quarantine_log.csv"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(128 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_existing_manifest(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    if manifest_path.exists():
        try:
            return pd.read_csv(manifest_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_to_csv(df_row: pd.DataFrame, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        df_row.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")


def validate_field_audio(audio_path: Path, 
                         speaker_id: str, 
                         command_label: str) -> Tuple[bool, str, Dict[str, Any]]:
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
    # 1. Consent verification
    if not verify_speaker_consent(speaker_id):
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
    if rms_db < -45.0:
        return False, f"abnormal_silence (RMS={rms_db:.1f} dBFS < -45.0 dBFS)", {
            "raw_duration": raw_dur, "rms_db": rms_db
        }

    # 7. Duplicate Check
    raw_sha = compute_sha256(audio_path)
    existing_df = load_existing_manifest()
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
    take_index: int = 1
) -> Dict[str, Any]:
    """
    Processes a single genuine human field recording:
      - Validates consent and audio quality.
      - Preserves untouched original in audio_raw/.
      - Standardizes 16 kHz mono 16-bit PCM 1.0s copy in audio_16k_standardized/.
      - Updates field_collection_manifest.csv.
    """
    input_path = Path(input_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file missing: {input_path}")

    command_label = command_label.upper()
    cmd_info = VALID_COMMANDS.get(command_label, {"class_id": -1, "word": command_label.lower()})
    class_id = cmd_info["class_id"]

    spk_meta = get_speaker_metadata(speaker_id) or {}
    dialect = spk_meta.get("dialect", "Unspecified")
    gender = spk_meta.get("gender", "UNKNOWN")
    dev_model = device_model or spk_meta.get("primary_device", "Unspecified Mobile")
    env = environment or spk_meta.get("primary_environment", "quiet_room")

    # Run validation
    is_valid, reason, metrics = validate_field_audio(input_path, speaker_id, command_label)

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
        append_to_csv(quarantine_row, QUARANTINE_LOG_PATH)
        return {
            "status": "QUARANTINED",
            "reason": reason,
            "recording_id": None
        }

    # Generate unique recording ID
    rec_id = f"field_{command_label.lower()}_{speaker_id}_{session_id}_take{take_index:03d}"

    # 1. Preserve Raw Original
    raw_dest_dir = RAW_AUDIO_DIR / speaker_id / session_id
    raw_dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dest_path = raw_dest_dir / f"{rec_id}{input_path.suffix.lower()}"
    shutil.copy2(input_path, raw_dest_path)

    # 2. Generate Standardized 16 kHz 1.0s Mono Copy
    std_dest_dir = STANDARDIZED_AUDIO_DIR / command_label
    std_dest_dir.mkdir(parents=True, exist_ok=True)
    std_dest_path = std_dest_dir / f"{rec_id}.wav"

    audio_resampled = metrics["audio"]
    standardized_audio = normalize_to_target_window(audio_resampled, target_samples=TARGET_SAMPLES)
    sf.write(str(std_dest_path), standardized_audio, TARGET_SR, subtype="PCM_16")

    sha_std = compute_sha256(std_dest_path)

    # 3. Create Manifest Record
    manifest_record = {
        "recording_id": rec_id,
        "source": "field_human_recording_campaign_v1",
        "source_version": "v1.0",
        "license": "Proprietary-Consented",
        "speaker_id": speaker_id,
        "dialect": dialect,
        "gender": gender,
        "command_label": command_label,
        "class_id": class_id,
        "transcript": cmd_info["word"],
        "session_id": session_id,
        "device_model": dev_model,
        "environment": env,
        "speaking_style": speaking_style,
        "raw_duration_sec": metrics["raw_duration"],
        "rms_db": metrics["rms_db"],
        "label_provenance": "human_field_recording",
        "raw_filepath": str(raw_dest_path).replace("\\", "/"),
        "processed_path": str(std_dest_path).replace("\\", "/"),
        "sha256_raw": metrics["sha256_raw"],
        "sha256_standardized": sha_std,
        "status": "APPROVED",
        "exclusion_reason": ""
    }

    append_to_csv(pd.DataFrame([manifest_record]), MANIFEST_PATH)
    print(f"[APPROVED] Ingested {rec_id} ({command_label}, {dialect}) -> {std_dest_path}")
    return manifest_record


def batch_ingest_directory(input_dir: Path, 
                           speaker_id: str, 
                           session_id: str,
                           command_label: Optional[str] = None,
                           device_model: Optional[str] = None,
                           environment: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Ingests all audio files from a directory for a specific speaker and session.
    If command_label is None, infers it from filename (e.g. trai_01.wav -> TRAI).
    """
    input_dir = Path(input_dir)
    results = []
    supported_exts = [".wav", ".m4a", ".mp3", ".ogg", ".flac", ".aac"]
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]
    files.sort()

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
            take_index=idx
        )
        results.append(res)

    approved = sum(1 for r in results if r["status"] == "APPROVED")
    print(f"Batch Ingestion Complete: {approved}/{len(results)} approved.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Single audio file to ingest")
    parser.add_argument("--dir", type=str, help="Directory of audio files to ingest")
    parser.add_argument("--speaker_id", type=str, required=True, help="Registered pseudonymous speaker ID")
    parser.add_argument("--command", type=str, choices=["TRAI", "PHAI", "LEN", "XUONG"], help="Command label")
    parser.add_argument("--session", type=str, default="SES_001", help="Session ID")
    parser.add_argument("--device", type=str, default="iPhone 13", help="Recording device model")
    parser.add_argument("--env", type=str, default="quiet_room", help="Acoustic environment")
    args = parser.parse_args()

    if args.file:
        if not args.command:
            raise ValueError("--command is required when ingesting a single file.")
        ingest_recording(Path(args.file), args.speaker_id, args.command, args.session, args.device, args.env)
    elif args.dir:
        batch_ingest_directory(Path(args.dir), args.speaker_id, args.session, args.command, args.device, args.env)
    else:
        print("Please provide --file or --dir. Use -h for help.")
