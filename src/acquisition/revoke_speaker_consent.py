"""
Standalone Operator CLI Tool: Consent Revocation & Data Quarantine Workflow.
Complies with docs/dataset_protocol.md Section 4.5 and docs/sound_command_recognition/campaign_20261013/collection_protocol.md:
  - Enforces explicit --confirm flag to prevent accidental withdrawal execution.
  - Supports --dry-run mode displaying affected files and records without disk mutation.
  - Quarantines associated raw and standardized audio into <quarantine_dir>/withdrawn_speakers/<speaker_id>/
    without hard deletion, preserving audit trail integrity.
  - Updates manifest rows to status="REVOKED_WITHDRAWN" with reason.
  - Updates companion consent ledger (consent_given=False, revocation timestamp, audit ID).
  - Logs withdrawal action to quarantine_log.csv.
  - Guarantees idempotent re-execution safety.
  - Explicit Governance Notice: Revocation permanently removes data from future training and splits.
    Previously trained model checkpoints cannot automatically purge learned representations
    without complete retraining from scratch.
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

import argparse
import json
import re
import secrets
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from src.acquisition.ingest_field_recordings import (
    HUMAN_BASE_DIR,
    MANIFEST_PATH as DEFAULT_MANIFEST_PATH,
    LEDGER_PATH as DEFAULT_LEDGER_PATH,
    RAW_AUDIO_DIR as DEFAULT_RAW_DIR,
    STANDARDIZED_AUDIO_DIR as DEFAULT_STD_DIR,
    QUARANTINE_DIR as DEFAULT_QUARANTINE_DIR,
    QUARANTINE_LOG_PATH as DEFAULT_QUARANTINE_LOG_PATH,
    append_to_csv,
    compute_sha256
)
from src.acquisition.speaker_consent import load_consent_ledger, save_consent_ledger


def execute_consent_revocation(
    speaker_id: str,
    reason: str = "participant_withdrawal",
    confirm: bool = False,
    dry_run: bool = False,
    manifest_path: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
    quarantine_dir: Optional[Path] = None,
    quarantine_log_path: Optional[Path] = None,
    raw_base_dir: Optional[Path] = None,
    std_base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes consent revocation and file quarantine workflow for a given speaker.
    If confirm is False or dry_run is True, performs a non-destructive simulation.
    """
    target_manifest_path = manifest_path if manifest_path is not None else DEFAULT_MANIFEST_PATH
    target_ledger_path = ledger_path if ledger_path is not None else DEFAULT_LEDGER_PATH
    target_quarantine_dir = quarantine_dir if quarantine_dir is not None else DEFAULT_QUARANTINE_DIR
    target_quarantine_log = quarantine_log_path if quarantine_log_path is not None else (target_quarantine_dir / "quarantine_log.csv")
    target_raw_base = raw_base_dir if raw_base_dir is not None else DEFAULT_RAW_DIR
    target_std_base = std_base_dir if std_base_dir is not None else DEFAULT_STD_DIR

    # 1. Input sanitization
    if not re.match(r"^[A-Za-z0-9_\-]+$", speaker_id):
        raise ValueError(f"Invalid characters in speaker_id '{speaker_id}'. Alphanumeric, underscores, and dashes allowed.")

    is_dry_run = dry_run or (not confirm)

    # 2. Inspect consent ledger
    ledger = load_consent_ledger(target_ledger_path)
    if speaker_id not in ledger:
        return {
            "status": "ERROR_SPEAKER_NOT_FOUND",
            "speaker_id": speaker_id,
            "message": f"Speaker ID '{speaker_id}' not found in consent ledger at {target_ledger_path}."
        }

    speaker_info = ledger[speaker_id]
    already_revoked = not speaker_info.get("consent_given", True)

    # 3. Discover manifest records and audio files
    manifest_rows_indices = []
    files_to_quarantine = []
    df_manifest = pd.DataFrame()

    if target_manifest_path.exists():
        try:
            df_manifest = pd.read_csv(target_manifest_path)
            if "speaker_id" in df_manifest.columns:
                matching_mask = (df_manifest["speaker_id"] == speaker_id)
                manifest_rows_indices = df_manifest[matching_mask].index.tolist()

                for idx in manifest_rows_indices:
                    row = df_manifest.loc[idx]
                    raw_fp = row.get("raw_filepath")
                    proc_fp = row.get("processed_path")
                    if raw_fp and pd.notna(raw_fp):
                        p = Path(str(raw_fp))
                        if p.exists() and p not in files_to_quarantine:
                            files_to_quarantine.append(p)
                    if proc_fp and pd.notna(proc_fp):
                        p = Path(str(proc_fp))
                        if p.exists() and p not in files_to_quarantine:
                            files_to_quarantine.append(p)
        except Exception as e:
            print(f"[WARN] Error inspecting manifest {target_manifest_path}: {e}")

    # Also scan disk raw and std directories for any orphan files belonging to speaker
    spk_raw_dir = target_raw_base / speaker_id
    if spk_raw_dir.exists() and spk_raw_dir.is_dir():
        for f in spk_raw_dir.rglob("*"):
            if f.is_file() and f not in files_to_quarantine:
                files_to_quarantine.append(f)

    if target_std_base.exists():
        for f in target_std_base.rglob(f"*{speaker_id}*"):
            if f.is_file() and f not in files_to_quarantine:
                files_to_quarantine.append(f)

    withdrawal_dest_dir = target_quarantine_dir / "withdrawn_speakers" / speaker_id

    # 4. Handle Dry-Run Mode
    if is_dry_run:
        return {
            "status": "DRY_RUN",
            "speaker_id": speaker_id,
            "already_revoked": already_revoked,
            "reason": reason,
            "manifest_records_to_update": len(manifest_rows_indices),
            "files_to_quarantine_count": len(files_to_quarantine),
            "files_to_quarantine": [str(p).replace("\\", "/") for p in files_to_quarantine],
            "quarantine_destination": str(withdrawal_dest_dir).replace("\\", "/"),
            "notice": (
                "DRY RUN COMPLETE: Zero files moved, zero records modified. "
                "Pass --confirm to permanently execute consent withdrawal."
            )
        }

    # 5. Handle Confirmed Execution (Idempotent)
    relocated_files = []
    withdrawal_dest_dir.mkdir(parents=True, exist_ok=True)

    # 5a. Relocate audio files to quarantine without deleting
    for src_file in files_to_quarantine:
        if not src_file.exists():
            continue
        # Preserve relative distinction between raw and standardized
        if "audio_raw" in str(src_file) or str(target_raw_base) in str(src_file):
            dest_subdir = withdrawal_dest_dir / "audio_raw"
        elif "audio_16k_standardized" in str(src_file) or str(target_std_base) in str(src_file):
            dest_subdir = withdrawal_dest_dir / "audio_16k_standardized"
        else:
            dest_subdir = withdrawal_dest_dir / "audio_other"

        dest_subdir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_subdir / src_file.name

        # If dest file already exists, avoid overwriting with a suffix
        if dest_file.exists() and dest_file != src_file:
            nonce = secrets.token_hex(2)
            dest_file = dest_subdir / f"{src_file.stem}_{nonce}{src_file.suffix}"

        try:
            shutil.move(str(src_file), str(dest_file))
            relocated_files.append({
                "original": str(src_file).replace("\\", "/"),
                "quarantined": str(dest_file).replace("\\", "/")
            })
        except Exception as e:
            print(f"[ERROR] Failed to move {src_file} -> {dest_file}: {e}")

    # Remove empty parent directory in raw_base if left behind
    if spk_raw_dir.exists() and not any(spk_raw_dir.iterdir()):
        try:
            spk_raw_dir.rmdir()
        except Exception:
            pass

    # 5b. Update Manifest records
    updated_manifest_rows = 0
    if not df_manifest.empty and manifest_rows_indices:
        df_manifest["status"] = df_manifest["status"].astype(object)
        df_manifest["exclusion_reason"] = df_manifest["exclusion_reason"].astype(object)
        df_manifest.loc[manifest_rows_indices, "status"] = "REVOKED_WITHDRAWN"
        df_manifest.loc[manifest_rows_indices, "exclusion_reason"] = f"consent_revoked: {reason}"
        df_manifest.to_csv(target_manifest_path, index=False, encoding="utf-8")
        updated_manifest_rows = len(manifest_rows_indices)

    # 5c. Append to Quarantine Log
    target_quarantine_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    quarantine_log_rows = []
    for rel in relocated_files:
        quarantine_log_rows.append({
            "timestamp": timestamp_iso,
            "speaker_id": speaker_id,
            "reason": f"consent_revocation: {reason}",
            "original_path": rel["original"],
            "quarantined_path": rel["quarantined"],
            "status": "REVOKED_WITHDRAWN"
        })

    if quarantine_log_rows:
        append_to_csv(pd.DataFrame(quarantine_log_rows), target_quarantine_log)

    # 5d. Update Consent Ledger
    audit_id = f"REV_{secrets.token_hex(4).upper()}"
    ledger[speaker_id]["consent_given"] = False
    ledger[speaker_id]["revoked_at"] = timestamp_iso
    ledger[speaker_id]["revocation_reason"] = reason
    ledger[speaker_id]["revocation_audit_id"] = audit_id
    save_consent_ledger(ledger, target_ledger_path)

    execution_status = "ALREADY_REVOKED" if already_revoked and not relocated_files else "REVOKED_CONFIRMED"

    return {
        "status": execution_status,
        "speaker_id": speaker_id,
        "audit_id": audit_id,
        "revoked_at": timestamp_iso,
        "reason": reason,
        "manifest_rows_updated": updated_manifest_rows,
        "files_quarantined_count": len(relocated_files),
        "quarantined_files": relocated_files,
        "quarantine_dest_dir": str(withdrawal_dest_dir).replace("\\", "/"),
        "governance_notice": (
            "Consent revocation executed successfully. All associated audio quarantined to "
            "withdrawn_speakers/ without deletion, and manifest status updated to REVOKED_WITHDRAWN. "
            "NOTICE: Revocation excludes this data from future training, testing, and distribution. "
            "Previously trained neural network checkpoints cannot automatically unlearn learned parameters; "
            "full model retraining is required to eliminate parameter-level representation."
        )
    }


def audit_speaker_residual_artifacts(
    speaker_id: str,
    manifest_path: Optional[Path] = None,
    split_manifest_paths: Optional[List[Path]] = None,
    raw_base_dir: Optional[Path] = None,
    std_base_dir: Optional[Path] = None,
    quarantine_dir: Optional[Path] = None,
    feature_cache_dirs: Optional[List[Path]] = None,
    backup_dirs: Optional[List[Path]] = None,
    model_checkpoint_dirs: Optional[List[Path]] = None,
    quarantine_log_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Read-only dry-run audit utility to inspect and report potential residual artifacts
    associated with a withdrawn pseudonymous speaker.

    Guarantees:
      - Read-only: Zero file modifications, zero deletions, zero moves.
      - Privacy-preserving: Does not print audio content or consent disclosure texts.
      - Transparent lineage: Distinguishes confirmed associations from uncertain/fuzzy matches.
      - Implementation gap reporting: Reports archives and caches with missing lineage.
    """
    if not re.match(r"^[A-Za-z0-9_\-]+$", speaker_id):
        raise ValueError(f"Invalid characters in speaker_id '{speaker_id}'. Alphanumeric, underscores, and dashes allowed.")

    target_manifest_path = manifest_path if manifest_path is not None else DEFAULT_MANIFEST_PATH
    target_raw_base = raw_base_dir if raw_base_dir is not None else DEFAULT_RAW_DIR
    target_std_base = std_base_dir if std_base_dir is not None else DEFAULT_STD_DIR
    target_quarantine_dir = quarantine_dir if quarantine_dir is not None else DEFAULT_QUARANTINE_DIR
    target_quarantine_log = quarantine_log_path if quarantine_log_path is not None else (target_quarantine_dir / "quarantine_log.csv")

    default_split_paths = [
        Path("data/manifests/train_manifest.csv"),
        Path("data/manifests/val_manifest.csv"),
        Path("data/manifests/test_manifest.csv"),
        Path("data/manifests/train_manifest_v2.csv"),
        Path("data/manifests/val_manifest_v2.csv"),
        Path("data/manifests/test_manifest_v2.csv"),
    ]
    splits_to_check = split_manifest_paths if split_manifest_paths is not None else default_split_paths
    feat_dirs = feature_cache_dirs if feature_cache_dirs is not None else [Path("data/features"), Path(".cache"), Path("scratch")]
    bkp_dirs = backup_dirs if backup_dirs is not None else [Path("data/backups")]
    model_dirs = model_checkpoint_dirs if model_checkpoint_dirs is not None else [Path("models/checkpoints"), Path("models/exported")]

    known_recording_ids = set()
    manifest_records_summary: Dict[str, Any] = {"total": 0, "statuses": {}, "confirmed_paths": []}

    # 1. Primary manifest audit
    if target_manifest_path.exists():
        try:
            df = pd.read_csv(target_manifest_path)
            if "speaker_id" in df.columns:
                matches = df[df["speaker_id"] == speaker_id]
                manifest_records_summary["total"] = len(matches)
                manifest_records_summary["statuses"] = matches["status"].value_counts().to_dict() if "status" in matches.columns else {}
                if "recording_id" in matches.columns:
                    for r_id in matches["recording_id"].dropna():
                        known_recording_ids.add(str(r_id).strip())
                for col in ["raw_filepath", "processed_path"]:
                    if col in matches.columns:
                        for p in matches[col].dropna():
                            manifest_records_summary["confirmed_paths"].append(str(p).replace("\\", "/"))
        except Exception as e:
            manifest_records_summary["error"] = str(e)

    # 2. Split manifests audit (confirmed)
    split_manifest_findings = []
    for sp_path in splits_to_check:
        if sp_path.exists():
            try:
                sp_df = pd.read_csv(sp_path)
                if "speaker_id" in sp_df.columns:
                    sp_matches = sp_df[sp_df["speaker_id"] == speaker_id]
                    if len(sp_matches) > 0:
                        split_manifest_findings.append({
                            "manifest_path": str(sp_path).replace("\\", "/"),
                            "matching_records": len(sp_matches),
                            "status": "CONFIRMED_RESIDUAL_SPLIT_RECORD"
                        })
            except Exception:
                pass

    # 3. Active audio files on disk (confirmed)
    active_audio_files = []
    spk_raw_dir = target_raw_base / speaker_id
    if spk_raw_dir.exists() and spk_raw_dir.is_dir():
        for f in spk_raw_dir.rglob("*"):
            if f.is_file():
                active_audio_files.append({"path": str(f).replace("\\", "/"), "category": "raw_audio", "match_confidence": "CONFIRMED_DIRECTORY"})
    if target_std_base.exists() and target_std_base.is_dir():
        for f in target_std_base.rglob(f"*{speaker_id}*"):
            if f.is_file():
                active_audio_files.append({"path": str(f).replace("\\", "/"), "category": "standardized_audio", "match_confidence": "CONFIRMED_FILENAME"})

    # 4. Quarantined audio files
    quarantined_audio_files = []
    quar_spk_dir = target_quarantine_dir / "withdrawn_speakers" / speaker_id
    if quar_spk_dir.exists() and quar_spk_dir.is_dir():
        for f in quar_spk_dir.rglob("*"):
            if f.is_file():
                quarantined_audio_files.append(str(f).replace("\\", "/"))

    # 5. Derived feature caches (.npy, spectrograms)
    feature_cache_findings = []
    unresolved_cache_lineage_count = 0
    for fd in feat_dirs:
        if fd.exists() and fd.is_dir():
            for f in fd.rglob("*.npy"):
                f_name = f.name
                matched_rec = any(r_id in f_name for r_id in known_recording_ids)
                if matched_rec:
                    feature_cache_findings.append({
                        "path": str(f).replace("\\", "/"),
                        "category": "feature_cache_npy",
                        "match_confidence": "CONFIRMED_RECORDING_LINEAGE"
                    })
                elif speaker_id in f_name or speaker_id in str(f):
                    feature_cache_findings.append({
                        "path": str(f).replace("\\", "/"),
                        "category": "feature_cache_npy",
                        "match_confidence": "UNCERTAIN_FILENAME_MATCH"
                    })
                else:
                    unresolved_cache_lineage_count += 1

    # 6. Backup archives
    backup_findings = []
    for bd in bkp_dirs:
        if bd.exists() and bd.is_dir():
            for archive_ext in ["*.tar.gz", "*.zip", "*.tar", "*.tgz"]:
                for arch in bd.glob(archive_ext):
                    backup_findings.append({
                        "archive_path": str(arch).replace("\\", "/"),
                        "category": "cold_backup_archive",
                        "lineage_status": "UNRESOLVED_ARCHIVE_LINEAGE",
                        "operational_note": (
                            "Immutable cold backup archives cannot be mutated; "
                            "destruction requires administrative policy SOP upon retention expiry."
                        )
                    })

    # 7. Model checkpoints
    model_findings = []
    for md in model_dirs:
        if md.exists() and md.is_dir():
            for model_ext in ["*.pt", "*.onnx", "*.pth", "*.bin"]:
                for m_file in md.rglob(model_ext):
                    model_findings.append({
                        "checkpoint_path": str(m_file).replace("\\", "/"),
                        "category": "trained_model_weights",
                        "lineage_status": "UNRESOLVABLE_CHECKPOINT_WEIGHTS",
                        "operational_note": (
                            "Neural network weights cannot unlearn vocal representations through deletion; "
                            "complete retraining from scratch is required under GATE-05."
                        )
                    })

    # 8. Quarantine log entries
    quarantine_log_matches = 0
    if target_quarantine_log.exists():
        try:
            q_df = pd.read_csv(target_quarantine_log)
            if "speaker_id" in q_df.columns:
                quarantine_log_matches = len(q_df[q_df["speaker_id"] == speaker_id])
        except Exception:
            pass

    return {
        "status": "RESIDUAL_AUDIT_COMPLETE",
        "speaker_id": speaker_id,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "files_modified": 0,
        "files_deleted": 0,
        "primary_manifest": manifest_records_summary,
        "split_manifests_affected": split_manifest_findings,
        "active_audio_files": active_audio_files,
        "quarantined_audio_files": quarantined_audio_files,
        "derived_feature_caches": feature_cache_findings,
        "unresolved_cache_lineage_count": unresolved_cache_lineage_count,
        "cold_backup_archives": backup_findings,
        "model_checkpoints": model_findings,
        "quarantine_log_entries_count": quarantine_log_matches,
        "governance_summary": {
            "confirmed_residuals_count": len(active_audio_files) + len(split_manifest_findings) + sum(1 for f in feature_cache_findings if f["match_confidence"] == "CONFIRMED_RECORDING_LINEAGE"),
            "uncertain_matches_count": sum(1 for f in feature_cache_findings if f["match_confidence"] == "UNCERTAIN_FILENAME_MATCH"),
            "quarantined_audio_count": len(quarantined_audio_files),
            "implementation_gaps": [
                "Feature caches without embedded recording lineage cannot be individually purged (cache flushing SOP required).",
                "Cold backup archives are immutable; cannot be selectively purged without re-archiving.",
                "Trained model weights cannot unlearn vocal traits; retraining budget commitment required."
            ]
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese KWS — Operator Consent Revocation and Data Quarantine CLI"
    )
    parser.add_argument(
        "--speaker-id", type=str, required=True,
        help="Registered pseudonymous speaker identifier to withdraw (e.g. SPK_H_N_8f2a1b)"
    )
    parser.add_argument(
        "--reason", type=str, default="participant_withdrawal",
        help="Reason for consent withdrawal"
    )
    parser.add_argument(
        "--confirm", action="store_true", default=False,
        help="Explicit confirmation to execute permanent data revocation and quarantine"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Simulate revocation and display affected files without modifying disk"
    )
    parser.add_argument(
        "--audit-residuals", action="store_true", default=False,
        help="Run read-only dry-run audit of residual artifacts for the speaker"
    )
    parser.add_argument(
        "--manifest-path", type=str, default=None,
        help="Override path to manifest CSV"
    )
    parser.add_argument(
        "--ledger-path", type=str, default=None,
        help="Override path to consent ledger JSON"
    )
    parser.add_argument(
        "--quarantine-dir", type=str, default=None,
        help="Override path to quarantine root directory"
    )

    args = parser.parse_args()

    manifest_p = Path(args.manifest_path) if args.manifest_path else None
    ledger_p = Path(args.ledger_path) if args.ledger_path else None
    quarantine_d = Path(args.quarantine_dir) if args.quarantine_dir else None

    if args.audit_residuals:
        audit_res = audit_speaker_residual_artifacts(
            speaker_id=args.speaker_id,
            manifest_path=manifest_p,
            quarantine_dir=quarantine_d
        )
        print("\n" + "=" * 65)
        print("RESIDUAL ARTIFACTS DRY-RUN AUDIT REPORT")
        print("=" * 65)
        print(json.dumps(audit_res, indent=2))
        print("=" * 65 + "\n")
        sys.exit(0)

    result = execute_consent_revocation(
        speaker_id=args.speaker_id,
        reason=args.reason,
        confirm=args.confirm,
        dry_run=args.dry_run,
        manifest_path=manifest_p,
        ledger_path=ledger_p,
        quarantine_dir=quarantine_d
    )

    print("\n" + "=" * 65)
    print("CONSENT REVOCATION EXECUTION SUMMARY")
    print("=" * 65)
    print(json.dumps(result, indent=2))
    print("=" * 65 + "\n")

    if result.get("status") in ["DRY_RUN", "REVOKED_CONFIRMED", "ALREADY_REVOKED"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
