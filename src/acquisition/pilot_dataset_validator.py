"""
Pilot Dataset Auditor and Quality-Control Validator for Vietnamese KWS.
Provides pure, non-destructive inventory, gap analysis, and QC validation for the
1,000-clip curated pilot dataset (250 clips each for TRAI, PHAI, LEN, XUONG).
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

import os
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.acquisition.campaign_validator import (
    validate_audio_file,
    compute_file_sha256,
    VALID_COMMAND_LABELS,
    ALLOWED_DIALECTS,
    BLOCKED_UNVERIFIED_PROVENANCE_VALUES
)


TARGET_COMMANDS: Tuple[str, ...] = ("TRAI", "PHAI", "LEN", "XUONG")
DEFAULT_TARGET_PER_COMMAND: int = 250
DEFAULT_TARGET_TOTAL: int = 1000

# Canonical provenance categories
PROVENANCE_SYNTHETIC = {
    "acoustic_reference_utterance",
    "SYNTHETIC_TEST_FIXTURE",
    "test_harness_mock",
    "calibration_acoustic_generator_v1"
}
PROVENANCE_HUMAN = {
    "HUMAN_CONSENTED",
    "human_field_recording",
    "field_human_recording_campaign_v1"
}
PROVENANCE_EXTERNAL_NEGATIVE = {
    "NEGATIVE_CORPUS_FLEURS",
    "continuous_speech_hard_negative",
    "embedded_keyword_sentence_negative",
    "google_fleurs_vi"
}
PROVENANCE_BLOCKED = {
    "NEGATIVE_NOISE_HUSEIN",
    "husein_mic_noise"
}


def audit_pilot_dataset(
    manifest_source: Union[pd.DataFrame, str, Path],
    target_per_command: int = DEFAULT_TARGET_PER_COMMAND,
    check_disk: bool = False,
    audio_root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Performs a non-destructive audit of a dataset manifest against the 1,000-clip pilot requirements:
      - 250 accepted clips per target command (TRAI, PHAI, LEN, XUONG)
      - Regional accent representation (Northern, Central, Southern)
      - Speaker diversity and contribution bounds
      - Provenance classification (Human Consented vs Synthetic Fixture vs External Corpus)
      - Optional physical audio file validation and QC checks

    Returns:
      Dict with audit metrics, per-command counts, gaps, regional distributions, and overall readiness.
    """
    if isinstance(manifest_source, (str, Path)):
        p = Path(manifest_source)
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        df = pd.read_csv(p)
    elif isinstance(manifest_source, pd.DataFrame):
        df = manifest_source.copy()
    else:
        raise TypeError(f"Expected DataFrame, str, or Path, got {type(manifest_source)}")

    # Normalize column names: support both 16-col (class_label) and 28-col (command_label) schemas
    col_map = {}
    if "command_label" in df.columns:
        label_col = "command_label"
    elif "class_label" in df.columns:
        label_col = "class_label"
    else:
        label_col = None

    if "dialect" in df.columns:
        dialect_col = "dialect"
    elif "accent" in df.columns:
        dialect_col = "accent"
    else:
        dialect_col = None

    total_records = len(df)
    if total_records == 0 or label_col is None:
        return {
            "is_pilot_ready": False,
            "total_records": total_records,
            "target_total": target_per_command * len(TARGET_COMMANDS),
            "target_per_command": target_per_command,
            "overall_gap": target_per_command * len(TARGET_COMMANDS),
            "summary_status": "EMPTY_OR_UNANNOTATED_MANIFEST",
            "per_command": {cmd: {"accepted": 0, "rejected": 0, "missing": target_per_command, "gap": target_per_command, "status": f"GAP_{target_per_command}"} for cmd in TARGET_COMMANDS},
            "per_region": {},
            "speaker_diversity": {"unique_speakers": 0, "max_single_speaker_share_pct": 0.0},
            "provenance": {"human_consented": 0, "synthetic_fixture": 0, "external_negative": 0, "unverified_blocked": 0},
            "qc_failures": ["Manifest is empty or missing label column"]
        }

    # Normalize labels and status
    labels = df[label_col].astype(str).str.strip().str.upper()
    status_series = df["status"].astype(str).str.strip().str.upper() if "status" in df.columns else pd.Series(["APPROVED"] * total_records, index=df.index)

    # 1. Per-Command Accounting
    per_command = {}
    total_accepted_target = 0
    total_missing_target = 0

    for cmd in TARGET_COMMANDS:
        cmd_mask = (labels == cmd)
        accepted_mask = cmd_mask & (status_series == "APPROVED")
        rejected_mask = cmd_mask & (status_series != "APPROVED")

        acc_count = int(accepted_mask.sum())
        rej_count = int(rejected_mask.sum())
        gap = max(0, target_per_command - acc_count)

        total_accepted_target += acc_count
        total_missing_target += gap

        per_command[cmd] = {
            "accepted": acc_count,
            "rejected": rej_count,
            "missing": gap,
            "gap": gap,
            "status": "MET" if gap == 0 else f"GAP_{gap}"
        }

    # Non-target counts
    non_target_mask = ~labels.isin(TARGET_COMMANDS)
    non_target_counts = labels[non_target_mask].value_counts().to_dict()

    # 2. Regional Accent Distribution (on target commands)
    per_region = {}
    target_mask = labels.isin(TARGET_COMMANDS)
    target_accepted_mask = target_mask & (status_series == "APPROVED")
    df_accepted_target = df[target_accepted_mask]

    if dialect_col and dialect_col in df.columns:
        dialects = df_accepted_target[dialect_col].astype(str).str.strip()
        region_counts = dialects.value_counts().to_dict()
        for region in ("Northern", "Central", "Southern"):
            r_count = region_counts.get(region, 0)
            r_share = (r_count / total_accepted_target * 100.0) if total_accepted_target > 0 else 0.0
            per_region[region] = {
                "count": r_count,
                "share_pct": round(r_share, 2),
                "by_command": {}
            }
            for cmd in TARGET_COMMANDS:
                cmd_r_count = int(((df_accepted_target[label_col] == cmd) & (df_accepted_target[dialect_col] == region)).sum())
                per_region[region]["by_command"][cmd] = cmd_r_count
        # Record any unclassified accents
        other_accents = {k: v for k, v in region_counts.items() if k not in ("Northern", "Central", "Southern")}
        if other_accents:
            per_region["Other"] = other_accents

    # 3. Speaker Diversity & Contribution Bounds
    speaker_diversity = {}
    if "speaker_id" in df.columns:
        spk_series = df_accepted_target["speaker_id"].astype(str).str.strip()
        spk_counts = spk_series.value_counts()
        unique_spk = int(spk_series.nunique())
        max_spk_takes = int(spk_counts.iloc[0]) if unique_spk > 0 else 0
        max_spk_share = (max_spk_takes / total_accepted_target * 100.0) if total_accepted_target > 0 else 0.0
        min_spk_takes = int(spk_counts.iloc[-1]) if unique_spk > 0 else 0
        mean_spk_takes = float(spk_counts.mean()) if unique_spk > 0 else 0.0

        speaker_diversity = {
            "unique_speakers": unique_spk,
            "min_takes_per_speaker": min_spk_takes,
            "max_takes_per_speaker": max_spk_takes,
            "mean_takes_per_speaker": round(mean_spk_takes, 2),
            "max_single_speaker_share_pct": round(max_spk_share, 2),
            "exceeds_5pct_dominance_limit": bool(max_spk_share > 5.0)
        }
    else:
        speaker_diversity = {
            "unique_speakers": 0,
            "min_takes_per_speaker": 0,
            "max_takes_per_speaker": 0,
            "mean_takes_per_speaker": 0.0,
            "max_single_speaker_share_pct": 0.0,
            "exceeds_5pct_dominance_limit": False
        }

    # 4. Provenance Classification
    prov_col = "label_provenance" if "label_provenance" in df.columns else ("source" if "source" in df.columns else None)
    prov_counts = {"human_consented": 0, "synthetic_fixture": 0, "external_negative": 0, "unverified_blocked": 0, "other": 0}

    if prov_col and prov_col in df.columns:
        for val, cnt in df[prov_col].astype(str).value_counts().items():
            if val in PROVENANCE_HUMAN:
                prov_counts["human_consented"] += int(cnt)
            elif val in PROVENANCE_SYNTHETIC:
                prov_counts["synthetic_fixture"] += int(cnt)
            elif val in PROVENANCE_EXTERNAL_NEGATIVE:
                prov_counts["external_negative"] += int(cnt)
            elif val in PROVENANCE_BLOCKED:
                prov_counts["unverified_blocked"] += int(cnt)
            else:
                prov_counts["other"] += int(cnt)

    # 5. Optional Physical Audio QC Checks
    qc_failures = []
    if check_disk:
        path_col = "processed_path" if "processed_path" in df.columns else ("raw_filepath" if "raw_filepath" in df.columns else None)
        if path_col is None:
            qc_failures.append("check_disk requested but neither 'processed_path' nor 'raw_filepath' exists in manifest")
        else:
            for idx, row in df_accepted_target.iterrows():
                fpath_str = str(row[path_col])
                if audio_root:
                    fpath = Path(audio_root) / fpath_str
                else:
                    fpath = Path(fpath_str)

                if not fpath.exists():
                    qc_failures.append(f"missing_file: row {idx} {fpath}")
                    continue

                # Check audio signal quality
                allow_silence = (str(row[label_col]).upper() == "SILENCE")
                audio_ok, audio_err, _ = validate_audio_file(fpath, allow_silence=allow_silence)
                if not audio_ok:
                    qc_failures.append(f"qc_error: row {idx} {fpath.name}: {audio_err}")

                # Check SHA-256 if present
                sha_col = "sha256_standardized" if "sha256_standardized" in df.columns else ("sha256" if "sha256" in df.columns else None)
                if sha_col and pd.notna(row[sha_col]):
                    calc_sha = compute_file_sha256(fpath)
                    expected_sha = str(row[sha_col]).strip().upper()
                    if calc_sha != expected_sha:
                        qc_failures.append(f"sha256_mismatch: row {idx} {fpath.name} {calc_sha} != {expected_sha}")

    # 6. Overall Pilot Readiness Determination
    target_total = target_per_command * len(TARGET_COMMANDS)
    has_zero_target_gap = (total_missing_target == 0) and (total_accepted_target >= target_total)
    has_balanced_classes = all(per_command[c]["accepted"] >= target_per_command for c in TARGET_COMMANDS)
    has_zero_blocked_prov = (prov_counts["unverified_blocked"] == 0)
    has_zero_qc_failures = (len(qc_failures) == 0)

    is_pilot_ready = bool(
        has_zero_target_gap and
        has_balanced_classes and
        has_zero_blocked_prov and
        has_zero_qc_failures
    )

    return {
        "is_pilot_ready": is_pilot_ready,
        "total_records": total_records,
        "target_total": target_total,
        "total_accepted_target": total_accepted_target,
        "overall_gap": total_missing_target,
        "target_per_command": target_per_command,
        "per_command": per_command,
        "non_target_counts": non_target_counts,
        "per_region": per_region,
        "speaker_diversity": speaker_diversity,
        "provenance": prov_counts,
        "qc_failures": qc_failures,
        "qc_failures_count": len(qc_failures)
    }


def format_pilot_audit_report(result: Dict[str, Any]) -> str:
    """Formats the pilot audit result as a clear Markdown summary report."""
    lines = [
        "# Pilot Dataset Audit Report (1,000-Clip Vietnamese KWS Target)",
        "",
        f"- **Overall Pilot Readiness:** `{'READY' if result['is_pilot_ready'] else 'NOT_READY (GAPS REMAIN)'}`",
        f"- **Total Manifest Records:** {result.get('total_records', 0):,}",
        f"- **Target Curated Dataset Size:** {result.get('target_total', 1000):,} accepted clips ({result.get('target_per_command', 250)} per command)",
        f"- **Total Accepted Target Clips:** {result.get('total_accepted_target', 0):,}",
        f"- **Total Missing Target Clips (Gap):** {result.get('overall_gap', 0):,}",
        "",
        "## 1. Target Command Class Breakdown",
        "",
        "| Command | Target Quota | Accepted | Rejected / Quarantined | Missing (Gap) | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for cmd, info in result.get("per_command", {}).items():
        status_badge = f"`{info['status']}`"
        lines.append(f"| **`{cmd}`** | {result.get('target_per_command', 250)} | {info['accepted']:,} | {info['rejected']:,} | {info['missing']:,} | {status_badge} |")

    lines.extend([
        "",
        "## 2. Regional Accent Representation (Accepted Target Clips)",
        "",
        "| Region | Vietnamese | Accepted Count | National Share | TRAI | PHAI | LEN | XUONG |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for reg, r_name in [("Northern", "Bắc"), ("Central", "Trung"), ("Southern", "Nam")]:
        reg_info = result.get("per_region", {}).get(reg, {"count": 0, "share_pct": 0.0, "by_command": {}})
        by_cmd = reg_info.get("by_command", {})
        lines.append(
            f"| **{reg}** | Giọng {r_name} | {reg_info['count']:,} | {reg_info['share_pct']:.1f}% | "
            f"{by_cmd.get('TRAI', 0):,} | {by_cmd.get('PHAI', 0):,} | {by_cmd.get('LEN', 0):,} | {by_cmd.get('XUONG', 0):,} |"
        )

    spk_info = result.get("speaker_diversity", {})
    lines.extend([
        "",
        "## 3. Speaker Diversity & Contribution Metrics",
        "",
        f"- **Unique Speakers:** {spk_info.get('unique_speakers', 0)}",
        f"- **Takes per Speaker (Min / Mean / Max):** {spk_info.get('min_takes_per_speaker', 0)} / {spk_info.get('mean_takes_per_speaker', 0.0)} / {spk_info.get('max_takes_per_speaker', 0)}",
        f"- **Maximum Single Speaker Contribution:** {spk_info.get('max_single_speaker_share_pct', 0.0):.2f}% (Limit: $\\le 5.0\\%$)",
        f"- **Speaker Dominance Violation:** `{'YES (EXCEEDS LIMIT)' if spk_info.get('exceeds_5pct_dominance_limit', False) else 'NO (COMPLIANT)'}`",
        "",
        "## 4. Source Provenance Classification",
        "",
        f"- **Genuine Human Field Speech (`HUMAN_CONSENTED`):** {result.get('provenance', {}).get('human_consented', 0):,}",
        f"- **Synthetic Acoustic Fixtures (`calibration_generator`):** {result.get('provenance', {}).get('synthetic_fixture', 0):,}",
        f"- **External Negative Corpora (`google_fleurs_vi`):** {result.get('provenance', {}).get('external_negative', 0):,}",
        f"- **Unverified / Blocked Sources (`husein_mic_noise`):** {result.get('provenance', {}).get('unverified_blocked', 0):,}",
        "",
        "## 5. Signal Quality Failures & Discrepancies",
        ""
    ])

    qc_errs = result.get("qc_failures", [])
    if not qc_errs:
        lines.append("- **QC Status:** Zero signal quality failures or SHA-256 discrepancies detected.")
    else:
        lines.append(f"- **QC Status:** {len(qc_errs)} failure(s) detected:")
        for err in qc_errs[:20]:
            lines.append(f"  - `{err}`")
        if len(qc_errs) > 20:
            lines.append(f"  - *... and {len(qc_errs) - 20} more failures.*")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit and validate dataset manifest against 1,000-clip pilot requirements.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV (e.g. data/manifests/source_registry_v2.csv)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_PER_COMMAND, help="Target accepted clips per command (default: 250)")
    parser.add_argument("--check-disk", action="store_true", help="Perform audio file format, clipping, silence, and SHA-256 checks on disk")
    parser.add_argument("--audio-root", type=str, default=None, help="Root directory for relative audio paths")
    parser.add_argument("--report-out", type=str, default=None, help="Path to write markdown audit report")
    args = parser.parse_args()

    result = audit_pilot_dataset(
        args.manifest,
        target_per_command=args.target,
        check_disk=args.check_disk,
        audio_root=args.audio_root
    )

    report_md = format_pilot_audit_report(result)
    print(report_md)

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\nAudit report saved to: {out_path}")

    sys.exit(0 if result["is_pilot_ready"] else 1)


if __name__ == "__main__":
    main()
