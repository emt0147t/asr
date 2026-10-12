"""
Field Recording Campaign Auditor and Statistics Reporter for Vietnamese KWS.
Analyzes data/raw/field_recordings/ metadata, validates quality gates,
computes progress toward target collection budgets, and generates:
reports/field_recording_campaign_audit.md.
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

import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.acquisition.speaker_consent import load_consent_ledger
from src.acquisition.ingest_field_recordings import MANIFEST_PATH, QUARANTINE_LOG_PATH, VALID_COMMANDS

TARGET_PER_COMMAND = 1500  # Initial target budget from docs/dataset_protocol.md


def audit_field_campaign(report_path: Path = Path("reports/field_recording_campaign_audit.md")) -> str:
    print("=" * 65)
    print("AUDITING HUMAN FIELD RECORDING CAMPAIGN")
    print("=" * 65)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = load_consent_ledger()
    df_manifest = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
    df_quarantine = pd.read_csv(QUARANTINE_LOG_PATH) if QUARANTINE_LOG_PATH.exists() else pd.DataFrame()

    total_registered_speakers = len(ledger)
    consented_speakers = sum(1 for s in ledger.values() if s.get("consent_given", False))

    # Speaker dialect and gender breakdown
    dialect_counts = {"Northern": 0, "Central": 0, "Southern": 0}
    gender_counts = {"MALE": 0, "FEMALE": 0, "OTHER": 0}
    for s in ledger.values():
        d = s.get("dialect", "Northern")
        g = s.get("gender", "MALE").upper()
        if d in dialect_counts:
            dialect_counts[d] += 1
        if g in gender_counts:
            gender_counts[g] += 1

    total_approved = len(df_manifest)
    total_quarantined = len(df_quarantine)

    # Command breakdown
    cmd_counts = {cmd: 0 for cmd in VALID_COMMANDS.keys()}
    total_duration_sec = 0.0

    if not df_manifest.empty and "command_label" in df_manifest.columns:
        counts = df_manifest["command_label"].value_counts().to_dict()
        for c in cmd_counts.keys():
            cmd_counts[c] = counts.get(c, 0)
        total_duration_sec = float(df_manifest["raw_duration_sec"].sum()) if "raw_duration_sec" in df_manifest.columns else 0.0

    total_duration_hr = total_duration_sec / 3600.0

    # Devices & Environments
    devices_dict = df_manifest["device_model"].value_counts().to_dict() if not df_manifest.empty and "device_model" in df_manifest.columns else {}
    env_dict = df_manifest["environment"].value_counts().to_dict() if not df_manifest.empty and "environment" in df_manifest.columns else {}

    # Quarantine reasons
    quarantine_reasons = df_quarantine["exclusion_reason"].value_counts().to_dict() if not df_quarantine.empty and "exclusion_reason" in df_quarantine.columns else {}

    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report_md = f"""# Vietnamese KWS — Human Field Recording Campaign Audit Report

**Project:** Vietnamese Four-Word Keyword Spotting System (`TRAI`, `PHAI`, `LEN`, `XUONG`)  
**Audit Scope:** Dedicated Human Field Recording Campaign (Strictly segregated from synthetic data)  
**Governance Standard:** `docs/dataset_protocol.md` Sections 4 & 7  
**Audit Timestamp:** {timestamp_str}  
**Status:** **ACTIVE COLLECTION CAMPAIGN AUDIT**  

---

## 1. Executive Summary & Progress Toward Target Budget

To resolve the 25% Northern `TRAI` recall deficit identified in Phase 3, this campaign collects **genuine human voice recordings** with verified informed consent across Northern, Central, and Southern dialects.

- **Registered & Consented Human Speakers:** **{consented_speakers} speakers** ({dialect_counts['Northern']} Northern, {dialect_counts['Central']} Central, {dialect_counts['Southern']} Southern).
- **Total Approved Command Utterances:** **{total_approved:,} recordings**.
- **Total Audio Duration Collected:** **{total_duration_sec:.1f} seconds** ({total_duration_hr:.3f} hours).
- **Target Planning Budget:** {TARGET_PER_COMMAND:,} recordings per command ({TARGET_PER_COMMAND * 4:,} total).
- **Overall Campaign Completion:** **{(total_approved / (TARGET_PER_COMMAND * 4)) * 100:.2f}%**.
- **Quarantined / Rejected Takes:** **{total_quarantined} takes** (quality gates triggered).

---

## 2. Command Collection Progress & Class Balance

| Command Label | Vietnamese Word | Target Class ID | Approved Takes | Target Budget | Progress (%) | Acoustic Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`TRAI`** | `trái` | 0 | **{cmd_counts['TRAI']}** | {TARGET_PER_COMMAND:,} | {(cmd_counts['TRAI'] / TARGET_PER_COMMAND) * 100:.1f}% | Affricate onset burst `/tɕ/` priority focus |
| **`PHAI`** | `phải` | 1 | **{cmd_counts['PHAI']}** | {TARGET_PER_COMMAND:,} | {(cmd_counts['PHAI'] / TARGET_PER_COMMAND) * 100:.1f}% | Fricative `/f/` tone hỏi contrast |
| **`LEN`** | `lên` | 2 | **{cmd_counts['LEN']}** | {TARGET_PER_COMMAND:,} | {(cmd_counts['LEN'] / TARGET_PER_COMMAND) * 100:.1f}% | Lateral onset `/l/` |
| **`XUONG`** | `xuống` | 3 | **{cmd_counts['XUONG']}** | {TARGET_PER_COMMAND:,} | {(cmd_counts['XUONG'] / TARGET_PER_COMMAND) * 100:.1f}% | Sibilant onset `/s/` tone sắc |
| **TOTAL** | — | — | **{total_approved}** | **{TARGET_PER_COMMAND * 4:,}** | **{(total_approved / (TARGET_PER_COMMAND * 4)) * 100:.2f}%** | — |

---

## 3. Demographics, Regional Dialect & Acoustic Stratification

### 3.1 Regional Dialect Distribution
- **Northern Dialect (Giọng Bắc):** {dialect_counts['Northern']} speakers ({(dialect_counts['Northern'] / max(1, total_registered_speakers)) * 100:.1f}%) — **Primary Target for TRAI/PHAI Resolution**.
- **Central Dialect (Giọng Trung):** {dialect_counts['Central']} speakers ({(dialect_counts['Central'] / max(1, total_registered_speakers)) * 100:.1f}%).
- **Southern Dialect (Giọng Nam):** {dialect_counts['Southern']} speakers ({(dialect_counts['Southern'] / max(1, total_registered_speakers)) * 100:.1f}%).

### 3.2 Contributor Gender & Age Balance
- **Gender:** {gender_counts['MALE']} Male ({(gender_counts['MALE'] / max(1, total_registered_speakers)) * 100:.1f}%), {gender_counts['FEMALE']} Female ({(gender_counts['FEMALE'] / max(1, total_registered_speakers)) * 100:.1f}%), {gender_counts['OTHER']} Other.
- **Target Protocol Standard:** 45%–55% female, 45%–55% male across all age tiers.

### 3.3 Hardware Devices Represented
"""
    if devices_dict:
        for dev, cnt in devices_dict.items():
            report_md += f"- **{dev}:** {cnt} takes\n"
    else:
        report_md += "- *No approved hardware recordings cataloged yet.*\n"

    report_md += f"""
### 3.4 Acoustic Environments Represented
"""
    if env_dict:
        for e, cnt in env_dict.items():
            report_md += f"- **{e}:** {cnt} takes\n"
    else:
        report_md += "- *No approved environment recordings cataloged yet.*\n"

    report_md += f"""
---

## 4. Quality Gate Validation & Quarantine Audit

Every ingested file passes automated quality gates in `src/acquisition/ingest_field_recordings.py`:

| Validation Gate | Standard Threshold | Quarantine Count | Action / Policy |
| :--- | :---: | :---: | :--- |
| **Informed Consent** | Verified in `consent_ledger.json` | 0 | Strict gate: unconsented audio rejected immediately |
| **Duration Bounds** | $0.25\text{{ s}} \le \text{{duration}} \le 2.50\text{{ s}}$ | {quarantine_reasons.get('truncated_utterance', 0) + quarantine_reasons.get('excessive_duration', 0)} | Rejects truncated cuts or continuous sentence bleed |
| **Digital Clipping** | $\ge 3$ consecutive samples $\ge 0.999$ | {quarantine_reasons.get('digital_clipping_detected', 0)} | Rejects microphone distortion / pre-amp overload |
| **Abnormal Silence** | RMS Energy $\ge -45.0$ dBFS | {sum(v for k, v in quarantine_reasons.items() if 'silence' in k)} | Rejects unvoiced or missing commands |
| **Duplicate Collisions**| SHA-256 collision detection | {quarantine_reasons.get('duplicate_recording_sha256_match', 0)} | Rejects duplicate audio uploads |

Total Quarantined Records: **{total_quarantined}** (logged in `data/raw/field_recordings/quarantined/quarantine_log.csv`).

---

## 5. Outstanding Gaps & Next Steps to Mark Campaign Complete

1. **Recruitment & Quota Deficits:**
   - **Northern Native Speakers:** Target minimum 40–50 distinct Northern speakers (currently: {dialect_counts['Northern']}) to provide robust coverage of native `/tɕ/` affricate acoustics.
   - **Mobile Form Factors:** Expand collections across physical Android (Samsung, Xiaomi) and iOS (iPhone 12–15) microphones.
   - **Acoustic Noise Scenarios:** Collect takes in reverberant tiled rooms, moving street noise, and background HVAC rumble.
2. **Strict Manifest Partitioning:**
   - As new speakers are ingested, partition them strictly by pseudonymous `speaker_id` (70% train, 15% validation, 15% test).
   - Ensure the Phase 3 frozen held-out test manifest (`data/manifests/test_manifest_v2.csv`) is preserved without modification until a complete genuine human test benchmark is scheduled.
3. **Evidence Required to Mark Campaign Complete:**
   - At least 1,500 verified recordings per command class across $\ge 100$ consented human speakers.
   - 0 speaker leakage verified across all splits.
   - Re-training and evaluation demonstrating $> 90\%$ Northern `TRAI` recall on genuine human utterances.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[REPORT GENERATED] Field campaign audit saved to: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    audit_field_campaign()
