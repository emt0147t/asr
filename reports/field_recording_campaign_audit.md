# Vietnamese KWS — Human Field Recording Campaign Audit Report

**Project:** Vietnamese Four-Word Keyword Spotting System (`TRAI`, `PHAI`, `LEN`, `XUONG`)  
**Audit Scope:** Dedicated Human Field Recording Campaign (Strictly segregated from synthetic data)  
**Governance Standard:** `docs/dataset_protocol.md` Sections 4 & 7  
**Audit Timestamp:** 2026-10-12 17:08 UTC  
**Status:** **ACTIVE COLLECTION CAMPAIGN AUDIT**  

---

## 1. Executive Summary & Progress Toward Target Budget

To resolve the 25% Northern `TRAI` recall deficit identified in Phase 3, this campaign collects **genuine human voice recordings** with verified informed consent across Northern, Central, and Southern dialects.

- **Registered & Consented Human Speakers:** **1 speakers** (1 Northern, 0 Central, 0 Southern).
- **Total Approved Command Utterances:** **21 recordings**.
- **Total Audio Duration Collected:** **15.8 seconds** (0.004 hours).
- **Target Planning Budget:** 1,500 recordings per command (6,000 total).
- **Overall Campaign Completion:** **0.35%**.
- **Quarantined / Rejected Takes:** **20 takes** (quality gates triggered).

---

## 2. Command Collection Progress & Class Balance

| Command Label | Vietnamese Word | Target Class ID | Approved Takes | Target Budget | Progress (%) | Acoustic Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`TRAI`** | `trái` | 0 | **11** | 1,500 | 0.7% | Affricate onset burst `/tɕ/` priority focus |
| **`PHAI`** | `phải` | 1 | **4** | 1,500 | 0.3% | Fricative `/f/` tone hỏi contrast |
| **`LEN`** | `lên` | 2 | **6** | 1,500 | 0.4% | Lateral onset `/l/` |
| **`XUONG`** | `xuống` | 3 | **0** | 1,500 | 0.0% | Sibilant onset `/s/` tone sắc |
| **TOTAL** | — | — | **21** | **6,000** | **0.35%** | — |

---

## 3. Demographics, Regional Dialect & Acoustic Stratification

### 3.1 Regional Dialect Distribution
- **Northern Dialect (Giọng Bắc):** 1 speakers (100.0%) — **Primary Target for TRAI/PHAI Resolution**.
- **Central Dialect (Giọng Trung):** 0 speakers (0.0%).
- **Southern Dialect (Giọng Nam):** 0 speakers (0.0%).

### 3.2 Contributor Gender & Age Balance
- **Gender:** 0 Male (0.0%), 1 Female (100.0%), 0 Other.
- **Target Protocol Standard:** 45%–55% female, 45%–55% male across all age tiers.

### 3.3 Hardware Devices Represented
- **Realtek Laptop Mic:** 13 takes
- **Test Harness Device:** 8 takes

### 3.4 Acoustic Environments Represented
- **quiet_room:** 21 takes

---

## 4. Quality Gate Validation & Quarantine Audit

Every ingested file passes automated quality gates in `src/acquisition/ingest_field_recordings.py`:

| Validation Gate | Standard Threshold | Quarantine Count | Action / Policy |
| :--- | :---: | :---: | :--- |
| **Informed Consent** | Verified in `consent_ledger.json` | 0 | Strict gate: unconsented audio rejected immediately |
| **Duration Bounds** | $0.25	ext{ s} \le 	ext{duration} \le 2.50	ext{ s}$ | 0 | Rejects truncated cuts or continuous sentence bleed |
| **Digital Clipping** | $\ge 3$ consecutive samples $\ge 0.999$ | 4 | Rejects microphone distortion / pre-amp overload |
| **Abnormal Silence** | RMS Energy $\ge -45.0$ dBFS | 4 | Rejects unvoiced or missing commands |
| **Duplicate Collisions**| SHA-256 collision detection | 4 | Rejects duplicate audio uploads |

Total Quarantined Records: **20** (logged in `data/raw/field_recordings/quarantined/quarantine_log.csv`).

---

## 5. Outstanding Gaps & Next Steps to Mark Campaign Complete

1. **Recruitment & Quota Deficits:**
   - **Northern Native Speakers:** Target minimum 40–50 distinct Northern speakers (currently: 1) to provide robust coverage of native `/tɕ/` affricate acoustics.
   - **Mobile Form Factors:** Expand collections across physical Android (Samsung, Xiaomi) and iOS (iPhone 12–15) microphones.
   - **Acoustic Noise Scenarios:** Collect takes in reverberant tiled rooms, moving street noise, and background HVAC rumble.
2. **Strict Manifest Partitioning:**
   - As new speakers are ingested, partition them strictly by pseudonymous `speaker_id` (70% train, 15% validation, 15% test).
   - Ensure the Phase 3 frozen held-out test manifest (`data/manifests/test_manifest_v2.csv`) is preserved without modification until a complete genuine human test benchmark is scheduled.
3. **Evidence Required to Mark Campaign Complete:**
   - At least 1,500 verified recordings per command class across $\ge 100$ consented human speakers.
   - 0 speaker leakage verified across all splits.
   - Re-training and evaluation demonstrating $> 90\%$ Northern `TRAI` recall on genuine human utterances.
