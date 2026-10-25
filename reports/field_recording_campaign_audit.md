# Vietnamese KWS — Human Field Recording Campaign Audit Report

**Project:** Vietnamese Four-Word Keyword Spotting System (`TRAI`, `PHAI`, `LEN`, `XUONG`)  
**Audit Scope:** Dedicated Human Field Recording Campaign (Strictly segregated from synthetic data)  
**Governance Standard:** `docs/dataset_protocol.md` Sections 4 & 7  
**Audit Timestamp:** 2026-10-13 03:20 UTC
**Status:** **ACTIVE COLLECTION CAMPAIGN AUDIT**  

---

## 1. Executive Summary: Genuine Human Field Recordings vs. Test Harness

In accordance with strict dataset governance mandates, **test-harness and automated unit-test records are strictly segregated from genuine human recordings**.

### 1.1 Genuine Consented Human Field Speech
- **Consented Human Speakers:** **0 speakers** (0 Northern, 0 Central, 0 Southern).
- **Approved Genuine Utterances:** **0 recordings**.
- **Total Genuine Audio Duration:** **0.0 seconds** (0.000 hours).
- **Target Planning Budget:** 1,500 recordings per command (6,000 total).
- **Campaign Completion vs. Budget:** **0.00%**.

### 1.2 Automated Test Harness & Pipeline Validation Status
- **Test-Harness Validation Takes:** **104 recordings** (generated during automated pipeline verification).
- **Total Quarantined / Rejected Takes:** **87 takes** (quality gates triggered: clipping, silence, duration bounds, duplicate collisions).

---

## 2. Command Collection Progress & Class Balance (Genuine Audio)

| Command Label | Vietnamese Word | Target Class ID | Approved Takes | Target Budget | Progress (%) | Acoustic Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`TRAI`** | `trái` | 0 | **0** | 1,500 | 0.0% | Affricate onset burst `/tɕ/` priority focus |
| **`PHAI`** | `phải` | 1 | **0** | 1,500 | 0.0% | Fricative `/f/` tone hỏi contrast |
| **`LEN`** | `lên` | 2 | **0** | 1,500 | 0.0% | Lateral onset `/l/` |
| **`XUONG`** | `xuống` | 3 | **0** | 1,500 | 0.0% | Sibilant onset `/s/` tone sắc |
| **TOTAL** | — | — | **0** | **6,000** | **0.00%** | — |

---

## 3. Demographics, Regional Dialect & Acoustic Stratification

### 3.1 Regional Dialect Distribution (Genuine Speakers)
- **Northern Dialect (Giọng Bắc):** 0 speakers (0.0%) — **Primary Target for TRAI/PHAI Resolution**.
- **Central Dialect (Giọng Trung):** 0 speakers (0.0%).
- **Southern Dialect (Giọng Nam):** 0 speakers (0.0%).

### 3.2 Contributor Gender Balance
- **Gender:** 0 Male, 0 Female, 0 Other.
- **Target Standard:** 45%–55% female, 45%–55% male across all regional tiers.

### 3.3 Hardware Devices Represented (Genuine Takes)
- *No genuine human hardware recordings cataloged yet.*

### 3.4 Acoustic Environments Represented (Genuine Takes)
- *No genuine human environment recordings cataloged yet.*

---

## 4. Quality Gate Validation & Quarantine Audit

Every ingested file passes automated quality gates in `src/acquisition/ingest_field_recordings.py`:

| Validation Gate | Standard Threshold | Quarantine Count | Action / Policy |
| :--- | :---: | :---: | :--- |
| **Informed Consent** | Verified in `consent_ledger.json` | 0 | Strict gate: unconsented audio rejected immediately |
| **Duration Bounds** | $0.25\text{ s} \le \text{duration} \le 2.50\text{ s}$ | 0 | Rejects truncated cuts or continuous sentence bleed |
| **Digital Clipping** | $\ge 3$ consecutive samples $\ge 0.999$ | 15 | Rejects microphone distortion / pre-amp overload |
| **Abnormal Silence** | RMS Energy $\ge -45.0$ dBFS | 15 | Rejects unvoiced or missing commands |
| **Duplicate Collisions**| SHA-256 collision detection | 14 | Rejects duplicate audio uploads |

Total Quarantined Records: **87** (logged in `data/raw/field_recordings/quarantined/quarantine_log.csv`).

---

## 5. File Format & Audio Manifest Integrity Audit

- **Standardization Compliance:** Mono, 16,000 Hz, 16-bit PCM, exactly 16,000 samples (1.00s).
- **Format Audit Result:** **PASSED (100% compliant)**.
- **Raw File Preservation:** Bit-level untouched raw files preserved with original container extensions (`.m4a`, `.wav`, etc.).
- **Security & Path Traversal:** Strict regex validation enforced on `speaker_id` and `session_id`.

Sample format verification log:
- [PASS] `field_trai_SPK_FIELD_N001_TESTSUITE_SES_PRESERVE_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_phai_SPK_FIELD_N001_TESTSUITE_SES_DUP_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_trai_SPK_FIELD_N001_TESTSUITE_SES_MOCK_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_trai_SPK_FIELD_N001_TESTSUITE_SES_PRESERVE_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_phai_SPK_FIELD_N001_TESTSUITE_SES_DUP_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_trai_SPK_FIELD_N001_TESTSUITE_SES_MOCK_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_trai_SPK_FIELD_N001_TESTSUITE_SES_MOCK_TEST_take002`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- [PASS] `field_len_SPK_FIELD_N001_TESTSUITE_SES_MOCK_TEST_take001`: Standardized 16kHz mono 16-bit PCM (16,000 samples, 32,044 bytes). Raw preserved: True.
- *... and 42 more verified standardized takes.*

---

## 6. Outstanding Gaps & Next Steps to Mark Campaign Complete

1. **Recruitment & Quota Deficits:**
   - **Northern Native Speakers:** Target minimum 40–50 distinct Northern speakers (currently: 0) to provide robust coverage of native `/tɕ/` affricate acoustics.
   - **Mobile Form Factors:** Expand collections across physical Android (Samsung, Xiaomi) and iOS (iPhone 12–15) microphones.
   - **Acoustic Noise Scenarios:** Collect takes in reverberant tiled rooms, moving street noise, and background HVAC rumble.
2. **Strict Manifest Partitioning:**
   - As new speakers are ingested, partition them strictly by pseudonymous `speaker_id` (70% train, 15% validation, 15% test).
   - Ensure the Phase 3 frozen held-out test manifest (`data/manifests/test_manifest_v2.csv`) is preserved without modification until a complete genuine human test benchmark is scheduled.
3. **Evidence Required to Mark Campaign Complete:**
   - At least 1,500 verified recordings per command class across $\ge 100$ consented human speakers.
   - 0 speaker leakage verified across all splits.
   - Re-training and evaluation demonstrating $> 90\%$ Northern `TRAI` recall on genuine human utterances.
