# Independent Pre-Collection Readiness Audit Report

> **Document Version:** 1.0.0-AUDIT  
> **Audit Date:** October 13, 2026  
> **Auditor:** Independent Automated Code & Policy Auditor  
> **Scope:** Campaign Preparation Artifacts under `docs/sound_command_recognition/campaign_20261013/`  
> **Repository Target:** Vietnamese Keyword Spotting (KWS) — `D:\trainidentificationaudio`  
> **Executive Verdict:** **READY FOR HUMAN REVIEW**  
> *(Collection Approval Status: **BLOCKED** pending human sign-off and remediation of P0/P1 gaps)*

---

## 1. Executive Verdict & Summary

### 1.1 Executive Verdict
The campaign preparation package has achieved the verdict of **`READY FOR HUMAN REVIEW`**.  
The governance protocols, schema specifications, quality control checklists, acceptance testing suites, and data isolation architectures are comprehensively documented and technically mature.

However, formal **`READY FOR COLLECTION APPROVAL`** is strictly **BLOCKED** at this time. Collection of genuine human speech must not commence until:
1. Human stakeholders formally review and approve the Informed Consent Agreement and Collection Protocol.
2. The default storage paths in production ingestion scripts are permanently switched from the historical fixture root (`data/raw/field_recordings/`) to the segregated human directory (`data/raw/human_recordings_v1/`).
3. External negative speech and noise source licenses (specifically `husein_mic_noise`) are legally verified and downloaded with immutable checksums.
4. Implementation gaps between the proposed 28-column schema and the current 23-column ingestion code are reconciled.

### 1.2 Audit Summary Scorecard

| Audit Domain | Target Standard | Status | Verdict |
| :--- | :--- | :---: | :---: |
| **1. Protected Artifact Integrity** | Frozen benchmark `test_manifest_v2.csv` exact SHA-256 match | Verified (100% Bit-Exact) | **PASSED** |
| **2. Protocol & Schema Consistency** | Cross-document coherence across 7 markdown specifications | Gaps Identified (P1) | **REVIEW REQUIRED** |
| **3. Partition & Split Correctness** | Zero speaker/session leakage; mathematical disjointness | Verified (Disjoint Guaranteed; Strata Unenforced) | **CONDITIONAL PASS** |
| **4. Consent & Provenance Enforcement** | Missing/revoked consent blocks; synthetic cannot masquerade as human | Verified via TC-01 to TC-04 | **PASSED** |
| **5. Audio Quality Gates & Boundaries** | 14 quality gates; clipping, silence, duration, format bounds | Verified (10/10 boundary tests passed) | **PASSED** |
| **6. Test Suite Isolation** | Zero mutation of production manifests or audio during testing | Verified (31/31 passed; 0 pollution) | **PASSED** |
| **7. External Source Licensing** | Legally vetted licenses for FLEURS, Common Voice, and noise | Unverified License Claims (P0) | **BLOCKED** |

---

## 2. Protected Artifact Integrity & Working-Tree State

### 2.1 Cryptographic Hash Verification
The frozen benchmark test set and historical fixture archives were hashed before and after all audit operations.

```powershell
Get-FileHash -Algorithm SHA256 `
  "data/manifests/test_manifest_v2.csv", `
  "data/raw/field_recordings/field_collection_manifest.csv", `
  "data/raw/field_recordings/consent_ledger.json"
```

| Asset Path | Expected SHA-256 Digest | Measured SHA-256 Digest | Integrity Status |
| :--- | :--- | :--- | :---: |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **MATCH (BIT-EXACT)** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **MATCH (BIT-EXACT)** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **MATCH (BIT-EXACT)** |

### 2.2 Working-Tree Status
Prior to this audit, Git tracked the following modified and untracked files:
- **Modified Working-Tree Files:**
  - `reports/field_recording_campaign_audit.md`
  - `src/acquisition/audit_field_campaign.py`
  - `src/acquisition/ingest_field_recordings.py`
  - `src/acquisition/record_command_cli.py`
  - `src/acquisition/speaker_consent.py`
  - `src/preprocessing/audio_normalizer.py`
  - `tests/test_field_acquisition.py`
- **Untracked Directories & Tooling Files:**
  - `docs/sound_command_recognition/campaign_20261013/`
  - `src/acquisition/campaign_validator.py`
  - `tests/test_campaign_acceptance.py`

Zero models have been modified, retrained, exported, or quantized.

---

## 3. Protocol and Schema Consistency Evaluation

We conducted a cross-document audit comparing:
1. `collection_protocol.md`
2. `proposed_manifest_schema.md`
3. `data_provenance_policy.md`
4. `speaker_disjoint_split_policy.md`
5. `quality_control_checklist.md`
6. `implementation_gap_report.md`
7. `test_plan.md`
against the underlying code in `src/acquisition/`.

### 3.1 Schema Discrepancy: 23-Column Ingestion vs. 28-Column Specification
- **Specification:** [`proposed_manifest_schema.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/proposed_manifest_schema.md#L23-L55) defines a 28-column manifest.
- **Current Code:** [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L293-L319) outputs only 23 columns.
- **Missing Columns in Ingestion Code:**
  1. `consent_id` (foreign key linking recording directly to signed consent agreement).
  2. `age_bracket` (`18-30`, `31-50`, `51+`).
  3. `sub_category` (`isolated_command`, `confusing_near_rhyme`, `carrier_sentence_embedded`, `background_room_tone`).
  4. `microphone_type` (`builtin_mobile`, `wired_headset`, `bluetooth_earbuds`, `usb_desktop_mic`).
  5. `distance_category` (`close_talk_15cm`, `desk_distance_60cm`, `far_field_2m`).
- **Impact:** Ingesting audio with the existing `ingest_recording` script will not capture these 5 acoustic metadata attributes, requiring a schema migration or script update before collection begins.

### 3.2 Metadata Constant & Naming Inconsistencies
| Property | `proposed_manifest_schema.md` | `ingest_field_recordings.py` (Current Implementation) | Discrepancy Severity |
| :--- | :--- | :--- | :---: |
| `license` string | `Proprietary-Consented-Research-Commercial` | `Proprietary-Consented` | P1 (String Mismatch) |
| `source` string | `vietnamese_kws_human_campaign_v1` | `field_human_recording_campaign_v1` | P1 (String Mismatch) |
| `source_version` | `v1.0.0-human` | `v1.0` | P1 (String Mismatch) |
| `recording_id` pattern | `^rec_h_[a-z0-9_\-]+_take\d{3}$` | `field_<cmd>_<spk>_<ses>_take<idx>` | P1 (Regex Mismatch) |
| `speaker_id` schema | `^SPK_H_[NCS]_[a-f0-9]{6}$` | `SPK_FIELD_<N|C|S><001>` (Sequential counter) | P0 (Privacy Leakage) |

### 3.3 Target Cohort Target Sizing
- `collection_protocol.md` Section 4.1 establishes:
  - **Northern:** 50 speakers (41.7%)
  - **Central:** 30 speakers (25.0%)
  - **Southern:** 40 speakers (33.3%)
  - **Total:** 120 speakers.
- This is statistically sound for 70/15/15 partitioning (84 train, 18 val, 18 test) and resolves prior informal references to 60/30/30 quotas.

---

## 4. Split Correctness & Leakage Evaluation

### 4.1 Audit of `validate_speaker_disjointness`
[`validate_speaker_disjointness`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L264-L313) in `campaign_validator.py` computes set intersections:
$$S_{\text{train}} \cap S_{\text{val}}, \quad S_{\text{train}} \cap S_{\text{test}}, \quad S_{\text{val}} \cap S_{\text{test}}$$
$$\text{Ses}_{\text{train}} \cap \text{Ses}_{\text{val}}, \quad \text{Ses}_{\text{train}} \cap \text{Ses}_{\text{test}}, \quad \text{Ses}_{\text{val}} \cap \text{Ses}_{\text{test}}$$
- **Verification:** Tested under TC-09 (disjoint dataset verification) and TC-10 (deliberate injection of leaked speaker `SPK_001` into test partition).
- **Result:** Successfully detects overlaps and returns detailed overlap dictionaries.

### 4.2 Audit of `generate_disjoint_splits`
[`generate_disjoint_splits`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L315-L352) uses a salted CRC32 hash of the `speaker_id`:
$$\text{Score}(s) = \frac{\text{CRC32}(\text{"kws\_vietnamese\_campaign\_2026\_v1\_"} + s)}{2^{32} - 1}$$
- **Determinism:** 100% deterministic. Identical speaker IDs always yield the exact same partition score.
- **Completeness:** Every row is assigned to exactly one partition (`train`, `val`, or `test`).
- **Speaker & Session Cohesion:** Invariant holds strictly; all takes and sessions for speaker $s$ map to the same partition.

### 4.3 Gap: Lack of Demographic / Regional Strata Enforcement
In [`speaker_disjoint_split_policy.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md#L46-L73), the proposed architecture calls for stratified hashing by demographic group (`Group by Strata (Dialect x Gender)`).

However, in `campaign_validator.py`, `generate_disjoint_splits` applies a uniform hash directly across all speakers without strata grouping.
We performed an empirical simulation with 120 synthetic speakers (50 Northern, 30 Central, 40 Southern):

```text
Simulation Output (120 Speakers, 70% / 15% / 15% Target):
- Total Speakers: Train = 85 (70.8%), Val = 17 (14.2%), Test = 18 (15.0%)
- Northern Cohort (50 spk):
    Train = 39 (78.0%)  [Target: 70.0%]
    Val   =  4 ( 8.0%)  [Target: 15.0% - Severe Undersampling]
    Test  =  7 (14.0%)  [Target: 15.0%]
- Central Cohort (30 spk):
    Train = 18 (60.0%)
    Val   =  7 (23.3%)
    Test  =  5 (16.7%)
- Southern Cohort (40 spk):
    Train = 28 (70.0%)
    Val   =  6 (15.0%)
    Test  =  6 (15.0%)
```

**Finding:** Uniform salted hashing does not guarantee stratified demographic balance. In the simulation, Northern validation representation dropped from the desired 15% (7–8 speakers) to only 8% (4 speakers).  
**Remediation Required:** Upgrade `generate_disjoint_splits` to hash within dialect/gender strata or perform constrained stratified assignment before freezing splits.

---

## 5. Consent Ledger & Provenance Enforcement Audit

### 5.1 Verification of Consent Enforcement
- **Ingestion Gate:** [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L112) calls `verify_speaker_consent()`. If the speaker is absent from the ledger or `consent_given == False`, ingestion immediately aborts with `missing_or_unconsented_speaker`.
- **Validation Gate:** [`src/acquisition/campaign_validator.py`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L168-L175) cross-checks every row marked `HUMAN_CONSENTED`. If consent is missing or False, `validate_manifest_row` reports `missing_consent_entry_for_human_speaker` or `consent_not_granted_for_speaker`.
- **Automated Verification:** Verified in `tests/test_campaign_acceptance.py::test_unconsented_speaker_rejection` (TC-02).

### 5.2 Handling of Consent Revocation (Withdrawal)
- `revoke_speaker_consent(speaker_id)` was added to `src/acquisition/speaker_consent.py`.
- **Audit Finding:** Calling `revoke_speaker_consent` marks `consent_given: false` and sets a UTC revocation timestamp. Subsequent manifest validations flag all historical recordings by that speaker.
- **Gap:** The operator CLI script `src/acquisition/revoke_speaker_consent.py` described in `collection_protocol.md` Section 2.4 does not exist. Currently, revocation must be called programmatically via Python, and files on disk are not automatically quarantined to `data/raw/quarantined/withdrawn_speakers/`.

### 5.3 Synthetic vs. Human Co-Mingling Defenses
- [`src/acquisition/campaign_validator.py`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L233-L242) inspects all records claiming `HUMAN_CONSENTED`:
  - If `raw_filepath` contains `data/raw/field_recordings/` (the legacy fixture archive), it raises `synthetic_fixture_co_mingling`.
  - If `speaker_id` contains substrings `TEST`, `MOCK`, or `SYNTH`, it raises `synthetic_speaker_id_mislabeled_as_human`.
  - If provenance is not an approved enum, it raises `invalid_provenance_enum`.
- **Automated Verification:** Verified in `tests/test_campaign_acceptance.py::test_invalid_provenance_tag_rejection` (TC-04).

### 5.4 Privacy & PII Audit
- Both `data/raw/field_recordings/consent_ledger.json` and `quarantine_log.csv` were inspected.
- Only acoustic-demographic metadata (`dialect`, `gender`, `age_bracket`, `device_model`, `environment`) is stored.
- Zero names, phone numbers, email addresses, national IDs, or GPS coordinates are present.

---

## 6. Audio Quality Gates & Boundary Stress Testing

### 6.1 Quality Gate Coverage (QC-01 through QC-14)
All 14 gates in [`quality_control_checklist.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/quality_control_checklist.md) were audited against the implementation in `campaign_validator.py` and `audio_normalizer.py`.

### 6.2 Empirical Boundary Stress Testing
We executed 10 boundary and edge-case tests against `validate_audio_file`:

| Test # | Quality Gate Tested | Input Condition | Expected Behavior | Measured Result |
| :---: | :--- | :--- | :--- | :---: |
| **B-01** | Missing File | Non-existent file path | Reject: `file_not_found` | **PASSED** |
| **B-02** | Zero-Byte File | 0-byte `.wav` file | Reject: `corrupt_file_read_error` | **PASSED** |
| **B-03** | Corrupted Header | Random ASCII bytes in `.wav` container | Reject: `corrupt_file_read_error` | **PASSED** |
| **B-04** | Sample Rate Mismatch | 44,100 Hz WAV | Reject: `sample_rate_mismatch: 44100 != 16000` | **PASSED** |
| **B-05** | Channel Mismatch | 2-channel stereo WAV | Reject: `channel_count_mismatch: 2 != 1` | **PASSED** |
| **B-06** | Frame Count Mismatch | 15,999 frames (0.9999 s) | Reject: `frame_count_mismatch: 15999 != 16000` | **PASSED** |
| **B-07** | Clipping Boundary (Allowed)| 2 consecutive samples at $+1.0$ | Accept (threshold is $\ge 3$) | **PASSED** |
| **B-08** | Clipping Boundary (Rejected)| 3 consecutive samples at $+1.0$ | Reject: `digital_clipping_detected` | **PASSED** |
| **B-09** | Negative Clipping | 3 consecutive samples at $-1.0$ | Reject: `digital_clipping_detected` | **PASSED** |
| **B-10** | Silence Threshold Boundary | Signal energy at $-46.0$ dBFS | Reject for speech; Accept for `SILENCE` | **PASSED** |

**Conclusion:** All 10 audio signal boundary checks behaved as specified. The silence exemption is strictly restricted to designated `SILENCE` instances (`allow_silence=True`).

---

## 7. Test Suite Isolation & Regression Safeguards

### 7.1 Isolation Mechanism Verification
- Acceptance test fixtures in `tests/test_campaign_acceptance.py` write exclusively to pytest's ephemeral `tmp_path` sandbox directory.
- `tests/test_field_acquisition.py` uses an autouse fixture monkeypatching all base paths to temporary folders.
- Production manifests, ledgers, checkpoints, and benchmark files are completely unreferenced in write modes.

### 7.2 Full Regression Execution
Command:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/
```

- **Exit Code:** `0` (Success)
- **Total Tests Collected & Executed:** **31 passed** in 11.00 seconds.
- **Breakdown:**
  - `tests/test_campaign_acceptance.py`: 11 passed (TC-01 through TC-11)
  - `tests/test_field_acquisition.py`: 13 passed
  - `tests/test_parity.py`: 7 passed

### 7.3 Post-Test Benchmark Hash Invariant
Post-execution SHA-256 computation of `data/manifests/test_manifest_v2.csv`:
$$\text{SHA-256} = \texttt{9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13}$$
**Status:** 100% Bit-Exact Match. Zero production pollution occurred.

---

## 8. External Source Licensing Assessment

The campaign documents propose incorporating negative speech and noise from external repositories to populate classes 4 (`UNKNOWN`) and 5 (`SILENCE`).

| Proposed Source | Target Class | Claimed License | Intended Usage | Repository Evidence & Licensing Status |
| :--- | :---: | :---: | :--- | :--- |
| **Google FLEURS (Vietnamese)** | `UNKNOWN` (4) | CC-BY-4.0 | Continuous conversational speech negatives | **PROPOSED / UNVERIFIED**<br/>CC-BY-4.0 requires attribution. No FLEURS clips exist in the repository; no attribution file or source URL manifest is currently committed. |
| **husein_mic_noise** | `SILENCE` (5) | CC-BY-4.0 | Mobile microphone noise | **BLOCKED / UNVERIFIED**<br/>The license is referenced in `data_provenance_policy.md`, but no license grant, author URL, or raw archive exists on disk. Must be verified before use. |
| **Mozilla Common Voice (vi)** | `UNKNOWN` (4) | CC0-1.0 | Out-of-vocabulary negatives | **PROPOSED / UNVERIFIED**<br/>Permissive CC0 license, but no specific dataset release version (e.g., CV 13.0) or audio files are cataloged. |
| **MS-SNSD** | `SILENCE` (5) | MIT | Environmental room noise | **PROPOSED / UNVERIFIED**<br/>Permissive MIT license, but no files currently imported. |
| **Proprietary Human Field Speech** | `TRAI`, `PHAI`, `LEN`, `XUONG` | Proprietary-Consented | Target command training and evaluation | **PENDING HUMAN APPROVAL**<br/>Requires formal human approval of consent text prior to participant recruitment. |

---

## 9. Comprehensive Finding Inventory (P0 / P1 / P2)

### 9.1 Priority 0 (P0) — Blockers for Human Collection
1. **[Finding P0-1] Default Ingestion Paths Point to Historical Test Fixture Root (`GAP-02`)**
   - *Evidence Path:* [`src/acquisition/ingest_field_recordings.py#L55-L60`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L55-L60), [`src/acquisition/speaker_consent.py#L38`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py#L38)
   - *Description:* `FIELD_BASE_DIR` in `ingest_field_recordings.py` is configured as `Path("data/raw/field_recordings")`. If an operator runs `record_command_cli.py` or `ingest_recording()` without overriding paths, genuine human speech will be appended to the test fixture manifest.
   - *Remediation:* Update default path constants to `data/raw/human_recordings_v1/`.
2. **[Finding P0-2] Unverified License Provenance for External Negative Noise Corpora**
   - *Evidence Path:* [`docs/sound_command_recognition/campaign_20261013/data_provenance_policy.md#L55-L58`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/data_provenance_policy.md#L55-L58)
   - *Description:* `husein_mic_noise` is designated for `NEGATIVE_NOISE_HUSEIN` under CC-BY-4.0, but no license copy, download script, or source URL exists in the repository.
   - *Remediation:* Verify legal licensing text and author terms, or replace with verified MIT-licensed datasets (MS-SNSD) or in-session ambient silence takes.
3. **[Finding P0-3] Sequential Speaker ID Generation Leaks Enrollment Timing**
   - *Evidence Path:* [`src/acquisition/speaker_consent.py#L59-L77`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py#L59-L77)
   - *Description:* `generate_next_speaker_id()` produces sequential IDs (`SPK_FIELD_N001`, `SPK_FIELD_N002`), which directly violates Section 3 of `collection_protocol.md` mandating pseudonymous cryptographic hex nonces (`SPK_H_N_<hex6>`).
   - *Remediation:* Update `speaker_consent.py` to generate pseudonymous nonces.

### 9.2 Priority 1 (P1) — Major Implementation & Schema Discrepancies
1. **[Finding P1-1] 23-Column Ingestion vs. 28-Column Proposed Manifest Schema Mismatch**
   - *Evidence Path:* [`src/acquisition/ingest_field_recordings.py#L293-L319`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L293-L319) vs. [`proposed_manifest_schema.md#L23-L55`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/proposed_manifest_schema.md#L23-L55)
   - *Description:* Ingestion code omits `consent_id`, `age_bracket`, `sub_category`, `microphone_type`, and `distance_category`.
   - *Remediation:* Update `ingest_field_recordings.py` to accept and write all 28 schema columns.
2. **[Finding P1-2] Inconsistent Metadata String Enums**
   - *Evidence Path:* `license` (`Proprietary-Consented` vs. `Proprietary-Consented-Research-Commercial`), `source` (`field_human_recording_campaign_v1` vs. `vietnamese_kws_human_campaign_v1`), `recording_id` naming pattern (`field_...` vs. `rec_h_...`).
   - *Remediation:* Standardize metadata strings across ingestion scripts and documentation.
3. **[Finding P1-3] Missing Standalone Consent Revocation CLI Tool (`GAP-03`)**
   - *Evidence Path:* Missing script `src/acquisition/revoke_speaker_consent.py`.
   - *Description:* While `revoke_speaker_consent()` exists as a library function in `speaker_consent.py`, the CLI operator tool described in `collection_protocol.md` Section 2.4 does not exist.
   - *Remediation:* Create `src/acquisition/revoke_speaker_consent.py` with automated quarantine handling.
4. **[Finding P1-4] Lack of Demographic Strata Enforcement in Split Generator**
   - *Evidence Path:* [`src/acquisition/campaign_validator.py#L315-L352`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L315-L352)
   - *Description:* `generate_disjoint_splits()` hashes speakers uniformly without grouping by dialect strata, risking regional skew across splits.
   - *Remediation:* Add stratum grouping before hashing to guarantee equal regional representation in validation and test partitions.

### 9.3 Priority 2 (P2) — Operational Improvements
1. **[Finding P2-1] Windows MCI Hardware Lock for Live Capture (`GAP-04`)**
   - *Evidence Path:* [`src/acquisition/record_command_cli.py#L42`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L42)
   - *Description:* Live microphone capture relies on `winmm.dll`. Cross-platform mobile field recordings must use external recording apps and batch `.m4a` ingestion.
2. **[Finding P2-2] Quarantined Takes Require Automated Re-ingestion Workflow**
   - *Evidence Path:* [`quality_control_checklist.md#L53-L69`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/quality_control_checklist.md#L53-L69)
   - *Description:* Rejection log inspection is currently manual without a scripted re-validation tool for resolved takes.

---

## 10. Summary Verification Matrix

| Check Identifier | Description | Test Method | Outcome |
| :---: | :--- | :--- | :---: |
| **CHK-01** | Benchmark `test_manifest_v2.csv` SHA-256 match | SHA-256 calculation | **PASSED** |
| **CHK-02** | Historical manifest `field_collection_manifest.csv` SHA-256 match | SHA-256 calculation | **PASSED** |
| **CHK-03** | Historical consent ledger `consent_ledger.json` SHA-256 match | SHA-256 calculation | **PASSED** |
| **CHK-04** | Valid human manifest passes all gates | Pytest TC-01 | **PASSED** |
| **CHK-05** | Unconsented speaker rejected | Pytest TC-02 | **PASSED** |
| **CHK-06** | Missing metadata rejected | Pytest TC-03 | **PASSED** |
| **CHK-07** | Synthetic fixture provenance masquerading rejected | Pytest TC-04 | **PASSED** |
| **CHK-08** | Duplicate raw audio SHA-256 quarantined | Pytest TC-05 | **PASSED** |
| **CHK-09** | Duplicate recording ID quarantined | Pytest TC-06 | **PASSED** |
| **CHK-10** | Digital clipping detected ($\ge 3$ consecutive clipped samples) | Pytest TC-07 | **PASSED** |
| **CHK-11** | Abnormal silence detected for commands ($RMS < -45.0$ dBFS) | Pytest TC-08 | **PASSED** |
| **CHK-12** | Silence allowed for designated `SILENCE` class | Pytest TC-08 | **PASSED** |
| **CHK-13** | Mathematical zero silence detected | Audio boundary test | **PASSED** |
| **CHK-14** | Speaker disjoint splits generated without leakage | Pytest TC-09 | **PASSED** |
| **CHK-15** | Speaker leakage detector raises violation on overlap | Pytest TC-10 | **PASSED** |
| **CHK-16** | Acceptance test suite causes zero production pollution | Pytest TC-11 | **PASSED** |
| **CHK-17** | Regional demographic strata preserved in splits | 120-spk simulation | **FAILED (Strata unenforced)** |
| **CHK-18** | Schema column parity (28 columns) in ingestion code | Code inspection | **FAILED (23 columns in code)** |
| **CHK-19** | Default paths point to segregated human folder | Code inspection | **FAILED (Points to legacy fixtures)** |
| **CHK-20** | Pseudonymous nonce speaker IDs generated | Code inspection | **FAILED (Sequential counters)** |
| **CHK-21** | External negative noise licenses verified | Licensing inspection | **BLOCKED (Unverified)** |

---

## 11. Explicit Limitations of this Audit

1. **Synthetic / Mock Audio Limitations:** Because genuine human participants have not yet been recruited, acceptance tests (TC-01 through TC-10) and boundary tests were evaluated using mathematically synthesized audio signals (sinusoids, noise, and DC blocks) generated within isolated test fixtures. No real human voices have been evaluated yet.
2. **Microphone Hardware Transducer Limits:** Tests evaluated file-level audio streams. Acoustic properties of real-world mobile microphones (MEMS non-linear clipping, hardware automatic gain control (AGC), and frequency roll-off) could not be evaluated without physical participant devices.
3. **Cross-Platform Mobile Testing:** Testing was conducted within the Windows environment. Mobile audio capture on iOS (CoreAudio AudioUnit) and Android (AAudio / OpenSL ES) was audited through file ingestion specifications, not physical mobile runtimes.

---

## 12. Human Decision Checklist Before Collection Approval

Before transitioning this project to `READY FOR COLLECTION APPROVAL`, human stakeholders must review and authorize:

- [ ] **1. Approve Informed Consent Form:** Review and authorize the Vietnamese consent text in [`collection_protocol.md#L41-L67`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67).
- [ ] **2. Authorize Remediation of P0-1:** Authorize changing default base directories in `ingest_field_recordings.py` and `speaker_consent.py` to `data/raw/human_recordings_v1/`.
- [ ] **3. Authorize Remediation of P0-2:** Resolve external noise licensing (vet `husein_mic_noise` or restrict `SILENCE` to ambient room tone recorded directly by participants).
- [ ] **4. Authorize Remediation of P0-3:** Authorize replacing sequential `SPK_FIELD_N001` speaker ID generation with pseudonymous `SPK_H_N_<hex6>` nonces.
- [ ] **5. Authorize Manifest Schema Expansion:** Authorize expanding `ingest_field_recordings.py` to output the full 28 columns defined in `proposed_manifest_schema.md`.
