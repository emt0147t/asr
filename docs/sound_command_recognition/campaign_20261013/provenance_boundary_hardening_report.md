# Provenance Allowlist & Manifest Write Boundary Hardening Report (RES-01)
## Final Security & Failure-Mode Audit

**Campaign Reference:** Vietnamese Keyword Spotting (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`)  
**Assessment Date:** 2026-10-14  
**Workspace:** `D:\trainidentificationaudio`  
**Target Remediation:** Residual Risk `RES-01` from `post_remediation_security_review.md`  
**Audit Verdict:** **`VERIFIED WITH RESIDUAL RISKS`**  
**Human Data Collection Status:** **`STRICTLY BLOCKED`**  

---

## 1. Executive Summary

This report delivers an independent, code-level security audit of the provenance allowlist and manifest write boundary implementation in the Vietnamese KWS field recording pipeline. The audit verifies whether technical controls prevent arbitrary caller-supplied provenance from entering permissive ingestion paths or being committed as approved manifest rows.

### Key Audit Findings:
1. **Enforced Boundary Gates:** Unknown or non-canonical provenance strings fail closed at the entry point of `ingest_recording()`, aborting before audio resampling, standardized file generation, raw copying, or manifest modification.
2. **Independent Lower-Level Write Validation:** Direct calls to `append_to_manifest()` independently validate data structures, required columns, provenance allowlists, status enums, and demographic consistency for human records.
3. **Unconditional Strict Metadata:** For human provenance (`human_field_recording` or `HUMAN_CONSENTED`), `effective_strict = True` is unconditionally enforced. Callers cannot pass `strict_metadata=False` to bypass demographic validation or inject fallback defaults.
4. **Independent Consent Gating:** Consent status is verified against `consent_ledger.json` before any human record can attain `APPROVED` status. Speakers with missing, invalid, or revoked consent are quarantined.
5. **Rollback & Pre-Existing File Preservation:** Ingestion unlinks only artifacts created during that specific execution attempt. A pre-existing destination collision safeguard prevents overwriting or unlinking existing files on disk.
6. **Cryptographic Baseline Preservation:** All three protected historical baseline files (`test_manifest_v2.csv`, `field_collection_manifest.csv`, `consent_ledger.json`) retain their bit-exact SHA-256 hashes.
7. **Physical Origin Limitation:** An allowlist validates caller metadata against authorized schemas; it does not physically prove the acoustic liveness or bio-metric origin of an audio file.

---

## 2. Files and Functions Actually Inspected

The following primary files and call paths were audited:

| Component / File | Inspected Functions / Blocks | Scope of Inspection |
| :--- | :--- | :--- |
| `src/acquisition/ingest_field_recordings.py` | `append_to_manifest()`, `append_to_csv()`, `validate_field_audio()`, `ingest_recording()`, `batch_ingest_directory()`, `__main__` CLI parser | Core ingestion lifecycle, allowlists, early quarantine, destination collision safeguard, rollback cleanup |
| `src/acquisition/campaign_validator.py` | `validate_manifest_row()`, `validate_manifest_dataset()`, `validate_speaker_disjointness()`, `ALLOWED_PROVENANCE_VALUES` | Manifest schema validation, synthetic fixture co-mingling detection, mock speaker ID rejection |
| `src/acquisition/speaker_consent.py` | `register_speaker()`, `verify_speaker_consent()`, `load_consent_ledger()`, `save_consent_ledger()`, `get_speaker_metadata()` | Consent verification, cryptographic nonce IDs, revocation tracking |
| `src/acquisition/revoke_speaker_consent.py` | `execute_consent_revocation()`, CLI execution entrypoint | Manifest status update to `REVOKED_WITHDRAWN`, physical audio relocation to quarantine |
| `src/acquisition/record_command_cli.py` | `run_recording_session()`, `AudioRecorder`, CLI options | CLI invocation wrapper, strict-mode propagation, mock mode provenance tagging |
| `tests/test_campaign_acceptance.py` | All test cases (TC-01 through TC-11, P0 tests, RES-01 tests) | Automated regression test coverage across all boundary and failure conditions |

---

## 3. Comprehensive Manifest-Write Paths

Every call path capable of writing manifest rows or generating standardized audio was identified:

```
[Path 1: Single Audio Ingestion]
Caller / CLI -> ingest_recording()
  ├─> Early Allowlist Check (ALLOWED_INGESTION_PROVENANCE)
  ├─> Strict Metadata Gate (effective_strict = True for human)
  ├─> Audio Quality & Consent Validation (validate_field_audio())
  ├─> Audio Standardization (sf.write to audio_16k_standardized/)
  └─> append_to_manifest() [Validates schema -> writes to manifest CSV]

[Path 2: Batch Audio Ingestion]
Caller / CLI -> batch_ingest_directory()
  └─> Iterates files -> invokes Path 1 (ingest_recording())

[Path 3: Interactive Field CLI]
CLI Operator -> record_command_cli.py -> run_recording_session()
  └─> Captures live/mock audio -> invokes Path 1 (ingest_recording())

[Path 4: Lower-Level Direct Manifest Write]
Direct Caller / Test -> append_to_manifest(record, manifest_path)
  └─> Independent validation (dict, core fields, ALLOWED_MANIFEST_PROVENANCE, status, demographics)
      └─> Appends row directly to manifest CSV

[Path 5: Consent Revocation Manifest Mutation]
Operator -> revoke_speaker_consent.py -> execute_consent_revocation()
  ├─> Reads existing manifest
  ├─> Updates matching speaker rows to status="REVOKED_WITHDRAWN"
  ├─> Quarantines audio files to <quarantine_dir>/withdrawn_speakers/<speaker_id>/
  └─> Rewrites manifest via df_manifest.to_csv()

[Path 6: Preprocessing & Benchmark Splits (Non-Acquisition)]
Developer -> src/preprocessing/build_manifests.py
  └─> Partitions data into train/val/test manifests (build-time only, distinct from field ingestion)
```

---

## 4. Trust Boundary Architecture & Invariants

### 4.1 Provenance Allowlists

Two authoritative sets govern ingestion and manifest storage:

```python
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
```

### 4.2 Invariant Verification Matrix

| Invariant Requirement | Implementation Mechanism | Enforcement Layer | Status |
| :--- | :--- | :--- | :--- |
| **1. Unknown Provenance Rejection** | Checked at lines 365–385 of `ingest_recording()` before file operations. | Ingestion Entry Gate | **VERIFIED** |
| **2. Independent Manifest Validation** | `append_to_manifest()` checks record type, core fields, provenance allowlist, and status enum. | Manifest Write Boundary | **VERIFIED** |
| **3. Mandatory Human Strict Mode** | `effective_strict = True` forced when `provenance in ("human_field_recording", "HUMAN_CONSENTED")`. | Ingestion Core | **VERIFIED** |
| **4. Explicit Consent Verification** | `verify_speaker_consent()` checks ledger `consent_given == True` and `revoked_at is None`. | Audio Validation Gate | **VERIFIED** |
| **5. Protection Against Untrusted Relabeling** | Provenance allowlist, mandatory strict metadata, ledger lookup, and dataset validator checks. | Multi-tier Pipeline | **VERIFIED** |
| **6. Synthetic Fixture Separation** | `prov_label` never promotes mock tags to `HUMAN_CONSENTED`; dataset validator rejects mock speaker IDs under human tags. | Pipeline & Dataset Validator | **VERIFIED** |
| **7. Granular Rollback & Pre-existing File Safety** | Destination collision check stops overwrites; `raw_created` and `std_created` track unlinking. | File System Boundary | **VERIFIED** |
| **8. Clean Quarantine Execution** | Quarantined records write exclusively to `quarantine_log.csv`; zero audio copies or manifest rows created. | Quarantine Handler | **VERIFIED** |
| **9. Historical Baseline Immutability** | Read-only separation; tests operate in isolated sandboxes (`tmp_path`). | Test Architecture | **VERIFIED** |

---

## 5. Automated Test Evidence & Failure-Mode Verification

Automated regression verification was executed using `pytest tests/test_campaign_acceptance.py`.

### 5.1 Test Execution Results

- **Command:** `pytest tests/test_campaign_acceptance.py -q`
- **Exit Code:** `0`
- **Total Tests Passed:** **42 passed in 1.98s**
- **Test Failures:** `0`
- **Test Skips / Timeouts:** `0`

### 5.2 Failure-Mode Test Matrix (11 Focused Boundary Tests)

Focused test run command:
`pytest tests/test_campaign_acceptance.py -k "test_revoked_consent or test_valid_consent_vs or test_existing_destination or test_unknown_provenance or test_custom_clean or test_synthetic_fixture or test_human_provenance or test_manifest_write_boundary or test_rejected_ingestion or test_existing_valid_fixture" -v`  
**Result:** `11 passed, 31 deselected in 0.81s` (Exit Code: `0`).

| Test Function | Target Failure Condition | Injected Fault / Scenario | Verified Behavior | Status |
| :--- | :--- | :--- | :--- | :--- |
| `test_unknown_provenance_fails_closed_to_quarantine` | Unknown Provenance | `provenance="unknown_external_audio"` | Quarantined immediately; 0 audio files copied; manifest untouched | **PASSED** |
| `test_custom_clean_source_cannot_enter_permissive_path` | Permissive Bypass Attempt | `provenance="custom_clean_source"`, `strict_metadata=False` | Permissive path blocked; quarantined; 0 audio artifacts created | **PASSED** |
| `test_synthetic_fixture_cannot_create_human_consented_manifest_row` | Fixture Relabeling Attempt | `provenance="SYNTHETIC_TEST_FIXTURE"` | Retains fixture provenance; dataset validator blocks mock IDs labeled as human | **PASSED** |
| `test_human_provenance_cannot_bypass_strict_metadata` | Strict Metadata Bypass Attempt | `provenance="human_field_recording"`, `strict_metadata=False` | Strict mode unconditionally forced; fails on incomplete metadata; quarantined | **PASSED** |
| `test_human_provenance_without_valid_consent_quarantined` | Unconsented Ingestion Attempt | Speaker with `consent_given=False` and unregistered speaker ID | Both quarantined (`missing_or_unconsented_speaker`); manifest untouched | **PASSED** |
| `test_manifest_write_boundary_rejects_invalid_provenance_and_fields` | Direct Invalid Manifest Write | Non-dict input, missing `speaker_id`, unauthorized provenance, invalid status enum | `TypeError` and `ValueError` raised; invalid records never committed to disk | **PASSED** |
| `test_rejected_ingestion_cleans_up_audio_artifacts` | Rollback on Commit Failure | Simulated disk error during `append_to_manifest()` | `RuntimeError` raised; raw and standardized audio files unlinked atomically | **PASSED** |
| `test_existing_valid_fixture_and_human_paths_retain_behavior` | Backward Compatibility | Valid human recording and test harness fixture | Both approved under respective canonical tags (`HUMAN_CONSENTED`, `test_harness_mock`) | **PASSED** |
| `test_revoked_consent_speaker_ingestion_blocked` | Post-Revocation Ingestion Attempt | Ingestion attempted for speaker whose consent was previously revoked | Quarantined immediately (`missing_or_unconsented_speaker`); manifest untouched | **PASSED** |
| `test_valid_consent_vs_invalid_metadata_isolated_failures` | Failure-Mode Isolation | (a) Valid consent + contradictory metadata; (b) Revoked consent + valid metadata | (a) Fails on `contradictory_metadata`; (b) Fails on `missing_or_unconsented_speaker` | **PASSED** |
| `test_existing_destination_files_during_rollback` | Pre-existing File Protection | (a) Pre-existing sibling file during rollback; (b) Collision with existing destination file | Sibling file preserved bit-exact; destination collision quarantined without overwrite | **PASSED** |

---

## 6. Protected Cryptographic Baseline Hash Verification

Baseline digests were recomputed using Windows PowerShell `Get-FileHash -Algorithm SHA256` and verified bit-for-bit:

| Protected File Path | Baseline SHA-256 Digest | Current SHA-256 Digest | Match Result |
| :--- | :--- | :--- | :--- |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |

---

## 7. Canonical Governance Gate Alignment

Technical test success does **NOT** authorize participant recruitment, speech recording, external corpus training, or model release. The status of the canonical 8 campaign gateways (from `governance_approval_action_tracker.md`) is recorded below:

| Gate ID | Gate Scope | Gate Type | Approving Authority | Current Status |
| :---: | :--- | :---: | :--- | :---: |
| **`GATE-01`** | **Overall Project Protocol Sign-Off** | Pre-Recruitment | Project Sponsor / Principal Investigator | `AWAITING HUMAN DECISION` |
| **`GATE-02`** | **Legal Vetting of Informed Consent Form** | Pre-Recruitment | Legal / Privacy Counsel | `AWAITING HUMAN DECISION` |
| **`GATE-03`** | **Cohort Quota (50N/30C/40S) & Budget Authorization** | Pre-Recruitment | Project Sponsor / Principal Investigator | `AWAITING HUMAN DECISION` |
| **`GATE-04`** | **FLEURS & MSWC Exact-Version License Archiving** | Pre-Training / Release | Legal Counsel & Project Sponsor | `OPEN` (Conditional on external training) |
| **`GATE-05`** | **Data Retention, Cache Purge & Retraining Policy** | Pre-Recruitment | Data Protection Officer & Tech Lead | `AWAITING HUMAN DECISION` |
| **`GATE-06`** | **Physical Hardware Inventory & Operator Calibration** | Pre-Recruitment | Field Operations Lead & Tech Lead | `OPEN` |
| **`GATE-07`** | **Technical Pipeline Strict-Mode Enforcement** | Infrastructure Baseline | Independent Technical Auditor | **`CLOSED WITH EVIDENCE`** |
| **`GATE-08`** | **Individual Contributor Consent Enrollment** | Runtime Per-Participant | Field Operator & Pipeline Invariant | `OPEN (Zero records enrolled)` |

> [!CAUTION]
> **Governance Separation Notice:**  
> `GATE-07` is closed based on documentary and automated code evidence. All other pre-recruitment gates (`GATE-01`, `GATE-02`, `GATE-03`, `GATE-05`, `GATE-06`) remain blocked pending formal human authority review. Participant recruitment and recording remain strictly prohibited.

---

## 8. Discrepancies Between Implementation and Previous Report

The audit identified two discrepancies in the earlier interim report:
1. **Gate Taxonomy Normalization:** The previous report referenced a legacy provisional gate numbering (e.g. labelling GATE-05 as Signal Quality and GATE-06 as Checksums). This report aligns strictly with the canonical 8-gate model defined in `governance_approval_action_tracker.md`.
2. **Pre-Existing File Rollback Vulnerability:** The previous implementation unconditionally unlinked `raw_dest_path` and `std_dest_path` during manifest write failure rollback. If a destination file had already existed prior to ingestion, rollback would have erroneously deleted it. This was remedied by adding an upfront destination collision check (`destination_file_already_exists_on_disk_{rec_id}`) and tracking `raw_created` and `std_created` flags to ensure rollback touches only newly created files.

---

## 9. Residual Risks and Untested Paths

1. **Acoustic Authenticity vs. Metadata Validation:** The ingestion pipeline verifies caller metadata against the consent ledger and quality specifications (RMS, clipping, duration). It does not perform bio-metric voice authentication or acoustic liveness detection to prove physical origin.
2. **Headless Execution Environment:** Live hardware microphone recording via Windows MCI (`AudioRecorder`) cannot be automated in headless environments; automated testing relies on mock audio generators and synthetic fixtures.
3. **Legacy Test Suite Hang:** The legacy `tests/test_field_acquisition.py` suite was omitted from this run due to the known Python 3.14 Windows `librosa` JIT compile hang. Acceptance and boundary testing were executed via `tests/test_campaign_acceptance.py`.

---

## 10. Final Audit Verdict

The provenance allowlists, entrypoint quarantine gates, manifest write boundary, and rollback mechanisms are correctly implemented, mathematically consistent, and covered by automated regression tests.

**Audit Verdict: `VERIFIED WITH RESIDUAL RISKS`**  
**Human Data Collection: `STRICTLY BLOCKED`**
