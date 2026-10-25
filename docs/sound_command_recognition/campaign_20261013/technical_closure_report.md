# Technical Closure Report: Human Speech Campaign Readiness

**Workspace:** `D:\trainidentificationaudio`  
**Date:** 2026-10-14  
**Author / Engineering Role:** AI Pair Programmer (Independent Forensic Audit & Technical Remediation)  
**Target Ingestion Target:** Genuine Human Field Speech Collection (`v1.0.0-human`)  
**Overall Technical Verdict:** **`TECHNICALLY READY FOR HUMAN SIGN-OFF`**  
**Collection Operation Status:** **`STRICTLY BLOCKED`** *(Pending formal human sign-off on consent ledger, sampling quota, and license approvals)*

---

## 1. Executive Summary

Following the comprehensive audit findings documented in [`governance_evidence_review.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md) and [`final_readiness_audit.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/final_readiness_audit.md), this report confirms the complete technical closure of all remaining engineering and pipeline vulnerabilities in the Vietnamese Keyword Spotting (KWS) data acquisition infrastructure.

### Summary of Closed Technical Gaps
1. **Production Fail-Closed Strict-Mode Enforced:** Traced and sealed the end-to-end call path from CLI to ingestion. In live human recording sessions (`mock_mode=False`) and external human file ingestion (`provenance="human_field_recording"`), strict metadata validation cannot be bypassed.
2. **Elimination of Silent Fallback Defaults:** Removed silent default injection (`"close_talk_15cm"`, `"builtin_mobile"`, `"18-30"`) in strict mode. Missing demographic or acoustic metadata triggers an immediate fail-closed rejection and quarantine.
3. **Interactive & Programmatic Registration Protocol:** Updated `interactive_registration()` and `register_speaker()` in [`record_command_cli.py`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py) and [`speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py) to collect and validate `microphone_type` and `distance_category`, persisting both to the informed consent ledger.
4. **Corpus Provenance & License Forensic Audit:** Completed forensic hash-level documentation of on-disk assets for Google FLEURS (`vi_vn`), MLCommons MSWC (`vi`), and Husein `mic-noise`. Documented disk reality versus code claims. Maintained strict blocking of `mic-noise`.
5. **Zero Test Pollution & Invariant Preservation:** 44/44 unit and regression tests pass in 100% isolated temporary sandboxes (`tmp_path`). Protected benchmark manifests and historical fixture ledgers remain bit-exact.

---

## 2. Cryptographic Hashes & Invariant Verification

Before and after all code changes and test runs, SHA-256 cryptographic digests were independently verified to guarantee absolute immutability of protected production and historical artifacts:

| Protected File Path | Expected SHA-256 Digest | Measured SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **`VERIFIED` (Bit-Exact)** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **`VERIFIED` (Bit-Exact)** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **`VERIFIED` (Bit-Exact)** |

*Production directory check:* `data/raw/human_recordings_v1` does not exist on disk. Zero unapproved test recordings or manifests have leaked into production storage.

---

## 3. Production Strict-Mode & Fail-Closed Call-Path Trace

### 3.1 Trace of Call Paths

```mermaid
flowchart TD
    CLI["CLI Invocation (record_command_cli.py)"] --> RegChoice{"Action Choice"}
    
    RegChoice -->|"--register"| RegFlow["interactive_registration() / CLI register"]
    RegFlow --> ValDemo["Validate Dialect, Gender, Age, Mic, Distance"]
    ValDemo --> LedgerWrite["register_speaker() -> consent_ledger_human.json"]
    
    RegChoice -->|Live Session (mock_mode=False)| LiveRec["run_recording_session()"]
    LiveRec --> EnforceStrict["Force effective_strict = True"]
    EnforceStrict --> MetaResolve["Resolve Mic/Dist from Ledger or CLI Args"]
    MetaResolve --> AudioRec["AudioRecorder (Windows MCI 44.1kHz -> 16kHz)"]
    AudioRec --> IngestCall["ingest_recording()"]
    
    RegChoice -->|"--ingest-file"| SingleFileIngest["Ingest External Audio File"]
    SingleFileIngest --> EnforceIngestStrict["Force effective_strict = True (human_field_recording)"]
    EnforceIngestStrict --> IngestCall
    
    IngestCall --> CheckConsent{"verify_speaker_consent()"}
    CheckConsent -->|False| QuarantConsent["Quarantine: unconsented_speaker"]
    CheckConsent -->|True| StrictGate{"effective_strict == True"}
    
    StrictGate -->|Verify Metadata| StrictAudit{"All Metadata Present & Consistent?"}
    StrictAudit -->|Missing / Invalid / Contradictory| QuarantMeta["Quarantine: Reason logged to quarantine_log.csv"]
    StrictAudit -->|Valid| QualityGate{"validate_field_audio() (Duration, Clipping, RMS, Duplicates)"}
    
    QualityGate -->|Fail| QuarantQuality["Quarantine: Audio quality rejection"]
    QualityGate -->|Pass| ManifestWrite["Append Canonical 28-Col Manifest Row & Save PCM"]
```

### 3.2 Key Architectural Remediations Implemented

1. **`record_command_cli.py` (Live Sessions):**
   - In `run_recording_session()`:
     ```python
     resolved_mic = microphone_type or spk_meta.get("primary_microphone")
     resolved_dist = distance_category or spk_meta.get("distance_category") or spk_meta.get("speaking_distance")
     effective_strict = True if not mock_mode else strict_metadata
     ```
   - When running a genuine recording session (`not mock_mode`), `strict_metadata` cannot be set to `False` through command-line flags. Live human recordings must pass strict metadata checks.
   - For simulated test harnesses (`mock_mode=True`), `effective_strict = strict_metadata`, preserving backwards compatibility with test fixtures.

2. **`record_command_cli.py` (`--ingest-file` Mode):**
   - Automatically computes:
     ```python
     effective_strict = True if not args.mock else args.strict
     prov = "human_field_recording" if not args.mock else "test_harness_mock"
     ```
   - Ingesting human recordings from external files cannot bypass strict mode unless explicitly marked as a mock harness.

3. **`ingest_field_recordings.py` (CLI Execution):**
   - CLI entry point enforces:
     ```python
     effective_strict = args.strict or (args.provenance == "human_field_recording")
     ```
   - Any execution targeting `"human_field_recording"` automatically enables strict checks.

4. **`ingest_field_recordings.py` (Elimination of Silent Defaults):**
   - In `ingest_recording()`:
     ```python
     age = age_bracket or spk_meta.get("age_bracket") or ("18-30" if not strict_metadata else None)
     mic_type = microphone_type or spk_meta.get("primary_microphone") or ("builtin_mobile" if not strict_metadata else None)
     dist_from_spk = spk_meta.get("distance_category") or spk_meta.get("speaking_distance")
     dist_cat = distance_category or dist_from_spk or ("close_talk_15cm" if not strict_metadata else None)
     ```
   - Under `strict_metadata=True`, if a field is absent from both arguments and `spk_meta`, it resolves to `None`.
   - The strict validator checks:
     - `(age_bracket is None and "age_bracket" not in spk_meta) or age not in [...]`
     - `(microphone_type is None and "primary_microphone" not in spk_meta) or mic_type not in VALID_MICROPHONE_TYPES`
     - `(distance_category is None and not dist_from_spk) or dist_cat not in VALID_DISTANCE_CATEGORIES`
   - Rejection reasons are recorded in `quarantine_log.csv` (e.g. `missing_or_invalid_distance_category_None`, `unregistered_or_invalid_microphone_type_None`).

5. **`interactive_registration()` Protocol:**
   - Prompts contributors interactively with explicit selection menus:
     - **Microphones (1-5):** `builtin_mobile`, `wired_headset`, `bluetooth_earbuds`, `laptop_mic`, `usb_desktop_mic`.
     - **Distances (1-3):** `close_talk_15cm`, `medium_talk_50cm`, `far_field_2m`.
   - Persists selections directly to `consent_ledger_human.json` as `primary_microphone`, `distance_category`, and `speaking_distance`.

---

## 4. Corpus Licenses & Provenance Matrix

The following forensic matrix provides exact evidence from the repository filesystem, distinguishing observed files from software claims:

| Corpus / Asset | Official Source | Version / Scope | Claimed License | Actual Disk Evidence | Disk SHA-256 Digest | Status & Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FLEURS (`vi_vn`)** | Google Research / HuggingFace (`google/fleurs`) | `vi_vn` subset (dev / test splits) | CC-BY-4.0 | TSV and archive files present in `data/raw/fleurs_vi/`. Standalone `LICENSE` file **not found** on disk. | `dev.tsv` (247,001 B): `C9BC17CEDE9765B1C75CB7A608F7066CB6BB6F8245CC75D721A5217A5DFFB414`<br>`test.tsv` (613,728 B): `EB744C2BE5F677B49C35527D428A4F647294BD0F67ABD790C296E91EFAAFFB93`<br>`dev.tar.gz` (214.5 MB): `8821A394C99069409B3CE7BB5CD14B10B709F2EE2FEA16F42B4701E1CC9EF673`<br>`test.tar.gz` (544.1 MB): `B359261216EF14B8EEA159ABCE794E270C2A0BE903218CF1E1E8F7C7C1565B5C` | **`PROVENANCE KNOWN — LICENSE FILE MISSING ON DISK`** *(Pending Human Approval: Downstream commercial usage requires inclusion of CC-BY-4.0 attribution notice)* |
| **MSWC (`vi`)** | Harvard / MLCommons Multilingual Spoken Words Corpus | `vi` split archive | CC-BY-4.0 | Splits archive and metadata present in `data/raw/mswc_vi/`. Standalone `LICENSE` file **not found** on disk. | `splits.tar.gz` (4,780 B): `964F485ACE314657B0F3C5F1BA7E97EF1814D0E1379DBEB0BD26E4526A9B8C00`<br>`splits_extracted/version.txt` (97 B): `42F75458F61C04A7832DCBE77932A6F2C72BB3634E881F6BAD0D67CDBE5DE824` | **`PROVENANCE KNOWN — LICENSE FILE MISSING ON DISK`** *(Pending Human Approval: Commercial training usage requires inclusion of CC-BY-4.0 attribution notice)* |
| **Husein `mic-noise`** | GitHub repository (`huseinzol05/malay-dataset`) | Audio noise augmentation samples | CC-BY-4.0 *(claimed in code comments)* | `data/raw/noise/mic-noise.zip` (45.9 MB) present on disk. Standalone `LICENSE` text **absent**. Terms of use unverified. | `mic-noise.zip`: `2757A2E1CE8AA9B2DA0718323A51137B08A70C2D0378D00E5A014D72FC26BD6B` | **`BLOCKED — UNVERIFIED LICENSE`** *(Downloader in `download_approved_sources.py` and validator in `campaign_validator.py` strictly block this source from all training and benchmarking)* |

---

## 5. Test Suite Verification & Sandbox Isolation

### 5.1 Test Execution Command & Summary
- **Execution Command:** `.\.venv\Scripts\python.exe -m pytest tests/`
- **Environment:** Python 3.11.9, pytest 9.1.1, Windows NT
- **Exit Code:** `0`
- **Duration:** 10.37 seconds
- **Results:** **44 passed, 0 failed, 0 errors, 0 warnings**

### 5.2 Test Inventory by Category

| Test Suite / File | Test Name | Description | Status |
| :--- | :--- | :--- | :--- |
| `test_campaign_acceptance.py` | `test_valid_human_manifest_acceptance` | TC-01: Compliant human manifest validation | **PASSED** |
| `test_campaign_acceptance.py` | `test_unconsented_speaker_rejection` | TC-02: Unconsented speaker exclusion | **PASSED** |
| `test_campaign_acceptance.py` | `test_missing_required_metadata_rejection` | TC-03: Missing required column rejection | **PASSED** |
| `test_campaign_acceptance.py` | `test_invalid_provenance_tag_rejection` | TC-04: Non-approved provenance tag rejection | **PASSED** |
| `test_campaign_acceptance.py` | `test_duplicate_audio_hash_quarantine` | TC-05: Byte-identical audio quarantine | **PASSED** |
| `test_campaign_acceptance.py` | `test_duplicate_recording_id_quarantine` | TC-06: Duplicate recording ID quarantine | **PASSED** |
| `test_campaign_acceptance.py` | `test_digital_clipping_gate` | TC-07: Clipping detection gate | **PASSED** |
| `test_campaign_acceptance.py` | `test_abnormal_silence_gate` | TC-08: Excessive silence gate (< -45 dBFS) | **PASSED** |
| `test_campaign_acceptance.py` | `test_speaker_disjoint_split_generator` | TC-09: Disjoint partition generator | **PASSED** |
| `test_campaign_acceptance.py` | `test_speaker_leakage_detector` | TC-10: Cross-split speaker leakage detector | **PASSED** |
| `test_campaign_acceptance.py` | `test_campaign_acceptance_zero_production_pollution` | TC-11: Zero production pollution validation | **PASSED** |
| `test_campaign_acceptance.py` | `test_p0_1_production_path_separation_and_defaults` | P0-1: Path separation from historical fixtures | **PASSED** |
| `test_campaign_acceptance.py` | `test_p0_2_source_license_gate` | P0-2: Unverified source blocker | **PASSED** |
| `test_campaign_acceptance.py` | `test_p0_3_pseudonymous_speaker_id_generation` | P0-3: Cryptographic nonce ID generator | **PASSED** |
| `test_campaign_acceptance.py` | `test_p1_1_manifest_schema_parity_28_columns` | P1-1: Canonical 28-column schema parity | **PASSED** |
| `test_campaign_acceptance.py` | `test_p1_2_stratified_disjoint_split_generator` | P1-2: Demographic stratification split | **PASSED** |
| `test_campaign_acceptance.py` | `test_p1_3_consent_revocation_workflow` | P1-3: Consent revocation & file deletion | **PASSED** |
| `test_campaign_acceptance.py` | `test_strict_human_metadata_validation` | P1-1b: Strict human metadata enforcement | **PASSED** |
| `test_campaign_acceptance.py` | `test_contradictory_metadata_rejection` | P1-1c: Contradictory metadata rejection | **PASSED** |
| `test_campaign_acceptance.py` | `test_revocation_partial_failure_recovery` | P1-3b: Revocation error handling | **PASSED** |
| `test_campaign_acceptance.py` | `test_recording_and_lineage_disjointness` | Task 1: Cross-split recording & lineage check | **PASSED** |
| `test_campaign_acceptance.py` | `test_fail_closed_production_strict_metadata` | **TC-Strict-01: Fail-closed strict validation** | **PASSED** |
| `test_campaign_acceptance.py` | `test_registration_flow_persists_mic_and_distance` | **TC-Strict-02: Interactive mic/dist persistence** | **PASSED** |
| `test_campaign_acceptance.py` | `test_recording_session_strict_fail_closed` | **TC-Strict-03: Session strict-mode enforcement** | **PASSED** |
| `test_field_acquisition.py` | 13 test functions | Legacy field acquisition pipeline tests | **PASSED** |
| `test_parity.py` | 7 test functions | Audio processing & numerical parity tests | **PASSED** |

---

## 6. Comprehensive Technical Audit Findings & Status Matrix

| Governance / Technical Domain | Sub-Item / Component | Current Implementation | Audit Finding & Status |
| :--- | :--- | :--- | :--- |
| **Cryptographic Baselines** | `test_manifest_v2.csv` | Protected frozen benchmark | **`VERIFIED`** (Hash `9E39...` bit-exact) |
| | `field_collection_manifest.csv` | Preserved historical fixture manifest | **`VERIFIED`** (Hash `B389...` bit-exact) |
| | `consent_ledger.json` | Preserved historical fixture consent ledger | **`VERIFIED`** (Hash `BE4B...` bit-exact) |
| **Pipeline Strict-Mode** | CLI to Ingestion Call-Path | `record_command_cli.py` & `ingest_field_recordings.py` | **`VERIFIED`** (Live sessions enforce `effective_strict=True`) |
| | Default Metadata Fallback | Ingestion strict validator | **`VERIFIED`** (Silent defaults removed; fail-closed quarantine) |
| | Interactive Registration | Numerical menu for Mic & Distance | **`VERIFIED`** (Persists mic and distance to consent ledger) |
| | Contradictory Metadata Gate | `campaign_validator.py` & ingestion pipeline | **`VERIFIED`** (Rejects conflicting distance/device combinations) |
| **Data Partitioning** | Stratified Disjoint Splitter | `generate_stratified_disjoint_splits()` | **`VERIFIED`** (Speaker, recording ID, and lineage disjoint) |
| | Stratum Rounding & Infeasible Bins | Small demographic group handling | **`NOT VERIFIED (STRUCTURAL DEFICIT)`** (Exact strata balance cannot be guaranteed mathematically when speaker counts per stratum are odd or small) |
| **Consent & Revocation** | Pseudonymous ID Generation | `generate_pseudonymous_speaker_id()` | **`VERIFIED`** (Cryptographic 6-hex nonce via `secrets` module) |
| | Consent Revocation CLI | `revoke_speaker_consent.py` | **`VERIFIED`** (Atomic file purging and manifest status updates) |
| | Cached Model Weights Policy | Consent revocation policy | **`VERIFIED`** (Explicitly documents that revoked data cannot be deleted from pre-trained model weights; requires retraining) |
| **External Corpus Licenses** | Google FLEURS (`vi_vn`) | Dev/Test split TSV/audio archives on disk | **`PENDING HUMAN APPROVAL`** (CC-BY-4.0 claimed; license text missing on disk) |
| | MLCommons MSWC (`vi`) | Splits archive on disk | **`PENDING HUMAN APPROVAL`** (CC-BY-4.0 claimed; license text missing on disk) |
| | Husein `mic-noise` | Noise augmentation samples | **`BLOCKED`** (Unverified license; downloader and pipeline strictly block usage) |
| **Operational Approvals** | Sampling Quota (50N/30C/40S) | Research campaign protocol proposal | **`PENDING HUMAN APPROVAL`** (Engineering proposal only; awaiting human stakeholder approval) |
| | Physical Human Recording | Live participant recruitment | **`STRICTLY BLOCKED`** (No recording until legal/IRB/quota sign-off) |

---

## 7. Next Actions for Authorizing Stakeholders

Before human speech recording begins, the project lead and compliance officers should complete the following sign-off steps:

1. **Formal Sign-off on Participant Sampling Quota:**  
   Review and formally approve the proposed `50N / 30C / 40S` regional dialect distribution, or supply revised targets.
2. **Approval of License Attribution Documentation:**  
   Review the FLEURS (`vi_vn`) and MSWC (`vi`) provenance documentation in Section 4. Place official CC-BY-4.0 attribution and notice files in the repository root or data directory prior to model distribution.
3. **Formal Approval of Informed Consent Notice:**  
   Confirm that the informed consent text in [`speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py) conforms to your organization's ethical review requirements and regional privacy regulations.
4. **Unblocking Campaign Execution:**  
   Following completion of steps 1–3, transition the operational status in [`collection_protocol.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md) from `STRICTLY BLOCKED` to `ACTIVE COLLECTION`.
