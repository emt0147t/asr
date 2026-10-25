# Campaign Acceptance Test Plan & Quality Assurance

> **Document Version:** 1.0.0-PROPOSAL  
> **Campaign Date:** October 13, 2026  
> **Testing Scope:** Automated Pre-Collection Acceptance Gates & Leakage Defenses  
> **Execution Invariant:** 100% Sandbox Execution (`tmp_path`); Zero Production Mutation

---

## 1. Test Strategy & Objectives

Before recruiting human participants or accepting contributed audio, the software pipeline must be independently verified using automated tests.

### Primary Test Objectives:
1. **Verify Quality Gate Logic:** Prove that every quality control check (QC-01 through QC-14) correctly accepts valid audio and rejects/quarantines flawed takes.
2. **Verify Provenance Integrity:** Prove that synthetic test fixtures cannot masquerade as `HUMAN_CONSENTED`.
3. **Verify Speaker Disjointness:** Prove that the split generator guarantees zero speaker overlap and detects accidental leakage.
4. **Verify Test Sandbox Isolation:** Guarantee that executing acceptance test suites leaves all existing production assets (`data/raw/field_recordings/`, `test_manifest_v2.csv`) 100% byte-for-byte untouched.

---

## 2. Test Case Inventory (`tests/test_campaign_acceptance.py`)

| Test ID | Test Function | Purpose / Verification Criterion | Expected Result |
| :---: | :--- | :--- | :---: |
| **TC-01** | `test_valid_human_manifest_acceptance` | Verifies that a fully compliant human dataset passing all 14 gates is accepted with `STATUS: APPROVED`. | PASS |
| **TC-02** | `test_unconsented_speaker_rejection` | Attempts ingestion with a `speaker_id` missing from the consent ledger or with `consent_given == False`. | Rejection: `missing_or_unconsented_speaker` |
| **TC-03** | `test_missing_required_metadata_rejection` | Ingests a record with missing mandatory fields (`dialect`, `device_model`, or `session_id`). | Rejection: `missing_required_metadata` |
| **TC-04** | `test_invalid_provenance_tag_rejection` | Ingests an audio file with synthetic characteristics tagged as `HUMAN_CONSENTED` without ledger verification. | Rejection: `unverified_provenance_attestation` |
| **TC-05** | `test_duplicate_audio_hash_quarantine` | Ingests two different takes containing identical audio waveforms (duplicate SHA-256). | Second take quarantined: `duplicate_recording_sha256_match` |
| **TC-06** | `test_duplicate_recording_id_quarantine` | Ingests two distinct recordings assigned the identical `recording_id`. | Second take quarantined: `duplicate_recording_id` |
| **TC-07** | `test_digital_clipping_gate` | Ingests an audio take with $\ge 3$ consecutive clipped samples at full scale. | Quarantined: `digital_clipping_detected` |
| **TC-08** | `test_abnormal_silence_gate` | Ingests an unvoiced audio take with energy $\text{RMS} < -45.0$ dBFS. | Quarantined: `abnormal_silence` |
| **TC-09** | `test_speaker_disjoint_split_generator` | Generates 70/15/15 splits across 12 human speakers and verifies mathematical disjointness. | $S_{\text{train}} \cap S_{\text{val}} = \emptyset \land S_{\text{train}} \cap S_{\text{test}} = \emptyset$ |
| **TC-10** | `test_speaker_leakage_detector` | Deliberately injects a shared speaker across train and test sets into the validator. | Validator raises: `speaker_leakage_detected` |
| **TC-11** | `test_campaign_acceptance_zero_production_pollution` | Snapshots production file hashes before and after TC-01 through TC-10 execution. | 100% Bit-Exact Match (`is_identical == True`) |

---

## 3. Test Execution Protocol

All tests must be run using the repository virtual environment:

```powershell
# 1. Run core regression suite (20 tests)
.\.venv\Scripts\python.exe -m pytest -v tests/test_field_acquisition.py tests/test_parity.py

# 2. Run campaign acceptance test suite
.\.venv\Scripts\python.exe -m pytest -v tests/test_campaign_acceptance.py

# 3. Verify benchmark invariant after all test runs
powershell -Command "(Get-FileHash data/manifests/test_manifest_v2.csv -Algorithm SHA256).Hash"
```

Expected Benchmark SHA-256:  
`9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`
