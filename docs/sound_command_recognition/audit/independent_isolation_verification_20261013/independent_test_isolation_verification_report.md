# Independent Forensic Verification Report: Test Isolation & Field-Data Integrity

> **Project:** Production-Oriented Vietnamese Keyword Spotting (KWS)  
> **Workspace:** `D:\trainidentificationaudio`  
> **Audit Date:** October 13, 2026  
> **Auditor:** Independent Quality & Forensic Verification Gate  
> **Benchmark Invariant:** `data/manifests/test_manifest_v2.csv`  
> **Benchmark SHA-256:** `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`  
> **Verification Verdict:** **PASS WITH OPEN ISSUES**

---

## 1. Executive Summary & Verification Scope

This independent verification was conducted to empirically evaluate the recent test-isolation repair in the Vietnamese Keyword Spotting repository. In accordance with strict audit policy, the previous completion claim was not accepted at face value; instead, the codebase, execution paths, cryptographic checksums, recursive file trees, and provenance records were independently probed.

### Primary Scope of Investigation:
1. **Source Code & Invocation Chain Audit:** Forensic examination of all acquisition functions (`speaker_consent.py`, `ingest_field_recordings.py`, `record_command_cli.py`, `audit_field_campaign.py`, `test_field_acquisition.py`) for path parameter propagation and early-bound default argument traps.
2. **Double-Run Production Immutability Verification:** Taking a complete cryptographic snapshot of production files and audio directories, executing the field acquisition test suite twice consecutively, and comparing the post-run state against the baseline.
3. **Full Regression Suite Verification:** Executing the complete repository test suite (`pytest -v tests/`, 20 tests) and confirming zero byte-level modifications to any production asset.
4. **Manifest Provenance Reconciliation (106 Rows vs. 99 Files):** Building an exhaustive, row-by-row reconciliation table correlating manifest entries, disk existence, physical file hashes, and test harness provenance.
5. **Frozen Benchmark Protection:** Continuous verification of `test_manifest_v2.csv`.

---

## 2. Files and Functions Inspected

The following modules, functions, and test fixtures were inspected for path parameter handling, isolation enforcement, and potential production leaks:

| File Inspected | Functions / Components Audited | Path Handling & Isolation Implementation |
| :--- | :--- | :--- |
| `src/acquisition/speaker_consent.py` | `load_consent_ledger`<br/>`save_consent_ledger`<br/>`register_speaker`<br/>`verify_speaker_consent`<br/>`get_speaker_metadata`<br/>`list_registered_speakers` | Evaluates `ledger_path if ledger_path is not None else LEDGER_PATH` dynamically at call time; supports explicit path injection and module monkeypatching. Enforces regex validation on `custom_speaker_id`. |
| `src/acquisition/ingest_field_recordings.py` | `load_existing_manifest`<br/>`validate_field_audio`<br/>`ingest_recording`<br/>`batch_ingest_directory` | Resolves target paths dynamically (`manifest_path`, `quarantine_path`, `raw_base_dir`, `std_base_dir`, `ledger_path`). Added parameter forwarding in `batch_ingest_directory`. Defends against path traversal via `re.match` and `.resolve()`. |
| `src/acquisition/record_command_cli.py` | `run_recording_session`<br/>`interactive_registration`<br/>`AudioRecorder` | Passes optional path parameters (`manifest_path`, `quarantine_path`, `raw_base_dir`, `std_base_dir`, `ledger_path`) through to `verify_speaker_consent`, `get_speaker_metadata`, and `ingest_recording`. Cleans up temp WAV files in `finally:` block. |
| `src/acquisition/audit_field_campaign.py` | `audit_field_campaign` | Defines module-level `REPORT_PATH`. Imports and resolves current module constants dynamically inside function body if arguments are `None`. Segregates genuine speakers from test fixtures via regex and consent ledger. |
| `src/preprocessing/audio_normalizer.py` | `load_and_resample` | Added fallback handler for compressed M4A/AAC decoding via `librosa`/`audioread`/`ffmpeg`. Preserves standard float32 16 kHz mono output. |
| `tests/test_field_acquisition.py` | `isolate_field_environment` (fixture)<br/>`setup_test_speaker` (fixture)<br/>12 core field acquisition tests<br/>`test_no_production_pollution_regression` | Implements `autouse=True` fixture that provisions an isolated sandbox in `tmp_path`, monkeypatches all path constants across core modules and test module globals. Refactors `setup_test_speaker` to sandbox ledger. Tests Actions 1–5 in regression test. |

---

## 3. Test Commands and Verbatim Execution Logs

### 3.1 Pre-Test Baseline Snapshot
- **Command:** `python scratch/audit_tools.py snapshot`
- **Exit Code:** `0`
- **Output:**
  ```
  Snapshot written to docs/sound_command_recognition/audit/independent_isolation_verification_20261013/baseline_snapshot.json
  Files recorded: 5
  Directory data/raw/field_recordings/audio_raw: 99 files
  Directory data/raw/field_recordings/audio_16k_standardized: 99 files
  ```

### 3.2 Field Acquisition Suite — Run 1
- **Command:** `.\.venv\Scripts\python.exe -m pytest -v tests/test_field_acquisition.py`
- **Exit Code:** `0`
- **Output:**
  ```
  ============================= test session starts =============================
  platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- D:\trainidentificationaudio\.venv\Scripts\python.exe
  cachedir: .pytest_cache
  rootdir: D:\trainidentificationaudio
  plugins: anyio-4.15.1, platformdirs-4.13.0
  collecting ... collected 13 items

  tests/test_field_acquisition.py::test_consent_enforcement PASSED         [  7%]
  tests/test_field_acquisition.py::test_quality_gate_clipping_rejection PASSED [ 15%]
  tests/test_field_acquisition.py::test_quality_gate_silence_rejection PASSED [ 23%]
  tests/test_field_acquisition.py::test_quality_gate_duration_bounds PASSED [ 30%]
  tests/test_field_acquisition.py::test_raw_preservation_and_standardization PASSED [ 38%]
  tests/test_field_acquisition.py::test_duplicate_sha256_detection PASSED  [ 46%]
  tests/test_field_acquisition.py::test_mock_cli_recording_session PASSED  [ 53%]
  tests/test_field_acquisition.py::test_audit_field_campaign_generation PASSED [ 61%]
  tests/test_field_acquisition.py::test_security_path_traversal_rejection PASSED [ 69%]
  tests/test_field_acquisition.py::test_duplicate_recording_id_rejection PASSED [ 76%]
  tests/test_field_acquisition.py::test_m4a_audio_ingestion_and_decoding PASSED [ 84%]
  tests/test_field_acquisition.py::test_windows_hardware_recorder_configuration PASSED [ 92%]
  tests/test_field_acquisition.py::test_no_production_pollution_regression PASSED [100%]

  ============================= 13 passed in 6.42s ==============================
  ```
- **Snapshot Comparison 1:**
  - `python scratch/audit_tools.py compare "After Field Tests Run 1"`
  - `is_identical`: **`True`** (0 files changed, 0 directories changed).

### 3.3 Field Acquisition Suite — Run 2 (Repetition Check)
- **Command:** `.\.venv\Scripts\python.exe -m pytest -v tests/test_field_acquisition.py`
- **Exit Code:** `0`
- **Output:**
  ```
  ============================= 13 passed in 5.59s ==============================
  ```
- **Snapshot Comparison 2:**
  - `python scratch/audit_tools.py compare "After Field Tests Run 2"`
  - `is_identical`: **`True`** (0 files changed, 0 directories changed).

### 3.4 Full Test Suite Execution
- **Command:** `.\.venv\Scripts\python.exe -m pytest -v tests/`
- **Exit Code:** `0`
- **Output:**
  ```
  ============================= test session starts =============================
  platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- D:\trainidentificationaudio\.venv\Scripts\python.exe
  cachedir: .pytest_cache
  rootdir: D:\trainidentificationaudio
  plugins: anyio-4.15.1, platformdirs-4.13.0
  collecting ... collected 20 items

  tests/test_field_acquisition.py::test_consent_enforcement PASSED         [  5%]
  tests/test_field_acquisition.py::test_quality_gate_clipping_rejection PASSED [ 10%]
  tests/test_field_acquisition.py::test_quality_gate_silence_rejection PASSED [ 15%]
  tests/test_field_acquisition.py::test_quality_gate_duration_bounds PASSED [ 20%]
  tests/test_field_acquisition.py::test_raw_preservation_and_standardization PASSED [ 25%]
  tests/test_field_acquisition.py::test_duplicate_sha256_detection PASSED  [ 30%]
  tests/test_field_acquisition.py::test_mock_cli_recording_session PASSED  [ 35%]
  tests/test_field_acquisition.py::test_audit_field_campaign_generation PASSED [ 40%]
  tests/test_field_acquisition.py::test_security_path_traversal_rejection PASSED [ 45%]
  tests/test_field_acquisition.py::test_duplicate_recording_id_rejection PASSED [ 50%]
  tests/test_field_acquisition.py::test_m4a_audio_ingestion_and_decoding PASSED [ 55%]
  tests/test_field_acquisition.py::test_windows_hardware_recorder_configuration PASSED [ 60%]
  tests/test_field_acquisition.py::test_no_production_pollution_regression PASSED [ 65%]
  tests/test_parity.py::test_feature_extractor_shape_and_finite PASSED     [ 70%]
  tests/test_parity.py::test_baseline_onnx_model_execution_and_shape PASSED [ 75%]
  tests/test_parity.py::test_tcresnet_onnx_model_execution_and_parity PASSED [ 80%]
  tests/test_parity.py::test_pytorch_onnx_numerical_parity_baseline PASSED [ 85%]
  tests/test_parity.py::test_golden_vectors_reproducibility PASSED         [ 90%]
  tests/test_parity.py::test_manifest_speaker_disjointness_v2 PASSED       [ 95%]
  tests/test_parity.py::test_manifest_class_completeness_v2 PASSED         [100%]

  ============================= 20 passed in 11.55s =============================
  ```
- **Snapshot Comparison 3 (After Full Test Suite):**
  - `python scratch/audit_tools.py compare "After Full Suite Run"`
  - `is_identical`: **`True`** (0 files changed, 0 directories changed).

---

## 4. Before/After Cryptographic Hashes and Directory Comparison

### 4.1 Production Core Files Immutability Matrix

| Production File | Baseline Size | Baseline SHA-256 | Post-Run Size | Post-Run SHA-256 | Immutability Status |
| :--- | :---: | :--- | :---: | :--- | :---: |
| `data/raw/field_recordings/field_collection_manifest.csv` | 75,969 bytes | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | 75,969 bytes | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **IDENTICAL (0 bytes / 0 rows changed)** |
| `data/raw/field_recordings/consent_ledger.json` | 914 bytes | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | 914 bytes | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **IDENTICAL (0 bytes changed)** |
| `data/raw/field_recordings/quarantined/quarantine_log.csv` | 13,850 bytes | `EC923FEB7D6E421BC420184D5BCE9663BD83923EF2C8E2544D31A02D9B2156FD` | 13,850 bytes | `EC923FEB7D6E421BC420184D5BCE9663BD83923EF2C8E2544D31A02D9B2156FD` | **IDENTICAL (0 bytes / 0 rows changed)** |
| `reports/field_recording_campaign_audit.md` | 6,781 bytes | `C98BA03AC3FE2BE2D0836981FA13B2D4986090A07B7AF75D628424C57B1CBB96` | 6,781 bytes | `C98BA03AC3FE2BE2D0836981FA13B2D4986090A07B7AF75D628424C57B1CBB96` | **IDENTICAL (0 bytes changed)** |
| `data/manifests/test_manifest_v2.csv` | 216,889 bytes | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | 216,889 bytes | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **IDENTICAL (Benchmark Invariant Protected)** |

### 4.2 Recursive Audio Directory Comparison

- **`data/raw/field_recordings/audio_raw/`**:
  - File count before: **99 files** (across `SPK_FIELD_N001_TESTSUITE` and `SPK_M4A_TEST` subdirectories).
  - File count after: **99 files**.
  - New files added: **0**.
  - Files removed: **0**.
  - Files modified (hash mismatch): **0**.
- **`data/raw/field_recordings/audio_16k_standardized/`**:
  - File count before: **99 files** (across `TRAI`, `PHAI`, `LEN`, `XUONG` subdirectories).
  - File count after: **99 files**.
  - New files added: **0**.
  - Files removed: **0**.
  - Files modified (hash mismatch): **0**.

---

## 5. Test Isolation Findings and Gaps Analysis

### 5.1 Effectiveness of Sandbox Mechanism
- The `@pytest.fixture(autouse=True)` in `tests/test_field_acquisition.py` completely isolates the file writes generated by all 13 field acquisition test functions.
- Every test receives a unique, isolated directory under `pytest`'s managed `tmp_path`.
- The monkeypatching strategy covers:
  - `src.acquisition.speaker_consent.LEDGER_PATH`
  - `src.acquisition.ingest_field_recordings.FIELD_BASE_DIR`
  - `src.acquisition.ingest_field_recordings.LEDGER_PATH`
  - `src.acquisition.ingest_field_recordings.RAW_AUDIO_DIR`
  - `src.acquisition.ingest_field_recordings.STANDARDIZED_AUDIO_DIR`
  - `src.acquisition.ingest_field_recordings.QUARANTINE_DIR`
  - `src.acquisition.ingest_field_recordings.QUARANTINE_LOG_PATH`
  - `src.acquisition.ingest_field_recordings.MANIFEST_PATH`
  - `src.acquisition.audit_field_campaign.LEDGER_PATH`
  - `src.acquisition.audit_field_campaign.MANIFEST_PATH`
  - `src.acquisition.audit_field_campaign.QUARANTINE_LOG_PATH`
  - `src.acquisition.audit_field_campaign.REPORT_PATH`
  - Current test module globals: `MANIFEST_PATH`, `QUARANTINE_LOG_PATH`, `RAW_AUDIO_DIR`, `STANDARDIZED_AUDIO_DIR`
- In addition, the runtime parameter forwarding added to `batch_ingest_directory` ensures that even directory-level batch ingestion commands can be sandboxed cleanly without relying solely on global monkeypatching.

### 5.2 Remaining Gaps / Limitations
1. **Unparameterized Standalone CLI Invocations:** Direct execution of `src/acquisition/record_command_cli.py` or `src/acquisition/ingest_field_recordings.py` via command line (`python src/... --file ...`) without `--manifest` or `--ledger` CLI flags will target production paths. This is intentional for field data collection, but operators must be aware that running the CLI interactively targets production.
2. **Pre-Existing Historical Test Fixtures on Disk:** The 106 manifest rows and 99 audio files accumulated prior to this repair remain in `data/raw/field_recordings/` (protected from accidental deletion). They must not be conflated with genuine human field data.

---

## 6. Manifest Provenance Reconciliation (106 Rows vs. 99 Files)

A dedicated forensic reconciliation table was generated at:  
[`docs/sound_command_recognition/audit/independent_isolation_verification_20261013/manifest_reconciliation_table.csv`](./manifest_reconciliation_table.csv)

### 6.1 Statistical Summary of Manifest Reconciliation

| Attribute | Verified Value | Evidence / Analysis |
| :--- | :---: | :--- |
| **Total Manifest Rows** | **106 rows** | Directly read from `data/raw/field_recordings/field_collection_manifest.csv` |
| **Raw Files Existing on Disk** | **106 / 106 (100%)** | Every path cited in `raw_filepath` resolves to an existing file on disk |
| **Standardized Files Existing on Disk** | **106 / 106 (100%)** | Every path cited in `processed_path` resolves to an existing file on disk |
| **Unique `raw_filepath` Values** | **99 unique paths** | Exactly 99 distinct physical file paths referenced across the 106 rows |
| **Unique `processed_path` Values** | **99 unique paths** | Exactly 99 distinct physical file paths referenced across the 106 rows |
| **Physical Raw Files on Disk** | **99 files** | Verified by recursive directory scan of `audio_raw/` (95 `.wav`, 4 `.m4a`) |
| **Physical Standardized Files on Disk** | **99 files** | Verified by recursive directory scan of `audio_16k_standardized/` (all `.wav`) |
| **Missing Files on Disk** | **0 files** | 0 missing files |
| **Rows with Exact Disk SHA-256 Match** | **99 rows (93.4%)** | Exactly matches the 99 distinct physical files currently on disk |
| **Rows with SHA-256 Mismatch vs. Disk** | **7 rows (6.6%)** | Explained by historical file overwrites (detailed below) |
| **Provenance Classification: Genuine Human** | **0 rows (0.0%)** | Zero recordings possess verified human consent or native field origin |
| **Provenance Classification: Test Fixture** | **106 rows (100.0%)** | 100% are synthetic voice simulations generated during test harness runs |

### 6.2 Forensic Explanation of the 7 SHA-256 Mismatches (Historical Overwriting)

The 7 rows where the SHA-256 recorded in the manifest differs from the file currently on disk are **not corrupted files**. They represent earlier test runs that wrote to static, non-randomized filenames before UUID suffixing was added:

1. **Group 1 (`SES_PRESERVE_TEST` — 4 rows pointing to 1 physical file):**
   - Manifest Row 1, Row 4, Row 10, Row 16 all target:  
     `audio_raw/SPK_FIELD_N001_TESTSUITE/SES_PRESERVE_TEST/field_trai_SPK_FIELD_N001_TESTSUITE_SES_PRESERVE_TEST_take001.wav`
   - In each of the first four unisolated test runs, `shutil.copy2` overwrote this exact file with a newly generated synthetic take containing unique random jitter.
   - The file on disk today is the output of the **fourth write** (Row 16).
   - Consequently, **Row 16 matches disk SHA-256**, while Rows 1, 4, and 10 record historical hashes from earlier takes that were subsequently overwritten (3 mismatches).
2. **Group 2 (`SES_DUP_TEST` — 4 rows pointing to 1 physical file):**
   - Manifest Row 2, Row 5, Row 11, Row 17 all target:  
     `audio_raw/SPK_FIELD_N001_TESTSUITE/SES_DUP_TEST/field_phai_SPK_FIELD_N001_TESTSUITE_SES_DUP_TEST_take001.wav`
   - Row 17 matches the current disk file; Rows 2, 5, and 11 represent overwritten takes (3 mismatches).
3. **Group 3 (`SES_MOCK_TEST` — 2 rows pointing to 1 physical file):**
   - Manifest Row 3 and Row 6 both target:  
     `audio_raw/SPK_FIELD_N001_TESTSUITE/SES_MOCK_TEST/field_trai_SPK_FIELD_N001_TESTSUITE_SES_MOCK_TEST_take001.wav`
   - Row 6 matches the current disk file; Row 3 represents an overwritten take (1 mismatch).

$$\text{Total Overwritten Takes} = 3 + 3 + 1 = \mathbf{7\text{ historical entries}}$$
$$\text{Total Manifest Rows (106)} - \text{Overwritten Entries (7)} = \mathbf{99\text{ Unique Physical Files on Disk}}$$

This accounts for every single row in the manifest and proves that no files are missing or unexplained.

### 6.3 Provenance Categorization
In accordance with strict dataset governance:
- **`UNVERIFIED_OR_TEST_FIXTURE` (106 rows, 100%):**  
  All 106 rows are labeled `UNVERIFIED_OR_TEST_FIXTURE`. 105 rows belong to test speaker `SPK_FIELD_N001_TESTSUITE` (device: `Test Harness Device`), and 1 row belongs to test speaker `SPK_M4A_TEST` (device: `Unspecified Mobile`). All audio signals were generated by `generate_mock_human_audio()` (a 150 Hz harmonic tone with jitter) or synthesized ffmpeg AAC streams.
- **Genuine Human Audio Takes: ZERO (0).**  
  No recording in `data/raw/field_recordings/` represents real human speech.

---

## 7. Frozen Benchmark Manifest Verification

The held-out test manifest `data/manifests/test_manifest_v2.csv` was verified before and after all audit operations:

```powershell
powershell -Command "(Get-FileHash data/manifests/test_manifest_v2.csv -Algorithm SHA256).Hash"
```
**Observed SHA-256:**
```
9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13
```
- **Byte Count:** Exactly 216,889 bytes.
- **Row Count:** Exactly 534 evaluation samples (36 TRAI, 36 PHAI, 36 LEN, 36 XUONG, 300 UNKNOWN, 90 SILENCE).
- **Match Status:** **100% BIT-EXACT INVARIANCE PRESERVED.**

---

## 8. Git Status and Uncommitted Files Summary

Current working tree status:

```
On branch master
Changes not staged for commit:
	modified:   docs/sound_command_recognition/audit/documentation_repair_log.md
	modified:   docs/sound_command_recognition/audit/final_evidence_consistency_report.md
	modified:   reports/field_recording_campaign_audit.md
	modified:   src/acquisition/audit_field_campaign.py
	modified:   src/acquisition/ingest_field_recordings.py
	modified:   src/acquisition/record_command_cli.py
	modified:   src/acquisition/speaker_consent.py
	modified:   src/preprocessing/audio_normalizer.py
	modified:   tests/test_field_acquisition.py

Untracked files:
	data/backups/
	data/processed/
	data/raw/
	docs/objective_1/
	docs/sound_command_recognition/
	scripts/
```

- No user files or commits have been reset or discarded.
- `git diff --check` passes cleanly without conflict markers.

---

## 9. Final Gate Verdict & Operational Recommendations

### Final Verdict: `PASS WITH OPEN ISSUES`

#### Justification for `PASS`:
1. **Test Isolation is 100% Operational:** Consecutive test runs (Run 1, Run 2, and the full 20-test repository suite) generated **zero byte modifications** and **zero new files** in production directories (`is_identical = True`).
2. **Regression Test Suite Passes (20/20):** Both the unit test suite and parity test suite execute with exit code 0.
3. **Discrepancy 106 vs. 99 Fully Reconciled:** The difference between 106 manifest rows and 99 files on disk is proven to be 7 historical file overwrites across static session IDs.
4. **Frozen Benchmark Invariant Intact:** `test_manifest_v2.csv` was preserved bit-for-bit.

#### Open Technical Issues (Requiring Subsequent Project Phases):
- **`ISSUE-01` (CRITICAL / P0):** Zero genuine human target command recordings exist in `data/raw/field_recordings/`. All 1,000 target command samples in the active splits remain synthetic parametric audio.
- **`ISSUE-02` (HIGH / P1):** Northern native Vietnamese `TRAI` recall is limited to 25.0% due to affricate onset `/tɕ/` modeling limits in synthetic audio.
- **`ISSUE-03` (HIGH / P2):** Missing TC-ResNet INT8 ONNX export (`vietnamese_kws_tcresnet_int8.onnx` is absent from disk).
- **`ISSUE-04` (OPERATIONAL):** Accumulated historical test fixtures (106 manifest rows, 99 audio files) remain in `data/raw/field_recordings/` and should be safely purged or archived prior to launching genuine human collection campaigns.
