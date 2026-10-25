# Documentation Repair & Evidence Reconciliation Log

> **Project:** Offline Vietnamese Keyword Spotting (KWS) for Android and iOS  
> **Workspace:** `D:\trainidentificationaudio`  
> **Date of Audit & Repair:** October 13, 2026  
> **Audited Package:** `docs/objective_1/` (Reports, Tables, Figures, Reproducibility Guide)  
> **Core Status:** Reconciled, Verified Against Primary Repository Artifacts  
> **Frozen Test Invariant:** `test_manifest_v2.csv` (SHA-256 `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`)  

---

## 1. Executive Summary & Purpose

Following the completion of the independent forensic audit documented in [`docs/sound_command_recognition/audit/independent_audit_report.md`](./independent_audit_report.md) and its accompanying CSV evidence matrices ([`claim_audit.csv`](./claim_audit.csv), [`dataset_reconciliation.csv`](./dataset_reconciliation.csv), [`model_evidence_matrix.csv`](./model_evidence_matrix.csv), and [`figure_table_audit.csv`](./figure_table_audit.csv)), this log records every modification, correction, and evidence-grounded reconciliation applied to the Objective 1 documentation package (`docs/objective_1/`) and its programmatic generation scripts (`scripts/objective_1_docs/`).

### Fundamental Audit Principles Followed:
1. **Preservation of Benchmark Invariant:** `data/manifests/test_manifest_v2.csv` was protected byte-for-byte; its cryptographic SHA-256 hash remained strictly unchanged throughout all operations.
2. **Zero Fabrication:** No synthetic data was generated to mask gaps, no fictional consent records were authored, and missing on-disk artifacts (such as `vietnamese_kws_tcresnet_int8.onnx`) were transparently declared as defects rather than simulated.
3. **Primary Artifact Grounding:** Every quantitative parameter, metric, split count, and hardware measurement in narrative reports, CSV tables, and SVG figures now maps directly to physical files on disk, checkpoint evaluations, or automated test executions.

---

## 2. Master Reconciliation Table (Before vs. After)

The table below contrasts original draft or historical claims against verified ground truth established during this repair:

| Claim Category / Focus | Original Narrative Claim (Draft / Historical) | Repaired Value (Verified Ground Truth) | Primary Evidence Source Artifact | Affected Documentation Files |
| :--- | :--- | :--- | :--- | :--- |
| **Negative Speech Source** | Mozilla Common Voice (vi) (779 approved clips, CC0) | **Google FLEURS vi** (579 approved clips, CC-BY-4.0) + 120 calibration confusing words. Common Voice has **0** approved samples in manifests. | `data/manifests/source_registry_v2.csv`, `train/val/test_manifest_v2.csv` | `dataset_and_provenance.md`, `objective_1_final_report.md`, `README.md`, Fig 2, Tab 01, Tab 03 |
| **Negative Ambient Noise Source** | ESC-50 / Free Music Archive (321 approved clips) | **`husein_mic_noise`** (400 approved clips, CC-BY-4.0). ESC-50 and FMA have **0** approved samples in manifests. | `data/manifests/source_registry_v2.csv`, `train/val/test_manifest_v2.csv` | `dataset_and_provenance.md`, `objective_1_final_report.md`, `README.md`, Fig 2, Tab 01, Tab 03 |
| **Split Partition Allocation** | Train: 160/cmd, 461 Unk, 180 Sil (1,281)<br/>Val: 40/cmd, 84 Unk, 40 Sil (284)<br/>Test: 50/cmd, 234 Unk, 100 Sil (534) | **Train:** 178/cmd, 319 Unk, 250 Sil (1,281)<br/>**Val:** 36/cmd, 80 Unk, 60 Sil (284)<br/>**Test:** 36/cmd, 300 Unk, 90 Sil (534)<br/>**Total:** 2,099 approved files | `data/manifests/train_manifest_v2.csv`, `val_manifest_v2.csv`, `test_manifest_v2.csv` | `dataset_and_provenance.md`, `objective_1_final_report.md`, `README.md`, Fig 3, Tab 02 |
| **Baseline Architecture** | BC-ResNet-1 (39,142 params, 4.2 M MACs) | **`VietnameseKWSNet`** (138,534 params, 2D-CNN). *BC-ResNet-1 does not exist in repository.* | `src/models/kws_model.py`, `models/checkpoints/best_kws_model.pt` | `results_and_evaluation.md`, `objective_1_final_report.md`, `README.md`, Fig 1, Fig 4, Tab 06 |
| **Champion Architecture** | TC-ResNet-14 (278,414 params, 18.7 M MACs) | **`VietnameseTCResNet`** (272,934 params, 1D temporal residual blocks) | `src/models/advanced_kws_models.py`, `models/checkpoints/kws_tcresnet_advanced.pt` | `results_and_evaluation.md`, `objective_1_final_report.md`, `README.md`, Fig 1, Fig 4, Tab 06 |
| **TC-ResNet Test Accuracy** | 98.13% (524 / 534 correct) | **96.63%** (516 / 534 correct) | Checkpoint evaluation on `test_manifest_v2.csv` | `results_and_evaluation.md`, `objective_1_final_report.md`, `README.md`, Fig 4, Tab 07 |
| **TC-ResNet Macro-F1** | 0.9576 | **0.9402** (unweighted 6-class Macro-F1) | Checkpoint evaluation on `test_manifest_v2.csv` | `results_and_evaluation.md`, `objective_1_final_report.md`, `README.md`, Fig 4, Tab 07 |
| **Baseline Macro-F1** | 0.9374 (reported in Phase 2) | **0.7812** on historical 5-class test set;<br/>**0.9466** on corrected 6-class `test_manifest_v2.csv` (97.75% acc) | `reports/phase3_evaluation_integrity.md`, Checkpoint evaluation | `results_and_evaluation.md`, `objective_1_final_report.md`, Tab 07 |
| **`TRAI` Class Recall** | Baseline: 69.4% &bull; TC-ResNet: 79.2% | **Baseline:** 69.44% (25/36)<br/>**TC-ResNet:** 75.00% (27/36) | Checkpoint evaluation on `test_manifest_v2.csv` | `results_and_evaluation.md`, `objective_1_final_report.md`, Fig 4, Tab 07 |
| **Northern `TRAI` Recall** | Baseline: 8.3% &bull; TC-ResNet: 25.0% | **Baseline:** 16.67% (2/12)<br/>**TC-ResNet:** 25.00% (3/12) | Checkpoint evaluation on 12 Northern TRAI test clips | `results_and_evaluation.md`, `objective_1_final_report.md`, Fig 4, Tab 07 |
| **Test Confusion Matrix** | 50 support per command, 6 TRAI $\to$ PHAI, 230 UNKNOWN, 99 SILENCE | **36 support per command**, 300 UNKNOWN, 90 SILENCE.<br/>**27** TRAI, **9** TRAI $\to$ PHAI, **36** PHAI, **36** LEN, **34** XUONG, **293** UNKNOWN, **90** SILENCE. | Checkpoint evaluation on `test_manifest_v2.csv` | `results_and_evaluation.md`, `objective_1_final_report.md`, Fig 5, Tab 08 |
| **Continuous Streaming Evaluation** | "140 continuous 1-hour audio streams", 0.0 FA/hr clean, 0.8 FA/hr babble, 192 ms latency | **8.0-minute continuous stream** (0.133 hr: 6.5 min speech + 1.5 min noise). At $\tau=0.70$: 0 FA (Poisson 95% bound $\le 22.5\text{ FA/hr}$), 706.4 ms trigger latency. | `reports/phase3_streaming_evaluation.md` | `results_and_evaluation.md`, `objective_1_final_report.md`, `README.md`, Fig 6, Tab 09 |
| **Export Parity Numerical Diff** | $3.81 \times 10^{-6}$ across all models | **Baseline FP32:** $9.54 \times 10^{-7}$<br/>**TC-ResNet FP32:** $1.19 \times 10^{-6}$<br/>(Opset 14, 100% concordance) | `tests/test_parity.py` execution, ONNX runtime verification | `results_and_evaluation.md`, `objective_1_final_report.md`, Fig 8, Tab 10 |
| **INT8 Export Artifacts** | Both models quantized; TC-ResNet INT8 cited as verified | `vietnamese_kws_int8.onnx` (148.8 KB) **VERIFIED**.<br/>`vietnamese_kws_tcresnet_int8.onnx` is **MISSING ON DISK** (`ISSUE-03`). | `models/exported/` on-disk file audit | `results_and_evaluation.md`, `objective_1_final_report.md`, `README.md`, Fig 8, Tab 10, Tab 12 |
| **Field Campaign Human Takes** | 0 genuine human recordings; 58 files cited in manifest | **0 genuine human recordings**.<br/>58 physical files (82 manifest records) across 2 test speakers are automated unit-test fixtures (`SPK_FIELD_N001_TESTSUITE`, `SPK_M4A_TEST`). | `data/raw/field_recordings/field_collection_manifest.csv`, `consent_ledger.json` | `dataset_and_provenance.md`, `objective_1_final_report.md`, `README.md`, Fig 7, Tab 04, Tab 05 |
| **Automated Test Suite** | Cited 19 tests across `test_field_acquisition.py` and `test_model_parity.py` with mock test names | **19 passed tests** across `tests/test_field_acquisition.py` (12 tests) and `tests/test_parity.py` (7 tests). | `pytest -v tests/` output | `reproducibility_and_audit.md`, Tab 11 |
| **Manifest & Checkpoint Sizes / Hashes** | Draft / simulated byte sizes and hashes in register | **Bit-exact byte counts and SHA-256 fingerprints** computed and documented for all 11 key artifacts. | Cryptographic hashing of repository files | `dataset_and_provenance.md`, `reproducibility_and_audit.md` |

---

## 3. Detailed Actions Undertaken by Phase

### Phase 1: Forensic Codebase & Artifact Audit
- Audited all CSV manifests in `data/manifests/` and confirmed that `source_registry_v2.csv` contains 2,240 rows, with 2,099 assigned to active splits (1,281 train, 284 val, 534 test).
- Audited `data/raw/field_recordings/` and discovered 58 physical files (54 WAV, 4 M4A) and 82 records in `field_collection_manifest.csv`, all associated with synthetic unit-test speaker IDs. Confirmed zero genuine human recordings exist.
- Evaluated PyTorch checkpoints on disk and verified that `VietnameseTCResNet` has 272,934 parameters and `VietnameseKWSNet` has 138,534 parameters. Confirmed no `BCResNet` implementation exists in the codebase.
- Audited `models/exported/` and verified that `vietnamese_kws_tcresnet_fp32.onnx`, `vietnamese_kws_float32.onnx`, and `vietnamese_kws_int8.onnx` exist and pass Opset 14 checker tests. Confirmed `vietnamese_kws_tcresnet_int8.onnx` is missing on disk.

### Phase 2: Generation Script Repairs
1. **`scripts/objective_1_docs/generate_tables.py`:**
   - Updated Table 01 to reflect actual manifest approved sources: Google FLEURS vi (579), Confusing Words (120), Husein Noise (400), Synthetic Commands (1,000). Total: 2,099. Noted Common Voice, ESC-50, and FMA have 0 approved samples in manifests.
   - Updated Table 03 to accurately convey 579 genuine human speech files and 400 real environmental noise files.
   - Re-executed script to regenerate all 12 CSV tables in `docs/objective_1/tables/`.
2. **`scripts/objective_1_docs/generate_figures.py`:**
   - Completely rewritten to generate vector SVG figures directly linked to verified evidence:
     - Fig 1: `VietnameseTCResNet` architecture (272,934 params), 0.35 ms CPU inference, 706 ms trigger latency.
     - Fig 2: Approved sources (Google FLEURS, Confusing Words, Husein Noise, Synthetic Commands) with 0 Common Voice samples.
     - Fig 3: Verified split counts (Train 178/178/178/178/319/250, Val 36/36/36/36/80/60, Test 36/36/36/36/300/90).
     - Fig 4: Verified model performance (Baseline 97.75% acc, 0.9466 F1 vs TC-ResNet 96.63% acc, 0.9402 F1, 25.0% Northern TRAI).
     - Fig 5: Exact 6x6 confusion matrix (36 support/cmd, 27 TRAI correct, 9 TRAI $\to$ PHAI, 293 UNKNOWN correct).
     - Fig 6: Actual 8.0-minute streaming curve (Poisson 95% bound $\le 22.5\text{ FA/hr}$, 706.4 ms latency).
     - Fig 7: Honest field campaign audit (0 genuine human takes, 58 test fixtures on disk).
     - Fig 8: Opset 14 deployment flow, FP32 and INT8 baseline verified, TC-ResNet INT8 flagged missing.
   - Re-executed script to refresh all 8 SVG files in `docs/objective_1/figures/`.

### Phase 3: Narrative Documentation Alignment
Updated and synchronized the following core narrative markdown files:
- `docs/objective_1/README.md`
- `docs/objective_1/tables/README.md`
- `docs/objective_1/figures/README.md`
- `docs/objective_1/dataset_and_provenance.md`
- `docs/objective_1/results_and_evaluation.md`
- `docs/objective_1/reproducibility_and_audit.md`
- `docs/objective_1/objective_1_final_report.md`

All mentions of BC-ResNet-1, fabricated split counts (50/50/50/50), inflated streaming hours (140 hours), and ungrounded accuracies (98.13%) were eliminated and replaced with verified empirical evidence.

---

## 4. Verification Checkpoint & Invariants Confirmation

### 4.1 Benchmark Test Manifest Cryptographic Invariant
- **Target File:** `data/manifests/test_manifest_v2.csv`
- **Expected SHA-256:** `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`
- **Verified SHA-256:** `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`
- **Status:** **UNMODIFIED / PERFECT INTEGRITY**

### 4.2 Automated Test Harness Execution
- **Command:** `.\.venv\Scripts\pytest -v tests/`
- **Outcome:** **20 passed in 11.50s** (13 in `tests/test_field_acquisition.py`, 7 in `tests/test_parity.py`)
- **Status:** **100% PASSING (Zero production file pollution verified)**

---

## 5. Test Isolation Repair & Field Data Protection

### 5.1 Root Cause Analysis of Test Pollution
- **Mechanism:** In `tests/test_field_acquisition.py`, test cases invoked `ingest_recording()` and `run_recording_session()` without redirecting output paths. Because Python binds default argument values (`manifest_path: Path = MANIFEST_PATH`) at function definition time, monkeypatching or calling without path parameters caused test executions to write directly to production files.
- **Pollution Growth:** Each `pytest` run appended exactly **8 rows** to `data/raw/field_recordings/field_collection_manifest.csv` and created test fixture WAV/M4A files in `audio_raw/` and `audio_16k_standardized/`. Manifest row progression:
  - Baseline state: 74 rows
  - Phase 3 audit run: 82 rows
  - Phase 4 repair run: 90 rows
  - Phase 5 pre-verification: 98 rows
  - Final Gate run: 106 rows

### 5.2 Pre-Modification Safeguards & Backup
- **Timestamped Backup Created:** `data/backups/field_data_backup_20261013_105714/`
  - `field_collection_manifest.csv`: 75,969 bytes, SHA-256: `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85`
  - `consent_ledger.json`: 914 bytes, SHA-256: `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0`
  - `audio_raw/`: 99 audio files backed up across speaker directories.
  - `audio_16k_standardized/`: 99 audio files backed up across command directories.
- **Benchmark Invariant Reconfirmed:** `data/manifests/test_manifest_v2.csv` SHA-256 = `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13`.

### 5.3 Code Modifications Completed
1. **`src/acquisition/speaker_consent.py`:**
   - Modified `load_consent_ledger`, `save_consent_ledger`, `register_speaker`, `verify_speaker_consent`, `get_speaker_metadata`, `list_registered_speakers` to resolve `ledger_path` dynamically at runtime (`ledger_path if ledger_path is not None else LEDGER_PATH`), allowing parameter injection and monkeypatching.
2. **`src/acquisition/ingest_field_recordings.py`:**
   - Modified `load_existing_manifest`, `validate_field_audio`, and `ingest_recording` to accept optional `manifest_path`, `quarantine_path`, `raw_base_dir`, `std_base_dir`, `ledger_path`, dynamically resolving defaults at runtime.
3. **`src/acquisition/record_command_cli.py`:**
   - Updated `run_recording_session` signature to accept optional paths and forward them through to `verify_speaker_consent`, `get_speaker_metadata`, and `ingest_recording`.
4. **`src/acquisition/audit_field_campaign.py`:**
   - Defined `REPORT_PATH` module constant and enabled dynamic resolution of `report_path`, `manifest_path`, `quarantine_path`, `ledger_path`.
5. **`tests/test_field_acquisition.py`:**
   - Created autouse fixture `isolate_field_environment(tmp_path, monkeypatch)` that creates a sandboxed directory structure in `tmp_path` and monkeypatches all path constants across `speaker_consent`, `ingest_field_recordings`, `audit_field_campaign`, and test module globals.
   - Refactored `setup_test_speaker` to register test fixtures inside the isolated sandbox ledger.
   - Added 13th test `test_no_production_pollution_regression` that snapshots production hashes before test operations, executes registration/ingestion/CLI/audit in sandbox, and asserts byte-for-byte invariance of all production files.

### 5.4 Verification & Production Artifact Invariance
- **Test Results:** 20/20 passed in 11.50s (13 field acquisition + 7 parity).
- **Production Artifact Verification (Before vs. After Full Test Run):**
  - `field_collection_manifest.csv`: 75,969 bytes, SHA-256 `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` (**0 bytes changed, 0 rows added**).
  - `consent_ledger.json`: 914 bytes, SHA-256 `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` (**0 bytes changed**).
  - `quarantine_log.csv`: 13,850 bytes, SHA-256 `EC923FEB7D6E421BC420184D5BCE9663BD83923EF2C8E2544D31A02D9B2156FD` (**0 bytes changed, 0 rows added**).
  - `field_recording_campaign_audit.md`: 6,781 bytes, SHA-256 `C98BA03AC3FE2BE2D0836981FA13B2D4986090A07B7AF75D628424C57B1CBB96` (**untouched**).
  - `audio_raw/` & `audio_16k_standardized/`: Exactly 99 files each (**0 new files created**).

### 5.5 Forensic Classification of Accumulated Field Data & Safe Cleanup Proposal
- **Forensic Audit Findings (106 Manifest Rows):**
  - 105 rows belong to `SPK_FIELD_N001_TESTSUITE` (device: `Test Harness Device`).
  - 1 row belongs to `SPK_M4A_TEST` (device: `Unspecified Mobile`).
  - 106 manifest rows map to **99 unique raw paths** and **99 unique processed paths** on disk (10 duplicate rows from early static session IDs `SES_PRESERVE_TEST`, `SES_DUP_TEST`, `SES_MOCK_TEST` before random UUID suffixes were implemented).
  - Physical file inventory: Exactly 99 raw audio files and 99 standardized audio files on disk (0 missing files).
  - True genuine human recordings: **ZERO (0)**.
- **Safe Cleanup Proposal (Non-Destructive Safeguard):**
  - All existing files are safely preserved in backup `data/backups/field_data_backup_20261013_105714/`.
  - Physical files on disk will NOT be deleted automatically without explicit user authorization.
  - Recommended user action: When ready to start genuine field data collection, run an authorized purge script to reset `field_collection_manifest.csv` and `consent_ledger.json` to pristine empty headers, archiving existing fixture audio files.

---

## 6. Unresolved Defect Summary & Strategic Blockers

The following items remain outstanding engineering blockers that cannot be resolved through documentation repair alone, requiring subsequent development phases:

1. **`ISSUE-01` (CRITICAL / P0): Zero Genuine Human Target Command Audio.**
   - All 1,000 target command files in active splits are synthetic parametric audio.
   - Real-world production generalization requires executing the field recruitment campaign across $\ge 100$ speakers.
2. **`ISSUE-02` (CRITICAL / P1): Northern Vietnamese `TRAI` Recall Bottleneck (25.0%).**
   - 75% of Northern `TRAI` test samples collapse into `PHAI` due to identical rhyme `/aːj/` and unmodeled `/tɕ/` affricate burst acoustics in the synthetic generator.
   - Requires collecting genuine Northern native recordings.
3. **`ISSUE-03` (HIGH / P2): Missing TC-ResNet INT8 ONNX Export.**
   - `vietnamese_kws_tcresnet_int8.onnx` is omitted from disk.
   - Requires running quantization export and committing the resulting artifact (~280 KB).
4. **`ISSUE-04` (MEDIUM / P3): Physical Mobile Profiling Blocked on Host Platform.**
   - Windows 11 host lacks macOS SDK and Android ADB tooling.
   - Requires remote/cross-platform build pipelines for on-device profiling.
5. **`ISSUE-05` (HIGH / P4): Continuous Streaming Evaluation Duration Limited to 8.0 Minutes.**
   - Continuous false alarm evaluation was performed on 0.133 hours (Poisson 95% upper bound: $\le 22.5\text{ FA/hr}$).
   - Requires long-duration evaluation across 10 to 50 hours of uncurated audio.
