# P0 Data Governance Remediation Report

> **Document Version:** 1.0.0-REMEDIATION  
> **Date:** October 13, 2026  
> **Workspace:** `D:\trainidentificationaudio`  
> **Source Audit:** [`docs/sound_command_recognition/campaign_20261013/pre_collection_readiness_audit.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/pre_collection_readiness_audit.md)  
> **Remediation Status:** **ALL THREE P0 BLOCKERS REMEDIATED & VERIFIED**  
> **Collection Approval Status:** **BLOCKED** *(Awaiting human protocol review and P1 schema migration)*

---

## 1. Executive Summary

This report documents the safe, auditable remediation of the three Priority 0 (P0) Data Governance Blockers identified during the independent readiness audit:
1. **P0-1 (Production Path Collision):** Rerouted default production acquisition directories from historical fixture paths (`data/raw/field_recordings/`) to a dedicated, segregated human root (`data/raw/human_recordings_v1/`).
2. **P0-2 (Source-License Gate):** Formally gated external sources; classified `husein_mic_noise` as `BLOCKED_UNVERIFIED_LICENSE` and blocked `NEGATIVE_NOISE_HUSEIN` from manifest validation, with authorized fallback to participant ambient room tones.
3. **P0-3 (Pseudonymous Speaker IDs):** Replaced sequential counters (`SPK_FIELD_N001`) with cryptographically random pseudonymous IDs (`SPK_H_<N|C|S>_<6-hex-nonce>`) using Python's `secrets` module, equipped with collision re-sampling.

All changes were implemented with 100% temporary sandbox isolation. Zero genuine human audio was recorded, no historical files were migrated or deleted, and all protected benchmarks remain bit-exact.

---

## 2. Protected Artifact Integrity Gate

The SHA-256 cryptographic hashes for the frozen benchmark manifest and historical fixture archives were calculated before and after all code changes and test runs:

```powershell
Get-FileHash -Algorithm SHA256 `
  "data/manifests/test_manifest_v2.csv", `
  "data/raw/field_recordings/field_collection_manifest.csv", `
  "data/raw/field_recordings/consent_ledger.json"
```

| Protected Asset Path | Expected Benchmark SHA-256 | Pre-Remediation SHA-256 | Post-Remediation SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |

Zero historical records or frozen test vectors were modified.

---

## 3. Detailed Remediation of P0 Blockers

### 3.1 P0-1: Safe Default Production Paths
- **Vulnerability Identified:**  
  Module constants in `src/acquisition/ingest_field_recordings.py` and `speaker_consent.py` previously defaulted to `data/raw/field_recordings/`. Launching ingestion without manual CLI overrides would append human records to the 106-row historical test fixture manifest.
- **Code Changes:**
  1. In [`src/acquisition/speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py#L38-L44):
     ```python
     HUMAN_RECORDINGS_ROOT = Path("data/raw/human_recordings_v1")
     LEGACY_FIXTURES_ROOT = Path("data/raw/field_recordings")

     # Default production path for genuine human consent ledger (P0-1 remediation)
     LEDGER_PATH = HUMAN_RECORDINGS_ROOT / "consent_ledger_human.json"
     # Historical fixture ledger path (preserved read-only for audit integrity)
     LEGACY_LEDGER_PATH = LEGACY_FIXTURES_ROOT / "consent_ledger.json"
     ```
  2. In [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L57-L72):
     ```python
     HUMAN_BASE_DIR = Path("data/raw/human_recordings_v1")
     LEGACY_FIELD_BASE_DIR = Path("data/raw/field_recordings")

     # Production default root for future genuine human recordings (P0-1 remediation)
     FIELD_BASE_DIR = HUMAN_BASE_DIR
     RAW_AUDIO_DIR = FIELD_BASE_DIR / "audio_raw"
     STANDARDIZED_AUDIO_DIR = FIELD_BASE_DIR / "audio_16k_standardized"
     QUARANTINE_DIR = FIELD_BASE_DIR / "quarantined"
     MANIFEST_PATH = FIELD_BASE_DIR / "human_collection_manifest_v1.csv"
     QUARANTINE_LOG_PATH = QUARANTINE_DIR / "quarantine_log.csv"
     LEDGER_PATH = FIELD_BASE_DIR / "consent_ledger_human.json"

     # Historical fixture archive paths (preserved read-only for audit integrity)
     LEGACY_MANIFEST_PATH = LEGACY_FIELD_BASE_DIR / "field_collection_manifest.csv"
     LEGACY_LEDGER_PATH = LEGACY_FIELD_BASE_DIR / "consent_ledger.json"
     LEGACY_QUARANTINE_LOG_PATH = LEGACY_FIELD_BASE_DIR / "quarantined" / "quarantine_log.csv"
     ```
- **Preservation of Overrides:** All functions continue to support explicit `manifest_path`, `ledger_path`, `raw_base_dir`, `std_base_dir`, and `quarantine_path` overrides, ensuring test fixtures run in sandboxes without touching production paths.

---

### 3.2 P0-2: Source-License Gate & External Noise Governance
- **Vulnerability Identified:**  
  `husein_mic_noise` was cited under CC-BY-4.0 without legal licensing verification or repository upstream grants.
- **Code & Policy Changes:**
  1. In [`src/acquisition/download_approved_sources.py`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L34-L71):
     Marked `mic_noise` as:
     ```python
     "license": "UNVERIFIED (Claimed CC-BY-4.0)",
     "license_verified": False,
     "status": "BLOCKED_UNVERIFIED_LICENSE",
     "license_notes": (
         "External repository lacks verified upstream legal license text and attribution grant. "
         "BLOCKED as an approved training source. "
         "Allowed fallback: Ambient room tone recorded during authorized collection sessions "
         "with verified participant consent and provenance (HUMAN_CONSENTED, sub_category: background_room_tone)."
     )
     ```
     Implemented `check_source_license_status(source_name)` to block any unverified source from download or acquisition pipelines.
  2. In [`src/acquisition/campaign_validator.py`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L34-L41):
     Added `BLOCKED_UNVERIFIED_PROVENANCE_VALUES`:
     ```python
     BLOCKED_UNVERIFIED_PROVENANCE_VALUES = {
         "NEGATIVE_NOISE_HUSEIN": (
             "External source 'husein_mic_noise' is BLOCKED: unverified licensing provenance. "
             "Allowed fallback: ambient room tone recorded during authorized collection sessions "
             "(HUMAN_CONSENTED, sub_category: background_room_tone)."
         )
     }
     ```
     `validate_manifest_row` automatically rejects any manifest record claiming `NEGATIVE_NOISE_HUSEIN` with error `unverified_source_license_blocked`.
  3. In [`docs/sound_command_recognition/campaign_20261013/data_provenance_policy.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/data_provenance_policy.md#L55-L60):
     Formally classified `NEGATIVE_NOISE_HUSEIN` as `BLOCKED / UNVERIFIED LICENSE` and documented the authorized fallback.

---

### 3.3 P0-3: Cryptographic Pseudonymous Speaker ID Generation
- **Vulnerability Identified:**  
  `generate_next_speaker_id` created sequential counter IDs (`SPK_FIELD_N001`), violating non-sequential pseudonymous requirements and leaking enrollment order.
- **Code Changes:**
  In [`src/acquisition/speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py#L57-L105):
  Implemented `generate_pseudonymous_speaker_id`:
  ```python
  SPEAKER_ID_REGEX = re.compile(r"^SPK_H_([NCS])_([a-f0-9]{6})$")

  def generate_pseudonymous_speaker_id(
      dialect: str,
      ledger: Optional[Dict[str, Any]] = None,
      max_attempts: int = 1000
  ) -> str:
      prefix_map = {"Northern": "N", "Central": "C", "Southern": "S"}
      prefix = prefix_map.get(dialect, "X")

      active_ledger = ledger if ledger is not None else {}
      for _ in range(max_attempts):
          nonce = secrets.token_hex(3).lower()
          candidate_id = f"SPK_H_{prefix}_{nonce}"
          if candidate_id not in active_ledger:
              return candidate_id

      raise RuntimeError(
          f"Speaker ID collision avoidance limit ({max_attempts}) reached. "
          "Active ledger may be congested."
      )
  ```
- **Properties & Invariants:**
  - **Entropy:** $16^6 = 16,777,216$ random combinations per regional prefix.
  - **No PII:** Contains zero personal names, dates of birth, phone numbers, or sequence numbers.
  - **Safe Collision Handling:** Re-samples if nonce already exists in active ledger; fails safely with `RuntimeError` if limit reached.
  - **Governance Clarification:** Pseudonymization separates voice assets from personal identities in storage. It does **not** claim biometric voice acoustics are anonymized.

---

## 4. Test Suite Execution & Verification

Three new acceptance tests were added to [`tests/test_campaign_acceptance.py`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py):
1. `test_p0_1_production_path_separation_and_defaults`: Verifies production paths point to `data/raw/human_recordings_v1/`, do not resolve into historical fixtures, and temporary path overrides function in isolated sandboxes without polluting production.
2. `test_p0_2_source_license_gate`: Verifies unverified sources (`mic_noise` / `NEGATIVE_NOISE_HUSEIN`) are blocked, verified sources (`fleurs_vi_dev`) pass, and manifest rows with unverified provenance are rejected.
3. `test_p0_3_pseudonymous_speaker_id_generation`: Verifies regex formatting (`^SPK_H_[NCS]_[a-f0-9]{6}$`), uniqueness across 1,000 samples (zero collisions), non-sequential integer distribution, collision re-sampling, and collision failure safety.

### Full Test Execution Log
Command:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\trainidentificationaudio
collected 34 items

tests/test_campaign_acceptance.py::test_valid_human_manifest_acceptance PASSED [  2%]
tests/test_campaign_acceptance.py::test_unconsented_speaker_rejection PASSED [  5%]
tests/test_campaign_acceptance.py::test_missing_required_metadata_rejection PASSED [  8%]
tests/test_campaign_acceptance.py::test_invalid_provenance_tag_rejection PASSED [ 11%]
tests/test_campaign_acceptance.py::test_duplicate_audio_hash_quarantine PASSED [ 14%]
tests/test_campaign_acceptance.py::test_duplicate_recording_id_quarantine PASSED [ 17%]
tests/test_campaign_acceptance.py::test_digital_clipping_gate PASSED     [ 20%]
tests/test_campaign_acceptance.py::test_abnormal_silence_gate PASSED     [ 23%]
tests/test_campaign_acceptance.py::test_speaker_disjoint_split_generator PASSED [ 26%]
tests/test_campaign_acceptance.py::test_speaker_leakage_detector PASSED  [ 29%]
tests/test_campaign_acceptance.py::test_campaign_acceptance_zero_production_pollution PASSED [ 32%]
tests/test_campaign_acceptance.py::test_p0_1_production_path_separation_and_defaults PASSED [ 35%]
tests/test_campaign_acceptance.py::test_p0_2_source_license_gate PASSED  [ 38%]
tests/test_campaign_acceptance.py::test_p0_3_pseudonymous_speaker_id_generation PASSED [ 41%]
tests/test_field_acquisition.py::test_consent_enforcement PASSED         [ 44%]
tests/test_field_acquisition.py::test_quality_gate_clipping_rejection PASSED [ 47%]
tests/test_field_acquisition.py::test_quality_gate_silence_rejection PASSED [ 50%]
tests/test_field_acquisition.py::test_quality_gate_duration_bounds PASSED [ 52%]
tests/test_field_acquisition.py::test_raw_preservation_and_standardization PASSED [ 55%]
tests/test_field_acquisition.py::test_duplicate_sha256_detection PASSED  [ 58%]
tests/test_field_acquisition.py::test_mock_cli_recording_session PASSED  [ 61%]
tests/test_field_acquisition.py::test_audit_field_campaign_generation PASSED [ 64%]
tests/test_field_acquisition.py::test_security_path_traversal_rejection PASSED [ 67%]
tests/test_field_acquisition.py::test_duplicate_recording_id_rejection PASSED [ 70%]
tests/test_field_acquisition.py::test_m4a_audio_ingestion_and_decoding PASSED [ 73%]
tests/test_field_acquisition.py::test_windows_hardware_recorder_configuration PASSED [ 76%]
tests/test_field_acquisition.py::test_no_production_pollution_regression PASSED [ 79%]
tests/test_parity.py::test_feature_extractor_shape_and_finite PASSED     [ 82%]
tests/test_parity.py::test_baseline_onnx_model_execution_and_shape PASSED [ 85%]
tests/test_parity.py::test_tcresnet_onnx_model_execution_and_parity PASSED [ 88%]
tests/test_parity.py::test_pytorch_onnx_numerical_parity_baseline PASSED [ 91%]
tests/test_parity.py::test_golden_vectors_reproducibility PASSED         [ 94%]
tests/test_parity.py::test_manifest_speaker_disjointness_v2 PASSED       [ 97%]
tests/test_parity.py::test_manifest_class_completeness_v2 PASSED         [100%]

============================= 34 passed in 8.27s ==============================
```

- **Exit Code:** `0`
- **Total Tests:** 34 passed, 0 failed, 0 warnings.
- **Git Diff Hygiene:** `git diff --check src/ tests/` exited with code 0 (no trailing whitespace or conflict markers).

---

## 5. Remaining Risks & Unresolved Licensing Evidence

1. **Unresolved Upstream Noise Licensing:**  
   `husein_mic_noise` remains blocked. The campaign will rely strictly on authorized ambient room tones recorded directly during participant sessions (`HUMAN_CONSENTED`, `sub_category: background_room_tone`).
2. **Pending P1 Enhancements (Deferred as Mandated):**
   - Ingestion code currently writes 23 columns; 28-column schema migration will occur during subsequent schema standardization.
   - Demographic strata grouping in `generate_disjoint_splits` remains uniform.
   - Dedicated standalone CLI for consent withdrawal (`revoke_speaker_consent.py`) remains to be authored.
3. **Participant Protection Invariant:**  
   Human collection remains strictly **BLOCKED** until stakeholders complete formal protocol review.

---

## 6. Exact Changed-File List

| File Path | Status | Summary of Changes |
| :--- | :---: | :--- |
| [`src/acquisition/speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py) | Modified | Changed default `LEDGER_PATH` to `data/raw/human_recordings_v1/consent_ledger_human.json`; implemented `generate_pseudonymous_speaker_id` using `secrets.token_hex(3)`. |
| [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py) | Modified | Changed default `FIELD_BASE_DIR` to `data/raw/human_recordings_v1/`; preserved legacy paths. |
| [`src/acquisition/download_approved_sources.py`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py) | Modified | Classified `mic_noise` as `BLOCKED_UNVERIFIED_LICENSE`; added `check_source_license_status()`. |
| [`src/acquisition/campaign_validator.py`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py) | Modified | Added `BLOCKED_UNVERIFIED_PROVENANCE_VALUES`; reject unverified provenance in `validate_manifest_row`. |
| [`docs/sound_command_recognition/campaign_20261013/data_provenance_policy.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/data_provenance_policy.md) | Modified | Updated Section 2.5: marked `NEGATIVE_NOISE_HUSEIN` as blocked and documented participant room tone fallback. |
| [`tests/test_campaign_acceptance.py`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py) | Modified | Added tests for P0-1, P0-2, and P0-3; total test suite expanded to 34 passing tests. |
| [`src/acquisition/audit_field_campaign.py`](file:///d:/trainidentificationaudio/src/acquisition/audit_field_campaign.py) | Modified | Cleaned trailing whitespace. |
| [`docs/sound_command_recognition/campaign_20261013/p0_remediation_report.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/p0_remediation_report.md) | Created | Generated this remediation report. |
