# Implementation Gap Report: Existing Codebase vs. Human Campaign Requirements

> **Document Version:** 1.0.0-PROPOSAL  
> **Campaign Date:** October 13, 2026  
> **Audit Focus:** `src/acquisition/` and `src/preprocessing/` modules vs. Production Recruitment Needs

---

## 1. Executive Summary

This report assesses the current acquisition codebase against the operational requirements of a multi-speaker, multi-region Vietnamese human data collection campaign.

The codebase successfully implements core primitives (basic MCI audio recording, 16 kHz standardization, quality gating for clipping/silence, and test sandbox isolation). However, **six technical and operational gaps** must be addressed before genuine human participant recruitment can safely proceed.

---

## 2. Detailed Technical Gap Inventory

### GAP-01 (CRITICAL / P0): Lack of `UNKNOWN` and `SILENCE` Class Ingestion Support
- **Current State:**  
  In [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L48-L53):
  ```python
  VALID_COMMANDS = {
      "TRAI": {"class_id": 0, "word": "trái"},
      "PHAI": {"class_id": 1, "word": "phải"},
      "LEN": {"class_id": 2, "word": "lên"},
      "XUONG": {"class_id": 3, "word": "xuống"}
  }
  ```
  Any attempt to ingest phonetically confusing words (`UNKNOWN`, class 4) or ambient noise (`SILENCE`, class 5) triggers line 117:
  ```python
  if label_upper not in VALID_COMMANDS:
      return False, f"invalid_command_label_{command_label}", {}
  ```
  The ingestion pipeline immediately quarantines all negative classes.
- **Impact:** The system cannot ingest human negative speech or background noise recordings.
- **Required Remediation:**  
  Expand `VALID_COMMANDS` to all 6 classes (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`) and support sub-category tracking (e.g., `confusing_near_rhyme`, `background_room_tone`).

---

### GAP-02 (HIGH / P1): Default Paths Point to Historical Test Fixture Root
- **Current State:**  
  Module constants in `src/acquisition/ingest_field_recordings.py` and `speaker_consent.py` default to:
  ```python
  FIELD_BASE_DIR = Path("data/raw/field_recordings")
  MANIFEST_PATH = FIELD_BASE_DIR / "field_collection_manifest.csv"
  LEDGER_PATH = Path("data/raw/field_recordings/consent_ledger.json")
  ```
  `data/raw/field_recordings/` already contains 106 historical test fixtures and 99 synthetic files.
- **Impact:** If an operator launches `record_command_cli.py` or `ingest_recording()` without explicit path overrides, genuine human recordings will be appended to the test fixture manifest, contaminating provenance.
- **Required Remediation:**  
  Configure production collection scripts to default to `data/raw/human_recordings_v1/`, preserving `data/raw/field_recordings/` as an untouched historical fixture archive.

---

### GAP-03 (HIGH / P1): Absence of Participant Consent Withdrawal Mechanism
- **Current State:**  
  `speaker_consent.py` implements `register_speaker()` and `verify_speaker_consent()`. However, there is no function or CLI tool to handle participant revocation of consent (`revoke_speaker_consent()`).
- **Impact:** If a participant requests withdrawal under ethical guidelines prior to dataset freeze, operators would need to manually edit JSON files, risking ledger corruption.
- **Required Remediation:**  
  Implement `src/acquisition/revoke_speaker_consent.py` to securely revoke consent, archive audio to a withdrawn partition, and update manifest records to `REVOKED_WITHDRAWN`.

---

### GAP-04 (HIGH / P2): Windows-Only Live Microphone Recording CLI
- **Current State:**  
  [`src/acquisition/record_command_cli.py`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L42) relies on Windows Multimedia API (`ctypes.windll.winmm`).
- **Impact:** The CLI cannot capture live audio on macOS, Linux, or mobile devices (Android/iOS). While `--ingest-file` supports external `.m4a` files, field collectors carrying mobile devices lack a streamlined batch transfer workflow.
- **Required Remediation:**  
  Formalize the mobile file-transfer ingestion protocol: contributors record sessions using native Voice Memos / Sound Recorder, export `.m4a` batches with structured filenames, and collectors ingest via `batch_ingest_directory()`.

---

### GAP-05 (MEDIUM / P2): Absence of Automated Stratified Split Generator
- **Current State:**  
  The repository does not contain an automated tool to partition a field collection manifest into 70% train, 15% val, and 15% test while enforcing strict speaker-disjointness and balancing dialect/gender strata.
- **Impact:** Manual splitting risks accidental speaker leakage or session splitting across partitions.
- **Required Remediation:**  
  Provide an automated partitioning script `src/acquisition/campaign_validator.py` with deterministic salted hashing.

---

### GAP-06 (MEDIUM / P2): Lack of Pre-Release Dataset Acceptance Gate
- **Current State:**  
  Testing is currently split between unit tests (`test_field_acquisition.py`) and model parity tests (`test_parity.py`). There is no comprehensive gate validator that checks an entire collection directory for consent completeness, file existence, SHA-256 consistency, format adherence, and zero synthetic contamination before freezing.
- **Impact:** Datasets could be released with subtle metadata omissions or formatting mismatches.
- **Required Remediation:**  
  Implement an automated dataset acceptance validator with pass/fail gate verdicts.

---

## 3. Prioritized Implementation Roadmap

| Priority | Gap ID | Focus Area | Effort | Prerequisite For |
| :---: | :--- | :--- | :---: | :--- |
| **P0** | `GAP-01` | Expand `ingest_recording` to support `UNKNOWN` & `SILENCE` | 0.5 day | Recording negative classes |
| **P1** | `GAP-02` | Configure dedicated `data/raw/human_recordings_v1/` destination | 0.5 day | First human recording take |
| **P1** | `GAP-03` | Implement consent withdrawal & revocation tooling | 0.5 day | Participant enrollment |
| **P2** | `GAP-04` | Document and test mobile M4A batch ingestion protocol | 1.0 day | iOS / Android mobile field capture |
| **P2** | `GAP-05` | Implement deterministic speaker-disjoint split generator | 0.5 day | Training partition creation |
| **P2** | `GAP-06` | Build automated dataset acceptance validator | 0.5 day | Dataset release freeze |
