# Final Targeted Verification Report: Strict Mode, Governance Gates, and Test Evidence

**Workspace:** `D:\trainidentificationaudio`  
**Document Version:** 1.0.0-TARGETED-VERIFICATION  
**Verification Date:** 2026-10-14  
**Auditor / Verification Role:** Independent Code & Governance Auditor (Pair Programming Assistant)  
**Deliverable Path:** [`docs/sound_command_recognition/campaign_20261013/final_targeted_verification.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/final_targeted_verification.md)  
**Governing Principle:** Read-Only Forensic Evidence Verification  

---

## 1. Definitive Verification Verdict & Operational Status

### Official Audit Verdict:
$$\mathbf{VERIFIED\ WITH\ OPEN\ TECHNICAL\ FINDINGS}$$

### Physical Human Data Collection Status:
$$\mathbf{Human\ Data\ Collection:\ STRICTLY\ BLOCKED}$$

*(Prohibited from participant recruitment, enrollment, or audio capture until all open technical findings and pre-recruitment governance gates are formally resolved and signed by authorized human leadership).*

---

## 2. Evidence Verification Table

The following table contrasts stated governance claims against primary technical and file-system evidence:

| Item / Claim | Primary File & Line Reference | Observed Forensic Reality | Confidence Level | Audit Finding & Classification |
| :--- | :--- | :--- | :---: | :--- |
| **CLI Strict-Mode Enforcement** | [`src/acquisition/record_command_cli.py#L80`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L80), [`#L381`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py#L381) | Live sessions (`mock_mode=False`) and `--ingest-file` compute `effective_strict = True`. | **HIGH** | **Verified Safeguard (CLI Layer).** CLI callers cannot disable strict mode for live recordings. |
| **Core Function Ingestion Bypass** | [`src/acquisition/ingest_field_recordings.py#L279`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L279), [`#L314-L328`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L314-L328) | Core function `ingest_recording()` defaults to `strict_metadata=False`. Provenance `"human_field_recording"` does not force `strict_metadata=True` in function body. | **HIGH** | **OPEN TECHNICAL FINDING.** Direct library callers can bypass strict validation and inject fallback defaults. |
| **Automated Test Suite Execution** | [`tests/test_campaign_acceptance.py`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py) | `pytest tests/test_campaign_acceptance.py -q` exits code 0; 24 passed in 1.88s. | **HIGH** | **Verified (Sandbox Automated Tests).** Acceptance suite passes cleanly under test fixtures. |
| **Field Acquisition Test Suite** | [`tests/test_field_acquisition.py`](file:///d:/trainidentificationaudio/tests/test_field_acquisition.py) | Suite previously timed out / hung at test 5 due to librosa import and audio compilation. | **HIGH** | **INCONCLUSIVE / PREVIOUSLY HUNG.** Not rerun during this verification to prevent unmonitored hangs. |
| **Physical Microphone Validation** | [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L98`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L98) | Hardware serial roster marked `UNKNOWN — ACTION REQUIRED`. Windows MCI recording untested on real mobile devices. | **HIGH** | **UNVERIFIED (HARDWARE REALITY).** Passing sandbox unit tests do not equal real-world microphone validation. |
| **Protected Artifact Immutability** | `data/manifests/test_manifest_v2.csv`, `field_collection_manifest.csv`, `consent_ledger.json` | Exact SHA-256 digests verified bit-for-bit before and after verification. | **HIGH** | **VERIFIED.** Cryptographic baseline 100% intact. |
| **FLEURS & MSWC Local Licenses** | `data/raw/fleurs_vi/`, `data/raw/mswc_vi/` | Audio and split metadata present; `LICENSE.txt` files missing on disk. | **HIGH** | **VERIFIED GAT-04 FINDING.** External corpora remain `NOT VERIFIED` on disk. |
| **Husein `mic-noise` Quarantine** | `data/raw/noise/mic-noise.zip` (45.9 MB), [`download_approved_sources.py#L38-L52`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L38-L52) | Pipeline throws runtime errors on ingestion; 45.9 MB unverified archive preserved on disk. | **HIGH** | **VERIFIED ACT-05A & ACT-05B.** Code block active; institutional storage disposition open. |
| **Cohort Sizing Quota (120 Speakers)** | [`collection_protocol.md#L105-L114`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L105-L114), [`human_governance_signoff_record.md#L18`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_signoff_record.md#L18) | Quota is an engineering proposal; sign-off record Section 4 is unexecuted. `consent_ledger_human.json` absent. | **HIGH** | **VERIFIED.** Purely hypothetical proposal; zero participants recruited. |
| **Pseudonymity Implementation** | [`src/acquisition/speaker_consent.py#L89-L100`](file:///d:/trainidentificationaudio/src/acquisition/speaker_consent.py#L89-L100) | Uses `secrets.token_hex(3)` to generate 6-hex-nonce IDs (`SPK_H_N_8f2a1b`). | **HIGH** | **VERIFIED.** Implements pseudorandom nonces; does not implement asymmetric cryptography. |

---

## 3. Test Evidence & Summary

### 3.1 Acceptance Test Suite Execution
- **Command:** `pytest tests/test_campaign_acceptance.py -q`
- **Execution Mode:** Bounded synchronous execution
- **Exit Code:** `0`
- **Output:**
  ```text
  ........................                                                 [100%]
  24 passed in 1.88s
  ```
- **Result:** **24 passed, 0 failed, 0 errors, 0 warnings**.

### 3.2 Field Acquisition Test Suite Status
- **Suite:** `tests/test_field_acquisition.py`
- **Audit Determination:** **`INCONCLUSIVE / PREVIOUSLY HUNG`**.
- **Rationale:** In previous verification attempts, this test suite hung at test index 5 (`test_raw_preservation_and_standardization`) due to `librosa.resample` import and compilation overhead on Python 3.14 / Windows NT. In compliance with safety constraint #4, this suite was **NOT rerun** without pre-bounded test isolation.

### 3.3 Sandbox Testing vs. Physical Microphone Validation
> [!IMPORTANT]
> **Prohibition Against Conflating Test Fixtures with Hardware Validation:**  
> All 24 passing acceptance tests execute in temporary sandbox environments (`tmp_path`) using synthetically mocked WAV files. They verify that Python algorithmic branches execute properly. They do **NOT** prove:
> 1. That physical smartphone microphones (iPhone 11–15, Samsung Galaxy S21–S24) record clean audio via Windows MCI or mobile drivers.
> 2. That automatic gain control (AGC) or hardware noise cancellation does not distort Vietnamese tones.
> 3. That field operators can execute recordings in reverberant rooms without clipping.

---

## 4. Strict-Mode Call-Graph & Core Function Bypass Finding

### 4.1 End-to-End Call Graph

```mermaid
flowchart TD
    subgraph CLI_Layer ["1. CLI Entry Points (Safeguards Enforced)"]
        CLI_Live["record_command_cli.py (Live Session)"] -->|Forces effective_strict=True| IngestFunc
        CLI_IngestFile["record_command_cli.py (--ingest-file)"] -->|Forces effective_strict=True| IngestFunc
        CLI_Batch["ingest_field_recordings.py (CLI main)"] -->|Forces effective_strict=True| IngestFunc
    end
    
    subgraph Python_Direct_Layer ["2. Python Library Invocation (BYPASS VULNERABILITY)"]
        DirectScript["Direct Python Call: ingest_recording()"] -->|Passes strict_metadata=False OR Omits Flag| IngestFunc
    end
    
    subgraph Core_Function ["3. Core Function: ingest_recording()"]
        IngestFunc["ingest_recording(input_path, speaker_id, ...)"]
        IngestFunc --> CheckFlag{"Is strict_metadata == True?"}
        
        CheckFlag -->|YES| StrictValidator["Strict Validation Active: Rejects missing mic, distance, age; rejects contradictory metadata"]
        CheckFlag -->|NO (Default)| FallbackInjection["Fallback Injection Active: age='18-30', mic='builtin_mobile', dist='close_talk_15cm'"]
        
        FallbackInjection --> ManifestWrite["Appends to Manifest with Fabricated Metadata: STATUS APPROVED"]
        StrictValidator -->|Pass| ManifestWrite
        StrictValidator -->|Fail| Quarantine["Quarantine Log: REJECTED"]
    end
```

### 4.2 Detailed Finding: Core Function Ingestion Bypass

- **Finding ID:** `TECH-FINDING-01`
- **Severity:** **`HIGH`** (Technical Pipeline Integrity)
- **Exact File & Line References:**
  1. [`src/acquisition/ingest_field_recordings.py#L273`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L273):
     ```python
     provenance: str = "human_field_recording",
     ```
  2. [`src/acquisition/ingest_field_recordings.py#L279`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L279):
     ```python
     strict_metadata: bool = False,
     ```
  3. [`src/acquisition/ingest_field_recordings.py#L314-L317`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L314-L317):
     ```python
     age = age_bracket or spk_meta.get("age_bracket") or ("18-30" if not strict_metadata else None)
     mic_type = microphone_type or spk_meta.get("primary_microphone") or ("builtin_mobile" if not strict_metadata else None)
     dist_from_spk = spk_meta.get("distance_category") or spk_meta.get("speaking_distance")
     dist_cat = distance_category or dist_from_spk or ("close_talk_15cm" if not strict_metadata else None)
     ```
  4. [`src/acquisition/ingest_field_recordings.py#L510`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L510):
     ```python
     def batch_ingest_directory(..., strict_metadata: bool = False)
     ```

- **Vulnerability Description:**
  The CLI entry points (`record_command_cli.py` and `ingest_field_recordings.py`) correctly enforce `effective_strict = True` whenever human recordings are processed. However, the underlying core library functions `ingest_recording()` and `batch_ingest_directory()` declare `strict_metadata: bool = False` as their default parameter. Crucially, the function body **does not inspect `provenance` to automatically promote `strict_metadata` to `True`**.

- **Consequence:**
  If a developer or ingestion automation script calls `ingest_recording()` directly in Python and omits `strict_metadata=True` (or explicitly passes `strict_metadata=False`), the function will:
  1. Silently inject default fallbacks (`"18-30"`, `"builtin_mobile"`, `"close_talk_15cm"`).
  2. Completely skip strict demographic and acoustic error validation (lines 328–348).
  3. Ingest the take as `APPROVED` into the manifest with fabricated metadata.

- **Recommended Remediation (For Future Implementation Phase):**
  Add a core-level guard at the very beginning of `ingest_recording()`:
  ```python
  if provenance == "human_field_recording" and not strict_metadata:
      # Option A: Fail-closed exception
      # raise ValueError("Human field recordings must be ingested with strict_metadata=True")
      # Option B: Automatic promotion
      strict_metadata = True
  ```

---

## 5. Gate Inconsistencies & Recommended Corrections

The cross-document review of the six primary governance documents identified the following discrepancies:

### 5.1 Checklist Item 12 Line Reference Discrepancy
- **Location:** [`docs/sound_command_recognition/campaign_20261013/human_governance_signoff_checklist.md#L182`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_signoff_checklist.md#L182)
- **Observed Text:** Cites `collection_protocol.md#L188-L215` as evidence for "Go/No-Go Gates & Stop Triggers".
- **Discrepancy:** Lines 188–215 of [`collection_protocol.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md) actually contain Section 8 (*Hardware & Acoustic Capture Protocol*) and Section 9 (*Budget Summary & Statistical Power*), rather than an explicit Go/No-Go gate table. The canonical gating framework is defined in Section 5 of `governance_approval_action_tracker.md`.
- **Recommended Correction:** Update Item 12 in `human_governance_signoff_checklist.md` to reference `governance_approval_action_tracker.md#L283-L330` for Go/No-Go gating, and add an explicit Campaign Stop Triggers sub-clause to `collection_protocol.md`.

### 5.2 Gate ID Naming Alignment in Decision Brief & Checklist
- **Location:** `human_governance_decision_brief.md` (Questions Q1–Q5) and `human_governance_signoff_checklist.md` (Items 1–12).
- **Discrepancy:** While `governance_approval_action_tracker.md` Section 5 and `human_governance_signoff_record.md` Section 3 use canonical gate IDs `GATE-01` through `GATE-08`, the Decision Brief uses `Q1`–`Q5` and the Checklist uses `Item 1`–`Item 12`.
- **Recommended Correction:** Embed explicit tags (e.g. `[GATE-01]`, `[GATE-02]`) in the headings of Checklist Items and Decision Brief Questions to make the 1-to-1 mapping unmistakable to reviewers.

### 5.3 Conditional Scope of GATE-04 (External Corpora)
- **Location:** `governance_approval_action_tracker.md` Table 5, `human_governance_signoff_checklist.md` Item 7.
- **Discrepancy:** Some descriptions could lead reviewers to believe that missing local `LICENSE.txt` files for Google FLEURS and MSWC block physical human participant recruitment.
- **Clarification:** Human field recording gathers authentic target commands, negative words (`UNKNOWN`), and room tones (`SILENCE`) exclusively from human contributors. FLEURS and MSWC are external negative speech corpora. Therefore, **`GATE-04` is conditional on multi-corpus model training and model release**, NOT a blocker for field recruitment.
- **Recommended Correction:** Explicitly annotate `GATE-04` as `Pre-Training / Multi-Corpus Blocker`, separating it from `Pre-Recruitment Blockers` (GATE-01, 02, 03, 05, 06, 07).

### 5.4 Distinction Between ACT-05A and ACT-05B
- **Verification:** The separation between `ACT-05A` (Technical software block on `husein_mic_noise`, status `CLOSED WITH EVIDENCE`) and `ACT-05B` (Institutional legal review of 45.9 MB on-disk archive storage, status `AWAITING HUMAN DECISION`) is properly maintained across all documents.
- **Finding:** Fully verified. Filesystem stability mandate (prohibiting deletion during this phase) is clearly stated.

### 5.5 Terminology Precision: Pseudonymization vs. Cryptographic Pseudonymity
- **Location:** `collection_protocol.md#L18`, `human_governance_decision_brief.md#L37`, `governance_approval_action_tracker.md#L178`.
- **Discrepancy:** The phrasing "cryptographic pseudonymity" could be misconstrued by legal counsel as implying digital signatures, public-key cryptography, or cryptographic hashes of identity.
- **Implementation Reality:** The system calls `secrets.token_hex(3)` in `speaker_consent.py#L96`, which generates a cryptographically random 6-character hex nonce (`SPK_H_N_8f2a1b`). No PII is collected or hashed.
- **Recommended Correction:** Formally adjust terminology to: **"Pseudonymization with random nonce IDs (generated via Python `secrets`); detached from direct personal identifiers (no names, telephone numbers, or citizen IDs collected); vocal recordings remain biometric personal data under applicable privacy laws."**

---

## 6. Quota and Split Arithmetic Verification

### 6.1 Mathematical Proof of Cohort-Level Arithmetic
- **Target Cohort:** 120 speakers.
- **Partition Ratio:** 70% Train / 15% Validation / 15% Test.
- **Exact Calculation:**
  $$120 \times 0.70 = 84.0 \quad (\text{Train speakers})$$
  $$120 \times 0.15 = 18.0 \quad (\text{Validation speakers})$$
  $$120 \times 0.15 = 18.0 \quad (\text{Test speakers})$$
  $$84 + 18 + 18 = 120 \quad (\text{Zero integer truncation drift})$$
- **Regional Quotas (50 Northern / 30 Central / 40 Southern):**
  - Northern (50): $50 \times 0.70 = 35$ Train, $50 \times 0.15 = 7.5 \rightarrow 7$ Val, $8$ Test ($35 + 7 + 8 = 50$).
  - Central (30): $30 \times 0.70 = 21$ Train, $30 \times 0.15 = 4.5 \rightarrow 4$ Val, $5$ Test ($21 + 4 + 5 = 30$).
  - Southern (40): $40 \times 0.70 = 28$ Train, $40 \times 0.15 = 6.0 \rightarrow 6$ Val, $6$ Test ($28 + 6 + 6 = 40$).
  - Sums: Train = $35+21+28=84$; Val = $7+4+6=17$; Test = $8+5+6=19$. Total = 120.

### 6.2 Hypothetical Arithmetic vs. Measured Reality

| Parameter | Theoretical / Hypothetical Model | Actual On-Disk Reality | Verification Status |
| :--- | :--- | :--- | :---: |
| **Total Enrolled Participants** | 120 participants | `0` participants (`consent_ledger_human.json` absent) | **Confirmed Zero** |
| **Regional Distribution** | 50 Northern, 30 Central, 40 Southern | `0` records across all regions | **Hypothetical Proposal Only** |
| **Demographic Strata Breakdown** | E.g. Central males aged 51+ ($n=2$) | `0` measured demographic strata | **Hypothetical Example Only** |
| **Human Recording Directory** | `data/raw/human_recordings_v1/` | **DOES NOT EXIST ON DISK** | **Confirmed Zero** |

**Confirmation:** No governance report claims or implies that actual human participants have been recruited or measured. The quota remains strictly **`PROPOSED — PENDING HUMAN APPROVAL`**.

---

## 7. Integrity, Repository State & Hashes

### 7.1 Exact `git status --short` Output

```text
 M reports/field_recording_campaign_audit.md
 M src/acquisition/audit_field_campaign.py
 M src/acquisition/download_approved_sources.py
 M src/acquisition/ingest_field_recordings.py
 M src/acquisition/record_command_cli.py
 M src/acquisition/speaker_consent.py
 M src/preprocessing/audio_normalizer.py
 M tests/test_field_acquisition.py
?? data/backups/
?? data/processed/
?? data/raw/
?? docs/objective_1/
?? docs/sound_command_recognition/
?? scripts/
?? src/acquisition/campaign_validator.py
?? src/acquisition/revoke_speaker_consent.py
?? tests/test_campaign_acceptance.py
```

### 7.2 Protected File Hashes (Before and After Verification)

Hashes independently computed via PowerShell `Get-FileHash`:

| Protected File Path | Baseline Required SHA-256 Digest | Measured Before Verification | Measured After Verification | Status |
| :--- | :--- | :--- | :--- | :---: |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT** |

### 7.3 Explicit Confirmation of Non-Modification
During this verification audit:
- Zero lines of source code or test files were modified.
- Zero git stage, commit, revert, or stash operations were performed.
- Zero mock or genuine consent records were created.
- Zero audio files were downloaded, modified, or deleted.

---

## 8. Remaining Blockers & Human Sign-Off Roadmap

Human data collection remains **`STRICTLY BLOCKED`** until the following discrete conditions are fulfilled:

1. **Resolution of `TECH-FINDING-01`:** Core ingestion function `ingest_recording()` must enforce `strict_metadata=True` internally whenever `provenance == "human_field_recording"`.
2. **Resolution of Blocker 1 (GATE-01):** Principal Investigator signs Section 4 of [`human_governance_signoff_record.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_signoff_record.md).
3. **Resolution of Blocker 2 (GATE-02):** Legal / Privacy Counsel signs Section 5 of [`human_governance_signoff_record.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_signoff_record.md) approving Vietnamese informed consent agreement.
4. **Resolution of Blocker 3 (GATE-03):** Executive resource allocation and quota of 120 participants approved in writing.
5. **Resolution of Blocker 5 (GATE-05):** Operational SOP governing intermediate cache purging and model retraining upon consent withdrawal adopted by DPO and ML Lead.
6. **Resolution of Blocker 6 (GATE-06):** Physical mobile device inventory roster completed and Field Operator Manual finalized.
7. **Resolution of Blocker 7 (ACT-05B):** Legal counsel memo deciding disposition of on-disk `husein_mic_noise` files.
8. **Runtime Invariant (GATE-08):** Per-participant enrollment in `consent_ledger_human.json` prior to any recording take being saved.

---

## 9. Final Conclusion

- **Audit Status:** Complete, evidence-based, and read-only.
- **Audit Verdict:** **`VERIFIED WITH OPEN TECHNICAL FINDINGS`**
- **Operational Reality:** **`Human data collection: STRICTLY BLOCKED`**
