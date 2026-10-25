# P0 Security Remediation Report: Ingestion Core Strict Metadata Enforcement

**Workspace:** `D:\trainidentificationaudio`  
**Campaign:** Vietnamese Keyword Spotting Campaign 20261013  
**Audit Finding Addressed:** `TECH-FINDING-01` (Core Metadata Invariant Bypass via Ingestion API)  
**Remediation Date:** 2026-10-14  
**Author:** Quality & Governance Engineering  
**Security Status:** `RESOLVED & VERIFIED`  
**Operational Status:** `HUMAN DATA COLLECTION: STRICTLY BLOCKED`  

---

## 1. Executive Summary

This remediation report certifies the resolution of critical vulnerability `TECH-FINDING-01`, identified in governance and technical security audits. 

Prior to this fix, while the Command-Line Interface (`record_command_cli.py` and `ingest_field_recordings.py` CLI parser) applied strict metadata enforcement for human recording sessions, the core ingestion APIs (`ingest_recording()` and `batch_ingest_directory()`) accepted an optional parameter `strict_metadata: bool = False`. If callers imported and invoked these functions directly without setting `strict_metadata=True`, the core defaulted to permissive mode, silently injected fabricated fallback strings (`"18-30"`, `"builtin_mobile"`, `"close_talk_15cm"`), completely skipped strict metadata validation gates, and approved non-compliant human recordings into canonical dataset manifests.

### Key Remediation Results
1. **Core Invariant Enforced at Ingestion Root:**  
   `ingest_recording()` and `batch_ingest_directory()` now evaluate an unconditional provenance security invariant:
   $$\text{effective\_strict} = \text{True if } \text{provenance} \in (\text{"human\_field\_recording"}, \text{"HUMAN\_CONSENTED"}) \text{ else bool}(\text{strict\_metadata})$$
   Callers cannot bypass strict metadata validation for human recordings under any circumstances, even when explicitly passing `strict_metadata=False`.
2. **Elimination of Fabricated Defaults for Human Data:**  
   Permissive fallback values are strictly disabled whenever `effective_strict` is active. Human recordings missing demographic brackets, microphone types, or distance categories fail closed and are quarantined.
3. **Mock Harness Isolation Preserved:**  
   Simulated test fixtures (`provenance="test_harness_mock"`) retain backwards-compatible permissive testing workflows when `strict_metadata=False`, preventing regression in synthetic pipeline unit tests while maintaining strict air-gapping against human data pollution.
4. **Automated Verification:**  
   The comprehensive acceptance test suite in `tests/test_campaign_acceptance.py` now includes 7 dedicated regression tests, passing with **31/31 passed (100%)** in 2.32 seconds.
5. **Cryptographic Immutability:**  
   All three protected baseline artifacts (`test_manifest_v2.csv`, `field_collection_manifest.csv`, `consent_ledger.json`) match their pre-remediation SHA-256 hashes bit-for-bit.
6. **Governance Posture:**  
   **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** Technical readiness does not constitute legal or administrative authorization. Genuine human data collection remains locked pending formal signatures on Gates G-01 through G-05.

---

## 2. Root Cause Analysis (`TECH-FINDING-01`)

### 2.1 The Vulnerability Path
In `src/acquisition/ingest_field_recordings.py`, the ingestion function was defined as:
```python
def ingest_recording(
    input_audio_path: Path,
    speaker_id: str,
    command_label: str,
    ...
    provenance: str = "human_field_recording",
    ...
    strict_metadata: bool = False,
) -> Dict[str, Any]:
```
When a caller executed:
```python
ingest_recording(audio_path, speaker_id, "TRAI", provenance="human_field_recording", strict_metadata=False)
```
or simply omitted `strict_metadata`:
1. **Permissive Fallback Injection:**
   ```python
   age = age_bracket or spk_meta.get("age_bracket") or ("18-30" if not strict_metadata else None)
   mic_type = microphone_type or spk_meta.get("primary_microphone") or ("builtin_mobile" if not strict_metadata else None)
   dist_cat = distance_category or dist_from_spk or ("close_talk_15cm" if not strict_metadata else None)
   ```
   Because `strict_metadata` evaluated to `False`, unverified defaults were injected into the record.
2. **Validation Gate Bypass:**
   ```python
   if strict_metadata:
       # Demographic, microphone, distance, and contradiction checks
   ```
   The entire block was skipped because `strict_metadata` was `False`.
3. **False Canonical Approval:**
   The unverified recording was copied into standardized audio directories and committed to the manifest with `status="APPROVED"` and `label_provenance="HUMAN_CONSENTED"`.
4. **Batch Propagation Vulnerability:**
   `batch_ingest_directory()` also defaulted `strict_metadata=False` and passed it down to `ingest_recording()`, compounding the bypass risk across entire directory ingestions.

---

## 3. Implementation Details

### 3.1 Core Invariant Implementation
In [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py), the function core was updated:

```python
# Core Security Invariant: Strict metadata validation is mandatory for human provenance.
# Callers cannot bypass strict validation for human recordings via strict_metadata=False.
is_human_provenance = provenance in ("human_field_recording", "HUMAN_CONSENTED")
effective_strict = True if is_human_provenance else bool(strict_metadata)

age = age_bracket or spk_meta.get("age_bracket") or ("18-30" if not effective_strict else None)
mic_type = microphone_type or spk_meta.get("primary_microphone") or ("builtin_mobile" if not effective_strict else None)
dist_from_spk = spk_meta.get("distance_category") or spk_meta.get("speaking_distance")
dist_cat = distance_category or dist_from_spk or ("close_talk_15cm" if not effective_strict else None)

# Default acoustic sub-category
if sub_category is not None:
    sub_cat = sub_category
elif command_label == "SILENCE":
    sub_cat = "background_room_tone"
else:
    sub_cat = "isolated_command"

# Strict metadata validation: prevents fabricated defaults from entering human dataset
if effective_strict:
    meta_errs = []
    if dialect not in ["Northern", "Central", "Southern"]:
        meta_errs.append(f"missing_or_invalid_dialect_{dialect}")
    if gender not in ["FEMALE", "MALE", "OTHER"]:
        meta_errs.append(f"missing_or_invalid_gender_{gender}")
    if (age_bracket is None and "age_bracket" not in spk_meta) or age not in ["18-30", "31-45", "46-60", "over_60", "31-50", "51+"]:
        meta_errs.append(f"unregistered_or_invalid_age_bracket_{age}")
    if (microphone_type is None and "primary_microphone" not in spk_meta) or mic_type not in VALID_MICROPHONE_TYPES:
        meta_errs.append(f"unregistered_or_invalid_microphone_type_{mic_type}")
    if (distance_category is None and not dist_from_spk) or dist_cat not in VALID_DISTANCE_CATEGORIES:
        meta_errs.append(f"missing_or_invalid_distance_category_{dist_cat}")

    # Contradictory metadata checks
    if command_label == "SILENCE" and sub_cat not in ["background_room_tone", "ambient_noise"]:
        meta_errs.append("contradictory_metadata_silence_command_with_non_silence_sub_category")
    if command_label in ["TRAI", "PHAI", "LEN", "XUONG"] and sub_cat in ["background_room_tone", "ambient_noise"]:
        meta_errs.append("contradictory_metadata_speech_command_with_room_tone_sub_category")
    if dist_cat in ["far_field_2m"] and mic_type in ["wired_headset", "earbuds", "bluetooth_earbuds"]:
        meta_errs.append("contradictory_metadata_wired_headset_with_far_field_distance")

    if meta_errs:
        reason_str = "; ".join(meta_errs)
        # Quarantine recording and abort before filesystem writes or manifest append
        ...
        return {"status": "QUARANTINED", "reason": reason_str, "recording_id": None}
```

### 3.2 Batch Ingestion Enforcement
In `batch_ingest_directory()`, the invariant is enforced prior to file iteration and passed to each individual file call:
```python
is_human_provenance = provenance in ("human_field_recording", "HUMAN_CONSENTED")
effective_strict = True if is_human_provenance else bool(strict_metadata)

for idx, audio_file in enumerate(files, start=1):
    res = ingest_recording(
        ...
        provenance=provenance,
        strict_metadata=effective_strict
    )
```

### 3.3 Test Harness Isolation
For unit tests operating on mock fixtures (`provenance="test_harness_mock"`), `is_human_provenance` evaluates to `False`. The behavior defaults to `bool(strict_metadata)`. Mock fixtures can run permissively when needed without risking human dataset pollution.

---

## 4. Test Verification Evidence

### 4.1 Regression Test Suite Structure
Seven new regression tests were added to [`tests/test_campaign_acceptance.py`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py):

| Test Case Name | Target Invariant Verified | Result |
|---|---|:---:|
| `test_direct_ingest_human_recording_strict_false_fails_closed` | Direct call with `provenance="human_field_recording", strict_metadata=False` fails closed and quarantines when mic/distance missing | **PASSED** |
| `test_direct_ingest_human_cannot_rely_on_fallback_defaults` | Default argument invocation without metadata rejects; confirms fallback strings (`18-30`, etc.) are never injected | **PASSED** |
| `test_batch_ingest_directory_human_recording_strict_false_fails_closed` | `batch_ingest_directory()` rejects incomplete human metadata even when caller passes `strict_metadata=False` | **PASSED** |
| `test_fully_specified_human_metadata_approved` | Fully specified human metadata (registered via ledger or explicit parameters) passes strict validation into 28-column manifest | **PASSED** |
| `test_mock_fixture_permissive_defaults_isolated` | Mock test harnesses (`provenance="test_harness_mock"`) remain isolated: permissive when False, strict when True | **PASSED** |
| `test_failed_strict_validation_leaves_manifest_untouched` | Validation rejection leaves manifest completely untouched with 0 approved rows added; quarantine log records exclusion | **PASSED** |
| `test_consent_revocation_and_lineage_with_strict_core` | End-to-end lifecycle under strict core: ingestion, approval, confirmed consent revocation (`REVOKED_CONFIRMED`), audio purging | **PASSED** |

### 4.2 Automated Test Execution Output
```
PS D:\trainidentificationaudio> pytest tests/test_campaign_acceptance.py -q
...............................                                          [100%]
31 passed in 2.32s
```
All 31 test cases executed within a bounded runtime of 2.32 seconds with zero test failures and zero hanging threads.

---

## 5. Cryptographic File Hash Invariant Table

All protected files in the repository were verified before and after remediation using Windows PowerShell `Get-FileHash -Algorithm SHA256`.

| Protected File Path | Baseline SHA-256 Hash | Post-Remediation SHA-256 Hash | Status |
|---|---|---|:---:|
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |

Zero bytes of benchmark data or historical field records were modified or polluted during this remediation.

---

## 6. Git Worktree Status

```
PS D:\trainidentificationaudio> git status --short
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

---

## 7. Operational Governance Gate Status

While `TECH-FINDING-01` is resolved and strict validation is enforced at the core, human data collection remains governed by the approval gates documented in [`governance_approval_action_tracker.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_approval_action_tracker.md):

| Gate ID | Governance Gate Name | Current Status | Blocker Description |
|:---:|---|:---:|---|
| **G-01** | Informed Consent Protocol & Form Legal Approval | `PENDING HUMAN APPROVAL` | Awaiting Project Sponsor & Legal sign-off on consent template. |
| **G-02** | Participant Recruitment Quota Approval (50N/30C/40S) | `PENDING HUMAN APPROVAL` | Proposed demographic quota requires authorized sponsor sign-off. |
| **G-03** | Technical Ingestion & Quality Control Infrastructure | `READY FOR REVIEW` | `TECH-FINDING-01` closed. Strict mode enforced at ingestion core. |
| **G-04** | External Training/Noise Corpus Licensing Verification | `PARTIALLY BLOCKED` | `mic-noise.zip` remains in quarantine pending license confirmation. |
| **G-05** | Production Data Ingestion & Model Training Authorization | `STRICTLY BLOCKED` | Dependent on G-01, G-02, G-03, and G-04 sign-offs. |

### Final Instruction to Operators
> [!CAUTION]
> **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**  
> No recording sessions with genuine human participants, recruitment flyers, consent executions, or corpus downloads may take place until formal written sign-offs on Gates G-01, G-02, G-04, and G-05 are obtained and recorded in `human_governance_signoff_record.md`.
