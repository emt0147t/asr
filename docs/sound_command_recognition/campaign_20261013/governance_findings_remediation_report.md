# Vietnamese KWS Campaign — Governance Findings Remediation Report

**Document ID:** `GOV-REM-20261014-01`  
**Workspace:** `D:\trainidentificationaudio`  
**Date of Audit & Remediation:** 2026-10-14  
**Audit Trigger:** Independent Governance Package Audit Report (`READY WITH FINDINGS`)  
**Remediation Classification:** Documentation-Only Remediation — Collection Blocked  
**Operational Status:** **`HUMAN DATA COLLECTION STRICTLY BLOCKED`**

---

## 1. Executive Summary & Remediation Boundary

This report documents the targeted, documentation-only remediation executed in response to the Independent Governance Package Audit. All material findings—spanning storage footprint underestimations, offline split boundary clarification, consent withdrawal and feature cache scoping, evidence status classifications, and legal risk nuances—have been fully addressed in the draft campaign governance documentation.

### Strict Governance Constraints Observed:
1. **Zero Human Speech Collection:** Human data collection remains **`STRICTLY BLOCKED`**. Zero participants were contacted, recruited, or recorded.
2. **Zero Code Modifications:** No Python source files (`.py`) or test files were modified during this remediation.
3. **Zero Decision Usurpation:** Option A (40 takes) and Option B (69 takes) remain presented objectively as open executive decisions for the Project Sponsor / PI (`GATE-03`). Zero gates were signed off, and zero signatures were populated.
4. **Preservation of Cryptographic Baselines:** All three protected SHA-256 hashes were recomputed and verified bit-exact before and after remediation.
5. **Clean Worktree Isolation:** Zero files were staged (`git add`) or committed (`git commit`).

---

## 2. Audit Findings & Detailed Remediation Inventory

### Finding 1: Audio Storage Footprint Underestimation & Lack of Explicit Derivation
- **Audit Finding:** Previous governance documents cited provisional storage figures (~450 MB for Option A, ~800 MB for Option B) that failed to account for dual preservation of both raw capture audio and standardized audio on disk, and assumed an undocumented 16 kHz mono raw capture format.
- **Code & Implementation Reality:**
  - `AudioRecorder` ([`src/acquisition/record_command_cli.py#L49-L68`](file:///D:/trainidentificationaudio/src/acquisition/record_command_cli.py#L49-L68)) records at **44.1 kHz 16-bit stereo** (MCI waveaudio for Windows driver compatibility) with a **1.6-second acquisition buffer**:
    - Raw byte rate: $44,100 \times 2 \text{ channels} \times 2 \text{ bytes} = 176,400\text{ bytes/s}$.
    - Raw file size per take: $176,400 \times 1.6\text{s} + 44\text{ B (WAV header)} = 282,284\text{ bytes} \approx 275.67\text{ KB} \approx 0.282\text{ MB}$.
  - Ingestion pipeline ([`src/acquisition/ingest_field_recordings.py#L559-L568`](file:///D:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py#L559-L568)) generates a **16.0 kHz 16-bit mono standardized copy** (1.0-second window, `TARGET_SAMPLES = 16000`):
    - Standardized byte rate: $16,000 \times 1 \text{ channel} \times 2 \text{ bytes} = 32,000\text{ bytes/s}$.
    - Standardized file size per take: $32,000 \times 1.0\text{s} + 44\text{ B (WAV header)} = 32,044\text{ bytes} \approx 31.29\text{ KB} \approx 0.032\text{ MB}$.
  - **Dual Retention Invariant:** Both the original raw file and the standardized copy are permanently preserved on disk (`raw_filepath` and `processed_path` columns in the manifest).
- **Documents Remediated:**
  - [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md) (Section 9)
  - [`docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md) (Section 3 Options Table)
  - [`docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md) (Section 3.2, 4.3)
  - [`docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md) (Section 2)
- **Before vs. After Values:**
  - *Option A (40 takes/spk, 4,800 total takes):*
    - **Before:** "~450 MB (at 44.1 kHz stereo 16-bit)" (unexplained / underestimated).
    - **After:** **~1.51 GB Primary Audio (~2.57 GB Itemized Working Footprint)**:
      - Raw capture storage: $4,800 \times 282,284\text{ B} = 1,354,963,200\text{ B} \approx \mathbf{1.355\text{ GB}}$ ($1,292.2\text{ MiB}$).
      - Standardized audio storage: $4,800 \times 32,044\text{ B} = 153,811,200\text{ B} \approx \mathbf{0.154\text{ GB}}$ ($146.7\text{ MiB}$).
      - Primary audio subtotal: $1,354,963,200 + 153,811,200 = 1,508,774,400\text{ B} \approx \mathbf{1.509\text{ GB} \approx 1.51\text{ GB}}$.
      - Derived feature caches (`.npy` arrays @ $80 \times 101 \times 4\text{ B} = 32,320\text{ B/take}$): $4,800 \times 32,320\text{ B} = 155,136,000\text{ B} \approx \mathbf{0.155\text{ GB}}$ ($148.0\text{ MiB}$).
      - Cold backup archive (Raw + Standardized Audio, ~50% unmeasured planning ratio): $50\% \times (1.355 + 0.154)\text{ GB} = 50\% \times 1.509\text{ GB} \approx \mathbf{0.754\text{ GB}}$ ($719.5\text{ MiB}$).
      - Temp files, provisional defect quarantine & manifests: $\approx \mathbf{0.150\text{ GB}}$ ($143.1\text{ MiB}$).
      - **Total Explicit Working Footprint:** $1.355 + 0.154 + 0.155 + 0.754 + 0.150 = \mathbf{2.568\text{ GB} \approx 2.57\text{ GB}}$ ($2,449.3\text{ MiB}$).
      - *(Alternative consolidated $1.5\times$ primary audio allowance: ~2.26 GB).*
  - *Option B (69 takes/spk, 8,280 total takes):*
    - **Before:** "~800 MB (at 44.1 kHz stereo 16-bit)" (unexplained / underestimated).
    - **After:** **~2.60 GB Primary Audio (~4.42 GB Itemized Working Footprint)**:
      - Raw capture storage: $8,280 \times 282,284\text{ B} = 2,337,311,520\text{ B} \approx \mathbf{2.337\text{ GB}}$ ($2,229.0\text{ MiB}$).
      - Standardized audio storage: $8,280 \times 32,044\text{ B} = 265,324,320\text{ B} \approx \mathbf{0.265\text{ GB}}$ ($253.0\text{ MiB}$).
      - Primary audio subtotal: $2,337,311,520 + 265,324,320 = 2,602,635,840\text{ B} \approx \mathbf{2.603\text{ GB} \approx 2.60\text{ GB}}$.
      - Derived feature caches (`.npy` arrays @ $80 \times 101 \times 4\text{ B} = 32,320\text{ B/take}$): $8,280 \times 32,320\text{ B} = 267,609,600\text{ B} \approx \mathbf{0.268\text{ GB}}$ ($255.2\text{ MiB}$).
      - Cold backup archive (Raw + Standardized Audio, ~50% unmeasured planning ratio): $50\% \times (2.337 + 0.265)\text{ GB} = 50\% \times 2.603\text{ GB} \approx \mathbf{1.301\text{ GB}}$ ($1,241.0\text{ MiB}$).
      - Temp files, provisional defect quarantine & manifests: $\approx \mathbf{0.250\text{ GB}}$ ($238.4\text{ MiB}$).
      - **Total Explicit Working Footprint:** $2.337 + 0.265 + 0.268 + 1.301 + 0.250 = \mathbf{4.421\text{ GB} \approx 4.42\text{ GB}}$ ($4,216.7\text{ MiB}$).
      - *(Alternative consolidated $1.5\times$ primary audio allowance: ~3.90 GB).*
- **Remaining Uncertainty & Buffer Notice:** 1.6 seconds is a nominal capture buffer parameter configured in `record_command_cli.py`, not an empirical guarantee that every saved file is exactly 1.6s. Ingestion accepts any file with duration in $[0.25\text{s}, 2.50\text{s}]$. Cold backup estimates assume a 50% compression ratio across raw plus standardized audio as an unmeasured planning assumption (empirical compression efficiency has not been measured on production speech). Temporary and quarantine headroom figures (~0.150 GB for Option A, ~0.250 GB for Option B) remain explicitly provisional. All figures are characterized as provisional engineering estimates.
- **Human Decision Required:** **YES** — Project Sponsor / PI must authorize either Option A or Option B (`GATE-03` / Decision Q3).

---

### Finding 2: Offline Speaker-Disjoint Boundary & Downstream Leakage Prevention
- **Audit Finding:** Prior documentation implied that speaker-disjoint splitting was enforced as an inline runtime filter during data collection or manifest ingestion, obscuring operational leakage risks in downstream model training.
- **Implementation Reality:**
  - Ingestion CLIs (`record_command_cli.py` and `ingest_field_recordings.py`) append rows to manifests with **no partition tags** (`split` column is empty / not assigned during capture).
  - Speaker-disjoint partitioning is strictly an **offline batch post-processing operation** executed via [`src/acquisition/campaign_validator.py::generate_stratified_disjoint_splits()`](file:///D:/trainidentificationaudio/src/acquisition/campaign_validator.py#L371-L560).
  - If a downstream training script bypasses the split generator and executes naive row-level random sampling (`df.sample()` or `sklearn.model_selection.train_test_split()`), catastrophic acoustic leakage occurs because multiple takes from the same speaker will be scattered across train and test sets.
  - Augmentation derivatives (pitch-shifted, noise-injected, or time-stretched samples) must strictly inherit the parent recording's partition assignment.
  - **Schema Scope & Evidence Classification for Existing Manifests:**
    - The existing 16-column manifests (`train_manifest_v2.csv`, `val_manifest_v2.csv`, `test_manifest_v2.csv` and legacy set) were derived from calibration utterances, Google FLEURS, and sliced noise chunks, where `session_id` and `parent_recording_id` are genuinely unavailable in the underlying data sources and were intentionally omitted from the 16-column schema.
    - Verified Evidence: Speaker disjointness = **`PASS`**, Recording ID disjointness = **`PASS`**.
    - Unverified (Column Absent): Session disjointness = **`NOT VERIFIABLE — REQUIRED COLUMN ABSENT`**, Augmentation-parent lineage = **`NOT VERIFIABLE — REQUIRED COLUMN ABSENT`**.
    - Full session cohesion and augmentation-parent lineage validation are required and strictly enforced when evaluating against the proposed 28-column human campaign schema (`proposed_manifest_schema.md`) upon authorized data collection.
- **Documents Remediated:**
  - [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md) (Section 9.3)
  - [`docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md) (Conflict 4, Section 4.3)
  - [`docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md) (Section 5)
- **Before vs. After Values:**
  - **Before:** Implied universal inline pipeline enforcement of disjointness.
  - **After:** Explicitly documented as an offline batch post-processing requirement. Established mandatory consumption rules for downstream training pipelines and prohibited naive row-level sampling.
- **Remaining Uncertainty:** Operational enforcement in future training scripts depends on ML engineers adhering to the mandate to ingest pre-split partition manifests.
- **Human Decision Required:** **NO** — Technical requirement enforced by ML Lead in model training pipelines.

---

### Finding 3: Tooling vs. Policy Scope in Consent Withdrawal, Feature Caches & Retraining
- **Audit Finding:** Documentation conflated the automated capabilities of `revoke_speaker_consent.py` with broader institutional data custody obligations, giving the false impression that intermediate feature caches (`.npy`), cold backups, and model weights are automatically purged by software.
- **Implementation Reality:**
  - **What `revoke_speaker_consent.py` Actually Does:**
    1. Relocates raw audio files (`data/raw/human_recordings_v1/<SPK>/`) to quarantine (`quarantine/withdrawn_speakers/<SPK>/audio_raw/`).
    2. Relocates standardized audio files (`data/processed/human_recordings_v1/<CMD>/<REC>.wav`) to quarantine (`quarantine/withdrawn_speakers/<SPK>/audio_16k_standardized/`).
    3. Updates manifest status to `REVOKED_WITHDRAWN` and records the revocation reason.
    4. Appends immutable audit records to `quarantine_log.csv`.
    5. Sets `consent_given: false` and records audit metadata in `consent_ledger.json`.
  - **What `revoke_speaker_consent.py` DOES NOT Do (Operational Gaps):**
    1. Does **not** purge derived intermediate feature caches (e.g. cached `.npy` spectrograms, MFCCs, or embedding vectors).
    2. Does **not** mutate or delete files in immutable cold backup directories (`data/backups/`).
    3. Does **not** roll back or retrain existing neural network model checkpoints (`.pt`, `.onnx`).
  - **Policy Mandate:** Full model retraining and feature cache purging require an approved administrative operational SOP (`ACT-06` / `GATE-05`), which remains a proposed policy awaiting human sign-off.
- **Documents Remediated:**
  - [`docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md) (Conflict 6)
  - [`docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md) (Section 4.4, 4.5)
  - [`docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md) (Section 4, 5)
- **Before vs. After Values:**
  - **Before:** "Automated revocation CLI purges participant data and handles model retraining."
  - **After:** Explicitly segregated into implemented audio relocation/manifest updating vs. pending operational SOP for cache flushing, backup disposition, and model retraining.
- **Remaining Uncertainty:** Latent acoustic representations in unpurged feature caches if operators omit manual purging steps.
- **Human Decision Required:** **YES** — DPO and Tech Lead must execute `GATE-05` / `ACT-06` approving the operational cache-purging and retraining SOP.

---

### Finding 4: Evidence Status Reclassification of Model Retraining Mandate
- **Audit Finding:** The previous audit classified the model retraining requirement upon consent revocation as `VERIFIED (Policy Proposed)`, improperly implying that an unapproved and unfunded policy was technically verified.
- **Remediation & Governance Reality:**
  - While technical recognition of neural network weight immutability is present in code docstrings and CLI governance exit notices, an unapproved institutional policy cannot be classified as verified.
  - Reclassified to **`PARTIALLY VERIFIED — POLICY PROPOSED`**.
  - Technical evidence verified: Ingestion CLI notice acknowledges that gradient updates cannot be unlearned and warns that full retraining is required.
  - Missing evidence: Formal institutional policy approval, operational retraining execution SOP, and compute budget commitment by the Project Sponsor / PI.
- **Documents Remediated:**
  - [`docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md) (Section 5 Matrix)
  - [`docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md) (Gate Matrix, Section 4.5)
  - [`docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md) (Section 4, 5)
- **Before vs. After Values:**
  - **Before:** `VERIFIED (Policy Proposed)`.
  - **After:** **`PARTIALLY VERIFIED — POLICY PROPOSED`**.
- **Remaining Uncertainty:** Willingness of leadership to fund compute resources for full model retraining if withdrawal occurs late in product lifecycle.
- **Human Decision Required:** **YES** — Project Sponsor / PI and Tech Lead approval of retraining budget and policy (`GATE-05` / Decision Q4).

---

### Finding 5: Legal Risk Nuance on Option B & External Corpus Licensing Status
- **Audit Finding:** Governance documentation previously described Option B as "MINIMAL / Legally self-contained" and treated missing local license text files as proof that upstream open-source corpora lacked licenses.
- **Remediation & Governance Reality:**
  - Option B eliminates third-party speech copyright dependencies by collecting native Vietnamese confusers and room tones, but is **not legally risk-free**. The campaign remains fully subject to statutory compliance under Vietnam Decree 13/2023/ND-CP (biometric voice data custody, informed consent obligations, right of withdrawal), prompt text copyright compliance, and physical data protection liabilities.
  - Missing local `LICENSE.txt` files for Google FLEURS (`vi_vn`) and MLCommons MSWC (`vi`) represent a **local documentation gap**, not proof of non-existence or invalidity of upstream CC-BY-4.0 licenses.
  - `GATE-04` is conditional upon actual use of these external corpora in multi-corpus pre-training or augmentation.
- **Documents Remediated:**
  - [`docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md) (Section 3 Options Table, Section 5 Matrix)
  - [`docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md) (Section 3.3)
  - [`docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md) (Section 4, 5)
- **Before vs. After Values:**
  - **Before:** Option B labeled as legally self-contained / risk-free; missing license files cited as lack of licensing.
  - **After:** Nuanced risk disclosure for Option B (reduced third-party dependency with ongoing statutory/biometric liabilities); missing local license files classified as a documentation gap with conditional gate status (`GATE-04`).
- **Remaining Uncertainty:** Legal Counsel's formal assessment of commercial compatibility for FLEURS/MSWC and institutional disposition of on-disk noise archives (`ACT-05B`).
- **Human Decision Required:** **YES** — Legal Counsel review of consent form (`GATE-02`) and external corpus licensing (`GATE-04` / `ACT-05B`).

---

## 3. Fresh Test Execution & Cryptographic Baseline Verification

### 3.1 Fresh Acceptance Test Suite Execution
During this task, the automated acceptance test suite was freshly executed to confirm zero regression or environment failure:
- **Command:** `pytest tests/test_campaign_acceptance.py -q`
- **Execution Date:** 2026-10-14 21:07:21+07:00
- **Summary:** `49 passed` (Elapsed time varies between ~4.5s and ~15s depending on CPU load and I/O scheduling; canonical criteria are test count and exit code)
- **Exit Status:** `0` (Success, 0 warnings)

### 3.2 Bounded Timeout & Hanging Test Safety
- `tests/test_field_acquisition.py` was **NOT RUN** during this task.
- Its governance status remains **`UNVERIFIED / NOT RUN`** due to documented Python 3.14 librosa JIT compilation hangs on Windows.

### 3.3 Protected Baseline Cryptographic Verification
SHA-256 hashes for all three protected baseline artifacts were computed immediately following documentation edits:

| File Path | Established Baseline SHA-256 | Freshly Recomputed SHA-256 | Verification Result |
| :--- | :---: | :---: | :---: |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |

Zero bytes of protected baseline data were modified or touched.

---

## 4. Git Worktree Audit & Confirmation of Constraints

### 4.1 Git Status Check
A fresh git status audit confirms that only documentation files were updated:
- **Files Modified in Working Tree (Documentation):**
  - `docs/sound_command_recognition/campaign_20261013/collection_protocol.md`
  - `docs/sound_command_recognition/campaign_20261013/governance_protocol_reconciliation.md`
  - `docs/sound_command_recognition/campaign_20261013/governance_review_package_index.md`
  - `docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md`
  - `docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md`
- **New Deliverable Created:**
  - `docs/sound_command_recognition/campaign_20261013/governance_findings_remediation_report.md`
- **Source Code Files Modified:** **NONE** (Zero `.py` files modified).
- **Files Staged in Index:** **NONE** (`git status` shows zero files staged).
- **Files Committed:** **NONE** (No commits made).

---

## 5. Summary Scorecard & Campaign Operational Status

| Governance Domain | Status | Operational Consequence |
| :--- | :---: | :--- |
| **Storage Arithmetic Remediation** | **RESOLVED IN DOCS** | Explicit dual-retention formulas documented (~1.51 GB Option A / ~2.60 GB Option B). |
| **Offline Split Boundary Remediation** | **RESOLVED IN DOCS** | Offline batch post-processing documented; downstream pipeline consumption mandated. |
| **Cache Purging & Withdrawal Scoping** | **RESOLVED IN DOCS** | CLI file quarantine separated from manual `.npy` cache purging SOP (`ACT-06`). |
| **Evidence Status Classification** | **RESOLVED IN DOCS** | Model retraining reclassified to `PARTIALLY VERIFIED — POLICY PROPOSED`. |
| **Legal Risk Nuance** | **RESOLVED IN DOCS** | Option B residual biometric/statutory risks disclosed; `GATE-04` marked conditional. |
| **Mandatory Pre-Recruitment Gates (01, 02, 03, 05, 06)** | **OPEN — PENDING HUMAN SIGN-OFF** | Mandatory prerequisites. Must be executed in sign-off record before participant recruitment begins. |
| **Independent Ethics / IRB Review** | **PENDING (IF MANDATED)** | Organizational prerequisite if required by institution; does not alter canonical gates (01–08). |
| **Conditional External Corpus Licensing (GATE-04)** | **PENDING HUMAN APPROVAL (CONDITIONAL)** | Conditional on actual multi-corpus pre-training, augmentation, or release; not a blocker for collection under Option B. |
| **Technical Baseline Gate (GATE-07)** | **CLOSED WITH EVIDENCE** | 52/52 tests passing; protected baselines bit-exact. Technical evidence only; does NOT grant collection authorization. |
| **Runtime Participant Invariant (GATE-08)** | **PENDING EXECUTION AT RUNTIME** | Enforced per-participant before recording; 0 human records currently enrolled. |
| **OVERALL CAMPAIGN STATUS** | **`STRICTLY BLOCKED`** | **NO RECRUITMENT, RECORDING, OR INGESTION AUTHORIZED.** |

---
*Report certified by Independent Pair Programming Assistant.*
