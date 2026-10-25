# Governance Protocol Reconciliation: Vietnamese KWS Campaign
## Discrepancy Resolution & Decision Framework (No Collection)

**Workspace:** `D:\trainidentificationaudio`  
**Document Version:** 1.0.0-PROTOCOL-RECONCILIATION  
**Assessment Date:** 2026-10-14  
**Target Ingestion Target:** Genuine Human Field Speech Collection (`v1.0.0-human`)  
**Auditor / Custodian Role:** Independent Code & Governance Auditor (Pair Programming Assistant)  
**Governing Principle:** Objective Conflict Registration Without Unauthorized Decision-Making  
**Physical Human Data Collection Status:** **`STRICTLY BLOCKED`**  
*(Prohibited until all mandatory pre-recruitment gates are formally resolved and authorized by designated human authorities)*

---

## 1. Executive Summary

A forensic review across all campaign governance documents, protocol specifications, technical reports, and pipeline scripts identified multiple material discrepancies regarding:
1. Participant recording volume (40 vs. 69 takes per speaker);
2. Negative (`UNKNOWN`) and room tone (`SILENCE`) targets;
3. Dependency on external speech and noise corpora (FLEURS, MSWC, Husein);
4. Train/validation/test split methodologies and speaker independence;
5. Demographic cohort quotas and fine-grained integer rounding constraints;
6. Retention limits, participant withdrawal mechanics, feature cache purging, and model retraining mandates.

This document compiles an objective **Conflict Register**, provides an **Options Table** evaluating alternative collection designs without selecting one on behalf of project leadership, establishes **Speaker-Independent Partitioning Requirements**, structures an **Evidence and Approval Matrix**, and outlines **Prioritized Action Items** for named human authorities.

---

## 2. Conflict Register

The following conflicting statements, numbers, and assumptions exist across the repository's governance and engineering documentation:

### Conflict 1: 40 Takes versus 69 Takes per Participant
- **Statement A (40 Takes — Keyword Focus):**  
  [`human_governance_decision_brief.md#L28`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md#L28) states: *"Each participant contributes 40 target command takes across speaking dynamics (normal, fast, slow, far-field), yielding a target yield of 4,800 authentic utterances."*  
  Similarly, [`speaker_disjoint_split_policy.md#L78-L80`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md#L78-L80) projects individual dialect yields assuming 40 takes per speaker (Northern: 1,400 train, 320 val, 280 test = 2,000 takes; Central: 800 train, 200 val, 200 test = 1,200 takes; Southern: 1,160 train, 200 val, 240 test = 1,600 takes; sum = 4,800 takes).  
  In [`src/acquisition/record_command_cli.py#L361`](file:///D:/trainidentificationaudio/src/acquisition/record_command_cli.py#L361), default commands are limited to `["TRAI", "PHAI", "LEN", "XUONG"]`.
- **Statement B (69 Takes — Full Multiclass Protocol):**  
  [`collection_protocol.md#L205-L220`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L205-L220) defines a total of 69 takes per speaker (40 target keywords + 25 `UNKNOWN` negative utterances + 4 `SILENCE` room tones), yielding **8,280 recordings (~2.38 hours)** across 120 participants.  
  In [`speaker_disjoint_split_policy.md#L81`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md#L81), the table TOTAL row states: `84 speakers (~5,796 takes) | 18 speakers (~1,242 takes) | 18 speakers (~1,242 takes)`, which equals 8,280 total takes ($84 \times 69 = 5,796$; $18 \times 69 = 1,242$).
- **Forensic Diagnosis:** An internal arithmetic mismatch existed in Table 3.1 of `speaker_disjoint_split_policy.md`, where the individual dialect rows were computed using 40 takes/speaker ($3,360 + 720 + 720 = 4,800$), while the summary row was copied from the 69 takes/speaker calculation ($5,796 + 1,242 + 1,242 = 8,280$).

---

### Conflict 2: Negative (`UNKNOWN`) and Room Tone (`SILENCE`) Target Counts
- **Statement A (Omitted / Zero Takes in Decision Brief):**  
  [`human_governance_decision_brief.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md) budgets zero human utterances for `UNKNOWN` and zero for `SILENCE`, focusing exclusively on the 4,800 target keyword takes.
- **Statement B (Explicit 3,480 Negative Takes in Protocol):**  
  [`collection_protocol.md#L150-L185`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L150-L185) specifies:
  - **`UNKNOWN`:** 25 takes per speaker (16 phonetic confusers such as *trắng, phải, lên, xuống*, 4 embedded carrier sentences, 5 conversational sentences) $\rightarrow$ **3,000 native negative takes** across 120 contributors.
  - **`SILENCE`:** 4 ambient room tones per speaker (1.6s duration each) $\rightarrow$ **480 authentic silence takes** across 120 contributors.
- **Forensic Diagnosis:** The Decision Brief presented an incomplete session burden to the Project Sponsor by omitting the 29 non-keyword takes required by the formal Collection Protocol.

---

### Conflict 3: External Corpus Dependencies (FLEURS, MSWC, Husein)
- **Statement A (Self-Contained Human Dataset — No External Speech Needed):**  
  [`final_governance_evidence_verification.md#L77`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/final_governance_evidence_verification.md#L77) asserts that external speech corpora (Google FLEURS, MLCommons MSWC) are **`NOT REQUIRED`** for the human evaluation benchmark because the 120 human participants generate 3,000 native `UNKNOWN` utterances and 480 `SILENCE` room tones, making the evaluation dataset fully self-contained.
- **Statement B (External Multi-Corpus Dependency for Training):**  
  [`human_governance_decision_brief.md#L79`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md#L79) (Decision Q5) and [`human_governance_signoff_record.md#L32-L33`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_signoff_record.md#L32-L33) list FLEURS (`vi_vn`) and MSWC (`vi`) as proposed sources requiring human sign-off (`[ ] ALLOWED [ ] PROHIBITED`) and local license archiving (`GATE-04`).
- **Statement C (Prohibited Noise Source with Unresolved Disk Archive):**  
  [`corpus_license_evidence_matrix.md#L73-L90`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/corpus_license_evidence_matrix.md#L73-L90) documents that `husein_mic_noise` is permanently blocked from software ingestion (`ACT-05A`), but the physical on-disk archive `data/raw/noise/mic-noise.zip` (45.9 MB, 39 MP3s) remains in quarantine pending institutional legal disposition (`ACT-05B`).

---

### Conflict 4: Split Methodology & Speaker Independence Across Components
- **Statement A (Strict Speaker-Disjoint Hash Partitioning via Offline Batch):**  
  [`speaker_disjoint_split_policy.md`](file:///D:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/speaker_disjoint_split_policy.md) and [`src/acquisition/campaign_validator.py#L371-L560`](file:///D:/trainidentificationaudio/src/acquisition/campaign_validator.py#L371-L560) implement deterministic speaker-level hashing (salted CRC32), enforcing zero speaker overlap ($S_{\text{train}} \cap S_{\text{val}} = \emptyset$, etc.), session cohesion, and augmentation lineage cohesion.
- **Statement B (Offline Boundary & Downstream Leakage Risk):**  
  Partition generation is strictly an **offline batch post-processing operation**, not an inline runtime check during audio recording or manifest ingestion. Ingestion CLIs write unpartitioned records to the manifest. If downstream model training scripts bypass `generate_stratified_disjoint_splits()` and apply naive row-level splitting (`df.sample()` or `train_test_split()`), catastrophic acoustic speaker leakage will occur.
- **Resolution Mandate:** Downstream training pipelines must explicitly load and validate the output generated by `generate_stratified_disjoint_splits()`. All transformed or augmented variants must strictly inherit the parent utterance's split assignment.

---

### Conflict 5: Cohort Quotas and Demographic Strata Arithmetic
- **Statement A (Clean Integer Partitions at Full Cohort Level):**  
  Across all documents, the 120-participant target (`50N / 30C / 40S`) divides into $84$ Train (70%), $18$ Validation (15%), and $18$ Test (15%) with zero truncation drift ($84 + 18 + 18 = 120$).
- **Statement B (Regional Sub-Tier Variations):**  
  - In `collection_protocol.md` and `speaker_disjoint_split_policy.md`: Central 30 is divided as **20 Train (66.7%), 5 Val (16.7%), 5 Test (16.7%)**, and Southern 40 is divided as **29 Train (72.5%), 5 Val (12.5%), 6 Test (15.0%)**.
  - In `governance_approval_action_tracker.md`: Central 30 is cited as $21/4/5$ ($70.0\% / 13.3\% / 16.7\%$) and Southern 40 as $28/6/6$ ($70.0\% / 15.0\% / 15.0\%$).
- **Statement C (Mathematical Impossibility in Fine-Grained Strata):**  
  In fine-grained demographic bins (Level 4: region × gender × age, e.g. Central males aged 51+, containing $\approx 2$ contributors), it is mathematically impossible to populate three disjoint partitions without at least one partition receiving **0 speakers**. The documentation must never claim demographic balance across partitions unless verified empirically by split diagnostic logs.

---

### Conflict 6: Retention, Withdrawal, Cache Purging, and Model Retraining
- **Tooling Implementation vs. Operational Policy Scope:**
  - *Automated Tool Scope (`revoke_speaker_consent.py`):* Safely relocates on-disk raw audio (`data/raw/human_recordings_v1/<SPK>/`) and standardized audio (`data/processed/human_recordings_v1/<CMD>/<REC>.wav`) to quarantine (`quarantine/withdrawn_speakers/<SPK>/`), updates manifest rows to `REVOKED_WITHDRAWN`, logs cryptographic audit entries to `quarantine_log.csv`, and sets `consent_given: false` in `consent_ledger.json`.
  - *Operational Tooling Gaps:* The automated CLI does **NOT** purge derived intermediate feature caches (`.npy` spectrograms, acoustic embeddings), does **NOT** alter immutable cold backups (`data/backups/`), and does **NOT** roll back or retrain existing neural network model checkpoints.
- **Pending Human Governance:**
  - The 3-year retention schedule, the manual administrative cache-flushing SOP (`ACT-06`), and the institutional model retraining commitment/budget (`GATE-05`) are **proposed policies pending human approval**, not implemented automated behaviors.

---

## 3. Options Table: 40 versus 69 Takes per Participant

The Project Sponsor / Principal Investigator must choose between two distinct collection designs:

| Dimension | Option A: Keyword Focus (40 Takes / Contributor) | Option B: Full Multiclass Protocol (69 Takes / Contributor) |
| :--- | :--- | :--- |
| **Utterance Breakdown per Contributor** | - 10x `TRAI`<br>- 10x `PHAI`<br>- 10x `LEN`<br>- 10x `XUONG`<br>*(0 `UNKNOWN`, 0 `SILENCE`)* | - 10x `TRAI`<br>- 10x `PHAI`<br>- 10x `LEN`<br>- 10x `XUONG`<br>- 25x `UNKNOWN` (confusers & sentences)<br>- 4x `SILENCE` (ambient room tones) |
| **Total Campaign Yield (120 Contributors)** | **4,800 recordings** (~1.33 hours total) | **8,280 recordings** (~2.38 hours total) |
| **Partition Allocation (Train / Val / Test)** | - Train (84 spk): **3,360 takes**<br>- Val (18 spk): **720 takes** (180 / cmd)<br>- Test (18 spk): **720 takes** (180 / cmd) | - Train (84 spk): **5,796 takes**<br>- Val (18 spk): **1,242 takes** (180/cmd + 450 UNK + 72 SIL)<br>- Test (18 spk): **1,242 takes** (180/cmd + 450 UNK + 72 SIL) |
| **Estimated Participant Session Duration** | **12–15 minutes** per participant | **25–35 minutes** per participant |
| **Reproducible Audio Storage Footprint** | **~1.51 GB Primary Audio (~2.57 GB Itemized Working Footprint)**<br>- Raw capture (44.1k stereo 16-bit, ~1.6s nominal @ 282.3 KB): **1.355 GB**<br>- Standardized (16k mono 16-bit, 1.0s @ 32.0 KB): **0.154 GB**<br>- Derived feature caches (`.npy` @ $80 \times 101 \times 4\text{ B}$): **0.155 GB**<br>- Cold backup archive (Raw + Standardized, ~50% unmeasured planning assumption): **0.754 GB** ($50\% \times 1.509\text{ GB}$)<br>- Temp files, provisional quarantine headroom & manifests: **0.150 GB**<br>- **Total Explicit Working Footprint:** **~2.57 GB** ($1.355 + 0.154 + 0.155 + 0.754 + 0.150$).<br>*(Alternative consolidated $1.5\times$ primary audio allowance: ~2.26 GB).* | **~2.60 GB Primary Audio (~4.42 GB Itemized Working Footprint)**<br>- Raw capture (44.1k stereo 16-bit, ~1.6s nominal @ 282.3 KB): **2.337 GB**<br>- Standardized (16k mono 16-bit, 1.0s @ 32.0 KB): **0.265 GB**<br>- Derived feature caches (`.npy` @ $80 \times 101 \times 4\text{ B}$): **0.268 GB**<br>- Cold backup archive (Raw + Standardized, ~50% unmeasured planning assumption): **1.301 GB** ($50\% \times 2.603\text{ GB}$)<br>- Temp files, provisional quarantine headroom & manifests: **0.250 GB**<br>- **Total Explicit Working Footprint:** **~4.42 GB** ($2.337 + 0.265 + 0.268 + 1.301 + 0.250$).<br>*(Alternative consolidated $1.5\times$ primary audio allowance: ~3.90 GB).* |
| **External Corpus Dependency** | **HIGH DEPENDENCY:** Requires external corpora (FLEURS/MSWC) to provide negative speech for `UNKNOWN` and external noise for `SILENCE`. | **ZERO SPEECH CORPUS DEPENDENCY:** Human participants provide 3,000 authentic negative speech samples and 480 native room tone samples. |
| **Legal & Licensing Risk Exposure** | **ELEVATED THIRD-PARTY EXPOSURE:** Subject to external CC-BY-4.0 attribution, upstream dataset integrity, and missing on-disk license texts for FLEURS and MSWC (`GATE-04`). | **REDUCED THIRD-PARTY DEPENDENCY (WITH RESIDUAL RISKS):** Eliminates upstream speech copyright dependencies, but **not legally risk-free**; subject to Vietnam Decree 13/2023/ND-CP biometric data compliance, prompt text copyright, and post-training consent withdrawal liabilities. |
| **Operator CLI Execution** | Standard CLI invocation (`--commands TRAI PHAI LEN XUONG --takes 10`). | Extended CLI script with negative speech prompts (`--commands TRAI PHAI LEN XUONG UNKNOWN SILENCE`). |
| **Financial Cost / Participant Honorarium** | Lower participant compensation burden. | Higher participant compensation required due to longer session time. |

> [!NOTE]
> **Consequence Summary:** Option A reduces participant recording time by ~50% but leaves the model dependent on third-party negative speech corpora that currently lack local on-disk license verification. Option B increases field time and compensation costs but produces an independent, self-contained native evaluation and training benchmark while retaining statutory privacy and consent governance obligations.

---

## 4. Speaker-Independent Partitioning Requirements

To ensure scientific validity and prevent acoustic leakage, all human field recording splits must satisfy the following criteria:

### 4.1 Formal Mathematical Invariants

Let $S_{\text{train}}, S_{\text{val}}, S_{\text{test}}$ be the sets of unique speaker IDs in each partition:
1. **Strict Speaker Disjointness:**
   $$S_{\text{train}} \cap S_{\text{val}} = \emptyset \quad \land \quad S_{\text{train}} \cap S_{\text{test}} = \emptyset \quad \land \quad S_{\text{val}} \cap S_{\text{test}} = \emptyset$$
2. **Session Cohesion:** All takes from a recording session must reside in exactly one partition:
   $$\forall s \in \text{Sessions}, \quad |\{\text{partition}(u) \mid u \in U \land \text{ses}(u) = s\}| = 1$$
3. **Lineage / Augmentation Cohesion:** Any augmented or resampled derivative of utterance $u$ must remain in the same partition as $u$:
   $$\forall u' \in \text{Augmentations}(u), \quad \text{partition}(u') = \text{partition}(u)$$

### 4.2 Prohibited Partitioning Practices
- **Row-level random splitting** (`sklearn.model_selection.train_test_split(df)` or `df.sample()`) is **strictly prohibited**. It causes speaker leakage by distributing different takes of the same speaker across train and test sets, artificially inflating accuracy scores.
- **Splitting multi-speaker external corpora** without speaker ID metadata (e.g. naive slicing of environmental noise files) must not be applied to human speech data.

### 4.3 Operational Pipeline Boundary & Leakage Defense
- **Offline Batch Processing:** Partition assignment is explicitly decoupled from data ingestion. Manifest ingestion scripts append rows with no partition tags. Splitting is executed offline via [`campaign_validator.py::generate_stratified_disjoint_splits()`](file:///D:/trainidentificationaudio/src/acquisition/campaign_validator.py#L371-L560).
- **Mandatory Downstream Validation:** Training pipelines must consume pre-split partition manifests generated by the disjoint split generator. Naive random sampling in training scripts represents an unmitigated operational leakage risk.
- **Augmentation Lineage Rule:** Any data augmentation pipeline generating perturbed samples must inherit the split of the parent recording (`sha256_raw` / `recording_id`).

---

## 5. Evidence and Approval Matrix

The table below distinguishes what is technically proven from proposed choices and outstanding legal/human requirements:

| Governance Item | Technical Evidence Verified | Proposed Protocol Choices | Missing Legal / License Evidence | Decisions Requiring Human Approval |
| :--- | :--- | :--- | :--- | :--- |
| **Purpose & Scope** | Ingestion pipeline enforces keywords `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE`. | Focus on edge mobile KWS; prohibition of biometric surveillance. | Formal IRB or institutional scope authorization. | **Project Sponsor / PI:** Formally approve research & commercial deployment scope (`GATE-01`). |
| **Informed Consent** | Revocation CLI works (`revoke_speaker_consent.py`); immutable audit logging verified. | Vietnamese consent disclosure text drafted in `collection_protocol.md`. | Formal written legal sign-off under Vietnam Decree 13/2023/ND-CP. | **Legal / Privacy Counsel:** Written approval of Vietnamese consent text (`GATE-02`). |
| **Cohort Quota & Sizing** | 70/15/15 arithmetic verified; CRC32 split generator validated in unit tests. | Target quota of 120 contributors (`50N / 30C / 40S`). | Formal budget allocation for recruitment honorarium. | **Project Sponsor / PI:** Authorize 120-speaker quota, operational budget, and resolve 40 vs 69 takes (`GATE-03`). |
| **External Corpora** | FLEURS & MSWC files present on disk with verified SHA-256 digests; Husein noise blocked. | Option to use FLEURS/MSWC for negative speech augmentation. | Standalone `LICENSE.txt` files missing on disk (documentation gap; not proof of lack of upstream license); Husein legal disposition pending. | **Legal Counsel & Data Steward:** Archive CC-BY-4.0 licenses on disk if external corpora are used; determine noise disposition (`GATE-04`, `ACT-05B`). |
| **Retention & Retraining Policy** | Revocation script updates manifest and moves audio; pre-existing files preserved; CLI notice warns retraining required. | Proposed 3-year retention from dataset freeze; mandatory retraining from scratch upon withdrawal. | Approved operational feature-cache purging SOP for administrators; legal commitment to retraining. | **`PARTIALLY VERIFIED — POLICY PROPOSED`**<br>**DPO & Tech Lead:** Endorse Data Retention, Cache Flushing & Retraining Policy (`GATE-05`). |
| **Hardware & Operator Guide** | Audio recorder supports 44.1k/16k conversion; strict mode catches invalid metadata. | Modern smartphone mix (iOS & Android); 3 distance categories (15 cm, 50 cm, 2 m). | **`UNVERIFIED`**<br>Approved physical hardware serial number roster and field operator calibration SOP pending physical device testing. | **Field Ops Lead & Tech Lead:** Approve physical hardware inventory and calibration guide (`GATE-06`). |
| **Individual Consent Enrollment** | Pipeline fails closed if speaker unconsented (`missing_or_unconsented_speaker`). | Runtime enrollment via `consent_ledger_human.json` using cryptographic nonce IDs. | Zero living human contributors registered on disk. | **Field Acquisition Operators:** Collect signed consent per contributor at runtime (`GATE-08`). |

---

## 6. Prioritized Decision Framework for Named Roles

To transition the campaign toward authorized execution, the following decisions must be made by the designated human authorities:

### 6.1 Project Sponsor / Principal Investigator (PI)
1. **Decision Q1 (Purpose & Scope):** Formally approve the collection and use of Vietnamese speech solely for edge KWS research, benchmarking, and product deployment (`GATE-01`).
2. **Decision Q3 (Quota, Takes & Budget):** 
   - Authorize the 120-participant recruitment target (`50N / 30C / 40S`) and commit necessary operational budget.
   - **Resolve Conflict 1:** Authorize either **Option A (40 takes / 4,800 recordings)** or **Option B (69 takes / 8,280 recordings)** per participant.
   - Acknowledge Level 4 integer rounding constraints (small strata $\le 2$ speakers cannot populate all three partitions).
3. **Decision Q4 (Retraining Commitment):** Formally commit to retraining models from scratch if a contributor withdraws consent prior to commercial release (`GATE-05`).

### 6.2 Legal / Privacy Counsel
1. **Decision Q2 (Consent Legal Vetting):** Review and formally execute Section 5 in `human_governance_signoff_record.md`, approving the Vietnamese Informed Consent Agreement (`GATE-02`).
2. **External Corpus Licensing:** Provide written guidance on FLEURS and MSWC CC-BY-4.0 attribution requirements (`GATE-04`) and issue a formal disposition directive for quarantined `husein_mic_noise` files on disk (`ACT-05B`).

### 6.3 Data Protection Officer (DPO) & Technical Custodian
1. **Data Retention & SOP:** Formally approve the 3-year retention schedule and the operator SOP for manual flushing of derived feature caches (`.npy`, spectrograms) upon participant withdrawal (`GATE-05`).

### 6.4 Field Operations Lead
1. **Hardware & Calibration:** Finalize the physical smartphone serial number roster and complete the Field Operator Distance Calibration Manual (`GATE-06`).

### 6.5 Independent Institutional Ethics / IRB Review (Additional Requirement)
- If institutional ethics committee (IRB) approval is independently required by the sponsoring or host institution, it must be documented and executed as a mandatory organizational prerequisite alongside `GATE-01` and `GATE-02` prior to recruitment.
- Institutional ethics review is an additional governance requirement that does not alter or repurpose canonical gate IDs (`GATE-01` through `GATE-08`).
- **No legal or institutional ethics approval has occurred to date.**

---

## 7. Final Governance Status & Gate Conditions

- **Mandatory Pre-Recruitment Approvals (GATES 01, 02, 03, 05, 06):** **`OPEN — PENDING HUMAN SIGN-OFF`**  
  *(All five gates must be formally executed in `human_governance_signoff_record.md` before participant recruitment or contact is permitted).*
- **Conditional External Corpus Licensing (GATE-04):** **`PENDING HUMAN APPROVAL (CONDITIONAL)`**  
  *(Conditional on actual multi-corpus pre-training, augmentation, or dataset release; not a blocker for collection if self-contained Option B is executed).*
- **Technical Infrastructure Baseline (GATE-07):** **`CLOSED WITH EVIDENCE`**  
  *(Technical baseline verified by 49/49 tests and bit-exact baselines; provides engineering evidence only and does NOT grant governance authorization to collect data).*
- **Per-Participant Runtime Invariant (GATE-08):** **`PENDING EXECUTION AT RUNTIME`**  
  *(Mandatory per-participant consent verification before recording required for each individual contributor).*
- **Campaign Execution State:** **`STRICTLY BLOCKED`**  
*(No participant recruitment or speech recording may occur until formal pre-recruitment signatures are executed).*

