# Human Governance Decision Brief: Vietnamese KWS Campaign

**Target Ingestion Target:** Genuine Human Field Speech Collection (`v1.0.0-human`)  
**Workspace:** `D:\trainidentificationaudio`  
**Date of Preparation:** 2026-10-14  
**Prepared By:** Independent Code & Governance Auditor (Pair Programming Assistant)  
**Document Purpose:** Executive Decision Summary for Project Sponsor/PI and Privacy/Legal Counsel  
**Current Governance Status:** **`PENDING HUMAN DECISION — COLLECTION STRICTLY BLOCKED`**

---

## 1. Executive Summary & Campaign Scope

The Vietnamese Keyword Spotting (KWS) system targets four edge commands: **`TRAI`** (trái), **`PHAI`** (phải), **`LEN`** (lên), and **`XUONG`** (xuống), along with **`UNKNOWN`** and **`SILENCE`**. Previous forensic audits revealed that **100% of existing target-command samples in active manifests are synthetic parametric fixtures**, resulting in severe real-world accuracy degradation—particularly on Northern `TRAI` recall (bottlenecked at 25.0% due to unmodeled native affricate burst `/tɕ/`).

The proposed campaign seeks to collect genuine, native Vietnamese human speech from physical mobile devices to build an evaluation and training dataset that reflects authentic acoustic phonetics, regional dialects, and real ambient environments.

---

## 2. Target Cohort Sizing & Quota Justification: Full Campaign vs. 1,000-Clip Pilot Proposal

### 2.1 Proposed Pilot Campaign (1,000-Clip Curated Dataset — Immediate Target)
In response to the Lecturer / PI directive, a focused pilot target is proposed:
- **Target Size:** Exactly **1,000 accepted audio clips** (250 clips each for `TRAI`, `PHAI`, `LEN`, `XUONG`).
- **Accent Distribution:** Balanced Vietnamese regional representation (Northern ~36%, Central ~32%, Southern ~32%; 90N / 80C / 80S accepted clips per command; marked for formal PI confirmation).
- **Cohort Options:**
  - *Model A (40 speakers):* 15 Northern, 13 Central, 12 Southern; 25 accepted takes per speaker.
  - *Model B (25 speakers):* 9 Northern, 8 Central, 8 Southern; 40 accepted takes per speaker.
- **Estimated Storage Footprint:** ~32.0 MB standardized (1,000 × 32.0 KB) + ~282.3 MB raw capture (1,000 × 282.3 KB nominal) = **~314.3 MB Primary Audio** (~500 MB total itemized working footprint including caches).
- **Core Engineering Benefit:** Minimal logistical overhead; immediately unblocks training and on-device evaluation of lightweight edge models (`TC-ResNet-14`) for gaming voice-command demonstrations (saying "LÊN" to jump) while deferring massive recruitment overhead.

### 2.2 Historical Proposals (Full Production Scope — Preserved for Reference)
The technical team's earlier engineering specifications for full commercial deployment remain documented:
- **Option A (4,800 Keyword Takes):** 120 native Vietnamese speakers (50N / 30C / 40S). 40 keyword takes per contributor across speaking dynamics. Footprint: **~1.51 GB Primary Audio** (~2.57 GB itemized working footprint; ~2.26 GB consolidated $1.5\times$ allowance).
- **Option B (8,280 Takes — Keywords + Negatives + Tone):** 120 native speakers contributing 40 target, 25 UNKNOWN, and 4 SILENCE takes each (69 takes total). Footprint: **~2.60 GB Primary Audio** (~4.42 GB itemized working footprint; ~3.90 GB consolidated $1.5\times$ allowance).

**Mathematical Split Rationale:**  
Under a 70% Train / 15% Validation / 15% Test split, the pilot dataset partitions cleanly into 700 Train, 150 Validation, and 150 Test clips with strictly disjoint speaker sets and zero demographic or session overlap.  
*Caveat on Fine-Grained Strata:* While full-cohort arithmetic is clean, fine-grained demographic strata (Level 4: dialect × gender × age) with $\le 2$ speakers cannot mathematically populate 3 disjoint partitions without empty bins. No claim of demographic balance across partitions is made without empirical verification from split diagnostic logs.

---

## 3. Scope of Data Collected & PII Protection

To strictly comply with data privacy standards (including Vietnam Decree 13/2023/ND-CP and GDPR principles):
- **Collected Data:** Standardized 16 kHz mono 16-bit PCM audio, raw container recordings (`.wav`, `.m4a`), non-identifying acoustic metadata (regional dialect, gender, age bracket [18–30, 31–45, 46–60, 51+], device model, microphone category, distance category, room acoustic label).
- **Prohibited PII:** **Zero direct personal identifiers.** No participant names, national ID numbers (CCCD/CMND), telephone numbers, email addresses, physical addresses, or biometric voiceprints for speaker identification are collected or stored.
- **Pseudonymization Scheme:** Every participant receives a cryptographic nonce ID (`SPK_H_<N|C|S>_<6-hex-nonce>`, generated via `secrets.token_hex(3)`), completely eliminating sequential enrollment leakage.

---

## 4. Risk Analysis

1. **Privacy & Vocal Acoustics:** Pseudonymization detaches personal contact details from stored files, but acoustic speech cannot be considered fully anonymous if voice samples could theoretically be matched out-of-band.
2. **Consent Withdrawal & Model Retraining (`PARTIALLY VERIFIED — POLICY PROPOSED`):** Deep learning models cannot mathematically unlearn gradient updates from removed training data. If a contributor revokes consent post-training, the model must be completely retrained from scratch prior to commercial distribution. Tooling in `revoke_speaker_consent.py` quarantines audio and updates manifests, but manual cache purging (`.npy`) and full retraining execution require an approved operational SOP and compute budget commitment.
3. **External Dataset Licensing & Quarantine:** Google FLEURS (`vi_vn`) and MLCommons MSWC (`vi`) lack local text license files on disk (a documentation gap, rather than proof of invalid upstream licensing; conditional on multi-corpus usage under GATE-04). External source `husein_mic_noise` has unverified upstream licensing; while permanently blocked from ingestion by code (ACT-05A), on-disk archives (`mic-noise.zip` 45.9 MB and 39 MP3s) remain preserved in quarantine pending institutional legal disposition (ACT-05B).
4. **Option B Legal Boundary:** Collecting 69 takes per contributor reduces dependence on external speech corpora, but is **not legally risk-free**; the campaign retains statutory liabilities under Vietnam Decree 13/2023/ND-CP (biometric data rights, right of withdrawal), prompt text copyright compliance, and ongoing data custody obligations.
5. **Demographic Bias Risk:** Inability to recruit across all proposed strata could leave validation bins under-represented.

---

## 5. Technical Readiness vs. Governance Gaps

### What Has Been Technically Verified (Ready)
- **52/52 Acceptance Tests Passing:** Full pipeline verified with zero production pollution in `tests/test_campaign_acceptance.py`.
- **Fail-Closed Production Strict-Mode:** Ingestion rejects missing, fallback, or contradictory metadata; interactive registration collects microphone type and distance category.
- **Cryptographic Invariants Preserved:** `test_manifest_v2.csv` (`9E39...`), `field_collection_manifest.csv` (`B389...`), and `consent_ledger.json` (`BE4B...`) remain bit-exact.
- **Data Segregation:** Historical synthetic fixtures (`data/raw/field_recordings/`) and human storage (`data/raw/human_recordings_v1/`) are physically separated.
- **Revocation Tooling:** Automated CLI quarantines audio and updates manifest and ledger states safely.
*(Technical Baseline Note: Technical verification under GATE-07 validates software pipeline readiness, but does NOT grant governance authorization to collect human speech data).*

### What Lacks Evidence & Awaits Human Decision (Blocked)
- **Mandatory Pre-Recruitment Gates Open:** All five pre-recruitment gates (`GATE-01` Purpose, `GATE-02` Consent Form, `GATE-03` Quota/Budget/Option, `GATE-05` Retention/Retraining SOP, `GATE-06` Hardware Roster/Calibration) remain OPEN and must be formally executed in `human_governance_signoff_record.md` before participant recruitment or contact is permitted.
- **Independent Institutional Ethics / IRB Review:** If independently mandated by institutional policy, formal ethics committee approval must be secured alongside `GATE-01` and `GATE-02` prior to recruitment. (Additional governance requirement; does not alter canonical gates `GATE-01` to `GATE-08`; no ethics or legal approval has occurred).
- **No Participant Consent Records on Disk:** Zero human participants have been contacted or registered (`GATE-08` runtime invariant pending execution).
- **Conditional External Corpus Licensing (`GATE-04`):** Standalone CC-BY-4.0 license files are not archived locally on disk for FLEURS and MSWC (conditional on multi-corpus training or evaluation).
- **On-Disk Noise Disposition Unresolved:** Physical retention vs. deprecation of 45.9 MB unverified noise files on disk awaits institutional legal directive (ACT-05B).

---

## 6. Key Decisions Required from Authorizing Stakeholders

The authorizing officers must explicitly answer and resolve the following five questions:

| # | Decision Item | Key Stakeholder | Options |
| :-: | :--- | :--- | :--- |
| **Q1** | **Purpose & Scope:** Do you formally approve the collection and use of Vietnamese speech solely for offline keyword spotting ML research, evaluation, and on-device product deployment (including game control)? | Project Sponsor / PI & Legal Counsel | [ ] Approve<br>[ ] Reject |
| **Q2** | **Informed Consent Notice:** Does the draft Vietnamese consent agreement in [`collection_protocol.md#L41-L67`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67) satisfy ethical and statutory requirements? | Legal / Privacy Officer | [ ] Approve<br>[ ] Request Edits<br>[ ] Reject |
| **Q3** | **Cohort Scope & Quota Authorization:** Which target dataset scale do you formally authorize for recruitment? | Project Sponsor / PI | [ ] **1,000-Clip Pilot** (250/cmd, 25-40 speakers)<br>[ ] **Full Option A** (4,800 takes, 120 speakers)<br>[ ] **Full Option B** (8,280 takes, 120 speakers)<br>[ ] Reject |
| **Q4** | **Model Retraining Mandate:** Do you accept the governance policy requiring complete model retraining from scratch if a contributor withdraws consent prior to commercial release? | Project Sponsor / PI & Technical Lead | [ ] Approve<br>[ ] Reject |
| **Q5** | **External Corpus Attribution:** Do you approve the inclusion of Google FLEURS (`vi_vn`) and MSWC (`vi`) subject to CC-BY-4.0 attribution notice archiving? | Legal Counsel & Data Steward | [ ] Approve<br>[ ] Reject |

---

## 7. Executive Sign-Off Determination

*(To be filled exclusively by the Authorizing Decision-Maker. Do NOT assume or pre-fill values.)*

### Final Decision:
`[  ] APPROVED`  
`[  ] APPROVED WITH CONDITIONS`  
`[  ] REJECTED`  
`Current System Status: PENDING HUMAN DECISION`

### Mandatory Conditions Attached (if Approved with Conditions):
1. `PENDING`
2. `PENDING`

### Decision-Maker Signature & Attribution:
- **Decision Maker Name:** `PENDING`
- **Role / Title:** `PENDING`
- **Organization / Unit:** `PENDING`
- **Decision Date:** `PENDING`
- **Signature / Verifiable Approval Token:** `PENDING`
- **Link to Signed Authorization Artifact:** `PENDING`
