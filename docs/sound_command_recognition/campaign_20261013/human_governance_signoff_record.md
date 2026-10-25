# Official Human Governance Sign-Off Record

**Campaign Identification:** Vietnamese Keyword Spotting Human Speech Acquisition (`v1.0.0-human`)  
**Workspace:** `D:\trainidentificationaudio`  
**Record Version:** 1.0.0-BLANK-RECORD  
**Status:** **`UNEXECUTED BLANK RECORD — PENDING HUMAN SIGNATURES`**  
**Governance Notice:** This document is the formal sign-off record. It must NOT be pre-filled, stamped, or marked as approved by AI agents or automated scripts. Only authorized human decision-makers may complete and execute this document.

---

## 1. Governance Review Overview

| Parameter | Specification Submitted for Review | Human Decision / Approved Value |
| :--- | :--- | :--- |
| **Protocol Version** | `collection_protocol.md` (Version 1.0.0-PROPOSAL) | `[ PENDING REVIEW ]` |
| **Campaign Dates** | Proposed: October 2026 | `[ PENDING SPECIFICATION ]` |
| **Target Commands** | `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `SILENCE` | `[ PENDING SPECIFICATION ]` |
| **Target Participants** | Proposed: 120 contributors (50 Northern, 30 Central, 40 Southern) | `[ PENDING SPECIFICATION ]` |
| **Authorized Purpose** | Offline Keyword Spotting ML research, evaluation, and product deployment | `[ PENDING SPECIFICATION ]` |
| **Data Retention Period** | Proposed: 3 years from dataset release freeze | `[ PENDING SPECIFICATION ]` |

---

## 2. Permitted & Prohibited Data Sources Determination

The decision-makers specify and authorize the inclusion or exclusion of the following data sources:

| Source Identifier | Proposed Role in Campaign | Human Authorization Decision |
| :--- | :--- | :---: |
| **Genuine Human Speech** | Primary target command recordings (`TRAI`, `PHAI`, `LEN`, `XUONG`) | `[  ] ALLOWED    [  ] PROHIBITED` |
| **Participant Room Tone** | Consented ambient background noise (`SILENCE`) | `[  ] ALLOWED    [  ] PROHIBITED` |
| **Google FLEURS (`vi_vn`)** | Negative speech samples for class `UNKNOWN` (CC-BY-4.0) | `[  ] ALLOWED    [  ] PROHIBITED` |
| **MLCommons MSWC (`vi`)** | Phonetic confusers / negative words (CC-BY-4.0) | `[  ] ALLOWED    [  ] PROHIBITED` |
| **Husein `mic-noise`** | Unverified environmental noise dataset | **`STRICTLY PROHIBITED (BLOCKED IN SOFTWARE — ACT-05A; DISK STATUS: ACT-05B PENDING)`** |
| **Synthetic Fixtures** | Parametric tones, TTS, or mock signals | **`PROHIBITED FROM HUMAN MANIFESTS`** |

---

## 3. Mandatory Pre-Conditions Required Before Field Recording Commences (Canonical Gates)

*(The authorizing officers record specific conditions that must be fulfilled before recruitment begins)*

1. `[  ]` **GATE-01 (Overall Protocol Sign-off):** Executive endorsement by Project Sponsor / PI of purpose, scope, and go/no-go thresholds.
2. `[  ]` **GATE-02 (Informed Consent Legal Vetting):** Formal written legal sign-off on the Vietnamese informed consent disclosure text.
3. `[  ]` **GATE-03 (Cohort Quota Authorization):** Formally approved budget and recruitment target of 120 contributors (50N/30C/40S), acknowledging Level 4 integer rounding constraints.
4. `[  ]` **GATE-04 (Corpus License Archiving):** Placement of official CC-BY-4.0 `LICENSE.txt` files on disk for FLEURS and MSWC, and signed attribution notice.
5. `[  ]` **GATE-05 (Data Retention & Retraining Policy):** Operational SOP governing intermediate cache purging (`.npy`), backup isolation, and complete retraining upon consent withdrawal.
6. `[  ]` **GATE-06 (Hardware Roster & Calibration Guide):** Finalized physical smartphone serial inventory and microphone distance calibration operator manual.
7. `[  ]` **GATE-07 (Technical Strict-Mode Verification):** Confirmed fail-closed CLI/ingestion strict-mode (`effective_strict=True`) passing automated regression suite (Status: `SATISFIED`).
8. `[  ]` **GATE-08 (Per-Participant Runtime Consent Invariant):** Mandatory individual contributor registration in `consent_ledger_human.json` prior to any recording take (Status: `RUNTIME INVARIANT`).

---

## 4. Formal Decision Determination

Select exactly one decision:

- `[  ]` **APPROVED:** All protocol specifications, participant quotas, consent forms, and data governance policies are approved without modification. Field recruitment may proceed upon completion of mandatory pre-conditions.
- `[  ]` **APPROVED WITH CONDITIONS:** Approved subject to the mandatory pre-conditions recorded in Section 3 above. No human recording may commence until all checked conditions are verified.
- `[  ]` **REJECTED:** The protocol is rejected. Human recruitment and data recording remain strictly prohibited. Specify reasons below.

**Rejection Reasons / Mandatory Modification Notes (if applicable):**  
`____________________________________________________________________________________________________`  
`____________________________________________________________________________________________________`  
`____________________________________________________________________________________________________`

---

## 5. Authorizing Signatures & Verifiable Attribution

*(To be signed physically or with verifiable cryptographic/e-signature by designated stakeholders)*

### Project Sponsor / Principal Investigator (PI)
- **Full Legal Name:** `__________________________________________________`
- **Official Job Title:** `__________________________________________________`
- **Department / Organization:** `__________________________________________________`
- **Decision Date (YYYY-MM-DD):** `____________________`
- **Signature / Digital Verification Token:**  
  ```
  [ PLACE VERIFIABLE SIGNATURE / STAMP / GPG TOKEN HERE ]
  ```

---

### Privacy Officer / Legal Counsel (if applicable)
- **Full Legal Name:** `__________________________________________________`
- **Official Job Title:** `__________________________________________________`
- **Department / Organization:** `__________________________________________________`
- **Decision Date (YYYY-MM-DD):** `____________________`
- **Signature / Digital Verification Token:**  
  ```
  [ PLACE VERIFIABLE SIGNATURE / STAMP / GPG TOKEN HERE ]
  ```

---

### Data Protection Officer / Technical Custodian
- **Full Legal Name:** `__________________________________________________`
- **Official Job Title:** `__________________________________________________`
- **Department / Organization:** `__________________________________________________`
- **Decision Date (YYYY-MM-DD):** `____________________`
- **Signature / Digital Verification Token:**  
  ```
  [ PLACE VERIFIABLE SIGNATURE / STAMP / GPG TOKEN HERE ]
  ```

---

## 6. Official Evidence & Document Links

1. **Collection Protocol:** [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md)
2. **Technical Closure Report:** [`docs/sound_command_recognition/campaign_20261013/technical_closure_report.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/technical_closure_report.md)
3. **Governance Evidence Review:** [`docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md)
4. **Sign-Off Checklist:** [`docs/sound_command_recognition/campaign_20261013/human_governance_signoff_checklist.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_signoff_checklist.md)
5. **Decision Brief:** [`docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/human_governance_decision_brief.md)
6. **Corpus License Matrix:** [`docs/sound_command_recognition/campaign_20261013/corpus_license_evidence_matrix.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/corpus_license_evidence_matrix.md)
