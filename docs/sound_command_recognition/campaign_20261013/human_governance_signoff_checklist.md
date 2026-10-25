# Human Governance Sign-Off Checklist: Vietnamese KWS Campaign

**Workspace:** `D:\trainidentificationaudio`  
**Document Version:** 1.0.0-SIGNOFF-CHECKLIST  
**Target Ingestion Target:** Genuine Human Field Speech Collection (`v1.0.0-human`)  
**Scope:** Governance, Legal, Privacy, Ethical, and Pipeline Integrity Verification  
**Operational Status:** **`STRICTLY BLOCKED`** *(Collection prohibited until all pending items receive verifiable human sign-off)*

---

## 1. Overview & Verification Legend

This checklist establishes the formal human governance sign-off criteria for the Vietnamese Keyword Spotting (`TRAI`, `PHAI`, `LEN`, `XUONG`) data collection campaign. It serves as the primary instrument for the Project Sponsor/PI, Legal/Privacy Counsel, and Data Protection Officer to audit readiness before authorizing participant recruitment.

### Status Definitions
- **`VERIFIED`**: Technical implementation, cryptographic integrity, or pipeline logic is proven by on-disk code, automated tests, and immutable checksums.
- **`NOT VERIFIED`**: Necessary artifacts, legal agreements, or empirical evidence are missing or incomplete.
- **`PENDING HUMAN APPROVAL`**: Engineering proposals or draft specifications are fully prepared, but require explicit, documented human authorization from designated stakeholders.
- **`BLOCKED`**: The item or data source is strictly prohibited from production use due to legal, privacy, or governance non-compliance.

---

## 2. Comprehensive Governance Checklist

### Item 1: Phê duyệt mục đích thu thập và sử dụng giọng nói (Approval of Purpose)
- **Requirement:** Formal written authorization of the specific purpose: machine learning research, acoustic benchmarking, offline KWS deployment on mobile devices (Android/iOS), and commercial product development. Prohibits biometric speaker identification or unauthorized surveillance applications.
- **Current Status:** **`PENDING HUMAN APPROVAL`**
- **Evidence File / Path:** [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L46-L48`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L46-L48)
- **What Evidence Proves:** The engineering protocol explicitly articulates and restricts the intended use to offline KWS research and technology evaluation.
- **What Evidence DOES NOT Prove:** Management, corporate legal counsel, or an institutional ethics board (IRB) has legally authorized this scope.
- **Risk if Incomplete:** Regulatory sanctions under data privacy statutes (e.g., Vietnam Decree 13/2023/ND-CP, GDPR Art. 5/6), mission creep, and legal exposure from unauthorized commercial exploitation.
- **Authorizing Role:** Project Sponsor / Principal Investigator (PI) & Legal Counsel.
- **Closing Condition:** Signed authorization endorsing the defined purpose and forbidding unauthorized scope expansion.

---

### Item 2: Mẫu Informed Consent tiếng Việt và cơ chế rút consent (Informed Consent Form & Withdrawal)
- **Requirement:** Legally sound standard Informed Consent Form in Vietnamese explaining voluntary participation, acoustic data collection, pseudonymous tracking, commercial/research licensing, and the unconditional right to withdraw prior to final dataset freeze; paired with an operational revocation mechanism.
- **Current Status:** **`PENDING HUMAN APPROVAL`** *(Tooling is `VERIFIED`; Legal text is `PENDING HUMAN APPROVAL`)*
- **Evidence File / Path:** 
  - Draft Consent Text: [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L41-L67)
  - Revocation CLI: [`src/acquisition/revoke_speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/revoke_speaker_consent.py)
  - Revocation Test: [`tests/test_campaign_acceptance.py::test_p1_3_consent_revocation_workflow`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py#L1072)
- **What Evidence Proves:** The automated CLI successfully changes consent status to `REVOKED_WITHDRAWN`, quarantines raw and standardized audio, logs audit events, and survives partial file loss idempotently. A Vietnamese template draft exists in the protocol.
- **What Evidence DOES NOT Prove:** The Vietnamese text has been vetted by qualified legal counsel; no living human participant has signed or been presented this form.
- **Risk if Incomplete:** Unenforceable consent agreement, invalidation of gathered speech corpora, participant complaints, and legal liability.
- **Authorizing Role:** Legal / Privacy Officer & Project Sponsor / PI.
- **Closing Condition:** Formal legal sign-off on the Vietnamese consent disclosure text and verification of contributor withdrawal SOP.

---

### Item 3: Quota tuyển mẫu 50N/30C/40S (Cohort Sizing Quota)
- **Requirement:** Clear sign-off on the demographic cohort sizing: 50 Northern, 30 Central, 40 Southern speakers (total 120 participants), with explicit acknowledgement that this is a technical engineering proposal until authorized.
- **Current Status:** **`PENDING HUMAN APPROVAL`**
- **Evidence File / Path:** [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L105-L114`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L105-L114)
- **What Evidence Proves:** Mathematical analysis confirms that `50N / 30C / 40S` enables exact 70% / 15% / 15% speaker-disjoint splitting at the cohort level (84 train, 18 val, 18 test) with zero truncation drift, covering critical regional phonetics (Northern `/tɕ/` affricate burst vs. Southern `/ʈ/` retroflex).
- **What Evidence DOES NOT Prove:** Budget, field personnel, compensation/honorarium, or operational feasibility have been approved by project leadership. Crucially, cohort arithmetic does **not guarantee demographic balance in fine-grained strata** (Level 4: `regional × gender × age`); small strata ($\le 2$ speakers, such as Central males aged 51+) cannot populate all 3 partitions without empty bins. No balance claim may be made without empirical verification from split diagnostic logs.
- **Risk if Incomplete:** Resource overruns, incomplete recruitment leading to distorted evaluation splits, or unmeasured regional/demographic bias.
- **Authorizing Role:** Project Sponsor / PI.
- **Closing Condition:** Written executive approval of the 120-participant target and dialect ratio, with documented acknowledgement of Level 4 integer rounding constraints.

---

### Item 4: Quy trình tuyển người và tiêu chí tham gia (Recruitment & Eligibility Screening)
- **Requirement:** Screening protocol ensuring participants are native Vietnamese speakers of the targeted dialect, $\ge 18$ years of age, possessing adequate vocal health (no severe speech impediments or acute respiratory conditions), recruited voluntarily without workplace or academic coercion.
- **Current Status:** **`PENDING HUMAN APPROVAL`**
- **Evidence File / Path:** [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L28-L37`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L28-L37)
- **What Evidence Proves:** Technical inclusion/exclusion criteria are explicitly documented.
- **What Evidence DOES NOT Prove:** Screening personnel have been trained, screener checklists exist, or recruitment channels have been ethically vetted.
- **Risk if Incomplete:** Inadvertent enrollment of minors ($<18$), dialect contamination from non-native speakers, or coercive enrollment violating research ethics.
- **Authorizing Role:** Project Sponsor / PI & Ethics / Privacy Reviewer.
- **Closing Condition:** Approval of field screening SOP, age-verification method, and voluntary recruitment channels.

---

### Item 5: Thời hạn lưu trữ, quyền truy cập, backup và cache (Retention, Access & Cache Purging)
- **Requirement:** Documented data retention period (e.g., 3 years or project lifetime), restricted role-based access control (RBAC), immutable audit logging, and explicit operational procedures to purge derived feature caches (`.npy`, spectrograms) and cold backups (`data/backups/`) when consent is revoked.
- **Current Status:** **`PENDING HUMAN APPROVAL`** *(Tooling is `VERIFIED`; Operational policy is `PENDING HUMAN APPROVAL`)*
- **Evidence File / Path:** 
  - Boundary Matrix: [`docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md#L151-L165`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md#L151-L165)
  - Automated Tool: [`src/acquisition/revoke_speaker_consent.py`](file:///d:/trainidentificationaudio/src/acquisition/revoke_speaker_consent.py)
- **What Evidence Proves:** The automated tool isolates raw/standardized audio and updates manifest/ledger. The forensic report documents that derived feature caches and cold backups are NOT touched by the CLI to preserve backup immutability.
- **What Evidence DOES NOT Prove:** An operational SOP exists for administrators to manually flush feature caches or handle long-term backup destruction upon retention expiry.
- **Risk if Incomplete:** Unconsented speech features persist in intermediate training caches; cold backups retain revoked data indefinitely in violation of privacy regulations.
- **Authorizing Role:** Data Protection Officer (DPO) & Technical Lead.
- **Closing Condition:** Approved Data Retention Policy, Access Control List, and Operator SOP for manual cache purging upon consent revocation.

---

### Item 6: Chính sách sử dụng dữ liệu trong huấn luyện và phát hành model (Model Training & Weight Governance)
- **Requirement:** Policy recognizing that neural network model weights (`.pt`, `.onnx`) cannot mathematically unlearn voice traits via gradient descent; mandating that if consent is revoked prior to release, the model must be retrained from scratch without the withdrawn data before commercial release.
- **Current Status:** **`PENDING HUMAN APPROVAL`**
- **Evidence File / Path:** [`docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md#L164-L165`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/governance_evidence_review.md#L164-L165)
- **What Evidence Proves:** The technical limitation of deep neural network weights is clearly acknowledged in the engineering governance documentation.
- **What Evidence DOES NOT Prove:** Organizational commitment has been made to budget for retraining if a revocation request occurs after model training.
- **Risk if Incomplete:** Intellectual property contamination of commercial models; breach of consent terms if models derived from withdrawn voices are deployed.
- **Authorizing Role:** Project Sponsor / PI & Legal Counsel.
- **Closing Condition:** Binding policy document requiring full retraining from scratch if consent is withdrawn prior to public or commercial model deployment.

---

### Item 7: Provenance và giấy phép FLEURS, MSWC và nguồn nhiễu (External Corpus Licensing)
- **Requirement:** Audited legal verification and attribution compliance for Google FLEURS (`vi_vn`) and MLCommons MSWC (`vi`), ensuring local preservation of official CC-BY-4.0 license text and commercial compatibility.
- **Current Status:** **`NOT VERIFIED`** *(Data artifacts verified on disk; License text files missing on disk)*
- **Evidence File / Path:** 
  - FLEURS Artifacts: `data/raw/fleurs_vi/` (`dev.tsv`, `test.tsv`, `dev.tar.gz`, `test.tar.gz`)
  - MSWC Artifacts: `data/raw/mswc_vi/` (`splits.tar.gz`, `splits_extracted/version.txt`)
  - Corpus Matrix: [`docs/sound_command_recognition/campaign_20261013/corpus_license_evidence_matrix.md`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/corpus_license_evidence_matrix.md)
- **What Evidence Proves:** Files exist on disk with exact SHA-256 hashes; source code asserts CC-BY-4.0.
- **What Evidence DOES NOT Prove:** Official `LICENSE` text files are stored locally on disk; formal legal review of commercial model training terms has been documented.
- **Risk if Incomplete:** Breach of open-source attribution requirements (CC-BY-4.0 Section 3), potential copyright challenges from upstream dataset creators.
- **Authorizing Role:** Legal Counsel & Data Steward.
- **Closing Condition:** Local placement of official CC-BY-4.0 license files in dataset directories and legal sign-off on attribution notices.

---

### Item 8: Xử lý nguồn dữ liệu bị chặn (Quarantine of Unverified External Noise — ACT-05A & ACT-05B)
- **Requirement:** Technical and operational prohibition of external sources lacking clear upstream licensing, specifically `husein_mic_noise`, coupled with legal disposition of on-disk archives.
- **Current Status:** **`VERIFIED` (Software Ingestion Blocked / Storage Retention Pending Legal Disposition)**
- **Evidence File / Path:** 
  - Downloader Gate (ACT-05A): [`src/acquisition/download_approved_sources.py#L38-L52`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L38-L52)
  - Validator Gate (ACT-05A): [`src/acquisition/campaign_validator.py#L39-L44`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L39-L44)
  - Gate Test (ACT-05A): [`tests/test_campaign_acceptance.py::test_p0_2_source_license_gate`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py#L1040)
  - Storage On Disk (ACT-05B): `data/raw/noise/mic-noise.zip` (45.9 MB) and 39 extracted MP3 files.
- **What Evidence Proves:** The acquisition script and manifest validator actively throw errors if `husein_mic_noise` is referenced; files in `data/raw/noise/` cannot enter the human speech pipeline (ACT-05A verified).
- **What Evidence DOES NOT Prove:** Upstream copyright permissions have not been verified; institutional legal disposition regarding the on-disk storage of the 45.9 MB archive and 39 MP3 files is pending counsel review (ACT-05B).
- **Filesystem Stability Mandate:** Files `data/raw/noise/mic-noise.zip` and the 39 extracted MP3s must **NOT be modified or deleted during this phase**, preserving filesystem stability and audit traceability until formal legal disposition is executed.
- **Risk if Incomplete:** Technical pipeline is secure. Corporate compliance risk if unverified files are retained indefinitely without institutional legal authorization.
- **Authorizing Role:** Technical Lead (Pipeline Control) & Legal Counsel (Data Custody).
- **Closing Condition:** Maintain strict software block (ACT-05A); obtain formal legal memo determining permanent quarantine retention or executing an authorized data purge procedure (ACT-05B).

---

### Item 9: Production Strict-Mode & Kiểm soát Metadata (Strict Metadata Pipeline)
- **Requirement:** Fail-closed CLI and ingestion pipeline that prevents any human audio from being ingested with missing, fallback, or contradictory demographic/acoustic metadata; interactive registration prompts for microphone and distance.
- **Current Status:** **`VERIFIED`**
- **Evidence File / Path:** 
  - CLI Implementation: [`src/acquisition/record_command_cli.py`](file:///d:/trainidentificationaudio/src/acquisition/record_command_cli.py)
  - Ingestion Implementation: [`src/acquisition/ingest_field_recordings.py`](file:///d:/trainidentificationaudio/src/acquisition/ingest_field_recordings.py)
  - Regression Tests: [`tests/test_campaign_acceptance.py::test_fail_closed_production_strict_metadata`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py#L1325), `test_registration_flow_persists_mic_and_distance`, `test_recording_session_strict_fail_closed`
- **What Evidence Proves:** Live sessions (`mock_mode=False`) and human file ingestion automatically force `strict_metadata=True`. Incomplete metadata immediately quarantines audio. Interactive registration prompts for and stores `microphone_type` and `distance_category`. All 44 tests pass.
- **What Evidence DOES NOT Prove:** Field operators will execute recordings in quiet physical spaces without acoustic reverberation.
- **Risk if Incomplete:** Negligible (technical controls are active and verified).
- **Authorizing Role:** Technical Lead.
- **Closing Condition:** Technical verification complete. Include operational instructions in the Field Operator Guide.

---

### Item 10: Phân vùng Dữ liệu Disjoint & Cách ly Lineage (Speaker & Lineage Disjointness)
- **Requirement:** Guaranteed mathematical separation between Train, Validation, and Test sets across speaker IDs, recording sessions, audio take files, and augmentation lineage parents.
- **Current Status:** **`VERIFIED`** *(Algorithmic disjointness is `VERIFIED`; Fine-grained demographic balance is `STRUCTURALLY CONSTRAINED`)*
- **Evidence File / Path:** 
  - Validator: [`src/acquisition/campaign_validator.py::validate_speaker_disjointness`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L320)
  - Split Generator: [`src/acquisition/campaign_validator.py::generate_stratified_disjoint_splits`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L390)
  - Regression Test: [`tests/test_campaign_acceptance.py::test_recording_and_lineage_disjointness`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py#L1236)
- **What Evidence Proves:** The validator detects and flags any cross-partition speaker, recording ID, or parent augmentation lineage overlap. Partitioning is strictly disjoint.
- **What Evidence DOES NOT Prove:** Perfect demographic balance in validation/test splits when total speakers in a stratum are small ($\le 2$ speakers, such as Central males aged 51+, which cannot populate 3 disjoint partitions without at least one empty bin). Demographic balance must never be claimed without empirical diagnostic verification.
- **Risk if Incomplete:** Overoptimistic benchmark evaluations due to acoustic data leakage across splits.
- **Authorizing Role:** Machine Learning Lead / Technical Lead.
- **Closing Condition:** Automated split validator integrated into the final release pipeline; diagnostic logs reviewed for small-stratum coverage.

---

### Item 11: Bảo toàn ba tệp được bảo vệ bằng SHA-256 (Protected Artifact Preservation)
- **Requirement:** Cryptographic immutability of the frozen benchmark test manifest `test_manifest_v2.csv` (`9E39...`), historical fixture manifest `field_collection_manifest.csv` (`B389...`), and historical fixture consent ledger `consent_ledger.json` (`BE4B...`).
- **Current Status:** **`VERIFIED`**
- **Evidence File / Path:** 
  - Benchmark Manifest: `data/manifests/test_manifest_v2.csv`
  - Historical Manifest: `data/raw/field_recordings/field_collection_manifest.csv`
  - Historical Ledger: `data/raw/field_recordings/consent_ledger.json`
- **What Evidence Proves:** Bit-exact SHA-256 match confirmed across all sessions. Zero bytes changed. Test suite executes with 100% temporary sandbox isolation (`tmp_path`).
- **What Evidence DOES NOT Prove:** Future unmonitored git operations cannot accidentally alter them without continuous verification hooks.
- **Risk if Incomplete:** Invalidation of previous scientific research, audit non-compliance, loss of frozen evaluation standard.
- **Authorizing Role:** Technical Lead / Repository Custodian.
- **Closing Condition:** Verified bit-exact match; pre-commit hash verification mandated.

---

### Item 12: Điều kiện Bắt đầu Thu thập & Tiêu chí Dừng Chiến dịch (Go/No-Go & Stop Triggers)
- **Requirement:** Documented gating conditions required before commencing participant recording (Go/No-Go), coupled with predefined early termination triggers (e.g., audio rejection rate $>20\%$, consent withdrawal rate $>5\%$, regional recruitment shortfall).
- **Current Status:** **`PENDING HUMAN APPROVAL`**
- **Evidence File / Path:** [`docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L188-L215`](file:///d:/trainidentificationaudio/docs/sound_command_recognition/campaign_20261013/collection_protocol.md#L188-L215)
- **What Evidence Proves:** Objective technical threshold criteria are clearly defined in the collection protocol specification.
- **What Evidence DOES NOT Prove:** Leadership and project operations have formally signed off on escalation pathways or stop triggers.
- **Risk if Incomplete:** Prolonged collection of unusable low-quality audio, unmanaged participant attrition, budget overruns.
- **Authorizing Role:** Project Sponsor / PI & Quality Assurance Lead.
- **Closing Condition:** Formal executive sign-off adopting the Go/No-Go gates and campaign termination thresholds.

---

## 3. Governance Summary Scorecard

| # | Governance Domain | Current Status | Authorizing Stakeholder | Sign-off Blocker? |
| :---: | :--- | :---: | :--- | :---: |
| 1 | Approval of Purpose & Scope | **`PENDING HUMAN APPROVAL`** | Project Sponsor / PI & Legal Counsel | **YES** |
| 2 | Informed Consent Text & Withdrawal | **`PENDING HUMAN APPROVAL`** | Legal / Privacy Officer & Project Sponsor | **YES** |
| 3 | Cohort Sizing Quota (50N/30C/40S) | **`PENDING HUMAN APPROVAL`** | Project Sponsor / PI | **YES** |
| 4 | Recruitment & Eligibility Screening | **`PENDING HUMAN APPROVAL`** | Project Sponsor / PI & Ethics Reviewer | **YES** |
| 5 | Data Retention, Access & Backup SOP | **`PENDING HUMAN APPROVAL`** | Data Protection Officer & Technical Lead | **YES** |
| 6 | Neural Network Weight Governance | **`PENDING HUMAN APPROVAL`** | Project Sponsor / PI & Legal Counsel | **YES** |
| 7 | External Corpus Licensing & Attribution | **`NOT VERIFIED`** | Legal Counsel & Data Steward | **YES** |
| 8 | Blocked Unverified Sources (`mic-noise`) | **`VERIFIED` (Software Gate Active / ACT-05B Legal Disposition Pending)** | Technical Lead & Legal Counsel | NO (Software Gate) / YES (Storage Review) |
| 9 | Production Strict-Mode Pipeline | **`VERIFIED`** | Technical Lead | NO |
| 10 | Disjoint Partitioning & Lineage Gate | **`VERIFIED` (Disjointness Proven; Demographic Balance Constrained)** | Machine Learning Lead | NO |
| 11 | Cryptographic Invariant Preservation | **`VERIFIED`** | Repository Custodian | NO |
| 12 | Go/No-Go Gates & Stop Triggers | **`PENDING HUMAN APPROVAL`** | Project Sponsor / PI & QA Lead | **YES** |

**Conclusion:** Items 8 (ACT-05A software gate), 9, 10, and 11 are technically verified and sealed. Items 1, 2, 3, 4, 5, 6, 7, 8 (ACT-05B on-disk file disposition), and 12 require documented human action before the campaign operational status can transition from `STRICTLY BLOCKED` to `ACTIVE`.
