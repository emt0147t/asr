# External Corpus License & Provenance Evidence Matrix

**Workspace:** `D:\trainidentificationaudio`  
**Document Version:** 1.0.0-LEGAL-MATRIX  
**Date of Audit:** 2026-10-14  
**Audit Standard:** Primary Forensic Verification (Filesystem Reality vs. Code Claims)  
**Governing Rule:** Code comments, variable strings, and repository folder names do NOT constitute legal license evidence. External sources lacking verified on-disk or documented legal grants are classified as `NOT VERIFIED` or `BLOCKED`.

---

## 1. Corpus License & Provenance Summary Table

| # | Corpus / Source | Scope / Version | Official Source URL | License Claimed | On-Disk License Text | Local Artifact SHA-256 Digest | Status & Decision |
| :-: | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **1** | **Google FLEURS (`vi_vn`)** | Vietnamese read speech (dev & test splits; 1,218 clips) | [HuggingFace Google FLEURS](https://huggingface.co/datasets/google/fleurs) | CC-BY-4.0 | **MISSING** (No `LICENSE` file on disk) | `dev.tsv` (247 KB): `C9BC17CEDE9765B1C75CB7A608F7066CB6BB6F8245CC75D721A5217A5DFFB414`<br>`test.tsv` (613 KB): `EB744C2BE5F677B49C35527D428A4F647294BD0F67ABD790C296E91EFAAFFB93`<br>`dev.tar.gz` (214.5 MB): `8821A394C99069409B3CE7BB5CD14B10B709F2EE2FEA16F42B4701E1CC9EF673`<br>`test.tar.gz` (544.1 MB): `B359261216EF14B8EEA159ABCE794E270C2A0BE903218CF1E1E8F7C7C1565B5C` | **`NOT VERIFIED`** *(Artifacts present on disk; License text missing on disk; Pending legal confirmation)* |
| **2** | **MLCommons MSWC (`vi`)** | Multilingual Spoken Words Corpus (Vietnamese keyword splits) | [MLCommons Spoken Words](https://mlcommons.org/datasets/spoken-words/) / [HuggingFace MSWC](https://huggingface.co/datasets/MLCommons/ml_spoken_words) | CC-BY-4.0 | **MISSING** (`version.txt` exists, but no `LICENSE` file) | `splits.tar.gz` (4.7 KB): `964F485ACE314657B0F3C5F1BA7E97EF1814D0E1379DBEB0BD26E4526A9B8C00`<br>`splits_extracted/version.txt` (97 B): `42F75458F61C04A7832DCBE77932A6F2C72BB3634E881F6BAD0D67CDBE5DE824` | **`NOT VERIFIED`** *(Metadata split present on disk; License text missing on disk; Pending legal confirmation)* |
| **3** | **Husein `mic-noise`** | Environmental microphone noise audio files (39 MP3s) | [HuggingFace huseinzol05/noise-dataset](https://huggingface.co/datasets/huseinzol05/noise-dataset) | Claimed CC-BY-4.0 *(in code comments)* | **ABSENT** (Terms of use unverified) | `mic-noise.zip` (45.9 MB): `2757A2E1CE8AA9B2DA0718323A51137B08A70C2D0378D00E5A014D72FC26BD6B` | **`BLOCKED`** *(Strictly blocked by code; prohibited from production manifests and training)* |
| **4** | **Calibration Confusing Words** | Synthetic tone negation files (`neg_chuông_...`) | Programmatically generated calibration audio | Internal Research Fixture | **N/A** (Synthetic fixture) | Individual 16-bit WAV calibration files in `data/raw/reference_confusing_words/` | **`NOT VERIFIED FOR HUMAN USE`** *(Restricted exclusively to automated calibration)* |
| **5** | **Genuine Ambient Room Tone** | Participant physical room noise (`sub_category: background_room_tone`) | Live participant microphone capture | Informed Consent Agreement (`v1.0-kws-research-commercial`) | Defined in `collection_protocol.md` | To be captured post-signoff | **`PENDING HUMAN APPROVAL`** *(Awaiting campaign sign-off)* |

---

## 2. Detailed Forensic Analysis by Corpus

### 2.1 Google FLEURS (`vi_vn` Subset)
- **Exact Corpus Name & Version:** Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech), Vietnamese (`vi_vn`) partition, v1.0.
- **Official Source URL:** `https://huggingface.co/datasets/google/fleurs`
- **Claimed License:** Creative Commons Attribution 4.0 International (CC-BY-4.0).
- **Actual Filesystem Evidence:**
  - Files present under `data/raw/fleurs_vi/`: `dev.tsv`, `test.tsv`, `dev.tar.gz`, `test.tar.gz`.
  - Standalone `LICENSE` file: **MISSING on disk**. No text file detailing the full legal grant was archived in the directory.
- **Verification Date:** 2026-10-14 (Filesystem inspection and cryptographic hashing).
- **Local Artifact Hashes:**
  - `dev.tsv`: `C9BC17CEDE9765B1C75CB7A608F7066CB6BB6F8245CC75D721A5217A5DFFB414` (247,001 bytes)
  - `test.tsv`: `EB744C2BE5F677B49C35527D428A4F647294BD0F67ABD790C296E91EFAAFFB93` (613,728 bytes)
  - `dev.tar.gz`: `8821A394C99069409B3CE7BB5CD14B10B709F2EE2FEA16F42B4701E1CC9EF673` (214,500,592 bytes)
  - `test.tar.gz`: `B359261216EF14B8EEA159ABCE794E270C2A0BE903218CF1E1E8F7C7C1565B5C` (544,139,865 bytes)
- **Permitted Scope under CC-BY-4.0:** Reproduction, distribution, public display, adaptation, and commercial use.
- **Mandatory Obligations & Restrictions:**
  - **Attribution:** Must cite the authors (Google Research, Conneau et al., 2022).
  - **License Notice:** Must retain a link or copy of the CC-BY-4.0 license.
  - **Notice of Changes:** Must indicate if audio was resampled, downmixed, or sliced.
- **Governance Audit Finding:**
  The dataset is well-established in the speech community under CC-BY-4.0. However, because no official `LICENSE.txt` file was preserved in `data/raw/fleurs_vi/`, the status remains **`NOT VERIFIED`** from an evidentiary standpoint.
- **Remediation Required:**
  Place the official CC-BY-4.0 text file in `data/raw/fleurs_vi/LICENSE.txt` and have legal counsel confirm compliance with attribution notices.

---

### 2.2 MLCommons Multilingual Spoken Words Corpus (MSWC `vi`)
- **Exact Corpus Name & Version:** MLCommons Multilingual Spoken Words Corpus (MSWC), Vietnamese (`vi`) subset, version 1.0.
- **Official Source URL:** `https://mlcommons.org/datasets/spoken-words/` / `https://huggingface.co/datasets/MLCommons/ml_spoken_words`
- **Claimed License:** Creative Commons Attribution 4.0 International (CC-BY-4.0).
- **Actual Filesystem Evidence:**
  - Archive `splits.tar.gz` and extracted directory `splits_extracted/` present in `data/raw/mswc_vi/`.
  - File `version.txt` contains: `"version 1.0, Multilingual Spoken Words Corpus"`.
  - Standalone `LICENSE` file: **MISSING on disk**.
- **Verification Date:** 2026-10-14 (Filesystem inspection and cryptographic hashing).
- **Local Artifact Hashes:**
  - `splits.tar.gz`: `964F485ACE314657B0F3C5F1BA7E97EF1814D0E1379DBEB0BD26E4526A9B8C00` (4,780 bytes)
  - `splits_extracted/version.txt`: `42F75458F61C04A7832DCBE77932A6F2C72BB3634E881F6BAD0D67CDBE5DE824` (97 bytes)
- **Permitted Scope under CC-BY-4.0:** Copying, adapting, and distributing for research and commercial machine learning.
- **Mandatory Obligations & Restrictions:**
  - Must provide attribution to Harvard University, MLCommons, and the Mozilla Common Voice contributors from which MSWC derives.
  - Must indicate modifications.
- **Governance Audit Finding:**
  Status remains **`NOT VERIFIED`** due to the absence of a local `LICENSE.txt` file and lack of documented formal legal review.
- **Remediation Required:**
  Place `data/raw/mswc_vi/LICENSE.txt` on disk and obtain legal confirmation of attribution text.

---

### 2.3 Husein Zolkepli `mic-noise`
- **Exact Corpus Name & Version:** `mic-noise` from GitHub repository `huseinzol05/malay-dataset` / HuggingFace `huseinzol05/noise-dataset`.
- **Official Source URL:** `https://huggingface.co/datasets/huseinzol05/noise-dataset`
- **Claimed License:** CC-BY-4.0 claimed in informal code comments.
- **Actual Filesystem Evidence:**
  - Archive `mic-noise.zip` (45,906,002 bytes) in `data/raw/noise/`.
  - 39 extracted MP3 files in `data/raw/noise/extracted/noise/mic-noise/` (e.g., `aXLsound - lo fi mic handling noise...`).
  - Standalone `LICENSE` file: **ABSENT**.
  - Provenance analysis reveals sample names matching commercial third-party sound libraries (`aXLsound`, `EFX EXT`), indicating potential uncredited scraping without upstream copyright grants.
- **Verification Date:** 2026-10-14.
- **Local Artifact Hash:**
  - `mic-noise.zip`: `2757A2E1CE8AA9B2DA0718323A51137B08A70C2D0378D00E5A014D72FC26BD6B`
- **Permitted Scope:** **UNVERIFIED / UNKNOWN**. Severe copyright contamination risk.
- **Governance Audit Finding:**
  Classified as **`BLOCKED`**. The acquisition pipeline ([`download_approved_sources.py#L42-L50`](file:///d:/trainidentificationaudio/src/acquisition/download_approved_sources.py#L42-L50)) and dataset validator ([`campaign_validator.py#L39-L44`](file:///d:/trainidentificationaudio/src/acquisition/campaign_validator.py#L39-L44)) explicitly throw exceptions if this source is ingested (**ACT-05A: Technical Control Verified**).
- **Remediation & On-Disk Storage Disposition:**
  Do NOT unblock. Prohibit all usage in training manifests. Rely exclusively on participant-captured ambient room tone. Regarding the on-disk archive `data/raw/noise/mic-noise.zip` (45.9 MB) and 39 extracted MP3 files, their long-term institutional disposition requires human legal counsel determination (**ACT-05B**). In accordance with the filesystem stability mandate, these physical files must **NOT be modified or deleted during this phase**.

---

### 2.4 Internal Calibration Audio (`reference_confusing_words`)
- **Exact Asset Name:** Calibration Negative Phonetic Confusers (`neg_chuông_...`).
- **Official Source:** Synthetically generated parametric tones created for test-suite verification.
- **Filesystem Evidence:** 10 WAV files in `data/raw/reference_confusing_words/`.
- **Status:** **`NOT VERIFIED FOR HUMAN USE`**. Permanently categorized as `SYNTHETIC_TEST_FIXTURE`.
- **Policy Rule:** Under the Data Provenance Policy, these files MUST NEVER be merged into human manifests or cited as human validation data.

---

### 2.5 Participant Ambient Room Tone (Proposed Replacement Noise)
- **Scope & Role:** Acoustic room tone and ambient silence for the `SILENCE` class (Class 5).
- **Source:** Captured directly from physical microphones during approved human recording sessions.
- **License Basis:** Covered under the participant's Informed Consent Agreement (`v1.0-kws-research-commercial`).
- **Status:** **`PENDING HUMAN APPROVAL`** (Awaiting formal campaign launch).
- **Governance Advantage:** Eliminates external copyright risk; perfectly matches the acoustic noise profile of target mobile hardware.

---

## 3. Mandatory Governance Directives for External Corpora

1. **Strict Prohibition Against Data Downloads:** No automated scripts or team members may download additional unapproved external audio sets during this phase.
2. **Mandatory License Archiving:** Prior to any training execution, official `LICENSE.txt` files for FLEURS and MSWC must be placed in their respective raw directories.
3. **Attribution File in Model Release:** Any model checkpoints (`.onnx`, `.pt`) trained using FLEURS or MSWC data must ship with an accompanying `ATTRIBUTION.md` detailing the CC-BY-4.0 notices.
4. **Maintenance of Technical Blocks & Preservation of On-Disk Files:** The algorithmic gates blocking `husein_mic_noise` in `download_approved_sources.py` and `campaign_validator.py` must remain permanently active (ACT-05A). On-disk archives must not be touched or deleted without formal legal authorization (ACT-05B).
