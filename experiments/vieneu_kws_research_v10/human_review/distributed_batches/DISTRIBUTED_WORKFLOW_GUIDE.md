# Operational Guide: Distributed Blinded Human-Listening Workflow

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Root:** `experiments/vieneu_kws_research_v10/human_review/distributed_batches/`  
**Review Model:** `variant_w10` (Frozen)  
**Governance Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Date:** 2026-10-18  

---

## 1. System Architecture & Blinded Package Inventory

The 120-clip synthetic Vietnamese KWS validation set has been partitioned into 6 disjoint, non-overlapping batches of 20 unique clips each using reproducible stratified allocation (seed `20261018`):

| Package Name | Contents | Assigned Unique Clips | Target Commands | Non-Targets |
| :--- | :--- | :---: | :---: | :---: |
| `reviewer_01_package.zip` | 20 audio WAVs, instructions, response template | 20 | 7 | 13 |
| `reviewer_02_package.zip` | 20 audio WAVs, instructions, response template | 20 | 7 | 13 |
| `reviewer_03_package.zip` | 20 audio WAVs, instructions, response template | 20 | 7 | 13 |
| `reviewer_04_package.zip` | 20 audio WAVs, instructions, response template | 20 | 7 | 13 |
| `reviewer_05_package.zip` | 20 audio WAVs, instructions, response template | 20 | 6 | 14 |
| `reviewer_06_package.zip` | 20 audio WAVs, instructions, response template | 20 | 6 | 14 |
| **Total Campaign** | **6 Independent Reviewer Packages** | **120** | **40** | **80** |

### Package Internal Structure
Each ZIP package contains strictly blinded, unlabelled assets:
```text
reviewer_XX/
├── audio/
│   ├── clip_01.wav
│   ├── clip_02.wav
│   └── ... clip_20.wav
├── reviewer_instructions.md
└── response_template.csv
```
- Audio files use strictly neutral filenames (`clip_01.wav` to `clip_20.wav`).
- Zero ground truth labels, model predictions, canonical classes, confusion family tags, or voice identities are included in reviewer packages.
- Byte-level SHA-256 integrity is verified ($100\%$ byte parity with original audio).

---

## 2. Secure Cloud Sharing via Google Drive / OneDrive

Because remote reviewers cannot access a local development workstation server (`127.0.0.1:8080`), packages must be distributed via authenticated cloud storage.

### Security Protocol:
1. **Isolated Reviewer Folders:**
   In Google Drive or Microsoft OneDrive, create 6 separate reviewer folders:
   - `Vietnamese_KWS_Review_R01/`
   - `Vietnamese_KWS_Review_R02/`
   - ... through `Vietnamese_KWS_Review_R06/`
2. **Upload Respective Packages Only:**
   Upload `reviewer_01_package.zip` to folder `Vietnamese_KWS_Review_R01/`, `reviewer_02_package.zip` to `Vietnamese_KWS_Review_R02/`, etc.
3. **Restricted Access Permissions:**
   - Set access strictly to **"Restricted / Specific people only"** using the reviewer's authenticated institutional/personal email address.
   - Do **NOT** use public "Anyone with link can view/edit" access.
4. **Quarantine Confidential Manifest:**
   - **NEVER** upload [`organizer_assignment_map.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/distributed_batches/organizer_assignment_map.csv) or [`review_manifest.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/review_manifest.csv) to any cloud folder accessible to reviewers.
   - The assignment map remains strictly on the organizer's local workstation.

---

## 3. Reviewer Workflow Guide

Send the following instructions to each reviewer along with their secure cloud link:

1. **Download & Extract:** Download your assigned ZIP package (e.g. `reviewer_01_package.zip`) and extract it locally.
2. **Review Setup:**
   - Read `reviewer_instructions.md` to familiarize yourself with the 6-point evaluation rubric.
   - Use wired or wireless over-ear / in-ear headphones in a quiet indoor room.
3. **Listen & Rate:**
   - Open `response_template.csv` in Excel, LibreOffice Calc, or any text editor.
   - Listen to `audio/clip_01.wav` through `audio/clip_20.wav`. Replay each clip as many times as needed.
   - Enter your ratings across the columns:
     1. `perceived_text`: Exact Vietnamese words with diacritics (or `[SILENCE]` / `[UNINTELLIGIBLE]`).
     2. `intelligibility_1to5`: Rating 1 to 5.
     3. `naturalness_1to5`: Rating 1 to 5.
     4. `pronunciation_correct_yn`: `Y`, `N`, or `Unsure`.
     5. `confidence_1to5`: Rating 1 to 5.
     6. `comments`: Optional notes on acoustic artifacts or swallowed consonants.
4. **Save & Return:**
   - Save the file as `reviewer_01_responses.csv` (UTF-8 encoding).
   - Upload the completed CSV back to your designated Google Drive / OneDrive folder or email it directly to the study organizer.

---

## 4. Response Validation Protocol

When a reviewer returns their completed CSV:

### Step 1: Place in Raw Staging Directory
Save the raw response file into:
`experiments/vieneu_kws_research_v10/human_review/distributed_batches/returned_responses/`  
*(e.g., `returned_responses/reviewer_01_responses.csv`)*

### Step 2: Validate Submission with `response_validation.py`
Run the automated validator using the `--batch` flag:
```powershell
.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/response_validation.py `
    experiments/vieneu_kws_research_v10/human_review/distributed_batches/returned_responses/reviewer_01_responses.csv `
    --batch
```

### Expected Output for Valid Submission:
```json
{
  "file": ".../reviewer_01_responses.csv",
  "is_complete": true,
  "valid_rows": 20,
  "invalid_rows": 0,
  "missing_clips": 0,
  "errors": []
}
```
If `is_complete` is `false`, review the errors list and request corrections from the reviewer.

---

## 5. Combining Responses without Overwriting Originals

Once all 6 reviewer submissions are validated, the study organizer runs `combine_distributed_responses.py`:

```powershell
.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/distributed_batches/combine_distributed_responses.py `
    --responses_dir experiments/vieneu_kws_research_v10/human_review/distributed_batches/returned_responses/ `
    --output_csv experiments/vieneu_kws_research_v10/human_review/responses/master_distributed_responses.csv
```

### Combination & Remapping Mechanics:
1. Reads all 6 files (`reviewer_01_responses.csv` through `reviewer_06_responses.csv`).
2. Validates each file for completeness (exactly 20 rows, valid ranges).
3. Joins with [`organizer_assignment_map.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/distributed_batches/organizer_assignment_map.csv) using `(reviewer_id, package_clip_id)` as the composite key.
4. Remaps local IDs (`clip_01` .. `clip_20`) back to master IDs (`review_clip_001` .. `review_clip_120`).
5. Assembles a 120-row master CSV sorted by original clip ID.
6. Raw returned files in `returned_responses/` remain preserved without modification.

---

## 6. Anti-Fabrication & Campaign Status Notice

- **No Ratings Fabricated:** All response templates remain blank. Zero synthetic or model-inferred ratings have been created.
- **Active Governance Banner:**  
  `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`
- Human listening validation is marked complete **only after** all 6 reviewers have returned authentic evaluations and the combined dataset has been analyzed against the quarantined answer key.
