# Phase 90 — Human Listening Workflow Operational Readiness Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Root:** `experiments/vieneu_kws_research_v10/`  
**Candidate Model:** `variant_w10` (Frozen)  
**Workflow Status:** `READY_FOR_HUMAN_REVIEW`  
**Submission Status:** `PENDING_HUMAN_RESPONSES`  
**Governance Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Continuous Open-Mic Verdict:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
**Date:** 2026-10-18  

---

## 1. Operational Status Summary

Phase 90 operationalizes the blinded human-listening validation package for the Vietnamese Keyword Spotting candidate `variant_w10`.

```
+---------------------------------------------------------------------------------------------------------+
|                                    PHASE 90 OPERATIONAL STATUS                                          |
+---------------------------------------------------------------------------------------------------------+
| Review Infrastructure:       | READY_FOR_HUMAN_REVIEW (Web app, endpoints, assets fully verified)       |
| Review Submissions:          | PENDING_HUMAN_RESPONSES (0 human submissions on file; blank template)    |
| Governance Verification:     | PENDING_HUMAN_LISTENER_VERIFICATION (Retained until audited)             |
| Natural Human Speech Status: | NOT_VALIDATED_ON_NATURAL_SPEECH (Synthetic VieNeu-TTS speech only)       |
| Open-Mic Stream Status:      | NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION (Streaming FA ~69/hr)      |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Audio Asset & Manifest Integrity Audit (Task 1)

All review audio clips were verified using the project environment (`.venv`) against the internal ground-truth ledger [`human_review/review_manifest.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/review_manifest.csv):

- **Target Audio Directory:** `experiments/vieneu_kws_research_v10/human_review/audio/`
- **Total Manifest Records:** 120 clips (`review_clip_001.wav` through `review_clip_120.wav`)
- **Audio Format:** 16 kHz mono, 16-bit PCM WAV, exactly 1.0 second duration (16,000 samples)
- **Hash Parity Results:**
  - **Matched:** **120 / 120** ($100.0\%$)
  - **Missing Files:** **0**
  - **SHA-256 Mismatches:** **0**

---

## 3. Application Startup Instructions & Response Schema (Task 2)

### A. Exact Launch Command
From the project workspace root (`D:\trainidentificationaudio`), execute:
```powershell
.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/review_app.py --port 8080
```

### B. Expected Terminal Startup Output
```text
Blinded Review Server running at http://127.0.0.1:8080/
Press Ctrl+C to terminate.
```

### C. Response Schema
The response files conform strictly to the following 7-field CSV schema:

| Column Header | Type | Valid Domain | Description |
| :--- | :--- | :--- | :--- |
| `review_clip_id` | String | `review_clip_001` .. `review_clip_120` | Anonymized clip identifier |
| `perceived_text` | String | Non-empty Vietnamese text | Exact words heard with diacritics, or `[SILENCE]`, `[UNINTELLIGIBLE]` |
| `intelligibility_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Comprehensibility (1 = Unintelligible, 5 = Crystal clear) |
| `naturalness_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | MOS naturalness (1 = Highly robotic/distorted, 5 = Human-like) |
| `pronunciation_correct_yn` | String | `Y`, `N`, `Unsure` | Phonetic correctness of tone contour and vowels |
| `confidence_1to5` | Integer | `1`, `2`, `3`, `4`, `5` | Listener confidence (1 = Pure guess, 5 = Certain) |
| `comments` | String | Optional freeform text | Notes on artifacts, swallowed syllables, or background noise |

### D. Step-by-Step Instructions for Reviewers
1. Ensure the web application is running at `http://127.0.0.1:8080/`.
2. Open a standard browser (Chrome, Edge, Firefox) and navigate to `http://127.0.0.1:8080/`.
3. In the top-left **Reviewer ID** input box, enter your assigned identifier (e.g. `reviewer_01`, `listener_alice`).
4. In a quiet environment using headphones, listen to the clip (press `Space` or click the audio player).
5. Complete all required fields (1 through 5) and optional comments (field 6).
6. Press `ArrowRight` or click **Next Clip →**. The app automatically persists your progress to `human_review/responses/<reviewer_id>_responses.csv`.
7. Repeat for all 120 clips. On Clip 120, click **Finish & Submit**.

---

## 4. Response Validator Execution (Task 3)

The existing validator [`human_review/response_validation.py`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/response_validation.py) was executed on the unmodified blank template [`human_review/response_template.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/response_template.csv):

```powershell
.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/response_validation.py experiments/vieneu_kws_research_v10/human_review/response_template.csv
```

### Validator Output:
```json
{
  "file": "experiments/vieneu_kws_research_v10/human_review/response_template.csv",
  "is_complete": false,
  "valid_rows": 0,
  "invalid_rows": 120,
  "errors": []
}
```

### Preservation & Anti-Fabrication Confirmation:
- The template file remains preserved with empty data fields.
- Zero placeholder or synthetic responses have been injected.
- The validator correctly rejects incomplete submissions and mandates all 120 clips.

---

## 5. Review Blinding & Scope Separation (Task 4)

### Double-Blind Verification:
1. **Zero Answer-Key Leakage:** The application script [`review_app.py`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/review_app.py) contains zero references, imports, or API routes serving `review_manifest.csv`.
2. **Anonymized Identifiers:** Files are named sequentially (`review_clip_001.wav` to `review_clip_120.wav`) with no voice names, lexical hints, or target class labels.
3. **No Model Exposure:** The web frontend displays no predictions, probabilities, or confidence scores from the CNN.

### Scope Distinction:
- **What this review measures:** Native listener perceptual clarity, intelligibility, tone preservation, and naturalness of **synthesized speech generated by VieNeu-TTS**.
- **What this review does NOT measure:** It does **NOT** measure the CNN model's accuracy on natural human speech.
- Evaluating synthesized audio ensures the training/testing data is phonetically valid Vietnamese. Testing CNN recognition on live human speakers requires a separate, consented human recording campaign.

---

## 6. Operational Parameters & Actions Remaining (Task 5)

| Parameter | Specification |
| :--- | :--- |
| **Local Application URL** | `http://127.0.0.1:8080/` |
| **Expected Response Location** | `experiments/vieneu_kws_research_v10/human_review/responses/<reviewer_id>_responses.csv` |
| **Validation Command** | `.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/response_validation.py experiments/vieneu_kws_research_v10/human_review/responses/<reviewer_id>_responses.csv` |
| **How to Terminate the App** | Send `Ctrl+C` in the terminal executing `review_app.py` |
| **Host Runtime Feasibility** | Verified: Pure Python HTTP server binds successfully to `127.0.0.1:8080` |

### Exact Manual Actions Remaining:
1. Provide human listeners with access to `http://127.0.0.1:8080/`.
2. Reviewers listen to the 120 clips via headphones and submit their ratings.
3. Run `response_validation.py` to confirm submission completeness.
4. Run `analyze_human_review.py` to merge responses with `review_manifest.csv` and compute MOS metrics.

Until real human listener responses are collected, audited, and validated, the status remains strictly:
$$\mathbf{PENDING\_HUMAN\_LISTENER\_VERIFICATION}$$
