# Blinded Vietnamese Speech Perceptual Review — Operational Manual
Workspace: `D:\trainidentificationaudio`  
Experiment: `experiments/vieneu_kws_research_v10/human_review/`  
Candidate Model: `variant_w10`  
Protocol Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. System Architecture & Blinding Invariants

The blinded human review system is strictly designed to evaluate perceptual audio quality without introducing confirmation bias or exposing ground truth.

### Key Governance Invariants:
1. **Zero Answer-Key Exposure:**
   - The ground-truth file (`review_manifest.csv`) contains canonical classes, target labels, phonetic transcriptions, and confusion family tags.
   - The web app (`review_app.py`) **never reads, imports, or references `review_manifest.csv`**.
   - Reviewer-facing assets expose only anonymous IDs (`review_clip_001` through `review_clip_120`).
2. **Local Loopback Network Isolation:**
   - The server binds exclusively to `127.0.0.1`.
   - Zero telemetry, analytics, CDNs, or external network requests are permitted.
3. **Data Integrity:**
   - All 120 audio files in `human_review/audio/` are immutable, 16 kHz mono 16-bit PCM WAVs.
   - SHA-256 checksums are pre-verified and matched to the manifest.

---

## 2. Server Startup & Verification

### Step 1: Launch Local Review Server
Run the local HTTP server in a terminal within the project environment:
```powershell
.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/review_app.py --port 8080
```
Expected output:
```text
Blinded Review Server running at http://127.0.0.1:8080/
Press Ctrl+C to terminate.
```

### Step 2: Verify Local Binding & Security
Confirm that the server is bound only to loopback:
```powershell
netstat -ano | findstr 8080
```
Verify that the listening address is strictly `127.0.0.1:8080`.

---

## 3. Reviewer Workflow Guide

1. **Accessing the App:**
   - Open a browser (Chrome, Edge, Firefox) and navigate to: `http://127.0.0.1:8080/`
2. **Reviewer Identification:**
   - Enter your designated Reviewer ID (e.g. `reviewer_01`, `listener_alice`) in the top-left box.
   - If resuming a session, entering your ID will automatically reload previously saved progress.
3. **Listening Environment:**
   - Use high-quality wired or wireless over-ear / in-ear headphones.
   - Maintain a quiet room environment.
   - Playback can be triggered via UI buttons or pressing the `Space` bar.
4. **Assessment Submission per Clip:**
   For each clip (`review_clip_001` to `review_clip_120`), provide:
   - **Perceived Text:** Exact Vietnamese words heard with proper diacritics. Use `[SILENCE]` for room ambient/quiet clips; use `[UNINTELLIGIBLE]` if speech is heard but indecipherable.
   - **Intelligibility (1–5):** 1 (Unintelligible) to 5 (Crystal clear).
   - **Naturalness (1–5):** 1 (Extremely distorted/robotic) to 5 (Completely human-like).
   - **Pronunciation & Tone Correctness:** `Y` (Correct), `N` (Defective/distorted), or `Unsure`.
   - **Confidence (1–5):** 1 (Pure guess) to 5 (Absolutely certain).
   - **Observations (Optional):** Specific acoustic anomalies, pitch glides, or tonal truncation.
5. **Autosave & Navigation:**
   - Moving to next (`ArrowRight` or "Next Clip →") automatically persists responses to `human_review/responses/<reviewer_id>_responses.csv`.
   - The user can click "Save Progress" at any point.

---

## 4. Response Validation Protocol

Once a reviewer completes the evaluation, their output CSV must be validated using `response_validation.py`:

```powershell
.venv\Scripts\python experiments/vieneu_kws_research_v10/human_review/response_validation.py `
    experiments/vieneu_kws_research_v10/human_review/responses/reviewer_01_responses.csv `
    --issues_output experiments/vieneu_kws_research_v10/human_review/responses/reviewer_01_issues.json
```

### Validation Checks:
1. **Completeness:** All 120 clips must have non-empty responses.
2. **Range Integrity:** Intelligibility, naturalness, and confidence must be integers $1 \le x \le 5$.
3. **Pronunciation Values:** Must be strictly `Y`, `N`, or `Unsure`.
4. **No Synthesized / Prefilled Responses:** Automated scripts or bots must never be used to fabricate responses. If missing, the status must strictly remain `PENDING_HUMAN_LISTENER_VERIFICATION`.

---

## 5. Post-Review Adjudication & Joining (Phase 86)
- Only after validation passes is the responses file joined against `review_manifest.csv` for analysis.
- The analysis script computes per-category intelligibility, naturalness MOS, pronunciation accuracy, and inter-rater agreement (Fleiss' Kappa or Krippendorff's Alpha if multiple reviewers participate).
