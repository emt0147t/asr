# Phase 85 — Blinded Listening Workflow Audit and Operation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment:** `experiments/vieneu_kws_research_v10/`  
**Candidate Model:** `variant_w10` (Frozen)  
**Governance Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Audit Date:** 2026-10-18  

---

## 1. Executive Summary & Blinding Integrity

Phase 85 conducted a rigorous forensic audit and hardening of the blinded human listening review system located in `experiments/vieneu_kws_research_v10/human_review/`.

### Key Verification Milestones:
1. **Audio Asset Integrity:** All 120 audio clips (`review_clip_001.wav` through `review_clip_120.wav`) were audited against the protected `review_manifest.csv`. All 120 files exist, are decodable 16 kHz 16-bit mono PCM WAVs, and exhibit 100% SHA-256 parity (0 mismatches).
2. **Double-Blind Isolation:** The web interface (`review_app.py`) was audited for information leakage. The server script and frontend code contain zero references, imports, or API routes serving the ground truth manifest `review_manifest.csv`. Clip identifiers (`review_clip_001` to `review_clip_120`) are purely sequential and contain no lexical, class, or speaker metadata.
3. **Local Loopback Security:** The application binds strictly to `127.0.0.1`. There are zero remote analytics, CDN dependencies, or external network requests.
4. **Vulnerability Discovery & Remediation:**
   - *Missing Import Defect:* Auditing revealed that `review_app.py` attempted to use `shutil.copyfileobj` without importing `shutil`, which would have crashed audio streaming requests.
   - *Input Sanitization:* Added regex sanitization `re.sub(r'[^a-zA-Z0-9_\-]', '', rev_raw)` to prevent path traversal in `reviewer_id`.
   - Both defects were resolved without modifying frozen audio assets or historical data.

---

## 2. Forensic Blinding & Interface Audit Checklist

| Audit Item | Specification Requirement | Verification Method | Result | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Ground Truth Isolation** | Zero exposure of transcripts, target labels, confusion family, model prediction | Code inspection & regex search of `review_app.py` | **PASSED** | Manifest is completely quarantined; server only knows `CLIP_IDS` |
| **Audio Naming Blinding** | Filenames must not reveal classes or transcripts | Directory listing & manifest check | **PASSED** | Strict naming: `review_clip_001.wav` to `review_clip_120.wav` |
| **Network Binding** | Bound strictly to `127.0.0.1` | Socket inspection (`server_address = ("127.0.0.1", port)`) | **PASSED** | No external interface exposure |
| **Telemetry / CDNs** | Zero external dependencies | HTML head/body inspection | **PASSED** | Pure vanilla JS and standard CSS; no external scripts or stylesheets |
| **Audio Integrity** | 120 WAV files matching SHA-256 hashes | SHA-256 hash recomputation over all 120 files | **PASSED** | 120/120 matched (0 corruptions) |
| **Progress Persistence** | Incremental autosave and session resuming | Automated POST `/api/save` & GET `/api/load` validation | **PASSED** | Saves per-clip to `responses/<id>_responses.csv` |
| **Path Traversal Protection** | Reviewer ID sanitization | Security inspection of file paths | **PASSED** | Regex-enforced alphanumeric/underscore/dash filter |

---

## 3. Reviewer Workflow & Rubric Verification

The review application implements the complete standardized 6-point evaluation rubric:

1. **Perceived Words / Syllables (`perceived_text`):** Freeform Vietnamese text entry with diacritics. Explicit tokens `[SILENCE]` and `[UNINTELLIGIBLE]` provided for edge cases.
2. **Intelligibility (1–5 scale):**
   - 5: Crystal clear comprehension
   - 4: Mostly clear; easily understood
   - 3: Moderately clear; noticeable synthetic distortion
   - 2: Barely intelligible; heavily garbled
   - 1: Completely unintelligible
3. **Naturalness (1–5 scale):**
   - 5: Completely human-like
   - 4: Natural with slight synthetic quality
   - 3: Acceptable synthetic cadence
   - 2: Robotic / unnatural tone
   - 1: Extremely distorted / buzzy
4. **Pronunciation & Tone Correctness:** Categorical choice (`Y` / `N` / `Unsure`) for assessing phoneme integrity and tonal contour.
5. **Confidence (1–5 scale):** Reviewer self-assessment of auditory confidence (1 = pure guess, 5 = absolutely certain).
6. **Observations / Notes:** Optional freeform field for logging clicks, buzz, tone flattening, or clipping.

---

## 4. Response Validation Tooling (`response_validation.py`)

To ensure high-integrity ingestion without data fabrication:
- Implemented `human_review/response_validation.py`.
- Enforces strict validation of all 120 clip IDs, required non-empty fields, valid rating integer domains ($[1, 5]$), and pronunciation tokens (`Y`, `N`, `Unsure`).
- Automatically quarantines malformed or incomplete rows into an issues report rather than silently modifying or dropping them.
- Tested against blank template (`response_template.csv`): correctly flagged 120 incomplete rows and rejected false completion claims.

---

## 5. Current Human Review Status

- **Total Human Responses on File:** **0**
- **Fabricated Responses Created:** **0**
- **Operational Status:** `PENDING_HUMAN_LISTENER_VERIFICATION`
- **Natural Speech Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`

The review interface is fully verified, secured, and ready for human listeners.
