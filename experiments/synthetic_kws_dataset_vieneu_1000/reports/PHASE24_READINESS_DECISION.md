# Phase 24 — Human Listening QA and Six-Class KWS Readiness Decision

**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/`  
**Phase ID:** Phase 24 — Human Listening QA & Six-Class Readiness Audit  
**Date:** October 17, 2026  
**Official Project Governance Status:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  

---

## 1. Multi-Dimensional Readiness Verdicts

| Evaluation Dimension | Verdict | Operational Status & Evidence |
| :--- | :---: | :--- |
| **1. Technical File Integrity** | **`PASS`** | 1,000/1,000 raw 48 kHz and KWS 16 kHz files verified on disk; 100% SHA-256 match; zero NaNs/Infs; zero clipping (max peak 0.9500); zero boundary discontinuities (edge amplitude 0.000000); 69/69 automated regression tests passed. |
| **2. Automated Tone Diagnostics** | **`PASS WITH LIMITATIONS`** | Autocorrelation $F_0$ contours demonstrate phonologically expected trajectories (+63.2 Hz for `TRAI`, +54.9 Hz for `XUONG`, -6.8 Hz for `LEN`, 82.0% interior dips for `PHAI`). **Limitation:** F0 extraction is diagnostic only and does not establish human perceptual correctness. |
| **3. Human Intelligibility Verification** | **`PENDING`** | 80-clip stratified blinded listening package prepared and quarantined with preserved key. Ratings remain completely blank (anti-fabrication invariant). Formal status: **`PENDING_HUMAN_LISTENER_VERIFICATION`**. |
| **4. Four-Class Baseline Readiness** | **`READY (SYNTHETIC HELD-OUT)`** | Technically ready for closed-set 4-class training/evaluation across the 17 train / 4 val / 4 test disjoint voice presets. |
| **5. Six-Class KWS Readiness** | **`BLOCKED (DATA GAP)`** | Cannot train an open-set or streaming KWS detector until `UNKNOWN` (non-target speech) and `SILENCE` (ambient/digital background) are synthesized under [`SIX_CLASS_DATA_READINESS_PLAN.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/SIX_CLASS_DATA_READINESS_PLAN.md). |
| **6. Natural-Speech Validation** | **`NOT VALIDATED`** | Synthesized purely from 25 preset conditioning vectors in VieNeu-TTS v3 Turbo. Does not evaluate natural human talkers, accents, room impulse responses, or field microphones. |

---

## 2. Key Unresolved Blockers

1. **Human Listening Responses Unsubmitted:**
   - The 80-clip review sheet [`blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv) awaits evaluation by a Vietnamese-proficient listener.
2. **Missing `UNKNOWN` and `SILENCE` Partitions for 6-Class KWS:**
   - 4-class models will suffer from catastrophic false triggers in continuous streaming audio without non-keyword speech and background silence classes.
3. **Absence of Real-World Environmental Noise:**
   - Signals are clean synthetic speech without additive room reverberation or ambient background noise.

---

## 3. Recommended Next Actions

1. **Conduct Human Listening Screening:**
   - Have an evaluator listen to the 80 clips in [`blinded_listening/audio/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/audio/) and record responses in `blinded_listening_review_sheet.csv`.
   - Run `python blinded_listening/analyze_blinded_review.py` to produce `HUMAN_LISTENING_EVALUATION_REPORT.md`.
2. **Generate Six-Class Expansion Data:**
   - Implement the specifications in `reports/SIX_CLASS_DATA_READINESS_PLAN.md` to generate 250 `UNKNOWN` utterances (using VieNeu-TTS v3 Turbo across the same 25 voices) and 250 `SILENCE` clips.
3. **Execute 6-Class Model Training & Streaming Evaluation:**
   - Retrain TC-ResNet-14 on the unified 1,500-sample 6-class dataset and conduct streaming false-alarm / latency stress testing.

---

## 4. Key Artifact Paths

- **Phase 24 Readiness Decision:** [`reports/PHASE24_READINESS_DECISION.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/PHASE24_READINESS_DECISION.md)
- **PHAI Tone Diagnostic Addendum:** [`reports/PHAI_TONE_DIAGNOSTIC_ADDENDUM.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/PHAI_TONE_DIAGNOSTIC_ADDENDUM.md)
- **Six-Class Data Readiness Plan:** [`reports/SIX_CLASS_DATA_READINESS_PLAN.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/SIX_CLASS_DATA_READINESS_PLAN.md)
- **Blinded Review Analysis Tool:** [`blinded_listening/analyze_blinded_review.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/analyze_blinded_review.py)
- **Blinded Listening Review Sheet:** [`blinded_listening/blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv)
- **Quarantined Answer Key:** [`blinded_listening/blinded_listening_key.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.json)
- **Master Dataset Manifest:** [`manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv)
