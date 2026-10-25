# Vietnamese Keyword Spotting: Human Listening Operations, Robustness Reproduction, and Acoustic Feasibility
## Research Campaign v10 (Phases 80–89)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Root:** `experiments/vieneu_kws_research_v10/`  
**Candidate Model:** `variant_w10` (Frozen)  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Open-Mic Status:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  

---

## 1. Executive Summary & Purpose

Campaign v10 delivers an exhaustive empirical and operational validation suite for the frozen Vietnamese KWS candidate `variant_w10`:
1. **Phases 80 & 87 (Metric Reconciliation & Robustness Reproduction):** Reconciles test ($N=720$) vs validation baseline ($N=360$), audits mathematical formulations across 13 acoustic stress conditions, and executes 14,040 CPU inferences. Flagged $+20\text{ dB}$ SNR ($94.37\%$ recall) as `BELOW_RECALL_TARGET`.
2. **Phases 81 & 85–86 (Human Listening Review):** Operates a hardened, double-blind local web app (`127.0.0.1:8080`), audits zero answer-key leakage, deploys automated response validation (`response_validation.py`), and records honest review accounting (0 responses fabricated, status `BLOCKED_NO_HUMAN_RESPONSES`).
3. **Phases 83 & 88 (Physical Acoustic Loopback):** Inspects physical audio endpoints and driver libraries at runtime. Marks `NOT_RUN` (verdict `UNRESOLVED`) due to lack of connected physical hardware, strictly forbidding software simulation of physical transducers.
4. **Phases 84 & 89 (Final Readiness Decision & Matrix):** Concludes that further model training on synthetic speech is saturated and not justified. Maps next operational unblocking gates to human listeners, physical transducer testing, and natural human speech.

---

## 2. Directory Architecture

```
experiments/vieneu_kws_research_v10/
├── artifacts/                           # Candidate metadata and manifests
├── human_review/                        # Blinded human listening review package
│   ├── README.md                        # Review overview
│   ├── review_app.py                    # Hardened local review web app (127.0.0.1:8080)
│   ├── reviewer_instructions.md         # Reviewer guidelines and scoring rubrics
│   ├── review_operations.md             # Operational procedures manual
│   ├── response_template.csv            # Blank response template (120 clips)
│   ├── review_manifest.csv              # Internal quarantined answer key
│   ├── response_validation.py           # Automated schema & data integrity validator
│   ├── analyze_human_review.py          # Response ingestion and adjudication script
│   ├── validated_responses.csv          # Validated responses (empty pending reviews)
│   ├── review_quality_summary.json      # Intake audit status (BLOCKED_NO_HUMAN_RESPONSES)
│   ├── audio/                           # 120 anonymized audio files (review_clip_001.wav - 120.wav)
│   └── responses/                       # Directory for saved reviewer submissions
├── robustness/                          # Robustness audit assets
│   ├── condition_manifest.json          # 13-condition mathematical formulation manifest v1
│   └── condition_manifest_v2.json       # Detailed formula, DSP, and gain/clipping ledger v2
├── acoustic_loopback/                   # Physical acoustic playback-to-microphone suite
│   ├── run_loopback_capture.py          # Device inspection and loopback capture script
│   ├── recording_manifest_template.csv  # Machine-readable capture metadata schema
│   └── captures/                        # Directory for raw captured physical audio (empty)
├── predictions/                         # Independently reproduced prediction outputs
│   ├── reproduced_phase74_metrics.json  # Phase 74 locked test reproduction (N=720)
│   ├── reproduced_phase77_metrics.json  # Phase 77 clean baseline reproduction (N=360)
│   ├── robustness_reproduced_summary.json# 13-condition summary across 3 seeds (14,040 inferences)
│   └── robustness_per_clip.csv          # Granular 14,040 inference per-clip ledger
├── reports/                             # Formal research reports
│   ├── PHASE80_METRIC_RECONCILIATION.md
│   ├── PHASE80_CONFIGURATION_AUDIT.md
│   ├── PHASE81_LISTENING_REVIEW_PROTOCOL.md
│   ├── PHASE82_ROBUSTNESS_AUDIT.md
│   ├── PHASE83_ACOUSTIC_PROTOCOL.md
│   ├── PHASE84_FINAL_DECISION.md
│   ├── PHASE84_READINESS_MATRIX.md
│   ├── PHASE85_REVIEW_WORKFLOW_AUDIT.md
│   ├── PHASE86_HUMAN_REVIEW_ANALYSIS.md
│   ├── PHASE87_ROBUSTNESS_REPRODUCTION.md
│   ├── PHASE88_ACOUSTIC_LOOPBACK.md
│   ├── PHASE89_FINAL_DECISION.md
│   └── PHASE89_READINESS_MATRIX.md
├── run_phase80_reconciliation.py        # Phase 80 execution script
├── run_phase81_prep.py                  # Phase 81 preparation script
├── run_phase82_robustness_audit.py      # Phase 82 audit script
├── run_phase87_reproduction.py          # Phase 87 independent reproduction script
└── README.md                            # Experiment package documentation
```

---

## 3. Quad-Partite Operational Verdicts

1. **Mode A (Bounded-Utterance Push-to-Talk):** `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED`  
   - Macro Command Recall: **$99.79\% \pm 0.36\%$** (TRAI: 100%, PHAI: 99.17%, LEN: 100%, XUONG: 100%)  
   - Non-Target FAR: **$2.03\% \pm 0.58\%$**  
   - 5-Class Accuracy: **$98.38\% \pm 0.40\%$**  
   - CPU Decision Latency: **$1.73 \pm 0.03\text{ ms}$**  
2. **Mode B (Human Perceptual Quality):** `PENDING_HUMAN_LISTENER_VERIFICATION` (0 human reviews collected; zero fabrication).  
3. **Mode C (Physical Acoustic Feasibility):** `NOT_RUN` / `UNRESOLVED` (No physical hardware attached; zero simulation).  
4. **Mode D (Continuous Open-Mic Mode):** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Stream false alarms $\sim 69.2\text{ FA/hour}$).  

---

## 4. Replication Commands

```powershell
# 1. Run Phase 85/86 Human Review Intake and Analysis
.venv\Scripts\python experiments\vieneu_kws_research_v10\human_review\analyze_human_review.py

# 2. Launch Local Blinded Review App (Browser: http://127.0.0.1:8080/)
.venv\Scripts\python experiments\vieneu_kws_research_v10\human_review\review_app.py --port 8080

# 3. Validate Reviewer CSV Submission
.venv\Scripts\python experiments\vieneu_kws_research_v10\human_review\response_validation.py `
    experiments\vieneu_kws_research_v10\human_review\responses\reviewer_01_responses.csv

# 4. Run Phase 87 Independent Robustness Reproduction (14,040 inferences)
.venv\Scripts\python experiments\vieneu_kws_research_v10\run_phase87_reproduction.py

# 5. Run Phase 88 Physical Loopback Check
.venv\Scripts\python experiments\vieneu_kws_research_v10\acoustic_loopback\run_loopback_capture.py

# 6. Run Automated Test Suites
.venv\Scripts\pytest tests\test_vieneu_research_v10.py
.venv\Scripts\pytest tests\test_vieneu_research_v10_p85_89.py
.venv\Scripts\pytest tests\ -k vieneu
```
