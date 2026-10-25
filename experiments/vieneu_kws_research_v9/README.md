# Vietnamese Keyword Spotting: Candidate Freeze, Blinded Listening, Robustness, and Acoustic Feasibility
## Research Campaign v9 (Phases 75–79)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Root:** `experiments/vieneu_kws_research_v9/`  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Overview & Directory Architecture

This experiment package packages the frozen Vietnamese Keyword Spotting candidate `variant_w10`, conducts an independent reproducibility audit, prepares a double-blind human perceptual evaluation package, executes a 13-condition synthetic robustness stress study, and provides formal tripartite operational readiness decisions.

### Directory Structure
```
experiments/vieneu_kws_research_v9/
├── artifacts/                           # Frozen candidate package
│   ├── candidate_manifest.json          # Machine-readable model & parameter manifest
│   ├── inference.py                     # Self-contained reproducible inference runner
│   └── checkpoints/                     # Cryptographically frozen weights (seeds 42, 43, 44)
│       ├── variant_w10_seed42.pt
│       ├── variant_w10_seed43.pt
│       └── variant_w10_seed44.pt
├── human_review/                        # Double-blind human listening evaluation
│   ├── README.md                        # Reviewer instructions & scoring rubrics
│   ├── reviewer_form.csv                # Sanitized reviewer form (blinded)
│   ├── review_manifest.csv              # Internal answer key & provenance ledger
│   └── audio/                           # 120 randomized clips (review_clip_001.wav - 120.wav)
├── robustness/                          # Robustness benchmark assets
│   └── robustness_clean_manifest.csv    # Frozen manifest of 360 baseline validation clips
├── acoustic_loopback/                   # Physical playback-to-microphone protocol
│   ├── README.md                        # Protocol specification and blocker disclosure
│   └── loopback_manifest.csv            # Machine-readable capture metadata schema
├── predictions/                         # Evaluation outputs
│   ├── robustness_summary.json          # 13-condition cross-seed & per-seed metrics
│   └── robustness_per_clip.csv          # 14,040 per-clip inference records
├── reports/                             # Formal phase research reports
│   ├── PHASE75_CANDIDATE_FREEZE.md
│   ├── PHASE75_REPRODUCIBILITY_AUDIT.md
│   ├── PHASE76_BLINDED_LISTENING_PROTOCOL.md
│   ├── PHASE77_ROBUSTNESS_REPORT.md
│   ├── PHASE78_ACOUSTIC_FEASIBILITY.md
│   ├── PHASE79_FINAL_DECISION.md
│   └── PHASE79_READINESS_MATRIX.md
├── run_phase75_freeze_audit.py          # Phase 75 execution script
├── run_phase76_blinded_review.py        # Phase 76 package preparation script
├── run_phase77_robustness.py            # Phase 77 robustness evaluation script
└── README.md                            # Experiment documentation
```

---

## 2. Tripartite Operational Verdicts

1. **Mode A (Bounded-Utterance Push-to-Talk):** `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED`  
   - Macro Command Recall: **99.79% $\pm$ 0.36%**  
   - Non-Target FAR: **2.03% $\pm$ 0.58%**  
   - 5-Class Accuracy: **98.38% $\pm$ 0.40%**  
   - CPU Latency (Batch 1): **1.73 $\pm$ 0.03 ms**  
2. **Mode B (Controlled Acoustic Feasibility):** `NOT_RUN` (Technical hardware blocker; zero fabrication).  
3. **Mode C (Continuous Open-Mic):** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Excessive stream false alarms in historical evaluations, ~69.2 FA/hr; strictly blocked).

---

## 3. Replication Commands

```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Run Phase 75 Candidate Freeze and Reproducibility Audit
python experiments\vieneu_kws_research_v9\run_phase75_freeze_audit.py

# 3. Run Phase 76 Blinded Human Review Package Preparation
python experiments\vieneu_kws_research_v9\run_phase76_blinded_review.py

# 4. Run Phase 77 Controlled Synthetic Robustness Evaluation
python experiments\vieneu_kws_research_v9\run_phase77_robustness.py

# 5. Run Unit and Regression Test Suites
pytest tests\test_vieneu_research_v9.py -v
pytest tests\ -k vieneu
```
