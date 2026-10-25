# Vietnamese Keyword Spotting Research Campaign v4 (Phases 50–54)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v4/`  
**Keywords:** `TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT` (`UNKNOWN` / `SILENCE`)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Directory Structure

```
experiments/vieneu_kws_research_v4/
├── README.md                                          <- Master overview and campaign guide
├── audit_findings_phase50.json                        <- Serialized bit-exact audit findings
├── run_phase50_audit.py                               <- Independent audit script
├── run_phase51_analysis.py                            <- Acoustic error taxonomy and review kit generator
├── build_bounded_manifests.py                         <- Bounded-utterance dataset manifest builder
├── evaluate_bounded_campaign.py                       <- Validation calibration and locked test evaluation script
├── calibration/
│   └── selected_bounded_utterance_configs.json        <- Calibrated thresholds for bounded mode
├── data/
│   ├── manifest.csv                                   <- 2,440-utterance master manifest
│   ├── splits/
│   │   ├── train_manifest.csv                         <- 1,360 clips (17 voices)
│   │   ├── val_manifest.csv                           <- 360 clips (4 voices)
│   │   └── test_manifest.csv                          <- 720 clips (4 held-out voices)
│   ├── error_taxonomy.csv                             <- Acoustic failure modes and phonetic taxonomy
│   └── phase51_analysis_summary.json                  <- Confusion family and voice error statistics
├── human_review/
│   ├── README.md                                      <- Blinded listening review instructions
│   ├── human_review_sheet.csv                         <- Blinded reviewer worksheet (40 items)
│   └── human_review_key.csv                           <- Locked ground-truth review key
├── models/
│   └── bounded_utterance/
│       ├── checkpoints/
│       │   ├── bounded_kws_net_seed42.pt              <- Candidate 3 checkpoint seed 42
│       │   ├── bounded_kws_net_seed43.pt              <- Candidate 3 checkpoint seed 43
│       │   └── bounded_kws_net_seed44.pt              <- Candidate 3 checkpoint seed 44
│       ├── logs/                                      <- Training progression logs
│       └── train_bounded_model.py                     <- Training script for Candidate 3
├── predictions/
│   ├── validation_summary.json                        <- Validation sweep metrics
│   ├── locked_test_summary.json                       <- Locked test results across all 3 candidates
│   └── per_utterance_predictions.csv                  <- Serialized individual test predictions
└── reports/
    ├── PHASE50_BENCHMARK_AUDIT.md                     <- 3-hour stream and metric audit report
    ├── PHASE50_ISSUES_AND_CORRECTIONS.md              <- Categorization and repetition findings
    ├── PHASE51_ERROR_ANALYSIS.md                      <- /aːj/ rhyme dilemma and error breakdown
    ├── PHASE53_CALIBRATION_AND_TEST.md                <- Bounded-utterance calibration and test report
    ├── PHASE54_FINAL_DECISION.md                      <- Dual-verdict certification
    └── PHASE54_READINESS_MATRIX.md                    <- Multi-dimensional readiness matrix
```

---

## 2. Dual Verdicts Summary

1. **Mode A: Bounded-Utterance Recognition (Push-to-Talk / User-Initiated Turn):**
   - **Verdict:** `BOUNDED_UTTERANCE_FEASIBLE_IN_SYNTHETIC_RESEARCH`
   - **Champion Architecture:** Candidate 2 (Cascade v1 Bounded Pipeline).
   - **Performance:** **95.83%** Command Recall (`TRAI` 99.17%, `PHAI` 100.0%, `XUONG` 93.33%, `LEN` 90.83%), **85.12%** Non-Target Rejection Rate, **5.81 ms** Latency.
2. **Mode B: Continuous Open-Mic Recognition (Always-Listening Audio Stream):**
   - **Verdict:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
   - **Status:** Unconstrained continuous open-mic monitoring remains blocked by sliding-window boundary false triggers (10.33 to 77.33 FA/hr). Success in bounded-utterance recognition does not transfer to open-mic streaming.

---

## 3. Reproduction Commands

```bash
# 1. Run Phase 50 Independent Audit
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v4/run_phase50_audit.py

# 2. Run Phase 51 Acoustic Analysis and Build Review Kit
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v4/run_phase51_analysis.py

# 3. Build Bounded Manifests
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v4/build_bounded_manifests.py

# 4. Train Candidate 3 Bounded Models
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v4/models/bounded_utterance/train_bounded_model.py

# 5. Run Calibration & Locked Test Evaluation
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v4/evaluate_bounded_campaign.py
```
