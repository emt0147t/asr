# Vietnamese Keyword Spotting Research Campaign v5 (Phases 55–59)
**Bounded-Utterance Improvement and Independent Evaluation**

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Root:** `experiments/vieneu_kws_research_v5/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Directory Structure

```
experiments/vieneu_kws_research_v5/
├── README.md                                          <- Campaign overview and reproduction guide
├── phase55_audit_findings.json                        <- Machine-readable audit and CPU benchmark results
├── run_phase55_audit.py                               <- Phase 55 metric and CPU latency audit script
├── build_dataset_v5.py                                <- Phase 56 VieNeu-TTS dataset generator
├── run_phase58_calibration.py                         <- Phase 58 validation calibration script
├── run_phase59_locked_test.py                         <- Phase 59 locked test evaluation script
├── calibration/
│   └── selected_configs.json                          <- Calibrated operating points for all variants
├── data/
│   ├── manifest.csv                                   <- 2,440-utterance master manifest
│   ├── splits/
│   │   ├── train_manifest.csv                         <- 1,360 clips (17 voices)
│   │   ├── val_manifest.csv                           <- 360 clips (4 voices)
│   │   └── test_manifest.csv                          <- 720 clips (4 held-out voices)
│   └── audio/                                         <- Raw 48k and standardized 16k waveforms
├── human_review/
│   └── README.md                                      <- Anti-fabrication protocol and listener guide
├── models/
│   ├── train_ablation_campaign.py                     <- Training script for Variants C and D
│   ├── candidate_c_standard/                          <- Checkpoints and training logs for Variant C
│   └── candidate_d_ablation/                          <- Checkpoints and training logs for Variant D
├── predictions/
│   ├── validation_summary.json                        <- Phase 58 validation sweep results
│   ├── locked_test_summary.json                       <- Phase 59 locked test results
│   └── per_clip_predictions.csv                       <- Serialized individual test clip predictions
└── reports/
    ├── PHASE55_BASELINE_AUDIT.md                      <- Source code and baseline audit report
    ├── PHASE55_METRIC_ISSUES.md                       <- Metric clarifications and CPU latency findings
    ├── PHASE56_DATASET_REPORT.md                      <- Dataset v5 construction report
    ├── PHASE57_TRAINING_ABLATION.md                   <- Controlled training ablation report
    ├── PHASE58_CALIBRATION_AND_ROBUSTNESS.md          <- Validation calibration report
    ├── PHASE59_LOCKED_TEST.md                         <- Locked test benchmark report
    ├── PHASE59_FINAL_DECISION.md                      <- Dual-verdict certification report
    └── PHASE59_READINESS_MATRIX.md                    <- Multi-dimensional readiness matrix
```

---

## 2. Dual Formal Verdicts

1. **Mode A (Bounded-Utterance Recognition):**
   - **Verdict:** `BOUNDED_UTTERANCE_RESEARCH_BENCHMARK_ADVANCED`
   - **Champion Architecture:** Finalist 1 (Cascade v1 Bounded Pipeline).
   - **Key Performance:** **92.29%** Command Recall (`TRAI`: 98.33%, `PHAI`: 90.83%, `LEN`: 90.83%, `XUONG`: 89.17%), **4.64%** False Acceptance Rate (meeting $\le 5.0\%$), **95.36%** Non-Target Rejection Accuracy, **4.15 ms** CPU Latency.
   - **Readiness:** Justifies proceeding to a subsequent controlled natural microphone-recording feasibility phase.

2. **Mode B (Continuous Open-Mic Recognition):**
   - **Verdict:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
   - **Status:** Unconstrained continuous open-mic monitoring remains blocked by high false trigger rates (10.33 to 77.33 FA/hr). Success in bounded-utterance mode does not confer open-mic streaming readiness.

---

## 3. Reproduction Commands

```bash
# 1. Run Phase 55 Baseline and Metric Audit
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v5/run_phase55_audit.py

# 2. Build Dataset v5 (VieNeu-TTS synthesis)
.\.venv_vieneu\Scripts\python.exe experiments/vieneu_kws_research_v5/build_dataset_v5.py

# 3. Train Candidate C & D Ablation Models
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v5/models/train_ablation_campaign.py

# 4. Run Validation Calibration & Robustness Profiling
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v5/run_phase58_calibration.py

# 5. Run Locked Test Evaluation
.\.venv\Scripts\python.exe experiments/vieneu_kws_research_v5/run_phase59_locked_test.py
```
