# Vietnamese Keyword Spotting Research Campaign v3 (Phases 45–49)

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Overview & Objective

This repository contains the dataset, models, evaluation pipelines, and research reports for the integrated research campaign (Phases 45–49) investigating long-horizon streaming robustness, independent evaluation integrity, and recall-preserving training in Vietnamese Keyword Spotting (KWS).

### Target Command Keywords
- `TRAI` (Trái - Left)
- `PHAI` (Phải - Right)
- `LEN` (Lên - Up)
- `XUONG` (Xuống - Down)
- Additional classes: `UNKNOWN` (non-target speech), `SILENCE` (ambient background)

### Predeclared Operational Targets
- **Target Command Event Recall:** $\ge 95\%$
- **Non-Target False Activations:** $\le 1.0\text{ event/hour}$
- **Mean Event Detection Latency:** $\le 600\text{ ms}$

---

## 2. Directory Layout

```text
experiments/vieneu_kws_research_v3/
├── README.md                                  # This document
├── audit_findings.json                        # Phase 45 programmatic audit findings
├── run_phase45_audit.py                       # Phase 45 independent audit script
├── build_benchmark_v3.py                      # Phase 46 dataset & 3-hour continuous stream generator
├── train_cascade_v3.py                        # Phase 47 recall-preserving cascade training script
├── evaluate_campaign_v3.py                    # Phase 48-49 3-way benchmark and validation calibration
├── data/
│   ├── manifest.csv                           # Master 2,440-utterance manifest
│   ├── text_inventory.csv                     # Disjoint 210-text inventory across 5 confusion families
│   ├── splits/
│   │   ├── train_manifest.csv                 # 17 voices, 1,360 utterances
│   │   ├── val_manifest.csv                   # 4 voices, 360 utterances
│   │   └── test_manifest.csv                  # 4 voices, 720 utterances
│   ├── audio/                                 # 2,440 16kHz mono synthetic WAV files
│   └── continuous_streams/                    # Unbroken continuous challenge streams
│       ├── val_stream_continuous_commands.wav
│       ├── val_stream_continuous_nontarget.wav
│       ├── test_stream_continuous_commands.wav
│       └── test_stream_continuous_nontarget_3h.wav (EXACTLY 10,800.0s = 3.00 hours)
├── models/
│   └── hard_negative_cascade_v3/
│       ├── checkpoints/                       # Stage 1 and Stage 2 weights (seeds 42, 43, 44)
│       └── logs/training_summary.json
├── predictions/
│   ├── validation_summary.json                # Phase 48 validation stream metrics
│   ├── locked_test_summary.json               # Phase 49 locked 3-hour test metrics
│   └── per_event_predictions.csv              # Granular timestamped detections
├── calibration/
│   └── selected_operating_points.json         # Validation operating point selection records
└── reports/
    ├── PHASE45_AUDIT.md                       # Independent evaluation & data integrity audit
    ├── PHASE45_DISCREPANCIES.md               # Historical discrepancy reconciliations
    ├── PHASE46_BENCHMARK_REPORT.md            # Fresh 3-hour continuous challenge benchmark
    ├── PHASE47_TRAINING_REPORT.md             # Recall-preserving training & ablations
    ├── PHASE48_CALIBRATION_AND_SELECTION.md   # Validation-only calibration & finalist selection
    └── PHASE49_LOCKED_EVALUATION.md           # Locked 3-hour continuous evaluation report
    └── PHASE49_FINAL_DECISION.md              # Authoritative decision & roadmap
```

---

## 3. Preservation & Governance Rules

1. **Preservation of Historical Baselines:** All prior datasets (`experiments/synthetic_kws_dataset_vieneu_1000/`, `6class/`, `6class_v2/`, `vieneu_kws_research_v2/`), checkpoints, manifests, golden fixtures, and reports (Phases 26–44) remain completely immutable.
2. **Field Audio Isolation:** Zero access, modification, or inclusion of field recordings (`data/raw/field_recordings/`) or human consent ledgers.
3. **Blinded Human Listening Evaluation Status:** 80-clip listening evaluation remains recorded as `PENDING_HUMAN_LISTENER_VERIFICATION`. All conclusions are strictly designated `NOT_VALIDATED_ON_NATURAL_SPEECH`.
4. **Git Workspace Discipline:** `git diff --cached` remains strictly empty (zero staged files).
