# Vietnamese Keyword Spotting Research Campaign v2 (Phases 40–44)

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v2/`  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Overview & Objective

This repository contains the code, data, models, and evaluation reports for the integrated research campaign (Phases 40–44) investigating the speech-active false alarm bottleneck in Vietnamese Keyword Spotting (KWS).

### Target Command Keywords
- `TRAI` (Trái - Left)
- `PHAI` (Phải - Right)
- `LEN` (Lên - Up)
- `XUONG` (Xuống - Down)
- Additional classes: `UNKNOWN` (non-target speech), `SILENCE` (ambient background)

### Primary Performance Requirements
- **Target Command Event Recall:** $\ge 95\%$
- **Non-Target False Activations:** $\le 1.0\text{ event/hour}$
- **Mean Event Detection Latency:** $\le 600\text{ ms}$

---

## 2. Directory Structure

```text
experiments/vieneu_kws_research_v2/
├── README.md                                  # This document
├── build_research_dataset.py                  # Phase 41 dataset synthesis & continuous stream generation
├── train_cascade_v2.py                        # Phase 42 Candidate A (Hard-Negative Stage 1 & 2) training script
├── train_ctc_kws.py                           # Phase 42 Candidate B (Character-Level CTC KWS) training script
├── evaluate_campaign.py                       # Phase 43 4-way candidate benchmark and validation calibration
├── data/
│   ├── manifest.csv                           # Master 2,000-utterance manifest
│   ├── text_inventory.csv                     # Disjoint 100-text inventory across 5 confusion families
│   ├── splits/
│   │   ├── train_manifest.csv                 # 17 voices, 1,360 utterances
│   │   ├── val_manifest.csv                   # 4 voices, 320 utterances
│   │   └── test_manifest.csv                  # 4 voices, 320 utterances
│   ├── audio/                                 # 2,000 16kHz mono synthetic WAV files
│   └── continuous_streams/                    # Unbroken 4-stream challenge benchmark
│       ├── val_stream_continuous_commands.wav
│       ├── val_stream_continuous_nontarget.wav
│       ├── test_stream_continuous_commands.wav
│       └── test_stream_continuous_nontarget.wav
├── models/
│   ├── hard_negative_cascade/
│   │   ├── checkpoints/                       # Stage 1 and Stage 2 retrained weights (seeds 42, 43, 44)
│   │   └── logs/training_summary.json
│   └── ctc_kws/
│       ├── ctc_model.py                       # VietnameseCharTokenizer and VietnameseCTCKWSNet definition
│       ├── checkpoints/                       # CTC checkpoints (seeds 42, 43, 44)
│       └── logs/training_summary.json
├── predictions/
│   ├── validation_summary.json                # Phase 43 validation stream metrics
│   ├── locked_test_summary.json               # Phase 43 locked held-out test stream metrics
│   └── per_event_predictions.csv              # Granular timestamped detections
├── calibration/
│   └── selected_operating_points.json         # Validation operating point selection records
└── reports/
    ├── PHASE40_PIPELINE_AUDIT.md              # Pipeline audit and protocol lock
    ├── PHASE41_DATASET_REPORT.md              # Hard-negative corpus and streaming benchmark report
    ├── PHASE42_MODEL_TRAINING_REPORT.md       # Multi-seed training and convergence report
    ├── PHASE43_LOCKED_EVALUATION_REPORT.md    # Multi-candidate streaming benchmark report
    └── PHASE44_FINAL_RESEARCH_DECISION.md     # Architectural verdict and transition roadmap
```

---

## 3. Preservation & Governance Rules

1. **Preservation of Historical Baselines:** All prior datasets (`experiments/synthetic_kws_dataset_vieneu_1000/`, `6class/`, `6class_v2/`), checkpoints, manifests, golden fixtures, and reports (Phases 26–39) remain completely immutable.
2. **Field Audio Isolation:** Zero access, modification, or inclusion of field recordings (`data/raw/field_recordings/`) or human consent ledgers.
3. **Blinded Human Listening Evaluation Status:** 80-clip listening evaluation remains recorded as `PENDING_HUMAN_LISTENER_VERIFICATION`. All conclusions are strictly designated `NOT_VALIDATED_ON_NATURAL_SPEECH`.
4. **Git Workspace Discipline:** `git diff --cached` remains strictly empty (zero staged files).
