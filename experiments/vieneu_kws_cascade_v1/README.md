# VieNeu-TTS Cascaded Vietnamese KWS (`vieneu_kws_cascade_v1`)

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Path:** `experiments/vieneu_kws_cascade_v1/`  
**Dataset Reference:** `experiments/synthetic_kws_dataset_vieneu_6class_v2/`  
**Stage 1 Reference:** `experiments/vieneu_kws_v2/` (CNN v2)  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## Overview

This repository directory contains the models, training scripts, evaluation code, predictions, and reports for the **Two-Stage Cascaded Keyword Spotting System** for Vietnamese speech commands (`TRAI`, `PHAI`, `LEN`, `XUONG`).

The system couples:
1. **Stage 1 (Proposer):** The frozen 6-class CNN v2 (`VietnameseKWSNet`, $359\text{k}$ parameters) evaluating $1.0\text{ s}$ audio windows every $100\text{ ms}$ ($1.26\text{ ms}$ CPU).
2. **Stage 2 (Temporal Verifier):** A compact Temporal Convolutional Network (`TemporalKWSVerifier`, $340\text{k}$ parameters) with dilated 1D convolutions (dilations $1, 2, 4, 8$) operating on $1.2\text{ s}$ acoustic crops to verify target commands against hard negatives and phonetically confusable words.

---

## Directory Structure

```
experiments/vieneu_kws_cascade_v1/
├── README.md                                 <- Master documentation (this file)
├── config.json                               <- Cascade architecture and calibration configuration
├── train_verifier.py                         <- Stage 2 verifier training with hard-negative mining
├── evaluate_cascade.py                       <- Cascade streaming evaluation and calibration
├── checkpoints/                              <- Stage 2 verifier checkpoints
│   ├── best_verifier_seed42.pt               <- Seed 42 checkpoint (Val Macro F1: 91.86%)
│   ├── best_verifier_seed43.pt               <- Seed 43 checkpoint (Val Macro F1: 92.14%)
│   └── best_verifier_seed44.pt               <- Seed 44 checkpoint (Val Macro F1: 91.68%)
├── logs/                                     <- Training loss and metric histories
│   ├── history_verifier_seed42.json
│   ├── history_verifier_seed43.json
│   └── history_verifier_seed44.json
├── calibration/                              <- Validation calibration sweeps
│   ├── validation_calibration_records.json   <- Grid sweep over (θ1, θ2)
│   └── selected_thresholds.json              <- Selected Pareto compromise thresholds
├── predictions/                              <- Evaluation output datasets
│   ├── evaluation_summary.json               <- Consolidated evaluation JSON across all seeds
│   ├── test_cascade_predictions_seed42.csv   <- Per-event streaming predictions
│   ├── test_cascade_predictions_seed43.csv
│   └── test_cascade_predictions_seed44.csv
└── reports/                                  <- Detailed analytical reports
    ├── EXISTING_METRICS_AUDIT.md             <- Independent audit of Phase 33–35 evaluation protocol
    ├── CASCADE_TRAINING_AND_EVALUATION_REPORT.md <- Complete training and held-out test report
    ├── STREAMING_READINESS_REPORT.md         <- Continuous streaming false alarm analysis
    └── NEXT_STEP_DECISION.md                 <- 8-dimension verdicts and future strategy
```

---

## Summary of Results

### Continuous Streaming Evaluation on Held-Out Test Data

| Metric | Stage 1 Alone (CNN v2) | Cascade (Stage 1 + Stage 2) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Command Stream Recall** | $99.80\% \pm 0.35\%$ | **$96.25\% \pm 2.73\%$** | Preserved ($>96\%$) |
| **Command Stream Precision** | $48.90\% \pm 1.05\%$ | **$97.27\% \pm 1.34\%$** | **$+48.37\%$** (Near-doubled!) |
| **Command Stream False Alarms** | $167.0 \pm 7.0$ | **$4.33 \pm 2.31$** | **$-97.4\%$ reduction** |
| **Mean Command Detection Latency** | $403.2\text{ ms} \pm 10.9\text{ ms}$ | **$403.7\text{ ms} \pm 11.9\text{ ms}$** | $+0.5\text{ ms}$ |
| **Non-Target Stream False Alarms** | $287.0 \pm 281.5$ | **$17.33 \pm 8.50$** | **$-94.0\%$ reduction** |
| **Non-Target Streaming FA / Hour** | $1,146.0 \pm 1,123.9\text{ FA/hr}$ | **$69.20 \pm 34.00\text{ FA/hr}$** | **$-94.0\%$ reduction** |
| **Single-Window CPU Latency** | $1.26\text{ ms}$ | **$4.47\text{ ms}$** (Total pipeline) | $<5\text{ ms}$ on CPU |

---

## Reproduction Instructions

To reproduce the training, calibration, and evaluation of the cascade from scratch:

```bash
# 1. Run unit and regression tests
python -m pytest tests/test_vieneu_kws_cascade.py -v

# 2. Train the Stage 2 Temporal Verifier across 3 seeds (seeds 42, 43, 44)
python experiments/vieneu_kws_cascade_v1/train_verifier.py

# 3. Run validation calibration and held-out test stream evaluation
python experiments/vieneu_kws_cascade_v1/evaluate_cascade.py
```
