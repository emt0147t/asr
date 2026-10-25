# VieNeu-TTS Six-Class KWS CNN v2

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Path:** `experiments/vieneu_kws_v2/`  
**Dataset Reference:** `experiments/synthetic_kws_dataset_vieneu_6class_v2/`  
**Governance Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## Overview

This repository directory contains the trained models, evaluation pipeline, predictions, and reports for the **second-generation six-class convolutional neural network (CNN v2)** for Vietnamese keyword spotting (commands: `TRAI`, `PHAI`, `LEN`, `XUONG`, plus `UNKNOWN` and `SILENCE`).

CNN v2 incorporates train-only multi-condition noise augmentation (white, pink, HVAC, babble at 5–20 dB SNR), SpecAugment, and time shifting to resolve the catastrophic broadband noise collapse observed in CNN v1.

---

## Directory Structure

```
experiments/vieneu_kws_v2/
├── README.md                                 <- Master documentation (this file)
├── checkpoints/                              <- Trained model weights
│   ├── best_checkpoint_seed42.pt             <- Checkpoint for Seed 42 (Epoch 24)
│   ├── best_checkpoint_seed43.pt             <- Checkpoint for Seed 43 (Epoch 25)
│   └── best_checkpoint_seed44.pt             <- Checkpoint for Seed 44 (Epoch 24)
├── logs/                                     <- Training loss and metric histories
│   ├── history_seed42.json
│   ├── history_seed43.json
│   └── history_seed44.json
├── calibration/                              <- Validation confidence thresholds
│   ├── validation_calibration_records.json   <- Sweep records (θ = 0.10 to 0.95)
│   └── selected_thresholds.json              <- Validation-selected Pareto thresholds
├── predictions/                              <- Evaluation output datasets
│   ├── evaluation_summary.json               <- Consolidated evaluation JSON across all seeds
│   ├── test_predictions_seed42.csv           <- Clean & calibrated predictions
│   ├── test_predictions_seed43.csv
│   └── test_predictions_seed44.csv
├── reports/                                  <- Detailed analytical reports
│   ├── EXISTING_CAMPAIGN_AUDIT.md            <- Part A: Independent audit of Phase 30–32
│   ├── V2_TRAINING_AND_ROBUSTNESS_REPORT.md  <- Comprehensive training & noise results
│   ├── STREAMING_READINESS_REPORT.md         <- Temporal streaming & false alarm analysis
│   └── NEXT_STEP_DECISION.md                 <- 7-dimension verdicts & cascaded architecture
├── train.py                                  <- Training pipeline with train-only augmentation
└── evaluate.py                               <- Calibration, noise testing, streaming simulation
```

---

## Summary of Results

### Clean Test Evaluation (Held-Out Test Split, N=320)

| Metric | Seed 42 ($\theta=0.60$) | Seed 43 ($\theta=0.50$) | Seed 44 ($\theta=0.70$) | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Calibrated Test Accuracy** | $93.12\%$ | $91.88\%$ | $94.69\%$ | **$93.23\% \pm 1.41\%$** |
| **Calibrated Macro F1** | $93.21\%$ | $91.97\%$ | $94.70\%$ | **$93.29\% \pm 1.36\%$** |
| **Target Command Recall** | $100.0\%$ | $99.38\%$ | $99.38\%$ | **$99.58\% \pm 0.36\%$** |
| **SILENCE False Activation** | $0.0\%$ ($0/80$) | $0.0\%$ ($0/80$) | $0.0\%$ ($0/80$) | **$0.0\% \pm 0.0\%$** |
| **UNKNOWN False Activation** | $27.50\%$ ($22/80$) | $31.25\%$ ($25/80$) | $20.00\%$ ($16/80$) | **$26.25\% \pm 5.73\%$** |
| **Single-Window CPU Latency** | $1.21\text{ ms}$ | $1.38\text{ ms}$ | $1.19\text{ ms}$ | **$1.26\text{ ms} \pm 0.10\text{ ms}$** |

### Noise Robustness (Mean Accuracy across Seeds)

| Condition | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **White Noise** | $85.8\%$ | $86.4\%$ | $84.5\%$ | $76.5\%$ | **$64.8\%$** *(vs 17.8% in v1)* |
| **Pink Noise** | $87.2\%$ | $86.2\%$ | $85.1\%$ | $78.0\%$ | **$65.5\%$** *(vs 16.8% in v1)* |
| **HVAC Hum** | $90.9\%$ | $90.5\%$ | $90.5\%$ | $89.8\%$ | **$88.5\%$** *(vs 72.1% in v1)* |
| **Babble Noise** | $87.8\%$ | $87.5\%$ | $84.2\%$ | $79.4\%$ | **$67.4\%$** *(vs 19.2% in v1)* |

### Streaming Simulation

| Metric | Target | Achieved (CNN v2) | Status |
| :--- | :---: | :---: | :---: |
| **Command Stream Recall** | $\ge 95\%$ | $98.1\% – 99.4\%$ | **PASS** |
| **Trigger Latency** | $\le 600\text{ ms}$ | $411.6 – 435.9\text{ ms}$ | **PASS** |
| **Silence Stream FA/hr** | $\le 1.0$ | $0.0\text{ FA/hr}$ | **PASS** |
| **Non-target Speech Stream FA/hr** | $\le 1.0$ | $223.6 – 1,130.1\text{ FA/hr}$ | **FAIL** |

---

## Reproduction Commands

To reproduce the training and evaluation of CNN v2 from scratch:

```bash
# 1. Run unit test suite
pytest tests/test_vieneu_kws_v2.py -v

# 2. Retrain all three seeds
python experiments/vieneu_kws_v2/train.py

# 3. Re-run validation calibration, held-out test evaluation, noise suites, and streaming
python experiments/vieneu_kws_v2/evaluate.py
```
