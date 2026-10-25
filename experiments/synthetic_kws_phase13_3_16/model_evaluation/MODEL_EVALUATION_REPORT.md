# Phase 15 Model Evaluation Report: TC-ResNet-14 on V5.2 Synthetic Release Candidate

**Architecture:** `VietnameseTCResNet` (TC-ResNet-14, 192 features)  
**Dataset:** `synthetic_kws_v5_2_release_candidate` (3,000 clips: 2,100 Train, 450 Val, 450 Test)  
**Date:** 2026-10-16  
**Verdict:** **`TRAINING_EVALUATION_PASS`**  

---

## 1. Executive Summary

Phase 15 executed controlled, reproducible training of the production `VietnameseTCResNet` architecture on the 3,000-clip V5.2 release candidate dataset across 3 independently seeded runs (Seeds 42, 43, 44). The frontend extracted standard 40-band Log-Mel filterbanks (16 kHz, 25ms window, 10ms hop). Checkpoints were selected strictly using validation loss on disjoint validation speakers.

Evaluation on the held-out test split (450 clips with completely unseen synthetic speakers and disjoint seeds) achieved **$100.00\% \pm 0.00\%$** test accuracy and **$100.00\%$** Macro-F1 across all 3 seeds. Probability calibration was excellent with Brier score of $0.0005 \pm 0.0003$ and Expected Calibration Error (ECE) of $0.0014 \pm 0.0007$.

However, continuous streaming target-absent evaluation revealed **$424.57\text{ false triggers / hour}$** on transitional window boundaries, reinforcing the foundational finding that high holdout classification accuracy on isolated clips does **NOT** imply uncalibrated rejection reliability on continuous audio.

---

## 2. Training Setup & Hyperparameters

* **Model:** TC-ResNet-14 (`VietnameseTCResNet`, 192 output features, dropout $p=0.2$)
* **Frontend:** `LogMelFeatureExtractor` ($40$ mels, $N_{FFT}=512$, $win=400$, $hop=160$, $f_{min}=20$, $f_{max}=8000$)
* **Input Tensor Shape:** `(Batch, 1, 40, 101)`
* **Loss Function:** PyTorch `CrossEntropyLoss`
* **Optimizer:** AdamW ($lr=1.0\times 10^{-3}$, weight decay $=1.0\times 10^{-4}$)
* **Schedule:** `CosineAnnealingLR` ($T_{max}=25$, $\eta_{min}=1.0\times 10^{-5}$)
* **Epochs & Batch Size:** 25 epochs, batch size $=64$
* **Validation Checkpoint Rule:** Save checkpoint with minimum validation loss (no test set leakage)

---

## 3. Multi-Seed Test Set Results (450 Held-Out Clips)

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Best Val Loss Epoch** | Epoch 25 | Epoch 25 | Epoch 20 | Epoch $23.3 \pm 2.4$ |
| **Best Val Loss** | $0.0001$ | $0.0001$ | $0.0002$ | $0.00013 \pm 0.00005$ |
| **Optimal Val Threshold $\tau^*$** | $0.500$ | $0.500$ | $0.500$ | $0.500 \pm 0.000$ |
| **Test Accuracy (Standard)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **$100.00\% \pm 0.00\%$** |
| **Test Accuracy (Gated $\tau^*$)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **$100.00\% \pm 0.00\%$** |
| **Test Macro F1** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **$100.00\% \pm 0.00\%$** |
| **Brier Score** | $0.0003$ | $0.0004$ | $0.0008$ | **$0.0005 \pm 0.0003$** |
| **ECE (10 bins)** | $0.0008$ | $0.0012$ | $0.0022$ | **$0.0014 \pm 0.0007$** |

### Per-Class Test Breakdown (Seed 42 Reference)

| Class | Precision | Recall | F1-Score | Support | Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | $1.0000$ | $1.0000$ | $1.0000$ | 75 | 0 |
| **`PHAI`** | $1.0000$ | $1.0000$ | $1.0000$ | 75 | 0 |
| **`LEN`** | $1.0000$ | $1.0000$ | $1.0000$ | 75 | 0 |
| **`XUONG`** | $1.0000$ | $1.0000$ | $1.0000$ | 75 | 0 |
| **`UNKNOWN`** | $1.0000$ | $1.0000$ | $1.0000$ | 75 | 0 |
| **`SILENCE`** | $1.0000$ | $1.0000$ | $1.0000$ | 75 | 0 |

### `TRAI` vs `PHAI` Error Analysis
* `TRAI` predicted as `PHAI`: **0**
* `PHAI` predicted as `TRAI`: **0**
* Symmetric error parity is strictly maintained (0/75 errors in each direction).

---

## 4. Synthetic Continuous Streaming False-Trigger Analysis

To evaluate false-alarm susceptibility during continuous audio playback:
* **Audio Stream:** 150 non-target test clips (75 `UNKNOWN` + 75 `SILENCE`) concatenated with random ambient gap pauses ($0.1\text{ s}$ to $0.4\text{ s}$) yielding a continuous $186.54\text{ second}$ target-absent stream.
* **Inference Pipeline:** 1.0s sliding window with a $100\text{ ms}$ hop (1,600 samples) across the stream.
* **Findings:**
  - Total Target-Absent Duration: $186.54\text{ s}$ ($3.11\text{ min}$)
  - False Trigger Count: **22 activations** (where transitional window boundary logits produced a target class prediction with confidence $\ge 0.500$)
  - **False Trigger Rate:** **$424.57\text{ false triggers / hour}$** at $\tau = 0.500$.
* **Interpretation:** While isolated test clip classification is $100\%$, streaming window boundary transitions between speech and ambient pauses frequently trigger false alarms if confidence thresholds are not adaptively elevated or smoothed with multi-frame temporal confirmation.

---

## 5. Scientific Limitations & Verdict

> [!WARNING]
> **Key Scientific Limitations:**
> 1. **Synthetic-Only Benchmark:** 100% test accuracy is achieved on synthetic held-out distributions (simulated vocal tracts). It **DOES NOT** prove model performance on human voices, noisy mobile microphones, or real-world reverberation.
> 2. **Streaming Calibration Required:** A simple static confidence threshold $\tau = 0.500$ is insufficient for streaming deployment and causes substantial boundary false alarms ($424.57\text{ FP/hr}$).
> 3. **Retraining on Real Speech Deferred:** Human data collection remains blocked pending formal authorization.

### Phase 15 Verdict: **`TRAINING_EVALUATION_PASS`**
