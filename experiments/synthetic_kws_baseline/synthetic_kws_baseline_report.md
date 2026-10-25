# Synthetic-Only Vietnamese KWS Baseline Smoke Test Report

**Date:** 2026-10-15  
**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments\synthetic_kws_baseline`  
**Architecture:** `VietnameseTCResNet` (272,548 parameters)  
**Validation Profile:** `benchmark` (Enforced via `validate_split_manifests_preflight`)  
**Data Scope:** Strictly synthetic calibration fixtures (`calibration_acoustic_generator_v1`)  
**Target Classes (4):** `TRAI` (0), `PHAI` (1), `LEN` (2), `XUONG` (3)  

---

## 1. Executive Summary & Smoke Test Purpose

This experiment is a **purely technical smoke test** designed to validate the end-to-end training and evaluation pipeline on clean, synthetic 16 kHz audio files.

> [!WARNING]
> **CRITICAL LIMITATIONS — NOT EVIDENCE OF REAL-WORLD GENERALIZATION:**
> - **Synthetic Acoustic Nature:** 100% of audio clips used in this experiment were parametrically synthesized by `calibration_acoustic_generator_v1`. They do not contain natural human vocal tract acoustics, irregular glottal pulses, or native articulation subtleties.
> - **Phonetic Gap:** Dialect tags in the synthetic generator do **not** capture native phonetics, such as the authentic Northern Vietnamese voiceless alveolo-palatal affricate `/tɕ/` burst (critical for distinguishing *Trái* from confusers) or Central/Southern tone neutralization.
> - **Acoustic Environment Gap:** The audio contains zero authentic room reverberation, ambient background noise, or mobile-device microphone frequency coloring.
> - **Closed-Set Classification:** This experiment uses a closed 4-class softmax. It contains **no** `UNKNOWN` (negative conversational speech) or `SILENCE` (ambient noise) classes and therefore provides **zero measurement of False Alarm Rate (FAR) or streaming trigger reliability**.
> - **Conclusion:** This smoke test proves that the neural network pipeline, feature extractor, and split validator run correctly without runtime defects. **It does NOT constitute evidence of real-world Vietnamese accent recognition or mobile deployment readiness.**

---

## 2. Dataset Partitioning & Split Validation

- **Total Clips Used:** 1,000 accepted clips (250 per command)
- **Train Partition:** 712 clips (178 per class) across 14 synthetic speakers (`SPK_CALIB_001`–`014`)
- **Validation Partition:** 144 clips (36 per class) across 3 synthetic speakers (`SPK_CALIB_015`–`017`)
- **Held-Out Test Partition:** 144 clips (36 per class) across 3 synthetic speakers (`SPK_CALIB_018`–`020`)
- **Speaker Overlap Across Splits:** **0 (Strictly Pairwise Disjoint)**
- **Preflight Invariant Check:** `validate_split_manifests_preflight(profile='benchmark')` returned `VERIFIED_DISJOINT`.

---

## 3. Training Configuration & Convergence

- **Optimizer:** AdamW (lr=0.001, weight_decay=1e-4)
- **Learning Rate Schedule:** CosineAnnealingLR (T_max=20)
- **Batch Size:** 32
- **Epochs Trained:** 20
- **Execution Device:** CPU
- **Total Training Wallclock Time:** 20.11 seconds
- **Best Validation Macro-F1:** 1.0000
- **Saved Checkpoint:** `experiments\synthetic_kws_baseline\checkpoints\best_synthetic_kws_model.pt`

---

## 4. Held-Out Test Evaluation Results (Unseen Synthetic Speakers)

- **Held-Out Test Accuracy:** **90.28%**
- **Held-Out Test Macro-F1:** **0.8990**

### 4.1 Per-Class Performance Breakdown

| Class ID | Command Label | Support | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **`TRAI`** | 36 | 1.0000 | 0.6111 | 0.7586 |
| 1 | **`PHAI`** | 36 | 0.7200 | 1.0000 | 0.8372 |
| 2 | **`LEN`** | 36 | 1.0000 | 1.0000 | 1.0000 |
| 3 | **`XUONG`** | 36 | 1.0000 | 1.0000 | 1.0000 |
| - | **Macro Average** | **144** | **0.9300** | **0.9028** | **0.8990** |

### 4.2 Confusion Matrix

```text
               Predicted
         TRAI    PHAI     LEN   XUONG
  TRAI      22      14       0       0
  PHAI       0      36       0       0
   LEN       0       0      36       0
 XUONG       0       0       0      36
```

---

## 5. Measured CPU Inference Latency

- **Average Inference Latency:** 1.50 ms
- **Median Latency (P50):** 1.42 ms
- **90th Percentile Latency (P90):** 1.64 ms
- **99th Percentile Latency (P99):** 3.01 ms
- **Note on Latency Claims:** While single-inference latency on desktop CPU is low, mobile-device streaming latency requires continuous ring-buffer sliding window benchmarking on target edge hardware (e.g. ARM Cortex / Snapdragon / Apple Silicon) under thermal throttling conditions.

---

## 6. Reproducibility Command

```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_baseline/train_and_eval_synthetic.py --validation_profile benchmark --epochs 20
```
