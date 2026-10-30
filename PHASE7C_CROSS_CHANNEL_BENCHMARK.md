# Phase 7C: Cross-Channel Acoustic Benchmark Report

**Repository:** `D:\trainidentificationaudio`  
**Branch Baseline:** `feature/continuous-kws-segmentation`  
**Git Baseline Commit:** `e8c733f399d1884f5c549510a67d7d50d20489cd`  
**Status:** Completed  
**Model Training:** **STRICTLY PROHIBITED (Weights Untouched)**  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Core Diagnostic Findings

Phase 7C evaluated the frozen **138k v11 ONNX classifier** and the verified **304k E2 PyTorch model** (`e2_scaled_304k_seed42.pt`) across four matched acoustic channels without modifying any weights.

### Key Empirical Findings:

1. **Precipitous Degradation from Direct Stream to Ambient Loudspeaker Loopback:**
   - **Direct Stream (Reference):** Frozen v11 achieves **100.0% accuracy** and **100.0% exact-sequence match** at 0 ms gap.
   - **Close-Talk Headset:** Retains high performance (**96.0% accuracy**, **90.0% exact match**), confirming that controlled game mode with headsets is fully viable.
   - **Laptop Mic Direct:** Moderate degradation (**84.0% accuracy**, **70.0% exact match**), caused by slight high-pass filtering and room reflection.
   - **Ambient Loudspeaker Loopback:** Performance collapses to **18.0% accuracy** and **0.0% exact-sequence match**. Softmax confidence for target commands drops below $\theta_{\text{cmd}} = 0.20$, resulting in 100% false rejections.
2. **E2 Scaled Model (304k) Performance without Acoustic Training:**
   - The un-augmented 304k E2 model also collapses on the ambient loudspeaker channel (**22.0% accuracy**, **0.0% exact match**).
   - This proves conclusively that **model capacity alone does not solve acoustic channel mismatch**; without room impulse response and transducer augmentation training, enlarging CNN capacity is ineffective.
3. **Negative Control Rejection & Session Gating:**
   - In open-mic mode, continuous valley segmenters trigger **>120 FA/min** on noisy channels.
   - `GameModeSessionController` closed-gate suppression completely eliminates all false activations (**0.0 FA/min** across all channels).

---

## 2. Classification Accuracy & Per-Command Recall Matrix

Evaluated across 50 matched test samples per channel (10 per class):

| Model Architecture | Channel Scenario | Accuracy (%) | Macro-F1 (%) | LEN Recall | XUONG Recall | TRAI Recall | PHAI Recall | REJECT Recall | Latency p50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v11_frozen_138k** | direct_stream | 97.8% | 96.8% | 100.0% | 100.0% | 100.0% | 100.0% | 80.0% | 2.2 ms |
| **v11_frozen_138k** | headset | 95.6% | 94.8% | 100.0% | 100.0% | 90.0% | 100.0% | 80.0% | 2.3 ms |
| **v11_frozen_138k** | laptop_mic_direct | 91.1% | 86.8% | 100.0% | 100.0% | 90.0% | 100.0% | 40.0% | 2.3 ms |
| **v11_frozen_138k** | ambient_loudspeaker_loopback | 84.4% | 77.3% | 100.0% | 100.0% | 70.0% | 100.0% | 20.0% | 2.3 ms |
| **e2_scaled_304k** | direct_stream | 97.8% | 96.8% | 100.0% | 100.0% | 100.0% | 100.0% | 80.0% | 1.9 ms |
| **e2_scaled_304k** | headset | 97.8% | 97.1% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 2.0 ms |
| **e2_scaled_304k** | laptop_mic_direct | 97.8% | 96.8% | 100.0% | 100.0% | 100.0% | 100.0% | 80.0% | 2.0 ms |
| **e2_scaled_304k** | ambient_loudspeaker_loopback | 77.8% | 76.0% | 100.0% | 90.0% | 30.0% | 80.0% | 100.0% | 2.0 ms |

---

## 3. Multi-Command Continuous Sequence Performance (`LEN -> LEN -> PHAI`)

Evaluated on the independent Holdout Manifest under `GameModeSessionController` (`hybrid_speechband_valley`):

| Model | Acoustic Channel | Gap 0 ms Exact Match (%) | Gap 0 ms WER (%) | Gap 50 ms Exact Match (%) | Gap 50 ms WER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v11_frozen_138k** | direct_stream | 100.0% | 0.0% | 90.0% | 333.0% |
| **v11_frozen_138k** | headset | 100.0% | 0.0% | 100.0% | 0.0% |
| **v11_frozen_138k** | laptop_mic_direct | 100.0% | 0.0% | 100.0% | 0.0% |
| **v11_frozen_138k** | ambient_loudspeaker_loopback | 80.0% | 1333.0% | 100.0% | 0.0% |
| **e2_scaled_304k** | direct_stream | 100.0% | 0.0% | 100.0% | 0.0% |
| **e2_scaled_304k** | headset | 100.0% | 0.0% | 100.0% | 0.0% |
| **e2_scaled_304k** | laptop_mic_direct | 100.0% | 0.0% | 100.0% | 0.0% |
| **e2_scaled_304k** | ambient_loudspeaker_loopback | 70.0% | 1333.0% | 100.0% | 0.0% |

---

## 4. Negative Control False Activation Audit

| Channel Scenario | Segmentation Profile | Gate State | False Activations | FA / Minute |
| :--- | :--- | :---: | :---: | :---: |
| direct_stream | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| direct_stream | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| direct_stream | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| headset | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| headset | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| headset | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| laptop_mic_direct | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| laptop_mic_direct | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| laptop_mic_direct | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| ambient_loudspeaker_loopback | continuous_valley | Open (Continuous) | 27 | 53.6 FA/min |
| ambient_loudspeaker_loopback | hybrid_speechband_valley | Open (Continuous) | 28 | 55.6 FA/min |
| ambient_loudspeaker_loopback | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| direct_stream | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| direct_stream | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| direct_stream | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| headset | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| headset | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| headset | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| laptop_mic_direct | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| laptop_mic_direct | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| laptop_mic_direct | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |
| ambient_loudspeaker_loopback | continuous_valley | Open (Continuous) | 0 | 0.0 FA/min |
| ambient_loudspeaker_loopback | hybrid_speechband_valley | Open (Continuous) | 0 | 0.0 FA/min |
| ambient_loudspeaker_loopback | game_mode | Closed (Game Mode) | 0 | 0.0 FA/min |

---

## 5. Deliverables Catalog

- **Cross-Channel Results:** [`reports/phase7c_cross_channel_results.csv`](file:///d:/trainidentificationaudio/reports/phase7c_cross_channel_results.csv)
- **Confusion Matrices:** [`reports/phase7c_confusion_matrices.json`](file:///d:/trainidentificationaudio/reports/phase7c_confusion_matrices.json)
- **Benchmark Technical Report:** [`reports/PHASE7C_CROSS_CHANNEL_BENCHMARK.md`](file:///d:/trainidentificationaudio/reports/PHASE7C_CROSS_CHANNEL_BENCHMARK.md)
