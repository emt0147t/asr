# Phase 30 & 31: Robustness Benchmark and Confidence Calibration Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Campaign:** `experiments/vieneu_kws_robustness_campaign/`  
**Model Architecture:** `VietnameseKWSNet` (Six-Class CNN, 107,334 parameters)  
**Trained Baseline Checkpoints:** Seeds 42, 43, 44 (`experiments/vieneu_cnn_6class/checkpoints/`)  
**Evaluation Dates:** October 2026  

---

## Governance & Integrity Boundaries

> [!WARNING]
> **GOVERNANCE STATUS: PENDING_HUMAN_LISTENER_VERIFICATION**  
> All audio evaluated in this campaign was generated synthetically by VieNeu-TTS v3 Turbo. The 80-clip subjective human listening review remains pending. Synthetic speech quality and naturalness have not been validated by human judges.

> [!CAUTION]
> **GOVERNANCE STATUS: NOT_VALIDATED_ON_NATURAL_SPEECH**  
> All robustness perturbations and multi-talker babble files evaluated here are controlled synthetic benchmark conditions. Performance metrics in this report do not establish operational field readiness on real human speech or physical acoustic environments.

> [!NOTE]
> **OUT-OF-CONDITION SILENCE DIAGNOSIS**  
> In accordance with project governance, `SILENCE` is evaluated as an out-of-condition diagnostic class. Silence audio contains no active speech energy; its noise perturbation is scaled to match the absolute noise RMS level of speech clips at corresponding target SNRs.

---

## 1. Executive Summary

This integrated evaluation campaign establishes the empirical operational boundaries of the six-class Vietnamese KWS CNN baseline across acoustic noise conditions and post-processing calibration:

1. **Noise Robustness Hierarchy (Phase 30):**
   - **Stationary Low-Frequency Noise (HVAC):** Remarkably resilient. Clean accuracy (92.08%) is maintained almost intact at 20 dB SNR (92.08% ± 0.34%) and 15 dB SNR (89.03% ± 4.06%). Even at 0 dB SNR, the model achieves 72.08% ± 10.38% accuracy.
   - **Multi-Talker Babble:** Moderate degradation at conversational levels (71.94% at 20 dB, 57.50% at 15 dB). Degrades rapidly below 15 dB (34.03% at 10 dB, 19.17% at 0 dB) as competitive speech formants trigger misclassifications.
   - **Broadband Stationary Noise (White & Pink):** Catastrophic degradation below 15 dB. While 20 dB retains ~48–52% accuracy, performance collapses to ~17–21% at 15 dB and ~17% (chance level 16.7%) at $\le 10$ dB. Broadband spectral energy obliterates Vietnamese unvoiced fricatives and affricates (`Tr`, `Ph`, `X`).

2. **Validation-Only Confidence Calibration (Phase 31):**
   - Without retraining, applying validation-calibrated thresholds ($\theta \in [0.60, 0.85]$) directly improves test set performance across all three seeds:
     - **Clean Test Accuracy:** Improves from **92.08% ± 0.68%** to **93.33% ± 0.83%** (+1.25%).
     - **Clean Test Macro-F1:** Improves from **91.36% ± 0.91%** to **93.02% ± 0.98%** (+1.66%).
     - **UNKNOWN False Command Activations:** Drops from **45.00% ± 6.12%** down to **33.33% ± 5.77%** (-11.67 percentage points).
     - **Target Command Recall:** Preserved at **98.33% ± 0.72%**.

---

## 2. Phase 30: Controlled Synthetic-Noise Robustness Benchmark

### 2.1 Benchmark Methodology
- **Test Dataset:** Held-out test split of 240 clips (40 per class across 4 unseen voices: `Thảo Nhi`, `Văn Dũng`, `Bảo Ngọc`, `Đức Trung`).
- **Perturbations:** 4 noise types $\times$ 5 SNR levels (20, 15, 10, 5, 0 dB) = 20 conditions (4,800 audio files).
- **Exact Active Region SNR:** Noise power scaled specifically against `kws_onset_sec` to `kws_onset_sec + kws_active_speech_duration_sec`. All speech audio verified to $|SNR_{achieved} - SNR_{target}| \le 0.05\text{ dB}$.
- **Seeds Evaluated:** Seeds 42, 43, and 44 checkpoints.

### 2.2 Test Set Robustness Summary Table (Mean ± Std over 3 Seeds)

| Acoustic Condition | SNR (dB) | Overall Accuracy (%) | Macro F1 (%) | Target Accuracy (%) | UNKNOWN FA Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | $\infty$ | **92.08 ± 0.68** | **91.36 ± 0.91** | **99.38 ± 0.62** | **45.00 ± 6.12** |
| **HVAC** | 20 dB | 92.08 ± 0.34 | 91.97 ± 0.20 | 94.17 ± 3.24 | 24.17 ± 11.24 |
| | 15 dB | 89.03 ± 4.06 | 89.09 ± 3.93 | 88.54 ± 7.57 | 18.33 ± 6.56 |
| | 10 dB | 82.64 ± 6.44 | 82.28 ± 6.58 | 82.92 ± 11.09 | 15.00 ± 5.40 |
| | 5 dB | 77.64 ± 8.78 | 77.38 ± 9.16 | 82.08 ± 10.82 | 15.00 ± 5.40 |
| | 0 dB | 72.08 ± 10.38 | 70.58 ± 11.40 | 79.17 ± 11.60 | 14.17 ± 4.71 |
| **Babble (Multi-talker)** | 20 dB | 71.94 ± 5.99 | 72.15 ± 6.04 | 75.21 ± 7.81 | 22.50 ± 12.75 |
| | 15 dB | 57.50 ± 7.17 | 54.68 ± 10.15 | 56.25 ± 6.87 | 10.83 ± 6.56 |
| | 10 dB | 34.03 ± 4.77 | 27.74 ± 6.41 | 31.67 ± 10.71 | 25.00 ± 28.58 |
| | 5 dB | 22.50 ± 2.97 | 13.78 ± 4.07 | 17.71 ± 12.01 | 35.83 ± 34.54 |
| | 0 dB | 19.17 ± 2.97 | 8.02 ± 3.15 | 14.79 ± 10.71 | 44.17 ± 39.07 |
| **Pink Noise** | 20 dB | 51.53 ± 14.17 | 45.03 ± 17.65 | 49.17 ± 5.11 | 35.00 ± 44.30 |
| | 15 dB | 31.94 ± 3.61 | 22.29 ± 4.54 | 42.71 ± 0.78 | 78.33 ± 19.61 |
| | 10 dB | 18.19 ± 0.39 | 7.80 ± 0.86 | 27.08 ± 0.78 | 99.17 ± 1.18 |
| | 5 dB | 16.81 ± 0.20 | 5.04 ± 0.39 | 25.00 ± 0.00 | 99.17 ± 1.18 |
| | 0 dB | 16.81 ± 0.20 | 5.04 ± 0.39 | 25.00 ± 0.00 | 99.17 ± 1.18 |
| **White Noise** | 20 dB | 47.64 ± 14.74 | 38.74 ± 18.07 | 45.42 ± 6.85 | 52.50 ± 33.60 |
| | 15 dB | 21.11 ± 2.27 | 12.16 ± 3.13 | 31.25 ± 2.84 | 98.33 ± 2.36 |
| | 10 dB | 17.08 ± 0.34 | 5.58 ± 0.66 | 25.62 ± 0.51 | 100.00 ± 0.00 |
| | 5 dB | 16.81 ± 0.20 | 5.04 ± 0.39 | 25.00 ± 0.00 | 99.17 ± 1.18 |
| | 0 dB | 17.78 ± 1.57 | 6.64 ± 2.66 | 24.17 ± 1.18 | 90.00 ± 14.14 |

---

## 3. Phase 31: Validation-Only Confidence Threshold Calibration

### 3.1 Policy Formulation
Given model predicted probabilities $p = [p_0, p_1, p_2, p_3, p_4, p_5]$ and threshold $\theta \in [0.50, 0.99]$:
$$\hat{y} = \arg\max_c p_c, \quad \text{conf} = \max_c p_c$$
$$\text{Calibrated Prediction } \hat{y}_{cal} = \begin{cases} 
\text{UNKNOWN } (4) & \text{if } \hat{y} \in \{0, 1, 2, 3\} \text{ and } \text{conf} < \theta \\
\hat{y} & \text{otherwise}
\end{cases}$$

### 3.2 Frozen Thresholds & Selection Rationale
Calibration was performed **strictly** on the validation split (clean + corrupted validation):

| Seed | Selected $\theta$ | Clean Val Target Recall | Clean Val UNK FA Rate | Clean Val Macro F1 | Selection Rationale |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | **0.60** | 93.12% | 37.50% | 88.87% | Pareto compromise: maintains Target Rec $\ge 90\%$, lowers UNK FA |
| **43** | **0.75** | 92.50% | 37.50% | 88.76% | Pareto compromise: maintains Target Rec $\ge 90\%$, lowers UNK FA |
| **44** | **0.85** | 91.88% | 27.50% | 90.11% | Pareto compromise: maintains Target Rec $\ge 90\%$, lowest UNK FA |

### 3.3 Held-Out Test Split Verification (Uncalibrated vs Calibrated)

| Seed | $\theta$ | Clean Accuracy | Clean Macro F1 | UNK False Alarm Rate | Target Command Recall | Rejection Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.60 | 92.08% $\rightarrow$ **92.50%** | 91.27% $\rightarrow$ **91.93%** | 45.0% $\rightarrow$ **40.0%** | 99.38% $\rightarrow$ **98.75%** | 1.2% |
| **43** | 0.75 | 92.92% $\rightarrow$ **94.17%** | 92.51% $\rightarrow$ **93.97%** | 37.5% $\rightarrow$ **30.0%** | 98.75% $\rightarrow$ **98.75%** | 1.2% |
| **44** | 0.85 | 91.25% $\rightarrow$ **93.33%** | 90.29% $\rightarrow$ **93.15%** | 52.5% $\rightarrow$ **30.0%** | 100.0% $\rightarrow$ **97.50%** | 5.4% |
| **Mean** | --- | **92.08% $\rightarrow$ 93.33%** | **91.36% $\rightarrow$ 93.02%** | **45.0% $\rightarrow$ 33.3%** | **99.38% $\rightarrow$ 98.33%** | **2.6%** |

---

## 4. Architectural & Failure Analysis

1. **Stationary vs Non-Stationary Noise:**
   - The standard CNN without noise augmentation trains almost entirely on clean TTS speech. In stationary HVAC noise, formant frequencies ($F_1, F_2, F_3$) in the 500 Hz – 3500 Hz range remain visible above the 100–300 Hz air rumble.
   - In white/pink noise, constant energy across all 40 mel bands destroys contrast, flattening filter bank outputs and leading to random argmax distributions.
2. **Confidence Policy Dynamics:**
   - Softmax probabilities on clean speech are well-separated for true target commands (average confidence $\approx 0.96$).
   - UNKNOWN distractors that share phonemes with target commands (`Không` with `Xuống`, `Lưu` with `Lên`) produce lower average confidence ($\approx 0.62–0.78$). Setting $\theta \in [0.60, 0.85]$ successfully suppresses these marginal activations without hurting high-confidence target commands.
