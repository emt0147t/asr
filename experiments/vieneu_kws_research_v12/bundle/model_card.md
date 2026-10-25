# Model Card: Vietnamese Bounded-Utterance Keyword Spotter CNN

**Model Name:** `CorrectedVietnameseKWSNet`  
**Version:** `v11_hard_negative_weighted_seed42` (Phase 12 Frozen Candidate)  
**Release Date:** October 2026  
**Artifact SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**License:** Research & Evaluation Prototype  
**Governance Label:** **`FROZEN_CNN_CANDIDATE`**

---

## 1. Model Details
- **Architecture:** Compact 4-layer 2D Convolutional Neural Network with Batch Normalization, Max Pooling, and non-parametric Adaptive Average Pooling `AdaptiveAvgPool2d((1, 1))` followed by a 2-layer MLP classifier with dropout ($p=0.3$).
- **Parameters:** 138,469 float32 parameters (565 KB checkpoint size).
- **Frontend:** Integrated 40-channel Log-Mel Spectrogram (16 kHz sampling rate, 25 ms window length, 10 ms hop length, Slaney normalization).
- **Target Keywords:** `TRAI` (trái), `PHAI` (phải), `LEN` (lên), `XUONG` (xuống).
- **Rejection Class:** `REJECT` (encompassing non-target speech, minimal pairs, prefix compounds, and ambient noise).
- **Calibrated Decision Threshold:** $\theta_{\text{cmd}}^* = 0.20$ (validation-calibrated for optimal recall-FAR trade-off).

---

## 2. Intended Use & Scope
- **Intended Use:** Bounded-utterance single-word keyword spotting in applications where audio capture is explicitly bounded by push-to-talk, manual activation, or an upstream high-precision Voice Activity Detector (VAD).
- **Out-of-Scope & Prohibited Uses:**
  - **Continuous Open-Mic Operation:** The model is **not** validated for autonomous continuous streaming audio. Sliding-window false alarms accumulate rapidly without a secondary verifier or temporal state machine.
  - **Natural Speech Production:** The model has **not** been validated on live human physical microphone recordings.
  - **Unconstrained Dictation or ASR:** The model recognizes strictly the 4 target commands and rejects all other speech.

---

## 3. Training & Preprocessing
- **Training Corpus:** 2,000 1-second 16 kHz audio clips synthesized across 8 diverse Vietnamese TTS voices (`Adam`, `Minh Triết`, `Mỹ Duyên`, `Ngọc Huyền`, `Quang Sơn`, `Thanh Bình`, `Thái Sơn`, `Thục Đoan`):
  - 800 Target Commands (200 per keyword).
  - 1,200 Non-Target Distractors (480 target-prefix/suffix compounds, 560 historical minimal pairs and neutral phrases, 160 ambient noise clips).
- **Augmentation:** SpecAugment (frequency masking 1–8 bins, time masking 1–15 frames), random time shifts ($\pm 80\text{ ms}$), gain variation ($\pm 3\text{ dB}$), additive Gaussian noise (SNR 12–28 dB).
- **Loss Formulation:** Cross-entropy with label smoothing ($0.05$) and sample weights ($w_{\text{cmd}} = 1.5, w_{\text{hn}} = 1.5, w_{\text{sil}} = 0.5$).

---

## 4. Evaluation & Performance Benchmarks
Evaluated on an independent, fresh locked test set of 640 clips across 4 unseen held-out voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`):

| Evaluation Dimension | Metric Value | 95% Wilson Confidence Interval | Predeclared Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Command Recall (Mean)** | **100.00%** (160 / 160) | [97.66%, 100.00%] | $\ge 95.0\%$ | **PASSED** |
| - `TRAI` Recall | 100.00% (40 / 40) | [91.24%, 100.00%] | $\ge 95.0\%$ | **PASSED** |
| - `PHAI` Recall | 100.00% (40 / 40) | [91.24%, 100.00%] | $\ge 95.0\%$ | **PASSED** |
| - `LEN` Recall | 100.00% (40 / 40) | [91.24%, 100.00%] | $\ge 95.0\%$ | **PASSED** |
| - `XUONG` Recall | 100.00% (40 / 40) | [91.24%, 100.00%] | $\ge 95.0\%$ | **PASSED** |
| **Non-Target FAR** | **0.63%** (3 / 480) | [0.21%, 1.82%] | $\le 5.0\%$ | **PASSED** |
| - Prefix Compounds ($n=112$) | 0.89% (1 / 112) | [0.16%, 4.88%] | Minimal | *xuống hồ* (conf 0.25) |
| - Suffix Compounds ($n=96$) | 0.00% (0 / 96) | [0.00%, 3.80%] | Minimal | None |
| - Minimal Pair Tone ($n=96$) | 2.08% (2 / 96) | [0.57%, 7.27%] | Minimal | *xưởng thủ công*, *xương rồng* |
| - Consonants / Coda ($n=96$) | 0.00% (0 / 96) | [0.00%, 3.80%] | Minimal | None |
| - Ambient Silence ($n=80$) | 0.00% (0 / 80) | [0.00%, 4.54%] | 0.00% | None |
| **5-Class Accuracy** | **99.53%** (637 / 640) | [98.63%, 99.84%] | $\ge 95.0\%$ | **PASSED** |
| **Decision Latency (CPU)** | **1.89 ± 0.47 ms** | p95: 2.41 ms | $\le 50.0\text{ ms}$ | **PASSED** |

---

## 5. Known Limitations & Potential Biases
1. **Selection Bias Risk:** Candidate ranking in Phase 11 was performed by evaluating all candidate variants on the locked test. While weights and thresholds were tuned strictly on training/validation data, selecting the top-performing seed introduces multiple-testing variance ("winner's curse"). A confirmatory study on an unseen test distribution is required.
2. **Acoustic Generalization:** Performance is benchmarked exclusively on synthetic neural TTS. Natural speech exhibits wider prosodic, tonal, and dialectal variations that may alter false trigger rates.
3. **Low-Confidence Boundary:** The 3 non-target false triggers occurred in the margin $0.22 \le p \le 0.28$. At threshold $\theta \ge 0.30$, all false alarms drop to $0.0\%$.

---

## 6. Governance Status
- **Engineering Status:** `FROZEN_CNN_CANDIDATE`
- **Synthetic Speech Quality:** `HUMAN_LISTENER_VERIFIED`
- **Natural Speech Microphone Validation:** `NOT_VALIDATED_ON_NATURAL_SPEECH`
- **Continuous Open-Mic Readiness:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
