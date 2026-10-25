# Model Card: Vietnamese Keyword Spotting CNN (Release Candidate v11)

**Release Identifier:** `variant_v11_hard_negative_weighted_seed42`  
**Bundle Version:** Phase 13.4 Final Release Candidate  
**Candidate Status:** `FROZEN_CNN_RELEASE_CANDIDATE`  
**Operational Assessment:** `NOT_VALIDATED_ON_NATURAL_SPEECH` | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
**Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**ONNX Export SHA-256:** `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`  
**Operating Decision Threshold ($\theta^*$):** `0.20`

---

## 1. Model Summary & Architecture

This repository releases the final frozen candidate CNN for four-keyword Vietnamese directional command spotting (`TRAI` [trái], `PHAI` [phải], `LEN` [lên], `XUONG` [xuống]) with a non-target rejection class (`REJECT`).

### Architecture Specifications (`CorrectedVietnameseKWSNet`)
- **Type:** 4-Block 2D Convolutional Neural Network with Adaptive Pooling.
- **Input Representation:** Log-Mel Spectrogram of shape `[1, 1, 40, 101]`.
- **Block 1:** Conv2D (1 $\to$ 32, k=3, p=1, bias=False) $\to$ BatchNorm2D(32) $\to$ ReLU $\to$ MaxPool2d(2, 2) [Shape: 20 $\times$ 50]
- **Block 2:** Conv2D (32 $\to$ 64, k=3, p=1, bias=False) $\to$ BatchNorm2D(64) $\to$ ReLU $\to$ MaxPool2d(2, 2) [Shape: 10 $\times$ 25]
- **Block 3:** Conv2D (64 $\to$ 64, k=3, p=1, bias=False) $\to$ BatchNorm2D(64) $\to$ ReLU $\to$ MaxPool2d(2, 2) [Shape: 5 $\times$ 12]
- **Block 4:** Conv2D (64 $\to$ 128, k=3, p=1, bias=False) $\to$ BatchNorm2D(128) $\to$ ReLU $\to$ AdaptiveAvgPool2d((1, 1)) [Shape: 1 $\times$ 1]
- **Classifier:** Dropout($p=0.3$) $\to$ Linear(128 $\to$ 64) $\to$ ReLU $\to$ Linear(64 $\to$ 5).
- **Total Parameters:** 135,813 (FP32 size $\approx 543\text{ KB}$).

---

## 2. Preprocessing & Frontend Contract

To guarantee exact parity with training and confirmatory evaluation, input audio MUST adhere to the following contract:
1. **Target Audio:** 16,000 Hz, 1 channel (mono), 32-bit float PCM.
2. **Padding / Truncation:** Pad right with zeros if shorter than 16,000 samples; truncate to 16,000 samples if longer.
3. **Amplitude Normalization:**
   $$\text{if } \max(|x|) > 1.0 \implies x \leftarrow \frac{x}{\max(|x|)}; \quad \text{else } x \text{ is unmodified.}$$
   *Floor amplification of quiet signals is strictly prohibited.*
4. **Log-Mel Frontend:**
   - Sample Rate: 16,000 Hz
   - STFT: $N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$, periodic Hann window, center reflection padding.
   - Filterbank: 40 triangular Slaney Mel filters ($20.0 - 8000.0\text{ Hz}$), Slaney area normalization.
   - Compression: Natural logarithm with $\epsilon = 1.0 \times 10^{-6}$.
   - Feature Tensor Shape: `[1, 1, 40, 101]`.

---

## 3. Decision Logic & Calibrated Operating Threshold

$$\text{Command Probabilities} = \text{Softmax}(\text{Logits})[0:4] = [p_{\text{TRAI}}, p_{\text{PHAI}}, p_{\text{LEN}}, p_{\text{XUONG}}]$$
$$\text{If } \max(p) \ge \theta^*_{\text{cmd}} = 0.20 \implies \text{Class} = \operatorname{argmax}(p); \quad \text{Else } \text{Class} = \text{REJECT}$$

---

## 4. Confirmatory Evaluation Performance (Phase 13.3)

Evaluated in a single immutable pass across 1,000 unseen clips under **Protocol Amendment 13.1-A** (8 novel Northern voices, 0% speaker leakage, 0% lexical text overlap):

| Metric | Target Acceptance SLA | Actual Confirmatory Outcome | Status | 95% Wilson Score CI |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | $\ge 95.0\%$ | **99.20%** (992 / 1000) | **PASSED** | [98.4%, 99.6%] |
| **Aggregate Command Recall** | $\ge 95.0\%$ | **99.00%** (198 / 200) | **PASSED** | [96.4%, 99.7%] |
| **TRAI Recall** | $\ge 95.0\%$ | **98.00%** (49 / 50) | **PASSED** | [89.5%, 99.6%] |
| **PHAI Recall** | $\ge 95.0\%$ | **100.00%** (50 / 50) | **PASSED** | [92.9%, 100.0%] |
| **LEN Recall** | $\ge 95.0\%$ | **98.00%** (49 / 50) | **PASSED** | [89.5%, 99.6%] |
| **XUONG Recall** | $\ge 95.0\%$ | **100.00%** (50 / 50) | **PASSED** | [92.9%, 100.0%] |
| **Non-Target False Alarm Rate** | $\le 5.0\%$ | **0.75%** (6 / 800) | **PASSED** | [0.3%, 1.6%] |
| **Phonetic Minimal Pairs FAR** | Diagnostic | **0.00%** (0 / 200) | **PASSED** | [0.0%, 1.9%] |
| **Syntactic Carrier Collisions FAR** | Diagnostic | **1.50%** (3 / 200) | **PASSED** | [0.5%, 4.3%] |
| **Conversational Non-Targets FAR** | Diagnostic | **1.00%** (3 / 300) | **PASSED** | [0.3%, 2.9%] |
| **Acoustic Background / Silence FAR**| Diagnostic | **0.00%** (0 / 100) | **PASSED** | [0.0%, 3.7%] |
| **CPU Latency (Single Clip)** | Mean/p95 $\le 50$ ms | Mean **1.80 ms**, p95 **2.12 ms** | **PASSED** | Total pass: 1.91 s |

---

## 5. Scope & Critical Operational Limitations

1. **`NOT_VALIDATED_ON_NATURAL_SPEECH`**: The candidate was confirmed exclusively on high-fidelity synthetic voices. It has not been validated on natural human speech recordings or physical microphone feeds.
2. **Regional Accent Generalization**: Confirmatory validation was strictly performed on 8 novel **Northern Vietnamese (Bắc)** voices under Amendment 13.1-A. Central (Trung) and Southern (Nam) dialect generalization is **out of scope and pending** until independent voices or consented human recordings become available.
3. **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**: The model operates strictly on isolated, pre-segmented 1.0-second audio buffers. Continuous open-mic listening requires streaming VAD, ring-buffering, and keyword boundary detection not present in this candidate.
4. **Carrier Sentence Collisions**: In 1.5% of synthetic carrier sentences embedding keywords, the model triggered an acceptance due to high localized acoustic energy.
