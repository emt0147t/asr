# Phase 40: Independent Pipeline Audit and Protocol Lock

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Path:** [`experiments/vieneu_kws_research_v2/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2)  
**Historical References Audited:**
- Stage 1 CNN v2: [`experiments/vieneu_kws_v2/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2)
- Stage 2 Temporal Verifier: [`experiments/vieneu_kws_cascade_v1/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1)
- VAD Ablation Campaign: [`experiments/vieneu_kws_vad_ablation_v1/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_vad_ablation_v1)
**Date:** 2026-10-18  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Audit Objectives

Phase 40 conducts a comprehensive, rigorous source-code and artifact audit of the Vietnamese keyword spotting pipeline before launching new model training or dataset construction. The objective is to resolve any architectural discrepancies between prior written reports and authoritative source code, verify exact mathematical formulas and denominators, explain historical anomalies, and formally lock the inference, streaming, and evaluation interfaces for Phases 41–44.

### Audit Verdict: **PASSED (ALL PROTOCOLS VERIFIED AND LOCKED)**

---

## 2. Authoritative Architecture & Parameter Counts

### 2.1 Stage 1 Model Architecture: `VietnameseKWSNet`
- **Source Code Definition:** [`src/models/kws_cnn.py`](file:///d:/trainidentificationaudio/src/models/kws_cnn.py#L11-L64)
- **Class Name:** `VietnameseKWSNet`
- **Architecture Structure:**
  - Block 1: `Conv2d(1, 32, k=3, p=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Shape: $40 \times 101 \rightarrow 20 \times 50$)
  - Block 2: `Conv2d(32, 64, k=3, p=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Shape: $20 \times 50 \rightarrow 10 \times 25$)
  - Block 3: `Conv2d(64, 64, k=3, p=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Shape: $10 \times 25 \rightarrow 5 \times 12$)
  - Block 4: `Conv2d(64, 128, k=3, p=1)` $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d((1, 1))` (Shape: $5 \times 12 \rightarrow 1 \times 1$)
  - Dense Head: `Dropout(0.3)` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(64, 6)`
- **Input Dimensions:** `(Batch, 1, 40, 101)` ($1.0\text{ s}$ audio at $16,000\text{ Hz}$)
- **Output Classes:** $6$ (`TRAI`: 0, `PHAI`: 1, `LEN`: 2, `XUONG`: 3, `UNKNOWN`: 4, `SILENCE`: 5)
- **Trainable Parameters:** **138,534** parameters (verified via `sum(p.numel() for p in model.parameters() if p.requires_grad)`).

---

### 2.2 Stage 2 Model Architecture: `TemporalKWSVerifier` (TCN)
- **Source Code Definition:** [`experiments/vieneu_kws_cascade_v1/train_verifier.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1/train_verifier.py#L52-L115)
- **Class Name:** `TemporalKWSVerifier`
- **Architecture Structure:**
  - **Frontend (2D CNN Frequency Reducer):**
    - `Conv2d(1, 32, k=(5, 3), p=(2, 1))` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d((2, 1))` (Mel $40 \rightarrow 20$)
    - `Conv2d(32, 64, k=(5, 3), p=(2, 1))` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d((2, 1))` (Mel $20 \rightarrow 10$)
    - `Conv2d(64, 64, k=(5, 3), p=(2, 1))` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d((1, None))` (Mel $10 \rightarrow 1$)
    - Output preserves time dimension $T=121$, reducing frequency to 1 channel: `(Batch, 64, 121)`.
  - **Temporal Sequence Backbone (TCN):**
    - 4 residual blocks with exponentially increasing dilation ($d = 1, 2, 4, 8$):
      - Block 1: `TCNResidualBlock(in=64, out=64, dilation=1)`
      - Block 2: `TCNResidualBlock(in=64, out=64, dilation=2)`
      - Block 3: `TCNResidualBlock(in=64, out=64, dilation=4)`
      - Block 4: `TCNResidualBlock(in=64, out=64, dilation=8)`
      - Each `TCNResidualBlock` consists of two dilated `Conv1d(k=3, p=dilation, d=dilation)` layers, `BatchNorm1d`, `Dropout(0.1)`, and residual addition.
  - **Classification Head:**
    - `AdaptiveAvgPool1d(1)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(64, 32)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(32, 5)`
- **Input Dimensions:** `(Batch, 1, 40, 121)` ($1.2\text{ s}$ acoustic context crop at $16,000\text{ Hz}$)
- **Output Classes:** $5$ (`TRAI`: 0, `PHAI`: 1, `LEN`: 2, `XUONG`: 3, `REJECT`: 4)
- **Trainable Parameters:** **340,773** parameters.

---

### 2.3 Critical Errata: Architectural Discrepancy Resolution
> [!IMPORTANT]
> **Errata Resolution on Stage 2 Verifier Architecture:**  
> In certain Phase 38 and Phase 39 summary documents (including `phase39_master_summary.md` and portions of `VAD_ABLATION_REPORT.md`), the Stage 2 verifier was colloquially described as a "BiGRU" verifier.  
> 
> **Authoritative Finding:** The source code and trained checkpoint files unequivocally establish that the Stage 2 model is a **Temporal Convolutional Network (TCN)** with dilated 1D residual convolutions (`TemporalKWSVerifier` using `TCNResidualBlock`). There are no recurrent GRU or LSTM units in the architecture.  
> 
> In accordance with governance rule #1, historical files remain unmodified, and this Phase 40 audit serves as the authoritative, permanent record establishing that the evaluated verifier is a **TCN**.

---

## 3. Acoustic Feature Extraction & Checkpoint Compatibility

Both Stage 1 and Stage 2 use the standardized `LogMelFeatureExtractor`:
- **Source Code Definition:** [`src/preprocessing/feature_extractor.py`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py)
- **Sampling Rate:** $16,000\text{ Hz}$, single-channel (mono), 16-bit PCM equivalent float32 normalized to $[-1.0, 1.0]$.
- **STFT Parameters:**
  - `n_fft`: $400$ samples ($25\text{ ms}$)
  - `win_length`: $400$ samples ($25\text{ ms}$, Hann window)
  - `hop_length`: $160$ samples ($10\text{ ms}$, $100\text{ frames/second}$)
- **Mel Filterbank:**
  - `n_mels`: $40$ triangular filterbanks
  - `f_min`: $20.0\text{ Hz}$
  - `f_max`: $8000.0\text{ Hz}$
- **Dynamic Compression & Per-Window Normalization:**
  - $\log(S_{\text{mel}} + 10^{-6})$
  - Frame-wise or window-wise standardization: $(X - \mu) / (\sigma + 10^{-6})$.
- **Tensor Dimensions:**
  - Stage 1 ($1.0\text{ s} = 16,000\text{ samples}$): `(Batch, 1, 40, 101)`
  - Stage 2 ($1.2\text{ s} = 19,200\text{ samples}$): `(Batch, 1, 40, 121)`

---

## 4. Continuous Streaming Simulation Engine & State Logic

### 4.1 Windowing, Hop, and Confirmation
- **Sliding Window:** $1.0\text{ s} = 16,000\text{ samples}$.
- **Hop Size:** $0.10\text{ s} = 1,600\text{ samples}$ ($10\text{ evaluations/second} \rightarrow 36,000\text{ evaluations/hour}$).
- **Consecutive Confirmation:** Requires $2$ consecutive sliding windows ($100\text{ ms}$ separation) predicting the same target command class $c \in \{0, 1, 2, 3\}$ where:
  $$P_1(c) \ge \tau_1 \quad \text{AND} \quad P_1(c) > P_1(\text{UNKNOWN}) \quad \text{AND} \quad P_1(c) > P_1(\text{SILENCE})$$
- **Candidate Proposal:** When consecutive hits reach 2, a preliminary candidate is declared.
- **Stage 2 Acoustic Context Crop:**
  - Crop length: $1.2\text{ s} = 19,200\text{ samples}$.
  - Pre-context offset: Starts $0.30\text{ s}$ ($4,800\text{ samples}$) prior to window end:
    $$c_{\text{start}} = \max(0, s_{\text{idx}} + 4800), \quad c_{\text{end}} = c_{\text{start}} + 19200$$
- **Stage 2 Confirmation Rule:**
  $$P_2(c) \ge \tau_2 \quad \text{AND} \quad P_2(c) > P_2(\text{REJECT})$$
- **Debounce & Lockout Mechanism:**
  Upon confirmation (or Stage 1 trigger if Stage 2 is bypassed), lockout is asserted until:
  $$\text{lockout\_sample} = s_{\text{idx}} + \text{int}(\text{debounce\_duration\_sec} \times 16,000)$$
  where $\text{debounce\_duration\_sec} = 0.60\text{ s}$ ($9,600\text{ samples}$). During lockout, all subsequent windows are suppressed and streak counters are reset to zero.

---

## 5. Event Matching, Deduplication, and Metric Formulations

### 5.1 Ground-Truth Matching Rules
- Ground-truth target command event: $[t_{\text{onset}}, t_{\text{end}}]$ with target class $L \in \{\text{TRAI}, \text{PHAI}, \text{LEN}, \text{XUONG}\}$.
- Detected trigger event: $t_{\text{trig}} = (s_{\text{idx}} + 16,000) / 16,000$ with predicted class $\hat{L}$.
- **Matching Window:**
  $$(t_{\text{onset}} - 0.20\text{ s}) \le t_{\text{trig}} \le (t_{\text{end}} + 0.60\text{ s})$$
- **Matching Conditions:**
  1. $\hat{L} == L$
  2. $t_{\text{trig}}$ within the matching window.
  3. Ground truth command is not previously matched (`gt_matched[idx] == False`).
- **Counts:**
  - $\text{True Positives (TP)} = \sum \text{gt\_matched}$
  - $\text{False Negatives (FN)} = N_{\text{targets}} - \text{TP}$
  - $\text{False Positives (FP)} = \text{Total Triggers} - \text{TP}$
- **Recall:** $\text{TP} / N_{\text{targets}}$
- **Precision:** $\text{TP} / (\text{TP} + \text{FP})$
- **Event F1:** $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$
- **Event Latency:** $(t_{\text{trig}} - t_{\text{onset}}) \times 1000.0\text{ ms}$.

### 5.2 Exact Denominators in False Activations / Hour
- **Continuous Stream Duration:** Audited from exact audio sample length $N_{\text{samples}}$ at $16,000\text{ Hz}$:
  $$\text{stream\_duration\_sec} = \frac{N_{\text{samples}}}{16000.0}, \quad \text{stream\_duration\_hours} = \frac{\text{stream\_duration\_sec}}{3600.0}$$
- **Exact Denominators for Frozen Challenge Streams:**
  - `val_stream_continuous_commands.wav`: $10,770,378\text{ samples} = 673.1486\text{ s} = 0.1869857\text{ hours}$
  - `val_stream_continuous_nontarget.wav`: $14,414,654\text{ samples} = 900.9159\text{ s} = 0.2502544\text{ hours}$
  - `test_stream_continuous_commands.wav`: $10,814,529\text{ samples} = 675.9081\text{ s} = 0.1877522\text{ hours}$
  - `test_stream_continuous_nontarget.wav`: $14,424,587\text{ samples} = 901.5367\text{ s} = 0.2504269\text{ hours}$
- **Calculation Rule:**
  $$\text{FA\_per\_hour} = \frac{\text{FP}}{\text{stream\_duration\_sec} / 3600.0}$$
  Both float division and high precision are verified.

---

## 6. VAD Gate Behavior & Investigation of Phase 39 Identical FP Counts

### 6.1 Gating Mechanism & Boundary Behavior
- When VAD is enabled in `evaluate_vad_ablation.py`:
  - Before Stage 1 inference, the window is assessed for active speech.
  - If declared inactive (`not is_sp`), the code executes:
    ```python
    streak_class = None
    consecutive_count = 0
    continue
    ```
  - This genuinely suppresses the window before Stage 1 forward evaluation, resetting any in-progress streak.
- **Boundary Vulnerability:**
  Soft voice onsets, unvoiced plosives (e.g. *t-*, *ch-*, *ph-*), or low-amplitude vowel transitions often fail the VAD spectral entropy / energy threshold for the first $1-2$ frames ($100-200\text{ ms}$). This resets the streak, causing the 2-frame consecutive trigger condition to miss short command utterances. This explains why target recall dropped by $-3.13\%$ in Condition D.

### 6.2 Why Condition B and Condition D Had Identical Cascade False Positives
- **Investigation:** In Phase 39, non-target false activations on the held-out test stream were identical with or without VAD:
  - Seed 42: $17$ FPs ($67.88\text{ FA/hr}$)
  - Seed 43: $26$ FPs ($103.82\text{ FA/hr}$)
  - Seed 44: $9$ FPs ($35.94\text{ FA/hr}$)
- **Audit Explanation:**
  1. The Stage 2 TCN verifier had already eliminated **100% of silence/background false alarms** ($0$ silence FPs in Condition B across all three seeds).
  2. Therefore, $100\%$ of residual cascade false alarms occurred during active non-target Vietnamese speech (distractor utterances containing rhyming or confusable syllables like *hướng*, *quên*, *thải*).
  3. Because VAD is explicitly designed to detect speech activity, it evaluated all of these distractor utterances as speech (`is_sp = True`), completely opening the gate and passing every single confusable distractor frame to Stage 1 and Stage 2!
  4. VAD cannot perform phonetic discrimination. Thus, Condition D passed the exact same distractor frames as Condition B, producing bitwise identical false alarms while degrading target command recall.

---

## 7. Protocol Lock for Phases 41–44

| Parameter | Locked Specification | Reference / Source |
| :--- | :---: | :---: |
| **Audio Format** | $16,000\text{ Hz}$, 1 channel, 16-bit PCM float32 | `LogMelFeatureExtractor` |
| **Window Length / Hop** | $1.0\text{ s}$ ($16,000$ samples) / $0.10\text{ s}$ ($1,600$ samples) | Standardized |
| **Mel Filterbanks** | $40$ bins, $20-8000\text{ Hz}$, $25\text{ ms}$ win, $10\text{ ms}$ hop | Standardized |
| **Stage 1 Architecture** | `VietnameseKWSNet` ($138,534$ params) | [`src/models/kws_cnn.py`](file:///d:/trainidentificationaudio/src/models/kws_cnn.py) |
| **Stage 2 Architecture** | `TemporalKWSVerifier` (TCN, $340,773$ params) | [`experiments/vieneu_kws_cascade_v1/train_verifier.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1/train_verifier.py) |
| **Consecutive Confirmation** | $2\times$ consecutive windows meeting $\tau_1$ | Standardized |
| **Debounce Lockout** | $0.60\text{ s}$ ($9,600$ samples) | Standardized |
| **Event Matching Window** | $[t_{\text{onset}} - 0.20\text{ s}, t_{\text{end}} + 0.60\text{ s}]$ | Standardized |
| **FA/hour Denominator** | $\text{Exact: } \text{stream\_duration\_sec} / 3600.0$ | Standardized |
| **Governance Status** | `PENDING_HUMAN_LISTENER_VERIFICATION` \| `NOT_VALIDATED_ON_NATURAL_SPEECH` | Standardized |

---

## 8. Audit Sign-Off
All historical artifacts remain strictly preserved. The inference and evaluation interfaces are verified and locked for Phases 41–44.
