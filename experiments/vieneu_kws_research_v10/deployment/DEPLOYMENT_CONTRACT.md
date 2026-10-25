# Deployment Contract: Vietnamese Bounded-Utterance Keyword Spotting (CNN variant_w10)

**Workspace:** `D:\trainidentificationaudio`  
**Candidate ID:** `variant_w10` (`CorrectedVietnameseKWSNet`)  
**Phase:** 91 (CNN Deployment Engineering)  
**Status:** `RESEARCH_FROZEN_CANDIDATE`  
**Governance Invariants:**
- `PENDING_HUMAN_LISTENER_VERIFICATION`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Continuous Open-Mic Mode)

---

## 1. Frozen Model Provenance and Checkpoint Integrity

The model artifact is frozen and immutable. The primary checkpoint used for deployment is Seed 42, calibrated in Phase 74 with decision threshold $\theta_{\text{cmd}} = 0.40$. Secondary seeds 43 and 44 are retained for cross-seed variance auditing.

| Checkpoint Name | Relative Path | Parameters | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| Primary (Seed 42) | `experiments/vieneu_kws_research_v9/artifacts/checkpoints/variant_w10_seed42.pt` | 138,469 | `7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed` |
| Secondary (Seed 43) | `experiments/vieneu_kws_research_v9/artifacts/checkpoints/variant_w10_seed43.pt` | 138,469 | `77f468c68b261de506a3fcb022c8d7cc68f8b0c2ad650852b7c300f274dda251` |
| Secondary (Seed 44) | `experiments/vieneu_kws_research_v9/artifacts/checkpoints/variant_w10_seed44.pt` | 138,469 | `bebab580e4caa6934c98a62b095611e4eef9221a7f4282d36f9a4b6928abde64` |

---

## 2. Input Audio Contract

### 2.1 File & Stream Format
- **Container / Format:** RIFF WAV container.
- **Encoding:** Uncompressed Linear PCM (16-bit signed integer or 32-bit floating point).
- **Target Sampling Rate:** $16,000 \text{ Hz}$ ($f_s = 16 \text{ kHz}$).
- **Target Channel Layout:** 1 channel (Mono).
- **Target Duration:** Exactly $1.000 \text{ second}$ ($16,000 \text{ samples}$).

### 2.2 Ingestion & Validation Rules

| Dimension | Strict Validation Mode | Permissive Mode (Offline Batch CLI) | On-Device C++ / Android Buffer |
| :--- | :--- | :--- | :--- |
| **File Existence** | Must exist, size $> 0$ bytes | Must exist, size $> 0$ bytes | Memory buffer pointer non-null |
| **Empty Audio** | Error (`EmptyAudioError`) | Error (`EmptyAudioError`) | Error (`EMPTY_BUFFER`) |
| **Header Parsing** | Valid RIFF/WAV header | Valid RIFF/WAV header | N/A (raw PCM buffer) |
| **Sample Rate** | Reject if $f_s \neq 16,000 \text{ Hz}$ (`InvalidSampleRateError`) | Torchaudio sinc/polyphase resample to $16,000 \text{ Hz}$ | Reject if hardware stream $f_s \neq 16,000 \text{ Hz}$ |
| **Channel Count** | Reject if $C \neq 1$ (`InvalidChannelCountError`) | Average across channels: $x = \frac{1}{C}\sum_{c=1}^C x_c$ | Reject if stream is not mono |
| **Duration / Length** | Reject if $N \neq 16,000$ (`InvalidDurationError`) | Pad right with zeros if $N < 16,000$; truncate if $N > 16,000$ | Fixed ring buffer: exactly $16,000$ samples |
| **Peak Normalization** | Clamped only if $\max(\|x\|) > 1.0$: $x \leftarrow \frac{x}{\max(\|x\|)}$ | Clamped only if $\max(\|x\|) > 1.0$: $x \leftarrow \frac{x}{\max(\|x\|)}$ | Clamped only if $\max(\|x\|) > 1.0$ |
| **Floor Amplification** | **STRICTLY PROHIBITED** (never amplify quiet noise floor) | **STRICTLY PROHIBITED** | **STRICTLY PROHIBITED** |

---

## 3. Frontend Acoustic Feature Extraction

The feature extraction front-end converts a 1-second 16 kHz mono waveform vector $x \in \mathbb{R}^{16000}$ into a 2D Log-Mel spectrogram representation $S_{\text{logmel}} \in \mathbb{R}^{1 \times 1 \times 40 \times 101}$.

### 3.1 Spectrogram Parameters
- **Transform:** Short-Time Fourier Transform (STFT) with squared magnitude ($\text{power} = 2.0$).
- **FFT Window Size ($N_{\text{fft}}$):** $400 \text{ samples}$ ($25.0 \text{ ms}$).
- **Window Length:** $400 \text{ samples}$ (Hann window).
- **Hop Size:** $160 \text{ samples}$ ($10.0 \text{ ms}$).
- **Time Framing:** $\text{center} = \text{True}$ with `pad_mode="reflect"`.
  - For $16,000 \text{ samples}$, centered padding produces $\lfloor 16000 / 160 \rfloor + 1 = 101 \text{ frames}$.
- **Frequency Range:** $f_{\min} = 20.0 \text{ Hz}$, $f_{\max} = 8,000.0 \text{ Hz}$ (Nyquist frequency at 16 kHz).
- **Mel Filterbanks ($N_{\text{mels}}$):** $40 \text{ triangular filters}$.
- **Mel Scale & Norm:** Slaney scale and Slaney triangular area normalization (`norm="slaney"`, `mel_scale="slaney"`).
- **Logarithm & Compression:**
  $$\mathbf{S}_{\text{logmel}} = \ln(\mathbf{S}_{\text{mel}} + \epsilon), \quad \epsilon = 1.0 \times 10^{-6}$$
- **Output Tensor Shape:** `[Batch=1, Channels=1, Mel=40, Time=101]`, `dtype=float32`.

---

## 4. Model Architecture Contract

### 4.1 Topology (`CorrectedVietnameseKWSNet`)
A compact 4-stage 2D convolutional neural network using non-parametric global pooling.

```
Input: [B, 1, 40, 101]
  │
  ├─ Conv1: Conv2d(1, 32, kernel=3, padding=1, bias=False) ──> [B, 32, 40, 101]
  ├─ BN1: BatchNorm2d(32) + ReLU
  ├─ Pool1: MaxPool2d(kernel=2, stride=2) ──────────────────> [B, 32, 20, 50]
  │
  ├─ Conv2: Conv2d(32, 64, kernel=3, padding=1, bias=False) ─> [B, 64, 20, 50]
  ├─ BN2: BatchNorm2d(64) + ReLU
  ├─ Pool2: MaxPool2d(kernel=2, stride=2) ──────────────────> [B, 64, 10, 25]
  │
  ├─ Conv3: Conv2d(64, 64, kernel=3, padding=1, bias=False) ─> [B, 64, 10, 25]
  ├─ BN3: BatchNorm2d(64) + ReLU
  ├─ Pool3: MaxPool2d(kernel=2, stride=2) ──────────────────> [B, 64, 5, 12]
  │
  ├─ Conv4: Conv2d(64, 128, kernel=3, padding=1, bias=False) ─> [B, 128, 5, 12]
  ├─ BN4: BatchNorm2d(128) + ReLU
  ├─ GlobalPool: AdaptiveAvgPool2d((1, 1)) ──────────────────> [B, 128, 1, 1]
  │
  ├─ Flatten(1) ─────────────────────────────────────────────> [B, 128]
  ├─ Dropout(p=0.3, eval=disabled)
  ├─ FC1: Linear(128, 64) + ReLU ────────────────────────────> [B, 64]
  └─ FC2: Linear(64, 5) ─────────────────────────────────────> [B, 5] (Logits)
```

- **Total Trainable Parameters:** $138,469$.
- **Model Checkpoint Size (PyTorch FP32):** $564 \text{ KB}$.
- **Exported ONNX FP32 Size:** $552 \text{ KB}$.
- **Exported ONNX INT8 Dynamic Quantized Size:** $145 \text{ KB}$.

---

## 5. Output Interpretation and Decision Logic

### 5.1 Class Mapping Table
The network outputs 5 logits corresponding exactly to the following zero-indexed class indices:

| Index | Label | Category | Description |
| :---: | :--- | :--- | :--- |
| `0` | `TRAI` | Target Command | Directional keyword: Left |
| `1` | `PHAI` | Target Command | Directional keyword: Right |
| `2` | `LEN` | Target Command | Directional keyword: Up |
| `3` | `XUONG` | Target Command | Directional keyword: Down |
| `4` | `REJECT` | Non-Target / Noise | Hard negatives, silence, background speech, non-target words |

### 5.2 Probability Computation
Logits $\mathbf{z} = [z_0, z_1, z_2, z_3, z_4]$ are mapped to probabilities $\mathbf{p}$ via Softmax:
$$p_i = \frac{e^{z_i}}{\sum_{j=0}^4 e^{z_j}}, \quad i \in \{0, 1, 2, 3, 4\}$$

### 5.3 Gating and Rejection Decision Rule
The command decision is governed by the calibrated threshold $\theta_{\text{cmd}} = 0.40$:

1. Compute maximum command probability:
   $$p_{\text{cmd\_max}} = \max \{p_0, p_1, p_2, p_3\}$$
   $$c_{\text{cmd\_idx}} = \arg\max_{i \in \{0,1,2,3\}} p_i$$

2. Evaluate threshold gate:
   $$\text{Decision} = \begin{cases} 
   \text{CLASSES}[c_{\text{cmd\_idx}}], & \text{if } p_{\text{cmd\_max}} \ge \theta_{\text{cmd}} \\ 
   \text{"REJECT"}, & \text{if } p_{\text{cmd\_max}} < \theta_{\text{cmd}} 
   \end{cases}$$

3. Confidence Reporting:
   - If a target command is accepted: $\text{Confidence} = p_{\text{cmd\_max}}$.
   - If rejected: $\text{Confidence} = p_4$ (the explicit non-target class probability).

---

## 6. Target Deployment Runtime Options

| Platform | Recommended Engine | Format | Features / Constraints |
| :--- | :--- | :--- | :--- |
| **Server / Offline CLI** | PyTorch / Torchaudio | `.pt` / `.onnx` | Direct Python pipeline, batch inference, reference implementation |
| **Android (Edge)** | ONNX Runtime Mobile | `.onnx` (FP32/INT8) | C++ JNI bridge for Mel-spectrogram, direct hardware acceleration (NNAPI/CPU) |
| **iOS (Edge)** | ONNX Runtime / CoreML | `.onnx` / `.mlpackage` | Accelerate vDSP for STFT/Mel, Swift wrapper, CoreML Neural Engine support |
| **Cross-Platform C++** | ONNX Runtime C++ API | `.onnx` (FP32/INT8) | Single unified C++ codebase, minimum third-party dependencies |

---

## 7. Status Disclaimers and Limitations

1. **Synthetic Training Scope:** This model was trained and evaluated solely on synthetic TTS utterances and synthetic noise augmentations.
2. **Pending Human Listener Verification:** Human listening evaluations remain `PENDING_HUMAN_LISTENER_VERIFICATION`.
3. **Natural Speech Non-Validation:** The model has `NOT_VALIDATED_ON_NATURAL_SPEECH` and may exhibit significant performance degradation on spontaneous natural speech, varied Vietnamese regional accents, or non-stationary acoustic environments.
4. **Continuous Open-Mic Verdict:** At 3,600 one-second windows per hour, a 2.0% false alarm rate produces approximately $70 \text{ false activations/hour}$. Therefore, this model is certified strictly for **Bounded-Utterance (Push-to-Talk or Recorded-File)** evaluation, and the open-mic verdict remains `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.
