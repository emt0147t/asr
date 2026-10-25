# Phase 6: Frozen KWS Model Exportability and Runtime-Parity Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> This export and runtime-parity audit operates exclusively on the frozen synthetic model checkpoint and synthetic acoustic fixtures. It does NOT access live microphones, collect human speech, or ingest external corpora. All numerical evaluations reflect synthetic software parity only.

---

## 1. Executive Summary & Audit Verdict

Phase 6 evaluated the exportability of the frozen primary candidate model (**V4 Arm B**: `arm_b_onset_no_resample.pt`), exported the neural network to ONNX (opset 17), and performed an exhaustive numerical parity audit comparing PyTorch against ONNX Runtime (CPU).

### Audit Verdict: `PARITY_AUDIT_PASS`
- **Source Checkpoint SHA-256**: `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` (Verified bit-exact).
- **Exported ONNX Graph**: `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx` (1,095,036 bytes, SHA-256: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625`).
- **Argmax Class Agreement Across All 356 Evaluated Test Inputs**: **100.00% (356 / 356)**.
- **Global Maximum Absolute Probability Difference**: **$7.75 \times 10^{-7}$** (Strictly within the $\le 1.0 \times 10^{-4}$ parity threshold).
- **Global Mean Absolute Probability Difference**: **$2.06 \times 10^{-8}$**.
- **Inference Runtime Speedup (Forward Pass)**: **3.05x** (PyTorch mean 0.925 ms vs. ONNX Runtime mean 0.303 ms on desktop CPU).

> [!CAUTION]
> **CRITICAL ARCHITECTURAL BOUNDARY: NEURAL GRAPH VS. RAW-AUDIO PIPELINE**
> The exported ONNX model (`arm_b_tcresnet14.onnx`) accepts pre-computed **Log-Mel spectrogram features** of shape `(Batch, 1, 40, 101)`. It does **NOT** accept raw 16 kHz WAV audio. An exported neural-network graph is not, by itself, a complete mobile inference engine. Mobile deployment requires an independently validated audio preprocessing frontend on Android (C++/NDK) or iOS (Accelerate/CoreAudio).

---

## 2. Export Configuration & Model Contract

The neural network was exported separately from the Log-Mel feature extractor to prevent coupling to PyTorch-specific STFT operators and to preserve exact numerical control over audio ingestion.

| Specification | Details | Value / Contract |
| :--- | :--- | :--- |
| **Model Architecture** | `VietnameseTCResNet` | 14-layer Temporal 1D Convolutional ResNet with Global Mean & Max Pooling |
| **Source Checkpoint** | PyTorch state dictionary | `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt` |
| **Source Checkpoint SHA-256**| SHA-256 Hash | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` |
| **Export Toolchain** | PyTorch $\rightarrow$ ONNX | PyTorch 2.6.0+cu124, ONNX 1.23.2, ONNX Runtime 1.31.0 |
| **Target ONNX Path** | Serialized Graph | `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx` |
| **Target ONNX SHA-256** | SHA-256 Hash | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` |
| **File Size** | Binary ONNX Model | 1,095,036 bytes (~1.04 MB) |
| **ONNX Opset Version** | Target IR Opset | **Opset 17** (Full mobile runtime compatibility across ONNX Runtime, TFLite, CoreML) |
| **Input Node** | Name / Dtype / Shape | `log_mel` / `float32` / `['batch_size', 1, 40, 101]` |
| **Output Node 1** | Name / Dtype / Shape | `logits` / `float32` / `['batch_size', 6]` (Raw linear projection) |
| **Output Node 2** | Name / Dtype / Shape | `probabilities` / `float32` / `['batch_size', 6]` (Normalized Softmax distribution) |
| **Class Ordering** | Canonical Index Map | `0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: UNKNOWN, 5: SILENCE` |

---

## 3. Comprehensive Parity Evaluation

The parity evaluation compared PyTorch (CPU) against ONNX Runtime (`CPUExecutionProvider`) across three distinct evaluation cohorts totaling 356 samples.

### Cohort 1: Canonical Synthetic Demonstration Fixtures (6 Clips)
Evaluated on the dedicated demonstration fixtures synthesized with speaker `SPK_DEMO_001`. Features extracted via canonical `LogMelFeatureExtractor`.

| Audio Fixture | Intended Command | PyTorch Pred (Conf) | ONNX Runtime Pred (Conf) | Argmax Agreement | Max Abs Prob Diff | Mean Abs Prob Diff |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `demo_01_trai.wav` | `TRAI` | `TRAI` (88.4011%) | `TRAI` (88.4011%) | **TRUE** | $7.45 \times 10^{-8}$ | $2.48 \times 10^{-8}$ |
| `demo_02_phai.wav` | `PHAI` | `PHAI` (99.3009%) | `PHAI` (99.3009%) | **TRUE** | $4.19 \times 10^{-9}$ | $1.40 \times 10^{-9}$ |
| `demo_03_len.wav` | `LEN` | `LEN` (96.8601%) | `LEN` (96.8601%) | **TRUE** | $3.17 \times 10^{-8}$ | $1.06 \times 10^{-8}$ |
| `demo_04_xuong.wav` | `XUONG` | `XUONG` (98.4418%) | `XUONG` (98.4418%) | **TRUE** | $2.79 \times 10^{-9}$ | $9.31 \times 10^{-10}$ |
| `demo_05_unknown.wav`| `UNKNOWN` | `UNKNOWN` (98.8710%) | `UNKNOWN` (98.8710%) | **TRUE** | $2.33 \times 10^{-9}$ | $7.76 \times 10^{-10}$ |
| `demo_06_silence.wav`| `SILENCE` | `SILENCE` (96.3572%) | `SILENCE` (96.3572%) | **TRUE** | $9.31 \times 10^{-9}$ | $3.10 \times 10^{-9}$ |
| **Cohort 1 Summary** | — | **6 / 6 (100.0%)** | **6 / 6 (100.0%)** | **100.0%** | **$7.45 \times 10^{-8}$** | **$6.93 \times 10^{-9}$** |

#### Probability Vector Comparison (Full Precision Verification)
- **`demo_01_trai.wav`**:
  - PyTorch: `[0.88401097, 0.04895696, 0.00204739, 0.00191244, 0.06148386, 0.00158838]`
  - ONNX:    `[0.88401090, 0.04895696, 0.00204739, 0.00191244, 0.06148393, 0.00158838]`
  - Max Error: $\Delta = 7.45 \times 10^{-8}$ (on `UNKNOWN` and `TRAI`).
- **`demo_06_silence.wav`**:
  - PyTorch: `[0.00147926, 0.00216298, 0.00693998, 0.00171097, 0.02413444, 0.96357238]`
  - ONNX:    `[0.00147926, 0.00216298, 0.00693998, 0.00171097, 0.02413444, 0.96357238]`
  - Max Error: $\Delta = 9.31 \times 10^{-9}$.

---

### Cohort 2: V4 Fresh Synthetic Holdout Benchmark (300 Clips)
Evaluated against the entire frozen V4 holdout dataset (`fresh_holdout_manifest_6class.csv`), comprising 50 clips per class across 5 calibration speakers (`SPK_CALIB_031` through `SPK_CALIB_035`).

- **Total Clips Evaluated**: 300 clips
- **Argmax Class Agreement**: **300 / 300 (100.00%)**
- **Overall Maximum Absolute Probability Difference**: **$7.75 \times 10^{-7}$** (Threshold: $\le 1.0 \times 10^{-4}$)
- **Overall Mean Absolute Probability Difference**: **$2.06 \times 10^{-8}$**
- **Per-Class Maximum Absolute Differences**:
  - `TRAI`: $7.75 \times 10^{-7}$
  - `PHAI`: $2.38 \times 10^{-7}$
  - `LEN`: $3.58 \times 10^{-7}$
  - `XUONG`: $2.38 \times 10^{-7}$
  - `UNKNOWN`: $7.45 \times 10^{-7}$
  - `SILENCE`: $1.19 \times 10^{-7}$

---

### Cohort 3: Deterministic Synthetic Feature Tensors (50 Inputs, Seed 20261016)
Evaluated across varied mathematical and statistical regimes directly fed to the model input node:
1. Standard normal distributions $\mathcal{N}(0, 1)$
2. Broad uniform distributions $\mathcal{U}(-10.0, 5.0)$
3. Ambient noise floor baselines $\mathcal{N}(-12.0, 0.1)$
4. Flat zeros (numerical silence floor)
5. High-amplitude transient bursts ($5.0 \cdot \mathcal{N}(0, 1)$)

- **Total Synthetic Tensors**: 50 inputs
- **Random Seed**: `20261016` (Documented and reproducible)
- **Argmax Class Agreement**: **50 / 50 (100.00%)**
- **Overall Maximum Absolute Difference**: **$2.38 \times 10^{-7}$**
- **Overall Mean Absolute Difference**: **$1.52 \times 10^{-8}$**

---

## 4. Latency Benchmarking & Profiling Boundaries

Latency measurements were profiled on a desktop x86_64 CPU over 100 consecutive timed runs following 25 unmeasured warm-up iterations.

| Metric | PyTorch CPU Forward Pass | ONNX Runtime CPU Forward Pass | Speedup Ratio |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | 0.925 ms | **0.303 ms** | **3.05x faster** |
| **Median (p50) Latency** | 0.912 ms | **0.298 ms** | 3.06x faster |
| **95th Percentile (p95)** | 0.989 ms | **0.420 ms** | 2.35x faster |
| **Standard Deviation** | 0.048 ms | 0.052 ms | — |

### Measurement Boundaries
- **In-Scope for Benchmark**: Pure neural network forward pass (inference graph execution from input tensor `(1, 1, 40, 101)` to output tensors `logits` and `probabilities`). Single-threaded execution (`intra_op_num_threads = 1`, `inter_op_num_threads = 1`).
- **Excluded from Benchmark**: Audio file I/O (disk read, WAV header parsing), PCM audio validation, and STFT/Log-Mel feature computation. In the complete desktop offline pipeline, feature extraction adds approximately 0.88 ms.

---

## 5. Mobile Preprocessing Contract & Deployment Gap

Mobile deployment cannot assume that feeding raw microphone audio into a mobile app will match Python output. The client runtime must implement an acoustic frontend matching the exact reference specification below:

```
[Raw Audio: 16,000 Hz, 16-bit PCM, 1 Channel]
                      |
           1. Amplitude Normalization (float32, [-1.0, 1.0])
                      |
           2. Framing: Window 400 (25ms), Hop 160 (10ms)
                      |
           3. Windowing: Periodic Hann / Hamming Window
                      |
           4. STFT: N_FFT = 512, Center = True, Pad Mode = Reflect
                      |
           5. Power Spectrum: |STFT|^2 (Power = 2.0)
                      |
           6. Mel Filterbank: 40 Mel Bands (20 Hz - 8,000 Hz, Slaney Normalized)
                      |
           7. Log Compression: ln(Mel + 1.0e-6)
                      |
[Output Tensor: Shape (1, 1, 40, 101), dtype=float32]
                      |
[ONNX Model: arm_b_tcresnet14.onnx]
```

### Critical Mobile Frontend Specifications

1. **Audio Buffering & Rejection**:
   - Sample Rate: Strictly **16,000 Hz**. Audio at 44.1 kHz or 48 kHz must be resampled using a high-quality sinc/polyphase filter before framing.
   - Channel Count: Strictly **mono**. If the device microphone captures stereo, client software must reject or explicitly mix to mono before windowing.
   - Buffer Duration: Exactly **16,000 samples** (1.000s).

2. **Short-Time Fourier Transform (STFT)**:
   - Analysis Window Length ($N_{\text{win}}$): 400 samples ($25\text{ ms}$).
   - Hop Length ($N_{\text{hop}}$): 160 samples ($10\text{ ms}$).
   - FFT Length ($N_{\text{FFT}}$): 512 samples (zero-padded from 400).
   - Padding Mode: `reflect` with `center = True`. For 16,000 samples, this produces exactly $1 + \lfloor 16000 / 160 \rfloor = 101$ time frames.

3. **Mel Filterbank Definition**:
   - Number of Mel Bands: **40 bands**.
   - Frequency Range: $F_{\text{min}} = 20.0\text{ Hz}$, $F_{\text{max}} = 8000.0\text{ Hz}$ (Nyquist).
   - Filter Formulation: **Slaney (area-normalized)** triangular filters (`norm="slaney"`, `mel_scale="slaney"`).

4. **Dynamic Range & Logarithm**:
   - Logarithm: **Natural logarithm** ($\ln$), not $\log_{10}$.
   - Epsilon Floor: $\epsilon = 1.0 \times 10^{-6}$.
   - Output Tensor Shape: `(1, 1, 40, 101)` corresponding to `(Batch, Channels, Mel_Bands, Time_Frames)`.

### Mobile Implementation Risk Factors
- **Filterbank Discrepancy**: Mobile DSP libraries often default to HTK-style Mel filterbanks rather than Slaney normalized filterbanks. HTK filters will introduce a systematic spectral tilt that degrades model accuracy.
- **Window Centering & Padding**: Failure to implement reflect padding or window centering will alter the frame alignment from 101 frames to 98 frames, causing shape rejection.
- **Log Epsilon Differences**: Using $\log_{10}$ or a different stabilizer ($\epsilon = 10^{-4}$ or clamping) will shift activations by several orders of magnitude.
- **Golden Test Requirement**: The mobile client's Log-Mel extractor must be verified against [`src/preprocessing/feature_extractor.py:extract_log_mel_numpy`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py) using identical test audio before running live inference.

---

## 6. Integration Safeguards & Game Adapter Invariants

1. **Decoupled Architecture**: The game-control adapter ([`game_adapter.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/game_adapter.py)) remains completely independent of the inference backend (whether PyTorch or ONNX Runtime).
2. **Non-Target Safety Invariant**: The adapter unconditionally maps `UNKNOWN` and `SILENCE` to `GameAction.NO_ACTION` with displacement $(0, 0)$ and $\text{is\_active} = \text{False}$.
3. **Confidence Thresholding**: The operational threshold of `0.50` is experimental and calibrated exclusively on synthetic acoustic fixtures. It has **not** been calibrated against human natural-speech false alarm rates.
4. **No Microphone / Streaming**: Live streaming audio, ring buffering, and microphone polling remain strictly absent from the codebase.

---

## 7. Real-World Limitations & Governance Boundary

> [!WARNING]
> ### Definitive Real-World Boundary Statement
> 1. **Synthetic-Only Proof of Correctness**: This audit proves that the PyTorch neural network and the exported ONNX graph are mathematically equivalent to within $7.75 \times 10^{-7}$ on synthetic acoustic data.
> 2. **Untested Human Voice**: The model has **never been evaluated on natural human voice**. Performance on real human speech is entirely unknown.
> 3. **Vietnamese Dialects**: Dialectal features modeled in the synthetic generator do not cover the acoustic richness, consonant shifts, or tonal mergers of real Vietnamese speakers across Northern, Central, and Southern regions.
> 4. **Hardware & Microphone Variances**: Mobile microphones exhibit variable AGC (automatic gain control), hardware clipping, non-linear distortion, and acoustic reverberation that were not present in synthetic training fixtures.
> 5. **Edge Thermal & Battery Feasibility**: Desktop CPU forward pass timings (0.303 ms) demonstrate algorithm efficiency on an x86 workstation. They do not constitute an embedded or mobile benchmark (e.g. Android NNAPI, iOS CoreML Neural Engine, INT8 quantization).
> 6. **Governance Rule**: **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** No human voice recordings may be acquired, uploaded, or processed without explicit regulatory and ethics protocol clearance.
