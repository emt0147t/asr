# Phase 7: KWS Frontend Golden-Vector Audit and Mobile Inference Readiness

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> This frontend audit and golden-vector verification operates strictly on synthetic audio fixtures and mathematical reference tensors. It does NOT access live microphones, collect human speech, or ingest external corpora. All findings reflect synthetic software and mathematical parity only.

---

## 1. Executive Summary & Verification Verdict

Phase 7 establishes the exact, executable mathematical contract for the Vietnamese Keyword Spotting (KWS) acoustic frontend, generates an immutable golden regression suite of 12 test cases, and audits numerical reproducibility across PyTorch and ONNX Runtime.

### Audit Verdict: `GOLDEN_VERIFICATION_PASS`
- **Source Checkpoint Verified**: `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt` (SHA-256: `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96`).
- **Exported ONNX Model Verified**: `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx` (SHA-256: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625`).
- **Golden Test Cases**: **12 canonical cases** (6 demo classes, 2 timing variants, 3 numerical edge cases, 1 historical holdout anchor).
- **Feature Reproducibility**: **12 / 12 cases bit-exact ($0.00 \times 10^{0}$ difference against saved golden tensors)**.
- **PyTorch vs. ONNX Runtime Prediction Parity**: **100% argmax agreement**, with maximum probability difference across all golden cases of **$2.09 \times 10^{-7}$** (Threshold: $\le 1.0 \times 10^{-4}$).
- **Frontend Discrepancy Resolved**: Ambiguous phrase `Periodic Hann / Hamming Window` resolved to **`Periodic Hann Window`** via executable code evidence (recorded as `DISC-09`).

---

## 2. Definitive Executable Frontend Contract

The exact Python frontend implementation was audited directly from executable code in [`src/preprocessing/feature_extractor.py`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py), [`experiments/synthetic_kws_demo/classifier.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/classifier.py), and [`experiments/synthetic_kws_demo/export/export_and_benchmark.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/export/export_and_benchmark.py).

| Parameter | Exact Contract Value | Code Location & Executable Verification |
| :--- | :--- | :--- |
| **Input Sample Rate** | `16,000 Hz` | Strictly enforced by `AudioValidator.validate_and_load`. Resampling is rejected. |
| **Channel Handling** | `1 (Mono)` | Multichannel/stereo audio is rejected with `InvalidChannelCountError`. |
| **Buffer Duration** | `16,000 samples (1.000s)` | Exactly 16,000 frames required (`InvalidDurationError` on mismatch). |
| **Waveform Scaling** | `float32 in [-1.0, 1.0]` | Scaled directly from PCM_16 by `soundfile.read(..., dtype='float32')`. No AGC or peak normalization. |
| **Window Length ($N_{\text{win}}$)** | `400 samples (25.0 ms)` | `T.MelSpectrogram(win_length=400)` |
| **Window Function** | **`Periodic Hann Window`** | `torch.hann_window(400, periodic=True)`. Absolute difference against periodic Hann is `0.0`. |
| **Hop Length ($N_{\text{hop}}$)** | `160 samples (10.0 ms)` | `T.MelSpectrogram(hop_length=160)` |
| **FFT Size ($N_{\text{FFT}}$)** | `512 samples` | `T.MelSpectrogram(n_fft=512)`. Zero-padded from 400 to 512. |
| **Centering & Padding Mode** | `center=True, pad_mode="reflect"` | Waveform is reflect-padded by $\lfloor 512/2 \rfloor = 256$ samples on both ends prior to STFT. |
| **STFT Normalization** | `normalized=False` | Unnormalized complex STFT magnitude. |
| **Power Exponent** | `power=2.0` | Power spectrogram: $|X(f, t)|^2 = \text{Re}^2 + \text{Im}^2$. |
| **Mel Filterbank Count** | `40 bands` | `T.MelSpectrogram(n_mels=40)` |
| **Mel Frequency Limits** | `F_min = 20.0 Hz, F_max = 8000.0 Hz` | Triangular filters spanning 20 Hz up to Nyquist (8,000 Hz). |
| **Mel Scale & Normalization** | `mel_scale="slaney", norm="slaney"` | Area-normalized triangular filters ($\frac{2}{f_{\text{right}} - f_{\text{left}}}$). |
| **Log Compression** | Natural Logarithm: $\ln(\text{Mel} + 10^{-6})$ | `torch.log(mel_spec + 1e-6)`. Base $e$, stabilizer $\epsilon = 1.0 \times 10^{-6}$. |
| **Post-Log Normalization** | None (Raw Log-Mel) | No CMVN (Cepstral Mean and Variance Normalization) applied. |
| **Tensor Dtype & Shape** | `float32`, `(1, 1, 40, 101)` | Layout: `(Batch, Channels, Mel_Bands, Time_Frames)`. |
| **Class Ordering** | 6 Classes | `0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: UNKNOWN, 5: SILENCE` |

### Resolution of Window Function Ambiguity (DISC-09)
In Phase 6 documentation, the window function was tentatively listed as `Periodic Hann / Hamming Window` and `window_type: Hamming (or Hann as verified)`.
Execution of the underlying torchaudio module (`T.MelSpectrogram`) reveals:
1. `mel_transform.spectrogram.window` is a `torch.Tensor` of shape `[400]`.
2. `torch.max(torch.abs(mel_transform.spectrogram.window - torch.hann_window(400, periodic=True))) == 0.0`.
3. `torch.max(torch.abs(mel_transform.spectrogram.window - torch.hamming_window(400))) == 0.080000`.
4. In [`src/preprocessing/feature_extractor.py:extract_log_mel_numpy`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py), `librosa.stft(..., window="hann")` also implements the periodic Hann window.

Therefore, the canonical window function is unequivocally the **Periodic Hann Window**:
$$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{400}\right), \quad n = 0, 1, \dots, 399$$
This discrepancy has been formally recorded in [`discrepancy_log.json`](discrepancy_log.json).

---

## 3. Golden Vector Inventory & Parity Results

All 12 golden test cases are stored immutably in [`experiments/synthetic_kws_demo/frontend_golden/`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_demo/frontend_golden/) with metadata in [`golden_manifest.json`](golden_manifest.json).

| Case ID | Case Name | Category | Intended | Source WAV SHA-256 | Golden Tensor SHA-256 | PyTorch Pred (Conf) | ONNX Pred (Conf) | Feat Diff | Prob Diff |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| `case_01` | TRAI Demo Fixture | canonical_demo | `TRAI` | `F4D8BFB19B68...1BFA` | `40C537BBC3F2...9EB8` | `TRAI` (88.40%) | `TRAI` (88.40%) | $0.00$ | $8.57 \times 10^{-8}$ |
| `case_02` | PHAI Demo Fixture | canonical_demo | `PHAI` | `313A900BD5E4...E675` | `57E5433CD504...9D78` | `PHAI` (99.30%) | `PHAI` (99.30%) | $0.00$ | $4.19 \times 10^{-9}$ |
| `case_03` | LEN Demo Fixture | canonical_demo | `LEN` | `B77DA0C254B8...76CE` | `C41474B37ED2...1A90` | `LEN` (96.86%) | `LEN` (96.86%) | $0.00$ | $3.17 \times 10^{-8}$ |
| `case_04` | XUONG Demo Fixture | canonical_demo | `XUONG` | `9A501FA097F5...0DB1` | `BC4CD8928DD7...031D` | `XUONG` (98.44%) | `XUONG` (98.44%) | $0.00$ | $1.35 \times 10^{-8}$ |
| `case_05` | UNKNOWN Demo Fixture | canonical_demo | `UNKNOWN` | `87D97EF62CE6...01EB7`| `2F4607EB1068...A11E` | `UNKNOWN` (98.87%) | `UNKNOWN` (98.87%) | $0.00$ | $6.98 \times 10^{-9}$ |
| `case_06` | SILENCE Demo Fixture | canonical_demo | `SILENCE` | `6AE179B7EDFD...462A` | `B4BCDC250DCC...5FCA` | `SILENCE` (96.36%) | `SILENCE` (96.36%) | $0.00$ | $1.19 \times 10^{-7}$ |
| `case_07` | TRAI 200ms Leading Pad | timing_variant | `TRAI` | `7F6E4A0C7502...890D` | `0F7CB1B00E8A...17D8` | `TRAI` (81.68%) | `TRAI` (81.68%) | $0.00$ | $1.27 \times 10^{-7}$ |
| `case_08` | PHAI Centered Pad | timing_variant | `PHAI` | `6908F87BFBFA...BC57` | `143E7D3B2556...853F` | `PHAI` (95.06%) | `PHAI` (95.06%) | $0.00$ | $2.61 \times 10^{-8}$ |
| `case_09` | Pure Digital Zeros | numerical_edge | `SILENCE` | `CFF4DDC97D81...EB16` | `AE9DEE80D881...5E30` | `TRAI` (30.23%)\* | `TRAI` (30.23%)\* | $0.00$ | $1.79 \times 10^{-7}$ |
| `case_10` | Low-Amplitude Ambient | numerical_edge | `SILENCE` | `1C95A0129BFF...D8CE` | `7AB22413DF38...6C2F` | `TRAI` (30.24%)\* | `TRAI` (30.24%)\* | $0.00$ | $1.19 \times 10^{-7}$ |
| `case_11` | Transient Unit Impulse | numerical_edge | `UNKNOWN` | `26CC6243A482...B1F8` | `70E105D15A37...FD91` | `TRAI` (31.37%)\* | `TRAI` (31.37%)\* | $0.00$ | $2.09 \times 10^{-7}$ |
| `case_12` | Holdout TRAI Anchor | holdout_anchor | `TRAI` | `9DF5C497B1A5...7206` | `2A5B9585A5ED...42EA` | `TRAI` (86.11%) | `TRAI` (86.11%) | $0.00$ | $1.79 \times 10^{-7}$ |

\* **Note on Numerical Edge Cases (Cases 09, 10, 11)**:
When presented with non-acoustic artificial inputs (exact mathematical zeros, ultra-low sub-audible noise, or a single delta impulse), the network produces diffuse, near-uniform class probabilities where the maximum probability is only ~30% (below the experimental confidence threshold of 0.50). Under the game adapter rules, any prediction below 0.50 is safely suppressed to `NO_ACTION`. In contrast, realistic ambient room tone with dither noise (`case_06`) is classified as `SILENCE` with 96.36% confidence.

---

## 4. Cross-Platform Mobile Frontend Evaluation & Technical Blockers

### Existing Toolchain Inspection
- **Compilers Found**: MinGW `gcc.exe` (via MSYS2 / Strawberry Perl).
- **Available Audio C/C++ Libraries**: None. The repository contains no third-party C/C++ FFT or DSP library (such as KissFFT, PFFFT, FFTW, or CMSIS-DSP).
- **Portable Implementation Path**: Rather than introducing an unvetted dependency, the verified Python reference ([`extract_log_mel_numpy`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py)) serves as the gold standard.

### Integration Protocol for Android & iOS Frontends
Before connecting a mobile audio capture pipeline to ONNX Runtime on Android (C++/NDK) or iOS (Swift/Accelerate), mobile developers must execute the following offline parity test:

```
[Golden Test WAV (e.g. case_01_trai_demo.wav)]
                     |
        Mobile C++ / Swift STFT & Mel
                     |
        [Candidate Mobile Feature Tensor] <--- Compare with ---> [Saved Golden Tensor (.npy)]
                     |                                                    |
                     +----------------> Check: Max Diff <= 1.0e-4? <------+
                                              |
                                             YES
                                              |
                               [Pass: Safe to feed ONNX Runtime]
```

### Technical Blockers for Full Mobile Deployment
1. **STFT Centering & Reflect Padding**: Many mobile DSP examples omit reflect padding or use asymmetric zero-padding, producing 98 or 100 frames instead of exactly 101 frames.
2. **Mel Filterbank Discrepancy**: Default Android/iOS libraries typically implement HTK triangular filters rather than Slaney area-normalized filters. A precomputed $40 \times 257$ float32 matrix derived from `librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, norm="slaney", mel_scale="slaney")` must be bundled as an asset.
3. **Log Epsilon Formula**: Must use natural logarithm $\ln(x + 10^{-6})$ rather than $\log_{10}(x)$ or $\max(\log(x), \text{floor})$.

---

## 5. Architectural Separation of Parity Stages

To prevent conflation of software export parity with device readiness, Phase 7 formalizes four distinct evaluation boundaries:

```
+-----------------------------------------------------------------------------+
| Stage 1: Python Reference Reproducibility                                   |
|   Status: COMPLETE (12/12 Golden Vectors reproduce with 0.00e+00 error)     |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Stage 2: PyTorch-to-ONNX Neural Graph Numerical Parity                      |
|   Status: COMPLETE (356/356 inputs match argmax; max prob diff: 7.75e-07)   |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Stage 3: Cross-Implementation Mobile Frontend Parity                        |
|   Status: UNVERIFIED (Awaiting C++/Swift DSP implementation against golden)  |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Stage 4: Physical Mobile Hardware & Acoustic Performance                    |
|   Status: UNTESTED (Microphone hardware, AGC, battery, thermal unmeasured)  |
+-----------------------------------------------------------------------------+
```

---

## 6. Real-World Limitations & Governance Invariant

> [!WARNING]
> ### Real-World Boundary & Governance Notice
> 1. **Synthetic Fixtures Only**: All golden vectors and audio inputs were synthesized using parametric formant models or captured from previous synthetic runs.
> 2. **Untested Human Voice**: The system has **never been evaluated on live or recorded human speech**.
> 3. **Vietnamese Dialects**: Dialectal variations in real human speakers (e.g. Hoi-Nga tone merger, coda neutralization) remain untested.
> 4. **Microphone Hardware & AGC**: Mobile microphone clipping, automatic gain control, and reverberation have not been validated.
> 5. **Governance Rule**: **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** No human recordings may be acquired, uploaded, or processed without explicit regulatory and ethics protocol clearance.
