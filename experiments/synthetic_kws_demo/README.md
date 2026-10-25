# Phase 5: Frozen Synthetic KWS Offline Inference & Game-Control Prototype

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> 
> **SYNTHETIC-DEMO-ONLY**: This inference engine uses a model trained exclusively on synthetic acoustic fixtures. Its high synthetic accuracy (99.33% on V4 native holdout) DOES NOT establish real-world speech accuracy, Vietnamese regional accent invariance, live microphone reliability, or mobile edge deployment performance. HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED. This prototype operates strictly offline on synthetic audio fixtures generated via parametric acoustic modeling. It does NOT access live microphones, ingest external speech corpora, or recruit human speakers. All reported metrics reflect synthetic holdout performance only.

---

## 1. Executive Summary & Prototype Scope

Phase 5 delivers a complete, modular, and strictly validated offline Keyword Spotting (KWS) inference engine and voice-driven game-control demonstration for Vietnamese commands (`TRAI`, `PHAI`, `LEN`, `XUONG`). 

### Core Architectural Separation
The prototype enforces strict decoupling across three independent layers:
1. **Inference Engine (`classifier.py`)**: Responsible for strict PCM audio validation, Log-Mel spectrogram extraction, and neural network forward pass using the frozen primary candidate model. Emits structured predictions with timing telemetry and synthetic caveats.
2. **Game-Control Adapter (`game_adapter.py`)**: Decoupled state translation layer that converts raw KWS predictions into discrete game actions (`MOVE_LEFT`, `MOVE_RIGHT`, `JUMP`, `CROUCH`, `NO_ACTION`). Enforces strict non-target safety suppression and configurable confidence thresholding.
3. **2D Grid Simulator (`game_simulator.py`)**: Lightweight character simulator on a bounded 2D grid containing obstacles and goal markers, executing discrete displacement deltas and reporting obstacle collision and boundary events.
4. **Command-Line Interface (`cli.py`)**: Interactive CLI supporting single-audio inference (human-readable or JSON) and full end-to-end sequential demonstration across all 6 acoustic classes.

---

## 2. Model Architecture & Preprocessing Specifications

The prototype loads the frozen **V4 Arm B** champion checkpoint without modification or retraining.

| Component | Specification | Description / Value |
| :--- | :--- | :--- |
| **Model Architecture** | `VietnameseTCResNet` | TC-ResNet-14 with 1D temporal convolutions, residual blocks, and 6-class linear projection |
| **Checkpoint Path** | `arm_b_onset_no_resample.pt` | `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt` |
| **Checkpoint SHA-256** | `6A0FD55CD6F1...CF1D96` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` |
| **Input Feature Format** | Log-Mel Spectrogram | `(Batch=1, Channels=1, Mel_Bands=40, Time_Frames=101)` |
| **Frontend Extractor** | `LogMelFeatureExtractor` | PyTorch native frontend reused from `src/preprocessing/feature_extractor.py` |
| **Sample Rate** | 16,000 Hz | Exact match required. Silent resampling is strictly prohibited |
| **FFT / Window Size** | 512 / 400 samples | 25.0 ms Hamming analysis window |
| **Hop Length** | 160 samples | 10.0 ms frame advance rate |
| **Mel Filterbank** | 40 filter banks | Slaney normalized triangular filterbank (0 Hz to 8,000 Hz Nyquist) |
| **Compression** | Natural Log | $\ln(\text{Mel} + 10^{-6})$ |
| **Audio Format** | Native V4 Alignment | Left-aligned active speech, right-padded with zeros to exactly 16,000 samples |
| **Ambient Preservation** | Unshifted Full Buffer | Full 16,000 samples preserved unshifted without speech-onset cropping |

### Strict Audio Input Validation Policy
The `AudioValidator` enforces zero tolerance for invalid audio inputs without silent degradation:
- **Sample Rate**: Must equal exactly 16,000 Hz. Audio at 8,000 Hz or 44,100 Hz raises `InvalidSampleRateError`.
- **Channel Count**: Must be 1 (mono). Stereo audio raises `InvalidChannelCountError`.
- **Duration / Frame Count**: Must be exactly 16,000 samples (1.000s). Shorter or longer audio raises `InvalidDurationError`.
- **File Integrity**: Zero-byte or truncated files raise `EmptyAudioError` or `CorruptedAudioError`.

---

## 3. Synthetic Demonstration & Test Fixture Inventory

All demonstration fixtures are generated from the parametric Onset-Aware Synthetic Acoustic Generator using dedicated demo speaker `SPK_DEMO_001` (disjoint from training and validation calibration speakers) and recorded in `experiments/synthetic_kws_demo/fixtures/fixture_manifest.csv`.

| Fixture Filename | Class ID | Label | Intended Action | Duration / Samples | Whole-File SHA-256 | Description |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `demo_01_trai.wav` | 0 | `TRAI` | `MOVE_LEFT` | 1.0s / 16,000 | `F4D8BFB19B6858CD0367A6CFEF7569E66F42D50EC4BE6E5225EDFCDAC9E91BFA` | Native onset-aware "trái" stop-closure burst |
| `demo_02_phai.wav` | 1 | `PHAI` | `MOVE_RIGHT` | 1.0s / 16,000 | `313A900BD5E44856E21F171092F6AF06C04F038F37FF8C6DB2137D71714FE675` | Native onset-aware "phải" continuous frication |
| `demo_03_len.wav` | 2 | `LEN` | `JUMP` | 1.0s / 16,000 | `B77DA0C254B86A8ED0A087AC06775C53896A621D9B0A2665B20399AE1D5876CE` | Native onset-aware "lên" level tone utterance |
| `demo_04_xuong.wav` | 3 | `XUONG` | `CROUCH` | 1.0s / 16,000 | `9A501FA097F563B3B7071F956A1521DFD975ED9DA83656C6C759C40FD45A0DB1` | Native onset-aware "xuống" sibilant utterance |
| `demo_05_unknown.wav` | 4 | `UNKNOWN` | `NO_ACTION` | 1.0s / 16,000 | `87D97EF62CE65EEF9E6C38AC44A48BE1B0D86AF1A3919C74E06A339B23F01EB7` | Out-of-vocabulary non-target confuser ("chào") |
| `demo_06_silence.wav` | 5 | `SILENCE` | `NO_ACTION` | 1.0s / 16,000 | `6AE179B7EDFD4689E62C8C673B880E9EDEF2A18287C6F12C092EF5BA19BE462A` | Full-buffer unshifted ambient dither noise |
| `invalid_stereo.wav` | -1 | `INVALID` | `ERROR_EXPECTED` | 1.0s / 16,000 | `38AC9F8E41FA8B63F69D0F04D38EC4FE76F6F9D70CD9C3E1482EAAEF52A88BD7` | Malformed: 2 channels (stereo) |
| `invalid_sample_rate_8k.wav`| -1 | `INVALID` | `ERROR_EXPECTED` | 1.0s / 8,000 | `872D1EAE2F7016BC24151CFFB09121E429B5C827117ED2C4E3DDFCE7943F2A4F` | Malformed: 8,000 Hz sample rate |
| `invalid_short_duration.wav`| -1 | `INVALID` | `ERROR_EXPECTED` | 0.5s / 8,000 | `ECB93BD01FDE647C99F1DE13D6AF413AE5755106DB5D6C76EB859AE49520443D` | Malformed: 8,000 samples duration (0.5s) |
| `invalid_empty.wav` | -1 | `INVALID` | `ERROR_EXPECTED` | 0.0s / 0 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` | Malformed: 0-byte empty file |

---

## 4. Game-Control Adapter Mechanics & Safety Suppression

The `CommandToActionAdapter` translates KWS model output probabilities into deterministic game mechanics:

```
[Raw Audio File] -> AudioValidator -> LogMelFeatureExtractor -> TC-ResNet-14
                                                                      |
                                                               [PredictionResult]
                                                                      |
                                                          [CommandToActionAdapter]
                                                          /          |           \
                                                    Target?     Confidence?     Non-Target?
                                                     /               |               \
                                        Active Displacement      Threshold Check    NO_ACTION (Suppressed)
                                                    \                |               /
                                                    [AdaptedAction (delta_x, delta_y)]
                                                                     |
                                                           [GridGameSimulator]
```

### Safety Guarantees
1. **Unconditional Non-Target Safety Suppression**: Any prediction of `UNKNOWN` or `SILENCE` unconditionally triggers `GameAction.NO_ACTION` with `delta_x = 0`, `delta_y = 0`, and `is_active = False`. The game character's position is completely unaffected.
2. **Experimental Confidence Thresholding**: Target commands must meet or exceed a configurable threshold (default `0.50`). Predictions below the threshold are suppressed to `NO_ACTION`.
3. **Debouncing Cooldown**: An optional debouncing cooldown (default `0.50s`) prevents accidental double-triggers from trailing speech frames.

---

## 5. Desktop CPU Latency Measurements

All latency benchmarks were measured on desktop CPU using Python's high-precision `time.perf_counter()` over the 6 canonical demonstration fixtures.

| Fixture | Command | Preprocessing (ms) | Inference (ms) | Total Latency (ms) | Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `demo_01_trai.wav` | `TRAI` | 2.52 ms | 1.77 ms | 4.29 ms | 88.40% |
| `demo_02_phai.wav` | `PHAI` | 0.45 ms | 1.32 ms | 1.77 ms | 99.30% |
| `demo_03_len.wav` | `LEN` | 0.65 ms | 1.58 ms | 2.23 ms | 96.86% |
| `demo_04_xuong.wav` | `XUONG` | 0.38 ms | 1.30 ms | 1.68 ms | 98.44% |
| `demo_05_unknown.wav` | `UNKNOWN` | 0.72 ms | 1.44 ms | 2.16 ms | 98.87% |
| `demo_06_silence.wav` | `SILENCE` | 0.58 ms | 1.34 ms | 1.92 ms | 96.36% |
| **Cohort Average** | — | **0.88 ms** | **1.46 ms** | **2.34 ms** | **96.37%** |

> [!NOTE]
> Desktop CPU latency (~2.34 ms total per 1-second audio frame) demonstrates that offline feature extraction and neural inference introduce negligible compute overhead on desktop hardware. This measurement reflects desktop Python/PyTorch execution and does NOT establish mobile edge latency, memory constraints, or NPU acceleration.

---

## 6. Reproducible Execution Commands

All commands are executed from the repository root `D:\trainidentificationaudio`.

### 1. Run Full Sequential Game Demonstration
Executes sequential simulation over all 6 audio fixtures, printing ASCII grid transitions and action reports:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/cli.py --play-demo
```

### 2. Single-File Inference (Standard Output)
Classifies a single WAV file, outputting label, confidence, per-class probabilities, and latency breakdown:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/cli.py --wav experiments/synthetic_kws_demo/fixtures/demo_01_trai.wav
```

### 3. Single-File Inference (JSON Output)
Returns structured, machine-parsable JSON including full telemetry, adapted action deltas, and governance warnings:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/cli.py --wav experiments/synthetic_kws_demo/fixtures/demo_03_len.wav --json
```

### 4. Custom Confidence Thresholding
Suppresses commands with lower confidence:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/cli.py --wav experiments/synthetic_kws_demo/fixtures/demo_01_trai.wav --threshold 0.95
```

### 5. Negative Audio Testing (Rejection Contract)
Demonstrates rejection of malformed stereo audio:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/cli.py --wav experiments/synthetic_kws_demo/fixtures/invalid_stereo.wav --json
```

### 6. Run Prototype Test Suite
Executes the 19 dedicated integration and unit tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_kws_demo.py -v
```

### 7. Run Full Campaign Acceptance Suite (TC-01 through TC-65)
Executes all 65 campaign acceptance tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_acceptance.py -q
```

---

## 7. Synthetic Boundary & Real-World Limitations

> [!WARNING]
> ### Critical Real-World Boundary Notice
> 1. **Synthetic-Only Baseline**: The underlying model (`arm_b_onset_no_resample.pt`) was trained and evaluated exclusively on synthetic speech generated via parametric formant and onset modeling. Its 99.33% accuracy applies strictly to the V4 synthetic holdout partition.
> 2. **Untested Natural Speech**: The system has **never been evaluated on natural human voice**. It cannot be assumed to generalize to human speech, informal colloquial speech, or unmodeled tones.
> 3. **Vietnamese Accent Limitations**: While synthetic dialect tokens (Northern, Central, Southern) were parameter-modeled, regional vowel shifts, glottalization, and tonal mergers in real human dialects remain untested.
> 4. **Live Microphone & Acoustics**: Acoustic effects such as room reverberation, microphone frequency distortion, clipping, clipping-recovery, background conversation, and varying signal-to-noise ratios (SNR) have not been verified.
> 5. **Edge Deployment Readiness**: Desktop CPU execution timings (~2.34 ms) demonstrate software correctness only. They do not benchmark thermal, battery, quantized INT8, or DSP/NPU performance on embedded microcontrollers or mobile operating systems.
> 6. **Human Data Governance**: **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** No human recordings may be acquired or ingested without unblocking governance protocols.

---

## 8. Phase 6 ONNX Export & Mobile Preprocessing Contract

Phase 6 exported the frozen V4 Arm B model to ONNX and completed an exhaustive runtime-parity audit against PyTorch. Full details are recorded in [`export/export_audit.md`](export/export_audit.md).

### Model Export Status
- **Exported Artifact**: [`export/arm_b_tcresnet14.onnx`](export/arm_b_tcresnet14.onnx) (1,095,036 bytes, Opset 17)
- **Model Checkpoint Source**: `arm_b_onset_no_resample.pt` (SHA-256: `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96`)
- **Exported ONNX SHA-256**: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625`
- **Audit Verdict**: **`PARITY_AUDIT_PASS`**
  - Argmax agreement: **100.00% (356 / 356 samples)** across demo fixtures, V4 holdout (300 clips), and synthetic tensors.
  - Global maximum absolute difference: **$7.75 \times 10^{-7}$** (Threshold: $\le 1.0 \times 10^{-4}$).
  - Isolated CPU forward pass: **0.303 ms** in ONNX Runtime vs. **0.925 ms** in PyTorch (3.05x speedup).

### Mobile Preprocessing Contract & Deployment Limitations
- **Graph Input Contract**: The ONNX model accepts pre-computed **Log-Mel spectrograms** of shape `(Batch, 1, 40, 101)`. It does **NOT** accept raw audio.
- **Audio Frontend Requirements**:
  - Sample rate: 16,000 Hz mono (1.000s / 16,000 samples).
  - STFT: $N_{\text{FFT}} = 512$, window length 400 (25ms), hop length 160 (10ms), reflect padding, center=True.
  - Mel Filterbank: 40 Slaney area-normalized triangular bands (20 Hz to 8,000 Hz).
  - Compression: Natural log $\ln(\text{Mel} + 1.0 \times 10^{-6})$.
- **Mobile Golden Parity Prerequisite**: An Android or iOS audio frontend must be independently compared and verified against [`extract_log_mel_numpy`](../../src/preprocessing/feature_extractor.py) before claiming mobile edge readiness.
- **Reproducible Script**: Run `python experiments/synthetic_kws_demo/export/export_and_benchmark.py` to regenerate the export and audit.

---

## 9. Phase 7 Golden-Vector Suite & Mobile Frontend Parity Contract

Phase 7 establishes an immutable regression suite of 12 golden test cases, documents the verified executable frontend contract, and resolves documentation ambiguities. Full details are recorded in [`frontend_golden/frontend_contract.md`](frontend_golden/frontend_contract.md).

### Golden-Vector Artifacts & Verdict
- **Master Manifest**: [`frontend_golden/golden_manifest.json`](frontend_golden/golden_manifest.json) (12 cases: 6 demo commands, 2 timing variants, 3 numerical edge cases, 1 historical holdout anchor).
- **Discrepancy Record**: [`frontend_golden/discrepancy_log.json`](frontend_golden/discrepancy_log.json) (DISC-09: Ambiguous window resolved to **Periodic Hann Window** `torch.hann_window(400, periodic=True)`).
- **Parity Report**: [`frontend_golden/feature_parity_report.json`](frontend_golden/feature_parity_report.json)
- **Suite Verdict**: **`GOLDEN_VERIFICATION_PASS`**
  - Feature recomputation difference: **$0.00 \times 10^{0}$** across all 12 cases against saved `.npy` tensors.
  - PyTorch vs. ONNX Runtime prediction difference: **$\le 2.09 \times 10^{-7}$** (Threshold: $\le 1.0 \times 10^{-4}$).

### Four-Stage Parity Status
1. **Stage 1: Python Reference Reproducibility** $\rightarrow$ **VERIFIED** ($0.00 \times 10^0$ feature error on 12/12 cases).
2. **Stage 2: PyTorch-to-ONNX Neural Graph Parity** $\rightarrow$ **VERIFIED** ($7.75 \times 10^{-7}$ global probability error on 356 samples).
3. **Stage 3: Cross-Implementation Mobile Frontend Parity** $\rightarrow$ **VERIFIED** in Phase 8 ($3.62 \times 10^{-5}$ max feature error on 12/12 golden cases; $1.79 \times 10^{-7}$ max prob error).
4. **Stage 4: Physical Mobile Hardware & Acoustic Performance** $\rightarrow$ **UNTESTED** (Microphone AGC, reverberation, and device edge thermals unmeasured).
- **Governance Notice**: **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**

---

## 10. Phase 8 Portable C/C++ Frontend Parity & Mobile Verification

Phase 8 implements and verifies a dependency-minimal, portable C/C++ audio preprocessing frontend (`kws_mobile_frontend`) reproducing the exact Python `LogMelFeatureExtractor` features for mobile deployment (Android NDK / iOS Swift). Full details are recorded in [`mobile_frontend/mobile_frontend_audit.md`](mobile_frontend/mobile_frontend_audit.md).

### Mobile Frontend Artifacts & Verdict
- **Public Header & Implementation**: [`mobile_frontend/include/kws_mobile_frontend.h`](mobile_frontend/include/kws_mobile_frontend.h) and [`mobile_frontend/src/kws_mobile_frontend.cpp`](mobile_frontend/src/kws_mobile_frontend.cpp)
- **Mel Filterbank Header & Asset**: [`mobile_frontend/include/mel_filterbank_40x257.h`](mobile_frontend/include/mel_filterbank_40x257.h) ($40 \times 257$ Slaney area-normalized table)
- **Standalone CLI Utility**: [`mobile_frontend/bin/run_frontend_cli.exe`](mobile_frontend/bin/run_frontend_cli.exe)
- **C++ Unit Test Runner**: [`mobile_frontend/bin/test_cxx_frontend.exe`](mobile_frontend/bin/test_cxx_frontend.exe) (15,185 assertions passed)
- **Audit Verdict**: **`HOST_CROSS_IMPLEMENTATION_PARITY_PASS`**
  - Host-level cross-implementation verification on Windows DLL against torchaudio reference.
  - Global maximum feature difference: **$3.62 \times 10^{-5}$** across all 12 golden vectors (Threshold: $\le 1.0 \times 10^{-4}$).
  - Global maximum ONNX probability difference: **$1.79 \times 10^{-7}$** (Threshold: $\le 1.0 \times 10^{-4}$).
  - Argmax agreement: **100.0% (12 / 12 cases)**.
  - Zero-signal exactness: **$0.00 \times 10^0$** feature difference on pure zeros.
  - Safety suppression: Artificial numerical edge cases (pure zeros, ambient noise floor, transient impulse) all produce confidence $< 0.32$ and trigger `NO_ACTION` at threshold 0.50.
  - Note: Windows host DLL test does not establish Android NDK or iOS edge execution. Physical device testing remains unverified.

### Reproducible Commands
```powershell
# 1. Build C++ frontend library, unit tests, and CLI runner
powershell -ExecutionPolicy Bypass -File experiments/synthetic_kws_demo/mobile_frontend/build.ps1

# 2. Run standalone C++ unit tests (15,185 assertions)
.\experiments\synthetic_kws_demo\mobile_frontend\bin\test_cxx_frontend.exe

# 3. Run golden comparison and ONNX parity harness
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/mobile_frontend/compare_with_golden.py
```

---

## 11. Phase 9 Android Offline KWS Integration and ONNX Runtime Validation

Phase 9 integrates the KWS pipeline into an Android application project under [`android_app/`](android_app/) reusing the canonical C++ Log-Mel frontend through JNI and Microsoft ONNX Runtime Mobile for offline synthetic fixture inference. Full details are recorded in [`android_app/PHASE9_ANDROID_AUDIT.md`](android_app/PHASE9_ANDROID_AUDIT.md).

### Android Architecture & Governance
- **Zero Permissions**: `AndroidManifest.xml` declares 0 permissions. Microphone capture (`RECORD_AUDIO`) and network (`INTERNET`) are strictly excluded.
- **Packaged Model**: [`android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx`](android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx) matches the frozen source model bit-exact (`8A5AC4C1...`).
- **JNI Bridge**: [`android_app/app/src/main/cpp/kws_jni_bridge.cpp`](android_app/app/src/main/cpp/kws_jni_bridge.cpp) provides memory-safe, bounds-checked bindings to the canonical C++ frontend.
- **NDK Build**: [`android_app/app/src/main/cpp/CMakeLists.txt`](android_app/app/src/main/cpp/CMakeLists.txt) compiles the native shared library referencing the canonical frontend directly.
- **Safety Adaptation**: [`android_app/app/src/main/java/com/example/synthetic_kws/GameActionAdapter.kt`](android_app/app/src/main/java/com/example/synthetic_kws/GameActionAdapter.kt) suppresses non-target commands and low-confidence edge cases to `NO_ACTION`.

### Gate-by-Gate Verification Status
- **Gate A — Host-side / static validation**: **`PASS`** (12/12 golden vectors match within $3.62 \times 10^{-5}$ feature error and $1.79 \times 10^{-7}$ prob error; edge cases suppressed).
- **Gate B — Android NDK compilation**: **`BLOCKED`** (Host lacks Android NDK; no target `.so` libraries compiled).
- **Gate C — APK build**: **`BLOCKED`** (Host lacks JDK, Gradle, and Android SDK; no APK assembled).
- **Gate D — Android runtime inference**: **`BLOCKED`** (Requires built APK and emulator/device via `adb`; no runtime execution occurred).
- **Gate E — Physical-device performance**: **`NOT_TESTED`** (Requires attached physical Android hardware).
- **Overall Verdict**: **`BLOCKED`** (The overall Phase 9 verdict is BLOCKED for the Android deployment objective while the required toolchain is unavailable; a project scaffold or Windows DLL run does NOT constitute an Android deployment).
- **Platform Scope**: Android verification does not establish iOS readiness (tracked as an independent future platform gate).

### Reproducible Host Validation Command
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/android_app/validate_android_pipeline.py
```




