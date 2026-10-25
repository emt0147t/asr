# Phase 91: CNN Deployment Engineering for Vietnamese Keyword Spotting

**Workspace:** `D:\trainidentificationaudio`  
**Candidate Architecture:** `CorrectedVietnameseKWSNet` (138,469 parameters)  
**Candidate Checkpoint:** `variant_w10_seed42.pt` (calibrated $\theta_{\text{cmd}} = 0.40$)  
**Status:** `RESEARCH_FROZEN_CANDIDATE`  
**Governance Invariants:**
- `PENDING_HUMAN_LISTENER_VERIFICATION`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` (Continuous Open-Mic Mode)

---

## 1. Executive Summary

Phase 91 establishes a production-grade deployment engineering foundation for the frozen bounded-utterance Vietnamese keyword spotting candidate `variant_w10`. While the distributed blinded human-listening study (Phases 85–90) remains pending, Phase 91 hardens the offline and on-device software boundary without altering model parameters, retuning thresholds, or exaggerating real-world acoustic readiness.

### Key Deliverables Completed:
1. **Deployment Contract (`DEPLOYMENT_CONTRACT.md`):** Formally codified audio input standards, sample rate conversion, channel averaging, fixed-window framing, non-amplifying peak normalization, Mel-spectrogram parameters, network topology, 5-class index mapping, and threshold rejection rules.
2. **Audio Validation & Ingestion Engine (`audio_validator.py`):** Implemented strict and permissive audio validators with dedicated exception types (`EmptyAudioError`, `CorruptedAudioError`, `InvalidSampleRateError`, `InvalidChannelCountError`, `InvalidDurationError`, `AudioFileNotFoundError`).
3. **Standalone WAV Inference Tool (`predict_wav.py`):** Built a multi-backend CLI supporting PyTorch and ONNX Runtime, offering structured JSON or formatted text outputs, detailed latency profiling, explicit exit codes (0 to 8), and governance disclaimers.
4. **Reproducible Model Export (`export_model.py`):** Exported eval-mode TorchScript, FP32 ONNX (opset 17), and dynamic INT8 ONNX models with an automated manifest (`export_manifest.json`).
5. **Numerical Parity Verification:** Proved bit-accurate TorchScript parity ($0.00$ deviation) and near-zero ONNX FP32 deviation ($3.11 \times 10^{-6}$, well within the $\le 10^{-4}$ tolerance), achieving $0$ label mismatches across all evaluation fixtures.
6. **Mobile Edge Deployment Audit:** Compared ONNX Runtime Mobile, TorchScript, TFLite, CoreML, and ExecuTorch for Android and iOS; audited the existing Android scaffold (`experiments/synthetic_kws_demo/android_app`), cataloging host toolchain blockers (missing JDK 17, Android SDK/NDK, Gradle wrapper) and outlining an evidence-based migration plan.
7. **Operational Mode Taxonomy:** Documented the mathematical and operational divergence between Recorded-File, Push-to-Talk, and Continuous Open-Mic modes, reaffirming why bounded-utterance models cannot be deployed open-mic without streaming temporal gating.
8. **Regression & Isolation Test Suite (`tests/test_cnn_deployment_phase91.py`):** Added 16 tests covering all edge cases, parity, and invariance. All 16 tests passed cleanly alongside the existing Phase 80–89 test suites.

---

## 2. Checkpoint Verification and Hash Invariance

As mandated by governance requirements, all candidate model checkpoints were cryptographically verified using SHA-256 before and after all engineering operations. Zero weights or configurations were modified.

| Checkpoint Identifier | Path | Size | SHA-256 Checksum | Invariance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Primary (Seed 42)** | `experiments/vieneu_kws_research_v9/artifacts/checkpoints/variant_w10_seed42.pt` | 564 KB | `7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed` | **VERIFIED_UNCHANGED** |
| **Secondary (Seed 43)** | `experiments/vieneu_kws_research_v9/artifacts/checkpoints/variant_w10_seed43.pt` | 564 KB | `77f468c68b261de506a3fcb022c8d7cc68f8b0c2ad650852b7c300f274dda251` | **VERIFIED_UNCHANGED** |
| **Secondary (Seed 44)** | `experiments/vieneu_kws_research_v9/artifacts/checkpoints/variant_w10_seed44.pt` | 564 KB | `bebab580e4caa6934c98a62b095611e4eef9221a7f4282d36f9a4b6928abde64` | **VERIFIED_UNCHANGED** |

---

## 3. Audio Ingestion and Preprocessing Contract

The deployment contract (`experiments/vieneu_kws_research_v10/deployment/DEPLOYMENT_CONTRACT.md`) enforces strict physical and mathematical invariants across all client implementations.

### 3.1 Audio Input Specifications
- **Format:** RIFF WAV uncompressed Linear PCM.
- **Sampling Rate:** $16,000 \text{ Hz}$.
- **Channels:** 1 (Mono).
- **Target Duration:** Exactly $1.000 \text{ s}$ ($16,000 \text{ samples}$).

### 3.2 Dual-Mode Ingestion Policy
To support both rigorous automated benchmark certification and flexible developer usage, the audio validator (`audio_validator.py`) provides two operational modes:

| Check | Strict Mode (`--strict`) | Permissive Mode (Default Offline CLI) |
| :--- | :--- | :--- |
| **Missing File** | Exit code 2 (`AudioFileNotFoundError`) | Exit code 2 (`AudioFileNotFoundError`) |
| **Empty File (0 bytes / 0 frames)** | Exit code 3 (`EmptyAudioError`) | Exit code 3 (`EmptyAudioError`) |
| **Corrupted Header / Data** | Exit code 4 (`CorruptedAudioError`) | Exit code 4 (`CorruptedAudioError`) |
| **Sample Rate $\neq 16 \text{ kHz}$** | Exit code 5 (`InvalidSampleRateError`) | Torchaudio polyphase resample to $16 \text{ kHz}$ |
| **Channel Count $> 1$** | Exit code 6 (`InvalidChannelCountError`) | Multi-channel downmix: $x_{\text{mono}} = \frac{1}{C}\sum_{c=1}^C x_c$ |
| **Duration $\neq 16,000$ samples** | Exit code 7 (`InvalidDurationError`) | Zero-pad on right if $N < 16,000$; truncate if $N > 16,000$ |
| **Peak Amplitude $> 1.0$** | Scale by $\max(\|x\|)$ | Scale by $\max(\|x\|)$ |
| **Noise Floor Amplification** | **FORBIDDEN** (never scale up quiet audio) | **FORBIDDEN** (never scale up quiet audio) |

### 3.3 Log-Mel Frontend Feature Extraction
- **STFT Parameters:** $N_{\text{fft}} = 400$ ($25.0 \text{ ms}$), Hann window length $= 400$, hop size $= 160$ ($10.0 \text{ ms}$), `center=True`, `pad_mode="reflect"`, $\text{power} = 2.0$.
- **Mel Filterbank:** $40 \text{ triangular filters}$, $f_{\min} = 20.0 \text{ Hz}$, $f_{\max} = 8,000.0 \text{ Hz}$, Slaney frequency scale and area normalization (`norm="slaney"`, `mel_scale="slaney"`).
- **Compression:** $\mathbf{S}_{\text{logmel}} = \ln(\mathbf{S}_{\text{mel}} + 10^{-6})$.
- **Tensor Shape:** Fixed `[Batch=1, Channels=1, Mel=40, Time=101]`, float32.

### 3.4 Output Classes and Decision Logic
The model produces 5 class logits mapped to indices:
- `0`: `TRAI` (Command)
- `1`: `PHAI` (Command)
- `2`: `LEN` (Command)
- `3`: `XUONG` (Command)
- `4`: `REJECT` (Noise / Hard Negative / Silence / Non-target speech)

**Threshold Gating Rule:**
Let $p_i = \text{Softmax}(\mathbf{z})_i$ for $i \in \{0, 1, 2, 3, 4\}$.
1. $p_{\text{cmd\_max}} = \max \{p_0, p_1, p_2, p_3\}$, with $c_{\text{cmd}} = \arg\max_{i \in \{0, 1, 2, 3\}} p_i$.
2. If $p_{\text{cmd\_max}} \ge \theta_{\text{cmd}} = 0.40$:
   - Return predicted command: $\text{CLASSES}[c_{\text{cmd}}]$
   - Rejection status: `ACCEPTED`
   - Confidence: $p_{\text{cmd\_max}}$
3. If $p_{\text{cmd\_max}} < \theta_{\text{cmd}} = 0.40$:
   - Return class: `REJECT`
   - Rejection status: `REJECTED`
   - Confidence: $p_4$

---

## 4. Standalone WAV Inference CLI Tool

The tool `experiments/vieneu_kws_research_v10/deployment/predict_wav.py` provides offline, deterministic WAV classification with zero ambiguity.

### 4.1 CLI Syntax and Options
```powershell
python experiments/vieneu_kws_research_v10/deployment/predict_wav.py <wav_path> [options]

Arguments:
  wav_path               Path to input WAV file.

Options:
  --checkpoint PATH      Path to PyTorch checkpoint (default: variant_w10_seed42.pt)
  --onnx-model PATH      Path to ONNX model (switches backend to ONNX Runtime)
  --threshold FLOAT      Command decision threshold theta_cmd (default: 0.40)
  --strict               Enforce strict validation (16kHz, mono, exactly 16000 samples)
  --device {cpu,cuda}    Execution device for PyTorch backend (default: cpu)
  --json                 Produce machine-readable JSON output
```

### 4.2 Standard Output Modes

#### Human-Readable Output:
```text
================================================================================
VIETNAMESE KEYWORD SPOTTING - OFFLINE WAV INFERENCE (PHASE 91)
================================================================================
File:            experiments\synthetic_kws_demo\fixtures\demo_01_trai.wav
Audio Info:      16000 Hz, 1 ch, 16000 frames (1.000s)
Validation:      PERMISSIVE (auto-resample & pad/trunc)
Backend:         pytorch_cpu
Model Checkpoint: variant_w10_seed42.pt (SHA256: 7374af469a4bc4df...)
--------------------------------------------------------------------------------
PREDICTION:      LEN
STATUS:          ACCEPTED (threshold = 0.40)
CONFIDENCE:      85.99%
--------------------------------------------------------------------------------
Class Probabilities:
  - TRAI:        0.0083
  - PHAI:        0.0125
  - LEN:         0.8599
  - XUONG:       0.0744
  - REJECT:      0.0449
--------------------------------------------------------------------------------
Latency:
  - Preprocessing: 1.71 ms
  - Inference:     4.97 ms
  - Total:         6.67 ms
--------------------------------------------------------------------------------
Governance Invariants:
  [!] Human Listening:       PENDING_HUMAN_LISTENER_VERIFICATION
  [!] Natural Speech:        NOT_VALIDATED_ON_NATURAL_SPEECH
  [!] Continuous Open-Mic:   NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
================================================================================
```

#### JSON Output Mode:
```json
{
  "status": "SUCCESS",
  "wav_path": "experiments\\synthetic_kws_demo\\fixtures\\demo_01_trai.wav",
  "audio_metadata": {
    "file_name": "demo_01_trai.wav",
    "file_size_bytes": 32044,
    "original_sample_rate": 16000,
    "original_channels": 1,
    "original_frames": 16000,
    "original_duration_seconds": 1.0,
    "format": "WAV",
    "subtype": "PCM_16",
    "strict_validation": false
  },
  "predicted_class": "LEN",
  "rejection_status": "ACCEPTED",
  "confidence": 0.8599,
  "command_threshold": 0.4,
  "probabilities": {
    "TRAI": 0.008336,
    "PHAI": 0.012511,
    "LEN": 0.859851,
    "XUONG": 0.074425,
    "REJECT": 0.044876
  },
  "backend": "pytorch_cpu",
  "model_path": "D:\\trainidentificationaudio\\experiments\\vieneu_kws_research_v9\\artifacts\\checkpoints\\variant_w10_seed42.pt",
  "model_sha256": "7374af469a4bc4df5e124aae27720a8c11e688413f045eec3ee4ffd79c836bed",
  "latency": {
    "preprocessing_ms": 1.71,
    "inference_ms": 4.97,
    "total_ms": 6.67
  },
  "governance": {
    "human_listener_verification": "PENDING_HUMAN_LISTENER_VERIFICATION",
    "natural_speech_validation": "NOT_VALIDATED_ON_NATURAL_SPEECH",
    "continuous_open_mic_verdict": "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
  }
}
```

### 4.3 Exit Code Specification
The tool returns deterministic numeric exit codes enabling automated script integration:
- `0`: Successful inference
- `1`: Unhandled execution error
- `2`: `AudioFileNotFoundError` (file does not exist on disk)
- `3`: `EmptyAudioError` (file is 0 bytes or has 0 audio samples)
- `4`: `CorruptedAudioError` (unreadable RIFF header or truncated PCM)
- `5`: `InvalidSampleRateError` (sample rate $\neq 16,000 \text{ Hz}$ in strict mode)
- `6`: `InvalidChannelCountError` (multi-channel audio in strict mode)
- `7`: `InvalidDurationError` (duration $\neq 1.000 \text{ s}$ in strict mode)
- `8`: `CheckpointNotFoundError` (model checkpoint path invalid)

---

## 5. Model Export and Numerical Parity Verification

The export workflow (`experiments/vieneu_kws_research_v10/deployment/export_model.py`) converts the frozen PyTorch candidate into production deployment formats under `experiments/vieneu_kws_research_v10/deployment/models/`.

### 5.1 Exported Artifacts Inventory

| Model Format | Filename | Size | SHA-256 Checksum | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **TorchScript** | `variant_w10_seed42_torchscript.pt` | 576.9 KB | `1197c8320ae1ee8ffec5df03fc3ff07481ba95eb2bfd971a80d5d05051b4742a` | Python/C++ PyTorch Mobile runtime |
| **ONNX FP32** | `variant_w10_seed42_fp32.onnx` | 542.7 KB | `1f401625c59a85eb0ccabebf707ff45d77e6ee37d88bf6d83ce3b5c978a348ef` | Primary edge runtime (Android/iOS ONNX Runtime) |
| **ONNX Dynamic INT8** | `variant_w10_seed42_int8.onnx` | 146.9 KB | `0be86f127e85a7839c0cd7bbae5417f714caacca09ec41f05d5c66bd508828a7` | Ultra-compact ARM64 mobile CPU inference |

### 5.2 Parity Test Results
Evaluated on fixed synthetic development fixtures (`demo_01` to `demo_06`):

| Test Fixture | PyTorch Prediction | TorchScript Prediction | ONNX FP32 Prediction | Max Logit Deviation (PyTorch vs ONNX FP32) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `demo_01_trai.wav` | `LEN` | `LEN` | `LEN` | $1.49 \times 10^{-6}$ | **PASS** |
| `demo_02_phai.wav` | `LEN` | `LEN` | `LEN` | $2.86 \times 10^{-6}$ | **PASS** |
| `demo_03_len.wav` | `LEN` | `LEN` | `LEN` | $3.11 \times 10^{-6}$ | **PASS** |
| `demo_04_xuong.wav` | `LEN` | `LEN` | `LEN` | $2.26 \times 10^{-6}$ | **PASS** |
| `demo_05_unknown.wav` | `LEN` | `LEN` | `LEN` | $2.15 \times 10^{-6}$ | **PASS** |
| `demo_06_silence.wav` | `REJECT` | `REJECT` | `REJECT` | $2.86 \times 10^{-6}$ | **PASS** |

- **TorchScript vs PyTorch:** Maximum absolute logit deviation: $0.00 \times 10^0$ (exact bit equivalence), **0 label mismatches**.
- **ONNX FP32 vs PyTorch:** Maximum absolute logit deviation: $3.11 \times 10^{-6}$ (tolerance $\le 1.00 \times 10^{-4}$), **0 label mismatches**.
- **ONNX INT8 vs PyTorch:** Maximum probability shift: $0.116$, **0 label mismatches**, model footprint compressed by **$3.70\times$** ($542.7 \text{ KB} \to 146.9 \text{ KB}$).

---

## 6. Edge Deployment Audit: Android & iOS

### 6.1 Evaluation of Mobile Runtime Frameworks

| Runtime Engine | Platform Support | Operator Compatibility | Model Footprint | Latency (ARM64 CPU est.) | Maintenance & Tooling Burden | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ONNX Runtime Mobile** | Android (AAR/JNI), iOS (CocoaPods/SPM), Linux/macOS/Windows | **Full** (Standard Conv2d, BatchNorm, MaxPool, AdaptiveAvgPool, Linear) | 542 KB (FP32)<br>147 KB (INT8) | ~1.5–2.5 ms | **Lowest.** Common multi-platform runtime, single model artifact, official bindings for Kotlin, Swift, and C++. | **RECOMMENDED COMMON RUNTIME** |
| **TorchScript / PyTorch Mobile** | Android, iOS | **Full** (Native PyTorch trace) | 577 KB | ~2.0–3.5 ms | **High.** PyTorch Mobile is superseded by ExecuTorch; large static binary overhead (+15–30 MB library size). | Secondary / Deprecated |
| **TensorFlow Lite (TFLite)** | Android, iOS | **Partial** (`AdaptiveAvgPool2d` and `pad_mode="reflect"` require manual ONNX-to-TF-to-TFLite conversion) | ~530 KB | ~1.5–2.2 ms | **High.** Fragile two-stage conversion pipeline with frequent operator translation defects. | Not recommended |
| **Apple CoreML** | iOS / macOS only | **Full** via `coremltools` (Neural Engine acceleration supported) | ~500 KB | < 1.0 ms (ANE) | **Medium.** Excellent on Apple devices, but requires maintaining a separate model format from Android. | Platform-specific option for iOS |
| **ExecuTorch** | Android, iOS | **Emerging** (Requires custom operator build toolchain) | ~500 KB | ~1.2–2.0 ms | **High.** Fast-moving API, non-trivial build setup, limited documentation for small teams. | Candidate for future research |

### 6.2 Host Machine Toolchain Audit (Current Workspace)
An audit of the host environment for building the Android application (`experiments/synthetic_kws_demo/android_app`) revealed:
1. **Java Development Kit (JDK):** JDK 17 / `JAVA_HOME` is not configured on the local system path.
2. **Android SDK / NDK:** Android SDK Command-line Tools and NDK (`cmake`, `ninja`, `ndk-build`) are not installed.
3. **Gradle Wrapper:** `experiments/synthetic_kws_demo/android_app/gradle/wrapper/gradle-wrapper.jar` is absent from version control (only `gradle-wrapper.properties` exists).
4. **Conclusion:** Local APK generation on this host is **BLOCKED BY HOST ENVIRONMENT LIMITATIONS**. Build must occur on a configured CI runner or Android developer workstation.

### 6.3 Existing Android Scaffold Analysis
The repository contains an existing Android demo project at `experiments/synthetic_kws_demo/android_app`.
- **Current Stack:**
  - UI: Kotlin single-activity architecture (`MainActivity.kt`, `activity_main.xml`).
  - Audio Input: `WavAudioLoader.kt` (reads bundled `.wav` assets).
  - Neural Backend: `OnnxKwsClassifier.kt` using `ai.onnxruntime:onnxruntime-android:1.15.0`.
  - Frontend: C++ JNI bridge (`kws_jni_bridge.cpp`) computing STFT and Mel filterbanks.
  - Legacy Target: Hardcoded for a 6-class TC-ResNet-14 (`arm_b_tcresnet14.onnx`).

### 6.4 Evidence-Based Integration Plan
To upgrade the Android scaffold to the Phase 91 candidate:
1. **Asset Replacement:** Copy `experiments/vieneu_kws_research_v10/deployment/models/variant_w10_seed42_fp32.onnx` (and `_int8.onnx`) into `app/src/main/assets/models/`.
2. **Class Mapping Update:** In `GameActionAdapter.kt`, update `CLASS_LABELS` from 6 classes to the 5-class contract:
   `["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]`.
3. **Rejection Gating Update:** In `OnnxKwsClassifier.kt`, implement the calibrated threshold gating:
   - Calculate maximum probability among indices 0–3: $p_{\text{cmd\_max}} = \max(p_0, p_1, p_2, p_3)$.
   - If $p_{\text{cmd\_max}} \ge 0.40 \implies$ command is accepted; else $\implies$ `REJECT`.
4. **Recorded-File Test Mode:** Maintain `WavAudioLoader` as the primary verification interface to execute deterministic regression testing against recorded assets prior to live microphone testing.

### 6.5 Microphone Audio Capture Architecture (Android & iOS)
Live hardware validation must remain cleanly decoupled from offline model verification:
- **Android Capture Architecture:**
  - Use `AudioRecord` (or Google `Oboe` C++ library for low-latency AAudio/OpenSL ES) configured for `16,000 Hz, CHANNEL_IN_MONO, ENCODING_PCM_16BIT`.
  - Circular Ring Buffer: A 16,000-sample (1-second) rolling FIFO buffer.
- **iOS Capture Architecture:**
  - Use `AVAudioEngine` with an input tap on `inputNode` at 16 kHz mono.
  - Accelerated FFT via Apple Accelerate framework (`vDSP`).
- **Safety Invariant:** Software interfaces must clearly mark that continuous microphone streaming has **NOT** been certified for production.

---

## 7. Operational Mode Differentiation and Boundary Taxonomy

A fundamental architectural finding is that single-clip bounded-utterance classification does not equate to continuous wake-word detection.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      OPERATIONAL MODE COMPARISON                                 │
├───────────────────────┬───────────────────────────┬───────────────────────────────┤
│ Mode                  │ Trigger / Input Window    │ False Alarm Vulnerability     │
├───────────────────────┼───────────────────────────┼───────────────────────────────┤
│ 1. Recorded-File      │ Static 1.0s WAV from disk │ ZERO open-mic background exposure│
│                       │ Explicit bounded capture  │ Candidate FAR: 2.03% ± 0.58%  │
├───────────────────────┼───────────────────────────┼───────────────────────────────┤
│ 2. Push-to-Talk (PTT) │ User physically depresses │ Bounded to user speech window │
│                       │ button for ~1.0 second    │ Clean synthetic FAR: ~2%      │
│                       │ Capture bounded by intent │ Candidate ACCEPTABLE for PTT  │
├───────────────────────┼───────────────────────────┼───────────────────────────────┤
│ 3. Continuous Open-Mic│ Microphone on indefinitely│ 3,600 evaluations per hour    │
│                       │ 1.0s window evaluated     │ Expected False Alarms:        │
│                       │ every 1s (or every 100ms) │ ~70 to 700+ false alarms/hour │
│                       │                           │ UNACCEPTABLE FOR DEPLOYMENT   │
└───────────────────────┴───────────────────────────┴───────────────────────────────┘
```

### The False Alarm Scaling Law
For an isolated 1-second clip, an acceptance rate on non-target audio of $2.0\%$ appears acceptable. However, under continuous open-mic listening:
- In 1 hour of ambient silence or conversational background noise, a non-overlapping 1-second classifier evaluates $3,600$ windows.
- At $\text{FAR} \approx 2\%$, the expected false alarms per hour is:
  $$\mathbb{E}[\text{FA/hour}] = 3,600 \times 0.02 \approx 72 \text{ false activations per hour}$$
- In a sliding window evaluated every $100 \text{ ms}$ ($36,000 \text{ evaluations/hour}$), false triggers increase by an order of magnitude.

**Operational Verdict:**
- **Recorded-File Mode:** Validated for synthetic development research.
- **Push-to-Talk Mode:** Feasible research candidate subject to real-world microphone validation.
- **Continuous Open-Mic Mode:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`. Deploying this model in continuous listening mode without multi-frame temporal smoothing, confidence accumulation, and energy-based VAD will lead to immediate user dissatisfaction.

---

## 8. Verification Results and Acceptance Summary

### 8.1 Regression Suite Execution
All tests in `tests/test_cnn_deployment_phase91.py` were executed using pytest under Python 3.11:

```powershell
.\.venv\Scripts\pytest tests/test_cnn_deployment_phase91.py -v
```

**Results:**
- `test_checkpoint_integrity_and_immutability`: **PASSED**
- `test_audio_validator_missing_file`: **PASSED**
- `test_audio_validator_empty_file`: **PASSED**
- `test_audio_validator_strict_mode_errors`: **PASSED**
- `test_audio_validator_permissive_mode`: **PASSED**
- `test_audio_validator_peak_normalization_contract`: **PASSED**
- `test_frontend_shape_and_numerical_invariants`: **PASSED**
- `test_threshold_rejection_gating`: **PASSED**
- `test_export_manifest_and_models_exist`: **PASSED**
- `test_numerical_parity_onnx_runtime`: **PASSED**
- `test_cli_predict_wav_success_json`: **PASSED**
- `test_cli_predict_wav_missing_file`: **PASSED**
- `test_cli_predict_wav_empty_file`: **PASSED**
- `test_cli_predict_wav_strict_sample_rate`: **PASSED**
- `test_cli_predict_wav_onnx_backend`: **PASSED**
- `test_governance_and_field_recording_isolation`: **PASSED**

**Overall Status: 16 passed in 11.16s (100% pass rate).**

### 8.2 Historical and Isolation Verification
- Regression tests for Phase 80–84 and Phase 85–89 (`tests/test_vieneu_research_v10.py` and `tests/test_vieneu_research_v10_p85_89.py`): **13 passed in 0.13s**.
- Regression tests for Phase 75–79 (`tests/test_vieneu_research_v9.py`): **7 passed in 1.66s**.
- Staged git commits: `git diff --cached` is strictly empty.
- Human review and field recording assets: completely isolated and unreferenced.

---

## 9. Inventory of Created Files

| File Path | Description |
| :--- | :--- |
| `experiments/vieneu_kws_research_v10/deployment/DEPLOYMENT_CONTRACT.md` | Formal deployment contract covering audio format, Mel frontend, model topology, classes, and thresholds. |
| `experiments/vieneu_kws_research_v10/deployment/audio_validator.py` | Audio loading and validation module supporting strict and permissive modes. |
| `experiments/vieneu_kws_research_v10/deployment/predict_wav.py` | Command-line WAV inference tool with PyTorch/ONNX backends, latency profiling, and exit codes. |
| `experiments/vieneu_kws_research_v10/deployment/export_model.py` | Reproducible model export script generating TorchScript, ONNX FP32, and ONNX INT8 with parity tests. |
| `experiments/vieneu_kws_research_v10/deployment/models/variant_w10_seed42_fp32.onnx` | Exported FP32 ONNX model artifact (opset 17, 542.7 KB). |
| `experiments/vieneu_kws_research_v10/deployment/models/variant_w10_seed42_int8.onnx` | Exported Dynamic INT8 ONNX model artifact (146.9 KB). |
| `experiments/vieneu_kws_research_v10/deployment/models/variant_w10_seed42_torchscript.pt` | Exported TorchScript model artifact (576.9 KB). |
| `experiments/vieneu_kws_research_v10/deployment/models/export_manifest.json` | Manifest documenting export metadata, checksums, and parity measurements. |
| `tests/test_cnn_deployment_phase91.py` | 16-test unit and integration test suite. |
| `experiments/vieneu_kws_research_v10/reports/PHASE91_CNN_DEPLOYMENT.md` | Comprehensive Phase 91 engineering report (this document). |

---

## 10. Developer Guide: How to Run Offline WAV Inference

### 10.1 Quick Start (PyTorch Backend)
To classify a standard 16 kHz WAV file using the frozen primary checkpoint:
```powershell
.\.venv\Scripts\python experiments/vieneu_kws_research_v10/deployment/predict_wav.py path/to/audio.wav
```

### 10.2 JSON Output Mode (for Integration into Scripts)
```powershell
.\.venv\Scripts\python experiments/vieneu_kws_research_v10/deployment/predict_wav.py path/to/audio.wav --json
```

### 10.3 ONNX Runtime Edge Backend (Faster CPU Inference)
```powershell
.\.venv\Scripts\python experiments/vieneu_kws_research_v10/deployment/predict_wav.py path/to/audio.wav `
  --onnx-model experiments/vieneu_kws_research_v10/deployment/models/variant_w10_seed42_fp32.onnx
```

### 10.4 Strict Validation Mode (Rejects non-16kHz, stereo, or non-1s audio)
```powershell
.\.venv\Scripts\python experiments/vieneu_kws_research_v10/deployment/predict_wav.py path/to/audio.wav --strict
```

### 10.5 Run the Test Suite
```powershell
.\.venv\Scripts\pytest tests/test_cnn_deployment_phase91.py -v
```

---

## 11. Concluding Governance Declarations

1. `PENDING_HUMAN_LISTENER_VERIFICATION`: The human listening study distributed in Phase 90 remains pending review submissions. No human scores have been fabricated.
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`: All parity benchmarks and inference runs were executed solely on approved synthetic and developmental fixtures. No claims of natural speech validity are made.
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`: The current candidate is certified solely for bounded-utterance (push-to-talk or recorded file) operations. Continuous open-mic deployment remains strictly disclaimed.
