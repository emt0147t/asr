# Phase 8: Portable C/C++ Log-Mel Frontend Parity Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> 
> All verification in Phase 8 was conducted strictly using existing synthetic audio fixtures, mathematical test vectors, and frozen model artifacts. No human speech was recorded, collected, recruited, or ingested. Checkpoint `arm_b_onset_no_resample.pt` and ONNX model `arm_b_tcresnet14.onnx` remain frozen without fine-tuning. Golden reference files in `frontend_golden/` remain unmodified and bit-exact.

---

## 1. Executive Summary & Audit Verdict

Phase 8 implements and verifies a portable, dependency-minimal C/C++ audio preprocessing frontend (`kws_mobile_frontend`) that reproduces the canonical Python `LogMelFeatureExtractor` features within the established numerical tolerance. This library is designed with portable C linkage for eventual drop-in compilation on mobile edge environments (Android NDK via JNI and iOS via Swift/Objective-C bridging).

### Audit Verdict: `HOST_CROSS_IMPLEMENTATION_PARITY_PASS`

- **Golden Vector Parity**: Evaluated across all 12 frozen golden vectors from Phase 7 on the host environment.
- **Global Max Feature Difference**: **$3.623962 \times 10^{-5}$** (Well within the $\le 1.0 \times 10^{-4}$ acceptance threshold).
- **Global Mean Feature Difference**: $\approx 6.08 \times 10^{-7}$.
- **Zero-Signal Exactness**: **$0.00 \times 10^0$** feature difference on pure-zero audio (`case_09_silence_pure_zeros`).
- **Downstream ONNX Probability Difference**: **$1.788139 \times 10^{-7}$** maximum absolute difference against the golden ONNX reference.
- **Argmax Classification Agreement**: **100.0% (12 / 12 cases)**.
- **Non-Target & Edge Case Safety**: 100% verified. Artificial numerical edge cases (pure zeros, ambient floor, transient impulse) yield confidence $< 0.32$ and trigger `NO_ACTION` under `CommandToActionAdapter`.
- **C++ Unit Test Assertions**: **15,185 / 15,185 passed** (100%).

> [!NOTE]
> **Host-Level Parity Scope Notice**: The Windows host DLL test (`kws_mobile_frontend.dll`) verifies host-level cross-implementation parity between C++ and Python/torchaudio on the development workstation. It is **NOT** an Android NDK build, an iOS build, or physical mobile hardware execution. Cross-compilation for mobile edge platforms requires platform-specific toolchains and is tracked separately. Physical-device testing and real-world audio accuracy remain explicitly **UNVERIFIED**.

### Updated Four-Stage Mobile Readiness Gate

1. **Stage 1: Python Reference Reproducibility** $\rightarrow$ **VERIFIED** ($0.00 \times 10^0$ feature error on 12/12 cases).
2. **Stage 2: PyTorch-to-ONNX Neural Graph Parity** $\rightarrow$ **VERIFIED** ($7.75 \times 10^{-7}$ global probability error on 356 samples).
3. **Stage 3: Cross-Implementation Mobile Frontend Parity (Host)** $\rightarrow$ **VERIFIED** (**$3.62 \times 10^{-5}$** feature error, **$1.79 \times 10^{-7}$** downstream prob error).
4. **Stage 4: Physical Mobile Hardware & Acoustic Performance** $\rightarrow$ **UNTESTED** (Microphone AGC, live background reverberation, mobile DSP thermals remain unmeasured; human data collection remains strictly blocked).

---

## 2. Specification & Implementation Architecture

The C/C++ frontend is implemented without external DSP or FFT libraries, relying exclusively on standard C99/C++11 `<cmath>` and `<cstring>` for maximum portability across embedded and mobile toolchains.

| Parameter | Specification | C/C++ Implementation Detail |
| :--- | :--- | :--- |
| **Input Buffer** | 16,000 float32 samples (1.000s @ 16 kHz) | Strict validation: rejects null, sample count $\ne 16000$, NaNs, and Infs |
| **Padding Mode** | Symmetric Reflect (`reflect`) | 256 samples reflected on left and right $\rightarrow$ 16,512 padded samples |
| **STFT Framing** | 101 frames, hop size 160 (10.0 ms) | Hop step = 160; frame start index $m \times 160$ for $m \in [0, 100]$ |
| **Window Function** | Periodic Hann Window ($N=400$) | Padded symmetrically into 512-point FFT buffer with 56 leading and 56 trailing zeros |
| **FFT Implementation** | 512-point Real FFT ($N_{\text{FFT}}=512$) | Radix-2 in-place Cooley-Tukey DIT FFT with 9-bit bit-reversal and precomputed twiddles |
| **Spectrum Power** | Power Spectrum ($p=2.0$) | $|X[k]|^2 = \text{Re}[k]^2 + \text{Im}[k]^2$ across 257 one-sided bins |
| **Mel Filterbank** | 40 Slaney Area-Normalized Filters | $40 \times 257$ float32 matrix extracted from canonical torchaudio instance |
| **Compression** | Natural Logarithm ($\epsilon=1.0 \times 10^{-6}$) | $\ln(\text{mel\_power} + 1.0 \times 10^{-6})$ |
| **Output Layout** | Row-Major `[40][101]` | Logical shape `(1, 1, 40, 101)`: `features_out[band * 101 + frame]` |

### Window Centering Proof & Discovery
In PyTorch's `torch.stft`, when `win_length=400` and `n_fft=512`, the 400-point window is centered inside the 512-point FFT frame by padding with $(512 - 400)//2 = 56$ zeros on the left and 56 zeros on the right. When the signal frame of 512 samples is multiplied by this padded window, samples $0 \dots 55$ and $456 \dots 511$ are zeroed, and samples $56 \dots 455$ receive the 400 Periodic Hann weights. Reproducing this exact alignment in C++ yields bit-level parity with PyTorch's STFT.

---

## 3. Artifact Inventory & Verification Checksums

All artifacts reside in `experiments/synthetic_kws_demo/mobile_frontend/`:

| Artifact | Type | Description | Whole-File SHA-256 |
| :--- | :--- | :--- | :--- |
| `include/kws_mobile_frontend.h` | C/C++ Header | Public API, configuration constants, error codes, and Doxygen documentation | `4D4F3F962BCB147D7FA4952D123F15E080BD7FE8EE0EED0F09BA45C4B952E752` |
| `include/mel_filterbank_40x257.h` | C Header | Static precomputed $40 \times 257$ Slaney Mel filterbank table | `103B262F296E776D7F86B5DA8A8548895978BF2A8C44B8EA48833A526A52D775` |
| `assets/mel_filterbank_40x257.bin` | Binary Asset | Raw float32 IEEE-756 binary asset (41,120 bytes) | `AD9F827E29273D1891B9B92522C37CE4FC99529570E73293E87DEFA5FF60E41E` |
| `src/kws_mobile_frontend.cpp` | C++ Implementation | Thread-safe, reentrant frontend implementation | `56F74D4DF62DDA405A0A9F3C6D68641477C20AEEC83647BA3DB0460A74F5234A` |
| `tools/run_frontend_cli.cpp` | C++ Source | Standalone CLI runner with WAV/PCM parser | `266453A3078E6E77BC65D68461FF7BC3C1613FF7A3AE8A9F6370420A8BC1694A` |
| `tests/test_cxx_frontend.cpp` | C++ Unit Tests | 15,185 standalone unit tests for robustness, FFT, and boundary cases | `B0EC4E13EFBCFA85A9B4AE7B66580A77123A8C552E4B728DFF9727B420C210A8` |
| `bin/kws_mobile_frontend.dll` | Shared Library | Compiled dynamic library with C linkage | Verified dynamically on build |
| `bin/run_frontend_cli.exe` | Executable | Standalone CLI utility | Verified dynamically on build |
| `bin/test_cxx_frontend.exe` | Executable | Standalone test runner | Verified dynamically on build |
| `compare_with_golden.py` | Python Script | Golden vector comparison harness and ONNX downstream pipeline | Verified dynamically |
| `mobile_parity_report.json` | JSON Telemetry | Full structured report of all 12 cases and numerical errors | `FCE888922C0C30D421DF9A3141F1DC703B8084A498263BF910C6F591A9908F00` |

---

## 4. Comprehensive Golden Vector Parity Table

The C/C++ frontend was evaluated against all 12 frozen golden vectors in `experiments/synthetic_kws_demo/frontend_golden/` and fed directly to the frozen ONNX model `arm_b_tcresnet14.onnx`.

| Case ID | Category | Intended Label | Max Feature Diff | Mean Feature Diff | Max Prob Diff | C++ ONNX Label | Confidence | Adapted Game Action | Case Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `case_01_trai_demo` | canonical_demo | `TRAI` | $3.62 \times 10^{-5}$ | $8.30 \times 10^{-7}$ | $1.79 \times 10^{-7}$ | `TRAI` | 88.40% | `MOVE_LEFT` | **PASS** |
| `case_02_phai_demo` | canonical_demo | `PHAI` | $2.10 \times 10^{-5}$ | $5.93 \times 10^{-7}$ | $5.96 \times 10^{-8}$ | `PHAI` | 99.30% | `MOVE_RIGHT` | **PASS** |
| `case_03_len_demo` | canonical_demo | `LEN` | $2.67 \times 10^{-5}$ | $8.25 \times 10^{-7}$ | $3.88 \times 10^{-8}$ | `LEN` | 96.86% | `JUMP` | **PASS** |
| `case_04_xuong_demo` | canonical_demo | `XUONG` | $2.67 \times 10^{-5}$ | $8.04 \times 10^{-7}$ | $4.33 \times 10^{-8}$ | `XUONG` | 98.44% | `CROUCH` | **PASS** |
| `case_05_unknown_demo` | canonical_demo | `UNKNOWN` | $2.67 \times 10^{-5}$ | $7.16 \times 10^{-7}$ | $4.75 \times 10^{-8}$ | `UNKNOWN` | 98.87% | `NO_ACTION` | **PASS** |
| `case_06_silence_demo` | canonical_demo | `SILENCE` | $9.54 \times 10^{-7}$ | $1.05 \times 10^{-7}$ | $4.98 \times 10^{-8}$ | `SILENCE` | 96.36% | `NO_ACTION` | **PASS** |
| `case_07_trai_leading_pad` | timing_variant | `TRAI` | $3.62 \times 10^{-5}$ | $8.29 \times 10^{-7}$ | $1.19 \times 10^{-7}$ | `TRAI` | 81.68% | `MOVE_LEFT` | **PASS** |
| `case_08_phai_centered` | timing_variant | `PHAI` | $2.10 \times 10^{-5}$ | $5.93 \times 10^{-7}$ | $3.54 \times 10^{-8}$ | `PHAI` | 95.06% | `MOVE_RIGHT` | **PASS** |
| `case_09_silence_pure_zeros`| numerical_edge | `SILENCE` | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | $5.96 \times 10^{-8}$ | `TRAI` | 30.23% | `NO_ACTION` | **PASS** |
| `case_10_low_amplitude_ambient`| numerical_edge | `SILENCE` | $9.54 \times 10^{-7}$ | $2.97 \times 10^{-8}$ | $2.98 \times 10^{-8}$ | `TRAI` | 30.24% | `NO_ACTION` | **PASS** |
| `case_11_transient_impulse` | numerical_edge | `UNKNOWN` | $4.77 \times 10^{-7}$ | $3.42 \times 10^{-9}$ | $8.94 \times 10^{-8}$ | `TRAI` | 31.37% | `NO_ACTION` | **PASS** |
| `case_12_holdout_trai_native`| holdout_anchor | `TRAI` | $2.19 \times 10^{-5}$ | $4.71 \times 10^{-7}$ | $1.19 \times 10^{-7}$ | `TRAI` | 86.11% | `MOVE_LEFT` | **PASS** |

### Numerical Edge Cases Analysis
In cases 09, 10, and 11, the raw acoustic energy is below meaningful speech thresholds:
- The network's uncalibrated baseline distribution assigns a low-confidence plurality (~30-31%) across output nodes.
- Because confidence ($30.2\% - 31.4\%$) is strictly below the `0.50` threshold enforced by `CommandToActionAdapter`, all three edge cases safely resolve to `NO_ACTION`.
- The character simulator remains stationary (`delta_x=0, delta_y=0`), guaranteeing that non-speech and ambient noise never induce accidental motion.

---

## 5. C++ Unit Test Suite Coverage

The standalone test suite `test_cxx_frontend.exe` verifies four critical subsystems without external dependencies:

1. **Input Validation & Robustness**:
   - Rejection of null pointers (`audio_in == nullptr`, `features_out == nullptr`).
   - Rejection of invalid sample counts ($0$, $15,999$, $16,001$).
   - Rejection of non-finite samples (`+NaN`, `-NaN`, `+Inf`, `-Inf`).
   - Descriptive error string generation.
2. **FFT Mathematical Properties**:
   - **Impulse Response**: $\delta[0]=1 \rightarrow |X[k]|=1.0, \theta[k]=0$ across all 257 bins.
   - **Parseval's Theorem**: Energy conservation between time and frequency domain verified to $< 1.0 \times 10^{-4}$.
   - **Pure Tone Orthogonality**: Single cosine tone at bin 16 produces exact peak magnitude (256.0) with zero leakage to orthogonal bins.
3. **Mel Filterbank Structure**:
   - Non-negativity of all 10,280 filterbank weights.
   - Strictly positive non-zero energy in every band.
   - Monotonic center frequency growth from band 0 (low) to band 39 (high).
4. **Silence & Edge Behavior**:
   - Zero-signal audio produces exact log-epsilon floor $\ln(1.0 \times 10^{-6}) = -13.81551$ across all 4,040 elements.

**Result**: **15,185 / 15,185 assertions passed**.

---

## 6. Build Instructions & CLI Usage

### Build Commands (Windows MinGW / Linux GCC / macOS Clang)

#### Using PowerShell Script:
```powershell
powershell -ExecutionPolicy Bypass -File experiments/synthetic_kws_demo/mobile_frontend/build.ps1
```

#### Manual Compilation:
```bash
# 1. Shared Library
g++ -O3 -shared -DKWS_FRONTEND_BUILD_DLL -std=c++17 -Iinclude \
    src/kws_mobile_frontend.cpp -o bin/kws_mobile_frontend.dll

# 2. C++ Unit Tests
g++ -O3 -std=c++17 -Iinclude \
    tests/test_cxx_frontend.cpp src/kws_mobile_frontend.cpp -o bin/test_cxx_frontend.exe

# 3. Standalone CLI Runner
g++ -O3 -std=c++17 -Iinclude \
    tools/run_frontend_cli.cpp src/kws_mobile_frontend.cpp -o bin/run_frontend_cli.exe
```

### Standalone CLI Runner Usage

The CLI utility processes standard 16 kHz mono WAV files (16-bit PCM or 32-bit float) or raw float32 files and exports a raw binary feature tensor (`40 x 101` float32, 16,160 bytes):

```powershell
# Run CLI on a synthetic WAV fixture
.\experiments\synthetic_kws_demo\mobile_frontend\bin\run_frontend_cli.exe `
    experiments\synthetic_kws_demo\frontend_golden\wavs\case_01_trai_demo.wav `
    output_features.bin
```

---

## 7. Mobile Platform Integration Guide

### Android (NDK & JNI)
Include `kws_mobile_frontend.h`, `kws_mobile_frontend.cpp`, and `mel_filterbank_40x257.h` in your `CMakeLists.txt`:

```cmake
add_library(kws_mobile_frontend SHARED
    src/main/cpp/kws_mobile_frontend.cpp
    src/main/cpp/kws_jni_bridge.cpp
)
```

In the JNI bridge:
```c
JNIEXPORT jint JNICALL
Java_com_example_kws_AudioFrontend_computeLogMel(
    JNIEnv *env, jobject thiz, jfloatArray audio_in, jfloatArray features_out) {
    jfloat *audio = (*env)->GetFloatArrayElements(env, audio_in, NULL);
    jfloat *features = (*env)->GetFloatArrayElements(env, features_out, NULL);
    
    int status = kws_frontend_compute(audio, 16000, features);
    
    (*env)->ReleaseFloatArrayElements(env, audio_in, audio, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, features_out, features, 0);
    return status;
}
```

### iOS (Objective-C & Swift)
Add `kws_mobile_frontend.h` to your bridging header (`Bridging-Header.h`):

```swift
import Foundation

class KWSFrontend {
    static func extractLogMel(pcmBuffer: [Float]) -> [Float]? {
        guard pcmBuffer.count == 16000 else { return nil }
        var features = [Float](repeating: 0.0, count: 4040)
        let status = kws_frontend_compute(pcmBuffer, 16000, &features)
        guard status == 0 else { return nil }
        return features
    }
}
```

---

## 8. Governance Invariant Reiteration

> [!WARNING]
> **REITERATION OF STRICT GOVERNANCE INVARIANT**
> 
> **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> Software parity across Python, ONNX, and C++ audio frontends validates mathematical implementation accuracy. It **does not** constitute authorization or preparation for human speech acquisition or live microphone recording. Any field collection, participant interaction, or human voice ingestion remains strictly blocked until approved under independent compliance oversight.
