# Phase 9: Android Offline KWS Integration and Build Blocker Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> 
> All Android application development, assets, and verification in Phase 9 were conducted strictly offline using existing synthetic audio fixtures and frozen model artifacts. Checkpoint `arm_b_onset_no_resample.pt` and ONNX model `arm_b_tcresnet14.onnx` remain frozen without retraining or quantization. No microphone access, live recording, participant interaction, or external dataset ingestion is implemented. Zero permissions are requested in `AndroidManifest.xml` (`RECORD_AUDIO` and `INTERNET` are strictly absent).

---

## 1. Executive Summary & Gate-by-Gate Status

Phase 9 integrates the Vietnamese Keyword Spotting (KWS) inference pipeline into an isolated Android application project under `experiments/synthetic_kws_demo/android_app/`. The project reuses the canonical C++ Log-Mel frontend (`kws_mobile_frontend`) through the Android NDK/JNI boundary and prepares offline neural inference on pre-packaged synthetic WAV fixtures via Microsoft ONNX Runtime Mobile.

Per the Phase 9 governance mandate, host-level cross-implementation parity, Android build success, Android emulator/device execution, and physical acoustic measurements are strictly distinguished across five distinct gates.

### Executive Verdict: `BLOCKED`

The overall Phase 9 verdict is **`BLOCKED`** for the Android deployment objective because the host workstation lacks the required Java Development Kit (JDK), Gradle build system, Android SDK, and Android NDK toolchains. While host-side static and numerical pipeline validation succeeded, **a project scaffold, static validator, or Windows DLL run does NOT constitute an Android deployment.**

### Android Build and Execution Gates Summary

| Gate | Scope | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Gate A** | **Host-side / static validation** | **`PASS`** | 12/12 golden vectors match within $3.62 \times 10^{-5}$ feature error and $1.79 \times 10^{-7}$ probability error; edge cases safely suppressed to `NO_ACTION`. |
| **Gate B** | **Android NDK compilation** | **`BLOCKED`** | Host lacks Android NDK (`ANDROID_NDK_HOME` not set). No `.so` shared libraries have been compiled for Android target ABIs (`arm64-v8a`, `x86_64`). |
| **Gate C** | **APK build** | **`BLOCKED`** | Host lacks Java Development Kit (`javac` not found), Gradle (`gradle` not found), and Android SDK (`ANDROID_HOME` not set). No APK has been assembled. |
| **Gate D** | **Android runtime inference** | **`BLOCKED`** | Requires built APK and available Android emulator or attached physical device via `adb` (`adb` not found). No runtime execution has occurred. |
| **Gate E** | **Physical-device performance** | **`NOT_TESTED`** | Requires attached physical Android hardware. No latency, memory, or thermal benchmarks have been measured on hardware. |

---

## 2. Actual Test Results and Evidence

All available host-side test suites were executed and verified directly on the repository:

| Test Suite / Script | Command | Exit Code | Result | Details |
| :--- | :--- | :---: | :---: | :--- |
| **Android Pipeline Validator** | `python validate_android_pipeline.py` | `0` | **PASS** | 12/12 golden cases passed; 4/4 audio rejection tests passed; asset hashes verified. |
| **Campaign Acceptance** | `pytest tests/test_campaign_acceptance.py -q` | `0` | **PASS** | **69 passed, 0 failed** in 32.93s (including TC-68 and TC-69). |
| **KWS Export Parity** | `pytest tests/test_kws_export_parity.py -q` | `0` | **PASS** | **13 passed, 0 failed** in 2.76s. |
| **Synthetic KWS Demo** | `pytest tests/test_synthetic_kws_demo.py -q` | `0` | **PASS** | **19 passed, 0 failed** in 2.59s. |
| **Git Diff Format Check** | `git diff --check` | `0` | **PASS** | Zero whitespace or formatting errors detected. |
| **Git Staged State** | `git diff --cached` | `0` | **PASS** | **0 files staged** (clean index). |

---

## 3. Host Toolchain Inspection & Blocker Analysis

An automated inspection of the development workstation environment was conducted on 2026-10-16:

```
=== Toolchain Inspection Results ===
  java             : NOT FOUND
  javac            : NOT FOUND
  gradle           : NOT FOUND
  cmake            : C:\Strawberry\c\bin\cmake.EXE (version 3.29.2)
  adb              : NOT FOUND
  emulator         : NOT FOUND
  JAVA_HOME        : NOT SET
  ANDROID_HOME     : NOT SET
  ANDROID_SDK_ROOT : NOT SET
  ANDROID_NDK_HOME : NOT SET
  GRADLE_HOME      : NOT SET
  Local Android SDK: False (C:\Users\DELL\AppData\Local\Android\Sdk does not exist)
  Gradle Wrapper   : False (No gradlew script; wrapper would still require Java)
```

### Exact Missing Toolchain Components
1. **Java Development Kit (JDK 17+)**: `java` and `javac` binaries are absent; `JAVA_HOME` environment variable is not configured.
2. **Gradle Build Tool**: `gradle` binary is absent; `GRADLE_HOME` is not set. No Gradle wrapper is bundled, and any Gradle wrapper would require a functioning Java runtime.
3. **Android SDK & Platform Tools**: Neither `ANDROID_HOME` nor `ANDROID_SDK_ROOT` is set. `%LOCALAPPDATA%\Android\Sdk` does not exist on disk.
4. **Android NDK (Side-by-side NDK r26+)**: `ANDROID_NDK_HOME` is not set; no cross-compiler toolchains for `aarch64-linux-android` or `x86_64-linux-android` exist.
5. **Android Debug Bridge (`adb`) & Emulator**: `adb` is absent from PATH; no Android Virtual Devices (AVDs) or hardware devices are connected.

In strict compliance with governance instructions ("Do not install new SDKs or modify environment configuration without explicit approval"), no external toolchain packages were installed.

---

## 4. Static Consistency Checks & Source Verification

Without claiming Android compilation or execution, the following static checks were performed and verified:

1. **Package & JNI Symbol Consistency**:
   - Application namespace: `com.example.synthetic_kws` in `app/build.gradle` and `AndroidManifest.xml`.
   - Kotlin JNI class: `com.example.synthetic_kws.KwsFrontendJni`.
   - C++ JNI bridge symbols in `kws_jni_bridge.cpp`:
     - `Java_com_example_synthetic_1kws_KwsFrontendJni_init`
     - `Java_com_example_synthetic_1kws_KwsFrontendJni_computeLogMel`
     - `Java_com_example_synthetic_1kws_KwsFrontendJni_getErrorString`
   - Native library name: `kws_mobile_frontend_jni` loaded via `System.loadLibrary("kws_mobile_frontend_jni")` and built by `CMakeLists.txt`.
2. **Canonical Source Code Sharing**:
   - `CMakeLists.txt` references the exact canonical C++ frontend at `${CMAKE_CURRENT_SOURCE_DIR}/../../../../mobile_frontend/src/kws_mobile_frontend.cpp` and headers in `${CMAKE_CURRENT_SOURCE_DIR}/../../../../mobile_frontend/include`. No second, mathematically distinct frontend was authored.
3. **Target ABIs & Dependencies**:
   - `app/build.gradle` specifies target ABIs `'arm64-v8a'` and `'x86_64'`.
   - ONNX Runtime Android dependency explicitly declared: `com.microsoft.onnxruntime:onnxruntime-android:1.17.0`.
   - Proguard rules (`proguard-rules.pro`) preserve JNI entry points and `ai.onnxruntime.**` symbols.
4. **Packaged ONNX Model Asset Integrity**:
   - Source: `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx`
   - Destination: `app/src/main/assets/models/arm_b_tcresnet14.onnx`
   - Whole-file SHA-256: `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` (Bit-exact match to frozen source model).
5. **Packaged Audio Fixture Integrity**:
   - All 11 WAV fixtures in `app/src/main/assets/audio/` match their source SHA-256 hashes recorded in `asset_manifest.json` and `frontend_golden/golden_manifest.json`.
6. **Zero-Permission Governance Enforcement**:
   - `AndroidManifest.xml` contains zero `<uses-permission>` tags.
   - Microphone recording permissions (`android.permission.RECORD_AUDIO`) and network permissions (`android.permission.INTERNET`) are strictly absent.
7. **Strict Audio Validation & Rejection Contract**:
   - `WavAudioLoader.kt` strictly requires 16,000 Hz sample rate, mono (1 channel), 16,000 samples (1.000s), and finite float values.
   - Static test harness verified safe rejection of:
     - Multichannel audio (`invalid_stereo.wav` $\rightarrow$ Rejected)
     - Non-16kHz sample rates (`invalid_sample_rate_8k.wav` $\rightarrow$ Rejected)
     - Non-1.0s durations (`invalid_short_duration.wav` $\rightarrow$ Rejected)
     - Empty/truncated files (`invalid_empty.wav` $\rightarrow$ Rejected)
8. **Action-Suppression Path for Numerical Edge Cases**:
   - Full pipeline (`WAV -> C++ -> ONNX -> GameActionAdapter`) verified:
     - `case_09_silence_pure_zeros`: Plurality confidence 30.23% ($< 50.0\%$) $\rightarrow$ Suppressed to `NO_ACTION`.
     - `case_10_low_amplitude_ambient`: Plurality confidence 30.24% ($< 50.0\%$) $\rightarrow$ Suppressed to `NO_ACTION`.
     - `case_11_transient_impulse`: Plurality confidence 31.37% ($< 50.0\%$) $\rightarrow$ Suppressed to `NO_ACTION`.
     - `case_05_unknown_demo`: Non-target class `UNKNOWN` (98.87%) $\rightarrow$ Unconditionally suppressed to `NO_ACTION`.
     - `case_06_silence_demo`: Non-target class `SILENCE` (96.36%) $\rightarrow$ Unconditionally suppressed to `NO_ACTION`.
     - Target commands (`case_01` to `case_04`, `case_07`, `case_08`, `case_12`): Confidences 81.68% - 99.30% $\rightarrow$ Dispatched to `MOVE_LEFT`, `MOVE_RIGHT`, `JUMP`, `CROUCH`.

---

## 5. End-to-End Pipeline Telemetry (12 Golden Vectors)

Evaluated under host static diagnostic mode using canonical assets and DLL:

| Case ID | Category | Intended | Max Feat Diff | Mean Feat Diff | Max Prob Diff | Pred Label | Conf | Adapted Action | Status |
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

---

## 6. Reproducible Build Plan to Resume When Toolchain is Available

When an authorized environment provisioned with JDK 17, Android SDK, and Android NDK is available, the build and validation process can be resumed as follows:

```bash
# 1. Export toolchain paths
export JAVA_HOME="/path/to/jdk-17"
export ANDROID_HOME="/path/to/android-sdk"
export ANDROID_NDK_HOME="${ANDROID_HOME}/ndk/26.1.10909125"
export PATH="${JAVA_HOME}/bin:${ANDROID_HOME}/platform-tools:${PATH}"

# 2. Navigate to Android project directory
cd experiments/synthetic_kws_demo/android_app

# 3. Generate Gradle wrapper (if not present)
gradle wrapper --gradle-version 8.2

# 4. Compile native NDK shared library and assemble APK
./gradlew assembleDebug

# 5. Verify built APK contents
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep "libkws_mobile_frontend_jni.so"
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep "arm_b_tcresnet14.onnx"

# 6. Install on emulator or attached device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 7. Launch application
adb shell am start -n com.example.synthetic_kws/.MainActivity

# 8. Inspect runtime logs for JNI and ONNX execution
adb logcat -s KwsFrontendJni:V MainActivity:D
```

---

## 7. Explicit Governance & Deployment Disclaimers

> [!WARNING]
> **MANDATORY GOVERNANCE AND READINESS DISCLAIMERS**
> 1. **No Android Deployment Has Occurred**: No Android APK build has succeeded, no `.so` library has been compiled by the Android NDK, no emulator execution has taken place, and no physical device measurement has been conducted on this host.
> 2. **No Human Speech Collected or Processed**: **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** No microphone hardware was accessed, no human voice recordings were made or processed, and no field participant recruitment was initiated.
> 3. **iOS Platform Boundary**: Android integration progress provides zero verification for iOS. iOS deployment remains a completely independent future platform gate.
