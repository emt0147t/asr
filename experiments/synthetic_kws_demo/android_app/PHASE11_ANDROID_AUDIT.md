# Phase 11: Android Toolchain Validation and Build Verification Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> 
> All Android application development, assets, and verification in Phase 11 were conducted strictly offline using existing synthetic audio fixtures and frozen model artifacts. Checkpoint `arm_b_onset_no_resample.pt` and ONNX model `arm_b_tcresnet14.onnx` remain frozen without retraining or quantization. No microphone access, live recording, participant interaction, or external dataset ingestion is implemented. Zero permissions are requested in `AndroidManifest.xml` (`RECORD_AUDIO` and `INTERNET` are strictly absent).

---

## 1. Executive Summary & Gate-by-Gate Status

Phase 11 resumes from Phase 10 to inspect the actual installed Android toolchain, validate the project's Gradle Wrapper, attempt real Android builds, and re-run all regression and governance checks.

Per governance requirements, host-side parity, Android NDK compilation, Android APK assembly, Android emulator/device execution, and physical-device measurements are strictly separated into five distinct verification gates.

### Executive Verdict: `BLOCKED`

The overall Phase 11 verdict remains **`BLOCKED`** for the Android deployment objective.
1. The development host has no installed Java Development Kit (JDK/javac), Android SDK, Android NDK, or adb tools.
2. The project's Gradle wrapper is incomplete because `gradle/wrapper/gradle-wrapper.jar` is missing from the repository, and cannot be generated locally without a functional Java/Gradle installation.
3. No fabricated wrapper jar or improvised scripts were added, respecting the explicit directive: *"Do not fabricate a wrapper JAR, manually improvise wrapper scripts, or choose arbitrary Gradle and plugin versions."*
4. All host-side regression suites (113 automated tests across 4 suites) pass with 100% success, confirming zero host regressions.

### Android Build and Execution Gates Summary

| Gate | Scope | Status | Evidence & Actual Operational State |
| :--- | :--- | :---: | :--- |
| **Gate A** | **Host-side / static validation** | **`PASS`** | 12/12 golden vectors match within $3.62 \times 10^{-5}$ feature error and $1.79 \times 10^{-7}$ probability error; edge cases safely suppressed to `NO_ACTION`. 69/69 campaign acceptance tests pass. |
| **Gate B** | **Android NDK compilation** | **`BLOCKED`** | Host lacks Android NDK (`ANDROID_NDK_HOME` not set). Command `cmake -DCMAKE_TOOLCHAIN_FILE=android.toolchain.cmake ...` failed with exit code 1 (`Could not find toolchain file: android`). |
| **Gate C** | **APK assembly** | **`BLOCKED`** | Host lacks Java runtime (`java`/`javac` not found) and `gradle/wrapper/gradle-wrapper.jar` is absent. Command `gradlew.bat assembleDebug` failed with exit code 1 (`JAVA_HOME is not set`). |
| **Gate D** | **Android runtime inference** | **`BLOCKED`** | Requires built APK and available Android emulator or attached physical device via `adb`. Command `adb devices` failed with exit code 1 (`'adb' is not recognized`). |
| **Gate E** | **Physical-device performance** | **`NOT_TESTED`** | Requires attached physical Android hardware. No latency, memory, or thermal benchmarks have been measured on hardware. |

---

## 2. Toolchain Inspection & Compatibility Matrix

An automated and manual inspection of system paths and environment variables was performed on 2026-10-16:

### Toolchain Inventory

| Toolchain Component | Expected by Project | Actual Status / Path Found | Version Found | Operational Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Java / JDK** | JDK 17 (`JavaVersion.VERSION_17`) | `None` (searched `PATH`, `C:\Program Files`, `C:\Program Files (x86)`) | N/A | **ABSENT** |
| **Java Compiler (javac)** | javac 17 | `None` | N/A | **ABSENT** |
| **Android SDK** | API 34 (`compileSdk 34`) | `None` (`ANDROID_HOME`, `ANDROID_SDK_ROOT` unset; `%LOCALAPPDATA%\Android\Sdk` missing) | N/A | **ABSENT** |
| **Platform-Tools (adb)** | adb client / server | `None` | N/A | **ABSENT** |
| **Build-Tools** | 34.0.0 | `None` | N/A | **ABSENT** |
| **Android NDK** | NDK r25+ / 26.1+ | `None` (`ANDROID_NDK_HOME` unset; no Clang cross-compilers) | N/A | **ABSENT** |
| **Android CMake** | 3.22.1 (`externalNativeBuild.cmake.version`) | System Strawberry Perl CMake (`C:\Strawberry\c\bin\cmake.EXE`) | 3.29.2 (Host Win32 only) | **INCOMPATIBLE / NO TOOLCHAIN** |
| **Host Gradle** | Gradle 8.2+ | `None` | N/A | **ABSENT** |
| **Android Emulator / Device** | AVD or attached USB device | `None` (`emulator.exe` missing; `adb` missing) | N/A | **ABSENT** |

### Project Configuration Compatibility Audit

Inspection of `build.gradle` and `app/build.gradle`:
- **Android Gradle Plugin (AGP)**: `8.2.2` (declared in root `build.gradle`)
- **Kotlin Android Plugin**: `1.9.22`
- **Gradle Version**: `8.2` (in `distributionUrl=https\://services.gradle.org/distributions/gradle-8.2-bin.zip`)
- **AGP / Gradle Compatibility**: AGP 8.2.2 officially requires Gradle 8.2 through 8.5. The specified Gradle 8.2 is **fully compatible**.
- **AGP / JDK Compatibility**: AGP 8.2.2 requires JDK 17 to execute. The project configures `JavaVersion.VERSION_17` and `jvmTarget = '17'`, which is **fully aligned**.
- **Target SDK**: `compileSdk 34`, `targetSdk 34`, `minSdk 24`. Fully compatible with Android 14 / AndroidX.
- **NDK Configuration**: ABIs `arm64-v8a` and `x86_64`; flags `-std=c++17 -O3 -Wall -Wextra`. Targets canonical `kws_mobile_frontend.cpp`.
- **Runtime Dependency**: `com.microsoft.onnxruntime:onnxruntime-android:1.17.0`.

---

## 3. Gradle Wrapper Validation

Inspection of `experiments/synthetic_kws_demo/android_app/`:

| File | Status | Size | Content / Diagnosis |
| :--- | :---: | :---: | :--- |
| `gradlew` | Present | 2,105 bytes | Standard Unix shell wrapper script. |
| `gradlew.bat` | Present | 2,742 bytes | Standard Windows batch wrapper script. |
| `gradle/wrapper/gradle-wrapper.properties` | Present | 250 bytes | Configured for `https\://services.gradle.org/distributions/gradle-8.2-bin.zip`. |
| `gradle/wrapper/gradle-wrapper.jar` | **MISSING** | 0 bytes | **Absent from disk.** |

### Analysis of the Missing Wrapper JAR:
- The Gradle wrapper consists of both scripts (`gradlew`/`gradlew.bat`) AND the bootstrap archive `gradle-wrapper.jar`.
- Under standard development workflows, running `./gradlew` executes `java -jar gradle/wrapper/gradle-wrapper.jar ...` to download and invoke the declared Gradle distribution.
- Because `gradle/wrapper/gradle-wrapper.jar` is missing, running the wrapper scripts would fail at the bootstrap stage even if Java were present.
- Attempting to generate the wrapper with host Gradle (`gradle wrapper --gradle-version 8.2 --distribution-type bin`) failed with exit code 1 because `gradle` is not installed on the workstation.
- Attempting to run `gradlew.bat wrapper` failed with exit code 1 because `JAVA_HOME is not set and no 'java' command could be found in your PATH`.
- Per user instructions (*"Do not fabricate a wrapper JAR, manually improvise wrapper scripts, or choose arbitrary Gradle and plugin versions. If a component is missing, report it and use an appropriate supported method to repair the wrapper after the required toolchain is available."*), no placeholder or mock JAR was created.
- **Remediation Procedure Once Toolchain is Installed**:
  Once JDK 17 and Gradle (or Android Studio) are installed on the workstation, run:
  ```cmd
  gradle wrapper --gradle-version 8.2 --distribution-type bin
  ```
  This will officially generate the verified, signed `gradle-wrapper.jar` matching Gradle 8.2 without manual improvisation.

---

## 4. Real Android Build Attempts & Command Failure Telemetry

### 1. Gate B: Native arm64-v8a Compilation & NDK Toolchain Path Verification
In Android projects managed by AGP (`app/build.gradle`), Gradle invokes CMake automatically passing the absolute toolchain path:
`${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake`. AGP does not rely on a relative path.
For manual command-line CMake invocations, using `-DCMAKE_TOOLCHAIN_FILE=android.toolchain.cmake` fails immediately because CMake requires the exact absolute path to the NDK file.

Attempting compilation with the correct absolute toolchain parameter:
```cmd
cmake -S experiments/synthetic_kws_demo/android_app/app/src/main/cpp -B experiments/synthetic_kws_demo/android_app/build_arm64 -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK_HOME%/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a
```
- **Exit Code**: `1`
- **Output / Error**:
  ```
  CMake Error at C:/Strawberry/c/share/cmake-3.29/Modules/CMakeDetermineSystem.cmake:152 (message):
    Could not find toolchain file: %ANDROID_NDK_HOME%/build/cmake/android.toolchain.cmake
  Call Stack (most recent call first):
    CMakeLists.txt:5 (project)

  CMake Error: CMake was unable to find a build program corresponding to "Ninja".  CMAKE_MAKE_PROGRAM is not set.
  CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
  -- Configuring incomplete, errors occurred!
  ```
- **Finding**: Because `ANDROID_NDK_HOME` is unset and no NDK directory exists on disk, the NDK toolchain file does not exist at either the relative or absolute path.

### 2. Gate C: APK Assembly via Gradle Wrapper Attempt
```cmd
cmd /c "experiments\synthetic_kws_demo\android_app\gradlew.bat assembleDebug"
```
- **Exit Code**: `1`
- **Output / Error**:
  ```
  ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.

  Please set the JAVA_HOME variable in your environment to match the
  location of your Java installation.
  ```
- **Finding**: Execution blocked immediately due to missing Java runtime; subsequent wrapper bootstrap is also blocked by missing `gradle-wrapper.jar`.

### 3. Gate D: Android Device / Emulator Execution Attempt
```cmd
cmd /c "adb devices"
```
- **Exit Code**: `1`
- **Output / Error**:
  ```
  'adb' is not recognized as an internal or external command,
  operable program or batch file.
  ```
- **Finding**: No Android platform-tools exist on the system.

---

## 5. Host Regression & Governance Verification

All host-side test suites and cryptographic baselines were executed and verified:

### 1. Test Suite Execution Summary (113 Total Tests: 113 Passed, 0 Failed)

| Suite | Scope | Command | Passed / Total | Duration | Exit Code |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `validate_android_pipeline.py` | Pipeline & golden vector parity | `python validate_android_pipeline.py` | 12 / 12 | 4.8s | `0` |
| `test_campaign_acceptance.py` | Full campaign acceptance (TC-01..TC-69) | `pytest tests/test_campaign_acceptance.py` | 69 / 69 | 28.15s | `0` |
| `test_kws_export_parity.py` | Export parity & ONNX contract | `pytest tests/test_kws_export_parity.py` | 13 / 13 | 2.99s | `0` |
| `test_synthetic_kws_demo.py` | Demo classifier & game adapter | `pytest tests/test_synthetic_kws_demo.py` | 19 / 19 | 2.59s | `0` |
| `git diff --check` | Formatting & whitespace | `git diff --check` | Clean | 0.2s | `0` |
| `git diff --cached` | Staged file check | `git diff --cached` | 0 files | 0.1s | `0` |

### 2. Cryptographic Baseline Verification (All Bit-Exact)

| Protected File | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **MATCH** |
| `field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **MATCH** |
| `consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **MATCH** |
| Checkpoint `arm_b_onset_no_resample.pt` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | **MATCH** |
| Export ONNX `arm_b_tcresnet14.onnx` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **MATCH** |
| Asset ONNX `arm_b_tcresnet14.onnx` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **MATCH** |
| Golden Manifest `golden_manifest.json` | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | **MATCH** |

### 3. Golden Vector Parity & Edge Case Telemetry (12 Cases)

| Case ID | Category | Intended | Max Feat Diff | Max Prob Diff | Pred Label | Conf | Adapted Action | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `case_01_trai_demo` | canonical_demo | `TRAI` | $3.62 \times 10^{-5}$ | $1.79 \times 10^{-7}$ | `TRAI` | 88.40% | `MOVE_LEFT` | **PASS** |
| `case_02_phai_demo` | canonical_demo | `PHAI` | $2.10 \times 10^{-5}$ | $5.96 \times 10^{-8}$ | `PHAI` | 99.30% | `MOVE_RIGHT` | **PASS** |
| `case_03_len_demo` | canonical_demo | `LEN` | $2.67 \times 10^{-5}$ | $3.88 \times 10^{-8}$ | `LEN` | 96.86% | `JUMP` | **PASS** |
| `case_04_xuong_demo` | canonical_demo | `XUONG` | $2.67 \times 10^{-5}$ | $4.33 \times 10^{-8}$ | `XUONG` | 98.44% | `CROUCH` | **PASS** |
| `case_05_unknown_demo` | canonical_demo | `UNKNOWN` | $2.67 \times 10^{-5}$ | $4.75 \times 10^{-8}$ | `UNKNOWN` | 98.87% | `NO_ACTION` | **PASS** |
| `case_06_silence_demo` | canonical_demo | `SILENCE` | $9.54 \times 10^{-7}$ | $4.98 \times 10^{-8}$ | `SILENCE` | 96.36% | `NO_ACTION` | **PASS** |
| `case_07_trai_leading_pad` | timing_variant | `TRAI` | $3.62 \times 10^{-5}$ | $1.19 \times 10^{-7}$ | `TRAI` | 81.68% | `MOVE_LEFT` | **PASS** |
| `case_08_phai_centered` | timing_variant | `PHAI` | $2.10 \times 10^{-5}$ | $3.54 \times 10^{-8}$ | `PHAI` | 95.06% | `MOVE_RIGHT` | **PASS** |
| `case_09_silence_pure_zeros`| numerical_edge | `SILENCE` | **$0.00 \times 10^{0}$** | $5.96 \times 10^{-8}$ | `TRAI` | 30.23% | `NO_ACTION` | **PASS** |
| `case_10_low_amplitude_ambient`| numerical_edge | `SILENCE` | $9.54 \times 10^{-7}$ | $2.98 \times 10^{-8}$ | `TRAI` | 30.24% | `NO_ACTION` | **PASS** |
| `case_11_transient_impulse` | numerical_edge | `UNKNOWN` | $4.77 \times 10^{-7}$ | $8.94 \times 10^{-8}$ | `TRAI` | 31.37% | `NO_ACTION` | **PASS** |
| `case_12_holdout_trai_native`| holdout_anchor | `TRAI` | $2.19 \times 10^{-5}$ | $1.19 \times 10^{-7}$ | `TRAI` | 86.11% | `MOVE_LEFT` | **PASS** |

All 12 golden WAVs and 12 golden `.npy` feature tensors match their manifest SHA-256 hashes bit-exact.

---

## 6. Actionable Next Steps to Resume Android Build

When an environment provisioned with JDK 17, Android SDK (API 34), Android NDK (r25+), and Android platform-tools is available:

1. **Install JDK 17 and Set Environment**:
   ```cmd
   set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot
   set PATH=%JAVA_HOME%\bin;%PATH%
   ```
2. **Install Android SDK and NDK**:
   ```cmd
   set ANDROID_HOME=C:\Users\%USERNAME%\AppData\Local\Android\Sdk
   set ANDROID_NDK_HOME=%ANDROID_HOME%\ndk\26.1.10909125
   set PATH=%ANDROID_HOME%\platform-tools;%PATH%
   ```
3. **Repair/Regenerate Gradle Wrapper JAR**:
   ```cmd
   cd experiments\synthetic_kws_demo\android_app
   gradle wrapper --gradle-version 8.2 --distribution-type bin
   ```
4. **Compile Native Frontend for arm64-v8a**:
   ```cmd
   cmake -S app/src/main/cpp -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK_HOME%/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a
   cmake --build build_arm64
   ```
5. **Assemble Debug APK**:
   ```cmd
   .\gradlew.bat assembleDebug
   ```
6. **Verify APK Contents**:
   Verify that `libkws_mobile_frontend_jni.so` and `assets/models/arm_b_tcresnet14.onnx` are packed inside `app-debug.apk`.
7. **Deploy and Run Offline Inference**:
   Install via `adb install app-debug.apk` and capture logs via `adb logcat`.
