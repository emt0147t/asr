# Android Offline Vietnamese KWS Demonstration App (Phase 9)

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> 
> This Android application operates strictly offline on pre-packaged synthetic audio fixtures. It contains zero microphone capture permissions, zero network permissions, and zero streaming acquisition code. All acoustic models and golden test vectors remain frozen.

---

## Overview

This project integrates the Vietnamese Keyword Spotting (KWS) inference engine into an Android application:
1. **Audio Preprocessing**: High-precision C++ Log-Mel frontend compiled via Android NDK through a thread-safe JNI bridge.
2. **Neural Inference**: Microsoft ONNX Runtime Mobile (`onnxruntime-android:1.17.0`) executing the frozen TC-ResNet-14 model (`arm_b_tcresnet14.onnx`).
3. **Safety Adaptation**: Decoupled game action adapter enforcing non-target safety suppression and a confidence threshold of 0.50.

---

## Directory Structure

```
android_app/
├── app/
│   ├── build.gradle                              # Application build config (minSdk 24, targetSdk 34)
│   ├── proguard-rules.pro                        # Proguard keep rules for JNI and ONNX Runtime
│   └── src/
│       └── main/
│           ├── AndroidManifest.xml               # Zero permissions (No RECORD_AUDIO, No INTERNET)
│           ├── assets/
│           │   ├── models/arm_b_tcresnet14.onnx  # Packaged ONNX model (verified bit-exact)
│           │   ├── audio/*.wav                   # Packaged synthetic demonstration fixtures
│           │   └── asset_manifest.json           # File metadata and whole-file SHA-256 hashes
│           ├── cpp/
│           │   ├── CMakeLists.txt                # NDK CMake build referencing canonical mobile_frontend
│           │   └── kws_jni_bridge.cpp            # Thread-safe JNI bridge implementation
│           ├── java/com/example/synthetic_kws/
│           │   ├── MainActivity.kt               # UI controller and interactive fixture selector
│           │   ├── KwsFrontendJni.kt             # Native method declarations
│           │   ├── WavAudioLoader.kt             # Strict 16kHz mono WAV parser
│           │   ├── OnnxKwsClassifier.kt          # ONNX Runtime model inference wrapper
│           │   └── GameActionAdapter.kt          # Thresholding and action mapping
│           └── res/
│               ├── layout/activity_main.xml      # Accessible mobile layout
│               └── values/strings.xml
├── build.gradle                                  # Root Gradle configuration
├── settings.gradle                               # Module registry
├── gradle.properties                             # Build parameters
├── validate_android_pipeline.py                  # Host-side validation test harness
├── android_parity_report.json                    # Structured test report
└── PHASE9_ANDROID_AUDIT.md                       # Comprehensive audit report
```

---

## Verification & Five-Gate Status

### Executive Verdict: `BLOCKED`

The overall Phase 9 verdict is **`BLOCKED`** for the Android deployment objective. While host-side static and numerical pipeline validation succeeded, the host workstation lacks the required Java Development Kit (JDK), Gradle, Android SDK, and Android NDK toolchains. **A project scaffold, static validator, or Windows DLL run does NOT constitute an Android deployment.**

| Gate | Description | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Gate A** | **Host-side / static validation** | **`PASS`** | Max feature error $3.62 \times 10^{-5}$, max prob error $1.79 \times 10^{-7}$; edge cases suppressed to `NO_ACTION`. |
| **Gate B** | **Android NDK compilation** | **`BLOCKED`** | Host lacks Android NDK (`ANDROID_NDK_HOME` not set). No target `.so` libraries compiled. |
| **Gate C** | **APK build** | **`BLOCKED`** | Host lacks JDK, Gradle, and Android SDK. No APK assembled. |
| **Gate D** | **Android runtime inference** | **`BLOCKED`** | Requires built APK and emulator/device via `adb`. No runtime execution occurred. |
| **Gate E** | **Physical-device performance** | **`NOT_TESTED`** | Requires attached physical Android hardware. No hardware measurements taken. |

---

## Host Validation Execution

To run the offline host validation script:
```powershell
.\.venv\Scripts\python.exe experiments/synthetic_kws_demo/android_app/validate_android_pipeline.py
```

