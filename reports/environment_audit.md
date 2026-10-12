# Environment Audit Report

**Project:** Production-Oriented Vietnamese Keyword Spotting System (TRAI, PHAI, LEN, XUONG)  
**Date:** 2026-10-12  
**Author:** Senior Machine Learning Engineer  
**Audit Status:** COMPLETE — PASSED (with mobile hardware constraints documented)

---

## 1. System Hardware & Operating System

| Component | Specification | Operational Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro (64-bit, Build 10.0.26100) | Available |
| **Host Shell** | PowerShell (Windows Terminal / ConHost) | Operational |
| **Processor (CPU)** | 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz (8 cores, 16 threads) | Available |
| **System Memory (RAM)** | 16.49 GB visible physical RAM (~1.4 GB free initially, memory footprint must be monitored) | Active monitoring required |
| **Graphics (GPU)** | NVIDIA RTX A2000 Laptop GPU (4,096 MiB VRAM), Driver 610.60 | CUDA 13.3 driver active |
| **Storage (Primary D:)** | `D:\trainidentificationaudio` (143.45 GB free space) | Dedicated workspace drive |
| **Storage (System C:)** | `C:\` (61.98 GB free space) | OS & package cache drive |

---

## 2. Core Toolchains & Runtime Environment

### 2.1 Python Runtime & Acceleration
- **Python Version:** Python 3.11.9 (`C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe`)
- **Virtual Environment:** `d:\trainidentificationaudio\.venv` (isolated, created with standard `venv`)
- **PyTorch:** `torch 2.6.0+cu124` with CUDA acceleration enabled (`torch.cuda.is_available() == True`, Device: `NVIDIA RTX A2000 Laptop GPU`)
- **Audio I/O & Signal Processing:**
  - `torchaudio`: 2.6.0+cu124
  - `soundfile`: 0.14.0 (backed by libsndfile)
  - `librosa`: 0.11.0
  - `scipy`: 1.17.1
  - `numpy`: 2.4.6
- **Data Engineering & Hugging Face Tools:**
  - `pandas`: 3.0.6
  - `datasets`: 5.1.0
  - `huggingface_hub`: 2.2.0
  - `tqdm`: 4.70.1
  - `requests`: 2.34.2
- **Model Export & Inference Engine:**
  - `onnx`: 1.23.2
  - `onnxruntime`: 1.31.0
  - `pytest`: 9.1.1 (testing framework)

### 2.2 Audio Processing Binaries
- **FFmpeg Version:** `ffmpeg version 9.0.2-full_build-www.gyan.dev`
- **Location:** In system `PATH`
- **Supported Encodings:** Full build with native support for Opus, FLAC, WAV (PCM 16/24/32-bit), MP3, AAC, and Whisper-compatible frontends. Used for reproducible sample rate normalization to 16 kHz mono 16-bit PCM.

### 2.3 Version Control
- **Git Version:** `git version 2.45.1.windows.1`

---

## 3. Mobile Deployment Toolchain Status

| Mobile Target | Toolchain Component | Host Detection | Status & Action Plan |
| :--- | :--- | :--- | :--- |
| **Android** | `adb` (Android Debug Bridge) | Not in PATH (`where adb` returned code 1) | **NOT AVAILABLE ON HOST** — Export models to mobile runtime format (ONNX / TFLite), verify latency on CPU/NNAPI runtime benchmarks; do not fabricate physical device logs. |
| **Android** | Java JDK / Android SDK | Not in PATH | **NOT AVAILABLE ON HOST** |
| **iOS** | `xcodebuild` / Xcode SDK | Not in PATH / Incompatible OS (Windows 11) | **BLOCKED ON HOST** — Physical iOS compilation and CoreML/XCTest execution require macOS with Xcode. In accordance with project policy, iOS on-device claims are quarantined and flagged as requiring macOS runner. |

---

## 4. Key Environmental Constraints & Mitigations

1. **Available RAM Constraint (16 GB total, ~1.4 GB free initially):**
   - *Mitigation:* Audio data loaders must stream or process in batches; avoid loading entire uncompressed audio datasets (>10 GB) directly into RAM. Use memory-mapped parquet/tar processing and cached 16 kHz WAV chunks.
2. **GPU VRAM Limit (4 GB):**
   - *Mitigation:* Batch size for CNN baseline KWS training set to 64 or 128 with mixed precision (fp16) or standard float32 to prevent CUDA Out-of-Memory.
3. **Encoding on Windows:**
   - *Mitigation:* Explicitly enforce UTF-8 across all Python scripts and PowerShell pipelines to avoid `charmap` codec errors with Vietnamese diacritics.

---

## 5. Verification Commands Run

```powershell
python --version
git --version
ffmpeg -version
nvidia-smi
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python -c "import onnxruntime, soundfile, librosa; print('Core audio/runtime libraries OK')"
```

All verification commands executed with return code 0.
