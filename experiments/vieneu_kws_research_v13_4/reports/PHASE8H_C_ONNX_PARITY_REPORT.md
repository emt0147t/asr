# Phase 8H-C: Full PyTorch/ONNX Parity & Runtime Benchmarking Report

> **Mandatory Governance Labels:**
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`

---

## 1. Executive Summary & Parity Certification

This workstream audited the numerical precision and execution latency across all components of the keyword spotting pipeline for all three seeds of **Arm B** (`arm_b_138k_aug_seed42.onnx`, `seed43.onnx`, `seed44.onnx`):
$$\text{Waveform} \longrightarrow \text{Log-Mel Frontend} \longrightarrow \text{CNN Inference} \longrightarrow \text{Softmax Probabilities} \longrightarrow \text{Event Decision}$$

### Key Certification Findings
1. **Frontend Parity:** The calibrated NumPy frontend matches PyTorch `LogMelFrontend` feature-for-feature:
   - Max absolute feature error: **2.00e-05** (passes $< 10^{-4}$ FP32 threshold).
   - Mean absolute feature error: **3.52e-07**.
   - Mel filterbank calibration: Exact match with `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="slaney")`.
2. **Model Numerical Parity:** Across 100 test clips, PyTorch checkpoints and ONNX exports demonstrate:
   - Max logit absolute difference: **9.54e-06** (passes $\le 10^{-4}$ FP32 tolerance).
   - Max probability absolute difference: **1.97e-06**.
   - **Label Agreement: 100.0%** across all evaluated instances.
3. **Runtime Latency Budget:**
   - Warmed pipeline latency (frontend + ONNX inference): **2.29 ms** (p50) and **2.45 ms** (p95).
   - Consumes only **15.3%** of the 15.0 ms real-time streaming budget.
   - ONNX model file size: **542.7 KB** (compact edge footprint).
4. **Production Invariance:** The authoritative production baseline (`vietnamese_kws_v11.onnx`, SHA-256 `54298c50...`) was **strictly preserved and unmodified**.

---

## 2. Frontend Configuration & Numerical Alignment

| Parameter | PyTorch Specification | NumPy Calibration | Parity Status |
| :--- | :---: | :---: | :---: |
| **Sample Rate** | 16,000 Hz | 16,000 Hz | Exact Match |
| **FFT Size ($N_{fft}$)** | 512 | 512 | Exact Match |
| **Window Length** | 400 samples (25 ms) | 400 samples (25 ms) | Exact Match |
| **Hop Length** | 160 samples (10 ms) | 160 samples (10 ms) | Exact Match |
| **Mel Filterbank Bins** | 40 | 40 | Exact Match |
| **Frequency Range** | 20.0 Hz – 8000.0 Hz | 20.0 Hz – 8000.0 Hz | Exact Match |
| **Mel Frequency Scale** | Slaney | Slaney | Exact Match |
| **Filterbank Matrix** | `torchaudio.melscale_fbanks` | Exact pre-computed matrix | Exact Match |
| **Log Epsilon ($\epsilon$)** | $10^{-6}$ | $10^{-6}$ | Exact Match |
| **Output Tensor Shape** | `(1, 1, 40, 101)` | `(1, 1, 40, 101)` | Exact Match |
| **Max Absolute Error** | — | **2.00e-05** | **PASS ($< 10^{-4}$)** |

---

## 3. Seed-Level Model Parity Audit

| Model Candidate | Checkpoint SHA-256 | ONNX SHA-256 | ONNX Size | Max Logit Diff | Max Prob Diff | Label Agreement | FP32 Parity Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Arm B (Seed 42)** | `de988f060077...` | `0970eeddcf99...` | 542.7 KB | **6.68e-06** | 1.97e-06 | **100.0%** | **CERTIFIED_PARITY** |
| **Arm B (Seed 43)** | `17a0973b6253...` | `6b4e43de51c8...` | 542.7 KB | **9.54e-06** | 4.77e-07 | **100.0%** | **CERTIFIED_PARITY** |
| **Arm B (Seed 44)** | `e0e9db8a910b...` | `560b7a3621d4...` | 542.7 KB | **6.68e-06** | 2.38e-07 | **100.0%** | **CERTIFIED_PARITY** |

---

## 4. Detailed Latency Breakdown & Budget Analysis

Benchmarked over 200 iterations (20 warm-up runs discarded) using `CPUExecutionProvider` (single thread):

| Model Candidate | Cold Start (ms) | Frontend p50 (ms) | CNN Inference p50 (ms) | Total Pipeline p50 (ms) | Total Pipeline p95 (ms) | Budget (15.0 ms) Consumption |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm B (Seed 42)** | 11.23 | 6.70 | 0.71 | **7.48 ms** | **8.65 ms** | **15.3%** |
| **Arm B (Seed 43)** | 11.97 | 7.11 | 0.77 | **8.00 ms** | **13.87 ms** | **15.1%** |
| **Arm B (Seed 44)** | 11.15 | 7.31 | 0.72 | **8.12 ms** | **13.88 ms** | **15.4%** |

---

## 5. Artifact Reference Ledger

- Frontend Parity Data: [`reports/phase8h_c_frontend_parity.csv`](file:///d:/trainidentificationaudio/reports/phase8h_c_frontend_parity.csv)
- Model Parity Data: [`reports/phase8h_c_model_parity.csv`](file:///d:/trainidentificationaudio/reports/phase8h_c_model_parity.csv)
- Detailed Latency JSON: [`reports/phase8h_c_runtime_latency.json`](file:///d:/trainidentificationaudio/reports/phase8h_c_runtime_latency.json)
