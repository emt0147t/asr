# Phase 8F-D: Arm B ONNX Export & Runtime Validation Report

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Commit Baseline:** `eea9caf`  
**Date:** 2026-10-31  

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Export Details

Workstream D exported the candidate Arm B models ([`CorrectedVietnameseKWSNet`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/scripts/train_eval_phase8.py#L75-L109), 138,469 parameters trained with acoustic channel augmentation) to standard FP32 ONNX format across all three seeds:
- `arm_b_138k_aug_seed42.onnx` (547.4 KB)
- `arm_b_138k_aug_seed43.onnx` (547.4 KB)
- `arm_b_138k_aug_seed44.onnx` (547.4 KB)

Export Location: `experiments/vieneu_kws_research_v13_4/checkpoints/phase8/onnx/`

### Preserved Invariants
- The production default ONNX model ([`vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx), SHA-256 `54298c50...`) was **strictly preserved and not overwritten**.
- `legacy_v11` remains the active production profile.

---

## 2. Numerical Parity Verification (PyTorch vs ONNX)

Recorded in [`reports/phase8f_d_onnx_parity.csv`](file:///d:/trainidentificationaudio/reports/phase8f_d_onnx_parity.csv):

| Model Artifact | Seed | ONNX File Size | Max Logit Abs Diff | Max Probability Abs Diff | Label Agreement (%) | Parity Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `arm_b_138k_aug_seed42.onnx` | 42 | 547.4 KB | $0.000003$ | $0.000001$ | **100.0%** | **PASS** (Tolerance $< 10^{-4}$) |
| `arm_b_138k_aug_seed43.onnx` | 43 | 547.4 KB | $0.000004$ | $0.000001$ | **100.0%** | **PASS** (Tolerance $< 10^{-4}$) |
| `arm_b_138k_aug_seed44.onnx` | 44 | 547.4 KB | $0.000003$ | $0.000001$ | **100.0%** | **PASS** (Tolerance $< 10^{-4}$) |

NumPy frontend ([`LogMelFrontendNumPy`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/infer.py)) demonstrated complete bitwise and numerical compatibility with the exported models.

---

## 3. Runtime Latency & Memory Footprint

Recorded in [`reports/phase8f_d_runtime_latency.json`](file:///d:/trainidentificationaudio/reports/phase8f_d_runtime_latency.json):

| Model Seed | Cold Latency (ms) | Inference p50 (ms) | Inference p95 (ms) | Frontend + Inference p50 (ms) | Frontend + Inference p95 (ms) | Latency SLA Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seed 42** | 6.8 ms | 0.81 ms | 1.12 ms | **1.35 ms** | **1.72 ms** | **PASS** ($< 20.0\text{ ms}$) |
| **Seed 43** | 6.5 ms | 0.80 ms | 1.09 ms | **1.34 ms** | **1.69 ms** | **PASS** ($< 20.0\text{ ms}$) |
| **Seed 44** | 6.7 ms | 0.81 ms | 1.15 ms | **1.35 ms** | **1.71 ms** | **PASS** ($< 20.0\text{ ms}$) |

- **Peak Memory:** $< 45\text{ MB}$ RSS on CPU.
- **File Footprint:** $547.4\text{ KB}$ per model ($6.7\times$ smaller than E2 304k models at $3.6\text{ MB}$).

---

## 4. Continuous KWS Game Mode Runtime Tests

All continuous sequence integration fixtures passed:
1. `LEN -> LEN -> PHAI` (0 ms intentional gap): **EXACT MATCH** (`LEN -> LEN -> PHAI`).
2. `LEN -> LEN -> PHAI` (50 ms intentional gap): **EXACT MATCH** (`LEN -> LEN -> PHAI`).
3. Repeated same-label commands (`LEN -> LEN`, `PHAI -> PHAI`): **PASSED** (properly split into two events).
4. Mixed chains (`TRAI -> XUONG -> LEN`): **PASSED**.
5. Duplicate event emission: **0 duplicate events**.
6. Session flush & reset: Clean lifecycle termination without orphan events.
