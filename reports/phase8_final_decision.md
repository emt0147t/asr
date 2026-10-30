# Phase 8: Final Training & Architecture Decision Document

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Final Status:** SELECT_AUGMENTED_138K  
**Date:** 2026-10-31  

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Formal Decision Declaration

### Decision: `SELECT_AUGMENTED_138K`

**Rationale:**  
Arm B (138k + Aug) achieved marked ambient channel gain (+15.0%) meeting acceptance criteria without expanding CNN parameter budget.

---

## 2. Factorial Matrix Performance Summary

| Architecture Arm | Parameters | Augmentation | Direct Stream Acc (%) | Headset Acc (%) | Laptop Mic Acc (%) | Ambient Loopback Acc (%) | Ambient WER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A (Baseline 138k)** | 138,469 | Clean Only | 96.0% | 95.0% | 96.3% | 73.7% | 8.9% |
| **Arm B (Augmented 138k)** | 138,469 | Acoustic Aug | 96.0% | 96.0% | 95.3% | **88.7%** | **2.6%** |
| **Arm C (Capacity 304k)** | 304,277 | Clean Only | 97.7% | 96.0% | 97.0% | 74.0% | 3.7% |
| **Arm D (Augmented 304k)** | 304,277 | Acoustic Aug | 97.0% | 96.7% | 97.0% | **92.0%** | **2.9%** |
| **Frozen v11 ONNX** | 138,469 | Historical | 97.0% | 96.0% | 92.0% | 84.0% | 4.4% |

---

## 3. Preregistered Hypotheses Verification

1. **$H_1$ (Augmentation Effect on 138k):** Validated. Arm B improves ambient loopback accuracy by +15.00 percentage points over Arm A.
2. **$H_2$ (Capacity vs. Augmentation Interaction):** Validated. Arm B (138k + Aug, 88.7%) outperforms Arm C (304k + Clean, 74.0%) by +14.67 percentage points on ambient loudspeaker loopback, confirming that channel augmentation dominates capacity scaling for acoustic mismatch.
3. **$H_3$ (Capacity Scaling Interaction):** Evaluated. Arm D achieved 92.0% ambient loopback accuracy vs 88.7% for Arm B.

---

## 4. Invariant Protection & Production Default Integrity

1. **Frozen v11 Invariance:** The authoritative v11 ONNX model (`54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`) and PyTorch checkpoint were untouched.
2. **Production Default Protection:** `legacy_v11` remains the default production path until field recordings are formally ingested and multi-speaker acceptance criteria are met.
3. **Quarantined Final Holdout:** All 55 trials from `phase6_1a_holdout_manifest.csv` were evaluated blind without influencing training or early stopping.
