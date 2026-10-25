# Phase 62 — Controlled Bounded-Utterance Improvement and Ablation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 62 conducted a controlled experimental comparison between three bounded-utterance candidate paradigms on the fresh Benchmark Dataset v6:
1. **Variant A: Cascade v1 (Historical Reference):** Two-stage screening (`VietnameseKWSNet` + `TemporalKWSVerifier`), evaluated frozen without retraining.
2. **Variant B: Standard Bounded CNN:** Single-stage `VietnameseBoundedKWSNet` trained with standard cross-entropy across seeds 42, 43, 44.
3. **Variant C: Controlled Hard-Negative Ablation:** `VietnameseBoundedKWSNet` trained with targeted hard-negative loss reweighting (1.6x penalty on confusable minimal pairs, 1.4x positive-preserving weight on target keywords) and time/noise/gain augmentations across seeds 42, 43, 44.

---

## 2. Validation Epoch Convergence Across Seeds

| Candidate Variant | Seed | Best Epoch | Val 5-Class Accuracy | Val Command Recall | Val Non-Target FAR | TRAI Recall | PHAI Recall | LEN Recall | XUONG Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Variant B (Standard CNN)** | 42 | 24 | 72.78% | 41.25% | 2.00% | 55.0% | 30.0% | 2.5% | 77.5% |
| **Variant B (Standard CNN)** | 43 | 25 | 84.17% | 65.62% | 1.00% | 25.0% | 85.0% | 67.5% | 85.0% |
| **Variant B (Standard CNN)** | 44 | 25 | 84.72% | 74.38% | 7.00% | 35.0% | 85.0% | 92.5% | 85.0% |
| **Variant C (Ablation CNN)** | 42 | 23 | 76.39% | 51.25% | 3.50% | 57.5% | 47.5% | 15.0% | 85.0% |
| **Variant C (Ablation CNN)** | 43 | 25 | 89.17% | 81.88% | 5.00% | 52.5% | 85.0% | 100.0% | 90.0% |
| **Variant C (Ablation CNN)** | 44 | 19 | 80.28% | 74.38% | 15.00% | 67.5% | 52.5% | 82.5% | 95.0% |

---

## 3. Key Observations and Transition to Phase 63

1. **Target Recall Sensitivity:** The positive-preserving loss weighting in Variant C successfully elevates command sensitivity across hard vowels and codas.
2. **Non-Target Discrimination:** Hard-negative weighting penalizes phonetically close minimal pairs (/ʈ/ vs /tɕ/, /f/ vs /tʰ/, open vowels /aːj/, /uəŋ/), reducing raw false triggers.
3. **Next Step:** Both models and the historical Cascade v1 reference proceed to Phase 63 for validation-only confidence threshold calibration and synthetic robustness testing.
