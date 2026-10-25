# Phase 67 — Controlled Bounded-Utterance Training Ablation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Checkpoints Directory:** `experiments/vieneu_kws_research_v7/models/checkpoints/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 67 executed a controlled experimental comparison between two principled architectural and loss interventions directly addressing the defects identified in Phase 65 and Phase 66:
1. **Candidate A (Corrected Baseline):** Replaces parametric attention pooling with translation-invariant `AdaptiveAvgPool2d((1, 1))` and implements exact inverse-class loss weighting ($w_{\text{cmd}} = 2.0$, $w_{\text{REJECT}} = 0.5$) to neutralize the 4:1 class imbalance.
2. **Candidate B (Moderately Weighted Hard-Negative CNN):** Builds upon Candidate A with bounded hard-negative sample weighting ($1.2\times$ on confusable minimal pairs, preserving $2.0\times$ on target commands) to enhance rejection without crushing command sensitivity.
3. **Historical Cascade v1:** Preserved as an immutable, frozen reference.

---

## 2. Validation Epoch Convergence Across Seeds

| Candidate Intervention | Seed | Best Epoch | Val 5-Class Accuracy | Val Command Recall | Val Non-Target FAR | TRAI Recall | PHAI Recall | LEN Recall | XUONG Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A (Corrected Baseline)** | 42 | 25 | 91.67% | 95.62% | 11.50% | 100.0% | 85.0% | 100.0% | 97.5% |
| **Candidate A (Corrected Baseline)** | 43 | 23 | 94.17% | 99.38% | 10.00% | 100.0% | 100.0% | 100.0% | 97.5% |
| **Candidate A (Corrected Baseline)** | 44 | 23 | 95.56% | 95.62% | 4.50% | 97.5% | 90.0% | 100.0% | 95.0% |
| **Candidate B (Moderate HN)** | 42 | 20 | 96.67% | 96.88% | 3.50% | 97.5% | 95.0% | 100.0% | 95.0% |
| **Candidate B (Moderate HN)** | 43 | 24 | 96.94% | 96.25% | 2.50% | 97.5% | 97.5% | 100.0% | 90.0% |
| **Candidate B (Moderate HN)** | 44 | 21 | 96.11% | 94.38% | 2.50% | 92.5% | 97.5% | 100.0% | 87.5% |

---

## 3. Key Findings from the Controlled Interventions

1. **Elimination of the Command Collapse Deficit:**
   - In Phase 62, unweighted cross-entropy (Variant B) collapsed command recall to 32.08% overall, and attention pooling suffered extreme seed instability (Seed 43 had 1.88% recall).
   - In Phase 67, Candidate A and Candidate B restored robust command sensitivity across all seeds, demonstrating that class-balanced loss weighting and global pooling completely eliminate the pathological zero-recall behavior.
2. **Target Recall Sensitivity:**
   - Inverse frequency weighting ensured that gradient updates from command instances were not drowned out by the 50% majority REJECT class.
3. **Next Step:** Both candidates proceed to Phase 68 alongside Historical Cascade v1 for validation calibration and synthetic robustness benchmarking.
