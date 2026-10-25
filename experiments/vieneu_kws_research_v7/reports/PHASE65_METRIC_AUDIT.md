# Phase 65 — Comprehensive Metric and Statistical Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Source Evaluation Artifact:** `experiments/vieneu_kws_research_v6/predictions/locked_test_summary.json`  
**Audit Artifact:** `experiments/vieneu_kws_research_v7/predictions/reproduced_metrics.json`  
**Status Banner:** `INDEPENDENT_AUDIT_VERIFIED` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 65 independently reconstructed every test denominator, per-keyword confusion matrix, precision, recall, F1, target false rejection rate, non-target false acceptance rate, 5-class accuracy, and single-clip CPU latency from the raw serialized prediction logs of the Phase 60–64 campaign.

### Key Reconciliations and Statistical Validation:
1. **Exact Denominator Verification:**
   - Target Commands: Exactly 160 clips (40 clips each for `TRAI`, `PHAI`, `LEN`, `XUONG`).
   - Non-Target Rejection: Exactly 560 clips (480 hard-negative distractor texts + 80 silence clips).
   - Total Locked Test Clips: Exactly 720 clips across 4 held-out speaker voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`).
2. **Mathematical Equivalence of Macro and Micro Recall:**
   - Because each of the 4 keyword classes has identical test support ($N = 40$), macro-average recall $\frac{1}{4}\sum_k \text{Recall}_k \equiv \text{Micro-Recall} = \frac{\text{Hits}}{160}$. Recomputation verified exact equivalence across all candidates and seeds.
3. **Statistical Aggregation Independence Principle:**
   - The 720 test clips were evaluated across three training seeds (42, 43, 44). Aggregation procedures that pool repeated observations of the same clips into 2,160 trials violate independence and artificially compress standard errors.
   - Phase 65 formally reports: (1) Per-seed 95% Wilson Score Confidence Intervals, and (2) Cross-seed Mean $\pm$ Sample Standard Deviation ($s$).

---

## 2. Reconstructed Performance Matrix Across Candidates and Seeds

| Candidate Architecture | Seed | Macro Command Recall | 95% Wilson CI (Recall) | Target FRR | Non-Target FAR | 95% Wilson CI (FAR) | 5-Class Accuracy | Mean CPU Latency (Batch 1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | 89.38% | [83.64%, 93.26%] | 10.62% | 4.46% | [3.04%, 6.51%] | 94.17% | 3.60 ms |
| **variant_a_cascade_v1** | 43 | 80.00% | [73.13%, 85.46%] | 20.00% | 4.82% | [3.33%, 6.92%] | 91.81% | 4.05 ms |
| **variant_a_cascade_v1** | 44 | 86.87% | [80.77%, 91.25%] | 13.13% | 8.21% | [6.21%, 10.78%] | 90.69% | 4.07 ms |
| **variant_b_standard_bounded_cnn** | 42 | 15.00% | [10.29%, 21.35%] | 85.00% | 0.00% | [0.00%, 0.68%] | 81.11% | 2.56 ms |
| **variant_b_standard_bounded_cnn** | 43 | 1.88% | [0.64%, 5.37%] | 98.12% | 0.00% | [0.00%, 0.68%] | 78.19% | 2.49 ms |
| **variant_b_standard_bounded_cnn** | 44 | 79.37% | [72.45%, 84.92%] | 20.63% | 7.68% | [5.75%, 10.18%] | 89.44% | 2.55 ms |
| **variant_c_hard_negative_ablation** | 42 | 61.25% | [53.52%, 68.45%] | 38.75% | 2.50% | [1.49%, 4.15%] | 89.44% | 2.54 ms |
| **variant_c_hard_negative_ablation** | 43 | 88.12% | [82.20%, 92.26%] | 11.88% | 9.11% | [6.99%, 11.78%] | 90.28% | 2.43 ms |
| **variant_c_hard_negative_ablation** | 44 | 46.88% | [39.31%, 54.59%] | 53.12% | 5.54% | [3.93%, 7.75%] | 83.89% | 2.41 ms |
| **variant_d_stage1_cnn** | 42 | 88.75% | [82.92%, 92.76%] | 11.25% | 13.39% | [10.82%, 16.46%] | 87.08% | 1.70 ms |
| **variant_d_stage1_cnn** | 43 | 75.62% | [68.42%, 81.63%] | 24.38% | 13.21% | [10.66%, 16.27%] | 84.31% | 1.64 ms |
| **variant_d_stage1_cnn** | 44 | 89.38% | [83.64%, 93.26%] | 10.62% | 18.04% | [15.07%, 21.43%] | 83.61% | 1.68 ms |

---

## 3. Cross-Seed Aggregated Statistics (Mean $\pm$ Sample Standard Deviation)

| Candidate Variant | Command Recall (Mean $\pm$ Std) | Non-Target FAR (Mean $\pm$ Std) | 5-Class Accuracy (Mean $\pm$ Std) | Single-Clip CPU Latency | Meets $\ge 95\%$ Recall & $\le 5\%$ FAR? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | **85.42% $\pm$ 4.85%** | **5.83% $\pm$ 2.07%** | 92.22% $\pm$ 1.77% | 3.91 $\pm$ 0.27 ms | NO |
| **variant_b_standard_bounded_cnn** | **32.08% $\pm$ 41.48%** | **2.56% $\pm$ 4.43%** | 82.92% $\pm$ 5.84% | 2.53 $\pm$ 0.04 ms | NO |
| **variant_c_hard_negative_ablation** | **65.42% $\pm$ 20.94%** | **5.71% $\pm$ 3.31%** | 87.87% $\pm$ 3.47% | 2.46 $\pm$ 0.07 ms | NO |
| **variant_d_stage1_cnn** | **84.58% $\pm$ 7.76%** | **14.88% $\pm$ 2.73%** | 85.00% $\pm$ 1.84% | 1.67 $\pm$ 0.03 ms | NO |

---

## 4. Reconstructed Confusion Matrices (Summed across Seeds 42, 43, 44)

### Confusion Matrix: `variant_a_cascade_v1`
```
Actual \ Pred │   TRAI   │   PHAI   │    LEN   │  XUONG   │  REJECT  │ Total
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
TRAI         │      113 │        0 │        0 │        0 │        7 │   120
PHAI         │        2 │       97 │        0 │        0 │       21 │   120
LEN          │        0 │        0 │      103 │        0 │       17 │   120
XUONG        │        0 │        0 │        0 │       97 │       23 │   120
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
REJECT       │       28 │        6 │       40 │       24 │     1582 │  1680
```

### Confusion Matrix: `variant_b_standard_bounded_cnn`
```
Actual \ Pred │   TRAI   │   PHAI   │    LEN   │  XUONG   │  REJECT  │ Total
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
TRAI         │        9 │       31 │        0 │        0 │       80 │   120
PHAI         │        1 │       41 │        0 │        1 │       77 │   120
LEN          │        0 │        0 │       40 │        1 │       79 │   120
XUONG        │        0 │        0 │        0 │       64 │       56 │   120
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
REJECT       │        4 │        0 │       32 │        7 │     1637 │  1680
```

### Confusion Matrix: `variant_c_hard_negative_ablation`
```
Actual \ Pred │   TRAI   │   PHAI   │    LEN   │  XUONG   │  REJECT  │ Total
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
TRAI         │       59 │       19 │        0 │        0 │       42 │   120
PHAI         │       22 │       58 │        0 │        0 │       40 │   120
LEN          │        0 │        0 │       79 │       35 │        6 │   120
XUONG        │        0 │        0 │        0 │      118 │        2 │   120
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
REJECT       │       14 │        0 │       50 │       32 │     1584 │  1680
```

### Confusion Matrix: `variant_d_stage1_cnn`
```
Actual \ Pred │   TRAI   │   PHAI   │    LEN   │  XUONG   │  REJECT  │ Total
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
TRAI         │      108 │        0 │        0 │        0 │       12 │   120
PHAI         │        0 │       73 │        0 │        0 │       47 │   120
LEN          │        0 │        0 │      105 │        0 │       15 │   120
XUONG        │        0 │        0 │        0 │      120 │        0 │   120
─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────
REJECT       │       68 │       26 │       83 │       73 │     1430 │  1680
```

