# Phase 60 — Metric Reconciliation and Statistical Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Reconciliation Artifact:** `experiments/vieneu_kws_research_v6/predictions/phase60_reproduced_metrics.json`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 60 independently audited and reconstructed every sample denominator, per-keyword confusion matrix cell, confidence interval calculation, and CPU latency component from the raw prediction logs of Phase 55–59.

### Key Audit Reconciliations:
1. **Mathematical Equivalence of Macro and Micro Recall:** Because the test split contains an identical count of 40 clips per target keyword ($N_{\text{cmd}} = 160$), macro-average recall ($\frac{1}{4} \sum_k \text{Recall}_k$) and micro-average recall ($\frac{\sum \text{Hits}}{160}$) are mathematically identical.
2. **Discrete Increment Step Sizes:**
   - Per-keyword change per clip: $\frac{1}{40} = 2.500\%$.
   - Command recall change per clip: $\frac{1}{160} = 0.625\%$.
   - Non-target FAR change per clip: $\frac{1}{560} = 0.17857\%$.
   - Overall test accuracy change per clip: $\frac{1}{720} = 0.13889\%$.
3. **Statistical Aggregation Principle:** The 720 test clips were repeatedly evaluated across three training seeds (42, 43, 44). Treating repeated observations of the same clips as 2,160 independent trials would falsely compress standard errors and confidence intervals. Therefore, Phase 60 reports:
   - Per-seed 95% Wilson Score Confidence Intervals.
   - Cross-seed Mean $\pm$ Sample Standard Deviation ($s$).
4. **Candidate 2 (Cascade v1) Reconciled Performance:**
   - **Command Recall:** $92.29\% \pm 1.44\%$ (Seed 42: $93.12\%$, Seed 43: $93.12\%$, Seed 44: $90.62\%$).
   - **Non-Target FAR:** $4.64\% \pm 2.11\%$ (Seed 42: $2.32\%$, Seed 43: $5.18\%$, Seed 44: $6.43\%$).
   - **Target FRR:** $7.71\% \pm 1.44\%$.
   - **5-Class Accuracy:** $94.68\% \pm 1.89\%$.
   - **Cross-Command Confusion:** Exactly **0** instances.

---

## 2. Comprehensive Test Denominator Reconstruction

Evaluated on `experiments/vieneu_kws_research_v5/data/splits/test_manifest.csv` ($N = 720$ clips across 4 held-out voices):

| Candidate | Seed | Target Clips (`TRAI`, `PHAI`, `LEN`, `XUONG`) | Non-Target Clips (`UNKNOWN`, `SILENCE`) | Correct Target Hits | Target False Rejections (Miss to REJECT) | Cross-Command Wrong Accepts | Non-Target False Accepts | Non-Target Correct Rejections | Total Verified Clips |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cascade v1** | 42 | 160 (40 each) | 560 (480 UNK + 80 SIL) | 149 | 11 | **0** | 13 | 547 | **720** |
| | 43 | 160 (40 each) | 560 (480 UNK + 80 SIL) | 149 | 11 | **0** | 29 | 531 | **720** |
| | 44 | 160 (40 each) | 560 (480 UNK + 80 SIL) | 145 | 15 | **0** | 36 | 524 | **720** |
| **Stage 1 CNN**| 42 | 160 (40 each) | 560 (480 UNK + 80 SIL) | 153 | 7 | **0** | 78 | 482 | **720** |
| | 43 | 160 (40 each) | 560 (480 UNK + 80 SIL) | 150 | 10 | **0** | 88 | 472 | **720** |
| | 44 | 160 (40 each) | 560 (480 UNK + 80 SIL) | 158 | 2 | **0** | 143 | 417 | **720** |

---

## 3. Recomputed Metrics with Statistical Confidence Intervals

### Candidate 2: Cascade v1 Bounded Pipeline

| Metric | Seed 42 | Seed 43 | Seed 44 | Cross-Seed Mean $\pm$ Std |
| :--- | :--- | :--- | :--- | :--- |
| **Macro Recall** | 93.12% [88.11%, 96.12%] | 93.12% [88.11%, 96.12%] | 90.62% [85.11%, 94.24%] | **92.29% $\pm$ 1.44%** |
| **Target FRR** | 6.88% [3.88%, 11.89%] | 6.88% [3.88%, 11.89%] | 9.38% [5.76%, 14.89%] | **7.71% $\pm$ 1.44%** |
| **Non-Target FAR**| **2.32%** [1.36%, 3.93%] | **5.18%** [3.63%, 7.34%] | **6.43%** [4.68%, 8.77%] | **4.64% $\pm$ 2.11%** |
| **Non-Target TNR**| 97.68% [96.07%, 98.64%] | 94.82% [92.66%, 96.37%] | 93.57% [91.23%, 95.32%] | **95.36% $\pm$ 2.11%** |
| **5-Class Accuracy**| 96.67% [95.09%, 97.75%] | 94.44% [92.52%, 95.88%] | 92.92% [90.81%, 94.57%] | **94.68% $\pm$ 1.89%** |

#### Per-Keyword Recall Breakdown (Cascade v1):
- **`TRAI`:** Seed 42 = 100.0% (40/40), Seed 43 = 95.0% (38/40), Seed 44 = 100.0% (40/40) $\rightarrow$ **Mean: 98.33% $\pm$ 2.89%**
- **`PHAI`:** Seed 42 = 85.0% (34/40), Seed 43 = 92.5% (37/40), Seed 44 = 95.0% (38/40) $\rightarrow$ **Mean: 90.83% $\pm$ 5.20%**
- **`LEN`:** Seed 42 = 95.0% (38/40), Seed 43 = 97.5% (39/40), Seed 44 = 80.0% (32/40) $\rightarrow$ **Mean: 90.83% $\pm$ 9.46%**
- **`XUONG`:** Seed 42 = 92.5% (37/40), Seed 43 = 87.5% (35/40), Seed 44 = 87.5% (35/40) $\rightarrow$ **Mean: 89.17% $\pm$ 2.89%**

---

## 4. Reconciled CPU Latency Decomposition

Benchmarking setup:
- **CPU:** Intel64 Family 6 Model 141 Stepping 1 (GenuineIntel)
- **Threads:** 4 threads (`torch.set_num_threads(4)`)
- **Batch Size:** 1 (Single clip sequential processing)
- **Trials:** 200 repetitions following 30 warmup iterations
- **Timer:** High-resolution `time.perf_counter()`

| Latency Pipeline Stage | Candidate 1: Stage 1 CNN Alone | Candidate 2: Cascade v1 (Worst-Case Path) |
| :--- | :--- | :--- |
| **1. Feature Extraction (Log-Mel)** | 0.659 ms (P90 = 1.010 ms) | Stage 1: 0.554 ms <br> Stage 2: 0.581 ms |
| **2. Neural Network Forward Pass** | 1.629 ms (P90 = 2.221 ms) | Stage 1: 1.473 ms <br> Stage 2: 4.740 ms <br> Total NN: **6.212 ms** (P90 = 7.702 ms) |
| **3. Rejection / Thresholding Logic** | 0.064 ms (P90 = 0.080 ms) | 0.039 ms (P90 = 0.076 ms) |
| **Total Post-Capture Decision Latency**| **2.354 ms** (Median = 2.129 ms, P90 = 3.153 ms) | **7.434 ms** (Median = 7.226 ms, P90 = 9.206 ms, P99 = 13.079 ms) |
| **Audio Capture Duration** | 1,000.000 ms | 1,000.000 ms |
| **Total End-to-End Perceived Latency**| **1,002.354 ms** (P90 = 1,003.153 ms) | **1,007.434 ms** (P90 = 1,009.206 ms) |
| **Compliance with $\le 50\,\text{ms}$ Post-Capture Budget** | **PASS** ($2.35\,\text{ms} \ll 50\,\text{ms}$) | **PASS** ($7.43\,\text{ms} \ll 50\,\text{ms}$) |
