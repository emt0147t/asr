# Phase 74 — Fresh Locked Test Benchmark Evaluation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Test Partition:** `experiments/vieneu_kws_research_v8/data/splits/test_manifest.csv` ($N = 720$ clips)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Voice Provenance Audit

Phase 74 evaluated the selected candidate architectures on the fresh locked 720-clip synthetic benchmark dataset v8.

### Speaker Voice Provenance Audit & Plain Limitation Disclosure:
- The 4 held-out speaker voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`) are strictly disjoint from the 17 training voices and 4 validation voices.
- **Explicit Disclosure:** An exhaustive audit of the approved VieNeu-TTS inventory confirmed that exactly 25 voices exist in total (17 train, 4 val, 4 test). There are no additional approved voices available. Therefore, while these 4 test voices are strictly held out from the v8 training split and evaluated on 80 brand-new textual prompts, they have been encountered in historical benchmarks (Phases 56–69). This evaluation proves generalization to **brand-new unseen text prompts on a fixed synthetic voice inventory**, and must NOT be claimed as zero-shot generalization to arbitrary unencountered human speakers.

---

## 2. Cross-Seed Locked Test Performance Matrix

| Candidate Architecture | Seed | Macro Command Recall | 95% Wilson CI (Recall) | Non-Target FAR | Raw FA (out of 560) | 95% Wilson CI (FAR) | 5-Class Accuracy | Mean CPU Latency (Batch 1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_w10** | 42 | 100.00% | [97.66%, 100.00%] | 2.68% | 15/560 | [1.63%, 4.37%] | 97.92% | 1.73 ms |
| **variant_w10** | 43 | 99.38% | [96.55%, 99.89%] | 1.61% | 9/560 | [0.85%, 3.03%] | 98.61% | 1.76 ms |
| **variant_w10** | 44 | 100.00% | [97.66%, 100.00%] | 1.79% | 10/560 | [0.97%, 3.26%] | 98.61% | 1.71 ms |
| **candidate_b_v7_baseline** | 42 | 100.00% | [97.66%, 100.00%] | 2.50% | 14/560 | [1.49%, 4.15%] | 98.06% | 1.72 ms |
| **candidate_b_v7_baseline** | 43 | 99.38% | [96.55%, 99.89%] | 2.86% | 16/560 | [1.77%, 4.59%] | 97.64% | 1.64 ms |
| **candidate_b_v7_baseline** | 44 | 100.00% | [97.66%, 100.00%] | 1.07% | 6/560 | [0.49%, 2.32%] | 99.17% | 1.85 ms |
| **cascade_v1** | 42 | 88.12% | [82.20%, 92.26%] | 1.07% | 6/560 | [0.49%, 2.32%] | 96.53% | 3.56 ms |
| **cascade_v1** | 43 | 81.25% | [74.50%, 86.54%] | 1.07% | 6/560 | [0.49%, 2.32%] | 95.00% | 4.12 ms |
| **cascade_v1** | 44 | 85.00% | [78.65%, 89.71%] | 3.04% | 17/560 | [1.90%, 4.81%] | 94.31% | 3.37 ms |

---

## 3. Statistical Aggregate Summary (Mean $\pm$ Sample Std Dev)

| Candidate Variant | Command Recall (Mean $\pm$ Std) | Non-Target FAR (Mean $\pm$ Std) | 5-Class Accuracy (Mean $\pm$ Std) | CPU Latency (Mean $\pm$ Std) | Meets $\ge 95\%$ Recall & $\le 5\%$ FAR? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_w10** | **99.79% $\pm$ 0.36%** | **2.03% $\pm$ 0.57%** | 98.38% $\pm$ 0.40% | 1.73 $\pm$ 0.03 ms | YES |
| **candidate_b_v7_baseline** | **99.79% $\pm$ 0.36%** | **2.14% $\pm$ 0.95%** | 98.29% $\pm$ 0.79% | 1.74 $\pm$ 0.11 ms | YES |
| **cascade_v1** | **84.79% $\pm$ 3.44%** | **1.73% $\pm$ 1.14%** | 95.28% $\pm$ 1.14% | 3.68 $\pm$ 0.39 ms | NO |
