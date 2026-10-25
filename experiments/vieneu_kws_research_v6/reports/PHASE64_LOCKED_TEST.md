# Phase 64 — Locked Test Benchmark Evaluation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Test Manifest:** `experiments/vieneu_kws_research_v6/data/splits/test_manifest.csv` ($N = 720$ clips across 4 held-out voices)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 64 conducted the definitive evaluation on the freshly synthesized Benchmark Dataset v6 locked test split ($N = 720$ clips: 160 commands, 480 non-target distractor texts, 80 silence clips) using 4 held-out speaker voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`).

All thresholds were frozen from Phase 63 validation calibration. Testing was conducted sequentially on CPU with batch size 1 across seeds 42, 43, 44.

---

## 2. Cross-Seed Locked Test Performance Matrix

| Candidate Architecture | Seed | Macro Command Recall | 95% Wilson CI (Recall) | Non-Target FAR | 95% Wilson CI (FAR) | 5-Class Accuracy | Mean CPU Latency (Batch 1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | 89.38% | [83.64%, 93.26%] | 4.46% | [3.04%, 6.51%] | 94.17% | 3.60 ms |
| **variant_a_cascade_v1** | 43 | 80.00% | [73.13%, 85.46%] | 4.82% | [3.33%, 6.92%] | 91.81% | 4.05 ms |
| **variant_a_cascade_v1** | 44 | 86.87% | [80.77%, 91.25%] | 8.21% | [6.21%, 10.78%] | 90.69% | 4.07 ms |
| **variant_b_standard_bounded_cnn** | 42 | 15.00% | [10.29%, 21.35%] | 0.00% | [0.00%, 0.68%] | 81.11% | 2.56 ms |
| **variant_b_standard_bounded_cnn** | 43 | 1.88% | [0.64%, 5.37%] | 0.00% | [0.00%, 0.68%] | 78.19% | 2.49 ms |
| **variant_b_standard_bounded_cnn** | 44 | 79.37% | [72.45%, 84.92%] | 7.68% | [5.75%, 10.18%] | 89.44% | 2.55 ms |
| **variant_c_hard_negative_ablation** | 42 | 61.25% | [53.52%, 68.45%] | 2.50% | [1.49%, 4.15%] | 89.44% | 2.54 ms |
| **variant_c_hard_negative_ablation** | 43 | 88.12% | [82.20%, 92.26%] | 9.11% | [6.99%, 11.78%] | 90.28% | 2.43 ms |
| **variant_c_hard_negative_ablation** | 44 | 46.88% | [39.31%, 54.59%] | 5.54% | [3.93%, 7.75%] | 83.89% | 2.41 ms |
| **variant_d_stage1_cnn** | 42 | 88.75% | [82.92%, 92.76%] | 13.39% | [10.82%, 16.46%] | 87.08% | 1.70 ms |
| **variant_d_stage1_cnn** | 43 | 75.62% | [68.42%, 81.63%] | 13.21% | [10.66%, 16.27%] | 84.31% | 1.64 ms |
| **variant_d_stage1_cnn** | 44 | 89.38% | [83.64%, 93.26%] | 18.04% | [15.07%, 21.43%] | 83.61% | 1.68 ms |

---

## 3. Statistical Aggregate Summary (Mean $\pm$ Sample Std Dev)

| Candidate Variant | Command Recall (Mean $\pm$ Std) | Non-Target FAR (Mean $\pm$ Std) | 5-Class Accuracy (Mean $\pm$ Std) | CPU Latency (Mean $\pm$ Std) | Meets Both Criteria? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | **85.42% $\pm$ 4.86%** | **5.83% $\pm$ 2.07%** | 92.22% $\pm$ 1.78% | 3.91 $\pm$ 0.27 ms | NO |
| **variant_b_standard_bounded_cnn** | **32.08% $\pm$ 41.47%** | **2.56% $\pm$ 4.43%** | 82.91% $\pm$ 5.84% | 2.53 $\pm$ 0.04 ms | NO |
| **variant_c_hard_negative_ablation** | **65.42% $\pm$ 20.93%** | **5.72% $\pm$ 3.31%** | 87.87% $\pm$ 3.47% | 2.46 $\pm$ 0.07 ms | NO |
| **variant_d_stage1_cnn** | **84.58% $\pm$ 7.77%** | **14.88% $\pm$ 2.74%** | 85.00% $\pm$ 1.84% | 1.67 $\pm$ 0.03 ms | NO |

---

## 4. Per-Keyword Recall Breakdown (Averaged across Seeds)

| Candidate Variant | TRAI Recall | PHAI Recall | LEN Recall | XUONG Recall | Cross-Command Confusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 94.17% | 80.83% | 85.83% | 80.83% | **0** |
| **variant_b_standard_bounded_cnn** | 7.50% | 34.17% | 33.33% | 53.33% | **0** |
| **variant_c_hard_negative_ablation** | 49.17% | 48.33% | 65.83% | 98.33% | **0** |
| **variant_d_stage1_cnn** | 90.00% | 60.83% | 87.50% | 100.00% | **0** |
