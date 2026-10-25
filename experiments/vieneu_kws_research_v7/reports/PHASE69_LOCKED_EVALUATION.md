# Phase 69 — Locked Test Benchmark Evaluation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Test Partition:** `experiments/vieneu_kws_research_v6/data/splits/test_manifest.csv` ($N = 720$ clips)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Voice Provenance Audit

Phase 69 evaluated the three bounded-utterance candidate architectures on the locked 720-clip synthetic benchmark partition.

### Speaker Voice Provenance Audit:
- The 4 speaker voices evaluated (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`) are strictly disjoint from the 17 training voices and 4 validation voices.
- **Explicit Disclosure:** These 4 held-out voices are from the approved 25-voice VieNeu-TTS inventory and were previously encountered in earlier benchmark splits (Phases 56–64). Therefore, this evaluation establishes generalization to **unseen, brand-new text prompts on a previously encountered voice inventory**, and must not be claimed as generalization to arbitrary unencountered human speakers.

---

## 2. Cross-Seed Locked Test Performance Matrix

| Candidate Architecture | Seed | Macro Command Recall | 95% Wilson CI (Recall) | Non-Target FAR | 95% Wilson CI (FAR) | 5-Class Accuracy | Mean CPU Latency (Batch 1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | 89.38% | [83.64%, 93.26%] | 4.46% | [3.04%, 6.51%] | 94.17% | 3.15 ms |
| **variant_a_cascade_v1** | 43 | 80.00% | [73.13%, 85.46%] | 4.82% | [3.33%, 6.92%] | 91.81% | 3.95 ms |
| **variant_a_cascade_v1** | 44 | 86.87% | [80.77%, 91.25%] | 8.21% | [6.21%, 10.78%] | 90.69% | 4.94 ms |
| **candidate_a_corrected_baseline** | 42 | 90.00% | [84.37%, 93.75%] | 5.71% | [4.08%, 7.96%] | 93.33% | 1.93 ms |
| **candidate_a_corrected_baseline** | 43 | 99.38% | [96.55%, 99.89%] | 8.75% | [6.68%, 11.38%] | 93.06% | 1.67 ms |
| **candidate_a_corrected_baseline** | 44 | 99.38% | [96.55%, 99.89%] | 7.14% | [5.29%, 9.58%] | 94.31% | 2.04 ms |
| **candidate_b_moderately_weighted_hard_negative** | 42 | 98.12% | [94.63%, 99.36%] | 5.18% | [3.63%, 7.34%] | 95.56% | 1.66 ms |
| **candidate_b_moderately_weighted_hard_negative** | 43 | 99.38% | [96.55%, 99.89%] | 6.96% | [5.14%, 9.38%] | 94.44% | 1.62 ms |
| **candidate_b_moderately_weighted_hard_negative** | 44 | 100.00% | [97.66%, 100.00%] | 5.54% | [3.93%, 7.75%] | 95.69% | 1.68 ms |

---

## 3. Statistical Aggregate Summary (Mean $\pm$ Sample Std Dev)

| Candidate Variant | Command Recall (Mean $\pm$ Std) | Non-Target FAR (Mean $\pm$ Std) | 5-Class Accuracy (Mean $\pm$ Std) | CPU Latency (Mean $\pm$ Std) | Meets $\ge 95\%$ Recall & $\le 5\%$ FAR? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | **85.42% $\pm$ 4.86%** | **5.83% $\pm$ 2.07%** | 92.22% $\pm$ 1.78% | 4.01 $\pm$ 0.90 ms | NO |
| **candidate_a_corrected_baseline** | **96.25% $\pm$ 5.42%** | **7.20% $\pm$ 1.52%** | 93.57% $\pm$ 0.66% | 1.88 $\pm$ 0.19 ms | NO |
| **candidate_b_moderately_weighted_hard_negative** | **99.17% $\pm$ 0.96%** | **5.89% $\pm$ 0.94%** | 95.23% $\pm$ 0.69% | 1.65 $\pm$ 0.03 ms | NO |
