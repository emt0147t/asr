# Phase 63 — Validation Calibration and Robustness Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Selected Configurations:** `experiments/vieneu_kws_research_v6/calibration/selected_configs.json`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 63 conducted rigorous, validation-only calibration and synthetic robustness profiling across 4 bounded-utterance candidate architectures on the newly synthesized Benchmark Dataset v6 validation split ($N = 360$ clips).

### Predeclared Acceptance Criteria:
- **Per-Keyword Target Recall:** $\ge 95.0\%$ for EACH keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`).
- **Non-Target False Acceptance Rate (FAR):** $\le 5.0\%$ on non-target distractor and silence clips.
- **CPU Latency Ceiling:** $\le 50\,\text{ms}$ post-capture decision latency.
- **Validation Operating Point Determination:** NO_FEASIBLE_VALIDATION_OPERATING_POINT.

---

## 2. Validation Calibration Performance Summary Across Seeds

| Candidate Variant | Seed | Calibrated Operating Parameters | Val Command Recall | Val Non-Target FAR | TRAI | PHAI | LEN | XUONG | Strict Pass? | Operating Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | `theta_s1=0.35, theta_s2=0.7` | 79.37% | 4.50% | 90.0% | 77.5% | 75.0% | 75.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_a_cascade_v1** | 43 | `theta_s1=0.35, theta_s2=0.8` | 70.63% | 5.00% | 82.5% | 62.5% | 72.5% | 65.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_a_cascade_v1** | 44 | `theta_s1=0.35, theta_s2=0.55` | 87.50% | 5.00% | 95.0% | 85.0% | 85.0% | 85.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_b_standard_bounded_cnn** | 42 | `theta_cmd=0.35` | 11.87% | 0.00% | 0.0% | 0.0% | 0.0% | 47.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_b_standard_bounded_cnn** | 43 | `theta_cmd=0.35` | 3.12% | 0.00% | 0.0% | 12.5% | 0.0% | 0.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_b_standard_bounded_cnn** | 44 | `theta_cmd=0.4` | 70.63% | 5.00% | 22.5% | 85.0% | 92.5% | 82.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_c_hard_negative_ablation** | 42 | `theta_cmd=0.35` | 43.13% | 2.50% | 50.0% | 40.0% | 5.0% | 77.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_c_hard_negative_ablation** | 43 | `theta_cmd=0.35` | 81.87% | 5.00% | 52.5% | 85.0% | 100.0% | 90.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_c_hard_negative_ablation** | 44 | `theta_cmd=0.6` | 39.37% | 3.50% | 0.0% | 2.5% | 62.5% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_d_stage1_cnn** | 42 | `theta_cmd=0.9` | 71.88% | 7.50% | 70.0% | 50.0% | 77.5% | 90.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_d_stage1_cnn** | 43 | `theta_cmd=0.9` | 55.63% | 8.00% | 52.5% | 40.0% | 57.5% | 72.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_d_stage1_cnn** | 44 | `theta_cmd=0.9` | 70.63% | 11.50% | 80.0% | 42.5% | 77.5% | 82.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |

---

## 3. Synthetic Noise & Gain Robustness Breakdown (Validation Split)

| Candidate Variant | Seed | Clean (Recall / FAR) | SNR 20dB (Recall / FAR) | SNR 15dB (Recall / FAR) | SNR 10dB (Recall / FAR) | Gain +3dB (Recall / FAR) | Gain -3dB (Recall / FAR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | 79.4% / 4.5% | 80.0% / 4.5% | 75.6% / 4.0% | 71.9% / 4.0% | 76.9% / 4.0% | 86.2% / 5.0% |
| **variant_a_cascade_v1** | 43 | 70.6% / 5.0% | 76.9% / 7.0% | 78.8% / 6.0% | 71.9% / 3.5% | 68.1% / 4.5% | 81.2% / 7.5% |
| **variant_a_cascade_v1** | 44 | 87.5% / 5.0% | 90.0% / 8.0% | 88.8% / 9.0% | 81.9% / 10.5% | 87.5% / 5.0% | 88.1% / 6.0% |
| **variant_b_standard_bounded_cnn** | 42 | 11.9% / 0.0% | 15.0% / 0.0% | 16.2% / 0.0% | 15.0% / 0.0% | 11.2% / 0.0% | 15.6% / 0.5% |
| **variant_b_standard_bounded_cnn** | 43 | 3.1% / 0.0% | 1.2% / 0.0% | 11.2% / 0.0% | 6.2% / 0.0% | 3.8% / 0.0% | 1.2% / 0.0% |
| **variant_b_standard_bounded_cnn** | 44 | 70.6% / 5.0% | 63.7% / 10.0% | 56.2% / 11.5% | 56.2% / 10.5% | 68.8% / 4.5% | 70.0% / 10.5% |
| **variant_c_hard_negative_ablation** | 42 | 43.1% / 2.5% | 50.6% / 6.5% | 51.2% / 7.0% | 33.8% / 5.5% | 41.9% / 2.0% | 46.2% / 2.5% |
| **variant_c_hard_negative_ablation** | 43 | 81.9% / 5.0% | 73.1% / 8.0% | 72.5% / 11.5% | 62.5% / 13.0% | 81.2% / 5.5% | 76.2% / 7.5% |
| **variant_c_hard_negative_ablation** | 44 | 39.4% / 3.5% | 41.9% / 2.5% | 38.1% / 1.5% | 11.9% / 1.5% | 40.6% / 3.5% | 35.0% / 4.0% |
| **variant_d_stage1_cnn** | 42 | 71.9% / 7.5% | 55.0% / 7.5% | 39.4% / 5.5% | 18.1% / 2.0% | 72.5% / 7.5% | 71.9% / 8.0% |
| **variant_d_stage1_cnn** | 43 | 55.6% / 8.0% | 30.6% / 4.0% | 11.9% / 0.0% | 2.5% / 0.0% | 54.4% / 7.0% | 51.2% / 8.0% |
| **variant_d_stage1_cnn** | 44 | 70.6% / 11.5% | 60.6% / 8.0% | 36.2% / 5.5% | 13.8% / 0.0% | 70.6% / 11.5% | 72.5% / 11.5% |

---

## 4. Finalist Selection for Locked Test Evaluation (Phase 64)

### Selection Rule Formulation:
1. Any candidate satisfying all four keyword recall targets ($\ge 95\%$) and $\text{FAR} \le 5.0\%$ automatically qualifies.
2. Candidates are ranked by composite score $S = 0.6 \cdot \text{Recall} + 0.4 \cdot (1 - \text{FAR})$.
3. The top 2 ranking candidates advance as finalists to Phase 64 locked test evaluation.

### Selected Finalists:
1. **Finalist 1:** `variant_a_cascade_v1`
2. **Finalist 2:** `variant_d_stage1_cnn`

These two configurations are locked into `calibration/selected_configs.json` and proceed directly to Phase 64.
