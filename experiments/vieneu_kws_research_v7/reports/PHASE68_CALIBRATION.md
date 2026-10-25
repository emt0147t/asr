# Phase 68 — Validation Calibration and Robustness Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Selected Configurations:** `experiments/vieneu_kws_research_v7/calibration/selected_configs.json`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 68 conducted validation-only confidence threshold calibration and synthetic robustness benchmarking across three candidate systems:
1. **Historical Cascade v1:** Frozen reference baseline (`VietnameseKWSNet` + `TemporalKWSVerifier`).
2. **Candidate A (Corrected Baseline):** 5-class `CorrectedVietnameseKWSNet` with inverse class weighting ($w_{\text{cmd}} = 2.0, w_{\text{REJECT}} = 0.5$).
3. **Candidate B (Moderate Hard-Negative):** 5-class `CorrectedVietnameseKWSNet` with moderate hard-negative weighting ($w_{\text{hn}} = 1.2, w_{\text{cmd}} = 2.0$).

### Predeclared Acceptance Criteria:
- **Per-Keyword Target Recall:** $\ge 95.0\%$ for EACH keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`).
- **Non-Target Clip FAR:** $\le 5.0\%$.
- **CPU Latency Ceiling:** $\le 50\,\text{ms}$ post-capture decision latency.
- **Validation Operating Point Determination:** `FEASIBLE_VALIDATION_OPERATING_POINT`.

---

## 2. Validation Calibration Performance Across Seeds

| Candidate Architecture | Seed | Calibrated Operating Parameters | Val Command Recall | Val Non-Target FAR | TRAI | PHAI | LEN | XUONG | Strict Pass? | Operating Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | `theta_s1=0.35, theta_s2=0.7` | 79.37% | 4.50% | 90.0% | 77.5% | 75.0% | 75.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_a_cascade_v1** | 43 | `theta_s1=0.35, theta_s2=0.8` | 70.63% | 5.00% | 82.5% | 62.5% | 72.5% | 65.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_a_cascade_v1** | 44 | `theta_s1=0.35, theta_s2=0.55` | 87.50% | 5.00% | 95.0% | 85.0% | 85.0% | 85.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_a_corrected_baseline** | 42 | `theta_cmd=0.65` | 77.50% | 4.00% | 72.5% | 47.5% | 100.0% | 90.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_a_corrected_baseline** | 43 | `theta_cmd=0.7` | 94.37% | 5.00% | 85.0% | 100.0% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_a_corrected_baseline** | 44 | `theta_cmd=0.5` | 94.37% | 3.50% | 95.0% | 87.5% | 100.0% | 95.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_b_moderately_weighted_hard_negative** | 42 | `theta_cmd=0.5` | 96.25% | 3.00% | 95.0% | 95.0% | 100.0% | 95.0% | YES | `FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_b_moderately_weighted_hard_negative** | 43 | `theta_cmd=0.35` | 96.88% | 5.00% | 97.5% | 97.5% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_b_moderately_weighted_hard_negative** | 44 | `theta_cmd=0.35` | 95.63% | 4.00% | 95.0% | 97.5% | 100.0% | 90.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |

---

## 3. Synthetic Noise and Gain Robustness (Validation Split)

| Candidate Variant | Seed | Clean (Recall / FAR) | SNR 20dB (Recall / FAR) | SNR 15dB (Recall / FAR) | SNR 10dB (Recall / FAR) | Gain +3dB (Recall / FAR) | Gain -3dB (Recall / FAR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **variant_a_cascade_v1** | 42 | 79.4% / 4.5% | 80.0% / 4.5% | 75.6% / 4.0% | 71.9% / 4.0% | 76.9% / 4.0% | 86.2% / 5.0% |
| **variant_a_cascade_v1** | 43 | 70.6% / 5.0% | 76.9% / 7.0% | 78.8% / 6.0% | 71.9% / 3.5% | 68.1% / 4.5% | 81.2% / 7.5% |
| **variant_a_cascade_v1** | 44 | 87.5% / 5.0% | 90.0% / 8.0% | 88.8% / 9.0% | 81.9% / 10.5% | 87.5% / 5.0% | 88.1% / 6.0% |
| **candidate_a_corrected_baseline** | 42 | 77.5% / 4.0% | 71.2% / 2.5% | 65.0% / 3.5% | 52.5% / 4.0% | 76.2% / 3.0% | 75.6% / 9.5% |
| **candidate_a_corrected_baseline** | 43 | 94.4% / 5.0% | 91.9% / 4.0% | 88.8% / 4.5% | 69.4% / 8.0% | 93.8% / 4.0% | 95.0% / 8.0% |
| **candidate_a_corrected_baseline** | 44 | 94.4% / 3.5% | 92.5% / 8.0% | 89.4% / 9.0% | 80.6% / 9.5% | 93.1% / 2.5% | 96.9% / 8.5% |
| **candidate_b_moderately_weighted_hard_negative** | 42 | 96.2% / 3.0% | 93.8% / 3.5% | 90.6% / 5.0% | 80.6% / 3.5% | 93.8% / 2.5% | 96.2% / 7.5% |
| **candidate_b_moderately_weighted_hard_negative** | 43 | 96.9% / 5.0% | 94.4% / 4.5% | 92.5% / 8.0% | 90.6% / 11.5% | 96.2% / 4.0% | 98.1% / 10.0% |
| **candidate_b_moderately_weighted_hard_negative** | 44 | 95.6% / 4.0% | 93.8% / 3.0% | 91.2% / 6.0% | 84.4% / 6.0% | 95.0% / 3.0% | 95.6% / 6.5% |

---

## 4. Finalist Selection for Locked Test Evaluation (Phase 69)

### Written Selection Rule:
1. Any candidate achieving $\ge 95.0\%$ recall across all 4 keywords and $\text{FAR} \le 5.0\%$ automatically qualifies.
2. Candidates are ranked by composite score $S = 0.6 \cdot \text{Recall} + 0.4 \cdot (1 - \text{FAR})$.
3. The top 2 ranking candidates advance as finalists to Phase 69 locked test evaluation.

### Selected Finalists:
1. **Finalist 1:** `candidate_b_moderately_weighted_hard_negative`
2. **Finalist 2:** `candidate_a_corrected_baseline`

All parameters and configurations are frozen into `calibration/selected_configs.json` and proceed to Phase 69.
