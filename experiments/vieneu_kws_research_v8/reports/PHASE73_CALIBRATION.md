# Phase 73 — Robust Calibration and Finalist Selection Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Written Selection Rule

Phase 73 compared six candidate systems on the 360-clip validation partition across clean and synthetic robustness conditions:
1. **`cascade_v1`:** Frozen historical reference baseline.
2. **`candidate_b_v7_baseline`:** Historical Candidate B baseline ($w_{\text{hn}} = 1.2$).
3. **`variant_w10`:** Fresh ablation variant ($w_{\text{hn}} = 1.0$).
4. **`variant_w12`:** Fresh ablation variant ($w_{\text{hn}} = 1.2$).
5. **`variant_w14`:** Fresh ablation variant ($w_{\text{hn}} = 1.4$, class mass parity).
6. **`variant_w16`:** Fresh ablation variant ($w_{\text{hn}} = 1.6$, stronger suppression).

### Written Finalist Selection Rule:
1. Prefer candidates where all seeds simultaneously satisfy $\ge 95.0\%$ recall for every keyword and $\le 5.0\%$ non-target FAR on validation.
2. If no candidate satisfies this strict condition across all seeds, rank candidates by composite score $S = 0.6 \cdot \text{Mean Recall} + 0.4 \cdot (1 - \text{Mean FAR})$.
3. Select at most two finalists to advance to locked test benchmarking in Phase 74 while honestly retaining the failure status.

---

## 2. Validation Calibration Performance Across Candidates

| Candidate Architecture | Seed | Operating Threshold | Val Recall | Val FAR | Raw FA (out of 200) | TRAI | PHAI | LEN | XUONG | Strict Pass? | Operating Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cascade_v1** | 42 | `theta_s1=0.35, theta_s2=0.7` | 79.37% | 4.50% | 9/200 | 90.0% | 77.5% | 75.0% | 75.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **cascade_v1** | 43 | `theta_s1=0.35, theta_s2=0.8` | 70.63% | 5.00% | 10/200 | 82.5% | 62.5% | 72.5% | 65.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **cascade_v1** | 44 | `theta_s1=0.35, theta_s2=0.55` | 87.50% | 5.00% | 10/200 | 95.0% | 85.0% | 85.0% | 85.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_b_v7_baseline** | 42 | `theta_cmd=0.5` | 96.25% | 3.00% | 6/200 | 95.0% | 95.0% | 100.0% | 95.0% | YES | `FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_b_v7_baseline** | 43 | `theta_cmd=0.35` | 96.88% | 5.00% | 10/200 | 97.5% | 97.5% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **candidate_b_v7_baseline** | 44 | `theta_cmd=0.35` | 95.63% | 4.00% | 8/200 | 95.0% | 97.5% | 100.0% | 90.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w10** | 42 | `theta_cmd=0.4` | 96.25% | 4.00% | 8/200 | 90.0% | 97.5% | 100.0% | 97.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w10** | 43 | `theta_cmd=0.5` | 97.50% | 4.00% | 8/200 | 95.0% | 100.0% | 100.0% | 95.0% | YES | `FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w10** | 44 | `theta_cmd=0.35` | 96.88% | 5.00% | 10/200 | 95.0% | 97.5% | 100.0% | 95.0% | YES | `FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w12** | 42 | `theta_cmd=0.4` | 96.25% | 5.00% | 10/200 | 92.5% | 97.5% | 100.0% | 95.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w12** | 43 | `theta_cmd=0.4` | 98.75% | 4.50% | 9/200 | 100.0% | 100.0% | 100.0% | 95.0% | YES | `FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w12** | 44 | `theta_cmd=0.45` | 95.00% | 4.50% | 9/200 | 95.0% | 92.5% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w14** | 42 | `theta_cmd=0.35` | 94.37% | 5.00% | 10/200 | 90.0% | 95.0% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w14** | 43 | `theta_cmd=0.45` | 98.12% | 3.00% | 6/200 | 100.0% | 95.0% | 100.0% | 97.5% | YES | `FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w14** | 44 | `theta_cmd=0.35` | 95.63% | 4.50% | 9/200 | 90.0% | 100.0% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w16** | 42 | `theta_cmd=0.4` | 92.50% | 4.00% | 8/200 | 90.0% | 87.5% | 100.0% | 92.5% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w16** | 43 | `theta_cmd=0.45` | 95.63% | 4.00% | 8/200 | 87.5% | 100.0% | 100.0% | 95.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |
| **variant_w16** | 44 | `theta_cmd=0.3` | 95.63% | 3.00% | 6/200 | 95.0% | 92.5% | 100.0% | 95.0% | NO | `NO_FEASIBLE_VALIDATION_OPERATING_POINT` |

---

## 3. Candidate Ranking and Finalist Selection

| Rank | Candidate Architecture | Mean Val Recall | Mean Val FAR | Composite Score $S$ | All Seeds Strict Pass? | Advance to Phase 74? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **variant_w10** | 96.88% | 4.33% | 0.9639 | NO | **YES (Finalist)** |
| 2 | **candidate_b_v7_baseline** | 96.25% | 4.00% | 0.9615 | NO | **YES (Finalist)** |
| 3 | **variant_w12** | 96.67% | 4.67% | 0.9613 | NO | **NO** |
| 4 | **variant_w14** | 96.04% | 4.17% | 0.9596 | NO | **NO** |
| 5 | **variant_w16** | 94.59% | 3.67% | 0.9529 | NO | **NO** |
| 6 | **cascade_v1** | 79.17% | 4.83% | 0.8557 | NO | **NO** |

---

## 4. Frozen Finalist Configurations for Phase 74

- **Finalist 1:** `variant_w10`
- **Finalist 2:** `candidate_b_v7_baseline`

All parameters and model checkpoints are frozen into `calibration/selected_configs.json` and proceed without modification into Phase 74.
