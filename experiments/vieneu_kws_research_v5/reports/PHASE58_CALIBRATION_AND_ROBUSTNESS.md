# Phase 58 — Validation Calibration and Robustness Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Calibration Configs:** `experiments/vieneu_kws_research_v5/calibration/selected_configs.json`  
**Validation Summary JSON:** `experiments/vieneu_kws_research_v5/predictions/validation_summary.json`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary and Formal Validation Decision

Phase 58 performed validation-only threshold calibration and controlled acoustic robustness profiling across all four candidate architectures on the newly generated validation dataset v5 (360 clips: 160 commands across 4 validation voices, 120 `UNKNOWN` distractors across 40 disjoint texts, 80 `SILENCE` clips).

### Predeclared Proposed Acceptance Criteria (Documented Prior to Test Inspection):
1. **Target Recall:** $\ge 95.0\%$ for EACH individual target keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`), assessed per seed and in the aggregate.
2. **False Acceptance Rate:** $\le 5.0\%$ on non-target clips ($\text{FAR} \le 0.05$).
3. **Decision Latency:** $\le 50.0\,\text{ms}$ per 1-second clip on CPU.

### Formal Validation Verdict:
$$\mathbf{NO\_FEASIBLE\_VALIDATION\_OPERATING\_POINT}$$

> [!WARNING]
> **Validation Target Assessment:**  
> Across all four variants and all three random seeds (42, 43, 44), **zero configurations** simultaneously satisfied both $\text{Recall}_k \ge 95.0\%$ for all four keywords AND $\text{FAR} \le 5.0\%$.  
> - When thresholds were tightened to achieve low false acceptance ($\text{FAR} \le 5\%$), command recalls settled between $86.9\%$ and $93.8\%$.  
> - When thresholds were lowered to achieve $\ge 95\%$ recall on difficult keywords, false acceptance rose to $9\%\text{--}15\%$.  
> In strict accordance with research governance, the criteria were **not relaxed post-hoc**. The formal verdict of `NO_FEASIBLE_VALIDATION_OPERATING_POINT` is declared.

---

## 2. Validation Calibration Results Across All Variants

Operating points selected using the predeclared validation selection objective:
$$\text{Maximize } \mathcal{S} = \text{Macro\_Recall} - 2.5 \times \text{FAR}$$

| Variant | Seed | Calibrated Parameters | Validation Macro Recall | Validation FAR | Per-Keyword Recall (`TRAI`, `PHAI`, `LEN`, `XUONG`) | Strict Criteria Met? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Variant A: Cascade v1** | 42 | $\theta_{s1}=0.30, \theta_{s2}=0.55$ | 87.50% | **4.00%** | `TRAI`: 90.0%, `PHAI`: 90.0%, `LEN`: 82.5%, `XUONG`: 87.5% | No (Recall < 95%) |
| | 43 | $\theta_{s1}=0.30, \theta_{s2}=0.20$ | 86.88% | **3.50%** | `TRAI`: 97.5%, `PHAI`: 82.5%, `LEN`: 82.5%, `XUONG`: 85.0% | No (Recall < 95%) |
| | 44 | $\theta_{s1}=0.30, \theta_{s2}=0.20$ | 93.75% | **3.00%** | `TRAI`: 97.5%, `PHAI`: 95.0%, `LEN`: 90.0%, `XUONG`: 92.5% | No (Recall < 95%) |
| **Variant A Mean** | — | — | **89.38%** | **3.50%** | **`TRAI`: 95.0%, `PHAI`: 89.2%, `LEN`: 85.0%, `XUONG`: 88.3%** | **Selection Score: 0.8063** |
| **Variant B: Stage 1 CNN** | 42 | $\theta_{\text{cmd}}=0.85$ | 85.62% | 9.00% | `TRAI`: 82.5%, `PHAI`: 77.5%, `LEN`: 87.5%, `XUONG`: 95.0% | No (FAR > 5%) |
| | 43 | $\theta_{\text{cmd}}=0.80$ | 75.62% | 11.50% | `TRAI`: 77.5%, `PHAI`: 60.0%, `LEN`: 77.5%, `XUONG`: 87.5% | No (FAR > 5%) |
| | 44 | $\theta_{\text{cmd}}=0.75$ | 95.00% | 13.00% | `TRAI`: 95.0%, `PHAI`: 92.5%, `LEN`: 95.0%, `XUONG`: 97.5% | No (FAR > 5%) |
| **Variant B Mean** | — | — | **85.41%** | **11.17%** | **`TRAI`: 85.0%, `PHAI`: 76.7%, `LEN`: 86.7%, `XUONG`: 93.3%** | **Selection Score: 0.5748** |
| **Variant C: BoundedKWSNet**| 42 | $\theta_{\text{rej}}=0.50$ | 85.00% | 10.50% | `TRAI`: 75.0%, `PHAI`: 80.0%, `LEN`: 85.0%, `XUONG`: 100.0% | No (FAR > 5%) |
| | 43 | $\theta_{\text{rej}}=0.30$ | 30.00% | **0.50%** | `TRAI`: 0.0%, `PHAI`: 7.5%, `LEN`: 12.5%, `XUONG`: 100.0% | No (Recall < 95%) |
| | 44 | $\theta_{\text{rej}}=0.40$ | 72.50% | 6.00% | `TRAI`: 12.5%, `PHAI`: 80.0%, `LEN`: 97.5%, `XUONG`: 100.0% | No (FAR > 5%) |
| **Variant C Mean** | — | — | **62.50%** | **5.67%** | **`TRAI`: 29.2%, `PHAI`: 55.8%, `LEN`: 65.0%, `XUONG`: 100.0%** | **Selection Score: 0.4833** |
| **Variant D: Weighted Abl** | 42 | $\theta_{\text{rej}}=0.30$ | 65.62% | 9.50% | `TRAI`: 65.0%, `PHAI`: 37.5%, `LEN`: 87.5%, `XUONG`: 72.5% | No (FAR > 5%) |
| | 43 | $\theta_{\text{rej}}=0.40$ | 60.62% | 7.00% | `TRAI`: 0.0%, `PHAI`: 52.5%, `LEN`: 90.0%, `XUONG`: 100.0% | No (FAR > 5%) |
| | 44 | $\theta_{\text{rej}}=0.40$ | 76.88% | 14.50% | `TRAI`: 22.5%, `PHAI`: 87.5%, `LEN`: 97.5%, `XUONG`: 100.0% | No (FAR > 5%) |
| **Variant D Mean** | — | — | **67.71%** | **10.33%** | **`TRAI`: 29.2%, `PHAI`: 59.2%, `LEN`: 91.7%, `XUONG`: 90.8%** | **Selection Score: 0.4188** |

---

## 3. Acoustic Robustness Stress Testing (Validation Split)

Using the calibrated operating points, models were tested under synthetic environmental degradation:
1. **Additive Noise at SNR = 20 dB:**
   - Variant A (Cascade v1): Recall = **84.38%**, FAR = **3.50%** (Robust preservation of low FAR).
   - Variant B (Stage 1 CNN): Recall = **77.50%**, FAR = **8.50%**.
   - Variant C: Recall = **58.75%**, FAR = **6.00%**.
2. **Additive Noise at SNR = 15 dB:**
   - Variant A (Cascade v1): Recall = **78.12%**, FAR = **4.00%** (Maintains FAR $\le 5\%$ even at 15 dB SNR).
   - Variant B (Stage 1 CNN): Recall = **70.62%**, FAR = **9.00%**.
3. **Gain Attenuation (-3 dB):**
   - Variant A (Cascade v1): Recall = **88.12%**, FAR = **3.50%**.
   - Variant B (Stage 1 CNN): Recall = **83.12%**, FAR = **10.50%**.

---

## 4. Finalist Selection for Locked Test Evaluation

Per the governance requirement to select no more than two finalists based strictly on written validation criteria:
- **Finalist 1:** **Variant A (Cascade v1 Bounded Pipeline)**  
  *Selection Rationale:* Highest validation selection score ($\mathcal{S} = 0.8063$), highest macro recall ($89.38\%$), lowest false acceptance rate ($3.50\%$, beating the $5\%$ ceiling), and exceptional noise stability.
- **Finalist 2:** **Variant B (Stage 1 CNN alone)**  
  *Selection Rationale:* Second highest validation selection score ($\mathcal{S} = 0.5748$), lowest complexity (138k parameters), and fastest latency (1.41 ms on CPU).

Both finalists and their calibrated operating points are frozen in `calibration/selected_configs.json` for Phase 59 evaluation.
