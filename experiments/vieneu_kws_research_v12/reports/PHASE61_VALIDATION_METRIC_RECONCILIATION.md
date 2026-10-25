# PHASE 61: VALIDATION METRIC RECONCILIATION AND PARETO FRONTIER AUDIT

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Audit Target:** Phase 55 & Phase 56 Validation Ledgers on `validation_v1.csv` ($N = 1,280$)  
**Status:** **AUDITED, RECONCILED, AND FORMALLY RE-ANALYZED**  

---

## 1. EXECUTIVE SUMMARY & RECONCILIATION FINDINGS

This audit independently re-evaluated the raw prediction outputs on `validation_v1.csv` ($N=1,280$) to resolve all metric discrepancies between Phase 55 and Phase 56 reports, standardize Macro-F1 calculation definitions, and recompute the multi-objective Pareto frontier.

### Key Forensic Findings:
1. **Validation Raw Predictions Confirmed:**
   - Every candidate's predictions on `validation_v1.csv` match `phase55_validation_comparison.json` and `PHASE56_PARETO_TABLE.csv` exactly.
   - For Baseline $v11$, true validation accuracy is **$78.05\%$** ($999/1280$), 4-command Macro-F1 is **$79.45\%$**, and 5-class Macro-F1 is **$78.86\%$**.
   - For Baseline $E1$, true validation accuracy is **$91.95\%$** ($1177/1280$), 4-command Macro-F1 is **$92.22\%$**, and 5-class Macro-F1 is **$92.14\%$**.
2. **Resolution of Transposed Table Numbers:**
   - In `PHASE56_59_FINAL_REPORT.md` (Section 2.1), the table listed $88.59\%$ accuracy and $88.63\%$ Macro-F1 for $v11$, and $91.64\%$ accuracy for $E1$.
   - **Root Cause Identified:** $88.59\%$ was Candidate $D_3\_C$'s `reject_recall` ($567/640 = 0.8859375$), which was erroneously transposed into the summary markdown table. The value $91.64\%$ was drawn from historical Phase 25 `D1_H0` runs rather than the canonical $E1$ checkpoint.
   - `phase55_validation_comparison.json` and `PHASE56_PARETO_TABLE.csv` were always correct; the discrepancy was isolated to the narrative summary text.
3. **Macro-F1 Calculation Standard:**
   - **4-Command Macro-F1:** $\frac{1}{4} (F_{1,\text{TRAI}} + F_{1,\text{PHAI}} + F_{1,\text{LEN}} + F_{1,\text{XUONG}})$. REJECT is excluded from the numerator/denominator of the average, but REJECT confusions are fully accounted for in the per-class precision and recall of each command.
   - **5-Class Macro-F1:** $\frac{1}{5} \sum_{c=1}^5 F_{1,c}$, incorporating REJECT as an explicit fifth class.
4. **Pareto Dominance Clarification:**
   - Under **4 core objectives** (Macro-F1, Accuracy, Min Command Recall, $-1 \times$ OOV FAR), $D_2\_H1$ **strictly Pareto-dominates** $D_3\_C$, Phase 26, $D_3\_A$, $D_3\_B$, $E1$, and $v11$.
   - Under **5 objectives** including `reject_precision`, $D_2\_H1$ does **NOT** strictly dominate $D_3\_C$ or $D_3\_A$, because $D_3\_C$ ($97.76\%$) and $D_3\_A$ ($98.77\%$) achieve higher REJECT precision than $D_2\_H1$ ($97.48\%$). Both $D_2\_H1$, $D_3\_A$, and $D_3\_C$ reside on the 5-objective Pareto frontier.

---

## 2. COMPREHENSIVE VALIDATION RECONCILIATION TABLE

All metrics re-verified on `validation_v1.csv` ($N=1,280$: 160 TRAI, 160 PHAI, 160 LEN, 160 XUONG, 640 REJECT):

| Candidate | Val Accuracy | 4-Cmd Macro-F1 | 5-Class Macro-F1 | Min Cmd Recall | OOV FAR | REJECT Precision | REJECT Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 28 $D_2\_H1$** | **93.91%** | **93.88%** | **93.91%** | **95.62%** | **9.22%** | 97.48% | **90.78%** |
| **Phase 26 $D_0\_H1$** | 93.12% | 93.22% | 93.19% | 95.00% | 10.78% | 97.27% | 89.22% |
| **Candidate $D_3\_C$** | 93.05% | 93.22% | 93.17% | 95.00% | 11.41% | 97.76% | 88.59% |
| **Candidate $D_3\_A$** | 92.89% | 92.96% | 92.96% | **95.62%** | 12.19% | **98.77%** | 87.81% |
| **Candidate $D_3\_B$** | 90.94% | 91.43% | 91.25% | 91.88% | 15.00% | 96.80% | 85.00% |
| **Baseline E1** | 91.95% | 92.22% | 92.14% | **95.62%** | 13.91% | 98.39% | 86.09% |
| **Baseline v11 (default)** | 78.05% | 79.45% | 78.86% | 78.75% | 34.22% | 91.32% | 65.78% |

---

## 3. CLASS-WISE METRIC DISAGGREGATION

### Per-Class F1, Precision, and Recall:

#### Phase 28 $D_2\_H1$ (seed 42):
- **TRAI:** Precision = $95.03\%$, Recall = $95.62\%$, F1 = $95.33\%$ ($153/160$)
- **PHAI:** Precision = $82.35\%$, Recall = $96.25\%$, F1 = $88.76\%$ ($154/160$)
- **LEN:** Precision = $92.77\%$, Recall = $96.25\%$, F1 = $94.48\%$ ($154/160$)
- **XUONG:** Precision = $94.12\%$, Recall = $100.00\%$, F1 = $96.97\%$ ($160/160$)
- **REJECT:** Precision = $97.48\%$, Recall = $90.78\%$, F1 = $94.01\%$ ($581/640$)

#### Candidate $D_3\_C$ (seed 42):
- **TRAI:** Precision = $95.60\%$, Recall = $95.00\%$, F1 = $95.30\%$ ($152/160$)
- **PHAI:** Precision = $80.83\%$, Recall = $97.50\%$, F1 = $88.39\%$ ($156/160$)
- **LEN:** Precision = $90.17\%$, Recall = $97.50\%$, F1 = $93.69\%$ ($156/160$)
- **XUONG:** Precision = $91.43\%$, Recall = $100.00\%$, F1 = $95.52\%$ ($160/160$)
- **REJECT:** Precision = $97.76\%$, Recall = $88.59\%$, F1 = $92.95\%$ ($567/640$)

#### Baseline $v11$ (seed 42):
- **TRAI:** Precision = $86.30\%$, Recall = $78.75\%$, F1 = $82.35\%$ ($126/160$)
- **PHAI:** Precision = $65.53\%$, Recall = $96.25\%$, F1 = $77.97\%$ ($154/160$)
- **LEN:** Precision = $63.98\%$, Recall = $94.38\%$, F1 = $76.26\%$ ($151/160$)
- **XUONG:** Precision = $72.77\%$, Recall = $91.88\%$, F1 = $81.22\%$ ($147/160$)
- **REJECT:** Precision = $91.32\%$, Recall = $65.78\%$, F1 = $76.48\%$ ($421/640$)

---

## 4. MULTI-OBJECTIVE PARETO FRONTIER RE-ANALYSIS

### 4.1 Four-Objective Core Formulation
Objectives to optimize simultaneously:
$$\mathbf{y}_{\text{core}} = \left(\max \text{Macro-F1}_{\text{4cmd}}, \max \text{Accuracy}, \max \text{Min Command Recall}, \min \text{OOV FAR}\right)$$

| Candidate | Status | Dominated By |
| :--- | :--- | :--- |
| **Phase 28 $D_2\_H1$** | **Pareto-Efficient (Optimal)** | **None (Dominates all other models)** |
| **Phase 26 $D_0\_H1$** | Subordinated | Phase 28 $D_2\_H1$ |
| **Candidate $D_3\_C$** | Subordinated | Phase 28 $D_2\_H1$ |
| **Candidate $D_3\_A$** | Subordinated | Phase 28 $D_2\_H1$ |
| **Candidate $D_3\_B$** | Subordinated | Baseline $E1$, Phase 28 $D_2\_H1$, Phase 26, $D_3\_A$, $D_3\_C$ |
| **Baseline E1** | Subordinated | Phase 28 $D_2\_H1$, $D_3\_A$ |
| **Baseline v11** | Subordinated | All modern models |

*Result:* Under the 4 core operating objectives, Phase 28 $D_2\_H1$ strictly dominates $D_3\_C$ on every dimension.

### 4.2 Five-Objective Formulation (Including REJECT Precision)
Objectives:
$$\mathbf{y}_{\text{5obj}} = \left(\max \text{Macro-F1}_{\text{4cmd}}, \max \text{Accuracy}, \max \text{Min Command Recall}, \min \text{OOV FAR}, \max \text{REJECT Precision}\right)$$

| Candidate | Status | Dominated By |
| :--- | :--- | :--- |
| **Phase 28 $D_2\_H1$** | **Pareto-Efficient** | **None** |
| **Candidate $D_3\_A$** | **Pareto-Efficient** | **None** (highest REJECT precision: $98.77\%$) |
| **Candidate $D_3\_C$** | **Pareto-Efficient** | **None** ($97.76\% > 97.48\%$ REJECT precision) |
| **Phase 26 $D_0\_H1$** | Subordinated | Phase 28 $D_2\_H1$ |
| **Baseline E1** | Subordinated | Candidate $D_3\_A$ |
| **Candidate $D_3\_B$** | Subordinated | Baseline $E1$, Phase 28 $D_2\_H1$, Phase 26, $D_3\_A$, $D_3\_C$ |
| **Baseline v11** | Subordinated | All modern models |

### 4.3 Why $D_3\_C$ Achieves Slightly Higher REJECT Precision:
- REJECT Precision is $\frac{\text{True Rejections}}{\text{True Rejections} + \text{False Rejections}}$.
- False rejections occur when actual command utterances are misclassified as REJECT.
- In $D_3\_C$, commands were predicted fewer times overall (it missed 8 TRAI, 4 PHAI, 4 LEN = 16 misses), but because its rejection threshold behavior was slightly more selective on certain boundary clips, its false rejection denominator was slightly lower, yielding $97.76\%$ precision vs $97.48\%$ for $D_2\_H1$.
- However, $D_2\_H1$ achieved higher REJECT recall ($90.78\% > 88.59\%$), lower OOV false alarms ($9.22\% < 11.41\%$), higher accuracy ($93.91\% > 93.05\%$), and higher Macro-F1 ($93.88\% > 93.22\%$).

---

## 5. AUDIT ADDENDUM DECLARATION

This report and [`PHASE61_PARETO_REANALYSIS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE61_PARETO_REANALYSIS.csv) constitute the official reconciliation addendum to Phase 56. All historical raw ledgers remain immutable.
