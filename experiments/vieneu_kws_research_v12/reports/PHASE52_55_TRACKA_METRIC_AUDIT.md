# PHASE 52: TRACK A DENOMINATOR AND CONFUSION-MATRIX FORENSIC AUDIT

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 52 (Track A Forensic Count Reconciliation & Audit)  
**Governing Evidence File:** [`phase52_trackA_reconciliation.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/phase52_trackA_reconciliation.json)  
**Per-Clip Raw Predictions:** [`phase52_per_clip_predictions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/phase52_per_clip_predictions.csv)  
**Audited Manifest:** [`PHASE50_LOCKED_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE50_LOCKED_MANIFEST.csv) (SHA-256: `b644a7e71badae8394b0e693fa45e1fdafb0a346b939c456b5f33e2941c80297`)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Default Canonical Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)  
**Audit Verification Status:** **100% BIT-FOR-BIT MATHEMATICAL RECONCILIATION (ZERO DISCREPANCIES)**

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No model retrained or capacity modified. All checkpoints ($v11$, $E1$, Phase 28 $D_2\_H1$, Phase 26 $D_0\_H1\_seed44$, Phase 28 $D_0\_H1\_seed42$) remain strictly frozen and unaltered.

---

## 1. Executive Summary & Audit Scope

This audit resolves all denominator, row-sum, accuracy, and prevalence questions regarding Track A of the Phase 51 locked confirmatory evaluation. Forward inference was re-executed from first principles on all 240 clips in [`PHASE50_LOCKED_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE50_LOCKED_MANIFEST.csv) using the five frozen model checkpoints under the exact locked protocol ($\theta_{\text{cmd}} = 0.20$, fixed Log-Mel frontend, REJECT-inclusive 5-class argmax).

### Core Audit Verdicts:
1. **Denominator Verification:** Track A contains exactly **240 scored clips**, comprising **208 OOV clips** and **32 target-command clips** (exactly 8 clips per command class across 4 SNR tiers: clean, 15 dB, 10 dB, 5 dB).
2. **Implied Values Confirmed:** The implied target-correct values hypothesized in the Phase 52 prompt were confirmed with zero error:
   - **Baseline $v11$:** Exactly **27 / 32** target clips correct ($84.38\%$)
   - **Baseline $E1$:** Exactly **20 / 32** target clips correct ($62.50\%$)
   - **Phase 28 $D_2\_H1$:** Exactly **11 / 32** target clips correct ($34.38\%$)
   - **Phase 26 $D_0\_H1\_seed44$:** Exactly **19 / 32** target clips correct ($59.38\%$)
   - **Phase 28 $D_0\_H1\_seed42$:** Exactly **21 / 32** target clips correct ($65.63\%$)
3. **Correctly Rejected OOV Count:**
   - **Baseline $v11$:** Exactly **71 / 208** ($34.13\%$)
   - **Baseline $E1$:** Exactly **125 / 208** ($60.10\%$)
   - **Phase 28 $D_2\_H1$:** Exactly **180 / 208** ($86.54\%$)
   - **Phase 26 $D_0\_H1\_seed44$:** Exactly **165 / 208** ($79.33\%$)
   - **Phase 28 $D_0\_H1\_seed42$:** Exactly **157 / 208** ($75.48\%$)
4. **Prevalence Distortion Unveiled:** In an evaluation corpus where **$86.67\%$ of samples belong to the REJECT class** ($208 / 240$), overall accuracy is an inherently deceptive metric. $D_2\_H1$'s reported $79.58\%$ overall accuracy is overwhelmingly driven by rejecting background OOV distractors, masking a collapse in target-command recall down to $34.38\%$.

---

## 2. Track A Manifest Composition & Denominators

The composition of [`PHASE50_LOCKED_MANIFEST.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/PHASE50_LOCKED_MANIFEST.csv) was verified by inspecting all 240 rows:

```
Total Track A Clips: 240 (100.0%)
  |-- Target Commands: 32 clips (13.33% prevalence)
  |     |-- TRAI:   8 clips (2 clean instances x 4 SNR tiers: clean, 15dB, 10dB, 5dB)
  |     |-- PHAI:   8 clips (2 clean instances x 4 SNR tiers)
  |     |-- LEN:    8 clips (2 clean instances x 4 SNR tiers)
  |     |-- XUONG:  8 clips (2 clean instances x 4 SNR tiers)
  |
  |-- OOV Distractors: 208 clips (86.67% prevalence)
        |-- test_oov_phonetic: 128 clips (32 phonetic instances x 4 SNR tiers)
        |-- test_oov_general:   80 clips (20 general OOV instances x 4 SNR tiers)
```

| Class / Category | Ground-Truth Label | Subgroup | Synthesis Instances | SNR Tiers | Total Scored Clips | Class Prevalence (%) |
|:-----------------|:------------------:|:---------|:-------------------:|:---------:|:------------------:|:--------------------:|
| **TRAI** | `TRAI` (Idx 0) | `command` | 2 | 4 | **8** | $3.33\%$ |
| **PHAI** | `PHAI` (Idx 1) | `command` | 2 | 4 | **8** | $3.33\%$ |
| **LEN** | `LEN` (Idx 2) | `command` | 2 | 4 | **8** | $3.33\%$ |
| **XUONG** | `XUONG` (Idx 3) | `command` | 2 | 4 | **8** | $3.33\%$ |
| **Phonetic OOV** | `REJECT` (Idx 4) | `test_oov_phonetic` | 32 | 4 | **128** | $53.33\%$ |
| **General OOV** | `REJECT` (Idx 4) | `test_oov_general` | 20 | 4 | **80** | $33.33\%$ |
| **Total Track A** | — | — | **60** | **4** | **240** | **100.00%** |

---

## 3. Confusion Matrices & Raw Metric Reconciliation

For each of the five models, confusion matrices were computed directly from [`phase52_per_clip_predictions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/phase52_per_clip_predictions.csv). The rows represent True Labels ($0\text{--}4$) and columns represent Predicted Labels ($0\text{--}4$):

### 3.1 Model Confusion Matrices

```
1. Baseline v11 (Default, seed 42):
   True \ Pred   TRAI   PHAI    LEN  XUONG  REJECT | Total | Recall
   TRAI             7      0      0      0       1 |     8 |  87.5%
   PHAI             0      6      0      0       2 |     8 |  75.0%
   LEN              0      0      8      0       0 |     8 | 100.0%
   XUONG            0      0      0      6       2 |     8 |  75.0%
   REJECT          15     24     66     32      71 |   208 |  34.1%

2. Baseline E1 (seed 42):
   True \ Pred   TRAI   PHAI    LEN  XUONG  REJECT | Total | Recall
   TRAI             2      0      0      0       6 |     8 |  25.0%
   PHAI             0      7      0      0       1 |     8 |  87.5%
   LEN              0      0      8      0       0 |     8 | 100.0%
   XUONG            0      0      0      3       5 |     8 |  37.5%
   REJECT           0      1     78      4     125 |   208 |  60.1%

3. Phase 28 D2_H1 Candidate (seed 42):
   True \ Pred   TRAI   PHAI    LEN  XUONG  REJECT | Total | Recall
   TRAI             1      0      0      0       7 |     8 |  12.5%
   PHAI             0      1      0      0       7 |     8 |  12.5%
   LEN              0      0      8      0       0 |     8 | 100.0%
   XUONG            0      0      0      1       7 |     8 |  12.5%
   REJECT           0      0     28      0     180 |   208 |  86.5%

4. Phase 26 Candidate D0_H1 (seed 44):
   True \ Pred   TRAI   PHAI    LEN  XUONG  REJECT | Total | Recall
   TRAI             2      0      0      0       6 |     8 |  25.0%
   PHAI             0      4      0      0       4 |     8 |  50.0%
   LEN              0      0      8      0       0 |     8 | 100.0%
   XUONG            0      0      0      5       3 |     8 |  62.5%
   REJECT           0      0     41      2     165 |   208 |  79.3%

5. Phase 28 D0_H1 Candidate (seed 42, exploratory):
   True \ Pred   TRAI   PHAI    LEN  XUONG  REJECT | Total | Recall
   TRAI             3      0      0      0       5 |     8 |  37.5%
   PHAI             0      7      0      0       1 |     8 |  87.5%
   LEN              0      0      8      0       0 |     8 | 100.0%
   XUONG            0      0      0      3       5 |     8 |  37.5%
   REJECT           1      0     48      2     157 |   208 |  75.5%
```

---

## 4. Derived Metric Verification Master Table

| Metric Definition | Formula | Baseline v11 (Default) | Baseline E1 | Phase 28 D2_H1 Candidate | Phase 26 Candidate (seed44) | Phase 28 D0_H1 Candidate |
|:------------------|:-------:|:----------------------:|:-----------:|:------------------------:|:---------------------------:|:------------------------:|
| **Scored Clips ($N$)** | $N_{\text{total}}$ | 240 | 240 | 240 | 240 | 240 |
| **Target Clips ($N_{\text{cmd}}$)** | Rows 0–3 sum | 32 | 32 | 32 | 32 | 32 |
| **OOV Clips ($N_{\text{oov}}$)** | Row 4 sum | 208 | 208 | 208 | 208 | 208 |
| **Target Correct Count** | $\sum_{i=0}^3 CM_{i,i}$ | **27 / 32** | **20 / 32** | **11 / 32** | **19 / 32** | **21 / 32** |
| **Implied Value Match** | Hypothesized in P52 | **MATCH (True)** | **MATCH (True)** | **MATCH (True)** | **MATCH (True)** | **MATCH (True)** |
| **Target-Only Accuracy** | Correct Cmd / 32 | **84.38%** | **62.50%** | **34.38%** | **59.38%** | **65.62%** |
| **TRAI Recall** | $CM_{0,0} / 8$ | 7 / 8 (87.5%) | 2 / 8 (25.0%) | 1 / 8 (12.5%) | 2 / 8 (25.0%) | 3 / 8 (37.5%) |
| **PHAI Recall** | $CM_{1,1} / 8$ | 6 / 8 (75.0%) | 7 / 8 (87.5%) | 1 / 8 (12.5%) | 4 / 8 (50.0%) | 7 / 8 (87.5%) |
| **LEN Recall** | $CM_{2,2} / 8$ | 8 / 8 (100.0%) | 8 / 8 (100.0%) | 8 / 8 (100.0%) | 8 / 8 (100.0%) | 8 / 8 (100.0%) |
| **XUONG Recall** | $CM_{3,3} / 8$ | 6 / 8 (75.0%) | 3 / 8 (37.5%) | 1 / 8 (12.5%) | 5 / 8 (62.5%) | 3 / 8 (37.5%) |
| **Target Misses (into REJECT)** | $\sum_{i=0}^3 CM_{i,4}$ | 5 / 32 (15.6%) | 12 / 32 (37.5%) | 21 / 32 (65.6%) | 13 / 32 (40.6%) | 11 / 32 (34.4%) |
| **Target Cross-Substitutions** | $\sum_{i \ne j, i,j < 4} CM_{i,j}$ | **0 / 32 (0.0%)** | **0 / 32 (0.0%)** | **0 / 32 (0.0%)** | **0 / 32 (0.0%)** | **0 / 32 (0.0%)** |
| **Correctly Rejected OOV** | $CM_{4,4}$ | **71 / 208 (34.1%)** | **125 / 208 (60.1%)** | **180 / 208 (86.5%)** | **165 / 208 (79.3%)** | **157 / 208 (75.5%)** |
| **OOV False Alarm Count** | $\sum_{j=0}^3 CM_{4,j}$ | **137 / 208** | **83 / 208** | **28 / 208** | **43 / 208** | **51 / 208** |
| **OOV False Alarm Rate** | FA / 208 | **65.87%** | **39.90%** | **13.46%** | **20.67%** | **24.52%** |
| **Total Correct Samples** | $\text{Cmd}_{\text{corr}} + \text{OOV}_{\text{corr}}$ | **98 / 240** | **145 / 240** | **191 / 240** | **184 / 240** | **178 / 240** |
| **Overall Accuracy** | Total Correct / 240 | **40.83%** | **60.42%** | **79.58%** | **76.67%** | **74.17%** |
| **4-Command Macro-F1** | Mean F1 (0–3) | **30.96%** | **46.13%** | **25.76%** | **50.35%** | **53.62%** |
| **5-Class Macro-F1** | Mean F1 (0–4) | **34.77%** | **51.40%** | **38.21%** | **57.38%** | **59.60%** |

---

## 5. Prevalence Distortion: Why Overall Accuracy Alone Is Deceptive

The mathematical breakdown reveals why evaluating a KWS system primarily on overall accuracy in an imbalanced test corpus is dangerous:

### 5.1 The Extreme Class Imbalance
In Track A:
$$P(\text{Class } = \text{REJECT}) = \frac{208}{240} = \mathbf{86.67\%}$$
$$P(\text{Class } \in \{\text{TRAI, PHAI, LEN, XUONG}\}) = \frac{32}{240} = \mathbf{13.33\%}$$

### 5.2 The "Dummy" Constant Reject Classifier Paradox
Consider a hypothetical model that predicts $\text{REJECT}$ on every sample regardless of input:
$$\text{Overall Accuracy}_{\text{dummy}} = \frac{0 + 208}{240} = \mathbf{86.67\%}$$
$$\text{Target-Command Recall}_{\text{dummy}} = \frac{0}{32} = \mathbf{0.00\%}$$

This trivial model would achieve an overall accuracy of **$86.67\%$**—higher than every single model trained in this entire repository!

### 5.3 Anatomy of D2_H1's Accuracy
- $D_2\_H1$ scored **$79.58\%$ overall accuracy** ($191 / 240$).
- Of those $191$ correct classifications:
  - **$180$** came from rejecting OOV distractors ($94.2\%$ of its correct predictions).
  - Only **$11$** came from identifying target commands ($5.8\%$ of its correct predictions).
- In reality, $D_2\_H1$ missed **$21$ out of $32$ target commands** ($65.6\%$ miss rate).
- If one relied solely on overall accuracy ($79.58\%$), one would conclude that $D_2\_H1$ was the superior system. In reality, on single-syllable commands generated by an unseen engine, it acts almost like the dummy reject classifier, shutting down keyword functionality.

### 5.4 The Metric Reconciliation Principle
To prevent deceptive conclusions:
1. **Target-Only Accuracy / Macro-F1** isolates sensitivity to intended keywords.
2. **OOV False Alarm Rate (FAR)** isolates immunity to unintended speech.
3. Both metrics must be reported simultaneously as a Pareto pair $(R_{\text{cmd}}, \text{FAR}_{\text{oov}})$, never collapsed into a single scalar accuracy.

---

## 6. Audit Verdict

All counts, denominators, confusion matrix cells, and accuracy scores reconcile with **100% mathematical precision**. No metric discrepancy or counting bug exists.

**Phase 52 Audit Gate:**
$$\mathbf{PASS\_TRACKA\_DENOMINATOR\_AUDIT\_CONFIRMED}$$
