# PHASE 58: SAMPLE-SIZE RATIONALE AND CLUSTER-AWARE INFERENCE DESIGN

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 58 (Statistical Inference Design & Power Audit)  
**Creation Timestamp:** `2026-10-25T15:15:00+00:00`  
**Evaluation Target:** Confirmatory Piper VAIS-1000 + ESC-50 Fold 3 Corpus  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Default Canonical Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)

---

## 1. The Statistical Unit of Inference: Solving Unit Conflation

Prior phases identified a pervasive methodological risk in acoustic evaluation: treating multiple SNR variants of the same spoken utterance as independent observations.

### Data Dependency Hierarchy in Phase 58:
```
Level 1: Lexical Vocabulary
  |-- 4 Target Commands (trái, phải, lên, xuống)
  |-- 16 Phonetic Confusers
  |-- 10 General OOVs
  |-- 6 Tone Probes

Level 2: Independent Clean Synthesis Instances (N_inst = 120)
  |-- 40 Command Instances (10 independent speaking rates per command: 0.85 to 1.30)
  |-- 40 Phonetic OOV Instances (16 words across multiple rates)
  |-- 25 General OOV Instances (10 words across multiple rates)
  |-- 15 Tone Probe Instances (6 tonal syllables across multiple rates)

Level 3: Derivative Acoustic Clips (N = 480 total: 420 primary + 60 tone probes)
  |-- Each synthesis instance mixed across 4 SNR tiers: clean, 15 dB, 10 dB, 5 dB
  |-- Highly correlated observations sharing identical underlying phonetic timing
```

**Statistical Rule:** In Phase 59, all paired comparisons, hypothesis tests, and confidence intervals are computed at **Level 2 (the synthesis instance cluster, $N_{\text{inst}} = 105$ primary)**, averaging performance across its 4 derivative SNR variants:
$$s_i = \frac{1}{4} \sum_{j=1}^4 \mathbb{I}(\hat{y}_{i,j} = y_i) \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$$

---

## 2. Statistical Power Analysis & Sample-Size Rationale

### 2.1 Power Calculation:
- **Sample Size:** $N_{\text{clusters}} = 105$ independent clean speech waveforms.
- **Hypothesis Test:** Two-sided paired $t$-test ($df = 104$) and paired Wilcoxon signed-rank test.
- **Significance Threshold:** $\alpha = 0.01$ (conservative Bonferroni-corrected significance level for 4 planned comparisons against $v11$).
- **Minimum Detectable Effect Size:**
  Assuming empirical cluster accuracy standard deviation $\sigma \le 18.0\text{ pp}$:
  $$t_{\text{stat}} = \frac{\Delta \sqrt{N}}{\sigma} = \frac{7.5 \times \sqrt{105}}{18.0} = \frac{7.5 \times 10.247}{18.0} = \mathbf{4.27}$$
  At $\alpha = 0.01$ with $df = 104$, critical $t_{0.005} \approx 2.62$.
  $$\text{Statistical Power } (1 - \beta) > \mathbf{99.0\%} \quad (\beta < 0.01)$$

**Conclusion:** The design provides overwhelming statistical power to detect a true performance difference of $\Delta \ge 7.5\text{ pp}$ across independent synthesis instances, without relying on spurious clip-level sample inflation.

---

## 3. Diagnostic Tone-Probe Uncertainty Hierarchy

When models achieve $0$ false alarms across the diagnostic tone probe cohort, the Rule of Three must be reported across all three aggregation levels:

| Level of Aggregation | Independent Units ($N$) | Definition | Exact Rule of Three 95% Upper Bound ($\text{UCB95}$) | Interpretation |
|:---------------------|:-----------------------:|:-----------|:----------------------------------------------------:|:---------------|
| **Level 3: Acoustic Clips** | $N = 60$ | SNR-augmented clips | **4.87%** ($1 - 0.05^{1/60}$) | Assumes clip independence; severely underestimates variance. |
| **Level 2: Synthesis Instances** | $\mathbf{N = 15}$ | **Clean speech waveforms** | **18.10%** ($1 - 0.05^{1/15}$) | **Primary Realistic Bound:** Accounts for correlation across SNRs. |
| **Level 1: Base Tonal Syllables** | $N = 6$ | Distinct tonal lexemes | **39.30%** ($1 - 0.05^{1/6}$) | Ultra-conservative bound across the 6 specific syllables. |

---

## 4. Pre-Registered Candidate Profiles & Pre-Inference Model List

In accordance with Phase 58 instructions, the exact models and evaluation criteria are locked prior to executing confirmatory inference:

### 4.1 Pre-Inference Model List:
1. `Baseline_v11_seed42`: Canonical default model (legacy reference).
2. `Baseline_E1_seed42`: Frozen E1-scale baseline benchmark.
3. `Phase28_D2_H1_Candidate`: Pre-registered Primary Candidate (Low-FA Profile).
4. `Phase26_Candidate_D0_H1_seed44`: Pre-registered Primary Candidate (Balanced Sensitivity Profile).
5. `Candidate_D3_C_seed42`: Explicitly Pre-Registered Developmental Challenger Candidate.

### 4.2 Pre-Registered Acceptance Criteria:

```
Profile 1: Low-FA Profile (Phase 28 D2_H1 Candidate)
  - Silence FA / min <= 1.0 (UCB95 <= 2.0)
  - Conversational OOV FA / min <= 6.0
  - Isolated OOV FAR <= 20.0%
  - Track A 4-Command Macro-F1 >= 50.0%

Profile 2: Balanced Sensitivity Profile (Phase 26 Candidate seed44)
  - Track A 4-Command Macro-F1 >= 60.0%
  - Track B Streaming Event F1 >= 50.0%
  - Silence FA / min <= 3.0
  - Isolated OOV FAR <= 25.0%

Challenger Profile: D3_C Developmental Challenger
  - Track A 4-Command Macro-F1 >= 60.0%
  - Track A Minimum Command Recall >= 40.0%
  - Track B Streaming Event F1 >= 50.0%
  - Silence FA / min <= 2.0
  - Isolated OOV FAR <= 25.0%
```
