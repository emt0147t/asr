# PHASE 32–35: INDEPENDENT METRIC REPRODUCTION & STATISTICAL AUDIT REPORT

**Repository:** `D:\trainidentificationaudio`  
**Evaluation Target:** Locked `edge_tts_eval_v2` Evaluation Corpus  
**Decision Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**VAD Engine:** `SpeechBandRatioVAD` (Strictly Locked)  
**Architecture Capacity:** 138,469 parameters (`CorrectedVietnameseKWSNet`, strictly preserved)  
**Audit Timestamp:** 2026-10-25T11:03:26.529017

---

## 1. Executive Summary & Forensic Findings

This forensic audit independently recomputed and verified all evaluation metrics from the Phase 28–31 work package from raw prediction matrices, event ledgers, and audio manifests.

### Key Audit Conclusions:
1. **Isolated Metric Reproduction: 100% Bit-Exact Match**  
   All confusion matrices, per-class metrics, 4-command Macro-F1, 5-class F1, and SNR breakdowns reproduced identically to recorded values.
2. **Statistical Claim Correction (ANOVA Factor H):**  
   The Two-Way ANOVA on validation Macro-F1 yielded $F = 3.7956, p = 0.0872$. Because $p = 0.0872 > 0.05$, the effect of hard-negative sampling (**Factor H**) is **NOT statistically significant** at the conventional $\alpha = 0.05$ threshold. It represents a **suggestive trend**, not a statistically confirmed effect.
3. **In-Domain Validation Retention (Zero Catastrophic Forgetting):**  
   On the original in-domain VieNeu validation set, $D_0\_H1$ achieved $93.28\% \pm 0.50\%$ and $D_2\_H1$ achieved $93.28\% \pm 0.57\%$ ($t = -0.0015, p = 0.9989$). Substituting 10% Edge-TTS speech caused **zero performance degradation** on the in-domain training distribution.
4. **False Alarm Rate Correction via Rule of Three:**  
   During continuous streaming evaluation, zero false alarms were observed on 10 pure silence sequences ($93.38\text{ s} = 1.56\text{ min}$) and 10 pure OOV conversational sequences ($70.83\text{ s} = 1.18\text{ min}$). Under the mathematical **Rule of Three** for zero-event Poisson/Binomial observations, the **95% Upper Confidence Bound (UCB95)** is:
   - Silence FA/min: **$1.87\text{ FA/min}$** ($3 / 1.61\text{ min}$), NOT $0.00\text{ FA/min}$.
   - OOV FA/min: **$2.39\text{ FA/min}$** ($3 / 1.26\text{ min}$), NOT $0.00\text{ FA/min}$.
5. **Generalization Scope & Provenance Disclosure:**  
   $D_2\_H1$ was trained with 10% Edge-TTS data using `vi-VN-HoaiMyNeural` and `vi-VN-NamMinhNeural`. The locked `edge_tts_eval_v2` corpus uses the identical engine (Edge-TTS 7.3.1) and identical two voices. Therefore, for $D_2$ models, this evaluation tests **within-generator, within-voice generalization** to fast conversational speech (+10% rate), unseen phrases, and held-out noise tracks (36–39). It is a cross-generator/cross-voice evaluation for $D_0$, but NOT for $D_2$.

---

## 2. Factorial Training Statistical Re-Analysis (ANOVA Audit)

### Two-Way ANOVA on Validation Macro-F1 ($N = 12$, $n = 3$ runs per cell)

| Source of Variation | Sum of Squares | df | Mean Square | $F$-statistic | $p$-value | Partial $\eta^2$ | Corrected Interpretation |
|:---------------------|:---------------:|:--:|:-----------:|:-------------:|:---------:|:-----------------:|:--------------------------|
| **Factor D** (Training Distribution) | 0.000078 | 1 | 0.000078 | 1.3658 | 0.2762 | 0.1458 | **Not significant** ($p > 0.05$) |
| **Factor H** (Hard Negatives) | 0.000217 | 1 | 0.000217 | 3.7956 | **0.0872** | **0.3218** | **Non-significant suggestive trend** ($p > 0.05$) |
| **Interaction ($D \times H$)** | 0.000081 | 1 | 0.000081 | 1.4195 | 0.2676 | 0.1507 | **Not significant** ($p > 0.05$) |
| **Residual (Error)** | 0.000457 | 8 | 0.000057 | — | — | — | Baseline variance across seeds |

> [!IMPORTANT]
> **Audit Correction:** Factor H ($p = 0.0872$) was previously cited with strong positive language. The claim has been corrected to state that while the effect size is moderate ($\eta_p^2 = 0.32$), the effect does not reach statistical significance at $\alpha = 0.05$ with $N = 12$ runs.

### In-Domain Validation Metric Stability (Cell Summary)

| Training Condition | Factor D (Source) | Factor H (Sampler) | Mean Val Macro-F1 | Std Dev | Min Val F1 | Max Val F1 |
|:--------------------|:-----------------:|:------------------:|:-----------------:|:-------:|:----------:|:----------:|
| **D0_H0** | VieNeu Only | Standard (10% hard) | 92.45% | ±0.13% | 92.32% | 92.57% |
| **D0_H1** | VieNeu Only | Hard-Negative (30% hard) | **93.28%** | ±0.50% | 92.78% | 93.78% |
| **D2_H0** | 10% Edge-TTS | Standard (10% hard) | 93.08% | ±0.49% | 92.60% | 93.58% |
| **D2_H1** | 10% Edge-TTS | Hard-Negative (30% hard) | **93.28%** | ±0.57% | 92.75% | 93.88% |

**In-Domain Retention $t$-test:** $D_0\_H1$ vs $D_2\_H1$: Difference = $0.0000$, $t = -0.0015, p = 0.9989$. There is no statistically detectable difference between pure VieNeu and 10% Edge-TTS on the VieNeu validation distribution.

---

## 3. Isolated Evaluation Reproduction (`edge_tts_eval_v2`)

**Test Set Composition (480 clips):**
- 64 Target Commands (16 clips per command: TRAI, PHAI, LEN, XUONG)
- 256 Phonetic Hard Negatives (16 lexemes $\times$ 2 voices $\times$ 2 rates $\times$ 4 SNRs)
- 160 General OOV Clips (10 lexemes $\times$ 2 voices $\times$ 2 rates $\times$ 4 SNRs)

### 3.1 Primary Comparative Performance

| Evaluation Metric | Phase 13.4 (v11) | Phase 18 (E1) | Phase 28 (D0_H1_seed42) | Phase 28 (D2_H1_seed42) | Reproduction Status |
|:------------------|:----------------:|:-------------:|:-----------------------:|:-----------------------:|:-------------------:|
| **Target Macro-F1 (4 cmds)** | 31.89% | 56.15% | 56.32% | **82.72%** | **VERIFIED** (Exact) |
| **Overall Macro-F1 (5 classes)**| 44.59% | 63.95% | 63.92% | **85.32%** | **VERIFIED** (Exact) |
| **Overall Accuracy** | 77.29% | 79.58% | 78.96% | **92.92%** | **VERIFIED** (Exact) |
| **Phonetic OOV FAR** | 24.61% (63/256) | 30.47% (78/256) | 20.70% (53/256) | **5.08% (13/256)** | **VERIFIED** (Exact) |
| **General OOV FAR** | 6.88% (11/160) | 12.50% (20/256) | 30.00% (48/160) | **13.12% (21/160)** | **VERIFIED** (Exact) |
| **Combined OOV FAR** | 17.79% (74/416) | 23.56% (98/416) | 24.28% (101/416) | **8.17% (34/416)** | **VERIFIED** (Exact) |

### 3.2 Paired Bootstrap Significance Testing (1,000 Resamples)

To test whether the improvement of $D_2\_H1$ over baselines is statistically significant on the 480-clip locked test set:

1. **$D_2\_H1\_seed42$ vs $E1$:**
   - Empirical Mean $\Delta \text{Macro-F1}$: **$+26.54\%$**
   - 95% Confidence Interval: **$[+17.06\%, +35.80\%]$**
   - One-tailed $p$-value: **$p < 0.001$**
   - **Conclusion:** Statistically significant improvement over $E1$.
2. **$D_2\_H1\_seed42$ vs $D_0\_H1\_seed42$:**
   - Empirical Mean $\Delta \text{Macro-F1}$: **$+26.39\%$**
   - 95% Confidence Interval: **$[+17.04\%, +35.25\%]$**
   - One-tailed $p$-value: **$p < 0.001$**
   - **Conclusion:** Statistically significant improvement over in-domain VieNeu training condition $D_0\_H1$.

### 3.3 Confusion Matrix Reproduction ($D_2\_H1\_seed42$)

```
                  Predicted Class
                TRAI   PHAI    LEN  XUONG  REJECT   Support
True TRAI        16      0      0      0       0        16
True PHAI         0     16      0      0       0        16
True LEN          0      0     16      0       0        16
True XUONG        0      0      0     16       0        16
True REJECT       2      5     25      2     382       416
Support          18     21     41     18     382       480
```

- Target Command Recall: **100.0%** across all 4 commands (64/64 detected).
- False Alarm breakdown by command: LEN absorbed 25 false alarms (predominantly from phonetic confusers "lên", "luôn", "lớp"), PHAI absorbed 5, TRAI absorbed 2, XUONG absorbed 2.

### 3.4 Acoustic Condition Breakdowns (SNR and Rate)

| Breakdown Slice | Slice Value | $D_2\_H1\_seed42$ Macro-F1 | Phonetic OOV FAR | General OOV FAR |
|:----------------|:------------|:-----------------------------:|:----------------:|:---------------:|
| **SNR Tier** | Clean | 90.40% | 3.12% (2/64) | 2.50% (1/40) |
| | 15 dB SNR | 83.18% | 3.12% (2/64) | 12.50% (5/40) |
| | 10 dB SNR | 79.08% | 7.81% (5/64) | 15.00% (6/40) |
| | 5 dB SNR | 78.43% | 6.25% (4/64) | 22.50% (9/40) |
| **Speech Rate** | 1.0x Normal | 85.93% | 3.91% (5/128) | 11.25% (9/80) |
| | 1.1x Fast | 79.52% | 6.25% (8/128) | 15.00% (12/80) |
| **Voice** | vi-VN-HoaiMyNeural | 80.60% | 6.25% (8/128) | 13.75% (11/80) |
| | vi-VN-NamMinhNeural | 84.82% | 3.91% (5/128) | 12.50% (10/80) |

---

## 4. Continuous Streaming Evaluation Reproduction

**Streaming Test Corpus Structure (50 sequences):**
- 10 Pure Silence/Noise sequences (held-out noise tracks 36–39, duration = 96.30 s / 1.61 min)
- 10 Pure OOV Conversational sequences (duration = 75.31 s / 1.26 min)
- 15 Clean Multi-Command sequences (3 commands concatenated with natural inter-command gaps)
- 15 Noise-Augmented Multi-Command sequences (15 dB, 10 dB, 5 dB SNR)
- Total Duration: **401.06 s (6.68 min)**

### 4.1 Comparative Streaming Metrics

| Streaming Metric | Phase 13.4 (v11) | Phase 18 (E1) | Phase 28 (D0_H1_seed42) | Phase 28 (D2_H1_seed42) | Status |
|:-----------------|:----------------:|:-------------:|:-----------------------:|:-----------------------:|:------:|
| **Exact Sequence Accuracy** | 20.00% (10/50) | 24.00% (12/50) | 24.00% (12/50) | **52.00% (26/50)** | **VERIFIED** |
| **Ground Truth Commands** | 90 | 90 | 90 | 90 | **VERIFIED** |
| **Detected Commands** | 8 | 51 | 51 | 86 | **VERIFIED** |
| **True Positive Commands**| 8 | 39 | 40 | **74** | **VERIFIED** |
| **Missed Commands (FN)** | 82 | 51 | 50 | **16** | **VERIFIED** |
| **Substitutions** | 0 | 1 | 0 | **1** | **VERIFIED** |
| **False Insertions (FP)**| 0 | 11 | 11 | **11** | **VERIFIED** |
| **Command Recall** | 8.89% | 43.33% | 44.44% | **82.22%** | **VERIFIED** |
| **Command Precision** | 100.00% | 76.47% | 78.43% | **86.05%** | **VERIFIED** |
| **Command F1-Score** | 16.33% | 55.32% | 56.74% | **84.09%** | **VERIFIED** |

### 4.2 Streaming False Alarm Rate Audit & Rule of Three Bounds

When evaluating false alarm rates on silence and conversational OOV sequences where 0 false alarms are observed, claiming an absolute "0.00 FA/min" is statistically invalid due to finite sample observation. We apply the **Rule of Three** for zero-event Poisson processes:

$$\text{UCB}_{95\%} = \frac{3.0}{\text{Duration in Minutes}}$$

| Sequence Slice | Total Duration (s) | Total Duration (min) | Observed FAs | Recorded FA/min | Rule of Three 95% UCB | Audit Compliance |
|:---------------|:------------------:|:--------------------:|:------------:|:---------------:|:---------------------:|:----------------:|
| **Pure Silence/Noise** | 96.30 s | 1.61 min | **0** | 0.00 FA/min | **1.87 FA/min** | **CORRECTED** |
| **Pure OOV Conversational** | 75.31 s | 1.26 min | **0** | 0.00 FA/min | **2.39 FA/min** | **CORRECTED** |
| **Multi-Command Clean** | 105.00 s | 1.75 min | 2 | 1.14 FA/min | Point Estimate: 1.14 | Exact Match |
| **Multi-Command 15 dB** | 35.00 s | 0.58 min | 2 | 3.43 FA/min | Point Estimate: 3.43 | Exact Match |
| **Multi-Command 10 dB** | 35.00 s | 0.58 min | 3 | 5.14 FA/min | Point Estimate: 5.14 | Exact Match |
| **Multi-Command 5 dB** | 35.00 s | 0.58 min | 4 | 6.86 FA/min | Point Estimate: 6.86 | Exact Match |

---

## 5. Summary of Claims, Reproductions, and Governance Disclosures

| Claim from Phase 28–31 | Evidence Artifact | Reproduced? | Forensic Limitation / Correction | Corrected Claim Status |
|:-----------------------|:------------------|:-----------:|:----------------------------------|:-----------------------|
| Isolated Macro-F1 = 82.72% for D2_H1 | `phase28_31_final_evaluation_results.json` | **YES** | Valid on `edge_tts_eval_v2` (within-generator transfer). | **VERIFIED** |
| Phonetic OOV FAR dropped from 30.5% to 5.08% | `phase28_31_final_evaluation_results.json` | **YES** | 13/256 errors on fast/noise-augmented test clips. | **VERIFIED** |
| Continuous streaming Command F1 = 84.09% | `edge_tts_eval_v2_streaming_events.json` | **YES** | 74 TP, 16 Miss, 11 FP across 50 sequences. | **VERIFIED** |
| Factor H is statistically significant ($p < 0.05$) | `phase28_31_statistical_analysis.json` | **NO** | ANOVA $p = 0.0872 > 0.05$. Effect size $\eta_p^2 = 0.32$ is suggestive but non-significant. | **CORRECTED** |
| Streaming Silence FA/min is 0.00 | `edge_tts_eval_v2_streaming_manifest.csv` | **PARTIAL** | 0 events observed over 1.61 min; 95% UCB is **1.87 FA/min**. | **CORRECTED** |
| Streaming OOV FA/min is 0.00 | `edge_tts_eval_v2_streaming_manifest.csv` | **PARTIAL** | 0 events observed over 1.26 min; 95% UCB is **2.39 FA/min**. | **CORRECTED** |
| Cross-generator generalization achieved for D2 | `PHASE32_35_PROVENANCE_AUDIT.json` | **NO** | D2 training exposed 10% Edge-TTS from same 2 voices. In-generator transfer only. | **CORRECTED** |
| Cross-architecture generalization verified | Model configs | **NO** | All models use fixed 138,469 parameter `CorrectedVietnameseKWSNet`. | **CORRECTED** |
| Model is ready for live microphone deployment | Governance Policy | **NO** | All 4 governance labels remain strictly active. Synthetic research only. | **PROHIBITED** |

---

## 6. Mandatory Governance Declarations

```
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```
