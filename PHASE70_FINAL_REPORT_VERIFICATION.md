# PHASE 70: FINAL REPORT EVIDENCE VERIFICATION
## Comprehensive Forensic Cross-Audit of Consolidated Research Reports against Authoritative Primary Ledgers

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 26, 2026  
**Audited Documents:**  
1. Root Consolidated Report: [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md) (and [`experiments/vieneu_kws_research_v12/reports/RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/RESEARCH_FINAL_REPORT.md))  
2. Root Executive Summary: [`FINAL_RESEARCH_SUMMARY.md`](file:///d:/trainidentificationaudio/FINAL_RESEARCH_SUMMARY.md) (and [`Vietnamese_KWS_Final_Research_Summary.md`](file:///d:/trainidentificationaudio/Vietnamese_KWS_Final_Research_Summary.md))  
**Primary Ground-Truth Evidentiary Sources:**  
- [`PHASE64_VALIDATION_RECONCILED_METRICS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_RECONCILED_METRICS.csv)  
- [`PHASE64_VALIDATION_MASTER_METRICS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_MASTER_METRICS.json)  
- [`PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv) ($N=1,280$ rows)  
- [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json)  
- [`PHASE63_CORRECTED_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE63_CORRECTED_FINAL_REPORT.md)  
- [`FINAL_CLAIM_EVIDENCE_TABLE.csv`](file:///d:/trainidentificationaudio/FINAL_CLAIM_EVIDENCE_TABLE.csv)  
- [`FINAL_SHA256_MANIFEST.json`](file:///d:/trainidentificationaudio/FINAL_SHA256_MANIFEST.json)  

---

## MANDATORY GOVERNANCE DECLARATIONS

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

1. **Synthetic-Only Research Boundary:** All empirical numbers, confusion matrices, false alarm bounds, and statistical tests reported herein derive strictly from synthetic text-to-speech audio mixtures and synthetic acoustic noise files.
2. **Defensive Model Retention:** Legacy baseline model $v11$ ([`vietnamese_kws_v11.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt), [`vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx)) remains the sole default reference model.
3. **Locked Inference Threshold:** Operating command threshold $\theta_{\text{cmd}} = 0.20$ is strictly immutable.
4. **Zero Physical Microphone or Human Validation:** No physical microphone streaming or human acoustic recordings were collected or validated.

---

## 1. Executive Forensic Audit Findings & Report Synchronizations

During the Phase 70 audit, every quantitative claim, numerator, denominator, metric definition, uncertainty bound, and statistical interpretation in the consolidated reporting suite was systematically cross-referenced against the raw prediction matrices and execution ledgers.

### Key Audit Discrepancies Identified and Corrected:
1. **Truncation of Root [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md):**  
   - *Finding:* The file in repository root was found to be truncated at 1,951 bytes, ending prematurely during the preamble without containing Sections 1 through 13.
   - *Resolution:* Root [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md) was replaced with the full, authoritative 23,259-byte consolidated scientific report synchronized with [`experiments/vieneu_kws_research_v12/reports/RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/RESEARCH_FINAL_REPORT.md).
2. **Nomenclature of [`FINAL_RESEARCH_SUMMARY.md`](file:///d:/trainidentificationaudio/FINAL_RESEARCH_SUMMARY.md):**  
   - *Finding:* The Vietnamese executive summary existed in root as [`Vietnamese_KWS_Final_Research_Summary.md`](file:///d:/trainidentificationaudio/Vietnamese_KWS_Final_Research_Summary.md), while governance checklists specified [`FINAL_RESEARCH_SUMMARY.md`](file:///d:/trainidentificationaudio/FINAL_RESEARCH_SUMMARY.md).
   - *Resolution:* Both filenames are now synchronized in the repository root to ensure seamless traceability across languages and build tools.
3. **Formal Ledger of Discrepancies:**  
   - All 13 core metrics have been compiled into [`PHASE70_CLAIM_DISCREPANCY_LEDGER.csv`](file:///d:/trainidentificationaudio/PHASE70_CLAIM_DISCREPANCY_LEDGER.csv).

---

## 2. Verification of Authoritative Validation Metrics (Phase 64 Reconciled)

Under the repository's locked decision rule ($\theta_{\text{cmd}} = 0.20$), every prediction row in [`validation_v1.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/validation_v1.csv) ($N = 1,280$: 640 REJECT clips, 160 each of `TRAI`, `PHAI`, `LEN`, `XUONG`) was recomputed from raw audio:

```
Authoritative Validation Performance Matrix (validation_v1.csv, N = 1,280, theta_cmd = 0.20):
+--------------------------------+---------------+----------+----------+----------+----------+-----------------+-----------------+---------------+
| Model / Candidate              | Correct / Tot | Accuracy | 4-Cmd F1 | 5-Cls F1 | Min Rec  | OOV FAR (N=640) | Rej Precision   | Rej Recall    |
+--------------------------------+---------------+----------+----------+----------+----------+-----------------+-----------------+---------------+
| Baseline v11 (Default)         | 999 / 1280    | 78.05%   | 79.45%   | 78.86%   | 78.75%   | 34.22% (219/640)| 91.32% (421/461)| 65.78%(421/640|
| Baseline E1 Frozen             | 1177 / 1280   | 91.95%   | 92.22%   | 92.14%   | 95.62%   | 13.91% (89/640) | 98.39% (551/560)| 86.09%(551/640|
| Phase 28 D2_H1 Candidate       | 1202 / 1280   | 93.91%   | 93.88%   | 93.91%   | 95.62%   | 9.22% (59/640)  | 97.48% (581/596)| 90.78%(581/640|
| Phase 26 D0_H1 seed44          | 1192 / 1280   | 93.12%   | 93.22%   | 93.19%   | 95.00%   | 10.78% (69/640) | 97.27% (571/587)| 89.22%(571/640|
| Phase 28 D0_H1 seed42          | 1198 / 1280   | 93.59%   | 93.78%   | 93.72%   | 96.88%   | 11.56% (74/640) | 99.12% (566/571)| 88.44%(566/640|
| Candidate D3_A seed42          | 1189 / 1280   | 92.89%   | 92.96%   | 92.96%   | 95.62%   | 12.19% (78/640) | 98.77% (562/569)| 87.81%(562/640|
| Candidate D3_B seed42          | 1164 / 1280   | 90.94%   | 91.43%   | 91.25%   | 91.88%   | 15.00% (96/640) | 96.80% (544/562)| 85.00%(544/640|
| Candidate D3_C seed42          | 1191 / 1280   | 93.05%   | 93.22%   | 93.17%   | 95.00%   | 11.41% (73/640) | 97.76% (567/580)| 88.59%(567/640|
+--------------------------------+---------------+----------+----------+----------+----------+-----------------+-----------------+---------------+
```

### Forensic Proof of the Historical 88.59% Accuracy Transcription Error:
- In historical Phase 55–56 text documentation, $v11$ validation accuracy was mistakenly reported as $88.59\%$ ($1,134 / 1,280$).
- *Forensic Cause:* $88.59375\%$ ($567 / 640$) is the exact REJECT recall of Candidate $D_3\_C$. A copy-paste transposition occurred where $D_3\_C$'s reject recall was placed into $v11$'s accuracy cell.
- Under locked policy $\theta_{\text{cmd}} = 0.20$, the true, authoritative validation accuracy of $v11$ is **$78.05\%$** ($999 / 1,280$).
- When evaluated across an unconstrained discrete threshold sweep ($\theta \in [0.10, 0.95]$, step $0.02$), the maximum observed accuracy for $v11$ was **$80.55\%$** ($1,031 / 1,280$) at $\theta = 0.28$. Under the operational policy $\theta = 0.20$, accuracy is $78.05\%$.

---

## 3. Verification of Confirmatory Test Results (Phase 59 Piper VAIS-1000)

Authoritative metrics reproduced from [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json) on unseen Piper VAIS-1000 audio mixed with ESC-50 Fold 3 noise:

### 3.1 Track A (Isolated Utterances, $N = 420$ Primary Clips):
- **Phase 28 $D_2\_H1$:**
  - Overall Accuracy: **$74.52\%$** ($313 / 420$)
  - 4-Command Macro-F1: **$53.76\%$**
  - 5-Class Macro-F1: **$59.65\%$**
  - Target Command Accuracy: **$42.50\%$** ($68 / 160$)
  - REJECT Recall: **$94.23\%$** ($245 / 260$)
  - Isolated OOV FAR: **$5.77\%$** ($15 / 260$)
  - Tone Probe FAR ($N=60$ probe clips): **$6.67\%$** ($4 / 60$)
- **Candidate $D_3\_C$:**
  - Overall Accuracy: **$69.29\%$** ($291 / 420$)
  - 4-Command Macro-F1: **$41.23\%$**
  - Isolated OOV FAR: **$13.08\%$** ($34 / 260$)
- **Baseline $v11$:**
  - Overall Accuracy: **$43.33\%$** ($182 / 420$)
  - Isolated OOV FAR: **$64.23\%$** ($167 / 260$)

### 3.2 Track B (Continuous Streaming Audio, 50 Sequences, 58 GT Commands, 218.201 s):
- **Phase 28 $D_2\_H1$:**
  - Exact Sequence Matches: **$23 / 50$** ($46.00\%$)
  - Command Event Recall: **$17.24\%$** ($10 / 58$ true positives; $41$ misses, $7$ substitutions)
  - Command Event Precision: **$21.74\%$** ($10 / 46$ detected command events)
  - Command Event F1: **$19.23\%$**
  - Silence False Alarm Rate: **$0.00\text{ FA/min}$** ($0$ false alarms in $0.86045\text{ min}$)
  - Conversational OOV False Alarm Rate: **$1.85\text{ FA/min}$** ($1$ false alarm in $0.54155\text{ min}$)
- **Candidate $D_3\_C$:**
  - Exact Sequence Matches: **$25 / 50$** ($50.00\%$)
  - Command Event Recall: **$32.76\%$** ($19 / 58$ true positives)
  - Command Event Precision: **$30.16\%$** ($19 / 63$ detected command events)
  - Command Event F1: **$31.40\%$**

### 3.3 Exact Poisson Uncertainty Formulations:
- **Silence Cohort ($k = 0$, $T = 0.86045\text{ min}$):**  
  $$\text{UCB}_{95\%} = \frac{-\ln(0.05)}{T} = \frac{2.99573}{0.86045} = \mathbf{3.48\text{ FA/min}}$$  
  *(Rule of Three approximation: $3.0 / 0.86045 = 3.49\text{ FA/min}$. Historical ad-hoc code value of $2.28\text{ FA/min}$ is formally repudiated).*
- **Conversational OOV Cohort ($k = 1$, $T = 0.54155\text{ min}$):**  
  $$\text{UCB}_{95\%} = \frac{\chi^2_{2(1+1), 0.95}}{2T} = \frac{\chi^2_{4, 0.95}}{2 \times 0.54155} = \frac{9.48773}{1.08310} = \mathbf{8.76\text{ FA/min}}$$  
  *(Historical ad-hoc code value of $5.47\text{ FA/min}$ is formally repudiated).*

---

## 4. Verification of the 4 Key Core Research Conclusions

1. **$D_2\_H1$ Achieves Low False-Alarm Rates on Evaluated Synthetic Sets:**  
   *Verified.* On Track A isolated evaluation, $D_2\_H1$ recorded an OOV FAR of $5.77\%$ ($15/260$), meeting the $\le 10.0\%$ Low-FA profile threshold. On Track B streaming, its point false-alarm rate on conversational OOV was $1.85\text{ FA/min}$ (meeting the $\le 2.0\text{ FA/min}$ point threshold).
2. **$D_2\_H1$ Suffers from Critical Command-Recall Collapse:**  
   *Verified.* $D_2\_H1$'s low false-alarm rate is achieved through aggressive over-rejection rather than selective phonetic discrimination. Under synthesizer domain shift:
   - Command recall on `TRAI` dropped to **$22.50\%$** ($9/40$; $29/40$ ground-truth `TRAI` clips were misclassified as REJECT).
   - In continuous streaming audio (Track B), $D_2\_H1$ missed $41$ of $58$ commands, yielding a command recall of only **$17.24\%$** and an Event F1 of **$19.23\%$**.
   - *Conclusion:* Strong OOV rejection does not imply usable command detection. $D_2\_H1$ is unsuitable for continuous operational deployment.
3. **Results Do Not Establish Natural-Speech Performance or Deployment Readiness:**  
   *Verified.* All training, validation, and testing were conducted exclusively with synthetic TTS audio (VieNeu, Edge-TTS, Meta MMS, Google Translate TTS, Piper VAIS-1000) and environmental noise files (ESC-50). Zero natural human speech was evaluated, and zero physical microphone streaming was performed.
4. **Baseline $v11$ Locked Validation Accuracy is $78.05\%$ ($999/1,280$):**  
   *Verified.* The authoritative prediction ledger [`PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv) conclusively establishes that $v11$ correctly classified $999$ of $1,280$ validation samples under $\theta_{\text{cmd}} = 0.20$.

---

## 5. Statistical Claim & Significance Verification ($\alpha = 0.01$)

All comparative significance claims were audited against the dependency-aware clustered paired $t$-test across $N_{\text{inst}} = 105$ primary waveforms ($df = 104$):

| Model Comparison | Mean Difference ($\Delta$) | $95\%$ Confidence Interval | $t$-statistic | $p$-value | Preregistered $\alpha = 0.01$ Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$D_2\_H1$ vs Baseline $v11$** | $+31.19\text{ pp}$ | $[+22.66, +39.72]\text{ pp}$ | $+7.2479$ | $7.73 \times 10^{-11}$ | **STATISTICALLY SIGNIFICANT** ($p < 0.01$) |
| **Phase 26 vs Baseline $v11$** | $+27.62\text{ pp}$ | $[+18.47, +36.77]\text{ pp}$ | $+5.9868$ | $3.09 \times 10^{-8}$ | **STATISTICALLY SIGNIFICANT** ($p < 0.01$) |
| **$D_3\_C$ vs Baseline $v11$** | $+25.95\text{ pp}$ | $[+16.68, +35.22]\text{ pp}$ | $+5.5514$ | $2.18 \times 10^{-7}$ | **STATISTICALLY SIGNIFICANT** ($p < 0.01$) |
| **$D_2\_H1$ vs $D_3\_C$** | $+5.24\text{ pp}$ | $[+0.41, +10.06]\text{ pp}$ | $+2.1533$ | $0.0336$ | **NOT SIGNIFICANT AT $\alpha = 0.01$** |

*Key Governance Rule:* Because $p = 0.0336 > 0.01$, the $+5.24\text{ pp}$ difference between $D_2\_H1$ and $D_3\_C$ **fails confirmatory statistical significance**. It is an exploratory trend nominally significant only at an uncorrected $\alpha = 0.05$.

---

## 6. Audit Verdict

All primary claims in [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md) and [`FINAL_RESEARCH_SUMMARY.md`](file:///d:/trainidentificationaudio/FINAL_RESEARCH_SUMMARY.md) have been verified and reconciled against raw evidence.

**Selected Phase 70 Audit Gate:**  
$$\mathbf{PASS\_WITH\_DOCUMENTED\_REPORT\_CORRECTIONS}$$
