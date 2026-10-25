# PHASE 60: FORENSIC AUDIT OF TRACK B FALSE-ALARM METRICS, DENOMINATORS, AND UNCERTAINTY BOUNDS

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Audit Target:** Phase 59 Track B Streaming False-Alarm Evaluation  
**Reference Corpus:** `PHASE58_STREAMING_MANIFEST.csv` (50 sequences), `PHASE58_EVENT_LEDGER.csv` (58 events)  
**Status:** **AUDITED AND RECONCILED WITH MATHEMATICAL EVIDENCE**  

---

## 1. EXECUTIVE SUMMARY & FORENSIC FINDING

An exhaustive forensic audit of the Track B streaming evaluation code (`eval_phase59_confirmatory_test.py`), audio manifests, event ledgers, and raw output files was conducted to reconcile:
1. Exact audio duration denominators for silence, conversational out-of-vocabulary (OOV), and contextual streaming.
2. The mathematical derivation of reported Upper Confidence Bounds (UCB95).
3. The discrepancy between exact Poisson rate bounds and the values reported in Phase 59 (2.28 for silence and 5.47 for OOV).

### Core Forensic Conclusions:
1. **Ad-Hoc Normal Heuristic Identified:** The evaluation code implemented an ad-hoc normal approximation formula with a variance clamp:
   $$\text{UCB95}_{\text{ad-hoc}} = \frac{k + 1.96 \sqrt{\max(1, k)}}{T_{\text{min}}}$$
   - For silence ($k = 0$, $T = 0.86045\text{ min}$): $\frac{0 + 1.96 \times 1}{0.86045} = \mathbf{2.28\text{ FA/min}}$.
   - For single-event OOV ($k = 1$, $T = 0.54155\text{ min}$): $\frac{1 + 1.96 \times 1}{0.54155} = \frac{2.96}{0.54155} = \mathbf{5.47\text{ FA/min}}$.
2. **Exact Poisson Upper Bounds Corrected:**
   - Under an exact one-sided Poisson 95% confidence interval ($\alpha = 0.05$):
     - For $k = 0$ events in $0.86045$ minutes: $\lambda_{\text{upper}} = \frac{-\ln(0.05)}{T} = \frac{2.99573}{0.86045} = \mathbf{3.48\text{ FA/min}}$ (Rule of Three approximation: $3 / 0.86045 = \mathbf{3.49\text{ FA/min}}$).
     - For $k = 1$ event in $0.54155$ minutes: $\lambda_{\text{upper}} = \frac{\chi^2_{4, 0.95}}{2 T} = \frac{4.74386}{0.54155} = \mathbf{8.76\text{ FA/min}}$.
3. **Duration Denominator Reconciliation:**
   - The actual total audio duration of the 50 streaming sequences in `PHASE58_STREAMING_MANIFEST.csv` is **$218.201$ seconds** ($3.6367$ minutes).
   - The figure of "$200.7$ seconds" in report text was an unrounded preliminary approximation that did not affect computation, as the script loaded individual waveform lengths from disk ($51.627$s silence, $32.493$s OOV, $134.081$s contextual commands).

---

## 2. DURATION DENOMINATOR RECONCILIATION

The 50 streaming sequences in `PHASE58_STREAMING_MANIFEST.csv` comprise three distinct sequence types:

| Sequence Type | Count | Exact Duration (s) | Exact Duration (min) | Previous Reported (min) | Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `pure_silence_noise` | 10 | 51.627 s | **0.86045 min** | 0.860 min | $+0.00045$ min |
| `pure_oov_conversation` | 10 | 32.493 s | **0.54155 min** | 0.542 min | $-0.00045$ min |
| `contextual_commands` | 30 | 134.081 s | **2.23468 min** | *(Omitted in text)* | N/A |
| **Total Streaming Audio** | **50** | **218.201 s** | **3.63668 min** | **200.7 s** | **$+17.501$ s** |

### Evaluation Scope:
- In `eval_phase59_confirmatory_test.py`, silence false alarms were accumulated strictly over `pure_silence_noise` clips ($51.627$s).
- Conversational OOV false alarms were accumulated strictly over `pure_oov_conversation` clips ($32.493$s).
- Background false alarms during contextual command speech were tracked under the `false_insertions` taxonomy ($24$ for $D_2\_H1$, $13$ for Phase 26, $27$ for $D_3\_C$).

---

## 3. MODEL-BY-MODEL RECONCILIATION TABLE

### 3.1 Silence Subgroup ($T = 0.86045\text{ min}$)

| Model | Observed Count ($k$) | Point Rate (FA/min) | Ad-Hoc Normal UCB95 (Reported) | Exact Poisson 95% UCB | Rule of Three ($3/T$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline v11 (default)** | 0 | 0.00 | 2.28 | **3.48** | 3.49 |
| **Baseline E1** | 0 | 0.00 | 2.28 | **3.48** | 3.49 |
| **Phase 28 $D_2\_H1$** | 0 | 0.00 | 2.28 | **3.48** | 3.49 |
| **Phase 26 $D_0\_H1$** | 0 | 0.00 | 2.28 | **3.48** | 3.49 |
| **Candidate $D_3\_C$** | 0 | 0.00 | 2.28 | **3.48** | 3.49 |

### 3.2 Conversational OOV Subgroup ($T = 0.54155\text{ min}$)

| Model | Observed Count ($k$) | Point Rate (FA/min) | Ad-Hoc Normal UCB95 (Reported) | Exact Poisson 95% UCB | Exact Poisson Multiplier $\chi^2_{2(k+1), 0.95}/2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 28 $D_2\_H1$** | 1 | **1.85** | 5.47 | **8.76** | 4.744 |
| **Phase 26 $D_0\_H1$** | 1 | **1.85** | 5.47 | **8.76** | 4.744 |
| **Candidate $D_3\_C$** | 1 | **1.85** | 5.47 | **8.76** | 4.744 |
| **Baseline E1** | 4 | **7.39** | 14.62 | **16.90** | 9.154 |
| **Baseline v11 (default)** | 15 | **27.70** | 41.72 | **42.65** | 23.097 |

---

## 4. MATHEMATICAL FORMULATION AUDIT

### 4.1 Root Cause of Reported Values (2.28 and 5.47)
The evaluation script defined:
```python
sil_ucb95 = (silence_seq_fa + 1.96 * np.sqrt(max(1, silence_seq_fa))) / sil_min
oov_ucb95 = (oov_seq_fa + 1.96 * np.sqrt(max(1, oov_seq_fa))) / oov_min
```
1. For $k = 0$:
   $$\text{UCB95}_{\text{silence}} = \frac{0 + 1.96 \times \sqrt{1}}{0.86045} = \frac{1.96}{0.86045} = 2.2779 \approx \mathbf{2.28}$$
2. For $k = 1$:
   $$\text{UCB95}_{\text{OOV}} = \frac{1 + 1.96 \times \sqrt{1}}{0.54155} = \frac{2.96}{0.54155} = 5.4658 \approx \mathbf{5.47}$$

### 4.2 Exact Poisson Confidence Limit Formulation
For a homogeneous Poisson process with rate $\lambda$ and exposure time $T$, the exact one-sided $100(1-\alpha)\%$ upper confidence bound is given by:
$$\lambda_{\text{upper}} = \frac{1}{2T} \chi^2_{2(k+1), 1-\alpha}$$
For $\alpha = 0.05$ (95% confidence):
- When $k = 0$, degrees of freedom $2(0+1) = 2$:
  $$\chi^2_{2, 0.95} = -2 \ln(0.05) \approx 5.99146 \implies \lambda_{\text{upper}} = \frac{5.99146}{2 \times 0.86045} = \mathbf{3.4816\text{ FA/min}}$$
- When $k = 1$, degrees of freedom $2(1+1) = 4$:
  $$\chi^2_{4, 0.95} \approx 9.48773 \implies \lambda_{\text{upper}} = \frac{9.48773}{2 \times 0.54155} = \frac{4.74386}{0.54155} = \mathbf{8.7598\text{ FA/min}}$$

### 4.3 Why the Rule of Three Cannot Apply to $k \ge 1$
The "Rule of Three" ($3/T$) is an algebraic approximation of $-\ln(0.05)/T \approx 2.9957/T$, derived strictly from the zero-term Poisson probability:
$$P(K = 0) = e^{-\lambda T} \le 0.05 \implies \lambda T \ge -\ln(0.05) \approx 3.00$$
When $k = 1$ event is observed, $P(K \le 1) = e^{-\lambda T}(1 + \lambda T) \le 0.05$, which solves to $\lambda T \ge 4.74386$. Applying $3/T$ to a nonzero event cohort is mathematically invalid.

---

## 5. IMPACT ON PRE-REGISTERED ENGINEERING CRITERIA

The pre-registered Low-FA profile for $D_2\_H1$ in Phase 55 specified:
- Silence false alarm rate: $\le 1.0\text{ FA/min}$.
- Conversational OOV false alarm rate: $\le 6.0\text{ FA/min}$.

### Point Estimates vs. Upper Confidence Bounds:
1. **Silence:** Observed rate is **$0.00\text{ FA/min}$** ($\le 1.0$, **PASS**). The 95% upper bound of $3.48\text{ FA/min}$ reflects limited sampling duration ($0.86$ min), not observed failures.
2. **Conversational OOV:** Observed rate is **$1.85\text{ FA/min}$** ($\le 6.0$, **PASS**). The point estimate comfortably satisfies the gate criteria. However, due to the short duration ($0.542$ min), the exact 95% Poisson upper bound is $8.76\text{ FA/min}$.
3. **Audit Verdict:** The Low-FA profile pass on point estimates is **CONFIRMED**, but the report must explicitly distinguish the observed point estimate ($1.85$) from the exact Poisson uncertainty bound ($8.76\text{ FA/min}$).

Full machine-readable ledger preserved in:
`experiments/vieneu_kws_research_v12/reports/PHASE60_FA_RATE_RECONCILIATION.json`.
