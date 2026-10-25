# PHASE 56: DEVELOPMENT CANDIDATE AND PARETO FORENSIC AUDIT

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 56 (Development Candidate & Pareto Optimization Audit)  
**Governing Evidence Files:**
- [`PHASE56_PARETO_TABLE.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE56_PARETO_TABLE.csv)
- [`PHASE56_METRIC_REPRODUCTION.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE56_METRIC_REPRODUCTION.json)
- [`PHASE54_D3_RUN_LEDGER.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE54_D3_RUN_LEDGER.csv)  
**Audited Manifest:** [`validation_v1.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/validation_v1.csv) ($N = 1,280$, SHA-256: `061d4bdf882eb7bc4803964ff0bb50ebbe6c5c7ce1a5cc6a00cb74431e5bb0ee`)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Default Canonical Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No model deployed. Checkpoints for $v11$, $E1$, Phase 28 $D_2\_H1$, and Phase 26 $D_0\_H1\_seed44$ remain strictly frozen and read-only.
> Historical evaluation corpora (Phase 43, Phase 51) were strictly quarantined and never entered model training or candidate selection.

---

## 1. Executive Summary & Audit Scope

This audit independently reconciles and audits all model candidates on the repository's permitted development validation split ([`validation_v1.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/validation_v1.csv), $N=1,280$). Every headline metric was reproduced from raw model logits and predictions under the canonical protocol ($\theta_{\text{cmd}} = 0.20$, fixed Log-Mel frontend, REJECT-inclusive 5-class argmax).

### Critical Audit Findings:
1. **100% Metric Reproduction:** All reported validation metrics for $v11$, $E1$, $D_2\_H1$, Phase 26 $D_0\_H1\_seed44$, and $D_3\_A/B/C$ were verified with zero discrepancies.
2. **Explicit Metric Reconciliation:**
   - **Phase 28 $D_2\_H1$:** 4-Command Macro-F1 = **$93.88\%$**, Min Recall = **$95.62\%$**, OOV FAR = **$9.22\%$**, Overall Accuracy = **$93.91\%$**.
   - **Phase 26 $D_0\_H1$:** 4-Command Macro-F1 = **$93.22\%$**, Min Recall = **$95.00\%$**, OOV FAR = **$10.78\%$**, Overall Accuracy = **$93.12\%$**.
   - **Candidate $D_3\_C$:** 4-Command Macro-F1 = **$93.22\%$**, Min Recall = **$95.00\%$**, OOV FAR = **$11.41\%$**, Overall Accuracy = **$93.05\%$**.
3. **Pareto Domination Reality on In-Domain Validation:**
   On existing validation data (`validation_v1.csv`), **$D_2\_H1$ strictly dominates $D_3\_C$** across all primary validation metrics:
   $$\text{Macro-F1}: 93.88\% > 93.22\% \quad | \quad \text{Min Recall}: 95.62\% > 95.00\% \quad | \quad \text{OOV FAR}: 9.22\% < 11.41\%$$
   Furthermore, Phase 26 achieves identical Macro-F1 ($93.22\%$) and identical minimum recall ($95.00\%$) to $D_3\_C$, while achieving a lower OOV FAR ($10.78\%$ vs $11.41\%$).
   **Scientific Integrity Directive:** $D_3\_C$ must NOT be promoted as superior on validation data.
4. **Three Distinct Conceptual Statuses:**
   This audit enforces strict separation between:
   - *Best Model on Existing Validation:* $D_2\_H1$.
   - *Candidate Developed for an Engineering Operating Regime:* $D_3\_C$ was developed under the hypothesis that removing the $1.25$ loss weight penalty on REJECT prevents out-of-domain sensitivity collapse; this is an exploratory candidate hypothesis, NOT an empirical victory on validation data.
   - *Model Confirmed on a New Test:* Can only be established after a locked, unobserved confirmation test on a genuinely independent speech source.
5. **Loss-Weight Hypothesis Demoted from Causal Certainty to Proposed Mechanism:**
   The $55.56\%$ gradient-energy calculation is acknowledged as a proposed mathematical mechanism, not proven causal proof, since loss reweighting interacts with stochastic optimization paths, learning rate schedules, and gradient variance.

---

## 2. Reproduced Master Validation Metrics Table ($N = 1,280$)

All metrics below were computed directly from model forward passes on [`validation_v1.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/validation_v1.csv) and logged in [`PHASE56_PARETO_TABLE.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE56_PARETO_TABLE.csv):

| Model Name | Role in Repository | Architecture Scale | Accuracy | 4-Cmd Macro-F1 | 5-Class Macro-F1 | Min Command Recall | OOV False Alarm Rate | Rejection Precision | TRAI Recall | PHAI Recall | LEN Recall | XUONG Recall | REJECT Recall | Checkpoint SHA-256 |
|:-----------|:-------------------|:------------------:|:--------:|:--------------:|:----------------:|:------------------:|:-------------------:|:-------------------:|:-----------:|:-----------:|:----------:|:------------:|:-------------:|:-------------------|
| **Baseline v11** | Canonical Default | Legacy CNN | 78.05% | 79.45% | 79.86% | 78.75% | 34.22% ($219/640$) | 91.32% | 78.8% | 85.0% | 88.8% | 90.6% | 65.78% | `3c4369d2f950...` |
| **Baseline E1** | Frozen Benchmark | 138,469 params | 91.95% | 92.22% | 92.51% | 95.62% | 13.91% ($89/640$) | 98.39% | 95.6% | 96.9% | 96.2% | 98.8% | 86.09% | `f3c0fe8f7cce...` |
| **Phase 28 D2_H1** | Frozen Candidate | 138,469 params | **93.91%** | **93.88%** | **94.01%** | **95.62%** | **9.22% (59/640)** | 97.48% | 95.6% | 96.2% | 96.2% | **100.0%** | **90.78%** | `c6102b4876dd...` |
| **Phase 26 D0_H1** | Frozen Candidate | 138,469 params | 93.12% | 93.22% | 93.07% | 95.00% | 10.78% ($69/640$) | 97.27% | 95.6% | 98.8% | 95.0% | 98.8% | 89.22% | `00b4fc71f863...` |
| **Candidate D3_A** | Dev Comparison | 138,469 params | 92.89% | 92.96% | 93.30% | **95.62%** | 12.19% ($78/640$) | **98.77%** | 95.6% | **98.1%** | **99.4%** | 98.8% | 87.81% | `56a362720498...` |
| **Candidate D3_B** | Dev Comparison | 138,469 params | 90.94% | 91.43% | 91.82% | 91.88% | 15.00% ($96/640$) | 96.80% | 91.9% | 96.9% | 98.8% | **100.0%** | 85.00% | `b2e0ac163ca2...` |
| **Candidate D3_C** | Dev Candidate | 138,469 params | 93.05% | 93.22% | 93.42% | 95.00% | 11.41% ($73/640$) | 97.76% | 95.0% | 97.5% | 97.5% | **100.0%** | 88.59% | `da5b414328c3...` |

---

## 3. Detailed Subgroup OOV Breakdown on Validation ($N = 640$ REJECT Clips)

To audit whether specific background types suffered disproportionate false alarms, the 640 REJECT samples in `validation_v1.csv` were decomposed into four distinct subgroups:

| Model | Total OOV FAR ($N=640$) | OOV Word FAR ($N=256$) | Conversation Speech FAR ($N=160$) | Environmental Noise FAR ($N=128$) | Ambient Silence FAR ($N=96$) |
|:------|:-----------------------:|:----------------------:|:----------------------------------:|:---------------------------------:|:----------------------------:|
| **Baseline v11 (Default)** | **34.22%** ($219/640$) | 37.11% ($95/256$) | 34.38% ($55/160$) | 32.81% ($42/128$) | 28.12% ($27/96$) |
| **Baseline E1** | **13.91%** ($89/640$) | 15.62% ($40/256$) | 13.12% ($21/160$) | 12.50% ($16/128$) | 12.50% ($12/96$) |
| **Phase 28 D2_H1** | **9.22% (59/640)** | **10.16% (26/256)** | **8.75% (14/160)** | **7.81% (10/128)** | **9.38% (9/96)** |
| **Phase 26 D0_H1** | **10.78%** ($69/640$) | 12.11% ($31/256$) | 10.62% ($17/160$) | 9.38% ($12/128$) | 9.38% ($9/96$) |
| **Candidate D3_A** | **12.19%** ($78/640$) | 13.67% ($35/256$) | 11.88% ($19/160$) | 10.94% ($14/128$) | 10.42% ($10/96$) |
| **Candidate D3_B** | **15.00%** ($96/640$) | 16.41% ($42/256$) | 15.00% ($24/160$) | 14.06% ($18/128$) | 12.50% ($12/96$) |
| **Candidate D3_C** | **11.41%** ($73/640$) | 12.89% ($33/256$) | 11.25% ($18/160$) | 10.16% ($13/128$) | 9.38% ($9/96$) |

---

## 4. The Pareto Domination Audit & Honest Model Status

### 4.1 Strict Mathematical Comparison
Comparing Phase 28 $D_2\_H1$, Phase 26 $D_0\_H1$, and $D_3\_C$ on the existing validation set:

```
                  Macro-F1 (Higher is better)    Min Recall (Higher)    OOV FAR (Lower is better)
Phase 28 D2_H1:            93.88%                      95.62%                     9.22%
Phase 26 D0_H1:            93.22%                      95.00%                    10.78%
Candidate D3_C:            93.22%                      95.00%                    11.41%

Pareto Dominance Assessment:
- D2_H1 vs D3_C:  D2_H1 wins on Macro-F1 (+0.66 pp), wins on Min Recall (+0.62 pp), wins on OOV FAR (-2.19 pp).
                  Verdict: D2_H1 strictly Pareto-dominates D3_C on validation data.
- P26 vs D3_C:    P26 ties on Macro-F1 (0.00 pp), ties on Min Recall (0.00 pp), wins on OOV FAR (-0.63 pp).
                  Verdict: Phase 26 Pareto-dominates D3_C on validation data.
```

### 4.2 Why D3_C Was Developed & What It Represents
If $D_3\_C$ does not beat $D_2\_H1$ or Phase 26 on validation data, why does it exist?
- **The Empirical Motive:** On out-of-domain evaluation (Phase 51, Google Translate TTS), $D_2\_H1$ collapsed in command sensitivity ($25.76\%$ Macro-F1, $34.38\%$ target-only accuracy), whereas Phase 26 preserved $50.35\%$ Macro-F1.
- **The Hypothesis:** $D_2\_H1$'s out-of-domain sensitivity collapse was hypothesized to stem from its $1.25$ loss weight on REJECT combined with dual-engine domain constriction. $D_3$ was trained to test whether relaxing this penalty ($1.05$ on commands vs $1.0$ on REJECT) would produce an architecture that maintains balanced sensitivity on unseen neural engines.
- **The Epistemic Distinction:**
  1. *Validation Status:* On in-domain validation, $D_3\_C$ is slightly inferior to $D_2\_H1$ because `validation_v1.csv` is composed of in-domain VieNeu audio where $D_2\_H1$'s tight boundaries are rewarded.
  2. *Development Role:* $D_3\_C$ is a **development challenger candidate** specifically tailored for out-of-domain sensitivity.
  3. *Confirmatory Requirement:* $D_3\_C$'s hypothesis can only be validated on an unobserved independent evaluation set (Phase 58–59), never by retroactively re-evaluating on Phase 51 or claiming victory on validation data.

---

## 5. Audit of the Loss-Weight Causal Claim

In Phase 53, it was calculated that combining a 50% REJECT batch prior with a $1.25$ cross-entropy loss weight resulted in REJECT accounting for $55.56\%$ of the nominal gradient energy in each mini-batch:
$$\frac{32 \times 1.25}{(32 \times 1.0) + (32 \times 1.25)} = \frac{40}{72} \approx 55.56\%$$

### Critical Audit Limitation:
- **Gradient Energy is a Proposed Mechanism, Not a Proof:**
  While this algebraic calculation accurately reflects the scalar weighting of the loss function, neural network optimization involves complex non-linear backpropagation dynamics. The gradient norm of each class also depends on the softmax probability $\mathbf{p}_i - \mathbf{y}_i$, the feature representation in the conv layers, and the optimizer's second-moment scaling in AdamW ($v_t$).
- **Controlled Finding:**
  When loss weights were adjusted to symmetric $[1.0, 1.0, 1.0, 1.0, 1.0]$ in $D_3\_A$ and $[1.05..1.0]$ in $D_3\_C$, validation command recall remained high ($\ge 95\%$), but OOV FAR increased by $2.2\text{ pp}$. This confirms that loss weighting directly shifts the operating trade-off between sensitivity and rejection, but does not prove it was the sole cause of out-of-domain behavior.

---

## 6. Audit Verdict

All metrics for all seven models reproduce with **100% numerical precision**. The original D3 pre-registration, run ledger, and failed criteria are fully preserved without retroactive revision.

**Phase 56 Audit Gate:**
$$\mathbf{PASS\_DEVELOPMENT\_CANDIDATE\_AUDIT\_CONFIRMED}$$
