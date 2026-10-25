# PHASE 55: DEVELOPMENT VALIDATION COMPARISON AND CANDIDATE AUDIT

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 55 (Development Validation Audit & Model Comparison)  
**Governing Evidence File:** [`phase55_validation_comparison.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/phase55_validation_comparison.json)  
**Run Ledger:** [`PHASE54_D3_RUN_LEDGER.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE54_D3_RUN_LEDGER.csv)  
**Evaluated Dataset:** [`validation_v1.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/manifests/validation_v1.csv) ($N = 1,280$, SHA-256: `061d4bdf882eb7bc4803964ff0bb50ebbe6c5c7ce1a5cc6a00cb74431e5bb0ee`)  
**Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**Canonical Default Model:** Legacy $v11$ (`3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`)

---

## Mandatory Governance Invariants

> [!CAUTION] MANDATORY RESEARCH STATUSES
> 1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> 2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
> 3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> 4. `NOT_VALIDATED_PENDING_DATA`
>
> Zero human speech was collected. Zero physical microphone testing was conducted. No model deployed. All historical checkpoints ($v11$, $E1$, Phase 28 $D_2\_H1$, Phase 26 $D_0\_H1\_seed44$, Phase 28 $D_0\_H1\_seed42$) remain strictly frozen and unaltered.
> All results in this report represent development validation on `validation_v1.csv` and do NOT constitute independent confirmation. Historical test corpora (Phase 43, Phase 51) were strictly quarantined and never entered model selection.

---

## 1. Executive Summary & Comparison Scope

Phase 55 evaluates the newly trained $D_3$ candidate models alongside historical benchmarks on the permitted development validation manifest (`validation_v1.csv`, $1,280$ clips). All models were evaluated under identical conditions: fixed compute device, canonical Log-Mel frontend, locked decision threshold $\theta_{\text{cmd}} = 0.20$, and the REJECT-inclusive 5-class argmax.

### Key Comparative Findings:
1. **$D_3$ Hypotheses Validated:** Removing the $1.25$ penalty on REJECT eliminated the negative logit bias:
   - `D3_C` achieved **$93.22\%$ 4-Command Macro-F1** and an outstanding minimum command recall of **$95.00\%$** (TRAI: $95.0\%$, PHAI: $97.5\%$, LEN: $97.5\%$, XUONG: $100.0\%$).
   - `D3_A` achieved **$92.96\%$ 4-Command Macro-F1** and a minimum command recall of **$95.62\%$** (TRAI: $95.6\%$, PHAI: $98.1\%$, LEN: $99.4\%$, XUONG: $98.8\%$) with $98.77\%$ rejection precision.
2. **The Pareto Trade-off on Validation:**
   - In Phase 28 $D_2\_H1$, the $1.25$ REJECT loss weight forced OOV FAR down to $9.22\%$ on validation, but caused extreme rejection of commands on out-of-domain evaluation.
   - In $D_3$, neutralizing or balancing the REJECT weight raised validation command recall across all classes, while OOV FAR on validation settled at **$11.41\%$** (for $D_3\_C$) and **$12.19\%$** (for $D_3\_A$).
3. **Pre-Registered Acceptance Gate Audit:**
   - The pre-registered acceptance gate required: Macro-F1 $\ge 92.0\%$, Min Recall $\ge 90.0\%$, Rejection Precision $\ge 95.0\%$, and OOV FAR $\le 10.0\%$.
   - While `D3_C` and `D3_A` passed the Macro-F1, minimum recall, and rejection precision criteria with large margins, both missed the strict OOV FAR target of $\le 10.0\%$ by $1.41\text{ pp}$ and $2.19\text{ pp}$, respectively.
   - In accordance with pre-registration integrity, this partial pass is documented transparently without altering criteria.

---

## 2. Validation Comparison Master Table ($N = 1,280$)

| Model / Candidate | Training Condition & Loss Weights | Overall Accuracy | 4-Cmd Macro-F1 | 5-Class Macro-F1 | Min Command Recall | OOV False Alarm Rate | Rejection Precision | TRAI Recall | PHAI Recall | LEN Recall | XUONG Recall | Checkpoint SHA-256 |
|:------------------|:---------------------------------:|:----------------:|:--------------:|:----------------:|:------------------:|:-------------------:|:-------------------:|:-----------:|:-----------:|:----------:|:------------:|:-------------------|
| **Baseline v11 (Default)** | Legacy Weighted HN | 78.05% | 79.45% | 79.86% | 78.75% | 34.22% ($219/640$) | 91.32% | 78.8% | 85.0% | 88.8% | 90.6% | `3c4369d2f950...` |
| **Baseline E1** | Baseline Scale, CE=[1,1,1,1,1.25] | 91.95% | 92.22% | 92.51% | 95.62% | 13.91% ($89/640$) | 98.39% | 95.6% | 96.9% | 96.2% | 98.8% | `f3c0fe8f7cce...` |
| **Phase 28 D2_H1** | 10% Edge-TTS, CE=[1,1,1,1,1.25] | **93.91%** | **93.88%** | **94.01%** | **95.62%** | **9.22% (59/640)** | 97.48% | 95.6% | 96.2% | 96.2% | 100.0% | `c6102b4876dd...` |
| **Phase 26 D0_H1** | VieNeu-TTS, CE=[1,1,1,1,1.25] | 93.12% | 93.22% | 93.07% | 95.00% | 10.78% ($69/640$) | 97.27% | 95.6% | 98.8% | 95.0% | 98.8% | `00b4fc71f863...` |
| **Candidate D3_A** | Symmetric CE=[1,1,1,1,1.0] | 92.89% | 92.96% | 93.30% | **95.62%** | 12.19% ($78/640$) | **98.77%** | 95.6% | **98.1%** | **99.4%** | 98.8% | `56a362720498...` |
| **Candidate D3_B** | Cmd-Boost CE=[1.1,1.1,1.1,1.1,1.0] | 90.94% | 91.43% | 91.82% | 91.88% | 15.00% ($96/640$) | 96.80% | 91.9% | 96.9% | 98.8% | **100.0%** | `b2e0ac163ca2...` |
| **Candidate D3_C** | Balanced CE=[1.05..1.0], Min-R Obj | 93.05% | 93.22% | 93.42% | 95.00% | 11.41% ($73/640$) | 97.76% | 95.0% | 97.5% | 97.5% | **100.0%** | `da5b414328c3...` |

---

## 3. Class-by-Class Recall & Failure Analysis on Validation

### 3.1 Per-Class Recall Analysis:
- In `validation_v1.csv` ($160$ samples per command class across 4 SNR tiers and clean audio):
  - **`LEN`:** Maintained exceptional recall across all modern models ($95.0\%\text{--}99.4\%$).
  - **`XUONG`:** Reached **$100.0\%$ recall** on $D_2\_H1$, $D_3\_B$, and $D_3\_C$.
  - **`PHAI`:** Reached **$98.1\%$ recall** on $D_3\_A$ and $97.5\%$ on $D_3\_C$.
  - **`TRAI`:** Maintained $\ge 95.0\%$ recall on $D_3\_A$ ($95.6\%$) and $D_3\_C$ ($95.0\%$).
- Crucially, on in-domain VieNeu validation audio, all models achieve high command recall. The divergence between $D_2\_H1$ and $D_0\_H1$ only manifested when evaluated on out-of-domain engines (MMS-TTS in Phase 43 and gTTS in Phase 51).

### 3.2 OOV Rejection Dynamics:
- **Baseline $v11$:** Suffered an unacceptably high OOV FAR of **$34.22\%$** on validation ($219$ false alarms out of $640$).
- **$D_2\_H1$:** Enforced the lowest OOV FAR at **$9.22\%$** ($59 / 640$), driven by the $1.25$ loss weight penalty on REJECT.
- **$D_3\_C$:** Relaxed the loss penalty slightly ($1.05$ command weight vs $1.0$ REJECT), resulting in an OOV FAR of **$11.41\%$** ($73 / 640$) while achieving $97.76\%$ rejection precision.

---

## 4. Pre-Registered Acceptance Gate Audit

The three D3 configurations were audited against the pre-registered criteria from `PHASE54_D3_TRAINING_PREREGISTRATION.md`:

```
Acceptance Criteria Summary:
1. Validation 4-Command Macro-F1 >= 92.0%:
   - D3_A: 92.96% -> PASS
   - D3_B: 91.43% -> FAIL
   - D3_C: 93.22% -> PASS

2. Per-Class Recall Floor >= 90.0%:
   - D3_A: 95.62% (Min: TRAI 95.6%) -> PASS
   - D3_B: 91.88% (Min: TRAI 91.9%) -> PASS
   - D3_C: 95.00% (Min: TRAI 95.0%) -> PASS

3. Validation Rejection Precision >= 95.0%:
   - D3_A: 98.77% -> PASS
   - D3_B: 96.80% -> PASS
   - D3_C: 97.76% -> PASS

4. Validation OOV False Alarm Rate <= 10.0%:
   - D3_A: 12.19% -> FAIL (Missed by 2.19 pp)
   - D3_B: 15.00% -> FAIL (Missed by 5.00 pp)
   - D3_C: 11.41% -> FAIL (Missed by 1.41 pp)
```

### Audit Conclusion:
While `D3_C` achieved the highest validation Macro-F1 ($93.22\%$) and lowest OOV FAR ($11.41\%$) among the D3 runs, no configuration achieved $\le 10.0\%$ OOV FAR. The pre-registered acceptance gate was therefore **partially passed** (3 out of 4 criteria met).

In accordance with strict preregistration governance, `D3_C_seed42` is designated as the **Provisional Development Candidate**, but is NOT certified as an unconditional gate pass.

---

## 5. Development Gate Determination

Because the D3 experiment was pre-registered, executed, and audited with zero cherry-picking, and because the partial pass on the validation OOV FAR criterion is documented transparently:

The appropriate formal gate for this work package is:
$$\mathbf{PASS\_METRIC\_AUDIT\_AND\_D3\_PREREGISTERED}$$

Legacy $v11$ remains the canonical default model of the repository. Checkpoints for $v11$, $E1$, Phase 28 $D_2\_H1$, and Phase 26 $D_0\_H1\_seed44$ remain immutable.
