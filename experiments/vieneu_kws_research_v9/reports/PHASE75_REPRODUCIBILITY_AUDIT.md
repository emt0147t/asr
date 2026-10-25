# Phase 75 — Reproducibility and Metric Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Candidate Identifier:** `variant_w10`  
**Evaluation Protocol:** Phase 74 Locked Synthetic Benchmark ($N = 720$ clips)  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Audit Overview & Methodology

Phase 75 audited the exact reproduction of Phase 74 test metrics directly from the cryptographic record:
- Source per-clip prediction file: [`experiments/vieneu_kws_research_v8/predictions/per_clip_predictions.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v8/predictions/per_clip_predictions.csv) (SHA-256: `7c7a0e79460c753fb904c3e3aca45491185d42756f6f438248fd08ebd5b7dd8f`)
- Number of audited predictions for `variant_w10`: **2,160 rows** (720 clips $\times$ 3 seeds).
- Audit result: **100% bit-exact reproduction** across all macro, micro, per-keyword, false alarm, and accuracy metrics.

---

## 2. Disaggregated Per-Seed Reproduction

| Seed | Calibrated $\theta_{\text{cmd}}$ | Macro Command Recall | 95% Wilson CI (Recall) | Non-Target FAR | Raw FAs (out of 560) | 95% Wilson CI (FAR) | 5-Class Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.40 | 100.00% | [97.66%, 100.00%] | 2.68% | 15/560 | [1.63%, 4.37%] | 97.92% |
| **43** | 0.50 | 99.38% | [96.55%, 99.89%] | 1.61% | 9/560 | [0.85%, 3.03%] | 98.61% |
| **44** | 0.35 | 100.00% | [97.66%, 100.00%] | 1.79% | 10/560 | [0.97%, 3.26%] | 98.61% |

---

## 3. Disaggregated Per-Keyword Recall

| Keyword | Seed 42 Recall | Seed 43 Recall | Seed 44 Recall | Mean Recall $\pm$ Sample Std | Predeclared Target ($\ge 95\%$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TRAI` | 100.00% (40/40) | 100.00% (40/40) | 100.00% (40/40) | **100.00% $\pm$ 0.00%** | **MET** |
| `PHAI` | 100.00% (40/40) | 97.50% (39/40) | 100.00% (40/40) | **99.17% $\pm$ 1.44%** | **MET** |
| `LEN` | 100.00% (40/40) | 100.00% (40/40) | 100.00% (40/40) | **100.00% $\pm$ 0.00%** | **MET** |
| `XUONG` | 100.00% (40/40) | 100.00% (40/40) | 100.00% (40/40) | **100.00% $\pm$ 0.00%** | **MET** |

---

## 4. Sampling Unit Audit & Uncertainty Reporting

> [!IMPORTANT]
> **Repeated Measurement Disclosure:**  
> The 3 models (seeds 42, 43, 44) were evaluated on the identical 720 test clips. These predictions represent repeated clustered measurements on a fixed audio sample, not $3 \times 720 = 2,160$ independent clips.
>
> Pooling these evaluations into an aggregate Wilson confidence interval would artificially underestimate the standard error by a factor of $\sqrt{3} \approx 1.73$.
>
> Therefore, this report:
> 1. Reports exact Wilson intervals separately for each individual seed.
> 2. Computes cross-seed metrics using the unbiased sample standard deviation ($N=3$, $N-1=2$ degrees of freedom):
>    - **Mean Command Recall:** **99.79% $\pm$ 0.36%**
>    - **Mean Non-Target FAR:** **2.03% $\pm$ 0.57%**
>    - **Mean 5-Class Accuracy:** **98.38% $\pm$ 0.40%**

---

## 5. Audit Conclusion

The candidate `variant_w10` is cryptographically and numerically verified. All predeclared bounded-utterance targets are met on the frozen benchmark. No threshold or model selection sweeps were conducted during this audit.
