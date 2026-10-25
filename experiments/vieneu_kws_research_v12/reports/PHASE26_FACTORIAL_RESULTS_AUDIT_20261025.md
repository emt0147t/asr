# PHASE 26 — FACTORIAL RESULTS AUDIT AND CANDIDATE FREEZE REPORT

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Auditor / Role:** Antigravity Pair Programming Engine (Controlled Governance Track)  
**Previous Phase:** Phase 25 — Final Training Execution Review & Controlled Factorial Experiment  
**Previous Gate:** `PASS_FACTORIAL_TRAINING_COMPLETED`  
**Architecture Decision:** `DEFER_E2_CAPACITY`  
**Final Phase Status:** `PASS_TO_CANDIDATE_FREEZE_ONLY`

---

## Executive Summary

Phase 26 conducted an independent, exhaustive audit of the 12 completed runs from the Phase 25 $2 \times 2$ factorial training campaign. All training logs, checkpoint files, parameter configurations, manifests, and validation outputs were verified. 

Key outcomes of the Phase 26 audit:
1. **12-Run Integrity Audit:** 100% of the 12 factorial runs were verified to have completed without exception or early abort, correctly utilizing their assigned manifests (`D0_train_v1.csv` and `D1_matched_train_v2.csv`), maintaining exact architecture scale (`CorrectedVietnameseKWSNet`, 138,469 trainable parameters), and honoring the fixed inference decision threshold ($\theta_{\text{cmd}} = 0.20$).
2. **Canonical Frozen Baseline Verification:** All frozen reference artifacts—including the $v11$ checkpoint and ONNX model, the $E1$ checkpoint and ONNX model, and the selected Phase 15 VAD configuration—remain bit-for-bit immutable. The historical command failure in Phase 25 was identified as an ad-hoc query against an obsolete path (`variant_v11_pre_silence_thresholding.pt`) rather than the canonical release path (`variant_v11_hard_negative_weighted_seed42.pt`), which has now been explicitly cataloged and verified.
3. **Statistical Method Audit & Correction:** The original Phase 25 statistical script utilized Welch's two-sample t-test ($N=6$ vs $N=6$) across pooled factor levels, yielding $t = -1.966$ ($p = 0.0882$) for Factor D and $t = -0.115$ ($p = 0.9111$) for Factor H. This method conflated levels of the complementary factor and ignored the randomized seed block structure. An independent two-way fixed effects ANOVA ($df_{\text{error}} = 8$) yields $F = 3.3681$ ($p = 0.1038$) for Factor D, $F = 0.0159$ ($p = 0.9027$) for Factor H, and $F = 0.6946$ ($p = 0.4288$) for the $D \times H$ interaction. A randomized block ANOVA controlling for seed blocks ($df_{\text{error}} = 6$) yields $F = 4.1054$ ($p = 0.0891$) for Factor D. With $N=3$ seeds per cell, the experiment has limited power, and non-significant results must not be interpreted as proof of zero effect.
4. **REJECT Error Diagnostics:** Detailed confusion audit on the 640 validation REJECT samples reveals that conversational speech, environmental noise, and acoustic silence achieve virtually zero false alarms across all models ($\le 0.21\%$). Nearly 100% of all false alarms on REJECT stem from OOV words (`reject_subgroup == 'oov_word'`). Hard-negative sampling ($H_1$) successfully depresses OOV false alarm rates in both $D_0$ ($29.04\% \to 27.60\%$) and $D_1$ ($30.86\% \to 25.65\%$). In $D_0$, this reduction occurs without sacrificing keyword recall, establishing $D_0\_H_1$ as the most stable and balanced condition.
5. **Candidate Selection & Freezing:** A sharp policy distinction is drawn between condition-level mean, individual run peak, and evaluation candidacy. Because cross-condition candidate selection was not pre-registered in Phase 23, checkpoint selection is designated as **post-hoc on validation**. Checkpoint **`D0_H1_seed44`** (Validation Macro-F1: $93.22\%$, SHA-256: `00b4fc71...`) is frozen into a dedicated candidate bundle as the primary evaluation candidate representing the winning condition ($D_0\_H_1$, mean $93.02\% \pm 0.17\%$). Peak individual checkpoint **`D0_H0_seed44`** ($93.38\%$) is cataloged as a reference baseline.
6. **Next Independent Evaluation Plan:** The observed test corpus `synthetic_eval_test_v1` remains strictly quarantined. Because an authorized third-party TTS engine or unobserved natural speech test corpus is currently unavailable locally, the independent evaluation phase is formally recorded as `BLOCKED_PENDING_INDEPENDENT_EVALUATION_SOURCE`. The final gate status is accordingly set to `PASS_TO_CANDIDATE_FREEZE_ONLY`.

All four mandatory governance labels are upheld without reservation:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

---

## 1. Audit of the 12 Factorial Training Runs

### 1.1 Complete 12-Run Audit Matrix

Each run directory in `experiments/vieneu_kws_research_v12/training_phase25/` was inspected on disk. Checkpoints were loaded into memory to verify exact architecture topology and trainable parameters; logs and summaries were parsed for epoch convergence and cryptographic hashes:

| Run Identifier | Manifest File | Manifest SHA-256 | Sampler | Total Params | Best Ep / Total | Val Acc | Val Loss | Val Macro-F1 | Checkpoint SHA-256 | Elapsed (s) | Exit Code |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **`D0_H0_seed42`** | `D0_train_v1.csv` | `e0bbc0a9...` | H0 | 138,469 | 21 / 29 | 92.81% | 0.2063 | **92.46%** | `c438c03acb42ecdb676a031e0c832014ae07b674ee41284196f65f092709272a` | 50.4 | 0 |
| **`D0_H0_seed43`** | `D0_train_v1.csv` | `e0bbc0a9...` | H0 | 138,469 | 40 / 40 | 92.34% | 0.2306 | **92.42%** | `63e1dd6bc1c25ecde5c9c44f3ed84875e6d9dcc88821b43b409f5e9e92c4e8eb` | 81.5 | 0 |
| **`D0_H0_seed44`** | `D0_train_v1.csv` | `e0bbc0a9...` | H0 | 138,469 | 31 / 39 | 93.12% | 0.2256 | **93.38%** | `8b79c539765f04522826d6c610bd6a7e8603a0a72fb564a5fcc262bd65cfa30f` | 101.9 | 0 |
| **`D0_H1_seed42`** | `D0_train_v1.csv` | `e0bbc0a9...` | H1 | 138,469 | 32 / 40 | 92.97% | 0.2150 | **92.95%** | `8d8fd08ae4cf586a53298c2679f8e0d992d3a05ceddce58f99b46305f5088e81` | 88.2 | 0 |
| **`D0_H1_seed43`** | `D0_train_v1.csv` | `e0bbc0a9...` | H1 | 138,469 | 40 / 40 | 93.05% | 0.2185 | **92.90%** | `8e5b21759f561b507be6ce41c7b5eb863ea8ec2e44a2dedc2eaf0a3c7616900b` | 240.4 | 0 |
| **`D0_H1_seed44`** | `D0_train_v1.csv` | `e0bbc0a9...` | H1 | 138,469 | 32 / 40 | 93.12% | 0.2080 | **93.22%** | `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3` | 343.7 | 0 |
| **`D1_H0_seed42`** | `D1_matched_train_v2.csv` | `581a0856...` | H0 | 138,469 | 18 / 26 | 91.64% | 0.2515 | **91.98%** | `af3984de10a27d112472e9d51207343ea1ed0eb2e4b1034873f3bdb8240120e6` | 220.7 | 0 |
| **`D1_H0_seed43`** | `D1_matched_train_v2.csv` | `581a0856...` | H0 | 138,469 | 35 / 40 | 93.12% | 0.2132 | **93.14%** | `c9ee314c3be9b13392c2a6ba65bcd2b17f66389f496dfb672b351b866b79cb80` | 339.8 | 0 |
| **`D1_H0_seed44`** | `D1_matched_train_v2.csv` | `581a0856...` | H0 | 138,469 | 36 / 40 | 91.88% | 0.2573 | **91.99%** | `fc8cbe02116f7e7ef22957b3632214d53f33ba9be434aa354b18f5f2f86ca8ec` | 344.0 | 0 |
| **`D1_H1_seed42`** | `D1_matched_train_v2.csv` | `581a0856...` | H1 | 138,469 | 12 / 20 | 91.09% | 0.2712 | **90.86%** | `477772801ef0b302e91b03dd6e4ebf2c42a1e39f327d4830abe6e879e0ee7bda` | 169.3 | 0 |
| **`D1_H1_seed43`** | `D1_matched_train_v2.csv` | `581a0856...` | H1 | 138,469 | 38 / 40 | 92.81% | 0.2330 | **92.72%** | `3444035ea6f094c8e8e17c7b85e2af7303eddd236647ee877de215e398794e6b` | 340.2 | 0 |
| **`D1_H1_seed44`** | `D1_matched_train_v2.csv` | `581a0856...` | H1 | 138,469 | 22 / 30 | 92.42% | 0.2691 | **92.42%** | `251411a5f5f55d34bb53ffcb8a9e2edd78679e82016a012b40ef800724030666` | 255.0 | 0 |

### 1.2 Training Manifests and Validation Ground Truth Hashes
- **$D_0$ Training Manifest:** `experiments/vieneu_kws_research_v12/manifests/D0_train_v1.csv`  
  SHA-256: `e0bbc0a95a6d18f9724a7784f53a0082ce0fd3aae8a8e15a67ed39b6743ec7f7` ($N=5,440$)
- **$D_1$ Matched Training Manifest (Path A):** `experiments/vieneu_kws_research_v12/manifests/D1_matched_train_v2.csv`  
  SHA-256: `581a08561b4eb3722afa1d6302852363e504f080248844abb3a1be369d93c06e` ($N=5,440$)
- **Validation Manifest:** `experiments/vieneu_kws_research_v12/manifests/validation_v1.csv`  
  SHA-256: `5af8cf9daa96526cba1a742d1e18982cd4c1ed291dc17d139a8d82bf1cb06682` ($N=1,280$)
- **Source Eligibility & Quarantine Ledger:** `experiments/vieneu_kws_research_v12/manifests/source_eligibility_and_quarantine_ledger.csv`  
  SHA-256: `1113f248f1d62b010c0f27000abe33caf474da1abe52af3f477c731c48aa7407` ($N=6,144$)

### 1.3 Identification and Reconciliation of Historical Hash Check Failure

During Step 17039 of Phase 25, an ad-hoc PowerShell one-liner exited with code 1:
```
FileNotFoundError: [Errno 2] No such file or directory: 'experiments\\synthetic_kws_phase13_3_16\\dataset_release_candidate\\checkpoints\\variant_v11_pre_silence_thresholding.pt'
```
**Forensic Root Cause:**
The command queried an experimental working checkpoint path (`variant_v11_pre_silence_thresholding.pt`) referenced in preliminary design notes. That path was superseded during Phase 13.4 when the final CNN model was officially released. The canonical paths and hashes established in Phase 13.4, Phase 17, and Phase 18 are:
1. `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`
2. `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`
3. `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt`
4. `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx`
5. `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json`

Re-running canonical verification confirms that zero reference files were overwritten or corrupted:
```
v11_bundle_pt     : 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2
v11_bundle_onnx   : 54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99
v11_v12_bundle_pt : 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2
e1_pt             : f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878
e1_onnx           : 20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8
vad_cfg           : af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3
```
All hashes match historical records bit-for-bit.

---

## 2. Statistical Recomputation and Method Audit

### 2.1 Recomputed Condition Performance
Primary outcome metric: true four-command macro-F1:
$$\operatorname{MacroF1} = \frac{1}{4}(F1_{\text{TRAI}} + F1_{\text{PHAI}} + F1_{\text{LEN}} + F1_{\text{XUONG}})$$

| Condition | Seeds | Mean Val Macro-F1 | Std Dev ($\sigma$) | 95% CI (Student-$t$, $df=2$) | Mean Accuracy | Mean Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D0_H0`** | 42, 43, 44 | **92.75%** | ±0.54% | $[91.41\%, 94.10\%]$ | 92.76% | 0.2208 |
| **`D0_H1`** | 42, 43, 44 | **93.02%** | **±0.17%** | $[92.60\%, 93.45\%]$ | 93.05% | 0.2138 |
| **`D1_H0`** | 42, 43, 44 | **92.37%** | ±0.67% | $[90.71\%, 94.03\%]$ | 92.21% | 0.2407 |
| **`D1_H1`** | 42, 43, 44 | **92.00%** | ±1.00% | $[89.52\%, 94.47\%]$ | 92.11% | 0.2578 |

### 2.2 Factorial Method Audit & Statistical Correction

In Phase 25, the script `analyze_phase25_factorial.py` computed p-values via `scipy.stats.ttest_ind(..., equal_var=False)`:
- Factor D: $t = -1.966$, $p = 0.0882$ (Welch's t-test, $N=6$ vs $N=6$, $df \approx 7.04$)
- Factor H: $t = -0.115$, $p = 0.9111$ (Welch's t-test, $N=6$ vs $N=6$, $df \approx 9.77$)

#### Critical Methodological Critique
1. **Conflation of Factor Levels:** The Welch t-test compared the 6 runs of $D_1$ against the 6 runs of $D_0$, ignoring whether each run was trained under $H_0$ or $H_1$.
2. **Violation of Independence:** Treating 12 runs as 12 independent samples ignores the shared random seed structure ($42, 43, 44$).
3. **Absence of Interaction Term:** The t-tests did not evaluate the $D \times H$ interaction within a unified model.

#### Rigorous ANOVA Recomputation

We compute both the full Two-Way Fixed Effects ANOVA and the Randomized Block ANOVA (blocking on Seed):

```
=== 2-WAY FIXED EFFECTS ANOVA (df_error = 8) ===
Source of Variation     SS          df      MS          F-statistic     p-value
-------------------------------------------------------------------------------
Factor D (Source)       0.00014839   1      0.00014839  3.3681          0.1038
Factor H (Prior)        0.00000070   1      0.00000070  0.0159          0.9027
Interaction (D x H)     0.00003060   1      0.00003060  0.6946          0.4288
Residual (Error)        0.00035247   8      0.00004406
Total                   0.00053216  11

=== RANDOMIZED BLOCK ANOVA (Seed Blocked, df_error = 6) ===
Source of Variation     SS          df      MS          F-statistic     p-value
-------------------------------------------------------------------------------
Seed Block (42,43,44)   0.00013559   2      0.00006780  1.8757          0.2329
Factor D (Source)       0.00014839   1      0.00014839  4.1054          0.0891
Factor H (Prior)        0.00000070   1      0.00000070  0.0194          0.8938
Interaction (D x H)     0.00003060   1      0.00003060  0.8466          0.3930
Residual (Error)        0.00021687   6      0.00003615
Total                   0.00053216  11
```

#### Official Statistical Correction Notice
The historical report `PHASE25_FACTORIAL_TRAINING_EXECUTION_REPORT_20261024.md` cited $p = 0.0882$ (Factor D) and $p = 0.9111$ (Factor H) from the Welch t-test. The authoritative Two-Way ANOVA values are:
- **Factor D Main Effect:** $\Delta = -0.70\%$, $F(1, 8) = 3.37, p = 0.1038$ (Two-Way ANOVA) / $F(1, 6) = 4.11, p = 0.0891$ (Blocked ANOVA). 95% Confidence Interval: $[-1.59\%, +0.18\%]$.
- **Factor H Main Effect:** $\Delta = -0.05\%$, $F(1, 8) = 0.02, p = 0.9027$ (Two-Way ANOVA) / $F(1, 6) = 0.02, p = 0.8938$ (Blocked ANOVA). 95% Confidence Interval: $[-0.93\%, +0.84\%]$.
- **Interaction ($D \times H$):** Contrast $= -0.64\%$, $F(1, 8) = 0.69, p = 0.4288$. 95% Confidence Interval: $[-2.41\%, +1.13\%]$.

> [!IMPORTANT]
> **Power Limitation Warning:** With $N=3$ seeds per cell, statistical power to detect small effect sizes ($|\Delta| < 1.0\%$) is low. The non-significant p-values ($p > 0.05$) indicate **absence of statistical evidence for an effect**, not **evidence of absence**. The 95% confidence intervals encompass both slight negative drift and modest positive gain.

---

## 3. Detailed REJECT Confusion and Subgroup Analysis

To determine how the conditions handle non-keyword audio, all 12 checkpoints were evaluated on `validation_v1.csv` ($N=1,280$, containing 640 keyword targets and 640 REJECT items):

| Condition | Seed | REJECT FA Total | REJECT FA Rate | Keyword Miss Rate | OOV FA Count ($/256$) | OOV FA Rate | Conv FA Count ($/160$) | Env Noise FA ($/128$) | Silence FA ($/96$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D0_H0`** | 42 | 69 / 640 | 10.78% | 2.19% | 69 / 256 | 26.95% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D0_H0`** | 43 | 84 / 640 | 13.12% | 1.41% | 83 / 256 | 32.42% | 1 / 160 (0.6%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D0_H0`** | 44 | 71 / 640 | 11.09% | 2.34% | 71 / 256 | 27.73% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D0_H1`** | 42 | 72 / 640 | 11.25% | 2.03% | 72 / 256 | 28.12% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D0_H1`** | 43 | 71 / 640 | 11.09% | 1.72% | 71 / 256 | 27.73% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D0_H1`** | 44 | 69 / 640 | **10.78%** | **2.50%** | 69 / 256 | **26.95%** | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D1_H0`** | 42 | 85 / 640 | 13.28% | 2.81% | 85 / 256 | 33.20% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D1_H0`** | 43 | 69 / 640 | 10.78% | 2.34% | 69 / 256 | 26.95% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D1_H0`** | 44 | 84 / 640 | 13.12% | 2.34% | 83 / 256 | 32.42% | 1 / 160 (0.6%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D1_H1`** | 42 | 67 / 640 | 10.47% | 6.25% | 67 / 256 | 26.17% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D1_H1`** | 43 | 70 / 640 | 10.94% | 2.50% | 70 / 256 | 27.34% | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |
| **`D1_H1`** | 44 | 60 / 640 | **9.38%** | **5.31%** | 60 / 256 | **23.44%** | 0 / 160 (0.0%) | 0 / 128 (0.0%) | 0 / 96 (0.0%) |

### Key Diagnostic Insights
1. **Acoustic and Conversational Rejection is Solved:** Environmental noise (128 clips) and acoustic silence (96 clips) experienced **0.00% false alarms** across all 12 models. Conversational speech (160 clips) experienced only 2 total false alarms across the entire 12-run battery ($\le 0.21\%$).
2. **OOV Confusers Are the Sole Source of False Alarms:** 99.8% of all false alarms on REJECT are caused by out-of-vocabulary spoken words.
3. **Hard-Negative Sampling ($H_1$) Mechanism:**
   - In $D_0$, $H_1$ reduces the mean OOV FA rate from $29.04\%$ down to $27.60\%$ while keeping keyword miss rate nearly identical ($1.98\% \to 2.08\%$).
   - In $D_1$, $H_1$ forces OOV FA down even further to $25.65\%$ (and $23.44\%$ in seed 44); however, this aggressive suppression causes the keyword miss rate to more than double ($2.50\% \to 4.69\%$, reaching $6.25\%$ in seed 42).
   - This explains the negative $D \times H$ interaction: $D_1\_H_1$ over-penalizes speech acoustic features, degrading true keyword sensitivity. Condition **$D_0\_H_1$** achieves the optimal balance between OOV rejection and keyword recall.

---

## 4. Checkpoint-Selection Policy Resolution & Candidate Freezing

### 4.1 Separation of Selection Decisions

We explicitly separate three distinct analytical questions:

```
[Decision 1: Winning Experimental Condition]
   -> Which factor combination produces the highest, most consistent validation performance?
   -> Answer: D0_H1 (Mean Val Macro-F1 = 93.02% ± 0.17%, lowest sigma across seeds).

[Decision 2: Highest Peak Individual Checkpoint]
   -> Which individual training trajectory attained the single highest validation score?
   -> Answer: D0_H0_seed44 (Best Val Macro-F1 = 93.38%, Epoch 31).

[Decision 3: Most Defensible Candidate for Independent Evaluation]
   -> Which model should be frozen to advance to the next independent evaluation?
   -> Answer: D0_H1_seed44 (from the winning condition D0_H1, Val Macro-F1 = 93.22%, Epoch 32).
```

### 4.2 Governance Disclosure on Selection Policy
In Phase 23 Section 5.3, the pre-registered model selection rule specified:
- *"The model state dict corresponding to the epoch with the highest `val_macro_f1` is saved as the official candidate checkpoint."*

This rule was defined **within-run** (selecting the best epoch during training). It did not specify an automated mathematical tie-breaking rule for selecting across conditions and seeds. Therefore, we explicitly record:
**Candidate selection across runs is post-hoc on validation.**

No evaluation on `synthetic_eval_test_v1` was used to influence this decision.

### 4.3 Candidate Comparison Table

| Candidate Checkpoint | Condition | Seed | Val Macro-F1 | Val Acc | TRAI F1 | PHAI F1 | LEN F1 | XUONG F1 | REJECT F1 | OOV FA Rate | KW Miss Rate | Checkpoint SHA-256 | Selection Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`D0_H1_seed44`** | `D0_H1` | 44 | **93.22%** | 93.12% | 94.62% | 88.00% | 94.04% | 96.25% | 93.18% | 26.95% | 2.50% | `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3` | **PRIMARY FROZEN CANDIDATE** (Representative of winning condition) |
| **`D0_H0_seed44`** | `D0_H0` | 44 | **93.38%** | 93.12% | 94.62% | 88.77% | 93.10% | 97.03% | 93.42% | 27.73% | 2.34% | `8b79c539765f04522826d6c610bd6a7e8603a0a72fb564a5fcc262bd65cfa30f` | **REFERENCE CANDIDATE** (Peak single-run score) |
| **`D0_H1_seed42`** | `D0_H1` | 42 | **92.95%** | 92.97% | 94.38% | 90.00% | 91.90% | 95.53% | 93.00% | 28.12% | 2.03% | `8d8fd08ae4cf586a53298c2679f8e0d992d3a05ceddce58f99b46305f5088e81` | Secondary replicate |
| **`D0_H1_seed43`** | `D0_H1` | 43 | **92.90%** | 93.05% | 92.26% | 89.94% | 93.08% | 96.32% | 93.19% | 27.73% | 1.72% | `8e5b21759f561b507be6ce41c7b5eb863ea8ec2e44a2dedc2eaf0a3c7616900b` | Secondary replicate |

### 4.4 Candidate Freezing Execution

Checkpoint **`D0_H1_seed44`** has been formally frozen to:
`experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/`

- **Frozen Checkpoint Path:** `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/candidate_checkpoint.pt`  
  SHA-256: `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3`
- **Candidate Metadata Path:** `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/candidate_metadata.json`
- **Checksums File:** `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/checksums.sha256`

The baseline $v11$ and $E1$ checkpoints and release models were **not** overwritten.

---

## 5. Conservative Evidence Interpretation

1. **Claim: D1 has a 10% Nano-derived treatment dose.**
   - *Supported.* $D_1$ contains 544 matched Nano samples (10.0% of $N=5,440$).
2. **Claim: The main effect of Factor D is $-0.70$ percentage points.**
   - *Supported as sample mean difference ($\bar{D}_1 - \bar{D}_0 = -0.70\%$).*
   - *Crucial Context:* Validation audio is synthesized primarily with VieNeu-TTS v3 Turbo voices. A slight performance drop on Turbo validation when substituting 10% of training data with Nano indicates an in-distribution domain gap, but **does not** prove that Nano data fails to improve out-of-distribution transfer to Nano or other acoustic domains.
3. **Claim: The main effect of Factor H is $-0.05$ percentage points.**
   - *Supported as global average difference ($\bar{H}_1 - \bar{H}_0 = -0.05\%$).*
   - *Crucial Context:* The global average conceals an interaction: $H_1$ is beneficial on $D_0$ ($+0.27\%$, variance reduced by $3.1\times$), while detrimental on $D_1$ ($-0.37\%$, increased keyword miss rate).
4. **Claim: D0_H1 has the highest condition-level mean.**
   - *Supported.* Mean validation macro-F1 is $93.02\% \pm 0.17\%$, outperforming $D0\_H0$ ($92.75\%$), $D1\_H0$ ($92.37\%$), and $D1\_H1$ ($92.00\%$).
5. **Claim: D0_H0_seed44 has the highest single-run validation score.**
   - *Supported.* Attained $93.38\%$, compared to $93.22\%$ for $D0\_H1\_seed44$.

---

## 6. Next Independent Evaluation Plan

### 6.1 Evaluation Source Availability Audit
- **Observed Test Set Quarantine:** `synthetic_eval_test_v1` is permanently classified as **OBSERVED** and cannot serve as an independent test set.
- **Local Engine Availability:** The repository currently has local offline models only for VieNeu-TTS v3 Turbo and VieNeu-TTS v3 Nano.
- **Third-Party Open-Source Engines:** External engines (such as Piper TTS Vietnamese or distinct acoustic vocoders) are not yet integrated or authorized locally. Cloud TTS APIs (e.g., `edge-tts`) require external network transmission and cannot guarantee offline reproducibility.
- **Audit Determination:** In accordance with Section 5 constraints:
```
================================================================================
STATUS: BLOCKED_PENDING_INDEPENDENT_EVALUATION_SOURCE
================================================================================
```
No newly assembled permutations of previously observed clips will be mischaracterized as independent validation.

### 6.2 Pre-Registered Protocol for Future Independent Test Corpus
When an authorized independent external TTS engine or natural speech test corpus becomes available, it must comply with this pre-registered protocol:
1. **Source Disjointness:** 0% overlap with any voice profile, speaker, or seed present in $D_0$, $D_1$, validation, or `synthetic_eval_test_v1`.
2. **Class Balance:** Exact 50:50 Target/REJECT balance.
3. **Decision Rule:** Strict $\theta_{\text{cmd}} = 0.20$ threshold; zero heuristic blacklists.
4. **Pre-Flight Hash Locking:** All manifest rows, audio WAVs, and metadata must be cryptographically hashed and committed to Git before running any model predictions.

---

## 7. Final Decision & Gate Sign-Off

All 12 training runs are fully audited and verified. Checkpoint provenance and architecture scale are confirmed. The winning condition ($D_0\_H_1$) and frozen candidate (`phase26_candidate_d0_h1_seed44`) have been cataloged without modifying baseline models. Because an independent external evaluation audio source is not currently available locally, the phase status is:

```
================================================================================
FINAL PHASE DECISION: PASS_TO_CANDIDATE_FREEZE_ONLY
================================================================================
```
