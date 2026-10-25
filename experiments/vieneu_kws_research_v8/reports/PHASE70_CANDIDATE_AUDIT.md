# Phase 70 — Candidate B Architecture and Checkpoint Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  
**Candidate Under Audit:** `candidate_b_moderately_weighted_hard_negative` (Phase 67/69 Checkpoints)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Audit Purpose

Phase 70 independently audits Candidate B (`CorrectedVietnameseKWSNet`, 138,469 parameters), inspecting its exact PyTorch checkpoints, inference logic, validation calibration records, locked-test predictions, and dataset manifests across seeds 42, 43, and 44.

### Predeclared Acceptance Criteria:
- **Recall Target:** $\ge 95.0\%$ for EVERY target keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`), assessed both per seed and in aggregate.
- **Non-Target Clip FAR Ceiling:** $\le 5.0\%$ (at most 28 false acceptances out of 560 non-target clips).
- **CPU Latency Budget:** $\le 50.0\,\text{ms}$ mean post-capture CPU decision latency.

### Audit Verdict Summary:
- **Bit-Exact Reproduction:** 100% agreement (0 mismatches across 2,160 evaluations).
- **Did All Seeds Meet Stated Target?** **NO.**
  - Seed 42 achieved only **92.50% recall on PHAI** (37/40 hits; 3 confused as TRAI).
  - All three seeds exceeded the 5.0% FAR ceiling (Seed 42: 5.18%, Seed 43: 6.96%, Seed 44: 5.54%).
  - Mean FAR is **5.89% ± 0.94%**.
  - Mean CPU latency is **1.65 ± 0.03 ms**, well within the 50 ms budget.

---

## 2. Checkpoint & Architecture Inspection

- **Architecture:** `CorrectedVietnameseKWSNet` with parameter-free `AdaptiveAvgPool2d((1, 1))` pooling.
- **Parameters:** Exactly 138,469 trainable parameters across 5 Conv2D blocks.
- **Checkpoints Audited:**
  - `experiments/vieneu_kws_research_v7/models/checkpoints/candidate_b_seed42.pt` (Epoch 20)
  - `experiments/vieneu_kws_research_v7/models/checkpoints/candidate_b_seed43.pt` (Epoch 24)
  - `experiments/vieneu_kws_research_v7/models/checkpoints/candidate_b_seed44.pt` (Epoch 21)
- **Loss Function Weights:** Inverse class weighting with moderate hard-negative weight ($w_{\text{cmd}}=2.0, w_{\text{hn}}=1.2, w_{\text{silence}}=0.5$).

---

## 3. Disaggregated Per-Seed & Per-Keyword Audit

| Seed | Threshold $\theta_{\text{cmd}}$ | Macro Recall | Micro Recall | TRAI | PHAI | LEN | XUONG | Non-Target FAR | Raw FA (out of 560) | Target False Rejections | Wrong Cmd Labels | 5-Class Accuracy | Mean CPU Latency | Strict Pass? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 42** | 0.5 | 98.12% | 98.12% | 100.0% | 92.5% | 100.0% | 100.0% | 5.18% | 29/560 | 0 | 3 | 95.56% | 1.56 ms | NO |
| **Seed 43** | 0.35 | 99.38% | 99.38% | 100.0% | 97.5% | 100.0% | 100.0% | 6.96% | 39/560 | 0 | 1 | 94.44% | 1.54 ms | NO |
| **Seed 44** | 0.35 | 100.00% | 100.00% | 100.0% | 100.0% | 100.0% | 100.0% | 5.54% | 31/560 | 0 | 0 | 95.69% | 1.64 ms | NO |
| **Mean $\pm$ Std** | — | **99.17% $\pm$ 0.96%** | **99.17%** | 100.0% | 96.7% | 100.0% | 100.0% | **5.89% $\pm$ 0.94%** | 33.0/560 | 0.0 | 1.3 | 95.23% $\pm$ 0.69% | 1.58 $\pm$ 0.05 ms | **NO** |

---

## 4. Key Diagnostic Insights from the Error Breakdown

1. **Zero Target False Rejections:** Across all 3 seeds (480 command evaluations total), exactly 0 target command clips were rejected into `REJECT`. The models never failed to detect speech when a command was spoken.
2. **Acoustic Confusion Between TRAI and PHAI:** In Seed 42, 3 clips of `PHAI` were misclassified as `TRAI`. In Seed 43, 1 clip of `PHAI` was misclassified as `TRAI`. This acoustic overlap is the sole reason Seed 42 fell below 95% on PHAI (92.5%).
3. **Non-Target False Acceptance Skew:** Across all seeds, non-target false acceptances were overwhelmingly concentrated in `XUONG` (19 in seed 42, 19 in seed 43, 13 in seed 44) and `TRAI` (7, 12, 5), with virtually zero false acceptances into `PHAI` (2, 0, 0). Vietnamese non-target words ending in velar nasal /ŋ/ and falling tones trigger false activations on `XUONG`.
4. **Denominators and Step Sizes:** On 560 non-target clips, each clip represents exactly $1/560 \approx 0.1786\%$. The target of $\le 5.0\%$ corresponds to $\le 28$ false alarms. The actual false alarm counts were 29, 39, and 31.

---

## 5. Statistical Sampling Unit Audit

- The benchmark test set consists of 720 unique clips (160 command clips, 560 non-target clips).
- Evaluating 3 random seeds on the same 720 clips produces 2,160 predictions, but these are **not 2,160 independent observations**.
- A naive binomial Wilson score interval calculated over $N=2,160$ would artificially underestimate the standard error by a factor of $\sqrt{3} \approx 1.73$.
- **Methodological Mandate:** Reports must present per-seed Wilson intervals for the independent 720-clip evaluations, accompanied by the sample standard deviation across seeds ($s = 0.94\%$ for FAR).
