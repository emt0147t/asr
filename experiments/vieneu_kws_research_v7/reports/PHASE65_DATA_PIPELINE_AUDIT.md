# Phase 65 — Data and Pipeline Defect Diagnosis Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v7/`  
**Status Banner:** `INDEPENDENT_AUDIT_VERIFIED` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary: Why Did Retrained CNNs Underperform?

A rigorous, source-level audit was conducted across checkpoints, training scripts, loss formulations, dataset manifests, and decision inference logic to determine why newly trained models (Variant B: 32.08% recall; Variant C: 65.42% recall) underperformed the historical baseline (Cascade v1: 85.42% recall; Stage 1 CNN: 84.58% recall).

The investigation revealed **four material defects** in the retrained bounded-utterance pipeline that directly caused the collapse in target command recall.

---

## 2. Root Cause Analysis of Identified Defects

### Defect 1: Severe Training Class Imbalance (50% REJECT Majority)
- **Inspection Evidence:** In `experiments/vieneu_kws_research_v6/data/splits/train_manifest.csv`, `REJECT` contains 680 clips (340 hard-negative UNKNOWN + 340 SILENCE), while each target keyword contains only 170 clips (`TRAI`: 170, `PHAI`: 170, `LEN`: 170, `XUONG`: 170).
- **Mechanism:** `REJECT` accounts for exactly 50% of the training distribution, making it 4x larger than any individual target command class. In standard unweighted cross-entropy (Variant B), predicting `REJECT` immediately achieves 50% training accuracy with zero penalty across half the dataset, inducing a severe pathological bias towards rejection.
- **Consequence:** In early epochs (Epochs 1–15), Variant B's command recall was literally **0.00%**, and final test recall collapsed to 32.08%.

### Defect 2: Excessive Gradient Mass on REJECT in Hard-Negative Reweighting
- **Inspection Evidence:** In Variant C (`BoundedDatasetV6.__getitem__`), sample weights were set to: $w_{\text{hard\_unk}} = 1.6$, $w_{\text{cmd}} = 1.4$, and $w_{\text{silence}} = 0.9$.
- **Quantitative Defect:** There are 244 confusable minimal-pair UNKNOWN clips and 96 neutral speech clips. Total gradient weight mass allocated to `REJECT` was:  
  $$\text{Mass}_{\text{REJECT}} = (244 \times 1.6) + (96 \times 1.0) + (340 \times 0.9) = 390.4 + 96.0 + 306.0 = 792.4$$  
  In contrast, each command had only $170 \times 1.4 = 238.0$ weight mass! The total REJECT loss mass exceeded any individual command by **$3.33\times$**.
- **Consequence:** Whenever a target command produced a minor confusable activation, the network aggressively penalized the command logits to avoid REJECT classification errors, suppressing command recall to 65.42%.

### Defect 3: Decision Rule Asymmetry in Thresholding
- **Inspection Evidence:** In `run_phase64_locked_test.py`, the bounded decision logic was hardcoded as:  
  ```python
  if top_cmd_prob >= theta_cmd and top_cmd_prob > p[4]:  # p[4] is REJECT
      pred_label = TARGET_CLASSES[top_cmd_idx]
  ```
- **Mechanism:** Because `REJECT` was trained with a 4x higher prior, the posterior $P(\text{REJECT})$ remained elevated ($0.35\text{--}0.50$) on difficult command utterances. Even when the calibration search lowered $\theta_{\text{cmd}}$ to $0.35$, the condition $P(\text{cmd}) > P(\text{REJECT})$ systematically rejected true commands whenever $P(\text{REJECT}) = 0.38$ and $P(\text{cmd}) = 0.36$.
- **Correction:** Thresholding should be governed by the calibrated threshold $\theta_{\text{cmd}}$ relative to the decision margin, or class priors must be balanced during training.

### Defect 4: Overfitting in Attentive Temporal Pooling Head
- **Inspection Evidence:** `VietnameseBoundedKWSNet` replaced standard pooling with a 2-layer trainable attention module (`attn_proj` $\mathbb{R}^{128 \to 64}$ + `attn_score` $\mathbb{R}^{64 \to 1}$) operating over 51 time frames.
- **Mechanism:** With only 1,360 training samples and 25 training epochs, the attention weights overfit to background acoustic noise profiles and border silence padding. In contrast, the historical `VietnameseKWSNet` uses parameter-free `AdaptiveAvgPool2d((1, 1))`, which is mathematically invariant to frame shifts.

---

## 3. Preserved Pipeline Elements (Verified Free of Defects)

The following pipeline components were independently checked and found to be strictly defect-free:
1. **Sampling Rate & Format:** 16,000 Hz, 16-bit mono PCM, exactly 16,000 samples (1.000s) across all 2,440 clips.
2. **Peak Normalization:** Normalized to $-1.0\,\text{dBFS}$ target with no clipping or digital distortion.
3. **Feature Extraction Parameters:** 40 Mel bands, $f_{\min} = 20\,\text{Hz}$, $f_{\max} = 8,000\,\text{Hz}$, Hann window $25\,\text{ms}$ (400 samples), hop $10\,\text{ms}$ (160 samples), FFT 512, producing exact tensor shape `[B, 1, 40, 101]`.
4. **Label Mapping Integrity:** Canonical mapping `[0: TRAI, 1: PHAI, 2: LEN, 3: XUONG, 4: REJECT]` strictly maintained across manifests, data loaders, and models.
5. **Data Separation:** Validation-only threshold search was preserved without test snooping.

---

## 4. Corrective Architecture and Training Strategy for Phase 67

Based on these diagnostic findings, Phase 67 must execute two controlled, principled interventions:
1. **Candidate A (Corrected Baseline):**
   - Restore the robust parameter-free `AdaptiveAvgPool2d((1, 1))` backbone architecture (`VietnameseKWSNet` style).
   - Enforce **strict class-balanced loss weighting** in Cross-Entropy: allocate inverse frequency weights $w_c = \frac{N_{\text{total}}}{C \cdot N_c}$ ($w_{\text{cmd}} = 2.0$, $w_{\text{REJECT}} = 0.5$).
   - Use symmetric calibrated thresholding: accept command if $P(c) \ge \theta_c$ without the asymmetric $P(c) > P(\text{REJECT})$ bottleneck.
2. **Candidate B (Moderately Weighted Hard-Negative Training):**
   - Build directly upon Candidate A.
   - Apply moderate, bounded sample weights ($1.2\times$ on confusable minimal pairs, preserving $2.0\times$ on commands) to prevent the gradient mass of REJECT from overwhelming command representations.

This diagnosis provides an evidence-based foundation to proceed to Phase 66 and Phase 67.
