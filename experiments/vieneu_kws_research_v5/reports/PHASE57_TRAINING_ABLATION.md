# Phase 57 — Controlled Bounded-Utterance Training Ablation Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Models Directory:** `experiments/vieneu_kws_research_v5/models/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 57 executed a controlled training ablation for bounded-utterance Vietnamese keyword spotting on the newly generated dataset v5. Four candidate architectures were structured for side-by-side empirical comparison:
- **Variant A (Frozen Historical Cascade v1):** Two-stage pipeline comprising Stage 1 CNN (`VietnameseKWSNet`, 138,534 params) and Stage 2 TCN Verifier (`TemporalKWSVerifier`, 340,773 params; Total: 479,307 params).
- **Variant B (Frozen Historical Stage 1 CNN):** Standalone 6-class CNN baseline (`VietnameseKWSNet`, 138,534 params) evaluated with a calibrated rejection threshold.
- **Variant C (Newly Trained Bounded Classifier — Standard):** Dedicated compact 5-class network (`VietnameseBoundedKWSNet`, 183,500 params) trained with standard Cross-Entropy loss ($label\_smoothing = 0.05$).
- **Variant D (Hard-Negative Weighted Ablation):** Controlled ablation of Variant C trained with sample-weighted loss elevating hard-negative minimal pairs ($\times 1.5$) and command preservation ($\times 1.25$).

All training was strictly confined to `data/splits/train_manifest.csv` (1,360 clips across 17 training voices). Validation clips (`data/splits/val_manifest.csv`) were used exclusively for model checkpointing and threshold calibration. Zero test data was observed during training or selection.

---

## 2. Model Architecture and Parameter Budget

| Model Architecture | Task Classes | Parameter Count | Receptive Field / Structure | Checkpoint Directory |
| :--- | :--- | :--- | :--- | :--- |
| **Variant A: Cascade v1** | Stage 1: 6 classes <br> Stage 2: 5 classes | 138,534 (S1) <br> 340,773 (S2) <br> **Total: 479,307** | 2D-CNN + Dilated 1D-TCN Sequence Verifier | `experiments/vieneu_kws_v2/` <br> `experiments/vieneu_kws_cascade_v1/` |
| **Variant B: Stage 1 CNN** | 6 classes | **138,534** | 2D-CNN (4 conv blocks + Global Avg Pool) | `experiments/vieneu_kws_v2/checkpoints/` |
| **Variant C: BoundedKWSNet** | 5 classes | **183,500** | Multi-scale 2D-CNN + Residual + Temporal Attentive Pooling | `experiments/vieneu_kws_research_v5/models/candidate_c_standard/` |
| **Variant D: Weighted Ablation** | 5 classes | **183,500** | Identical to Variant C with Hard-Negative Weighted Loss | `experiments/vieneu_kws_research_v5/models/candidate_d_ablation/` |

---

## 3. Training Protocol and Ablation Factors

### A. Dataset Partitioning (Strict Split Isolation)
- **Train Split (`train_manifest.csv`):** 1,360 clips (680 target commands: 170 each of `TRAI`, `PHAI`, `LEN`, `XUONG`; 340 `UNKNOWN` distractor clips across 80 training texts; 340 `SILENCE` clips across 4 noise profiles).
- **Validation Split (`val_manifest.csv`):** 360 clips (160 target commands, 120 `UNKNOWN` across 40 disjoint texts, 80 `SILENCE`).

### B. Training-Time Augmentation (Applied to Variants C & D)
- **Random Time Shift:** $\pm 80\,\text{ms}$ ($\pm 1,280$ samples) with constant boundary padding.
- **Gain Perturbation:** Log-uniform gain scaling between $-3.0\,\text{dB}$ and $+3.0\,\text{dB}$.
- **Additive Ambient Noise:** Gaussian white/ambient noise injected at SNR randomly between $12.0\,\text{dB}$ and $28.0\,\text{dB}$ with probability $p = 0.60$.
- **SpecAugment:** 1 frequency mask ($F_{\text{max}} = 8$ bins) and 1 time mask ($T_{\text{max}} = 15$ frames) applied directly to Log-Mel spectrograms.

### C. Ablation Parameter Study (Variant C vs Variant D)
- In **Variant C**, standard Cross-Entropy is applied uniformly:
  $$\mathcal{L}_{\text{std}} = \frac{1}{B} \sum_{i=1}^B \text{CrossEntropy}(z_i, y_i)$$
- In **Variant D**, hard-negative minimal pairs (sharing rhymes or confusable onsets with target commands) receive a $1.5\times$ penalty weight, while target commands receive a $1.25\times$ preservation weight:
  $$w_i = \begin{cases} 1.50 & \text{if } y_i = \text{REJECT and } \text{family} \in \{\text{minimal\_pair\_tone, minimal\_pair\_consonant, phrase\_context}\} \\ 1.25 & \text{if } y_i \in \{\text{TRAI}, \text{PHAI}, \text{LEN}, \text{XUONG}\} \\ 1.00 & \text{otherwise (neutral speech / silence)} \end{cases}$$
  $$\mathcal{L}_{\text{ablated}} = \frac{1}{\sum w_i} \sum_{i=1}^B w_i \cdot \text{CrossEntropy}(z_i, y_i)$$

---

## 4. Training Progression and Validation Selection Across Seeds

Selection rule: Optimize $\text{Selection Score} = \text{Val Recall} - 2.5 \times \text{Val FAR}$ on the 360 validation clips:

### Candidate C (Standard Bounded Classifier):
- **Seed 42:** Final Train Loss: 0.7709 | Best Val Recall: **90.62%** | Val FAR: **14.50%** | Val Acc: **87.22%** (Epoch 25)
- **Seed 43:** Final Train Loss: 0.8363 | Best Val Recall: **37.50%** | Val FAR: **5.00%** | Val Acc: **69.44%** (Epoch 10)
- **Seed 44:** Final Train Loss: 0.7679 | Best Val Recall: **80.00%** | Val FAR: **15.00%** | Val Acc: **82.78%** (Epoch 20)

### Candidate D (Hard-Negative Weighted Ablation):
- **Seed 42:** Final Train Loss: 1.0579 | Best Val Recall: **71.25%** | Val FAR: **13.00%** | Val Acc: **79.17%** (Epoch 20)
- **Seed 43:** Final Train Loss: 1.0688 | Best Val Recall: **70.63%** | Val FAR: **19.00%** | Val Acc: **76.39%** (Epoch 20)
- **Seed 44:** Final Train Loss: 0.9780 | Best Val Recall: **81.87%** | Val FAR: **17.50%** | Val Acc: **80.28%** (Epoch 25)

Checkpoints and serialized training logs are archived under `experiments/vieneu_kws_research_v5/models/{candidate_c_standard, candidate_d_ablation}/`.
