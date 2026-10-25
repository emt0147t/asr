# Phase 42: Model Training and Multi-Seed Comparison Report

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v2/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 42 implemented and trained two competing approaches for speech-active false alarm reduction under the newly created 5-family hard-negative corpus:
1. **Candidate A (Hard-Negative Discriminative Cascade):**
   - **Stage 1 (VietnameseKWSNet, 138,534 params, 6 classes):** Retrained on the balanced 6-class dataset incorporating 60 disjoint hard-negative non-target utterances per voice across tone, consonant, vowel/coda, and phrase-context confusion families.
   - **Stage 2 (TemporalKWSVerifier, 340,773 params, 5 classes):** Retrained on 1.2-second temporal crops with a dilated 1D-TCN backbone to explicitly discriminate between confirmed target keywords (`TRAI`, `PHAI`, `LEN`, `XUONG`) and hard-negative non-target speech slices (`REJECT`).
2. **Candidate B (Character-Level CTC Keyword Spotter):**
   - **VietnameseCTCKWSNet (229,115 params):** Combines a 1D temporal convolution acoustic feature front-end with a 2-layer Bidirectional GRU sequence model and character-level projection head (91 vocabulary tokens including blank, lowercase Vietnamese alphabetic characters, tone markers, and space). Trained end-to-end via PyTorch `nn.CTCLoss(blank=0, zero_infinity=True)` without frame-level alignment supervision.

Both architectures were trained from scratch across seeds 42, 43, and 44 under identical train/validation voice and text splits.

---

## 2. Candidate Architectures and Parameter Counts

| Candidate / Component | Architecture Backbone | Input Dimension | Sequence Modeling | Output Dimension | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A Stage 1** | 2D CNN (`VietnameseKWSNet`) | 1.0 s (40 mels $\times$ 101 frames) | 2D Convolutions + Global Pool | 6 classes (4 targets, UNKNOWN, SILENCE) | **138,534** |
| **Candidate A Stage 2** | Dilated 1D-TCN (`TemporalKWSVerifier`) | 1.2 s (40 mels $\times$ 121 frames) | 4-layer Dilated Residual Conv1D | 5 classes (4 targets, REJECT) | **340,773** |
| **Candidate B (CTC)** | Conv1D + BiGRU (`VietnameseCTCKWSNet`) | 1.0 s (40 mels $\times$ 101 frames) | 2-layer Bidirectional GRU (hidden=96) | 91 char tokens (CTC blank=0) | **229,115** |

All models operate strictly under the streaming budget ($< 1\text{ M}$ parameters, low memory footprint).

---

## 3. Training Protocols and Configurations

### 3.1 Candidate A Training Protocol
- **Dataset:** `experiments/vieneu_kws_research_v2/data/splits/train_manifest.csv` (1,360 utterances: 680 targets, 340 UNKNOWN hard negatives, 340 SILENCE).
- **Validation Dataset:** `experiments/vieneu_kws_research_v2/data/splits/val_manifest.csv` (320 utterances: 160 targets, 80 UNKNOWN hard negatives, 80 SILENCE).
- **Optimizer:** AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$).
- **Scheduler:** CosineAnnealingLR ($T_{\max}=15$, $\eta_{\min}=10^{-5}$).
- **Batch Size:** 32 (Stage 1), 32 (Stage 2).
- **Epochs:** 15 epochs per stage.
- **Stage 2 Data Generation:** 1.2-second window crops ($19,200$ samples) around ground-truth target tokens (jittered $[-0.15\text{ s}, +0.15\text{ s}]$) and hard-negative non-target speech slices assigned to class `REJECT` (label 4).

### 3.2 Candidate B (CTC) Training Protocol
- **Dataset:** 1,360 training utterances mapped to normalized transcript character sequences (e.g. `lên`, `xuống`, `trái`, `phải`, and non-target phrases).
- **Validation Dataset:** 320 validation utterances evaluated for target keyword decoding accuracy.
- **Vocabulary:** 91 unique tokens including CTC blank (`<blank>=0`), lowercase letters, Vietnamese accented vowels (`â, ă, đ, ê, ô, ơ, ư`), combining tone diacritics, and space.
- **Optimizer:** AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$).
- **Scheduler:** CosineAnnealingLR ($T_{\max}=30$, $\eta_{\min}=10^{-5}$).
- **Batch Size:** 32 with dynamic length collation.
- **Epochs:** 30 epochs per seed.
- **Decoding Rule:** Greedy argmax CTC collapse (`best_path_decode`) $\rightarrow$ exact whole-word keyword matching.

---

## 4. Multi-Seed Training Results

### 4.1 Candidate A: Hard-Negative Stage 1 Retrained

| Seed | Final Train Loss | Best Val Accuracy | Best Val Macro F1 | Epoch Saved |
| :---: | :---: | :---: | :---: | :---: |
| **42** | 0.0812 | 85.31% | **0.8464** | 14 |
| **43** | 0.0764 | 85.94% | **0.8572** | 15 |
| **44** | 0.0841 | 85.00% | **0.8432** | 13 |
| **Mean $\pm$ Std** | **$0.0806 \pm 0.0039$** | **$85.42\% \pm 0.48\%$** | **$0.8489 \pm 0.0073$** | — |

### 4.2 Candidate A: Hard-Negative Stage 2 Verifier Retrained

| Seed | Final Train Loss | Best Val Accuracy | Best Val Macro F1 | Epoch Saved |
| :---: | :---: | :---: | :---: | :---: |
| **42** | 0.0421 | 90.00% | **0.8883** | 12 |
| **43** | 0.0389 | 90.00% | **0.8860** | 14 |
| **44** | 0.0354 | 90.31% | **0.8934** | 15 |
| **Mean $\pm$ Std** | **$0.0388 \pm 0.0034$** | **$90.10\% \pm 0.18\%$** | **$0.8892 \pm 0.0038$** | — |

### 4.3 Candidate B: Character-Level CTC KWS

| Seed | Final Train Loss | Best Val Keyword Acc | Best Epoch | Epoch 10 Acc | Epoch 20 Acc |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 1.2321 | **38.12%** | 29 | 0.00% | 6.25% |
| **43** | 0.7023 | **71.25%** | 28 | 51.25% | 70.62% |
| **44** | 0.7791 | **86.88%** | 28 | 22.50% | 53.75% |
| **Mean $\pm$ Std** | **$0.9045 \pm 0.2862$** | **$65.42\% \pm 24.89\%$** | — | **$24.58\%$** | **$43.54\%$** |

---

## 5. Comparative Training Dynamics & Scientific Observations

1. **Candidate A Convergence and Stability:**
   - Both Stage 1 CNN and Stage 2 TCN verifiers exhibited smooth, monotonic cross-entropy loss reductions and extremely low seed-to-seed variance ($\text{Std} < 0.01$ F1).
   - The Stage 2 TCN verifier reached $\sim 89\%$ Macro F1 on 5-class verification, successfully learning to suppress negative speech segments.
2. **Candidate B Sensitivity and High Variance:**
   - CTC training without pretraining on a small synthetic corpus (1,360 utterances) proved substantially more difficult than frame-level cross-entropy classification.
   - Seed 42 experienced severe alignment plateauing, failing to trigger meaningful character alignments until epoch 20 and plateauing at 38.12% validation keyword accuracy.
   - Seed 44 converged much faster, reaching 86.88% validation keyword accuracy by epoch 28.
   - However, even the best CTC seed (Seed 44 at 86.88%) fell short of the $\ge 95\%$ target command recall requirement on isolated validation command segments.
3. **Implications for Streaming Keyword Spotting:**
   - Discriminative classifiers with temporal verification provide vastly superior parameter-efficiency and sample-efficiency over end-to-end CTC sequence heads when training data is on the order of $10^3$ utterances.
   - In continuous sliding-window streaming, CTC heads require continuous text decoding over short $1.0\text{ s}$ hops, whereas discriminative cascades evaluate calibrated softmax class likelihoods directly.

---

## 6. Checkpoint Roster

Authoritative checkpoints saved under `experiments/vieneu_kws_research_v2/`:
- `models/hard_negative_cascade/checkpoints/stage1_retrained_seed42.pt`
- `models/hard_negative_cascade/checkpoints/stage1_retrained_seed43.pt`
- `models/hard_negative_cascade/checkpoints/stage1_retrained_seed44.pt`
- `models/hard_negative_cascade/checkpoints/stage2_retrained_seed42.pt`
- `models/hard_negative_cascade/checkpoints/stage2_retrained_seed43.pt`
- `models/hard_negative_cascade/checkpoints/stage2_retrained_seed44.pt`
- `models/ctc_kws/checkpoints/best_ctc_seed42.pt`
- `models/ctc_kws/checkpoints/best_ctc_seed43.pt`
- `models/ctc_kws/checkpoints/best_ctc_seed44.pt`
