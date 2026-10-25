# PHASE 28–31 — CROSS-TTS EVALUATION PREREGISTRATION & LOCKED EXPERIMENT PROTOCOL

**Date:** October 25, 2026  
**Workspace:** `D:\trainidentificationaudio`  
**Work Package:** Phase 28–31 — Cross-TTS Error Audit, Domain-Augmented Training and Locked Evaluation  
**Target:** `experiments/vieneu_kws_research_v12/reports/PHASE28_31_EVALUATION_PREREGISTRATION.md`  
**Current Candidate:** `D0_H1_seed44` (`00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3`)  
**Architecture Policy:** `DEFER_E2_CAPACITY` (Strictly fixed at 138,469 parameters)  
**Inference Threshold:** $\theta_{\text{cmd}} = 0.20$ strictly locked  

---

## 0. Mandatory Governance Declarations

- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

This protocol governs strictly offline evaluation using newly synthesized speech from an authorized, reproducible text-to-speech engine entirely external and independent of VieNeu-TTS. It does not evaluate or authorize deployment on natural human speech.

---

## 1. Scientific Objectives & Hypotheses

### 1.1 Context and Motivation
Phase 27 evaluated the Phase 26 candidate (`D0_H1_seed44`) on an independent TTS distribution (Microsoft Edge-TTS 7.3.1) and revealed two primary findings:
1. When tested under unsegmented sliding-window evaluation without spectral band gating, false alarms were inflated due to mic handling noise in low-SNR segments; re-evaluating with the canonical Profile B `SpeechBandRatioVAD` from `selected_phase15_vad_config.json` cuts silence FA by $>97\%$ down to **1.52 FA/min**.
2. However, cross-TTS OOV false alarm rate remained high (31.25% overall OOV FA, 34.82% on phonetic hard negatives, 47.88 FA/min on continuous OOV speech), indicating that models trained solely on VieNeu-TTS acoustic representations suffer from out-of-domain acoustic brittleness.

Stage B prepared $D_2$ by substituting 10.0% ($N=544$) of the training data with Edge-TTS audio using strictly non-test training lexemes. This work package tests whether domain-augmented training ($D_2$) combined with hard-negative mining ($H_1$) resolves cross-TTS degradation while preserving high in-domain precision.

### 1.2 Formal Research Hypotheses
1. **$H_{\text{domain}}$ (Factor D Main Effect):**  
   Models trained on $D_2$ (10.0% Edge-TTS domain augmentation) will exhibit significantly lower OOV false alarm rates on independent Edge-TTS speech compared to models trained on $D_0$, without degrading target command recall.
2. **$H_{\text{hard}}$ (Factor H Main Effect):**  
   Models trained with $H_1$ (phonetic hard negative sampling) will achieve significantly higher Macro-F1 and lower phonetic OOV false alarms than $H_0$ models under independent acoustic conditions.
3. **$H_{\text{interaction}}$ ($D \times H$ Interaction):**  
   The combination $D_2 \times H_1$ will yield synergistic gains, achieving the lowest streaming OOV FA/min and the highest Macro-F1 across all 4 experimental conditions.

---

## 2. Experimental Design ($2 \times 2$ Factorial)

The controlled experiment follows a balanced $2 \times 2$ factorial design across 3 random seeds (42, 43, 44), yielding exactly 12 training runs:

| Condition | Training Distribution | Negative Sampling Strategy | Seeds |
| :--- | :--- | :--- | :--- |
| **D0_H0** | $D_0$ (100% VieNeu-TTS Turbo, $N=5,440$) | Standard random negative sampling | 42, 43, 44 |
| **D0_H1** | $D_0$ (100% VieNeu-TTS Turbo, $N=5,440$) | Phonetic hard-negative oversampling | 42, 43, 44 |
| **D2_H0** | $D_2$ (10.0% Edge-TTS Augmented, $N=5,440$) | Standard random negative sampling | 42, 43, 44 |
| **D2_H1** | $D_2$ (10.0% Edge-TTS Augmented, $N=5,440$) | Phonetic hard-negative oversampling | 42, 43, 44 |

### 2.1 Model Architecture & Capacity Freeze
- Model: `CorrectedVietnameseKWSNet` (Phase 24–26 architecture)
- Parameters: **Strictly 138,469 parameters** (`DEFER_E2_CAPACITY` maintained)
- Frontend: Canonical Slaney Mel-spectrogram (80 bins, 16 kHz mono, hop=160, win=400)
- Input: 80 Mel bins $\times$ 101 time frames (1.00s audio)

### 2.2 Training Hyperparameters (Pre-Registered)
- Optimizer: AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 0.01$)
- Scheduler: CosineAnnealingLR ($T_{\max} = 40$, $\eta_{\min} = 10^{-5}$)
- Batch size: 64
- Max epochs: 35
- Early stopping: Patience 8 epochs based on `val_macro_f1`
- Loss function: CrossEntropyLoss with label smoothing 0.05

---

## 3. Evaluation Corpus Specification (`edge_tts_eval_v2`)

To ensure complete independence and prevent test-set reuse:
1. **Lexical Isolation:** Zero lexemes from training ($D_0$, $D_1$, $D_2$) or Phase 21/Phase 27 test sets are reused for OOV confusers.
2. **Prosodic Variation:** Test audio includes both normal rate and conversational rate ($+10\%$) variations across male and female Edge-TTS voices (`vi-VN-HoaiMyNeural` and `vi-VN-NamMinhNeural`).
3. **Acoustic Noise Isolation:** Mixed with held-out mic noise tracks 36–39, which have zero exposure in any training or prior evaluation manifest.

### 3.1 Lexicon Table

| Category | Count | Lexemes | Description |
| :--- | :--- | :--- | :--- |
| **Target Commands** | 4 | `trái`, `phải`, `lên`, `xuống` | 4 in-domain commands |
| **Phonetic Confusers (TRAI)** | 4 | `trước`, `trường`, `trưa`, `trẻ` | Onset /r/ and vowel confusions |
| **Phonetic Confusers (PHAI)** | 4 | `phim`, `phút`, `phương`, `phát` | Onset /f/ and tone confusions |
| **Phonetic Confusers (LEN)** | 4 | `lớp`, `luôn`, `làm`, `lòng` | Onset /l/ and rhyme confusions |
| **Phonetic Confusers (XUONG)** | 4 | `xem`, `xanh`, `xuyên`, `xa` | Onset /s/ and glide confusions |
| **General In-Domain OOVs** | 10 | `cây`, `hoa`, `núi`, `sông`, `biển`, `đèn`, `quạt`, `bóng`, `áo`, `quần` | Unseen everyday Vietnamese nouns |

### 3.2 Isolated Test Corpus (`edge_tts_eval_v2_isolated_test_manifest.csv`)
- 4 Commands $\times$ 2 Voices $\times$ 2 Rates $\times$ 4 SNR tiers (Clean, 15 dB, 10 dB, 5 dB) = 64 command clips.
- 16 Phonetic OOVs $\times$ 2 Voices $\times$ 2 Rates $\times$ 4 SNR tiers = 256 phonetic OOV clips.
- 10 General OOVs $\times$ 2 Voices $\times$ 2 Rates $\times$ 4 SNR tiers = 160 general OOV clips.
- **Total isolated clips:** Exactly **480 audio clips** ($1.00\text{s}$ duration each).

### 3.3 Streaming Test Corpus (`edge_tts_eval_v2_streaming_manifest.csv`)
- **50 continuous streaming sequences** (nominal duration 8.0s to 16.0s, total $\approx 10.5$ minutes).
- Stratified into:
  - 10 Pure background/silence sequences (baseline false activation test).
  - 10 Pure OOV conversational sequences (conversational speech rejection test).
  - 15 Clean multi-command sequences with interspersed OOVs.
  - 15 Noise-mixed multi-command sequences (SNR 15 dB, 10 dB, 5 dB).
- Inter-utterance gaps: $0.3\text{s}$ to $1.5\text{s}$.
- Noise tracks: Exclusively held-out mic-noise tracks 36–39.

---

## 4. Canonical Metric Definitions & Decision Standards

1. **Four-Command True Macro-F1:**
   $$\operatorname{Macro-F1} = \frac{1}{4} \sum_{c \in \{\text{TRAI, PHAI, LEN, XUONG}\}} F1_c$$
2. **Overall Macro-F1 (5-Class):**
   $$\operatorname{Macro-F1}_5 = \frac{1}{5} \sum_{c \in \{\text{TRAI, PHAI, LEN, XUONG, REJECT}\}} F1_c$$
3. **OOV False Alarm Rate:**
   $$\text{FAR}_{\text{OOV}} = \frac{\sum \text{False Alarms on OOV}}{\text{Total OOV Clips (416)}}$$
4. **Phonetic Hard Negative False Alarm Rate:**
   $$\text{FAR}_{\text{phonetic}} = \frac{\sum \text{False Alarms on Phonetic Hard OOVs}}{\text{Total Phonetic OOV Clips (256)}}$$
5. **Streaming False Activation Rates:**
   - Silence FA/min $= \frac{\text{False Alarms in Pure Silence Sequences}}{\text{Total Duration of Silence (min)}}$
   - OOV FA/min $= \frac{\text{False Alarms in OOV Sequences}}{\text{Total Duration of OOV Speech (min)}}$
6. **Streaming Precision, Recall, and Latency:**
   - Event tolerance window: $[-0.15\text{s}, +0.45\text{s}]$ relative to speech event end.
   - Mean latency: Time from speech onset to trigger frame.

---

## 5. Candidate Selection & Unblinding Protocol

1. **Pre-Test Candidate Policy:**
   - Candidate selection MUST take place prior to unblinding `edge_tts_eval_v2`.
   - The winning training condition will be selected based on Factorial ANOVA and mean validation Macro-F1 across the 3 seeds.
   - The best individual seed run of that condition will be designated as `phase28_31_candidate`, its weights frozen, and its ONNX model exported.
2. **Benchmarking Suite:**
   The frozen test suite will evaluate:
   - Baseline $v11$ (Phase 11 release)
   - Baseline $E1$ (Phase 14 release)
   - Phase 26 Candidate (`D0_H1_seed44`)
   - Phase 28–31 Winning Candidate (`phase28_31_candidate`)
3. **Success Criteria for Advancing Gate:**
   - Candidate Macro-F1 on `edge_tts_eval_v2` $\ge 70.0\%$.
   - Candidate Phonetic OOV FAR $< 25.0\%$ (significant improvement over Phase 27's $34.8\%$).
   - Silence FA/min $< 3.0$ FA/min under canonical `SpeechBandRatioVAD`.
   - Streaming command event recall $\ge 80.0\%$.
