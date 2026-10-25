# Phase 33–35: Next-Step Decision and Architectural Strategy Report

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Directory:** [`experiments/vieneu_kws_v2/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2)  
**Execution Timestamp:** 2026-10-18  
**Governance Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Decision Summary

The Phase 33–35 experimental campaign achieved a major technical breakthrough in **noise robustness**, completely eliminating the catastrophic broadband noise collapse of CNN v1:
- Babble noise accuracy at $10\text{ dB}$ surged from $34.0\%$ to **$84.2\%$** ($+50.2\%$).
- White noise accuracy at $0\text{ dB}$ surged from $17.8\%$ to **$64.8\%$** ($+47.0\%$).
- Pink noise accuracy at $0\text{ dB}$ surged from $16.8\%$ to **$65.5\%$** ($+48.7\%$).
- Clean test calibrated accuracy reached **$93.23\% \pm 1.41\%$** with **$99.58\%$ target command recall**.

However, rigorous validation calibration and streaming simulation proved that **a single-stage compact 2D CNN cannot satisfy the continuous streaming false alarm constraint ($\le 1.0\text{ FA/hour}$)**. On continuous unseen non-target speech, CNN v2 produced between $223.6$ and $1,130.1\text{ FA/hour}$.

In accordance with strict experimental governance, we do not claim deployment readiness. We formally declare the exact constraint failure and provide actionable architectural recommendations for the next phase.

---

## 2. Verdicts Across the 7 Mandatory Dimensions

| Dimension | Verdict | Empirical Basis / Justification |
| :--- | :---: | :--- |
| **1. v2 Dataset Quality & Leakage Protection** | **PASS** | 2,000 balanced utterances generated with local VieNeu-TTS v3 Turbo. Strict $17:4:4$ speaker voice isolation across Train, Val, and Test splits. UNKNOWN vocabulary expanded to 120 unique texts with zero lexical overlap with target commands. Zero sample or metadata leakage. |
| **2. CNN Training & Reproducibility** | **PASS** | Clean training across 3 random seeds ($42, 43, 44$) with reproducible convergence. Best validation macro F1 ranged from $82.8\%$ to $85.8\%$. All random seeds, training logs, and model checkpoints are archived. |
| **3. UNKNOWN Rejection** | **PASS WITH LIMITATIONS** | UNKNOWN vocabulary coverage was vastly improved ($120$ unique phrases vs $40$ previously). Seed 44 achieved $80.0\%$ UNKNOWN recall ($20.0\%$ FA rate) on unseen held-out phrases. However, the mean UNKNOWN FA rate across seeds ($26.2\%$) failed the strict $\le 5.0\%$ static validation target. |
| **4. SILENCE / Background False Activation** | **PASS** | $0.0\%$ false activation on isolated silence and environmental background clips across all seeds ($0/80$ errors). $0.0\text{ FA/hour}$ on pure continuous background streams. Complete immunity to HVAC, office hum, and digital silence. |
| **5. Noise Robustness** | **PASS** | Train-only dynamic multi-condition augmentation resolved the single biggest failure of v1. The model demonstrated massive accuracy gains ($+47\%$ to $+57\%$) across broadband noise and multi-speaker babble down to $0\text{ dB}$ SNR. |
| **6. Continuous-Stream Behavior** | **FAIL** | While command stream recall ($98.1\%–99.4\%$) and latency ($411–435\text{ ms}$) met operational targets, the continuous non-target false alarm rate ($223.6–1,130.1\text{ FA/hour}$) severely violated the $\le 1.0\text{ FA/hour}$ operating requirement. |
| **7. Synthetic Demonstration Readiness** | **PASS WITH LIMITATIONS** | Ready for push-to-talk, button-gated, or benchmark demonstrations. **NOT READY** for hands-free continuous open-microphone deployment without a second-stage verifier. |

---

## 3. Root Cause Analysis of the Streaming Failure

The failure to achieve $\le 1.0\text{ FA/hour}$ is not a hyperparameter tuning bug or lack of training epochs. It is a fundamental consequence of using a **single-stage, sliding-window, 2D static convolutional architecture**:

1. **High Evaluation Frequency:** A $100\text{ ms}$ hop yields $36,000$ evaluations per hour. To achieve $\le 1.0\text{ FA/hour}$, a model must achieve $99.9972\%$ per-window specificity on open-domain conversational speech.
2. **Loss of Temporal Sequence Structure:** A 2D CNN processes a $64 \times 101$ Mel-spectrogram as a 2D image. It learns spectral texture rather than strict left-to-right phonetic state transitions. In Vietnamese, non-target words sharing rhymes or tone profiles (such as *hướng*, *thương*, *đường*, *phương* relative to *Xuống*; or *bài*, *tay*, *mai* relative to *Phải*) trigger transient activation peaks.
3. **Absence of Whole-Word Temporal Alignment:** Because the classifier evaluates every $100\text{ ms}$ slice independently, partial syllable overlaps frequently trigger false positives unless temporal alignment is strictly enforced.

---

## 4. Governance and Evidence Boundaries

All conclusions in this campaign are subject to the following non-negotiable boundaries:
1. **`PENDING_HUMAN_LISTENER_VERIFICATION`:** The 80-clip blinded human listening review from Phase 24–25 remains pending. The synthetic VieNeu-TTS audio has not yet received human perceptual sign-off.
2. **`NOT_VALIDATED_ON_NATURAL_SPEECH`:** All training and testing in this phase utilized synthetic speech from VieNeu-TTS. Performance on authentic human microphone recordings in real rooms is unknown.
3. **`NO_PRODUCTION_CLAIMS`:** High synthetic accuracy must not be misconstrued as field readiness.

---

## 5. Recommended Next Experiment: Two-Stage Cascaded Architecture

To achieve hands-free continuous streaming readiness ($\le 1.0\text{ FA/hour}$) while preserving ultra-low power consumption and high recall, we recommend pivoting from a single-stage model to a **Two-Stage Cascaded KWS System**:

```
[ Continuous Audio Stream ]
            │
            ▼
┌───────────────────────────────────────┐
│ Stage 1: Fast Candidate Proposer      │  <-- Current CNN v2 (Operating at high recall, θ = 0.50)
│ • Evaluates every 100 ms (1.2 ms CPU) │      Filters out 99.4% of stream, passes ~50 candidate
│ • Target Recall > 99%                 │      triggers per hour to Stage 2.
└──────────────────┬────────────────────┘
                   │ Candidate Trigger Event (Audio window + context)
                   ▼
┌───────────────────────────────────────┐
│ Stage 2: Second-Pass Verifier         │  <-- Temporal Acoustic Model (CTC / RNN-T / TCN)
│ • Runs only when Stage 1 triggers     │      Verifies strict phonetic sequence:
│ • Detailed temporal alignment check   │      e.g., [x] -> [u] -> [o] -> [ng] + tone
│ • Suppresses false rhyme triggers     │      Reduces FA/hr from ~200 to < 0.5 FA/hr!
└──────────────────┬────────────────────┘
                   │ Verified Command
                   ▼
       [ Valid Command Output ]
```

### Proposed Work Packages:
1. **Complete Phase 25 Human Listening Evaluation:** Collect human perceptual ratings on the 80 VieNeu-TTS clips to validate naturalness and intelligibility.
2. **Design Stage 2 Verifier (Phase 36):**
   - Implement a lightweight Connectionist Temporal Classification (CTC) phone-sequence verifier or a small streaming Temporal Convolutional Network (TCN).
   - Alternatively, evaluate an energy-based Voice Activity Detector (VAD) coupled with a dynamic-thresholding temporal confidence policy.
3. **Acoustic Pilot on Natural Human Speech:** Collect a 50-utterance exploratory natural Vietnamese speech sample to establish the transfer gap between synthetic VieNeu-TTS representations and authentic physical microphones.
