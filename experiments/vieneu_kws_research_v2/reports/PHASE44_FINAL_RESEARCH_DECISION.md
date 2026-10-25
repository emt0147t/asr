# Phase 44: Final Research Decision and Campaign Synthesis

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v2/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Authoritative Architectural Decision

### Official Verdict: **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**

Based on empirical evidence across 4 candidate architectures, 3 random seeds (42, 43, 44), and 4 continuous streaming challenge conditions (totaling $2,577.5\text{ s} \approx 0.716\text{ hours}$ of streaming evaluation), **none of the investigated candidate systems simultaneously satisfies the operational criteria**:
- Target command event recall $\ge 95\%$
- Non-target false activations $\le 1.0\text{ event/hour}$
- Mean event detection latency $\le 600\text{ ms}$

Consequently, all candidate configurations fail validation calibration and must be recorded as:
`NO_FEASIBLE_VALIDATION_OPERATING_POINT`.

---

## 2. Evidence-Based Answers to Research Questions

### Question 1: Did hard-negative training solve the speech-active false alarm bottleneck?
**Answer: No.**  
While hard-negative training produced clear, measurable qualitative benefits—most notably **complete 100% elimination of tonal false alarms** (`minimal_pair_tone`) and an **87.5% reduction in phrase-context false alarms**—it failed to resolve the fundamental problem:
- The non-target false alarm rate in continuous speech dropped from $102.80\text{ FA/hr}$ (Historical Cascade) to $62.72\text{ FA/hr}$ (Candidate 3). While an improvement, $62.72\text{ FA/hr}$ is still **over 60 times higher** than the requirement of $\le 1.0\text{ FA/hr}$.
- Crucially, this reduction in false alarms came at a fatal cost to detection recall: mean target command recall degraded from **$97.50\%$ down to $82.50\%$**, falling severely below the $\ge 95\%$ threshold.
- In summary, discriminative boundary shifting under a fixed acoustic feature space merely trades off false alarms against false rejections; it does not solve the underlying phonetic overlap between monosyllabic command words and Vietnamese speech.

### Question 2: Did character-level CTC solve the speech-active false alarm bottleneck?
**Answer: No.**  
Character-level CTC sequence modeling trained from scratch on this scale proved fundamentally ill-suited:
- **Severe optimization instability:** Across identical configurations varying only random seed, validation keyword accuracy swung wildly from $38.12\%$ (Seed 42) to $71.25\%$ (Seed 43) to $86.88\%$ (Seed 44).
- **Streaming sliding-window failure:** In continuous sliding-window streaming ($1.0\text{ s}$ window, $100\text{ ms}$ hop), a CTC model that learns to emit characters exhibits frequent spurious character activations on non-target speech, resulting in an unmanageable **$475.66\text{ FA/hr}$** for Seed 43. Conversely, under-converged seeds (42 and 44) achieved low false alarms only because they suffered near-total deafness to target commands (recall of $12.5\%$ and $38.1\%$, with mean latency exceeding $725\text{ ms}$).
- Training end-to-end CTC sequence heads without pretraining requires tens of thousands of diverse natural speech hours, which is completely mismatched to a 2,000-utterance synthetic domain.

### Question 3: Is Candidate 1 (Historical Cascade v1) or any other system deployable?
**Answer: None is deployable for continuous open-microphone operation.**  
- **Candidate 1 (Historical Cascade v1)** remains the highest-performing architecture on command recall ($97.50\%$) and latency ($404.6\text{ ms}$), but its continuous non-target false alarm rate of $\sim 102.8\text{ FA/hr}$ would cause a false trigger once every $\sim 35\text{ seconds}$ in continuous non-target speech environments.
- **Candidate 2 (Hard-Negative S1 alone)** is disastrous ($296.2\text{ FA/hr}$), demonstrating that a single-stage detector without temporal verification cannot function in streaming audio.
- **Candidate 3 (Hard-Negative Cascade)** fails recall ($82.50\%$) and remains at $62.72\text{ FA/hr}$.
- **Candidate 4 (CTC KWS)** fails both recall, latency, and stability.

---

## 3. Discrepancy & Metric Resolution Summary

| Discrepancy / Question | Audit / Empirical Finding in Phases 40–44 | Authoritative Status |
| :--- | :--- | :--- |
| **Stage 2 Architecture Identity** | Prior text referred colloquially to "BiGRU". Code audit authoritatively proved Stage 2 is a **4-layer Dilated 1D Temporal Convolutional Network (TCN)** with 340,773 parameters. | **Resolved: Dilated 1D-TCN** |
| **Stage 1 Parameter Count** | `VietnameseKWSNet` has exactly **138,534** parameters. | **Resolved: 138,534 params** |
| **Stage 2 Parameter Count** | `TemporalKWSVerifier` has exactly **340,773** parameters. | **Resolved: 340,773 params** |
| **Candidate B (CTC) Params** | `VietnameseCTCKWSNet` has exactly **229,115** parameters. | **Resolved: 229,115 params** |
| **VAD Gating Effect** | Phase 39 confirmed VAD degraded recall without reducing false alarms. | **Resolved: VAD unviable for cascade** |
| **Stream Duration Sufficiency** | $688.7\text{ s}$ challenge non-target stream ($0.1913\text{ hr}$) provides high density ($80$ distractors) but mathematical lower bound on 0 FP is $5.23\text{ FA/hr}$. | **Documented: Multi-hour stream required for $\le 1.0\text{ FA/hr}$ certification** |
| **Human Listening Evaluation** | The 80-clip listening test remains pending human review. | **Status: `PENDING_HUMAN_LISTENER_VERIFICATION`** |
| **Natural Speech Validation** | All training and testing in Phases 26–44 utilized synthetic VieNeu-TTS audio. | **Status: `NOT_VALIDATED_ON_NATURAL_SPEECH`** |

---

## 4. Why Synthetic-Data Metrics Cannot Establish Natural Speech Readiness

1. **TTS Artifacts & Phonetic Homogeneity:**  
   VieNeu-TTS synthesizes clear, studio-grade speech with regularized pitch contours, consistent formant transitions, and absent natural articulation anomalies (co-articulation, mumbling, throat-clearing, breath noise, speaker-room acoustic interaction). Models trained exclusively on synthetic audio learn sharp, un-generalizable acoustic boundaries that fail catastrophically when presented with natural human speech variability.
2. **Adversarial Linguistic Distribution:**  
   While the 5-family hard-negative corpus effectively tested minimal pairs and tone contrasts, natural spoken Vietnamese conversational dialogue features colloquial reductions, rapid phrase blending, dialectal vowel shifts (Northern vs. Central vs. Southern phonology), and ambient non-stationary acoustic noise that cannot be synthesized by rule.
3. **Formal Verification Gate:**  
   No model produced in this research campaign may be deployed to production, loaded into user-facing hardware, or represented as validated on natural speech until:
   - The 80-clip human listening benchmark is completed and verified.
   - Natural human field recordings with verified informed consent are unlocked, transcribed, and evaluated under an approved institutional protocol.

---

## 5. Technical Roadmap to Reach $\le 1.0\text{ FA/hr}$

To bridge the gap between the current state ($62.7\text{–}102.8\text{ FA/hr}$) and the operational goal ($\le 1.0\text{ FA/hr}$) without degrading target recall below $95\%$:

1. **Transition from Shallow Spectrogram Features to Pretrained Foundation Representations:**
   - Small CNNs (138k params) operating directly on 40-channel log-Mel filterbanks lack the phonetic capacity to reliably separate monosyllabic words with identical vowels and codas (e.g., *lên* vs. *lệnh* vs. *lên ngôi*).
   - Use a lightweight, distilled self-supervised speech encoder (e.g., pruned Wav2Vec2-Vietnamese, HuBERT-compact, or Conformer-CTC) as a frozen feature extractor. Pretrained representations capture contextual phonetic embeddings that robustly distinguish lexical boundaries.
2. **Continuous Multi-Frame Hypothesis Integration:**
   - Instead of sliding-window classification with fixed 2-frame lockout debounce, implement an explicit Hidden Markov Model (HMM) or Dynamic Time Warping (DTW) keyword decoder over posterior probabilities. Decoding keyword posteriors with explicit duration and silence penalties provides substantial immunity to transient acoustic bursts.
3. **Expanding the Hard-Negative Inventory to 1,000+ Phrases:**
   - The 100-text inventory demonstrated high efficacy for tonal and phrase context, but the vowel/coda and consonant categories remained undersaturated. An expanded dictionary of 1,000+ phonetically balanced non-target phrases is needed for discriminative verifier training.
4. **Natural Speech Validation Pipeline:**
   - Complete the human listening evaluation on the 80 synthetic review clips.
   - Proceed to natural human speech data collection under strict institutional privacy and consent governance.
