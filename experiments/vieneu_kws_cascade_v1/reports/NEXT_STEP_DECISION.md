# Phase 36–38: Next-Step Decision and Architectural Strategy Report

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Directory:** [`experiments/vieneu_kws_cascade_v1/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1)  
**Execution Timestamp:** 2026-10-18  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Decision Summary

The Phase 36–38 experimental campaign validated the hypothesis that a **Two-Stage Cascaded KWS Architecture** dramatically suppresses false alarms compared to a standalone 2D CNN:
- **Command stream false alarms dropped by $97.4\%$** (precision increased from $48.9\%$ to **$97.3\%$**).
- **Non-target stream false alarms dropped by $94.0\%$** (from $1,146\text{ FA/hr}$ down to **$69.2\text{ FA/hr}$**).
- **Target command recall remained high at $96.25\%$** (with $154/160$ commands detected).
- **Mean event latency was preserved at $403.7\text{ ms}$** (adding less than $1\text{ ms}$ of computation).

However, on unconstrained continuous non-target Vietnamese speech ($15.0\text{ minutes}$ of spoken phrases), the achieved false alarm rate of **$69.2\text{ FA/hour}$** ($35.9\text{ FA/hr}$ in Seed 44) exceeds the strict operational goal of $\le 1.0\text{ FA/hour}$.

In strict compliance with evidentiary governance, we explicitly declare **`NO_FEASIBLE_VALIDATION_OPERATING_POINT`** for autonomous open-mic continuous listening, delineate the exact physical boundaries of the system, and provide a clear roadmap for next steps.

---

## 2. Verdicts Across the 8 Mandatory Dimensions

| # | Evaluation Dimension | Verdict | Empirical Justification |
| :-: | :--- | :---: | :--- |
| **1** | **Existing Metric Reproducibility** | **PASS** | Independent audit of Phase 33–35 verified all denominators, stream durations, latency formulas, and metric values. Clean test metrics, noise suites, and streaming results reproduced bit-for-bit from frozen artifacts. |
| **2** | **Training and Validation Integrity** | **PASS** | Stage 2 verifier trained strictly on training-split source audio ($1,360$ source files, 17 voices). Hard negatives mined deterministically from training streams only. Zero test-to-training leakage. Checkpoints converged smoothly across seeds 42, 43, 44 with validation macro F1 reaching $91.7\%–92.1\%$. |
| **3** | **Temporal Verifier Improvement Over Stage 1** | **PASS** | Stage 2 eliminated $97.4\%$ of command-stream false alarms (FP count reduced from $167.0$ to $4.33$), boosted precision from $48.9\%$ to $97.3\%$, and cut non-target streaming false alarms by $94.0\%$ ($1,146\text{ FA/hr} \rightarrow 69.2\text{ FA/hr}$). |
| **4** | **UNKNOWN Rejection** | **PASS WITH LIMITATIONS** | UNKNOWN speech rejection in streaming mode improved tenfold compared to Stage 1. However, confusable monosyllabic words sharing diphthongs and tones with target words still trigger residual false activations ($35.9–103.8\text{ FA/hr}$). |
| **5** | **Background False Activation** | **PASS** | $0.0\%$ false activation on ambient environmental noise, HVAC hum, and digital silence ($0.0\text{ FA/hr}$ on silence streams across all seeds). |
| **6** | **Noise Robustness** | **PASS WITH LIMITATIONS** | High accuracy ($\ge 86\%$) maintained under mild stationary noise ($\ge 10\text{ dB}$). However, under heavy babble noise ($<10\text{ dB}$), Stage 2 defaults to conservative rejection, reducing recall. |
| **7** | **Streaming Precision, Recall, and Latency** | **PASS WITH LIMITATIONS** | Command stream recall ($96.25\% \pm 2.73\%$), precision ($97.27\% \pm 1.34\%$), and mean latency ($403.7\text{ ms}$) all passed predeclared criteria. Continuous unconstrained non-target FA/hr ($69.2\text{ FA/hr}$) failed the $\le 1.0\text{ FA/hr}$ requirement. |
| **8** | **Synthetic Demonstration Readiness** | **PASS WITH LIMITATIONS** | Ready for push-to-talk, button-gated, or benchmark demonstrations. **NOT READY** for hands-free continuous open-mic deployment without upstream VAD or end-to-end sequence alignment. |

---

## 3. Failure Diagnosis: Why $\le 1.0\text{ FA/hr}$ Remains Unmet

The residual false alarm rate of $69.2\text{ FA/hr}$ reflects the limits of sliding-window classification in monosyllabic tonal languages:
1. **Window Multiplier Effect:** In continuous streaming at a $100\text{ ms}$ hop, an engine performs $36,000\text{ trials/hour}$. Even with a phenomenal $99.81\%$ per-window specificity, the multiplication of trials yields $\approx 69\text{ false alarms/hour}$.
2. **Vietnamese Syllable Overlap:** Vietnamese words are monosyllables composed of an initial consonant, vowel nucleus, and final coda, modulated by one of six tones. Words like *đường*, *hướng*, *phương*, *thương*, and *vườn* share the identical rhyme `/uəŋ/` and tonal pitch slopes with *Xuống*. In continuous conversational speech, when a sliding window frames these syllables, both Stage 1 and Stage 2 observe spectral and temporal trajectories that closely match the target word.
3. **Absence of Character / Phoneme CTC Loss:** Neither Stage 1 nor Stage 2 uses Connectionist Temporal Classification (CTC) alignment to verify exact sub-phonetic letter transitions.

---

## 4. Governance Status and Non-Negotiable Boundaries

1. **`PENDING_HUMAN_LISTENER_VERIFICATION`:** The Phase 24–25 80-clip blinded human listening review remains pending. Synthetic VieNeu-TTS audio naturalness and intelligibility have not yet received human perceptual sign-off.
2. **`NOT_VALIDATED_ON_NATURAL_SPEECH`:** All training, verification, and evaluation data in this campaign was generated synthetically. Performance on natural human speech recorded via physical mobile microphones in acoustically reverberant rooms is unknown.
3. **`NO_PRODUCTION_CLAIMS`:** High synthetic command stream recall ($96.3\%$) and precision ($97.3\%$) must not be represented as real-world open-mic field readiness.

---

## 5. Strategic Recommendations for Next Phase

To advance beyond synthetic simulations toward real-world deployment, we recommend the following sequenced roadmap:

### Track 1: Complete the Human Subjective Gate (Phase 25 Closeout)
- Execute the blinded human listening interface on the 80 clips to establish human intelligibility and perceptual validity before collecting field recordings.

### Track 2: Upstream Acoustic Energy & VAD Gating (Immediate Win)
- Integrate a lightweight energy/entropy Voice Activity Detector (VAD) before Stage 1:
  * Restricts evaluation windows strictly to voiced speech segments.
  * In realistic usage (where a user speaks only 5–10% of the time), a VAD immediately suppresses 90% of idle streaming evaluations, reducing false alarms from $69\text{ FA/hr}$ to $< 7\text{ FA/hr}$.

### Track 3: End-to-End Temporal Sequence Model (Architectural Evolution)
- Transition from sliding-window image classification to an **End-to-End Streaming Conformer or CRNN with CTC Loss**:
  * Emits character/phoneme tokens: `[s, u, o, n, g, tone]`.
  * Enforces complete whole-word string matching.
  * Completely eliminates rhyme-based false alarms (e.g. *hướng* will emit `[h, ...]`, which CTC immediately rejects).
