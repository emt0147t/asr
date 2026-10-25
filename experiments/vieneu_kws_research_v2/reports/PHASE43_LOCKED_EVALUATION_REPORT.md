# Phase 43: Locked Continuous-Stream Benchmark and Calibration Report

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v2/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 43 evaluated four candidate systems across random seeds 42, 43, and 44 on continuous streaming audio using the newly synthesized hard-negative challenge benchmark:
1. **Candidate 1 (Historical Cascade v1):** Frozen baseline (Stage 1 CNN v2 + Stage 2 TCN verifier v1).
2. **Candidate 2 (Hard-Negative Stage 1):** Retrained 6-class CNN v2 alone with hard-negative exposure.
3. **Candidate 3 (Hard-Negative Cascade):** Retrained Stage 1 CNN + Retrained Stage 2 TCN verifier.
4. **Candidate 4 (Character-Level CTC KWS):** End-to-end character CTC sequence model.

Evaluation was performed on 4 distinct continuous streams:
- `val_stream_continuous_commands.wav`: 601.4 s (160 validation target command events)
- `val_stream_continuous_nontarget.wav`: 697.8 s (80 validation hard-negative distractor events)
- `test_stream_continuous_commands.wav`: 589.6 s (160 locked test target command events)
- `test_stream_continuous_nontarget.wav`: 688.7 s (80 locked test hard-negative distractor events)

### Primary Operational Criteria
- **Target Event Recall:** $\ge 95\%$
- **Non-Target False Activations:** $\le 1.0\text{ event/hour}$
- **Mean Event Detection Latency:** $\le 600\text{ ms}$

### Calibration Finding
On validation stream search, **no operating point** simultaneously met all three requirements for any candidate. Consequently, the calibration protocol locked:
`NO_FEASIBLE_VALIDATION_OPERATING_POINT` across all candidates.

---

## 2. Locked Held-Out Test Stream Results

All metrics below are computed on the frozen held-out test streams (`test_stream_continuous_commands.wav` and `test_stream_continuous_nontarget.wav`).
The false alarm rate is computed with exact denominator: $\text{FA/hr} = \frac{\text{False Positives}}{\text{Stream Duration (seconds)} / 3600.0}$. Test non-target stream duration = $688.7\text{ s} = 0.1913\text{ hours}$.

### 2.1 Multi-Seed Summary Table

| Candidate | Seed | Target Recall | Mean Latency | NT FPs | NT FA / hr | Primary Requirements Met? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1: Historical Cascade v1** | 42 | 99.38% (159/160) | 395.6 ms | 19 | 99.31 | Failed ($\text{FA/hr} > 1.0$) |
| | 43 | 97.50% (156/160) | 431.6 ms | 20 | 104.54 | Failed ($\text{FA/hr} > 1.0$) |
| | 44 | 95.63% (153/160) | 386.7 ms | 20 | 104.54 | Failed ($\text{FA/hr} > 1.0$) |
| | **Mean $\pm$ Std** | **$97.50\% \pm 1.88\%$** | **$404.6 \pm 23.8\text{ ms}$** | **$19.7$** | **$102.80 \pm 3.02$** | **Failed** |
| **Candidate 2: Hard-Negative S1 alone** | 42 | 95.00% (152/160) | 449.7 ms | 54 | 282.26 | Failed ($\text{FA/hr} > 1.0$) |
| | 43 | 99.38% (159/160) | 388.4 ms | 67 | 350.21 | Failed ($\text{FA/hr} > 1.0$) |
| | 44 | 99.38% (159/160) | 427.4 ms | 49 | 256.13 | Failed ($\text{FA/hr} > 1.0$) |
| | **Mean $\pm$ Std** | **$97.92\% \pm 2.53\%$** | **$421.8 \pm 31.0\text{ ms}$** | **$56.7$** | **$296.20 \pm 48.56$** | **Failed** |
| **Candidate 3: Hard-Negative Cascade** | 42 | 69.37% (111/160) | 424.4 ms | 17 | 88.86 | Failed (Recall & FA/hr) |
| | 43 | 92.50% (148/160) | 387.9 ms | 10 | 52.27 | Failed (Recall & FA/hr) |
| | 44 | 85.62% (137/160) | 421.2 ms | 9 | 47.04 | Failed (Recall & FA/hr) |
| | **Mean $\pm$ Std** | **$82.50\% \pm 11.89\%$** | **$411.2 \pm 20.2\text{ ms}$** | **$12.0$** | **$62.72 \pm 22.81$** | **Failed** |
| **Candidate 4: Character-Level CTC KWS** | 42 | 12.50% (20/160) | 777.4 ms | 0 | 0.00 | Failed (Recall & Latency) |
| | 43 | 76.88% (123/160) | 502.2 ms | 91 | 475.66 | Failed (Recall & FA/hr) |
| | 44 | 38.12% (61/160) | 725.5 ms | 0 | 0.00 | Failed (Recall & Latency) |
| | **Mean $\pm$ Std** | **$42.50\% \pm 32.41\%$** | **$668.4 \pm 146.3\text{ ms}$** | **$30.3$** | **$158.55 \pm 274.62$** | **Failed** |

---

## 3. Validation Stream Reference Results

For strict integrity verification, validation stream metrics (`val_stream_continuous_commands.wav` and `val_stream_continuous_nontarget.wav`) are reported below:

| Candidate | Mean Val Target Recall | Mean Val Latency | Mean Val NT FPs | Mean Val NT FA / hr |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate 1: Historical Cascade v1** | 92.08% | 397.6 ms | 22.0 | 113.50 |
| **Candidate 2: Hard-Negative Stage 1** | 90.62% | 419.9 ms | 54.0 | 278.60 |
| **Candidate 3: Hard-Negative Cascade** | 73.96% | 413.4 ms | 14.0 | 72.23 |
| **Candidate 4: Character-Level CTC KWS** | 41.04% | 679.4 ms | 26.0 | 134.14 |

---

## 4. Linguistic Confusion Breakdown on Held-Out Non-Target Stream

The 80 held-out non-target challenge events span five controlled linguistic confusion families. The table below lists the total false positive activations across all 3 random seeds categorized by their trigger family:

| Linguistic Confusion Family | Candidate 1 (Historical) | Candidate 2 (S1 alone) | Candidate 3 (HN Cascade) | Candidate 4 (CTC KWS) |
| :--- | :---: | :---: | :---: | :---: |
| **`minimal_pair_vowel_coda`** | 22 | 63 | **17** | 38 |
| **`minimal_pair_consonant`** | 13 | 34 | **9** | 13 |
| **`phonetic_phrase_context`** | 16 | 31 | **2** | 17 |
| **`neutral_speech`** | 8 | 33 | **8** | 17 |
| **`minimal_pair_tone`** | 0 | 9 | **0** | 6 |
| **Total False Positives (3 seeds)** | **59** | **170** | **36** | **91** |
| **Average FPs / Seed** | **19.7** | **56.7** | **12.0** | **30.3** |

### Key Linguistic Findings:
1. **Suppression of Tone Confusions:** Both cascade systems (Candidate 1 and Candidate 3) achieved complete (0 FP) suppression of `minimal_pair_tone` distractors (e.g. *trại*, *phải* tonal variants). Tone differences are effectively captured by spectrogram temporal contour modeling in the verifier.
2. **Persistence of Vowel/Coda and Consonant Confusions:** The vast majority of surviving false alarms stem from `minimal_pair_vowel_coda` (words differing only by nuclear vowel or final coda consonant, e.g. *tra*, *trau*, *pha*, *lenh*) and `minimal_pair_consonant` (words differing only by initial onset, e.g. *quai*, *sai*, *ken*, *buong*).
3. **Phrase-Context Rejection:** Candidate 3 dramatically reduced phrase-context false alarms from 16 down to only 2 triggers across 3 seeds. Hard-negative training taught the model not to trigger when command-like syllables are embedded inside longer natural phrases.
4. **The Recall–Rejection Tradeoff:** Candidate 3 reduced total false positives by nearly 40% relative to Candidate 1 (from 59 down to 36). However, this aggressive boundary shifting caused the Stage 2 verifier to reject genuine command utterances when spoken with slightly fast or atypical phonetics, driving target recall down to $82.50\%$.

---

## 5. Candidate Analysis and Failure Modes

### Candidate 1: Historical Cascade v1
- **Strengths:** Excellent target recall ($97.50\%$) and rapid detection latency ($404.6\text{ ms}$).
- **Failure Mode:** Inadequate non-target rejection under adversarial phonetic confusion. Generates $\sim 102.8\text{ FA/hr}$ on dense non-target speech, far above the $1.0\text{ FA/hr}$ threshold.

### Candidate 2: Hard-Negative Stage 1 alone
- **Strengths:** High target recall ($97.92\%$).
- **Failure Mode:** Catastrophic false alarm rate ($296.2\text{ FA/hr}$). A single-stage 2D-CNN sliding at 100 ms intervals without temporal verification triggers on transient energy peaks that superficially resemble target command spectra.

### Candidate 3: Hard-Negative Retrained Cascade
- **Strengths:** Best speech-active false alarm rate among discriminative models ($62.72\text{ FA/hr}$), zero tone false alarms, near-zero phrase-context false alarms.
- **Failure Mode:** Severe command recall degradation ($82.50\%$ mean recall, dipping to $69.37\%$ on Seed 42). Fails the $\ge 95\%$ target recall threshold.

### Candidate 4: Character-Level CTC KWS
- **Strengths:** High theoretical selectivity.
- **Failure Mode:** Catastrophic training instability and streaming hallucination under small data. Seeds 42 and 44 failed to learn robust alignments (recall $< 40\%$). Seed 43 learned character outputs but suffered massive spurious substring matches across sliding windows ($475.7\text{ FA/hr}$).

---

## 6. Conclusion

None of the four candidates achieved the joint operational target ($\ge 95\%$ recall, $\le 1.0\text{ FA/hr}$, $\le 600\text{ ms}$ latency). The speech-active false alarm bottleneck cannot be resolved purely by threshold adjustments or naive hard-negative augmentation without degrading command recall.
