# Phase 50 — Issues, Anomalies, and Corrective Actions

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v4/`  
**Status:** `VERIFIED_INDEPENDENT_AUDIT` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Issue: Separation of Error Categorization from Event-Matching Tolerance

### Description
In continuous streaming keyword spotting evaluations, two distinct temporal concepts were previously susceptible to conceptual blurring:
1. **Event-Matching Tolerance Window:** The temporal distance permitted between a detected event timestamp and a ground-truth command center ($|t_{\text{det}} - t_{\text{ref}}| \le 0.50\,\text{s}$) to count as a True Positive.
2. **False Alarm Error Categorization:** The acoustic context where a False Positive occurs (whether it was triggered by background noise, speech onset/offset transitions, or non-target speech).

### Corrective Action
In Phase 50, we formalized a rigorous, non-overlapping partition of the evaluation stream:
- $T_{\text{active}} = \bigcup_{i} [t_{i, \text{onset}}, t_{i, \text{end}}]$
- $T_{\text{boundary}} = \bigcup_{i} \Big( [t_{i, \text{onset}} - 0.20\,\text{s}, t_{i, \text{onset}}) \cup (t_{i, \text{end}}, t_{i, \text{end}} + 0.60\,\text{s}] \Big) \setminus T_{\text{active}}$
- $T_{\text{gap}} = [0, T_{\text{total}}] \setminus (T_{\text{active}} \cup T_{\text{boundary}})$

Every false alarm is assigned to exactly one disjoint set. This cleanly isolates ambient noise triggers from phonetic transition triggers.

---

## 2. Issue: Repetition Structure of the 3-Hour Challenge Stream

### Description
The 3-hour continuous challenge stream (`test_stream_continuous_nontarget_3h.wav`) contains 960 speech events across 10,800 seconds. It was necessary to audit whether utterances were repeated uniformly or skewed toward specific phonetic classes.

### Audit Findings & Resolution
The audit confirmed:
- Exactly 480 unique synthetic utterances from 4 held-out test voices (120 utterances per voice).
- The stream consists of exactly 2 passes across all 480 utterances ($480 \times 2 = 960$ events).
- Each utterance is repeated exactly 2 times with different pink noise background samples and variable randomized inter-utterance gap durations (5.0s to 14.0s).
- The distribution of confusion families is preserved identically across passes (neutral speech: 23.3%, consonant minimal pairs: 20.0%, vowel/coda minimal pairs: 20.0%, phrase context: 20.0%, tonal minimal pairs: 16.7%).

This validates that the 3-hour stream is statistically stationary and does not artificially weight any single speaker or confusion class.

---

## 3. Acoustic Finding: Boundary-Margin Vulnerability vs Silence Immunity

### Audit Finding
In Phase 50 analysis of Candidate 3 (Recall-Preserving Cascade):
- **Gap False Alarms:** Exactly **0** (0.00%) across all three seeds (42, 43, 44). The cascade is completely immune to pure pink noise and background silence.
- **Boundary Margin False Alarms:** Constitute **60.7% to 88.5%** of all false alarms ($11/13$, $17/28$, $46/52$).
- **Active Speech False Alarms:** Constitute **11.5% to 39.3%** of all false alarms ($2/13$, $11/28$, $6/52$).

### Acoustic Explanation
False alarms in Candidate 3 are almost entirely triggered by window-edge boundary effects: as the sliding window transitions off a non-target word (e.g., *sải*, *thải*), the trailing vowel decay and coda consonant (/aːj/) match the Stage 1 CNN filterbank pattern while the TCN verifier temporal receptive field still sees residual speech energy before silence kicks in.

---

## 4. Architectural Architectural Conclusion: Bounded-Utterance vs Continuous Open-Mic

### Strategic Realignment
In continuous open-mic streaming, the model evaluates approximately 36,000 sliding window inferences per hour (100 ms stride). Even a specificity of 99.9% yields 36 false alarms per hour.

In **Bounded-Utterance Recognition (Mode A)**:
- Audio capture is explicitly bounded (e.g., push-to-talk, VAD-isolated segment, or short audio clip input).
- A single classification decision is rendered on the full utterance with a dedicated rejection class.
- The window-edge transition vulnerability is eliminated because evaluation happens over the bounded segment.

Therefore, Phase 52–54 establishes bounded-utterance recognition as a distinct, rigorous evaluation track while maintaining the open-mic verdict of `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.
