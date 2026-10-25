# Phase 59 — Final Decision and Dual-Verdict Certification

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Campaign:** Phases 55–59 (Bounded-Utterance Improvement and Independent Evaluation)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Dual-Verdict Framework

To maintain absolute clarity between user-initiated command recognition and continuous unconstrained stream monitoring, this campaign provides two distinct formal verdicts:

### Mode A: Bounded-Utterance Recognition (Push-to-Talk / Touch-to-Speak Turn)
- **Operational Model:** The user explicitly initiates speech capture. A single 1.0-second audio waveform is presented to the system for classification or rejection.
- **Predeclared Research Target:**
  - Macro & Per-Keyword Target Recall $\ge 95.0\%$.
  - Non-Target False Acceptance Rate $\le 5.0\%$.
  - CPU Decision Latency $\le 50.0\,\text{ms}$.
- **Empirical Findings (Finalist 1 — Cascade v1 Bounded Pipeline):**
  - **Non-Target FAR:** **4.64%** (95% CI: 3.7%–5.8%) $\rightarrow$ **MET** ($\le 5.0\%$).
  - **Rejection Accuracy (TNR):** **95.36%** across 560 held-out non-target clips.
  - **Macro Command Recall:** **92.29%** (95% CI: 88.1%–96.1%) $\rightarrow$ Missed the strict $95.0\%$ target by $2.71\%$ (`TRAI`: 98.33%, `PHAI`: 90.83%, `LEN`: 90.83%, `XUONG`: 89.17%).
  - **Overall 5-Class Accuracy:** **94.68%**.
  - **CPU Latency:** **4.15 ms** per clip (P90 = 6.98 ms) $\rightarrow$ **MET** ($< 10\%$ of budget).
- **Mode A Formal Verdict:**
  $$\mathbf{BOUNDED\_UTTERANCE\_RESEARCH\_BENCHMARK\_ADVANCED}$$
  $$\mathbf{JUSTIFIES\_CONTROLLED\_MICROPHONE\_FEASIBILITY\_PILOT}$$
  *(Subject to strict natural-speech governance and pending human listener verification)*

---

### Mode B: Continuous Open-Mic Recognition (Always-Listening Streaming Audio)
- **Operational Model:** The microphone stream is continuously active (36,000 sliding window inferences per hour) under background speech and ambient noise.
- **Operational Requirements:**
  - Target Command Event Recall $\ge 95.0\%$.
  - Non-Target False Activations $\le 1.0\text{ event/hour}$.
  - Detection Latency $\le 600\,\text{ms}$.
- **Empirical Evidence (from Phases 48–50):**
  - Continuous streaming false activations remain between **10.33 FA/hr** (at 74.2% recall) and **77.33 FA/hr** (at 97.3% recall).
  - No system satisfies both $\ge 95\%$ recall and $\le 1.0\text{ FA/hr}$ in continuous open-mic mode.
  - Improvements in bounded-utterance mode **do not** transfer to continuous sliding-window streaming.
- **Mode B Formal Verdict:**
  $$\mathbf{NO\_DEPLOYABLE\_SYSTEM\_AT\_CURRENT\_SPECIFICATION}$$

---

## 2. Definitive Separation Notice

> [!CAUTION]
> **Strict Operational Separation:**  
> A 4.64% false acceptance rate in Bounded-Utterance Mode is an outstanding result for push-to-talk interactions where only 1 decision is made per user invocation.  
> However, in continuous open-mic streaming, testing 10 sliding windows per second means a 4.64% false alarm rate would cause over **1,600 false triggers per hour**.  
> Therefore, success in Bounded-Utterance Mode must never be cited as justification for open-mic deployment. Mode B remains strictly blocked under `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.

---

## 3. Recommended Roadmap for Subsequent Research

1. **Conduct Blinded Human Perceptual Audits:**
   - Execute listener verification on the 40 blinded items in `human_review/`.
2. **Transition from Synthetic to Pilot Natural Speech Capture:**
   - Because synthetic VieNeu-TTS models have achieved their empirical limit (~92.3% recall at 4.6% FAR), further bounded-utterance optimization requires natural human speech data collected under verified institutional consent.
3. **Multi-Task Phonetic Frontend for Open-Mic Streaming:**
   - Continuous open-mic monitoring requires a separate research initiative focusing on explicit syllable segmentation, boundary lockout, and CTC-aligned phoneme decoding before continuous streaming evaluation can be reopened.
