# Phase 54 — Final Decision and Dual-Verdict Certification

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v4/`  
**Campaign:** Phases 50–54 (Benchmark Audit, Acoustic Error Analysis, Bounded-Utterance Recognition)  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Dual-Verdict Framework

To resolve the architectural divergence identified across Phases 45–53, the evaluation was partitioned into two operationally distinct modes:

### Mode A: Bounded-Utterance Recognition (Push-to-Talk / User-Initiated Turn)
- **Operational Model:** The user explicitly initiates speech capture (e.g., button press, touch-to-speak, or isolated VAD segment). A single 1.0-second waveform is presented to the system.
- **Evaluation Criteria:**
  - Target Command Recall $\ge 95.0\%$.
  - Non-target / Hard-negative Rejection Rate $\ge 80.0\%$ (FAR $\le 20.0\%$).
  - Inference Latency per clip $\le 50\,\text{ms}$.
- **Empirical Results (Candidate 2 — Cascade v1 Bounded Pipeline):**
  - **Mean Command Recall:** **95.83%** (Hits: 157, 153, 150 out of 160 across seeds 42, 43, 44).
  - **Per-Keyword Recall:** `TRAI` = **99.17%**, `PHAI` = **100.00%**, `XUONG` = **93.33%**, `LEN` = **90.83%**.
  - **Non-Target Rejection Rate:** **85.12%** (Rejects: 489, 484, 457 out of 560 clips across seeds).
  - **Non-Target False Acceptance Rate (FAR):** **14.88%**.
  - **Inference Latency:** **5.81 ms** per clip.
- **Mode A Formal Verdict:**
  $$\mathbf{BOUNDED\_UTTERANCE\_FEASIBLE\_IN\_SYNTHETIC\_RESEARCH}$$
  *(Subject to pending human perceptual audit and natural speech validation)*

---

### Mode B: Continuous Open-Mic Recognition (Always-Listening Unconstrained Audio)
- **Operational Model:** The microphone stream is continuously active (36,000 sliding window inferences per hour) under background speech and ambient noise.
- **Predeclared Criteria:**
  - Target Command Event Recall $\ge 95.0\%$.
  - Non-Target False Activations $\le 1.0\text{ event/hour}$.
  - Mean Detection Latency $\le 600\,\text{ms}$.
- **Empirical Results (from Phase 49 & Phase 50 Audit):**
  - **Historical Cascade v1:** Recall = $97.29\%$, Wall-Clock FA Rate = **77.33 FA/hr** (Failed by $77.3\times$).
  - **Recall-Preserving Cascade (Candidate 3):** Wall-Clock FA Rate = **10.33 FA/hr** (best seed 4.33 FA/hr), but Command Recall dropped to **74.17%** (`TRAI` = 58.3%, `PHAI` = 59.2%).
  - Zero models simultaneously met both $\ge 95\%$ recall and $\le 1.0\text{ FA/hr}$.
- **Mode B Formal Verdict:**
  $$\mathbf{NO\_DEPLOYABLE\_SYSTEM\_AT\_CURRENT\_SPECIFICATION}$$

---

## 2. Definitive Separation Notice

> [!CAUTION]
> **Strict Operational Separation:**  
> Success in Bounded-Utterance Recognition (Mode A) **does not** imply or confer readiness for Continuous Open-Mic Recognition (Mode B).  
> The 14.88% false acceptance rate acceptable in push-to-talk interactions causes severe false trigger rates (10–77 FA/hr) when sliding windows are evaluated 10 times per second across hours of unconstrained conversation.  
> Under no circumstances may Candidate 2 or Candidate 3 be deployed as an unconstrained open-mic keyword detector.

---

## 3. Recommended Next Steps

1. **Conduct Blinded Human Perceptual Review:**
   - Execute listening evaluation on `experiments/vieneu_kws_research_v4/human_review/human_review_sheet.csv` using the protocol in `human_review/README.md`.
2. **Field Data Acquisition & Natural Speech Validation:**
   - Synthetic VieNeu-TTS data has enabled rigorous architectural exploration, but real-world mobile deployment requires multi-speaker natural Vietnamese speech recordings collected under proper institutional consent.
3. **Multi-Task Phonetic Frontend for Open-Mic Research:**
   - To achieve $\le 1.0\text{ FA/hr}$ in continuous streaming, future open-mic models should incorporate explicit acoustic-phonetic syllable decomposition or joint CTC alignment before decision gating.
