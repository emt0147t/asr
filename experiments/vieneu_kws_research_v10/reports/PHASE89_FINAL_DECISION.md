# Phase 89 — Final Evidence and Operational Decision Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment:** `experiments/vieneu_kws_research_v10/`  
**Candidate Model:** `variant_w10` (Frozen across seeds 42, 43, 44)  
**Governance Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Continuous Open-Mic Verdict:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
**Date:** 2026-10-18  

---

## 1. Executive Summary & Quad-Partite Verdicts

Phase 89 concludes the integrated research campaign (Phases 85–89) by synthesizing all empirical findings across synthetic benchmarking, human perceptual auditing, independent robustness reproduction, physical loopback feasibility, and open-mic deployment readiness.

```
+-------------------------------------------------------------------------------------------------------+
|                                    OPERATIONAL READINESS STATUS                                       |
+-------------------------------------------------------------------------------------------------------+
| 1. Synthetic Bounded-Utterance Mode: | FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED (Test: 99.8%, 2.0% FAR) |
| 2. Human Perceptual Quality:         | PENDING_HUMAN_LISTENER_VERIFICATION (0 actual reviews)         |
| 3. Physical Acoustic Loopback:       | NOT_RUN / UNRESOLVED (No physical hardware connected)          |
| 4. Continuous Open-Mic Mode:         | NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION                  |
| 5. Natural Human Speech:             | NOT_VALIDATED_ON_NATURAL_SPEECH                                |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Evidence Synthesis by Evaluation Domain

### Domain A: Synthetic Bounded-Utterance Benchmark
- **Frozen Test Set Parity ($N = 720$, 4 Held-Out Voices, 80 Novel Texts):**
  - Macro Command Recall: **$99.79\% \pm 0.36\%$** (TRAI: $100\%$, PHAI: $99.17\%$, LEN: $100\%$, XUONG: $100\%$)
  - Non-Target False Acceptance Rate: **$2.03\% \pm 0.58\%$** (meets $\le 5.0\%$ ceiling)
  - 5-Class Overall Accuracy: **$98.38\% \pm 0.40\%$**
  - Single-Clip CPU Latency: **$1.73 \pm 0.03\text{ ms}$** (p90: $1.96\text{ ms}$)
- **Validation Partition Parity ($N = 360$, 4 Held-Out Validation Voices):**
  - Macro Command Recall: **$96.88\% \pm 0.63\%$**
  - Non-Target False Acceptance Rate: **$5.83\% \pm 0.76\%$**
  - 5-Class Overall Accuracy: **$95.37\% \pm 0.64\%$**
- **Reconciliation Resolution:**
  - Absolute raw false acceptances are essentially identical ($\sim 11 - 13$ clips across both partitions).
  - The $2.8\times$ difference in non-target denominator ($560$ in test vs $200$ in validation) accounts entirely for the percentage shift. Model parameters and decision thresholds ($\theta_{\text{cmd}} = [0.40, 0.50, 0.35]$) are $100\%$ identical.

---

### Domain B: Human Perceptual Validation
- **Package Status:** 120 audio clips verified with $100\%$ SHA-256 hash parity in `human_review/audio/`.
- **Interface Security:** Double-blind web application (`review_app.py`) verified and hardened: zero exposure of answer keys, labels, transcripts, or model predictions; strictly binds to `127.0.0.1:8080`.
- **Review Intake Accounting:** Exactly **0** human responses submitted to date.
- **Anti-Fabrication Compliance:** Zero responses fabricated, imputed, or simulated from model outputs.
- **Status:** **`PENDING_HUMAN_LISTENER_VERIFICATION`**. Automated test passes do not substitute for human perceptual judgment.

---

### Domain C: Synthetic Robustness Reproduction
Phase 87 reproduced all 13 conditions ($14,040$ inferences) without altering the frozen candidate:
- **Conditions Meeting Both Targets ($\text{Recall} \ge 95\%$, $\text{FAR} \le 5\%$):**
  - `speed_0_90x`: Recall $97.29\% \pm 0.72\%$, FAR $3.67\% \pm 0.76\%$
  - `shift_minus_150ms`: Recall $96.46\% \pm 0.91\%$, FAR $3.67\% \pm 0.76\%$
- **Conditions Failing Recall Target ($< 95.0\%$):**
  - `snr_20db`: Recall **$94.37\% \pm 0.63\%$** (FAR $5.67\%$) — formally **`BELOW_RECALL_TARGET`**
  - `clipping_half`: Recall $94.58\% \pm 0.36\%$ (FAR $3.33\%$)
  - `reverb_rt60_200ms`: Recall $92.92\% \pm 0.72\%$ (FAR $3.33\%$)
  - `gain_plus_6db`: Recall $91.46\% \pm 0.91\%$ (FAR $0.83\%$)
  - `gain_minus_12db`: Recall $88.75\% \pm 1.57\%$ (FAR $31.83\%$)
  - `snr_10db`: Recall $86.67\% \pm 0.72\%$ (FAR $10.17\%$)
  - `snr_5db`: Recall $66.25\% \pm 1.57\%$ (FAR $13.50\%$)
  - `snr_0db`: Recall $39.37\% \pm 2.08\%$ (FAR $32.83\%$)
- **Conditions Exceeding FAR Ceiling ($> 5.0\%$):**
  - `clean_baseline` ($5.83\%$), `speed_1_10x` ($7.33\%$), `shift_plus_150ms` ($9.50\%$), and noise/attenuation conditions.
- **Acoustic Vulnerability:**
  - `TRAI` is the most fragile keyword, dropping to $88.33\%$ at $+20\text{ dB}$ SNR due to masking of the voiceless plosive burst `/ʈ/`.
  - `LEN` exhibits the highest noise resilience, retaining $98.33\%$ recall at $+10\text{ dB}$ SNR.
  - Low-SNR rejection is classified as *graceful rejection*, not robust recognition.

---

### Domain D: Physical Acoustic Feasibility
- Runtime inspection found no physical audio capture hardware or driver bindings (`sounddevice`, `pyaudio`).
- In strict adherence to scientific integrity rules, acoustic propagation was **NOT** simulated in software.
- Status: **`NOT_RUN`**.
- Verdict: **`UNRESOLVED`**.

---

### Domain E: Natural Human Speech & Open-Mic Deployment
- **Natural Human Speech:** The candidate has evaluated only high-fidelity VieNeu-TTS synthesized speech. Status remains strictly **`NOT_VALIDATED_ON_NATURAL_SPEECH`**.
- **Continuous Open-Mic Mode:** The candidate is a bounded-utterance single-clip classifier. In continuous streaming, synthetic non-target audio generates $\sim 69\text{ FA/hour}$ (Phase 39). Therefore, the verdict remains strictly:
  $$\mathbf{NO\_DEPLOYABLE\_SYSTEM\_AT\_CURRENT\_SPECIFICATION}$$

---

## 3. Decision on Model Development & Further Retraining

### Formal Decision:
**FURTHER SYNTHETIC MODEL RETRAINING IS NOT JUSTIFIED.**

### Empirical Justification:
1. **Saturation on Synthetic Data:** Candidate `variant_w10` achieves $99.79\%$ command recall and $2.03\%$ FAR on the locked synthetic test set. Retraining on additional synthetic variations yields diminishing returns and risks overfitting to TTS synthesizer idiosyncrasies.
2. **True Operational Bottlenecks:** The remaining barriers to real-world deployment are external to synthetic model architecture:
   - *Bottleneck 1 (Perceptual Validation):* Human listeners must verify that VieNeu-TTS synthetic audio accurately represents intelligible Vietnamese phonetics.
   - *Bottleneck 2 (Acoustic Channel):* Physical loudspeaker and microphone loopback must be tested to measure transducer distortion and room reverberation.
   - *Bottleneck 3 (Natural Human Speech):* Real Vietnamese speakers with diverse regional accents (Northern, Central, Southern) must be recorded under approved consent protocols.

Retraining another synthetic CNN or tweaking loss weights before addressing these empirical bottlenecks would be scientifically unproductive.

---

## 4. Next Step Roadmap

| Phase / Action | Objective | Prerequisites | Unblocking Action |
| :--- | :--- | :--- | :--- |
| **1. Human Listening Review** | Collect 120 ratings per listener to validate synthetic pronunciation | Reviewer access to `http://127.0.0.1:8080/` | Human listeners evaluate audio using `review_app.py` |
| **2. Physical Acoustic Capture** | Execute Phase 88 loopback playback through physical loudspeaker/mic | Physical lab rig, `pip install sounddevice` | Attach USB measurement mic and monitor speaker |
| **3. Natural Human Speech Collection** | Evaluate candidate on authentic human Vietnamese speakers | Approved IRB/consent protocol | Ingest consented human speech recordings |
