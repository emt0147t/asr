# Phase 89 — Deployment Readiness Matrix

**Workspace:** `D:\trainidentificationaudio`  
**Experiment:** `experiments/vieneu_kws_research_v10/`  
**Candidate Model:** `variant_w10` (Frozen)  
**Date:** 2026-10-18  

---

## 1. Multi-Dimensional Readiness Matrix

The table below presents the audited operational status across all project specifications and deployment criteria:

| Dimension / Mode | Specification Requirement | Measured Candidate Metric | Compliance Status | Operational Gate Status |
| :--- | :--- | :--- | :---: | :--- |
| **Synthetic Bounded-Utterance (Test)** | Recall $\ge 95\%$ for every keyword | TRAI: 100%, PHAI: 99.17%, LEN: 100%, XUONG: 100% (Mean: 99.79%) | **COMPLIANT** | `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED` |
| **Synthetic Non-Target FAR (Test)** | FAR $\le 5.0\%$ across non-targets | Measured FAR: **$2.03\% \pm 0.58\%$** | **COMPLIANT** | `FEASIBLE_BOUNDED_CANDIDATE_ESTABLISHED` |
| **Decision Latency (CPU)** | Latency $\le 100\text{ ms}$ | Single-clip: **$1.73 \pm 0.03\text{ ms}$** (p90: $1.96\text{ ms}$) | **COMPLIANT** | `EXCEEDS_EFFICIENCY_TARGET` |
| **Model Size / Footprint** | Low parameter footprint ($\le 1\text{M}$) | 138,469 parameters ($541\text{ KB}$ weights) | **COMPLIANT** | `EXCEEDS_EFFICIENCY_TARGET` |
| **Clean Baseline (Validation)** | Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: $96.88\%$, FAR: $5.83\%$ | **PARTIAL** | Non-target FAR exceeds $5\%$ due to smaller denominator |
| **Mild Noise (+20 dB SNR)** | Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: **$94.37\%$**, FAR: $5.67\%$ | **NON-COMPLIANT** | `BELOW_RECALL_TARGET` |
| **Moderate Noise (+10 dB SNR)**| Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: $86.67\%$, FAR: $10.17\%$ | **NON-COMPLIANT** | `BELOW_RECALL_AND_EXCEEDS_FAR` |
| **Digital Clipping (|x| > 0.5)** | Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: $94.58\%$, FAR: $3.33\%$ | **PARTIAL** | Recall narrowly misses ($94.58\%$ vs $95\%$) |
| **Reverberation ($T_{60} = 0.2\text{s}$)**| Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: $92.92\%$, FAR: $3.33\%$ | **PARTIAL** | Recall misses ($92.92\%$ vs $95\%$) |
| **Slow Tempo (0.90x)** | Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: $97.29\%$, FAR: $3.67\%$ | **COMPLIANT** | `MEETS_BOUNDED_TARGETS` |
| **Early Temporal Shift (-150ms)** | Recall $\ge 95\%$, FAR $\le 5\%$ | Recall: $96.46\%$, FAR: $3.67\%$ | **COMPLIANT** | `MEETS_BOUNDED_TARGETS` |
| **Human Perceptual Review** | Confirmed intelligible by native listeners | 0 verified ratings on file | **BLOCKED** | `PENDING_HUMAN_LISTENER_VERIFICATION` |
| **Physical Acoustic Loopback** | Verified through loudspeaker & microphone | Hardware unattached / drivers missing | **BLOCKED** | `NOT_RUN` (Verdict: `UNRESOLVED`) |
| **Natural Human Speech** | Validated on human speech recordings | Synthetic TTS audio only | **BLOCKED** | `NOT_VALIDATED_ON_NATURAL_SPEECH` |
| **Continuous Open-Mic Mode** | False alarms $\le 1\text{ event/hour}$ | Stream FA: $\sim 69\text{ events/hour}$ (P39) | **NON-COMPLIANT** | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` |

---

## 2. Gate Decision Summary

### Gate 1: Bounded-Utterance Synthetic Feasibility — **UNLOCKED**
Candidate `variant_w10` achieves $99.79\%$ command recall and $2.03\%$ FAR on the locked synthetic test set. Within push-to-talk bounded synthetic interaction, it meets the performance requirements.

### Gate 2: Human Perceptual Verification — **LOCKED**
Review interface and audio package are verified and ready, but 0 human evaluations are on file.
*Action to Unlock:* Complete blinded listening evaluations using `review_app.py`.

### Gate 3: Physical Transducer Feasibility — **LOCKED**
Loudspeaker-to-microphone loopback could not run due to lack of connected physical audio devices.
*Action to Unlock:* Connect reference microphone and speaker in physical acoustic environment.

### Gate 4: Continuous Open-Mic Deployment — **LOCKED**
Single-clip bounded KWS models trigger false alarms during conversational speech and background noise. Continuous open-mic operation requires streaming temporal accumulation or two-stage verification that is not yet established.
*Action to Unlock:* Develop streaming temporal confidence integration or wake-word gating.

### Gate 5: Natural Human Speech — **LOCKED**
Zero human speech recordings evaluated in this phase.
*Action to Unlock:* Ingest approved, consented field recordings of human Vietnamese speakers.
