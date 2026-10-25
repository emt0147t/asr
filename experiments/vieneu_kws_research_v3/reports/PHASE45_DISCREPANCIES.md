# Phase 45: Discrepancy Log and Resolution Record

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Overview

This document reconciles all technical discrepancies, historical errata, and methodological constraints identified across Phases 36–44, providing unambiguous resolutions for the Phase 45–49 campaign.

---

## 2. Reconciled Discrepancies

### Discrepancy 1: Verifier Architecture Colloquial Designation ("BiGRU" vs. Dilated 1D-TCN)
- **Origin:** Reports in Phases 36–39 colloquially described Stage 2 as a "BiGRU verifier" based on early design notes.
- **Physical Reality:** The actual code in [`experiments/vieneu_kws_cascade_v1/train_verifier.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1/train_verifier.py) implements class `TemporalKWSVerifier`. It contains a 4-layer Dilated 1D Temporal Convolutional Network (TCN) with residual connections, batch normalization, and adaptive pooling, totaling **340,773 parameters**. It contains zero recurrent or GRU layers.
- **Resolution:** All subsequent phases and reports must refer to Stage 2 strictly as a **Dilated 1D Temporal Convolutional Network (TCN)**.

### Discrepancy 2: CTC Sequence Model Instability Under Small-Data Regimes
- **Origin:** Candidate B (CTC) validation keyword accuracy showed high variance across seeds (Seed 42: $38.12\%$, Seed 43: $71.25\%$, Seed 44: $86.88\%$).
- **Physical Reality:** Character-level CTC loss without pretraining on small datasets ($1,360$ training samples) suffers from severe blank-token collapse and alignment saddle points during early epochs. Seed 42 took 20 epochs before predicting any non-blank tokens.
- **Resolution:** CTC training will not be repeated blindly without architectural modifications or pretrained speech representations. For Phase 47, focus will be placed on Candidate A (Recall-Preserving Cascade) and investigating local pretrained speech encoders if available.

### Discrepancy 3: Condition Identifiers in the `voice_id` Column for Background/Silence Files
- **Origin:** Inspection of `manifest.csv` showed 21 unique `voice_id` values in training, but 4 of them were ambient noise labels (`near_silence`, `office_ambient`, `hvac_hum`, `airflow_pink`).
- **Physical Reality:** Speech samples strictly used 17 train, 4 val, and 4 test TTS speaker IDs (`vie_voice_001` through `vie_voice_025`), which are 100% mutually disjoint. Silence/background samples were labeled by condition type in the same column for database simplicity.
- **Resolution:** In Phase 46, schema separation will distinguish `speaker_voice_id` (strictly null for silence) from `acoustic_condition_tag` to prevent audit ambiguity.

### Discrepancy 4: Statistical Confidence Lower Bound on Short Non-Target Streams
- **Origin:** In Phase 43, the non-target challenge stream duration was $688.72\text{ s} = 0.1913\text{ hours}$ ($11.48\text{ minutes}$).
- **Physical Reality:** Under Poisson process assumptions, the 95% one-sided upper confidence bound for a process with 0 observed events over exposure duration $T$ hours is:
  $$\lambda_{95\%} = \frac{-\ln(0.05)}{T} \approx \frac{2.996}{T}$$
  For $T = 0.1913\text{ hours}$, 0 observed events yields a 95% upper bound of:
  $$\lambda_{95\%} = \frac{2.996}{0.1913} \approx 15.66\text{ FA/hour}$$
  Therefore, an 11.5-minute test stream cannot mathematically demonstrate a claim of $\le 1.0\text{ FA/hour}$ even if zero false alarms occur.
- **Resolution:** To establish $\le 1.0\text{ FA/hour}$ with 95% statistical confidence, the test stream must provide at least $T \ge 3.0\text{ hours}$ ($10,800\text{ seconds}$) of continuous exposure with 0 false activations:
  $$\lambda_{95\%} = \frac{2.996}{3.0} \approx 0.999\text{ FA/hour} \le 1.0\text{ FA/hour}$$
  This mathematical requirement forms the core rationale for Phase 46's long-horizon 3-hour benchmark.
