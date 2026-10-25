# Phase 48: Validation-Only Calibration and Candidate Selection Report

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 48 performed strict validation-only calibration and finalist selection across three competing configurations:
1. **Candidate 1:** Historical Cascade v1 (immutable baseline).
2. **Candidate 2:** Recall-Preserving Stage 1 alone (CNN v3).
3. **Candidate 3:** Recall-Preserving Hard-Negative Cascade (Stage 1 CNN v3 + Stage 2 TCN Verifier v3).

All parameter selection, threshold searches ($\theta_1 \in [0.40, 0.70]$, $\theta_2 \in [0.40, 0.75]$), debounce lockout rules, and confidence policies were evaluated strictly on the validation streams:
- `val_stream_continuous_commands.wav`: 514.95 s (160 validation target command events)
- `val_stream_continuous_nontarget.wav`: 478.46 s (120 validation non-target distractor events)

### Operational Feasibility Verdict: **`NO_FEASIBLE_VALIDATION_OPERATING_POINT`**
Across all evaluated thresholds, random seeds (42, 43, 44), and candidates, **no operating point met the three mandatory operational targets simultaneously**:
- Target event recall $\ge 95\%$
- Non-target false activations $\le 1.0\text{ event/hour}$
- Mean event detection latency $\le 600\text{ ms}$

Consequently, `NO_FEASIBLE_VALIDATION_OPERATING_POINT` was declared and recorded in [`calibration/selected_operating_points.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v3/calibration/selected_operating_points.json).

---

## 2. Multi-Seed Validation Stream Results

| Candidate Architecture | Seed | Target Recall | Mean Latency | Non-Target FPs | Stream Duration | Non-Target FA / hr | Feasible? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1: Historical Cascade** | 42 | 93.13% (149/160) | 775.8 ms | 24 | 0.1329 hr | 180.58 | Failed ($\text{FA/hr} > 1.0$, Lat $> 600$) |
| | 43 | 90.00% (144/160) | 684.4 ms | 28 | 0.1329 hr | 210.68 | Failed (All 3 criteria) |
| | 44 | 92.50% (148/160) | 745.0 ms | 31 | 0.1329 hr | 233.25 | Failed (All 3 criteria) |
| | **Mean $\pm$ Std** | **$91.88\% \pm 1.66\%$** | **$735.1 \pm 46.5\text{ ms}$** | **$27.7$** | **$0.1329\text{ hr}$** | **$208.17 \pm 26.43$** | **Failed** |
| **Candidate 2: Stage 1 alone** | 42 | 93.13% (149/160) | 577.9 ms | 88 | 0.1329 hr | 662.13 | Failed ($\text{FA/hr} > 1.0$, Rec $< 95\%$) |
| | 43 | 94.38% (151/160) | 580.1 ms | 72 | 0.1329 hr | 541.74 | Failed ($\text{FA/hr} > 1.0$, Rec $< 95\%$) |
| | 44 | 94.38% (151/160) | 600.5 ms | 61 | 0.1329 hr | 458.97 | Failed (All 3 criteria) |
| | **Mean $\pm$ Std** | **$93.96\% \pm 0.72\%$** | **$586.2 \pm 12.5\text{ ms}$** | **$73.7$** | **$0.1329\text{ hr}$** | **$554.28 \pm 102.16$** | **Failed** |
| **Candidate 3: Recall-Preserving Cascade** | 42 | 50.00% (80/160) | 756.4 ms | 2 | 0.1329 hr | 15.05 | Failed (Recall & Latency) |
| | 43 | 65.00% (104/160) | 694.2 ms | 5 | 0.1329 hr | 37.62 | Failed (Recall, Latency & FA/hr) |
| | 44 | 80.00% (128/160) | 710.2 ms | 6 | 0.1329 hr | 45.15 | Failed (Recall, Latency & FA/hr) |
| | **Mean $\pm$ Std** | **$65.00\% \pm 15.00\%$** | **$720.3 \pm 32.3\text{ ms}$** | **$4.3$** | **$0.1329\text{ hr}$** | **$32.61 \pm 15.67$** | **Failed** |

---

## 3. Threshold Calibration Dynamics

1. **Stage 1 Proposal Threshold ($\theta_1$):**
   - At $\theta_1 = 0.50$, Stage 1 yields high candidate recall ($> 93\%$) on validation commands. Raising $\theta_1 > 0.65$ reduces false alarms marginally but drops target command recall catastrophically below $80\%$.
2. **Stage 2 Verification Threshold ($\theta_2$):**
   - At $\theta_2 = 0.55$, Candidate 3 suppresses over $84\%$ of non-target false activations (from 27.7 FPs down to 4.3 FPs on validation non-target stream).
   - However, even at $\theta_2 = 0.55$, the verifier rejects $35\%$ of genuine commands on validation streams, demonstrating that the decision boundary cannot cleanly separate target monosyllabic words from dense phonetic hard negatives.
3. **Debounce Lockout ($0.60\text{ s}$):**
   - A $0.60\text{ s}$ debounce lockout effectively eliminates duplicate triggers following confirmed detections without causing blindspots on subsequent commands (inter-utterance gaps are $> 1.5\text{ s}$).

---

## 4. Finalist Selection for Locked Testing

Pursuant to the pre-registered protocol:
- **Finalist 1:** Candidate 1 (Historical Cascade v1) — Preserved as the immutable comparative benchmark.
- **Finalist 2:** Candidate 3 (Recall-Preserving Cascade) — Selected as the primary experimental candidate due to superior non-target rejection ($32.6\text{ FA/hr}$ vs $208.2\text{ FA/hr}$).
- Candidate 2 (Stage 1 alone) is carried forward as an ablation reference to document the single-stage failure mode.

All models, thresholds, and configurations were locked prior to running the frozen 3-hour test benchmark.
