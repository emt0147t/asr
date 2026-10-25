# Experiment: Controlled VAD Gating Ablation for Vietnamese KWS Cascade v1

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Path:** [`experiments/vieneu_kws_vad_ablation_v1/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_vad_ablation_v1)  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Overview

This directory contains the code, configurations, calibration records, prediction summaries, and analysis reports for **Phase 39: Controlled Voice Activity Detection (VAD) Gating Ablation**.

The objective of this ablation is to empirically test whether pre-filtering streaming audio with Voice Activity Detection before the two-stage Vietnamese KWS cascade improves continuous-stream false alarm behavior towards the requirements:
- **Target Command Recall:** $\ge 95\%$
- **Non-Target False Activations:** $\le 1.0\text{ FA/hour}$
- **Mean Event Detection Latency:** $\le 600\text{ ms}$

### Final Empirical Verdict: **REJECT VAD GATING**
- **Non-Target False Alarms:** VAD provided **$0.0\%$ reduction** in cascade false alarms ($69.21\text{ FA/hr}$ with or without VAD) because $100\%$ of residual cascade false alarms occur during active speech.
- **Target Command Recall:** VAD caused onset/tail clipping, degrading recall by **$-3.13\%$** ($96.25\% \rightarrow 93.13\%$) and causing Seeds 43 and 44 to fail the $\ge 95\%$ threshold.
- **Recommendation:** Retain Condition B (Cascade without VAD) as champion.

---

## 2. Directory Structure

```
experiments/vieneu_kws_vad_ablation_v1/
├── README.md                              <- Master reproduction guide and summary
├── config.json                            <- Frozen experiment and VAD configurations
├── vad_engine.py                          <- VAD implementations (Energy & SpeechBand)
├── evaluate_vad_ablation.py               <- Streaming evaluation script across A, B, C, D
├── calibration/
│   └── validation_records.json            <- Validation calibration grid search and infeasibility declaration
├── predictions/
│   └── vad_ablation_summary.json          <- Full event triggers, matches, and error decompositions (seeds 42, 43, 44)
└── reports/
    ├── EXISTING_PIPELINE_AUDIT.md         <- Independent audit of durations, rates, and event matching
    ├── VAD_ABLATION_REPORT.md             <- Comprehensive ablation findings and comparative tables
    └── NEXT_STEP_DECISION.md              <- Formal decision, root-cause diagnosis, and future roadmap
```

---

## 3. Evaluated Operating Conditions

Across initialization seeds 42, 43, and 44:
- **Condition A (Stage 1 Alone):** CNN v2 streaming baseline ($1.0\text{ s}$ window, $100\text{ ms}$ hop, $2\times$ confirmation).
- **Condition B (Two-Stage Cascade):** CNN v2 followed by BiGRU Temporal Verifier ($1.2\text{ s}$ crop).
- **Condition C (Stage 1 + VAD):** CNN v2 gated by front-end Speech-Band VAD.
- **Condition D (Cascade + VAD):** Two-Stage Cascade gated by front-end Speech-Band VAD.

---

## 4. Key Performance Summary (Held-Out Test Streams)

| Metric | Condition A (S1 Alone) | Condition B (Cascade) | Condition C (S1 + VAD) | Condition D (Cas + VAD) | Target Specification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Target Command Recall** | $99.79\% \pm 0.36\%$ | **$96.25\% \pm 2.73\%$** | $96.67\% \pm 0.36\%$ | $93.13\% \pm 2.73\%$ | $\ge 95.0\%$ |
| **Command Precision** | $48.89\% \pm 1.05\%$ | **$97.29\% \pm 1.37\%$** | $49.22\% \pm 1.03\%$ | $97.20\% \pm 1.41\%$ | N/A |
| **Command F1 Score** | $0.6563$ | **$0.9674$** | $0.6522$ | $0.9510$ | N/A |
| **Mean Detection Latency** | $403.2\text{ ms}$ | **$403.7\text{ ms}$** | $406.2\text{ ms}$ | $406.4\text{ ms}$ | $\le 600\text{ ms}$ |
| **Non-Target FA / Hour** | $1,146.0\text{ FA/hr}$ | **$69.21\text{ FA/hr}$** | $1,138.1\text{ FA/hr}$ | **$69.21\text{ FA/hr}$** | $\le 1.0\text{ FA/hr}$ |
| **Non-Target Raw FPs** | $287.0\text{ FPs}$ | **$17.33\text{ FPs}$** | $285.0\text{ FPs}$ | **$17.33\text{ FPs}$** | Minimal |
| - Speech-Active FPs | $104.0\text{ FPs}$ | **$17.33\text{ FPs}$** | $102.3\text{ FPs}$ | **$17.33\text{ FPs}$** | N/A |
| - Silence/Background FPs| $183.0\text{ FPs}$ | **$0.00\text{ FPs}$** | $182.7\text{ FPs}$ | **$0.00\text{ FPs}$** | N/A |

---

## 5. Reproduction Instructions

### 5.1 Run Automated Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_vieneu_vad_ablation.py -v
```

### 5.2 Re-Run Full Ablation Evaluation
```powershell
.\.venv\Scripts\python.exe experiments/vieneu_kws_vad_ablation_v1/evaluate_vad_ablation.py
```

### 5.3 Run Full VieNeu Regression Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -k vieneu -v
```
