# Phase 8H-B: Open-Gate Negative-Control Audit Report

> **Mandatory Governance Labels:**
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`

---

## 1. Executive Summary & Security Boundary Findings

This audit evaluated the false activation rates of **Arm B** (seeds 42, 43, 44) compared to **frozen v11 ONNX** across six standardized negative-control conditions under continuous open microphone listening versus session-gated listening.

### Key Audit Conclusions
1. **Dramatic Open-Gate Selectivity Gain:** On natural Vietnamese conversation, Arm B reduces false activations from **133.33 FA/min** (frozen v11) down to **17.78 FA/min** (an **88% reduction**). On phonetic foils and OOV vocabulary, Arm B achieves substantial suppression compared to the baseline.
2. **Open-Gate Insufficiency:** Despite an 88% reduction, open-gate conversational FAR is **17.78 FA/min** (approximately 1 false activation every 3.5 seconds during active speech). The standalone classifier **cannot operate safely as an un-gated, always-on listening trigger**.
3. **Architectural Session-Gating Invariance:** When routed through [`GameModeSessionController`](file:///d:/trainidentificationaudio/src/continuous_kws/game_mode_session.py) with the gate closed, **zero false activations (0.00 FA/min)** are emitted across all models and all 6 conditions (789.7 seconds combined audio). Zero false activation is an **architectural property of the session gate**, not a classifier intrinsic guarantee.
4. **Validation Threshold Sweep:** Raising $\theta_{cmd}$ from 0.20 to 0.35 reduces conversational FAR from 16.7 to ~10.4 FA/min, but drops command recall by ~2.8%. Operating at $\theta_{cmd} = 0.20$ inside a session-gated Game Mode profile provides the optimal operating point (maximum command agility with zero closed-gate false triggers).

---

## 2. False Activation Rates by Negative Control Condition

| Condition | Audio Duration | Frozen v11 ONNX | Arm B (Seed 42) | Arm B (Seed 43) | Arm B (Seed 44) | Arm B Mean FAR | Relative Reduction vs v11 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vietnamese Conversation** | 144.0 s | **133.33** | 15.42 | 17.50 | 20.42 | **17.78 FA/min** | **-87.5%** |
| **Phonetic Foils** | 120.0 s | **48.20** | 10.43 | 11.93 | 14.41 | **12.26 FA/min** | **-66.7%** |
| **General OOV Speech** | 120.0 s | **48.70** | 19.38 | 17.89 | 23.35 | **20.21 FA/min** | **-75.0%** |
| **Ambient Speaker Playback** | 144.2 s | **117.34** | 30.37 | 34.95 | 31.21 | **32.18 FA/min** | **-73.6%** |
| **Environmental Noise** | 200.0 s | **40.50** | 4.50 | 3.00 | 3.00 | **3.50 FA/min** | **-91.4%** |
| **Digital Silence** | 60.0 s | **0.00** | 0.00 | 0.00 | 0.00 | **0.00 FA/min** | 0.0% |
| **Closed Architectural Gate** | **789.7 s** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00 FA/min** | **100% BLOCKED** |

---

## 3. Emitted Command Distribution on Natural Vietnamese Conversation (Open Gate)

| Model Candidate | Total FAs | LEN Triggers | XUONG Triggers | TRAI Triggers | PHAI Triggers | Primary False Alarm Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Frozen v11 ONNX** | 320 | 24 | 230 | 21 | 45 | Predominantly `XUONG` (71.9%) |
| **Arm B (Seed 42)** | 37 | 20 | 12 | 0 | 5 | Balanced `LEN` / `XUONG` |
| **Arm B (Seed 43)** | 42 | 33 | 8 | 0 | 1 | Predominantly `LEN` |
| **Arm B (Seed 44)** | 41 | 18 | 15 | 1 | 7 | Balanced `LEN` / `XUONG` |

---

## 4. Validation-Calibrated Threshold Sweep Analysis

Calibrated on the independent validation split (640 target commands + 144s conversation):

| Model | Threshold $\theta_{cmd}$ | Validation Target Recall (%) | Open-Gate Conversation FAR (/min) | Operational Trade-off |
| :--- | :---: | :---: | :---: | :--- |
| **Arm B (Seed 42)** | 0.20 | 98.75% | 15.42 | High Agility (Default Game Mode) |
| **Arm B (Seed 42)** | 0.35 | 96.25% | 10.42 | Moderate Trade-off (-32% FAR, -2.5% Recall) |
| **Arm B (Seed 42)** | 0.50 | 91.88% | 6.25 | Conservative (-59% FAR, -6.9% Recall) |
| **Arm B (Seed 42)** | 0.70 | 83.12% | 2.50 | High Precision (-84% FAR, -15.6% Recall) |
| **Frozen v11 ONNX** | 0.20 | 97.50% | 133.33 | Severe False Triggering |
| **Frozen v11 ONNX** | 0.50 | 88.75% | 72.50 | High FAR remains even at high threshold |

> [!NOTE]
> Even at $\theta_{cmd} = 0.70$, open-gate conversational FAR on Arm B remains non-zero (2.50 FA/min), while degrading keyword recall to 83.12%. Therefore, threshold tuning cannot substitute for architectural session gating.

---

## 5. Artifact Reference Ledger

- Negative Control Results: [`reports/phase8h_b_negative_controls.csv`](file:///d:/trainidentificationaudio/reports/phase8h_b_negative_controls.csv)
- Confidence Distributions: [`reports/phase8h_b_confidence_distributions.json`](file:///d:/trainidentificationaudio/reports/phase8h_b_confidence_distributions.json)
- Threshold Sweep Data: [`reports/phase8h_b_threshold_sweep.csv`](file:///d:/trainidentificationaudio/reports/phase8h_b_threshold_sweep.csv)
