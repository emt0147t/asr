# Phase 21 — Extended Synthetic Speaker Generalization and Streaming Reliability

**Workspace:** `D:\trainidentificationaudio`  
**Evaluation Scope:** `experiments/synthetic_kws_phase20_22/phase21_robustness/`  
**Execution Date:** 2026-10-16  
**Generator / Model:** V5.2 Candidate Generator / Frozen TC-ResNet-14 (`v5_2_tcresnet14_seed_42.pt`)  
**Scripts:**
- [`phase21_generate_heldout.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_generate_heldout.py)
- [`phase21_streaming_benchmark.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_streaming_benchmark.py)

---

## 1. Executive Summary & Gate Verdict

| Experiment Component | Evaluation Scope | Measured Value | Performance Standard | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Speaker Generalization** | 24 newly sampled held-out profiles, 288 clips | **100.00%** (288/288), F1 = 1.0000 | $\ge 90.0\%$ | **PASS** |
| **Statistical Uncertainty** | Wilson score 95% CI across 288 clips | $[98.68\%, 100.00\%]$ | Documented | **PASS** |
| **Target-Absent Streaming** | 3.00 Hours (10,800.0s), 107,946 windows | **0 False Triggers (0.00 FP/hr)** | $\le 1.0\text{ FP/hr}$ | **PASS** |
| **Target-Absent Upper Bound** | Poisson / Rule of Three 95% upper limit | $\le 1.00\text{ FP/hr}$ | Documented | **PASS** |
| **Target-Present Detection** | 200 ground truth targets across 8 streams | **200/200 Detected (100.00% recall)** | $\ge 95.0\%$ recall | **PASS** |
| **Streaming Event Precision** | True Positives / (TP + FP on gaps) | **100.00%** (0 false activations) | $\ge 90.0\%$ precision | **PASS** |
| **Detection Latency** | Full empirical distribution ($N=200$) | Mean: **0.506s**, Median: **0.503s**, P95: **0.636s** | $\le 0.700\text{ s}$ | **PASS** |
| **Counterfactual TRAI vs PHAI** | 50 matched pairs holding non-phonetics fixed | **100.00%** (50/50 pairs distinguished) | $\ge 95.0\%$ | **PASS** |
| **Human Data Governance** | Human collection blocked, no field audio | Strictly maintained, 0 unapproved bytes | Enforced | **PASS** |

### Gate Verdict
```
================================================================================
GATE VERDICT: PHASE21_PASS_WITH_LIMITATIONS
================================================================================
```
The frozen TC-ResNet-14 acoustic model and dual-threshold hysteresis streaming detector demonstrate comprehensive generalization across 24 held-out speaker profiles, complete suppression of false triggers over 3.00 hours of continuous target-absent audio, and 100.00% target recall across 200 continuous keyword events within the parametric synthetic speech synthesizer family. The official status remains **`NOT_VALIDATED_ON_NATURAL_SPEECH`**.

---

## 2. Extended Held-Out Synthetic Speaker Profile Evaluation

### 2.1 Profile Design and Disjoint Sampling
The prior test set was constrained to two speaker profiles. To stress generalization across diverse physical vocal tracts, 24 newly sampled speaker profiles (`SPK_HELD_01` through `SPK_HELD_24`) were instantiated using disjoint seed domains ($700,000+$):

1. **High Female & Child Profiles (6):**
   - VTL scale: 0.82 – 0.89, Base F0: 218 – 255 Hz, Open Quotient: 0.42 – 0.65, Tilt: $-13.5$ to $-7.5\text{ dB/oct}$.
2. **Mid Female Profiles (6):**
   - VTL scale: 0.91 – 0.97, Base F0: 175 – 210 Hz, Open Quotient: 0.49 – 0.62, Tilt: $-12.5$ to $-9.0\text{ dB/oct}$.
3. **Mid Male Profiles (6):**
   - VTL scale: 1.01 – 1.06, Base F0: 130 – 155 Hz, Open Quotient: 0.44 – 0.66, Tilt: $-14.0$ to $-8.0\text{ dB/oct}$.
4. **Deep Male & Resonant Profiles (6):**
   - VTL scale: 1.08 – 1.18, Base F0: 92 – 122 Hz, Open Quotient: 0.46 – 0.70, Tilt: $-15.0$ to $-9.0\text{ dB/oct}$.

### 2.2 Dataset Manifest & Performance
- **Generated Clips:** 288 (12 per profile: 2 `TRAI`, 2 `PHAI`, 2 `LEN`, 2 `XUONG`, 2 `UNKNOWN`, 2 `SILENCE`).
- **Manifest:** [`phase21_heldout_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/phase21_heldout_manifest.csv) (100% parameter serialization and unique SHA-256 hashes).
- **Classification Results (Frozen Seed 42 Checkpoint):**
  - **Overall Accuracy:** **100.00%** (288 / 288 correct)
  - **Macro-F1:** **1.0000**
  - **Wilson Score 95% Confidence Interval:** $[98.68\%, 100.00\%]$
  - **Rule of Three Error Bound:** $\le \frac{3}{288} = 1.04\%$
  - **Per-Profile Accuracy:** 100.00% across all 24 speaker profiles.
  - **Confusion Analysis:**
    - `TRAI` misclassified as `PHAI`: 0
    - `PHAI` misclassified as `TRAI`: 0
    - `UNKNOWN` false alarms to target classes: 0
    - `SILENCE` false alarms to target classes: 0

---

## 3. Extended Synthetic Target-Absent Streaming (3.00 Hours)

### 3.1 Streaming Architecture & Setup
- **Evaluated Duration:** **10,800.0 seconds (3.00 Hours)** of actual continuous synthesized audio.
- **Number of Independent Streams:** 6 streams of 1,800.0s (30.0 minutes) each.
- **Acoustic Content:** Realistic concatenations of UNKNOWN speech tokens, SILENCE regimes (room tone, low-frequency hum, dither, digital zero), and inter-speech pauses (0.2s to 1.5s).
- **Inference Volume:** 107,946 sliding window evaluations (1.0s window, 100ms hop).

### 3.2 Stream-Level Results

| Stream Identifier | Seed Base | Audio Duration | Windows Evaluated | False Triggers | False Trigger Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ta_stream_01` | 800000 | 1,800.0 s (30m) | 17,991 | 0 | 0.00 FP/hr |
| `ta_stream_02` | 815000 | 1,800.0 s (30m) | 17,991 | 0 | 0.00 FP/hr |
| `ta_stream_03` | 830000 | 1,800.0 s (30m) | 17,991 | 0 | 0.00 FP/hr |
| `ta_stream_04` | 845000 | 1,800.0 s (30m) | 17,991 | 0 | 0.00 FP/hr |
| `ta_stream_05` | 860000 | 1,800.0 s (30m) | 17,991 | 0 | 0.00 FP/hr |
| `ta_stream_06` | 875000 | 1,800.0 s (30m) | 17,991 | 0 | 0.00 FP/hr |
| **Total / Overall** | — | **10,800.0 s (3.00 h)** | **107,946** | **0** | **0.00 FP/hr** |

- **CSV Artifact:** [`stream_level_target_absent.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/stream_level_target_absent.csv)
- **Uncertainty Analysis:**
  Under the Poisson assumption for rare independent events, observing zero events over $T = 3.00\text{ hours}$ yields a one-sided 95% upper bound of:
  $$\lambda_{95\%} \le \frac{3.0}{T} = \frac{3.0}{3.00} = 1.00\text{ False Triggers / Hour}$$
  This tightens the Phase 18 bound ($\le 3.00\text{ FP/hr}$) by a factor of 3.

---

## 4. Continuous Target-Present Keyword Event Detection (200 Events)

### 4.1 Evaluation Setup
- **Ground Truth Target Events:** **200 target events** across 8 independent streams (25 targets per stream).
- **Class Balance:** 56 `TRAI`, 48 `PHAI`, 48 `LEN`, 48 `XUONG` (randomized balanced scheduling).
- **Inter-Target Distractors:** 2.0s to 5.5s intervals containing 1 to 3 UNKNOWN distractor words, pauses, and ambient noise transitions.
- **Event Matching Tolerance:** Event decision time $t_{decision} = t_{win} + 1.0\text{ s} \in [t_{start}, t_{end} + 0.60\text{ s}]$.

### 4.2 Stream-Level Detection Results

| Stream Identifier | Targets Inserted | True Positives | False Positives | Missed | Recall | Precision | Mean Latency | Median Latency | P95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tp_stream_01` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.507 s | 0.504 s | 0.640 s |
| `tp_stream_02` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.516 s | 0.518 s | 0.638 s |
| `tp_stream_03` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.506 s | 0.502 s | 0.635 s |
| `tp_stream_04` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.521 s | 0.520 s | 0.638 s |
| `tp_stream_05` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.495 s | 0.493 s | 0.635 s |
| `tp_stream_06` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.494 s | 0.492 s | 0.632 s |
| `tp_stream_07` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.517 s | 0.519 s | 0.641 s |
| `tp_stream_08` | 25 | 25 | 0 | 0 | 100.0% | 100.0% | 0.494 s | 0.493 s | 0.633 s |
| **Consolidated** | **200** | **200** | **0** | **0** | **100.00%** | **100.00%** | **0.506 s** | **0.503 s** | **0.636 s** |

- **CSV Artifacts:**
  - Stream-level: [`stream_level_target_present.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/stream_level_target_present.csv)
  - Event-level: [`event_level_target_present.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/event_level_target_present.csv) (200 ground truth rows)
- **Per-Keyword Recall Breakdown:**
  - `TRAI`: 56 / 56 (100.0%)
  - `PHAI`: 48 / 48 (100.0%)
  - `LEN`: 48 / 48 (100.0%)
  - `XUONG`: 48 / 48 (100.0%)
- **Latency Distribution ($N=200$):**
  - Mean: $0.506\text{ s}$
  - Median: $0.503\text{ s}$
  - 90th Percentile: $0.600\text{ s}$
  - 95th Percentile: $0.636\text{ s}$
  - Standard Deviation: $0.081\text{ s}$

---

## 5. Factorial Counterfactual Study (TRAI vs PHAI)

### 5.1 Design of Matched Pairs
To evaluate whether the model discriminates phonetic contrasts rather than incidental synthesis settings, 50 matched counterfactual pairs were generated:
- **Shared Parameters per Pair:** Speaker voice profile, vocal tract length scale, base fundamental frequency, utterance duration, onset offset, speech level, background noise type, and background noise level.
- **Isolated Intervention:** Synthesized keyword phonetics (`TRAI` /ʈaːɪ/ vs `PHAI` /faːɪ/).
- **CSV Artifact:** [`counterfactual_trai_phai_results.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_phase20_22/phase21_robustness/counterfactual_trai_phai_results.csv)

### 5.2 Results
- **Pairs Evaluated:** 50
- **Correctly Distinguished Pairs:** **50 / 50 (100.00%)**
- **Average Confidence:**
  - `TRAI` predictions on `TRAI` audio: Mean confidence = 0.9984
  - `PHAI` predictions on `PHAI` audio: Mean confidence = 0.9991
- **Cross-Keyword Probability Leakage:** In all 50 pairs, the competing keyword received $< 0.001$ probability.

---

## 6. Frozen Streaming Detector Configuration

All streaming evaluations utilized the exact detector parameters established and audited in Phase 18/20 without post-hoc tuning:

```json
{
  "tau_on": 0.80,
  "tau_off": 0.35,
  "smoothing_frames": 3,
  "min_consecutive_frames": 2,
  "cooldown_sec": 0.600,
  "window_size_samples": 16000,
  "hop_size_samples": 1600
}
```

---

## 7. Limitations & Transition to Research Release Candidate

1. **Synthetic Generation Support:**
   The observed 100% accuracy, zero false triggers, and 100% event recall demonstrate robust generalization across diverse parametric configurations within the synthesizer. They do **not** establish performance on real human speakers or unconstrained conversational acoustic environments.
2. **Poisson Rate Bound:**
   Zero false triggers over 3.00 hours establishes that the synthetic false trigger rate is bounded above by $1.00\text{ FP/hr}$ at 95% confidence.
3. **Official Status:**
   The project remains **`NOT_VALIDATED_ON_NATURAL_SPEECH`**.
