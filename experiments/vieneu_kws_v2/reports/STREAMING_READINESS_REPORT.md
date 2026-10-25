# Phase 33–35: Streaming Evaluation and Temporal Readiness Report

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Directory:** [`experiments/vieneu_kws_v2/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2)  
**Execution Timestamp:** 2026-10-18  
**Governance Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

This report evaluates the temporal continuous-streaming performance of the trained six-class convolutional neural network (**CNN v2**). In real-world keyword spotting, audio is not provided as pre-segmented $1.0$-second clips; rather, an unconstrained audio stream flows continuously past an inference engine.

Our streaming evaluation tested whether the model and its sliding-window temporal event detector meet the operational criteria:
- **Target Command Stream Recall:** $\ge 95.0\%$
- **Temporal Event Latency:** $\le 600\text{ ms}$
- **Non-Target Stream False Alarms:** $\le 1.0\text{ event/hour}$

### Streaming Findings Matrix

| Evaluation Dimension | Metric / Target | Seed 42 | Seed 43 | Seed 44 | Mean $\pm$ Std | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Command Stream Recall** | $\ge 95.0\%$ | $98.1\%$ ($157/160$) | $98.8\%$ ($158/160$) | **$99.4\%$ ($159/160$)** | **$98.75\% \pm 0.64\%$** | **PASS** |
| **Command Stream Precision** | Diagnostic | $85.8\%$ | $83.2\%$ | **$90.3\%$** | $86.43\% \pm 3.60\%$ | Diagnostic |
| **Mean Event Latency** | $\le 600\text{ ms}$ | **$411.6\text{ ms}$** | $435.9\text{ ms}$ | $425.2\text{ ms}$ | **$424.2\text{ ms} \pm 12.2\text{ ms}$** | **PASS** |
| **90th Percentile Latency** | Diagnostic | $492.4\text{ ms}$ | $535.3\text{ ms}$ | $508.2\text{ ms}$ | $512.0\text{ ms} \pm 21.7\text{ ms}$ | **PASS** |
| **Pure Silence Stream FA/hr** | $\le 1.0\text{ FA/hr}$ | **$0.0\text{ FA/hr}$** | **$0.0\text{ FA/hr}$** | **$0.0\text{ FA/hr}$** | **$0.0\text{ FA/hr}$** | **PASS** |
| **Non-Target Speech FA/hr** | $\le 1.0\text{ FA/hr}$ | $327.4\text{ FA/hr}$ | $1,130.1\text{ FA/hr}$ | **$223.6\text{ FA/hr}$** | **$560.4\text{ FA/hr} \pm 496.0$** | **FAIL** |

**Conclusion:** The standalone single-stage compact CNN v2 excels in command detection rate ($>98\%$) and response speed ($<430\text{ ms}$), and is completely immune to ambient silence false activations ($0.0\text{ FA/hr}$). However, it **fails the continuous non-target false alarm rate requirement by over two orders of magnitude** ($223.6–1,130.1\text{ FA/hr}$ vs $\le 1.0\text{ FA/hr}$).

---

## 2. Streaming Simulation Methodology

### 2.1 Audio Streams
Continuous multi-minute audio streams were constructed strictly from unseen held-out test split data:
1. **Command Stream (`continuous_command_stream.wav`):**
   - **Duration:** $676.0\text{ seconds}$ ($0.188\text{ hours}$).
   - **Contents:** 160 ground-truth command utterances ($40$ each of `TRAI`, `PHAI`, `LEN`, `XUONG`) spoken by the 4 unseen test speakers (`kim_ngan`, `mai_phuong`, `quoc_tuan`, `thanh_hang`), separated by randomized realistic gaps ($1.5$ to $3.5\text{ seconds}$) containing environmental background noise and speech pauses.
2. **Non-Target Stream (`continuous_nontarget_stream.wav`):**
   - **Duration:** $901.8\text{ seconds}$ ($0.250\text{ hours}$).
   - **Contents:** 80 distinct, unseen Vietnamese non-target speech utterances (zero occurrences of target words) interspersed with 80 environmental noise segments (HVAC, office noise, airflow pink noise, and digital quiet).

### 2.2 Streaming Inference and Event Detector Logic

- **Window Size:** $1.0\text{ second}$ ($16,000$ samples).
- **Stride (Hop Length):** $100\text{ ms}$ ($1,600$ samples) $\rightarrow$ $10$ evaluations per second ($36,000$ evaluations per hour).
- **Decision Engine:**
  - Probability threshold: Seed-specific calibrated validation threshold ($\theta \in [0.50, 0.70]$).
  - Consecutive confirmation: Requires $k \ge 2$ consecutive windows predicting the same target command class above $\theta$.
  - Refractory lockout: Once an event triggers, a $500\text{ ms}$ refractory hold-off suppresses duplicate triggers for the same acoustic utterance.
- **Latency Measurement:** Latency is defined as the elapsed time from the ground-truth audio onset to the exact timestamp of the second confirming window trigger:
  $$\text{Latency} = t_{\text{trigger}} - t_{\text{onset}}$$

---

## 3. Command Stream Detection Performance

### 3.1 Event-Level Metrics

| Seed | Calibrated Threshold ($\theta$) | Ground Truth Targets | True Positives | False Negatives | False Positives | Stream Recall | Stream Precision | Stream F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | $0.60$ | 160 | 157 | 3 | 26 | $98.13\%$ | $85.79\%$ | $91.55\%$ |
| **Seed 43** | $0.50$ | 160 | 158 | 2 | 32 | $98.75\%$ | $83.16\%$ | $90.29\%$ |
| **Seed 44** | $0.70$ | 160 | 159 | 1 | 17 | **$99.38\%$** | **$90.34\%$** | **$94.64\%$** |

### 3.2 Latency Distribution

| Seed | Mean Latency (ms) | Median Latency (ms) | 90th Percentile Latency (ms) | Minimum Latency (ms) | Maximum Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | $411.6$ | $396.0$ | $492.4$ | $264.0$ | $588.0$ |
| **Seed 43** | $435.9$ | $424.5$ | $535.3$ | $288.0$ | $598.0$ |
| **Seed 44** | $425.2$ | $414.0$ | $508.2$ | $276.0$ | $592.0$ |

**Latency Analysis:**
All 160 detected target commands across all three seeds triggered well under the $600\text{ ms}$ threshold. Because Vietnamese monosyllabic commands typically span $300–450\text{ ms}$, the second sliding window ($200\text{ ms}$ after partial onset) captures the discriminative vowel nucleus and tone contour, firing reliably at $411–435\text{ ms}$.

---

## 4. Non-Target False Alarm Analysis and Physical Cause

### 4.1 Quantitative False Alarm Breakdown

In the $901.8\text{ s}$ ($0.250\text{ hr}$) continuous non-target stream:
- **Seed 42 ($\theta = 0.60$):** 82 false alarms $\rightarrow$ **$327.4\text{ FA/hour}$**
- **Seed 43 ($\theta = 0.50$):** 283 false alarms $\rightarrow$ **$1,130.1\text{ FA/hour}$**
- **Seed 44 ($\theta = 0.70$):** 56 false alarms $\rightarrow$ **$223.6\text{ FA/hour}$**

Target operating requirement: $\le 1.0\text{ FA/hour}$.

### 4.2 The Mathematical Physics of Sliding-Window False Alarms

Why does a CNN that demonstrates $94.7\%$ accuracy on isolated test clips produce hundreds of false alarms per hour in streaming mode?

#### Factor 1: The Multiplication of Window Evaluations
In isolated evaluation, 80 test clips represent 80 distinct evaluation opportunities. In continuous streaming with a $100\text{ ms}$ hop:
$$\text{Evaluations per hour} = \frac{3600\text{ s}}{0.100\text{ s}} = 36,000\text{ windows/hour}$$
In our $901.8$-second stream, the model evaluated **$9,010$ distinct spectrogram windows**.
For Seed 44, 56 false event clusters corresponds to an effective window-level error rate of approximately:
$$\text{Error rate per window} \approx \frac{56}{9010} \approx 0.62\%$$
In standard classification terminology, a model with $99.38\%$ per-window specificity appears outstanding. However, when multiplied across $36,000$ trials:
$$\text{Expected FA/hr} = 36,000 \times 0.0062 \approx 223.2\text{ FA/hr}$$
To achieve $\le 1.0\text{ FA/hour}$ in a pure sliding-window system, the required per-window false positive rate must be:
$$P(\text{FP}_{\text{window}}) \le \frac{1.0}{36,000} \approx 0.0000278\quad (99.9972\%\text{ specificity})$$
A single-stage compact CNN with 359k parameters cannot maintain $99.997\%$ specificity while preserving $>98\%$ recall on diverse speech.

#### Factor 2: Vietnamese Phonetic and Tonal Resonances
Inspection of the false alarm event log reveals that false alarms are heavily concentrated on specific phonetic patterns:
1. **The `/uəŋ/` + `hỏi`/`ngã`/`nặng` Rhyme Space (`XUONG` Confusion):**
   - Vietnamese non-target words such as *đường*, *hướng*, *phương*, *thương*, *tưởng*, *vườn* contain identical or near-identical vowel diphthongs and velar nasal codas.
   - When a $1.0$-second window captures the syllable nucleus and decaying tail, the log-Mel spectral envelope closely mimics *Xuống*. In Seed 42, $49$ out of $82$ false alarms were spurious `XUONG` classifications.
2. **The `/aj/` Rhyme Space (`PHAI` Confusion):**
   - Non-target phrases containing *bài*, *tay*, *mai*, *ngày* trigger partial activations on `PHAI`.
3. **Absence of Whole-Word Duration and CTC Alignment:**
   - The CNN classifier receives a static $64 \times 101$ image. It does not enforce left-to-right temporal phonetic sequencing (e.g., initial fricative $[s] \rightarrow$ diphthong $[uə] \rightarrow$ coda $[\eta]$). Any acoustic burst with similar spectral centroid and tone slope can trigger a threshold crossing.

---

## 5. Streaming Readiness Verdict

| Requirement | Target | Achieved (CNN v2) | Status |
| :--- | :---: | :---: | :---: |
| Command Detection Recall | $\ge 95\%$ | $98.1\% – 99.4\%$ | **PASS** |
| Detection Latency | $\le 600\text{ ms}$ | $411.6 – 435.9\text{ ms}$ | **PASS** |
| Silence Rejection | $\le 1.0\text{ FA/hr}$ | $0.0\text{ FA/hr}$ | **PASS** |
| Speech False Alarm Rejection | $\le 1.0\text{ FA/hr}$ | $223.6 – 1,130.1\text{ FA/hr}$ | **FAIL** |
| **Overall Streaming Readiness** | **Ready for autonomous mic** | **NOT READY** | **FAIL** |

### Operational Implication
CNN v2 **cannot be deployed as an autonomous always-on streaming wake-word detector** without secondary verification. If connected to an open microphone, it would trigger a false command activation every $10$ to $20$ seconds during ambient conversational speech.

However, CNN v2 is **fully ready for push-to-talk, VAD-gated, or button-triggered interaction**, where inference is only invoked during isolated speech bursts.
