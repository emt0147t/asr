# Phase 36: Independent Audit of Existing Evaluation Protocol and Metrics

**Workspace:** `d:\trainidentificationaudio`  
**Audited Artifacts:**
- [`experiments/vieneu_kws_v2/evaluate.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/evaluate.py)
- [`experiments/vieneu_kws_v2/config.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/config.json)
- [`experiments/vieneu_kws_v2/predictions/evaluation_summary.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/predictions/evaluation_summary.json)
- [`experiments/vieneu_kws_v2/calibration/selected_thresholds.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/calibration/selected_thresholds.json)
- [`experiments/vieneu_kws_v2/reports/V2_TRAINING_AND_ROBUSTNESS_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/reports/V2_TRAINING_AND_ROBUSTNESS_REPORT.md)
- [`experiments/vieneu_kws_v2/reports/STREAMING_READINESS_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/reports/STREAMING_READINESS_REPORT.md)
- [`experiments/vieneu_kws_v2/reports/NEXT_STEP_DECISION.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/reports/NEXT_STEP_DECISION.md)
- [`experiments/vieneu_kws_v2/streaming_simulation/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/streaming_simulation)
**Execution Date:** 2026-10-18  
**Auditor:** Antigravity Pair-Programming Agent (Autonomous Research Suite)  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Audit Summary

An independent verification and mathematical recalculation of the Phase 33–35 evaluation pipeline was conducted. All raw audio streams, ground truth files, confidence threshold calibrations, and evaluation scripts were audited against strict evidentiary standards.

### Audit Verdict: **PASSED WITH NOTED OBSERVATIONS**
- **Metric Reproducibility:** All reported metrics (accuracy, F1, recalls, latencies, and false activations per hour) were independently reproduced from the frozen artifacts and code.
- **Denominator Integrity:** False activations per hour are calculated by dividing the exact count of deduplicated event triggers by the true eligible audio stream duration.
- **Zero Leakage / Calibration Isolation:** Threshold selection was executed strictly on the validation partition (`val_manifest.csv` and `val_stream_*`). No test-split data was utilized during calibration.
- **Infeasibility Transparency:** The reported infeasibility of the strict validation constraints ($\le 1.0\text{ FA/hour}$, $\le 5\%\text{ UNK FA}$) was verified. There was no silent relaxation.

---

## 2. Detailed Verification of the 7 Audit Dimensions

### 2.1 False Activation Denominators and Duration Calculations

The calculation of false activations per hour was verified in `match_events_and_evaluate()`:
$$\text{Stream Hours} = \frac{\text{Duration in Samples}}{16000 \times 3600}$$
$$\text{FA/hr} = \frac{\text{False Positives Count}}{\text{Stream Hours}}$$

Exact physical durations of the streaming audio files were measured with `soundfile.info()`:
- **Test Non-Target Stream (`test_stream_continuous_nontarget.wav`):**
  - Sample count: $14,424,587\text{ samples}$
  - Sample rate: $16,000\text{ Hz}$
  - Duration: $901.5367\text{ seconds} = 0.2504269\text{ hours}$
  - Seed 42: $82\text{ FPs} \rightarrow 82 / 0.2504269 = \mathbf{327.44\text{ FA/hr}}$ (matches reported $327.4$)
  - Seed 43: $283\text{ FPs} \rightarrow 283 / 0.2504269 = \mathbf{1,130.07\text{ FA/hr}}$ (matches reported $1,130.1$)
  - Seed 44: $56\text{ FPs} \rightarrow 56 / 0.2504269 = \mathbf{223.62\text{ FA/hr}}$ (matches reported $223.6$)
- **Validation Non-Target Stream (`val_stream_continuous_nontarget.wav`):**
  - Sample count: $14,414,654\text{ samples}$
  - Duration: $900.9159\text{ seconds} = 0.2502544\text{ hours}$

**Verdict:** The reported false activations per hour use actual detected event counts divided by exact eligible stream duration. Verified without error.

---

### 2.2 Deduplication, Consecutive Hits, and Lockout Logic

The streaming engine (`SlidingWindowStreamingEngine`) operates with:
- Window size: $16,000\text{ samples}$ ($1.0\text{ s}$)
- Hop size: $1,600\text{ samples}$ ($100\text{ ms}$)
- Streak confirmation: $k = 2$ consecutive windows predicting the identical target class with confidence $\ge \theta$.
- Debounce refractory lockout: $0.60\text{ s}$ ($600\text{ ms}$).

When an event triggers:
```python
trigger_time = t_sec + (self.window_samples / self.sample_rate)
lockout_until_sec = trigger_time + self.debounce_sec
current_streak_class = None
consecutive_count = 0
```
While `t_sec < lockout_until_sec`, incoming window classifications cannot increment the streak or trigger a new event. 

**Observation on Command Stream FPs:**
In the command stream evaluation ($675.9\text{ s}$ duration, $160$ genuine target commands), the model produced $26\text{ FPs}$ (Seed 42), $32\text{ FPs}$ (Seed 43), and $17\text{ FPs}$ (Seed 44). Audit of the ground truth reveals that the command stream contains $63$ interleaved distractor (`UNKNOWN`) utterances. The false alarms observed on the command stream originate almost exclusively from these interleaved UNKNOWN distractors triggering false command activations, rather than multiple triggers on the same command utterance.

**Verdict:** Repeated and overlapping windows from the same acoustic burst are properly deduplicated and not counted as independent events.

---

### 2.3 Event Matching and Latency Calculations

The matching logic in `match_events_and_evaluate()` compares each triggered event timestamp $t_{\text{trig}}$ against ground-truth targets:
- Match window: $[t_{\text{onset}} - 0.20\text{ s},\, t_{\text{end}} + 0.60\text{ s}]$.
- Ground-truth target is matched only if `pred_label == gt_class_label` and `not gt_matched[idx]`.
- Latency is computed as:
  $$\text{Latency (ms)} = (t_{\text{trig}} - t_{\text{onset}}) \times 1000$$
- Each ground-truth target is matched at most once (`gt_matched[idx] = True`).

**Measured Latencies (Seed 42):**
- Mean: $411.6\text{ ms}$
- Median: $396.0\text{ ms}$
- 90th percentile: $492.4\text{ ms}$
- Min / Max: $264.0\text{ ms} / 588.0\text{ ms}$

**Verdict:** Event matching, latency calculation, and one-to-one ground truth matching are mathematically consistent and sound.

---

### 2.4 Documented Denominators Across Evaluation Suites

| Evaluation Mode | Evaluated Item | Denominator | Documented Value |
| :--- | :--- | :--- | :---: |
| **Clean Test Static** | Overall Accuracy | Total clips in test split | $320$ |
| | Target Command Recall | Total target command clips in test | $160$ ($40 \times 4$) |
| | UNKNOWN FA Rate | Total UNKNOWN clips in test | $80$ |
| | SILENCE FA Rate | Total SILENCE clips in test | $80$ |
| | Target Rejection Rate | Total target command clips in test | $160$ |
| **Noise Robustness** | Overall Accuracy per SNR | Total clips in test split | $320$ |
| **Streaming Command** | Command Recall | Ground-truth targets in command stream | $160$ |
| | Command Precision | Total triggers in command stream ($\text{TP} + \text{FP}$) | $\text{TP} + \text{FP}$ |
| **Streaming Non-Target** | FA / Hour | Eligible stream duration in hours | $0.2504\text{ h}$ |

**Verdict:** All denominators are explicitly documented, unambiguous, and consistent across reports.

---

### 2.5 Validation Isolation and Non-Leakage

Audit of `run_evaluation_campaign()` confirms:
1. `val_manifest_p` (`experiments/synthetic_kws_dataset_vieneu_6class_v2/splits/val_manifest.csv`) and `val_stream_*` were loaded first.
2. Candidate thresholds $\theta \in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98]$ were evaluated only on validation data.
3. The selected thresholds ($\theta = 0.60$ for Seed 42, $0.50$ for Seed 43, $0.70$ for Seed 44) were frozen into `calibration/selected_thresholds.json`.
4. The test split (`test_manifest.csv` and `test_stream_*`) was evaluated strictly using these fixed thresholds without any retuning.

**Verdict:** Zero test-to-training or test-to-calibration leakage occurred.

---

### 2.6 Robustness Gain Reproducibility

The test noise generation was re-inspected. In `generate_noisy_test_batch()`:
- Noise waveforms are generated deterministically per seed.
- Active RMS scaling is computed:
  $$s_{\text{rms}} = \sqrt{\frac{1}{N} \sum x^2}$$
  $$\text{target } n_{\text{rms}} = s_{\text{rms}} \times 10^{-\text{SNR} / 20}$$
- Peak clipping prevention scales audio if peak exceeds $0.98$.
- Accuracy gains over CNN v1 ($+50.2\%$ on Babble 10 dB, $+47.0\%$ on White 0 dB) are fully reproducible and driven by the multi-condition noise augmentation applied during training.

**Verdict:** Robustness gains are mathematically verified and reproducible.

---

## 3. Findings and Design Requirements for Phase 37–38 Cascade

The audit confirms that the standalone CNN v2 is an effective **Stage 1 candidate proposer**:
1. It achieves $\ge 99.4\%$ target command recall.
2. It triggers at low latency ($411–435\text{ ms}$).
3. However, under non-target continuous speech, it proposes between $223$ and $1,130\text{ candidate events/hour}$.

### Design Rules for Stage 2 (Temporal Verifier):
1. **Decoupled Candidate Extraction:** Stage 1 will output candidate event proposals $(t_{\text{start}}, t_{\text{end}}, \hat{c}, \text{conf})$.
2. **Contextual Acoustic Crop:** Stage 2 will receive a fixed temporal acoustic crop centered on the proposed event (e.g., $1.2\text{ s}$ to $1.5\text{ s}$ of audio), providing temporal context around the candidate.
3. **Temporal Architecture:** Stage 2 must process frame-level temporal representations (such as a 1D Temporal Convolutional Network over log-Mel frames) rather than collapsing temporal sequence structure into a single global average pool.
4. **Target Classes + REJECT:** Stage 2 must output probabilities over $\{\text{TRAI}, \text{PHAI}, \text{LEN}, \text{XUONG}, \text{REJECT}\}$.
5. **Hard Negative Mining on Train-Only Data:** Stage 2 must be trained on genuine target commands plus hard negatives mined by running Stage 1 on the training split audio. No validation or test audio may be used for training.
