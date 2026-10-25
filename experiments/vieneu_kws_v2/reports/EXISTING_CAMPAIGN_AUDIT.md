# Independent Audit of the Phase 30–32 Robustness Campaign

**Workspace:** `D:\trainidentificationaudio`  
**Audited Directory:** `experiments/vieneu_kws_robustness_campaign/`  
**Audit Date:** October 2026  
**Auditor:** Antigravity Agentic Integrity Auditor  

---

## Executive Summary

An exhaustive independent audit was conducted on the source code, manifests, audio perturbation pipelines, confidence calibration scripts, streaming simulator, and evaluation reports generated during the Phase 30–32 integrated campaign.

### Key Audit Findings

| Audit Dimension | Status | Summary Finding |
| :--- | :---: | :--- |
| **1. Active Region SNR Calculation** | **VERIFIED** | Bounded correctly to `[kws_onset_sec, kws_onset_sec + kws_active_speech_duration_sec]`. Achieved SNR matches requested SNR to $\le 0.05$ dB on all speech clips. |
| **2. Noise Mixing & Amplitude Scaling** | **VERIFIED** | Speech RMS and noise RMS computed accurately; pre-mix and post-mix active region power numerically verified. |
| **3. Output Formatting & Preprocessing** | **VERIFIED** | 16 kHz mono 16-bit PCM WAV, peak amplitudes $\le 0.98$ (no digital clipping), 40-bin log-Mel parameters strictly matched to training. |
| **4. Metric Denominators & Label Mapping** | **VERIFIED** | Label IDs $0..5$ consistently mapped. Denominators for recall, false-activation rate, and macro metrics are mathematically exact. |
| **5. Prediction Count Verification** | **VERIFIED** | All reported rates (45.0% UNK FA rate, 99.38% target recall, 92.08% accuracy) reconcile exactly with row counts in `predictions/*.csv`. |
| **6. Streaming & Latency Implementation** | **VERIFIED WITH DISCLOSURE** | Sliding-window hop (100 ms), consecutive-hit logic ($N=2$), debounce (0.60s), and event matching are internally consistent. Re-used UNKNOWN clips in continuous stream represent correlated observations. |
| **7. Calibration Constraint Discrepancy** | **DISCREPANCY IDENTIFIED** | **Stated constraint (Target Rec $\ge 95\%$, UNK FA $\le 5\%$) was unachievable.** The script silently fell back to a relaxed Pareto compromise (Target Rec $\ge 90\%$, UNK FA $27.5\%–37.5\%$). |

---

## Detailed Audit Investigation

### 1. Active Region SNR & Mixing Fidelity (`noise_generator.py`)
- **Inspection:** In `noise_generator.py`, lines 146–180:
  ```python
  onset_sample = int(row["kws_onset_sec"] * 16000)
  dur_samples = int(row["kws_active_speech_duration_sec"] * 16000)
  active_clean = clean_audio[onset_sample : onset_sample + dur_samples]
  speech_rms = np.sqrt(np.mean(active_clean**2))
  target_noise_rms = speech_rms * (10.0 ** (-target_snr_db / 20.0))
  ```
- **Verification:** Slicing strictly excludes leading and trailing silence. Post-mix verification:
  `achieved_snr_db = 20.0 * np.log10(speech_rms / achieved_noise_rms)`
  On the 3,840 speech clips in the corrupted test manifest, $|achieved\_snr\_db - target\_snr\_db| == 0.000\text{ dB}$, confirming mathematical precision.
- **Silence Nuance:** For `SILENCE` samples (which have zero speech), noise was scaled to the mean active speech RMS of the speech dataset (`mean_ref_speech_rms * 10^(-target_snr / 20)`). In the manifest, `achieved_snr_db` was recorded as `0.0`. This is physically correct (no speech present), but should be explicitly documented to avoid confusion with 0 dB SNR speech.

### 2. Audio Formatting & Preprocessing Pipeline
- All generated audio was stored as 16,000-sample, 16 kHz, single-channel 16-bit PCM WAV.
- Peak amplitude is governed by a safety limiter (`if peak > 0.98: mixed = mixed * (0.98 / peak)`). No waveform clipping or integer overflow occurred.
- Feature extraction uses `LogMelFeatureExtractor` (`n_mels=40`, `win_length=400`, `hop_length=160`, `f_min=20`, `f_max=8000`), exactly identical to Phase 29 training.

### 3. Metric Calculation & Ground-Truth Counts
- **Label Mapping:**
  `0: TRAI`, `1: PHAI`, `2: LEN`, `3: XUONG`, `4: UNKNOWN`, `5: SILENCE`.
- **Clean Test Counts (240 clips total, 40 per class):**
  - Targets (160 clips): Seed 42 misclassified 1 clip (`TRAI` $\rightarrow$ `UNKNOWN`), giving $159/160 = 99.38\%$ target recall.
  - UNKNOWN (40 clips): Seed 42 classified 22 as `UNKNOWN` ($55.0\%$ recall) and 18 as target commands ($18/40 = 45.0\%$ false activation rate).
  - SILENCE (40 clips): Seed 42 classified 40 as `SILENCE` ($100.0\%$ recall, $0.0\%$ false activation).
  - Overall accuracy: $(159 + 22 + 40)/240 = 221/240 = 92.08\%$.
- All numbers in the report match raw sample-by-sample predictions without rounding distortion.

### 4. Root-Cause Analysis of Calibration Constraint Discrepancy
- **The Discrepancy:** The prompt for Phase 31 specified:
  - Minimum target command recall: $\ge 95\%$
  - Maximum false-command activation rate on UNKNOWN: $\le 5\%$
  - Maximum false-command activation rate on SILENCE: $\le 5\%$
- **The Code Logic (`calibrate_threshold.py` lines 177–214):**
  The script evaluated candidate thresholds $\theta \in [0.50, 0.99]$. On the clean validation set, at threshold $\theta = 0.50$, UNK FA rate was $37.5\%$. At threshold $\theta = 0.99$, UNK FA rate was still $22.5\%$ (because several UNKNOWN utterances such as `Không` were predicted with $> 99\%$ confidence by the model), while Target Recall plummeted to $< 70\%$.
- **Consequence:** `feasible_entries` was empty. Rather than raising a hard failure or declaring the operating point infeasible, the code executed a fallback:
  ```python
  compromise_pool = [e for e in results if e["clean"]["target_recall"] >= 0.90]
  best_entry = min(compromise_pool, key=lambda x: (x["clean"]["unknown_false_activation_rate"], -x["clean"]["macro_f1"]))
  ```
  This selected $\theta = 0.60$ (Seed 42), $\theta = 0.75$ (Seed 43), and $\theta = 0.85$ (Seed 44), which yielded UNK FA rates of $27.5\%–37.5\%$.
- **Audit Conclusion:** The stated requirement of $\le 5\%$ UNK FA was **never satisfied**. The original CNN v1 could not achieve this constraint due to severe acoustic-phonetic overlap in the limited 40-word UNKNOWN set.
- **Rule for Phase 33–35 (v2):** In v2, if the strict operating requirements cannot be satisfied, the evaluation must formally declare **NO FEASIBLE OPERATING POINT EXISTS**. Silent relaxation of constraints is strictly prohibited.

### 5. Continuous Streaming Simulator Integrity (`streaming_simulator.py`)
- **Sliding Window:** Step size 1,600 samples (100 ms) and window size 16,000 samples (1.0s) correctly process continuous audio.
- **Hysteresis & Debounce:** $N=2$ consecutive frames $\ge \theta$ required to fire. Lockout period is set to $T_{trigger} + 0.60$s.
- **Matching Window:** Ground truth commands are matched within $[t_{onset} - 0.20s, t_{end} + 0.60s]$.
- **Limitation Disclosed:** The non-target stream (20.0 minutes) was formed by sampling from the 40 test UNKNOWN utterances and repeating them between silence pauses and babble bursts. As a result, the 52–425 false alarms reported per seed represent correlated errors on repeated confusable clips rather than 425 independent linguistic failures. In v2, the non-target stream will be constructed from a vastly expanded vocabulary of non-target speech.

---

## Action Items for Phase 33–35 (v2) Pipeline

1. **Massively Expand UNKNOWN Vocabulary:** Expand from 40 words to $\ge 100$ distinct non-target texts (50+ train, 25+ val, 25+ test, disjoint across splits) and $\ge 500$ accepted audio samples.
2. **Train-Only Acoustic Augmentation:** Train the CNN with dynamic multi-condition noise injection (HVAC, babble, white/pink at 5–20 dB SNR) so the representation learns noise invariance.
3. **Strict Validation Constraint Enforcement:** If validation calibration cannot achieve Target Recall $\ge 95\%$ and False Alarms $\le 1\text{ event/hr}$, report the exact constraint failure without relaxing thresholds.
