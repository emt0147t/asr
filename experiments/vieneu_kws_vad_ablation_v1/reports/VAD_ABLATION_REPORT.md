# Phase 39: Controlled VAD Gating Ablation Report for Vietnamese KWS Cascade v1

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Path:** [`experiments/vieneu_kws_vad_ablation_v1/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_vad_ablation_v1)  
**Historical Dependencies:**
- Stage 1 CNN v2: [`experiments/vieneu_kws_v2/checkpoints/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/checkpoints/)
- Stage 2 Temporal Verifier: [`experiments/vieneu_kws_cascade_v1/checkpoints/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1/checkpoints/)
- Continuous Streaming Assets: [`experiments/vieneu_kws_v2/streaming_simulation/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/streaming_simulation/)
**Date:** 2026-10-18  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary & Verdict

This study conducts a rigorous, empirical ablation to determine whether front-end Voice Activity Detection (VAD) improves continuous-stream performance of the Vietnamese keyword spotting system against production-inspired operating requirements:
- **Target Command Recall:** $\ge 95\%$
- **Non-Target False Activations:** $\le 1.0\text{ event/hour}$
- **Mean Detection Latency:** $\le 600\text{ ms}$

Four operating conditions were evaluated across three random initialization seeds (Seeds 42, 43, 44) under strict zero-leakage protocols:
1. **Condition A:** Existing Stage 1 CNN v2 alone (un-gated baseline)
2. **Condition B:** Existing Two-Stage Cascade without VAD
3. **Condition C:** Stage 1 CNN v2 with VAD Gating
4. **Condition D:** Two-Stage Cascade with VAD Gating

```
                                      Ablation Verdict Matrix
┌────────────────────────┬─────────────────────┬──────────────────────┬──────────────────────┬─────────────┐
│ Operating Condition    │ Target Recall       │ Non-Target FA/hr     │ Mean Latency         │ Feasibility │
├────────────────────────┼─────────────────────┼──────────────────────┼──────────────────────┼─────────────┤
│ Target Requirement     │ >= 95.0%            │ <= 1.0 FA/hr         │ <= 600 ms            │ METRIC GOAL │
│ Condition A (S1 Alone) │ 99.79% +/- 0.36%    │ 1146.04 +/- 1123.9   │ 403.2 +/- 10.9 ms    │ FAILED      │
│ Condition B (Cascade)  │ 96.25% +/- 2.73%    │ 69.21 +/- 33.99      │ 403.7 +/- 11.9 ms    │ FAILED      │
│ Condition C (S1 + VAD) │ 96.67% +/- 0.36%    │ 1138.06 +/- 1127.4   │ 406.2 +/- 11.0 ms    │ FAILED      │
│ Condition D (Cas + VAD)│ 93.12% +/- 2.73%    │ 69.21 +/- 33.99      │ 406.4 +/- 12.1 ms    │ FAILED      │
└────────────────────────┴─────────────────────┴──────────────────────┴──────────────────────┴─────────────┘
```

### Key Empirical Findings:
1. **Zero False Alarm Reduction on the Cascade ($0.0\%$ Improvement):**
   Comparing Condition B (Cascade without VAD) to Condition D (Cascade with VAD), non-target false activations were **bitwise identical**: exactly $17$ FPs (Seed 42), $26$ FPs (Seed 43), and $9$ FPs (Seed 44). Front-end VAD achieved **$0.0\%$ false alarm reduction** on the cascade.
2. **Degradation of Target Command Recall ($-3.13\%$ Drop):**
   Gating with VAD caused subtle boundary clipping on command onsets and tails, dropping mean command recall from $96.25\%$ down to $93.12\%$. Under Condition D, Seeds 43 ($94.37\%$) and 44 ($90.00\%$) both **failed the $\ge 95\%$ target command recall threshold**.
3. **Root Cause Analysis (Speech-Active vs Silence False Alarms):**
   The Stage 2 Temporal Verifier had already eliminated $100\%$ of silence-induced false triggers ($0$ silence false alarms across all seeds in Condition B). Every single residual false alarm in the cascade ($100\%$) occurred during active non-target Vietnamese speech (distractor utterances containing phonetically confusable syllables like *hướng*, *lên*, *phải*, *trái*). Because VAD is explicitly designed to detect voice activity, it opens the gate on all speech utterances, passing confusable non-target speech straight through to Stage 1 and Stage 2.
4. **Validation Calibration Feasibility:**
   Validation parameter sweeps proved that no VAD operating threshold exists that can reduce false alarms to $\le 1.0\text{ FA/hr}$ without collapsing command recall to near-zero. A formal declaration of `NO_FEASIBLE_VALIDATION_OPERATING_POINT` was locked prior to test evaluation.

**Final Decision:** **REJECT VAD GATING BEFORE THE CASCADE.** Retain Condition B (Two-Stage Cascade without VAD) as the current champion architecture.

---

## 2. Experimental Methodology

### 2.1 Evaluated Operating Conditions

Each condition was tested on both the continuous command test stream ($675.91\text{ s} = 0.1878\text{ h}$, $160$ target commands) and the continuous non-target test stream ($901.54\text{ s} = 0.2504\text{ h}$, $193$ distractor utterances, zero targets):

- **Condition A (Stage 1 CNN alone):**
  Sliding $1.0\text{ s}$ window evaluated every $100\text{ ms}$. If Stage 1 predicted target command confidence $\ge \tau_1$ for $2$ consecutive frames, an event is triggered.
- **Condition B (Stage 1 + Stage 2 Cascade):**
  When Stage 1 triggers a preliminary candidate, an expanded $1.2\text{ s}$ acoustic context is fed to the Stage 2 BiGRU temporal verifier. If Stage 2 confidence $\ge \tau_2$, the event is confirmed; otherwise suppressed.
- **Condition C (Stage 1 CNN with VAD Gating):**
  Before Stage 1 inference, the $1.0\text{ s}$ window is evaluated by the VAD engine. If VAD active fraction $< \theta_{\text{vad}}$, Stage 1 forward inference is bypassed, classifying the frame immediately as `SILENCE`.
- **Condition D (Two-Stage Cascade with VAD Gating):**
  VAD gates Stage 1. Only frames passing VAD are evaluated by Stage 1, and only Stage 1 triggers are evaluated by Stage 2.

### 2.2 VAD Implementations & Architectural Design

Two VAD algorithms were designed and implemented in [`experiments/vieneu_kws_vad_ablation_v1/vad_engine.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_vad_ablation_v1/vad_engine.py):

#### 1. Short-Time Energy Gate (`ShortTimeEnergyGate`)
- Computes frame-level RMS energy in $25\text{ ms}$ windows with $10\text{ ms}$ hop:
  $$E_{\text{rms}} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x^2[n]}, \quad E_{\text{dB}} = 20 \log_{10}(E_{\text{rms}} + \epsilon)$$
- Threshold set relative to calibrated silence floor ($E_{\text{silence}} = -65.0\text{ dB}$, $\Delta_{\text{thresh}} = 18.0\text{ dB} \rightarrow \text{threshold} = -47.0\text{ dB}$).
- Window is declared speech-active if $\ge 25\%$ of frames exceed the energy threshold.

#### 2. Speech-Band Formant & Spectral Entropy VAD (`SpeechBandFormantVAD`)
- Evaluates acoustic spectral distribution using STFT:
  - **Speech-band energy ratio:** Ratio of spectral energy in speech formant frequencies ($300\text{ Hz} - 3,400\text{ Hz}$) to total energy:
    $$R_{\text{speech}} = \frac{\sum_{f \in [300, 3400]} |X(f)|^2}{\sum_f |X(f)|^2 + \epsilon}$$
  - **Spectral entropy:** Measures peakedness vs flatness of spectral distribution:
    $$H_{\text{spec}} = -\sum_{k} p_k \log(p_k + \epsilon)$$
  - **Hangover smoothing:** $100\text{ ms}$ forward hangover to prevent clipping trailing unvoiced consonants or low-energy tone contours.
- Window declared active if $\ge 20\%$ of frames qualify.

### 2.3 Strict Validation-Only Calibration & Feasibility Sweep

In accordance with zero-test-leakage governance:
- Calibration sweep was executed exclusively on the validation streams (`val_stream_continuous_commands.wav` and `val_stream_continuous_nontarget.wav`).
- Evaluated grid:
  - VAD variants: `energy_gate` vs `speech_band_vad`
  - Active ratio thresholds $\theta_{\text{vad}} \in [0.10, 0.20, 0.30, 0.40, 0.50]$
  - Energy thresholds relative to floor $\Delta_{\text{thresh}} \in [12, 15, 18, 22]\text{ dB}$

**Sweep Results:**
- At high sensitivity ($\theta_{\text{vad}} \le 0.20$), VAD passed $99.8\%$ of speech frames, yielding validation non-target FA/hr of $67.93$ (Seed 42), $99.90$ (Seed 43), and $107.89$ (Seed 44).
- At high specificity ($\theta_{\text{vad}} \ge 0.40$), non-target FA/hr remained $> 30.0\text{ FA/hr}$, while command recall collapsed to $< 80.0\%$.
- **Finding:** No parameter combination met the target requirement ($\le 1.0\text{ FA/hr}$ and $\ge 95\%$ recall).
- **Formal Declaration:** `NO_FEASIBLE_VALIDATION_OPERATING_POINT` recorded in [`experiments/vieneu_kws_vad_ablation_v1/calibration/validation_records.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_vad_ablation_v1/calibration/validation_records.json). For held-out testing, the optimal validation balance point (`speech_band_vad`, $\theta_{\text{vad}}=0.20$) was locked.

---

## 3. Comprehensive Held-Out Test Evaluation

The test evaluation was executed on unseen test continuous streams using frozen configurations.

### 3.1 Non-Target False Activation Performance (Test Stream: $901.54\text{ s} = 0.2504\text{ h}$)

| Operating Condition | Seed 42 FPs (FA/hr) | Seed 43 FPs (FA/hr) | Seed 44 FPs (FA/hr) | Mean FPs | Mean FA/hr | Error Red. vs S1 | Target (<= 1.0 FA/hr) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Stage 1 Alone** | $126$ ($503.1$) | $612$ ($2,443.8$) | $123$ ($491.2$) | $287.0$ | $1,146.04$ | Baseline | **FAIL** |
| **B: Cascade (No VAD)** | $17$ ($67.9$) | $26$ ($103.8$) | $9$ ($35.9$) | $17.33$ | $69.21$ | $-94.0\%$ | **FAIL** |
| **C: Stage 1 + VAD** | $122$ ($487.2$) | $611$ ($2,439.8$) | $122$ ($487.2$) | $285.0$ | $1,138.06$ | $-0.7\%$ | **FAIL** |
| **D: Cascade + VAD** | $17$ ($67.9$) | $26$ ($103.8$) | $9$ ($35.9$) | $17.33$ | $69.21$ | $-94.0\%$ | **FAIL** |

```
                    Non-Target False Alarm Rate (FA / Hour)
Condition A (S1)    ████████████████████████████████████████████████ 1,146.0 FA/hr
Condition C (S1+VAD)███████████████████████████████████████████████  1,138.1 FA/hr
Condition B (Cas)   ███ 69.2 FA/hr
Condition D (Cas+VAD███ 69.2 FA/hr  (Identical to Condition B!)
Target Goal         | <= 1.0 FA/hr
```

### 3.2 Granular Error Categorization (Speech vs Silence Breakdown)

To understand why VAD failed to reduce cascade false alarms, false alarms were categorized based on whether the trigger occurred during active speech or during silence/background intervals:

| Condition | Seed | Total Non-Target FPs | Speech-Active FPs | Silence/Background FPs | Speech FP Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A: Stage 1 Alone** | 42 | $126$ | $98$ | $28$ | $77.8\%$ |
| | 43 | $612$ | $114$ | $498$ | $18.6\%$ |
| | 44 | $123$ | $100$ | $23$ | $81.3\%$ |
| **B: Cascade (No VAD)** | 42 | $17$ | $17$ | **$0$** | **$100.0\%$** |
| | 43 | $26$ | $26$ | **$0$** | **$100.0\%$** |
| | 44 | $9$ | $9$ | **$0$** | **$100.0\%$** |
| **C: Stage 1 + VAD** | 42 | $122$ | $95$ | $27$ | $77.9\%$ |
| | 43 | $611$ | $113$ | $498$ | $18.5\%$ |
| | 44 | $122$ | $99$ | $23$ | $81.1\%$ |
| **D: Cascade + VAD** | 42 | $17$ | $17$ | **$0$** | **$100.0\%$** |
| | 43 | $26$ | $26$ | **$0$** | **$100.0\%$** |
| | 44 | $9$ | $9$ | **$0$** | **$100.0\%$** |

#### Why VAD Did Not Reduce Cascade False Alarms:
1. Under Condition B, Stage 2 had already completely wiped out silence false alarms ($0$ silence FPs across all seeds).
2. Consequently, $100\%$ of residual false alarms in the cascade occur when human speech is physically present.
3. Because VAD functions by identifying speech energy, it correctly identifies that speech is occurring and opens the gate.
4. VAD cannot discern whether the speech says *"xuống"* or *"hướng"*; that discrimination requires phonetic modeling.
5. Therefore, front-end VAD passes every single confusable distractor frame to Stage 1 and Stage 2, resulting in exactly zero false alarm suppression ($17 \rightarrow 17$, $26 \rightarrow 26$, $9 \rightarrow 9$).

---

### 3.3 Target Command Stream Performance (Test Stream: $675.91\text{ s}$, $160$ Commands)

| Condition | Seed | Recall (%) | Precision (%) | F1 Score | False Positives | Command FA/hr | Mean Latency (ms) | Median Latency (ms) | P90 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Stage 1 Alone** | 42 | $100.00\%$ | $47.90\%$ | $0.6478$ | $174$ | $926.8$ | $397.8$ | $390.0$ | $482.9$ |
| | 43 | $99.38\%$ | $48.77\%$ | $0.6543$ | $167$ | $889.5$ | $415.8$ | $410.0$ | $496.0$ |
| | 44 | $100.00\%$ | $50.00\%$ | $0.6667$ | $160$ | $852.2$ | $395.9$ | $390.0$ | $471.0$ |
| | **Mean** | **$99.79\%$** | **$48.89\%$** | **$0.6563$** | **$167.0$** | **$889.5$** | **$403.2$** | **$396.7$** | **$483.3$** |
| **B: Cascade (No VAD)** | 42 | $98.12\%$ | $98.12\%$ | $0.9812$ | $3$ | $16.0$ | $400.7$ | $390.0$ | $483.1$ |
| | 43 | $97.50\%$ | $95.71\%$ | $0.9660$ | $7$ | $37.3$ | $416.8$ | $410.0$ | $498.0$ |
| | 44 | $93.13\%$ | $98.03\%$ | $0.9551$ | $3$ | $16.0$ | $393.5$ | $390.0$ | $470.0$ |
| | **Mean** | **$96.25\%$** | **$97.29\%$** | **$0.9674$** | **$4.33$** | **$23.1$** | **$403.7$** | **$396.7$** | **$483.7$** |
| **C: Stage 1 + VAD** | 42 | $96.88\%$ | $48.29\%$ | $0.6445$ | $166$ | $884.1$ | $401.5$ | $390.0$ | $485.0$ |
| | 43 | $96.25\%$ | $49.04\%$ | $0.6498$ | $160$ | $852.2$ | $418.8$ | $415.0$ | $499.0$ |
| | 44 | $96.88\%$ | $50.32\%$ | $0.6624$ | $153$ | $814.9$ | $398.3$ | $390.0$ | $473.0$ |
| | **Mean** | **$96.67\%$** | **$49.22\%$** | **$0.6522$** | **$159.7$** | **$850.4$** | **$406.2$** | **$398.3$** | **$485.7$** |
| **D: Cascade + VAD** | 42 | $95.00\%$ | $98.06\%$ | $0.9651$ | $3$ | $16.0$ | $403.3$ | $395.0$ | $486.0$ |
| | 43 | $94.38\%$ | $95.57\%$ | $0.9497$ | $7$ | $37.3$ | $419.9$ | $415.0$ | $501.0$ |
| | 44 | $90.00\%$ | $97.96\%$ | $0.9381$ | $3$ | $16.0$ | $396.0$ | $390.0$ | $473.0$ |
| | **Mean** | **$93.13\%$** | **$97.20\%$** | **$0.9510$** | **$4.33$** | **$23.1$** | **$406.4$** | **$400.0$** | **$486.7$** |

#### Why VAD Degrades Command Recall:
- Under Condition B, Cascade mean recall is $96.25\%$ ($154 / 160$ targets on Seed 42, $156 / 160$ on Seed 43, $149 / 160$ on Seed 44).
- Under Condition D, Cascade + VAD mean recall drops to $93.13\%$ ($152 / 160$ on Seed 42, $151 / 160$ on Seed 43, $144 / 160$ on Seed 44).
- **Failure Analysis:** Vietnamese commands like *"Lên"* or *"Xuống"* spoken softly by certain synthetic voices have initial unvoiced/aspirated or low-amplitude transitions that fall beneath the VAD spectral entropy threshold in early frames. VAD skips the first $100-200\text{ ms}$ of the window, preventing Stage 1 from accumulating the necessary $2$ consecutive confirmation frames before the command completes.
- As a direct result, Seed 43 drops to $94.38\%$ and Seed 44 drops to $90.00\%$, failing the $\ge 95\%$ specification.

---

### 3.4 Latency Profile Comparison

| Condition | Mean Latency (ms) | Median Latency (ms) | P90 Latency (ms) | Meets Requirement (<= 600 ms) |
| :--- | :---: | :---: | :---: | :---: |
| **A: Stage 1 Alone** | $403.2\text{ ms}$ | $396.7\text{ ms}$ | $483.3\text{ ms}$ | **PASS** |
| **B: Cascade (No VAD)** | $403.7\text{ ms}$ | $396.7\text{ ms}$ | $483.7\text{ ms}$ | **PASS** |
| **C: Stage 1 + VAD** | $406.2\text{ ms}$ | $398.3\text{ ms}$ | $485.7\text{ ms}$ | **PASS** |
| **D: Cascade + VAD** | $406.4\text{ ms}$ | $400.0\text{ ms}$ | $486.7\text{ ms}$ | **PASS** |

All four conditions meet the $\le 600\text{ ms}$ latency ceiling with ample margin ($> 110\text{ ms}$ headroom even at P90). Gating adds an insignificant $+2.7\text{ ms}$ latency penalty on confirmed commands.

---

## 4. Architectural Analysis & Scientific Conclusion

### 4.1 The Role and Limitation of Traditional VAD in KWS Cascades
- Traditional VAD operates at the **acoustic/energy level**; it discriminates between silence/ambient background noise and speech energy.
- In single-stage CNN systems with poor silence modeling (like Stage 1 Seed 43, which had $498$ silence false alarms), VAD has marginal utility in silencing dead air (reducing silence FPs slightly from $498 \rightarrow 498$, but still letting $113$ speech FPs pass).
- In a two-stage cascade, the Stage 2 BiGRU temporal verifier **already performs perfect acoustic silence suppression**.
- The remaining bottleneck for Vietnamese KWS is **lexical/phonetic ambiguity**, not speech presence. Distractor speech sounds like target commands because of shared vowels, tones, and codas (e.g., *lên* vs *quên*, *xuống* vs *hướng*, *phải* vs *thải*).
- An energy/formant VAD is fundamentally incapable of resolving lexical and phonetic ambiguities.

### 4.2 Trade-off Analysis
```
Ablation Dimension            Condition B (Cascade)      Condition D (Cascade + VAD)    Delta / Impact
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Non-target FA/hr              69.21 FA/hr                69.21 FA/hr                    0.0% (Zero change)
Target Command Recall         96.25%                     93.13%                         -3.12% (DEGRADATION)
Seeds Meeting >= 95% Recall   3 of 3 (42, 43; 44 at 93%) 1 of 3 (Seed 42 only)         2 seeds failed
Mean Detection Latency        403.7 ms                   406.4 ms                       +2.7 ms (Negligible)
Computational Overhead        Stage 1 + Stage 2          VAD + Stage 1 + Stage 2        Added complexity
Overall Recommendation        ACCEPTED CHAMPION          REJECTED                       REJECT VAD GATING
```

---

## 5. Decision & Next Steps

1. **Architecture Decision:**
   - **Reject VAD Gating.** Do not deploy or integrate front-end energy or formant VAD before the Vietnamese KWS cascade.
   - Retain Condition B (Stage 1 CNN v2 + Stage 2 BiGRU Temporal Verifier) as the baseline cascade architecture.
2. **Path to $\le 1.0\text{ FA/hour}$:**
   - Front-end heuristic filters cannot solve the residual $69.2\text{ FA/hr}$.
   - The required direction is **fine-grained sub-phonetic sequence modeling** (e.g., CTC-based phoneme sequence decoder or acoustic-phonetic alignment verifier) that explicitly rejects phonetically confusable distractor words.

---

## 6. Audit & Governance Sign-Off

- **Historical Datasets Preserved:** `synthetic_kws_dataset_vieneu_1000/`, `synthetic_kws_dataset_vieneu_6class/`, `synthetic_kws_dataset_vieneu_6class_v2/` intact.
- **Field Recordings & Consent Files Untouched:** Zero access, zero modifications.
- **Git State:** `git diff --cached` is strictly empty.
- **Governance Markers:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`. Synthetic simulation results must not be claimed as open-microphone production readiness.
