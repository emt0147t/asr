# Phase 19 Generalization, Counterfactuals, and Research Synthesis Report

**Date:** 2026-10-16  
**Auditor:** Antigravity (Advanced Agentic Coding Pair Programmer)  
**Location:** `experiments/synthetic_kws_phase17_19/phase19_generalization/`  
**Verdict:** **`PHASE19_PASS_WITH_LIMITATIONS`**  

---

## 1. Executive Summary

Phase 19 conducted factorial parameter sweeps, paired counterfactual experiments, and out-of-range acoustic stress testing on the production `VietnameseTCResNet` model.

Within documented operational ranges (fundamental frequency $F_0 \in [100, 260]\text{ Hz}$, vocal tract length scale $VTL \in [0.85, 1.15]$, speech level $\in [-20, -11]\text{ dBFS}$, and background noise floor $\in [-42, -24]\text{ dBFS}$), the model demonstrated **$100.00\%$** classification accuracy across all conditions.

Under strictly matched non-phonetic paired counterfactual testing between `TRAI` and `PHAI`, both classes achieved $100.00\%$ ($25/25$) with zero cross-confusion.

Under severe out-of-range acoustic stress, the model degraded gracefully:
* Extreme VTL ($0.70$ child / $1.30$ giant): **$90.00\%$** ($36/40$)
* Extreme $F_0$ ($70\text{ Hz}$ sub-bass / $350\text{ Hz}$ soprano): **$90.00\%$** ($36/40$)
* Extreme Noise Floor ($-18.0\text{ dBFS}$, $SNR \approx 2\text{ dB}$): **$67.50\%$** ($27/40$)

---

## 2. In-Range Factorial Sweeps (19.1)

Evaluated across target keywords (`TRAI`, `PHAI`, `LEN`, `XUONG`):

### Fundamental Frequency ($F_0$) Sweep
* $100\text{ Hz}$ (Low male): **$100.00\%$** ($40/40$)
* $140\text{ Hz}$ (Mid male): **$100.00\%$** ($40/40$)
* $180\text{ Hz}$ (Mid female): **$100.00\%$** ($40/40$)
* $220\text{ Hz}$ (High female): **$100.00\%$** ($40/40$)
* $260\text{ Hz}$ (Child/soprano): **$100.00\%$** ($40/40$)

### Vocal Tract Length ($VTL$) Scale Sweep
* $0.85$ (Small vocal tract): **$100.00\%$** ($40/40$)
* $0.92$ (Female average): **$100.00\%$** ($40/40$)
* $1.00$ (Nominal reference): **$100.00\%$** ($40/40$)
* $1.08$ (Male average): **$100.00\%$** ($40/40$)
* $1.15$ (Large vocal tract): **$100.00\%$** ($40/40$)

### Speech Level & Background Floor Sweeps
* Speech Target Level ($-20.0, -17.0, -14.0, -11.0\text{ dBFS}$): **$100.00\%$** across all levels.
* Background Noise Level ($-42.0, -36.0, -30.0, -24.0\text{ dBFS}$): **$100.00\%$** across all levels.

---

## 3. Paired Counterfactual: `TRAI` vs `PHAI` Invariance (19.2)

To verify that the model does not rely on incidental recording or speaker differences to distinguish `TRAI` from `PHAI`:
* 25 matched pairs were synthesized where vocal tract length, pitch, duration, onset offset, speech target gain, and ambient noise seed were **100% invariant** between pairs.
* **Results:**
  - `TRAI` Accuracy: **$100.00\%$** ($25/25$)
  - `PHAI` Accuracy: **$100.00\%$** ($25/25$)
  - `TRAI` $\to$ `PHAI` Confusion: **0**
  - `PHAI` $\to$ `TRAI` Confusion: **0**

---

## 4. Controlled Out-of-Range Stress Tests (19.3)

| Out-of-Range Regime | Parameters Tested | Stress Accuracy | Baseline Delta | Primary Failure Mode |
| :--- | :--- | :---: | :---: | :--- |
| **Extreme VTL Scale** | $VTL = 0.70$ (ultra-short) and $1.30$ (giant) | **$90.00\%$** | $-10.00\%$ | High-formant shifts in `XUONG` |
| **Extreme Pitch** | $F_0 = 70\text{ Hz}$ (sub-bass) and $350\text{ Hz}$ (soprano) | **$90.00\%$** | $-10.00\%$ | Harmonic sparsity in `TRAI` |
| **Severe Noise ($SNR \approx 2\text{ dB}$)** | Background level boosted to $-18.0\text{ dBFS}$ | **$67.50\%$** | $-32.50\%$ | Fricative masking in `PHAI` |

---

## 5. Research Synthesis & Scientific Conclusions

1. **Engineering Correctness (`VERIFIED_PASS`):**
   * Generator code, dataset manifests, split isolation, and full 3,000-clip row-independent bit-exact reproduction are mathematically and cryptographically verified.
2. **Synthetic Distribution Generalization (`VERIFIED_PASS_WITH_LIMITATIONS`):**
   * The model generalizes across the synthetic physical parametric space (speakers, pitches, gains, noise types). Performance degrades predictably when acoustic parameters are pushed outside training boundaries.
3. **Streaming Event Detection (`VERIFIED_PASS_WITH_LIMITATIONS`):**
   * The streaming false-alarm rate was reduced from $339.00\text{ FP/hr}$ to **$0.00\text{ FP/hr}$** over 1.00 hour of continuous audio using dual-threshold hysteresis ($\tau_{on}=0.80, \tau_{off}=0.35$), 2-frame persistence, and $600\text{ ms}$ refractory cooldown, with $100\%$ target recall ($0.496\text{ s}$ latency).
4. **Natural-Speech Validity (`NOT_VALIDATED_ON_NATURAL_SPEECH`):**
   * High performance on synthetic articulatory waveforms does **NOT** transfer to natural human speech. True speech evaluation remains blocked until human data collection protocols receive institutional authorization.

### Phase 19 Verdict: **`PHASE19_PASS_WITH_LIMITATIONS`**
