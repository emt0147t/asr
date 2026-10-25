# PHASE 65: PHASE 59 CONFIRMATORY AUDIT VERIFICATION & POISSON BOUND RECONCILIATION

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Audit Target:** Phase 59 Confirmatory Evaluation Manifests, Event Ledgers, Track A/B Counts, and Corrected Poisson Bounds  
**Governing Manifests:** `PHASE58_CONFIRMATORY_MANIFEST.csv` (420 clips), `PHASE58_TONE_PROBE_MANIFEST.csv` (60 clips), `PHASE58_STREAMING_MANIFEST.csv` (50 sequences), `PHASE58_EVENT_LEDGER.csv` (58 events)  
**Status:** **AUDITED, CROSS-VERIFIED, AND MATHEMATICALLY RECONCILED**  

---

## 1. EXECUTIVE SUMMARY & VERIFICATION SCOPE

This audit cross-references all Phase 59 confirmatory test counts, raw audio files, and event ledgers against the reported values in `PHASE59_FINAL_EVALUATION_RESULTS.json` and `PHASE59_CONFIRMATORY_METRICS.csv`.

### Key Verification Resolutions:
1. **Track A Sample Counts Confirmed:**
   - Primary Track A: Exactly **420 clips** (160 target commands, 160 phonetic distractors, 100 conversational OOVs).
   - Command representation: 4 target commands $\times$ 10 speaking rates ($0.85$ to $1.30$) $\times$ 4 SNR tiers (`clean`, 20 dB, 10 dB, 0 dB) = 160 clips ($N_{\text{inst}} = 40$ unique waveforms).
   - Tone probes: Exactly **60 clips** (15 base instances $\times$ 4 SNRs).
2. **Track B Audio Duration & Subgroups Confirmed:**
   - Manifest duration sum: Exactly **$218.201$ seconds** ($3.63668$ minutes).
   - `pure_silence_noise`: 10 sequences, **$51.627$ seconds** ($0.86045$ minutes).
   - `pure_oov_conversation`: 10 sequences, **$32.493$ seconds** ($0.54155$ minutes).
   - `contextual_commands`: 30 sequences, **$134.081$ seconds** ($2.23468$ minutes).
   - Annotated Ground-Truth Commands: Exactly **58 events**.
3. **Exact Poisson Uncertainty Bounds Reconciled:**
   - The ad-hoc normal approximation formula previously used in `eval_phase59_confirmatory_test.py` is permanently replaced by exact one-sided Poisson confidence limits.
   - For silence ($k = 0$, $T = 0.86045\text{ min}$): Exact 95% Poisson rate is **$3.48\text{ FA/min}$** ($- \ln(0.05) / T$); Rule of Three is **$3.49\text{ FA/min}$** ($3 / T$).
   - For conversational OOV ($k = 1$, $T = 0.54155\text{ min}$): Exact 95% Poisson rate is **$8.76\text{ FA/min}$** ($\chi^2_{4, 0.95} / (2T)$); point estimate is **$1.85\text{ FA/min}$** ($1 / T$).

---

## 2. MANIFEST & AUDIO FILE INTEGRITY TABLE

| Evaluation Track | Manifest File Path | Clip Count | Evaluated Audio Duration | SHA-256 Manifest Checksum | Audit Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Track A Primary** | `manifests/PHASE58_CONFIRMATORY_MANIFEST.csv` | 420 | 420.0 s (1.0s padded) | `9cca06f2c31d0b677d394798b95e35f5...` | **VERIFIED (100%)** |
| **Track A Tone Probes** | `manifests/PHASE58_TONE_PROBE_MANIFEST.csv` | 60 | 60.0 s (1.0s padded) | `1bd694843f9edba63b0c00437472be22...` | **VERIFIED (100%)** |
| **Track B Streaming** | `manifests/PHASE58_STREAMING_MANIFEST.csv` | 50 | 218.201 s (exact sum) | `035ad2795aefe375a7cd79c2ca49daf1...` | **VERIFIED (100%)** |
| **Track B Events** | `manifests/PHASE58_EVENT_LEDGER.csv` | 58 | N/A (58 timestamps) | `e8ca1cf80c8c58bc543546ee505d699a...` | **VERIFIED (100%)** |

---

## 3. COMPREHENSIVE MODEL EVALUATION AUDIT ON PHASE 59 DATA

### 3.1 Track A: Isolated Performance (420 Clips)
All metrics computed from raw logits under locked $\theta_{\text{cmd}} = 0.20$:

| Model | Overall Accuracy | 4-Cmd Macro-F1 | Target Acc (160 cmds) | OOV FAR (260 OOVs) | Tone Probe FAR (60 clips) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 28 $D_2\_H1$** | **74.52%** ($313/420$) | **53.76%** | 42.50% ($68/160$) | **5.77%** ($15/260$) | 6.67% ($4/60$) |
| **Phase 26 $D_0\_H1$** | 70.95% ($298/420$) | 39.02% | 40.00% ($64/160$) | 10.00% ($26/260$) | **3.33%** ($2/60$) |
| **Candidate $D_3\_C$** | 69.29% ($291/420$) | 41.23% | 40.62% ($65/160$) | 13.08% ($34/260$) | 5.00% ($3/60$) |
| **Baseline E1** | 63.33% ($266/420$) | 46.96% | **56.88%** ($91/160$) | 32.69% ($85/260$) | 25.00% ($15/60$) |
| **Baseline v11 (default)** | 43.33% ($182/420$) | 41.72% | 55.62% ($89/160$) | 64.23% ($167/260$) | 63.33% ($38/60$) |

#### Exact Per-Class Recall Breakdown:
- **Phase 28 $D_2\_H1$:** TRAI: **$22.5\%$** ($9/40$), PHAI: $55.0\%$ ($22/40$), LEN: $47.5\%$ ($19/40$), XUONG: $45.0\%$ ($18/40$), REJECT: $\mathbf{94.2\%}$ ($245/260$).
- **Candidate $D_3\_C$:** TRAI: **$2.5\%$** ($1/40$), PHAI: $15.0\%$ ($6/40$), LEN: $90.0\%$ ($36/40$), XUONG: $55.0\%$ ($22/40$), REJECT: $86.9\%$ ($226/260$).
- **Phase 26 $D_0\_H1$:** TRAI: **$0.0\%$** ($0/40$), PHAI: $77.5\%$ ($31/40$), LEN: $82.5\%$ ($33/40$), XUONG: **$0.0\%$** ($0/40$), REJECT: $90.0\%$ ($234/260$).
- **Baseline E1:** TRAI: **$0.0\%$** ($0/40$), PHAI: $97.5\%$ ($39/40$), LEN: $100.0\%$ ($40/40$), XUONG: $30.0\%$ ($12/40$), REJECT: $67.3\%$ ($175/260$).
- **Baseline v11:** TRAI: $42.5\%$ ($17/40$), PHAI: $75.0\%$ ($30/40$), LEN: $60.0\%$ ($24/40$), XUONG: $45.0\%$ ($18/40$), REJECT: $35.8\%$ ($93/260$).

---

### 3.2 Track B: Continuous Streaming Performance (50 Sequences, 58 Events)

| Model | Exact Seq Matches | Command Event F1 | Command Recall | Command Precision | Silence FA/min | Exact Poisson 95% UCB | Conversational OOV FA/min | Exact Poisson 95% UCB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate $D_3\_C$** | **25 / 50 (50.0%)** | **31.40%** | **32.76% (19/58)** | 30.16% | **0.00** | **3.48** | **1.85** | **8.76** |
| **Phase 28 $D_2\_H1$** | 23 / 50 (46.0%) | 19.23% | 17.24% (10/58) | 21.74% | **0.00** | **3.48** | **1.85** | **8.76** |
| **Phase 26 $D_0\_H1$** | 22 / 50 (44.0%) | 28.57% | 22.41% (13/58) | **39.39%** | **0.00** | **3.48** | **1.85** | **8.76** |
| **Baseline E1** | 22 / 50 (44.0%) | 30.08% | 34.48% (20/58) | 26.67% | **0.00** | **3.48** | 7.39 | 16.90 |
| **Baseline v11** | 13 / 50 (26.0%) | 9.95% | 17.24% (10/58) | 6.99% | **0.00** | **3.48** | 27.70 | 42.65 |

---

## 4. AUDIT OF STATISTICAL COMPARISONS AT PREREGISTERED $\alpha = 0.01$

Under dependency-aware clustered paired $t$-tests ($N_{\text{inst}} = 105$ primary waveforms, $df = 104$, $t_{\text{crit}} = 2.623$ at $\alpha = 0.01$):

1. **$D_2\_H1$ vs Baseline $v11$:** $+31.19$ pp ($95\%$ CI: $[+22.66, +39.72]$, $t = 7.2479$, $p = 7.73 \times 10^{-11}$). **STATISTICALLY SIGNIFICANT**.
2. **Phase 26 vs Baseline $v11$:** $+27.62$ pp ($95\%$ CI: $[+18.47, +36.77]$, $t = 5.9868$, $p = 3.09 \times 10^{-8}$). **STATISTICALLY SIGNIFICANT**.
3. **$D_3\_C$ vs Baseline $v11$:** $+25.95$ pp ($95\%$ CI: $[+16.68, +35.22]$, $t = 5.5514$, $p = 2.18 \times 10^{-7}$ ). **STATISTICALLY SIGNIFICANT**.
4. **$D_2\_H1$ vs $D_3\_C$:** $+5.24$ pp ($95\%$ CI: $[+0.41, +10.06]$, $t = 2.1533$, $p = 0.0336$). **NOT SIGNIFICANT AT PREREGISTERED $\alpha = 0.01$**. ($p = 0.0336 > 0.01$).
5. **$D_3\_C$ vs Phase 26:** $-1.67$ pp ($95\%$ CI: $[-7.83, +4.49]$, $t = -0.5366$, $p = 0.5927$). **NOT SIGNIFICANT**.

---

## 5. AUDIT VERDICT

All Phase 59 confirmatory test counts, waveforms, event matches, and Poisson confidence limits have been reconciled and verified against the raw manifests.
