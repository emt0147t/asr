# PHASE 62: AUDIT OF INFERENCE, TTS PROVENANCE CLAIMS, PROTOCOL BOUNDS, AND ACCEPTANCE CRITERIA

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 25, 2026  
**Audit Target:** Phase 55 Preregistration, Phase 57–59 TTS Provenance, Statistical Significance Protocol, and Profile Acceptance Decisions  
**Status:** **AUDITED, SCOPE NARROWED, AND FORMALLY RECONCILED**  

---

## 1. EXECUTIVE SUMMARY & FORENSIC SCOPE AUDIT

This audit provides a rigorous, evidentiary evaluation of the claims made during Phases 56–59 regarding speech engine independence, environmental noise diversity, statistical significance thresholds, and candidate engineering acceptance.

### Key Forensic Resolutions:
1. **TTS Independence Claim Narrowed:**
   - Piper TTS as an inference runtime was previously piloted in Phase 36–39 (`PHASE36_39_INDEPENDENT_SOURCE_READINESS.md`).
   - The specific checkpoint evaluated in Phase 59, `vi_VN-vais1000-medium.onnx`, is derived from the IEEE DataPort VAIS-1000 corpus (CC-BY 4.0, professional Hanoi studio recording) using the VITS end-to-end architecture.
   - Piper VAIS-1000 audio has **zero historical exposure** in training or validation splits for any model in this repository.
   - *Evidentiary Correction:* Piper VAIS-1000 is an independent, unobserved test corpus and voice, but it is **not** an engine family completely uninspected by the repository, as Piper had been screened during Phase 36–39.
2. **ESC-50 Noise Taxonomy Overlap:**
   - Acoustic separation across folds (Fold 1 in Phase 43, Fold 2 in Phase 51, Fold 3 in Phase 58–59) is confirmed at the waveform level ($\max |r| = 0.0947 < 0.1500$).
   - However, the underlying environmental categories (32: door knock, 35: washing machine, 36: vacuum cleaner, 38: clock tick) are semantically identical across all three phases, originating from Karol Piczak's ESC-50 dataset.
3. **Statistical Significance Protocol Audit:**
   - In `PHASE58_SAMPLE_SIZE_AND_CLUSTER_DESIGN.md` (Section 2.1), the preregistered significance threshold was explicitly set to **$\alpha = 0.01$** (to account for family-wise multiple comparisons against $v11$).
   - Pairwise result for $D_2\_H1$ vs $D_3\_C$: $p = 0.0336$.
   - *Protocol Correction:* At the preregistered $\alpha = 0.01$ threshold, the difference between $D_2\_H1$ and $D_3\_C$ is **NOT statistically significant** ($p = 0.0336 > 0.01$). It was nominally significant only at an uncorrected $\alpha = 0.05$ exploratory level.
4. **Low-FA Profile Decision Clarified:**
   - Phase 28 $D_2\_H1$ satisfies all four registered point-estimate criteria for the Low-FA profile (Silence $\le 1.0\text{ FA/min}$, OOV $\le 6.0\text{ FA/min}$, Isolated OOV $\le 20.0\%$, Macro-F1 $\ge 50.0\%$).
   - *Critical Engineering Caveat:* Passing the Low-FA profile is **NOT** evidence of reliable continuous keyword spotting. $D_2\_H1$ achieved only **$22.5\%$ recall on `trái`**, **$17.24\%$ recall in streaming**, and a streaming Command-Event F1 of only **$19.23\%$**. It avoids false alarms through extreme conservative rejection.
5. **Sample-Size Power Analysis Audit:**
   - The reported $>99\%$ statistical power at $\alpha=0.01$ applies strictly to the aggregate accuracy estimand across the $N_{\text{inst}} = 105$ primary waveforms.
   - It does **not** generalize to individual command recalls, where sample size is only $N = 10$ waveforms per command class.

---

## 2. MANIFEST & PREREGISTRATION INTEGRITY AUDIT

All underlying preregistration documents and confirmatory manifests were verified for cryptographic integrity:

| File Artifact | Recorded Role | SHA-256 Checksum | Verification Status |
| :--- | :--- | :--- | :---: |
| `PHASE55_CONFIRMATORY_EVALUATION_PREREGISTRATION.md` | Protocol Registration | `763d93ca46906a6db2f812ecf720e84fdca582247f1aee808526cbb78487fc4d` | **VERIFIED (Bit-for-bit)** |
| `PHASE58_CONFIRMATORY_MANIFEST.csv` | Track A Manifest (420 clips) | `9cca06f2c31d0b677d394798b95e35f565e1d88f49ea1de7a928303b6bbff803` | **VERIFIED (Bit-for-bit)** |
| `PHASE58_TONE_PROBE_MANIFEST.csv` | Tone Probe Manifest (60 clips)| `1bd694843f9edba63b0c00437472be228afa152328cb48a6dd78536a5da9bc82` | **VERIFIED (Bit-for-bit)** |
| `PHASE58_STREAMING_MANIFEST.csv` | Track B Manifest (50 seqs) | `035ad2795aefe375a7cd79c2ca49daf12d0f8afcbc4d0b6ff7a43f83065f0a9b` | **VERIFIED (Bit-for-bit)** |
| `PHASE58_EVENT_LEDGER.csv` | Track B Ground Truth Events | `e8ca1cf80c8c58bc543546ee505d699a5bb17dc83b5bb4d46f50937f070eb48f` | **VERIFIED (Bit-for-bit)** |

---

## 3. SPEECH GENERATOR PROVENANCE & HISTORICAL EXPOSURE

### 3.1 Historical Audit of Piper TTS in Repository
1. **Phase 36–39 Screening:** Piper TTS was audited as a potential evaluation source in `PHASE36_39_INDEPENDENT_SOURCE_READINESS.md`. At that time, `vi_VN-25hours_single-low` was flagged for phonemizer diacritic omissions, leading to the selection of MMS-TTS and gTTS for Phases 43 and 51.
2. **Phase 57 Screening & Selection:** In Phase 57, the `vi_VN-vais1000-medium.onnx` voice was acquired from `rhasspy/piper-voices`. Pilot testing verified complete tone coverage and bit-for-bit determinism ($\max |\Delta| = 0.000000$ at `noise_scale=0.0`).
3. **Training & Validation Separation:**
   - Training datasets: VieNeu-TTS (Phase 13–26), Edge-TTS augmented (Phase 28–31 D2).
   - Validation dataset: `validation_v1.csv` (VieNeu-TTS exclusively).
   - Prior evaluations: Phase 43 (MMS-TTS), Phase 51 (gTTS).
   - **Conclusion:** Piper VAIS-1000 audio is completely held out and unexposed to all 5 neural models during training and hyperparameter tuning. However, descriptions of Piper as a "completely novel engine uninspected by the project" are narrowed: Piper was screened as a runtime candidate in Phase 36–39 and formally deployed with the VAIS-1000 voice in Phase 57–59.

---

## 4. STATISTICAL PROTOCOL & HYPOTHESIS TESTING AUDIT

### 4.1 Significance Threshold Classification
In `PHASE58_SAMPLE_SIZE_AND_CLUSTER_DESIGN.md`:
$$\alpha = 0.01 \quad (\text{Bonferroni-adjusted for 4 planned comparisons against legacy } v11)$$

### 4.2 Pairwise Clustered Paired $t$-Test Results ($df = 104$, $t_{\text{crit}} = 2.623$):
| Hypothesis Comparison | Mean Delta ($\Delta$) | $t$-statistic | $p$-value | Significance at Preregistered $\alpha=0.01$ | Significance at Uncorrected $\alpha=0.05$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$D_2\_H1$ vs Baseline $v11$** | $+31.19$ pp | $+7.2479$ | $7.73 \times 10^{-11}$ | **STATISTICALLY SIGNIFICANT** | Statistically Significant |
| **Phase 26 vs Baseline $v11$** | $+27.62$ pp | $+5.9868$ | $3.09 \times 10^{-8}$ | **STATISTICALLY SIGNIFICANT** | Statistically Significant |
| **$D_3\_C$ vs Baseline $v11$** | $+25.95$ pp | $+5.5514$ | $2.18 \times 10^{-7}$ | **STATISTICALLY SIGNIFICANT** | Statistically Significant |
| **$D_2\_H1$ vs $D_3\_C$** | $+5.24$ pp | $+2.1533$ | $0.0336$ | **NOT SIGNIFICANT** | Statistically Significant |
| **$D_3\_C$ vs Phase 26** | $-1.67$ pp | $-0.5366$ | $0.5927$ | **NOT SIGNIFICANT** | Not Significant |

### 4.3 Protocol Audit Finding:
The difference between $D_2\_H1$ and $D_3\_C$ ($+5.24$ pp, $p = 0.0336$) **fails to meet the preregistered significance threshold of $\alpha = 0.01$**. 
While $D_2\_H1$ showed higher point accuracy, the hypothesis that $D_2\_H1$ is strictly superior to $D_3\_C$ on overall synthetic accuracy cannot be claimed at the preregistered $\alpha = 0.01$ level. It represents an exploratory trend that is nominally significant only at $\alpha = 0.05$.

---

## 5. RECONCILIATION OF ENGINEERING ACCEPTANCE CRITERIA

### 5.1 Pre-Registered Low-FA Profile Audit ($D_2\_H1$):
- Registered point-estimate requirements:
  - Silence false alarms $\le 1.0\text{ FA/min}$ -> Observed: **$0.00$** (**PASS**)
  - Conversational OOV false alarms $\le 6.0\text{ FA/min}$ -> Observed: **$1.85$** (**PASS**)
  - Isolated OOV FAR $\le 20.0\%$ -> Observed: **$5.77\%$** (**PASS**)
  - 4-Command Macro-F1 $\ge 50.0\%$ -> Observed: **$53.76\%$** (**PASS**)
- **Verdict:** The Low-FA profile pass is **MAINTAINED** under its pre-registered criteria.

### 5.2 Mandatory Operational Limitations of $D_2\_H1$:
A model that passes the Low-FA profile under synthetic conditions is **not** suitable for continuous production KWS:
1. **Acoustic Collapse on `trái`:** Track A recall for `trái` was only **$22.5\%$** ($9/40$ correct; $29/40$ misclassified as REJECT).
2. **Poor Streaming Sensitivity:** In Track B continuous streaming, $D_2\_H1$ detected only **10 of 58 ground-truth events**, resulting in a command recall of **$17.24\%$** and a Command-Event F1 of **$19.23\%$**.
3. **Mechanism of False-Alarm Suppression:** $D_2\_H1$'s low false-alarm rate is achieved primarily through aggressive over-rejection rather than fine-grained phonetic discrimination.

### 5.3 Challenger Profile Audit ($D_3\_C$):
- Pre-registered requirements: Macro-F1 $\ge 60\%$, Min Recall $\ge 40\%$, Streaming Event F1 $\ge 50\%$.
- Observed: Macro-F1 = $41.23\%$, Min Recall = $2.50\%$ (on `trái`), Streaming Event F1 = $31.40\%$.
- **Verdict:** Formally **FAILED** on 3 of 5 registered profile targets.

---

## 6. POWER ANALYSIS RE-AUDIT

In `PHASE58_SAMPLE_SIZE_AND_CLUSTER_DESIGN.md`, statistical power was calculated as $>99\%$ at $\alpha=0.01$ for a true effect size of $\Delta = 15$ pp.
- **Unit of Analysis:** Independent synthesis instance waveforms ($N_{\text{inst}} = 105$).
- **Variance Assumption:** Estimated cluster variance across SNR variants $\sigma^2 \le 0.04$ ($\sigma \le 0.20$).
- **Observed Cluster Standard Deviation:** In Phase 59 paired inference, the observed cluster standard deviation of paired differences was $s = 0.249$.
- **Recomputed Power at Observed Variance:**
  $$\text{SE} = \frac{0.249}{\sqrt{105}} \approx 0.0243 \implies t_{\text{stat}} = \frac{0.15}{0.0243} \approx 6.17 \gg t_{\text{crit}} (2.623) \implies \text{Power } > 99.9\%$$
- **Generalization Boundary:** This statistical power applies **exclusively to the 105-cluster aggregate accuracy estimand**. It does **not** provide statistical power to resolve differences in individual command recalls ($N=10$ waveforms per command class). Future protocols must allocate larger per-class waveform quotas to achieve power for class-specific hypotheses.
