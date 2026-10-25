# Phase 13.1 — V5 Pilot Counterfactual and Shortcut Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> This audit investigates the existing 120-clip synthetic V5 pilot for class-correlated nuisance artifacts, label-contract ambiguity, and reproducibility gaps.
> No human speech was collected, ingested, processed, or evaluated. No model retraining or fine-tuning was performed.
> All legacy checkpoints, ONNX models, golden vectors, and V2 manifests remain strictly frozen and cryptographically verified.

---

## 1. Executive Summary & Verdict

| Audit Dimension | Result / Finding | Status |
| :--- | :--- | :---: |
| **Audit Scope** | Counterfactual analysis, shortcut investigation, reproducibility audit, and label policy on V5 pilot | **COMPLETE** |
| **Final Audit Verdict** | **`PASS_WITH_LIMITATIONS`** | **APPROVED** |
| **V4 Stop-Closure Shortcut** | Absent. The 25ms hard zero-clamp (`output[:400] *= 0.05`) is completely gone; `TRAI` closure has acoustic leakage (mean peak $= 0.5067 \ge 0.02$, min peak $= 0.0905$). | **VERIFIED ABSENT** |
| **Incidental Nuisance Shortcuts** | **Identified and Quantified**: Peak-normalization ($0.85$) on high crest-factor fricatives (/f/ in `PHAI`) depresses full-clip RMS ($\eta^2 = 0.6042$). Under SNR-relative mixing, this also depresses pre-onset background floor ($\eta^2 = 0.1806$). | **IDENTIFIED & DISCLOSED** |
| **Counterfactual Independence** | Word-swap tests confirm phonetic features persist across identical nuisance conditions. Fixed-floor counterfactuals prove pre-onset cue can be decoupled. | **PASS** |
| **Full Reproducibility (120 clips)** | 120 / 120 ($100.0\%$) clips bit-exact match on disk and regenerate identically via `reproduce_pilot_clip` using master seed. | **100% BIT-EXACT** |
| **Manifest Metadata Completeness** | Generator manifest omits `open_quotient`, `spectral_tilt_db`, `jitter_pct`, and `pilot_seed_base`, preventing row-independent reconstruction without master seed. | **LIMITATION DISCLOSED** |
| **Label Contract Semantics** | Established formal two-tier taxonomy distinguishing literal silence from 5 ambient non-speech regimes. | **SPECIFIED** |
| **Similarity Without Overclaiming** | Raw Log-Mel similarity is inflated by negative DC energy floor ($0.930 \pm 0.042$ within, $0.913 \pm 0.033$ between). Mean-centering unmasks genuine acoustic dispersion ($0.528 \pm 0.267$ within). | **QUANTIFIED** |
| **Test Suite & Baseline Preservation** | 118 / 118 tests passed across 4 test suites; all 7 cryptographic baselines and 12 golden test fixtures verified bit-exact. | **VERIFIED** |

---

## 2. Cryptographic Baseline & Evidence Integrity

All 7 cryptographic baselines and all 12 Golden Test fixtures were verified bit-for-bit:

| Protected Artifact | Expected SHA-256 Baseline | Actual Computed SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_demo/android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json` | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | **BIT-EXACT MATCH** |

- **Golden Vectors**: All 12 Golden WAVs and 12 Golden Log-Mel `.npy` tensors verified bit-exact.
- **Git Staged Diff**: `git diff --cached` is strictly empty.

---

## 3. Class-Conditional Acoustic Distribution Audit

Full distributions were computed across all 120 pilot clips (20 per class) and serialized into [`phase13_1_audit/class_conditional_acoustic_audit.json`](file:///d:/trainidentificationaudio/phase13_1_audit/class_conditional_acoustic_audit.json).

### 3.1. Robust Quantiles Summary Table (Median [IQR] and Range)

| Feature | `TRAI` | `PHAI` | `LEN` | `XUONG` | `UNKNOWN` | `SILENCE` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spectral Centroid (Hz)** | $1402\ [1316-1518]$ | $2905\ [2794-2999]$ | $1043\ [883-1200]$ | $2753\ [2566-2965]$ | $1859\ [1328-2621]$ | $1057\ [0-2646]$ |
| **HF Ratio ($>3\text{kHz}$)** | $0.007\ [0.004-0.011]$ | $0.239\ [0.187-0.301]$ | $0.000\ [0.000-0.001]$ | $0.063\ [0.038-0.096]$ | $0.026\ [0.004-0.091]$ | $0.027\ [0.000-0.369]$ |
| **Full-Clip RMS** | $0.168\ [0.138-0.191]$ | $0.071\ [0.063-0.082]$ | $0.210\ [0.187-0.228]$ | $0.148\ [0.129-0.165]$ | $0.143\ [0.117-0.168]$ | $0.002\ [0.000-0.006]$ |
| **First 25ms RMS** | $0.012\ [0.008-0.015]$ | $0.005\ [0.003-0.007]$ | $0.013\ [0.009-0.017]$ | $0.010\ [0.006-0.014]$ | $0.009\ [0.006-0.012]$ | $0.002\ [0.000-0.006]$ |
| **First 10ms RMS** | $0.012\ [0.008-0.015]$ | $0.005\ [0.003-0.007]$ | $0.012\ [0.009-0.016]$ | $0.010\ [0.006-0.014]$ | $0.009\ [0.006-0.012]$ | $0.002\ [0.000-0.006]$ |
| **Body 25ms Peak** | $0.621\ [0.129-0.846]$ | $0.573\ [0.380-0.722]$ | $0.435\ [0.244-0.658]$ | $0.697\ [0.572-0.781]$ | $0.620\ [0.472-0.730]$ | $0.000\ [0.000-0.011]$ |
| **Onset Offset (s)** | $0.180\ [0.125-0.236]$ | $0.150\ [0.083-0.208]$ | $0.141\ [0.106-0.190]$ | $0.138\ [0.087-0.203]$ | $0.177\ [0.094-0.230]$ | $0.000\ [0.000-0.000]$ |
| **Active Duration (s)** | $0.460\ [0.432-0.516]$ | $0.431\ [0.377-0.495]$ | $0.485\ [0.421-0.528]$ | $0.468\ [0.428-0.528]$ | $0.466\ [0.411-0.541]$ | $1.000\ [1.000-1.000]$ |

### 3.2. Variance Decomposition & ANOVA (Speech Classes)

One-way ANOVA across the 5 speech classes (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`):

| Feature | $F$-Statistic | $p$-Value | $\eta^2$ (Variance Explained) | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **`hf_energy_ratio`** | $63.64$ | $4.73 \times 10^{-26}$ | **$0.7282$** | **Legitimate Phonetic**: Frication in /f/ and sibilance in /s/ naturally generate high-frequency acoustic energy compared to vowels and laterals. |
| **`spectral_centroid_hz`** | $55.49$ | $4.79 \times 10^{-24}$ | **$0.7003$** | **Legitimate Phonetic**: Alveolar/front vowels (/e/ in `lên`) vs open central (/a/ in `trái`) vs labiodental constriction (/f/). |
| **`full_clip_rms`** | $36.25$ | $2.26 \times 10^{-18}$ | **$0.6042$** | **Incidental Nuisance Shortcut**: Peak-normalization ($0.85$) interacts with unvoiced fricative crest factor to artificially attenuate `PHAI` by $\sim 6\text{ dB}$ relative to `LEN`. |
| **`first_25ms_rms`** | $5.23$ | $7.47 \times 10^{-4}$ | **$0.1806$** | **Incidental Pre-Onset Cue**: Background noise is scaled to maintain constant SNR relative to speech RMS ($target\_bg = speech\_rms / snr$). Lower speech RMS for `PHAI` drives quieter pre-onset background. |
| **`onset_offset_sec`** | $0.69$ | $0.598$ | **$0.0284$** | **No Class Shortcut**: Uniform random distribution ($[0.025, 0.320]\text{ s}$) successfully eliminates temporal clustering. |
| **`duration_sec`** | $0.71$ | $0.590$ | **$0.0289$** | **No Class Shortcut**: Utterance duration is balanced across all speech classes. |

### 3.3. Key Pairwise Effect Sizes (Cohen's $d$) & Overlap

- **`PHAI` vs. `LEN` (RMS Separation)**:
  - Full-Clip RMS: Cohen's $d = -4.38$, distribution overlap $= 0.00\%$. Completely separated!
  - First 25ms RMS: Cohen's $d = -1.78$, distribution overlap $= 28.1\%$.
- **`PHAI` vs. `TRAI` (Phonetic Distinction)**:
  - HF Energy Ratio: Cohen's $d = 4.82$, distribution overlap $= 0.00\%$.
  - Spectral Centroid: Cohen's $d = 7.88$, distribution overlap $= 0.00\%$.
- **`TRAI` vs. `LEN` (Onset & Vowel Transition)**:
  - Spectral Centroid: Cohen's $d = 1.25$, distribution overlap $= 42.5\%$.
  - Full-Clip RMS: Cohen's $d = -1.24$, distribution overlap $= 41.2\%$.

---

## 4. Controlled Counterfactual Generator Audit

Synthetic diagnostic examples were generated in [`phase13_1_audit/diagnostic_audio/`](file:///d:/trainidentificationaudio/phase13_1_audit/diagnostic_audio) to test whether class identity can be decoupled from incidental settings.

### 4.1. Experiment 1: Target Word Swap under Fixed Nuisance

A single acoustic context was held strictly constant across all 4 target words:
- **Fixed Voice Profile**: `SPK_DIAG_BASE` ($vtl=1.00, f_0=160\text{ Hz}, O_q=0.55, \tau=-10\text{ dB/oct}, \text{jitter}=0.75\%$).
- **Fixed Timing & Gain**: $t_{\text{onset}} = 0.150\text{ s}$, $T_{\text{dur}} = 0.450\text{ s}$, $\text{peak} = 0.85$.
- **Fixed Ambient Background**: `room_tone` at $\text{SNR} = 25.0\text{ dB}$, fixed noise seed ($77000$).

| Target Word | Clip RMS | First 25ms RMS | Body First 25ms Peak | Spectral Centroid (Hz) | HF Energy Ratio ($>3\text{kHz}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | $0.1558$ | $0.0120$ | $0.8500$ | $1312.0$ | $0.0086$ |
| **`PHAI`** | **$0.0840$** | **$0.0077$** | $0.7368$ | **$2952.0$** | **$0.2105$** |
| **`LEN`** | **$0.2145$** | **$0.0165$** | $0.7574$ | **$907.2$** | **$0.0003$** |
| **`XUONG`** | $0.1548$ | $0.0133$ | $0.7043$ | $2629.3$ | $0.0531$ |

**Finding**: Even with identical speaker, timing, gain, and noise configuration, `PHAI` exhibits $60\%$ lower clip RMS ($0.084$ vs $0.215$ for `LEN`). This confirms that the RMS difference is intrinsic to the generator's peak-normalization step applied across high crest-factor fricatives.

### 4.2. Experiment 2: Nuisance Sweeps across Target Classes

| Sweep Dimension | Settings Evaluated | Phonetic Persistence | Nuisance Interaction |
| :--- | :--- | :--- | :--- |
| **Noise Type** | `room_tone`, `dither`, `hum` at $\text{SNR} = 25\text{ dB}$ | `PHAI` centroid remains $\sim 2900\text{ Hz}$ and HF ratio $\sim 0.23$ across all noise types; `LEN` centroid remains $< 1000\text{ Hz}$ in tone/hum and rises to $1932\text{ Hz}$ in broadband dither. | Noise type adds broad intra-class variation without collapsing class contrast. |
| **SNR Level** | $15\text{ dB}, 25\text{ dB}, 35\text{ dB}$ | Intended phonetic formants remain clearly separable at all SNRs. | At $15\text{ dB}$, pre-onset background RMS increases proportionally for all classes. |
| **Speaker Profile** | Female-like ($f_0=220, vtl=1.10$), Neutral ($f_0=160, vtl=1.00$), Male-like ($f_0=115, vtl=0.90$) | Class ordering in centroid ($LEN < TRAI < XUONG < PHAI$) is preserved across all simulated speaker profiles. | Speaker VTL shifts centroid by $\pm 100-250\text{ Hz}$ within class. |

### 4.3. Experiment 3: Fixed Background Noise Floor Isolation Test

To confirm whether pre-onset background RMS differences can be eliminated, an identical $-46\text{ dBFS}$ room tone floor was injected into all 4 classes without SNR-relative scaling:

| Class | Standard Generator First 25ms RMS | Fixed Noise Floor First 25ms RMS | Decoupling Result |
| :--- | :---: | :---: | :---: |
| `TRAI` | $0.01425$ | **$0.00518$** | **Identical across all classes** |
| `PHAI` | $0.00874$ | **$0.00518$** | **Identical across all classes** |
| `LEN` | $0.01690$ | **$0.00518$** | **Identical across all classes** |
| `XUONG` | $0.01271$ | **$0.00518$** | **Identical across all classes** |

**Conclusion**: The pre-onset background energy shortcut is an artifact of SNR-relative background scaling. When background floor is decoupled from speech RMS, the pre-onset cue vanishes completely while phonetic contrast remains unaffected.

---

## 5. Full 120-Clip Reproducibility & Manifest Audit

Every row in [`experiments/synthetic_kws_v5_candidate/manifest.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v5_candidate/manifest.csv) was audited:

- **Total Rows Audited**: 120.
- **WAV Existence on Disk**: 120 / 120 ($100.0\%$).
- **Disk SHA-256 Match**: 120 / 120 ($100.0\%$).
- **Regenerated SHA-256 Match**: 120 / 120 ($100.0\%$) via [`reproduce_pilot_clip`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v5_candidate/generator.py#L448-L536) using `pilot_seed_base=50000`.

### Identified Reproducibility Gap (Manifest Completeness):
While the generator reproduces 100% of samples bit-exact when the Python generator script is executed, an external auditor possessing **only** `manifest.csv` cannot reproduce the exact waveforms because:
1. `open_quotient`, `spectral_tilt_db`, and `jitter_pct` are generated dynamically in memory and omitted from the CSV columns.
2. `pilot_seed_base` (the master seed) is not recorded in the manifest.
3. `vtl_scale` is rounded to 4 decimals and `f0_base` to 2 decimals in the CSV, introducing floating-point discrepancies if parsed as raw text.

**Recommendation for V5.1**: Serialize all glottal profile parameters (`open_quotient`, `spectral_tilt_db`, `jitter_pct`) and unrounded floats directly into future manifest releases.

---

## 6. Label Semantics & Silence Policy

Documented in [`phase13_1_audit/silence_label_policy.md`](file:///d:/trainidentificationaudio/phase13_1_audit/silence_label_policy.md):

1. **Current `SILENCE` Contract**: Represents non-vocal ambient presence across 5 regimes (`exact_zeros`, `low_level_dither`, `room_tone`, `synthetic_hum`, `non_speech_transient`).
2. **Current `UNKNOWN` Contract**: Represents non-target human speech across 20 documented Vietnamese syllables.
3. **No In-Flight Relabeling**: Existing pilot samples must remain under their 6 canonical labels to preserve inference graph compatibility.
4. **Proposed Future Taxonomy (V5.1 / V6)**: Introduce an explicit `acoustic_subclass` manifest column to distinguish digital silence from HVAC room tone, mains hum, and impulsive transients.

---

## 7. Pairwise Log-Mel Similarity Distribution Audit

Evaluated on $40 \times 101$ Log-Mel representations ($4040$ dimensions per clip):

| Pair Group | Number of Pairs | Raw Cosine Mean $\pm$ Std | Raw Range | Mean-Centered Mean $\pm$ Std | Mean-Centered Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `TRAI` within | 190 | $0.9083 \pm 0.0415$ | $[0.7819, 0.9758]$ | $0.5531 \pm 0.2287$ | $[-0.1472, 0.9088]$ |
| `PHAI` within | 190 | $0.9292 \pm 0.0297$ | $[0.8426, 0.9915]$ | $0.5445 \pm 0.1909$ | $[0.1105, 0.9380]$ |
| `LEN` within | 190 | $0.9298 \pm 0.0351$ | $[0.8216, 0.9929]$ | $0.6825 \pm 0.1666$ | $[0.0695, 0.9748]$ |
| `XUONG` within | 190 | $0.9182 \pm 0.0340$ | $[0.8292, 0.9920]$ | $0.5904 \pm 0.1678$ | $[0.1477, 0.9463]$ |
| `UNKNOWN` within | 190 | $0.9131 \pm 0.0378$ | $[0.7719, 0.9852]$ | $0.5729 \pm 0.1579$ | $[0.0045, 0.9465]$ |
| `SILENCE` within | 190 | $0.9844 \pm 0.0118$ | $[0.9494, 1.0000]$ | $0.2255 \pm 0.3718$ | $[-0.1127, 0.9995]$ |
| **All Within-Class** | **1,140** | **$0.9305 \pm 0.0416$** | **$[0.7719, 1.0000]$** | **$0.5282 \pm 0.2673$** | **$[-0.1472, 0.9995]$** |
| **All Between-Class** | **6,000** | **$0.9126 \pm 0.0328$** | **$[0.7696, 0.9910]$** | **$0.4583 \pm 0.2584$** | **$[-0.2456, 0.9640]$** |

### Scientific Interpretation:
- **Raw Similarity Inflation**: Raw cosine similarity is elevated ($0.913$ between classes) because of the global negative DC baseline of Log-Mel filterbanks.
- **Mean-Centered Unmasking**: Subtracting the mean vector unmasks true spectro-temporal variance ($0.528 \pm 0.267$ within-class), proving that V5 avoids the static template collapse of V4 ($>0.97$) while maintaining distinct cluster centers.

---

## 8. Test Suite Execution & Acceptance

```
tests/test_synthetic_kws_v5_generator.py .................               [17 passed]
tests/test_kws_export_parity.py .............                            [13 passed]
tests/test_synthetic_kws_demo.py ...................                     [19 passed]
tests/test_campaign_acceptance.py ...................................... [69 passed]
============================= 118 passed in 41.5s =============================
```

- **Exit codes**: 0 across all runs.
- **Failures / Errors**: 0.
- **Git status**: Clean, no staged changes (`git diff --cached` is empty).

---

## 9. Final Deliverables Manifest

1. [`phase13_1_audit/class_conditional_acoustic_audit.json`](file:///d:/trainidentificationaudio/phase13_1_audit/class_conditional_acoustic_audit.json): Distribution quantiles, Cohen's $d$ effect sizes, ANOVA, and attribution statistics.
2. [`phase13_1_audit/counterfactual_generator_audit.json`](file:///d:/trainidentificationaudio/phase13_1_audit/counterfactual_generator_audit.json): Controlled counterfactual experiments (word swaps, nuisance sweeps, fixed-floor test).
3. [`phase13_1_audit/reproducibility_audit.json`](file:///d:/trainidentificationaudio/phase13_1_audit/reproducibility_audit.json): 120-row reproduction verification and manifest schema gap analysis.
4. [`phase13_1_audit/silence_label_policy.md`](file:///d:/trainidentificationaudio/phase13_1_audit/silence_label_policy.md): Acoustic label taxonomy and semantic boundary specification.
5. [`phase13_1_audit/similarity_audit.json`](file:///d:/trainidentificationaudio/phase13_1_audit/similarity_audit.json): Raw and mean-centered Log-Mel similarity distributions across 7,140 pairs.
6. [`phase13_1_audit/diagnostic_audio/`](file:///d:/trainidentificationaudio/phase13_1_audit/diagnostic_audio): 20 temporary counterfactual diagnostic WAVs.
7. [`phase13_1_audit/PHASE13_1_SHORTCUT_AUDIT.md`](file:///d:/trainidentificationaudio/phase13_1_audit/PHASE13_1_SHORTCUT_AUDIT.md): This comprehensive audit report.
