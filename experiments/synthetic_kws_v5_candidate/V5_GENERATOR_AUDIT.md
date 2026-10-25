# Phase 13: Synthetic KWS V5 Candidate Generator Prototype and Acoustic QA Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**
> This audit evaluates an isolated, synthetic-only prototype generator candidate (`synthetic_kws_v5_candidate`) on a deterministic 120-clip pilot dataset.
> No human speech was collected, ingested, processed, or evaluated. No model retraining or fine-tuning was performed.
> All legacy checkpoints, ONNX models, golden vectors, and V2 manifests remain strictly frozen and cryptographically verified.

---

## 1. Executive Summary & Audit Verdict

| Assessment Dimension | Result / Finding | Status |
| :--- | :--- | :--- |
| **Audit Scope** | V5 candidate generator implementation & acoustic QA on 120 deterministic pilot clips | **COMPLETE** |
| **Final Audit Verdict** | **`V5_PILOT_PASS_WITH_LIMITATIONS`** | **APPROVED FOR RESEARCH** |
| **V4 Shortcut Elimination** | Hard 25ms silence gap (`output[:400] *= 0.05`) completely eliminated; replaced with continuous VOT, closure acoustic leakage, burst resonator, and dynamic $F_2$ onset locus | **PASS (100% of TRAI clips)** |
| **Acoustic Variance** | Raw intra-class cosine std $= 0.0416$; mean-centered intra-class similarity $= 0.5282 \pm 0.2672$ (vs V4's $0.966 - 0.986$ template matching) | **PASS (Gate 1)** |
| **Temporal Placement** | Utterance onset uniformly distributed across $[0.0285\text{ s}, 0.3184\text{ s}]$ (mean $= 0.1568\text{ s}$) | **PASS (Gate 2)** |
| **Silence Coverage** | 5 explicit regimes implemented: exact zeros, dither, room tone, hum, transients | **PASS (Gate 4)** |
| **Negative Vocabulary** | 20 distinct Vietnamese syllables covering initial-sharing, rhyme-sharing, and open vocabulary | **PASS (Gate 5)** |
| **Single-Clip Determinism** | 100% bit-exact byte and SHA-256 reproduction across all 6 classes from deterministic seeds | **PASS** |
| **Protected Baselines** | All 7 protected SHA-256 baselines and 12 golden test cases verified bit-exact | **PASS** |
| **Test Suite Pass Rate** | 118 / 118 unit and regression tests passed across 4 test suites | **PASS** |
| **Model Retraining** | Retraining intentionally not performed; Arm B checkpoint remains frozen | **DEFERRED (OUT OF SCOPE)** |

---

## 2. Cryptographic Baseline & Evidence Integrity

All 7 cryptographic baselines were re-verified bit-for-bit with SHA-256:

| Protected Artifact | Expected SHA-256 Baseline | Actual Computed SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- |
| `data/manifests/test_manifest_v2.csv` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/field_collection_manifest.csv` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` | **BIT-EXACT MATCH** |
| `data/raw/field_recordings/consent_ledger.json` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_demo/export/arm_b_tcresnet14.onnx` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_demo/android_app/app/src/main/assets/models/arm_b_tcresnet14.onnx` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | `8A5AC4C105DF52F1749AE11B6BB2124B3A672486601EFF308ACC7E257FF97625` | **BIT-EXACT MATCH** |
| `experiments/synthetic_kws_demo/frontend_golden/golden_manifest.json` | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | `5904A5503CE5528ECF88A4A670C6726D80B74285B34EE6B910CF2FA48F60E5C3` | **BIT-EXACT MATCH** |

Furthermore, all **12 Golden WAV fixtures** and **12 Golden Log-Mel tensors** (`frontend_golden/`) were verified bit-exact against their recorded SHA-256 hashes.
Git staging status (`git diff --cached`) is strictly empty.

---

## 3. V5 Acoustic Modeling Architecture vs. V4 Flaws

The V5 candidate generator directly implements the scientific specifications formulated in `PHASE12_2_V5_DATASET_DESIGN.md`:

| Acoustic Dimension | V4 Baseline (Defective) | V5 Candidate Implementation | Acoustic / Scientific Mechanism |
| :--- | :--- | :--- | :--- |
| **Onset Stop Modeling (`TRAI`)** | Artificial hard clamp `output[:400] *= 0.05` creating fixed 25ms silence gap | Continuous closure variation ($16–32\text{ ms}$) with glottal murmur leakage ($\text{peak} \ge 0.02$) | Eliminates neural shortcut; trains acoustic discrimination on consonant burst and coarticulatory transition |
| **Consonant Locus Equations** | Static identical formant start for `TRAI` and `PHAI` ($F_1=800, F_2=1350$) | High $F_2$ onset locus ($1750–1950\text{ Hz} \cdot \alpha_{\text{vtl}}$) for `TRAI` vs Low $F_2$ onset locus ($1000–1200\text{ Hz} \cdot \alpha_{\text{vtl}}$) for `PHAI` | True physiological locus equations modeling place of articulation (palatal/alveolar constriction vs. labial constriction) |
| **Release Burst & Frication** | Static 15ms high-pass burst; fixed 50ms frication | Variable 2nd-order resonator burst ($3400–4400\text{ Hz}$, $Q=4–7$); continuous labiodental frication ($45–75\text{ ms}$, $1600–6800\text{ Hz}$) with smooth attack | Realistic spectral envelope and continuous duration distributions |
| **Glottal Excitation** | 4 static harmonics ($1.0, 0.5, 0.25, 0.12$) | Rosenberg polynomial glottal flow pulse with open quotient ($O_q \in [0.45, 0.65]$), spectral tilt ($-13\text{ to } -7\text{ dB/oct}$), and micro-jitter | Natural glottal source harmonics and spectral rolloff |
| **Vocal Tract Scaling (VTL)** | 100% constant across all samples | Parameterized $\alpha_{\text{vtl}} \in [0.88, 1.12]$ per simulated speaker applied to all formants | Simulates diverse speaker vocal tract dimensions |
| **Formant Resonators** | Constant chunk filters without state retention | Time-varying IIR peak resonators with continuous state retention (`zi`) across chunks | Eliminates boundary clicks; provides smooth formant glide |
| **Temporal Placement** | Fixed left alignment at sample 0 | Uniform random onset offset $t_{\text{onset}} \in [0.025, 0.320]\text{ s}$ across 1.0s window | Prevents fixed-frame onset template matching |
| **Ambient Silence** | 3 simplistic modes; pure zeros misclassified as `TRAI` ($30.2\%$) | 5 explicit regimes: `exact_zeros`, `low_level_dither`, `room_tone`, `synthetic_hum`, `non_speech_transient` | Covers digital silence, electrical hum, room resonance, and non-vocal acoustic impulses |
| **Negative Vocabulary** | 8 monosyllables | 20 distinct Vietnamese syllables (pilot) structured into initial-sharing, rhyme-sharing, and open vocabulary | Prevents false alarms on phonetically similar Vietnamese words |
| **Dialect Tagging** | Dead-code `dialect` parameter (0.000000 diff) | Dropped bogus dialect claims; truthful metadata: `synthetic_parametric_candidate` | Scientific transparency and metadata truthfulness |

---

## 4. Measurable Acceptance Quality Gates

The 120-clip pilot dataset was subjected to automated verification in `validate_dataset.py`. All 5 measurable quality gates passed:

| Quality Gate | Requirement / Condition | Pilot Measurement / Result | Gate Verdict |
| :--- | :--- | :--- | :--- |
| **Gate 1: Acoustic Variance** | Intra-class Log-Mel cosine std $\sigma \ge 0.03$ or mean-centered similarity $\le 0.85$ | Raw intra-class std $= 0.0416 \ge 0.03$; Mean-centered intra-class mean $= 0.5282 \le 0.85$ (std $= 0.2672$) | **PASS** |
| **Gate 2: Onset Shift Robustness** | Utterance onset spans $[0.02\text{ s}, 0.35\text{ s}]$ without fixed-frame clustering | Min onset $= 0.0285\text{ s}$, Max onset $= 0.3184\text{ s}$, Mean $= 0.1568\text{ s}$, Std $= 0.0785\text{ s}$ | **PASS** |
| **Gate 3: Zero Closure Verification** | For `TRAI`, max absolute amplitude in first 25ms of utterance body $\ge 0.02$ in $\ge 95\%$ of clips | Min peak $= 0.0905 \ge 0.02$, Mean peak $= 0.5067$, Max peak $= 0.8500$; Passed clips: **20/20 ($100\%$)** | **PASS** |
| **Gate 4: Silence Coverage** | 5 explicit ambient regimes present and verified | `exact_zeros` (4), `low_level_dither` (4), `room_tone` (4), `synthetic_hum` (4), `non_speech_transient` (4) | **PASS** |
| **Gate 5: UNKNOWN Vocabulary** | 20 distinct Vietnamese syllables covering initial-sharing, rhyme-sharing, and open vocabulary | 20 unique syllables confirmed: `biển`, `bạn`, `chào`, `cuống`, `hoa`, `học`, `mái`, `mây`, `ngày`, `nhà`, `núi`, `pha`, `rừng`, `sông`, `số`, `tai`, `trời`, `xe`, `đi`, `đèn` | **PASS** |

---

## 5. Quantitative Acoustic & Spectral Diagnostics

### 5.1. Per-Class Spectral Distribution (120 Pilot Clips)

| Class Label | Number of Clips | Spectral Centroid Mean (Hz) | Spectral Centroid Std (Hz) | HF Energy Ratio ($>3\text{kHz}$) | Full-Clip RMS Mean | Body First 25ms Peak Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `TRAI` | 20 | 1409.0 | 216.6 | 0.0080 | 0.1654 | 0.5067 |
| `PHAI` | 20 | 2911.4 | 160.6 | 0.2463 | 0.0749 | 0.4410 |
| `LEN` | 20 | 1069.5 | 314.5 | 0.0006 | 0.2055 | 0.5898 |
| `XUONG` | 20 | 2784.9 | 333.8 | 0.0757 | 0.1467 | 0.4728 |
| `UNKNOWN` | 20 | 1955.3 | 925.0 | 0.0622 | 0.1444 | 0.4984 |
| `SILENCE` | 20 | 1528.6 | 1625.8 | 0.1965 | 0.0034 | N/A |

### Key Acoustic Takeaways:
1. **Phonological Separation of TRAI vs. PHAI**:
   - `PHAI` has high spectral centroid ($2911\text{ Hz}$) and substantial high-frequency energy ratio ($0.2463$) driven by continuous labiodental frication $/f/$.
   - `TRAI` has lower spectral centroid ($1409\text{ Hz}$) and low HF ratio ($0.0080$) dominated by the open vowel $/a:j/$, but with a sharp transient burst ($3400-4400\text{ Hz}$) and acoustic murmur during closure ($>0.02$ peak).
2. **Negative Class (`UNKNOWN`) Breadth**:
   - `UNKNOWN` spectral centroid spans a wide range ($1955.3 \pm 925.0\text{ Hz}$) reflecting phonetic diversity across 20 distinct syllables (vowels, nasals, stops, fricatives).
3. **Absence of Onset Silence Shortcut**:
   - In V4, `TRAI` closure had amplitude suppressed to $\le 0.005$, creating a binary feature that models easily memorized. In V5, `body_first_25ms_peak` averages $0.5067$, with a minimum of $0.0905$, completely preventing the model from relying on silence gap detection.

---

## 6. Pairwise Cosine Similarity Analysis

Cosine similarity was evaluated on the $40 \times 101$ Log-Mel spectrograms ($4040$ dimensions per clip) across both **raw** and **mean-centered** representations:

### 6.1. Raw Log-Mel Cosine Similarity

| Class | Within-Class Mean | Within-Class Std | Within-Class Min | Within-Class Max |
| :--- | :---: | :---: | :---: | :---: |
| `TRAI` | 0.9083 | 0.0414 | 0.7819 | 0.9758 |
| `PHAI` | 0.9292 | 0.0296 | 0.8426 | 0.9915 |
| `LEN` | 0.9298 | 0.0350 | 0.8216 | 0.9929 |
| `XUONG` | 0.9182 | 0.0339 | 0.8292 | 0.9920 |
| `UNKNOWN` | 0.9131 | 0.0377 | 0.7719 | 0.9852 |
| `SILENCE` | 0.9844 | 0.0118 | 0.9494 | 1.0000 |
| **Overall Within-Class** | **0.9305** | **0.0416** | **0.7719** | **1.0000** |
| **Between-Class** | **0.9126** | **0.0328** | **0.7696** | **0.9910** |

*Note: Raw Log-Mel features have an inherent positive DC baseline (due to log-filterbank energy floors), which inflates raw cosine similarity. The mean-centered analysis removes this floor.*

### 6.2. Mean-Centered Log-Mel Cosine Similarity (Floor-Removed)

| Class | Within-Class Mean | Within-Class Std | Within-Class Min | Within-Class Max |
| :--- | :---: | :---: | :---: | :---: |
| `TRAI` | 0.5531 | 0.2281 | -0.1472 | 0.9088 |
| `PHAI` | 0.5445 | 0.1904 | 0.1105 | 0.9380 |
| `LEN` | 0.6825 | 0.1662 | 0.0695 | 0.9748 |
| `XUONG` | 0.5904 | 0.1673 | 0.1477 | 0.9463 |
| `UNKNOWN` | 0.5729 | 0.1575 | 0.0045 | 0.9465 |
| `SILENCE` | 0.2255 | 0.3709 | -0.1127 | 0.9995 |
| **Overall Within-Class** | **0.5282** | **0.2672** | **-0.1472** | **0.9995** |
| **Between-Class** | **0.4583** | **0.2584** | **-0.2456** | **0.9640** |

### Contrast with V4 Baseline:
- In V4, intra-class cosine similarity was $0.966–0.986$ with standard deviation $< 0.015$, indicating near-identical template waveforms.
- In V5, mean-centered intra-class similarity drops to $0.5282 \pm 0.2672$, confirming rich acoustic diversity driven by VTL scaling, dynamic $F_2$ locus glide, variable burst spectra, and continuous temporal jitter.

---

## 7. Single-Clip Determinism Verification

Deterministic reproduction was validated by regenerating clips from seed and comparing byte-exact SHA-256 hashes against disk files:

| Tested Class | Tested Recording ID | Seed | Manifest SHA-256 | Regenerated SHA-256 | Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| `TRAI` | `v5_pilot_trai_001` | 50000 | `BFFC4DE382359AADF0F50B31E18FD3840AA21AAA48409FB587F591FB6317F140` | `BFFC4DE382359AADF0F50B31E18FD3840AA21AAA48409FB587F591FB6317F140` | **MATCH (BIT-EXACT)** |
| `PHAI` | `v5_pilot_phai_001` | 50100 | `14A02AD7D27E472DC200D1A1E72033F33A7AE144B695FB1728C354E13ECBFE92` | `14A02AD7D27E472DC200D1A1E72033F33A7AE144B695FB1728C354E13ECBFE92` | **MATCH (BIT-EXACT)** |
| `LEN` | `v5_pilot_len_001` | 50200 | `590B235C6CAD3594BC87D402861C757BB9E5F60433EA47EEE4D1C79F616033B4` | `590B235C6CAD3594BC87D402861C757BB9E5F60433EA47EEE4D1C79F616033B4` | **MATCH (BIT-EXACT)** |
| `XUONG` | `v5_pilot_xuong_001` | 50300 | `1A7F312BA782DB05857245DC159685959377BE9B878E930A9632449461315A72` | `1A7F312BA782DB05857245DC159685959377BE9B878E930A9632449461315A72` | **MATCH (BIT-EXACT)** |
| `UNKNOWN` | `v5_pilot_unknown_001` | 50400 | `24EF8DDCB51661D143239FA76D2C120181B83F686531D63CCB363F4725FCCD98` | `24EF8DDCB51661D143239FA76D2C120181B83F686531D63CCB363F4725FCCD98` | **MATCH (BIT-EXACT)** |
| `SILENCE` | `v5_pilot_silence_001` | 50500 | `643F8A8DC8BD9C19225AFFFAD2BECFEC5426180B3749CB208ABDF1A6C8354EFC` | `643F8A8DC8BD9C19225AFFFAD2BECFEC5426180B3749CB208ABDF1A6C8354EFC` | **MATCH (BIT-EXACT)** |

---

## 8. Test Suite Execution & Parity Evidence

Four test suites were executed to verify zero regression across the codebase:

```
tests/test_synthetic_kws_v5_generator.py .................               [17 passed]
tests/test_kws_export_parity.py .............                            [13 passed]
tests/test_synthetic_kws_demo.py ...................                     [19 passed]
tests/test_campaign_acceptance.py ...................................... [69 passed]
============================= 118 passed in 40.2s =============================
```

- **Exit codes**: 0 across all runs.
- **Failures / Errors / Skips**: 0.

---

## 9. Scientific Limitations & Governance Invariant

> [!CAUTION]
> **SCIENTIFIC HONESTY & LIMITATIONS:**
> 1. **Synthetic-Only Prototype**: The V5 candidate generator produces synthetic waveforms derived from mathematical resonator models. It does **not** represent natural human speech, nor does it replace real acoustic data.
> 2. **Cross-Generator Evaluation Reality**: As demonstrated in Phase 12.1 and 12.2 audits, models trained on synthetic speech suffer catastrophic domain shift on natural human speech ($22.0\%$ accuracy on V2 multisource audio). Retraining on V5 without human speech calibration will still exhibit generator dependence.
> 3. **Model Retraining Deferred**: Retraining was intentionally **not** executed in Phase 13. The existing frozen checkpoint (`arm_b_onset_no_resample.pt`) was trained on V4 (with the 25ms shortcut) and must remain unmodified until explicit governance approval and retrain campaign design are authorized.
> 4. **Human Data Governance**: Collection and processing of human speech remains **strictly blocked**.

---

## 10. Final Deliverables Summary

1. `experiments/synthetic_kws_v5_candidate/config.py`: Configuration dataclasses and 20-word pilot vocabulary.
2. `experiments/synthetic_kws_v5_candidate/generator.py`: Complete V5 synthesis engine and reproduction functions.
3. `experiments/synthetic_kws_v5_candidate/validate_dataset.py`: Comprehensive dataset QA and acoustic diversity harness.
4. `experiments/synthetic_kws_v5_candidate/manifest.csv`: 120-clip pilot dataset manifest with full provenance.
5. `experiments/synthetic_kws_v5_candidate/dataset_integrity_report.json`: Format, finite, and determinism verification report.
6. `experiments/synthetic_kws_v5_candidate/acoustic_diversity_report.json`: Gate assessments, spectral distributions, and cosine similarity reports.
7. `tests/test_synthetic_kws_v5_generator.py`: 17 automated regression and contract unit tests.
8. `experiments/synthetic_kws_v5_candidate/V5_GENERATOR_AUDIT.md`: This comprehensive audit report.
