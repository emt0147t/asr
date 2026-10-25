# PHASE 22 — POST-EVALUATION STATISTICAL AND CLAIM AUDIT

**Date:** 2026-10-24  
**Workspace:** `D:\trainidentificationaudio`  
**Previous Phase:** Phase 21 — Synthetic Audio Provenance, Diversity and New Evaluation Corpus  
**Previous Gate:** `PASS_TO_NEW_LOCKED_SYNTHETIC_EVALUATION`  
**Phase Type:** Independent Offline Statistical Audit and Claim Verification  
**Final Decision:** `PASS_TO_TRAINING_DESIGN_REVIEW`  

---

## 0. MANDATORY GOVERNANCE DECLARATIONS & INVARIANTS

All investigations, statistical calculations, and claim assessments strictly maintain the mandatory governance declarations:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

### Strict Governance Constraints Followed
1. **Zero Retraining or Tuning:** Checkpoints `variant_v11...pt` and `exp_v12_e1...pt`, as well as ONNX artifacts `vietnamese_kws_v11.onnx` and `vietnamese_kws_v12_e1.onnx`, remained byte-for-byte unmodified. Zero fine-tuning, training, or E2 execution was initiated.
2. **Fixed Decision Threshold:** The canonical decision threshold $\theta_{\text{cmd}} = 0.20$ remained completely invariant. No post-hoc threshold adjustment or command-specific tuning was permitted.
3. **No OOV Blacklisting or VAD Alteration:** `SpeechBandRatioVAD` parameters remained identical to `selected_phase15_vad_config.json`. No word blacklisting or negative filtering heuristics were introduced.
4. **Observed Test Set Status:** `synthetic_eval_test_v1` is formally treated as an **observed test set** from this point forward; it cannot be reused as a blind test set in any subsequent iteration.
5. **Zero Human Speech Collection:** No physical microphones, human speech recordings, or crowdsourced voices were collected.

---

## 1. EXECUTIVE SUMMARY & AUDIT SCOPE

Phase 22 conducted an independent, multi-dimensional statistical audit of the Phase 21 evaluation results:
1. **Reconciliation of Performance Metrics:** Independently verified all counts (matched, missed, substitutions, false alarms) and recomputed exact 5x5 confusion matrices, precision, recall, and F1 scores.
2. **Clarification of F1 Formulation:** Resolved the distinction between Global Aggregate / Micro-F1 and True Macro-F1. In Phase 21, the metric labeled "Command Macro-F1: 89.16%" was mathematically the **Micro-averaged / Global Aggregate F1**. The **True Macro-F1** (unweighted arithmetic mean of per-command F1) is **$0.8788$ ($87.88\%$)** on the test corpus and **$0.8354$ ($83.54\%$)** on the dev corpus.
3. **OOV Denominator & False Alarm Rate Audit:** Disentangled the component clip pool (96 OOV clips) from the actual sequence insertions (16 OOV events). Audited all 8 false activations and discovered that **100% of false alarms were triggered exclusively by phonetic confusers spoken by a single high-pitch female voice (`Ái Hân`)**, while everyday general OOV words achieved a **100.0% rejection rate**.
4. **Uncertainty Quantification:** Established 95% Wilson Score confidence intervals for exact-sequence accuracy and performed cluster-robust bootstrap resampling (1,000 resamples clustered at sequence level).
5. **Technical Assessment of TTS Diversity:** Audited the relationship between `VieNeu-TTS v3 Turbo` and `VieNeu-TTS v3 Nano`, correcting overclaims of "cross-architecture generalization" to **"controlled acoustic-engine transfer across VieNeu variants"**.

---

## 2. EXACT COMMAND METRICS RECOMPUTATION & CONFUSION MATRIX RECONCILIATION

### 2.1 Development Corpus (`synthetic_eval_dev_v1`, 20 Sequences)

- **Exact-Sequence Accuracy:** 8 / 20 = **40.00%**
- **Target Events Total:** 64
- **Matched Commands:** 48
- **Missed Commands:** 16
- **Substitutions:** 0
- **OOV False Activations:** 0 (100% rejection across 8 inserted OOV events)
- **Silence False Activations:** 0
- **Duplicate Triggers:** 1

#### 5x5 Confusion Matrix (Dev Corpus)
*Rows = Ground Truth Reference, Columns = Detected Command (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT/Miss`)*

| Reference Label | Pred `TRAI` | Pred `PHAI` | Pred `LEN` | Pred `XUONG` | REJECT / Miss | Total Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | **8** | 0 | 0 | 0 | 8 | 16 |
| **`PHAI`** | 0 | **8** | 0 | 0 | 4 | 12 |
| **`LEN`** | 0 | 0 | **18** | 0 | 0 | 18 |
| **`XUONG`** | 0 | 0 | 0 | **14** | 4 | 18 |
| **`NEGATIVE` (OOV/Silence)** | 0 | 0 | 0 | 0 | - | - |

#### Per-Class Metrics (Dev Corpus)
| Command Class | TP | FP | FN | Support | Precision | Recall | $F_1$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | 8 | 0 | 8 | 16 | 100.00% | 50.00% | 0.6667 |
| **`PHAI`** | 8 | 0 | 4 | 12 | 100.00% | 66.67% | 0.8000 |
| **`LEN`** | 18 | 0 | 0 | 18 | 100.00% | 100.00% | 1.0000 |
| **`XUONG`** | 14 | 0 | 4 | 18 | 100.00% | 77.78% | 0.8750 |
| **True Macro-F1 (Unweighted Mean)** | - | - | - | - | - | - | **0.8354** |
| **Support-Weighted Macro-F1** | - | - | - | - | - | - | **0.8440** |
| **Global Aggregate / Micro-F1** | 48 | 0 | 16 | 64 | **100.00%** | **75.00%** | **0.8571** |

---

### 2.2 Locked Test Corpus (`synthetic_eval_test_v1`, 40 Sequences)

- **Exact-Sequence Accuracy:** 22 / 40 = **55.00%**
- **Target Events Total:** 128
- **Matched Commands:** 111
- **Missed Commands:** 15
- **Substitutions:** 2 (`TRAI` $\rightarrow$ `PHAI`: 1, `PHAI` $\rightarrow$ `TRAI`: 1)
- **OOV False Activations:** 8 (all 8 from phonetic confusers)
- **Silence False Activations:** 0
- **Duplicate Triggers:** 0

#### 5x5 Confusion Matrix (Test Corpus)
*Rows = Ground Truth Reference, Columns = Detected Command (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT/Miss`)*

| Reference Label | Pred `TRAI` | Pred `PHAI` | Pred `LEN` | Pred `XUONG` | REJECT / Miss | Total Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | **21** | 1 | 0 | 0 | 10 | 32 |
| **`PHAI`** | 1 | **20** | 0 | 0 | 3 | 24 |
| **`LEN`** | 0 | 0 | **35** | 0 | 1 | 36 |
| **`XUONG`** | 0 | 0 | 0 | **35** | 1 | 36 |
| **`NEGATIVE` (OOV/Silence)** | 1 | 3 | 3 | 1 | - | - |

#### Per-Class Metrics (Test Corpus)
| Command Class | TP | FP | FN | Support | Precision | Recall | $F_1$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | 21 | 2 (1 sub, 1 OOV) | 11 (1 sub, 10 miss) | 32 | 91.30% | 65.63% | 0.7636 |
| **`PHAI`** | 20 | 4 (1 sub, 3 OOV) | 4 (1 sub, 3 miss) | 24 | 83.33% | 83.33% | 0.8333 |
| **`LEN`** | 35 | 3 (0 sub, 3 OOV) | 1 (0 sub, 1 miss) | 36 | 92.11% | 97.22% | 0.9459 |
| **`XUONG`** | 35 | 1 (0 sub, 1 OOV) | 1 (0 sub, 1 miss) | 36 | 97.22% | 97.22% | 0.9722 |
| **True Macro-F1 (Unweighted Mean)** | - | - | - | - | - | - | **0.8788** |
| **Support-Weighted Macro-F1** | - | - | - | - | - | - | **0.8866** |
| **Global Aggregate / Micro-F1** | 111 | 10 | 17 | 128 | **91.74%** | **86.72%** | **0.8916** |

### 2.3 Mathematical Audit of Macro-F1 vs Global Aggregate F1

In Phase 21 Section 5.1, the summary table reported:
$$\text{Precision} = \frac{111}{121} = 0.9174, \quad \text{Recall} = \frac{111}{128} = 0.8672, \quad \text{Command F1} = \frac{2 \times 0.9174 \times 0.8672}{0.9174 + 0.8672} = 0.8916$$
This formula corresponds strictly to **Global Aggregate / Micro-F1**, where all positive detections and errors are pooled globally before computing the harmonic mean.

The **True Macro-F1**, defined as the unweighted arithmetic mean of per-class F1 scores:
$$\text{Macro-F1} = \frac{1}{4} \left( F1_{\text{TRAI}} + F1_{\text{PHAI}} + F1_{\text{LEN}} + F1_{\text{XUONG}} \right) = \frac{0.7636 + 0.8333 + 0.9459 + 0.9722}{4} = \mathbf{0.8788}$$
- **Discrepancy:** $0.8916 - 0.8788 = +0.0128$ ($+1.28\%$).
- **Cause:** Global Aggregate F1 assigns higher weight to classes with larger support and higher performance (`LEN` and `XUONG`, each with 35 TP out of 36), while True Macro-F1 gives equal 25% weight to `TRAI` ($F_1 = 0.7636$).
- **Audit Verdict:** Both numbers are statistically valid under their precise definitions. All future reports must explicitly distinguish between **True Macro-F1 (87.88%)** and **Micro/Aggregate F1 (89.16%)**.

---

## 3. OOV OCCURRENCE, REJECTION RATES & DENOMINATOR AUDIT

### 3.1 Clarification of OOV Counts (Pool vs Inserted Events)
A potential source of confusion in streaming benchmarks is the distinction between available component clips and actively inserted streaming events:
- **Component Pool (`synthetic_eval_components_manifest.csv`):**
  - Dev: 40 clips (20 words $\times$ 2 voices)
  - Test: 96 clips (13 phonetic confusers $\times$ 4 voices = 52 clips; 11 general OOV $\times$ 4 voices = 44 clips)
- **Actual Sequence Insertions:**
  - Dev: exactly **8 OOV events** inserted across 20 sequences.
  - Test: exactly **16 OOV events** inserted across 40 sequences:
    - **13 Phonetic Confusers**
    - **3 General Everyday OOV words**

### 3.2 OOV Rejection Rates
- **General Everyday OOV Rejection Rate:** $3 / 3 = \mathbf{100.00\%}$ (zero false alarms across all general speech).
- **Phonetic Confusers Rejection Rate:** $5 / 13 = \mathbf{38.46\%}$ (8 false activations out of 13 inserted confusers).
- **Overall OOV Event Rejection Rate:** $(16 - 8) / 16 = \mathbf{50.00\%}$.

### 3.3 Complete Trace of All 8 False Activations
Every false activation in the test corpus was traced back to its audio origin:

| Sequence ID | Condition & Gap | Spoken Word | OOV Category | Speaker Voice | Falsely Emitted Command | Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `synthetic_test_seq_0008` | `snr15`, 150 ms | `len` | Phonetic Confuser | `Ái Hân` | `LEN` | 0.9954 |
| `synthetic_test_seq_0009` | `snr10`, 200 ms | `lân` | Phonetic Confuser | `Ái Hân` | `LEN` | 0.6937 |
| `synthetic_test_seq_0009` | `snr10`, 200 ms | `lền` | Phonetic Confuser | `Ái Hân` | `LEN` | 0.9944 |
| `synthetic_test_seq_0017` | `snr20`, 300 ms | `phái` | Phonetic Confuser | `Ái Hân` | `PHAI` | 0.9988 |
| `synthetic_test_seq_0018` | `snr15`, 400 ms | `phay` | Phonetic Confuser | `Ái Hân` | `PHAI` | 0.9608 |
| `synthetic_test_seq_0019` | `snr10`, 100 ms | `phai` | Phonetic Confuser | `Ái Hân` | `PHAI` | 0.9722 |
| `synthetic_test_seq_0019` | `snr10`, 100 ms | `trại` | Phonetic Confuser | `Ái Hân` | `TRAI` | 0.8444 |
| `synthetic_test_seq_0029` | `snr10`, 300 ms | `xuồng` | Phonetic Confuser | `Ái Hân` | `XUONG` | 0.9827 |

#### Key Diagnostic Finding
1. **Speaker Clustering:** **8 out of 8 (100.0%) false activations occurred on speaker `Ái Hân`** (a female voice characterized by higher fundamental frequency $F_0$).
2. **Zero False Activations on Other Voices:** Voices `Hữu Quân`, `Xuân Tiên`, and `Mạnh Dũng` experienced **0 false activations**, successfully rejecting all phonetic confusers and general OOV words.
3. **Lexical Clustering:** All 8 false activations involved monosyllabic words sharing initial consonants and vowel formants with target keywords, differing primarily by lexical tone or subtle vowel shifts (`phái`/`phay`/`phai` $\rightarrow$ `PHAI`, `len`/`lân`/`lền` $\rightarrow$ `LEN`, `xuồng` $\rightarrow$ `XUONG`, `trại` $\rightarrow$ `TRAI`).

### 3.4 Duration-Normalized False Alarm Rates (Test Corpus)
- **Total Audio Duration:** 114.75 seconds (1.91 minutes)
- **Target Speech Duration:** 64.90 seconds (1.08 minutes)
- **Pure OOV Speech Duration:** 6.15 seconds (0.1025 minutes)
- **Pure Silence Duration:** 43.70 seconds (0.7283 minutes)
- **Total Negative Duration (OOV + Silence):** 49.85 seconds (0.8308 minutes)

| Negative Audio Substrate | Duration (Minutes) | False Activations | False Activation Rate |
| :--- | :---: | :---: | :---: |
| **Total Negative Audio (OOV + Silence)** | 0.8308 min | 8 | **9.63 FA / min** |
| **Pure OOV Speech** | 0.1025 min | 8 | **78.05 FA / min** |
| **Pure Silence & Non-Speech Noise** | 0.7283 min | 0 | **0.00 FA / min** |

---

## 4. STATISTICAL UNCERTAINTY QUANTIFICATION

Because test evaluations operate on finite sequence samples ($n=20$ dev, $n=40$ test), point estimates alone are insufficient for engineering decisions.

### 4.1 Exact-Sequence Accuracy 95% Wilson Score Intervals
The Wilson score interval provides robust coverage for binomial proportions without normal approximation distortion:
- **Dev Corpus ($n=20$, $k=8$):**
  $$\hat{p} = 40.00\% \quad \Longrightarrow \quad \mathbf{95\% \text{ CI: } [21.88\%, 61.34\%]}$$
- **Test Corpus ($n=40$, $k=22$):**
  $$\hat{p} = 55.00\% \quad \Longrightarrow \quad \mathbf{95\% \text{ CI: } [39.83\%, 69.29\%]}$$

### 4.2 Cluster-Robust Bootstrap (1,000 Resamples at Sequence Level)
Because commands within the same sequence share acoustic conditions, background noise tracks, and temporal context, observations are clustered at the sequence level. We executed 1,000 cluster-bootstrap iterations with fixed seed `20261024`:

#### Development Corpus Bootstrap Distribution ($n=20$ clusters)
| Metric | Bootstrap Mean | Bootstrap Std | 95% Bootstrap Empirical CI |
| :--- | :---: | :---: | :---: |
| **Exact-Sequence Accuracy** | 40.06% | 11.00% | **[20.00%, 60.12%]** |
| **Command Precision** | 97.91% | 2.13% | **[92.86%, 100.00%]** |
| **Command Recall** | 74.82% | 5.35% | **[64.06%, 85.08%]** |
| **Global Aggregate F1** | 84.72% | 3.73% | **[76.92%, 91.67%]** |
| **True Macro-F1** | 81.98% | 4.23% | **[72.63%, 89.77%]** |
| **Total False Activations** | 0.00 | 0.00 | **[0.0, 0.0]** |

#### Test Corpus Bootstrap Distribution ($n=40$ clusters)
| Metric | Bootstrap Mean | Bootstrap Std | 95% Bootstrap Empirical CI |
| :--- | :---: | :---: | :---: |
| **Exact-Sequence Accuracy** | 55.38% | 7.80% | **[40.00%, 70.00%]** |
| **Command Precision** | 93.36% | 2.60% | **[88.09%, 98.10%]** |
| **Command Recall** | 88.36% | 3.10% | **[81.48%, 93.98%]** |
| **Global Aggregate F1** | 90.75% | 2.00% | **[86.67%, 94.45%]** |
| **True Macro-F1** | 89.74% | 2.29% | **[85.12%, 93.89%]** |
| **Total False Activations** | 8.04 | 3.20 | **[2.0, 15.0]** |

---

## 5. LATENCY PROFILE DISTRIBUTION

Latency measures the elapsed duration from reference keyword start time to VAD segment emission time:
- **Dev Corpus Latency ($n=48$ matched events):**
  - Median ($P_{50}$): **451.30 ms**
  - $P_{95}$: **682.46 ms**
- **Test Corpus Latency ($n=111$ matched events):**
  - Median ($P_{50}$): **320.00 ms**
  - $P_{95}$: **825.00 ms**
- **Audit Verification:** $P_{95}$ latency on the locked test corpus (825.0 ms) successfully satisfies the pre-registered latency ceiling requirement ($\le 850.0\text{ ms}$).

---

## 6. CROSS-TTS GENERALIZATION TECHNICAL ASSESSMENT & CLAIM AUDIT

Phase 21 demonstrated successful evaluation across generators. However, scientific rigor requires auditing the exact relationship between the training synthesizer and the evaluation synthesizer.

### 6.1 Architectural Comparison: VieNeu-TTS v3 Turbo vs VieNeu-TTS v3 Nano

| Technical Dimension | Training Generator: `v3turbo` | Evaluation Generator: `v3nano` | Shared vs Disjoint |
| :--- | :--- | :--- | :--- |
| **Codebase & Origin** | `pnnbao-ump/VieNeu-TTS` | `pnnbao-ump/VieNeu-TTS` | **Shared genealogy** |
| **Text Normalization** | Rule-based Vietnamese normalizer | Rule-based Vietnamese normalizer | **Identical** |
| **Phonemizer** | Upstream Vietnamese G2P | Upstream Vietnamese G2P | **Identical** |
| **Acoustic Estimator** | PyTorch Autoregressive Flow / Transformer | Lightweight ONNX vector regressor | **Disjoint Architecture** |
| **Vocoder Architecture** | Continuous neural vocoder (HiFi-GAN style) | Discrete neural codec decoder | **Disjoint Architecture** |
| **Native Sampling Rate** | $48,000\text{ Hz}$ | $24,000\text{ Hz}$ | **Disjoint (Both resampled to 16 kHz)** |
| **Speaker Embeddings** | 25 fixed preset voices | 4 held-out voice presets | **Strictly Disjoint Partition** |

### 6.2 Overclaim Correction & Scoping
- **Original Claim in Phase 21:** *"Established strong cross-architecture generalization."*
- **Audit Assessment:** Calling this "cross-architecture generalization" without qualification is an **overclaim**. While the acoustic regressor, vocoder family, sampling rate, and voice embeddings are architecturally distinct, both models share the exact same linguistic frontend, tone mapping, and text normalization routines from the `VieNeu-TTS` project.
- **Redacted & Precise Claim:**
  > *"The evaluation demonstrates **controlled acoustic-engine transfer across VieNeu-TTS model variants (from v3 Turbo to v3 Nano) across unobserved speaker embeddings and distinct vocoder decoders**. It does NOT establish generalization to independent external TTS families (e.g., Piper, VALL-E, FastSpeech2), nor does it validate robustness to natural human speech."*

---

## 7. CHRONOLOGY, PROVENANCE & CRYPTOGRAPHIC REGISTRY

### 7.1 Chronological Pipeline Execution Order
Audit of file modification timestamps confirmed strict sequential integrity:
1. `2026-10-24 19:53:18` — `audit_synthetic_provenance_phase21.py` executed.
2. `2026-10-24 19:53:56` — `PHASE21_SYNTHETIC_EVALUATION_PREREGISTRATION.md` created & committed.
3. `2026-10-24 19:58:05` — `build_phase21_synthetic_corpora.py` generated audio & manifests.
4. `2026-10-24 19:58:53` — `validate_phase21_corpora.py` confirmed 0 split leakage.
5. `2026-10-24 20:00:54` — `run_phase21_evaluation.py` evaluated frozen E1 ONNX.
6. `2026-10-24 20:04:15` — `PHASE21_SYNTHETIC_DATA_AND_EVALUATION_REPORT.md` written.
7. `2026-10-24 21:29:40` — `audit_phase21_statistics.py` generated `phase22_statistical_audit.json`.

### 7.2 Cryptographic Verification of All 11 Immutable Artifacts
All hashes verified bit-for-bit:

| Artifact Name | Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| `v11_checkpoint` | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **VERIFIED** |
| `v11_onnx` | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **VERIFIED** |
| `e1_checkpoint` | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **VERIFIED** |
| `e1_onnx` | `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` | `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8` | **VERIFIED** |
| `selected_vad_config` | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | **VERIFIED** |
| `dataset_v12_manifest` | `experiments/vieneu_kws_research_v12/manifests/dataset_v12_manifest.csv` | `bca9cb604d6b21668a70d4690297bffac25be935bd3843de3c3d3435dfb77806` | **VERIFIED** |
| `synthetic_dev_manifest` | `experiments/vieneu_kws_research_v12/manifests/synthetic_eval_dev_manifest.csv` | `57abdfcf3ba126f1372ec280af4b0e1e00dd81ef9f8da9ac6decb3ba8bc3cae3` | **VERIFIED** |
| `synthetic_test_manifest` | `experiments/vieneu_kws_research_v12/manifests/synthetic_eval_test_manifest.csv` | `1093629730c2b1d1d022d2a6623de1c0c3302fc1e271d16691a55b68a1087d32` | **VERIFIED** |
| `synthetic_components_manifest` | `experiments/vieneu_kws_research_v12/manifests/synthetic_eval_components_manifest.csv` | `b590a7eab92e21c584b59a90dda688c40a00dd32ad1ab3fdee313db417294d04` | **VERIFIED** |
| `phase21_preregistration_md` | `experiments/vieneu_kws_research_v12/reports/PHASE21_SYNTHETIC_EVALUATION_PREREGISTRATION.md` | `2ba0408f96c00b849484598b358dea55835a02ad59ac50996584c4c43286631c` | **VERIFIED** |
| `phase21_evaluation_json` | `experiments/vieneu_kws_research_v12/reports/phase21_synthetic_evaluation_results.json` | `c035134ec0cc4ad080cb41bc26251e223173b1265288972d803178c6d6d22a7e` | **VERIFIED** |

---

## 8. FINAL DECISION & SIGN-OFF

The statistical audit has independently reconciled all evaluation metrics, clarified the distinction between Micro/Aggregate F1 (89.16%) and True Macro-F1 (87.88%), verified OOV denominators, identified the specific female speaker clustering of phonetic false activations (`Ái Hân`), quantified statistical uncertainty with 95% Wilson and Bootstrap CIs, and corrected cross-generator claims.

With the synthetic evaluation profile of E1 fully documented and `synthetic_eval_test_v1` marked as an observed benchmark, the system is eligible to proceed to a formal training design review for Experiment 2 (E2).

### Final Gate Status
**`PASS_TO_TRAINING_DESIGN_REVIEW`**

*Notice: This gate authorizes the planning and design review of training methodologies (addressing phonetic confuser discrimination on female voices like Ái Hân and noise robustness). It does NOT authorize un-gated model training, physical microphone testing, or natural human speech deployment.*
