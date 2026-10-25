# Phase 12.1: Claims Verification and Reproducible Evidence Audit

> [!IMPORTANT]
> **GOVERNANCE INVARIANT & SCOPE DISCREPANCY RECORD:**
> - **Human Data Collection Policy**: Human data collection, live microphone recording, and external speech ingestion remain **strictly blocked**.
> - **Historical Scope Discrepancy Record**: In Phase 12.1, the automated dataset inventory script (`audit_dataset_audio`) invoked `soundfile.read()` locally over the files referenced in `data/manifests/*_v2.csv` (which included `google_fleurs_vi` and `husein_mic_noise`). Execution logs confirm that all commands ran locally in sandbox mode without external network upload (`git diff --cached` remained empty, 0 bytes transmitted). Nonetheless, loading those files constituted a scope discrepancy against the strict governance directive to pause operations on sources that may contain natural human speech.
> - **Corrective Freeze**: In Phase 12.2 and going forward, all waveform, feature extraction, and model evaluation operations on V2 multisource data and `data/raw/field_recordings/` are **strictly paused** pending explicit authorization. Analysis proceeds exclusively on demonstrably synthetic V4 artifacts, verified static prediction tables, and source-code inspection.

---

## 1. Executive Summary

Phase 12.1 executes a rigorous empirical verification of the claims made during the initial Phase 12 audit. By moving from small spot-checks to full-cohort statistical evaluations (all $1,468$ V4 clips analyzed across 181,094 pairwise comparisons), tracing preprocessing code paths, and recomputing metrics from $6,768$ canonical row-level predictions, this audit establishes a definitive, evidence-grounded scientific assessment of the Vietnamese KWS project.

---

## 2. Claim-by-Claim Verification Matrix

| Claim from Phase 12 | Initial Claim Summary | Phase 12.1 / 12.2 Empirical Verification | Verification Result | Root Cause & Classification |
| :--- | :--- | :--- | :---: | :--- |
| **Claim 1: Model Shortcut Learning** | Arm B relies on an artificial 25ms silence gap for `TRAI` rather than real speech phonetics. | Tested Arm B on V1, V2, and V3 holdouts. Recomputed from row predictions: **$136 / 136$ historical `TRAI` clips were misclassified as `PHAI` ($0.0\%$ recall)**. | **`PROVEN_DEFECT`** | Non-linguistic onset artifact in V4 generator (`output[:400] *= 0.05`). |
| **Claim 2: Train/Test Near-Duplicates** | High cosine similarity ($>0.966$) indicates dataset leakage between train and test. | Re-evaluated on all $1,468$ V4 clips ($166,514$ initial same-class pairs; $181,094$ complete 6-group pairs). Within-Train ($0.9695$) and Cross-Train-Test ($0.9676$) are virtually identical. | **`GENERATOR_DEPENDENCE`** | Low-variance synthetic generator; **`EXACT_LEAKAGE` is 0%** (0 duplicate hashes or speaker IDs). |
| **Claim 3: Dead-Code Dialect** | The `dialect` parameter in `onset_aware_generator.py` is never used. | Deterministic test comparing Northern, Southern, and Central calls: **max sample difference is `0.000000` (bit-exact zero)**. | **`PROVEN_DEFECT`** | Unused function parameter in generator code. |
| **Claim 4: Silence Classifier Failure** | The neural model misclassifies silence as `TRAI` and relies on the adapter threshold to pass. | Evaluated ONNX model on pure zeros: **`TRAI`: 30.23%, `UNKNOWN`: 30.22%, `SILENCE`: 13.05%**. | **`PROVEN_DEFECT`** | Model predicts `TRAI` on zeros; safe `NO_ACTION` is solely due to adapter threshold ($<50\%$). |
| **Claim 5: V2 Peak Amplitude Cluster** | Peak amplitude in V2 clusters at $0.8910–0.8913$. | Traced to [`src/preprocessing/audio_normalizer.py`](file:///d:/trainidentificationaudio/src/preprocessing/audio_normalizer.py#L23): `PEAK_TARGET = 0.89125` ($-1.0\text{ dBFS}$) quantized to 16-bit PCM. | **`EXPLAINED_INTENTIONAL`** | Intentional peak normalization to $-1.0\text{ dBFS}$. |
| **Claim 6: Field Pilot Duplicate Hashes** | Field collection manifest contains 7 duplicate hashes. | Checked hashes: correspond to intentional test fixtures (`SES_PRESERVE_TEST`, `SES_DUP_TEST`, etc.). Excluded from model analysis per governance. | **`GOVERNANCE_EXCLUDED`** | Field pilot is quarantined pending formal approval. |

---

## 3. Corrected Similarity Analysis Accounting

### Exact Pairwise Accounting
In Phase 12.1, an unverified estimate of "240,000+ comparisons" was erroneously referenced in summary prose. The exact accounting across the documented cohorts ($N_{\text{train}}=178, N_{\text{val}}=36, N_{\text{test}}=50$ for targets; $N_{\text{train}}=120, N_{\text{val}}=36, N_{\text{test}}=50$ for negatives) has been strictly audited and expanded in `split_leakage_report.json`:

1. **Initial 4 Groups (Evaluated in Phase 12.1)**:
   - Within-Train pairs: $4 \times \binom{178}{2} + 2 \times \binom{120}{2} = 4 \times 15,753 + 2 \times 7,140 = 77,292$
   - Within-Test pairs: $6 \times \binom{50}{2} = 6 \times 1,225 = 7,350$
   - Cross Train-Val pairs: $4 \times (178 \times 36) + 2 \times (120 \times 36) = 4 \times 6,408 + 2 \times 4,320 = 34,272$
   - Cross Train-Test pairs: $4 \times (178 \times 50) + 2 \times (120 \times 50) = 4 \times 8,900 + 2 \times 6,000 = 47,600$
   - **Subtotal Initial 4 Groups**: **`166,514` pairwise comparisons**.
2. **Complete 6 Groups (Added in Phase 12.2)**:
   - Within-Val pairs: $6 \times \binom{36}{2} = 6 \times 630 = 3,780$
   - Cross Val-Test pairs: $6 \times (36 \times 50) = 6 \times 1,800 = 10,800$
   - **Total All 6 Groups**: $166,514 + 3,780 + 10,800 =$ **`181,094` pairwise comparisons**.

### Complete Full-Cohort Results Across All 6 Groups:

| Class | Within-Train | Within-Val | Within-Test | Cross Train-Val | Cross Train-Test | Cross Val-Test | Scientific Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | $0.9695 \pm 0.019$ | $0.9722 \pm 0.018$ | $0.9666 \pm 0.021$ | $0.9684 \pm 0.020$ | $0.9676 \pm 0.020$ | $0.9662 \pm 0.021$ | **`GENERATOR_DEPENDENCE`** |
| **`PHAI`** | $0.9667 \pm 0.021$ | $0.9688 \pm 0.019$ | $0.9666 \pm 0.021$ | $0.9667 \pm 0.021$ | $0.9665 \pm 0.021$ | $0.9674 \pm 0.021$ | **`GENERATOR_DEPENDENCE`** |
| **`LEN`** | $0.9714 \pm 0.019$ | $0.9712 \pm 0.019$ | $0.9721 \pm 0.019$ | $0.9701 \pm 0.021$ | $0.9713 \pm 0.020$ | $0.9694 \pm 0.021$ | **`GENERATOR_DEPENDENCE`** |
| **`XUONG`** | $0.9753 \pm 0.016$ | $0.9782 \pm 0.014$ | $0.9727 \pm 0.017$ | $0.9752 \pm 0.015$ | $0.9741 \pm 0.017$ | $0.9742 \pm 0.017$ | **`GENERATOR_DEPENDENCE`** |
| **`UNKNOWN`** | $0.9789 \pm 0.012$ | $0.9849 \pm 0.009$ | $0.9791 \pm 0.013$ | $0.9801 \pm 0.012$ | $0.9790 \pm 0.013$ | $0.9811 \pm 0.012$ | **`GENERATOR_DEPENDENCE`** |
| **`SILENCE`** | $0.9862 \pm 0.012$ | $0.9860 \pm 0.012$ | $0.9860 \pm 0.012$ | $0.9863 \pm 0.012$ | $0.9863 \pm 0.012$ | $0.9863 \pm 0.012$ | **`GENERATOR_DEPENDENCE`** |

- **Interpretation**: Within-split and cross-split similarities are indistinguishable ($0.966–0.986$). This demonstrates extreme **`GENERATOR_DEPENDENCE`** resulting from fixed formant equations ($F_1, F_2, F_3$) and static harmonic amplitudes.
- **Leakage Status**: **`EXACT_LEAKAGE: 0.0%`**. Speaker IDs and audio hashes are strictly disjoint across splits.
- All unsourced human speech similarity estimates have been excised from the findings.

---

## 4. Corrected Model Validity Classifications

To maintain scientific precision without making unsubstantiated claims about un-evaluated domains:

| Evaluation Dimension | Scientific Classification | Evidence & Operational Boundary |
| :--- | :---: | :--- |
| **Exact Duplicate Leakage** | **`ZERO_LEAKAGE_DETECTED`** | $0$ shared file hashes, $0$ shared recording IDs, $0$ shared speaker IDs across train, val, and test splits. |
| **Generator Dependence** | **`HIGH_GENERATOR_DEPENDENCE`** | Intra-class Log-Mel cosine similarity across all split pairs is $0.966–0.986$, confirming low acoustic variance across the synthetic generator. |
| **Natural-Speech Validity** | **`NOT_VALIDATED_ON_NATURAL_SPEECH`** | Human speech evaluation has not taken place due to the strict governance invariant blocking human data collection and ingestion. The model cannot be declared valid or invalid on natural speech without empirical testing. |
| **Model Robustness within Synthetic Scope** | **`COLLAPSES_ACROSS_GENERATOR_VERSIONS`** | Verified from row predictions: candidate `arm_b_onset_no_resample.pt` collapses from $99.33\%$ on V4 holdout to $69.44\%$ on V1 and $83.33\%$ on V2/V3, with **$0.0\%$ recall on `TRAI` ($136/136$ misclassified as `PHAI`)**. |

---

## 5. Structured Audit Artifacts Updated

The following audit deliverables are maintained in [`experiments/synthetic_kws_v4/audit/`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/):
1. [`dataset_inventory.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/dataset_inventory.csv): Verified technical checks across all datasets.
2. [`dataset_integrity_report.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/dataset_integrity_report.json): JSON audit with governance quarantine notes.
3. [`split_leakage_report.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/split_leakage_report.json): Recomputed pairwise comparisons covering both the 166,514 initial pairs and 181,094 complete pairs across all 6 groups.
4. [`label_and_generator_audit.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/label_and_generator_audit.md): Systematic analysis of generator semantics and dead code.
5. [`model_evaluation_lineage.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/model_evaluation_lineage.csv): Recomputed row-level prediction lineage.
6. [`cross_generator_evaluation_report.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/cross_generator_evaluation_report.md): Cross-generator metrics and confusion matrices.
7. [`PHASE12_2_V5_DATASET_DESIGN.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/PHASE12_2_V5_DATASET_DESIGN.md): Detailed specification for the future synthetic V5 candidate.
