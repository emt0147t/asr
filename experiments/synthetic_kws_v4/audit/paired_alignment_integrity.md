# Synthetic Vietnamese KWS V4: Paired Alignment & Checkpoint Verification Report

**Date:** 2026-10-15  
**Workspace:** `D:\trainidentificationaudio`  
**Audit Directory:** `experiments\synthetic_kws_v4\audit`  
**Governance Invariant:** `HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED`  
**Audit Verdict:** `AUDIT_PASS_WITH_LIMITATIONS` *(Provisional `AUDIT_BLOCKED` resolved following paired alignment verification and causal bounding)*

> [!CAUTION]
> **GOVERNANCE INVARIANT — STRICT ENFORCEMENT:**  
> **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**  
> Zero human participants have been recruited or contacted. Zero human speech samples have been recorded or ingested. All evaluations reported herein are performed strictly on programmatic synthetic fixtures. Successful technical benchmarks do NOT constitute authorization or approval to collect human field data.

---

## 1. Checkpoint Identity Verification: Arm A Control vs. V3 Champion

We conducted an exhaustive, tensor-by-tensor comparison between the control model of V4 and the winning model of V3:
- **Model Checkpoint 1 (Arm A):** `experiments/synthetic_kws_v4/checkpoints/arm_a_control.pt`
- **Model Checkpoint 2 (V3 Champion):** `experiments/synthetic_kws_v3/checkpoints/best_tcresnet14_v3.pt`

### 1.1 Empirical Verification Metrics ([`checkpoint_identity.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/checkpoint_identity.json))

| Verification Parameter | Arm A Control Checkpoint | V3 Champion Checkpoint | Comparison Result |
| :--- | :---: | :---: | :---: |
| **File Path** | `.../arm_a_control.pt` | `.../best_tcresnet14_v3.pt` | Distinct files |
| **Whole-File SHA-256** | `3170BD75BC83C3A860F68E0975E1EBB51821EE3BAD0C2C2D58E005E7DE171E23` | `6930BEDA2EFD03EBFE86A502A6019A765C0F097531C3A11F6DF6F94410B53941` | Differ at container level |
| **Metadata Keys** | `['arm_name', 'epoch', 'model_state_dict', 'num_classes', 'best_metrics']` | `['epoch', 'model_name', 'model_class', 'num_classes', 'model_state_dict', ...]` | Explain file hash difference |
| **State Dict Tensors** | 74 tensors | 74 tensors | Exact key match (74/74) |
| **Per-Tensor Bitwise Equality** | 74 / 74 identical | 74 / 74 identical | **100% bit-exact equality** |
| **Max Absolute Tensor Difference** | **0.000000e+00** | **0.000000e+00** | **Identical weights** |
| **Canonical Predictions Equality** | 1,692 predictions | 1,692 predictions | **1,692 / 1,692 identical (100.0%)** |

### 1.2 Experimental Design Rationale
In [`run_ablation.py`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/run_ablation.py#L226-L234), Arm A was explicitly defined as the **Control Arm** to benchmark against the onset-aware models (Arm B and Arm C). It was trained on the identical V3 training manifest (`train_manifest_6class.csv`) with the identical architecture (`VietnameseTCResNet`), optimizer (AdamW, lr=1e-3, weight_decay=1e-4), scheduler, augmentation pipeline, and random seed (`set_seed(42)`). Because the training pipeline is deterministic on CPU, Arm A converged to the exact same optimal parameters at epoch 9 as V3 Champion. 

Their identical diagnostic metrics across all evaluation partitions are **expected by experimental design** and verify deterministic reproducibility.

---

## 2. Verification of Alignment-Only Transformations & Paired Sample Preservation

To verify that temporal shift operations did not alter or distort underlying acoustic information, we audited all 444 paired audio files ([`alignment_invariance_results.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/alignment_invariance_results.json)):
1. **Baseline Pairs (144):** Native center-aligned clips (`pad_left` ~246 ms) transformed to left-aligned clips (`pad_left` = 0 ms).
2. **Onset-Aware Pairs (300):** Native left-aligned clips (`pad_left` = 0 ms) transformed to center-aligned clips (`pad_left` ~217–255 ms).

### 2.1 Paired Sample Preservation Metrics
- **Total Pairs Audited:** 444 / 444
- **Order-Preserving Non-Padding Sample Equality:** **100.00% (444 / 444 pairs bit-exact identical)**
- **Maximum Absolute Sample Difference:** **$0.000000$** across all preserved active speech samples.
- **Active Speech Samples Truncated:** **0 samples** across all 444 files.
- **Signal Integrity:** The temporal transformation consisted exclusively of trimming or inserting digital exact zeros ($0.0$). The underlying speech samples were preserved with zero corruption or resampling artifacts.

### 2.2 Active-Relative HF Energy Ratio Invariance (>3.5 kHz)
In the prior preliminary table, aggregate HF ratios were reported as 20.51% (native left) vs 15.81% (shifted centered). We traced this aggregate discrepancy down to individual classes:

| Class Label | Sample Count | Native Left-Aligned HF Ratio | Shifted Center-Aligned HF Ratio | Max Pairwise Difference | Invariance Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`TRAI`** | 50 | 0.1430% | 0.1430% | **0.000000%** | **BIT-EXACT INVARIANT** |
| **`PHAI`** | 50 | 32.6886% | 32.6886% | **0.000000%** | **BIT-EXACT INVARIANT** |
| **`LEN`** | 50 | 0.0639% | 0.0639% | **0.000000%** | **BIT-EXACT INVARIANT** |
| **`XUONG`** | 50 | 61.9184% | 61.9184% | **0.000000%** | **BIT-EXACT INVARIANT** |
| **`UNKNOWN`** | 50 | 0.0622% | 0.0622% | **0.000000%** | **BIT-EXACT INVARIANT** |
| **`SILENCE`** | 50 | 28.2083% | 0.0004% | 28.2079% | Fallback window artifact |

#### Resolution of the HF Ratio Discrepancy:
For all 250 speech samples, the active-relative HF ratio is **100% bit-exact invariant** between native and centered alignments. The macro-average difference occurred strictly because the `SILENCE` class contains no samples exceeding the 0.005 threshold. When evaluated under native alignment, its fallback window (samples 0–799) measured Gaussian dither (HF ratio 28.2%), whereas in centered alignment, samples 0–799 contained newly padded digital zeros ($0.0$, HF ratio 0.0%). Active speech acoustics were preserved with zero alteration.

---

## 3. Disaggregation of Three Distinct Onset Definitions

To eliminate ambiguity in temporal reporting, we disaggregate and measure three distinct onset parameters across all diagnostic conditions:

1. **Definition 1 (First Nonzero Sample):** Earliest sample index where $a[t] \neq 0.0$.
2. **Definition 2 (First Threshold-Crossing Sample):** Earliest sample index where $|a[t]| \ge 0.005$.
3. **Definition 3 (Synthetic Release-Burst Time):** Nominal time where high-frequency transient noise was injected (sample 400 relative to utterance start in V4 `TRAI`).

### 3.1 Measured Temporal Positions by Condition and Class

| Experimental Condition | Class | Def 1: First Nonzero ($a \neq 0$) | Def 2: Threshold ($|a| \ge 0.005$) | Def 3: Synthetic Burst Release | Waveform Stage |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A1-B2: Baseline Centered** | `TRAI` | Sample 3,908.4 (244.3 ms) | Sample 3,914.9 (244.7 ms) | *N/A (No burst modeled)* | Standardized Processed WAV |
| | `PHAI` | Sample 4,092.9 (255.8 ms) | Sample 4,099.4 (256.2 ms) | *N/A (No burst modeled)* | Standardized Processed WAV |
| **A1-B1: Baseline Left** | `TRAI` | Sample 0.0 (0.0 ms) | Sample 0.0 (0.0 ms) | *N/A (No burst modeled)* | Shifted Diagnostic Input |
| | `PHAI` | Sample 0.0 (0.0 ms) | Sample 0.0 (0.0 ms) | *N/A (No burst modeled)* | Shifted Diagnostic Input |
| **A2-B1: Onset Left (Native)**| `TRAI` | **Sample 0.0 (0.0 ms)** | **Sample 400.3 (25.0 ms)** | **Sample 400.0 (25.0 ms)** | Native Processed WAV |
| | `PHAI` | Sample 1.1 (0.1 ms) | Sample 8.3 (0.5 ms) | *N/A (Fricative)* | Native Processed WAV |
| **A2-B2: Onset Centered** | `TRAI` | **Sample 4,081.8 (255.1 ms)** | **Sample 4,482.1 (280.1 ms)** | **Sample 4,481.8 (280.1 ms)**| Shifted Diagnostic Input |
| | `PHAI` | Sample 4,095.6 (256.0 ms) | Sample 4,102.8 (256.4 ms) | *N/A (Fricative)* | Shifted Diagnostic Input |

### 3.2 Verification of the Centered V4 Aggregate Onset Figure
In the preliminary audit report, centered V4 audio was reported as having mean onset near sample 3,478.5. This occurred because the aggregate calculation averaged threshold crossings across all 300 clips, including UNKNOWN and SILENCE (where dither or fallback crossings pulled down the average). When disaggregated by command:
- In centered V4 `TRAI`, the closure gap begins at sample **4,081.8 (255.1 ms)** (Definition 1).
- The threshold-crossing release burst occurs at sample **4,482.1 (280.1 ms)** (Definition 2).
- The difference is exactly **$400.3\text{ samples}$ ($25.0\text{ ms}$)**, confirming that the 25 ms closure gap and release burst are preserved at the expected relative positions.

---

## 4. Proportional Causal Conclusions & Scientific Boundaries

We describe the experimental findings with rigorous causal boundaries:

### 4.1 What the Evidence Establishes:
1. **Arm B's Failure on Baseline `TRAI` is Robust to Alignment:**
   - Shifting baseline clips to left-alignment (speech at $t=0$) does **not** rescue `TRAI` recall (remains **0.0%**, 36/36 classified as `PHAI`).
   - Under both center-aligned and left-aligned conditions, Arm B rejects baseline `TRAI`.
2. **Alignment Moderates Performance Across Other Classes:**
   - In Arm B, `XUONG` jumped from **16.7% to 100.0%** when left-aligned, increasing accuracy from 54.17% to 75.00%.
   - In Arm A and V3 Champion, center-aligning onset-aware clips caused `TRAI` recall to collapse from 34.0% to **0.0%** (100% classified as `UNKNOWN`).
   - In Arm C, left-aligning baseline audio dropped `TRAI` recall from 52.8% to **0.0%**.

### 4.2 What the Evidence Does NOT Prove (Causal Boundary):
- **The 25 ms closure gap alone cannot be claimed as the isolated, proven cause of cross-generator failure.**
- In this diagnostic, the baseline generator and onset-aware generator differ not only in closure/burst modeling, but also in generator implementation, formant trajectories, pitch curves, and speaker cohorts. 
- Because we cannot synthesize an onset-aware counterpart from the baseline generator without simultaneously altering other acoustic properties, **isolated acoustic causality remains `UNRESOLVED`**.
- This test constitutes **paired alignment-sensitivity testing plus cross-generator generalization testing**, not a fully crossed factorial experiment on identical utterances.

---

## 5. Comprehensive Discrepancy Classification Log

| ID | Issue Description | Status | Scientific Classification & Resolution |
| :--- | :--- | :---: | :--- |
| **DISC-01** | Arm A validation accuracy reported as 96.76% vs 82.87%. | **RESOLVED** | Proven to reflect two distinct validation distributions: baseline generator (`val_manifest_6class.csv`, 100% at best ep, 96.76% at ep 25) vs onset-aware generator (`val_manifest_onset_aware.csv`, 82.87%). |
| **DISC-02** | Arm B zero `TRAI` recall on older holdouts: closure causality vs alignment. | **UNRESOLVED** *(Causality)* / **RESOLVED** *(Observational)* | Observational fact is verified: Arm B achieves 0.0% recall on baseline `TRAI` under both left and center alignment. However, isolated causal attribution to the 25 ms closure alone is unproven because generator identity, onset parameters, and acoustic formants co-vary across generators. |
| **DISC-03** | Parent-recording lineage in historical benchmark schema. | **NOT_VERIFIABLE** | Historical 16-column schema lacks `parent_recording_id` and `session_id`. Lineage cannot be automatically validated for historical benchmarks and is classified `NOT_VERIFIABLE`. |
| **DISC-04** | Pitch augmentation vs rate resampling terminology. | **RESOLVED** | Confirmed implementation uses linear rate resampling, which alters duration, pitch, and formants simultaneously. |
| **DISC-05** | Arithmetic contradictions in Markdown table recalls vs accuracies. | **RESOLVED** | Corrected with canonical confusion matrices (Arm B Hist 144: 16.7%; Arm C Hist 144: 5.6%; Arm A Val Onset: 75.0%), reconciling all accuracies and target miss counts with zero residual error. |
| **DISC-06** | Waveform closure RMS (0.05551) vs silence threshold (< 0.005) contradiction. | **RESOLVED** | Traced to an onset detection bug in `reproduce_and_audit.py` (`|a| > 0.01`) that skipped the 25 ms closure gap and measured samples 400 to 800 (RMS = 0.05550). True closure window (samples 0 to 400) has RMS = 0.000382, peak amplitude = 0.001740, and 100.00% of samples $< 0.005$. |
| **DISC-07** | Temporal alignment discrepancy across audio processing stages. | **RESOLVED** | Fully documented across the 5-stage pipeline: older generator speech starts at $t=0$ in raw audio, but was centered (~246 ms leading silence) in `audio_16k_1s`; V4 generator was right-padded (speech at $t=0$, burst at sample 400 = 25 ms). Both loaded as 16,000 samples into PyTorch. |

---

## 6. Audit Test & Repository Safety Status

1. **Acceptance Test Suite:**
   - Automated regression test suite: **62 passed in 26.58s** (`tests/test_campaign_acceptance.py`), verifying checkpoint tensor identity, paired sample preservation, alignment invariance of active-relative HF ratios, distinct onset definitions, and discrepancy classifications.
2. **Repository Cleanliness:**
   - `git diff --check`: Exit code 0 (clean).
   - `git status --short`: Preserved all user modifications; zero unprompted deletions.
   - `git diff --cached`: Exit code 0 (**0 files staged**).
3. **Protected File Baseline Hashes:**
   - `data/manifests/test_manifest_v2.csv`: `9E3904C8EA427244F7280A41140D7EFD19FD9625B359290ADB7ADCD4FA2E1C13` (**MATCH**)
   - `data/raw/field_recordings/field_collection_manifest.csv`: `B389282F9853A74484B39356EE85E5BD7E74C0C46481300524C7BD2894AA8C85` (**MATCH**)
   - `data/raw/field_recordings/consent_ledger.json`: `BE4BFDD7CEC0C87A1DC97455E5783054EF8093C638279C44940C896F5BDA2BA0` (**MATCH**)
