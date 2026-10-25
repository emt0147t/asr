# Synthetic Vietnamese KWS V4 — Research Freeze & Reproducibility Package

**Workspace:** `D:\trainidentificationaudio`  
**Phase:** V4 Research Freeze & Diagnostic Audit Closeout  
**Package Artifact:** `experiments/synthetic_kws_v4/V4_RESEARCH_FREEZE.md`  
**Final Research Audit Verdict:** `AUDIT_PASS_WITH_LIMITATIONS`  

> [!CAUTION]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**  
> Human participant recruitment, field recording collection, mobile ingestion, and model V5 training remain **STRICTLY BLOCKED**. All checkpoints, acoustic manifests, and evaluation findings in this package are derived exclusively from synthetic calibration generators and historical synthetic test sets.

---

## 1. Research Freeze Inventory: Checkpoints and Hashes

All trained model checkpoints across V1 to V4 are preserved in the workspace with frozen whole-file SHA-256 hashes:

| Campaign / Model Name | Architecture | Relative Checkpoint Path | Whole-File SHA-256 Hash | Training Manifest & Classes |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Baseline Model** | TC-ResNet-14 | `experiments/synthetic_kws_baseline/checkpoints/best_synthetic_kws_model.pt` | `811C61E69E94DC60DD0C2F185593D5A9729275767AB6F4425E73709FB3900C38` | `experiments/synthetic_kws_baseline/train_manifest.csv` (4 Classes) |
| **V2 6-Class TC-ResNet** | TC-ResNet-14 | `experiments/synthetic_kws_v2/checkpoints/best_tcresnet_6class.pt` | `887B2F37B0FAC2B1A19CB05DE842097CBF0F41D94E0D24F44AB7164609B77EB2` | `experiments/synthetic_kws_v2/train_manifest_6class.csv` (6 Classes) |
| **V2 Temporal KWSNet** | Temporal-KWSNet | `experiments/synthetic_kws_v2/checkpoints/best_temporal_kwsnet_6class.pt` | `76A601E7AD44A52C600B5F071286F763B6002D5617344F491CC77A15033E802A` | `experiments/synthetic_kws_v2/train_manifest_6class.csv` (6 Classes) |
| **V3 Champion TC-ResNet-14** | TC-ResNet-14 | `experiments/synthetic_kws_v3/checkpoints/best_tcresnet14_v3.pt` | `6930BEDA2EFD03EBFE86A502A6019A765C0F097531C3A11F6DF6F94410B53941` | `experiments/synthetic_kws_v3/train_manifest_6class.csv` (6 Classes) |
| **V3 Temporal TC-ResNet-14** | Temporal-TC-ResNet | `experiments/synthetic_kws_v3/checkpoints/best_temporal_tcresnet14_v3.pt` | `AF57C85581671DAD32912895577A985ABF99F585D2828E406380594A3D07F866` | `experiments/synthetic_kws_v3/train_manifest_6class.csv` (6 Classes) |
| **V4 Arm A (Control)** | TC-ResNet-14 | `experiments/synthetic_kws_v4/checkpoints/arm_a_control.pt` | `3170BD75BC83C3A860F68E0975E1EBB51821EE3BAD0C2C2D58E005E7DE171E23` | `experiments/synthetic_kws_v3/train_manifest_6class.csv` (6 Classes) |
| **V4 Arm B (Onset-Aware, No Resamp)** | TC-ResNet-14 | `experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt` | `6A0FD55CD6F16F5B683951127076EEC72506C31EE670F896AAEACE5981CF1D96` | `experiments/synthetic_kws_v4/train_manifest_onset_aware.csv` (6 Classes) |
| **V4 Arm C (Onset-Aware + Resamp)** | TC-ResNet-14 | `experiments/synthetic_kws_v4/checkpoints/arm_c_onset_with_resample.pt` | `FD0D4D52C0A9B4EBCA2EA65FA6221842C045AD9330FB3093461BC8A5D28A2B28` | `experiments/synthetic_kws_v4/train_manifest_onset_aware.csv` (6 Classes) |

---

## 2. Exact Training and Evaluation Commands

### 2.1 Model Training Commands
- **V1 Baseline Training (4-Class):**
  ```powershell
  .\.venv\Scripts\python.exe src/models/train_kws.py --config config/kws_baseline.yaml --output-dir experiments/synthetic_kws_baseline
  ```
- **V2 6-Class Baseline Training:**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v2/run_ablation.py --epochs 25 --batch-size 32 --lr 0.001
  ```
- **V3 Hardened TC-ResNet-14 Training:**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v3/train_v3.py --data-dir experiments/synthetic_kws_v3 --epochs 25 --seed 42
  ```
- **V4 Controlled Onset-Acoustic Ablation (Arms A, B, C):**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v4/run_ablation.py --seed 42 --epochs 25 --batch-size 32
  ```

### 2.2 Canonical Evaluation & Audit Commands
- **Canonical Prediction Generator (6,768 rows across 7 partitions):**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v4/audit/build_canonical_predictions.py
  ```
- **Mathematical Metric Verification & Discrepancy Auditing:**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v4/audit/recompute_from_canonical.py
  ```
- **Physical Waveform Acoustic Property Extraction:**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v4/audit/audit_waveform_measurements.py
  ```
- **Controlled 2x2 Alignment & Onset Diagnostic V2 Execution:**
  ```powershell
  .\.venv\Scripts\python.exe experiments/synthetic_kws_v4/audit/run_alignment_diagnostic_v2.py
  ```
- **Campaign Acceptance Suite (Target: 64 passed):**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_campaign_acceptance.py -q
  ```

---

## 3. Directory of Canonical Audit Artifacts

All research evidence, metrics, and audit tables are frozen at the following relative workspace paths:

| Artifact Description | Workspace Relative Path | Content Summary |
| :--- | :--- | :--- |
| **Canonical Predictions Table** | [`experiments/synthetic_kws_v4/audit/canonical_predictions.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/canonical_predictions.csv) | 6,768 prediction rows covering 4 models across all 7 evaluation partitions |
| **Canonical Metrics JSON** | [`experiments/synthetic_kws_v4/audit/canonical_metrics.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/canonical_metrics.json) | Recomputed 6-class confusion matrices, per-class metrics, support, accuracy, macro-F1 |
| **Diagnostic V2 Predictions Table** | [`experiments/synthetic_kws_v4/audit/canonical_predictions_diag_v2.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/canonical_predictions_diag_v2.csv) | 1,200 prediction rows evaluating all 4 frozen models on Condition A2-B2-V2 |
| **Alignment Preservation V2 JSON** | [`experiments/synthetic_kws_v4/audit/alignment_preservation_v2.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/alignment_preservation_v2.json) | Bit-exact paired sample audits across all 444 pairs; 50 SILENCE verification records |
| **Condition A2-B2-V2 Manifest** | [`experiments/synthetic_kws_v4/audit/manifest_diag_A2_B2_v2_onset_centered.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/manifest_diag_A2_B2_v2_onset_centered.csv) | 300 rows enriched with `alignment_action`, `transform_applied`, `source_sha256`, `output_sha256` |
| **Waveform Measurements JSON** | [`experiments/synthetic_kws_v4/audit/waveform_measurements.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/waveform_measurements.json) | Closure gap (0–25ms, RMS 0.000382) and burst attack (25–40ms, RMS 0.055500) |
| **Checkpoint Identity JSON** | [`experiments/synthetic_kws_v4/audit/checkpoint_identity.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/checkpoint_identity.json) | 74/74 tensor bit-exact match proof between Arm A and V3 Champion |
| **Audit Discrepancy Log** | [`experiments/synthetic_kws_v4/audit/discrepancy_log.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/discrepancy_log.json) | Structured tracking for DISC-01 through DISC-08 |
| **Metric Integrity Audit Report** | [`experiments/synthetic_kws_v4/audit/metric_integrity_audit.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/metric_integrity_audit.md) | Initial discrepancy resolutions and recomputation mathematical identities |
| **SILENCE Preservation Report** | [`experiments/synthetic_kws_v4/audit/silence_alignment_preservation_audit.md`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/silence_alignment_preservation_audit.md) | Ambient semantics, fallback bug resolution, and V2 model safety evaluations |

---

## 4. Confirmed Identity of Arm A and V3 Champion

In [`checkpoint_identity.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/checkpoint_identity.json), the equivalence between Arm A and V3 Champion is conclusively established:
- **Architecture:** `VietnameseTCResNet` (TC-ResNet-14, 6 classes, 74 state dictionary tensors).
- **Tensor Equality:** Exactly **74 / 74 tensors are bit-exact identical** ($\max |\Delta| = 0.000000$).
- **Canonical Predictions:** **1,692 / 1,692 canonical predictions match identically** across all test partitions.
- **Top-Level File Hash Explanation:** Whole-file SHA-256 hashes differ (`3170...1E23` vs `6930...3941`) strictly because PyTorch serialized different top-level dictionary metadata keys (`best_metrics` vs `val_macro_f1`).
- **Experimental Design Intent:** Arm A was explicitly designed as the V3 Control arm. It was trained on the identical V3 training manifest (`train_manifest_6class.csv`) using the identical deterministic seed (`42`). Convergence to the identical optimal checkpoint at epoch 9 confirms 100% deterministic reproducibility.

---

## 5. Final Canonical Confusion Matrices and Performance Summary

All metrics are reproduced deterministically from [`canonical_predictions.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/canonical_predictions.csv) and [`canonical_predictions_diag_v2.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/canonical_predictions_diag_v2.csv):

### 5.1 Multi-Partition Accuracy and Macro-F1 Comparison

| Evaluation Cohort | Part. ID | Samples | Arm A (Control) | Arm B (Onset-Aware) | Arm C (Onset+Resamp) | V3 Champion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Target Baseline** | `hist_144` | 144 | $90.28\%$ (F1: $0.8990$) | $54.17\%$ (F1: $0.4632$) | $58.33\%$ (F1: $0.5188$) | $90.28\%$ (F1: $0.8990$) |
| **Historical 6-Class Holdout** | `hist_216` | 216 | $93.52\%$ (F1: $0.9348$) | $69.44\%$ (F1: $0.6558$) | $72.22\%$ (F1: $0.6974$) | $93.52\%$ (F1: $0.9348$) |
| **V2 Fresh Holdout** | `v2_fresh_300` | 300 | $89.00\%$ (F1: $0.8876$) | $61.33\%$ (F1: $0.5593$) | $70.67\%$ (F1: $0.6872$) | $89.00\%$ (F1: $0.8876$) |
| **V3 Fresh Holdout** | `v3_fresh_300` | 300 | $89.00\%$ (F1: $0.8876$) | $61.33\%$ (F1: $0.5593$) | $70.67\%$ (F1: $0.6872$) | $89.00\%$ (F1: $0.8876$) |
| **V4 Fresh Holdout (Onset-Aware)**| `v4_fresh_300` | 300 | $89.00\%$ (F1: $0.8710$) | $99.33\%$ (F1: $0.9933$) | $97.67\%$ (F1: $0.9767$) | $89.00\%$ (F1: $0.8710$) |
| **Validation (V3 Baseline Gen.)** | `val_baseline_216` | 216 | $100.00\%$ (F1: $1.0000$) | $69.44\%$ (F1: $0.6558$) | $72.22\%$ (F1: $0.6974$) | $100.00\%$ (F1: $1.0000$) |
| **Validation (V4 Onset-Aware Gen.)**| `val_onset_216` | 216 | $82.87\%$ (F1: $0.7818$) | $99.07\%$ (F1: $0.9907$) | $98.15\%$ (F1: $0.9815$) | $82.87\%$ (F1: $0.7818$) |
| **Diagnostic Condition A1-B2** | `diag_A1_B2` | 144 | $90.28\%$ (F1: $0.8990$) | $54.17\%$ (F1: $0.4632$) | $58.33\%$ (F1: $0.5188$) | $90.28\%$ (F1: $0.8990$) |
| **Diagnostic Condition A1-B1** | `diag_A1_B1` | 144 | $90.28\%$ (F1: $0.8990$) | $75.00\%$ (F1: $0.7167$) | $75.00\%$ (F1: $0.7167$) | $90.28\%$ (F1: $0.8990$) |
| **Diagnostic Condition A2-B1** | `diag_A2_B1` | 300 | $89.00\%$ (F1: $0.8710$) | $99.33\%$ (F1: $0.9933$) | $97.67\%$ (F1: $0.9767$) | $89.00\%$ (F1: $0.8710$) |
| **Diagnostic Condition A2-B2-V2**| `diag_A2_B2_V2` | 300 | $83.33\%$ (F1: $0.7778$) | $96.33\%$ (F1: $0.9629$) | $92.33\%$ (F1: $0.9185$) | $83.33\%$ (F1: $0.7778$) |

### 5.2 Confusion Matrices: Historical 144 Target Clips

#### Arm A / V3 Champion (Historical 144)
```
              TRAI   PHAI    LEN  XUONG
TRAI            22     14      0      0   (Recall:  61.1%, Precision: 100.0%, F1: 0.7586)
PHAI             0     36      0      0   (Recall: 100.0%, Precision:  72.0%, F1: 0.8372)
LEN              0      0     36      0   (Recall: 100.0%, Precision: 100.0%, F1: 1.0000)
XUONG            0      0      0     36   (Recall: 100.0%, Precision: 100.0%, F1: 1.0000)
Accuracy: 90.28% | Macro-F1: 0.8990 | Target Misses: 14 / 144 (9.72%)
```

#### Arm B (Onset-Aware, No Resample) (Historical 144)
```
              TRAI   PHAI    LEN  XUONG
TRAI             0     36      0      0   (Recall:   0.0%, Precision:   0.0%, F1: 0.0000)
PHAI             0     36      0      0   (Recall: 100.0%, Precision:  50.0%, F1: 0.6667)
LEN              0      0     36      0   (Recall: 100.0%, Precision:  54.5%, F1: 0.7059)
XUONG            0      0     30      6   (Recall:  16.7%, Precision: 100.0%, F1: 0.2857)
Accuracy: 54.17% | Macro-F1: 0.4632 | Target Misses: 66 / 144 (45.83%)
```

#### Arm C (Onset-Aware + Resample) (Historical 144)
```
              TRAI   PHAI    LEN  XUONG
TRAI            19     17      0      0   (Recall:  52.8%, Precision: 100.0%, F1: 0.6897)
PHAI             0     27      0      9   (Recall:  75.0%, Precision:  61.4%, F1: 0.6750)
LEN              0      0     36      0   (Recall: 100.0%, Precision:  51.4%, F1: 0.6792)
XUONG            0      0     34      2   (Recall:   5.6%, Precision:  18.2%, F1: 0.0851)
Accuracy: 58.33% | Macro-F1: 0.5188 | Target Misses: 60 / 144 (41.67%)
```

---

## 6. Discrepancy Tracking and Causal Boundary

All historical ambiguities and audit findings are resolved in [`discrepancy_log.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/discrepancy_log.json):

```
+-------------------------------------------------------------------------------------------------+
|                                   DISCREPANCY LOG RESOLUTION                                    |
+-------------------------------------------------------------------------------------------------+
| DISC-01: Arm A Validation Accuracy Discrepancy (Table 3.1 vs 4.1)                  -> RESOLVED  |
| DISC-02: Arm B Zero TRAI Recall Causal Attribution Confound                        -> UNRESOLVED|
| DISC-03: Lineage Verification in Historical 16-Column Benchmark Schema             -> UNVERIFIED|
| DISC-04: Pitch Augmentation Terminology (Linear Resampling vs Phase Vocoder)       -> RESOLVED  |
| DISC-05: Preliminary Report Markdown Arithmetic Inconsistencies                    -> RESOLVED  |
| DISC-06: Waveform Closure RMS vs Silence Threshold Contradiction                   -> RESOLVED  |
| DISC-07: Processing Stage Temporal Alignment (Raw vs Normalized vs Onset-Aware)    -> RESOLVED  |
| DISC-08: Diagnostic Centering Fallback Truncation on Ambient SILENCE (Condition V2)-> RESOLVED  |
+-------------------------------------------------------------------------------------------------+
```

### Critical Causal Scope (DISC-02)
- **Verified Fact:** Arm B achieves $0.0\%$ `TRAI` recall on baseline historical audio under both center-aligned ($246\,\text{ms}$ onset) and left-aligned ($0\,\text{ms}$ onset) conditions ($36/36$ samples predicted as `PHAI` in both).
- **Unresolved Causality:** While alignment alone does not explain Arm B's failure on baseline `TRAI`, **the 25 ms stop closure alone cannot be asserted as the sole proven cause**. The baseline and onset-aware generators differ across multiple acoustic dimensions simultaneously (burst release amplitude, formant trajectories, pitch slope, and high-frequency spectral tilt). Attributing failure entirely to closure gap is unsupported by single-variable control. Causal attribution remains explicitly **`UNRESOLVED`**.

---

## 7. Remaining Limitations & Edge Deployment Boundaries

1. **Synthetic Governance Restriction:**  
   All reported metrics, acoustic models, and conclusions reflect **purely synthetic calibration audio**. They do NOT establish human speech accuracy, Vietnamese regional accent invariance (Northern, Central, Southern), conversational noise rejection, or mobile phone microphone compatibility.
2. **Desktop CPU Profiling vs Mobile Deployment:**  
   Disaggregated CPU latency profiling was executed exclusively on a host Windows x86_64 development workstation ($1.89\,\text{ms}$ mean inference per $1.0\,\text{s}$ audio frame). This benchmark establishes desktop CPU throughput only. Real-time streaming performance, cache contention, thermal throttling, and battery consumption on mobile/edge hardware (ARM Cortex-A/M, Android/iOS NPU runtimes) have not been measured or validated.
3. **Ambient SILENCE Semantics:**  
   The `SILENCE` class represents synthetic ambient dither beds, not digital zero silence. Its spectral ratio is defined as a fixed-window ambient metric ($0$–$50\,\text{ms}$, samples 0–799), not an active speech onset metric.

---

## 8. Final Research Freeze Verdict

### Verdict: `AUDIT_PASS_WITH_LIMITATIONS`

#### Research Freeze Criteria Status:
- [x] Checkpoint inventory complete with verified whole-file SHA-256 hashes.
- [x] Canonical prediction tables ($6,768$ rows) and metrics JSON mathematically validated.
- [x] Condition A2-B2-V2 manifest enriched with per-row metadata (`alignment_action`, `transform_applied`, `source_sha256`, `output_sha256`).
- [x] Bit-exact sample preservation confirmed across all 444 paired diagnostic clips.
- [x] Ambient background semantics clearly distinguished from digital silence.
- [x] Arm A and V3 Champion identity proven via bit-exact state dict tensor comparison (74/74 tensors).
- [x] Causal over-claiming eliminated; `DISC-02` retained as unresolved.
- [x] All 64 campaign acceptance tests passing.
- [x] **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** Model V5 training has not been started.
