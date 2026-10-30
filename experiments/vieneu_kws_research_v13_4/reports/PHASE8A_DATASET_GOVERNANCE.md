# Phase 8A: Dataset Governance & Split Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `91d91dbf6ec23486cb6f95e0a62a48d590360bad`  
**Status:** Completed & Validated  
**Leakage Gate Decision:** **LEAKAGE-SAFE GATE PASSED**  
**Date:** 2026-10-31

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Leakage Gate Determination

Phase 8A performed an exhaustive forensic audit of all datasets, audio provenance records, speaker identities, and derivative trees across the Vietnamese Continuous KWS project before training authorization.

### Audit Verdict: `LEAKAGE-SAFE GATE PASSED`

1. **Zero Speaker Overlap between Clean Training and Evaluation:**
   - Forensic analysis of `experiments/vieneu_kws_research_v12/manifests/train_manifest.csv` (5,440 samples), `val_manifest.csv` (1,280 samples), and `test_manifest.csv` (1,280 samples) established that the 700 training speakers share **0 overlap** with the 164 validation speakers and 168 test speakers.
2. **Absolute Quarantine of Confirmatory & Holdout Audio:**
   - The confirmatory speech corpus (`spk_north_01` to `spk_north_08`) was verified to have **zero overlap** with the v12 training manifest (`intersection = set()`).
   - The Phase 6.1A holdout manifest strictly quarantines Northern speakers **`spk_north_03`**, **`spk_north_06`**, and **`spk_north_08`**.
   - No derivatives of these three speakers exist in any training split.
3. **Source-to-Derivative Integrity in Phase 7B Augmentation:**
   - Phase 7B augmented training files ($N=200$) were strictly derived from the 5 permitted speakers (`spk_north_01, 02, 04, 05, 07`).
   - Phase 7B holdout files ($N=60$) were strictly derived from the 3 quarantined speakers (`spk_north_03, 06, 08`).
   - **Zero cross-split contamination exists** in the derivative tree.
4. **Sufficiency Determination:**
   - While the 200 Phase 7B augmented files are high-quality, 200 files alone cannot train a 138k or 304k parameter CNN from scratch without catastrophic overfitting.
   - For Phase 8 training, the augmentation engine must transform the full 5,440-sample training corpus during training (or generate matched transformed versions of the base corpus), while preserving the un-augmented validation set for early stopping and the quarantined holdout manifest for final benchmarking.

---

## 2. Dataset Partition Audit Matrix

Recorded in [`reports/phase8a_dataset_audit.csv`](file:///d:/trainidentificationaudio/reports/phase8a_dataset_audit.csv):

| Dataset Partition | Total Files | Independent Source Clips | Unique Speakers | Known Speakers (%) | Overlap with Holdout | Governance Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v12_train_clean`** | 5,440 | 5,440 | 701 | 100.0% (speech) | 0 | **LEAKAGE_SAFE** |
| **`v12_val_clean`** | 1,280 | 1,280 | 165 | 100.0% (speech) | 0 | **LEAKAGE_SAFE** |
| **`v12_test_clean`** | 1,280 | 1,280 | 169 | 100.0% (speech) | 0 | **LEAKAGE_SAFE** |
| **`v13_confirmatory_raw`**| 1,000 | 1,000 | 8 | 80.0% (20% noise)| 3 (Quarantined) | **PARTITIONED** |
| **`phase7b_aug_train`** | 200 | 128 | 5 | 100.0% | 0 | **LEAKAGE_SAFE** |
| **`phase7b_aug_holdout`** | 60 | 72 | 3 | 100.0% | 3 | **QUARANTINED** |
| **`phase6_1a_holdout`** | 55 | 55 | 3 | 100.0% | 3 | **QUARANTINED** |

---

## 3. Explicit Source-to-Derivative Provenance Mapping

The complete provenance tree has been serialized into [`reports/phase8a_provenance_graph.json`](file:///d:/trainidentificationaudio/reports/phase8a_provenance_graph.json):
- **Graph Nodes:** 1260 entities (physical source recordings and generated derivatives).
- **Graph Edges:** 260 derivation links with explicit parameter sets (random seed, microphone EQ profile, RT60, gain scaling, and target SNR).
- **Cross-Split Verification:** Validated that no edge connects a training derivative to a holdout evaluation node.

---

## 4. Deliverables Index

- **Dataset Audit CSV:** [`reports/phase8a_dataset_audit.csv`](file:///d:/trainidentificationaudio/reports/phase8a_dataset_audit.csv)
- **Split Manifest CSV:** [`reports/phase8a_split_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8a_split_manifest.csv) (8260 rows)
- **Provenance Graph JSON:** [`reports/phase8a_provenance_graph.json`](file:///d:/trainidentificationaudio/reports/phase8a_provenance_graph.json)
- **Technical Report:** [`reports/PHASE8A_DATASET_GOVERNANCE.md`](file:///d:/trainidentificationaudio/reports/PHASE8A_DATASET_GOVERNANCE.md)
