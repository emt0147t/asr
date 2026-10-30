# Phase 8H-A: Final Holdout Governance & Evaluation Report

> **Mandatory Governance Labels:**
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`

---

## 1. Executive Summary & Holdout Integrity Verdict

This workstream audited the provenance of the sealed holdout manifest ([`phase8f_b_final_holdout_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8f_b_final_holdout_manifest.csv), 1,340 instances, SHA-256 `251546a5...`) and conducted the formal blind evaluation of the selected candidate (**Arm B**, 138k parameters, Phase 7B acoustic channel augmentation, seeds 42, 43, 44).

### Key Provenance Findings
1. **Zero Source Audio Overlap:** The holdout instances share **0 source audio files** with the training set (`TRAIN_CLEAN` / `AUG_TRAIN`), validation set (`VAL_CLEAN`), Phase 7B augmentations, or historical Phase 6.1A benchmarks.
2. **Zero Speaker Overlap:** The 169 speakers represented in the holdout are strictly disjoint from the 706 training speakers and 165 validation speakers.
3. **Historical Data Reclassification:** The Phase 6.1A holdout remains classified as a historical development benchmark.
4. **Data Characterization & Limitation:** The 1,340 holdout samples represent clean, studio-standardized speech across unseen speakers (1,280 isolated utterances + 60 concatenated multi-command sequences). While statistically independent and disjoint from model training, it is **not a natural multi-speaker continuous conversation dataset in the wild**. True conversational generalization remains `NOT_VALIDATED_PENDING_DATA` until the human field recording campaign is executed.

---

## 2. Seed-Level Performance on Sealed Holdout (1,340 Instances)

| Model Candidate | Seed | Single Acc (%) | Target Acc (%) | Reject Acc (%) | Macro F1 (%) | Seq Exact Match (%) | Seq WER (%) | 0ms Gap Exact (%) | 50ms Gap Exact (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm B (Seed 42)** | 42 | **93.59%** | 98.75% | 88.44% | **93.65%** | **100.0%** | **0.0%** | 100.0% | 100.0% |
| **Arm B (Seed 43)** | 43 | **94.92%** | 98.75% | 91.09% | **95.08%** | **100.0%** | **0.0%** | 100.0% | 100.0% |
| **Arm B (Seed 44)** | 44 | **92.66%** | 99.69% | 85.62% | **93.1%** | **100.0%** | **0.0%** | 100.0% | 100.0% |
| **Arm B Mean** | mean | **93.72%** | 99.06% | 88.38% | **93.94%** | **100.0%** | **0.0%** | 100.0% | 100.0% |
| **Arm B Std Dev** | std | ±1.14% | ±0.54% | ±2.74% | ±1.02% | ±0.0% | ±0.0% | ±0.0% | ±0.0% |

---

## 3. Per-Class F1 Scores (Mean across Seeds 42, 43, 44)

| Class | Label Type | Mean Precision (%) | Mean Recall (%) | Mean F1 (%) | Std Dev F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TRAI** | Target Command | 91.82% | 97.92% | **94.75%** | ±0.85% |
| **PHAI** | Target Command | 91.65% | 100.0% | **95.63%** | ±1.43% |
| **LEN** | Target Command | 80.44% | 99.79% | **89.02%** | ±2.83% |
| **XUONG** | Target Command | 95.2% | 98.54% | **96.82%** | ±0.38% |
| **REJECT** | Reject Foil | 99.25% | 88.38% | **93.48%** | ±1.32% |

---

## 4. Continuous Sequence Timing Tolerance

| Inter-Command Gap | Total Trials | Exact Matches | Exact Match % | Sequence WER % |
| :--- | :---: | :---: | :---: | :---: |
| **0 ms (Continuous Utterance)** | 20 | 20 | **100.0%** | 0.0% |
| **50 ms (Natural Conversational Pause)** | 20 | 20 | **100.0%** | 0.0% |
| **100 ms (Extended Deliberate Pause)** | 20 | 20 | **100.0%** | 0.0% |

---

## 5. Artifact Reference Ledger

- Provenance Audit: [`reports/phase8h_a_holdout_provenance.csv`](file:///d:/trainidentificationaudio/reports/phase8h_a_holdout_provenance.csv)
- Seed Level Metrics: [`reports/phase8h_a_seed_level_results.csv`](file:///d:/trainidentificationaudio/reports/phase8h_a_seed_level_results.csv)
- Confusion Matrices: [`reports/phase8h_a_confusion_matrices.json`](file:///d:/trainidentificationaudio/reports/phase8h_a_confusion_matrices.json)
