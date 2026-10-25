# Objective 1: Tables Catalog & Specifications

> **Path:** `docs/objective_1/tables/`  
> **Count:** 12 CSV Tables  
> **Generation Script:** `scripts/objective_1_docs/generate_tables.py`  
> **Audit Status:** Audited and verified against primary manifests, logs, and checkpoints.

---

## Table Directory & Overview

| Table File | Title / Focus | Primary Metrics / Dimensions | Key Findings / Highlights |
| :--- | :--- | :--- | :--- |
| [`table_01_dataset_source_inventory.csv`](./table_01_dataset_source_inventory.csv) | Candidate Dataset Source & Licensing Inventory | Source name, URL, License, Commercial status, Samples, Status | Google FLEURS (579 clips, CC-BY-4.0) approved for UNKNOWN; Husein noise (400 clips, CC-BY-4.0) approved for SILENCE; 1,000 synthetic calibration commands; VIVOS rejected (CC-BY-NC-SA); VietSpeech quarantined; Common Voice has 0 approved samples in manifests. |
| [`table_02_class_split_distribution.csv`](./table_02_class_split_distribution.csv) | Class Partition Allocation by Split (V2) | Class Name, Class ID, Train, Val, Test, Total Counts | Train (1,281: 178/178/178/178/319/250), Val (284: 36/36/36/36/80/60), Test (534: 36/36/36/36/300/90). Total approved pool: 2,099 audio files. |
| [`table_03_genuine_vs_fixture_breakdown.csv`](./table_03_genuine_vs_fixture_breakdown.csv) | Ground-Truth Acoustic Provenance Breakdown | Category, Genuine Speech, Real Noise, Synthetic Ref, Harness Fixtures, Quarantined | 100% of 1,000 target commands are synthetic calibration speech. 100% of field collection records are automated test fixtures (58 files on disk). Genuine human speech exists only in UNKNOWN (579 FLEURS clips); real noise in SILENCE (400 Husein clips). |
| [`table_04_field_campaign_progress.csv`](./table_04_field_campaign_progress.csv) | Human Field Recording Campaign Progress | Command Label, Vietnamese Word, Class ID, Genuine Takes, Target Budget, Status | Planned budget: 6,000 recordings (1,500/cmd). Audited genuine human takes: 0. Status: Awaiting human participant recruitment; collection infrastructure verified. |
| [`table_05_speaker_dialect_distribution.csv`](./table_05_speaker_dialect_distribution.csv) | Regional Dialect Stratification & Deficit | Dialect, Target Quota, Genuine Audited, Unit Test Mocks, Gap | Target: Northern (50%, 3,000 takes), Central (25%, 1,500), Southern (25%, 1,500). Genuine audited: 0 across all dialects. Test fixtures: mock fixtures only. |
| [`table_06_model_architecture_comparison.csv`](./table_06_model_architecture_comparison.csv) | Model Architectural Specification Matrix | Architecture, Params, MACs, Opset, FP32 Size, INT8 Size, Latency, Status | Phase 2 Baseline 2D-CNN (`VietnameseKWSNet`: 138,534 params, 555.4 KB FP32, 148.8 KB INT8, 0.25 ms); Temporal-CNN (292,582 params, 1,150.0 KB FP32, 1.44 ms); TC-ResNet-14 Champion (`VietnameseTCResNet`: 272,934 params, 1,093.7 KB FP32, missing INT8, 0.35 ms). BC-ResNet-1 does not exist. |
| [`table_07_accuracy_and_metrics_comparison.csv`](./table_07_accuracy_and_metrics_comparison.csv) | Diagnostic Model Performance on Frozen Test Set | Evaluation Metric, Baseline 2D-CNN, TC-ResNet-14, Recall by Class | Test set (N=534): Baseline 2D-CNN achieves 97.75% acc, 0.9466 Macro-F1, 69.44% TRAI, 16.67% Northern TRAI; Champion TC-ResNet-14 achieves 96.63% acc, 0.9402 Macro-F1, 75.00% TRAI, 25.0% Northern TRAI. Historical baseline: 97.52% acc, 0.7812 true 6-class F1 due to 0 SILENCE support. |
| [`table_08_confusion_matrix_tcresnet.csv`](./table_08_confusion_matrix_tcresnet.csv) | TC-ResNet-14 Confusion Matrix on Frozen Test Set | True Class vs Predicted (TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE) | Support: 36/cmd, 300 UNKNOWN, 90 SILENCE. 27 TRAI correct, 9 misclassified as PHAI (shared rhyme `/aːj/`). 36 PHAI, 36 LEN, 34 XUONG (2 to LEN), 293 UNKNOWN (7 to SILENCE), 90 SILENCE correct. |
| [`table_09_streaming_performance_comparison.csv`](./table_09_streaming_performance_comparison.csv) | Continuous Streaming Detection Trade-offs | Threshold $\tau$, False Alarms (Speech/Noise), FA/hr, Missed Cmd Rate, Latency | Evaluated on 8.0-minute continuous stream (0.133 hr). At $\tau = 0.7$: 117/144 TP, 0 FA clean/noise (Poisson 95% bound $\le 22.5$ FA/hr), 706.4 ms mean trigger latency. |
| [`table_10_onnx_model_benchmarks.csv`](./table_10_onnx_model_benchmarks.csv) | On-Device Export & ONNX Benchmarks | Model Artifact, Format, On-Disk Size, Max Abs Diff, CPU Latency, Status | TC-ResNet FP32 (1,093.7 KB, opset 14, max diff $1.19\times 10^{-6}$, 0.35 ms CPU); TC-ResNet INT8 (missing on disk); Baseline FP32 (555.4 KB, opset 14, max diff $9.54\times 10^{-7}$, 0.25 ms CPU); Baseline INT8 (148.8 KB, opset 14, genuine INT8 ops, 0.91 ms CPU). |
| [`table_11_automated_test_results.csv`](./table_11_automated_test_results.csv) | Automated Verification & Test Harness Register | Test File, Test Case Function, Test Purpose, Execution Status | 19 passing tests: 12 field acquisition tests (`tests/test_field_acquisition.py`) + 7 deployment parity & taxonomy tests (`tests/test_parity.py`). |
| [`table_12_outstanding_issues_and_severity.csv`](./table_12_outstanding_issues_and_severity.csv) | Outstanding Defects, Risks, and Blockers Register | Issue ID, Severity, Category, Description, Impact, Action | P0: 0 genuine field recordings; P1: Northern TRAI recall deficit (25.0%); P2: Missing INT8 TC-ResNet file; P3: Windows iOS tooling blocked; P4: Streaming false alarm duration limited to 8.0 minutes. |

---

## Reproduction & Verification

To programmatically re-generate and update all 12 tables:

```powershell
python scripts/objective_1_docs/generate_tables.py
```
