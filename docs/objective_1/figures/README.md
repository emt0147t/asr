# Objective 1: Figures Catalog & Visual Specifications

> **Path:** `docs/objective_1/figures/`  
> **Count:** 8 Publication-Grade Vector SVG Figures  
> **Generation Script:** `scripts/objective_1_docs/generate_figures.py`  
> **Format:** Pure Vector SVG with embedded typography, dark-theme styling, and responsive `viewBox` coordinates. Zero binary dependencies.

---

## Figure Gallery & Technical Specifications

### Figure 1: End-to-End Real-Time Vietnamese KWS System Architecture
- **File:** [`figure_01_end_to_end_system_architecture.svg`](./figure_01_end_to_end_system_architecture.svg)
- **Scope:** Complete signal and decision pipeline from microphone audio capture (16 kHz PCM), framing/Mel-spectrogram (40 filterbanks, 10 ms hop), 1D temporal residual backbone (`VietnameseTCResNet`, 272,934 parameters), sliding circular ring buffer, EMA posterior smoothing ($\alpha = 0.60$), dual-threshold gating ($\tau = 0.70$, margin $= 0.15$), and refractory lockout timer (600 ms). Documents single-frame CPU latency of 0.35 ms and end-to-end sliding trigger latency of 706 ms.
- **Dimensions:** 1200 &times; 680 px

### Figure 2: Dataset Acquisition, Provenance Tracking & Ingestion Lineage
- **File:** [`figure_02_dataset_acquisition_provenance_flow.svg`](./figure_02_dataset_acquisition_provenance_flow.svg)
- **Scope:** Complete data lineage tracking. Documents approved model training corpora: 1,000 synthetic calibration commands + 579 Google FLEURS vi CC-BY-4.0 distractors + 120 calibration confusing words + 400 Husein CC-BY-4.0 mic noise samples (Total: 2,099 approved files). Discloses rejected/quarantined sources (VIVOS rejected, VietSpeech quarantined, Common Voice with 0 approved samples). Details field campaign track with 0 genuine human takes and 58 physical test fixtures quarantined.
- **Dimensions:** 1200 &times; 700 px

### Figure 3: Dataset Class Distribution Partitioned by Split (V2)
- **File:** [`figure_03_class_distribution_by_split.svg`](./figure_03_class_distribution_by_split.svg)
- **Scope:** Grouped bar chart comparing the allocation of all 6 classes across Train (1,281 samples: 178/178/178/178/319/250), Validation (284 samples: 36/36/36/36/80/60), and Test (534 samples: 36/36/36/36/300/90) splits. Accurately reflects true manifest counts with 36 test samples per command, 300 UNKNOWN, and 90 SILENCE. Total volume: 2,099 approved files.
- **Dimensions:** 1100 &times; 600 px

### Figure 4: Comparative Performance: Baseline 2D-CNN vs. TC-ResNet-14
- **File:** [`figure_04_model_accuracy_f1_trai_comparison.svg`](./figure_04_model_accuracy_f1_trai_comparison.svg)
- **Scope:** Diagnostic performance comparison across five critical dimensions on the frozen test set: Overall Accuracy (Baseline 2D-CNN `VietnameseKWSNet` 97.75% vs TC-ResNet-14 `VietnameseTCResNet` 96.63%), Macro-F1 (0.9466 vs 0.9402), Overall `TRAI` Recall (69.44% vs 75.00%), Northern Dialect `TRAI` Recall (16.67% vs 25.00%), and Parameters (138,534 vs 272,934).
- **Dimensions:** 1150 &times; 620 px

### Figure 5: TC-ResNet-14 Confusion Matrix on Frozen Test Set (N = 534)
- **File:** [`figure_05_confusion_matrix_tcresnet.svg`](./figure_05_confusion_matrix_tcresnet.svg)
- **Scope:** Full 6&times;6 confusion matrix heatmap on the frozen test set (`test_manifest_v2.csv`). Color-coded by accuracy and error severity. Explicitly illustrates the 9 misclassifications of `TRAI` as `PHAI` (27 correct out of 36) stemming from shared rhyme phonetics (`/aːj/`), while documenting strong rejection on `UNKNOWN` (97.67%, 293/300) and `SILENCE` (100.0%, 90/90).
- **Dimensions:** 1000 &times; 680 px

### Figure 6: Continuous Streaming Detection Trade-off Curve
- **File:** [`figure_06_streaming_roc_mcr_latency.svg`](./figure_06_streaming_roc_mcr_latency.svg)
- **Scope:** Operating curves plotting Command Detection Rate (%) against False Alarms per Hour (FA/hr) across decision thresholds $\tau \in [0.60, 0.95]$ from the actual 8.0-minute continuous streaming benchmark (0.133 hr: 6.5 min speech + 1.5 min noise). Illustrates the operating point ($\tau = 0.70$, 117/144 TP, 0 observed FA, Poisson 95% upper bound $\le 22.5$ FA/hr, 706.4 ms latency) alongside streaming engine parameters.
- **Dimensions:** 1150 &times; 620 px

### Figure 7: Human Field-Recording Campaign Audit & Data Gap Analysis
- **File:** [`figure_07_field_campaign_target_progress.svg`](./figure_07_field_campaign_target_progress.svg)
- **Scope:** Transparent gap audit contrasting Planned Minimum Quotas (6,000 recordings, 100+ speakers, 50% North / 25% Central / 25% South) against Audited Genuine Human Speech (0 recordings, 0 speakers, 0.0% completion). Explicitly audits the 58 test-harness files on disk (82 manifest records across two test fixtures) to prevent false progress claims.
- **Dimensions:** 1150 &times; 640 px

### Figure 8: On-Device Deployment, Quantization & Numerical Parity Flow
- **File:** [`figure_08_deployment_parity_pipeline.svg`](./figure_08_deployment_parity_pipeline.svg)
- **Scope:** Architecture and verification flow for edge deployment: PyTorch Golden Checkpoint $\to$ ONNX FP32 Export $\to$ INT8 Dynamic Quantization $\to$ ONNX Runtime Mobile & CoreML. Documents verified parity tolerance (Baseline FP32: $9.54\times 10^{-7}$, TC-ResNet FP32: $1.19\times 10^{-6}$, 100% class match) and flags the missing `vietnamese_kws_tcresnet_int8.onnx` file.
- **Dimensions:** 1150 &times; 640 px

---

## Reproduction & Verification

To re-generate and refresh all 8 vector SVG figures:

```powershell
python scripts/objective_1_docs/generate_figures.py
```
