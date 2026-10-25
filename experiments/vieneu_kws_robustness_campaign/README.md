# Phase 30–32: Integrated Robustness, Confidence Calibration, and Streaming Evaluation Campaign

**Directory:** `experiments/vieneu_kws_robustness_campaign/`  
**Dataset Reference:** `experiments/synthetic_kws_dataset_vieneu_6class/`  
**Model Checkpoints:** `experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed{42,43,44}.pt`  
**Governance Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Directory Structure

```text
experiments/vieneu_kws_robustness_campaign/
├── config.json                               # Master configuration for Phases 30-32
├── noise_generator.py                        # Deterministic noise corruption & SNR engine
├── evaluate_robustness.py                    # Phase 30 evaluation across 4 noises x 5 SNRs
├── confidence_policy.py                      # Phase 31 post-processing confidence rejection function
├── calibrate_threshold.py                    # Phase 31 validation-only grid search & test comparison
├── streaming_simulator.py                    # Phase 32 sliding-window stream simulation engine
├── babble_sources/
│   └── synthetic_vietnamese_babble_30s.wav   # 30-second multi-talker Vietnamese babble pool
├── audio_corrupted/                          # Deterministic corrupted WAV files (val & test)
├── corrupted_val_manifest.csv                # 4,800 validation corrupted samples manifest
├── corrupted_test_manifest.csv               # 4,800 test corrupted samples manifest
├── calibration/
│   ├── calibration_seed{42,43,44}.json       # Full validation grid search curves per seed
│   ├── selected_thresholds.json              # Frozen operating thresholds per seed
│   └── test_calibration_comparison.json     # Test split metrics before vs after calibration
├── streaming_simulation/
│   ├── continuous_command_stream.wav         # 11.1 min continuous command stream
│   ├── command_stream_ground_truth.csv       # Exact ground-truth event timeline
│   ├── continuous_nontarget_stream.wav       # 20.0 min continuous non-target stream
│   └── nontarget_stream_ground_truth.csv     # Non-target distractor timeline
├── predictions/
│   ├── robustness_results.json               # Full tabular metrics for all 21 conditions x 3 seeds
│   ├── test_robustness_predictions_seed*.csv # Per-sample predictions across all noise conditions
│   ├── calibrated_test_predictions_seed*.csv # Per-sample calibrated predictions on held-out test
│   └── streaming_results.json                # Continuous streaming simulation metrics
└── reports/
    ├── ROBUSTNESS_AND_CALIBRATION_REPORT.md  # Comprehensive Phase 30 & 31 report
    ├── STREAMING_EVALUATION_REPORT.md        # Comprehensive Phase 32 streaming report
    └── RECOMMENDED_NEXT_EXPERIMENT.md        # Post-Phase 32 prioritized roadmap
```

---

## 2. Quick Reproduction Guide

All commands run under `.venv` from the workspace root (`D:\trainidentificationaudio`):

```powershell
# 1. Generate corrupted audio datasets (4 noises x 5 SNRs)
.\.venv\Scripts\python.exe experiments/vieneu_kws_robustness_campaign/noise_generator.py

# 2. Run Phase 30 robustness benchmark
.\.venv\Scripts\python.exe experiments/vieneu_kws_robustness_campaign/evaluate_robustness.py

# 3. Run Phase 31 confidence threshold calibration
.\.venv\Scripts\python.exe experiments/vieneu_kws_robustness_campaign/calibrate_threshold.py

# 4. Run Phase 32 continuous streaming simulation
.\.venv\Scripts\python.exe experiments/vieneu_kws_robustness_campaign/streaming_simulator.py

# 5. Run automated test suite
.\.venv\Scripts\pytest.exe tests/test_vieneu_robustness_campaign.py -v
```

---

## 3. Key Findings Summary

| Evaluation Dimension | Uncalibrated Baseline | Calibrated Policy | Impact / Takeaway |
| :--- | :---: | :---: | :--- |
| **Clean Test Accuracy** | 92.08% ± 0.68% | **93.33% ± 0.83%** | +1.25% improvement |
| **Clean Test Macro-F1** | 91.36% ± 0.91% | **93.02% ± 0.98%** | +1.66% improvement |
| **UNKNOWN False Activation** | 45.00% ± 6.12% | **33.33% ± 5.77%** | -11.67% false alarms suppressed |
| **Target Command Recall** | 99.38% ± 0.62% | **98.33% ± 0.72%** | High recall preserved |
| **HVAC Noise (0 dB SNR)** | 72.08% ± 10.38% | --- | Formant preservation retains functionality |
| **White/Pink Noise ($\le 15$ dB)** | 16.8% – 31.9% | --- | Catastrophic collapse; requires augmentation |
| **Babble Noise (15 dB SNR)** | 57.50% ± 7.17% | --- | Competitive formants degrade classification |
| **Streaming Command Recall** | 98.1% – 100.0% | **97.5% – 98.8%** | Robust command capture in continuous audio |
| **Streaming Mean Latency** | 428.6 – 449.3 ms | **458.2 – 471.2 ms** | Acceptable interactive UI delay (< 500 ms) |
| **Non-Target Stream FA/hr** | 230 – 1,274 FA/hr | **155 – 1,190 FA/hr** | Calibration reduces FA/hr; requires 2nd stage |
