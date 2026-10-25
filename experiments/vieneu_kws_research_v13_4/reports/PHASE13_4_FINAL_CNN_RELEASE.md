# Phase 13.4: Final Vietnamese Bounded-Utterance CNN Release Candidate Report

**Release Status:** `FROZEN_CNN_RELEASE_CANDIDATE`  
**Execution Date:** 2026-10-19  
**Candidate Identifier:** `variant_v11_hard_negative_weighted_seed42`  
**Candidate Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**ONNX Export SHA-256:** `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`  
**Calibrated Operating Threshold ($\theta^*$):** `0.20`  
**Authoritative Operational State:** `FROZEN_CNN_RELEASE_CANDIDATE` | `NOT_VALIDATED_ON_NATURAL_SPEECH` | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`

---

## 1. Executive Summary

Following the successful execution and passage of all 8 preregistered synthetic confirmatory criteria in Phase 13.3, Phase 13.4 finalizes the frozen Vietnamese bounded-utterance CNN candidate into a self-contained, reproducible release bundle located at [`experiments/vieneu_kws_research_v13_4/bundle/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/).

The candidate model (`CorrectedVietnameseKWSNet`, 135,813 parameters) spots four Vietnamese directional commands (`TRAI` [trái], `PHAI` [phải], `LEN` [lên], `XUONG` [xuống]) with a non-target rejection class (`REJECT`).

### Key Release Highlights
1. **Unbroken Checkpoint Integrity:** The candidate checkpoint SHA-256 was audited pre- and post-release, remaining byte-identical: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`.
2. **Phase 13.3 Confirmatory Verification:** The actual Phase 13.3 metrics file (`confirmatory_evaluation_metrics.json`) was audited, confirming:
   - Overall Accuracy: **99.20% (992/1000)** (95% CI: 98.4% – 99.6%)
   - Aggregate Command Recall: **99.00% (198/200)** (95% CI: 96.4% – 99.7%)
   - Per-Command Recall: `TRAI` 49/50 (98.0%), `PHAI` 50/50 (100.0%), `LEN` 49/50 (98.0%), `XUONG` 50/50 (100.0%)
   - Non-Target False Alarm Rate: **0.75% (6/800)** (95% CI: 0.3% – 1.6%)
   - Single-Clip CPU Latency: Mean **1.80 ms**, p95 **2.12 ms**.
3. **Native ONNX Export with High Numerical Parity:** Exported directly from the frozen v11 checkpoint (not legacy v10 artifacts) to standard FP32 ONNX (`vietnamese_kws_v11.onnx`, SHA-256: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`). Maximum absolute logit difference vs PyTorch on development fixtures is **$3.58 \times 10^{-6}$** (well within the $\le 10^{-4}$ tolerance) with 100% decision parity.
4. **Benchmarked CPU Latency:** Single-clip latency averaged **1.39 ms** (PyTorch) and **2.27 ms** (ONNX Runtime), satisfying the $\le 50.0\text{ ms}$ real-time SLA by more than an order of magnitude.
5. **Acceptance Label:** **`FROZEN_CNN_RELEASE_CANDIDATE`**.

---

## 2. Release Bundle Contents & Checksum Manifest

The versioned release candidate bundle in [`experiments/vieneu_kws_research_v13_4/bundle/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/) comprises:

| File Name | SHA-256 Hash | Purpose / Content |
| :--- | :--- | :--- |
| `checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | Frozen PyTorch model weights |
| `onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | FP32 ONNX model (opset 17) |
| `onnx/export_parity_manifest.json` | `d6be9f8f0e6d9c832b8d0890e614874c298e6d358da19a0ddc670c45ffe2db80` | Verification report of PyTorch vs ONNX parity |
| `infer.py` | `e3d651acfff2aa395b5d1ada92d60e00f2dea733b151977b4b3eaeac2149f3a1` | Standalone CLI for single & batch WAV inference |
| `benchmark_latency.py` | `d2c6e5d4599fdb3c5a5544980984a5b9cf82ac88dde5f4845ad32232c27d7f0c` | Benchmarking utility for CPU latency |
| `latency_benchmark_results.json` | `6b802cead1e246db63c66da7cbe7b39f729503d9003735f6c0ae7df3439be10a` | Benchmarked latency statistics |
| `model_config.json` | `c23f64424be720a4839cbe37dfffeb7ebf47c348807a93703aa569165188fffc` | Model architecture, layers, and class metadata |
| `preprocessing_contract.json` | `ca577646d4d54e03e8623bccb594339fb0d10c15edad3ce3ecd253462d01296e` | Preprocessing, STFT, and Mel scale contract |
| `requirements.txt` | `a15fc04b7945396e1df0e6b7cf572b8d95932b2fa1ca679ae04944e13326b89e` | Minimal runtime Python dependencies |
| `environment_lock.json` | `c2eedfc9a78cc996f89a155f03b430f98395360fb4d7bf41300d61bd893c3948` | System and dependency version lock |
| `MODEL_CARD.md` | `ea81aaaf0a7f9c616f3c3fcb5e2d649a673fc46975dd7887dcb034b5bad4081e` | Full model card and operational boundaries |
| `README.md` | `0b26c6101d83150b2ef26b1cd071049110e354b9d53eda0001bf1776e6bc44fa` | Step-by-step reproduction and inference guide |
| `checksums.json` | *(Bundle Manifest)* | SHA-256 index of all bundle files |

---

## 3. ONNX Export and Numerical Parity Verification

The model was exported from the frozen v11 checkpoint using `torch.onnx.export` (opset 17, dynamic batch dimension) directly into [`bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx).

Numerical parity was validated across non-locked development fixtures (`experiments/vieneu_kws_research_v13_1/fixtures/`):

| Fixture Name | Logits Max Abs Diff | Probs Max Abs Diff | PyTorch Prediction | ONNX Prediction | Parity Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dev_01_digital_silence.wav` | $3.58 \times 10^{-6}$ | $8.94 \times 10^{-7}$ | `REJECT` | `REJECT` | **PASS** |
| `dev_02_quiet_ambient_room.wav` | $7.15 \times 10^{-7}$ | $4.66 \times 10^{-9}$ | `REJECT` | `REJECT` | **PASS** |
| `dev_03_attenuated_trai_peak010.wav` | $1.67 \times 10^{-6}$ | $2.98 \times 10^{-7}$ | `REJECT` | `REJECT` | **PASS** |
| `dev_04_attenuated_phai_peak005.wav` | $1.67 \times 10^{-6}$ | $2.38 \times 10^{-7}$ | `REJECT` | `REJECT` | **PASS** |
| `dev_05_normal_len_peak089.wav` | $1.88 \times 10^{-6}$ | $4.77 \times 10^{-7}$ | `XUONG` | `XUONG` | **PASS** |
| `dev_06_normal_xuong_peak089.wav` | $1.49 \times 10^{-6}$ | $3.58 \times 10^{-7}$ | `XUONG` | `XUONG` | **PASS** |
| `dev_07_high_amplitude_trai_peak180.wav` | $1.59 \times 10^{-6}$ | $3.43 \times 10^{-7}$ | `LEN` | `LEN` | **PASS** |
| `dev_08_high_amplitude_xuong_peak250.wav` | $1.91 \times 10^{-6}$ | $5.07 \times 10^{-7}$ | `XUONG` | `XUONG` | **PASS** |

All fixtures satisfied numerical tolerance ($\le 10^{-4}$) with **100% decision parity**.

---

## 4. CPU Latency Benchmark

Single-clip WAV-to-prediction CPU latency was measured over 10 warm-up runs followed by 100 benchmark iterations on development fixtures:

| Backend | Mean Latency | Median (p50) | 95th Percentile (p95) | Max Latency | SLA Target ($\le 50$ ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (CPU)** | **1.39 ms** | 1.42 ms | **1.51 ms** | 1.64 ms | **PASSED** |
| **ONNX Runtime (CPU)** | **2.27 ms** | 2.39 ms | **2.48 ms** | 2.80 ms | **PASSED** |

---

## 5. Scope, Disclaimers, and Mandatory Operational Boundaries

In strict compliance with project governance:
1. **`FROZEN_CNN_RELEASE_CANDIDATE`**: This candidate has achieved confirmatory validation on unseen carrier phrases across 8 novel Northern-accented synthetic voices under Amendment 13.1-A.
2. **`NOT_VALIDATED_ON_NATURAL_SPEECH`**: The model has **not** undergone natural human speech validation. Acoustic behavior under physical microphone distortion and conversational human speech remains unproven.
3. **Regional Dialect Limitations**: Validation was conducted strictly on Northern Vietnamese (Bắc) voices. Regional generalization across Central (Trung) and Southern (Nam) dialects remains **explicitly out of scope and pending**.
4. **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**: The model operates strictly on isolated 1.0-second pre-segmented audio windows. Continuous streaming open-mic deployment requires streaming VAD, ring-buffering, and keyword boundary modeling not present in this candidate.
5. **Preservation of Past Artifacts**: No historical checkpoints, test datasets, answer keys, review packages, or consent records were altered. `git diff --cached` remains strictly empty.
