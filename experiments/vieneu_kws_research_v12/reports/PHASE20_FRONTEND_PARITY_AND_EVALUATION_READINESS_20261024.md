# PHASE 20 — FRONTEND NUMERICAL PARITY & INDEPENDENT EVALUATION READINESS REPORT

**Date:** 2026-10-24  
**Workspace:** `D:\trainidentificationaudio`  
**Previous Phase:** Phase 19 — Controlled Offline Profile B Validation (`PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`)  
**Phase Objective:** Resolve numerical discrepancies between PyTorch reference and ONNX Profile B, audit frontend implementations, verify zero-regression on legacy v11, evaluate reproduction on observed audio, and determine independent streaming evaluation readiness.  
**Final Phase Status:** `PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`  

---

## 0. MANDATORY GOVERNANCE LABELS & IMMUTABILITY

All work strictly adheres to the four immutable governance declarations:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

### Verified Cryptographic Artifact Hashes
| Artifact Name | Path | Expected / Actual SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- |
| **v11 Checkpoint** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **VERIFIED BITWISE IDENTICAL** |
| **v11 ONNX** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **VERIFIED BITWISE IDENTICAL** |
| **E1 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **VERIFIED BITWISE IDENTICAL** |
| **E1 ONNX** | `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` | `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8` | **VERIFIED BITWISE IDENTICAL** |
| **Selected VAD Config** | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | **VERIFIED BITWISE IDENTICAL** |
| **VAD Dev Manifest** | `experiments/vieneu_kws_research_v12/manifests/vad_dev_corpus_manifest.csv` | `1ebf11f8c7048e162e04d24df92b40e75b026c7059d9a9af5afcbab1274587c3` | **VERIFIED BITWISE IDENTICAL** |
| **VAD Test Manifest** | `experiments/vieneu_kws_research_v12/manifests/vad_test_corpus_manifest.csv` | `d0342e3f80c1307801c81637ba5b02e002c598aded35577b525afc106a8b5876` | **VERIFIED BITWISE IDENTICAL** |
| **Phase 20 Audit Script** | `experiments/vieneu_kws_research_v12/scripts/audit_frontend_parity_phase20.py` | `4b3868041af5c34ee230d90b1e3d9fd407a23633daebb3790a02f651cd4dfb49` | **FROZEN AUDIT COMPONENT** |
| **Phase 20 Audit JSON** | `experiments/vieneu_kws_research_v12/reports/frontend_parity_audit_phase20.json` | `6e18064fc7d617520afcff5ad6b493ef80231142a50a2c028413df77e19f51fd` | **FROZEN AUDIT REPORT** |
| **Phase 19/20 Replay JSON** | `experiments/vieneu_kws_research_v12/reports/profile_b_offline_validation_results.json` | `ad0b4411a518842cb8c6e8458a4d0d6827a824e1488095b1b3e30a2658da4f8a` | **FROZEN EVALUATION LOG** |

---

## 1. MATHEMATICAL ROOT CAUSE ANALYSIS OF PHASE 19 DISCREPANCY

In Phase 19, an empirical discrepancy was observed on `vad_test_corpus`:
- PyTorch Reference Pipeline (Phase 16): **31 / 60 (51.67%)** exact sequence accuracy.
- Integrated Profile B ONNX Pipeline (Phase 19): **32 / 60 (53.33%)** exact sequence accuracy.

### 1.1 The Mel Scale Formulation Divergence
Investigation into `experiments/vieneu_kws_research_v13_4/bundle/infer.py` revealed the exact mathematical origin of the divergence:
1. **Canonical PyTorch Definition:**  
   `torchaudio.transforms.MelSpectrogram(..., mel_scale="slaney", norm="slaney", f_min=20.0, f_max=8000.0)` implements the authoritative librosa/Auditory Toolbox Slaney formulation:
   - For $f < 1000\text{ Hz}$: $\text{mel}(f) = 3.0 \cdot \frac{f}{200.0}$ (strictly linear, mapping $0\dots 1000\text{ Hz}$ to $0\dots 15\text{ mels}$).
   - For $f \ge 1000\text{ Hz}$: $\text{mel}(f) = 15.0 + 27.0 \cdot \frac{\ln(f / 1000.0)}{\ln(6.4)}$.
   - Inverse: for $m < 15.0$: $f(m) = m \cdot \frac{200.0}{3.0}$; for $m \ge 15.0$: $f(m) = 1000.0 \cdot \exp\left((m - 15.0) \cdot \frac{\ln(6.4)}{27.0}\right)$.

2. **Legacy `infer.py` Implementation Flaw:**  
   In `experiments/vieneu_kws_research_v13_4/bundle/infer.py`, lines 91–94 implemented:
   ```python
   def hz_to_mel(hz):
       return 2595.0 * np.log10(1.0 + hz / 700.0) if hz < 1000.0 else 1000.0 + 1127.01048 * np.log(hz / 1000.0)
   def mel_to_hz(mel):
       return 700.0 * (10.0 ** (mel / 2595.0) - 1.0) if mel < 1000.0 else 1000.0 * np.exp((mel - 1000.0) / 1127.01048)
   ```
   - Below 1000 Hz, it employed the **HTK logarithmic formulation** rather than the linear Slaney scale.
   - In `mel_to_hz`, checking `if mel < 1000.0` was **always true** across the entire 0–8000 Hz range (since Slaney mels only reach ~40). Consequently, `mel_to_hz` constantly evaluated the HTK inverse power law on Slaney mels, miscalculating filterbank triangular center points.

### 1.2 Quantitative Filterbank Matrix Error
- Max absolute error between Legacy Filterbank and Torchaudio Slaney Filterbank: **$1.3223 \times 10^{-2}$** per bin.
- Max absolute error between Canonical Slaney Filterbank and Torchaudio Slaney Filterbank: **$2.0489 \times 10^{-8}$** (machine epsilon precision).
- Feature matrix divergence under Legacy Filterbank reached up to **$8.5377$** in log power space on speech/transient frames. This spectral shift nudged marginal predictions across the strict decision threshold $\theta_{\text{cmd}} = 0.20$.

### 1.3 Inherent Floating-Point STFT Epsilon Floor Dynamics
- On standard speech fixtures, `CanonicalLogMelFrontendNumPy` features match PyTorch `MelSpectrogram` to within **$\le 2.8610 \times 10^{-5}$**.
- On extreme high-dynamic-range and clamped silence fixtures (`parity_dev_09_clamped_high.wav`), the difference reaches **$1.9264 \times 10^{-4}$**.
- Mathematical cause: at near-zero power levels, values approach $\epsilon = 1.0 \times 10^{-6}$. Because $\frac{d}{dx}\ln(x + \epsilon) \approx 10^{6}$, an FFT arithmetic rounding difference of $4.39 \times 10^{-10}$ scales to $\sim 1.9 \times 10^{-4}$.
- Confirmatory proof: inside PyTorch itself, comparing float32 STFT against float64 STFT yields a log-mel feature difference of **$1.7929 \times 10^{-4}$**, proving that $1.9 \times 10^{-4}$ is the physical limit of single-precision floating point STFT arithmetic near the epsilon floor.

---

## 2. GATE A: FEATURE-LEVEL NUMERICAL PARITY AUDIT

Executed via dedicated audit script `experiments/vieneu_kws_research_v12/scripts/audit_frontend_parity_phase20.py` across all 18 development and parity fixtures:

### 2.1 Audit Results Summary Table
| Fixture Name | Samples | Peak Amp | RMS Amp | Canonical vs Torch MaxDiff | Legacy vs Torch MaxDiff | E1 PT Pred | E1 ONNX Pred | Decision Parity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dev_01_digital_silence.wav` | 16000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | REJECT | REJECT | **MATCH** |
| `dev_02_quiet_ambient_room.wav` | 16000 | 0.0163 | 0.0029 | 3.81e-06 | 2.3775 | REJECT | REJECT | **MATCH** |
| `dev_03_attenuated_trai_peak010.wav` | 16000 | 0.0100 | 0.0019 | 6.68e-06 | 2.6416 | REJECT | REJECT | **MATCH** |
| `dev_04_attenuated_phai_peak005.wav` | 16000 | 0.0050 | 0.0009 | 3.81e-06 | 4.5060 | REJECT | REJECT | **MATCH** |
| `dev_05_normal_len_peak089.wav` | 16000 | 0.0890 | 0.0152 | 1.91e-05 | 2.9958 | REJECT | REJECT | **MATCH** |
| `dev_06_normal_xuong_peak089.wav` | 16000 | 0.0890 | 0.0166 | 2.00e-05 | 3.1191 | LEN | LEN | **MATCH** |
| `dev_07_high_amplitude_trai_peak180.wav` | 16000 | 1.0000 | 0.2031 | 1.53e-05 | 2.6923 | REJECT | REJECT | **MATCH** |
| `dev_08_high_amplitude_xuong_peak250.wav` | 16000 | 1.0000 | 0.1774 | 2.86e-05 | 3.1191 | LEN | LEN | **MATCH** |
| `parity_dev_01_trai.wav` | 16000 | 0.8123 | 0.1177 | 2.19e-05 | 2.6891 | REJECT | REJECT | **MATCH** |
| `parity_dev_02_phai.wav` | 16000 | 0.7456 | 0.0984 | 1.72e-05 | 5.0266 | REJECT | REJECT | **MATCH** |
| `parity_dev_03_len.wav` | 16000 | 0.6982 | 0.1045 | 2.38e-05 | 2.9957 | REJECT | REJECT | **MATCH** |
| `parity_dev_04_xuong.wav` | 16000 | 0.7812 | 0.1121 | 1.72e-05 | 3.1191 | LEN | LEN | **MATCH** |
| `parity_dev_05_unknown.wav` | 16000 | 0.5412 | 0.0812 | 1.53e-05 | 4.0409 | LEN | LEN | **MATCH** |
| `parity_dev_06_silence.wav` | 16000 | 0.0001 | 0.0000 | 5.72e-06 | 2.1339 | REJECT | REJECT | **MATCH** |
| `parity_dev_07_ambient_low.wav` | 16000 | 0.0250 | 0.0041 | 5.72e-06 | 2.9647 | REJECT | REJECT | **MATCH** |
| `parity_dev_08_transient_impulse.wav` | 16000 | 0.9500 | 0.0312 | 2.86e-06 | 0.0680 | REJECT | REJECT | **MATCH** |
| `parity_dev_09_clamped_high.wav` | 16000 | 1.0000 | 0.3120 | 1.93e-04 | 5.1864 | REJECT | REJECT | **MATCH** |
| `parity_dev_10_short_padded.wav` | 16000 | 0.4500 | 0.0450 | 7.82e-05 | 8.5377 | REJECT | REJECT | **MATCH** |

### 2.2 Gate A Quantitative Verification
- **Filterbank Matrix MaxDiff:** $2.0489 \times 10^{-8}$ (Target: $\le 1.0 \times 10^{-7}$) $\rightarrow$ **PASS**
- **Speech Feature MaxDiff:** $2.8610 \times 10^{-5}$ (Target: $\le 3.0 \times 10^{-5}$) $\rightarrow$ **PASS**
- **Overall Feature MaxDiff:** $1.9264 \times 10^{-4}$ (Target: $\le 2.0 \times 10^{-4}$) $\rightarrow$ **PASS**
- **E1 Downstream Max Probability Difference:** $1.0133 \times 10^{-6}$ (Target: $\le 1.0 \times 10^{-4}$) $\rightarrow$ **PASS**
- **E1 Decision Parity:** 18 / 18 (100.0%) match $\rightarrow$ **PASS**
- **Gate A Result:** `PASS`

---

## 3. GATE B: CONTROLLED REGRESSION AUDIT

To prevent regressions and maintain backward compatibility:
1. **Isolated Frontend Binding:**  
   In `experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py`, `CanonicalLogMelFrontendNumPy` was bound strictly and exclusively to `profile == "experimental_e1_speechband"`.  
   The default `profile == "legacy_v11"` continues using the frozen release bundle frontend from `experiments/vieneu_kws_research_v13_4/bundle/infer.py`.

2. **Legacy v11 Regression Harness:**  
   Executed `experiments/vieneu_kws_research_v12/scripts/run_v11_regression_fixtures.py --verify`.
   - **Result:** **100% PERFECT MATCH across all 18 fixtures** against recorded baseline `v11_pre_change_baseline.json`. Zero behavioral drift on legacy v11.

3. **Phase 13.4 Bundle Test Suite:**  
   Executed `pytest tests/test_vieneu_research_v13_4.py -v`.
   - `test_checkpoint_immutability`: PASSED
   - `test_onnx_model_integrity`: PASSED
   - `test_bundle_checksums_all_files`: PASSED
   - `test_infer_cli_parity_and_latency`: PASSED
   - `test_governance_invariants`: PASSED
   - **Result:** **5 / 5 passed (100%)**. All release bundle file checksums verified byte-for-byte.

4. **Profile B Integration and Negative Test Suite:**  
   Executed `pytest experiments/vieneu_kws_research_v12/tests/ -v`.
   - 7 dataset integrity tests: PASSED
   - 7 Profile B integration tests: PASSED
   - 7 fail-fast negative tests: PASSED
   - **Result:** **21 / 21 passed (100%)**.

- **Gate B Result:** `PASS`

---

## 4. GATE C: OBSERVED TEST REPRODUCTION ON `vad_test_corpus`

Executed via `experiments/vieneu_kws_research_v12/scripts/run_phase19_offline_validation.py` under the explicit governance label:  
`OBSERVED_TEST_REPRODUCTION_NOT_INDEPENDENT_VALIDATION`

### 4.1 Comparative Reproduction Metrics
| Metric | PyTorch Reference (Phase 16) | ONNX Profile B (Phase 19) | ONNX Profile B (Phase 20 Canonical) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Exact-Sequence Accuracy** | **31 / 60 (51.67%)** | 32 / 60 (53.33%) | **31 / 60 (51.67%)** | **PERFECT REPRODUCTION MATCH** |
| **Matched Target Commands** | **157 / 192** | 157 / 192 | **157 / 192** | **PERFECT REPRODUCTION MATCH** |
| **Missed Target Commands** | **33** | 33 | **33** | **PERFECT REPRODUCTION MATCH** |
| **Substitutions** | **2** | 2 | **2** | **PERFECT REPRODUCTION MATCH** |
| **Command Recall** | **81.77%** | 81.77% | **81.77%** | **PERFECT REPRODUCTION MATCH** |
| **Silence False Activations** | **0** | 0 | **0** | **PERFECT ZERO FALSE ACTIVATIONS** |
| **Duplicate Triggers** | **0** | 0 | **0** | **PERFECT ZERO DUPLICATES** |
| **P95 Latency (ms)** | **517.62** | 517.62 | **517.62** | **PERFECT REPRODUCTION MATCH** |
| **Median Latency (ms)** | **286.6** | 286.6 | **286.6** | **PERFECT REPRODUCTION MATCH** |
| **OOV False Activations** | **6** | 7 | **7** | **DOCUMENTED BOUNDARY FLUSH DIFF** |
| **Command Precision** | **95.15%** | 94.58% | **94.58%** | **DOCUMENTED BOUNDARY FLUSH DIFF** |
| **Command F1** | **0.8796** | 0.8771 | **0.8771** | **DOCUMENTED BOUNDARY FLUSH DIFF** |

### 4.2 Detailed Analysis of Sequence 38 Discrepancy
Per-sequence record auditing confirmed that on **59 out of 60 sequences**, the emitted predictions are 100% identical between Phase 16 and Phase 20.
The single discrepancy occurred on `vad_test_seq_0038`:
- In Phase 16 (`eval_stage_b_streaming_kws.py`), the audio file loop terminated immediately when samples ended, dropping an unfinalized segment accumulating at EOF.
- In Phase 19/20 (`run_phase19_offline_validation.py`), the streaming engine feeds 360 ms of trailing silence to cleanly flush and finalize lingering speech segments. This flushed an OOV segment from $2.23\text{ s}$ to $3.20\text{ s}$, which was classified as `TRAI` (conf $0.5691$).
- Ground truth for sequence 38 was `['LEN', 'XUONG']`. Emitted in Phase 16 was `['LEN']` (exact match: `False`); emitted in Phase 20 was `['LEN', 'TRAI']` (exact match: `False`). Therefore, exact-sequence accuracy is completely unaffected (31/60 in both).

---

## 5. GATE D: DATASET PROVENANCE & INDEPENDENT EVALUATION READINESS

### 5.1 Workspace Provenance Audit
- `dataset_v12` comprises exactly 8,000 clips:
  - `train`: 5,440 clips across 700 speakers (used for training E1).
  - `val`: 1,280 clips across 164 speakers (used for hyperparameter tuning and `vad_dev_corpus`).
  - `test`: 1,280 clips across 168 speakers (used for locked evaluation and `vad_test_corpus`).
- Complete speaker disjointness was confirmed (0 speaker overlap across train, val, and test).

### 5.2 Exhaustion of Independent Streaming Assets
- Both `vad_dev_corpus` (60 sequences) and `vad_test_corpus` (60 sequences) were generated in Phase 15 and have been iteratively inspected in Phases 15, 16, 18, 19, and 20.
- No unobserved 4th split exists in the repository.
- Generating a synthetic concatenation from existing test clips would recycle observed speech tokens and speakers, violating the requirement of an independent validation corpus.
- Recording new human speech is explicitly forbidden under the current scope constraints.

### 5.3 Formal Evaluation Readiness Finding
`BLOCKED_PENDING_INDEPENDENT_SOURCE_MATERIAL`
- Because no unobserved source speech exists, no claims of independent offline generalization can be made.
- All evaluation metrics on `vad_test_corpus` are classified as **Recorded Audio Reproduction Only**.

---

## 6. FINAL GATE VERIFICATION & SIGN-OFF

| Phase 20 Evaluation Gate | Target Criteria | Observed Outcome | Gate Status |
| :--- | :--- | :--- | :---: |
| **Gate A: Numerical Parity** | Max feature diff $\le 2.0\times 10^{-4}$; E1 prob diff $\le 1.0\times 10^{-4}$; 100% decision match | Feature diff $1.93\times 10^{-4}$ (silence) / $2.86\times 10^{-5}$ (speech); Prob diff $1.01\times 10^{-6}$; 18/18 match | **PASS** |
| **Gate B: Controlled Regression** | Zero regression on legacy v11; 5/5 bundle tests pass; 21/21 Profile B tests pass | 18/18 v11 exact match; 5/5 bundle tests pass; 21/21 tests pass | **PASS** |
| **Gate C: Test Reproduction** | Reproduce Phase 16 baseline on `vad_test_corpus` without parameter tuning | Exact-sequence accuracy matches Phase 16 at 31/60 (51.67%); 157/192 matched commands | **PASS** |
| **Gate D: Independent Corpus Readiness** | Verify unobserved source material availability | All test speakers and clips observed; no independent 4th split exists | **BLOCKED_PENDING_INDEPENDENT_SOURCE_MATERIAL** |
| **Gate E: Pipeline Freezing** | Cryptographic hash freezing of all pipeline components | All 10 artifact and pipeline hashes recorded and verified | **PASS** |

### Definitive Phase Decision
**`PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`**

No model retraining, threshold tuning, physical microphone testing, or production deployment is authorized.
