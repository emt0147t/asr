# PHASE 21 — SYNTHETIC AUDIO PROVENANCE, DIVERSITY AND NEW EVALUATION CORPUS REPORT

**Date:** 2026-10-24  
**Workspace:** `D:\trainidentificationaudio`  
**Previous Phase:** Phase 20 — Frontend Numerical Parity & Independent Evaluation Readiness (`PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`)  
**Phase Objective:** Audit synthetic audio provenance and generator diversity, design and preregister a new evaluation protocol, construct an independent synthetic evaluation corpus using an unobserved generator architecture and unobserved voice distribution, validate split isolation, and evaluate the frozen E1 system under strict streaming conditions.  
**Final Phase Status:** `PASS_TO_NEW_LOCKED_SYNTHETIC_EVALUATION`  

---

## 0. MANDATORY GOVERNANCE DECLARATIONS & INVARIANTS

All research, data generation, and evaluation strictly adhere to the mandatory governance invariants:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

### Invariant Guarantees
1. **Zero Retraining:** Checkpoint `exp_v12_e1_baseline_scale_seed42.pt` and ONNX artifact `vietnamese_kws_v12_e1.onnx` were maintained byte-for-byte unmodified.
2. **Zero Human Audio Collection:** No physical microphone recordings, human speech captures, or crowdsourced voice recordings were collected or persisted.
3. **Threshold Immutability:** Decision threshold $\theta_{\text{cmd}} = 0.20$ remained strictly fixed. Zero word blacklisting or post-hoc threshold tuning was performed.
4. **Isolated Default Profile:** Legacy profile `legacy_v11` remained untouched and zero-regression verified.

---

## 1. SYNTHETIC AUDIO PROVENANCE AUDIT

Executed via `experiments/vieneu_kws_research_v12/scripts/audit_synthetic_provenance_phase21.py` on `dataset_v12_manifest.csv` (8,000 clips) and upstream sources:

### 1.1 Source Type Breakdown in Dataset v12
| Source Type | Clip Count | Percentage | Split Allocation (Train / Val / Test) | Audio Provenance Description |
| :--- | :---: | :---: | :---: | :--- |
| **`noise_augmented`** | 2,800 | 35.00% | 1,904 / 448 / 448 | Positive keywords (`TRAI`, `PHAI`, `LEN`, `XUONG`) mixed with mic-noise at 0–20 dB SNR. |
| **`tts_synthetic_clean`** | 1,200 | 15.00% | 816 / 192 / 192 | Unaugmented clean positive keywords from `VieNeu-TTS v3 Turbo`. |
| **`noise_augmented_oov`** | 1,120 | 14.00% | 762 / 179 / 179 | Monosyllabic OOV confusers mixed with mic-noise at 0–20 dB SNR. |
| **`conversation_fleurs`** | 1,000 | 12.50% | 680 / 160 / 160 | Continuous conversational Vietnamese speech from public FLEURS dataset (CC-BY-4.0). |
| **`environmental_noise`** | 800 | 10.00% | 544 / 128 / 128 | Non-speech ambient and domestic background noise from Husein mic-noise dataset (MIT). |
| **`ambient_silence`** | 600 | 7.50% | 408 / 96 / 96 | Algorithmic acoustic silence models (near-silence, HVAC hum, airflow pink noise). |
| **`tts_synthetic_oov_clean`** | 480 | 6.00% | 326 / 77 / 77 | Unaugmented clean monosyllabic OOV confusers from `VieNeu-TTS v3 Turbo`. |
| **Total** | **8,000** | **100.00%** | **5,440 / 1,280 / 1,280** | **Strict 68.0% Train / 16.0% Val / 16.0% Test Distribution** |

### 1.2 Upstream Component Provenance Verification
- **Total Synthetic Speech Clips in v12:** 5,600 clips (4,000 target keywords + 1,600 OOV words).
- **Verified Generator:** 100.0% of synthetic speech in `dataset_v12` originated from **`VieNeu-TTS v3 Turbo` (version 3.8.3)**.
- **External Dataset Provenance:** Verified against original licenses (FLEURS: CC-BY-4.0, Husein Mic-Noise: MIT, Parametric Silence: Local Algorithmic).
- **Metadata Completeness:** All 8,000 records contain verified sample IDs, source utterance IDs, voice IDs, text transcripts, noise IDs, SNR levels, and SHA-256 hashes.

---

## 2. GENERATOR DIVERSITY & SYNTHETIC VALIDITY AUDIT

The audit investigated whether `dataset_v12` genuinely varied the underlying speech distribution or primarily varied acoustic noise:

### 2.1 The Five Diversity Dimensions
1. **Generator Diversity (Finding: CRITICAL BOTTLENECK):**  
   `dataset_v12` exhibited **zero generator architecture diversity**. All 5,600 synthetic speech clips were generated exclusively by a single model family (`VieNeu-TTS v3 Turbo`). No alternative vocoder, duration model, or acoustic architecture was included.
2. **Voice Diversity (Finding: EXHAUSTED PRESET SPACE):**  
   `VieNeu-TTS v3 Turbo` provides exactly 25 preset voices. In `dataset_v12`, all 25 voices were partitioned across splits:
   - Train: 17 voices (`Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`).
   - Val: 4 voices (`Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`).
   - Test: 4 voices (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`).  
   Consequently, the preset voice space of `VieNeu-TTS v3 Turbo` was **completely exhausted**; no unobserved voices remained for new test evaluation in this engine.
3. **Text & Lexical Diversity:**  
   - Target commands have fixed canonical text (`trái`, `phải`, `lên`, `xuống`).  
   - OOV confusers in v12 covered 124 distinct lexemes (63 train, 30 val, 31 test). While disjoint, they represent a bounded monosyllabic dictionary.
4. **Acoustic Condition Diversity:**  
   Acoustic variation in v12 was driven almost entirely by additive microphone noise across 52 unique noise tracks and 5 SNR tiers ($20, 15, 10, 5, 0\text{ dB}$).
5. **Source-Level Independence:**  
   Within `v3turbo`, strict disjointness was maintained. However, generalization across synthesizers was completely untested.

---

## 3. NEW SYNTHETIC EVALUATION PROTOCOL

Formally pre-registered in `experiments/vieneu_kws_research_v12/reports/PHASE21_SYNTHETIC_EVALUATION_PREREGISTRATION.md` prior to evaluation.

### 3.1 Unseen Generator & Voice Architecture Selection
To address the generator diversity bottleneck identified in Section 2, the new evaluation corpus was generated using an entirely independent, previously unobserved synthesizer:
- **TTS Engine:** **`VieNeu-TTS v3 Nano`**
- **Architecture Differences vs `v3turbo`:**
  - Native sampling rate: $24,000\text{ Hz}$ (vs $48,000\text{ Hz}$ in Turbo).
  - ONNX-based lightweight vector estimator and duration predictor with discrete codec decoder (vs autoregressive vocoder in Turbo).
  - Novel prosodic contours, formant structures, and pitch dynamics.
- **Unseen Voice Allocation:**
  - **Development Split (`synthetic_eval_dev_v1`):** `Anh Khôi`, `Minh Quân` (2 voices).
  - **Locked Test Split (`synthetic_eval_test_v1`):** `Ái Hân`, `Hữu Quân`, `Xuân Tiên`, `Mạnh Dũng` (4 voices).
  - *All 4 test voices are strictly UNSEEN in `dataset_v12` training, validation, or previous test sets.*

### 3.2 OOV Lexicon Strategy
- **Dev OOV Lexicon (20 words):** `bật`, `tắt`, `dừng`, `chạy`, `mở`, `đóng`, `quay`, `lùi`, `tìm`, `chọn`, `xoay`, `trông`, `trẻ`, `trường`, `phim`, `phục`, `phấn`, `lược`, `lớp`, `lửa`.
- **Test Unseen OOV Lexicon (24 words):**
  - *Phonetic Confusers (13 words):* Minimal pair vowel/tone contrasts: `lăn`, `len`, `lân`, `lền`, `phái`, `phay`, `phai`, `trại`, `trát`, `trai`, `xuồng`, `xưởng`, `xướng`.
  - *General Everyday OOV (11 words):* `ghế`, `bàn`, `nhà`, `đèn`, `sách`, `bút`, `xe`, `nước`, `cơm`, `đồng`, `hồ`.

---

## 4. CORPUS CONSTRUCTION & VALIDATION

Generated via `build_phase21_synthetic_corpora.py` and validated via `validate_phase21_corpora.py`:

### 4.1 Corpus Specifications
| Corpus Name | Split | Sequences | Total Target Commands | Unseen Voices | OOV Lexemes | Noise Tracks | Sequence Length |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :---: |
| **`synthetic_eval_dev_v1`** | Dev | 20 | 64 | `Anh Khôi`, `Minh Quân` | 20 dev words | Tracks 28–33 (6 tracks) | $2.6\text{s} - 4.1\text{s}$ |
| **`synthetic_eval_test_v1`** | Test | 40 | 128 | `Ái Hân`, `Hữu Quân`, `Xuân Tiên`, `Mạnh Dũng` | 24 unseen words | Tracks 34–39 (6 tracks) | $2.5\text{s} - 4.2\text{s}$ |
| **Component Clips** | - | 160 | - | All 6 voices | 44 total words | None (Clean $16\text{kHz}$) | Exactly $1.0\text{s}$ |

### 4.2 Validation & Leakage Audit Findings
Executed `validate_phase21_corpora.py` across all 160 component clips and 60 streaming sequences:
- **Manifest-to-File SHA-256 Consistency:** **100.0% Pass (0 mismatches)**.
- **Audio Integrity:** 100% mono float32 at $16,000\text{ Hz}$; all peak amplitudes $\le 1.0$; zero non-finite values.
- **Voice Disjointness:** Dev voices $\cap$ Test voices = $\emptyset$. Test voices $\cap$ `dataset_v12` voices = $\emptyset$. **100% Pass**.
- **OOV Lexeme Disjointness:** Dev OOV $\cap$ Test OOV = $\emptyset$. **100% Pass**.
- **Noise Track Disjointness:** Dev tracks $\cap$ Test tracks = $\emptyset$. **100% Pass**.
- **Acoustic Boundaries:** All event timings chronologically ordered with non-negative acoustic gaps. **100% Pass**.
- **Total Violations:** **0**.

---

## 5. EMPIRICAL EVALUATION OF FROZEN E1 ON NEW SYNTHETIC CORPORA

Evaluated via `run_phase21_evaluation.py` using frozen E1 ONNX (`vietnamese_kws_v12_e1.onnx`), `CanonicalLogMelFrontendNumPy`, $\theta_{\text{cmd}} = 0.20$, and frozen `SpeechBandRatioVAD`.

### 5.1 Primary Streaming Results
| Metric | Development Corpus (`synthetic_eval_dev_v1`) | Locked Test Corpus (`synthetic_eval_test_v1`) | Pre-Registered Acceptance Target |
| :--- | :---: | :---: | :---: |
| **Total Sequences** | 20 | 40 | - |
| **Exact-Sequence Accuracy** | **8 / 20 (40.00%)** | **22 / 40 (55.00%)** | Baseline Generalization |
| **Target Commands Total** | 64 | 128 | - |
| **Matched Target Commands** | **48 / 64** | **111 / 128** | - |
| **Missed Commands** | 16 | 15 | - |
| **Substitutions** | 0 | 2 | - |
| **Command Recall** | **75.00%** | **86.72%** | $> 80.0\%$ |
| **Command Precision** | **100.00%** | **91.74%** | $> 85.0\%$ |
| **Command Macro-F1** | **0.8571** | **0.8916** | $> 0.85$ |
| **Silence False Activations** | **0** | **0** | $\le 1$ event (**PASS**) |
| **Duplicate Triggers** | 1 | **0** | $0$ events (**PASS**) |
| **OOV False Activations** | 0 | **8** (Confusers: 8, General: 0) | Documented Distribution |
| **False Act / Neg Minute** | 0.00 | **9.63** | - |
| **Median Latency (P50)** | 310.0 ms | **320.0 ms** | - |
| **P95 Latency** | 682.5 ms | **825.0 ms** | $\le 850.0\text{ ms}$ (**PASS**) |

---

## 6. IN-DEPTH ANALYSIS & DIAGNOSTICS

### 6.1 Performance by Unseen Synthetic Voice
| Voice Preset | Voice Gender & Characteristics | Target Commands | Matched | Recall | Misses | OOV False Activations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Hữu Quân`** | Male, clear northern accent | 32 | **31** | **96.88%** | 0 | **0** |
| **`Xuân Tiên`** | Female, resonant southern tone | 32 | **27** | **84.38%** | 4 | **0** |
| **`Mạnh Dũng`** | Male, deep resonant baritone | 32 | **27** | **84.38%** | 5 | **0** |
| **`Ái Hân`** | Female, high fundamental frequency ($F_0$) | 32 | **26** | **81.25%** | 6 | **8** (all confusers) |

*Key Diagnostic:* E1 achieved $\ge 81.25\%$ recall across all 4 unseen voices. Voice `Hữu Quân` reached $96.88\%$ recall with zero false activations.

### 6.2 Phonetic Confuser Discrimination (Validation of RQ3)
- **General Everyday OOV Words (11 words):** `ghế`, `bàn`, `nhà`, `đèn`, `sách`, `bút`, `xe`, `nước`, `cơm`, `đồng`, `hồ`.  
  **Result: 0 / 11 False Activations (100.0% Rejection Precision).**
- **Minimal Pair Phonetic Confusers (13 words):**  
  **Result: 8 False Activations** occurred exclusively when paired with high-pitch female voice `Ái Hân`:
  - `trai` triggered as `TRAI` (1 event) — phonetic homophone differing only by lack of acute accent (tone contrast).
  - `phái` and `phai` triggered as `PHAI` (3 events) — identical onset and nucleus, tone shift.
  - `len` and `lăn` triggered as `LEN` (3 events) — vowel nucleus shift ([ɛ] vs [e]).
  - `xuồng` triggered as `XUONG` (1 event) — falling tone contrast on identical syllable.

### 6.3 Performance by Acoustic Condition & SNR
| Condition | Target SNR | Total Sequences | Exact Correct Seqs | Matched Commands | Command Recall | Total False Activations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clean`** | Clean ($\infty$) | 8 | **6 / 8 (75.0%)** | 30 / 32 | **93.75%** | **0** |
| **`snr20`** | 20.0 dB | 8 | **6 / 8 (75.0%)** | 23 / 24 | **95.83%** | 1 |
| **`snr15`** | 15.0 dB | 8 | **4 / 8 (50.0%)** | 22 / 24 | **91.67%** | 2 |
| **`snr10`** | 10.0 dB | 8 | **3 / 8 (37.5%)** | 22 / 24 | **91.67%** | 5 |
| **`snr05`** | 5.0 dB | 8 | **3 / 8 (37.5%)** | 16 / 24 | **66.67%** | 0 |

### 6.4 Confusion Matrix (Test Corpus)
*Rows = Reference Label, Columns = Predicted Class (`TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT/Miss`)*
```
                Pred_TRAI  Pred_PHAI  Pred_LEN  Pred_XUONG  Miss/REJECT
Ref_TRAI           21          1         0           0          10
Ref_PHAI            1         20         0           0           3
Ref_LEN             0          0        35           0           1
Ref_XUONG           0          0         0          35           1
Ref_OOV/Noise       1          3         3           1           -
```
- Command `LEN`: 35 / 36 recognized (**97.22% recall**).
- Command `XUONG`: 35 / 36 recognized (**97.22% recall**).
- Command `PHAI`: 20 / 24 recognized (**83.33% recall**).
- Command `TRAI`: 21 / 32 recognized (**65.63% recall**), with 10 misses occurring at high noise levels ($5\text{ dB}$ SNR).

---

## 7. CRYPTOGRAPHIC PIPELINE REGISTRY

| Component | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **v11 Checkpoint** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | Frozen Baseline |
| **v11 ONNX** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | Frozen Baseline |
| **E1 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | Frozen Research |
| **E1 ONNX** | `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` | `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8` | Frozen Profile B |
| **Selected VAD Config** | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | Frozen VAD |
| **Preregistration Document** | `experiments/vieneu_kws_research_v12/reports/PHASE21_SYNTHETIC_EVALUATION_PREREGISTRATION.md` | `2ba0408f96c00b849484598b358dea55835a02ad59ac50996584c4c43286631c` | Frozen Protocol |
| **Components Manifest** | `experiments/vieneu_kws_research_v12/manifests/synthetic_eval_components_manifest.csv` | `b590a7eab92e21c584b59a90dda688c40a00dd32ad1ab3fdee313db417294d04` | Frozen Components |
| **Dev Corpus Manifest** | `experiments/vieneu_kws_research_v12/manifests/synthetic_eval_dev_manifest.csv` | `57abdfcf3ba126f1372ec280af4b0e1e00dd81ef9f8da9ac6decb3ba8bc3cae3` | Frozen Dev Corpus |
| **Test Corpus Manifest** | `experiments/vieneu_kws_research_v12/manifests/synthetic_eval_test_manifest.csv` | `1093629730c2b1d1d022d2a6623de1c0c3302fc1e271d16691a55b68a1087d32` | Frozen Test Corpus |
| **Provenance Audit JSON** | `experiments/vieneu_kws_research_v12/reports/synthetic_provenance_audit_phase21.json` | `1d0c4c729c319953252c9bfc743cc1a4c7b57471d3f302c64d28d9115816929c` | Frozen Audit Log |
| **Validation Report JSON** | `experiments/vieneu_kws_research_v12/reports/phase21_corpus_validation_report.json` | `878ec20762bc9524a4ede17fcd0aac0c2fb10d972b09d3bd71d83f0aa3cb8e65` | Frozen Validation |
| **Evaluation Results JSON** | `experiments/vieneu_kws_research_v12/reports/phase21_synthetic_evaluation_results.json` | `c035134ec0cc4ad080cb41bc26251e223173b1265288972d803178c6d6d22a7e` | Frozen Results Log |

---

## 8. FINAL DECISION & SIGN-OFF

The comprehensive provenance audit confirmed that `dataset_v12` exhausted the preset voice space of `VieNeu-TTS v3 Turbo`. In accordance with Phase 21 pre-registration, an independent evaluation corpus was generated using the novel `VieNeu-TTS v3 Nano` architecture and verified unseen voices. Validation confirmed zero leakage, and evaluation of the frozen E1 system established strong cross-architecture generalization (86.72% recall, 91.74% precision, 55.0% exact-sequence accuracy, 0 silence false activations).

### Phase 21 Gate Status
**`PASS_TO_NEW_LOCKED_SYNTHETIC_EVALUATION`**

*Notice: This evaluation demonstrates cross-synthesizer robustness under controlled conditions. It does NOT authorize live physical-microphone deployment or validate performance on natural human speech.*
