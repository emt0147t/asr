# Vietnamese Synthetic KWS — VieNeu-TTS 1,000-Utterance Dataset Audit and Closeout Report

**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/`  
**Dataset Version:** 1.0.0 (Research Release Candidate)  
**Task Continuation ID:** `task-8705`  
**Date:** October 17, 2026  
**Final Technical Verdict:** **`PASS WITH LIMITATIONS`**  
**Technical Signal Integrity Verdict:** **`PASS`**  
**Subjective Intelligibility Status:** **`PENDING_HUMAN_LISTENER_VERIFICATION`**  
**Official Project Governance Status:** **`NOT_VALIDATED_ON_NATURAL_SPEECH`**  

---

## 1. Executive Summary & Verdict Decision

This report provides the final independent audit and verification for the generation and validation of the **1,000-Utterance Vietnamese Synthetic Keyword Spotting (KWS) Dataset**. Synthesized entirely using the open-source neural TTS engine **VieNeu-TTS v3 Turbo** (`pnnbao-ump/VieNeu-TTS-v3-Turbo`) executed locally on CPU via ONNX Runtime, this dataset establishes an open-weights, zero-cost, privacy-preserving acoustic corpus for Vietnamese command recognition.

### Verdict: `PASS WITH LIMITATIONS`
- **Technical & Acoustic Signal Integrity:** **`PASS`**
  - Exactly 1,000 audio pairs on disk matching 1,000 manifest rows with 100% SHA-256 byte-level agreement.
  - Exactly 250 utterances per canonical keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`).
  - Exactly 40 utterances per preset voice across 25 preset voices (10 per command).
  - Held-out preset split partition (680 train, 160 val, 160 test) with zero voice leakage.
  - Zero digital clipping (peak $\le 0.9500$), zero NaNs/Infs, zero phoneme truncation ($\ge 50$ ms margins).
  - Independent re-synthesis confirms bit-exact determinism ($\Delta = 0.00000000\text{e}+00$).
  - 69/69 automated regression and baseline tests pass with zero failures.
- **Scientific & Operational Limitations:**
  1. **Synthetic Nature of Presets:** Preset voices represent synthetic neural conditioning vectors ($\mathbf{e}_{\text{spk}} \in \mathbb{R}^{192}$, $\mathbf{C}_{\text{ref}}$) within the VieNeu latent space, not independent human participants.
  2. **Held-out Split Scope:** The test partition evaluates generalization to unseen *synthetic voice presets*. It **must not** be construed as evidence of generalization to natural human speakers, field microphones, or acoustic reverberant spaces.
  3. **Lexical Tone Diagnostics vs Perceptual Ground Truth:** Objective $F_0$ pitch tracking confirms phonologically expected rising and dipping pitch slopes across classes, but objective pitch slopes do not constitute proof of human lexical tone perception.
  4. **Human Review Status:** A 80-sample stratified blinded listening package has been generated and quarantined. Subjective intelligibility remains formally `PENDING_HUMAN_LISTENER_VERIFICATION`.

---

## 2. Phase A: Generation Completion & Inventory Verification

### 2.1 Process Lifecycle & Task State
- Task `task-8705` ran to completion in the dedicated `.venv_vieneu` environment without interruption (exit code 0, elapsed time 630.7s).
- No secondary or concurrent `generate_dataset.py` process was executed.

### 2.2 Filesystem vs. Manifest Audit
An exhaustive cross-reference between `manifest.csv` and the filesystem confirms perfect one-to-one correspondence:

| Inventory Dimension | Manifest Record Count | Filesystem File Count | Discrepancy | Status |
| :--- | :---: | :---: | :---: | :-: |
| **Raw Master Audio (48 kHz mono)** | 1,000 rows | 1,000 files in `audio/raw_48k/` | 0 missing, 0 unreferenced | **PASS** |
| **KWS Audio (16 kHz 1.0s mono)** | 1,000 rows | 1,000 files in `audio/kws_16k/` | 0 missing, 0 unreferenced | **PASS** |
| **SHA-256 Checksum Verification** | 1,000 entries | 1,000 matching disk digests | 0 mismatches | **PASS** |
| **Stale / Temporary Files** | N/A | 0 temporary files found | None | **PASS** |

### 2.3 Per-Class and Per-Voice Distribution
- **Per-Class Breakdown:**
  - `TRAI` (`Trái`): exactly 250 records (25.0%)
  - `PHAI` (`Phải`): exactly 250 records (25.0%)
  - `LEN` (`Lên`): exactly 250 records (25.0%)
  - `XUONG` (`Xuống`): exactly 250 records (25.0%)
- **Per-Voice Breakdown:**
  - 25 installed preset voices: `Adam, Adam bựa, Hải Đăng, Kim Thanh, Mai Anh, Minh Triết, Minh Đức, Mỹ Duyên, Ngọc Huyền, Ngọc Linh, Ngọc Trân, Phạm Tuyên, Quang Sơn, Quốc Tuấn, Quỳnh Anh, Thái Sơn, Thanh Bình, Thiền Tâm Đức, Thiện Minh, Thùy Dung, Thục Đoan, Trúc Ly, Xuân Vĩnh, Đoan Trang, Đức Trí`.
  - Every voice preset has exactly **40 accepted records** (10 records per command class).

---

## 3. Phase B: Audio Integrity & Complete-Word Coverage

### 3.1 Signal Health Audit
Every accepted audio clip was checked for numeric validity, format compliance, and boundary integrity:
- **Raw Waveforms:** Readable, mono, 48,000 Hz, finite values, zero NaNs, zero Infs, no near-silent audio (peak $\ge 0.05$).
- **KWS Waveforms:** Readable, mono, 16,000 Hz, exactly 16,000 samples (1.0000 second).
- **Anti-Aliasing Resampling:** Resampling from 48 kHz to 16 kHz is performed via exact polyphase filtering (`scipy.signal.resample_poly(raw, 1, 3)`), applying a Kaiser-window FIR anti-aliasing filter at the 8 kHz Nyquist rate.
- **Boundary Continuity:** Boundary sample amplitudes at sample index 0 and 15,999 were verified across all 1,000 files. Max edge amplitude observed: **`0.000000`** (zero boundary clicks or step discontinuities).

### 3.2 Complete-Word Coverage & Margins
- **Speech Window Placement:**
  - Active speech duration is constrained to $0.20\text{s} \le D_{\text{speech}} \le 0.85\text{s}$.
  - Target onset is set to sample 2,400 (0.150s).
  - When speech duration exceeds 0.75s, onset is shifted backward while preserving minimum leading margin $\ge 50$ ms and minimum trailing margin $\ge 50$ ms.
  - Across all 1,000 samples: Leading margin $\ge 50.0$ ms, trailing margin $\ge 50.0$ ms. Truncation count: **0**.

### 3.3 Root-Cause Resolution of Historical Pilot Defects
1. **Truncation on `PHAI` (`vieneu_pilot_phai_trucly_f_north`):**
   - *Pilot Defect:* Final diphthong off-glide was cut off by heuristic `peak * 0.02` silence trimming.
   - *Resolution:* Heuristic trimming was eliminated; natural neural vocoder audio envelope is preserved intact. Trúc Ly's 10 `PHAI` samples in the 1,000 dataset have active speech durations from 0.44s to 0.58s, fitting completely within the 1.0s window with > 250 ms trailing silence.
2. **Bass Distortion on Deep Voices (`Phạm Tuyên`):**
   - *Pilot Defect:* Low-frequency F0 (~85–110 Hz) produced inter-sample clipping and harmonic distortion.
   - *Resolution:* Headroom capped at 0.9500 peak amplitude, with polyphase antialiasing filter mitigating sub-harmonic energy buildup.

---

## 4. Phase C: Determinism & Acoustic Diversity Audit

### 4.1 Implementation of `OnnxV3LiteEngine._sample`
Inspection of `.venv_vieneu/Lib/site-packages/vieneu/_v3_turbo_engine/onnx_runtime_lite.py` reveals:
```python
def _sample(self, logits, temperature, top_k, top_p, rep_pen, prev):
    ...
    # top_k filtering, softmax, top_p nucleus filtering
    return int(cand[np.random.choice(cand.shape[-1], p=p)])
```
Stochastic token sampling is mediated entirely by `np.random.choice` on candidate top-$p$ probabilities. Prior to inference, `generate_dataset.py` explicitly sets `np.random.seed(current_seed)`.

### 4.2 Independent Re-Synthesis Verification
To verify that global NumPy seeding guarantees bit-exact reproducibility, an independent verification script ([`verify_determinism.py`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/verify_determinism.py)) re-synthesized samples across all 4 classes from scratch:

| Sample ID | Class | Voice Preset | Seed | Max Absolute Raw Diff | Max Absolute KWS Diff | SHA-256 Match | Determinism Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :-: |
| `vnkws_trai_0001` | `TRAI` | Adam bựa | 100000 | 0.00000000e+00 | 0.00000000e+00 | 100% Match | **BIT_EXACT** |
| `vnkws_phai_0001` | `PHAI` | Adam bựa | 110000 | 0.00000000e+00 | 0.00000000e+00 | 100% Match | **BIT_EXACT** |
| `vnkws_len_0001` | `LEN` | Adam bựa | 120000 | 0.00000000e+00 | 0.00000000e+00 | 100% Match | **BIT_EXACT** |
| `vnkws_xuong_0001`| `XUONG`| Adam bựa | 130000 | 0.00000000e+00 | 0.00000000e+00 | 100% Match | **BIT_EXACT** |

*Finding:* Bit-exact determinism is confirmed on identical hardware and software runtimes.

### 4.3 Acoustic Diversity & Near-Duplicate Analysis
- **Exact Duplicate Hashes:** 0 duplicate raw hashes, 0 duplicate KWS hashes.
- **Waveform Cross-Correlation Analysis:**
  Evaluated across all **4,500 pairwise comparisons** of utterances sharing both identical voice and identical command word:
  - Mean pairwise correlation: **0.0071**
  - Minimum correlation: **-0.5449**
  - Maximum correlation: **0.7321**
  - Near-duplicate candidates ($\text{correlation} > 0.90$): **0**
- **Acoustic Metric Distributions:**
  - Active Speech Duration: $0.6099 \pm 0.1073$ s (Range: $[0.320, 0.800]$ s)
  - Speech RMS: $-21.88 \pm 2.25$ dBFS
  - Spectral Centroid: $85.7 \pm 43.5$ Hz
  - Peak Amplitude: $0.5086 \pm 0.1142$ (Max: 0.9500)

---

## 5. Phase D: Label & Tone Quality Diagnostics

### 5.1 Objective Fundamental Frequency ($F_0$) Contour Analysis
Using short-time autocorrelation pitch tracking across all 1,000 utterances:

| Command | Vietnamese Tone | Mean $F_0$ | Mean $F_0$ Slope ($\Delta F_0$) | Dipping Rate | Acoustic Tone Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`TRAI`** | sắc (high-rising) | 193.0 Hz | **+63.2 Hz** | 20.8% | Pronounced rising pitch trajectory matching *sắc* tone |
| **`PHAI`** | hỏi (dipping) | 180.3 Hz | **+13.1 Hz** | **34.8%** | Prominent pitch dip during vowel onset before recovery |
| **`LEN`** | ngang (mid-level) | 173.0 Hz | **-6.8 Hz** | 20.0% | Stable, flat fundamental frequency trajectory |
| **`XUONG`**| sắc (high-rising) | 204.1 Hz | **+54.9 Hz** | 26.4% | Distinct rising pitch trajectory over diphthong `/uə/` |

- **Gender Registers:** Female voices average $F_0 = 239.3 \pm 40.3$ Hz; male voices average $F_0 = 147.0 \pm 35.9$ Hz.
- **Caveat:** These $F_0$ contours represent acoustic diagnostic evidence. Perceptual tone correctness is formally unconfirmed until human listeners score the blinded review sheet.

### 5.2 Stratified Blinded Listening Package (80 Samples)
A standardized, quarantined listening test package was created under [`blinded_listening/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/):
- **Sample Count:** Exactly 80 clips (20 `TRAI`, 20 `PHAI`, 20 `LEN`, 20 `XUONG`).
- **Voice Coverage:** All **25 preset voices** represented (100% voice coverage).
- **Split Distribution:** 52 train, 14 val, 14 test.
- **Anonymized Audio:** Files named `blind_eval_01.wav` to `blind_eval_80.wav` with all metadata concealed.
- **Preserved Key:** Quarantined in [`blinded_listening_key.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.json) and [`blinded_listening_key.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.csv).
- **Blank Evaluation Sheet:** [`blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv) evaluating command identification, defect flags (clipping, distortion, truncation, buzzer), perceived tone, intelligibility (1-5), and naturalness (1-5).
- **Listener Guide:** [`BLINDED_LISTENING_GUIDE.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/BLINDED_LISTENING_GUIDE.md).

---

## 6. Phase E: Split Integrity & Disjoint Preset Policy

The dataset enforces a strict preset-disjoint partitioning strategy:

```
                          [25 Preset Voices (1,000 Utterances)]
                                           |
               +---------------------------+---------------------------+
               |                           |                           |
               v                           v                           v
     [Train Partition]             [Val Partition]             [Test Partition]
        17 Voices                      4 Voices                    4 Voices
       680 Utterances                160 Utterances              160 Utterances
     (170 per command)              (40 per command)            (40 per command)
```

- **Voice Overlap Check:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 shared voices)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 shared voices)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 shared voices)
- **Scientific Limitation:** This split evaluates out-of-preset synthetic generalization. It does **not** evaluate generalization to natural human speakers, field recording devices, or room acoustics.

---

## 7. Phase F: Test Results & Project Integrity

### 7.1 Test Suite Summary
All test suites executed with exit code 0:
1. `validate_dataset.py` $\to$ **PASS** (1,000/1,000 verified)
2. `analyze_diversity.py` $\to$ **PASS** (0 duplicates, 0 near-duplicates)
3. `verify_determinism.py` $\to$ **PASS** (4/4 bit-exact)
4. `pytest` Regression Suites $\to$ **69 passed in 6.21s**:
   - `tests/test_vieneu_dataset_1000.py`: 5 passed
   - `tests/test_vieneu_local_pilot.py`: 5 passed
   - `tests/test_tts_pilot.py`: 5 passed
   - `tests/test_parity.py`: 7 passed
   - `tests/test_kws_export_parity.py`: 13 passed
   - `tests/test_synthetic_kws_v5_2_candidate.py`: 5 passed
   - `tests/test_synthetic_kws_v5_1_candidate.py`: 12 passed
   - `tests/test_synthetic_kws_v5_generator.py`: 17 passed

### 7.2 Git State Integrity
- `git diff --cached` is strictly empty. No files staged or committed.
- The 13 historical user modifications in the working tree were strictly preserved and untouched.

---

## 8. Provenance, Licensing, and Dependencies

| Component | Upstream Origin | License | Local Verification Status |
| :--- | :--- | :--- | :-: |
| **VieNeu-TTS v3 Turbo** | `pnnbao97/VieNeu-TTS` | Apache-2.0 | Verified local CPU inference via ONNX Runtime |
| **MOSS Vocoder** | `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX` | Apache-2.0 | Verified bundled local model weights |
| **sea-g2p** | `pnnbao97/sea-g2p` | Apache-2.0 | Verified Rust phonemizer binary |
| **External Cloud APIs** | None | N/A | Verified zero cloud calls; $0.00 cost |

---

## 9. Artifact Path Index

| Artifact | File Path |
| :--- | :--- |
| **Master Manifest (1,000 rows)** | [`manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/manifest.csv) |
| **Master Raw 48 kHz Audio** | [`audio/raw_48k/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/audio/raw_48k/) |
| **Standardized KWS 16 kHz Audio**| [`audio/kws_16k/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/audio/kws_16k/) |
| **Split Manifests** | [`splits/train_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/splits/train_manifest.csv)<br>[`splits/val_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/splits/val_manifest.csv)<br>[`splits/test_manifest.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/splits/test_manifest.csv) |
| **Blinded Listening Package (80)** | [`blinded_listening/`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/) |
| **Preserved Answer Key** | [`blinded_listening/blinded_listening_key.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_key.json) |
| **Listener Review Sheet** | [`blinded_listening/blinded_listening_review_sheet.csv`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/blinded_listening_review_sheet.csv) |
| **Rejection Log (33 retries)** | [`reports/rejection_log.json`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/rejection_log.json) |
| **Tone & F0 Diagnostic Report** | [`reports/TONE_AND_INTELLIGIBILITY_AUDIT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/TONE_AND_INTELLIGIBILITY_AUDIT.md) |
| **Validation Report** | [`reports/DATASET_VALIDATION_REPORT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/DATASET_VALIDATION_REPORT.md) |
| **Diversity Report** | [`reports/DATASET_DIVERSITY_REPORT.md`](file:///D:/trainidentificationaudio/experiments/synthetic_kws_dataset_vieneu_1000/reports/DATASET_DIVERSITY_REPORT.md) |
