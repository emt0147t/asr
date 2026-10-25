# SILENCE Alignment Preservation Audit and V2 Diagnostic Report

**Workspace:** `D:\trainidentificationaudio`  
**Audit Target:** Ambient SILENCE Semantics, Alignment Preservation Invariants, and V2 Diagnostic Evaluation  
**Status:** Audit Closed — Ambient Background Semantics Clarified and Finalized  
**Audit Verdict:** `AUDIT_PASS_WITH_LIMITATIONS`  

> [!CAUTION]
> **GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.**  
> Human participant recruitment, field speech collection, live acoustic recording, and model V5 training remain **STRICTLY BLOCKED**. All analyses, acoustic transformations, and model evaluations conducted in this audit strictly utilize synthetic fixtures and historical benchmark calibration corpora.

---

## 1. Executive Summary & Semantic Clarifications

During the paired alignment diagnostic audit of Synthetic KWS V4, an implementation flaw was investigated in the active speech centering logic of `experiments/synthetic_kws_v4/audit/run_alignment_diagnostic.py` (lines 203–210):

```python
act = np.where(np.abs(a) >= 0.005)[0]
last_idx = act[-1] if len(act) > 0 else 8000
speech = a[:last_idx + 1]
pad_tot = 16000 - len(speech)
pad_l = pad_tot // 2
pad_r = pad_tot - pad_l
a_center = np.pad(speech, (pad_l, pad_r))[:16000]
```

### 1.1 Ambient SILENCE Semantics and the Absence of Onset
A synthetic `SILENCE` recording is **not digital silence** (which would be an exact zero $0.0$ flatline). It is a **synthetic ambient room noise and pseudo-random dither fixture** spanning all $16,000$ samples with a mean RMS of $0.002082$.

Crucially:
- **25 of the 50 clips** (all odd takes: take001, take003, take005, take007, take009) have **zero samples** exceeding the threshold $0.005$ ($\text{count}(\|a\| \ge 0.005) = 0$).
- **25 of the 50 clips** (all even takes: take000, take002, take004, take006, take008) have sporadic random noise spikes reaching $\ge 0.005$ scattered across the buffer.
- Neither cohort possesses a linguistically or acoustically defensible speech onset, release burst, or utterance boundary.

### 1.2 Rectification of Alignment Terminology
Prior preliminary notes loosely referred to Condition A2-B2 as "onset-centered." This report explicitly clarifies:
- The 50 `SILENCE` fixtures were **NOT uniformly center-aligned**.
- When no defensible onset can be identified in an ambient background signal, attempting to "center" it by threshold cropping is mathematically and physically invalid.
- Condition A2-B2-V2 **deliberately preserves the full $16,000$-sample ambient recording unshifted** (`pad_l = 0`, `pad_r = 0`, `shift = 0`).
- For speech and confuser classes (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, 250 clips), active speech is centered within the $1.0\,\text{s}$ buffer.

### 1.3 Label Policy Clarification
Manifests and checkpoints retain the historical label `SILENCE` solely for backward compatibility with trained model dictionary heads (`num_classes=6`) and historical benchmark schemas. Future production pipelines should relabel this class as `BACKGROUND` or `AMBIENT_NOISE` to prevent confusion with digital zero silence.

---

## 2. Independent Verification of Final Preservation Claims

### 2.1 Independent Confirmation for All 50 SILENCE Pairs
Every one of the 50 synthetic `SILENCE` pairs was independently audited between the source holdout audio (`data/fresh_holdout/`) and Condition A2-B2-V2 output (`audit/diagnostic_audio_v2/A2_B2_onset_centered/`):

- **Sample Count:** Exactly $16,000$ samples in source, exactly $16,000$ samples in V2 output ($50/50$ pairs).
- **Sample Rate & Channels:** $16,000\,\text{Hz}$, $1$ channel (mono) in both source and output ($50/50$ pairs).
- **Sample Sequence Equality:** Bit-exact identical sample sequence ($a_{\text{v2}}[i] == a_{\text{src}}[i]$ for all $i \in [0, 15999]$). Maximum absolute difference $\max |\Delta| = 0.000000$ across all 50 clips.
- **Zero Omission & Zero Alteration:** $0$ samples omitted, $0$ samples altered, $0$ zero-padding inserted ($50/50$ pairs).
- **Whole-File SHA-256 Equality:** Whole-file SHA-256 matches bit-for-bit between source and V2 output across all $50/50$ pairs.
- **Stable Pair Identifiers:** Mapped from `SILENCE_PAIR_000` to `SILENCE_PAIR_049` in [`alignment_preservation_v2.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/alignment_preservation_v2.json).

#### Representative Verification Sample (SILENCE Cohort)
| Pair ID | Recording ID | Source & V2 SHA-256 (Matching) | Sample Count | Sample Rate | Max Abs Diff ($\max \|\Delta\|$) | Preservation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SILENCE_PAIR_000` | `fresh_v4_silence_SPK_CALIB_031_take000` | `EC6CF7051C47E20F49FC21487EFAC6A386D22DEFFCEEB20F4F7A02C1F7DF5803` | 16,000 | 16 kHz | $0.000000$ | **BIT-EXACT INTACT** |
| `SILENCE_PAIR_001` | `fresh_v4_silence_SPK_CALIB_031_take001` | `D923C2BC3E4A7F4846F9EB4AE99E74B088F5E97FE218764B5666014AF39FA157` | 16,000 | 16 kHz | $0.000000$ | **BIT-EXACT INTACT** |
| `SILENCE_PAIR_002` | `fresh_v4_silence_SPK_CALIB_031_take002` | `DC30CE8593C297F16C62C6FF2A4DFF4D78B2A633758BA5782B8710CF5829623D` | 16,000 | 16 kHz | $0.000000$ | **BIT-EXACT INTACT** |
| `SILENCE_PAIR_003` | `fresh_v4_silence_SPK_CALIB_031_take003` | `B777A382103565860714DE88E52A7453D7653F50DFDE598F6586236965E502B0` | 16,000 | 16 kHz | $0.000000$ | **BIT-EXACT INTACT** |
| `SILENCE_PAIR_004` | `fresh_v4_silence_SPK_CALIB_031_take004` | `EAEFF16E7F28EEA750BEF97825C992FE586BA0AE63DF9B172EE2B8D8FDF52E84` | 16,000 | 16 kHz | $0.000000$ | **BIT-EXACT INTACT** |
| ... | *(45 additional verified pairs)* | *(100% SHA-256 match)* | 16,000 | 16 kHz | $0.000000$ | **BIT-EXACT INTACT** |

### 2.2 Distinctions for Active Speech Diagnostic Pairs (394 Pairs)
For the 394 speech pairs (144 baseline historical pairs + 250 V4 speech/confuser pairs):
- **Active Speech Content:** The contiguous non-padding speech signal is $100\%$ bit-exact preserved in value and sequence order ($\max |\Delta| = 0.000000$ on all speech samples). Zero active speech samples were truncated.
- **Outer Padding Repositioning:** Leading and trailing zero-padding was intentionally adjusted to shift alignment (e.g. trimming leading silence for left-alignment, or symmetric zero-padding for center-alignment).
- **Integrity Boundary:** The report **does not claim the entire $1.0\,\text{s}$ WAV file is bit-exact** for speech pairs across alignments; outer padding differences are by design to test model alignment sensitivity.

---

## 3. Model Predictions vs Acoustic Equivalence

All four frozen checkpoints were evaluated on Condition A2-B2-V2 and compared directly against the historical buggy A2-B2 evaluation:

| Model Arm | Condition A2-B2-V2 Accuracy | Buggy A2-B2 Accuracy | Predictions Equal Count | Predictions Changed Count | Confusion Matrix Difference ($\Delta \text{CM}$) | SILENCE Recall (V2) | SILENCE False Alarms on Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm A (Control)** | $83.33\%$ ($250/300$) | $83.33\%$ ($250/300$) | **300 / 300** | **0** | $\mathbf{0}_{6 \times 6}$ | $100.0\%$ ($50/50$) | $0 / 50$ ($0.0\%$) |
| **Arm B (Onset-Aware)** | $96.33\%$ ($289/300$) | $96.33\%$ ($289/300$) | **300 / 300** | **0** | $\mathbf{0}_{6 \times 6}$ | $100.0\%$ ($50/50$) | $0 / 50$ ($0.0\%$) |
| **Arm C (Onset + Resamp)**| $92.33\%$ ($277/300$) | $92.33\%$ ($277/300$) | **300 / 300** | **0** | $\mathbf{0}_{6 \times 6}$ | $100.0\%$ ($50/50$) | $0 / 50$ ($0.0\%$) |
| **V3 Champion** | $83.33\%$ ($250/300$) | $83.33\%$ ($250/300$) | **300 / 300** | **0** | $\mathbf{0}_{6 \times 6}$ | $100.0\%$ ($50/50$) | $0 / 50$ ($0.0\%$) |

> [!IMPORTANT]
> **Scientific Finding: Prediction Equality Does NOT Imply Acoustic Equivalence.**  
> The fact that every model yielded identical predictions ($300/300$) and identical confusion matrices between the buggy and corrected transforms **does NOT prove that a $50\%$ truncated audio file with $8,000$ digital zeros is acoustically equivalent to a continuous $16,000$-sample ambient recording**.  
> Rather, TC-ResNet-14 classified both representations as `SILENCE` because its log-mel filterbank energies remained strictly below activation thresholds for all target keyword classes in both cases. Rectifying the transform was required for data integrity, spectral measurement accuracy, and sound experimental methodology.

---

## 4. Finalized Scientific Reporting & Disaggregated Dimensions

To maintain complete rigor, the audit distinguishes five separate scientific dimensions:

```
+-------------------------------------------------------------------------------------------------+
|                                    FIVE SCIENTIFIC DIMENSIONS                                    |
+-------------------------------------------------------------------------------------------------+
| 1. Metric Reproducibility  | Exact mathematical identities verified across all confusion matrices|
| 2. Waveform Preservation   | Bit-exact speech segment preservation; full-buffer ambient dither   |
| 3. Alignment Sensitivity   | Shift effects separated from onset acoustics (XUONG & TRAI impact)  |
| 4. Cross-Gen Generalization| Failure of models to transfer across synthetic generators           |
| 5. Acoustic Causality      | DISC-02: 25ms stop closure is NOT proven sole cause (UNRESOLVED)    |
+-------------------------------------------------------------------------------------------------+
```

1. **Metric Arithmetic & Reproducibility:**  
   Verified $100\%$ consistent across all $6,768$ canonical prediction rows. Trace equals accuracy $\times$ support; row sums match class support; target command misses match error counts.
2. **Waveform Preservation:**  
   Active speech segments are $100\%$ preserved in order without active sample truncation ($\max |\Delta| = 0.0$); ambient `SILENCE` is $100\%$ preserved in full $16\,\text{k}$ buffer without cropping ($\max |\Delta| = 0.0$).
3. **Sensitivity to Temporal Alignment:**  
   The 2x2 diagnostic demonstrated that models exhibit varying degrees of temporal alignment sensitivity: `XUONG` on Arm B jumped from $16.7\%$ to $100.0\%$ when left-aligned; Arm A `TRAI` collapsed from $34.0\%$ to $0.0\%$ when onset-aware audio was center-aligned.
4. **Cross-Generator Generalization:**  
   Models trained on the baseline generator fail to generalize to the onset-aware generator, and vice versa.
5. **Unresolved Acoustic Causality (DISC-02):**  
   The diagnostic proved that Arm B's $0.0\%$ `TRAI` recall on baseline audio persists under both center-aligned and left-aligned conditions ($0.0\%$ in both). However, **the 25 ms stop closure alone cannot be declared the proven cause of this failure**. The baseline and onset-aware generators co-vary across multiple acoustic dimensions simultaneously (burst frequencies, formant transitions, pitch contours, and spectral tilt). Causal attribution remains explicitly **`UNRESOLVED`**.

### 4.1 Disaggregated Spectral Metrics
The aggregate class metrics do not treat `SILENCE` threshold crossings as an onset:
- **Speech Classes (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`):** Measured via **`active_onset_relative_hf_energy_ratio`** (evaluated on the first $50\,\text{ms}$ / $800$ samples following the threshold-detected onset $\ge 0.005$). Invariant between native and centered alignments ($\max |\Delta| = 0.00000000$).
- **Ambient `SILENCE` Class:** Measured via **`fixed_window_ambient_noise_spectral_ratio`** (evaluated on a fixed initial window of samples 0–799, $0.0$–$50.0\,\text{ms}$). Invariant between native and V2 unshifted preservation ($\max |\Delta| = 0.00000000$, value $0.282083$).

---

## 5. Discrepancy Log Status

In [`discrepancy_log.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_v4/audit/discrepancy_log.json):
- **`DISC-01` (Arm A Validation Accuracy Discrepancy):** `RESOLVED`
- **`DISC-02` (Arm B Zero TRAI Recall & Alignment Confound):** **`UNRESOLVED`** (causal claim regarding 25 ms closure is not proven).
- **`DISC-03` (Lineage Verification in Historical Schema):** `NOT_VERIFIABLE`
- **`DISC-04` (Pitch Augmentation Terminology Discrepancy):** `RESOLVED`
- **`DISC-05` (Arithmetic Contradictions in Preliminary Report):** `RESOLVED`
- **`DISC-06` (Waveform Closure RMS Contradiction):** `RESOLVED`
- **`DISC-07` (Temporal Alignment Discrepancy Across Stages):** `RESOLVED`
- **`DISC-08` (SILENCE Centering Truncation Fallback):** `RESOLVED` via Condition A2-B2-V2.

---

## 6. Regression Testing and Safety Verification

1. **Focused Acceptance Tests:**  
   [`tests/test_campaign_acceptance.py`](file:///d:/trainidentificationaudio/tests/test_campaign_acceptance.py) verifies:
   - Threshold-free ambient noise is never truncated by a centering heuristic.
   - Ambient fixtures preserve all $16,000$ samples unshifted ($\max |\Delta| = 0.0$).
   - Waveform integrity claims distinguish speech content preservation from outer zero-padding.
   - Report text does not claim `SILENCE` was center-aligned.
   - `DISC-02` remains `UNRESOLVED` and final audit verdict is `AUDIT_PASS_WITH_LIMITATIONS`.
2. **Full Acceptance Test Suite:**  
   `pytest tests/test_campaign_acceptance.py -q`: **64 passed in 30.45s** (exit code 0).
3. **Protected Baselines SHA-256 Check:** All 3 protected baselines match exactly:
   - `data/manifests/test_manifest_v2.csv`: `9E3904C8EA427244...` (MATCH)
   - `data/raw/field_recordings/field_collection_manifest.csv`: `B389282F9853A744...` (MATCH)
   - `data/raw/field_recordings/consent_ledger.json`: `BE4BFDD7CEC0C87A...` (MATCH)
4. **Git Workspace Hygiene:**
   - `git diff --check`: 0 errors / clean.
   - `git diff --cached`: **0 files staged** (clean).

---

## 7. Final Closeout Verdict

### Verdict: `AUDIT_PASS_WITH_LIMITATIONS`

#### Technical Summary:
1. All documented data-integrity, waveform preservation, and canonical metric identity checks pass with zero discrepancies.
2. The semantic distinction between synthetic ambient background fixtures and digital silence is formally established.
3. Full-buffer unshifted preservation of ambient audio is empirically verified with matching source/output SHA-256 hashes ($50/50$).
4. The acoustic causality regarding Arm B and the 25 ms stop closure is properly scoped and retained as unresolved (`DISC-02`).
5. **HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.** Model V5 training has not been started.
