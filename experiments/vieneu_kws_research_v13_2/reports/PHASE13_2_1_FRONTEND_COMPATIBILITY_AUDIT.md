# Phase 13.2.1: Pre-Synthesis Frontend Compatibility Audit Report

**Status:** `PRE_SYNTHESIS_GATE_PASSED`  
**Date:** 2026-10-19  
**Candidate Identifier:** `variant_v11_hard_negative_weighted_seed42.pt`  
**Verified Candidate SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**Authoritative Frontend Specification:** $N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$, 40 Slaney Mel filters ($20.0 - 8000.0\text{ Hz}$), Power $= 2.0$, $\ln(\cdot + 10^{-6})$  
**Operational Assessment:** `FROZEN_CNN_CANDIDATE` | `NOT_VALIDATED_ON_NATURAL_SPEECH` | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`

---

## 1. Executive Summary

This audit resolves the pre-synthesis compatibility blocker identified prior to executing Phase 13.2 audio synthesis and one-shot confirmatory evaluation. Specifically, a discrepancy existed between historical deployment prose and the amended evaluation protocol:
- **Phase 92 Deployment Report:** Described $N_{\text{fft}} = 400$ and a $40 \times 201$ Slaney filterbank.
- **Phase 13.2 Confirmatory Protocol Summary:** Specified $N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, and $H_{\text{len}} = 160$ ($40 \times 257$ filterbank).

Through an exhaustive audit of executable training source code, inference modules, and numerical verification on development fixtures, this audit establishes that:
1. **The Frozen Checkpoint was Trained with $N_{\text{fft}} = 512$:**  
   The frozen champion `variant_v11_hard_negative_weighted_seed42.pt` was trained strictly using `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, mel_scale="slaney", norm="slaney", power=2.0)`.
2. **Phase 92 Described a Superseded Baseline Candidate:**  
   Phase 92 deployment engineering targeted the legacy Phase 9/10 candidate `variant_w10_seed42.pt`, which used $N_{\text{fft}} = 400$. When Phase 11 trained the current champion (`variant_v11`), the pipeline adopted standard power-of-2 FFT ($N_{\text{fft}} = 512$) with 400-sample windowing.
3. **Phase 13.1 and Phase 13.2 Specifications are 100% Bitwise Identical to Training:**  
   The remediated Phase 13.1 inference engine (`experiments/vieneu_kws_research_v13_1/bundle/infer.py`) and Phase 13.2 protocol reproduce the training frontend with **zero numerical divergence** ($\text{MaxDiff} = 0.00\text{e}+00$) across all development fixtures.
4. **Pure-NumPy Portable Frontend Reaches Complete Numerical Parity:**  
   A portable pure-NumPy frontend implementing the exact PyTorch centered-window STFT formulation achieves a maximum absolute difference of $\le 2.00 \times 10^{-5}$ on log-Mel features, $\le 7.15 \times 10^{-7}$ on model logits, and **100% decision parity**.

**Gate Verdict:** `PRE_SYNTHESIS_GATE_PASSED`. Confirmatory audio synthesis for Phase 13.2 is unblocked.

---

## 2. File-and-Line Evidence Audit

The provenance of frontend parameters across all relevant phases was audited directly from executable source code:

| Artifact / File Path | Exact Lines | Parameters Implemented | Purpose / Context | Authoritative for v11? |
| :--- | :--- | :--- | :--- | :---: |
| `experiments/vieneu_kws_research_v11/train_v11.py` | [Lines 36–53, 266](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v11/train_v11.py#L36-L53) | `n_fft=512, win_length=400, hop_length=160, n_mels=40, power=2.0, mel_scale="slaney", norm="slaney", eps=1e-6` | Training module that produced the frozen checkpoint | **YES (Authoritative)** |
| `experiments/vieneu_kws_research_v11/evaluate_v11.py` | [Lines 38, 227](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v11/evaluate_v11.py#L38-L45) | `LogMelFrontend()` default: `n_fft=512, win_length=400, hop_length=160, n_mels=40` | Validation & evaluation code for Phase 11 | **YES** |
| `experiments/vieneu_kws_research_v12/bundle/infer.py` | [Lines 29–48](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/bundle/infer.py#L29-L48) | `n_fft=512, win_length=400, hop_length=160, n_mels=40` | Packaged Phase 12 inference script | **YES** |
| `experiments/vieneu_kws_research_v13_1/bundle/infer.py` | [Lines 30–50](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_1/bundle/infer.py#L30-L50) | `n_fft=512, win_length=400, hop_length=160, n_mels=40` | Remediated reference inference script | **YES** |
| `experiments/vieneu_kws_research_v13_1/bundle/preprocessing_contract.json` | [Lines 18–24](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_1/bundle/preprocessing_contract.json#L18-L24) | `n_fft: 512, win_length: 400, hop_length: 160, n_mels: 40, f_min: 20.0, f_max: 8000.0` | Preregistered preprocessing contract | **YES** |
| `experiments/vieneu_kws_research_v13_2/protocol/AMENDED_CONFIRMATORY_PROTOCOL_V13_2.md` | [Section 5](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_2/protocol/AMENDED_CONFIRMATORY_PROTOCOL_V13_2.md#L60-L85) | `N_fft=512, win_length=400, hop_length=160, 40 Slaney bands` | Confirmatory evaluation protocol specification | **YES** |
| `experiments/vieneu_kws_research_v10/deployment/portable_frontend.py` | [Lines 17–45](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/deployment/portable_frontend.py#L17-L45) | `n_fft=400, win_length=400, hop_length=160, 40 Slaney bands` | Historical Phase 92 deployment export for `variant_w10` | **NO (Superseded Baseline)** |
| `reports/PHASE92_CROSS_PLATFORM_PREPROCESSING_PARITY.md` | [Lines 32–48](file:///d:/trainidentificationaudio/reports/PHASE92_CROSS_PLATFORM_PREPROCESSING_PARITY.md#L32-L48) | Prose description: `N_fft=400, 40x201 filterbank` | Report documenting Phase 92 parity on `variant_w10` | **NO (Superseded Baseline)** |

---

## 3. Discrepancy Root-Cause Analysis

### 3.1 Historical Evolution: Phase 9/10 (`variant_w10`) vs Phase 11/12/13 (`variant_v11`)
- In Phase 9 and 10, the candidate model `variant_w10` used $N_{\text{fft}} = 400$ ($201$ frequency bins) to match the window length exactly ($W_{\text{len}} = 400$). The Phase 91 and Phase 92 deployment work was commissioned specifically to evaluate on-device parity for that legacy baseline.
- In Phase 11, the architecture was retrained as `variant_v11` with hard-negative weighting and switched to standard power-of-2 FFT ($N_{\text{fft}} = 512$) with $W_{\text{len}} = 400$ and $H_{\text{len}} = 160$ ($257$ frequency bins). Power-of-2 FFTs are significantly faster and standard across optimized DSP hardware and mobile BLAS/vDSP backends.
- The Phase 92 report prose reflected the old baseline because Phase 92 concluded before Phase 11 was trained. It was never updated to describe Phase 11.

### 3.2 PyTorch STFT Window Padding Semantics
When $W_{\text{len}} = 400 < N_{\text{fft}} = 512$, PyTorch's `torch.stft` performs symmetric zero-padding of the Hann window itself:
$$\text{pad\_left} = \lfloor(512 - 400) / 2\rfloor = 56, \quad \text{pad\_right} = 56$$
The 512-sample padded window is applied to each 512-sample frame extracted from the reflect-padded waveform (where reflect-padding is $N_{\text{fft}} // 2 = 256$ on left and right).
When implemented identically in pure NumPy, numerical divergence between PyTorch and NumPy drops from $4.65$ to **$2.00 \times 10^{-5}$**.

---

## 4. Empirical Frontend Verification Results

The compatibility audit script [`experiments/vieneu_kws_research_v13_2/audit_frontend_compatibility.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_2/audit_frontend_compatibility.py) was executed across the non-locked development fixtures (`dev_01`, `dev_02`, `dev_05`, `dev_06`, `dev_07`).

### 4.1 Numerical Divergence Table

| Fixture Name | Preprocessing Frontend | Spec Shape | Spec MaxDiff vs Train | Logits MaxDiff vs Train | Prediction | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `dev_01_digital_silence.wav` | **Training (512)** | `[1, 1, 40, 101]` | `0.0 (Ref)` | `0.0 (Ref)` | `REJECT` | **REFERENCE** |
| | Remediated 13.1 (512) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `REJECT` | **EXACT MATCH** |
| | Phase 92 (400) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `REJECT` | MATCH |
| | Portable NumPy (512) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `REJECT` | PORTABLE MATCH |
| `dev_02_quiet_ambient_room.wav` | **Training (512)** | `[1, 1, 40, 101]` | `0.0 (Ref)` | `0.0 (Ref)` | `REJECT` | **REFERENCE** |
| | Remediated 13.1 (512) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `REJECT` | **EXACT MATCH** |
| | Phase 92 (400) | `[1, 1, 40, 101]` | `3.71e-01` | `3.78e-02` | `REJECT` | MATCH |
| | Portable NumPy (512) | `[1, 1, 40, 101]` | `3.81e-06` | `4.77e-07` | `REJECT` | PORTABLE MATCH |
| `dev_05_normal_len_peak089.wav` | **Training (512)** | `[1, 1, 40, 101]` | `0.0 (Ref)` | `0.0 (Ref)` | `XUONG` | **REFERENCE** |
| | Remediated 13.1 (512) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `XUONG` | **EXACT MATCH** |
| | Phase 92 (400) | `[1, 1, 40, 101]` | `1.55e+00` | `4.89e-02` | `XUONG` | MATCH |
| | Portable NumPy (512) | `[1, 1, 40, 101]` | `1.91e-05` | `7.15e-07` | `XUONG` | PORTABLE MATCH |
| `dev_06_normal_xuong_peak089.wav` | **Training (512)** | `[1, 1, 40, 101]` | `0.0 (Ref)` | `0.0 (Ref)` | `XUONG` | **REFERENCE** |
| | Remediated 13.1 (512) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `XUONG` | **EXACT MATCH** |
| | Phase 92 (400) | `[1, 1, 40, 101]` | `1.74e+00` | `1.17e-01` | `XUONG` | MATCH |
| | Portable NumPy (512) | `[1, 1, 40, 101]` | `2.00e-05` | `5.07e-07` | `XUONG` | PORTABLE MATCH |
| `dev_07_high_amp_trai_peak180.wav` | **Training (512)** | `[1, 1, 40, 101]` | `0.0 (Ref)` | `0.0 (Ref)` | `LEN` | **REFERENCE** |
| | Remediated 13.1 (512) | `[1, 1, 40, 101]` | `0.00e+00` | `0.00e+00` | `LEN` | **EXACT MATCH** |
| | Phase 92 (400) | `[1, 1, 40, 101]` | `1.32e+00` | `5.36e-02` | `LEN` | MATCH |
| | Portable NumPy (512) | `[1, 1, 40, 101]` | `1.53e-05` | `7.15e-07` | `LEN` | PORTABLE MATCH |

### 4.2 Key Findings
1. **Remediated Frontend Bitwise Identity:**  
   The remediated Phase 13.1 frontend produces identical log-Mel spectrograms ($\text{MaxDiff} = 0.00\text{e}+00$) and identical logits ($\text{MaxDiff} = 0.00\text{e}+00$) to the exact training module across all fixtures.
2. **Phase 92 Mismatch:**  
   The $N_{\text{fft}} = 400$ frontend introduces log-Mel spectral distortion up to $1.74$ and logit differences up to $0.117$. This confirms that running $N_{\text{fft}} = 400$ on `variant_v11` would constitute an unauthorized frontend perturbation.
3. **Cross-Platform Portability Parity:**  
   The NumPy portable frontend replicates PyTorch $N_{\text{fft}} = 512$ with $\le 2.00 \times 10^{-5}$ spectral divergence and zero decision discrepancies, providing an exact blueprint for future C++/mobile implementations.

---

## 5. Formal Clarification & Preregistration Decision

1. **Authoritative Frontend Confirmed:**  
   The authoritative, immutable frontend specification for `variant_v11_hard_negative_weighted_seed42.pt` is **$N_{\text{fft}} = 512$, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$** with 40 Slaney Mel filters ($20.0 - 8000.0\text{ Hz}$).
2. **Phase 92 Prose Deprecated for Phase 11/12/13 Models:**  
   The description in `PHASE92_CROSS_PLATFORM_PREPROCESSING_PARITY.md` stating $N_{\text{fft}} = 400$ is documented as historically limited to the superseded `variant_w10` baseline. It is formally rejected as non-authoritative for `variant_v11`.
3. **No Breaking Modifications Required:**  
   Because both `experiments/vieneu_kws_research_v13_1/bundle/infer.py` and `experiments/vieneu_kws_research_v13_2/protocol/AMENDED_CONFIRMATORY_PROTOCOL_V13_2.md` already specify and execute $N_{\text{fft}} = 512$, no code or protocol amendments are necessary.
4. **Gate Status:**  
   **`PRE_SYNTHESIS_GATE_PASSED`**. The synthesis pipeline in Phase 13.2 may proceed to generate audio using the preregistered TTS engines and protocol.

---

## 6. Checkpoint & Governance Invariants

- **Candidate Checkpoint:** `experiments/vieneu_kws_research_v12/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`
- **Candidate SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (Byte-identical, verified)
- **Authoritative Threshold:** $\theta^* = 0.20$ (Frozen)
- **Natural Speech Validation:** `NOT_VALIDATED_ON_NATURAL_SPEECH`
- **System Readiness:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
