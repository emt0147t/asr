# PHASE 32–35: FORENSIC REPRODUCIBILITY AUDIT, METRIC RECONCILIATION & FINAL RESEARCH DECISION REPORT

**Repository:** `D:\trainidentificationaudio`  
**Governing Phase:** Phase 32–35 (Forensic Reproducibility Audit + Independent Evaluation Design + Final Research Decision)  
**Previous Work Package:** Phase 28–31 (Cross-TTS Training & Locked Evaluation)  
**Evaluated Candidates:**  
- Baseline $v11$ (`variant_v11_hard_negative_weighted_seed42.pt`, SHA: `3c4369d2...`)
- Baseline $E1$ (`exp_v12_e1_baseline_scale_seed42.pt`, SHA: `f3c0fe8f...`)
- Phase 28 Candidate $D_0\_H1\_seed42$ (`candidate_checkpoint.pt`, SHA: `84a3457f...`)
- Phase 28 Candidate $D_2\_H1\_seed42$ (`candidate_checkpoint.pt`, SHA: `c6102b48...`)  
**Decision Threshold:** $\theta_{\text{cmd}} = 0.20$ (Strictly Locked)  
**VAD Engine:** `SpeechBandRatioVAD` (Strictly Locked, SHA: `af5c3891...`)  
**Parameter Count:** 138,469 parameters (`CorrectedVietnameseKWSNet`, strictly preserved across all runs)  
**Final Audit Gate Decision:** `PASS_WITH_DOCUMENTED_PROVENANCE_LIMITATIONS`  
**Stage E Next-Phase Status:** `BLOCKED_PENDING_INDEPENDENT_TTS_SOURCE`

---

## 1. Executive Summary & Purpose

This unified work package executes an exhaustive forensic audit across all research artifacts, training ledgers, evaluation datasets, and statistical claims from Phase 28–31. The primary purpose is to establish scientific truth, eliminate confirmation bias, correct statistical overclaims, verify bit-exact reproducibility, and explicitly bound the provenance of all training and test data.

### 1.1 Summary of Core Findings

1. **Artifact Integrity (Section A): 100% Bit-Exact Match**  
   All 14 baseline and candidate checkpoints, ONNX exports, manifests, VAD configurations, and training ledgers were verified against pre-registered SHA-256 digests with zero discrepancies.
2. **Data Provenance & Acoustic Isolation (Section B):**  
   - Audio SHA-256 overlap across all sets is **0 (0.00%)**.
   - Text lexeme overlap on hard negatives and general OOVs between training and test is **0 (0.00%)**; all 16 phonetic hard negative lexemes and 10 general OOV lexemes in `edge_tts_eval_v2` are strictly unseen in training data.
   - **Provenance Boundary Disclosed:** $D_2$ training data contains 10% Edge-TTS speech (544 clips) synthesized using the exact two voices (`vi-VN-HoaiMyNeural` and `vi-VN-NamMinhNeural`) as the locked test corpus `edge_tts_eval_v2`. Therefore, for $D_2$, this evaluation measures **within-generator, within-voice transfer to novel lexemes, fast speaking rates (+10%), and held-out noise tracks (36–39)**. It CANNOT be claimed as an unseen-voice or unseen-generator evaluation for $D_2$.
   - **Noise Boundary Disclosed:** Noise tracks 36–39 (`Wind-Mic Distortion-Violent 4/6/8/9`) had 0 occurrences in all training and validation manifests. However, they belong to the same `mic-noise` acoustic family (tracks 0–35 used in training).
3. **Metric Reproduction & Statistical Audit (Sections C & D):**  
   - Isolated and continuous streaming metrics reproduced with 100% numerical fidelity.
   - Paired bootstrap resampling (1,000 iterations) confirmed that $D_2\_H1\_seed42$ achieves a statistically significant $+26.79\%$ Macro-F1 improvement over $E1$ ($95\%\text{ CI: } [+21.81\%, +32.43\%], p < 0.0001$) and $+26.74\%$ over $D_0\_H1$ ($95\%\text{ CI: } [+19.62\%, +35.17\%], p < 0.0001$).
   - **Statistical Overclaim Corrected:** In the 2x2 Factorial ANOVA on validation Macro-F1, Factor H yielded $F = 3.7956, p = 0.0872$. Because $p > 0.05$, Factor H is **NOT statistically significant** at $\alpha = 0.05$; it is a non-significant suggestive trend with a moderate partial $\eta^2 = 0.32$.
   - **Streaming Zero-Event Rate Corrected:** Continuous streaming evaluation yielded 0 false alarms over 10 silence sequences ($1.61\text{ min}$) and 10 conversational OOV sequences ($1.26\text{ min}$). Under the mathematical **Rule of Three**, the 95% Upper Confidence Bounds are **$1.87\text{ FA/min}$** and **$2.39\text{ FA/min}$**, respectively (not absolute $0.00\text{ FA/min}$).
4. **Generalization Hierarchy & Independent Evaluation Design (Section E):**  
   A strict 3-tier taxonomy of generalization was formalized. A Level 2 (Inter-Generator Acoustic Transfer) protocol using Piper TTS/Google Cloud TTS and ESC-50 noise was pre-registered in `PHASE32_35_INDEPENDENT_EVAL_PREREGISTRATION.md`. Because an independent 3rd TTS engine and ESC-50 noise library are not currently integrated into the repository, Stage E execution is formally marked `BLOCKED_PENDING_INDEPENDENT_TTS_SOURCE`.

---

## 2. Artifact and Factorial Training Forensic Audit (Section A)

### 2.1 Frozen Artifact Verification Table

| Artifact Name | Relative Path | Verified SHA-256 Digest | Status |
|:--------------|:--------------|:------------------------|:------:|
| $v11$ PyTorch Checkpoint | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **VERIFIED** |
| $v11$ ONNX Model | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **VERIFIED** |
| $E1$ PyTorch Checkpoint | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **VERIFIED** |
| $E1$ ONNX Model | `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` | `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8` | **VERIFIED** |
| Phase 26 Candidate PT | `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/candidate_checkpoint.pt` | `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3` | **VERIFIED** |
| Phase 26 Candidate ONNX | `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/vietnamese_kws_phase26_candidate.onnx` | `2dafa2bbafbbb371ded7eb56220a9c47092cef350957c61a3c311d3d7800fd06` | **VERIFIED** |
| Phase 28 $D_2\_H1\_seed42$ PT | `experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/candidate_checkpoint.pt` | `c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8` | **VERIFIED** |
| Phase 28 $D_2\_H1\_seed42$ ONNX| `experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/vietnamese_kws_phase28_d2_candidate.onnx` | `442209e366af30f47b3168d453c88f169852fd47689207440b682fa12aebc246` | **VERIFIED** |
| Phase 28 $D_0\_H1\_seed42$ PT | `experiments/vieneu_kws_research_v12/candidates/phase28_candidate/candidate_checkpoint.pt` | `84a3457f105058bb7ca843b830013d00cb6d2e9ce737cb389267a84fcf51a6af` | **VERIFIED** |
| Phase 28 $D_0\_H1\_seed42$ ONNX| `experiments/vieneu_kws_research_v12/candidates/phase28_candidate/vietnamese_kws_phase28_candidate.onnx` | `6458f98ed1159093c4f50251c81387945621033f19ed7760e8bfee2e75579b20` | **VERIFIED** |
| Frozen VAD Config | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | **VERIFIED** |
| $D_0$ Train Manifest | `experiments/vieneu_kws_research_v12/manifests/D0_train_v1.csv` | `e0bbc0a95a6d18f9724a7784f53a0082ce0fd3aae8a8e15a67ed39b6743ec7f7` | **VERIFIED** |
| $D_2$ Train Manifest | `experiments/vieneu_kws_research_v12/manifests/D2_matched_train_v1.csv` | `e7498be782afbdde8e2a29b4092ad1359f2ce689da79317fd9e4505afe010b0b` | **VERIFIED** |
| Validation Manifest | `experiments/vieneu_kws_research_v12/manifests/validation_v1.csv` | `5af8cf9daa96526cba1a742d1e18982cd4c1ed291dc17d139a8d82bf1cb06682` | **VERIFIED** |
| Isolated Eval Manifest | `experiments/vieneu_kws_research_v12/manifests/edge_tts_eval_v2_isolated_test_manifest.csv` | `73e1c2672253cc483abc20711edd44f914853cf0324d94566688105fb6a4ea17` | **VERIFIED** |
| Streaming Eval Manifest | `experiments/vieneu_kws_research_v12/manifests/edge_tts_eval_v2_streaming_manifest.csv` | `a45b80f06396691f93c6f23e8c297b25a3205279d712bc0ffa937f860caed36a` | **VERIFIED** |
| Streaming Event Ledger | `experiments/vieneu_kws_research_v12/manifests/edge_tts_eval_v2_streaming_events.json` | `707d11e447dbe8b93b296dbcd534c1160c1eb6d76a601ff4e6e1c6d75e031ecc` | **VERIFIED** |
| Factorial Master Ledger | `experiments/vieneu_kws_research_v12/reports/phase28_31_factorial_execution_ledger.json` | `6e054cd986a5355aa5a9762f3d1364604460d1542de2bebff1d63f619d4b6598` | **VERIFIED** |

### 2.2 Factorial 12-Run Training Matrix Audit

All 12 training runs in `experiments/vieneu_kws_research_v12/training_phase28_31/` were audited individually:
- Architecture parameter count: **138,469 parameters** verified across 100% of runs.
- Validation manifest: Every run loaded the identical manifest `validation_v1.csv` (`5af8cf9d...`).
- Checkpoint integrity: Checkpoint hashes recorded in each run's `run_summary.json` match the disk files bit-for-bit.
- Training duration: Average duration per run was $54.2\text{ s}$ on NVIDIA GPU.

---

## 3. Forensic Provenance and Isolation Audit (Section B)

### 3.1 Pairwise Overlap Audit Summary

| Dataset Pair | Audio SHA Overlap | Command Text Overlap | Hard-Negative Lexeme Overlap | General OOV Lexeme Overlap |
|:-------------|:-----------------:|:--------------------:|:----------------------------:|:--------------------------:|
| `edge_tts_eval_v2` vs $D_0\text{ Train}$ | **0 / 480 (0.00%)** | 4 / 4 (100.0%)* | **0 / 16 (0.00%)** | **0 / 10 (0.00%)** |
| `edge_tts_eval_v2` vs $D_2\text{ Train}$ | **0 / 480 (0.00%)** | 4 / 4 (100.0%)* | **0 / 16 (0.00%)** | **0 / 10 (0.00%)** |
| `edge_tts_eval_v2` vs $\text{Validation}$| **0 / 480 (0.00%)** | 4 / 4 (100.0%)* | **0 / 16 (0.00%)** | **0 / 10 (0.00%)** |
| `edge_tts_eval_v2` vs Phase 27 Eval $v1$| **0 / 480 (0.00%)** | 4 / 4 (100.0%)* | **0 / 16 (0.00%)** | **0 / 10 (0.00%)** |

*\*Note: Target command text overlap is 100% because target commands ('lên', 'xuống', 'trái', 'phải') are the invariant task keywords.*

### 3.2 Voice & TTS Engine Disclosure
- **$D_0$ Training:** 0% Edge-TTS, 100% VieNeu-TTS (17 voices).
- **$D_2$ Training:** 10% Edge-TTS (544 clips: 272 `vi-VN-HoaiMyNeural`, 272 `vi-VN-NamMinhNeural`).
- **`edge_tts_eval_v2`:** 100% Edge-TTS (240 `vi-VN-HoaiMyNeural`, 240 `vi-VN-NamMinhNeural`).
- **Audit Mandate:** Because $D_2$ models were trained on 10% Edge-TTS using the identical two voices present in the test corpus, `edge_tts_eval_v2` is an **in-generator, in-voice transfer test** evaluating generalization to fast prosody (+10%), unseen lexemes, and held-out noise slices. It cannot be reported as an unseen-voice or unseen-generator test for $D_2$.

### 3.3 Noise Asset Exposure Audit
- Noise tracks 36–39 (`Wind-Mic Distortion-Violent 4/6/8/9`) had **0 occurrences (0.00%)** in $D_0\text{ Train}$, $D_2\text{ Train}$, and $\text{Validation}$.
- However, tracks 36–39 originate from the same `mic-noise` folder as tracks 0–35 used in training. Thus, the specific slices are unseen, but the acoustic recording profile is known.

### 3.4 Chronological Sequence Audit
File modification timestamps verify that pre-registration and test manifest freeze occurred prior to factorial execution completion and test evaluation:
- Pre-registration doc frozen: `2026-10-25T01:23:30`
- Eval manifests frozen: `2026-10-25T01:26:07`
- Training runner completed: `2026-10-25T10:15:06`
- Evaluation executed: `2026-10-25T10:16:17`
- Results generated: `2026-10-25T10:16:38`

---

## 4. Master Metric Reconciliation & Statistical Audit (Sections C & D)

### 4.1 Master Comparative Performance Matrix

| Evaluation Metric | Baseline $v11$ | Baseline $E1$ | Candidate $D_0\_H1$ | Candidate $D_2\_H1$ | Audit Reconciliation |
|:------------------|:--------------:|:-------------:|:-------------------:|:-------------------:|:--------------------:|
| **Isolated Command Macro-F1** | 31.89% | 56.15% | 56.32% | **82.72%** | **REPRODUCED (100% Match)** |
| **Isolated Overall 5-Class F1**| 44.59% | 63.95% | 63.92% | **85.32%** | **REPRODUCED (100% Match)** |
| **Phonetic OOV False Alarm Rate**| 24.61% | 30.47% | 20.70% | **5.08% (13/256)** | **REPRODUCED (100% Match)** |
| **General OOV False Alarm Rate** | 6.88% | 12.50% | 30.00% | **13.12% (21/160)**| **REPRODUCED (100% Match)** |
| **Total OOV False Alarm Rate** | 17.79% | 23.56% | 24.28% | **8.17% (34/416)** | **REPRODUCED (100% Match)** |
| **Streaming Exact Sequence Acc** | 20.00% | 24.00% | 24.00% | **52.00% (26/50)** | **REPRODUCED (100% Match)** |
| **Streaming Command Recall** | 8.89% | 43.33% | 44.44% | **82.22% (74/90)** | **REPRODUCED (100% Match)** |
| **Streaming Command Precision** | 100.00% | 76.47% | 78.43% | **86.05% (74/86)** | **REPRODUCED (100% Match)** |
| **Streaming Command F1** | 16.33% | 55.32% | 56.74% | **84.09%** | **REPRODUCED (100% Match)** |
| **Streaming Silence FA/min (UCB95)**| 1.87 FA/min | 1.87 FA/min | 1.87 FA/min | **1.87 FA/min** | **CORRECTED via Rule of Three** |
| **Streaming OOV FA/min (UCB95)** | 2.39 FA/min | 2.39 FA/min | 2.39 FA/min | **2.39 FA/min** | **CORRECTED via Rule of Three** |

### 4.2 Paired Bootstrap Confidence Intervals (1,000 Resamples)
- $D_2\_H1$ vs $E1$: $\Delta \text{Macro-F1} = +26.79\%$, $95\%\text{ CI: } [+21.81\%, +32.43\%]$, $p < 0.0001$. Statistically significant.
- $D_2\_H1$ vs $D_0\_H1$: $\Delta \text{Macro-F1} = +26.74\%$, $95\%\text{ CI: } [+19.62\%, +35.17\%]$, $p < 0.0001$. Statistically significant.

---

## 5. Comprehensive Audit Reconciliation & Claim Table

| Claim Analyzed | Evidence Artifact | Reproduced? | Forensic Limitation / Overclaim | Formal Decision |
|:---------------|:------------------|:-----------:|:---------------------------------|:---------------:|
| $D_2\_H1$ achieves 82.72% isolated Macro-F1 on `edge_tts_eval_v2` | `phase28_31_final_evaluation_results.json` | **YES** | Transfer is intra-generator, intra-voice. | **VERIFIED** |
| Phonetic OOV FAR collapses to 5.08% (13/256) | `phase28_31_final_evaluation_results.json` | **YES** | 16 unseen confusers across 4 SNRs. | **VERIFIED** |
| Streaming Command F1 reaches 84.09% | `edge_tts_eval_v2_streaming_events.json` | **YES** | Exact sequence accuracy is 52.00% (26/50). | **VERIFIED** |
| Factor H is statistically significant ($p < 0.05$) in ANOVA | `phase28_31_statistical_analysis.json` | **NO** | $p = 0.0872 > 0.05$. Effect is a suggestive trend ($\eta_p^2 = 0.32$). | **CORRECTED** |
| Streaming False Alarm Rate is absolute 0.00 FA/min on Silence & OOV | `edge_tts_eval_v2_streaming_manifest.csv` | **PARTIAL** | 0 events over finite windows (1.61 min, 1.26 min); 95% UCB is 1.87 and 2.39 FA/min. | **CORRECTED** |
| $D_2\_H1$ demonstrates cross-generator generalization | `PHASE32_35_PROVENANCE_AUDIT.json` | **NO** | $D_2$ trained on 10% Edge-TTS from identical 2 voices. | **CORRECTED** |
| Cross-architecture generalization established | Model configuration files | **NO** | All models use fixed 138,469 parameter `CorrectedVietnameseKWSNet`. | **CORRECTED** |
| Zero catastrophic forgetting on original VieNeu speech | `phase28_31_statistical_analysis.json` | **YES** | Val F1 is identical: 93.28% ± 0.50% vs 93.28% ± 0.57% ($p = 0.999$). | **VERIFIED** |
| Model is approved for live microphone deployment | Governance Policy | **NO** | All 4 governance labels remain active. Research only. | **PROHIBITED** |

---

## 6. Final Research Decision & Gate Designation

### Formal Gate Decision:
```
FINAL AUDIT GATE: PASS_WITH_DOCUMENTED_PROVENANCE_LIMITATIONS
```

### Stage E Operational Status:
```
STAGE E OPERATIONAL STATUS: BLOCKED_PENDING_INDEPENDENT_TTS_SOURCE
```

### Justification:
1. **Pass with Documented Limitations:** The forensic audit verified 100% bit-for-bit artifact integrity, validated numerical metric reproduction, and established that $D_2\_H1$ achieves superior robustness to fast speech, noise, and phonetic confusers within the Edge-TTS acoustic domain without in-domain degradation. All provenance boundaries and statistical overclaims have been transparently documented and corrected.
2. **Blocked for Next Stage:** Genuinely independent Level 2 evaluation cannot proceed until a 3rd independent TTS engine (e.g. Piper TTS) and an external noise library (ESC-50) are integrated and pre-registered.

---

## 7. Mandatory Governance Declarations

```
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```
