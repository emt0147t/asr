# PHASE 71: REPOSITORY RELEASE AND REPRODUCIBILITY CHECKLIST
## Formal Verification of Repository Cleanliness, Cryptographic Integrity, and Release Readiness

**Repository:** `D:\trainidentificationaudio`  
**Date:** October 26, 2026  
**Evaluator:** Antigravity Research Governance Engine  
**Release Target:** Final Academic Research Closeout (Phase 70–72)  

---

## MANDATORY GOVERNANCE DECLARATIONS

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

1. **Synthetic-Only Research Boundary:** All metrics and checkpoints reflect synthetic audio mixtures and ESC-50 noise files.
2. **Defensive Model Retention:** Legacy $v11$ remains the sole default reference model.
3. **Locked Operating Threshold:** Calibrated command threshold override $\theta_{\text{cmd}} = 0.20$ is locked across all pipelines.
4. **Permanent Artifact Quarantine:** Phase 43, Phase 51, and Phase 59 evaluation corpora remain permanently quarantined from training.

---

## 1. Git Repository Status & Workspace Inspection

- **Current Git Branch:** `master`
- **Latest Commit Hash:** `471f1af` (`baseline: initial commit before end-to-end field verification`)
- **Working Tree Analysis:**
  - *Tracked Modified Files:* Code in `src/` and `tests/` preserved for compatibility.
  - *Untracked New Files:* New versioned research deliverables (`RESEARCH_FINAL_REPORT.md`, `FINAL_RESEARCH_SUMMARY.md`, `REPRODUCIBILITY_GUIDE.md`, audit logs, manifests, and catalogs).
  - *Secrets & Sensitive Data Inspection:* Zero API keys, credentials, or private tokens detected.
  - *Prohibited Data Inspection:* Zero human voice recordings or unauthorized third-party proprietary datasets present.

---

## 2. Checkpoint & ONNX Cryptographic Integrity Check

Every reference checkpoint and export ONNX model was independently verified against reference checksums:

| Component | File Path | Parameters | Reference SHA-256 Checksum | Actual Computed SHA-256 | Verification Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **v11 Checkpoint** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | 138,469 | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **MATCH (PASS)** |
| **v11 ONNX** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | 138,469 | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **MATCH (PASS)** |
| **E1 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | 138,469 | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **MATCH (PASS)** |
| **D2_H1 Checkpoint** | `experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/candidate_checkpoint.pt` | 138,469 | `c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8` | `c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8` | **MATCH (PASS)** |
| **D2_H1 ONNX** | `experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/vietnamese_kws_phase28_d2_candidate.onnx` | 138,469 | `442209e366af30f47b3168d453c88f169852fd47689207440b682fa12aebc246` | `442209e366af30f47b3168d453c88f169852fd47689207440b682fa12aebc246` | **MATCH (PASS)** |
| **Phase 26 D0_H1** | `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/candidate_checkpoint.pt` | 138,469 | `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3` | `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3` | **MATCH (PASS)** |
| **Candidate D3_C** | `experiments/vieneu_kws_research_v12/training_phase54_d3/D3_C_seed42/best_checkpoint.pt` | 138,469 | `da5b414328c33644b07c0667c3f9926637baa5f1672119f86393182c53c38ba6` | `da5b414328c33644b07c0667c3f9926637baa5f1672119f86393182c53c38ba6` | **MATCH (PASS)** |

---

## 3. Canonical Configuration & Inference Threshold Verification

- **Default Model Identifier:** `variant_v11_hard_negative_weighted_seed42` (Legacy $v11$).
- **Configuration Contract:** [`experiments/vieneu_kws_research_v13_4/bundle/model_config.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/model_config.json) confirms:
  ```json
  "decision_logic": {
    "policy": "argmax_over_commands_with_calibrated_threshold",
    "calibrated_theta_cmd": 0.2,
    "formula": "If max(prob[TRAI], prob[PHAI], prob[LEN], prob[XUONG]) >= 0.20 then argmax command else REJECT"
  }
  ```
- **Frontend Architecture:** [`LogMelFrontend`](file:///d:/trainidentificationaudio/frontend.py) (16,000 Hz, 400 window length, 160 hop length, 40 mel filterbanks, log compression) verified across all model pipelines.
- **Voice Activity Detection:** [`SpeechBandRatioVAD`](file:///d:/trainidentificationaudio/vad.py) (300–3,400 Hz band energy ratio $\ge 0.65$) verified fixed.

---

## 4. Test Corpus Quarantine & Data Isolation Verification

Cross-manifest waveform intersection audit between development manifests and historical evaluation corpora:

| Dataset / Split | Manifest Path | Sample Count | Overlap with Training Manifests | Quarantine Status |
| :--- | :--- | :---: | :---: | :---: |
| **Training (D2 Matched)** | `experiments/vieneu_kws_research_v12/manifests/D2_matched_train_v1.csv` | 5,440 | Baseline | Active Development Corpus |
| **Validation (v1)** | `experiments/vieneu_kws_research_v12/manifests/validation_v1.csv` | 1,280 | Disjoint | Active Development Corpus |
| **Phase 43 (MMS-TTS)** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_LOCKED_MANIFEST.csv` | 480 | **0 clips (0.00%)** | **PERFECTLY QUARANTINED** |
| **Phase 51 (gTTS)** | `experiments/vieneu_kws_research_v12/manifests/PHASE50_LOCKED_MANIFEST.csv` | 240 | **0 clips (0.00%)** | **PERFECTLY QUARANTINED** |
| **Phase 59 (Piper Track A)** | `experiments/vieneu_kws_research_v12/manifests/PHASE58_CONFIRMATORY_MANIFEST.csv` | 420 | **0 clips (0.00%)** | **PERFECTLY QUARANTINED** |
| **Phase 59 (Piper Track B)** | `experiments/vieneu_kws_research_v12/manifests/PHASE58_STREAMING_MANIFEST.csv` | 50 | **0 clips (0.00%)** | **PERFECTLY QUARANTINED** |

*Verdict:* Zero test waveform leakage into training. All held-out evaluation datasets remain cryptographically unexposed to model training routines.

---

## 5. Reproducibility Guide & Environment Readiness

- **Guide Location:** [`REPRODUCIBILITY_GUIDE.md`](file:///d:/trainidentificationaudio/REPRODUCIBILITY_GUIDE.md)
- **Verified Environment:** Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4 operational.
- **Reproduction Scripts Verified:**
  1. `run_phase64_forensic_validation_audit.py` -> Reproduces 1,280 aligned rows and exact metrics for all 8 models.
  2. `eval_phase59_confirmatory_test.py` -> Reproduces Track A (420 clips) and Track B (50 streaming sequences) metrics.
  3. Clustered paired permutation / $t$-test script -> Reproduces $df = 104$ statistical bounds.

---

## 6. Master Artifact Catalog Summary

The catalog of all 28 key deliverables is compiled in [`PHASE71_FINAL_ARTIFACT_CATALOG.csv`](file:///d:/trainidentificationaudio/PHASE71_FINAL_ARTIFACT_CATALOG.csv):
- 7 Frozen Checkpoints and ONNX Models (all SHA-256 verified)
- 6 Primary and Historical Research Reports
- 4 Audit Guides and Evidence Ledgers
- 6 Quarantined Dataset and Streaming Manifests
- 5 Raw Prediction and Aligned Prediction Matrices

---

## 7. Versioned Git Commit & Tag Proposal

The repository is fully auditable, consistent, and free of sensitive information.

### Proposed Git Tag:
`v1.0.0-research-release-p72`

### Proposed Commit Header & Message:
```text
release(research): final scientific report, reproducibility package, and governance freeze

- Synchronize consolidated research reports (RESEARCH_FINAL_REPORT.md, FINAL_RESEARCH_SUMMARY.md)
- Complete Phase 70 forensic report evidence audit (PHASE70_FINAL_REPORT_VERIFICATION.md)
- Complete Phase 71 release checklist and final artifact catalog (PHASE71_FINAL_ARTIFACT_CATALOG.csv)
- Complete Phase 72 defense package with talk track and defense Q&A
- Maintain legacy v11 as default reference model with locked theta_cmd = 0.20
- Lock all frozen checkpoints, quarantined test manifests, and raw evaluation ledgers
- Enforce mandatory governance labels:
  EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
  NOT_VALIDATED_ON_NATURAL_SPEECH
  NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
  NOT_VALIDATED_PENDING_DATA
```

*Note on Execution:* In strict compliance with guidelines, no destructive git operations or remote pushes were executed.

---

## 8. Release Verification Verdict

**Selected Phase 71 Release Gate:**  
$$\mathbf{PASS\_FINAL\_RESEARCH\_PACKAGE\_VERIFIED}$$
