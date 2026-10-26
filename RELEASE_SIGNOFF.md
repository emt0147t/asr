# VIETNAMESE KEYWORD SPOTTING — FINAL RELEASE SIGNOFF
## Formal Record of Dedicated Research-Release Commit, Tag Verification, and Repository Governance Freeze

**Repository Path:** `D:\trainidentificationaudio`  
**Execution Timestamp:** October 26, 2026 (04:12:38 Local Time)  
**Evaluator & Governance Agent:** Antigravity Research Governance Engine  
**Final Governance Closeout Gate:** `PASS_WITH_DOCUMENTED_REPORT_CORRECTIONS`  

---

## MANDATORY GOVERNANCE DECLARATIONS

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

1. **Synthetic-Only Boundary:** All models, metrics, and experimental inferences derive strictly from synthetic text-to-speech audio mixtures and standard environmental noise files.
2. **Defensive Baseline Retention:** Legacy baseline model $v11$ ([`vietnamese_kws_v11.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)) remains the immutable default reference model.
3. **Locked Operating Decision Policy:** Calibrated command threshold override $\theta_{\text{cmd}} = 0.20$ is strictly fixed across all production and evaluation configurations.
4. **No Deployment or Natural-Speech Claim:** Zero human audio was collected, zero physical microphone testing was conducted, and no system is approved for production deployment.

---

## 1. Release Commit & Tag Verification Metadata

| Attribute | Recorded Verification Value |
| :--- | :--- |
| **Git Branch** | `master` |
| **Dedicated Release Commit Hash** | `928fc85db7e8992c0d8fb253d7fc8bc2bb313fa2` (Short: `928fc85`) |
| **Commit Subject** | `release(research): final scientific report, reproducibility package, and governance freeze` |
| **Local Annotated Tag** | `v1.0.0-research-release-p72` |
| **Tag Object Hash** | `8e9db27a7972afde77441d0fa9660878decdfcb6` |
| **Tag Target Commit Hash** | `928fc85db7e8992c0d8fb253d7fc8bc2bb313fa2` (Verified Match with `HEAD`) |
| **Tag Annotation Message** | `"Vietnamese KWS final research release: verified artifacts, reproducibility guides, defense package, and governance freeze"` |
| **Working Tree Status** | Clean (`nothing to commit, working tree clean` immediately verified post-tagging) |
| **Remote Action** | Strictly Withheld (Zero push executed; local release tag only) |

---

## 2. Exact Git Commands Executed During Release Operation

```powershell
# 1. Inspect repository state, branch, and commit history
git status
git branch
git log -1 --oneline
git tag -l

# 2. Update .gitignore to exclude virtual environments, raw WAV audio files, and temporary directories
# (Ensures 0 secrets, credentials, or large binary datasets enter git history)

# 3. Stage all verified research deliverables, reports, manifests, tests, and models
git add .

# 4. Create dedicated research-release commit
git commit -m "release(research): final scientific report, reproducibility package, and governance freeze

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
  NOT_VALIDATED_PENDING_DATA"

# 5. Create local annotated release tag
git tag -a v1.0.0-research-release-p72 -m "Vietnamese KWS final research release: verified artifacts, reproducibility guides, defense package, and governance freeze"

# 6. Verify commit hash, tag target, tag annotation, and clean working tree
git show-ref --tags -d
git status
```

---

## 3. Cryptographic Verification of Frozen Checkpoints & ONNX Models

Every reference model was independently hashed and matched against authoritative values:

| Model Identifier | Relative File Path | Architecture Parameters | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Legacy v11 (Default)** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | 138,469 | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` |
| **Legacy v11 ONNX** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | 138,469 | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` |
| **Baseline E1 Frozen** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | 138,469 | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` |
| **Phase 28 D2_H1 Candidate** | `experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/candidate_checkpoint.pt` | 138,469 | `c6102b4876ddeaa453547c211c5cfc322511c957e18fd488a705a164b18601c8` |
| **Phase 28 D2_H1 ONNX** | `experiments/vieneu_kws_research_v12/candidates/phase28_d2_candidate/vietnamese_kws_phase28_d2_candidate.onnx` | 138,469 | `442209e366af30f47b3168d453c88f169852fd47689207440b682fa12aebc246` |
| **Phase 26 D0_H1 seed44** | `experiments/vieneu_kws_research_v12/candidates/phase26_candidate_d0_h1_seed44/candidate_checkpoint.pt` | 138,469 | `00b4fc71f863788648d5ea388a99c4cbd4441d9b9f63adc5e56154e329df63f3` |
| **Candidate D3_C seed42** | `experiments/vieneu_kws_research_v12/training_phase54_d3/D3_C_seed42/best_checkpoint.pt` | 138,469 | `da5b414328c33644b07c0667c3f9926637baa5f1672119f86393182c53c38ba6` |

---

## 4. Test Manifest Quarantine Verification

Zero test sample leakage into training sets was cryptographically verified:
- `PHASE43_LOCKED_MANIFEST.csv` (480 clips): **0 overlap (0.00%)** with training.
- `PHASE50_LOCKED_MANIFEST.csv` (240 clips): **0 overlap (0.00%)** with training.
- `PHASE58_CONFIRMATORY_MANIFEST.csv` (420 clips): **0 overlap (0.00%)** with training.
- `PHASE58_STREAMING_MANIFEST.csv` (50 sequences): **0 overlap (0.00%)** with training.

---

## 5. Core Scientific Findings & Evidence Summary

1. **Validation Performance ($N=1,280$ rows, $\theta_{\text{cmd}}=0.20$):**
   - $v11$ accuracy: **$78.05\%$** ($999/1,280$), 4-cmd Macro-F1 **$79.45\%$**, Min Recall **$78.75\%$** (`TRAI`), OOV FAR **$34.22\%$** ($219/640$).
   - Historical claim of $88.59\%$ accuracy ($1,134/1,280$) was a copy-paste error from Candidate $D_3\_C$'s REJECT recall ($567/640 = 88.59375\%$).
   - Highest observed accuracy in discrete threshold sweep was $80.55\%$ at $\theta=0.28$. Operational threshold remains locked at $\theta=0.20$.
2. **Confirmatory Test (Phase 59 Piper VAIS-1000):**
   - $D_2\_H1$ passed Low-FA point gates ($5.77\%$ isolated FAR, $1.85\text{ FA/min}$ streaming).
   - $D_2\_H1$ suffered severe sensitivity collapse: `TRAI` recall was **$22.50\%$** ($9/40$), streaming command recall was **$17.24\%$** ($10/58$), and Event F1 was **$19.23\%$**.
   - *Core Takeaway:* Strong OOV rejection does not imply strong keyword recall.
3. **Statistical Inference ($\alpha=0.01$):**
   - $D_2\_H1$ vs $D_3\_C$ ($+5.24\text{ pp}$, $p=0.0336$) is **NOT statistically significant** at preregistered $\alpha=0.01$ ($0.0336 > 0.01$).
4. **Exact Poisson Uncertainty Bounds:**
   - Silence ($k=0, T=0.86\text{m}$): $\text{UCB}_{95\%} = \mathbf{3.48\text{ FA/min}}$.
   - Conversational OOV ($k=1, T=0.54\text{m}$): $\text{UCB}_{95\%} = \mathbf{8.76\text{ FA/min}}$.

---

## 6. Release Artifact Inventory

All release artifacts are indexed and permanent:
- [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md) — 13-section consolidated research report
- [`FINAL_RESEARCH_SUMMARY.md`](file:///d:/trainidentificationaudio/FINAL_RESEARCH_SUMMARY.md) — Executive summary
- [`REPRODUCIBILITY_GUIDE.md`](file:///d:/trainidentificationaudio/REPRODUCIBILITY_GUIDE.md) — End-to-end reproduction guide
- [`FINAL_CLAIM_EVIDENCE_TABLE.csv`](file:///d:/trainidentificationaudio/FINAL_CLAIM_EVIDENCE_TABLE.csv) — 10-claim evidence table
- [`PHASE70_CLAIM_DISCREPANCY_LEDGER.csv`](file:///d:/trainidentificationaudio/PHASE70_CLAIM_DISCREPANCY_LEDGER.csv) — 13-claim forensic ledger
- [`FINAL_SHA256_MANIFEST.json`](file:///d:/trainidentificationaudio/FINAL_SHA256_MANIFEST.json) — Cryptographic checksum ledger
- [`PHASE71_FINAL_ARTIFACT_CATALOG.csv`](file:///d:/trainidentificationaudio/PHASE71_FINAL_ARTIFACT_CATALOG.csv) — 28-item master catalog
- [`PHASE71_RELEASE_CHECKLIST.md`](file:///d:/trainidentificationaudio/PHASE71_RELEASE_CHECKLIST.md) — Release verification checklist
- [`PHASE72_DEFENSE_SCRIPT.md`](file:///d:/trainidentificationaudio/PHASE72_DEFENSE_SCRIPT.md) — 8–10 min Vietnamese talk track
- [`PHASE72_DEFENSE_QA.md`](file:///d:/trainidentificationaudio/PHASE72_DEFENSE_QA.md) — 20 evidence-based defense Q&As

---

## 7. Sign-off Declaration

The Vietnamese KWS research repository `D:\trainidentificationaudio` is formally sealed, cryptographically verified, and released under tag `v1.0.0-research-release-p72`. No further commits, modifications, or remote pushes are authorized.
