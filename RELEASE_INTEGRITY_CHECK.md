# RELEASE INTEGRITY CHECK: FINAL POST-RELEASE AUDIT
## Comprehensive Verification of Git Tree, Tag Lineage, Manifest Checksums, and Model Invariants

**Repository:** `D:\trainidentificationaudio`  
**Execution Timestamp:** October 26, 2026 (11:00:00 Local Time)  
**Evaluator & Governance Engine:** Antigravity Research Governance Engine  
**Release Audit Status:** Fully Audited, Reconciled, and Sealed  

---

## MANDATORY GOVERNANCE DECLARATIONS

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

1. **Synthetic-Only Research Boundary:** All checkpoints, predictions, metrics, and risk assessments are strictly bounded to synthetic text-to-speech audio mixtures and standard environmental noise files.
2. **Defensive Baseline Retention:** Legacy baseline model $v11$ ([`vietnamese_kws_v11.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt), [`vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx)) remains the sole default reference model.
3. **Locked Operating Decision Policy:** Calibrated command threshold override $\theta_{\text{cmd}} = 0.20$ is strictly fixed across all production and evaluation configurations.
4. **No Deployment or Natural-Speech Claim:** Zero human audio was collected, zero physical microphone testing was conducted, and no system is approved for production deployment.

---

## 1. Initial Release Reference Audit (`v1.0.0-research-release-p72`)

Inspection of the initial release commit and tag yielded the following exact Git references:

| Git Attribute | Reference Value | Audit Finding |
| :--- | :--- | :--- |
| **Current Branch** | `master` | Nominal tracking branch |
| **Base Release Commit** | `928fc85db7e8992c0d8fb253d7fc8bc2bb313fa2` (Short: `928fc85`) | Verified initial release commit |
| **Annotated Tag** | `v1.0.0-research-release-p72` | Verified present; not overwritten |
| **Tag Object Hash** | `8e9db27a7972afde77441d0fa9660878decdfcb6` | Verified annotated tag object |
| **Tag Target Commit** | `928fc85db7e8992c0d8fb253d7fc8bc2bb313fa2` | Verified match with base release commit |
| **Tree Check (`RELEASE_SIGNOFF.md`)** | `git ls-tree -r --name-only v1.0.0-research-release-p72 -- RELEASE_SIGNOFF.md` | **EMPTY (Absent from initial tag)** |
| **Manifest Check (`FINAL_SHA256_MANIFEST.json`)** | Root `RESEARCH_FINAL_REPORT.md` entry: `7b0c6865...` | **STALE HASH (Pre-synchronization value)** |

### Forensic Explanation of Discrepancies:
1. `RELEASE_SIGNOFF.md` was created immediately following the base commit and tag creation in Phase 71 to record the resulting commit hash, leaving it untracked in the working tree.
2. In Phase 70, root `RESEARCH_FINAL_REPORT.md` was synchronized with the canonical full report (hash: `c90c0d504df8d4ea...`), but `FINAL_SHA256_MANIFEST.json` retained the pre-synchronization hash (`7b0c6865c8abcd...`) for the root path.

---

## 2. Working Tree State & Ignored Patterns Inspection

```powershell
# 1. Short status
git status --short
# (Pre-follow-up commit: ?? RELEASE_SIGNOFF.md; Post-follow-up commit: clean)

# 2. Ignored patterns status
git status --short --ignored
```

- **Clean Working Tree:** Post-reconciliation working tree contains zero untracked or modified files.
- **Ignored Artifacts:**
  - Virtual environments: `.venv/`, `.venv_vieneu/` (properly ignored via `.venv*/`).
  - Raw audio waveforms: All `*.wav` files (2.95 GB across experiments and data directories) cleanly excluded from Git tracking.
  - Large binary datasets: `data/raw/`, `data/processed/`, `data/candidate_tts/`, `data/backups/` properly ignored.
  - Test and compiler caches: `.pytest_cache/`, `__pycache__/`, `*.pyc`.
- **Ignore Check:** `git check-ignore -v RELEASE_SIGNOFF.md` confirmed that `RELEASE_SIGNOFF.md` is NOT ignored and is intended for repository tracking.

---

## 3. Sealed Follow-Up Release Lineage (`v1.0.1-research-release-p72`)

In strict accordance with post-release governance guidelines:
1. Tag `v1.0.0-research-release-p72` was **NOT amended, moved, or deleted**.
2. A dedicated follow-up commit was created containing strictly the intended release-record files:
   - [`RELEASE_SIGNOFF.md`](file:///d:/trainidentificationaudio/RELEASE_SIGNOFF.md) (release signoff record)
   - [`FINAL_SHA256_MANIFEST.json`](file:///d:/trainidentificationaudio/FINAL_SHA256_MANIFEST.json) (corrected root report hash)
   - [`experiments/vieneu_kws_research_v12/reports/FINAL_SHA256_MANIFEST.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/FINAL_SHA256_MANIFEST.json)
   - [`PHASE71_FINAL_ARTIFACT_CATALOG.csv`](file:///d:/trainidentificationaudio/PHASE71_FINAL_ARTIFACT_CATALOG.csv) (updated with signoff document and new manifest hash)
3. Zero unrelated or unexplained changes were staged.
4. An annotated sealed release tag was created: `v1.0.1-research-release-p72`.

### Sealed Release Metadata:
- **Sealed Release Commit Hash:** `d5af658a9afe523b306e5ca6c0159789f1c67fad` (Short: `d5af658`)
- **Sealed Annotated Tag:** `v1.0.1-research-release-p72`
- **Tag Object Hash:** `e596ef0c420b0a76a0e589f1fe192cf696794dc1`
- **Tag Target Commit Hash:** `d5af658a9afe523b306e5ca6c0159789f1c67fad` (Verified match with `HEAD`)
- **Tag Annotation Message:** `"Vietnamese KWS final sealed research release: includes RELEASE_SIGNOFF.md, synchronized manifest hashes, and governance freeze"`
- **Tag Tree Verification:**
  ```powershell
  git ls-tree -r --name-only v1.0.1-research-release-p72 -- RELEASE_SIGNOFF.md
  # Output: RELEASE_SIGNOFF.md (VERIFIED PRESENT)
  ```

---

## 4. Manifest Integrity Audit (28 of 28 Items Matched)

Every entry in [`FINAL_SHA256_MANIFEST.json`](file:///d:/trainidentificationaudio/FINAL_SHA256_MANIFEST.json) was audited against actual disk files:

| Artifact Path | Expected Hash | Actual Computed Hash | Match Status |
| :--- | :--- | :--- | :---: |
| `RESEARCH_FINAL_REPORT.md` | `c90c0d504df8d4ea...` | `c90c0d504df8d4ea...` | **MATCH (100%)** |
| `experiments/.../RESEARCH_FINAL_REPORT.md` | `c90c0d504df8d4ea...` | `c90c0d504df8d4ea...` | **MATCH (100%)** |
| `REPRODUCIBILITY_GUIDE.md` | `454e889ab953bac4...` | `454e889ab953bac4...` | **MATCH (100%)** |
| `FINAL_ARTIFACT_CATALOG.csv` | `b367cb66079a80cd...` | `b367cb66079a80cd...` | **MATCH (100%)** |
| `FINAL_CLAIM_EVIDENCE_TABLE.csv` | `2453a871bfe7d508...` | `2453a871bfe7d508...` | **MATCH (100%)** |
| `v11 Checkpoint (.pt)` | `3c4369d2f95059b9...` | `3c4369d2f95059b9...` | **MATCH (100%)** |
| `v11 ONNX Model (.onnx)` | `54298c503f51c829...` | `54298c503f51c829...` | **MATCH (100%)** |
| `E1 Checkpoint (.pt)` | `f3c0fe8f7ccebb8f...` | `f3c0fe8f7ccebb8f...` | **MATCH (100%)** |
| `D2_H1 Candidate Checkpoint (.pt)` | `c6102b4876ddeaa4...` | `c6102b4876ddeaa4...` | **MATCH (100%)** |
| `Phase 26 D0_H1 Checkpoint (.pt)` | `00b4fc71f8637886...` | `00b4fc71f8637886...` | **MATCH (100%)** |
| `D3_C Candidate Checkpoint (.pt)` | `da5b414328c33644...` | `da5b414328c33644...` | **MATCH (100%)** |
| `validation_v1.csv` Manifest | `5af8cf9daa96526c...` | `5af8cf9daa96526c...` | **MATCH (100%)** |
| `PHASE58_CONFIRMATORY_MANIFEST.csv` | `9cca06f2c31d0b67...` | `9cca06f2c31d0b67...` | **MATCH (100%)** |
| `PHASE58_STREAMING_MANIFEST.csv` | `035ad2795aefe375...` | `035ad2795aefe375...` | **MATCH (100%)** |
| `PHASE58_EVENT_LEDGER.csv` | `e8ca1cf80c8c58bc...` | `e8ca1cf80c8c58bc...` | **MATCH (100%)** |
| *13 Additional Report/Prediction Files* | *Verified identical* | *Verified identical* | **MATCH (100%)** |

*Audit Finding:* **Zero mismatches across all 28 entries.**

---

## 5. Model Invariants & Configuration Check

- **Default Model:** Legacy $v11$ (`CorrectedVietnameseKWSNet`, 138,469 parameters, [`variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)) remains the immutable baseline.
- **Operating Threshold:** $\theta_{\text{cmd}} = 0.20$ override locked in [`experiments/vieneu_kws_research_v13_4/bundle/model_config.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/model_config.json).
- **Frontend & VAD Pipeline:** [`LogMelFrontend`](file:///d:/trainidentificationaudio/frontend.py) (40 mel filterbanks, 16kHz) and [`SpeechBandRatioVAD`](file:///d:/trainidentificationaudio/vad.py) (300–3,400Hz energy ratio $\ge 0.65$) fixed.
- **Test Corpus Quarantine:** Cryptographically verified $0\%$ overlap between training sets and Phase 43, Phase 50/51, Phase 58/59 evaluation corpora.

---

## 6. Discrepancy & Resolution Summary

| Discrepancy ID | Item | Condition in `v1.0.0` | Resolution in `v1.0.1` |
| :--- | :--- | :--- | :--- |
| **DISC-01** | `RELEASE_SIGNOFF.md` | Absent from tagged commit tree (created post-tag) | Staged and committed in `d5af658`; verified in `v1.0.1` tag tree |
| **DISC-02** | `FINAL_SHA256_MANIFEST.json` | Root `RESEARCH_FINAL_REPORT.md` listed pre-sync hash | Updated to synchronized canonical hash (`c90c0d50...`); all 28 entries match |

---

## 7. Final Sealed Release Verdict

The repository `D:\trainidentificationaudio` has been forensically verified across Git tree state, tag targets, cryptographic checksums, data quarantine boundaries, and operational configurations.

**Final Release Status:**  
$$\mathbf{PASS\_FINAL\_RELEASE\_VERIFIED\_AND\_SEALED}$$

- **Base Release Tag:** `v1.0.0-research-release-p72` (`928fc85db7e8992c0d8fb253d7fc8bc2bb313fa2`)
- **Sealed Release Tag:** `v1.0.1-research-release-p72` (`d5af658a9afe523b306e5ca6c0159789f1c67fad`)
- **Working Tree:** `clean` (`nothing to commit, working tree clean`)
- **Remote Push:** **Withheld** (strictly local release)
