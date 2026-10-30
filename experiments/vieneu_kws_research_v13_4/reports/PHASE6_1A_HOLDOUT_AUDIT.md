# Phase 6.1A: Dataset Independence & Holdout Manifest Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `d5af658a9afe523b306e5ca6c0159789f1c67fad`  
**Date:** 2026-10-30  
**Status:** Completed  
**Workstream:** Phase 6.1A (Workstream A)

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Frozen Artifact Integrity Verification

Prior to performing the dataset independence audit, the frozen candidate model checkpoints were independently re-hashed:
- **PyTorch Checkpoint:** `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
  `SHA-256: 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (Verified Match)
- **ONNX Model:** `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`  
  `SHA-256: 54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (Verified Match)

---

## 2. Inventory of Available Target Recordings

Audit of `experiments/vieneu_kws_research_v13_3/data/confirmatory_evaluation_manifest.csv` and the audio directory `experiments/vieneu_kws_research_v13_3/data/audio/`:
- **Available Source Files per Target Class:**
  - `LEN`: 50 audio files (`v13_conf_cmd_len_001.wav` .. `v13_conf_cmd_len_050.wav`)
  - `PHAI`: 50 audio files (`v13_conf_cmd_phai_001.wav` .. `v13_conf_cmd_phai_050.wav`)
  - `TRAI`: 50 audio files (`v13_conf_cmd_trai_001.wav` .. `v13_conf_cmd_trai_050.wav`)
  - `XUONG`: 50 audio files (`v13_conf_cmd_xuong_001.wav` .. `v13_conf_cmd_xuong_050.wav`)
  - **Total Command Files:** 200 files.
- **Speaker Demographics:** Exactly 8 speakers, all with Northern Vietnamese accent (`Bắc`):
  - Female (3 speakers, 20 clips per class):
    - `spk_north_01` (Trúc Ly): clips 001..007 (7 clips)
    - `spk_north_02` (Ngọc Linh): clips 008..014 (7 clips)
    - `spk_north_03` (Đoan Trang): clips 015..020 (6 clips)
  - Male (5 speakers, 30 clips per class):
    - `spk_north_04` (Thiện Minh): clips 021..026 (6 clips)
    - `spk_north_05` (Bảo Khang): clips 027..032 (6 clips)
    - `spk_north_06` (Hải Đăng): clips 033..038 (6 clips)
    - `spk_north_07` (Quang Minh): clips 039..044 (6 clips)
    - `spk_north_08` (Hoàng Nam): clips 045..050 (6 clips)

---

## 3. Forensic Audit of Phase 6 Locked Trial Manifest

The 200 trials in [`phase6_locked_trial_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_locked_trial_manifest.csv) were deconstructed down to individual audio clip tokens in [`phase6_1a_source_clip_overlap.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1a_source_clip_overlap.csv):

### Findings on Sample Overlap & Repetition:
1. **Mathematical Data Ceiling:**
   - Evaluating 30 trials of `LEN -> LEN -> PHAI` requires 60 instances of `LEN`.
   - Because only 50 distinct `LEN` recordings exist, **it is mathematically impossible to achieve 30 mutually disjoint trials from this corpus**.
   - Consequently, trials 26–30 in the Phase 6 manifest recycled `len_001` through `len_010`.
2. **0 ms vs 50 ms Trial Coupling:**
   - The 30 trials in `primary_len_len_phai_gap50` used **100% identical audio clip combinations** as `primary_len_len_phai_gap0`, differing only in the inter-command silence gap length.
3. **Cross-Category Clip Reuse:**
   - The 10 trials in `secondary_len_len` reused the exact same initial clip pairs as trials 1–10 of `primary_len_len_phai`.
   - The 10 trials in `elongated_len` reused clips `len_001` through `len_010`.
   - Long sequence stress tests ($N=3, 5, 10, 20$) cyclically reused clips 001..015.
4. **Tuning Data Contamination Disclosure:**
   - During Phase 6 factor investigation, the low cutoff frequency (100 Hz vs 300 Hz) was tuned by observing performance on diagnostic trials 5, 11, and 20, and running a sweep over all 30 trials of `len_len_phai`.
   - Therefore, the Phase 6 reported exact-match results (96.7% and 90.0%) were evaluated on the **exact same pool of clips used to discover the 100 Hz bandpass necessity**.
   - These results represent **fit to the confirmatory tuning pool**, rather than independent held-out generalization.

---

## 4. Phase 6.1A Holdout Manifest Specification

To provide an uncompromised evaluation partition, a new documented holdout manifest was constructed: [`phase6_1a_holdout_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1a_holdout_manifest.csv).

### Partitioning Methodology:
1. **Reserved Holdout Speakers:**
   - Female: `spk_north_03` (Đoan Trang, clips 015..020)
   - Male: `spk_north_06` (Hải Đăng, clips 033..038)
   - Male: `spk_north_08` (Hoàng Nam, clips 045..050)
   - *Total reserved clips:* 18 clips per class (72 clips total).
   - *Direct Diagnostic Overlap:* **0 trials** in this holdout partition contain any of the clips directly analyzed in the Phase 6 diagnostic scripts (`len_009`, `len_010`, `phai_005`, `len_021`, `len_022`, `phai_011`, `len_039`, `len_040`, `phai_020`).
2. **Holdout Manifest Composition (55 trials total):**
   - **Primary Demo Target ($N=3$, 30 trials):**
     - 15 trials at 0 ms gap (`holdout_len_len_phai_gap0_t01` .. `t15`).
     - 15 trials at 50 ms gap (`holdout_len_len_phai_gap50_t01` .. `t15`).
     - Features structured subsets:
       - Speaker-isolated blocks (F-F-M, M-M-F, M-M-M).
       - Novel cross-speaker combinations never paired in Phase 6 (e.g. `spk_north_03` female onset transitioning into `spk_north_06` male sonorant).
   - **Secondary Patterns (20 trials):**
     - 10 trials of `LEN -> LEN` (5 at 0 ms, 5 at 50 ms).
     - 10 trials of `TRAI -> LEN -> XUONG` (5 at 0 ms, 5 at 50 ms).
   - **Elongated Single Command Control (5 trials):**
     - 5 trials of sustained `LEN` ($L=750$ ms) using male holdout speaker `spk_north_08`.

### Data Limitation Statement:
While the new Holdout Manifest successfully guarantees **zero overlap with diagnostic failure files** and presents **novel cross-speaker transitions**, the total pool of recorded clips in this repository remains bounded at 50 per class. A true large-scale statistical validation will require a newly collected, independent recording campaign with unseen speakers and natural conversational environments.

---

## 5. Summary Deliverables

- Detailed Source Clip Overlap Table: [`phase6_1a_source_clip_overlap.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1a_source_clip_overlap.csv)
- Documented Holdout Manifest: [`phase6_1a_holdout_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1a_holdout_manifest.csv)
- Machine-Readable Audit Summary: [`phase6_1a_holdout_audit.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1a_holdout_audit.json)
