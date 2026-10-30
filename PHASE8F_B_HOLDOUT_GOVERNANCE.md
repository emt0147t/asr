# Phase 8F-B: Final Holdout Governance & Source Overlap Audit

**Repository:** `D:\trainidentificationaudio`  
**Base Branch:** `feature/continuous-kws-segmentation`  
**Commit Baseline:** `eea9caf`  
**Date:** 2026-10-31  

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Holdout Reclassification

### Reclassification of `phase6_1a_holdout_manifest.csv`
- In Phase 6, Phase 7, and Phase 8, the 55 continuous sequence trials defined in `phase6_1a_holdout_manifest.csv` were evaluated repeatedly across threshold sweeps, acoustic channel characterizations, and factorial training evaluations.
- **Formal Status Reclassification:** `phase6_1a_holdout_manifest.csv` is formally reclassified as a **Historical Development Benchmark**, **NOT an untouched final holdout**.
- Evaluating models on `phase6_1a_holdout_manifest.csv` evaluates performance against a known development target, but cannot establish independent generalization to genuinely unseen natural speech.

---

## 2. Exhaustive Source Overlap Audit

Recorded in [`reports/phase8f_b_source_overlap.csv`](file:///d:/trainidentificationaudio/reports/phase8f_b_source_overlap.csv):

| Corpus A | Corpus B | Sample Count A | Sample Count B | Audio Overlap | Speaker Overlap | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **V12 Training Set** | **V12 Validation Set** | 5,440 | 1,280 | **0** | **0** | Disjoint Split |
| **V12 Training Set** | **V12 Test Benchmark** | 5,440 | 1,280 | **0** | **0** | Disjoint Split |
| **V12 Validation Set** | **V12 Test Benchmark** | 1,280 | 1,280 | **0** | **0** | Disjoint Split |
| **V12 Training Set** | **Phase 7B Aug Sources** | 5,440 | 260 | **0** | **0** | Disjoint Split |
| **Phase 7B Aug Sources**| **Phase 6.1A Benchmark** | 260 | 55 | **0** | **0** | Speaker Disjoint |
| **V12 Training Set** | **Phase 6.1A Benchmark** | 5,440 | 55 | **0** | **0** | Fully Disjoint |

### Key Audit Findings
1. **Zero Training Leakage:** The 5,440 training samples and their speakers share **0 audio files** and **0 speaker overlap** with the validation set, test benchmark, Phase 7B augmentations, and Phase 6.1A benchmark.
2. **Confirmatory Speaker Separation:** The 8 confirmatory speakers (`spk_north_01` to `08`) were partitioned so that `spk_north_01, 02, 04, 05, 07` were used for Phase 7B training augmentations, while `spk_north_03, 06, 08` were strictly quarantined for Phase 6.1A continuous evaluation.
3. **Repeated Exposure of Quarantined Speakers:** Although `spk_north_03, 06, 08` never leaked into training, their repeated evaluation across Phases 6–8 means they can no longer be claimed as a blind holdout.

---

## 3. Absence of Unseen Natural Continuous Speech Corpus

A forensic search across all repository directories confirmed:
1. **Field Recordings Campaign:** `data/raw/field_recordings/` contains only test harness calibration runs (`SPK_FIELD_N001_TESTSUITE` and `SPK_CALIB_001` to `020`), which are synthetic calibration utterances.
2. **FLEURS Vietnamese Corpus:** `data/raw/fleurs_vi/audio_extracted/test/` contains 857 general Vietnamese sentences, but contains zero target KWS command keywords (`TRAI`, `PHAI`, `LEN`, `XUONG`).
3. **Conclusion:** **No genuinely untouched, natural conversational multi-command continuous evaluation corpus currently exists in the repository.**

### Mandatory Requirements for Future Human Recording Campaign
Before natural continuous speech generalization can be validated:
1. **Speaker Cohort:** $\ge 10$ native Vietnamese speakers with zero prior exposure to the project.
2. **Dialectal Balance:** At least 3 Northern (Hanoi), 3 Central (Hue/Da Nang), and 4 Southern (Ho Chi Minh City) speakers.
3. **Gender Balance:** Equal distribution (5 Female, 5 Male).
4. **Acoustic Transducers & Distances:** Simultaneous multi-microphone recording:
   - Close-talk headset ($15\text{ cm}$)
   - Laptop integrated microphone ($50\text{ cm}$)
   - Far-field smart speaker microphone ($1.5\text{ m}$)
5. **Acoustic Environments:** Quiet office ($	ext{SNR} > 25\text{ dB}$), reverberant meeting room ($	ext{RT60} > 0.4\text{ s}$), and domestic ambient noise ($	ext{SNR} = 12\text{--}18\text{ dB}$).
6. **Task Protocol:** Unprompted conversational speech containing intentional keyword bursts, conversational foils, and natural pacing.

---

## 4. Sealed Final Holdout Manifest

To preserve experimental integrity for future verification, a new sealed manifest has been constructed and cryptographically locked:
- **Manifest Path:** [`reports/phase8f_b_final_holdout_manifest.csv`](file:///d:/trainidentificationaudio/reports/phase8f_b_final_holdout_manifest.csv)
- **Manifest Composition:**
  - **1,280 single-utterance instances** from `experiments/vieneu_kws_research_v12/manifests/test_manifest.csv` (168 unseen speakers, zero training overlap).
  - **60 multi-command continuous sequence trials** constructed from test speaker audio.
- **Total Instances:** 1,340 evaluation instances.
- **Governance Constraint:** **Candidate models are strictly PROHIBITED from being evaluated on this sealed holdout during Phase 8F.**
