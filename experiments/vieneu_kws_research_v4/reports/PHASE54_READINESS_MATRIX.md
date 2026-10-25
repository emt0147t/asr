# Phase 54 — System Readiness and Deployment Matrix

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v4/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Multi-Dimensional Readiness Matrix

| Dimension | Continuous Open-Mic Mode (Mode B) | Bounded-Utterance Mode (Mode A) | Evidence & Notes |
| :--- | :--- | :--- | :--- |
| **Operational Specification** | Always-listening 24/7 background audio stream | Push-to-talk / User-initiated short capture turn | Explicit architectural separation formalized in Phase 50–54 |
| **Target Command Recall** | Required: $\ge 95.0\%$  <br> Achieved: **74.17%** (Candidate 3) / **97.29%** (Historical) | Required: $\ge 95.0\%$  <br> Achieved: **95.83%** (Candidate 2) | Bounded capture eliminates sliding window truncation |
| **False Activation / Acceptance** | Required: $\le 1.0\text{ FA/hr}$  <br> Achieved: **10.33 FA/hr** (Best: 4.33 FA/hr) | Required: $\text{FAR} \le 20.0\%$  <br> Achieved: **14.88%** (TNR = **85.12%**) | Candidate 2 rejects 476/560 non-target clips |
| **Detection / Inference Latency** | Required: $\le 600\,\text{ms}$  <br> Achieved: **320 ms** | Required: $\le 50\,\text{ms}$  <br> Achieved: **5.81 ms** | Fast GPU / mobile CPU inference |
| **Phonetic Rhyme `/aːj/` Handling** | **FAILED** (Severe over-suppression on `TRAI`/`PHAI`) | **FEASIBLE** (`TRAI`: 99.17%, `PHAI`: 100.0%) | Full-clip joint temporal evaluation resolves the `/aːj/` dilemma |
| **Noise Robustness** | Tested under pink noise ($0.0\text{ gap FAs}$) | Tested under additive pink/white noise + time shift | Ambient silence immunity verified |
| **Natural Speech Validation** | **NOT VALIDATED** (Synthetic VieNeu only) | **NOT VALIDATED** (Synthetic VieNeu only) | Requires approved field recordings |
| **Human Perceptual Verification**| **PENDING** (Listening kit created) | **PENDING** (Listening kit created) | Sealed key in `human_review/human_review_key.csv` |
| **Formal Deployment Verdict** | **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`** | **`BOUNDED_UTTERANCE_FEASIBLE_IN_SYNTHETIC_RESEARCH`** | Clear operational boundary |

---

## 2. Component Architecture Reference

### Candidate 2: Cascade v1 Bounded Pipeline (Recommended for Bounded Experimentation)
- **Stage 1 CNN:** `VietnameseKWSNet` (138,534 parameters, 2D Log-Mel CNN).
  - Inputs: (1, 40, 101) Log-Mel Spectrogram (16 kHz, 25 ms window, 10 ms hop).
  - Checkpoint: `experiments/vieneu_kws_v2/checkpoints/best_checkpoint_seed{42,43,44}.pt`.
- **Stage 2 Verifier:** `TemporalKWSVerifier` (340,773 parameters, Dilated 1D-TCN).
  - Inputs: Multi-frame Log-Mel sequence.
  - Checkpoint: `experiments/vieneu_kws_cascade_v1/checkpoints/best_verifier_seed{42,43,44}.pt`.
- **Total Parameters:** 479,307 parameters (~1.9 MB uncompressed FP32).
- **Execution Cost:** 5.81 ms per decision turn.

---

## 3. Regulatory and Integrity Adherence

1. **Synthetic Data Boundaries:** All experiments were strictly isolated within synthetic VieNeu-TTS datasets.
2. **Zero Modification of Historical Baselines:** Checked and verified across all historical directories.
3. **No Unapproved Field Recordings:** `data/raw/field_recordings/` and consent files remained untouched.
4. **Git Staging Cleanliness:** `git diff --cached` contains 0 staged files.
