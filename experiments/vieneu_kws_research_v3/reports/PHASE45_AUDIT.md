# Phase 45: Independent Evaluation and Data Integrity Audit

Workspace: `D:\trainidentificationaudio`  
Experiment Directory: `experiments/vieneu_kws_research_v3/`  
Date: 2026-10-18  
Status: `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 45 conducted an exhaustive, independent programmatic audit of the Phase 40–44 keyword spotting pipeline, codebase, dataset manifests, trained model checkpoints, and evaluation artifacts prior to any new training in Phase 47.

All ten mandatory audit criteria were investigated using dedicated automated verification tools (`run_phase45_audit.py`). The audit confirms that:
1. Architectural definitions, parameter counts, and checkpoint tensor geometries are fully verified and reconciled.
2. The evaluation pipeline adheres strictly to causal streaming protocols, with valid event deduplication, lockout debouncing, and mathematically exact false-alarm denominators.
3. Machine-readable summary metrics reported in Phase 43 replicate with 100% precision from granular event prediction logs.
4. Data manifests maintain strict isolation: zero voice leakage and zero non-target text leakage across train, validation, and test splits.
5. All observed false alarms across all models occur strictly during speech-active intervals ($100\%$ during speech, $0\%$ in inter-utterance silence gaps).

---

## 2. Detailed Audit Findings Across 10 Points

### Point 1: Architectural Reconciliations and Checkpoint Integrity
- **Stage 1 Detector:** Class [`VietnameseKWSNet`](file:///d:/trainidentificationaudio/src/models/kws_cnn.py), 2D Convolutional neural network, exactly **138,534 parameters**. Checkpoint `stage1_retrained_seed42.pt` contains 6 output logits corresponding to `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, and `SILENCE`.
- **Stage 2 Verifier:** Class [`TemporalKWSVerifier`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1/train_verifier.py), 4-layer Dilated 1D Temporal Convolutional Network (TCN), exactly **340,773 parameters**. Checkpoint `stage2_retrained_seed42.pt` contains 5 output logits (4 target commands + `REJECT`). Inspection verified `has_tcn = True` and `has_gru = False`.
- **Candidate B (CTC):** Class [`VietnameseCTCKWSNet`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/models/ctc_kws/ctc_model.py), Conv1D frontend + 2-layer Bidirectional GRU + character linear projection, exactly **229,115 parameters** over a 91-token vocabulary. Checkpoint `best_ctc_seed42.pt` is fully verified.

### Point 2: Preprocessing Configuration and Input Tensor Shapes
- Extractor: [`LogMelFeatureExtractor`](file:///d:/trainidentificationaudio/src/preprocessing/feature_extractor.py) with sample rate $16,000\text{ Hz}$, $40$ Mel filterbanks, window length $400$ ($25\text{ ms}$), hop length $160$ ($10\text{ ms}$).
- **1.0-second Audio Window ($16,000$ samples):** Produces feature tensor $[B, 1, 40, 101]$. Stage 1 forward output is $[B, 6]$; CTC forward output is $[101, B, 91]$.
- **1.2-second Audio Window ($19,200$ samples):** Produces feature tensor $[B, 1, 40, 121]$. Stage 2 forward output is $[B, 5]$.
- All tensor geometries strictly match model expectations across all seeds.

### Point 3 & 5: Metric Recomputation and Exact FA/hour Denominator
Summary metrics were recomputed from scratch using [`per_event_predictions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/predictions/per_event_predictions.csv) and authoritative ground-truth JSON files.
- **Non-Target Stream Duration:** $688.72225\text{ s} = 0.1913117\text{ hours}$.
- **Denominator Formula:** Strictly $\text{FA/hr} = \frac{\text{False Positives}}{\text{Stream Duration (seconds)} / 3600.0}$.
- **Recomputed vs Documented Metrics:**
  - Candidate 1 (Historical Cascade): Seed 42 = $99.31\text{ FA/hr}$ ($19$ FPs), Seed 43 = $104.54\text{ FA/hr}$ ($20$ FPs), Seed 44 = $104.54\text{ FA/hr}$ ($20$ FPs). Exact match ($0.000\%$ discrepancy).
  - Candidate 2 (Hard-Negative S1 alone): Seed 42 = $282.26\text{ FA/hr}$ ($54$ FPs), Seed 43 = $350.21\text{ FA/hr}$ ($67$ FPs), Seed 44 = $256.13\text{ FA/hr}$ ($49$ FPs). Exact match.
  - Candidate 3 (Hard-Negative Cascade): Seed 42 = $88.86\text{ FA/hr}$ ($17$ FPs), Seed 43 = $52.27\text{ FA/hr}$ ($10$ FPs), Seed 44 = $47.04\text{ FA/hr}$ ($9$ FPs). Exact match.
  - Candidate 4 (CTC KWS): Seed 42 = $0.00\text{ FA/hr}$ ($0$ FPs), Seed 43 = $475.66\text{ FA/hr}$ ($91$ FPs), Seed 44 = $0.00\text{ FA/hr}$ ($0$ FPs). Exact match.

### Point 4: Streaming Logic, Event Deduplication, and Lockout Debounce
- Matching tolerance: A trigger at time $t_{\text{trig}}$ is matched to ground truth if $t_{\text{onset}} - 0.20\text{ s} \le t_{\text{trig}} \le t_{\text{end}} + 0.60\text{ s}$.
- Deduplication: Once a ground truth event is matched, subsequent triggers falling in its window are classified as `false_positives`.
- Debounce: Confirmed events enforce a $0.60\text{ s}$ ($9,600$ sample) lockout during which no new streaks may initiate.

### Point 6: Strict Validation-Only Threshold Selection
Inspection of [`calibration/selected_operating_points.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/calibration/selected_operating_points.json) confirms that threshold tuning was executed exclusively on validation streams (`val_stream_continuous_commands.wav` and `val_stream_continuous_nontarget.wav`).
Because no threshold pair satisfied the joint criteria ($\text{Recall} \ge 95\%$, $\text{FA/hr} \le 1.0$, $\text{Latency} \le 600\text{ ms}$) on validation data, the protocol recorded `NO_FEASIBLE_VALIDATION_OPERATING_POINT` without altering criteria or leaking test data.

### Point 7: Data Manifest Integrity and Zero Split Leakage
- **Speech Voices:**
  - Train: 17 voices (`vie_voice_001` .. `vie_voice_017`)
  - Validation: 4 voices (`vie_voice_018` .. `vie_voice_021`)
  - Test: 4 voices (`vie_voice_022` .. `vie_voice_025`)
  - Voice overlap across splits: **0 voices (100% disjoint)**.
- **Non-Target Text Inventory:**
  - Train: 60 unique texts across 5 confusion families
  - Validation: 20 unique texts across 5 confusion families
  - Test: 20 unique texts across 5 confusion families
  - Text overlap across splits: **0 texts (100% disjoint)**.
- **File Paths:** 2,000 distinct WAV files with zero cross-split file sharing.

### Point 8: Temporal Alignment of Non-Target False Activations
Cross-referencing all 356 non-target false positive triggers across all models and seeds against ground-truth audio intervals revealed:
- **Triggers occurring during speech or margin $[t_{\text{onset}} - 0.20, t_{\text{end}} + 0.60]$:** **$356 / 356$ ($100.0\%$)**
- **Triggers occurring during inter-utterance silence gaps:** **$0 / 356$ ($0.0\%$)**
This rigorously establishes that false activations are exclusively speech-active phonetic confusions, not silence/background noise instabilities.

### Point 9: Machine-Readable Output Reproducibility
All metrics in [`reports/PHASE43_LOCKED_EVALUATION_REPORT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/reports/PHASE43_LOCKED_EVALUATION_REPORT.md) match the saved JSON outputs [`locked_test_summary.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/predictions/locked_test_summary.json) and [`validation_summary.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/predictions/validation_summary.json) with complete arithmetic precision.

### Point 10: CTC Implementation & Sequence Decoding Audit
- **Sequence Lengths:** Input acoustic frames $T=101$; maximum transcript character length is $\sim 15$ characters. The condition $T_{\text{in}} \ge T_{\text{target}}$ is satisfied with ample margin.
- **Loss Function:** `torch.nn.CTCLoss(blank=0, zero_infinity=True)` computes valid finite scalar losses.
- **Decoding Rule:** Standard CTC collapse (`best_path_decode`) successfully collapses duplicate tokens and removes blanks (`[l, l]` $\rightarrow$ `"l"`, `[l, blank, l]` $\rightarrow$ `"ll"`).
- **Instability Diagnosis:** CTC sequence modeling trained from scratch on 1,360 samples exhibits severe optimization sensitivity, frequently falling into blank-collapse local minima. Under sliding window streaming, spurious character emissions cause severe false activations ($475.66\text{ FA/hr}$).

---

## 3. Audit Verdict

The evaluation pipeline, metric accounting, and split isolation are verified free of blocking defects. The research campaign is authorized to proceed to Phase 46 (Benchmark Generation) and Phase 47 (Recall-Preserving Model Training).
