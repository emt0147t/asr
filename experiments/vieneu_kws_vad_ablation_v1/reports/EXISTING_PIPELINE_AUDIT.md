# Phase 39A: Independent Audit of Existing Evaluation Pipeline and Metrics

**Workspace:** `d:\trainidentificationaudio`  
**Experiment Path:** [`experiments/vieneu_kws_vad_ablation_v1/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_vad_ablation_v1)  
**Audited Systems:**
- Stage 1 Baseline: [`experiments/vieneu_kws_v2/evaluate.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/evaluate.py)
- Cascade Pipeline: [`experiments/vieneu_kws_cascade_v1/evaluate_cascade.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_cascade_v1/evaluate_cascade.py)
- Streaming Simulation Assets: [`experiments/vieneu_kws_v2/streaming_simulation/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_v2/streaming_simulation)
**Execution Date:** 2026-10-18  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

In preparation for Phase 39 (Voice Activity Detection Gating Ablation), an independent audit of the streaming inference engine, event definitions, ground-truth matching logic, debounce mechanisms, and metric calculations was conducted.

### Audit Verdict: **PASSED (PROTOCOL VERIFIED)**
- **Stream Durations and Hop Rates:** Audited directly from raw WAV file sample counts at $16,000\text{ Hz}$.
- **Event Definition Equivalence:** Both Stage 1 alone and the Stage 1 + Stage 2 Cascade use the exact same sliding window, consecutive confirmation, debounce lockout, and matching tolerances.
- **Deduplication vs Raw Windows:** Verified that false alarm rates count discrete multi-window activation events, not individual $100\text{ ms}$ sliding windows.
- **Per-Seed Granularity:** Confirmed that target command recall requirements must be assessed both per-seed and across seeds.

---

## 2. Technical Audit Details

### 2.1 Sample Rates, Frame Sizes, and Exact Stream Durations

| Stream Asset | Sample Count ($16\text{ kHz}$) | Physical Duration (s) | Duration in Hours | Ground-Truth Targets | Distractor Clips |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Commands** (`val_stream_continuous_commands.wav`) | $10,770,378$ | $673.1486\text{ s}$ | $0.1870\text{ h}$ | $160$ ($40 \times 4$) | $62$ |
| **Validation Non-Target** (`val_stream_continuous_nontarget.wav`) | $14,414,654$ | $900.9159\text{ s}$ | $0.2503\text{ h}$ | $0$ | $164$ |
| **Test Commands** (`test_stream_continuous_commands.wav`) | $10,814,529$ | $675.9081\text{ s}$ | $0.1878\text{ h}$ | $160$ ($40 \times 4$) | $63$ |
| **Test Non-Target** (`test_stream_continuous_nontarget.wav`) | $14,424,587$ | $901.5367\text{ s}$ | $0.2504\text{ h}$ | $0$ | $193$ |

**Sliding Window Dimensions:**
- Window size: $1.0\text{ s} = 16,000\text{ samples}$ ($101$ Log-Mel frames, $win\_length=400$, $hop\_length=160$).
- Hop size: $0.10\text{ s} = 1,600\text{ samples}$ ($10$ evaluations per second $\rightarrow 36,000$ evaluations per hour).
- Stage 2 acoustic crop: $1.2\text{ s} = 19,200\text{ samples}$ ($121$ Log-Mel frames).

---

### 2.2 Event Detection, Matching, and Latency Calculations

In `match_events_and_evaluate()`:
- Ground-truth target command interval: $[t_{\text{onset}}, t_{\text{end}}]$.
- A detected trigger at $t_{\text{trig}}$ with predicted label $L$ is matched to a ground-truth command if:
  $$(t_{\text{onset}} - 0.20\text{ s}) \le t_{\text{trig}} \le (t_{\text{end}} + 0.60\text{ s}) \quad \text{AND} \quad L == \text{gt\_label} \quad \text{AND} \quad \text{gt\_matched}[\text{idx}] == \text{False}$$
- **True Positives (TP):** Count of successfully matched ground-truth commands.
- **False Negatives (FN):** Unmatched ground-truth commands ($N_{\text{targets}} - \text{TP}$).
- **Target Recall:** $\text{TP} / N_{\text{targets}}$.
- **False Positives (FP):** Count of triggered events that do not match any ground-truth command.
- **Precision:** $\text{TP} / (\text{TP} + \text{FP})$.
- **Detection Latency:** $(t_{\text{trig}} - t_{\text{onset}}) \times 1000\text{ ms}$.
- **False Activations / Hour:** $\text{FP} / \text{stream\_duration\_hours}$.

---

### 2.3 Debounce, Hysteresis, and Duplicate Suppression

In `CascadedStreamingEngine`:
- **Consecutive Confirmations:** Requires $k = 2$ consecutive $100\text{ ms}$ windows predicting the identical target class with confidence $\ge \theta_1$.
- **Refractory Lockout:** Upon event trigger (whether Stage 1 alone or Cascade), a debounce lockout is set:
  $$\text{lockout\_until\_sample} = s_{\text{idx}} + (\text{debounce\_sec} \times \text{sample\_rate})$$
  where $\text{debounce\_sec} = 0.60\text{ s}$ ($9,600$ samples).
- During lockout, incoming window predictions cannot increment a streak or fire an event.
- **Verification:** Because of this lockout mechanism, multiple overlapping $100\text{ ms}$ windows from the same acoustic burst are collapsed into exactly **one activation event**. False alarm counts strictly reflect distinct trigger events, not window counts.

---

### 2.4 Event Definition Equivalence between Stage 1 Alone and Cascade

Audit of `evaluate_cascade.py` confirms:
- Both Stage 1 alone (evaluated with `bypass_stage2=True`) and the Cascade (evaluated with Stage 2 enabled) utilize the exact same sliding-window loop, same hop size ($100\text{ ms}$), same streak threshold ($k=2$), same debounce duration ($0.60\text{ s}$), and same ground-truth matching tolerance ($[-0.20\text{ s}, +0.60\text{ s}]$).
- The only difference is that when Stage 2 is active, a Stage 1 candidate proposal must also be confirmed by the Stage 2 Temporal Verifier ($P_2(c) \ge \theta_2 \text{ and } P_2(c) > P_2(\text{REJECT})$) before an event is emitted.
- This guarantees a completely fair, unbiased comparison.

---

### 2.5 Per-Seed Assessment Requirement

In Phase 36–38:
- Overall mean command recall was $96.25\%$.
- However, per-seed analysis revealed:
  * Seed 42: $98.1\%$
  * Seed 43: $97.5\%$
  * Seed 44: $93.1\%$ (fell below the $\ge 95\%$ target).
- **Rule for Phase 39:** In accordance with the user instructions, any seed with command recall below $95\%$ will be explicitly flagged, even if the across-seed mean meets the threshold.

---

## 3. Scope and Directives for Phase 39B–39E (VAD Ablation)

1. **VAD Architecture Selection:**
   - Environment audit reveals `torchaudio.functional.vad` (SoX speech-band VAD) is available.
   - We will implement and compare two distinct VAD variants:
     * **Variant 1 (`EnergyGate`):** Short-time energy detector using frame-level RMS relative to an adaptive or calibrated noise floor.
     * **Variant 2 (`SpeechBandVAD`):** Frequency-selective speech-band VAD combining speech formant energy ratio ($300–3400\text{ Hz}$ vs out-of-band), spectral flatness/entropy, and hangover smoothing.
2. **Four Conditions to Benchmark:**
   - **Condition A:** Stage 1 CNN alone (un-gated baseline).
   - **Condition B:** Two-stage Cascade without VAD.
   - **Condition C:** Stage 1 CNN with VAD gating.
   - **Condition D:** Two-stage Cascade with VAD gating.
3. **Strict Validation-Only Parameter Tuning:**
   - All VAD thresholds, hangover frames, and energy floors will be tuned on validation streams only.
   - Test streams will be evaluated exactly once with frozen configurations.
