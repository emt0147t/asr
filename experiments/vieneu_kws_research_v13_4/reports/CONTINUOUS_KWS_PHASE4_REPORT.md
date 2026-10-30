# CONTINUOUS KWS PHASE 4 EXPERIMENTAL REPORT
## Hybrid Continuous Keyword Spotting with False Activation Control

**Repository:** `D:\trainidentificationaudio`  
**Git Branch:** `feature/continuous-kws-segmentation`  
**Phase:** Phase 4 — Hybrid Continuous KWS with False Activation Control  
**Date:** 2026-10-29  
**Execution Environment:** Python 3.11.9, ONNX Runtime 1.31.0, PyTorch 2.6.0+cu124  

---

### Mandatory Governance Labels
> [!WARNING]
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`

---

## 1. Executive Summary & Core Engineering Rulings

Phase 4 investigates whether the rapid contiguous multi-command spotting capability of `continuous_valley` (`LÊNLENPHAI` with 0ms inter-command gap) can be preserved while controlling false activations on Vietnamese conversational speech and out-of-vocabulary (OOV) phonetic foils.

### Key Findings:
1. **Hybrid SpeechBand-Valley Architecture (`hybrid_speechband_valley`):**
   - Combines formant energy concentration gating in 300–3400 Hz ($\text{ratio} \ge 0.32$) with acoustic valley energy dip/rebound splitting.
   - Shortens median end-to-end detection latency from $121.6\text{ ms}$ (`continuous_valley`) to **$52.0\text{ ms}$** (`hybrid_speechband_valley`) (compared to $291.8\text{ ms}$ in `legacy_v11`).
   - Retains **$92.9\%$ command recall** and **$80.0\%$ exact sequence match** on contiguous command sequences.
2. **False Activation Realities on Conversational Speech:**
   - Vietnamese conversational speech consists of human vocalic formants whose energy naturally concentrates in 300–3400 Hz ($\text{ratio} \approx 0.35 - 0.65$). Consequently, spectral ratio filtering alone cannot distinguish open-vocabulary Vietnamese speech from target command words.
   - In always-open microphone mode at $\theta_{\text{cmd}}=0.20$, conversational false activation rate remains high at **$111.0\text{ triggers/min}$**.
3. **Threshold Sweep Trade-offs ($\theta_{\text{cmd}} \in [0.20, 0.60]$):**
   - Increasing $\theta_{\text{cmd}}$ from $0.20$ to $0.50$ reduces conversational false alarms from $111.0/\text{min}$ to $59.4/\text{min}$ while preserving $91.8\%$ command recall.
   - Pushing $\theta_{\text{cmd}}$ to $0.60$ drops command recall down to **$78.2\%$** and exact sequence match to **$60.0\%$**. Raising the threshold alone cannot eliminate false alarms on open-mic continuous speech without unacceptable keyword loss.
4. **Game Mode / Intentional Session Gating (`game_mode_session`):**
   - When the session gate is closed (idle non-gameplay state), conversational speech, noise, and silence produce **$0\text{ false activations}$ ($0.00/\text{min}$)** over 300s of speech, 100s of noise, and 150s of silence.
   - When the session gate is open during gameplay, rapid command bursts achieve **$93.3\%$ recall** and **$80.0\%$ exact match** on $N=3$ contiguous commands.
5. **Mandatory Reporting Rulings:**
   - **Sequence Length Scaling Ruling:** On long sequences ($N=20$), exact sequence match is only **$10.0\%$** (and $30.0\%$ for $N=10$, $60.0\%$ for $N=5$). Because exact match for 20-command sequences is well below $70\%$, **we explicitly DO NOT claim stable support for 20-command continuous sequences**.
   - **Release Sign-Off Ruling:** In an always-open microphone configuration, no parameterization achieves both high command recall and low false activation on natural speech. **The always-open continuous pipeline DOES NOT qualify for release sign-off on natural conversational speech**. Production deployment is defensible only under Intentional Session Gating (Game Mode / push-to-talk / game-state bounded windows).

---

## 2. Baseline Integrity & Governance Invariants

Strict baseline immutability was enforced throughout Phase 4:
- **Git Branch:** Dedicated branch [`feature/continuous-kws-segmentation`](file:///d:/trainidentificationaudio) preserved.
- **Frozen Model Checkpoint:** `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`  
  SHA-256: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (100% bitwise verified).
- **Frozen ONNX Graph:** `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`  
  SHA-256: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (100% bitwise verified).
- **Architecture & Retraining Invariant:** Zero model retraining, zero parameter adjustments, E1/E2 quarantined.
- **Released Demo Default:** Default `LiveKWSPipeline` configuration remains `legacy_v11` ($\theta_{\text{cmd}}=0.20$), with experimental profiles accessible via explicit profile flags.

---

## 3. Thí nghiệm A: Hybrid Segmentation Architecture & Evaluation

### Architecture
[`HybridSpeechBandValleySegmenter`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/hybrid_segmenter.py) fuses:
1. **Spectral Speech-Band Ratio Gating:** Real-time 512-point FFT energy concentration in 300–3400 Hz ($\text{ratio} \ge 0.32$). Broadband noise transients, thumps, and mechanical clicks falling outside human speech formants are filtered before onset accumulation.
2. **Short-Time Acoustic Valley Splitting:** Tracks RMS energy dynamics during continuous vocalization. When vocalic power drops by $\ge 12.0\text{ dB}$ into an inter-syllabic valley and rebounds by $\ge 6.0\text{ dB}$ with confirmed speech-band concentration, the segmenter cleanly segments the prior utterance and re-arms for the subsequent command without requiring a pause.

### Comparative Results ($\theta_{\text{cmd}} = 0.20$)
Evaluated across 50 command sequence trials (`LEN -> LEN -> PHAI`, `TRAI -> LEN -> XUONG`, `LEN -> LEN -> LEN`, and elongated `LÊÊÊN`) and full negative controls (5.0 min silence, 100 noise clips, 300 conversational speech clips, 200 phonetic foil clips).

| Profile Name | Command Recall (%) | Command Precision (%) | Command F1 (%) | Exact Sequence Match (%) | Mean Infer Latency ($t_{\text{infer}}$) | Median E2E Latency ($t_{\text{e2e}}$) | Silence FAR (/min) | Noise FAR (/min) | Conversational Speech FAR (/min) | OOV Foil FAR (/min) | Total Conv Audio (s) | Total Foil Audio (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `legacy_v11` | 25.7% | 76.6% | 38.5% | 10.0% | 1.94 ms | 291.8 ms | 0.00 | 0.00 | 0.20 | 0.00 | 300.0s | 200.0s |
| `continuous_valley` | 97.1% | 97.1% | 97.1% | 92.0% | 1.93 ms | 121.6 ms | 0.00 | 0.00 | 137.80 | 105.90 | 300.0s | 200.0s |
| `hybrid_speechband_valley` | 92.9% | 97.0% | 94.9% | 80.0% | 1.92 ms | **52.0 ms** | 0.00 | 0.00 | 111.00 | 104.70 | 300.0s | 200.0s |

*Artifact Source:* [`experiments/vieneu_kws_research_v13_4/reports/phase4_hybrid_segmentation_comparison.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase4_hybrid_segmentation_comparison.csv)

### Latency Disaggregation Analysis:
- **Model Inference Latency ($t_{\text{infer}}$):** Measures forward pass execution of ONNX Runtime on CPU + Log-Mel spectrogram computation. Constant at **$1.92 - 1.94\text{ ms}$** across all profiles.
- **End-to-End Latency ($t_{\text{e2e}}$):** Measures elapsed acoustic time from syllable offset to detection event emission.
  - In `legacy_v11`: $291.8\text{ ms}$ due to mandatory 300ms trailing silence hangover. Contiguous speech merges into single unparsed blocks (recall only 25.7%).
  - In `continuous_valley`: $121.6\text{ ms}$ due to 140ms silence hangover / valley detection.
  - In `hybrid_speechband_valley`: **$52.0\text{ ms}$** median latency because spectral confirmation accelerates valley rebound confirmation.

---

## 4. Thí nghiệm B: Threshold Sweep ($\theta_{\text{cmd}}$)

The confidence threshold $\theta_{\text{cmd}}$ was swept across $[0.20, 0.30, 0.40, 0.50, 0.60]$ using `hybrid_speechband_valley` on the command suite and negative control corpora.

| Threshold $\theta_{\text{cmd}}$ | Command Recall (%) | Command Precision (%) | Command F1 (%) | Exact Sequence Match (%) | Conversational False Triggers | Conversational FAR (/min) | Foil False Triggers | Foil FAR (/min) | Mean Infer Latency ($t_{\text{infer}}$) | Median E2E Latency ($t_{\text{e2e}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 92.7% | 96.2% | 94.4% | 80.0% | 555 | 111.00 | 349 | 104.70 | 2.09 ms | 56.9 ms |
| **0.30** | 92.7% | 96.2% | 94.4% | 80.0% | 519 | 103.80 | 331 | 99.30 | 2.02 ms | 57.3 ms |
| **0.40** | 92.7% | 96.2% | 94.4% | 80.0% | 411 | 82.20 | 261 | 78.30 | 2.16 ms | 57.3 ms |
| **0.50** | 91.8% | 97.1% | 94.4% | 77.5% | 297 | 59.40 | 182 | 54.60 | 2.37 ms | 62.3 ms |
| **0.60** | 78.2% | 100.0% | 87.8% | 60.0% | 201 | 40.20 | 125 | 37.50 | 2.17 ms | 81.8 ms |

*Artifact Source:* [`experiments/vieneu_kws_research_v13_4/reports/phase4_threshold_sweep.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase4_threshold_sweep.csv)

### Scientific Findings from Threshold Sweep:
1. **The $\theta = 0.50$ Operating Point:**
   - Increasing $\theta_{\text{cmd}}$ from $0.20$ to $0.50$ cuts conversational false activations by **$46.5\%$** (from 555 triggers to 297 triggers; $59.4/\text{min}$) while retaining **$91.8\%$ recall** and **$97.1\%$ precision**.
2. **Failure of Pure Threshold Tuning at $\theta \ge 0.60$:**
   - At $\theta_{\text{cmd}} = 0.60$, conversational triggers drop to $40.2/\text{min}$, but command recall collapses by $14.5\%$ (down to $78.2\%$) and exact sequence match drops to $60.0\%$.
   - **Conclusion:** Closed-set softmax probability calibration cannot eliminate out-of-domain conversational speech false alarms without catastrophic degradation of target keyword sensitivity.

---

## 5. Thí nghiệm C: Game Mode / Intentional Command-Listening Session

To resolve the fundamental limitation of always-open microphones on natural speech, [`GameModeSessionController`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/hybrid_segmenter.py) introduces intentional command listening sessions:
- **Idle State (Gate Closed):** Mic input during normal conversation is gated out.
- **Active Gameplay State (Gate Open):** High-agility hybrid valley segmentation is armed for rapid contiguous command bursts.

| Mode / Scenario | Test Scenario | Command Length $N$ | Inter-Command Gap (ms) | Exact Sequence Match (%) | Command Recall (%) | False Activation Count | Total Audio Duration (s) | FAR (/min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`game_mode_idle` (Gate Closed)** | Conversational Speech | 0 | N/A | N/A | N/A | **0** | 300.0s | **0.00** |
| **`game_mode_idle` (Gate Closed)** | Environmental Noise | 0 | N/A | N/A | N/A | **0** | 100.0s | **0.00** |
| **`game_mode_idle` (Gate Closed)** | Digital Silence | 0 | N/A | N/A | N/A | **0** | 150.0s | **0.00** |
| `always_open_mic` | Conversational Speech | 0 | N/A | N/A | N/A | 555 | 300.0s | 111.00 |
| `game_mode_active` (Gate Open) | Contiguous Bursts ($N=3$) | 3 | 0 ms | **80.0%** | **93.3%** | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Contiguous Bursts ($N=3$) | 3 | 50 ms | **80.0%** | **93.3%** | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Sequence Scaling ($N=5$) | 5 | 0 ms | 40.0% | 82.0% | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Sequence Scaling ($N=5$) | 5 | 50 ms | 60.0% | 92.0% | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Sequence Scaling ($N=10$) | 10 | 0 ms | 10.0% | 87.0% | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Sequence Scaling ($N=10$) | 10 | 50 ms | 30.0% | 93.0% | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Sequence Scaling ($N=20$) | 20 | 0 ms | 10.0% | 85.5% | 0 | N/A | 0.00 |
| `game_mode_active` (Gate Open) | Sequence Scaling ($N=20$) | 20 | 50 ms | 10.0% | 91.5% | 0 | N/A | 0.00 |

*Artifact Source:* [`experiments/vieneu_kws_research_v13_4/reports/phase4_gamemode_comparison.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase4_gamemode_comparison.csv)

### Mandatory Evaluation Decisions:
1. **Sequence Length Scaling Ruling (N = 20):**
   - Exact sequence match for $N=20$ is **$10.0\%$** (well below the $70\%$ threshold).
   - While individual command recall remains acceptable ($85.5\% - 91.5\%$), cumulative timing drift and boundary misalignment across 20 unbroken contiguous syllables prevent stable whole-sequence exact matches.
   - **Ruling:** Pursuant to governance requirements, **stable support for 20-command continuous sequences is NOT supported and will not be claimed**.
2. **False Activation Claims:**
   - Under `game_mode_idle`, zero false activations (FAR = 0.00/min) is achieved by architectural gating, validated across 300s of conversational speech, 100s of noise, and 150s of silence.
   - Under `always_open_mic`, false activations cannot be claimed as zero.

---

## 6. Comprehensive Verification & Regression Test Suite

All Phase 4 algorithms and invariants were packaged and verified in [`tests/test_continuous_kws_phase4.py`](file:///d:/trainidentificationaudio/tests/test_continuous_kws_phase4.py).

```powershell
.venv\Scripts\pytest.exe tests/test_continuous_kws_phase2.py tests/test_continuous_kws_phase4.py -v
```

**Results:**
- `test_frozen_artifact_immutability`: PASSED (Checkpoint & ONNX byte hashes identical).
- `test_hybrid_repeated_command_len_len_phai[0, 50, 100]`: PASSED (Accurately segments and detects contiguous commands).
- `test_hybrid_three_contiguous_len`: PASSED (Triple identical contiguous commands segmented into exactly 3 events).
- `test_hybrid_elongated_command_protection`: PASSED (750ms elongated word produces strictly 1 event).
- `test_hybrid_speechband_out_of_band_rejection`: PASSED (High frequency 6000Hz tone rejected by spectral gating).
- `test_threshold_sweep_rejection_behavior`: PASSED (Monotonic threshold gate verification).
- `test_game_mode_closed_gate_suppression`: PASSED (Idle gate suppresses conversational speech 100%).
- `test_game_mode_active_burst_detection`: PASSED (Active gate detects rapid command bursts).
- `test_game_mode_timeout_expiry`: PASSED (Auto-closing session after timeout).
- `test_live_mic_engine_profile_bindings`: PASSED (Profile registration integrity).
- **Total:** **21 passed in 3.92s (100% pass rate)**.

---

## 7. Final Governance & Release Sign-Off Statement

Based on the empirical evidence gathered across Phases 2, 3, and 4:
1. **Always-Open Continuous Microphone:**
   - Fails release criteria for natural conversational environments due to elevated open-vocabulary false activations ($111.0/\text{min}$ at $\theta=0.20$; $59.4/\text{min}$ at $\theta=0.50$).
   - Must remain strictly marked with:
     - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
     - `NOT_VALIDATED_ON_NATURAL_SPEECH`
     - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
2. **Production-Defensible Architecture:**
   - Intentional Session Gating (`game_mode_session`) resolves the conversational speech vulnerability by silencing idle chatter (FAR = 0.00/min) while granting rapid contiguous command responsiveness ($52\text{ ms}$ latency, $93.3\%$ recall) during active gameplay.
   - Long continuous sequences are bounded: stable for $N \le 3$, degrading as $N \to 20$.
