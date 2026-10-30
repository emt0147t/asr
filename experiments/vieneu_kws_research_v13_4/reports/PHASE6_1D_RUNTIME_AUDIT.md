# Phase 6.1D: Runtime Lifecycle & Regression Audit Report

**Repository:** `D:\trainidentificationaudio`  
**Branch:** `feature/continuous-kws-segmentation`  
**Baseline Git Commit:** `d5af658a9afe523b306e5ca6c0159789f1c67fad`  
**Date:** 2026-10-30  
**Status:** Completed  
**Workstream:** Phase 6.1D (Workstream D)

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Frozen Artifact Integrity Verification

- **PyTorch Checkpoint:** `variant_v11_hard_negative_weighted_seed42.pt`  
  `SHA-256: 3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (Verified Match)
- **ONNX Model:** `vietnamese_kws_v11.onnx`  
  `SHA-256: 54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (Verified Match)

---

## 2. Executive Summary of Regression Scenarios

All 10 critical runtime lifecycle and regression scenarios passed successfully (logged in [`phase6_1d_runtime_regressions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1d_runtime_regressions.csv)):

| Scenario ID | Test Scope / Description | Expected Behavior | Actual Outcome | Status |
| :--- | :--- | :--- | :--- | :---: |
| **`REG-01-REPEATED-BURSTS`** | Consecutive same-label separation (`LEN -> LEN -> LEN`) | Emit 3 separate events without debounce cooldown | Emitted 3 events: `['LEN', 'LEN', 'LEN']` | **PASSED** |
| **`REG-02-EVENT-ORDERING`** | Directional rapid transition ordering (`TRAI -> LEN -> XUONG`) | Chronological FIFO preservation | Emitted `['TRAI', 'LEN', 'XUONG']` in order | **PASSED** |
| **`REG-03-DUPLICATE-PREVENTION`**| Double-emission suppression from single cut | $\Delta t \ge 160$ ms between distinct cuts | 3 events emitted, min $\Delta t = 220$ ms | **PASSED** |
| **`REG-04-ELONGATED-OVERSPLIT`** | Sustained vowel over-splitting resistance ($L=750$ ms) | Exactly 1 event emitted for `LÊÊÊN` | Emitted 1 event: `['LEN']` (0 over-splits) | **PASSED** |
| **`REG-05-FLUSH-IDEMPOTENCE`** | Stream termination `flush()` single-fire contract | Emits pending speech once; subsequent calls return `None` | Flush 1: Valid tuple; Flush 2: `None` | **PASSED** |
| **`REG-06-SESSION-GATE-ISOLATION`**| Session gate closed discard vs gate open processing | Closed: 100% `None`; Open: processes & flushes | Closed: 100% `None`; Open: Finalized | **PASSED** |
| **`REG-07-CHUNK-SIZE-RESILIENCY`**| Arbitrary block sizes (160, 320, 50, 71, 0, 500 samples) | Zero exceptions, automatic padding / pass-through | Zero unhandled exceptions | **PASSED** |
| **`REG-08-BELOW-MIN-SPEECH-REJECTION`**| Impulse click rejection (< 100 ms) | 0 events emitted during chunking or flush | 0 segments emitted | **PASSED** |
| **`REG-09-TEMPORAL-MATCHING-BIJECTION`**| Monotonic DP global conflict resolution | Bijective 1-to-1 matching, zero double-counting | $TP \le 2$, strict injection verified | **PASSED** |
| **`REG-10-METRIC-SEPARATION`** | Standard sequence WER vs temporal event metrics | Purely symbolic Levenshtein alignment decoupled from latencies | Exact Levenshtein edit distance verified | **PASSED** |

---

## 3. Patch Summary & Runtime Bug Remediations

During Phase 6 and Phase 6.1D auditing, three specific runtime issues were analyzed and addressed:

### Patch 1: Bandpass Cutoff Low Boundary Accommodation
- **Component:** [`HybridSpeechBandValleySegmenter`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/hybrid_segmenter.py)
- **Bug/Limitation:** Standard telephony speech band `(300.0, 3400.0)` rejected male low-pitched voices ($F_0 \approx 110-180$ Hz), artificially deleting `LEN` commands.
- **Root Cause:** In adult male speech, voiced sonorants and nasal codas have primary formant energy concentrated below 300 Hz.
- **Fix:** Expanded the lower filterbank cutoff from 300.0 Hz to 100.0 Hz (`speech_band_hz=(100.0, 3400.0)`).
- **Regression Test:** `test_consecutive_same_label_separation_and_order` in [`tests/test_game_mode_phase6_1d.py`](file:///d:/trainidentificationaudio/tests/test_game_mode_phase6_1d.py).

### Patch 2: Flush Idempotence & State Hygiene
- **Component:** `flush()` in [`HybridSpeechBandValleySegmenter`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/hybrid_segmenter.py) and [`AcousticValleySegmenter`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/acoustic_valley_segmenter.py).
- **Bug/Limitation:** Multiple consecutive calls to `flush()` at stream termination risked emitting duplicate utterances if internal state was not atomically cleared.
- **Root Cause:** Prior code did not guarantee that `self.reset()` executed atomically before returning the finalized tuple.
- **Fix:** Saved segment info locally, reset state variables (`WAITING`, empty chunk array, zero sample counters), and returned the finalized tuple. Subsequent calls immediately return `None`.
- **Regression Test:** `test_flush_idempotence_and_no_double_emission` in [`tests/test_game_mode_phase6_1d.py`](file:///d:/trainidentificationaudio/tests/test_game_mode_phase6_1d.py).

### Patch 3: Session Gating Clean Slate Isolation
- **Component:** `close_session()` in [`GameModeSessionController`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/hybrid_segmenter.py#L307).
- **Bug/Limitation:** Closing the session gate while audio was in the `ACCUMULATING` state could leak residual speech into the next session.
- **Root Cause:** Closing the session did not forcefully reset the underlying segmenter state.
- **Fix:** Added explicit `self.segmenter.reset()` inside `close_session()`, wiping preroll and accumulated buffers upon gate closure.
- **Regression Test:** `test_session_isolation_and_cross_session_leakage` in [`tests/test_game_mode_phase6_1d.py`](file:///d:/trainidentificationaudio/tests/test_game_mode_phase6_1d.py).

---

## 4. Summary Deliverables

- Detailed Runtime Regressions CSV: [`phase6_1d_runtime_regressions.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_1d_runtime_regressions.csv)
- Standalone Test Suite: [`tests/test_game_mode_phase6_1d.py`](file:///d:/trainidentificationaudio/tests/test_game_mode_phase6_1d.py) (All 7 tests passing)
