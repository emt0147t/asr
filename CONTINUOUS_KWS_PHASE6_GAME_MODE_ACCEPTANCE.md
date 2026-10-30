# Phase 6: Game Mode Acceptance & Live Event Validation

**Location:** `experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE6_GAME_MODE_ACCEPTANCE.md`  
**Decision:** `ACCEPT_CONTROLLED_DEMO`  
**Date:** 2026-10-30  
**Status:** Completed

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## Quick Reference Summary

The comprehensive Phase 6 Game Mode acceptance audit has concluded with the explicit operational verdict: **`ACCEPT_CONTROLLED_DEMO`**.

### Primary Target ($N=3$: `LEN -> LEN -> PHAI` over 30 independent trials per gap condition)
- **0 ms gap:** **29/30 (96.7%) Exact Match**, Sequence WER: **1.1%**, 0 duplicate events.
- **50 ms gap:** **27/30 (90.0%) Exact Match** (`game_mode_session`) / **28/30 (93.3%)** (`continuous_valley`), Sequence WER: **2.2% - 3.3%**, 0 duplicate events.
- **Both gap conditions exceed the $\ge 90.0\%$ acceptance threshold.**

### Over-Split & Negative Control Verification
- **Elongated single commands ($L=750$ ms):** **0 over-splits** across 10 trials.
- **Conversational Chatter FA (300.0s):** **0 False Activations** under Game Mode session gating (closed gate during idle/chatter).
- **Phonetic Foils FA (200.0s):** **0 False Activations** under Game Mode session gating.
- **Environmental Noise (100.0s) & Digital Silence (300.0s):** **0 False Activations**.

### Detailed Report & Data Artifacts
Please inspect the full master documentation and data files at:
- Master Report: [`CONTINUOUS_KWS_PHASE6_GAME_MODE_ACCEPTANCE.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/CONTINUOUS_KWS_PHASE6_GAME_MODE_ACCEPTANCE.md)
- Locked Trial Manifest: [`phase6_locked_trial_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_locked_trial_manifest.csv)
- Trial-Level Results: [`phase6_trial_level_results.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_trial_level_results.csv)
- Failure Analysis: [`phase6_failure_analysis.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_failure_analysis.csv)
- Live Latency Benchmark: [`phase6_live_latency_results.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase6_live_latency_results.json)
- Unit Test Suite: [`tests/test_game_mode_phase6.py`](file:///d:/trainidentificationaudio/tests/test_game_mode_phase6.py)
