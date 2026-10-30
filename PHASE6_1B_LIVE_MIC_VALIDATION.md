# Phase 6.1B: Physical Live-Microphone Event Validation

**Full Report Location:** [`experiments/vieneu_kws_research_v13_4/reports/PHASE6_1B_LIVE_MIC_VALIDATION.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/PHASE6_1B_LIVE_MIC_VALIDATION.md)  
**Status:** Completed  
**Workstream:** Phase 6.1B  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## Key Results Summary

1. **Physical Microphone Validation:** Evaluated on physical host hardware using `sounddevice` with `Microphone (2- Realtek(R) Audio)` and `Speakers (2- Realtek(R) Audio)`.
2. **Offline Replay vs Physical Live-Mic Discrepancy:**
   - Synthetic direct-digital injection achieves **96.7% (0 ms)** and **90.0% (50 ms)** exact match.
   - Physical acoustic over-the-air playback through laptop speaker into microphone achieves **0/10 (0.0%)** exact match on both gap conditions due to severe acoustic distortion, room reverberation, and loudspeaker coloration on the frozen v11 CNN classifier.
3. **Latency Profile:**
   - Chunk transit + Segmenter + ONNX inference: **2.15 ms** (p50)
   - Command offset to event emission: **497.6 ms** (p50)
   - Command onset to event emission: **757.9 ms** (p50)
   - Buffer overflows: **0** across all physical runs.
4. **Session Gating & Flush:**
   - Gate closed completely blocks 100% of loud acoustic playback (**0 false events**).
   - Abrupt stream termination recovers cut-off audio via `flush()`.
