# Phase 59 — System Readiness and Deployment Matrix

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Multi-Dimensional Readiness Matrix

| Evaluation Dimension | Mode A: Bounded-Utterance Mode | Mode B: Continuous Open-Mic Mode | Comparison & Evidence |
| :--- | :--- | :--- | :--- |
| **Operational Interaction** | Push-to-Talk / User-initiated short capture turn | Unconstrained continuous 24/7 background audio | Explicit separation formalized in Phases 50–59 |
| **Target Command Recall** | Required: $\ge 95.0\%$  <br> Achieved: **92.29%** (Cascade v1) / **96.04%** (Stage 1) | Required: $\ge 95.0\%$  <br> Achieved: **74.17%** (Cand 3) / **97.29%** (Historical) | Evaluated on 720 held-out locked test clips |
| **False Activation / FAR** | Required: $\text{FAR} \le 5.0\%$  <br> Achieved: **4.64%** (TNR = **95.36%**) | Required: $\le 1.0\text{ FA/hr}$  <br> Achieved: **10.33 to 77.33 FA/hr** | Cascade v1 achieved $\le 5\%$ FAR goal on test |
| **Decision Latency** | Required: $\le 50.0\,\text{ms}$ on CPU  <br> Achieved: **4.15 ms** (Mean) / **6.98 ms** (P90) | Required: $\le 600\,\text{ms}$  <br> Achieved: **320 ms** sliding window lag | Benchmarked with CPU single-clip batch=1 |
| **Rhyme `/aːj/` Acoustic Dilemma**| **CONTROLLED:** `TRAI` recall is **98.33%**; `PHAI` is **90.83%**; FAs into `PHAI` are only **0.36%** | **UNRESOLVED:** Sliding-window boundary edges trigger false alarms | Bounded joint evaluation suppresses window edge artifacts |
| **Noise Robustness** | Tested under SNR 20dB, 15dB, and gain $-3\text{dB}$ | Tested under pink noise stream ($0.0\text{ gap FAs}$) | Ambient noise immunity verified |
| **Natural Speech Generalization** | **NOT VALIDATED** (Synthetic VieNeu only) | **NOT VALIDATED** (Synthetic VieNeu only) | Needs multi-speaker field data |
| **Human Perceptual Review** | **PENDING** (Listening package created) | **PENDING** (Listening package created) | Awaiting external listener sheets |
| **Formal Deployment Verdict** | **`BOUNDED_UTTERANCE_RESEARCH_FEASIBLE`** | **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`** | Dual-verdict boundary |

---

## 2. Component Specifications (Finalist 1: Cascade v1 Bounded Pipeline)

```
                            Input 1.0-Second Utterance (16 kHz Mono)
                                             │
                                             ▼
                             Log-Mel Feature Extraction (1.4 ms)
                                             │
                                             ▼
                         Stage 1 CNN: VietnameseKWSNet (138k params)
                                             │
                        ┌────────────────────┴────────────────────┐
                        │ Command Prob < θ_s1                     │ Command Prob >= θ_s1
                        ▼                                         ▼
                 Output: REJECT                       Extract 1.2s Audio Crop (19.2k samples)
                                                                  │
                                                                  ▼
                                                      Stage 2 TCN: TemporalKWSVerifier (340k params)
                                                                  │
                                             ┌────────────────────┴────────────────────┐
                                             │ S2 Conf < θ_s2                          │ S2 Conf >= θ_s2
                                             ▼                                         ▼
                                      Output: REJECT                         Output: Target Command
                                                                             (TRAI / PHAI / LEN / XUONG)
```

- **Total Network Parameters:** 479,307 parameters (~1.9 MB FP32 uncompressed).
- **RAM Footprint:** $< 12\,\text{MB}$ peak inference buffer.
- **CPU Execution Cost:** 4.15 ms on host CPU.
