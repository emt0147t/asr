# Phase 64 — Operational Readiness Matrix

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. System Readiness Matrix Across Operating Modes

| Evaluation Dimension | Mode A: Bounded-Utterance (Push-to-Talk) | Mode B: Continuous Open-Mic (Always-Listening) |
| :--- | :--- | :--- |
| **Acoustic Paradigm** | Discrete 1.0s speech capture | Unsegmented, continuous audio stream |
| **Target Recall Target** | $\ge 95.0\%$ per keyword | $\ge 95.0\%$ event recall |
| **False Positive Ceiling** | $\le 5.0\%$ clip FAR | $\le 1.0$ false alarm / hour |
| **Latency Ceiling** | $\le 50\,\text{ms}$ post-capture CPU | $\le 600\,\text{ms}$ mean event detection |
| **Best Current Candidate** | Cascade v1 / Bounded CNN | Two-Stage Cascade v1 |
| **Demonstrated Performance** | $\approx 92\text{--}96\%$ recall, $\approx 4.6\text{--}8.5\%$ FAR, $\sim 2\text{--}7\,\text{ms}$ CPU | $\sim 96.25\%$ recall, $\sim 69.2$ false alarms/hour |
| **Deployment Verdict** | **FEASIBLE RESEARCH PROTOTYPE** | **NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION** |
| **Deployment Eligibility** | Gated to synthetic research evaluation only | **STRICTLY BLOCKED** |
| **Human Validation Status** | `PENDING_HUMAN_LISTENER_VERIFICATION` | `PENDING_HUMAN_LISTENER_VERIFICATION` |
