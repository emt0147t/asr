# Phase 88 — Conditional Physical Playback-to-Microphone Acoustic Test Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment:** `experiments/vieneu_kws_research_v10/acoustic_loopback/`  
**Candidate Model:** `variant_w10` (Frozen)  
**Execution Status:** `NOT_RUN`  
**Acoustic Feasibility Verdict:** `UNRESOLVED`  
**Natural Speech Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Date:** 2026-10-18  

---

## 1. Executive Summary & Runtime Hardware Audit

Phase 88 evaluated the feasibility of conducting real physical loudspeaker-to-microphone acoustic loopback testing for the frozen Vietnamese KWS candidate `variant_w10`.

At execution time, the automated audit script [`run_loopback_capture.py`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/acoustic_loopback/run_loopback_capture.py) inspected the operating system and Python environment for available physical audio playback and recording hardware.

### Runtime Audit Results:
- **Python Audio Backends (`sounddevice` / `pyaudio`):** Not installed in virtual environment `.venv`.
- **Physical Input Endpoints Detected by Software:** 0
- **Physical Output Endpoints Detected by Software:** 0
- **Physical Acoustic Rig Status:** Not attached or calibrated.
- **Execution Status:** **`NOT_RUN`**

---

## 2. Strict Anti-Fabrication & Governance Declaration

In strict compliance with campaign research rules:
1. **Zero Software Simulation:** Physical acoustic propagation (loudspeaker transducer response, room boundary reflections, microphone polar pattern, air absorption) was **NOT** simulated using software filters, artificial reverb, or synthetic noise.
2. **Zero Fabricated Recordings:** Exactly 0 files have been placed in `acoustic_loopback/captures/`. The recording manifest template [`recording_manifest_template.csv`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v10/acoustic_loopback/recording_manifest_template.csv) remains empty.
3. **Verdict Remains Unresolved:** Without empirical acoustic measurements, physical acoustic feasibility cannot be claimed. The verdict is explicitly marked **`UNRESOLVED`**.
4. **Natural Speech Distinction:** Playback-to-microphone loopback tests synthetic TTS waveforms transmitted through physical air; it does **not** evaluate natural human speech articulation, coarticulation, or prosody.

---

## 3. Physical Hardware Blockers & Prerequisites

To execute Phase 88 in a physical acoustic laboratory, the following hardware and software stack must be provisioned:

| Prerequisite Layer | Item Required | Purpose / Specification | Current State |
| :--- | :--- | :--- | :--- |
| **Driver / Library** | `sounddevice` or `pyaudio` | Low-latency audio I/O streaming from Python | Missing in `.venv` |
| **Transducer (Output)** | Calibrated reference loudspeaker | Flat frequency response ($80\text{ Hz} - 16\text{ kHz}$) (e.g. Genelec 8010A or equivalent) | None attached |
| **Transducer (Input)** | Calibrated measurement microphone | Omnidirectional or cardioid reference microphone (e.g. miniDSP UMIK-1) | None attached |
| **Physical Geometry** | Fixed measurement rig | Calibrated distances: $0.5\text{ m}$ (near-field), $1.0\text{ m}$ (standard desktop), $2.0\text{ m}$ (far-field) | Unavailable |
| **Acoustic Environment** | Calibrated room | Documented RT60 reverberation time and ambient background noise level ($\le 35\text{ dBA}$) | Unmeasured |

---

## 4. Laboratory Execution Protocol (When Unblocked)

When physical hardware is connected:
1. **Source Selection:** Select 120 approved synthetic evaluation clips from held-out partitions (60 command clips, 60 hard-negative non-targets).
2. **Development Calibration Run:** Play 10 calibration sweeps/tones to adjust preamp gain, ensuring peak headroom of $-3.0\text{ dBFS}$ without clipping.
3. **Capture Execution:** Stream clips sequentially, recording through the microphone at $16\text{ kHz}$ 16-bit mono PCM into `acoustic_loopback/captures/`.
4. **Metadata Ledger:** Populate `recording_manifest.csv` with capture timestamp, device IDs, physical distance, measured SNR, clipping flag, and SHA-256 hashes of both source and capture.
5. **Frozen Inference:** Score captured waveforms with `variant_w10` using frozen thresholds ($\theta_{\text{cmd}} = [0.40, 0.50, 0.35]$). Prohibit threshold tuning on held-out loopback test captures.

Until this physical protocol is executed, Phase 88 remains strictly **`NOT_RUN`**.
