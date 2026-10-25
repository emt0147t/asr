# Phase 78 — Controlled Acoustic Playback-to-Microphone Feasibility Study

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v9/`  
**Execution Status:** **`NOT_RUN`**  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Operating Status & Technical Blocker Disclosure

Per research governance protocol, Phase 78 is permitted strictly when suitable physical loudspeaker and microphone hardware is available and can be tested without collecting human speech.

### Hardware & Environment Audit
- **Execution Environment:** Windows Subsystem / Automated Agentic Environment (`D:\trainidentificationaudio`).
- **Audio Capture Hardware:** No physical microphone or loopback audio input interface is connected or accessible to the process.
- **Software Dependencies:** Low-level physical audio I/O drivers/libraries (`sounddevice`, `pyaudio`) are absent from the runtime environment.
- **Formal Status Determination:**
  $$\mathbf{Status:}\quad \mathbf{NOT\_RUN}$$

> [!CAUTION]
> **Non-Fabrication Policy:**  
> In strict accordance with research integrity requirements, physical microphone captures must **never** be simulated in code or fabricated with synthetic noise and claimed as real physical captures. Because the necessary hardware and capture permissions are unavailable, Phase 78 is honestly marked as `NOT_RUN`.

---

## 2. Standard Acoustic Playback-to-Microphone Protocol Specification

When physical laboratory hardware becomes available, the following protocol must be executed without modification:

### A. Controlled Acoustic Loopback Setup
1. **Playback Source:** High-fidelity studio reference monitor (calibrated at 65 dBA SPL at 1 meter).
2. **Microphone Interface:** Calibrated omnidirectional measurement microphone positioned at fixed standard distances (0.5 m, 1.0 m, 2.0 m).
3. **Room Acoustic Envelope:** Controlled conference room or semi-anechoic enclosure with documented reverberation time ($T_{60} \le 0.35\text{ s}$).
4. **Input Audio Material:** Approved synthetic audio clips from the frozen benchmark only (no human speech permitted).

### B. Machine-Readable Hardware & Capture Metadata Ledger
Every capture must log:
- Timestamp (ISO 8601 UTC)
- Capture Device ID, Manufacturer, Model, Serial Number
- Sample Rate (16,000 Hz, 16-bit PCM)
- Preamp Gain (dB) and Measured SPL (dBA)
- Physical Speaker Distance and Angle
- Ambient Background Noise Floor (dBA)
- SHA-256 Digest of the raw captured `.wav` file

### C. Data Safety and Isolation
- All acoustic loopback captures must reside exclusively in:
  `experiments/vieneu_kws_research_v9/acoustic_loopback/captures/`
- Zero access to protected human field recordings or consent ledgers is permitted under any circumstance.

---

## 3. Anticipated Metric Comparisons & Failure Mode Analysis

When Phase 78 is run in a physical acoustic setup, the frozen candidate `variant_w10` should be compared between direct synthetic files and captured loopback audio:
- **Expected Degradation Vectors:**
  1. High-frequency roll-off from loudspeaker / microphone frequency response curves.
  2. Room resonances (standing waves) in low-frequency vowel formants (/uə/ in `XUONG`).
  3. Ambient air conditioning HVAC noise altering silent boundary regions.
- **Decision Criteria for Feasibility:**
  - Macro command recall retention $\ge 85.0\%$ of original clean baseline.
  - Non-target false alarms on captured ambient silence $\le 1.0\%$.

---

## 4. Governance & Compliance Sign-Off

- Physical Audio Collected: **0 clips** (None)
- Human Speech Collected: **0 clips** (Strictly Forbidden)
- Synthetic Data Disguised as Hardware Capture: **0 clips** (Strictly Zero Fabrication)
- Continuous Open-Mic Status: **STRICTLY BLOCKED (`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`)**
