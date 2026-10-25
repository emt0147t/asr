# Phase 83 — Physical Acoustic Loopback Feasibility Protocol Report

**Workspace:** `D:\trainidentificationaudio`  
**Package:** `experiments/vieneu_kws_research_v10/acoustic_loopback/`  
**Execution Status:** **`NOT_RUN`**  
**Status Banners:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Operating Status & Hardware Blocker Disclosure

Per research governance protocol, Phase 83 investigates whether physical loudspeaker playback and microphone re-recording can be executed without collecting human speech.

### Runtime Hardware & Backend Inspection:
1. **Audio Capture Hardware:** No physical microphone, studio reference loudspeaker, or audio capture peripheral is attached or accessible to the automated Windows subshell runtime.
2. **Audio Backend Drivers:** Software audio driver packages (`sounddevice`, `pyaudio`) are not present in the runtime environment.
3. **Formal Determination:**
   $$\mathbf{Status:}\quad \mathbf{NOT\_RUN}$$

> [!CAUTION]
> **Strict Non-Fabrication Policy:**  
> In accordance with research integrity standards:
> - Physical microphone captures must **never** be simulated in software (e.g. by applying synthetic noise or equalization filters in Python and claiming it represents physical microphone hardware).
> - Fake loopback `.wav` files were **not** generated.
> - Because physical hardware is unavailable to this automated environment, Phase 83 is recorded honestly as `NOT_RUN`.

---

## 2. Standard Physical Loopback Laboratory Protocol

When physical acoustic hardware is connected, the following protocol must be executed without modification using `run_loopback_capture.py`:

### A. Laboratory Environment & Acoustic Calibration
1. **Playback Loudspeaker:** Nearfield studio monitor (e.g. Genelec 8010A or Yamaha HS5) placed on an acoustic isolation stand.
2. **Measurement Microphone:** Calibrated omnidirectional measurement microphone (e.g. miniDSP UMIK-1 or Earthworks M23).
3. **Test Distances:** Standardized distances of 0.5 m, 1.0 m, and 2.0 m on-axis (0 degrees) and off-axis (30 degrees, 45 degrees).
4. **Sound Pressure Level:** Playback calibrated to 65 dBA SPL at 1 meter using a sound level meter.
5. **Room Envelope:** Controlled room with documented reverberation time ($T_{60} \le 0.35\text{ s}$) and background noise floor $\le 35\text{ dBA}$.

### B. Prohibitions & Human Speech Protection
- **No Human Speech:** Under no circumstance may human speech, conversations, or unscripted environmental voices be recorded.
- **Approved Audio Only:** Only approved synthetic clips from the frozen benchmark may be played and captured.
- **Data Isolation:** All captured recordings must be stored in:
  `experiments/vieneu_kws_research_v10/acoustic_loopback/captures/`
  with zero access to protected historical field data.

### C. Machine-Readable Metadata Ledger
Every physical capture must record in `recording_manifest_template.csv`:
- Unique `capture_id` and UTC timestamp.
- Source synthetic clip ID and SHA-256 hash.
- Playback device model and recording device model.
- Sample rate (16,000 Hz, 16-bit PCM mono), duration, and distance.
- Preamp gain (dB) and digital peak level.
- Clipping detection flag (`Y/N`).
- Captured file SHA-256 digest and status (`success`, `clipped`, or `dropped`).

---

## 3. Anticipated Model Evaluation Protocol

When physical captures are obtained:
1. **Calibration Set vs. Test Set Isolation:** First evaluate on a dedicated development subset. Keep a separate captured loopback test subset untouched until configuration is frozen.
2. **No Automatic Retuning:** Evaluate the frozen candidate `variant_w10` at its fixed calibrated thresholds ($\theta_{\text{cmd}} = [0.40, 0.50, 0.35]$).
3. **Evaluation Framing:** Report results strictly as an *acoustic playback-to-microphone feasibility experiment*, not as natural human speech validation.
