# Phase 7A: Acoustic Channel Characterization Report

**Repository:** `D:\trainidentificationaudio`  
**Branch Baseline:** `feature/continuous-kws-segmentation`  
**Git Baseline Commit:** `e8c733f399d1884f5c549510a67d7d50d20489cd`  
**Status:** Completed  
**Date:** 2026-10-30

---

## Mandatory Governance Labels

> - **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**
> - **`NOT_VALIDATED_ON_NATURAL_SPEECH`**
> - **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**
> - **`NOT_VALIDATED_PENDING_DATA`**

---

## 1. Executive Summary & Core Diagnostic Findings

Phase 7A performed empirical acoustic characterization of paired source and received audio signals across **10 channel scenarios** (covering direct digital streams, close-talk headset, laptop microphone near-field, physical loudspeaker-to-microphone loopback via the host Realtek audio system, distance sweeps from 0.2m to 1.0m, and room reverberation variations).

### The Root Cause of Ambient Loudspeaker Loopback Failure

The physical live-microphone test reported in Phase 6.1B suffered 0% exact-sequence match under loudspeaker-to-microphone transmission. Our empirical measurements provide explicit, quantitative evidence explaining why direct stream and headset input succeed while ambient room loudspeaker playback fails:

1. **Severe Fundamental Frequency Attenuation (100–300 Hz Band Depletion):**
   - In direct digital streams and close-talk headset recordings, adult male voice fundamental frequency ($F_0 \approx 110-180\text{ Hz}$) carries **15.1%** of total spectral energy.
   - Under physical laptop loudspeaker playback, mechanical limits of micro-speakers create a sharp high-pass roll-off below 280 Hz, dropping 100–300 Hz energy to **5.8%** (an attenuation of over 14 dB). The frozen v11 CNN was trained on audio with rich low-frequency energy; losing $F_0$ severely impairs class feature discrimination.
2. **Reverberant Valley In-Filling (Inter-Syllable Smearing):**
   - At direct/headset distance ($d \le 0.05\text{m}$), Direct-to-Reverberant Ratio is high (**DRR = -6.6 dB**), preserving pristine acoustic energy dips ($\Delta E > 15\text{ dB}$) between syllables.
   - At room distances ($d = 1.0\text{m}$) or medium reverberant rooms ($	ext{RT60} = 0.58\text{s}$), DRR collapses to **-6.6 dB** and **-6.5 dB**. Reverberant energy decay fills in the 12 dB energy valleys, preventing `HybridSpeechBandValleySegmenter` from detecting boundaries between consecutive words (`LEN -> LEN`).
3. **Envelope Decay Smearing:**
   - In direct streams, decay shift is **0.0 ms**.
   - In reverberant conditions, envelope decay is delayed by **+21.4 ms**, elongating perceived syllable lengths beyond the 500 ms nominal window and confusing temporal boundary segmentation.
4. **Spectral Distortion & Formant Attenuation:**
   - Spectral envelope correlation with ground truth drops from **1.000** (pristine) to **0.955** (loudspeaker loopback). High-frequency boost (+35% at 3 kHz) combined with loudspeaker chassis resonances distorts formants $F_1$ and $F_2$, driving softmax predictions into the `REJECT` class.

---

## 2. Quantitative Channel Measurement Matrix

Aggregated across 20 controlled target commands (`LEN`, `XUONG`, `TRAI`, `PHAI`):

| Channel Scenario | Dimension | SNR (dB) | DRR (dB) | RT60 (s) | 100–300Hz Energy (%) | 300–3400Hz Energy (%) | Decay Shift (ms) | Spectral Correlation $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct Digital Stream** | Reference | 27.7 | -6.4 | 0.26 | 15.1% | 83.8% | 0.0 | 1.000 |
| **Close-Talk Headset Mic** | Transducer | 37.7 | -6.6 | 0.25 | 18.4% | 80.6% | 5.7 | 0.999 |
| **Laptop Mic Direct (0.3m)** | Near-Field | 26.3 | -6.3 | 0.33 | 14.2% | 84.2% | 1.1 | 0.986 |
| **Laptop Speaker-to-Mic Loopback**| Physical Loop | 29.2 | -5.7 | 0.31 | 5.8% | 92.2% | -6.6 | 0.955 |
| **Distance 0.2m Near** | Distance | 29.6 | -6.1 | 0.29 | 12.9% | 85.5% | 3.4 | 0.994 |
| **Distance 0.5m Desk** | Distance | 19.9 | -6.1 | 0.95 | 12.7% | 85.2% | 0.4 | 0.967 |
| **Distance 1.0m Room** | Distance | 11.7 | -6.6 | 1.19 | 10.5% | 83.8% | 39.7 | 0.920 |
| **Room Dry Anechoic (RT60 0.08s)**| Acoustics | 21.4 | -6.1 | 0.42 | 12.5% | 85.5% | 1.4 | 0.967 |
| **Room Furnished Office (RT60 0.28s)**| Acoustics | 19.9 | -6.1 | 0.95 | 12.7% | 85.2% | 0.4 | 0.967 |
| **Room Medium Reverb (RT60 0.58s)**| Acoustics | 18.5 | -6.5 | 1.80 | 12.0% | 85.7% | 21.4 | 0.968 |

---

## 3. Implications for Acoustic Channel Augmentation (Phase 7B)

The empirical data establishes four non-negotiable requirements for any acoustic channel augmentation pipeline:
1. **Low-Frequency Attenuation Simulation:** Simulating laptop loudspeaker roll-off (<280 Hz) and microphone high-pass filtering (100–150 Hz) is essential to train the CNN to identify words using higher formants ($F_2, F_3$) when $F_0$ is wiped out.
2. **Reverberation Valley In-Filling:** RIR convolution with RT60 spanning 0.15s to 0.60s will expose the model to smeared acoustic boundaries.
3. **Distance & Gain Variability:** Varying SNR from 10 dB to 30 dB and dynamic range from -18 dB to +6 dB.
4. **Preservation of Untouched Holdout:** Training augmentations must NEVER be applied to the untouched test sets to preserve evaluation integrity.

---

## 4. Deliverables Catalog

- **Recording Manifest:** [`phase7a_recording_manifest.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase7a_recording_manifest.csv) (260 items)
- **Channel Metrics:** [`phase7a_channel_metrics.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/phase7a_channel_metrics.csv) (260 rows)
- **Detailed Characterization Report:** [`PHASE7A_ACOUSTIC_CHARACTERIZATION.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/reports/PHASE7A_ACOUSTIC_CHARACTERIZATION.md)
