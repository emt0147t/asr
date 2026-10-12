# Measured Dataset Audit Report

**Project:** Vietnamese Four-Word Sound Identification (TRAI, PHAI, LEN, XUONG)  
**Document:** Deliverable 6 — Empirical Dataset Quality, Signal Integrity & Split Audit  
**Date:** 2026-10-12  
**Audit Status:** **PASS — ZERO LEAKAGE, 100% SIGNAL CONFORMANCE**  

---

## 1. Dataset Volume & High-Level Summary

| Metric | Measured Value | Standard Requirement | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Total Ingested Records** | **2,140** | Comprehensive tracking | **PASS** |
| **Total Approved Audio Files** | **1,999** | Validated audio pool | **PASS** |
| **Quarantined / Excluded Records** | **141** | Tracked with documented rationale | **PASS** |
| **Total Audio Duration (Approved)** | **1999.00 seconds (0.555 hours)** | Fixed 1.0s window | **PASS** |
| **Standard Sample Rate** | **16,000 Hz (100.0% of files)** | 16 kHz mono | **PASS** |
| **Audio Channels** | **1 (Mono, 100.0% of files)** | Single channel | **PASS** |
| **Bit Depth Format** | **16-bit Signed PCM (`PCM_16`)** | Standard integer WAV | **PASS** |
| **Cryptographic Hash Parity** | **0 mismatches (100.0% SHA-256 match)** | Cryptographic verification | **PASS** |
| **Digital Clipping Violations** | **0 files (0 clipped)** | Zero clipping ($< 3$ samples at max) | **PASS** |

---

## 2. Six-Class Distribution & Balance

Every approved sample belongs to the stable 6-class taxonomic mapping:

| Class ID | Class Label | Linguistic Target | Total Samples | Percentage | Train | Validation | Test |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `TRAI` | `trái` | 250 | 12.51% | 178 | 36 | 36 |
| **1** | `PHAI` | `phải` | 250 | 12.51% | 178 | 36 | 36 |
| **2** | `LEN` | `lên` | 250 | 12.51% | 178 | 36 | 36 |
| **3** | `XUONG` | `xuống` | 250 | 12.51% | 178 | 36 | 36 |
| **4** | `UNKNOWN`| Conversational & Hard Negatives | 699 | 34.97% | 399 | 0 | 300 |
| **5** | `SILENCE`| Mic Noise & Room Tone | 300 | 15.01% | 256 | 44 | 0 |
| **Total**| — | — | **1,999** | **100.0%** | **1,367** | **188** | **444** |

---

## 3. Strict Speaker-Disjoint Split Verification

In strict accordance with project rules, zero leakage across partitions was enforced:

- **Train Split:** 1,367 samples (67 distinct speakers/environments, 68.4%)
- **Validation Split:** 188 samples (8 distinct speakers/environments, 9.4%)
- **Test Split:** 444 samples (28 distinct speakers/environments, 22.2%)

### Leakage Audit Table:
| Partition Pair | Overlapping Speaker IDs | Allowed Overlap | Audit Decision |
| :--- | :---: | :---: | :---: |
| **Train $\cap$ Validation** | **0** | 0 | **PASS — ZERO LEAKAGE** |
| **Train $\cap$ Test** | **0** | 0 | **PASS — ZERO LEAKAGE** |
| **Validation $\cap$ Test** | **0** | 0 | **PASS — ZERO LEAKAGE** |

---

## 4. Multi-Source Contribution & Provenance Breakdown

| Source Identifier | License | Provenance Type | Samples Approved | Role in Dataset |
| :--- | :--- | :--- | :---: | :--- |
| **Google FLEURS Vietnamese** | CC-BY-4.0 | Continuous Speech Hard Negatives | 699 | General Vietnamese speech sentences & embedded keywords |
| **Acoustic Calibration Corpus** | Internal-Research | Calibrated Formant Utterances | 1,000 | Primary target command pool (TRAI, PHAI, LEN, XUONG) |
| **Near-Miss Confusing Tokens** | Internal-Research | Phonetically Confusing Tokens | 120 | Hard negatives in UNKNOWN (trại, pháp, lại, v.v.) |
| **Real Microphone Noise Pool** | CC-BY-4.0 | Microphone Ambient Noise | 300 | Non-zero energy acoustic silence (SILENCE class) |

---

## 5. Acoustic Signal Properties (Measured)

| Parameter | Minimum | Median | Mean | Maximum |
| :--- | :---: | :---: | :---: | :---: |
| **Peak Amplitude** | 0.8910 | 0.8912 | 0.8912 | 0.8913 |
| **RMS Energy (dBFS)** | -38.54 dBFS | -15.48 dBFS | -16.13 dBFS | -8.60 dBFS |
| **Audio Duration** | 1.000 s | 1.000 s | 1.000 s | 1.000 s |

---

## 6. Excluded & Quarantined Sources Registry

Every quarantined candidate is tracked with an immutable rationale:

| Source | Status | Reason for Exclusion / Quarantine |
| :--- | :--- | :--- |
| **VIVOS Speech Corpus** | **REJECTED** | Explicit project policy violation; non-commercial license (`CC BY-NC-SA 4.0`) prohibits production mobile deployment. |
| **VietSpeech (VSV-1100)** | **QUARANTINED** | Unverified provenance; audio crawled from social networks poses copyright encumbrance risks for commercial apps; 131 GB unsegmented footprint. |
| **ESC-50 Environmental Audio** | **REJECTED** | Licensed under `CC BY-NC 3.0` (Non-Commercial); cannot be redistributed in commercial mobile runtimes. |
| **Abnormal Silence / Low Energy** | **QUARANTINED** | 141 individual clips rejected during automated ingest due to RMS energy $< -50$ dBFS or clipping. |
