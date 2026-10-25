# PHASE 42: FRESH ENVIRONMENTAL NOISE ACQUISITION & PROVENANCE REPORT

**Repository Source:** [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)  
**Dataset License:** Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)  
**Categories Selected:** `washing_machine`, `vacuum_cleaner`, `keyboard_typing`, `clock_tick`  
**Standardized Format:** 16,000 Hz, 16-bit Mono PCM WAV  
**Deduplication Scope:** Audited against all 40 historical `mic-noise` tracks (0–39)  
**Audit Status:** **VERIFIED_UNSEEN_PROVENANCE_READY**

---

## 1. Executive Summary & Selection Rationale

Phase 42 resolved the historical noise contamination blocker identified in Phase 37 (where tracks 34–39 were exposed in prior evaluation corpora). To guarantee complete independence from all prior training and evaluation activity, fresh environmental noise was acquired from the official ESC-50 dataset.

### Selection Criteria:
1. **Domestic & Office Acoustical Relevance:** Four indoor categories directly representative of ambient home and office voice-assistant deployments were chosen:
   - `washing_machine`: Low-frequency continuous mechanical rotation.
   - `vacuum_cleaner`: High-energy broadband acoustic noise.
   - `keyboard_typing`: Transient impulsive indoor disturbances.
   - `clock_tick`: Periodic impulse events in quiet rooms.
2. **Provenance & Attribution:** Two distinct takes per category from Fold 1 were retrieved from the official Karol Piczak repository, cross-referenced with their original Freesound source IDs.
3. **Multi-Tier Deduplication Audit:** Every acquired file was audited against all 40 historical `mic-noise` tracks across three detection tiers:
   - Tier 1: Byte-exact SHA-256 match.
   - Tier 2: Decoded 16kHz PCM integer hash match.
   - Tier 3: Normalized cross-correlation check on 3.0-second audio windows.

---

## 2. Acquired ESC-50 Environmental Noise Inventory

| Asset ID | Category | Original Filename | Freesound ID | Native SR | Raw SHA-256 | Processed 16k SHA-256 |
|:---------|:---------|:------------------|:------------:|:---------:|:------------|:----------------------|
| `esc50_1-100210-A-36` | `vacuum_cleaner` | `1-100210-A-36.wav` | `100210` | 44100 Hz | `f97e97b0fe0afce3...` | `1b60d5fe0209450c...` |
| `esc50_1-100210-B-36` | `vacuum_cleaner` | `1-100210-B-36.wav` | `100210` | 44100 Hz | `acfdccbb58647655...` | `4a8ba8d51fe2eeb4...` |
| `esc50_1-137-A-32` | `keyboard_typing` | `1-137-A-32.wav` | `137` | 44100 Hz | `d68e4716f74f073b...` | `4c0dc77e7919966e...` |
| `esc50_1-21896-A-35` | `washing_machine` | `1-21896-A-35.wav` | `21896` | 44100 Hz | `1b4210c50a01caba...` | `d1e6e8140208bdc6...` |
| `esc50_1-21934-A-38` | `clock_tick` | `1-21934-A-38.wav` | `21934` | 44100 Hz | `8d26e40e0e382b0d...` | `3372d3bbe7a999c7...` |
| `esc50_1-21935-A-38` | `clock_tick` | `1-21935-A-38.wav` | `21935` | 44100 Hz | `4d7ad80cf2733e2d...` | `a15145645f536ee7...` |
| `esc50_1-23996-A-35` | `washing_machine` | `1-23996-A-35.wav` | `23996` | 44100 Hz | `cbfd9239fcd7e10a...` | `7275d91b026abee5...` |
| `esc50_1-53501-A-32` | `keyboard_typing` | `1-53501-A-32.wav` | `53501` | 44100 Hz | `7edf3a6b856987de...` | `782f4daca705dc34...` |

---

## 3. Deduplication and Overlap Audit Results

| Asset ID | Category | Byte Duplication | Decoded PCM Match | Max Cross-Correlation | Nearest Historical Track | Audit Verdict |
|:---------|:---------|:----------------:|:-----------------:|:--------------------:|:-------------------------|:-------------:|
| `esc50_1-100210-A-36` | `vacuum_cleaner` | **False** | **False** | 0.0301 | `Wind-Mic Distortion-Violent 1.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-100210-B-36` | `vacuum_cleaner` | **False** | **False** | 0.0366 | `Wind-Mic Distortion-Violent 12.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-137-A-32` | `keyboard_typing` | **False** | **False** | 0.0152 | `Microphone_Touch_and_Movement_2.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-21896-A-35` | `washing_machine` | **False** | **False** | 0.1160 | `Microphone_Touch_and_Movement_2.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-21934-A-38` | `clock_tick` | **False** | **False** | 0.0391 | `RE201 018 CNTCT TRANSFORMER HUM Sustained H1A.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-21935-A-38` | `clock_tick` | **False** | **False** | 0.0173 | `RE201 016 CNTCT CASSETTE Tray Tape Movement Motor Engage H1A.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-23996-A-35` | `washing_machine` | **False** | **False** | 0.0805 | `RE201 018 CNTCT TRANSFORMER HUM Sustained H1A.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |
| `esc50_1-53501-A-32` | `keyboard_typing` | **False** | **False** | 0.0079 | `EFX EXT Mic Cable Coil Actions.mp3` | **VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE** |

---

## 4. Formal Gate Decision for Phase 42

```
NOISE PROVENANCE GATE: PASS
VERDICT: VERIFIED_FRESH_UNSEEN_ENVIRONMENTAL_NOISE
SUMMARY:
- 8 newly acquired, verified environmental noise assets from ESC-50.
- 0% byte duplication, 0% decoded PCM duplication against all 40 historical mic-noise tracks.
- Maximum cross-correlation across all pairs is strictly < 0.15 (statistically independent acoustic sources).
- Processed 16kHz mono audio stored in data/processed/noise/esc50_16k/.
```

---

## 5. Mandatory Governance Declarations

```
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```
