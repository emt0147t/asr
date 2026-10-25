# Phase 50 — Independent Benchmark and Metric Audit Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v4/`  
**Audit Script:** `experiments/vieneu_kws_research_v4/run_phase50_audit.py`  
**Audit Findings JSON:** `experiments/vieneu_kws_research_v4/audit_findings_phase50.json`  
**Status:** `VERIFIED_INDEPENDENT_AUDIT` | `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

This independent audit evaluated the integrity of the 3-hour continuous challenge stream, verified the audio metadata and bit-exact SHA-256 hashes, audited repetition and diversity statistics across synthetic non-target speech, conducted non-overlapping false-alarm error categorization, and performed full metric recomputation against Phase 49 prediction logs.

### Key Audit Conclusions:
1. **Audio & Hash Integrity:** All continuous streams (Validation Commands, Validation Non-target, Test Commands, Test Non-target 3-Hour) matched their recorded metadata and SHA-256 checksums bit-for-bit.
2. **Benchmark Repetition & Diversity:** The 3-hour continuous non-target stream contains exactly 960 speech events comprising 480 unique synthetic utterances spoken across 4 held-out test voices (Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí; 240 events each), structured in exactly 2 full passes (2.0 repetitions per utterance). The non-target speech spans 5 confusion families (neutral speech, consonant minimal pairs, vowel/coda minimal pairs, phrase contexts, and tonal minimal pairs).
3. **Non-Overlapping False Alarm Categorization:** By partitioning the timeline into three strictly disjoint sets—`active_speech`, `boundary_margins`, and `inter_utterance_gaps`—we proved:
   - The Historical Cascade v1 suffered 68.6%–72.2% of its false activations directly within **active speech**, failing to discriminate hard negatives.
   - The Recall-Preserving Stage 1 fired heavily on **boundary margins** (66.8%–74.9%), indicating high sensitivity to speech onset/offset energy transitions.
   - The Recall-Preserving Cascade (Candidate 3) eliminated virtually all **inter-utterance gap** false alarms ($0.0\%$ gap FAs across all seeds), with residual FAs primarily residing in boundary margins ($60.7\%\text{--}88.5\%$) and a minor fraction in active speech.
4. **Metric Reproduction:** Exact 100% reproduction of all Phase 49 reported metrics across all 3 candidates and all 3 random seeds (42, 43, 44), with zero metric discrepancies.

---

## 2. Stream Audio Metadata & SHA-256 Verification

| Stream ID | File Path | Duration (s) | Duration (hr) | Sample Rate | Frames | Ground Truth Events | SHA-256 Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `val_commands` | `.../val_stream_continuous_commands.wav` | 514.951s | 0.1430 hr | 16,000 Hz | 8,239,214 | 160 | `MATCH` (`e58a7b4d...`) |
| `val_nontarget` | `.../val_stream_continuous_nontarget.wav` | 478.458s | 0.1329 hr | 16,000 Hz | 7,655,331 | 120 | `MATCH` (`771be7e5...`) |
| `test_commands` | `.../test_stream_continuous_commands.wav` | 519.149s | 0.1442 hr | 16,000 Hz | 8,306,389 | 160 | `MATCH` (`7ee89c95...`) |
| `test_nontarget_3h` | `.../test_stream_continuous_nontarget_3h.wav` | 10,800.000s | 3.0000 hr | 16,000 Hz | 172,800,000 | 960 | `MATCH` (`1d325a33...`) |

All waveform files are mono 16-bit PCM at 16,000 Hz. The 3-hour benchmark duration is precisely 10,800.00 seconds.

---

## 3. Repetition and Diversity Audit of the 3-Hour Benchmark

- **Total Speech Events:** 960
- **Unique Synthetic Utterance Files:** 480
- **Unique Non-Target Text Transcripts:** 120
- **Unique Held-Out Test Voices:** 4
- **Pass Structure:** Exactly 2 passes across all 480 test utterances.
  - Min repetitions per utterance: 2
  - Max repetitions per utterance: 2
  - Mean repetitions: 2.0
- **Test Voice Distribution:**
  - `Kim Thanh`: 240 events (25.0%)
  - `Quốc Tuấn`: 240 events (25.0%)
  - `Quỳnh Anh`: 240 events (25.0%)
  - `Đức Trí`: 240 events (25.0%)
- **Confusion Family Distribution:**
  - `neutral_speech`: 224 events (23.33%)
  - `minimal_pair_consonant`: 192 events (20.00%)
  - `minimal_pair_vowel_coda`: 192 events (20.00%)
  - `phonetic_phrase_context`: 192 events (20.00%)
  - `minimal_pair_tone`: 160 events (16.67%)

The benchmark provides a well-balanced acoustic test covering tonal variations (hỏi, ngã, sắc, huyền, nặng), initial consonant confusions (e.g., tr vs ch, ph vs th), vowel rhymes (e.g., /aːj/, /ɛŋ/, /uəŋ/), and conversational non-target speech.

---

## 4. Non-Overlapping False Alarm Categorization

To resolve the distinction between event-matching tolerance and error categorization, the continuous timeline is partitioned into three disjoint, non-overlapping subsets:
1. **`active_speech`**: The exact interval during which an utterance is spoken:
   $$T_{\text{active}} = \bigcup_{i} [t_{i, \text{onset}}, t_{i, \text{end}}]$$
2. **`boundary_margins`**: Slices immediately preceding onset or trailing offset, capturing coarticulation tails and sliding window overlap without falling inside speech:
   $$T_{\text{boundary}} = \bigcup_{i} \Big( [t_{i, \text{onset}} - 0.20\,\text{s}, t_{i, \text{onset}}) \cup (t_{i, \text{end}}, t_{i, \text{end}} + 0.60\,\text{s}] \Big) \setminus T_{\text{active}}$$
3. **`inter_utterance_gaps`**: Background noise/silence intervals outside active speech and boundary margins:
   $$T_{\text{gap}} = [0, T_{\text{total}}] \setminus (T_{\text{active}} \cup T_{\text{boundary}})$$

### Categorization Results Across Evaluated Models:

| Model Architecture | Seed | Total FAs | Active Speech FAs (%) | Boundary Margin FAs (%) | Gap FAs (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Cascade v1** | 42 | 207 | 142 (68.60%) | 63 (30.43%) | 2 (0.97%) |
| | 43 | 245 | 177 (72.24%) | 66 (26.94%) | 2 (0.82%) |
| | 44 | 244 | 176 (72.13%) | 68 (27.87%) | 0 (0.00%) |
| **Recall-Preserving Stage 1** | 42 | 710 | 230 (32.39%) | 474 (66.76%) | 6 (0.85%) |
| | 43 | 570 | 136 (23.86%) | 427 (74.91%) | 7 (1.23%) |
| | 44 | 543 | 129 (23.76%) | 406 (74.77%) | 8 (1.47%) |
| **Recall-Preserving Cascade** | 42 | 13 | 2 (15.38%) | 11 (84.62%) | **0 (0.00%)** |
| | 43 | 28 | 11 (39.29%) | 17 (60.71%) | **0 (0.00%)** |
| | 44 | 52 | 6 (11.54%) | 46 (88.46%) | **0 (0.00%)** |

### Key Acoustic Insights:
- In the **Recall-Preserving Cascade (Candidate 3)**, background pink noise in inter-utterance gaps caused **zero false alarms** ($0.00\%$). The cascade's Stage 2 TCN verifier successfully suppressed ambient noise triggers.
- The majority of residual false activations in the cascade ($60.7\%\text{--}88.5\%$) occurred in the **boundary margins** ($t_{\text{end}} \text{ to } t_{\text{end}} + 0.60\,\text{s}$). This corresponds to window-edge effects where the trailing phone of a non-target utterance lingers in the sliding window buffer alongside noise.
- Active speech false alarms in Candidate 3 were reduced from $142\text{--}177$ down to $2\text{--}11$, demonstrating the potency of the hard-negative training.

---

## 5. Metric Reproduction Verification

Recomputation of all primary metrics from serialized prediction JSON files confirms exact numerical agreement with Phase 49 reports:

| Model | Seed | Target Recall | Non-Target FPs | Wall FA Rate (FA/hr) | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Historical Cascade v1 | 42 | 98.75% | 207 | 69.00 FA/hr | `MATCH` |
| | 43 | 96.25% | 245 | 81.67 FA/hr | `MATCH` |
| | 44 | 96.88% | 244 | 81.33 FA/hr | `MATCH` |
| Recall-Preserving Stage 1 | 42 | 96.88% | 710 | 236.67 FA/hr | `MATCH` |
| | 43 | 99.38% | 570 | 190.00 FA/hr | `MATCH` |
| | 44 | 98.12% | 543 | 181.00 FA/hr | `MATCH` |
| Recall-Preserving Cascade | 42 | 60.62% | 13 | 4.33 FA/hr | `MATCH` |
| | 43 | 70.00% | 28 | 9.33 FA/hr | `MATCH` |
| | 44 | 91.87% | 52 | 17.33 FA/hr | `MATCH` |

**Mean across seeds for Recall-Preserving Cascade:**
- Mean Command Recall: **74.17%** ($60.62\%, 70.00\%, 91.87\%$)
- Mean Wall-Clock FA Rate: **10.33 FA/hr** ($4.33, 9.33, 17.33$)
- Perfect reproduction confirmed (`is_perfect_reproduction: true`, 0 mismatches).
