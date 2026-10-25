# Phase 41: Hard-Negative Corpus and Locked Challenge Benchmark Report

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Path:** [`experiments/vieneu_kws_research_v2/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2)  
**Data Directory:** [`experiments/vieneu_kws_research_v2/data/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/data)  
**Date:** 2026-10-18  
**Governance Marker:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 41 constructs a new, phonetically targeted Vietnamese hard-negative corpus and frozen continuous streaming challenge benchmarks to directly attack the lexical confusion bottleneck identified in Phase 39.

### Key Milestones:
1. **Linguistic Confusion Families:** Designed 100 strictly disjoint non-target Vietnamese texts spanning 5 documented confusion families:
   - `minimal_pair_tone`: Same segmental phonemes, contrasting tone.
   - `minimal_pair_consonant`: Same rhyme and tone, contrasting onset consonant.
   - `minimal_pair_vowel_coda`: Similar vowel nucleus or coda consonant.
   - `phonetic_phrase_context`: 2-syllable compound phrases embedding command keywords.
   - `neutral_speech`: Common everyday conversational and system words.
2. **Strict Voice & Text Disjointness:**
   - 17 Train voices, 4 Validation voices, 4 Test voices (25 total voices, zero overlap).
   - 60 Train non-target texts, 20 Validation texts, 20 Test texts (100 total texts, zero overlap).
3. **Dataset Scale:** 2,000 utterances (16 kHz mono, 1.0s window):
   - Train (68%): 1,360 utterances (680 targets, 340 UNKNOWN hard negatives, 340 SILENCE)
   - Validation (16%): 320 utterances (160 targets, 80 UNKNOWN hard negatives, 80 SILENCE)
   - Test (16%): 320 utterances (160 targets, 80 UNKNOWN hard negatives, 80 SILENCE)
4. **Locked Challenge Streaming Benchmarks:**
   - Validation command stream: $601.4\text{ s} = 0.1671\text{ h}$ ($160$ targets)
   - Validation non-target stream: $697.8\text{ s} = 0.1938\text{ h}$ ($80$ distractor events)
   - Test command stream: $589.6\text{ s} = 0.1638\text{ h}$ ($160$ targets)
   - Test non-target stream: $688.7\text{ s} = 0.1913\text{ h}$ ($80$ distractor events)

---

## 2. Linguistic Confusion Families & Text Inventory

The text inventory is tracked in [`experiments/vieneu_kws_research_v2/data/text_inventory.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v2/data/text_inventory.csv).

| Target Command | Split | Minimal Pair Tone | Minimal Pair Consonant | Minimal Pair Vowel/Coda | Phrase Context | Neutral Speech |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `TRAI` (/ʈaːj˧˥/) | Train | *trai*, *trài*, *trải*, *trại* | *chái*, *sái* | *trát*, *tráo*, *trách*, *trang* | *trái cây*, *bên trái* | *bật*, *tắt*, *mở*, *đóng* |
| `PHAI` (/faːj˧˩˨/) | Train | *phai*, *phài*, *phái*, *phại* | *thải*, *thái*, *khải* | *phát*, *phạt*, *pháp* | *phải chăng*, *bên phải* | *chạy*, *dừng*, *nghe*, *nói* |
| `LEN` (/len˧˧/) | Train | *lèn*, *lén*, *lẻn*, *lẹn* | *nén*, *khen*, *chen*, *đen* | *lớp*, *lực* | *đi lên*, *lên đường* | *hát*, *đọc*, *viết*, *xin chào* |
| `XUONG` (/suəŋ˧˥/) | Train | *xuồng*, *xuổng*, *xướng* | *sương*, *sườn*, *hướng*, *buông*, *chuông* | *xuất*, *xong* | *đi xuống*, *xuống dốc* | -- |
| `TRAI` | Val | *trãi* | *mái* | *tròn* | *trại giam* | *xem* |
| `PHAI` | Val | *phãi* | *mải* | *phay* | *không phải* | *tìm* |
| `LEN` | Val | *lẽn* | *quên* | *lưng* | *tiến lên* | *chọn* |
| `XUONG` | Val | *xuỗng* | *hưởng* | *xanh* | *quê hương* | *bắt đầu* |
| `TRAI` | Test | -- | *lái* | *trước*, *trắng* | *con trai* | *hỏi* |
| `PHAI` | Test | -- | *bải* | *pháo*, *phần* | *phần mềm* | *lưu* |
| `LEN` | Test | -- | *nem* | *leng*, *lòng* | *bước lên* | *xóa* |
| `XUONG` | Test | *xưởng* | -- | *xốp*, *xúc* | *nhà xưởng* | *cảm ơn* |

---

## 3. Disjoint Split Partitioning & Overlap Audit

### 3.1 Voice Roster Disjointness
- **Train (17 voices):** `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`.
- **Validation (4 voices):** `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`.
- **Test (4 voices):** `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`.
- **Overlap Audit:**
  - $\text{Train} \cap \text{Val} = \emptyset$
  - $\text{Train} \cap \text{Test} = \emptyset$
  - $\text{Val} \cap \text{Test} = \emptyset$

### 3.2 Non-Target Text Disjointness
- Non-target text IDs are strictly disjoint between Train (60 texts), Val (20 texts), and Test (20 texts).
- **Target Command Identities:** Shared across splits (40 targets per voice across all 4 commands).

---

## 4. Frozen Asset Checksums & Reproducibility Hashes

| Asset File | Relative Path | SHA-256 Hash |
| :--- | :--- | :--- |
| Master Manifest | `data/manifest.csv` | `c019e8f13fca13df44daebb4d755853b3958b793b56e9ff32f63ebced9c815d2` |
| Train Manifest | `data/splits/train_manifest.csv` | `082191da8ad09c68d4d007ae1ede8af8524aeb407ae0f8285979785d12e32ff6` |
| Validation Manifest | `data/splits/val_manifest.csv` | `b36337f8ae4fe88bb243983189f46e56d5cf73a0afa29e290940cd716fc445f7` |
| Test Manifest | `data/splits/test_manifest.csv` | `d6542941f06bddb0cd6a92582d2cca50c0936ec9bf57ee8dc109d66c4163138f` |
| Text Inventory | `data/text_inventory.csv` | `c193274ff29a898c318e58ade1ec9d2fee59a63bf5def130db7182e782872046` |
| Val Commands Stream | `data/continuous_streams/val_stream_continuous_commands.wav` | `c9010273e96092db87bf889ec47bc08c046e2236c3078139e4219b2db74ef07d` |
| Val Non-Target Stream | `data/continuous_streams/val_stream_continuous_nontarget.wav` | `2ccdac898f5d0dfd45cd08daac212d4156c0efcfb5f5768bbfcdb4bf188859ac` |
| Test Commands Stream | `data/continuous_streams/test_stream_continuous_commands.wav` | `29352991346b14b7cca1c40ed7e3ad823bac5875d130cf81432e4c806e5ab785` |
| Test Non-Target Stream | `data/continuous_streams/test_stream_continuous_nontarget.wav` | `a94b32db5ffb1efa2c4489ddec1636e576b07f410a099c3216dcad6a4c2f0b77` |

---

## 5. Streaming Duration Disclosure

> [!NOTE]
> **Stream Duration & False Alarm Uncertainty Disclosure:**  
> The generated continuous test non-target stream has an exact physical duration of **$688.7\text{ s} = 0.1913\text{ hours}$** ($11.48\text{ minutes}$), incorporating 80 unique test distractor utterances and varied silence/background noise intervals without artificial looping.  
> 
> Because this stream duration is less than 3 hours, achieving zero false alarms over this stream establishes an empirical upper bound of approximately $\le 5.2\text{ FA/hour}$ ($1\text{ FP} / 0.1913\text{ h}$), rather than the strict production requirement of $\le 1.0\text{ FA/hour}$. As required by campaign governance, this limitation is explicitly stated and no duration is artificially inflated.
