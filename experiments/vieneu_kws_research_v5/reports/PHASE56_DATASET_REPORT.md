# Phase 56 — Bounded-Utterance Dataset Construction Report (Dataset v5)

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Directory:** `experiments/vieneu_kws_research_v5/`  
**Dataset Directory:** `experiments/vieneu_kws_research_v5/data/`  
**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Executive Summary

Phase 56 constructed a brand-new, comprehensive synthetic benchmark dataset (`research_v5.0_bounded`) specifically designed for rigorous bounded-utterance keyword spotting evaluation.

### Key Milestones Achieved:
1. **Zero Contamination from Historical Test Data:** All 720 locked test audio files (160 target commands, 480 hard-negative distractors, 80 silence files) were newly generated with fresh random seeds, distinct temperature profiles, and newly defined texts. Not a single audio clip or text from prior test sets was recycled for training or calibration.
2. **Strict Text Disjointness Across All Splits:** 200 distinct Vietnamese non-target texts were curated across tonal minimal pairs, consonant minimal pairs, vowel/coda variations, phrase contexts, and natural conversational sentences:
   - **Train:** 80 distinct texts
   - **Validation:** 40 distinct texts
   - **Test:** 80 distinct texts
   - **Intersection:** Exactly 0 text overlap between any pair of splits ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
3. **Disjoint Voice Allocations:** Preserved the 25-voice speaker partition across 17 training voices, 4 validation voices, and 4 held-out test voices (Kim Thanh, Quốc Tuấn, Quỳnh Anh, Đức Trí).
4. **Balanced Target Commands:** Every split contains strictly balanced target command instances across `TRAI`, `PHAI`, `LEN`, and `XUONG` ($170$ per keyword in Train, $40$ per keyword in Val, $40$ per keyword in Test).
5. **Acoustic Background Diversity:** 500 synthetic silence clips distributed evenly across four acoustic profiles (`near_silence`, `hvac_hum`, `office_ambient`, `airflow_pink`).

---

## 2. Dataset Split Breakdown

| Split | Target Commands (`TRAI`, `PHAI`, `LEN`, `XUONG`) | Non-Target `UNKNOWN` Distractors | Synthetic `SILENCE` Clips | Total Utterances | Allocated Speaker Voices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 680 (170 per command) | 340 (cycling 80 disjoint texts) | 340 (85 per noise profile) | **1,360** | 17 Train Voices |
| **Val** | 160 (40 per command) | 120 (cycling 40 disjoint texts) | 80 (20 per noise profile) | **360** | 4 Val Voices |
| **Test** | 160 (40 per command) | 480 (cycling 80 disjoint texts) | 80 (20 per noise profile) | **720** | 4 Held-Out Test Voices |
| **Total** | **1,000** | **940** | **500** | **2,440** | **25 Voices** |

---

## 3. Linguistic and Phonetic Confusion Taxonomy

The 200 non-target texts span five primary acoustic and phonetic categories:

### A. Minimal Pair Tone Confusions:
- Focuses on pitch contour confusions on open rhymes (/aːj/, /ɛŋ/, /uəŋ/):
  - `TRAI` targets: *trài* (huyền), *trải* (hỏi), *trại* (nặng), *tráng* (sắc with nasal coda), *trải dài*, *con trai*.
  - `PHAI` targets: *phài* (huyền), *phái* (sắc), *phại* (nặng), *phàn* (huyền nasal), *phai mờ*, *thời gian phai*.
  - `LEN` targets: *lèn* (huyền), *lén* (sắc), *lẻn* (hỏi), *lẹn* (nặng), *lén lút*, *lẻn vào*.
  - `XUONG` targets: *xuồng* (huyền), *xuổng* (hỏi), *xướng* (sắc), *xưởng mộc*, *xuồng ba lá*, *nhà xưởng*.

### B. Minimal Pair Consonant Confusions:
- Contrasts initial phoneme bursts, fricatives, and nasals against target onsets:
  - Retroflex /ʈ/ vs Affricate /tɕ/: *chái, chài, chải, chén*.
  - Retroflex /ʈ/ vs Sibilant /s/: *sái, sen*.
  - Labiodental /f/ vs Aspirate Dental /tʰ/: *thải, thái*.
  - Labiodental /f/ vs Velar Aspirate /kʰ/: *khải*.
  - Liquid /l/ vs Nasal /n/: *nén, nem, men*.
  - Fricative /s/ vs Glottal /h/ & Bilabial /b/: *hướng, buông, chuông, vuông, luồng*.

### C. Phonetic Phrase Contexts & Compounds:
- Targets embedded inside compound words and locative phrases:
  - *trái tim, trái phiếu, bên trái, rẽ trái*
  - *phải chăng, bên phải, rẽ phải, phân phái*
  - *đi lên, lên đường, leo lên, đứng lên*
  - *đi xuống, xuống dốc, trượt xuống, ngồi xuống*

### D. General Conversational & Home-Automation Neutral Speech:
- Monosyllabic and disyllabic conversational phrases:
  - Device controls: *bật, tắt, mở, đóng, chạy, dừng, xem, tìm, chọn, hủy, quay lại, tiếp tục, âm lượng, nhiệt độ, quạt gió, đèn trần, ti vi, máy giặt, điều hòa, tủ lạnh*.
  - Sentences: *xin chào bạn, tạm biệt nhé, cảm ơn nhiều, làm ơn giúp tôi, bây giờ là mấy giờ, thời tiết thế nào, kiểm tra email, gọi điện cho mẹ, dẫn đường về nhà, khóa cửa chính, chụp một bức ảnh*.

---

## 4. Audio Quality and Normalization Integrity

- **Sampling Rate:** 16,000 Hz, 16-bit mono PCM.
- **Duration:** Exactly 16,000 samples (1.000 second).
- **Speech Centering:** Energy-based voice activity centering ensures active speech is centered in the 1.0s window with $\ge 150\,\text{ms}$ padding.
- **Normalization:** Peak normalized to $-1.0\,\text{dBFS}$.
- **Bit Integrity:** Every audio file was assigned a unique SHA-256 hash stored in `data/manifest.csv`.
