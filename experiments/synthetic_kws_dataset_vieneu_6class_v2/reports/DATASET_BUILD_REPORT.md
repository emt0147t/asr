# Six-Class Vietnamese KWS Dataset v2 Build Report

**Dataset Path:** `experiments/synthetic_kws_dataset_vieneu_6class_v2/`  
**Engine:** VieNeu-TTS v3 Turbo (Local ONNX CPU Runtime)  
**Total Utterances:** 2,000  
**Status:** **FROZEN & VERIFIED**  

---

## 1. Class Composition

| Class Label | Class ID | Utterance Count | Train (68%) | Val (16%) | Test (16%) | Texts Inventory |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `TRAI` | 0 | 250 | 170 | 40 | 40 | 'Trái' (/ʈaːj˧˥/) |
| `PHAI` | 1 | 250 | 170 | 40 | 40 | 'Phải' (/faːj˧˩˨/) |
| `LEN` | 2 | 250 | 170 | 40 | 40 | 'Lên' (/len˧˧/) |
| `XUONG` | 3 | 250 | 170 | 40 | 40 | 'Xuống' (/suəŋ˧˥/) |
| `UNKNOWN` | 4 | 500 | 340 | 80 | 80 | 120 distinct disjoint non-target texts |
| `SILENCE` | 5 | 500 | 340 | 80 | 80 | 4 background noise families |
| **Total** | -- | **2,000** | **1,360** | **320** | **320** | **Balanced 6-Class Architecture** |

---

## 2. Voice Allocation & Disjointness

- **Train (17 voices):** `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`
- **Val (4 voices):** `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`
- **Test (4 voices):** `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`

Zero voice overlap exists across Train, Validation, and Test splits.

---

## 3. UNKNOWN Vocabulary Breadth

- **Train Split:** 60 unique non-target texts (monosyllables, disyllables, phrases)
- **Validation Split:** 30 unique non-target texts (disjoint from train and test)
- **Test Split:** 30 unique non-target texts (disjoint from train and val)
- **Hard Negative Onset / Coda Coverage:** Extensive coverage of confusable consonants (`/tr/`, `/ph/`, `/l/`, `/x/`) and vowels.
