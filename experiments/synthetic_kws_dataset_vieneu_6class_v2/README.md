# Synthetic Vietnamese KWS Dataset v2 (Six-Class Architecture)

**Dataset Directory:** `experiments/synthetic_kws_dataset_vieneu_6class_v2/`  
**Engine:** VieNeu-TTS v3 Turbo (Local CPU/ONNX Runtime)  
**Total Scale:** 2,000 utterances (16 kHz mono, 1.0 second window)  
**Governance Status:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Class Composition & Split Balance (68% / 16% / 16%)

| Class Label | Class ID | Category | Total | Train (68%) | Validation (16%) | Test (16%) | Vocabulary / Acoustic Basis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `TRAI` | 0 | Target | 250 | 170 | 40 | 40 | Vietnamese command `Trái` (/ʈaːj˧˥/) |
| `PHAI` | 1 | Target | 250 | 170 | 40 | 40 | Vietnamese command `Phải` (/faːj˧˩˨/) |
| `LEN` | 2 | Target | 250 | 170 | 40 | 40 | Vietnamese command `Lên` (/len˧˧/) |
| `XUONG` | 3 | Target | 250 | 170 | 40 | 40 | Vietnamese command `Xuống` (/suəŋ˧˥/) |
| `UNKNOWN` | 4 | Distractor | 500 | 340 | 80 | 80 | 120 distinct non-target texts (disjoint splits) |
| `SILENCE` | 5 | Background | 500 | 340 | 80 | 80 | 4 background families (disjoint realizations) |
| **Total** | -- | -- | **2,000** | **1,360** | **320** | **320** | **Strict Voice-Disjoint Allocation** |

---

## 2. Voice Allocation Roster (25 Preset Voices)

- **Train (17 voices):**
  `Adam`, `Adam bựa`, `Hải Đăng`, `Minh Triết`, `Minh Đức`, `Mỹ Duyên`, `Ngọc Huyền`, `Ngọc Linh`, `Quang Sơn`, `Thanh Bình`, `Thiền Tâm Đức`, `Thiện Minh`, `Thái Sơn`, `Thục Đoan`, `Trúc Ly`, `Xuân Vĩnh`, `Đoan Trang`
- **Validation (4 voices):**
  `Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`
- **Test (4 voices):**
  `Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`

Zero speaker overlap exists across Train, Validation, and Test splits.

---

## 3. UNKNOWN Vocabulary Breadth (120 Distinct Texts)

- **Train Split (60 unique texts):**
  `Dừng`, `Bật`, `Tắt`, `Mở`, `Đóng`, `Chạy`, `Vào`, `Ra`, `Trời`, `Phát`, `Trại`, `Trạm`, `Tròn`, `Trước`, `Trắng`, `Phòng`, `Phút`, `Phần`, `Phím`, `Phố`, `Làm`, `Lấy`, `Lúc`, `Luôn`, `Lại`, `Lời`, `Xuồng`, `Xem`, `Xong`, `Xanh`, `Xa`, `Xuất`, `Gặp`, `Hẹn`, `Nhớ`, `Quên`, `Biết`, `Hiểu`, `Đọc`, `Viết`, `Hát`, `Đàn`, `Mua`, `Bán`, `Ăn`, `Uống`, `Ngủ`, `Thức`, `Xin chào`, `Cảm ơn`, `Tạm biệt`, `Đồng ý`, `Từ chối`, `Bắt đầu`, `Kết thúc`, `Tiếp tục`, `Trợ giúp`, `Cài đặt`, `Thông báo`, `Hệ thống`.
- **Validation Split (30 unique texts, disjoint):**
  `Đi`, `Đứng`, `Nghe`, `Nói`, `Hỏi`, `Tìm`, `Chọn`, `Xoay`, `Lùi`, `Quay`, `Trông`, `Trẻ`, `Trường`, `Phim`, `Phục`, `Phấn`, `Lược`, `Lớp`, `Lửa`, `Xốp`, `Xúc`, `Xinh`, `Đếm`, `Cân`, `Đo`, `Xem lại`, `Làm lại`, `Quay lại`, `Bỏ qua`, `Xác nhận`.
- **Test Split (30 unique texts, disjoint):**
  `Chào`, `Cứu`, `Hủy`, `Lưu`, `Xóa`, `Phóng`, `Thu`, `Khóa`, `Được`, `Không`, `Trưa`, `Trà`, `Trách`, `Phở`, `Pháo`, `Phép`, `Lá`, `Làng`, `Lạc`, `Xôi`, `Xương`, `Xích`, `Nhập`, `Xuất bản`, `Gửi đi`, `Nhận tin`, `Tìm kiếm`, `Chia sẻ`, `Tải về`, `Đăng nhập`.

---

## 4. SILENCE Acoustic Families

1. **`near_silence`:** Micro-dither floor (RMS $\approx -60$ dB, highpass filtered).
2. **`hvac_hum`:** 50 Hz / 60 Hz hum + harmonics (100/120 Hz, 150/180 Hz) + low-pass ventilation rumble.
3. **`office_ambient`:** Diffuse room acoustics (bandpass filtered ambient room noise).
4. **`airflow_pink`:** Pink noise (1/f spectral tilt) simulating continuous airflow.

Every silence clip is an independent, non-zero realization with a unique SHA-256 hash. Digital zero waveforms are excluded.
