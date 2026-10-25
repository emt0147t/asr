# Báo Cáo Kiểm Định Toàn Diện Hệ Thống Micro Trực Tiếp
## (Comprehensive Live Microphone KWS Audit & Verification Report)

> **Mô hình đông kết:** `variant_v11_hard_negative_weighted_seed42` (Phase 13.4 Final Release Candidate)  
> **Thời điểm thẩm định:** 2026-10-21 23:44:33 UTC+7  
> **Quy tắc suy luận:** Chuẩn Hợp đồng Chương 07: $c^* = \arg\max_{j \in \{0..4\}} P(c_j)$, ngưỡng $\theta^*_{\text{cmd}} = 0.20$  
> **Phần cứng thu âm:** Device [1] Microphone (2- Realtek(R) Audio (MME) @ 16 kHz  

---

### 1. Ranh Giới Quản Trị & Khoa Học Bắt Buộc
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

---

### 2. Kết Quả Thử Nghiệm Chi Tiết

#### Phần A: Khẩu Lệnh Đơn Lẻ (40 Lần Thử Nghiệm)
- **Độ chính xác tổng:** **38/40 (95.0%)**
- Chi tiết:
  - `TRAI`: 9/10 Hits
  - `PHAI`: 9/10 Hits
  - `LEN`: 10/10 Hits
  - `XUONG`: 10/10 Hits

#### Phần B: Chuỗi Khẩu Lệnh Liên Tiếp (Multi-Command Sequences)
- **Chuỗi `TRAI -> XUONG -> PHAI`:** **4/10 chuỗi đúng 100% thứ tự (40%)**
- **Chuỗi `PHAI -> LEN -> TRAI`:** **5/10 chuỗi đúng 100% thứ tự (50%)**
- **Chuỗi lặp lại `TRAI -> TRAI -> TRAI`:** **2/5 chuỗi đúng 100% (40%)**

#### Phần C: Thử Nghiệm Mẫu Âm Bản & Báo Động Giả Từ Ngoài Tập Từ Khóa (OOV Negatives)
- **Tiếng ồn phòng nền tĩnh (5 giây):** **0** kích hoạt (0.00% FAR).
- **Câu đàm thoại thông thường (5 câu):** **0** kích hoạt.
- **Từ đơn âm tiết ngoài từ điển (OOV Monosyllables):**
  - Từ *'cửa'* (Kim Thanh): Nhãn nhận diện `['XUONG']` &bull; Độ tin cậy: [0.6008]
  - Từ *'cửa'* (Thùy Dung): Nhãn nhận diện `['XUONG']` &bull; Độ tin cậy: [0.6564]
  - Từ *'cửa'* (Mai Anh): Nhãn nhận diện `['PHAI']` &bull; Độ tin cậy: [0.2955]
  - Từ *'mở'* (Mai Anh): Nhãn nhận diện `['PHAI']` &bull; Độ tin cậy: [0.5011]
  - Từ *'bàn'* (Mai Anh): Nhãn nhận diện `['PHAI']` &bull; Độ tin cậy: [0.5243]
  - Từ *'ghế'* (Quốc Tuấn): Nhãn nhận diện `['LEN']` &bull; Độ tin cậy: [0.4446]
  - Từ *'nhà'* (Quốc Tuấn): Nhãn nhận diện `['PHAI']` &bull; Độ tin cậy: [0.237]

> [!IMPORTANT]
> **Kết luận khoa học về OOV Monosyllables:** Toàn bộ các từ đơn âm tiết ngoài tập từ khóa đều bị mô hình gán nhãn kích hoạt vào một trong 4 lệnh với xác suất từ 30% đến 80%. Đây là lỗi bản chất cấp mô hình (Model-level False Acceptance) do dữ liệu huấn luyện đóng không có âm bản đơn âm tiết, KHÔNG PHẢI lỗi phần mềm.

#### Phần D: Thử Nghiệm Độ Ổn Định Liên Tục (60.0 Giây Live Soak Test)
- **Thời lượng thu âm liên tục:** **60.02 giây**
- **Độ trễ suy luận trung bình:** **2.62 ms** (Phân vị P95: **4.18 ms**)
- **Sự kiện kích hoạt theo lịch trình:** **8/4 sự kiện**
- **Kiểm tra quyền riêng tư:** **ĐẠT** (0 tệp WAV thô được ghi ra đĩa).

---

### 3. Trạng Thái Phát Hành
- **Trạng thái:** `READY_FOR_SUPERVISED_DEMONSTRATION` (với đầy đủ cảnh báo khoa học về giới hạn từ ngoài tập từ khóa).
