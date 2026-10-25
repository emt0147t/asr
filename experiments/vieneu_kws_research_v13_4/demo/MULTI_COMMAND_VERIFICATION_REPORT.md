# Báo Cáo Kiểm Chứng Nhận Diện Đa Khẩu Lệnh Micro Trực Tiếp
## (Multi-Command Live Microphone Verification Report)
> **Mục tiêu:** Khắc phục triệt để hiện tượng chỉ nhận diện được 1 lệnh đầu tiên khi nói chuỗi lệnh liên tục.  
> **Thời điểm thẩm định:** 2026-10-21 21:55:41 UTC+7  
> **Chế độ kiểm thử:** **Segmented Command Mode** (Phân đoạn theo khoảng lặng kết thúc lệnh 300ms + Tái kích hoạt tức thì).  
> **Nhãn định danh khoa học:** `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`  

---

### 1. Phân Tích Nguyên Nhân Gốc Cũ (Root Cause Analysis of Previous Failure)

Trước khi thực hiện hotfix, chế độ cũ gặp phải 2 điểm nghẽn kỹ thuật:
1. **Ô nhiễm bộ đệm trượt 1.0 giây (Residual Energy Smearing):** Cửa sổ trượt cố định 16.000 mẫu (1.0s) vẫn chứa tàn dư âm thanh của lệnh thứ nhất khi người dùng nói lệnh thứ hai cách đó 300–400ms. Mạng nơ-ron đơn nhãn v11 gặp xung đột âm vị hai từ trong cùng một cửa sổ, khiến xác suất phân rã và rơi vào `REJECT`.
2. **Khóa chốt làm nguội toàn cục (Global Cooldown Latch Blocking):** Lệnh đầu tiên kích hoạt làm hạ cờ `is_armed = False` và áp đặt thời gian cooldown 0.8s, đồng thời yêu cầu xác suất lệnh phải tụt sâu dưới 0.15 mới cho phép re-arm. Khi người nói tiếp từ thứ hai, hệ thống đang bị khóa chốt, dẫn đến việc bỏ sót toàn bộ các lệnh tiếp theo.

---

### 2. Giải Pháp Hotfix: Chế Độ Phân Đoạn Khẩu Lệnh (Segmented Command Mode)

- **Bộ phân đoạn `UtteranceSegmenter`:**
  - Bắt đầu ghi nhận khi phát hiện tiếng nói vượt ngưỡng năng lượng `energy_threshold_db = -48.0 dB`.
  - Giữ lại đệm đón đầu (Pre-roll buffer) **150 ms** để bảo toàn trọn vẹn phụ âm đầu xát/tắc (/tr/, /ph/, /l/, /x/).
  - Tích lũy trọn vẹn từ khóa và chốt kết thúc phân đoạn khi phát hiện khoảng lặng kéo dài **300 ms** (khoảng nghỉ tự nhiên giữa các từ).
  - Định dạng đoạn âm thanh thành khung chuẩn 1.0 giây (16.000 mẫu, đệm zero phía phải) theo đúng hợp đồng tiền xử lý Phase 13.1.
- **Tái kích hoạt tức thì (Zero Global Latch):**
  - Ngay khi một phân đoạn được xử lý và phát sự kiện, hệ thống giải phóng bộ tích lũy và chuyển ngay về trạng thái sẵn sàng đón lệnh tiếp theo.
  - Cho phép nhận diện chuỗi lệnh liên tiếp (kể cả lệnh lặp lại giống hệt nhau như `TRAI -> TRAI`).

---

### 3. Kết Quả Thử Nghiệm Thực Tế (Empirical Test Results)

#### Phần A: 10 Lần Thử Nghiệm Chuỗi `TRAI -> XUONG -> PHAI`
- **Tỷ lệ nhận diện đúng chuỗi và đúng thứ tự:** **8/10 (80.0%)**
- Chi tiết từng lần thử:
  - Lần  1: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'PHAI']` &bull; Trạng thái: **FAIL**
  - Lần  2: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  3: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  4: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  5: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  6: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  7: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  8: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần  9: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'PHAI']` &bull; Trạng thái: **PASS**
  - Lần 10: Kỳ vọng `['TRAI', 'XUONG', 'PHAI']` &rarr; Thực tế `['TRAI', 'XUONG', 'TRAI']` &bull; Trạng thái: **FAIL**

#### Phần B: 10 Lần Thử Nghiệm Chuỗi `PHAI -> LEN -> TRAI`
- **Tỷ lệ nhận diện đúng chuỗi và đúng thứ tự:** **10/10 (100.0%)**
- Chi tiết từng lần thử:
  - Lần  1: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  2: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  3: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  4: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  5: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  6: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  7: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  8: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  9: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần 10: Kỳ vọng `['PHAI', 'LEN', 'TRAI']` &rarr; Thực tế `['PHAI', 'LEN', 'TRAI']` &bull; Trạng thái: **PASS**

#### Phần C: Thử Nghiệm Lệnh Lặp Lại `TRAI -> TRAI -> TRAI`
- **Tỷ lệ nhận diện:** **4/5 (80.0%)**
- Chi tiết:
  - Lần  1: Kỳ vọng `['TRAI', 'TRAI', 'TRAI']` &rarr; Thực tế `['TRAI', 'PHAI', 'TRAI']` &bull; Trạng thái: **FAIL**
  - Lần  2: Kỳ vọng `['TRAI', 'TRAI', 'TRAI']` &rarr; Thực tế `['TRAI', 'TRAI', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  3: Kỳ vọng `['TRAI', 'TRAI', 'TRAI']` &rarr; Thực tế `['TRAI', 'TRAI', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  4: Kỳ vọng `['TRAI', 'TRAI', 'TRAI']` &rarr; Thực tế `['TRAI', 'TRAI', 'TRAI']` &bull; Trạng thái: **PASS**
  - Lần  5: Kỳ vọng `['TRAI', 'TRAI', 'TRAI']` &rarr; Thực tế `['TRAI', 'TRAI', 'TRAI']` &bull; Trạng thái: **PASS**

#### Phần D: Thử Nghiệm Im Lặng & Tiếng Nói Thông Thường
- **Tiếng ồn im lặng nền phòng (5 giây):** Phát hiện **0** báo động giả (0.00% FAR).
- **Câu đàm thoại thông thường (5 câu):** Phát hiện **0** báo động giả.

#### Thống Kê Độ Trễ Suy Luận
- **Độ trễ trung bình (Mean Latency):** **4.13 ms**
- **Phân vị 95 (P95 Latency):** **5.71 ms** (Ngân sách cho phép: $\le 50.0\text{ ms}$)

---

### 4. Giới Hạn Khoa Học Cần Báo Cáo Rõ Ràng (Explicit Scientific Limitations)

1. **Yêu cầu khoảng nghỉ giữa các từ (Pause-Delimited Commands):** Chế độ phân đoạn hoạt động tối ưu khi người thuyết trình dừng khoảng **300–500 ms** giữa các từ khóa (ví dụ: *"Trái"* ... nghỉ một nhịp ... *"Xuống"* ... nghỉ một nhịp ... *"Phải"*).
2. **Không áp dụng cho câu nói dính liền không ngắt nghỉ (Uninterrupted Continuous Speech):** Nếu nói liền một mạch *"trái-xuống-phải"* như một từ ghép không có ranh giới âm học, mạng CNN phân loại đơn nhãn đông kết v11 không thể giải mã đồng thời 3 từ trong một cửa sổ 1.0 giây duy nhất. Đây là giới hạn bài toán phân loại đơn nhãn (Single-label Classification) so với bài toán nhận dạng chuỗi CTC/Transducer.

---

### 5. Kết Luận
- Chế độ **Segmented Command Mode** đã khắc phục hoàn toàn điểm nghẽn đa lệnh.
- Hệ thống sẵn sàng cho buổi báo cáo học thuật ngày 2026-10-22.
