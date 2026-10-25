# Hướng Dẫn Dành Cho Người Đánh Giá Nghe Giấu Tên (Blinded Reviewer Instructions)

**Loại đánh giá:** Đánh giá cảm thụ âm thanh người nghe hai chiều ẩn danh (Double-Blind Human Perceptual Listening Test)  
**Số lượng mẫu âm thanh:** 120 đoạn âm thanh (hoặc 20 đoạn nếu nhận gói phân phối chia nhỏ, mỗi đoạn 1.0 giây, 16 kHz mono)  
**Trạng thái quản trị:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Mục Đích Nghiên Cứu & Cơ Chế Giấu Tên (Blinding)

Bạn đang tham gia đánh giá độ rõ ràng, độ tự nhiên và độ chuẩn xác phát âm của các đoạn âm thanh tiếng Việt ngắn (1.0 giây) được tổng hợp tự động.  
Để loại bỏ hoàn toàn định kiến xác nhận (confirmation bias), thử nghiệm này được thiết kế **hoàn toàn giấu tên hai chiều (strictly double-blind)**:
- Bạn chỉ nhận được mã số thứ tự ngẫu nhiên của đoạn âm thanh (ví dụ: `review_clip_001` đến `review_clip_120`, hoặc `clip_01` đến `clip_20`).
- Bạn **KHÔNG** biết từ mục tiêu dự kiến ban đầu là gì, không biết danh mục nào, và không biết máy tính hay mô hình AI nhận diện ra từ nào.
- Vui lòng đánh giá khách quan dựa hoàn toàn trên những gì tai bạn nghe được thực tế.

---

## 2. Quy Trình Thực Hiện Đánh Giá

1. **Chuẩn bị môi trường nghe:**  
   - Đeo tai nghe có dây hoặc không dây (over-ear hoặc in-ear) trong phòng yên tĩnh.  
   - Bạn có thể nghe lại mỗi đoạn âm thanh nhiều lần tùy thích trước khi chấm điểm.
2. **Phương thức nhập dữ liệu:**  
   - Đánh giá trực tiếp qua giao diện web nội bộ tại `http://127.0.0.1:8080/` HOẶC điền vào file bảng tính `response_template.csv` (sử dụng Microsoft Excel, LibreOffice Calc hoặc trình chỉnh sửa văn bản).
3. **Quy tắc nhập liệu:**  
   - Đối với mỗi đoạn âm thanh, bạn phải hoàn thành đầy đủ **5 tiêu chí [BẮT BUỘC]** và có thể điền thêm **1 tiêu chí [TÙY CHỌN]**.

---

## 3. Thang Điểm & Tiêu Chí Đánh Giá Chi Tiết

> **LƯU Ý QUAN TRỌNG VỀ ĐỊNH DẠNG:**  
> Giữ nguyên chính xác tên các cột tiếng Anh trong file CSV. Các giá trị đặc biệt trong dấu ngoặc vuông hoặc mã ký tự tiếng Anh phải được nhập **chính xác từng chữ** như hướng dẫn dưới đây để hệ thống tự động kiểm tra tính hợp lệ.

---

### Tiêu Chí 1: Từ/Âm tiết nghe được — Cột: `perceived_text` **[BẮT BUỘC ĐÁNH GIÁ]**
- **Yêu cầu:** Gõ chính xác từ hoặc các âm tiết tiếng Việt có đầy đủ dấu thanh mà bạn nghe thấy (Ví dụ: *trái*, *phải*, *lên*, *xuống*, *trại*, *bánh rán*, *xưởng*, *tài*...).
- **Giữ nguyên mã tiếng Anh trong các trường hợp đặc biệt sau:**
  - Nếu đoạn âm thanh hoàn toàn yên tĩnh hoặc chỉ có tiếng ồn phòng/tiếng rè nền không có tiếng người: Nhập chính xác mã `[SILENCE]`
  - Nếu nghe thấy có tiếng nói nhưng bị méo mó, biến dạng đến mức không thể nhận diện được từ nào: Nhập chính xác mã `[UNINTELLIGIBLE]`

---

### Tiêu Chí 2: Độ rõ ràng/Dễ hiểu (Intelligibility) — Cột: `intelligibility_1to5` **[BẮT BUỘC ĐÁNH GIÁ]**
- **Thang điểm:** Chọn 1 số nguyên từ `1` đến `5`:
  - `5` = Rất rõ ràng, nghe hiểu ngay lập tức mà không cần gắng sức.
  - `4` = Khá rõ ràng, nghe hiểu dễ dàng, chỉ có tạp âm rất nhỏ.
  - `3` = Tương đối rõ, nhận ra được từ nhưng âm thanh có biến dạng nhận thấy rõ.
  - `2` = Khó nghe, âm thanh bị nghẹt/méo nhiều, phần lớn phải đoán.
  - `1` = Hoàn toàn không thể hiểu được, không nhận diện được âm tiết nào.

---

### Tiêu Chí 3: Độ tự nhiên (Naturalness) — Cột: `naturalness_1to5` **[BẮT BUỘC ĐÁNH GIÁ]**
- **Thang điểm:** Chọn 1 số nguyên từ `1` đến `5`:
  - `5` = Hoàn toàn tự nhiên như giọng người thật nói.
  - `4` = Tự nhiên, chỉ có cảm giác nhân tạo rất nhẹ.
  - `3` = Giọng máy chấp nhận được, ngữ điệu nhận ra được nhưng rõ chất tổng hợp.
  - `2` = Giọng máy cứng nhắc/robot, cao độ hoặc tiết tấu bất tự nhiên.
  - `1` = Rất rè, méo mó cơ học, âm thanh điện tử khó chịu.

---

### Tiêu Chí 4: Tính chuẩn xác phát âm & Thanh điệu — Cột: `pronunciation_correct_yn` **[BẮT BUỘC ĐÁNH GIÁ]**
- **Giá trị hợp lệ:** Nhập chính xác một trong 3 mã ký tự tiếng Anh sau:
  - `Y` (Yes) = Phát âm đúng ngữ âm tiếng Việt, thanh điệu (sắc, huyền, hỏi, ngã, nặng, không dấu) chuẩn xác, không bị nuốt âm đuôi.
  - `N` (No) = Phát âm sai thanh điệu, sai nguyên âm/phụ âm, hoặc bị cắt cụt đuôi âm tiết.
  - `Unsure` = Không chắc chắn, phát âm mơ hồ giữa các thanh điệu hoặc không nghe rõ để kết luận.

---

### Tiêu Chí 5: Độ tự tin của người nghe — Cột: `confidence_1to5` **[BẮT BUỘC ĐÁNH GIÁ]**
- **Thang điểm:** Chọn 1 số nguyên từ `1` đến `5`:
  - `5` = Tuyệt đối chắc chắn về những gì mình nghe được.
  - `4` = Khá tự tin.
  - `3` = Tự tin ở mức trung bình.
  - `2` = Ít tự tin, có sự phân vân.
  - `1` = Đoán mò hoàn toàn.

---

### Tiêu Chí 6: Ghi chú quan sát — Cột: `comments` **[TÙY CHỌN - KHÔNG BẮT BUỘC]**
- Bạn có thể ghi chú ngắn gọn các hiện tượng âm thanh đặc biệt nếu nhận thấy (Ví dụ: *"tiếng kim loại rè"*, *"nuốt âm ng cuối"*, *"lên giọng bất thường"*, *"tiếng quạt gió nền"*...).
- Nếu không có nhận xét thêm, có thể để trống.

---

## 4. Tóm Tắt Quy Định Nhập Liệu Trong File CSV

| Tên Cột Trong File (Giữ nguyên tiếng Anh) | Mức Độ | Định Dạng / Giá Trị Hợp Lệ | Ví Dụ |
| :--- | :---: | :--- | :--- |
| `review_clip_id` | **[BẮT BUỘC]** | Giữ nguyên mã số có sẵn của file | `clip_01` hoặc `review_clip_001` |
| `perceived_text` | **[BẮT BUỘC]** | Chữ tiếng Việt có dấu, hoặc `[SILENCE]`, `[UNINTELLIGIBLE]` | `trái`, `lên`, `[SILENCE]` |
| `intelligibility_1to5` | **[BẮT BUỘC]** | Số nguyên từ `1` đến `5` | `5` |
| `naturalness_1to5` | **[BẮT BUỘC]** | Số nguyên từ `1` đến `5` | `4` |
| `pronunciation_correct_yn` | **[BẮT BUỘC]** | Đúng một trong 3 mã: `Y`, `N`, `Unsure` | `Y` |
| `confidence_1to5` | **[BẮT BUỘC]** | Số nguyên từ `1` đến `5` | `5` |
| `comments` | **[TÙY CHỌN]** | Văn bản tự do hoặc để trống | `âm ng bị nuốt nhẹ` |

---

## 5. Lưu & Gửi Kết Quả

1. Sau khi hoàn thành tất cả các dòng, hãy lưu file với bảng mã **CSV UTF-8** để bảo toàn đầy đủ dấu tiếng Việt.
2. Kiểm tra để đảm bảo **không còn ô trống** ở 5 cột bắt buộc nào.
3. Đổi tên file kết quả theo cú pháp: `<reviewer_id>_responses.csv` (Ví dụ: `reviewer_01_responses.csv`) và gửi lại cho ban tổ chức theo đường dẫn lưu trữ bảo mật đã được cung cấp.
