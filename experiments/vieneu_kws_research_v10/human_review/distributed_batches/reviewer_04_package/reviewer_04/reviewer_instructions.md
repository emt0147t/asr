# Hướng Dẫn Dành Cho Người Đánh Giá Nghe Giấu Tên (Blinded Reviewer Instructions)
## Gói đánh giá: reviewer_04

**Loại đánh giá:** Đánh giá cảm thụ âm thanh người nghe hai chiều ẩn danh (Double-Blind Human Perceptual Listening Test)  
**Số lượng mẫu âm thanh:** 20 đoạn (`clip_01.wav` đến `clip_20.wav`, 1.0 giây mỗi đoạn, 16 kHz mono)  
**Trạng thái quản trị:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`

---

## 1. Mục Đích Nghiên Cứu & Cơ Chế Giấu Tên (Blinding)
Bạn đang đánh giá độ rõ ràng, độ tự nhiên và độ chuẩn xác phát âm của các đoạn âm thanh tiếng Việt ngắn (1.0 giây) được tổng hợp tự động.
Để loại bỏ hoàn toàn định kiến xác nhận (confirmation bias), thử nghiệm này được thiết kế **hoàn toàn giấu tên hai chiều (strictly double-blind)**:
- Bạn chỉ nhận được mã số thứ tự ngẫu nhiên của đoạn âm thanh (`clip_01` đến `clip_20`).
- Bạn **KHÔNG** biết từ mục tiêu dự kiến ban đầu là gì, không biết danh mục nào, và không biết máy tính hay mô hình AI nhận diện ra từ nào.
- Vui lòng đánh giá khách quan dựa hoàn toàn trên những gì tai bạn nghe được thực tế.

---

## 2. Quy Trình Thực Hiện Đánh Giá
1. Giải nén gói zip này trên máy tính cá nhân của bạn.
2. Đeo tai nghe trong phòng yên tĩnh, nghe từng file âm thanh trong thư mục `audio/` (từ `clip_01.wav` đến `clip_20.wav`). Bạn có thể nghe lại mỗi đoạn nhiều lần tùy thích.
3. Mở file `response_template.csv` bằng Microsoft Excel, LibreOffice Calc hoặc trình chỉnh sửa văn bản.
4. Với mỗi đoạn âm thanh, bạn phải hoàn thành đầy đủ **5 tiêu chí [BẮT BUỘC]** và có thể điền thêm **1 tiêu chí [TÙY CHỌN]**:

### Tiêu Chí Đánh Giá Chi Tiết:
1. **Từ/Âm tiết nghe được (`perceived_text`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - Gõ chính xác từ hoặc các âm tiết tiếng Việt có đầy đủ dấu thanh mà bạn nghe thấy (Ví dụ: *trái*, *phải*, *lên*, *xuống*, *trại*, *bánh rán*).
   - **Giữ nguyên mã tiếng Anh nếu gặp:**
     - Nếu đoạn âm thanh chỉ có tiếng ồn phòng/rè nền hoặc hoàn toàn yên lặng: Nhập chính xác `[SILENCE]`
     - Nếu nghe thấy có tiếng nói nhưng bị méo mó, biến dạng không nhận ra từ nào: Nhập chính xác `[UNINTELLIGIBLE]`
2. **Độ rõ ràng/Dễ hiểu (`intelligibility_1to5`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - `5` = Rất rõ ràng; nghe hiểu ngay lập tức không cần gắng sức.
   - `4` = Khá rõ ràng; nghe hiểu dễ dàng, tạp âm rất nhỏ.
   - `3` = Tương đối rõ; hiểu được từ nhưng có biến dạng nhận thấy rõ.
   - `2` = Khó nghe; âm thanh nghẹt/méo nhiều, phần lớn phải đoán.
   - `1` = Hoàn toàn không thể hiểu được; không nhận diện được âm tiết nào.
3. **Độ tự nhiên (`naturalness_1to5`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - `5` = Hoàn toàn tự nhiên như giọng người thật nói.
   - `4` = Tự nhiên, chỉ có cảm giác nhân tạo rất nhẹ.
   - `3` = Giọng máy chấp nhận được, ngữ điệu nhận ra được.
   - `2` = Giọng máy cứng nhắc/robot, cao độ hoặc tiết tấu bất thường.
   - `1` = Rất rè, méo mó cơ học, âm thanh điện tử khó chịu.
4. **Tính chuẩn xác phát âm & Thanh điệu (`pronunciation_correct_yn`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - Nhập chính xác một trong 3 mã tiếng Anh:
     - `Y` (Yes) = Phát âm đúng ngữ âm tiếng Việt, thanh điệu chuẩn xác, không nuốt âm đuôi.
     - `N` (No) = Phát âm sai thanh điệu, sai nguyên âm/phụ âm, hoặc bị cắt cụt đuôi âm tiết.
     - `Unsure` = Không chắc chắn, phát âm mơ hồ giữa các thanh điệu.
5. **Độ tự tin của người nghe (`confidence_1to5`) [BẮT BUỘC ĐÁNH GIÁ]:**
   - `5` = Tuyệt đối chắc chắn về những gì mình nghe được.
   - `4` = Khá tự tin.
   - `3` = Tự tin ở mức trung bình.
   - `2` = Ít tự tin, có sự phân vân.
   - `1` = Đoán mò hoàn toàn.
6. **Ghi chú quan sát (`comments`) [TÙY CHỌN - KHÔNG BẮT BUỘC]:**
   - Ghi chú các hiện tượng âm thanh đặc biệt (Ví dụ: "tiếng kim loại rè", "nuốt âm ng cuối", "tiếng quạt gió"). Có thể để trống nếu không có nhận xét.

---

## 3. Lưu & Gửi Kết Quả
1. Lưu file kết quả của bạn với tên `reviewer_04_responses.csv` (định dạng CSV UTF-8 để giữ đúng dấu tiếng Việt).
2. Kiểm tra đảm bảo cả 20 dòng đều có đủ dữ liệu ở các cột bắt buộc.
3. Gửi lại file `reviewer_04_responses.csv` cho ban tổ chức.
