# Chương 00: Tóm tắt Điều hành (Executive Summary)

## 1. Bản chất Dự án & Mục tiêu Cốt lõi
Dự án Phát hiện Từ khóa Tiếng Việt (Vietnamese Keyword Spotting - KWS) được khởi xướng nhằm xây dựng một mô hình học sâu dạng Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) siêu nhẹ, chạy cục bộ (on-device edge AI) với độ trễ thấp để nhận diện 4 khẩu lệnh điều khiển robot/thiết bị bằng tiếng Việt:
1. `TRAI` (Trái - Left)
2. `PHAI` (Phải - Right)
3. `LEN` (Lên - Up)
4. `XUONG` (Xuống - Down)

Mọi âm thanh ngoài 4 khẩu lệnh này—bao gồm tạp âm môi trường, tiếng nói tự nhiên thông thường, các từ ngữ tiếng Việt phát âm gần giống (confusable words), hoặc các câu mệnh lệnh dài—đều phải bị từ chối dứt khoát vào lớp `REJECT` (hoặc phân loại vào `UNKNOWN` / `BACKGROUND_NOISE`).

## 2. Checkpoint Đông kết & Hiện trạng Đóng gói (Candidate Status)
Tại thời điểm hoàn tất giai đoạn [Phase 13.4](file:///d:/trainidentificationaudio/reports/PHASE13_4_FINAL_CNN_RELEASE.md), dự án đã chính thức đóng băng ứng viên phát hành cuối cùng:

> **Định danh Checkpoint:** `variant_v11_hard_negative_weighted_seed42.pt`  
> **Trạng thái chuẩn y:** `FROZEN_CNN_RELEASE_CANDIDATE`  
> **Đường dẫn Lưu trữ Chuẩn:** [`experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)  
> **Kích thước:** 479.524 bytes  
> **Mã băm SHA-256 (Byte-Exact):** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
> **Mô hình ONNX Native:** [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx) (SHA-256: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`)  
> **Ngưỡng Tối ưu (Authoritative Threshold):** $\theta_{\text{cmd}} = 0.20$

Mô hình đạt được tính tương đương toán học tuyệt đối giữa PyTorch và ONNX Runtime với độ lệch logit cực đại chỉ $\Delta_{\text{max}} = 3.58 \times 10^{-6}$ (thỏa mãn ngưỡng sai số khắt khe $< 10^{-4}$).

## 3. Hiệu năng Thẩm định Thực chứng Cốt lõi
Hiệu năng kỹ thuật của mô hình được thẩm định qua 2 mốc độc lập lớn:
1. **Kiểm thử Khoá Phase 11 (Phase 11 Locked Test - 480 clips):**
   - **Command Recall:** 99.17% (119/120 clips lệnh kích hoạt chính xác)
   - **False Alarm Rate (FAR):** 0.63% (3/480 clips kích hoạt nhầm)
   - **Accuracy Tổng:** 98.75%
2. **Đánh giá Xác nhận Độc lập Đăng ký trước Phase 13.3 (Preregistered Synthetic Confirmatory Evaluation - 1.000 clips):**
   - Thử nghiệm trên 8 giọng đọc tổng hợp Vieneu giọng Bắc hoàn toàn mới chưa từng xuất hiện trong tập huấn luyện/kiểm thử trước đó, với các câu chứa từ ngữ âm học gây nhiễu chưa từng thấy (unseen lexical carriers).
   - **Độ chính xác Tổng thể (Accuracy):** 99.20% (992/1000 clips) [Vượt ngưỡng yêu cầu $\ge 95.0\%$]
   - **Độ nhạy Lệnh (Command Recall):** 99.00% (198/200 clips) [Vượt ngưỡng yêu cầu $\ge 95.0\%$]
     - `TRAI`: 49/50 (98.0%)
     - `PHAI`: 50/50 (100.0%)
     - `LEN`: 49/50 (98.0%)
     - `XUONG`: 50/50 (100.0%)
   - **Tỷ lệ Báo động Giả (Non-Target FAR):** 0.75% (6/800 clips) [Thỏa mãn yêu cầu khắt khe $\le 2.0\%$]
   - **Tỷ lệ Phân loại Nhầm giữa các Lệnh (Confusion Rate):** 0.00% (0/200 clips)
   - **Độ trễ Suy luận (CPU Latency trên Intel i7/Windows):** Mean = 1.80 ms, P95 = 2.12 ms (Budget cho phép $\le 50.0$ ms).
   - **Kết luận:** Đạt 8/8 tiêu chí nghiệm thu tiên quyết (ALL GATES PASSED).

## 4. Các Giới hạn Vận hành Bắt buộc (Mandatory Operational Boundaries)
Nhà nghiên cứu và kỹ sư vận hành phải tuân thủ nghiêm ngặt 3 nguyên tắc pháp lý và kỹ thuật sau:
1. **Chưa thẩm định trên Giọng nói Tự nhiên (`NOT_VALIDATED_ON_NATURAL_SPEECH`):**
   Toàn bộ kết quả định lượng đạt được đều thực hiện trên âm thanh tổng hợp từ công cụ Vieneu TTS v3-Turbo kết hợp nhiễu nền ESC-50/DNS. Thử nghiệm nghe mù trên người ở Phase 90 chỉ có đúng **01 người đánh giá** hoàn thành (5 người khác chưa nộp phản hồi). Chưa có tập dữ liệu giọng người thật nói tự nhiên nào được thu âm có bản quyền/đồng thuận đầy đủ để kiểm định mô hình. Tuyệt đối không được quảng bá mô hình đã sẵn sàng cho giọng người thực tế.
2. **Chưa hỗ trợ Micro Mở Liên tục (`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`):**
   Mô hình được thiết kế theo cơ chế **đoạn phát âm đóng khung (Bounded-Utterance)**: nhận đầu vào là đoạn âm thanh đúng 1.0 giây ($16.000$ mẫu). Mô hình **chưa tích hợp** bộ phát hiện tiếng nói (VAD - Voice Activity Detector) hay cơ chế cửa sổ trượt (sliding-window streaming). Nếu đưa luồng âm thanh micro liên tục mà không có cơ chế cắt câu chuẩn xác, mô hình sẽ gặp hiện tượng kích hoạt lặp hoặc báo động giả do nhiễu mép cửa sổ.
3. **Phạm vi Giọng Địa phương Hạn hẹp (Accent Scope Limitation):**
   Mô hình chỉ mới được xác nhận độc lập trên các biến thể phát âm thuộc **giọng Miền Bắc (Northern Accent)**. Giọng Miền Trung (Central) và Miền Nam (Southern) hoàn toàn chưa được kiểm định độc lập do thiếu giọng tổng hợp độc lập và chưa có dữ liệu thu âm người thật có đồng thuận.
