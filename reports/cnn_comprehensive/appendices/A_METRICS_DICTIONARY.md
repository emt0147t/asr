# Phụ Lục A: Từ Điển Thuật Ngữ & Chỉ Số Đo Lường (Metrics Dictionary)

Tài liệu này định nghĩa chi tiết các chỉ số toán học và thuật ngữ kỹ thuật được sử dụng xuyên suốt toàn bộ dự án KWS tiếng Việt.

---

## 1. Các Chỉ Số Đo Lường Hiệu Năng Mô Hình KWS

### 1.1. Độ Nhạy Nhận Lệnh (Command Recall / Sensitivity)
Tỷ lệ nhận diện chính xác các khẩu lệnh hợp lệ trên tổng số mẫu khẩu lệnh thực tế:
$$\text{Recall}_{\text{cmd}} = \frac{\sum_{c \in \{0, 1, 2, 3\}} \text{TP}_c}{\sum_{c \in \{0, 1, 2, 3\}} (\text{TP}_c + \text{FN}_c)}$$
Trong đó:
- $\text{TP}_c$ (True Positive): Số mẫu thuộc lệnh $c$ được mô hình nhận diện đúng lớp $c$.
- $\text{FN}_c$ (False Negative): Số mẫu thuộc lệnh $c$ bị mô hình bỏ sót (phân loại nhầm thành lệnh khác hoặc bị đẩy vào `REJECT`).

### 1.2. Tỷ Lệ Báo Động Giả (False Alarm Rate - FAR)
Tỷ lệ các mẫu âm thanh ngoại lai (không phải khẩu lệnh) bị mô hình nhận diện nhầm thành một trong 4 lệnh:
$$\text{FAR} = \frac{\sum_{c \in \{0, 1, 2, 3\}} \text{FP}_c}{N_{\text{non-target}}}$$
Trong đó:
- $\text{FP}_c$ (False Positive): Số mẫu ngoại lai bị gán nhầm thành lệnh $c$.
- $N_{\text{non-target}}$: Tổng số mẫu âm thanh không phải lệnh (bao gồm tiếng ồn, từ gây nhiễu, câu hội thoại, v.v.).

### 1.3. Độ Chính Xác Toàn Diện (Overall Accuracy)
Tỷ lệ các dự đoán hoàn toàn chính xác trên toàn bộ tập kiểm thử (bao gồm cả các lệnh hợp lệ và các mẫu từ chối hợp lệ):
$$\text{Accuracy} = \frac{\sum_{i=0}^4 \text{TP}_i}{N_{\text{total}}}$$
Trong đó lớp thứ 4 là lớp từ chối `REJECT`.

### 1.4. Điểm F1-Score Cho Từng Lớp Lệnh
Trung bình điều hòa giữa Độ chuẩn xác (Precision) và Độ nhạy (Recall):
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
$$F_{1, c} = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 1.5. Tỷ Lệ Lẫn Lộn Giữa Các Lệnh (Command Confusion Rate)
Tỷ lệ một khẩu lệnh hợp lệ này bị nhận diện nhầm sang một khẩu lệnh hợp lệ khác:
$$\text{Confusion Rate} = \frac{\sum_{c_i \ne c_j, c_i, c_j \in \{0,1,2,3\}} \text{Count}(y=c_i, \hat{y}=c_j)}{N_{\text{target}}}$$
Trong Phase 13.3, tỷ lệ này đạt mức tuyệt đối: **$0.00\%$** (0/200 clips).

---

## 2. Các Chỉ Số Đo Lường Triển Khai & Đẳng Cấu

### 2.1. Độ Lệch Logit Cực Đại (Max Absolute Logit Difference - $\Delta_{\text{max}}$)
Đo lường sai số số học giữa đầu ra của mô hình PyTorch và mô hình ONNX Runtime khi cùng nhận một ma trận đầu vào $X$:
$$\Delta_{\text{max}} = \max_{i=1 \dots N, j=0 \dots 4} |z_{i, j}^{\text{PyTorch}} - z_{i, j}^{\text{ONNX}}|$$
Yêu cầu nghiệm thu kỹ thuật: $\Delta_{\text{max}} < 1.0 \times 10^{-4}$. Checkpoint đông kết đạt $\Delta_{\text{max}} = 3.58 \times 10^{-6}$.

### 2.2. Phân Vị Độ Trễ (Latency Percentiles: Mean, P50, P95)
- **Mean Latency:** Thời gian suy luận trung bình tính trên toàn bộ các mẫu kiểm thử.
- **P50 Latency (Median):** Thời gian mà $50\%$ số mẫu suy luận hoàn thành nhanh hơn hoặc bằng giá trị này.
- **P95 Latency:** Thời gian mà $95\%$ số mẫu suy luận hoàn thành nhanh hơn hoặc bằng giá trị này (đại diện cho độ trễ trong tình huống tải xấu nhất).

---

## 3. Các Nhãn Trạng Thái Chuẩn Hóa (Canonical Status Labels)
- `FROZEN_CNN_RELEASE_CANDIDATE`: Ứng viên mô hình CNN đã được đóng băng toàn bộ trọng số, cấu hình và mã suy luận; không được phép can thiệp chỉnh sửa.
- `VALIDATED_HIGH_CONFIDENCE`: Đã được chứng minh bằng thực nghiệm độc lập với độ tin cậy thống kê cao.
- `NOT_VALIDATED_ON_NATURAL_SPEECH`: Chưa được thẩm định trên dữ liệu giọng người thật nói tự nhiên trong môi trường đời thực.
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`: Chưa có hệ thống hoàn chỉnh sẵn sàng triển khai cho chế độ micro mở thời gian thực.
- `NOT_VALIDATED_PENDING_DATA`: Chưa được thẩm định do thiếu dữ liệu kiểm thử độc lập.
- `INVALIDATED_INCOMPLETE_STUDY`: Nghiên cứu bị hủy giá trị thống kê do dữ liệu thu thập không đầy đủ so với kế hoạch ban đầu.
