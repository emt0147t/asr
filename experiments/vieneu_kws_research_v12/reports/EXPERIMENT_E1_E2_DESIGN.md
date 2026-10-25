# Thiết Kế Thử Nghiệm Mô Hình KWS v12: E1 & E2
## (Vietnamese KWS CNN v12 Retraining Protocol: E1 Baseline Scale & E2 Scaled Capacity)

> **Mã kế hoạch thử nghiệm:** `exp_v12_e1_e2_plan_20261022`  
> **Bộ dữ liệu nền tảng:** `Vietnamese KWS Dataset v12` (8,000 clips: 4,000 Keywords + 4,000 REJECT)  
> **Trạng thái cổng huấn luyện:** SẴN SÀNG KHỞI ĐỘNG SAU KHI VƯỢT QUA KIỂM ĐỊNH QA GATE DATASET v12  

---

### 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Trong mọi văn bản, báo cáo và phân tích kết quả, các nhãn quản trị bắt buộc sau phải luôn được hiển thị:

- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

**Nguyên tắc bất biến:**
1. Checkpoint v11 (`3c4369...`) và ONNX (`54298c...`) được giữ nguyên bản 100%, không bị ghi đè hay thay thế.
2. Không thực hiện huấn luyện mô hình khi chưa có báo cáo QA Gate PASS đầy đủ 9 tiêu chí.
3. Thử nghiệm **E1 bắt buộc phải hoàn thành, đánh giá toàn diện trước khi được phép xem xét E2**. Nghiêm cấm tăng số lượng tham số (model bloat) khi chưa có bằng chứng thực nghiệm về việc kiến trúc E1 bị bão hòa dung lượng (capacity saturation).

---

### 2. Thiết Kế Thử Nghiệm E1: Kiểm Chứng Sức Mạnh Của Cải Thiện Dữ Liệu (Baseline Scale)

#### 2.1 Giả Thuyết Khoa Học (Hypothesis)
Hiện tượng nhận diện nhầm lệnh (false acceptance) trên giọng nói tự nhiên và từ phủ định đồng âm ("cửa" bị nhận là `PHAI`) ở phiên bản v11 xuất phát chủ yếu từ **sự thiếu hụt dữ liệu phủ định đa dạng (negative sample starvation) và mất cân bằng phân phối âm học**, chứ không phải do thiếu dung lượng tham số mô hình.

Bằng việc huấn luyện lại trên tập dữ liệu Dataset v12 (gồm 1,600 từ OOV, 1,000 đoạn hội thoại FLEURS, 800 đoạn tiếng ồn môi trường, 600 đoạn im lặng/nền phòng) với đúng quy mô kiến trúc v11 (chính xác 138,469 tham số có thể huấn luyện):
- Tỷ lệ kích hoạt sai trên lời nói tự nhiên sẽ giảm >= 80%.
- Từ phủ định "cửa" và các từ OOV chưa từng thấy trong tập kiểm thử ("ghế") sẽ bị loại bỏ với độ chính xác >= 96%.
- Độ nhạy nhận diện từ khóa sạch vẫn duy trì >= 95% ở ngưỡng quyết định chuẩn $\theta^*_{\text{cmd}} = 0.20$.

#### 2.2 Kiến Trúc Mô Hình E1
- **Tên mô hình:** `VietnameseKWS_CNN_v12_E1` (lớp `CorrectedVietnameseKWSNet`)
- **Số tham số:** Chính xác **138,469 tham số huấn luyện** (580 phần tử đệm BatchNorm; tổng state_dict: 139,049 phần tử), khớp 100% kiến trúc v11 có thẩm quyền.
- **Đặc trưng đầu vào:** Log-Mel Spectrogram shape `[1, 1, 40, 101]` từ tín hiệu Mono 16 kHz, 16,000 mẫu (1.0 giây).
- **Cấu trúc mạng:**
  - Conv2D Block 1: Conv2d(1, 32, kernel=3, pad=1, bias=False) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2).
  - Conv2D Block 2: Conv2d(32, 64, kernel=3, pad=1, bias=False) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2).
  - Conv2D Block 3: Conv2d(64, 64, kernel=3, pad=1, bias=False) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2).
  - Conv2D Block 4: Conv2d(64, 128, kernel=3, pad=1, bias=False) -> BatchNorm2d(128) -> ReLU -> AdaptiveAvgPool2d((1, 1)).
  - Linear Block: Flatten(128) -> Dropout(0.3) -> Linear(128, 64) -> ReLU -> Linear(64, 5).
- **5 Lớp đầu ra:** `TRAI`, `PHAI`, `LEN`, `XUONG`, `REJECT`.

#### 2.3 Chiến Lược Huấn Luyện
- **Bộ tối ưu (Optimizer):** `AdamW` (learning rate: $1 \times 10^{-3}$, weight decay: $0.01$).
- **Lịch biểu học tập:** `CosineAnnealingLR` ($T_{\max} = 40$, $\eta_{\min} = 10^{-5}$).
- **Kích thước Batch:** 64.
- **Số Epoch tối đa:** 40 epochs.
- **Dừng sớm (Early Stopping):** Kiên nhẫn (patience) = 8 epochs dựa trên `val_macro_f1`.
- **Hàm mất mát:** `WeightedCrossEntropyLoss` với trọng số tăng cường cho lớp REJECT ($w_{\text{REJECT}} = 1.25$, $w_{\text{CMD}} = 1.0$).
- **Tăng cường trực tiếp (Data Augmentation):** SpecAugment (Frequency masking max 4 kênh, Time masking max 10 khung, Random time shift $\pm 400$ mẫu).

---

### 3. Thiết Kế Thử Nghiệm E2: Mở Rộng Dung Lượng Kiến Trúc (Chỉ Thực Hiện Khi E1 Đạt Giới Hạn)

#### 3.1 Điều Kiện Kích Hoạt Cổng E2 (Gate Condition)
Thử nghiệm E2 **CHỈ ĐƯỢC PHÉP TIẾN HÀNH** nếu và chỉ nếu kết quả thực nghiệm của E1 thỏa mãn một trong các điều kiện sau:
1. Loss huấn luyện của E1 bị nghẽn (plateau) ở mức $> 0.25$ và không thể hội tụ thêm dù đã thử nghiệm điều chỉnh learning rate (underfitting do thiếu dung lượng).
2. Macro F1 trên tập kiểm định của E1 đạt $< 0.92$ do nhầm lẫn giữa các âm trắc/âm tắc phức tạp.
3. Hội đồng nghiên cứu xác nhận bằng văn bản rằng E1 đã khai thác tối đa dung lượng biểu diễn 148k tham số.

#### 3.2 Kiến Trúc Đề Xuất Cho E2 (Nếu Được Phê Duyệt)
- **Tên mô hình:** `VietnameseKWS_CNN_v12_E2`
- **Số tham số:** ~419,333 tham số (Wide Residual 2D-CNN).
- **Cấu trúc mạng:** Kênh tích chập mở rộng [48, 96, 192] kết hợp 2 khối Residual Connection để bảo toàn thông tin thời gian - tần số chi tiết, FC layer ẩn 256 nơ-ron.
- **Ràng buộc độ trễ biên:** Độ trễ suy luận trên CPU laptop không vượt quá 15 ms, dung lượng ONNX model không vượt quá 2.0 MB.

---

### 4. Hệ Thống Tiêu Chí Đánh Giá Thực Nghiệm (Evaluation Battery)

Mọi mô hình (E1 và E2 nếu có) phải được đánh giá chéo trên cả 2 phương thức:

#### 4.1 Đánh Giá Ngoại Tuyến (Offline Evaluation trên Manifests Test)
1. **Macro F1 & Per-Class PR:** Đo lường trên toàn bộ 1,280 mẫu test của Dataset v12.
2. **Độ nhạy từ khóa sạch (Clean Target Recall):** Tại ngưỡng chuẩn $\theta^*_{\text{cmd}} = 0.20$, yêu cầu $\ge 95\%$.
3. **Độ bền âm học theo SNR (Robustness across SNR):** Độ chính xác phân rã theo 5 mức nhiễu (20 dB, 15 dB, 10 dB, 5 dB, 0 dB).
4. **Tỷ lệ từ chối OOV chưa từng thấy (Unseen OOV Rejection Rate):** Đo lường riêng biệt trên 30 từ OOV chưa từng xuất hiện trong tập huấn luyện (bao gồm "ghế"), yêu cầu $\ge 96\%$.

#### 4.2 Đánh Giá Dòng Tín Hiệu Liên Tục (Streaming Evaluation trên Multi-Command Corpus)
1. **Độ chính xác chuỗi đa lệnh (Sequential Sequence Recall):** Đánh giá trên 40 luồng phát kiểm thử (10 hoán vị, 8 lặp lại, 12 chèn từ nhiễu, 10 ép khoảng cách).
2. **Khả năng tách lệnh liên tiếp lặp lại (Repeated Command Resolution):** Đo lường việc nhận diện độc lập 2 lệnh giống nhau cách nhau 350-500ms mà không bị nuốt lệnh.
3. **Tỷ lệ kích hoạt nhầm trên giờ (False Alarm Rate per Hour - FAR/h):** Đo trên 1,000 mẫu hội thoại tự nhiên FLEURS kết hợp, chuẩn hóa theo giờ nói chuyện.
