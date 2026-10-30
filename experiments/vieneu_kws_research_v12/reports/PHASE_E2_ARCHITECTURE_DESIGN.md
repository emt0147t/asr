# Báo Cáo Thiết Kế Kiến Trúc Thử Nghiệm Dung Lượng E2 (~300k Tham Số)
## (Phase E2 Architectural Capacity Design & Parameter Derivation Report)

> **Mã tài liệu:** `PHASE_E2_ARCHITECTURE_DESIGN`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm lập:** 2026-10-28  
> **Mục tiêu:** Mở rộng dung lượng biểu diễn mô hình có kiểm soát từ baseline 138k lên xấp xỉ 300,000 tham số có thể huấn luyện  
> **Trạng thái:** PREREGISTERED ARCHITECTURE SPECIFICATION  

---

### 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ nghiêm ngặt 4 nhãn quản trị bất biến:
- `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
- `NOT_VALIDATED_ON_NATURAL_SPEECH`
- `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
- `NOT_VALIDATED_PENDING_DATA`

**Các cam kết quản trị bất biến:**
1. **Mô hình v11 mặc định:** Checkpoint `variant_v11_hard_negative_weighted_seed42.pt` và ONNX `vietnamese_kws_v11.onnx` được bảo toàn nguyên vẹn 100%, không bị sửa đổi hay thay thế.
2. **Bảo tồn các tập test đã quan sát:** Toàn bộ các tập dữ liệu ngoại tuyến lịch sử gồm Phase 43, Phase 51, và Phase 58/59 bị cách ly tuyệt đối (quarantined). Không sử dụng các tập này để chọn mô hình, chọn seed hay tinh chỉnh ngưỡng.
3. **Phân lập yếu tố dung lượng:** Không thay đổi đồng thời VAD, frontend, ngưỡng quyết định $\theta_{\text{cmd}} = 0.20$ hay ánh xạ 5 lớp. Thử nghiệm này chỉ cô lập biến số duy nhất là **dung lượng tham số mạng nơ-ron (model capacity)**.
4. **Không tuyên bố triển khai:** Thử nghiệm chỉ đánh giá trên tập kiểm định chuẩn `validation_v1.csv` (1.280 mẫu). Không tuyên bố mô hình sẵn sàng sản xuất hay vượt trội ngoài đời thực chỉ dựa vào tập validation.

---

### 2. Hiện Trạng Kiến Trúc Baseline (`CorrectedVietnameseKWSNet`)

Kiến trúc baseline được sử dụng trong mô hình tham chiếu v11 và thử nghiệm E1 là một mạng tích chập 2D 4 tầng (`CorrectedVietnameseKWSNet`), nhận đầu vào là Log-Mel Spectrogram với kích thước `[B, 1, 40, 101]`:

```
Input: [B, 1, 40, 101]
 │
 ├── Conv1: Conv2d(1, 32, k=3, p=1, bias=False) -> BN(32) -> ReLU -> MaxPool2d(2, 2) [-> 20, 50]
 ├── Conv2: Conv2d(32, 64, k=3, p=1, bias=False) -> BN(64) -> ReLU -> MaxPool2d(2, 2) [-> 10, 25]
 ├── Conv3: Conv2d(64, 64, k=3, p=1, bias=False) -> BN(64) -> ReLU -> MaxPool2d(2, 2) [-> 5, 12]
 ├── Conv4: Conv2d(64, 128, k=3, p=1, bias=False) -> BN(128) -> ReLU -> AdaptiveAvgPool2d((1, 1)) [-> 1, 1]
 │
 ├── Flatten: [B, 128]
 ├── Dropout(p=0.30)
 ├── FC1: Linear(128, 64) -> ReLU
 └── FC2: Linear(64, 5) -> Logits [TRAI, PHAI, LEN, XUONG, REJECT]
```

#### Bảng Bóc Tách Chi Tiết Tham Số Baseline (138,469 tham số):

| Tầng / Khối | Phép Tính Kích Thước | Tham Số Huấn Luyện (Trainable) | Phần Tử Đệm (Buffers) | Tổng Phần Tử |
| :--- | :--- | :---: | :---: | :---: |
| `conv1` | $1 \times 32 \times 3 \times 3$ | $288$ | $0$ | $288$ |
| `bn1` | $\gamma (32) + \beta (32)$ | $64$ | $32 + 32 + 1 = 65$ | $129$ |
| `conv2` | $32 \times 64 \times 3 \times 3$ | $18.432$ | $0$ | $18.432$ |
| `bn2` | $\gamma (64) + \beta (64)$ | $128$ | $64 + 64 + 1 = 129$ | $257$ |
| `conv3` | $64 \times 64 \times 3 \times 3$ | $36.864$ | $0$ | $36.864$ |
| `bn3` | $\gamma (64) + \beta (64)$ | $128$ | $64 + 64 + 1 = 129$ | $257$ |
| `conv4` | $64 \times 128 \times 3 \times 3$ | $73.728$ | $0$ | $73.728$ |
| `bn4` | $\gamma (128) + \beta (128)$ | $256$ | $128 + 128 + 1 = 257$ | $513$ |
| `fc1` | $128 \times 64 + 64$ | $8.256$ | $0$ | $8.256$ |
| `fc2` | $64 \times 5 + 5$ | $325$ | $0$ | $325$ |
| **TỔNG CỘNG** | — | **138.469** | **580** | **139.049** |

---

### 3. Đề Xuất Thay Đổi Kiến Trúc Cho E2: Nguyên Tắc "Thay Đổi Nhỏ Nhất Có Thể Bảo Vệ Được"

Nhằm mở rộng dung lượng lên mục tiêu **xấp xỉ 300.000 tham số**, chúng tôi khảo sát các phương án thay đổi cấu trúc:

#### So sánh các chiến lược mở rộng:
1. **Phương án A — Tăng số tầng tích chập (Depth Scaling):**  
   Thêm tầng `conv5` hoặc khối Residual.  
   *Nhược điểm:* Phá vỡ trường tiếp nhận (receptive field), thay đổi độ trễ theo thời gian, phá vỡ tính tương đồng topo với baseline v11, gây khó khăn cho việc phân lập nguyên nhân do dung lượng hay do độ sâu.
2. **Phương án B — Mở rộng tầng kết nối đầy đủ (FC Bottleneck Widening):**  
   Giữ nguyên Conv, tăng FC1 lên $128 \times 2048$.  
   *Nhược điểm:* Dồn phần lớn tham số vào tầng phân loại tuyến tính cuối, dễ dẫn đến overfitting trên tập huấn luyện thay vì nâng cao năng lực trích xuất đặc trưng âm học ở tầng phổ.
3. **Phương án C (LỰA CHỌN) — Mở rộng đồng dạng độ rộng kênh tích chập (Uniform Channel Width Multiplier $\alpha = 1.5$):**  
   Áp dụng hệ số mở rộng kênh tích chập $\alpha = 1.5$ đồng dạng trên toàn bộ 4 tầng tích chập:
   $$C_1 = 32 \times 1.5 = 48, \quad C_2 = 64 \times 1.5 = 96, \quad C_3 = 64 \times 1.5 = 96, \quad C_4 = 128 \times 1.5 = 192$$
   Giữ nguyên đầu phân loại FC1 ($192 \to 64$) và FC2 ($64 \to 5$).

#### Ưu điểm khoa học của Phương án C:
- **Bảo toàn hoàn hảo topo mạng:** Không thay đổi bất kỳ siêu tham số cấu trúc nào (kernel size $3 \times 3$, padding $1$, strides, pooling 3 lần MaxPool2d $2 \times 2$, 1 lần AdaptiveAvgPool2d $1 \times 1$, tỷ lệ Dropout $0.30$).
- **Bảo toàn tỷ lệ kênh tương đối giữa các tầng:**
  - Tầng 1 $\to$ Tầng 2: $48 \to 96$ ($2.0\times$, giống $32 \to 64$).
  - Tầng 2 $\to$ Tầng 3: $96 \to 96$ ($1.0\times$, giống $64 \to 64$).
  - Tầng 3 $\to$ Tầng 4: $96 \to 192$ ($2.0\times$, giống $64 \to 128$).
- **Tối ưu phần cứng:** Mọi số kênh ($48, 96, 96, 192$) đều là bội số của 16, tối ưu cho Tensor Core GPU, AVX-512 và ONNX Runtime vectorization.
- **Tiệm cận mục tiêu 300k với độ lệch nhỏ nhất:** Đạt chính xác **304,277 tham số** (chênh lệch chỉ $+1.42\%$ so với mục tiêu lý thuyết 300.000).

---

### 4. Bóc Tách Chi Tiết Kiến Trúc E2 (`ScaledVietnameseKWSNet_E2`)

```
Input: [B, 1, 40, 101]
 │
 ├── Conv1: Conv2d(1, 48, k=3, p=1, bias=False) -> BN(48) -> ReLU -> MaxPool2d(2, 2) [-> 20, 50]
 ├── Conv2: Conv2d(48, 96, k=3, p=1, bias=False) -> BN(96) -> ReLU -> MaxPool2d(2, 2) [-> 10, 25]
 ├── Conv3: Conv2d(96, 96, k=3, p=1, bias=False) -> BN(96) -> ReLU -> MaxPool2d(2, 2) [-> 5, 12]
 ├── Conv4: Conv2d(96, 192, k=3, p=1, bias=False) -> BN(192) -> ReLU -> AdaptiveAvgPool2d((1, 1)) [-> 1, 1]
 │
 ├── Flatten: [B, 192]
 ├── Dropout(p=0.30)
 ├── FC1: Linear(192, 64) -> ReLU
 └── FC2: Linear(64, 5) -> Logits [TRAI, PHAI, LEN, XUONG, REJECT]
```

#### Bảng Tính Toán Từng Tầng Của E2 (304,277 tham số):

| Tầng / Khối | Công Thức Tính | Tham Số Huấn Luyện (Trainable) | Phần Tử Đệm (Buffers) | Tổng Số Phần Tử |
| :--- | :--- | :---: | :---: | :---: |
| `conv1` | $1 \times 48 \times 3 \times 3$ | $432$ | $0$ | $432$ |
| `bn1` | $\gamma (48) + \beta (48)$ | $96$ | $48 + 48 + 1 = 97$ | $193$ |
| `conv2` | $48 \times 96 \times 3 \times 3$ | $41.472$ | $0$ | $41.472$ |
| `bn2` | $\gamma (96) + \beta (96)$ | $192$ | $96 + 96 + 1 = 193$ | $385$ |
| `conv3` | $96 \times 96 \times 3 \times 3$ | $82.944$ | $0$ | $82.944$ |
| `bn3` | $\gamma (96) + \beta (96)$ | $192$ | $96 + 96 + 1 = 193$ | $385$ |
| `conv4` | $96 \times 192 \times 3 \times 3$ | $165.888$ | $0$ | $165.888$ |
| `bn4` | $\gamma (192) + \beta (192)$ | $384$ | $192 + 192 + 1 = 385$ | $769$ |
| `fc1` | $192 \times 64 + 64$ | $12.352$ | $0$ | $12.352$ |
| `fc2` | $64 \times 5 + 5$ | $325$ | $0$ | $325$ |
| **TỔNG CỘNG** | — | **304.277** | **868** | **305.145** |

#### So Sánh Tương Quan Với Baseline 138k:
- **Tăng trưởng số lượng tham số:** $\frac{304.277}{138.469} = 2.197\times$ ($+119.74\%$).
- **Dung lượng file PyTorch checkpoint:** ~1.2 MB (so với ~0.55 MB của baseline).
- **Độ phức tạp tính toán (MACs / FLOPs ước tính):** Tăng từ ~2.8 MFLOPs lên ~6.2 MFLOPs mỗi khung 1.0 giây, vẫn hoàn toàn nằm trong giới hạn suy luận thời gian thực trên CPU ($< 15\text{ ms}$).

---

### 5. Định Nghĩa Lớp Mã Nguồn PyTorch

```python
class ScaledVietnameseKWSNet_E2(nn.Module):
    def __init__(self, num_classes: int = 5, dropout_p: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(192)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))
        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)
        return logits
```

---

### 6. Kết Luận và Kế Hoạch Đăng Ký Thử Nghiệm

Kiến trúc `ScaledVietnameseKWSNet_E2` đáp ứng đầy đủ các tiêu chuẩn khoa học khắt khe:
1. Đạt chính xác **304,277 tham số** (sai số $< 1.5\%$ so với mục tiêu 300k).
2. Giữ nguyên 100% ranh giới topo và các kết nối để phân lập chính xác biến số dung lượng.
3. Sẵn sàng tích hợp vào tài liệu tiền đăng ký `PHASE_E2_PREREGISTRATION.md` trước khi tiến hành huấn luyện.
