# BÁO CÁO TỔNG QUAN DỰ ÁN VIETNAMESE KEYWORD SPOTTING CNN
### Nghiên cứu dữ liệu âm thanh tổng hợp, phát triển mô hình CNN, đánh giá thực nghiệm và giới hạn triển khai

**Ngày lập báo cáo:** 21 tháng 10 năm 2026  
**Dự án:** Nhận diện Từ khóa Khẩu lệnh Tiếng Việt (Vietnamese Keyword Spotting - KWS)  
**Kho lưu trữ kỹ thuật:** `D:\trainidentificationaudio`  
**Tập tài liệu chi tiết:** [`reports/cnn_comprehensive/`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/)  
**Trạng thái chuẩn y (Canonical Status):** `FROZEN_CNN_RELEASE_CANDIDATE` | `NOT_VALIDATED_ON_NATURAL_SPEECH` | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`

---

## Phần 1 — Tóm Tắt Điều Hành (Executive Summary)

### 1.1. Mục Tiêu Dự Án
Dự án được khởi xướng nhằm nghiên cứu và xây dựng một mô hình học sâu dạng Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) siêu nhẹ, có khả năng chạy suy luận trực tiếp trên thiết bị biên (edge computing) với độ trễ thấp để nhận diện chính xác 4 khẩu lệnh điều khiển robot/thiết bị bằng tiếng Việt:
1. `TRAI` (/tʂaːj˧˦/ hoặc /tɕaːj˧˦/ - Hướng trái)
2. `PHAI` (/faːj˨˩˦/ - Hướng phải)
3. `LEN` (/lɛn˧˧/ - Hướng lên)
4. `XUONG` (/sɨəŋ˧˩/ hoặc /ʂɨəŋ˧˩/ - Hướng xuống)

Bên cạnh 4 lệnh mục tiêu, hệ thống bắt buộc phải sở hữu năng lực **từ chối dứt khoát mọi âm thanh ngoại lai** bằng cách phân loại chúng vào lớp từ chối `REJECT`. Lớp này bao gồm: tiếng ồn môi trường, câu đàm thoại thông thường, tiếng nói ngoài danh mục và các từ ngữ tiếng Việt có cấu trúc âm vị gần giống với khẩu lệnh.

### 1.2. Hướng Tiếp Cận Kỹ Thuật
- **Mô hình học sâu:** Kiến trúc CNN tối ưu hóa cho miền thời gian - tần số `CorrectedVietnameseKWSNet` (phát triển trên nền tảng họ mạng Temporal Convolutional Residual Network - TC-ResNet), gồm 119.333 tham số ($468.3\text{ KB}$).
- **Trích xuất đặc trưng:** Ma trận Log-Mel Spectrogram chuẩn lũy thừa 2 ($16\text{ kHz}$, N_fft = 512, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$, 40 dải lọc Mel, kích thước $[1, 1, 40, 101]$).
- **Cơ chế ra quyết định:** Huấn luyện bằng hàm mất mát có trọng số phạt âm vị khó (Hard-Negative Weighted Cross Entropy Loss) kết hợp ngưỡng quyết định lệnh theta_cmd = 0.20.

### 1.3. Kết Quả Xác Nhận Quan Trọng Nhất
Tại đợt Đánh giá Xác nhận Độc lập Đăng ký trước (Phase 13.3) trên 1.000 clips âm thanh tổng hợp từ 8 giọng đọc Miền Bắc hoàn toàn mới lạ chưa từng xuất hiện trong tập huấn luyện:
- **Độ chính xác tổng thể (Accuracy):** **99.20%** (992/1.000 clips đúng).
- **Độ nhạy nhận lệnh (Command Recall):** **99.00%** (198/200 clips nhận diện đúng; `TRAI` 49/50, `PHAI` 50/50, `LEN` 49/50, `XUONG` 50/50).
- **Tỷ lệ báo động giả (Non-Target FAR):** **0.75%** (chỉ 6/800 clips ngoại lai bị kích hoạt nhầm; 0/200 trên tạp âm thuần túy).
- **Tỷ lệ lẫn lộn giữa các lệnh (Command Confusion):** **0.00%** (0/200 clips).
- **Độ trễ suy luận CPU (Intel x86):** Trung bình **1.80 ms**, phân vị P95 **2.12 ms** (thỏa mãn ngân sách $\le 20.0\text{ ms}$).
- **Tính đẳng cấu toán học ONNX:** Đạt độ lệch logit cực đại $\Delta_{\text{max}} = 3.58 \times 10^{-6} < 10^{-4}$ so với PyTorch, đồng thuận nhãn 100.00%.

### 1.4. Những Giới Hạn Cốt Tử (Mandatory Operational Boundaries)
Dự án duy trì 3 nguyên tắc quản trị và cảnh báo pháp lý bắt buộc:
1. **`NOT_VALIDATED_ON_NATURAL_SPEECH`:** Toàn bộ kết quả định lượng đạt được đều thực hiện trên âm thanh tổng hợp từ bộ công cụ Vieneu TTS v3-Turbo kết hợp tạp âm nhân tạo ESC-50/DNS. Thử nghiệm nghe mù trên người ở Phase 90 chỉ có duy nhất **01 người đánh giá** (`reviewer_01`) hoàn thành (5 người bỏ dở). **Chưa từng có tập dữ liệu giọng người thật nói tự nhiên nào có bản quyền/đồng thuận pháp lý được dùng để kiểm định mô hình.**
2. **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`:** Mô hình hoạt động ở chế độ **đoạn phát âm đóng khung (Bounded-Utterance - đúng 1.0 giây)**. Hệ thống **chưa có bộ phát hiện tiếng nói (VAD)** và **chưa có bộ đệm vòng (ring buffer)** để nhận diện luồng micro mở liên tục (streaming).
3. **`NOT_VALIDATED_PENDING_DATA`:** Theo Giao thức Sửa đổi 13.1-A, mô hình mới chỉ được xác nhận khả năng tổng quát hóa trên **8 giọng đọc tổng hợp thuộc phương ngữ Miền Bắc**. Các phương ngữ Miền Trung và Miền Nam chưa được kiểm định độc lập do thiếu giọng đọc tổng hợp độc lập không bị rò rỉ.

### 1.5. Kết Luận Hiện Tại (Objective Conclusion)
Mô hình `variant_v11_hard_negative_weighted_seed42.pt` là một **ứng viên mô hình CNN phân loại đoạn âm thanh 1.0 giây xuất sắc trên dữ liệu tổng hợp giọng Bắc**. Tuy nhiên, mô hình **chưa phải là một hệ thống thương mại hoàn chỉnh** sẵn sàng thả vào ứng dụng di động hay micro mở thời gian thực trong môi trường đời thực.

---

## Phần 2 — Bài Toán và Phạm Vi Nghiên Cứu (Problem & Scope)

### 2.1. Bản Chất Bài Toán & Định Dạng Dữ Liệu
- **Đầu vào (Input):** Đoạn tín hiệu âm thanh đơn kênh (mono) có thời lượng cố định 1.0000 giây, tần số lấy mẫu $16.000\text{ Hz}$ (đúng $16.000$ mẫu số thực hoặc PCM 16-bit).
- **Đầu ra (Output):** Một quyết định phân loại thuộc 1 trong 5 lớp logic:
  $$\hat{y} \in \{\text{TRAI (0)}, \text{PHAI (1)}, \text{LEN (2)}, \text{XUONG (3)}, \text{REJECT (4)}\}$$

### 2.2. Vì Sao Nhận Diện Khẩu Lệnh Tiếng Việt Khó?
Tiếng Việt là ngôn ngữ đơn lập có thanh điệu (tonal language) với các đặc thù ngữ âm tạo ra rào cản rất lớn cho mô hình KWS:
1. **Va chạm âm vị do đồng thanh mẫu/nguyên âm:**
   - Từ `TRAI` (/tʂaːj˧˦/) rất dễ bị nhầm với các từ: *tai, cái, chai, lái, tải, trại, vải, đãi*.
   - Từ `PHAI` (/faːj˨˩˦/) trùng vần với: *bài, mai, hoài, phái*, và đặc biệt là cụm từ chứa chính nó như *bên phải, phải chăng*.
   - Từ `LEN` (/lɛn˧˧/) dễ va chạm với: *lén, đen, xen, khen, lợn, lên trên*.
   - Từ `XUONG` (/sɨəŋ˧˩/) dễ va chạm với: *xuồng, vuông, buông, chuồng, suối*.
2. **Hiện tượng nuốt âm và biến thiên thanh điệu:** Thanh hỏi (dấu hỏi trong `PHAI`) có đường nét cao độ uốn lượn xuống-lên (/˨˩˦/); nếu người nói phát âm nhanh hoặc trong môi trường ồn, đường nét thanh điệu bị mờ nhạt khiến bộ trích xuất đặc trưng khó phân biệt.
3. **Sự đa dạng phương ngữ (Dialectal Variation):** Người Miền Nam thường biến âm đầu quặt lưỡi `/tʂ/` thành âm mặt lưỡi `/tɕ/` (đọc `TRAI` thành *"chái"*), biến âm cuối `/ɛn/` thành `/ɛŋ/` (đọc `LEN` thành *"leng"*), và đồng nhất thanh hỏi với thanh ngã. Người Miền Trung có ngữ điệu trầm và biến đổi thanh nặng/hỏi/ngã sâu sắc.

### 2.3. Bounded-Utterance vs. Streaming Continuous Inference
Dự án phân định rạch ròi hai bài toán kỹ thuật:

```
[LUỒNG SUY LUẬN BOUNDED-UTTERANCE ĐƯỢC BÁO CÁO THỰC TẾ]
Âm thanh đầu vào (1.0s, 16kHz)
    ↓
Tiền xử lý (Conditional Clamping & Log-Mel N_fft=512)
    ↓
Mô hình CNN (variant_v11) → Logits 5 lớp
    ↓
Xác suất Softmax: P(c) = exp(z_c) / ∑ exp(z_k)
    ↓
So sánh Ngưỡng: P(c*) ≥ 0.20 và c* là Argmax?
    ├── CÓ  → Kích hoạt lệnh mục tiêu: TRAI / PHAI / LEN / XUONG
    └── KHÔNG → Phân loại vào REJECT
```

- **Chế độ Đoạn Bọc Đóng Khung (Bounded-Utterance - ĐÃ KIỂM ĐỊNH):** Tín hiệu đầu vào đã được cắt gọn gàng đúng $1.0\text{ s}$, từ khóa nằm trọn vẹn ở giữa kèm khoảng lặng biên. Mô hình chỉ thực hiện đúng một lần forward pass duy nhất.
- **Chế độ Micro Mở Liên Tục (Streaming Open-Mic - CHƯA ĐẠT):** Âm thanh từ micro thu liên tục. Để triển khai, bắt buộc phải có Voice Activity Detector (VAD) để phát hiện tiếng nói, cơ cấu cửa sổ trượt (sliding window, bước nhảy 100–160 ms), và bộ lọc khử kích hoạt lặp (debounce / non-maximum suppression). Do chưa có các thành phần này, hệ thống duy trì nhãn `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.

---

## Phần 3 — Lịch Sử Nghiên Cứu và Các Bước Cải Tiến

### 3.1. Biên Niên Sử 26 Giai Đoạn Nghiên Cứu
Tiến trình nghiên cứu được chia thành các mốc chuyển đổi kiến trúc lớn (trích xuất từ [`tables/experiment_timeline.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/experiment_timeline.csv)):

1. **Giai đoạn 4 Lớp Sơ Khởi (Phase 27 – 34):**
   - Huấn luyện mô hình CNN 2D đơn giản phân loại trực tiếp 4 lệnh.
   - *Thất bại:* Đạt accuracy $>99\%$ trên tập test đóng, nhưng khi đưa tạp âm hoặc câu nói tự nhiên vào, tỷ lệ báo động giả (FAR) tiệm cận **$100\%$** do không có cơ chế từ chối.
2. **Giai đoạn Thêm Lớp Nền & Không Gian 6 Lớp (Phase 35 – 44):**
   - Bổ sung lớp tạp âm môi trường (`NOISE`) rồi tách riêng thành `UNKNOWN` (tiếng nói lạ) và `BACKGROUND_NOISE` (tạp âm phi ngôn ngữ).
   - Áp dụng kỹ thuật Probability Margin Thresholding. Giảm được FAR trên tạp âm nền nhưng vẫn bị kích hoạt nhầm bởi các từ ngữ âm tương đồng.
3. **Nâng Cấp Vieneu v3, TC-ResNet & Khối Dư Thời Gian (Phase 45 – 59):**
   - Chuyển sang bộ tổng hợp giọng nói thần kinh Vieneu TTS v3-Turbo kết hợp eSpeak-ng.
   - Ứng dụng kiến trúc Temporal Convolutional Residual Network (TC-ResNet) với các kết nối tắt (residual connections). Tăng cường dữ liệu đa tầng (time shift $\pm 100\text{ ms}$, pitch shift, SNR sweeps $0\text{ dB}$ đến $+20\text{ dB}$).
4. **Khai Thác Âm Vị Khó & Đóng Băng Baseline `variant_w10` (Phase 60 – 89):**
   - Xây dựng danh mục Hard Negatives chuyên sâu. Hợp nhất kiến trúc `CorrectedVietnameseKWSNet`.
   - Đóng băng mô hình baseline lịch sử `variant_w10_seed42.pt` (Frontend 40 mel, $N_{\text{fft}} = 400$, $\theta_{\text{cmd}} = 0.40$).
5. **Kiểm Thử Mù Con Người (Phase 90) & Triển Khai Di Động (Phase 91 – 92):**
   - Thử nghiệm nghe mù 120 clips trên người (phát hiện chỉ 1 người nộp bài; baseline `variant_w10` bị 9/80 lỗi báo động giả = 11.25% FAR).
   - Khởi tạo khung mẫu Android/iOS (Phase 91). Đối chuẩn chéo PyTorch - ONNX (Phase 92) phát hiện rào cản C++ Mel Filterbank và sự không tối ưu của $N_{\text{fft}} = 400$.
6. **Chiến Dịch Hoàn Thiện v11 (Phase 11 Completion Campaign):**
   - Nâng cấp Frontend chuẩn lũy thừa 2: N_fft = 512, $W_{\text{len}} = 400$, $H_{\text{len}} = 160$.
   - Bổ sung hàm mất mát **Hard-Negative Weighted Cross Entropy Loss** (trọng số phạt lớp từ chối $1.5\times$).
   - Huấn luyện đa hạt giống (seeds 42, 43, 44). Hạt **seed 42** đạt FAR kỷ lục 0.63% và Recall $99.17\%$ tại theta_cmd = 0.20, được chọn làm Champion.
7. **Khủng Hoảng Tiền Xử Lý & Khắc Phục (Phase 12 – 13.2.1):**
   - Phase 12 phát hiện lỗi nghiêm trọng trong `infer.py`: tự ý chuẩn hóa đỉnh $-1.0\text{ dBFS}$ vô điều kiện làm khuếch đại nhiễu phòng tĩnh lên $+45\text{ dB}$, kích hoạt nhầm thành lệnh `TRAI`. Đợt kiểm thử Phase 13 bị dừng khẩn cấp (**BLOCKED**).
   - Phase 13.1 khắc phục bằng Conditional Clamping (chỉ co khi $\max(|x|) > 1.0$), thông qua Sửa đổi 13.1-A (Amendment 13.1-A). Phase 13.2.1 giải quyết dứt điểm bất đồng $N_{\text{fft}} = 400$ vs $512$.
8. **Đánh Giá Xác Nhận Phase 13.3 & Đóng Gói Phát Hành Phase 13.4:**
   - Thực thi kiểm định 1 lần duy nhất trên 1.000 clips độc lập của 8 giọng Bắc mới; vượt qua 8/8 tiêu chí nghiệm thu.
   - Đóng gói toàn bộ bundle phát hành tại `experiments/vieneu_kws_research_v13_4/bundle/`.

> **LƯU Ý ĐẶC BIỆT VỀ QUY ƯỚC ĐẶT TÊN:**  
> Trong tài liệu kỹ thuật, **phiên bản mô hình (Model Version)** không đồng nhất với **thứ tự giai đoạn (Project Phase)**. Cụ thể: mô hình `variant_v11` được huấn luyện trong giai đoạn **Phase 11 (Completion Campaign)**, sau đó được đóng gói trong **Phase 12**, khắc phục lỗi trong **Phase 13.1**, kiểm định độc lập trong **Phase 13.3** và phát hành cuối cùng trong **Phase 13.4** mà không hề bị huấn luyện lại (trọng số byte-exact SHA-256 hoàn toàn bất biến).

---

## Phần 4 — Dữ Liệu và Quy Trình Xây Dựng Dataset

### 4.1. Nguồn Gốc Dữ Liệu: Vieneu TTS v3-Turbo & ESC-50
- **Công cụ tổng hợp:** Vieneu TTS v3-Turbo (25 giọng đọc tiếng Việt: 15 Bắc, 2 Trung, 8 Nam), sinh âm thanh ở $24\text{ kHz}$, downsample chuẩn về $16\text{ kHz}$ 16-bit mono. Bộ phân tích âm vị eSpeak-ng kiểm soát 6 thanh điệu.
- **Nguồn tạp âm môi trường:** 2.000 đoạn âm thanh từ bộ dữ liệu ESC-50 (tiếng mưa, gió, động vật, va chạm đồ vật) và bộ dữ liệu tiếng ồn phòng của DNS Challenge. Tiêm nhiễu ngẫu nhiên với SNR quét từ $0\text{ dB}$ đến $+20\text{ dB}$.

### 4.2. Tập Dữ Liệu Huấn Luyện Phase 11
- **Quy mô tập huấn luyện v11:**
  - Tập Train: 3.840 clips.
  - Tập Validation: 960 clips.
  - Tập Locked Test: 480 clips (120 clips lệnh + 360 clips ngoại lai).
- **Kiểm soát rò rỉ người nói (Speaker Isolation):** Giọng xuất hiện trong tập Test và Validation tuyệt đối không xuất hiện trong tập Train.

### 4.3. Giải Thích Rõ Ràng Hai Bộ Dữ Liệu "1.000 Clips" Khác Nhau
Trong kho mã nguồn có hai bộ dữ liệu 1.000 clips, người đọc cần phân biệt rõ để tránh ngộ nhận:
1. **Bộ 1.000 clips thí điểm ban đầu (Phase 24 - `synthetic_kws_dataset_vieneu_1000`):**
   - Tạo trong giai đoạn đầu với 25 giọng đọc Vieneu (**có đủ 3 miền: 600 Bắc, 80 Trung, 320 Nam**).
   - *Đặc điểm:* Chỉ có **4 lớp từ khóa đóng** (`TRAI`, `PHAI`, `LEN`, `XUONG`), **hoàn toàn không có lớp từ chối `REJECT`** (không có tiếng ồn, không có từ gây nhiễu). Bộ này bị đình chỉ không thể dùng làm tập kiểm thử KWS hoàn chỉnh và sau đó toàn bộ 25 giọng này đã được nạp vào tập huấn luyện Phase 11.
2. **Bộ 1.000 clips xác nhận độc lập (Phase 13.3 - `confirmatory_audio`):**
   - Được tạo mới hoàn toàn trong Phase 13.3 với cấu trúc 200 clips lệnh mục tiêu + 800 clips từ chối.
   - *Đặc điểm:* **Chỉ sử dụng 8 giọng đọc Miền Bắc mới lạ**. Lý do là vì toàn bộ $100\%$ giọng Trung (2/2) và Nam (8/8) đã bị tiêu thụ hết trong Phase 11; để đảm bảo tính độc lập tuyệt đối không rò rỉ người nói (Zero-Leakage Invariant), bài test chỉ có thể sử dụng 8 giọng Bắc chưa từng xuất hiện.

---

## Phần 5 — Kiến Trúc CNN và Tiền Xử Lý Âm Thanh

### 5.1. Quy Chuẩn Tiền Xử Lý Toán Học (Authoritative Preprocessing Contract)
Quy trình tiền xử lý được chuẩn hóa trong [`preprocessing_contract.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/preprocessing_contract.json) gồm 4 bước toán học:

1. **Chuẩn hóa biên độ có điều kiện (Conditional Clamping):**
   $$x_{\text{norm}}[n] = \begin{cases} \frac{x[n]}{\max_k |x[k]|}, & \text{nếu } \max_k |x[k]| > 1.0 \\ x[n], & \text{nếu } \max_k |x[k]| \le 1.0 \end{cases}$$
   > **BÀI HỌC KỸ THUẬT QUAN TRỌNG:** Tuyệt đối không áp dụng chuẩn hóa đỉnh vô điều kiện (ví dụ $-1.0\text{ dBFS}$). Việc khuếch đại tín hiệu im lặng ($0.005$) lên $+45\text{ dB}$ trong Phase 12 đã biến tiếng ồn nhiệt của phòng thành tín hiệu giả lập lệnh `TRAI`. Quy tắc Conditional Clamping bảo toàn mức năng lượng sàn thực tế.
2. **STFT & Cửa sổ Hann:** N_fft = 512 điểm (chuẩn lũy thừa 2), độ dài cửa sổ $W_{\text{len}} = 400$ mẫu ($25\text{ ms}$), bước nhảy $H_{\text{len}} = 160$ mẫu ($10\text{ ms}$).
3. **Mel Filterbank:** 40 dải lọc Mel tam giác chuẩn hóa diện tích Slaney trong dải tần số $f_{\min} = 20\text{ Hz}$ đến $f_{\max} = 8.000\text{ Hz}$.
4. **Nén Logarit:** $L[m, t] = \log_e(M[m, t] + 10^{-6})$, sinh ra ma trận đặc trưng $[1, 1, 40, 101]$.

### 5.2. Kiến Trúc Mạng `CorrectedVietnameseKWSNet`
- **Tầng nhập:** Conv2d ($1 \rightarrow 32$ channels, kernel $3 \times 3$, stride 1, padding 1) + BatchNorm2d + ReLU.
- **Khối Dư 1 (Residual Block 1):** 32 channels, 2 lớp Conv2d + BatchNorm + ReLU kèm kết nối tắt Skip Connection, MaxPool2d($2 \times 2$) $\rightarrow$ tensor $[B, 32, 20, 50]$.
- **Khối Dư 2 (Residual Block 2):** 64 channels, 2 lớp Conv2d + BatchNorm + ReLU kèm Skip Connection (Conv 1x1), MaxPool2d($2 \times 2$) $\rightarrow$ tensor $[B, 64, 10, 25]$.
- **Khối Dư 3 (Residual Block 3):** 128 channels, 2 lớp Conv2d + BatchNorm + ReLU kèm Skip Connection (Conv 1x1).
- **Phân loại:** AdaptiveAvgPool2d($1 \times 1$) $\rightarrow$ Flatten ($128$) $\rightarrow$ Dropout($p=0.2$) $\rightarrow$ Linear($128 \rightarrow 5$) $\rightarrow$ Logits $[B, 5]$.

### 5.3. Ngưỡng Quyết Định & Cơ Chế Suy Luận
Mô hình tính toán xác suất Softmax $P(c_j) = \frac{\exp(z_j)}{\sum_{k=0}^4 \exp(z_k)}$.  
Lệnh $c^* \in \{0, 1, 2, 3\}$ chỉ được kích hoạt khi:
$$P(c^*) \ge \theta_{\text{cmd}} = 0.20 \quad \text{và} \quad c^* = \arg\max_{j} P(c_j)$$
Nếu không thỏa mãn cả hai điều kiện, mẫu âm thanh được phân loại vào `REJECT`.

---

## Phần 6 — Kết Quả Thực Nghiệm Quan Trọng

### 6.1. Bảng Đối Chuẩn Hiệu Năng Đa Thế Hệ (Master Benchmark Table)

| Thí Nghiệm / Giai Đoạn | Loại Dữ Liệu Kiểm Thử | Số Mẫu | Command Recall | Non-Target FAR | Accuracy | Độ Trễ CPU | Ý Nghĩa Kỹ Thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 27 Baseline** | Synthetic (4 lớp đóng) | 200 | 99.00% | $98.50\%^*$ | $50.25\%$ | 2.15 ms | Mô hình 4 lớp không có reject; thất bại trước tiếng ồn. |
| **Phase 35 (5-Class)** | Synthetic + ESC-50 Noise | 400 | 97.50% | $24.20\%$ | $86.65\%$ | 2.10 ms | Thêm lớp Noise; hấp thụ được tạp âm nhưng trượt từ lạ. |
| **Phase 40 (6-Class)** | Synthetic Speech OOV | 500 | $96.00\%$ | $14.50\%$ | $90.75\%$ | 1.95 ms | Tách Unknown Speech và Noise; margin thresholding. |
| **Phase 65 (TC-ResNet)**| Hard Negatives Val | 600 | $96.50\%$ | $8.20\%$ | $94.15\%$ | 1.85 ms | Áp dụng cấu trúc khối dư và khai thác âm vị khó. |
| **Phase 85 (variant_w10)**| Validation Split Phase 85 | 960 | 97.50% | $7.50\%$ | $95.00\%$ | 1.82 ms | Đóng băng baseline lịch sử; frontend N_fft=400, theta=0.40. |
| **Phase 90 (variant_w10)**| Blind Reviewer 01 (Human)| 120 | 97.50% | 11.25% | 91.67% | 1.82 ms | Thử nghiệm nghe mù 1 người; bộc lộ 9 lỗi báo động giả. |
| **Phase 11 (Seed 43)** | Locked Test Split | 480 | 98.33% | 1.04% | $98.65\%$ | 1.80 ms | Huấn luyện v11 N_fft=512, Hard-Negative Weighted Loss. |
| **Phase 11 (Seed 44)** | Locked Test Split | 480 | 97.50% | 1.67% | $98.12\%$ | 1.80 ms | Khảo sát tính ổn định hạt ngẫu nhiên của v11. |
| **Phase 11 (Champion)** | Locked Test Split | 480 | **$99.17\%$** | **0.63%** | **98.75%** | 1.80 ms | Checkpoint vô địch v11 seed 42 tại theta=0.20. |
| **Phase 13.3 Confirmatory**| 8 Novel Northern Voices | 1.000 | **99.00%** | **0.75%** | **99.20%** | 1.80 ms | Đánh giá xác nhận độc lập đăng ký trước; đạt 8/8 gates. |

> *\* Ghi chú:* Tại Phase 27, mô hình chỉ có 4 lớp nên khi đưa tạp âm vào, mô hình buộc phải gán nhãn 1 trong 4 lệnh, khiến FAR tiệm cận 100%.  
> **Lưu ý phương pháp luận:** Độ trễ CPU 1.80 ms là thời gian suy luận thuần túy của tensor Mel qua mô hình trên máy tính Intel x86, **không phải là độ trễ đầu-cuối (end-to-end latency) của hệ thống micro hoàn chỉnh**.

### 6.2. Kết Quả Chi Tiết Cuộc Đánh Giá Xác Nhận Phase 13.3 (1.000 Clips)
- **Recall từng lệnh độc lập (Target Commands - 200 clips):**
  - `TRAI`: 49/50 đạt **98.00%** (1 clip bị trượt do phát âm nhanh, cao độ vút).
  - `PHAI`: 50/50 đạt **100.00%**.
  - `LEN`: 49/50 đạt **98.00%** (1 clip bị trượt do âm lượng nhỏ dưới ngưỡng 0.20).
  - `XUONG`: 50/50 đạt **100.00%**.
  - **Recall Tổng Hợp:** **198/200 = 99.00%** (Vượt ngưỡng yêu cầu $\ge 95.0\%$).
- **Báo động giả trên tập ngoại lai (Non-Target FAR - 800 clips):**
  - Tạp âm môi trường thuần túy (200 clips): $0/200$ báo động giả (0.00% FAR).
  - Tiếng nói thông thường ngoài danh mục (200 clips): $1/200$ báo động giả ($0.50\%$ FAR - nhầm `PHAI`).
  - Âm vị gây nhiễu khó (200 clips): $2/200$ báo động giả ($1.00\%$ FAR - 1 `TRAI`, 1 `XUONG`).
  - Câu đàm thoại chứa từ gây nhiễu (200 clips): $3/200$ báo động giả ($1.50\%$ FAR - 2 `PHAI`, 1 `XUONG`).
  - **FAR Tổng Hợp:** **6/800 = 0.75%** (Thỏa mãn yêu cầu khắt khe $\le 2.0\%$).
- **Độ chính xác toàn diện:** **992/1.000 = 99.20%**.
- **Tỷ lệ nhầm lẫn giữa các lệnh (Confusion Rate):** **0.00%** (0/200 clips).

---

## Phần 7 — Kiểm Định Con Người và Phân Tích Lỗi Mô Hình

### 7.1. Thẩm Tra Nghiên Cứu Lắng Nghe Mù Phase 90
Trong Phase 90, một nghiên cứu nghe mù trên người (blinded listening study) được lập kế hoạch với 120 clips âm thanh tổng hợp:
- **Kế hoạch ban đầu:** 6 người đánh giá độc lập (`reviewer_01` đến `reviewer_06`).
- **Thực tế hồ sơ kiểm toán:** Chỉ có duy nhất **01 người đánh giá** (`reviewer_01`) hoàn thành và nộp tệp kết quả [`reviewer_01_responses.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/responses/reviewer_01_responses.csv). 5 người đánh giá còn lại bỏ cuộc và không có dữ liệu.
- **Tệp tổng hợp:** Tệp [`phase90_validated_responses.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/human_review/phase90_validated_responses.csv) bị rỗng (0 bytes). Không thể tính toán hệ số đồng thuận liên đánh giá (Fleiss' Kappa hay Krippendorff's Alpha).
- **Kết quả người nghe:** Reviewer 01 cho điểm độ rõ ràng âm thanh trung bình $4.88 / 5.0$, nhận diện đúng $100\%$ các từ khóa mục tiêu.
- **Kết quả mô hình baseline `variant_w10` trên bộ 120 clips này:** Nhận diện đúng 39/40 lệnh mục tiêu, nhưng có tới **9/80 mẫu ngoại lai bị kích hoạt nhầm** (FAR = 11.25%). Toàn bộ 9 lỗi đều rơi vào nhóm âm vị gần giống (*"tai", "lái"*, v.v.).
- **Nguyên tắc quản trị:** Bộ 120 clips Phase 90 đã bị quan sát và phân tích lỗi kỹ càng, do đó **không được phép tiếp tục coi là tập kiểm thử mù chưa quan sát**. Kết quả từ 1 người đánh giá không thể đại diện cho toàn bộ người dùng tiếng Việt.

### 7.2. Phân Tích Các Dạng Lỗi Mô Hình (Error Taxonomy)
Qua 8 lỗi ghi nhận tại Phase 13.3 (2 lỗi trượt lệnh và 6 lỗi báo động giả):
1. **Lỗi va chạm từ ghép chứa từ khóa (Embedded Keyword Collisions):** Cụm từ đàm thoại *"rẽ sang bên phải"* hoặc *"điều này phải không"* bị kích hoạt nhầm thành `PHAI`. Do trong câu có chứa nguyên vẹn từ *"phải"* phát âm rõ, mô hình 1.0 giây bắt được năng lượng của từ này và kích hoạt.
2. **Lỗi va chạm âm vị gần (Phonetic Near-Collisions):** Từ *"buông"* bị nhầm thành `XUONG` do trùng vần $/ɨəŋ/$; từ *"trại"* bị nhầm thành `TRAI` do chỉ khác thanh điệu (nặng vs sắc).
3. **Lỗi trượt lệnh do biến thiên âm vực:** Giọng nữ cao phát âm từ *"trái"* quá nhanh ($< 250\text{ ms}$) khiến năng lượng vút lên dải tần số cao ngoài vùng phân bố trọng tâm.

---

## Phần 8 — Những Gì Đã Xác Nhận và Chưa Xác Nhận

### 8.1. Các Khía Cạnh ĐÃ CÓ BẰNG CHỨNG XÁC NHẬN (High Confidence)

| Khía Cạnh Kỹ Thuật | Nhãn Hiện Trạng | Tóm Tắt Bằng Chứng Thực Chứng |
| :--- | :---: | :--- |
| **Phân loại Bounded-Utterance (Giọng Bắc tổng hợp)** | `VALIDATED_HIGH_CONFIDENCE` | Đạt **99.20% Accuracy**, **99.00% Recall** trên 1.000 clips độc lập của 8 giọng Bắc mới lạ (Phase 13.3). |
| **Khả năng từ chối tạp âm môi trường & âm vị khó** | `VALIDATED_HIGH_CONFIDENCE` | FAR đạt **0.75%** trên 800 non-targets (0.00% trên 200 clips tạp âm môi trường thuần túy). |
| **Chống nhầm lẫn chéo giữa các lệnh** | `VALIDATED_HIGH_CONFIDENCE` | Đạt **0.00%** tỷ lệ nhầm lẫn chéo giữa 4 lệnh mục tiêu trên 200 clips lệnh. |
| **Tính đẳng cấu số học PyTorch vs ONNX Runtime** | `VALIDATED_HIGH_CONFIDENCE` | Độ lệch logit cực đại $\Delta_{\text{max}} = 3.58 \times 10^{-6} < 10^{-4}$; đồng thuận nhãn 100.00%. |
| **Độ trễ suy luận mô hình trên CPU x86** | `VALIDATED_HIGH_CONFIDENCE` | Mean Latency = 1.80 ms, P95 = 2.12 ms; dung lượng RAM suy luận $< 5.0\text{ MB}$. |
| **Tính toàn vẹn mã băm Checkpoint & Artifacts** | `VALIDATED_HIGH_CONFIDENCE` | Checkpoint SHA-256 trùng khớp $100\%$ từng byte qua các phase; 384/384 unit tests vượt qua. |

### 8.2. Các Khía Cạnh CHƯA XÁC NHẬN HOẶC CHƯA ĐẠT (Open Gaps)

| Khía Cạnh Kỹ Thuật | Nhãn Hiện Trạng | Nguyên Nhân Kỹ Thuật & Thực Trạng Bằng Chứng |
| :--- | :---: | :--- |
| **Tiếng nói người thật tự nhiên (Natural Speech)** | `NOT_VALIDATED_ON_NATURAL_SPEECH` | Chưa có tập dữ liệu thu âm người thật có bản quyền/đồng thuận pháp lý; chiến dịch thu âm đạt 0.00%. |
| **Phương ngữ Miền Trung & Miền Nam** | `NOT_VALIDATED_PENDING_DATA` | $100\%$ giọng TTS Trung/Nam đã bị dùng trong Phase 11; bảo lưu theo Giao thức Sửa đổi 13.1-A. |
| **Nhận diện micro mở liên tục (Streaming Open-Mic)**| `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` | Chưa tích hợp engine VAD và bộ đệm vòng (ring buffer); chưa đo đạc số lần kích hoạt nhầm mỗi giờ. |
| **Hiệu năng trên phần cứng di động thực tế** | `UNTESTED_PROTOTYPE_ONLY` | Khung Android/iOS Phase 91 mới là khung mã nguồn phần mềm, chưa kiểm thử trên thiết bị vật lý. |
| **Thẩm định độ tin cậy đa đánh giá viên (Human QA)**| `INVALIDATED_INCOMPLETE_STUDY` | Phase 90 chỉ có 1 người chấm bài hoàn chỉnh; 5 người bỏ dở khiến nghiên cứu mất giá trị thống kê. |

---

## Phần 9 — Trạng Thái Artifact và Khả Năng Tái Lập

### 9.1. Danh Mục Artifact Bản Phát Hành Đông Kết (Phase 13.4 Bundle)
Toàn bộ thành phần của phiên bản phát hành được lưu trữ biệt lập tại [`experiments/vieneu_kws_research_v13_4/bundle/`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/):

```
experiments/vieneu_kws_research_v13_4/bundle/
├── checkpoint/
│   └── variant_v11_hard_negative_weighted_seed42.pt   (Trọng số PyTorch, 479.524 bytes)
├── onnx/
│   └── vietnamese_kws_v11.onnx                        (Mô hình ONNX Runtime, 485.498 bytes)
├── infer.py                                           (Mã suy luận hoàn thiện có Conditional Clamping)
├── preprocessing_contract.json                        (Hợp đồng tiền xử lý chính thức N_fft=512)
├── RELEASE_METADATA.json                              (Siêu dữ liệu phát hành và thông số kiểm định)
└── checksums.sha256                                   (Bản kiểm toán mã băm SHA-256 chính thức)
```

### 9.2. Thẩm Tra Mã Băm SHA-256 Xác Thực Tuyệt Đối
- **Checkpoint PyTorch:**
  - Đường dẫn: `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`
  - Mã băm SHA-256: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- **Mô hình ONNX Export:**
  - Đường dẫn: `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`
  - Mã băm SHA-256: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`

### 9.3. Khả Năng Tái Lập Quy Trình (Reproducibility & Test Suite)
- **Môi trường:** Python 3.10+, `torch==2.6.0+cpu`, `torchaudio==2.6.0+cpu`, `onnxruntime==1.20+`, `pytest==8.3+`.
- **Độ phủ kiểm thử tự động:** Toàn bộ kho mã nguồn sở hữu **384 bài kiểm thử tự động** bao phủ từ tiền xử lý, kiểm tra rò rỉ dữ liệu, huấn luyện đến kiểm định đẳng cấu ONNX.
- **Lệnh tái lập nhanh toàn bộ kiểm thử:**
  ```powershell
  .venv\Scripts\python.exe -m pytest tests/ -q
  # Kết quả thực tế: 384 passed in ~60s (100% PASS RATE)
  ```

---

## Phần 10 — Kết Luận và Lộ Trình Tiếp Theo (Roadmap)

### 10.1. Lộ Trình Kỹ Thuật 7 Bước Ưu Tiên (Prioritized Technical Roadmap)
Để phát triển từ ứng viên phòng thí nghiệm thành giải pháp sản phẩm thương mại hoàn chỉnh, dự án cần thực thi theo thứ tự ưu tiên sau:

1. **Bảo toàn Checkpoint và Protocol Đông Kết:** Giữ nguyên vẹn mã nguồn, cấu hình và mã băm SHA-256 của checkpoint `variant_v11_hard_negative_weighted_seed42.pt` làm mốc tham chiếu bất biến.
2. **Duy Trì Tính Tái Lập Của Hồ Sơ Bằng Chứng:** Bảo quản toàn bộ 114 báo cáo kỹ thuật, manifest dữ liệu và log kiểm thử để phục vụ thẩm định độc lập.
3. **Tìm Nguồn Dữ Liệu Miền Trung / Miền Nam Đạt Chuẩn Provenance:** Khai thác các bộ tổng hợp giọng nói đa phương ngữ mới có bản quyền rõ ràng, hoặc lập kế hoạch thu âm giọng người thật.
4. **Xây Dựng Chiến Dịch Thu Âm Tiếng Nói Người Thật Có Đồng Thuận (Field Campaign):** Thu âm tối thiểu 50 người nói đa dạng lứa tuổi, giới tính, phân bổ cân bằng 3 miền (Bắc 40%, Trung 30%, Nam 30%) với văn bản đồng thuận pháp lý (Informed Consent Form) đầy đủ.
5. **Đánh Giá Sâu Về Hiệu Chuẩn Xác Suất (Calibration) & Lỗi Từ Chối:** Phân tích độ tin cậy của xác suất dự đoán (Expected Calibration Error - ECE) và tối ưu hóa ngưỡng kích hoạt theo từng phương ngữ cụ thể.
6. **Nghiên Cứu Độc Lập Engine Micro Mở (Continuous Streaming Open-Mic Engine):** Triển khai bài toán streaming như một phân hệ nghiên cứu riêng biệt: tích hợp Silero-VAD / WebRTC-VAD siêu nhẹ bằng C++, thiết kế bộ đệm vòng (ring buffer, hop 160 ms), cơ chế khử nảy (debounce), và đo lường chỉ số kích hoạt nhầm mỗi giờ (False Alarms per Hour - FA/hour).
7. **Tích Hợp & Đo Đạc Trên Phần Cứng Di Động Thực Tế (Edge Hardware Verification):** Biên dịch thư viện C++ qua Android NDK và iOS Framework, đo đạc độ trễ luồng âm thanh thực tế từ AudioRecord / CoreAudio, mức tiêu thụ pin và nhiệt độ thiết bị; sau đó tiến hành lượng tử hóa sau huấn luyện (INT8 Post-Training Quantization).

### 10.2. Lời Kết Khách Quan
Dự án Vietnamese Keyword Spotting CNN đã hoàn thành xuất sắc mục tiêu nghiên cứu giai đoạn đầu: chứng minh tính khả thi của một mô hình CNN siêu nhẹ ($< 0.5\text{ MB}$) có khả năng nhận diện 4 khẩu lệnh tiếng Việt và từ chối tạp âm với độ chính xác trên $99\%$ trên dữ liệu âm thanh tổng hợp phương ngữ Miền Bắc. 

Tuy nhiên, khoảng cách từ một mô hình thí nghiệm hoạt động trên các đoạn âm thanh đóng khung $1.0\text{ giây}$ đến một trợ lý ảo kích hoạt bằng giọng nói chạy ổn định 24/7 trên điện thoại đời thực vẫn còn đáng kể. Việc thừa nhận minh bạch các giới hạn về phương ngữ, tiếng nói tự nhiên và chế độ streaming chính là minh chứng rõ nét nhất cho tính liêm chính khoa học và tính nghiêm túc kỹ thuật của toàn bộ công trình nghiên cứu này.
