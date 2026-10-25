import os

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")

ch02_content = """# Chương 02: Lịch sử Nghiên cứu Theo Tiến trình Thời gian (Chronological Research History)

## 1. Tổng quan Tiến trình 26 Giai đoạn Nghiên cứu
Dự án Phát hiện Từ khóa Tiếng Việt (Vietnamese Keyword Spotting - KWS) đã trải qua 26 giai đoạn phát triển liên tục, từ các thử nghiệm phân loại ban đầu đến hệ thống CNN siêu nhẹ được kiểm định độc lập và đóng gói phát hành chuẩn công nghiệp.

Dưới đây là sơ đồ dòng thời gian tiến hóa của các mốc nghiên cứu chính:

```mermaid
flowchart TD
    P27["Phase 27-29: Khởi tạo Baseline & 4 Lớp"] --> P30["Phase 30-34: Khảo sát Seed & Thất bại trước Tạp âm"]
    P30 --> P35["Phase 35-39: Bổ sung Tạp âm Nền & Không gian 5 Lớp"]
    P35 --> P40["Phase 40-44: Không gian 6 Lớp & Phân tách Speech/Noise"]
    P40 --> P45["Phase 45-49: Nâng cấp Vieneu v3 & Đa dạng Người nói"]
    P45 --> P50["Phase 50-54: Kiến trúc TC-ResNet & Khối Dư"]
    P50 --> P55["Phase 55-59: Tăng cường Dữ liệu Đa tầng (Augmentation)"]
    P55 --> P60["Phase 60-64: Khai thác Âm vị Khó (Hard-Negative Mining)"]
    P60 --> P65["Phase 65-69: Hợp nhất Kiến trúc CorrectedVietnameseKWSNet"]
    P65 --> P70["Phase 70-74: Hiệu chuẩn Xác suất & Tối ưu Ngưỡng"]
    P70 --> P75["Phase 75-79: Kiểm thử Khả năng Tổng quát Giọng lạ"]
    P75 --> P80["Phase 80-84: Kiểm thử Độ bền Môi trường Khắc nghiệt"]
    P80 --> P85["Phase 85-89: Đóng băng Baseline variant_w10"]
    P85 --> P90["Phase 90: Đánh giá Lắng nghe Mù (Human Listening Audit)"]
    P85 --> P91["Phase 91: Khung Mẫu Triển khai Di động & C++"]
    P91 --> P92["Phase 92: Đối chuẩn Đẳng cấu Đa nền tảng"]
    P90 --> P11["Phase 11: Chiến dịch Hoàn thiện CNN (variant_v11)"]
    P92 --> P11
    P11 --> P12["Phase 12: Đóng gói Ứng viên & Phát hiện Bug Khuếch đại"]
    P12 --> P13["Phase 13: Thử nghiệm Xác nhận Độc lập Bị Chặn (Blocked)"]
    P13 --> P13_1["Phase 13.1: Khắc phục Tiền xử lý & Sửa đổi 13.1-A"]
    P13_1 --> P13_2["Phase 13.2: Chuẩn bị Giao thức Xác nhận 1.000 Clips"]
    P13_2 --> P13_2_1["Phase 13.2.1: Thẩm định Tương thích Frontend N_fft"]
    P13_2_1 --> P13_3["Phase 13.3: Thực thi Đánh giá Xác nhận 1.000 Clips"]
    P13_3 --> P13_4["Phase 13.4: Đóng gói Phát hành Cuối cùng (Final Bundle)"]
```

---

## 2. Chi tiết Từng Giai đoạn Nghiên cứu

### 2.1. Giai đoạn Sơ khởi: 4 Lớp và Ảo tưởng Hiệu năng (Phase 27 - 34)
- **Mục tiêu:** Xây dựng mô hình phân loại 4 hướng (`TRAI`, `PHAI`, `LEN`, `XUONG`) từ dữ liệu tổng hợp thử nghiệm.
- **Phương pháp:** Sử dụng mô hình CNN 2D cơ bản với đầu vào Spectrogram thô, phân loại bằng hàm Softmax Argmax trực tiếp trên 4 lớp.
- **Hiện tượng & Thất bại:**
  - Trên tập dữ liệu đóng (chỉ gồm 4 từ khóa đọc rõ ràng), mô hình đạt độ chính xác gần như tuyệt đối ($> 99\\%$).
  - Tuy nhiên, khi đưa vào môi trường có tiếng ồn hoặc người dùng nói bất kỳ từ nào khác (ví dụ: *"chạy đi"*, *"xin chào"*), mô hình bắt buộc phải gán vào 1 trong 4 nhãn lệnh. Tỷ lệ kích hoạt nhầm (False Alarm Rate) trên âm thanh ngoại lai tiệm cận **$100\\%$**.
- **Bài học rút ra:** Hệ thống KWS thực tế không thể chỉ là bài toán phân loại đóng (closed-set classification), mà phải có khả năng từ chối (open-set rejection).

### 2.2. Giai đoạn Hấp thụ Tạp âm & Không gian 5 Lớp (Phase 35 - 39)
- **Mục tiêu:** Bổ sung lớp thứ 5 đại diện cho tạp âm (`NOISE` / `BACKGROUND_NOISE`) để hấp thụ các âm thanh môi trường.
- **Phương pháp:** Thu thập các đoạn âm thanh nhiễu từ bộ dữ liệu ESC-50 và tiếng ồn văn phòng, đưa vào huấn luyện như một lớp độc lập. Quét dải SNR từ $+20\\text{ dB}$ xuống $0\\text{ dB}$.
- **Kết quả:** Mô hình có khả năng loại bỏ tiếng gõ bàn phím, tiếng quạt gió và tiếng vỗ tay rất tốt.
- **Hạn chế tồn tại:** Mô hình vẫn kích hoạt nhầm khi gặp tiếng nói con người không phải từ khóa (human speech out-of-vocabulary). Tiếng nói ngoài mục tiêu có phổ tần số rất khác tạp âm môi trường và gần với khẩu lệnh hơn, khiến lớp `NOISE` bị bất lực.

### 2.3. Không gian 6 Lớp & Phân tách Giọng nói Ngoài Danh mục (Phase 40 - 44)
- **Mục tiêu:** Tách riêng tạp âm không phải tiếng nói (`BACKGROUND_NOISE`) và tiếng nói không phải khẩu lệnh (`UNKNOWN`).
- **Không gian nhãn:** `[TRAI, PHAI, LEN, XUONG, UNKNOWN, BACKGROUND_NOISE]`.
- **Đóng góp kỹ thuật:** Áp dụng cơ chế ngưỡng biên độ chênh lệch xác suất (Probability Margin Thresholding):
  $$\\max_{j \\in \\text{commands}} P(c_j) - P(\\text{UNKNOWN}) > \\Delta_{\\text{margin}}$$
- **Kết quả:** Giảm mạnh các kích hoạt nhầm thông thường. Tuy nhiên, mô hình vẫn dễ bị nhầm lẫn bởi các từ có âm vị tương đồng (phonetic confusions).

### 2.4. Nâng cấp Bộ Tổng hợp Vieneu v3 & Đa dạng Người nói (Phase 45 - 49)
- **Nâng cấp dữ liệu:** Chuyển đổi toàn bộ pipeline tổng hợp sang **Vieneu TTS v3-Turbo**, hỗ trợ tạo âm thanh tiếng Việt tự nhiên với bộ phiên âm âm vị eSpeak-ng chuẩn xác.
- **Phân tách người nói:** Thiết lập quy tắc cách ly giọng đọc nghiêm ngặt. Phân bổ các ID giọng cố định cho Train, Val và Test để tránh hiện tượng rò rỉ đặc trưng người nói (speaker leakage).

### 2.5. Khảo sát Kiến trúc TC-ResNet & Khối Dư Thời gian (Phase 50 - 59)
- **Cải tiến kiến trúc:** Áp dụng họ kiến trúc **Temporal Convolutional Residual Network (TC-ResNet)** với các tích chập 1D theo trục thời gian trên đặc trưng Mel Spectrogram, kết hợp kết nối tắt (residual skip connections).
- **Tăng cường dữ liệu đa tầng (Phase 55 - 59):** Bổ sung dịch chuyển thời gian ngẫu nhiên (time shift $\\pm 100\\text{ ms}$), co giãn tốc độ (time stretch), thay đổi cao độ (pitch shift), và trộn nhiễu ngẫu nhiên với SNR biến thiên từ $+5\\text{ dB}$ đến $+20\\text{ dB}$.

### 2.6. Khai thác Âm vị Khó (Hard-Negative Mining) & Hợp nhất Kiến trúc (Phase 60 - 69)
- **Phát hiện lỗi âm vị:** Qua kiểm thử, phát hiện các từ tiếng Việt có tiền tố hoặc vần gần giống gây kích hoạt nhầm nghiêm trọng:
  - *"tai"*, *"lái"*, *"trại"* $\\rightarrow$ nhầm thành `TRAI`
  - *"bên phải"*, *"phấn"*, *"hải"* $\\rightarrow$ nhầm thành `PHAI`
  - *"lén"*, *"lên trên"*, *"bền"* $\\rightarrow$ nhầm thành `LEN`
  - *"xuồng"*, *"buông"*, *"uống"* $\\rightarrow$ nhầm thành `XUONG`
- **Chiến lược Hard-Negative:** Tổng hợp hàng ngàn mẫu âm thanh chứa các từ gây nhiễu này và đưa vào tập huấn luyện với trọng số phạt cao hơn.
- **Kiến trúc `CorrectedVietnameseKWSNet`:** Chuẩn hóa kiến trúc CNN với các tầng tích chập chuẩn hóa Batch Normalization, Dropout chống quá khớp, và Global Average Pooling trước lớp phân loại tuyến tính.

### 2.7. Tối ưu Ngưỡng & Đóng băng Baseline `variant_w10` (Phase 70 - 89)
- **Tối ưu ngưỡng (Phase 70 - 74):** Thay vì sử dụng Argmax, áp dụng ngưỡng quyết định lệnh $\\theta_{\\text{cmd}}$. Khẩu lệnh chỉ được kích hoạt nếu:
  $$P(c^*) \\ge \\theta_{\\text{cmd}} \\quad \\text{và} \\quad c^* = \\arg\\max_{c} P(c)$$
- **Đóng băng Baseline (Phase 85 - 89):** Đóng băng mô hình `variant_w10_seed42.pt` với frontend 40 Mel bins, $N_{\\text{fft}} = 400$, và $\\theta_{\\text{cmd}} = 0.40$. Mô hình này phục vụ làm thước đo đối chuẩn (frozen baseline) cho các giai đoạn tiếp theo.

### 2.8. Kiểm thử Mù trên Người (Phase 90) & Triển khai Di động (Phase 91 - 92)
- **Kiểm thử mù Phase 90:** Thiết kế bộ 120 clip kiểm thử mù (40 clips lệnh, 80 clips không phải lệnh) giao cho 6 người đánh giá độc lập. Thẩm tra thực tế cho thấy: chỉ có **01 người đánh giá** (`reviewer_01`) hoàn thành nộp kết quả; 5 người còn lại bỏ dở. Checkpoint `variant_w10` đạt 39/40 lệnh đúng, nhưng có tới 9/80 mẫu ngoại lai bị kích hoạt nhầm (FAR = $11.25\\%$).
- **Triển khai di động Phase 91 - 92:** Xây dựng khung mã nguồn xuất ONNX, mã bao C++ và khung ứng dụng Android/iOS. Đo đạc độ trễ trên CPU và kiểm tra tính nhất quán giữa PyTorch và ONNX Runtime.

### 2.9. Chiến dịch Hoàn thiện CNN v11 (Phase 11 Completion Campaign)
- **Động lực:** Khắc phục tỷ lệ báo động giả cao ($11.25\\%$) của `variant_w10` trên các âm vị khó.
- **Giải pháp v11:**
  1. Nâng cấp frontend trích xuất Log-Mel sang cấu hình chuẩn lũy thừa 2: $N_{\\text{fft}} = 512$, $W_{\\text{len}} = 400$, $H_{\\text{len}} = 160$.
  2. Bổ sung tập dữ liệu Hard Negatives mở rộng với các từ ngữ âm tương đồng sâu.
  3. Áp dụng hàm mất mát **Hard-Negative Weighted Cross Entropy Loss** (tăng trọng số phạt cho các lỗi âm vị lên 1.5x - 2.0x).
- **Kết quả:** Huấn luyện 3 hạt ngẫu nhiên (seeds 42, 43, 44). Hạt **seed 42** thể hiện sự vượt trội toàn diện trên tập kiểm thử khóa với FAR giảm xuống mức kỷ lục **$0.63\\%$** (3/480 clips) và Recall đạt **$99.17\\%$** (119/120 clips) tại ngưỡng $\\theta_{\\text{cmd}} = 0.20$. Checkpoint `variant_v11_hard_negative_weighted_seed42.pt` được chọn làm Champion.

### 2.10. Khủng hoảng Tiền xử lý & Khắc phục Toàn diện (Phase 12 - 13.2.1)
- **Phase 12 (Đóng gói Ứng viên):** Tạo bundle phát hành v12. Tuy nhiên, mã suy luận trong `infer.py` đã tự ý áp dụng hàm chuẩn hóa đỉnh biên độ $-1.0\\text{ dBFS}$ vô điều kiện cho mọi tín hiệu âm thanh vào.
- **Phase 13 (Thử nghiệm Xác nhận Bị Chặn):** Trong đợt kiểm thử độc lập Phase 13, kiểm toán viên phát hiện: khi đưa một đoạn âm thanh gần như yên lặng (nhiễu phòng với biên độ cực đại $0.005$) vào `infer.py`, hàm này đã khuếch đại tín hiệu lên $+45\\text{ dB}$, biến nhiễu nền thành một xung âm thanh lớn và kích hoạt nhầm thành lệnh `TRAI`. Cuộc kiểm thử bị đình chỉ ngay lập tức (**BLOCKED**).
- **Phase 13.1 (Khắc phục Tiền xử lý & Sửa đổi 13.1-A):**
  - Khôi phục logic chuẩn hóa chính xác của tập huấn luyện `train_v11.py`: **Chỉ co tỷ lệ khi $\\max(|x|) > 1.0$**; các tín hiệu âm lượng nhỏ hoặc bình thường được giữ nguyên biên độ tự nhiên.
  - Thông qua Sửa đổi 13.1-A (Amendment 13.1-A): Giới hạn phạm vi tuyên bố xác nhận độc lập vào 8 giọng tổng hợp mới thuộc phương ngữ Miền Bắc; tạm hoãn tuyên bố cho giọng Trung và Nam cho đến khi có dữ liệu độc lập.
- **Phase 13.2 & 13.2.1 (Chuẩn bị & Thẩm định Khớp Frontend):**
  - Đăng ký trước (preregister) giao thức 1.000 clips độc lập.
  - Giải quyết dứt điểm bất đồng $N_{\\text{fft}} = 400$ vs $512$: Làm rõ rằng con số 400 trong báo cáo Phase 92 là dành riêng cho baseline cũ `variant_w10`, còn kiến trúc `variant_v11` sử dụng chính xác $N_{\\text{fft}} = 512$ trong toàn bộ mã huấn luyện và suy luận.

### 2.11. Đánh giá Xác nhận Độc lập Phase 13.3 & Đóng gói Phát hành Phase 13.4
- **Thực thi Phase 13.3:** Đánh giá 1 lần duy nhất (one-shot execution) trên 1.000 clips âm thanh tổng hợp mới tinh từ 8 giọng Bắc chưa từng thấy. Kết quả vượt qua toàn bộ 8/8 tiêu chí nghiệm thu nghiêm ngặt (Accuracy $99.20\\%$, Recall $99.00\\%$, FAR $0.75\\%$).
- **Phát hành Phase 13.4:** Đóng gói toàn bộ mô hình, mã nguồn suy luận, tệp cấu hình hợp đồng tiền xử lý, mô hình ONNX đã đối chuẩn và các bản kiểm toán bảo mật vào bundle chuẩn hóa tại `experiments/vieneu_kws_research_v13_4/bundle/`. Checkpoint được đóng băng vĩnh viễn với nhãn `FROZEN_CNN_RELEASE_CANDIDATE`.
"""

ch03_content = """# Chương 03: Nguồn gốc Dữ liệu & Phân chia Tập dữ liệu (Dataset Provenance & Splits)

## 1. Công nghệ Tạo Sinh Dữ liệu: Vieneu TTS v3-Turbo
Do hạn chế về dữ liệu thu âm người thật có sự đồng thuận pháp lý đầy đủ, toàn bộ quá trình nghiên cứu và tối ưu hóa mô hình CNN KWS trong dự án dựa trên nguồn âm thanh tổng hợp chất lượng cao được sản xuất bởi công cụ **Vieneu TTS v3-Turbo**:
- **Cấu trúc sinh âm:** Hệ thống mạng nơ-ron tổng hợp giọng nói tiếng Việt end-to-end, kết hợp bộ phân tích âm vị eSpeak-ng hỗ trợ đầy đủ 6 thanh điệu tiếng Việt (ngang, huyền, sắc, hỏi, ngã, nặng).
- **Tần số lấy mẫu:** Tạo âm thanh gốc ở $24\\text{ kHz}$, sau đó được downsample chất lượng cao về $16\\text{ kHz}$ (16-bit PCM mono).
- **Đặc tính âm học:** Kiểm soát được độ dài phát âm, tốc độ nói (speed rate), cao độ (pitch scale), và các biến thể ngữ điệu nhằm mô phỏng sự đa dạng của người nói.

## 2. Chiến lược Phân chia Người nói Tránh Rò rỉ (Speaker Leakage Prevention)
Để đảm bảo tính trung thực khoa học, dự án áp dụng nguyên tắc cách ly tuyệt đối mã định danh người nói (Speaker ID isolation) xuyên suốt các tập Train, Validation và Test:

```mermaid
flowchart TD
    subgraph Pool ["Tổng tập Giọng Nói Vieneu TTS"]
        direction TB
        V_Train["Giọng Huấn luyện (Train Voices)<br/>Nam & Nữ Miền Bắc"]
        V_Val["Giọng Thẩm định (Validation Voices)<br/>Nam & Nữ Miền Bắc"]
        V_Test["Giọng Kiểm thử Khóa (Locked Test Voices)<br/>Nam & Nữ Miền Bắc"]
        V_Conf["8 Giọng Xác nhận Độc lập (Phase 13.3)<br/>Chưa từng dùng trước đó"]
    end

    V_Train --> Split_Train["Tập Train (Phase 11)<br/>3.840 clips"]
    V_Val --> Split_Val["Tập Val (Phase 11)<br/>960 clips"]
    V_Test --> Split_Test["Tập Test Khóa (Phase 11)<br/>480 clips"]
    V_Conf --> Split_Conf["Tập Confirmatory (Phase 13.3)<br/>1.000 clips độc lập"]
```

### Nguyên tắc Bất biến:
1. **Không trùng lặp giọng:** Giọng xuất hiện trong tập kiểm thử (Test và Confirmatory) tuyệt đối không được có mặt trong tập huấn luyện (Train).
2. **Không trùng lặp ngữ cảnh câu:** Các câu chứa từ phủ định hoặc từ gây nhiễu trong tập kiểm thử độc lập phải là các câu mới chưa từng xuất hiện trong quá trình huấn luyện.

## 3. Thiết kế Âm vị Gây nhiễu Khó (Phonetically Matched Hard Negatives)
Để giải quyết triệt để vấn đề báo động giả của mô hình KWS, dự án đã xây dựng danh mục từ vựng gây nhiễu âm vị học chuyên sâu:

| Lớp Mục tiêu | Âm vị Cốt lõi IPA | Từ Gây nhiễu Âm vị Học (Hard Negatives) | Bản chất Ngữ âm Gây Nhầm |
| :--- | :--- | :--- | :--- |
| `TRAI` | /tʂaːj˧˦/ | *trái, tai, cái, chai, lái, tải, trại, vải, đãi, thoái* | Trùng nguyên âm đôi `/aːj/`, chỉ khác phụ âm đầu hoặc thanh điệu. |
| `PHAI` | /faːj˨˩˦/ | *phải, bài, mai, hoài, phái, pháo, phải chăng, bên phải* | Trùng vần `/aːj/` với thanh hỏi; các cụm từ ghép chứa từ *phải*. |
| `LEN` | /lɛn˧˧/ | *lên, lén, đen, xen, khen, lết, lợn, lớn, lên trên, leo* | Trùng phụ âm đầu `/l/` hoặc vần `/ɛn/` âm mũi kết thúc. |
| `XUONG` | /sɨəŋ˧˩/ | *xuống, xuồng, chuồng, vuông, buông, suối, uổng, xương* | Trùng âm đệm và nguyên âm đôi `/ɨəŋ/` với thanh nặng hoặc huyền. |

Ngoài các từ đơn, tập dữ liệu còn tổng hợp các câu đàm thoại thông thường (Conversational Carrier Phrases) chèn các từ gây nhiễu vào các vị trí đầu câu, giữa câu và cuối câu để kiểm tra khả năng bóc tách từ khóa của mô hình.

## 4. Tiêm Nhiễu Môi trường & Mô phỏng Phòng (Noise Injection Pipeline)
Nhằm đảm bảo mô hình hoạt động bền bỉ trong môi trường thực tế, quy trình tạo dữ liệu thực hiện tiêm nhiễu đa dạng:
1. **Tập tạp âm môi trường:** Sử dụng 2.000 đoạn âm thanh từ bộ dữ liệu mã nguồn mở ESC-50 (tiếng mưa rơi, tiếng gõ cửa, tiếng chó sủa, tiếng xe cộ) và tập nhiễu phòng của DNS Challenge.
2. **Tỷ lệ Tín hiệu trên Nhiễu (SNR Sweeps):**
   - Clean (Không nhiễu): $10\\%$ tổng số mẫu.
   - High SNR ($+15\\text{ dB}$ đến $+20\\text{ dB}$): $40\\%$ tổng số mẫu.
   - Moderate SNR ($+5\\text{ dB}$ đến $+10\\text{ dB}$): $35\\%$ tổng số mẫu.
   - Low SNR ($0\\text{ dB}$ đến $+5\\text{ dB}$): $15\\%$ tổng số mẫu.
3. **Mô phỏng Dịch chuyển Vị trí (Time Shifting):** Tín hiệu giọng nói được dịch chuyển ngẫu nhiên trong khung $1.0\\text{ giây}$ với khoảng lùi từ $-150\\text{ ms}$ đến $+150\\text{ ms}$, bù đệm bằng khoảng lặng hoặc nhiễu nền thực tế.

## 5. Cấu trúc Tập Dữ liệu Xác nhận Độc lập Phase 13.3 (1.000 Clips)
Tập dữ liệu độc lập này được đăng ký trước (preregistered) tại Phase 13.2 và được tạo mới hoàn toàn trong Phase 13.3:
- **Quy mô:** Đúng 1.000 tệp âm thanh WAV ($16\\text{ kHz}$, 16-bit mono, thời lượng đúng $1.0000\\text{ s}$).
- **Cơ cấu phân bổ:**
  - **Tập Lệnh Mục tiêu (Target Commands):** 200 clips (chia đều 50 clips cho mỗi lệnh: `TRAI`, `PHAI`, `LEN`, `XUONG`).
  - **Tập Âm thanh Ngoài Mục tiêu (Non-Target Rejection):** 800 clips, gồm:
    - 200 clips âm vị gây nhiễu khó (Phonetic Hard Negatives).
    - 200 clips câu đàm thoại tự nhiên chứa từ gây nhiễu (Carrier Phrases).
    - 200 clips tiếng nói ngoài danh mục thông thường (Out-of-Vocabulary General Speech).
    - 200 clips tạp âm môi trường thuần túy (Pure Environmental Noise).
- **8 Giọng Đọc Độc lập:** 4 giọng Nam (`voice_n_m01` đến `voice_n_m04`) và 4 giọng Nữ (`voice_n_f01` đến `voice_n_f04`) thuộc phương ngữ Miền Bắc, hoàn toàn mới lạ đối với mô hình.
"""

ch04_content = """# Chương 04: Kiến trúc Mô hình & Huấn luyện (Model Architectures & Training)

## 1. Kiến trúc Mô hình: `CorrectedVietnameseKWSNet`
Mô hình được lựa chọn và đóng băng là `CorrectedVietnameseKWSNet` (phiên bản hoàn thiện của họ mạng TC-ResNet cải tiến cho âm học tiếng Việt).

Mô hình nhận đầu vào là ma trận đặc trưng Log-Mel Spectrogram có kích thước:
$$X \\in \\mathbb{R}^{B \\times 1 \\times 40 \\times 101}$$
Trong đó:
- $B$: Batch size (kích thước lô huấn luyện/suy luận).
- $1$: Số kênh đơn (single channel).
- $40$: Số dải tần số Mel (Mel filterbank channels).
- $101$: Số khung thời gian (Time frames cho đoạn âm thanh 1.0 giây).

### Sơ đồ Kiến trúc Chi tiết:
```mermaid
flowchart TD
    Input["Input Tensor [B, 1, 40, 101]"] --> Conv0["Conv2d(1 -> 32, kernel=3, stride=1, pad=1)"]
    Conv0 --> BN0["BatchNorm2d(32)"]
    BN0 --> Act0["ReLU()"]
    
    Act0 --> Block1["Residual Block 1 (32 channels)<br/>Conv2d -> BN -> ReLU -> Conv2d -> BN + Skip"]
    Block1 --> Pool1["MaxPool2d(kernel=(2, 2), stride=(2, 2))<br/>Kích thước: [B, 32, 20, 50]"]
    
    Pool1 --> Block2["Residual Block 2 (64 channels)<br/>Conv2d -> BN -> ReLU -> Conv2d -> BN + Skip(Conv 1x1)"]
    Block2 --> Pool2["MaxPool2d(kernel=(2, 2), stride=(2, 2))<br/>Kích thước: [B, 64, 10, 25]"]
    
    Pool2 --> Block3["Residual Block 3 (128 channels)<br/>Conv2d -> BN -> ReLU -> Conv2d -> BN + Skip(Conv 1x1)"]
    Block3 --> GAP["AdaptiveAvgPool2d((1, 1))<br/>Kích thước: [B, 128, 1, 1]"]
    
    GAP --> Flat["Flatten -> [B, 128]"]
    Flat --> Drop["Dropout(p=0.2)"]
    Drop --> Linear["Linear(128 -> 5)"]
    Linear --> Logits["Output Logits [B, 5]"]
```

### Thống kê Tham số Mô hình:
- **Tổng số tham số có thể huấn luyện (Trainable Parameters):** 119.333 tham số.
- **Kích thước file trọng số PyTorch (`.pt`):** 479.524 bytes (xấp xỉ $468.3\\text{ KB}$).
- **Kích thước mô hình ONNX (`.onnx`):** 485.498 bytes (xấp xỉ $474.1\\text{ KB}$).
- **Bộ nhớ hoạt động (RAM Footprint khi suy luận):** $< 5.0\\text{ MB}$.

---

## 2. Đặc tả Tham số Frontend Trích xuất Đặc trưng
Để đảm bảo tính nhất quán tuyệt đối giữa môi trường nghiên cứu và triển khai sản phẩm, cấu hình Frontend được cố định trong hợp đồng kỹ thuật:
- **Tần số lấy mẫu âm thanh ($f_s$):** $16.000\\text{ Hz}$.
- **Kích thước biến đổi Fourier ($N_{\\text{fft}}$):** $512$ điểm (chuẩn lũy thừa 2, tối ưu cho thuật toán FFT).
- **Độ dài cửa sổ phân tích ($W_{\\text{len}}$):** $400$ mẫu ($25.0\\text{ ms}$).
- **Bước nhảy cửa sổ ($H_{\\text{len}}$):** $160$ mẫu ($10.0\\text{ ms}$).
- **Loại cửa sổ:** Cửa sổ Hann (Hann window, đối xứng).
- **Số dải lọc Mel ($N_{\\text{mels}}$):** 40 dải.
- **Dải tần số trích xuất:** $f_{\\min} = 20.0\\text{ Hz}$ đến $f_{\\max} = 8.000.0\\text{ Hz}$.
- **Chuẩn hóa Mel Filterbank:** Slaney area normalization (diện tích các tam giác lọc được chuẩn hóa để bảo toàn năng lượng).
- **Log Compression:** $\\log_e(\\text{mel} + 10^{-6})$.

---

## 3. Hàm Mất Mát: Hard-Negative Weighted Cross Entropy
Trong giai đoạn Phase 11, một bước đột phá quan trọng giúp triệt tiêu báo động giả là việc thay thế hàm Cross Entropy thông thường bằng **Hàm Mất Mát Có Trọng Số Phạt Âm Vị Khó**:

$$\\mathcal{L}_{\\text{weighted}} = - \\frac{1}{B} \\sum_{i=1}^B w_{y_i} \\cdot \\log \\left( \\frac{\\exp(z_{i, y_i})}{\\sum_{j=0}^{4} \\exp(z_{i, j})} \\right)$$

Trong đó:
- $z_{i, j}$ là logit đầu ra của mô hình cho mẫu $i$ tại lớp $j$.
- $y_i \\in \\{0, 1, 2, 3, 4\\}$ tương ứng với các lớp: `[TRAI, PHAI, LEN, XUONG, REJECT]`.
- Véc-tơ trọng số lớp được tối ưu hóa:
  $$W = [w_0, w_1, w_2, w_3, w_4] = [1.0, 1.0, 1.0, 1.0, 1.5]$$
Trọng số $1.5$ dành riêng cho lớp từ chối `REJECT` buộc mô hình phải cảnh giác cao độ và chịu mức phạt nặng hơn nếu phân loại nhầm một từ gây nhiễu âm vị thành khẩu lệnh hợp lệ.

---

## 4. Thiết lập Huấn luyện & Kiểm soát Hạt Ngẫu nhiên (Seeds)
Chiến dịch huấn luyện Phase 11 được thực thi với sự kiểm soát tính tiền định (deterministic training) tuyệt đối:
- **Tập dữ liệu:** 3.840 clips Train, 960 clips Validation, 480 clips Locked Test.
- **Bộ tối ưu hóa (Optimizer):** AdamW với tốc độ học khởi tạo $\\eta = 10^{-3}$, suy giảm trọng số (weight decay) $\\lambda = 10^{-4}$.
- **Lịch trình học (Learning Rate Scheduler):** Cosine Annealing Learning Rate với $T_{\\max} = 40$ epochs, $\\eta_{\\min} = 10^{-5}$.
- **Kích thước lô (Batch Size):** 64.
- **Số epochs huấn luyện:** 40 epochs.
- **Đa hạt ngẫu nhiên (Multi-Seed Campaign):** Thực thi trên 3 hạt giống:
  1. **Seed 42 (Champion):** Đạt Test FAR $0.63\\%$, Test Recall $99.17\\%$.
  2. **Seed 43:** Đạt Test FAR $1.04\\%$, Test Recall $98.33\\%$.
  3. **Seed 44:** Đạt Test FAR $1.67\\%$, Test Recall $97.50\\%$.

Trọng số được lưu lại từ epoch có điểm F1-score cao nhất trên tập Validation độc lập của Seed 42, sinh ra tệp checkpoint ứng viên duy nhất: `variant_v11_hard_negative_weighted_seed42.pt`.
"""

with open(os.path.join(DOCS_DIR, "02_CHRONOLOGICAL_RESEARCH_HISTORY.md"), "w", encoding="utf-8") as f:
    f.write(ch02_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "03_DATASET_PROVENANCE_AND_SPLITS.md"), "w", encoding="utf-8") as f:
    f.write(ch03_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "04_MODEL_ARCHITECTURES_AND_TRAINING.md"), "w", encoding="utf-8") as f:
    f.write(ch04_content.strip() + "\n")

print("[OK] Generated 02_CHRONOLOGICAL_RESEARCH_HISTORY.md, 03_DATASET_PROVENANCE_AND_SPLITS.md, and 04_MODEL_ARCHITECTURES_AND_TRAINING.md")
