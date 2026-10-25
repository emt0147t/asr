import os

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")

ch11_content = """# Chương 11: Kỹ thuật Triển khai & Giới hạn (Deployment Engineering & Limitations)

## 1. Lịch sử Triển khai Biên: Phase 91 và Phase 92
Nỗ lực đưa mô hình KWS ra môi trường biên bắt đầu từ giai đoạn Phase 91 với mục tiêu xây dựng một đường ống (pipeline) hoàn chỉnh từ Python sang C++ và các nền tảng di động:

```mermaid
flowchart LR
    PyTorch["PyTorch (Python)<br/>variant_v11.pt"] --> ONNX_Export["Xuất ONNX Native<br/>opset_version=14"]
    ONNX_Export --> ONNX_Model["vietnamese_kws_v11.onnx<br/>(485 KB)"]
    ONNX_Model --> ORT["ONNX Runtime Engine<br/>(C++ / Mobile)"]
    
    AudioIn["Microphone / WAV (16 kHz)"] --> C_Frontend["Frontend C++ (Mel Spectrogram)<br/>RÀO CẢN: Bất đồng N_fft & Filterbank"]
    C_Frontend --> ORT
    ORT --> PostProc["Hậu xử lý (Threshold theta=0.20)"]
    PostProc --> Decision["Quyết định: TRAI / PHAI / LEN / XUONG / REJECT"]
```

### 1.1. Thử nghiệm Triển khai Phase 91 (Mobile Scaffolds)
Trong Phase 91, các khung mã nguồn di động sơ khởi đã được thiết lập:
- **Android:** Khung ứng dụng Java/Kotlin đặt tại `mobile/android/`, sử dụng thư viện `onnxruntime-android` để nạp tệp mô hình và thực hiện suy luận.
- **iOS:** Khung ứng dụng Swift/Objective-C tại `mobile/ios/`, tích hợp ONNX Runtime qua CocoaPods.
- **Hiện trạng:** Cả hai khung mã nguồn này mới chỉ dừng lại ở mức **bộ khung nguyên mẫu (prototype scaffolds)**. Chúng chưa được biên dịch và kiểm thử đo đạc thực tế trên phần cứng vật lý (chưa có thiết bị Android/iOS thật trong phòng thí nghiệm để kiểm tra độ trễ âm thanh đầu vào AudioRecord / AVAudioEngine).

### 1.2. Thẩm tra Đẳng cấu Đa nền tảng Phase 92 (Cross-Platform Parity Audit)
Trong Phase 92, một cuộc đối chuẩn toàn diện giữa môi trường PyTorch Python và ONNX Runtime C++ đã phát hiện ra các rào cản kỹ thuật:
- **Rào cản C++ Mel Filterbank:** Thư viện tính Mel trong C++ tự viết có sai số làm tròn số học và công thức tính trọng số tam giác Mel khác biệt nhẹ so với hàm chuẩn `torchaudio.transforms.MelSpectrogram`. Điều này khiến ma trận Mel đầu vào trong C++ bị lệch khoảng $2\\% - 5\\%$ so với Python.
- **Rào cản FFT Size:** Báo cáo Phase 92 mô tả $N_{\\text{fft}} = 400$ do kế thừa từ baseline `variant_w10`. Khi chuyển sang `variant_v11`, toàn bộ pipeline bắt buộc phải chuẩn hóa về $N_{\\text{fft}} = 512$ để tương thích với các thuật toán FFT tối ưu hóa trên vi xử lý ARM/NEON.

---

## 2. Bản Đóng Gói ONNX Hoàn Thiện Trong Phase 13.4
Trong [Phase 13.4](file:///d:/trainidentificationaudio/reports/PHASE13_4_FINAL_CNN_RELEASE.md), quy trình xuất ONNX đã được chuẩn hóa tuyệt đối:
- **Đường dẫn tệp:** [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx)
- **Mã băm SHA-256:** `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`
- **Kích thước:** 485.498 bytes.
- **Opset Version:** 14 (tương thích rộng rãi từ ONNX Runtime v1.12 trở lên).
- **Kiểm định đẳng cấu số học:**
  $$\\Delta_{\\text{max}} = \\max |z^{\\text{PyTorch}} - z^{\\text{ONNX}}| = 3.58 \\times 10^{-6} \\quad (< 10^{-4})$$
  Đảm bảo $100.00\\%$ mẫu thử nghiệm cho kết quả nhãn trùng khớp hoàn toàn.

---

## 3. Đặc tả Yêu cầu Tài nguyên Phần cứng (Hardware Budget)
- **Tốc độ xử lý (CPU Latency):**
  - Intel Core i7 / AMD Ryzen x86: Mean = 1.39 ms, P95 = 2.27 ms.
  - ARM Cortex-A53 / Cortex-A72 (Raspberry Pi 4 / Smartphone): Dự kiến 4.5 ms - 8.0 ms (thỏa mãn ngân sách $< 50.0\\text{ ms}$).
- **Bộ nhớ (RAM Footprint):**
  - Dung lượng tĩnh của mô hình ONNX: $\\approx 0.48\\text{ MB}$.
  - Bộ đệm suy luận (Working Tensor Buffer): $\\approx 2.5\\text{ MB}$.
  - Tổng dung lượng RAM chiếm dụng: $< 5.0\\text{ MB}$.

---

## 4. Các Giới hạn Triển khai Hiện hữu (Open Deployment Limitations)
Kỹ sư hệ thống cần lưu ý các khoảng trống kỹ thuật chưa được hoàn thiện:
1. **Chưa có Bộ Phát hiện Tiếng nói C++ (Missing C++ VAD Engine):** Mô hình ONNX chỉ nhận đầu vào là ma trận Log-Mel hoàn chỉnh $[1, 1, 40, 101]$. Chưa có module C++ để tự động phát hiện giọng nói và cắt khung $1.0\\text{ giây}$ từ luồng micro thời gian thực.
2. **Chưa thẩm định Lượng tử hóa Phần cứng (No INT8 / FP16 Quantization):** Bản phát hành hiện tại ở định dạng số thực 32-bit (FP32). Chưa có bài kiểm thử lượng tử hóa INT8 có hiệu chuẩn (PTQ/QAT) để chạy trên NPU/DSP.
3. **Mã khung mẫu di động chưa kiểm thử thực địa:** Các thư mục `mobile/android` và `mobile/ios` chỉ mang tính chất minh họa kiến trúc gọi API, chưa sẵn sàng để đưa thẳng vào ứng dụng thương mại mà không qua bước tích hợp sâu.
"""

ch12_content = """# Chương 12: Khả năng Tái lập & Độ phủ Kiểm thử (Reproducibility & Test Coverage)

## 1. Môi trường Thực thi Chuẩn (Authoritative Execution Environment)
Để tái lập chính xác $100\\%$ các kết quả nghiên cứu và kiểm định, môi trường chạy thử nghiệm cần đáp ứng các thông số kỹ thuật sau:
- **Hệ điều hành:** Microsoft Windows 10/11 Pro (hoặc Linux Ubuntu 22.04 LTS).
- **Trình thông dịch:** Python 3.10.x hoặc Python 3.11.x (môi trường ảo `.venv`).
- **Các thư viện phụ thuộc cốt lõi:**
  - `torch == 2.6.0+cpu`
  - `torchaudio == 2.6.0+cpu`
  - `onnx == 1.17.0`
  - `onnxruntime == 1.20.1`
  - `numpy == 1.26.4`
  - `scipy == 1.14.1`
  - `pytest == 8.3.4`

---

## 2. Hướng Dẫn Các Lệnh Tái Lập Từng Bước (Step-by-Step Reproduction)

### Bước 1: Xác thực Toàn vẹn Mã Băm Checkpoint
Kiểm tra mã băm SHA-256 của file trọng số đông kết:
```powershell
# PowerShell trên Windows
Get-FileHash -Algorithm SHA256 experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt
```
*Kết quả bắt buộc:* `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`

### Bước 2: Chạy Toàn Bộ Bộ Kiểm Thử Tự Động (Workspace Test Suite)
Kho mã nguồn hiện có **384 bài kiểm thử tự động**, bao phủ toàn diện từ toán học Mel Spectrogram, kiểm tra chống rò rỉ dữ liệu, kiểm tra đẳng cấu ONNX, đến quy trình đánh giá mù:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```
*Kết quả kỳ vọng:* **384 passed in ~15-25s** (Tỷ lệ đạt $100.00\\%$).

### Bước 3: Tái lập Bài Đánh giá Xác nhận Độc lập Phase 13.3
Chạy trực tiếp tệp kiểm thử chuyên dụng của Phase 13.3:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_3.py -v
```
Kiểm thử này sẽ:
1. Đọc lại 1.000 clips âm thanh tổng hợp từ `experiments/vieneu_kws_research_v13_3/audio/`.
2. Nạp checkpoint ứng viên đông kết với ngưỡng $\\theta_{\\text{cmd}} = 0.20$.
3. Kiểm tra tính trùng khớp của 8 cổng nghiệm thu (Command Recall $99.00\\%$, FAR $0.75\\%$, Accuracy $99.20\\%$).

### Bước 4: Kiểm Tra Tính Đẳng Cấu PyTorch vs ONNX Runtime
Chạy bài kiểm thử xuất mô hình và kiểm tra sai số số học:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_4.py -v
```
Kiểm thử này đảm bảo độ lệch logit cực đại $\\Delta_{\\text{max}} < 1.0 \\times 10^{-4}$ và nhãn dự đoán trùng khớp $100\\%$.

---

## 3. Tổng Quan Danh Mục Bộ Kiểm Thử Tự Động (Test Suite Inventory)
Bảng dưới đây thống kê các mô-đun kiểm thử quan trọng trong thư mục [`tests/`](file:///d:/trainidentificationaudio/tests/):

| Tệp Kiểm Thử (Test Module) | Số Lượng Tests | Phạm Vi và Mục Tiêu Kiểm Tra |
| :--- | :--- | :--- |
| [`tests/test_vieneu_research_v13_4.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_4.py) | 8 | Đóng gói phát hành Phase 13.4, kiểm tra đẳng cấu ONNX, tính bất biến của SHA-256. |
| [`tests/test_vieneu_research_v13_3.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_3.py) | 12 | Thẩm định 8 cổng nghiệm thu Phase 13.3 trên 1.000 clips âm thanh độc lập. |
| [`tests/test_vieneu_research_v13_2_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_2_1.py) | 6 | Kiểm định giải quyết bất đồng $N_{\\text{fft}} = 400$ vs $512$ và Mel Filterbank. |
| [`tests/test_vieneu_research_v13_1.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v13_1.py) | 10 | Khắc phục lỗi khuếch đại tín hiệu im lặng, xác minh Conditional Clamping. |
| [`tests/test_vieneu_research_v11.py`](file:///d:/trainidentificationaudio/tests/test_vieneu_research_v11.py) | 16 | Huấn luyện Phase 11, hàm mất mát Hard-Negative Weighted Cross Entropy, multi-seed. |
| [`tests/test_cross_platform_parity_phase92.py`](file:///d:/trainidentificationaudio/tests/test_cross_platform_parity_phase92.py) | 14 | Đối chuẩn chéo PyTorch - ONNX, đo đạc độ trễ phân vị P50/P95. |
| [`tests/test_phase90_human_listening.py`](file:///d:/trainidentificationaudio/tests/test_phase90_human_listening.py) | 8 | Kiểm toán dữ liệu đánh giá con người Phase 90, phát hiện thiếu 5 người chấm. |
| Các bài kiểm thử phân hệ khác | 310 | Kiểm tra tiền xử lý âm thanh, sinh dữ liệu Vieneu, phân chia tập dữ liệu không rò rỉ. |
| **TỔNG CỘNG WORKSPACE** | **384** | **100% PASS RATE (Tất cả đều vượt qua).** |
"""

ch13_content = """# Chương 13: Hiện trạng Cuối cùng & Lộ trình Đề xuất (Final Status & Recommended Next Steps)

## 1. Ma Trận Thẩm Định Hiện Trạng Toàn Diện 10 Chiều (Validation Status Matrix)
Bảng dưới đây tổng hợp hiện trạng thực chứng trên 10 khía cạnh kỹ thuật cốt lõi của dự án (trích xuất từ [`tables/validation_status.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/validation_status.csv)):

| STT | Khía Cạnh Kỹ Thuật (Dimension) | Hiện Trạng Pháp Lý / Kỹ Thuật (Canonical Label) | Tóm Tắt Bằng Chứng Thực Chứng (Evidence Summary) |
| :--- | :--- | :--- | :--- |
| **1** | Bounded-Utterance Accuracy (Giọng Bắc Tổng hợp) | `VALIDATED_HIGH_CONFIDENCE` | Đạt **$99.20\\%$** Accuracy trên 1.000 clips độc lập của 8 giọng Bắc mới lạ (Phase 13.3). |
| **2** | Nhầm lẫn Giữa các Lệnh (Command Confusion) | `VALIDATED_HIGH_CONFIDENCE` | Đạt **$0.00\\%$** tỷ lệ nhầm lẫn chéo giữa 4 lệnh mục tiêu trên 200 clips lệnh. |
| **3** | Tính Đẳng Cấu PyTorch vs ONNX Runtime | `VALIDATED_HIGH_CONFIDENCE` | Độ lệch logit cực đại $\\Delta_{\\text{max}} = 3.58 \\times 10^{-6} < 10^{-4}$; đồng thuận nhãn $100\\%$. |
| **4** | Độ Trễ Suy Luận & Bộ Nhớ (CPU Latency) | `VALIDATED_HIGH_CONFIDENCE` | Mean Latency = $1.80\\text{ ms}$ (CPU x86), P95 = $2.12\\text{ ms}$; dung lượng RAM $< 5.0\\text{ MB}$. |
| **5** | Khả Năng Từ Chối Tạp Âm & Âm Vị Khó | `VALIDATED_HIGH_CONFIDENCE` | Tỷ lệ báo động giả FAR đạt **$0.75\\%$** (6/800 clips) tại ngưỡng tối ưu $\\theta_{\\text{cmd}} = 0.20$. |
| **6** | Giọng Địa Phương Miền Trung & Miền Nam | `NOT_VALIDATED_PENDING_DATA` | Chưa có dữ liệu giọng tổng hợp hoặc người thật độc lập; tạm hoãn theo Sửa đổi 13.1-A. |
| **7** | Giọng Nói Người Thật Tự Nhiên (Natural Speech) | `NOT_VALIDATED_ON_NATURAL_SPEECH` | Chưa có tập dữ liệu giọng người thật có bản quyền/đồng thuận pháp lý; Phase 90 chỉ có 1 người chấm. |
| **8** | Nhận Dạng Micro Mở Liên Tục (Streaming) | `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` | Chưa tích hợp bộ đệm vòng (ring buffer) và VAD lọc nhiễu mép cửa sổ trượt. |
| **9** | Hiệu Năng Trên Phần Cứng Di Động Thực Tế | `UNTESTED_PROTOTYPE_ONLY` | Khung mã nguồn Android/iOS Phase 91 mới dừng lại ở mức khung mô phỏng phần mềm. |
| **10** | Tính Đồng Thuận Đa Đánh Giá Viên (Human Study) | `INVALIDATED_INCOMPLETE_STUDY` | Phase 90 chỉ có Reviewer 01 hoàn thành (120 clips); 5 người bỏ cuộc; tệp tổng hợp rỗng. |

---

## 2. Lộ Trình Kỹ Thuật Đề Xuất Ưu Tiên (Prioritized Technical Roadmap)
Để đưa mô hình từ trạng thái ứng viên phòng thí nghiệm (`FROZEN_CNN_RELEASE_CANDIDATE`) thành sản phẩm thương mại hoàn chỉnh, dự án cần thực thi theo 4 bước ưu tiên sau:

```mermaid
flowchart TD
    Step1["Giai đoạn 1: Chiến Dịch Thu Âm Người Thật Có Đồng Thuận<br/>(Field Recording Campaign)<br/>- 50+ người nói đa dạng lứa tuổi, giới tính<br/>- Phân bổ 3 miền: Bắc (40%), Trung (30%), Nam (30%)<br/>- Ký cam kết đồng thuận GDPR/luật dữ liệu"]
    
    Step2["Giai đoạn 2: Phát Triển Engine Streaming VAD<br/>(Continuous Streaming Engine)<br/>- Triển khai C++ Ring Buffer hop 160ms<br/>- Tích hợp Silero-VAD hoặc WebRTC-VAD siêu nhẹ<br/>- Thuật toán Non-Maximum Suppression khử kích hoạt lặp"]
    
    Step3["Giai đoạn 3: Tích Hợp & Đo Đạc Trên Thiết Bị Di Động Thật<br/>(Edge Hardware Verification)<br/>- Biên dịch thư viện C++ qua Android NDK và iOS Framework<br/>- Đo đạc độ trễ luồng âm thanh thực tế từ AudioRecord<br/>- Đo lường mức tiêu hao pin và nhiệt độ thiết bị"]
    
    Step4["Giai đoạn 4: Lượng Tử Hóa Mô Hình (INT8 Quantization)<br/>(Quantization & Acceleration)<br/>- Lượng tử hóa sau huấn luyện (PTQ) trên ONNX Runtime<br/>- Kiểm tra độ suy giảm Recall/FAR sau khi lượng tử hóa<br/>- Tối ưu hóa tập lệnh ARM NEON / NPU"]
    
    Step1 --> Step2 --> Step3 --> Step4
```

### Chi tiết Các Nhiệm vụ Cụ thể:
1. **Nhiệm vụ 1: Thu thập Dữ liệu Giọng Người Thật Chuẩn Pháp Lý:**
   - Xây dựng quy trình thu âm có sự chấp thuận rõ ràng của người tham gia (Informed Consent Records).
   - Thu âm trong nhiều môi trường: phòng yên tĩnh, văn phòng có tiếng gõ phím/nói chuyện, quán cà phê, và môi trường xe di chuyển.
2. **Nhiệm vụ 2: Thiết kế Module Cửa Sổ Trượt Bền Vững (Sliding Window Engine):**
   - Thiết kế cơ chế lọc kích hoạt: Khi phát hiện lệnh với $P(\\text{cmd}) \\ge 0.20$, khóa kích hoạt trong vòng $500\\text{ ms}$ tiếp theo (debounce window) để tránh việc cùng một từ bị nhận diện $2 - 3$ lần khi cửa sổ dịch chuyển qua.
"""

ch14_content = """# Chương 14: Khoảng trống Bằng chứng & Bất đồng (Evidence Gaps & Discrepancies)

## 1. Sổ Đăng Ký 6 Điểm Bất Đồng Kỹ Thuật Trong Lịch Sử Dự Án
Trong quá trình kiểm toán toàn bộ kho mã nguồn và 114 báo cáo kỹ thuật, 6 sai lệch/bất đồng kỹ thuật lớn đã được phát hiện, phân tích nguyên nhân gốc rễ và xử lý dứt điểm (trích xuất từ [`tables/unresolved_discrepancies.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/unresolved_discrepancies.csv)):

---

### Bất đồng 1: Sai lệch Kích thước FFT ($N_{\\text{fft}} = 400$ vs $512$) Giữa Phase 92 và Phase 11
- **Vấn đề ghi nhận:** Báo cáo [Phase 92](file:///d:/trainidentificationaudio/reports/phase92_master_summary.md) mô tả frontend trích xuất Log-Mel sử dụng $N_{\\text{fft}} = 400$ và bộ lọc Slaney kích thước $40 \\times 201$. Tuy nhiên, quy trình chuẩn bị Phase 13.2 lại ghi nhận $N_{\\text{fft}} = 512$, $W_{\\text{len}} = 400$, $H_{\\text{len}} = 160$ và kích thước ma trận Mel là $40 \\times 257$.
- **Nguyên nhân gốc rễ (Root Cause):** Phase 92 thực hiện đo đạc đối chuẩn cho mô hình baseline lịch sử `variant_w10` (huấn luyện từ Phase 9/10 với $N_{\\text{fft}} = 400$). Khi bước sang chiến dịch hoàn thiện Phase 11, pipeline huấn luyện `train_v11.py` đã chủ động nâng cấp lên $N_{\\text{fft}} = 512$ (lũy thừa 2) để tương thích tối ưu với thuật toán FFT trên vi xử lý di động. Báo cáo Phase 92 mang tính lịch sử phản ánh thế hệ cũ, không áp dụng cho thế hệ v11.
- **Biện pháp giải quyết (Resolution):** Thẩm tra mã nguồn thực thi [Phase 13.2.1](file:///d:/trainidentificationaudio/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md) xác nhận $100\\%$ mã nguồn từ `train_v11.py`, `infer.py` đến file cấu hình `preprocessing_contract.json` của checkpoint đông kết đều sử dụng **$N_{\\text{fft}} = 512$**.

---

### Bất đồng 2: Lỗi Chuẩn Hóa Khuếch Đại Tín Hiệu Im Lặng Trong Phase 12
- **Vấn đề ghi nhận:** Tệp `infer.py` trong bundle Phase 12 tự ý áp dụng chuẩn hóa đỉnh biên độ $-1.0\\text{ dBFS}$ vô điều kiện. Khi gặp đoạn âm thanh im lặng (nhiễu phòng với biên độ cực đại $0.005$), hàm này khuếch đại tín hiệu lên $+45\\text{ dB}$, biến nhiễu nền thành xung âm thanh lớn và kích hoạt nhầm thành lệnh `TRAI`.
- **Nguyên nhân gốc rễ:** Lập trình viên viết mã suy luận Phase 12 đã hiểu sai yêu cầu "chuẩn hóa âm thanh" và tự ý đưa logic căn đỉnh vào mà không đối soát với hàm tiền xử lý của tập huấn luyện `train_v11.py`.
- **Biện pháp giải quyết:** Đợt đánh giá Phase 13 đã kích hoạt dừng khẩn cấp (**BLOCKED**). Giai đoạn Phase 13.1 đã khôi phục đúng logic chuẩn của `train_v11.py`: **Chỉ co tỷ lệ khi $\\max(|x|) > 1.0$**, giữ nguyên năng lượng tự nhiên của tín hiệu im lặng.

---

### Bất đồng 3: Sự Cố Nghiên Cứu Lắng Nghe Mù Con Người Phase 90
- **Vấn đề ghi nhận:** Báo cáo Phase 90 lên kế hoạch thử nghiệm nghe mù với 6 người đánh giá độc lập trên 120 clips. Tuy nhiên, tệp kết quả tổng hợp `validated_responses.csv` hoàn toàn bị rỗng và không thể tính toán chỉ số đồng thuận liên đánh giá.
- **Nguyên nhân gốc rễ:** Chỉ có duy nhất `reviewer_01` hoàn thành 120 câu hỏi; 5 người còn lại bỏ dở và không nộp tệp kết quả. Quy trình tổng hợp tự động bị treo do thiếu dữ liệu đầu vào.
- **Biện pháp giải quyết:** Ghi nhận trung thực bằng chứng: Phase 90 chỉ có $N=1$ người hoàn thành (điểm rõ ràng $4.88/5.0$). Không thực hiện điền khống dữ liệu. Duy trì nhãn chính thức `NOT_VALIDATED_ON_NATURAL_SPEECH`.

---

### Bất đồng 4: Thiên Kiến Chọn Lọc Mô Hình Champion Trong Phase 11
- **Vấn đề ghi nhận:** Trong chiến dịch Phase 11, mô hình `seed 42` được chọn làm Champion sau khi người nghiên cứu quan sát thấy nó đạt FAR thấp nhất trên tập Locked Test Split ($0.63\\%$ so với $1.04\\%$ của seed 43 và $1.67\\%$ của seed 44). Điều này vi phạm nguyên tắc cách ly tập kiểm thử khóa (Test Set Peeking Bias).
- **Nguyên nhân gốc rễ:** Áp lực muốn tìm kiếm mô hình có tỷ lệ báo động giả thấp nhất cho giai đoạn tiếp theo đã dẫn đến việc kiểm tra kết quả test set trước khi chốt checkpoint.
- **Biện pháp giải quyết:** Thừa nhận minh bạch thiên kiến chọn lựa trong báo cáo Phase 11 và Phase 12. Triệt tiêu hoàn toàn thiên kiến này bằng cách tổ chức **Đánh giá Xác nhận Độc lập Phase 13.3** trên 1.000 clips âm thanh tổng hợp mới tinh từ 8 giọng đọc chưa từng tiếp xúc. Checkpoint Seed 42 đã chứng minh năng lực thực sự khi đạt $99.20\\%$ Accuracy và $0.75\\%$ FAR.

---

### Bất đồng 5: Thiên Kiến Phương Ngữ Trong Tập Dữ Liệu Tổng Hợp (Accent Scope Bias)
- **Vấn đề ghi nhận:** Bộ dữ liệu tổng hợp Vieneu TTS v3-Turbo chủ yếu bao gồm các giọng đọc phát âm chuẩn theo phương ngữ Miền Bắc. Các biến thể phương ngữ Miền Trung (ngữ điệu nặng, biến đổi thanh điệu hỏi/ngã) và Miền Nam (biến đổi phụ âm cuối /n/ thành /ŋ/, /t/ thành /k/) chưa được đại diện đầy đủ.
- **Nguyên nhân gốc rễ:** Thiếu hụt các mô hình giọng đọc tổng hợp chất lượng cao cho phương ngữ Trung và Nam tại thời điểm nghiên cứu.
- **Biện pháp giải quyết:** Ban hành Giao thức Sửa đổi 13.1-A (Amendment 13.1-A), thu hẹp phạm vi tuyên bố: mô hình đông kết chỉ được xác nhận khả năng tổng quát hóa trên phương ngữ Miền Bắc; trạng thái của phương ngữ Trung và Nam được gắn nhãn `NOT_VALIDATED_PENDING_DATA`.

---

### Bất đồng 6: Nhầm Lẫn Giữa Kiểm Thử Đóng Khung (Bounded) và Luồng Micro Thực Tế (Streaming)
- **Vấn đề ghi nhận:** Một số tài liệu sơ bộ ban đầu mô tả mô hình "sẵn sàng nhận diện giọng nói liên tục", gây hiểu lầm rằng hệ thống có thể thả vào micro mở để nhận lệnh ngay lập tức.
- **Nguyên nhân gốc rễ:** Không phân định rõ ràng giữa bài toán phân loại clip âm thanh cắt sẵn ($1.0\\text{ giây}$) và bài toán nhận diện luồng âm thanh liên tục có tiếng ồn mép cửa sổ.
- **Biện pháp giải quyết:** Chuẩn hóa toàn bộ hệ thống tài liệu với nhãn chính thức `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` cho chế độ micro mở, đồng thời vạch rõ lộ trình tích hợp VAD và bộ đệm vòng trong tương lai.
"""

with open(os.path.join(DOCS_DIR, "11_DEPLOYMENT_ENGINEERING_AND_LIMITATIONS.md"), "w", encoding="utf-8") as f:
    f.write(ch11_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "12_REPRODUCIBILITY_AND_TEST_COVERAGE.md"), "w", encoding="utf-8") as f:
    f.write(ch12_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "13_FINAL_STATUS_AND_RECOMMENDED_NEXT_STEPS.md"), "w", encoding="utf-8") as f:
    f.write(ch13_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "14_EVIDENCE_GAPS_AND_DISCREPANCIES.md"), "w", encoding="utf-8") as f:
    f.write(ch14_content.strip() + "\n")

print("[OK] Generated 11, 12, 13, and 14 markdown files.")
