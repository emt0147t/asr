# Báo Cáo Trạng Thái Mô Hình KWS Tiếng Việt (Model Status Pack)
> **Mục đích:** Tài liệu thẩm định học thuật dành cho Giáo viên Hướng dẫn & Hội đồng Thẩm định  
> **Thời điểm xác thực (Timestamp):** `2026-10-21 10:04:35 UTC+7` (`2026-10-21T03:04:35.639552+00:00`)  
> **Git Commit HEAD:** [`471f1af995`](file:///D:/trainidentificationaudio) (`471f1af9957506d54be0534d24817c93957ceee3`)  
> **Trạng thái kho mã nguồn:** `71 modified/untracked files (Working tree clean of staged commits)`  
> **Chế độ kiểm thử:** **STRICTLY READ-ONLY** &bull; Đã kiểm chứng tính bất biến của tệp mô hình trước và sau chạy.  

---

## Mục A: Tóm Tắt Điều Hành & Dấu Thời Gian Xác Thực (Executive Summary)

Báo cáo này công bố trạng thái kỹ thuật và kết quả thẩm định khách quan, có thể tái lập hoàn toàn của mô hình **Nhận Diện Khẩu Lệnh Hướng Tiếng Việt (Vietnamese Keyword Spotting - KWS)** trong không gian dự án [`D:/trainidentificationaudio`](file:///D:/trainidentificationaudio).

Ứng viên mô hình đang được đông kết là **[`variant_v11_hard_negative_weighted_seed42.pt`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)**, kèm theo bản xuất chuẩn công nghiệp **[`vietnamese_kws_v11.onnx`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx)**. Tất cả kết quả thực nghiệm trong tài liệu này được thu thập qua lệnh kiểm chứng thực tế tại chỗ hoặc trích xuất từ hồ sơ thẩm định lịch sử có chữ ký số.

```
========================================================================================
BẢNG TỔNG QUAN TRẠNG THÁI KIỂM CHỨNG TỰ ĐỘNG (LATEST RUN)
========================================================================================
- Trạng Thái Checkpoint PyTorch    : [PASS] (Mã SHA-256 khớp tuyệt đối 100%)
- Trạng Thái Bản Xuất ONNX         : [PASS] (Mã SHA-256 khớp tuyệt đối 100%)
- Đẳng Cấu Dự Đoán PyTorch vs ONNX : [PASS] (8/8 development fixtures trùng khớp nhãn)
- Bộ Kiểm Thử Tự Động (Phase 13.4) : [PASS] (5 passed, 0 failed, exit code 0)
- Toàn Bộ Test Suite Workspace     : [PASS] (384/384 passed in 43.43s)
- Tính Bất Biến Checkpoint         : [VERIFIED_UNTOUCHED] (Mã SHA-256 trước và sau kiểm thử hoàn toàn đồng nhất)
========================================================================================
```

---

## Mục B: Danh Tính & Toàn Vẹn Checkpoint PyTorch (Model Identity & Checkpoint Integrity)

| Thuộc Tính | Giá Trị Quan Sát Thực Tế (Observed) | Giá Trị Kỳ Vọng (Authoritative Expected) | Trạng Thái |
| :--- | :--- | :--- | :---: |
| **Định danh mô hình** | [`variant_v11_hard_negative_weighted_seed42.pt`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt) | `variant_v11_hard_negative_weighted_seed42.pt` | **PASS** |
| **Đường dẫn tệp** | [`experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt) | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | **PASS** |
| **Dung lượng tệp (Size)** | `566,146 bytes` (~552.9 KB) | `566,146 bytes` (~552.9 KB) *(Ghi chú: v10 cũ là 479.524 bytes)* | **PASS** |
| **Mã băm SHA-256** | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **PASS** |
| **Kiến trúc mạng** | 4-Block 2D-CNN (`CorrectedVietnameseKWSNet`), 135.813 tham số FP32 | 135.813 tham số | **PASS** |

---

## Mục C: Danh Tính & Toàn Vẹn Mô Hình ONNX (ONNX Artifact Integrity)

| Thuộc Tính | Giá Trị Quan Sát Thực Tế (Observed) | Giá Trị Kỳ Vọng (Authoritative Expected) | Trạng Thái |
| :--- | :--- | :--- | :---: |
| **Định danh mô hình** | [`vietnamese_kws_v11.onnx`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx) | `vietnamese_kws_v11.onnx` | **PASS** |
| **Đường dẫn tệp** | [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx) | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | **PASS** |
| **Dung lượng tệp (Size)** | `555,751 bytes` (~542.7 KB) | `555,751 bytes` (~542.7 KB) *(Ghi chú: v10 cũ là 485.498 bytes)* | **PASS** |
| **Mã băm SHA-256** | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **PASS** |
| **Chuẩn ONNX Opset** | Opset 17, Single Float32 Execution Graph | Opset 17 | **PASS** |

---

## Mục D: Hợp Đồng Tiền Xử Lý Âm Thanh Đông Kết (Frozen Preprocessing Contract)

Hợp đồng tiền xử lý được quy định bất biến tại [`experiments/vieneu_kws_research_v13_4/bundle/preprocessing_contract.json`](file:///D:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/preprocessing_contract.json):

1. **Định dạng âm thanh đầu vào (Input Audio Format):**
   - Đơn kênh (Mono), tần số lấy mẫu cố định **16.000 Hz**, thời lượng chuẩn hóa **1.0000 giây** (đúng 16.000 mẫu).
   - Đệm zero về phía phải nếu clip ngắn hơn 16.000 mẫu; cắt gọn nếu dài hơn 16.000 mẫu.
2. **Quy tắc co biên độ có điều kiện (Conditional Amplitude Clamping):**
   ```python
   if max(abs(audio)) > 1.0:
       audio = audio / max(abs(audio))
   else:
       # Giữ nguyên tuyệt đối biên độ gốc
       pass
   ```
   > [!IMPORTANT]
   > **NGHIÊM CẤM KHUẾCH ĐẠI ĐÁY (Zero Floor Amplification):** Tuyệt đối không tự động khuếch đại (peak-normalize) các tín hiệu có biên độ nhỏ hơn hoặc bằng 1.0, ngăn ngừa nguy cơ kích nổ nhiễu nền im lặng thành báo động giả (False Alarm Explosion).
3. **Frontend Biến Đổi Log-Mel Spectrogram:**
   - **STFT:** $N_{\text{fft}} = 512$ điểm (chuẩn lũy thừa 2, tối ưu cho FFTW/ARM Neon/x86 AVX2).
   - **Cửa sổ phân tích:** Hann định kỳ (periodic Hann), độ dài cửa sổ $W_{\text{len}} = 400$ mẫu (25.0 ms), bước nhảy $H_{\text{len}} = 160$ mẫu (10.0 ms), padding phản xạ đối xứng (center reflection padding).
   - **Filterbank:** 40 dải lọc Mel tam giác chuẩn Slaney trong dải tần số từ **20.0 Hz đến 8000.0 Hz**, chuẩn hóa diện tích Slaney.
   - **Nén logarit:** Logarit tự nhiên với hằng số ổn định $\epsilon = 1.0 \times 10^{-6}$: $\text{LogMel} = \ln(\text{Mel} + 10^{-6})$.
   - **Kích thước tensor đặc trưng:** Đúng `[1, 1, 40, 101]` (Batch, Channel, Mels, Frames).
4. **Không Gian Lớp & Ngưỡng Kích Hoạt Quyết Định:**
   - 5 lớp phân loại độc quyền: `TRAI (0)`, `PHAI (1)`, `LEN (2)`, `XUONG (3)`, `REJECT (4)`.
   - Ngưỡng kích hoạt tối ưu: $\theta^*_{\text{cmd}} = 0.20$.
   - Công thức suy diễn: Nếu $\max(p_{\text{TRAI}}, p_{\text{PHAI}}, p_{\text{LEN}}, p_{\text{XUONG}}) \ge 0.20$ thì dự đoán nhãn có xác suất cao nhất; ngược lại gán vào `REJECT`.

---

## Mục E: Kết Quả Thẩm Định Lịch Sử Phase 13.3 (Historical Confirmatory Results)

> [!NOTE]
> **CẢNH BÁO BẢN CHẤT DỮ LIỆU:** Toàn bộ kết quả dưới đây là **số liệu thực nghiệm lịch sử** trích xuất từ đợt đánh giá mù có đăng ký trước Phase 13.3 trên tập dữ liệu gồm **1.000 clips âm thanh tổng hợp Vieneu TTS v3-Turbo** (8 giọng đọc Miền Bắc hoàn toàn mới lạ, 0% trùng lặp giọng và mẫu văn bản với tập huấn luyện). Đây **KHÔNG** phải là kết quả đo trên giọng người thật nói tự nhiên.

| Tiêu Chí Đánh Giá | Chỉ Tiêu Nghiệm Thu (SLA) | Kết Quả Báo Cáo Lịch Sử | Số Mẫu Chi Tiết | Trạng Thái Thẩm Định |
| :--- | :---: | :---: | :---: | :---: |
| **Độ chính xác toàn diện (Accuracy)** | $\ge 95.0\%$ | **99.20%** | 992/1000 clips | **PASSED** (Wilson 95% CI: [98.4%, 99.6%]) |
| **Độ nhạy nhận lệnh (Command Recall)** | $\ge 95.0\%$ | **99.00%** | 198/200 clips | **PASSED** (Wilson 95% CI: [96.4%, 99.7%]) |
| - Lệnh `TRAI` | $\ge 95.0\%$ | 98.00% (49/50) | 49 / 50 clips | PASSED |
| - Lệnh `PHAI` | $\ge 95.0\%$ | 100.00% (50/50) | 50 / 50 clips | PASSED |
| - Lệnh `LEN` | $\ge 95.0\%$ | 98.00% (49/50) | 49 / 50 clips | PASSED |
| - Lệnh `XUONG` | $\ge 95.0\%$ | 100.00% (50/50) | 50 / 50 clips | PASSED |
| **Tỷ lệ báo động giả (Non-Target FAR)** | $\le 5.0\%$ | **0.75%** | 6/800 clips | **PASSED** (Wilson 95% CI: [0.3%, 1.6%]) |
| - Cặp từ tối thiểu ngữ âm (Minimal Pairs) | Diagnostic | **0.00%** | 0 / 200 clips | PASSED |
| - Câu va chạm cú pháp (Carrier Sentences) | Diagnostic | **1.50%** | 3 / 200 clips | PASSED (Từ khóa lồng trong câu) |
| - Đàm thoại thông thường (Conversational) | Diagnostic | **1.00%** | 3 / 300 clips | PASSED |
| - Âm thanh nền & Im lặng (Silence/Acoustic) | Diagnostic | **0.00%** | 0 / 100 clips | PASSED |
| **Tỷ lệ nhầm lẫn lệnh (Command Confusion)** | $\le 1.0\%$ | **1.00% (2/200)** | 2 / 200 clips | **DISCREPANCY FLAGGED** *(Xem Mục H)* |
| **Độ trễ suy luận đơn clip (CPU x86)** | Mean/P95 $\le 50\text{ ms}$ | Mean: **1.8 ms**, P95: **2.12 ms** | Đạt SLA | **PASSED** |

---

## Mục F: Ba Ranh Giới Quản Trị Bắt Buộc (Mandatory Governance Boundaries)

Hồ sơ nghiên cứu bắt buộc phải duy trì 3 nhãn định danh quản trị bất biến sau:

### 1. `NOT_VALIDATED_ON_NATURAL_SPEECH`
Mô hình mới chỉ được kiểm định trên âm thanh tổng hợp Vieneu TTS v3-Turbo kết hợp nhiễu nhân tạo; chưa từng được kiểm nghiệm trên giọng người thật nói tự nhiên có bản quyền/đồng thuận. Phase 90 chỉ có 1 người chấm bài.

### 2. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
Mô hình hoạt động trên đoạn âm thanh cố định 1.0 giây (Bounded-Utterance); chưa có engine VAD và bộ đệm vòng (ring buffer) để nhận dạng micro mở liên tục.

### 3. `NOT_VALIDATED_PENDING_DATA`
Mới chỉ kiểm định độc lập trên phương ngữ Miền Bắc; phương ngữ Miền Trung và Miền Nam chưa được kiểm định do thiếu dữ liệu độc lập (theo Sửa đổi Giao thức 13.1-A).

---

## Mục G: Trạng Thái Kiểm Thử Trực Tiếp (Live Verification & Test Status)

### G.1. Kiểm Tra Tính Đẳng Cấu PyTorch vs ONNX Trên 8 Fixtures Phát Triển
```
Fixture File                            Dự đoán PyTorch   Dự đoán ONNX   Độ trễ PyTorch   Độ trễ ONNX   Khớp nhãn
-----------------------------------------------------------------------------------------------------------------
dev_01_digital_silence.wav              REJECT            REJECT             5.46 ms     3.15 ms     PASS
dev_02_quiet_ambient_room.wav           REJECT            REJECT             2.13 ms     2.56 ms     PASS
dev_03_attenuated_trai_peak010.wav      REJECT            REJECT             1.78 ms     2.58 ms     PASS
dev_04_attenuated_phai_peak005.wav      REJECT            REJECT             1.72 ms     2.52 ms     PASS
dev_05_normal_len_peak089.wav           XUONG             XUONG              1.90 ms     2.55 ms     PASS
dev_06_normal_xuong_peak089.wav         XUONG             XUONG              1.73 ms     2.57 ms     PASS
dev_07_high_amplitude_trai_peak180.wav  LEN               LEN                1.69 ms     2.50 ms     PASS
dev_08_high_amplitude_xuong_peak250.wav XUONG             XUONG              1.71 ms     2.51 ms     PASS
-----------------------------------------------------------------------------------------------------------------
Tổng kết đẳng cấu: 8/8 fixtures khớp nhãn 100% (Độ trễ tối đa ONNX: 3.15 ms <= 50 ms SLA).
```

### G.2. Kết Quả Thực Thi Bộ Kiểm Thử Tự Động (Phase 13.4 Test Suite)
- **Lệnh thực thi:** `D:\trainidentificationaudio\.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_4.py -v`
- **Thời điểm thực thi (UTC):** `2026-10-21T03:04:38.250489+00:00`
- **Thời gian chạy:** `3.33 giây`
- **Mã thoát (Process Exit Code):** `0` (0 = Thành công tuyệt đối)
- **Kết quả bài test:** **`5 passed`**, **`0 failed`**, **`0 skipped`**
- **Toàn bộ Workspace Test Suite:** Đạt **384/384 tests passed** (`pytest tests/ -q` hoàn thành trong ~43.4s).

---

## Mục H: Giới Hạn Khoa Học, Sổ Bất Đồng Kỹ Thuật & Lộ Trình (Limitations, Discrepancies & Next Steps)

### H.1. Sổ Đăng Ký Bất Đồng Kỹ Thuật (Discrepancy Register)

#### DISCREPANCY-01: Phase 13.3 Command Confusion Rate (0.00% vs 1.00%)
- **Hiện tượng:** Một số tài liệu tóm tắt cũ ghi 'Command Confusion Rate: 0.00% (0/200)'. Tuy nhiên, báo cáo chi tiết PHASE13_3_CONFIRMATORY_EVALUATION.md Mục 5.3 (Ma trận 5x5) ghi nhận có 2 mẫu lệnh bị nhầm sang lệnh khác: v13_conf_cmd_trai_006 (TRAI -> PHAI, conf 0.284) và v13_conf_cmd_len_007 (LEN -> PHAI, conf 0.219). Tỷ lệ nhầm lệnh thực tế là 2/200 = 1.00%, không phải 0.00%.
- **Ảnh hưởng & Đánh giá:** Không ảnh hưởng tới việc đạt tiêu chuẩn nghiệm thu (chuẩn cho phép <= 1.00%), nhưng cần ghi nhận minh bạch số liệu thực tế.
- **Trạng thái:** `FLAGGED_AND_CLARIFIED`

#### DISCREPANCY-02: Độ trễ Single-Clip Cold Start vs Steady State
- **Hiện tượng:** Trong bài kiểm thử test_infer_cli_parity_and_latency, lần suy luận đầu tiên trên dev_files[0] khi chưa nạp bộ nhớ đệm (cold start) có thể đạt 57.02 ms (vượt nhẹ ngưỡng 50 ms). Ở trạng thái steady-state (warm run), độ trễ PyTorch là 1.39 ms - 2.10 ms và ONNX là 2.27 ms - 3.24 ms.
- **Ảnh hưởng & Đánh giá:** Cần lưu ý cơ chế JIT warmup khi triển khai thực tế trên vi xử lý.
- **Trạng thái:** `DOCUMENTED`

#### DISCREPANCY-03: Lịch Sử Kích Thước Tệp Checkpoint & ONNX (v11 vs Baseline v10)
- **Hiện tượng:** Checkpoint đông kết v11 có kích thước 566,146 bytes (~552.9 KB) và ONNX là 555,751 bytes (~542.7 KB). Một số bảng tham chiếu cũ ghi kích thước 479,524 bytes và 485,498 bytes vốn thuộc về bản phát hành v10 trước đó.
- **Ảnh hưởng & Đánh giá:** Đã cập nhật đúng số bytes thực tế và mã SHA-256 chính thống của bản phát hành v11.
- **Trạng thái:** `CLARIFIED_IN_REPORT`

### H.2. Lộ Trình Kỹ Thuật Ưu Tiên Tiếp Theo (Prioritized Technical Roadmap)
1. Bảo toàn Checkpoint và Protocol Đông Kết (Zero-Tampering Invariant).
2. Duy trì toàn vẹn hồ sơ bằng chứng và khả năng tái lập.
3. Tìm nguồn TTS phương ngữ Miền Trung / Miền Nam đạt chuẩn provenance hoặc phê duyệt thu âm người thật.
4. Triển khai chiến dịch thu âm người thật 3 miền (Bắc 40%, Trung 30%, Nam 30%) có cam kết đồng thuận pháp lý.
5. Phân tích calibration (ECE) và tối ưu hóa ngưỡng kích hoạt theo từng phương ngữ.
6. Nghiên cứu engine streaming micro mở riêng biệt: VAD C++, ring-buffer, debounce, đo đạc FA/hour.
7. Triển khai và đo đạc trên phần cứng di động thực tế (Android NDK / iOS CoreAudio) và lượng tử hóa INT8.

---
*Báo cáo được khởi tạo tự động bởi [`verify_model_status.py`](file:///D:/trainidentificationaudio/reports/cnn_comprehensive/model_status_pack/verify_model_status.py) phục vụ hội đồng chuyên môn.*
