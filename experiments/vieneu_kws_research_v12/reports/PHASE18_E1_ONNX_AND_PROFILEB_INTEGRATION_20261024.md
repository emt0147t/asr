# Báo Cáo Triển Khai Xuất ONNX Cho E1 Và Tích Hợp Hồ Sơ B Thử Nghiệm
## (Phase 18 — E1 ONNX Export and Controlled Profile B Integration)

> **Mã báo cáo:** `PHASE18_E1_ONNX_AND_PROFILEB_INTEGRATION_20261024`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm nghiệm thu:** 2026-10-24  
> **Giai đoạn trước:** Phase 17 — Safe Integration Architecture Design (`PASS_TO_IMPLEMENTATION_PLAN`)  
> **Quyết định cổng nghiệm thu (Final Gate Decision):** **`PASS_TO_CONTROLLED_OFFLINE_PROFILE_VALIDATION`**  
> **Các nhãn quản trị bắt buộc:**
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`

---

## 1. Tóm Tắt Chấp Hành & Ranh Giới Quản Trị Khoa Học

Phase 18 thực thi kế hoạch tích hợp đã được phê duyệt ở Phase 17 theo cơ chế kiểm soát cổng nghiêm ngặt (gated implementation). Mục tiêu cốt lõi là:
1. Xuất mô hình checkpoint E1 (`exp_v12_e1_baseline_scale_seed42.pt`) sang tệp ONNX độc lập (`vietnamese_kws_v12_e1.onnx`) và kiểm chứng sai số số học $\le 10^{-5}$ so với PyTorch gốc trên các fixture chuẩn.
2. Ghi nhận đường cơ sở hồi quy (regression baseline) của hồ sơ mặc định `legacy_v11` trước khi can thiệp mã nguồn ứng dụng.
3. Hiện thực hóa Hồ sơ B thử nghiệm (`experimental_e1_speechband`) trong mã nguồn ứng dụng demo (`live_mic_app.py`, `live_mic_engine.py`) theo hình thức opt-in rõ ràng qua tham số dòng lệnh `--profile`.
4. Thiết lập cơ chế kiểm tra fail-fast toàn diện: kiểm tra sự tồn tại và mã băm SHA-256 của các tệp artifact trước khi khởi động bộ thu âm; cấm tuyệt đối cơ chế âm thầm hạ cấp (silent fallback) về v11 khi gặp lỗi ở Profile B.
5. Chứng minh không suy thoái (zero regression) $100\%$ đối với hồ sơ mặc định `legacy_v11`.
6. Thực hiện kiểm thử tích hợp ngoại tuyến (offline integration tests) không qua microphone vật lý, xử lý triệt để lỗi tương thích PyTorch frontend tại dòng 602.
7. Công bố minh bạch các giới hạn thực nghiệm: độ chính xác cả chuỗi lệnh $51.67\%$ và $6/24$ kích hoạt ảo OOV từ Phase 16.

---

## 2. Bảng Đối Soát Mật Mã Cổ Vật (Cryptographic Checksum Verification)

Toàn bộ các tệp mô hình, cấu hình và dữ liệu đã được xác minh mã băm SHA-256 byte-for-byte:

| Tên Cổ Vật / Tệp Dữ Liệu | Đường Dẫn Thực Tế | SHA-256 Kỳ Vọng / Ghi Nhận | Trạng Thái Toàn Vẹn |
| :--- | :--- | :---: | :---: |
| **Checkpoint v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **100% Khớp Nguyên Bản** |
| **ONNX v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **100% Khớp Nguyên Bản** |
| **Checkpoint E1 đóng băng** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **100% Khớp Nguyên Bản** |
| **ONNX E1 mới xuất (Phase 18)** | `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` | `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8` | **Đã Khóa & Xác Minh** |
| **Cấu hình VAD Phase 15** | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | **100% Khớp Nguyên Bản** |
| **Manifest VAD Dev Corpus** | `experiments/vieneu_kws_research_v12/manifests/vad_dev_corpus_manifest.csv` | `1ebf11f8c7048e162e04d24df92b40e75b026c7059d9a9af5afcbab1274587c3` | **100% Khớp Nguyên Bản** |
| **Manifest VAD Test Corpus** | `experiments/vieneu_kws_research_v12/manifests/vad_test_corpus_manifest.csv` | `d0342e3f80c1307801c81637ba5b02e002c598aded35577b525afc106a8b5876` | **100% Khớp Nguyên Bản** |

---

## 3. Kết Quả Kiểm Định Chi Tiết Từng Cổng (Gate Verification Audits)

### 3.1 Gate A: Xuất ONNX Cho E1 & Kiểm Định Sai Số Số Học (Numerical Parity)
- **Tệp xuất:** `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx`
- **Dung lượng:** $542.7\text{ KB}$ ($555,751\text{ bytes}$)
- **Cấu hình ONNX:** Opset 17, Input tensor `log_mel` $[B, 1, 40, 101]$ float32, Output `logits` $[B, 5]$ và `probabilities` $[B, 5]$, dynamic batch axis.
- **Tiêu chuẩn kiểm định sai số:** $\le 10^{-5}$ trên toàn bộ 10 development fixtures tại `experiments/vieneu_kws_research_v12/fixtures/`.
- **Kết quả thực tế:**
  - Sai số logit tuyệt đối cực đại ($\max |\text{diff}_{\text{logits}}|$): **$9.54 \times 10^{-6}$** (Đạt chuẩn $\le 1.0 \times 10^{-5}$).
  - Sai số xác suất tuyệt đối cực đại ($\max |\text{diff}_{\text{probs}}|$): **$8.34 \times 10^{-7}$** (Đạt chuẩn $\le 1.0 \times 10^{-5}$).
  - Tỷ lệ sai khác nhãn quyết định: **$0/10$ ($100\%$ đồng nhất hoàn hảo)**.
  - Báo cáo chi tiết được lưu trữ tại: `experiments/vieneu_kws_research_v12/onnx/e1_onnx_parity_report.json`.
- **Kết luận Gate A:** **`PASS`**

### 3.2 Gate B: Ghi Nhận Đường Cơ Sở Hồi Quy Profile Mặc Định (Legacy v11 Baseline)
- **Kịch bản kiểm thử:** `experiments/vieneu_kws_research_v12/scripts/run_v11_regression_fixtures.py --record`.
- **Phạm vi kiểm thử:** Tổng hợp 18 audio fixtures từ hai bộ:
  - Phase 12 Fixtures (10 tệp): `parity_dev_01_trai.wav` đến `parity_dev_10_short_padded.wav`.
  - Phase 13.1 Fixtures (8 tệp): `dev_01_digital_silence.wav` đến `dev_08_high_amplitude_xuong_peak250.wav`.
- **Kết quả:** Ghi nhận toàn bộ nhãn dự đoán, xác suất 5 lớp và độ tin cậy vào `experiments/vieneu_kws_research_v12/reports/v11_pre_change_baseline.json` trước khi chỉnh sửa bất kỳ tệp ứng dụng nào.
- **Kết luận Gate B:** **`PASS`**

### 3.3 Gate C: Cải Tiến Ứng Dụng & Tích Hợp Hồ Sơ B (Opt-in Profile B Implementation)
Các sửa đổi đã được thực hiện trực tiếp trên mã nguồn demo:

1. **`experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py`:**
   - **Tích hợp `SpeechBandRatioSegmenterAdapter`:** Bọc lớp `SpeechBandRatioVAD` (từ `phase15_vad/vad_segmenters.py`) với giao diện xử lý chunk tương thích hoàn toàn với `UtteranceSegmenter`, tự động tính toán `chunk_time_sec` từ số mẫu tích lũy và chuyển đổi định dạng kết quả sang tuple `(audio_1s, duration_ms)`.
   - **Khắc phục lỗi tương thích PyTorch dòng 602:** Hiệu đính lệnh tiền xử lý PyTorch thành `spec = self.inferencer.frontend.compute(audio_1s)`, tương thích với API của `LogMelFrontendNumPy` và loại bỏ hoàn toàn lỗi `TypeError: 'LogMelFrontendNumPy' object is not callable`.
   - **Hỗ trợ 2 hồ sơ độc lập:** Tham số `profile: str = "legacy_v11"` được bổ sung vào `LiveKWSPipeline`. Khi `profile == "experimental_e1_speechband"`, pipeline tự động cấu hình VAD dải formant từ `selected_phase15_vad_config.json` và nạp mô hình E1 ONNX.
   - **Mở rộng viễn thám `DetectionEvent`:** Bổ sung các trường `profile_name`, `model_artifact_hash`, `vad_config_hash`, `vad_algorithm` vào sự kiện kích hoạt và hàm `get_status_dict()`.

2. **`experiments/vieneu_kws_research_v13_4/demo/live_mic_app.py`:**
   - **Bổ sung tham số CLI `--profile`:** Cung cấp 2 lựa chọn tường minh `legacy_v11` (mặc định) và `experimental_e1_speechband`.
   - **Kiểm tra fail-fast nghiêm ngặt trước khi thu âm:**
     - Profile A: Xác minh sự tồn tại và mã băm của `vietnamese_kws_v11.onnx` (`54298c...`).
     - Profile B: Xác minh sự tồn tại và mã băm của `vietnamese_kws_v12_e1.onnx` (`20ecc2...`) và `selected_phase15_vad_config.json` (`af5c38...`).
     - Nếu phát hiện tệp thiếu hoặc sai mã băm, chương trình in thông báo lỗi chi tiết ra `sys.stderr` và lập tức thoát với exit code `1`. **Tuyệt đối không tự động âm thầm hạ cấp (silent fallback) về v11.**
   - **Giao diện trực quan:** Hiển thị huy hiệu `[ HỒ SƠ B THỬ NGHIỆM: E1 ONNX + SPEECHBAND VAD (PHASE 18) ]` trên Rich Terminal Dashboard và giao diện đồ họa Tkinter.
- **Kết luận Gate C:** **`PASS`**

### 3.4 Gate D: Xác Minh Hồi Quy Profile Mặc Định (Legacy v11 Regression Verification)
- **Kịch bản kiểm thử:** `experiments/vieneu_kws_research_v12/scripts/run_v11_regression_fixtures.py --verify`.
- **Kết quả:**
  - So sánh $18/18$ fixtures giữa mã nguồn sau chỉnh sửa và tệp baseline đã lưu.
  - Tỷ lệ khớp nhãn dự đoán: **$100\%$ ($18/18$)**.
  - Độ lệch xác suất 5 lớp cực đại: **$< 10^{-6}$** (Không có bất kỳ sai lệch nào).
  - Kiểm thử toàn bộ bundle Phase 13.4 (`pytest tests/test_vieneu_research_v13_4.py`): **$5/5$ tests passed ($100\%$)**.
- **Kết luận Gate D:** **`PASS`** (Zero regression).

### 3.5 Gate E: Kiểm Thử Tích Hợp Ngoại Tuyến Hồ Sơ B (Controlled Offline Integration Tests)
- **Tệp kiểm thử:** `experiments/vieneu_kws_research_v12/tests/test_profile_b_integration.py`.
- **Tổng số ca kiểm thử:** 7 ca kiểm thử tự động, kết quả: **$7/7\text{ PASSED}$ ($100\%$)**.
  1. `test_artifacts_exist_and_hashes_match`: Xác minh mã băm các tệp Profile B.
  2. `test_profile_b_initialization_and_metadata`: Xác minh khởi tạo và siêu dữ liệu Profile B.
  3. `test_offline_chunked_feeding_simulation`: Mô phỏng đẩy luồng audio 30ms ngoại tuyến không cần micro vật lý.
  4. `test_profile_b_inference_on_all_dev_fixtures`: Đánh giá suy luận trên toàn bộ 10 fixture chuẩn.
  5. `test_cli_fail_fast_invalid_profile`: Xác minh CLI từ chối tên profile không hợp lệ.
  6. `test_cli_fail_fast_corrupted_vad_hash`: Xác minh fail-fast và cấm silent fallback khi cấu hình VAD bị hỏng mã băm.
  7. `test_disclose_phase16_empirical_limitations`: Kiểm tra tính hiện diện của việc công bố giới hạn thực nghiệm Phase 16.
- **Kết luận Gate E:** **`PASS`**

---

## 4. Công Bố Minh Bạch Giới Hạn Thực Nghiệm (Empirical Limitations Disclosure)

Kế thừa các kết quả độc lập từ Phase 16, các giới hạn kỹ thuật của hệ thống được công bố đầy đủ:
1. **Độ chính xác cả chuỗi lệnh (Exact-Sequence Accuracy):** Trên tập kiểm thử bị khóa (Phase 16), chuỗi lệnh streaming chỉ đạt **$51.67\%$** độ chính xác hoàn hảo cả chuỗi. Đây là giới hạn đo lường khách quan trong điều kiện nhiều lệnh liên tiếp ở khoảng nghỉ ngắn.
2. **Kích hoạt ảo trên từ ngoài tập từ vựng (OOV False Activations):** Ghi nhận **$6/24$** trường hợp nhận nhầm từ OOV thành lệnh (tần suất $42.80\text{ lần/phút OOV}$, tập trung vào các từ có thanh sắc: `ghế`, `Pháo`, `Phép`, `Khóa`, `Xôi`).
3. **Quy định bất biến:** Tuyệt đối không can thiệp danh sách đen từ cứng (hardcoded blacklist) và không tùy tiện nâng ngưỡng $\theta_{\text{cmd}} = 0.20$ để làm đẹp số liệu.

---

## 5. Hướng Dẫn Vận Hành & Khởi Động Hồ Sơ Demo

### 5.1 Khởi Động Mặc Định (Profile A: Legacy v11)
```bash
# Terminal Dashboard mặc định
.venv\Scripts\python.exe experiments/vieneu_kws_research_v13_4/demo/live_mic_app.py

# Giao diện Desktop GUI
.venv\Scripts\python.exe experiments/vieneu_kws_research_v13_4/demo/live_mic_app.py --gui
```

### 5.2 Khởi Động Hồ Sơ Thử Nghiệm (Profile B: Experimental E1 + SpeechBand VAD)
```bash
# Terminal Dashboard thử nghiệm
.venv\Scripts\python.exe experiments/vieneu_kws_research_v13_4/demo/live_mic_app.py --profile experimental_e1_speechband

# Giao diện Desktop GUI thử nghiệm
.venv\Scripts\python.exe experiments/vieneu_kws_research_v13_4/demo/live_mic_app.py --profile experimental_e1_speechband --gui
```

---

## 6. Quyết Định Cổng Nghiệm Thu Cuối Cùng (Final Gate Decision)

Toàn bộ 5 cổng kỹ thuật (Gate A, B, C, D, E) đều đạt $100\%$ tiêu chuẩn nghiệm thu độc lập:
- Sai số ONNX của E1 $\le 10^{-5}$ ($9.54 \times 10^{-6}$).
- Bảo toàn $100\%$ hành vi của v11 mặc định (zero regression).
- Khắc phục triệt để lỗi tương thích dòng 602.
- Kiểm tra fail-fast và cấm silent fallback hoạt động chính xác.
- Bảo toàn trọn vẹn mã băm của toàn bộ các cổ vật đóng băng.

**QUYẾT ĐỊNH CHÍNH THỨC:**  
# **`PASS_TO_CONTROLLED_OFFLINE_PROFILE_VALIDATION`**
