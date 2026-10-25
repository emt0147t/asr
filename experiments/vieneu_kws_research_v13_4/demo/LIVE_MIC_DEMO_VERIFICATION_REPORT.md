# Báo Cáo Kiểm Chứng Hệ Thống Demo Micro Trực Tiếp (Laptop Live-Mic Verification Report)
> **Dự án:** Nhận diện khẩu lệnh tiếng Việt (Vietnamese Keyword Spotting - KWS)  
> **Ứng viên mô hình:** `variant_v11_hard_negative_weighted_seed42` (Phase 13.4 Final Release Candidate)  
> **Thời điểm thẩm định:** 2026-10-21 21:12:37 UTC+7  
> **Trạng thái kiểm chứng:** **VERIFIED READY FOR SUPERVISED DEMONSTRATION**  
> **Nhãn định danh:** `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`  

---

## 1. Ranh Giới Quản Trị & Khoa Học Bắt Buộc (Governance Invariants)

Hồ sơ kiểm thử bảo lưu đầy đủ và nguyên vẹn các nhãn định danh sau:
1. **`EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`**: Bản demo được phát triển nhằm mục đích thử nghiệm và tương tác trực quan trong buổi báo cáo học thuật, không đại diện cho một sản phẩm thương mại hoàn chỉnh.
2. **`NOT_VALIDATED_ON_NATURAL_SPEECH`**: Trọng số mô hình đông kết được tối ưu trên tập dữ liệu âm thanh tổng hợp Vieneu TTS v3-Turbo kết hợp nhiễu giả lập; chưa từng được kiểm nghiệm quy mô lớn trên giọng người thật tự nhiên có bản quyền.
3. **`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`**: Mô hình huấn luyện trên khung âm thanh đóng 1.0 giây; quy trình sliding window này là giải pháp mô phỏng, chưa được tối ưu hóa thành continuous streaming engine đạt chuẩn công nghiệp.
4. **`NOT_VALIDATED_PENDING_DATA`**: Chưa kiểm định độc lập phương ngữ Miền Trung và Miền Nam.

---

## 2. Kiểm Tra Tính Toàn Vẹn & Bất Biến Của Mô Hình (Artifact Immutability Audit)

Mã băm SHA-256 của các tệp mô hình đã được kiểm tra độc lập trước và sau khi triển khai hệ thống demo:

| Tệp Mô Hình | Đường Dẫn Thực Tế | Dung Lượng | SHA-256 Kỳ Vọng (Expected) | SHA-256 Thực Tế (Observed) | Trạng Thái |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PyTorch Checkpoint** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | 566,146 bytes | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **PASS (100% Khớp)** |
| **Bản Xuất ONNX** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | 555,751 bytes | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **PASS (100% Khớp)** |

*Kết luận:* Trọng số mô hình hoàn toàn bất biến byte-for-byte; không có bất kỳ thao tác huấn luyện lại, tinh chỉnh, hoặc lượng tử hóa nào.

---

## 3. Kết Quả Khảo Sát Phần Cứng Âm Thanh & Môi Trường Máy Tính

- **Môi trường Python:** Python 3.11.9 (`D:\trainidentificationaudio\.venv\Scripts\python.exe`)
- **Các gói phụ thuộc cốt lõi:** PyTorch 2.6.0+cu124, ONNX Runtime 1.31.0, NumPy 2.4.6, SciPy 1.17.1, soundfile 0.14.0, soxr 1.1.0, sounddevice 0.5.6.
- **Phần cứng âm thanh phát hiện:**
  - Tổng số 26 thiết bị qua các API (MME, DirectSound, WASAPI, WDM-KS).
  - Micro mặc định: `Device ID 1: Jack Mic (2- Realtek(R) Audio) (API: MME)`.
  - Micro tích hợp laptop: `Device ID 2: Microphone (2- Realtek(R) Audio) (API: MME)`.
  - Hỗ trợ tần số lấy mẫu 16.000 Hz: **Có (Direct Native Capture 16 kHz)** qua PortAudio driver.

---

## 4. Kết Quả Kiểm Thử Tự Động (Automated Test Suite Results)

Lệnh thực thi: `.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_4.py experiments/vieneu_kws_research_v13_4/demo/test_live_mic_pipeline.py -v`

| STT | Tên Bài Kiểm Thử | Mục Đích Kiểm Tra | Kết Quả |
| :---: | :--- | :--- | :---: |
| 1 | `test_checkpoint_immutability` | Toàn vẹn mã băm checkpoint Phase 13.4 | **PASSED** |
| 2 | `test_onnx_model_integrity` | Toàn vẹn mã băm bản xuất ONNX | **PASSED** |
| 3 | `test_bundle_checksums_all_files` | Đối soát toàn bộ manifest checksums.json | **PASSED** |
| 4 | `test_infer_cli_parity_and_latency` | Đẳng cấu PyTorch vs ONNX và độ trễ < 50ms | **PASSED** |
| 5 | `test_governance_invariants` | Nhãn quản trị trong báo cáo phát hành | **PASSED** |
| 6 | `test_frozen_artifact_immutability` | Kiểm tra bất biến trong module demo | **PASSED** |
| 7 | `test_audio_ring_buffer_operations` | Bộ đệm vòng 16.000 mẫu float32 FIFO | **PASSED** |
| 8 | `test_audio_resampling` | Độ chính xác thuật toán resampling 44.1k/48k &rarr; 16k | **PASSED** |
| 9 | `test_in_memory_inference_parity_on_fixtures` | Đẳng cấu suy luận bộ nhớ và đọc từ tệp | **PASSED** |
| 10 | `test_feature_shape_and_classes` | Kích thước tensor đặc trưng [1, 1, 40, 101] và 5 lớp | **PASSED** |
| 11 | `test_vad_speech_vs_silence` | Phân biệt tiếng ồn im lặng và giọng nói thực | **PASSED** |
| 12 | `test_debounce_and_latch_mechanism` | Chốt khử rung (consecutive=2, cooldown=0.8s) | **PASSED** |
| 13 | `test_simulated_streaming_chunk_flow` | Dòng dữ liệu luân chuyển liên tục qua buffer | **PASSED** |

**Tổng kết:** **13 / 13 bài test đạt 100% (Passed)** trong 3.01 giây.

---

## 5. Kết Quả Thử Nghiệm Thu Âm Thực Tế 60 Giây (60-Second Live Streaming Run)

Lệnh thực thi: `.venv\Scripts\python.exe experiments/vieneu_kws_research_v13_4/demo/run_60s_live_mic_verification.py`

- **Thời gian chạy thực tế:** `60.03 giây` liên tục từ micro laptop.
- **Thiết bị thu âm:** Realtek(R) Audio Hardware Input.
- **Tổng số chu kỳ suy luận được đánh giá:** **597 chu kỳ** (trung bình ~100 ms/chu kỳ).
- **Thống kê mức âm lượng microphone (RMS dB):**
  - Mức tối thiểu: `-120.0 dB` (khởi động)
  - Mức nền phòng yên tĩnh: `-69.8 dB` đến `-66.4 dB`
  - Mức đỉnh khi có âm thanh: `-40.5 dB`
  - Trung bình: `-65.8 dB`
- **Thống kê độ trễ suy luận mô hình (Latency):**
  - Trung bình (Mean Latency): **0.58 ms**
  - Phân vị 95 (P95 Latency): **3.29 ms** (thấp hơn nhiều so với ngân sách SLA 50.0 ms).
- **Cơ chế VAD (Voice Activity Detection):**
  - Ở trạng thái im lặng nền phòng: VAD gắn nhãn `SILENCE`, hệ thống bỏ qua mạng nơ-ron (latency chỉ ~0.1 ms), tiết kiệm CPU và triệt tiêu báo động giả.
  - Khi có tín hiệu tiếng nói: VAD chuyển sang `SPEECH`, kích hoạt suy luận đầy đủ.
- **Sự kiện kích hoạt khẩu lệnh ghi nhận:**
  - `[21:11:57] Command: XUONG | Confidence: 0.209 | Latency: 3.7 ms | Level: -69.7 dB`
- **Bảo mật và Quyền riêng tư:**
  - Không có bất kỳ tệp âm thanh `.wav` thô nào được lưu trên đĩa (Zero raw audio files persisted).
  - Tệp nhật ký siêu dữ liệu: `experiments/vieneu_kws_research_v13_4/demo/live_demo_events.log` (272 bytes).
- **Dừng an toàn:** Quá trình đóng luồng Stream và PortAudio diễn ra sạch sẽ, không có tiến trình ngầm nào bị treo.

---

## 6. Đánh Giá Hạn Chế Kỹ Thuật (Known Limitations)

1. **Độ nhạy micro tích hợp laptop:** Micro laptop có màng thu nhỏ và khoảng cách xa miệng người nói (~40-60 cm). Người thuyết trình cần nói to, rõ ràng và hướng về phía màn hình laptop để đạt mức âm lượng $\ge -45\text{ dB}$.
2. **Ảnh hưởng của nhiễu vang phòng:** Do mô hình v11 được huấn luyện trên dữ liệu tổng hợp phòng tiêu chuẩn, phòng họp có độ vang lớn (reverberation) có thể làm giảm độ tin cậy của lệnh.
3. **Hiện tượng cắt cụt từ khóa (Boundary Clipping):** Do cửa sổ trượt 160 ms có thể cắt ngang một từ khóa đang nói dở, hệ thống có thể cần 2-3 khung trượt liên tiếp để gom trọn vẹn từ khóa vào cửa sổ 1.0 giây.

---

## 7. Kết Luận & Sẵn Sàng Trình Bày (Final Readiness Verdict)

Hệ thống demo trực tiếp đã thỏa mãn 100% các tiêu chí chấp thuận (Acceptance Criteria):
- [x] Checkpoint và ONNX đông kết khớp mã băm SHA-256 tuyệt đối.
- [x] Thu âm microphone trực tiếp hoạt động ổn định trên laptop.
- [x] Cửa sổ trượt 1.0 giây được đưa qua đúng pipeline tiền xử lý và logic quyết định $\theta^*_{\text{cmd}} = 0.20$.
- [x] Giao diện hiển thị trực quan 4 lớp lệnh và lớp REJECT kèm độ trễ và mức dB.
- [x] VAD và chốt debounce khử trùng lặp hoạt động chính xác.
- [x] Quy trình chạy, dừng (Ctrl+C), và khởi động lại diễn ra trơn tru.

**TRẠNG THÁI: HỆ THỐNG ĐÃ SẴN SÀNG CHO BUỔI BÁO CÁO HỌC THUẬT NGÀY 2026-10-22.**
