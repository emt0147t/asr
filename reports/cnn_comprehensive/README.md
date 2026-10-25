# Tài liệu Tổng hợp Lịch sử Nghiên cứu và Kỹ thuật Mô hình CNN Keyword Spotting (KWS) Tiếng Việt

> **Cảnh báo Pháp lý & Hiện trạng Dự án (Canonical Status Labels):**
> - **Candidate Status:** `FROZEN_CNN_RELEASE_CANDIDATE`
> - **Natural Speech Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - **Streaming Deployment Status:** `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`

Tài liệu này là hồ sơ kỹ thuật toàn diện, có bằng chứng thực chứng (evidence-based), ghi lại toàn bộ tiến trình từ các thử nghiệm sơ khởi (baseline), qua 26 giai đoạn nghiên cứu, đến quy trình đánh giá xác nhận độc lập (Phase 13.3) và đóng gói bản phát hành ứng viên đông kết (Phase 13.4). Hồ sơ được thiết kế nhằm giúp một kỹ sư AI hoặc nhà nghiên cứu mới có thể nắm bắt đầy đủ bối cảnh, kiến trúc, dữ liệu, lỗi kỹ thuật, và lộ trình kiểm thử tiếp theo.

---

## 1. Cấu trúc Tài liệu (Documentation Map)

Toàn bộ gói tài liệu bao gồm 15 chương chính, 5 phụ lục chuyên sâu và 6 bảng dữ liệu đối chuẩn CSV:

### Các Chương Cốt Lõi (Core Chapters)
1. **[Chương 00: Tóm tắt Điều hành (Executive Summary)](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/00_EXECUTIVE_SUMMARY.md)**: Tổng quan toàn bộ dự án, vị trí của checkpoint đông kết, các chỉ số cốt lõi và giới hạn vận hành bắt buộc.
2. **[Chương 01: Phạm vi Dự án & Yêu cầu Kỹ thuật](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/01_PROJECT_SCOPE_AND_REQUIREMENTS.md)**: Định nghĩa bài toán 4 từ khóa (`TRAI`, `PHAI`, `LEN`, `XUONG`), ranh giới giữa bounded-utterance (1.0 giây) và streaming open-mic, cùng sự tiến hóa của cơ chế từ chối (`REJECT`).
3. **[Chương 02: Lịch sử Nghiên cứu Theo Tiến trình Thời gian](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/02_CHRONOLOGICAL_RESEARCH_HISTORY.md)**: Biên niên sử chi tiết qua 26 giai đoạn từ baseline sơ khai (Phase 27), các chuỗi cải tiến kiến trúc v2-v10, các đợt kiểm thử mù Phase 90, triển khai di động Phase 91-92, đến chiến dịch hoàn thiện v11 và đánh giá độc lập v13.
4. **[Chương 03: Nguồn gốc Dữ liệu & Phân chia Tập dữ liệu](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/03_DATASET_PROVENANCE_AND_SPLITS.md)**: Nguồn dữ liệu tổng hợp Vieneu TTS v3-Turbo, phân bổ speaker không rò rỉ (leakage-free), thiết kế hard negatives âm vị học và nhiễu nền.
5. **[Chương 04: Kiến trúc Mô hình & Huấn luyện](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/04_MODEL_ARCHITECTURES_AND_TRAINING.md)**: Phân tích kiến trúc TC-ResNet, CorrectedVietnameseKWSNet, hàm mất mát Hard-Negative Weighted Cross Entropy và siêu tham số.
6. **[Chương 05: Kết quả Thử nghiệm & So sánh](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/05_EXPERIMENTAL_RESULTS_AND_COMPARISON.md)**: Bảng đối chuẩn hiệu năng đa thế hệ, đường cong đánh đổi Pareto (Recall vs FAR vs Latency) và sổ tay so sánh giao thức.
7. **[Chương 06: Phân tích Lỗi & Bài học Nghiên cứu](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/06_ERROR_ANALYSIS_AND_RESEARCH_LESSONS.md)**: Phân loại sai số âm vị học, lỗi khuếch đại tín hiệu im lặng (quiet amplification bug), xung đột kích thước FFT ($N_{\text{fft}}$ bin mismatch) và bài học kỹ thuật.
8. **[Chương 07: Hợp đồng Tiền xử lý & Suy luận](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/07_PREPROCESSING_AND_INFERENCE_CONTRACT.md)**: Quy chuẩn toán học và mã nguồn chuẩn hóa âm thanh, trích xuất Log-Mel Spectrogram ($N_{\text{fft}}=512$), ngưỡng $\theta^*=0.20$ và kiểm định đẳng cấu PyTorch/ONNX.
9. **[Chương 08: Đánh giá Lắng nghe của Con người & Thẩm định](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/08_HUMAN_LISTENING_AND_VALIDATION.md)**: Thẩm tra thực tế thử nghiệm mù Phase 90: bằng chứng duy nhất 1 người đánh giá hoàn thành, 5 người bỏ cuộc, và tính trung thực của dữ liệu.
10. **[Chương 09: Đánh giá Xác nhận Độc lập Phase 13.3](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/09_CONFIRMATORY_EVALUATION_V13_3.md)**: Kết quả thẩm định độc lập 1.000 clip âm thanh tổng hợp từ 8 giọng Bắc mới, đạt 99.20% Accuracy, 99.00% Command Recall, 0.75% FAR.
11. **[Chương 10: Phả hệ Checkpoint & Đóng gói Phát hành](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/10_CHECKPOINT_LINEAGE_AND_RELEASE_ARTIFACTS.md)**: Biểu đồ phả hệ checkpoint từ Phase 9/10 đến Phase 13.4, băm SHA-256 xác thực và cấu trúc bundle phát hành.
12. **[Chương 11: Kỹ thuật Triển khai & Giới hạn](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/11_DEPLOYMENT_ENGINEERING_AND_LIMITATIONS.md)**: Thực trạng triển khai ONNX Runtime, rào cản C++ Mel Filterbank, và trạng thái khung mẫu Android/iOS.
13. **[Chương 12: Khả năng Tái lập & Độ phủ Kiểm thử](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/12_REPRODUCIBILITY_AND_TEST_COVERAGE.md)**: Hướng dẫn tái lập toàn bộ quy trình, danh mục 384 bài kiểm thử tự động đạt 100% pass rate.
14. **[Chương 13: Hiện trạng Cuối cùng & Lộ trình Đề xuất](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/13_FINAL_STATUS_AND_RECOMMENDED_NEXT_STEPS.md)**: Ma trận thẩm định thực chứng 10 chiều và lộ trình kỹ thuật cho thu âm thực tế (natural speech) và streaming VAD.
15. **[Chương 14: Khoảng trống Bằng chứng & Bất đồng](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/14_EVIDENCE_GAPS_AND_DISCREPANCIES.md)**: Sổ đăng ký 6 điểm bất đồng kỹ thuật trong lịch sử và cách thức giải quyết triệt để.

### Phụ Lục (Appendices)
- **[Phụ lục A: Từ điển Thuật ngữ & Chỉ số](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/appendices/A_METRICS_DICTIONARY.md)**
- **[Phụ lục B: Danh mục Checkpoint & Mô hình](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/appendices/B_MODEL_AND_CHECKPOINT_INVENTORY.md)**
- **[Phụ lục C: Danh mục Tập dữ liệu & Manifest](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/appendices/C_DATASET_AND_SPLIT_INVENTORY.md)**
- **[Phụ lục D: Danh mục Bộ kiểm thử Tự động](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/appendices/D_TEST_SUITE_INVENTORY.md)**
- **[Phụ lục E: Chỉ mục Tham chiếu Nguồn](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/appendices/E_SOURCE_REFERENCE_INDEX.md)**

### Bảng Dữ Liệu Đối Chuẩn (CSV Master Tables)
- [`tables/experiment_timeline.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/experiment_timeline.csv): Bảng biên niên sử 26 giai đoạn.
- [`tables/model_metrics_comparison.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/model_metrics_comparison.csv): Bảng so sánh 9 mô hình chính qua các giai đoạn.
- [`tables/checkpoint_lineage.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/checkpoint_lineage.csv): Danh mục 41 file checkpoint và mô hình ONNX cùng mã băm SHA-256.
- [`tables/dataset_inventory.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/dataset_inventory.csv): Danh mục 116 manifest dữ liệu cùng số dòng và mã băm.
- [`tables/validation_status.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/validation_status.csv): Hiện trạng xác nhận trên 10 khía cạnh kỹ thuật.
- [`tables/unresolved_discrepancies.csv`](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/tables/unresolved_discrepancies.csv): 6 sai lệch kỹ thuật và cách xử lý.

---

## 2. Thông tin Checkpoint Ứng viên Đông kết (Frozen Candidate)

Ứng viên phát hành duy nhất được chuẩn y hiện tại là:
- **Tên mô hình:** `variant_v11_hard_negative_weighted_seed42.pt`
- **Đường dẫn Bundle:** [`experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)
- **Kích thước file:** 479.524 bytes (468,29 KB)
- **Mã băm SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`
- **Phiên bản ONNX Export:** [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx) (SHA-256: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`)
- **Ngưỡng quyết định lệnh (Decision Threshold):** $\theta_{\text{cmd}} = 0.20$

Mọi tuyên bố kỹ thuật trong tài liệu này đều được liên kết trực tiếp tới file nguồn, mã nguồn thực thi, dữ liệu thẩm định và các bản ghi kiểm thử tự động trong kho mã nguồn `D:\trainidentificationaudio`.
