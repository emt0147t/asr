import os

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")

readme_content = """# Tài liệu Tổng hợp Lịch sử Nghiên cứu và Kỹ thuật Mô hình CNN Keyword Spotting (KWS) Tiếng Việt

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
7. **[Chương 06: Phân tích Lỗi & Bài học Nghiên cứu](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/06_ERROR_ANALYSIS_AND_RESEARCH_LESSONS.md)**: Phân loại sai số âm vị học, lỗi khuếch đại tín hiệu im lặng (quiet amplification bug), xung đột kích thước FFT ($N_{\\text{fft}}$ bin mismatch) và bài học kỹ thuật.
8. **[Chương 07: Hợp đồng Tiền xử lý & Suy luận](file:///d:/trainidentificationaudio/reports/cnn_comprehensive/07_PREPROCESSING_AND_INFERENCE_CONTRACT.md)**: Quy chuẩn toán học và mã nguồn chuẩn hóa âm thanh, trích xuất Log-Mel Spectrogram ($N_{\\text{fft}}=512$), ngưỡng $\\theta^*=0.20$ và kiểm định đẳng cấu PyTorch/ONNX.
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
- **Ngưỡng quyết định lệnh (Decision Threshold):** $\\theta_{\\text{cmd}} = 0.20$

Mọi tuyên bố kỹ thuật trong tài liệu này đều được liên kết trực tiếp tới file nguồn, mã nguồn thực thi, dữ liệu thẩm định và các bản ghi kiểm thử tự động trong kho mã nguồn `D:\\trainidentificationaudio`.
"""

exec_summary_content = """# Chương 00: Tóm tắt Điều hành (Executive Summary)

## 1. Bản chất Dự án & Mục tiêu Cốt lõi
Dự án Phát hiện Từ khóa Tiếng Việt (Vietnamese Keyword Spotting - KWS) được khởi xướng nhằm xây dựng một mô hình học sâu dạng Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) siêu nhẹ, chạy cục bộ (on-device edge AI) với độ trễ thấp để nhận diện 4 khẩu lệnh điều khiển robot/thiết bị bằng tiếng Việt:
1. `TRAI` (Trái - Left)
2. `PHAI` (Phải - Right)
3. `LEN` (Lên - Up)
4. `XUONG` (Xuống - Down)

Mọi âm thanh ngoài 4 khẩu lệnh này—bao gồm tạp âm môi trường, tiếng nói tự nhiên thông thường, các từ ngữ tiếng Việt phát âm gần giống (confusable words), hoặc các câu mệnh lệnh dài—đều phải bị từ chối dứt khoát vào lớp `REJECT` (hoặc phân loại vào `UNKNOWN` / `BACKGROUND_NOISE`).

## 2. Checkpoint Đông kết & Hiện trạng Đóng gói (Candidate Status)
Tại thời điểm hoàn tất giai đoạn [Phase 13.4](file:///d:/trainidentificationaudio/reports/PHASE13_4_FINAL_CNN_RELEASE.md), dự án đã chính thức đóng băng ứng viên phát hành cuối cùng:

> **Định danh Checkpoint:** `variant_v11_hard_negative_weighted_seed42.pt`  
> **Trạng thái chuẩn y:** `FROZEN_CNN_RELEASE_CANDIDATE`  
> **Đường dẫn Lưu trữ Chuẩn:** [`experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)  
> **Kích thước:** 479.524 bytes  
> **Mã băm SHA-256 (Byte-Exact):** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
> **Mô hình ONNX Native:** [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx) (SHA-256: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`)  
> **Ngưỡng Tối ưu (Authoritative Threshold):** $\\theta_{\\text{cmd}} = 0.20$

Mô hình đạt được tính tương đương toán học tuyệt đối giữa PyTorch và ONNX Runtime với độ lệch logit cực đại chỉ $\\Delta_{\\text{max}} = 3.58 \\times 10^{-6}$ (thỏa mãn ngưỡng sai số khắt khe $< 10^{-4}$).

## 3. Hiệu năng Thẩm định Thực chứng Cốt lõi
Hiệu năng kỹ thuật của mô hình được thẩm định qua 2 mốc độc lập lớn:
1. **Kiểm thử Khoá Phase 11 (Phase 11 Locked Test - 480 clips):**
   - **Command Recall:** 99.17% (119/120 clips lệnh kích hoạt chính xác)
   - **False Alarm Rate (FAR):** 0.63% (3/480 clips kích hoạt nhầm)
   - **Accuracy Tổng:** 98.75%
2. **Đánh giá Xác nhận Độc lập Đăng ký trước Phase 13.3 (Preregistered Synthetic Confirmatory Evaluation - 1.000 clips):**
   - Thử nghiệm trên 8 giọng đọc tổng hợp Vieneu giọng Bắc hoàn toàn mới chưa từng xuất hiện trong tập huấn luyện/kiểm thử trước đó, với các câu chứa từ ngữ âm học gây nhiễu chưa từng thấy (unseen lexical carriers).
   - **Độ chính xác Tổng thể (Accuracy):** 99.20% (992/1000 clips) [Vượt ngưỡng yêu cầu $\\ge 95.0\\%$]
   - **Độ nhạy Lệnh (Command Recall):** 99.00% (198/200 clips) [Vượt ngưỡng yêu cầu $\\ge 95.0\\%$]
     - `TRAI`: 49/50 (98.0%)
     - `PHAI`: 50/50 (100.0%)
     - `LEN`: 49/50 (98.0%)
     - `XUONG`: 50/50 (100.0%)
   - **Tỷ lệ Báo động Giả (Non-Target FAR):** 0.75% (6/800 clips) [Thỏa mãn yêu cầu khắt khe $\\le 2.0\\%$]
   - **Tỷ lệ Phân loại Nhầm giữa các Lệnh (Confusion Rate):** 0.00% (0/200 clips)
   - **Độ trễ Suy luận (CPU Latency trên Intel i7/Windows):** Mean = 1.80 ms, P95 = 2.12 ms (Budget cho phép $\\le 50.0$ ms).
   - **Kết luận:** Đạt 8/8 tiêu chí nghiệm thu tiên quyết (ALL GATES PASSED).

## 4. Các Giới hạn Vận hành Bắt buộc (Mandatory Operational Boundaries)
Nhà nghiên cứu và kỹ sư vận hành phải tuân thủ nghiêm ngặt 3 nguyên tắc pháp lý và kỹ thuật sau:
1. **Chưa thẩm định trên Giọng nói Tự nhiên (`NOT_VALIDATED_ON_NATURAL_SPEECH`):**
   Toàn bộ kết quả định lượng đạt được đều thực hiện trên âm thanh tổng hợp từ công cụ Vieneu TTS v3-Turbo kết hợp nhiễu nền ESC-50/DNS. Thử nghiệm nghe mù trên người ở Phase 90 chỉ có đúng **01 người đánh giá** hoàn thành (5 người khác chưa nộp phản hồi). Chưa có tập dữ liệu giọng người thật nói tự nhiên nào được thu âm có bản quyền/đồng thuận đầy đủ để kiểm định mô hình. Tuyệt đối không được quảng bá mô hình đã sẵn sàng cho giọng người thực tế.
2. **Chưa hỗ trợ Micro Mở Liên tục (`NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`):**
   Mô hình được thiết kế theo cơ chế **đoạn phát âm đóng khung (Bounded-Utterance)**: nhận đầu vào là đoạn âm thanh đúng 1.0 giây ($16.000$ mẫu). Mô hình **chưa tích hợp** bộ phát hiện tiếng nói (VAD - Voice Activity Detector) hay cơ chế cửa sổ trượt (sliding-window streaming). Nếu đưa luồng âm thanh micro liên tục mà không có cơ chế cắt câu chuẩn xác, mô hình sẽ gặp hiện tượng kích hoạt lặp hoặc báo động giả do nhiễu mép cửa sổ.
3. **Phạm vi Giọng Địa phương Hạn hẹp (Accent Scope Limitation):**
   Mô hình chỉ mới được xác nhận độc lập trên các biến thể phát âm thuộc **giọng Miền Bắc (Northern Accent)**. Giọng Miền Trung (Central) và Miền Nam (Southern) hoàn toàn chưa được kiểm định độc lập do thiếu giọng tổng hợp độc lập và chưa có dữ liệu thu âm người thật có đồng thuận.
"""

scope_content = """# Chương 01: Phạm vi Dự án & Yêu cầu Kỹ thuật (Project Scope & Requirements)

## 1. Định nghĩa Bài toán (Problem Statement)
Hệ thống Phát hiện Từ khóa (Keyword Spotting - KWS) là thành phần tiền xử lý giọng nói quan trọng trên thiết bị biên. Mục tiêu của dự án là nhận diện tức thời 4 lệnh định hướng trong tiếng Việt từ luồng âm thanh thu âm đơn kênh (mono) ở tần số lấy mẫu chuẩn $16\\text{ kHz}$:
- `TRAI` (/tʂaːj˧˦/ hoặc /tɕaːj˧˦/)
- `PHAI` (/faːj˨˩˦/)
- `LEN` (/lɛn˧˧/)
- `XUONG` (/sɨəŋ˧˩/ hoặc /ʂɨəŋ˧˩/)

Bên cạnh 4 lệnh hợp lệ, hệ thống phải liên tục giám sát và loại bỏ mọi âm thanh không mong muốn bằng cách phân loại chúng vào lớp từ chối (`REJECT` hoặc tập hợp các lớp phụ `UNKNOWN` và `BACKGROUND_NOISE`).

## 2. Bounded-Utterance vs. Streaming Inference
Một trong những bài học kỹ thuật quan trọng nhất của dự án là sự phân biệt rõ ràng giữa hai chế độ suy luận:

```mermaid
flowchart TD
    subgraph BoundedUtterance ["Chế độ Đoạn Bọc Đóng Khung (Bounded-Utterance) - ĐÃ ĐẠT CHUẨN"]
        A1["Âm thanh đầu vào"] --> B1["Cắt đúng 1.0 giây (16.000 mẫu)"]
        B1 --> C1["Chuẩn hóa biên độ (max abs > 1.0)"]
        C1 --> D1["Log-Mel Spectrogram (40 mel bins, N_fft=512)"]
        D1 --> E1["Mô hình CNN (variant_v11)"]
        E1 --> F1{"P(cmd) >= 0.20 ?"}
        F1 -->|"Có"| G1["Kích hoạt Lệnh (TRAI/PHAI/LEN/XUONG)"]
        F1 -->|"Không"| H1["Từ chối (REJECT)"]
    end

    subgraph Streaming ["Chế độ Luồng Micro Mở (Streaming Open-Mic) - CHƯA TRIỂN KHAI"]
        A2["Luồng Micro Liên tục (48/16 kHz)"] --> B2["Bộ đệm Vòng (Circular Ring Buffer)"]
        B2 --> C2["Bộ Phát hiện Giọng nói (VAD Engine)"]
        C2 --> D2["Cửa sổ trượt (Hop 100-200ms)"]
        D2 --> E2["Lọc Khử Nảy (Debounce & State Machine)"]
        E2 --> F2["Mô hình CNN KWS"]
        F2 --> G2["Trạng thái: NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"]
    end
```

### 2.1. Chế độ Bounded-Utterance (Hiện trạng Thẩm định)
- **Cấu trúc dữ liệu:** Mỗi mẫu kiểm thử là một tệp âm thanh đơn lẻ có thời lượng cố định $1.0\\text{ giây}$ ($16.000$ mẫu PCM 16-bit mono).
- **Căn chỉnh khẩu lệnh:** Khẩu lệnh mục tiêu nằm trọn vẹn trong khoảng thời gian này, được căn giữa hoặc phân bố ngẫu nhiên kèm khoảng lặng đệm (padding) tối thiểu $100\\text{ ms}$ ở hai đầu.
- **Tính toán xác suất:** Mô hình chỉ thực hiện một lần chuyển tiếp (forward pass) duy nhất trên ma trận Log-Mel Spectrogram kích thước $[1, 1, 40, 101]$ để đưa ra quyết định cuối cùng.

### 2.2. Chế độ Streaming Open-Mic (Chưa sẵn sàng)
- Trong môi trường thực tế, micro ghi âm liên tục và không có ranh giới bắt đầu/kết thúc từ trước.
- Để triển khai dạng streaming, hệ thống bắt buộc phải có:
  1. Module VAD để phát hiện khi nào người dùng bắt đầu nói và dừng nói.
  2. Cơ cấu cửa sổ trượt (sliding window) với bước nhảy (hop length) từ $100\\text{ ms}$ đến $250\\text{ ms}$.
  3. Bộ máy trạng thái (state machine) để khử kích hoạt lặp (debounce / non-maximum suppression) khi cùng một từ lệnh kéo dài qua nhiều khung hình trượt.
- **Kết luận hiện trạng:** Kho mã nguồn hiện nay mới chỉ xây dựng các tệp kịch bản kiểm thử giả lập sơ khai cho streaming ([`tests/test_kws_streaming.py`](file:///d:/trainidentificationaudio/tests/test_kws_streaming.py)), chưa có kiến trúc bộ đệm vòng (ring buffer) tối ưu hóa C++ hay bộ lọc khử nhiễu hoàn chỉnh. Do đó, hệ thống được gắn nhãn chính thức: `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION` đối với yêu cầu micro mở liên tục.

## 3. Sự Tiến hóa của Cơ chế Từ chối (Rejection Mechanism Evolution)

| Thế hệ Mô hình | Không gian Nhãn (Label Space) | Cơ chế Từ chối (Rejection Strategy) | Điểm yếu Cốt tử |
| :--- | :--- | :--- | :--- |
| **Giai đoạn Đầu (Phase 27 - 35)** | 4 Lớp (`TRAI`, `PHAI`, `LEN`, `XUONG`) | Softmax Argmax đơn thuần không có ngưỡng | Mô hình buộc phải gán nhãn 1 trong 4 lệnh cho mọi tiếng động (FAR $\\approx 100\\%$ khi có tạp âm). |
| **Giai đoạn Thêm Lớp Nền (Phase 36 - 60)** | 5 Lớp (`TRAI`, `PHAI`, `LEN`, `XUONG`, `NOISE`) | Softmax Argmax trên 5 lớp | Tạp âm môi trường được hấp thụ, nhưng tiếng nói người nói từ khác (speech out-of-vocabulary) vẫn bị kích hoạt nhầm. |
| **Giai đoạn 6 Lớp Phân tách (Phase 61 - 89)** | 6 Lớp (`TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, `BACKGROUND_NOISE`) | Softmax Argmax + Ngưỡng Margin $\\Delta P > 0.3$ | Phân biệt được tiếng nói lạ và tiếng ồn nền, tuy nhiên nhạy cảm với sự chênh lệch phân phối xác suất. |
| **Giai đoạn Hoàn thiện v11 - v13 (Hiện tại)** | 5 Lớp Logic / 2 Lớp Quyết định (`COMMAND` vs `REJECT`) | **Hard-Negative Weighted Cross Entropy + Ngưỡng Quyết định $\\theta_{\\text{cmd}} = 0.20$** | **Khắc phục triệt để:** Bắt buộc $P(\\text{cmd}) \\ge \\theta_{\\text{cmd}}$ VÀ $P(\\text{cmd}) = \\max_j P(c_j)$. Giảm tỷ lệ FAR từ $7.5\\%$ xuống $0.63\\% - 0.75\\%$. |

## 4. Yêu cầu Kỹ thuật Cốt lõi (System Targets)
Theo đặc tả thiết kế hệ thống KWS công nghiệp:
- **Command Recall (Độ nhạy nhận lệnh):** $\\ge 95.0\\%$ trên tất cả các lệnh mục tiêu độc lập.
- **False Alarm Rate (Tỷ lệ báo động giả):** $\\le 1.0\\%$ (chấp nhận tối đa $\\le 2.0\\%$ trong các điều kiện kiểm thử nhiễu khắc nghiệt).
- **Inference Latency (Độ trễ suy luận):** $\\le 20.0\\text{ ms}$ trên CPU ARM Cortex-A / x86 phổ thông; không vượt quá $50.0\\text{ ms}$.
- **Bộ nhớ Mô hình (Model Footprint):** Kích thước file trọng số $< 2.0\\text{ MB}$; RAM hoạt động $< 10.0\\text{ MB}$.
"""

with open(os.path.join(DOCS_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "00_EXECUTIVE_SUMMARY.md"), "w", encoding="utf-8") as f:
    f.write(exec_summary_content.strip() + "\n")

with open(os.path.join(DOCS_DIR, "01_PROJECT_SCOPE_AND_REQUIREMENTS.md"), "w", encoding="utf-8") as f:
    f.write(scope_content.strip() + "\n")

print("[OK] Generated README.md, 00_EXECUTIVE_SUMMARY.md, and 01_PROJECT_SCOPE_AND_REQUIREMENTS.md")
