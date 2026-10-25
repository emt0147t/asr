# Kế Hoạch Đăng Ký Trước Thử Nghiệm Phân Đoạn VAD & Dòng Lệnh
## (Phase 15 — Pre-Registration of Controlled VAD & Streaming Investigation Protocol)

> **Mã tài liệu:** `PHASE15_VAD_PREREGISTRATION_20261024`  
> **Repository:** `D:\trainidentificationaudio`  
> **Ngày đăng ký trước (Pre-registered Date):** 2026-10-24  
> **Giai đoạn nghiên cứu:** Phase 15 — Offline Controlled VAD & Streaming Investigation  
> **Trạng thái cổng tiên quyết (Prerequisite Gate):** `PASS_TO_VAD_DESIGN` (Phase 14)  
> **Quy chế thực thi:** Giao thức kiểm soát cô lập ngoại tuyến (Isolated Offline Research Protocol), đăng ký trước phương pháp, số liệu và tiêu chí lựa chọn TRƯỚC KHI tiến hành đo đạc hoặc đánh giá.

---

## 1. Ranh Giới Quản Trị & Cam Kết Khoa Học Tuyệt Đối

Báo cáo đăng ký trước này tuân thủ vô điều kiện 4 nhãn quản trị bất biến:
1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
4. `NOT_VALIDATED_PENDING_DATA`

### Cam Kết Ràng Buộc Trong Phase 15:
- **Tuyệt đối không huấn luyện lại hay tinh chỉnh (No retraining / No fine-tuning):** Không tác động vào trọng số v11, E1 hay khởi động E2.
- **Bảo toàn nguyên bản mô hình và mã băm mật mã:**
  - Checkpoint v11 (`variant_v11_hard_negative_weighted_seed42.pt`): `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`.
  - ONNX v11 (`vietnamese_kws_v11.onnx`): `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`.
  - Checkpoint E1 (`exp_v12_e1_baseline_scale_seed42.pt`): `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878`.
- **Bảo toàn demo mặc định:** Không sửa đổi tệp mã nguồn `live_mic_engine.py` hay gói phát hành Phase 13.4. Toàn bộ mã nguồn Phase 15 nằm độc lập tại thư mục `experiments/vieneu_kws_research_v12/phase15_vad/`.
- **Giữ nguyên hợp đồng suy luận canonical:** Ngưỡng lệnh cố định $\theta_{\text{cmd}} = 0.20$, phân loại 5 lớp đồng thời với `REJECT` tham gia vào argmax, frontend Mel 40-chiều giữ nguyên.
- **Tuyệt đối chống rò rỉ và ô nhiễm benchmark:** Không sử dụng tập test 1,280 mẫu đã bị quan sát hoặc 40 chuỗi streaming cũ để tinh chỉnh tham số VAD. Tập kiểm thử `vad_test_corpus` được khóa mã băm SHA-256 trước khi đánh giá và chỉ được chạy một lần duy nhất (one-time locked evaluation) sau khi đã chọn xong cấu hình tối ưu trên `vad_dev_corpus`.
- **Không thu âm giọng người mới mà chưa có phê duyệt chính thức.**
- **Không tuyên bố hệ thống sẵn sàng triển khai thực tế (No production readiness claim).**

---

## 2. Câu Hỏi Nghiên Cứu & Giả Thuyết Khoa Học (Research Questions & Hypotheses)

### 2.1 Câu Hỏi Cốt Lõi (Primary Research Question)
Nguyên nhân gốc rễ dẫn đến thất bại trong việc nhận diện chuỗi lệnh liên tục (streaming multi-command failures) là do:
1. **Lỗi phân đoạn âm thanh (VAD Segmentation Error):** Bộ tách phân đoạn gộp các từ phát âm liền kề thành một đoạn duy nhất (Merge Error), hoặc cắt non khi từ chưa dứt (Premature Cut), hoặc kích hoạt ảo do nhiễu (Spurious Trigger);
2. **Lỗi phân loại của mô hình CNN (Classification Error):** Bản thân mô hình nhận nhầm từ lệnh sang lớp khác hoặc từ chối nhầm vào `REJECT`;
3. **Hay cả hai yếu tố cùng tương tác cộng hưởng?**

### 2.2 Các Giả Thuyết Đăng Ký Trước (Pre-registered Hypotheses)
- **Giả thuyết H1 (VAD Hangover Bottleneck):**  
  Thuật toán VAD hiện tại với tham số thời gian chờ im lặng cố định $300\text{ ms}$ (`silence_interval_ms = 300`) là nguyên nhân trực tiếp gây ra $100\%$ lỗi dính cụm (Merge Rate $\approx 100\%$) khi khoảng cách phát âm giữa 2 từ liên tiếp nhỏ hơn $250\text{ ms}$ (đặc biệt tại nhóm $100\text{ ms}, 150\text{ ms}, 200\text{ ms}$).
- **Giả thuyết H2 (Adaptive Hangover Disentanglement):**  
  Cơ chế VAD hai ngưỡng kết hợp thời gian chờ thích ứng (Dual-Threshold Adaptive Hangover) có khả năng phân tách chính xác các từ phát âm ở khoảng cách ngắn ($100 - 250\text{ ms}$), làm giảm tỷ lệ dính cụm xuống dưới $20\%$ mà không làm tăng tỷ lệ cắt non trên các từ có âm kéo dài.
- **Giả thuyết H3 (Speech-Band Formant Suppression):**  
  Theo dõi tỷ số năng lượng tập trung trong dải formant tiếng nói ($300 - 3400\text{ Hz}$) giúp loại trừ kích hoạt ảo do nhiễu môi trường dải rộng (quạt gió, tiếng ù biến áp, tiếng thở micro) so với bộ phát hiện năng lượng toàn dải đơn thuần.
- **Giả thuyết H4 (Modularity of Segmentation vs Classification):**  
  Độ chính xác phân loại của mô hình CNN đông băng trên các phân đoạn đã được VAD cắt thành công (IoU $\ge 0.70$) sẽ duy trì ở mức cao tương đương đánh giá offline cô lập ($\ge 90\%$), chứng minh rằng cải thiện VAD là điều kiện cần và đủ để giải quyết phần lớn lỗi bỏ lỡ trong luồng đa lệnh.

---

## 3. Không Gian Thuật Toán Ứng Viên & Tham Số Cấu Hình

Ba thuật toán phân đoạn được khảo sát độc lập:

### 3.1 Ứng Viên A: Baseline Hiện Tại (Fixed-Silence Energy Gate VAD)
Mô phỏng chính xác logic phân đoạn của `UtteranceSegmenter` trong `live_mic_engine.py`:
- Kích thước khối âm thanh: $20\text{ ms}$ ($320\text{ samples}$ @ $16\text{ kHz}$).
- Vùng đệm đầu (Pre-roll buffer): $150\text{ ms}$ ($2,400\text{ samples}$).
- Ngưỡng năng lượng kích hoạt tĩnh: $-48.0\text{ dB}$.
- Bám vết sàn nhiễu động: $\text{floor} \leftarrow 0.95 \times \text{floor} + 0.05 \times \text{db}$.
- Ngưỡng kích hoạt hiệu dụng: $\text{thresh} = \max(-48.0, \min(\text{floor} + 5.0, -32.0))$.
- Thời gian chờ im lặng cố định: `silence_interval_ms = 300` ($4,800\text{ samples}$).
- Thời lượng tiếng nói tối thiểu: $120\text{ ms}$ ($1,920\text{ samples}$).
- Thời lượng phân đoạn tối đa: $1200\text{ ms}$ ($19,200\text{ samples}$).

### 3.2 Ứng Viên B: Dual-Threshold Adaptive Hangover VAD
- Kích thước khối âm thanh: $20\text{ ms}$ ($320\text{ samples}$).
- Pre-roll buffer: $150\text{ ms}$ ($2,400\text{ samples}$).
- Ước lượng sàn nhiễu bất đối xứng (Asymmetric Noise Tracking):  
  Nếu $\text{db} < \text{floor}$: $\text{floor} \leftarrow 0.85 \times \text{floor} + 0.15 \times \text{db}$ (hạ nhanh khi yên tĩnh).  
  Nếu $\text{db} \ge \text{floor}$: $\text{floor} \leftarrow 0.98 \times \text{floor} + 0.02 \times \text{db}$ (tăng rất chậm tránh bám vào tiếng nói).
- Ngưỡng trễ (Hysteresis Thresholds):
  - Ngưỡng bật (Onset / Speech Start): $\text{thresh}_{\text{onset}} = \text{clip}(\text{floor} + 10.0\text{ dB}, -45.0\text{ dB}, -30.0\text{ dB})$.
  - Ngưỡng tắt (Offset / Speech End): $\text{thresh}_{\text{offset}} = \text{clip}(\text{floor} + 4.0\text{ dB}, -52.0\text{ dB}, -40.0\text{ dB})$.
- Thời gian chờ thích ứng (Adaptive Hangover):
  - Thiết lập cơ sở: `hangover_base_ms = 180` ($180\text{ ms}$).
  - Thích ứng động: Khi năng lượng xung quanh duy trì SNR cao ($\ge 15\text{ dB}$), giảm hangover xuống $140-160\text{ ms}$ để đón đầu các khoảng nghỉ hẹp; khi SNR thấp, nới hangover lên $200-240\text{ ms}$.
- Thời lượng tiếng nói tối thiểu: $100\text{ ms}$.
- Thời lượng phân đoạn tối đa: $1200\text{ ms}$.

### 3.3 Ứng Viên C: Speech-Band Energy-Ratio VAD (Heuristic Dải Formant Tiếng Nói)
Được định danh chính xác là **Exploratory Speech-Band Energy Ratio Heuristic**:
- Chiều dài khung: $25\text{ ms}$ ($400\text{ samples}$), bước nhảy $10\text{ ms}$ ($160\text{ samples}$), cửa sổ Hanning, FFT 512 điểm.
- Dải tần tiếng nói: $300\text{ Hz} - 3400\text{ Hz}$.
- Tỷ số năng lượng dải formant:
  $$R_{\text{speech}} = \frac{\sum_{f=300}^{3400} |X(f)|^2}{\sum_{f=0}^{8000} |X(f)|^2 + 10^{-8}}$$
- Tiêu chí kích hoạt kép: Một khung được coi là tiếng nói khi:
  $$\text{RMS}_{\text{dB}} \ge \text{thresh}_{\text{energy}} \quad \text{VÀ} \quad R_{\text{speech}} \ge 0.32$$
- Bộ đếm làm mượt (Hangover frames): 8 khung ($80\text{ ms}$).
- Pre-roll buffer: $150\text{ ms}$.
- Phân đoạn kết thúc khi $R_{\text{speech}}$ liên tục dưới ngưỡng quá $160\text{ ms}$.

---

## 4. Thiết Kế Tập Dữ Liệu Kiểm Soát VAD (Corpus Construction & Split Policy)

Nhằm đảm bảo tính độc lập khoa học tuyệt đối và tránh ô nhiễm dữ liệu, hai tập âm thanh mới được sinh theo quy chế phân vùng nghiêm ngặt từ `split_policy.json`:

### 4.1 Quy Mô & Phân Bố Khoảng Nghỉ (Gap Calibration)
Mỗi tập bao gồm chính xác **60 chuỗi âm thanh liên tục** (tổng cộng 120 chuỗi cho 2 tập), phân bố cân bằng trên 6 khoảng nghỉ danh định:

| Khoảng Nghỉ Danh Định (Nominal Gap) | Số Chuỗi Trong `vad_dev_corpus` | Số Chuỗi Trong `vad_test_corpus` | Mục Đích Đo Lường |
| :---: | :---: | :---: | :--- |
| **`100 ms`** | 10 | 10 | Kiểm tra khả năng tách từ khi nói cực nhanh / dồn dập |
| **`150 ms`** | 10 | 10 | Đánh giá vùng nghẽn của ngưỡng 300ms |
| **`200 ms`** | 10 | 10 | Đo lường vùng quá độ dính cụm |
| **`250 ms`** | 10 | 10 | Ranh giới chuyển tiếp tách từ chuẩn |
| **`300 ms`** | 10 | 10 | Điểm cân bằng của baseline |
| **`400 ms`** | 10 | 10 | Khoảng nghỉ tự nhiên tiêu chuẩn |
| **Tổng cộng** | **60 chuỗi** | **60 chuỗi** | **120 chuỗi hoàn toàn độc lập** |

### 4.2 Cân Bằng Môi Trường Nhiễu Âm Học
Mỗi nhóm 10 chuỗi tại mỗi khoảng cách được phân bổ trên 5 mức nhiễu:
- Clean: 2 chuỗi (20%)
- SNR 20 dB: 2 chuỗi (20%)
- SNR 15 dB: 2 chuỗi (20%)
- SNR 10 dB: 2 chuỗi (20%)
- SNR 5 dB: 2 chuỗi (20%)

### 4.3 Cách Ly Tuyệt Đối Nguồn Dữ Liệu (Split Isolation Rules)
1. **Giọng đọc (Voice Disjoint):**
   - `vad_dev_corpus`: Chỉ sử dụng 4 giọng độc quyền của tập Val (`Mai Anh`, `Ngọc Trân`, `Phạm Tuyên`, `Thùy Dung`).
   - `vad_test_corpus`: Chỉ sử dụng 4 giọng độc quyền của tập Test (`Kim Thanh`, `Quốc Tuấn`, `Quỳnh Anh`, `Đức Trí`).
   - Giao thoa giọng đọc: $\text{Dev Voices} \cap \text{Test Voices} = \emptyset$.
2. **Từ vựng OOV xen kẽ (OOV Lexeme Disjoint):**
   - `vad_dev_corpus`: Chỉ dùng OOV tập Val (`đi`, `đứng`, `nghe`, `nói`, `hỏi`, `tìm`, `xoay`, `lùi`).
   - `vad_test_corpus`: Chỉ dùng OOV tập Test (`ghế`, `xúc`, `xinh`, `đếm`, `cân`, `đo`, `trưa`, `trà`).
   - Giao thoa từ OOV: $\text{Dev OOV} \cap \text{Test OOV} = \emptyset$.
3. **Bản ghi nhiễu môi trường (Noise Asset Disjoint):**
   - `vad_dev_corpus`: Chỉ dùng các track nhiễu Val (indices `28, 29, 30, 31, 32, 33`).
   - `vad_test_corpus`: Chỉ dùng các track nhiễu Test (indices `34, 35, 36, 37, 38, 39`).
   - Giao thoa track nhiễu: $\text{Dev Noise} \cap \text{Test Noise} = \emptyset$.

### 4.4 Quy Trình Cắt Tỉa Khoảng Lặng & Đo Lường Khoảng Nghỉ Âm Học Thực Tế
- Để tránh sai số do khoảng lặng đầu/đuôi (leading/trailing silence) của tệp gốc làm sai lệch khoảng nghỉ danh định, quy trình sinh áp dụng hàm cắt tỉa có kiểm toán (`auditable trimming`):
  - Xác định ranh giới âm học hoạt động dựa trên ngưỡng năng lượng $-40\text{ dB}$ và bọc viền đệm nhỏ $20\text{ ms}$ ($320\text{ samples}$).
  - Ghi nhận độc lập cả hai giá trị: `nominal_gap_ms` (khoảng cách chèn) và `effective_acoustic_gap_ms` (khoảng cách thực tế giữa điểm kết thúc tiếng nói từ $N$ và điểm bắt đầu tiếng nói từ $N+1$).

---

## 5. Phương Pháp & Tiêu Chí Đánh Giá Hai Giai Đoạn (Two-Stage Evaluation Protocol)

### 5.1 Giai Đoạn A: Đánh Giá Phân Đoạn Thuần Túy (Stage A — Segmentation-Only Evaluation)
Đánh giá độ chính xác của ranh giới thời gian mà VAD tạo ra so với mốc thời gian sự kiện thực tế (Ground-Truth Reference Intervals $[t_{\text{start}}, t_{\text{end}}]$), **hoàn toàn không liên quan đến nhãn dự đoán của mô hình**:

- **Quy tắc khớp 1-đối-1 (One-to-One Matching Rule):**
  - Tính chỉ số trùng lặp theo thời gian (Temporal Intersection over Union - IoU):
    $$\text{IoU}(S_{\text{pred}}, R_{\text{gt}}) = \frac{\max(0, \min(t_{\text{pred\_end}}, t_{\text{gt\_end}}) - \max(t_{\text{pred\_start}}, t_{\text{gt\_start}}))}{\max(t_{\text{pred\_end}}, t_{\text{gt\_end}}) - \min(t_{\text{pred\_start}}, t_{\text{gt\_start}})}$$
  - Một phân đoạn được coi là khớp (Matched) khi $\text{IoU} \ge 0.50$ với đúng 1 sự kiện chuẩn. Mỗi sự kiện chuẩn chỉ được khớp tối đa với 1 phân đoạn dự đoán.
- **Các Chỉ Số Bắt Buộc Báo Cáo:**
  1. `Event Precision`, `Event Recall`, `Event F1`.
  2. `Merge Count` & `Merge Rate`: Số lượng và tỷ lệ phân đoạn dự đoán bao trùm $> 1$ sự kiện thực tế (lỗi dính cụm).
  3. `Split Count` & `Split Rate`: Số lượng sự kiện thực tế bị cắt thành $\ge 2$ phân đoạn dự đoán (lỗi vỡ mảnh).
  4. `Missed Event Count`: Số sự kiện thực tế không được phân đoạn nào bao phủ.
  5. `Premature Cut Count`: Số phân đoạn bị ngắt trước khi sự kiện thực tế kết thúc ($t_{\text{pred\_end}} < t_{\text{gt\_end}} - 80\text{ ms}$).
  6. `Segment Count Bias`: $\Delta N = N_{\text{pred}} - N_{\text{gt}}$.
  7. `Onset / Offset Absolute Error (MAE)`: Sai số tuyệt đối của thời điểm bắt đầu và kết thúc (báo cáo giá trị trung vị Median và phân vị P95 tính bằng mili-giây).

### 5.2 Giai Đoạn B: Đánh Giá Phân Đoạn Kết Hợp Mô Hình Đóng Băng (Stage B — End-to-End Evaluation)
Nạp các phân đoạn do VAD tạo ra vào đường ống suy luận chính thức (v11 và E1):
- Ngưỡng lệnh: $\theta_{\text{cmd}} = 0.20$.
- 5 lớp tham gia argmax đầy đủ.
- Định dạng phân đoạn: chuẩn hóa đúng 1.0 giây ($16,000\text{ samples}$), độ dài thực tế được chèn đệm số 0 về phía sau theo đúng hợp đồng Phase 13.4.
- **Các Chỉ Số Bắt Buộc Báo Cáo:**
  1. `Exact-Sequence Accuracy`: Tỷ lệ chuỗi có toàn bộ thứ tự lệnh nhận diện đúng 100%.
  2. `Command Precision`, `Recall`, `F1`.
  3. `Missed Commands` & `Substitutions`.
  4. `False Command Activations on OOV`: Số lần nhận nhầm từ OOV thành lệnh.
  5. `False Command Activations during Silence`: Số lần kích hoạt ảo trên nền im lặng/nhiễu.
  6. `Duplicate Triggers`: Số lần kích hoạt lặp lại trên cùng 1 từ.
  7. `Trigger Latency`: Độ trễ từ mốc bắt đầu nói thực tế đến khi VAD hoàn tất phân đoạn (Median, P95).
  8. `False Activations Per Minute`: Tần suất kích hoạt sai trên phút âm thanh âm bản.

---

## 6. Quy Tắc Lựa Chọn Thuật Toán Trên Dev & Giao Thức Khóa Test

### 6.1 Tiêu Tụ Tuyển Chọn Tiên Quyết Trên `vad_dev_corpus`
Mọi so sánh thuật toán và tinh chỉnh tham số chỉ được thực hiện trên tập `vad_dev_corpus`. Quy tắc lựa chọn được đăng ký trước theo thứ tự ưu tiên:
1. **Tiêu chuẩn số 1 (Primary Criterion):** Tối đa hóa `Event-Level Segmentation F1` ở Giai đoạn A.
2. **Tiêu chuẩn phá vỡ thế cân bằng 1 (Tie-breaker 1):** Tối thiểu hóa `Merge Rate` tại các khoảng nghỉ ngắn ($\le 250\text{ ms}$).
3. **Tiêu chuẩn phá vỡ thế cân bằng 2 (Tie-breaker 2):** Tối thiểu hóa `False Command Activations` (trên cả OOV và Silence) ở Giai đoạn B.
4. **Tiêu chuẩn phá vỡ thế cân bằng 3 (Tie-breaker 3):** Tối thiểu hóa độ trễ kích hoạt `P95 Trigger Latency`.

### 6.2 Giao Thức Đánh Giá Khóa Một Lần Trên `vad_test_corpus`
- Sau khi xác định thuật toán và cấu hình chiến thắng trên Dev, xuất tệp cấu hình JSON và khóa mã băm SHA-256.
- Thực thi đúng **MỘT LẦN DUY NHẤT** trên `vad_test_corpus` cho:
  - Cấu hình Baseline Hiện Tại (A).
  - Cấu hình Chiến Thắng Tuyển Chọn (Winning Candidate).
- Toàn bộ kết quả đầu ra được bảo lưu nguyên vẹn, **tuyệt đối không điều chỉnh lại bất kỳ siêu tham số nào sau khi đã nhìn thấy kết quả Test**.

---

## 7. Điều Kiện Thất Bại & Kích Hoạt Trạng Thái BLOCKED

Trạng thái kiểm toán sẽ chuyển sang `BLOCKED_PENDING_EVIDENCE` hoặc `FAIL_EXPERIMENT` nếu xảy ra bất kỳ điều kiện nào sau đây:
1. **Rò rỉ phân vùng (Data Leakage):** Phát hiện bất kỳ sự trùng lặp nào về giọng đọc, từ vựng OOV hoặc file nhiễu giữa Dev và Test.
2. **Sai lệch mã băm mật mã (Cryptographic Integrity Failure):** Mã băm của checkpoint v11, ONNX v11, hoặc checkpoint E1 không khớp với giá trị kỳ vọng.
3. **Suy thoái nghiêm trọng (Performance Regression):** Ứng viên mới làm giảm `Command Precision` trên tập Dev xuống dưới $70\%$ hoặc làm tăng `False Activations` gấp đôi so với Baseline.
4. **Lỗi giao diện đường ống (Interface Breakage):** Không thể trích xuất phân đoạn hoặc nạp vào frontend mà không làm biến dạng tín hiệu âm thanh 16 kHz.

---

## 8. Chữ Ký Khóa Giao Thức Đăng Ký Trước (Protocol Lock Signature)

- **Mã định danh giao thức:** `PHASE15_VAD_PROTOCOL_v1.0`
- **Môi trường thực thi:** Python 3.11.9, PyTorch 2.6.0+cu124, SoundFile 0.13.1, NumPy 1.26.4.
- **Thời điểm đóng dấu giao thức:** 2026-10-24 10:30:00 (GMT+7).
- **Trạng thái:** **PROTOCOL OFFICIALLY LOCKED — PROCEED TO CORPUS GENERATION**.
