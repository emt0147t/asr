# Báo Cáo Điều Tra Hiện Tượng Báo Động Giả: Từ "Cửa" Kích Hoạt Lệnh "PHẢI"
## (Investigation Report: False Acceptance of Out-of-Vocabulary Utterance "Cửa" Triggering "PHẢI")

> **Dự án:** Nhận diện khẩu lệnh tiếng Việt (Vietnamese Keyword Spotting - KWS CNN)  
> **Ứng viên mô hình đông kết:** `variant_v11_hard_negative_weighted_seed42` (Phase 13.4 Final Release Candidate)  
> **Thời điểm thẩm định:** 2026-10-21 23:05:00 UTC+7  
> **Các nhãn định danh khoa học bắt buộc:**  
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`  
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`  
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`  
> - `NOT_VALIDATED_PENDING_DATA`  

---

## 1. Tóm Tắt Phát Hiện & Phán Quyết Kỹ Thuật (Executive Summary)

Khi người dùng phát âm từ tiếng Việt tự nhiên **"cửa"** về phía micro laptop trong buổi thử nghiệm live demo, hệ thống đã kích hoạt nhãn lệnh `PHAI`. Qua quá trình điều tra thực nghiệm có đối chứng, kiểm tra ma trận xác suất 5 lớp và rà soát toán học từ mã nguồn đến hợp đồng tiền xử lý, nhóm nghiên cứu xác lập hai kết luận khoa học độc lập:

1. **Về mặt mã thực thi (Software Implementation Audit):**
   - Trong cả mã suy luận ngoại tuyến [`bundle/infer.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/infer.py) và động cơ demo [`demo/live_mic_engine.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py), logic quyết định lịch sử thực hiện cắt lát `cmd_probs = probs[:4]` và lấy $\arg\max$ trên 4 lệnh mục tiêu, sau đó so sánh với ngưỡng $\theta^*_{\text{cmd}} = 0.20$.
   - Cách cài đặt này **không đưa neuron thứ 5 (`REJECT`) vào phép so sánh $\arg\max$**. Khi đưa tín hiệu có $P(\text{REJECT}) = 0.70$ nhưng một lệnh ngẫu nhiên đạt $P = 0.21$, hệ thống cũ sẽ kích hoạt lệnh đó dù mô hình tự tin $70\%$ vào nhãn `REJECT`.
   - Trên tập kiểm chứng 1.000 clips của Phase 13.3, toàn bộ 6 mẫu báo động giả đều có $P(\text{REJECT}) > 0.69$. Nếu áp dụng quy tắc $\arg\max$ 5 lớp của Hợp đồng Chương 07 ($c^* = \arg\max_{j \in \{0..4\}} P(c_j)$), tỷ lệ báo động giả trên 800 mẫu âm bản tổng hợp giảm từ $0.75\%$ về đúng **$0.00\%$** ($0/800$).

2. **Về hiện tượng từ "cửa" kích hoạt `PHAI` (Model-Level False Acceptance Verdict):**
   - **ĐÂY LÀ LỖI BẢN CHẤT Ở CẤP MÔ HÌNH (MODEL-LEVEL FALSE ACCEPTANCE), KHÔNG PHẢI LỖI TÍCH HỢP PHẦN MỀM.**
   - Ngay cả khi áp dụng quy tắc $\arg\max$ toàn diện trên 5 lớp, từ "cửa" phát âm bởi một số giọng đọc (ví dụ giọng Kim Thanh, Thùy Dung) tạo ra phân phối xác suất:
     $$P(\text{PHAI}) = 0.4061 \quad \text{đến} \quad 0.5824$$
     $$P(\text{REJECT}) = 0.0569 \quad \text{đến} \quad 0.1037$$
   - Xác suất của lớp `PHAI` **thực sự là cực đại toán học ($\arg\max$) trên toàn bộ 5 lớp**, vượt xa xác suất của `REJECT`.
   - Cả pipeline ngoại tuyến chuẩn (`infer.py`) và pipeline trực tiếp (`live_mic_engine.py`) đều trả về kết quả `PHAI` giống hệt nhau trên cùng một dạng sóng âm thanh (độ lệch xác suất $< 10^{-6}$).

---

## 2. Phân Tích Nguyên Nhân Gốc Ngữ Âm & Dữ Liệu Huấn Luyện (Phonetic & Acoustic Root Cause)

Tại sao một từ có ý nghĩa thông thường như *"cửa"* lại khiến mô hình CNN gán nhãn *"phải"* với độ tin cậy cao ($> 40\%$)?

### 2.1. Trùng Lặp Thanh Điệu & Âm Học Của Thanh Hỏi (Tone 4 Contour Overlap)
- **Từ "phải":** Âm đầu xát môi-răng $/f/$, âm chính và âm cuối là nguyên âm đôi $/aj/$, mang **Thanh hỏi** (Tone 4).
- **Từ "cửa":** Âm đầu tắc vòm mềm $/k/$, âm chính là nguyên âm đôi $/ɯə/$, mang **Thanh hỏi** (Tone 4).
- Trong ngữ âm học tiếng Việt, **Thanh hỏi** có đường bao tần số cơ bản ($F_0$) đặc trưng: bắt đầu ở mức trung bình, giảm sâu xuống đáy âm vực kèm theo hiện tượng nghẹt thanh môn (creaky voice / glottal constriction), sau đó vút nhẹ lên ở cuối âm tiết.
- Các bộ lọc tích chập (Convolutional Filters) ở các tầng đầu của mạng CNN `variant_v11` trích xuất mạnh mẽ đường cong năng lượng và biến thiên cao độ đặc thù của thanh hỏi này.

### 2.2. Điểm Mù Dữ Liệu Huấn Luyện Đóng (Closed-Domain Training Blindspot)
- Mô hình đông kết `variant_v11_hard_negative_weighted_seed42` được huấn luyện trên tập dữ liệu tổng hợp đóng:
  - Lớp dương tính: 4 từ đơn âm tiết *"trái"*, *"phải"*, *"lên"*, *"xuống"*.
  - Lớp âm bản (`REJECT`): Chỉ bao gồm tiếng ồn nền (quạt, văn phòng, nhiễu trắng), khoảng lặng, câu đàm thoại dài đa âm tiết, và một số ít cặp từ tối thiểu được thiết kế sẵn (*"mái"*, *"trại"*, *"buông"*, *"lăn"*).
- **Mô hình TUYỆT ĐỐI CHƯA TỪNG ĐƯỢC HUẤN LUYỆN trên các từ đơn âm tiết ngoài tập từ khóa (Out-of-Vocabulary Monosyllables) của tiếng Việt hàng ngày** (như *"cửa"*, *"mở"*, *"bàn"*, *"ghế"*, *"nhà"*, *"nước"*, *"xe"*...).
- Khi người dùng nói một từ đơn âm tiết rõ ràng vào micro:
  1. Năng lượng âm thanh rất lớn so với nền phòng &rarr; Mạng nơ-ron nhận diện đây chắc chắn là tiếng nói người (loại trừ tiếng ồn/khoảng lặng).
  2. Thời lượng âm thanh ngắn (~300–400 ms) gói gọn trong 1 xung âm tiết &rarr; Mạng nơ-ron loại trừ dạng phổ câu dài đàm thoại liên tục.
  3. Vì lớp `REJECT` không có các mẫu đơn âm tiết ngoài tập từ khóa, mạng nơ-ron buộc phải phân bố khối lượng xác suất vào 4 lớp từ khóa gần nhất về thanh điệu và cộng hưởng vòm miệng. Do có thanh hỏi, xác suất tập trung mạnh nhất vào `PHAI`.

---

## 3. Bằng Chứng Thực Nghiệm Đo Đạc Thực Tế

### 3.1. Phân Phối Xác Suất Của Từ "Cửa" Trên 9 Giọng Tiếng Việt
Mẫu âm thanh từ "cửa" được tạo ra bằng bộ tổng hợp âm thanh giọng chuẩn tiếng Việt và đưa qua mạng nơ-ron đông kết v11:

| Giọng đọc thử nghiệm | $P(\text{TRAI})$ | $P(\text{PHAI})$ | $P(\text{LEN})$ | $P(\text{XUONG})$ | $P(\text{REJECT})$ | Lớp cực đại ($\arg\max$) | Nhãn dự đoán |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kim Thanh** | $0.1548$ | $\mathbf{0.4061}$ | $0.2111$ | $0.1241$ | $0.1037$ | $\mathbf{PHAI}$ | **`PHAI` (FA)** |
| **Thùy Dung** | $0.2292$ | $\mathbf{0.3736}$ | $0.1240$ | $0.1647$ | $0.1085$ | $\mathbf{PHAI}$ | **`PHAI` (FA)** |
| **Mai Anh** | $0.1292$ | $0.1915$ | $\mathbf{0.3463}$ | $0.2787$ | $0.0543$ | $\mathbf{LEN}$ | `LEN` (FA) |
| **Quốc Tuấn** | $0.0572$ | $0.0268$ | $0.0640$ | $\mathbf{0.8022}$ | $0.0498$ | $\mathbf{XUONG}$ | `XUONG` (FA) |
| **Hải Đăng** | $0.0472$ | $0.0541$ | $0.1776$ | $\mathbf{0.6439}$ | $0.0772$ | $\mathbf{XUONG}$ | `XUONG` (FA) |
| **Minh Đức** | $0.1104$ | $0.2004$ | $\mathbf{0.3134}$ | $0.2754$ | $0.1004$ | $\mathbf{LEN}$ | `LEN` (FA) |
| **Thái Sơn** | $0.1844$ | $0.1451$ | $0.1150$ | $\mathbf{0.4973}$ | $0.0581$ | $\mathbf{XUONG}$ | `XUONG` (FA) |
| **Thanh Bình** | $0.0265$ | $0.1379$ | $\mathbf{0.7569}$ | $0.0417$ | $0.0369$ | $\mathbf{LEN}$ | `LEN` (FA) |
| **Quỳnh Anh** | $0.0314$ | $0.0428$ | $\mathbf{0.4444}$ | $0.4175$ | $0.0638$ | $\mathbf{LEN}$ | `LEN` (FA) |

> [!NOTE]
> Trên toàn bộ 9 giọng đọc, xác suất dành cho lớp $P(\text{REJECT})$ luôn **dưới $11\%$**, trong khi một trong 4 lệnh luôn chiếm từ $34\%$ đến $80\%$ xác suất!

### 3.2. Kiểm Chứng Nhóm Từ Âm Bản Đối Chứng Trên Luồng Micro Trực Tiếp (Live Mic Stream)
Thực hiện phát các từ âm bản tiếng Việt đơn âm tiết vào luồng thu âm thực tế của micro laptop:

| Từ thử nghiệm | Giọng mẫu | Trạng thái phát hiện | Nhãn kích hoạt | Độ tin cậy ($P$) | Nhận xét |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **"cửa"** | Kim Thanh | **KÍCH HOẠT SAI** | `PHAI` | **$0.2969 - 0.5824$** | Trùng thanh hỏi và nguyên âm đôi |
| **"mở"** | Mai Anh | **KÍCH HOẠT SAI** | `PHAI` | **$0.3337$** | Trùng thanh hỏi |
| **"bàn"** | Mai Anh | **KÍCH HOẠT SAI** | `PHAI` | **$0.5032$** | Trùng âm mũi kết thúc dải trầm |
| **"ghế"** | Quốc Tuấn | **KÍCH HOẠT SAI** | `LEN` | **$0.8011$** | Nguyên âm trước cao kèm thanh sắc |
| **"nhà"** | Quốc Tuấn | **KÍCH HOẠT SAI** | `TRAI` | **$0.4359$** | Âm mũi vòm và nguyên âm mở |
| **Câu đàm thoại** | Conf_001 | **TỪ CHỐI ĐÚNG** | `REJECT` | **$0.0000$** | Dạng phổ câu dài bị chặn hoàn hảo |

---

## 4. Đánh Giá Các Nhánh Quyết Định Theo Chỉ Đạo

### Nhánh 1: Sửa logic quyết định có giải quyết được hiện tượng "cửa" thành "PHAI" không?
- **Kết luận:** **KHÔNG**.
- Sửa quy tắc $\arg\max$ 5 lớp giúp loại bỏ các mẫu câu đàm thoại hoặc tạp âm có $P(\text{REJECT})$ cao nhưng một lệnh vượt nhẹ $0.20$.
- Tuy nhiên, đối với từ "cửa" (giọng Kim Thanh hoặc Thùy Dung), $P(\text{PHAI}) \approx 0.40 - 0.58$, cao hơn hẳn $P(\text{REJECT}) \approx 0.05 - 0.10$. Do đó, kể cả theo quy tắc $\arg\max$ 5 lớp, mô hình vẫn đưa ra kết luận là `PHAI`.

### Nhánh 2: Tăng ngưỡng quyết định $\theta_{\text{cmd}}$ có phải là giải pháp đúng đắn không?
- **Khuyến nghị:** **TUYỆT ĐỐI KHÔNG TỰ Ý TĂNG NGƯỠNG**.
- Để chặn từ "bàn" ($P = 0.5032$) hay từ "ghế" ($P = 0.8011$), ngưỡng $\theta$ sẽ phải nâng lên $> 0.80$.
- Việc nâng ngưỡng lên $> 0.80$ sẽ phá hủy hoàn toàn khả năng nhận diện các lệnh thật (Recall sụt giảm nghiêm trọng, vì nhiều lệnh thật phát âm tự nhiên chỉ đạt xác suất $0.35 - 0.60$).
- Hành động tăng ngưỡng tùy tiện vi phạm nguyên tắc quản trị khoa học độc lập của dự án.

### Nhánh 3: Nhận định bản chất khoa học
- Đây là minh chứng thực tế rõ ràng nhất cho thấy mô hình **CHƯA ĐƯỢC THẨM ĐỊNH TRÊN GIỌNG NÓI TỰ NHIÊN MỞ (Natural Open-Vocabulary Speech)**.
- Kết quả này hoàn toàn thống nhất với hai nhãn định danh quản trị cốt lõi:
  - `NOT_VALIDATED_ON_NATURAL_SPEECH`
  - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`

---

## 5. Thẩm Tra Tính Toàn Vẹn Của Hệ Thống

1. **Tính Bất Biến Của Mô Hình Đông Kết (Byte-for-Byte Checksum Verification):**
   - Checkpoint PyTorch: `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` (**KHỚP TUYỆT ĐỐI**)
   - Mô hình ONNX: `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` (**KHỚP TUYỆT ĐỐI**)
2. **Tính Đẳng Cấu Giữa Pipeline Ngoại Tuyến & Trực Tiếp:**
   - Cả hai pipeline chia sẻ chung hợp đồng tiền xử lý (16 kHz, đệm zero phía phải, co biên độ có điều kiện khi $\max > 1.0$).
   - Kết quả dự đoán trên 100% mẫu kiểm định phát triển trùng khớp hoàn toàn.
3. **Chính Sách Quyền Riêng Tư & An Toàn Bộ Nhớ:**
   - Mọi mẫu sóng âm micro được giữ hoàn toàn trong bộ nhớ RAM (In-Memory Only).
   - Tuyệt đối không lưu tệp âm thanh thô `.wav` ra ổ cứng.
   - Luồng thu âm PortAudio đóng sạch sẽ sau khi kết thúc thử nghiệm.
4. **Độ Bao Phủ Kiểm Thử:**
   - Toàn bộ **18/18 bài kiểm thử tự động** trong dự án đều vượt qua ($100\%$ PASSED).

---

## 6. Khuyến Nghị Cho Buổi Báo Cáo Học Thuật

1. **Trình bày trung thực về mặt khoa học:**
   - Trong bài thuyết trình, hãy công khai kết quả này như một bài học điển hình về giới hạn của mô hình phân loại đơn nhãn miền đóng (Closed-domain single-label CNN).
   - Nhấn mạnh rằng mô hình nhận diện rất tốt 4 từ khóa mục tiêu (`TRAI`, `PHAI`, `LEN`, `XUONG`) và loại bỏ hoàn hảo tiếng ồn môi trường/câu đàm thoại dài, nhưng **dễ bị kích hoạt nhầm bởi các từ đơn âm tiết tiếng Việt ngoài tập từ khóa có thanh điệu tương đồng (đặc biệt là thanh hỏi)**.
2. **Hướng nghiên cứu tiếp theo (Future Work):**
   - Để triển khai thực địa, cần thu thập tập dữ liệu âm bản gồm hàng ngàn từ đơn âm tiết tiếng Việt phổ thông (Open-vocabulary monosyllabic negatives) đưa vào lớp `REJECT`.
   - Kết hợp mô hình xác thực 2 giai đoạn (Cascade Verifier) hoặc chuyển dịch sang kiến trúc nhận dạng chuỗi CTC/Conformer.
