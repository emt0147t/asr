# PHASE 72: DEFENSE PACKAGE — 20 EVIDENCE-BASED DEFENSE QUESTIONS & ANSWERS
## Rigorous Technical Q&A Guide for Academic Committee and Expert Peer Review

**Dự án:** Vietnamese Keyword Spotting (KWS CNN)  
**Repository:** `D:\trainidentificationaudio`  
**Ngày hoàn thiện:** 26/10/2026  
**Mục tiêu:** Cung cấp câu trả lời có bằng chứng xác thực (evidence-based), bảo vệ trung thực ranh giới nghiên cứu và đối thoại kỹ thuật với Hội đồng bảo vệ.

---

### CÂU HỎI 1: Tại sao nhóm nghiên cứu không thu thập dữ liệu giọng nói người thật hoặc thử nghiệm trực tiếp trên microphone vật lý?
**Trả lời:**
- *Cơ sở bằng chứng:* Bốn nhãn quản trị bắt buộc của dự án là `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`, `NOT_VALIDATED_ON_NATURAL_SPEECH`, `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`, và `NOT_VALIDATED_PENDING_DATA`.
- *Giải thích:* Mục tiêu khoa học của đề tài được giới hạn nghiêm ngặt ở cấp độ **khảo sát âm học thuật toán trên môi trường tổng hợp có kiểm soát**. Việc thu thập giọng người thật đòi hỏi quy trình phê duyệt đạo đức (IRB/Ethics), chuẩn hóa thiết bị thu âm, phân tầng đại diện vùng miền (Bắc - Trung - Nam) và thỏa thuận đồng thuận người tham gia (consent form) chặt chẽ. Nếu thu âm vội vã vài giọng nói tự tạo trong phòng thí nghiệm, dữ liệu sẽ bị nhiễu do hiệu ứng người nói (speaker leakage) và tạo ra ảo tưởng về tính sẵn sàng triển khai. Do đó, nhóm kiên quyết giữ ranh giới nghiên cứu trung thực là synthetic-only.

---

### CÂU HỎI 2: Tại sao ứng viên Phase 28 D2_H1 đạt độ chính xác tới 93.91% trên tập kiểm định nhưng trong luồng streaming lại chỉ đạt 17.24% độ nhạy lệnh?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE64_VALIDATION_MASTER_METRICS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_MASTER_METRICS.json) và [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json).
- *Giải thích:* Đây là biểu hiện kinh điển của **sự đánh đổi giữa từ chối (rejection bias) và độ nhạy (sensitivity)**. Trong huấn luyện D2_H1, hàm mất mát cross-entropy đã đặt trọng số phạt $1.25\times$ lên lớp REJECT kết hợp với việc lấy mẫu hard negatives dày đặc. Cơ chế này ép mô hình ưu tiên gán nhãn REJECT để triệt tiêu báo động giả.
- Trên tập kiểm định cô lập (isolated), các clip từ khóa có độ dài cắt gọt đúng 1.0 giây, âm lượng chuẩn hóa nên mô hình vẫn bắt được. Nhưng trong luồng liên tục (streaming Track B), từ khóa nằm đan xen giữa các khoảng lặng và ngữ cảnh đàm thoại; các đặc trưng âm học ở ranh giới khung trượt không hoàn toàn khớp với mẫu huấn luyện, khiến bộ phân loại luôn chọn phương án an toàn nhất là REJECT. Kết quả là D2_H1 bỏ sót $41$ trên tổng số $58$ khẩu lệnh ($82.76\%$ miss rate).

---

### CÂU HỎI 3: Tại sao từ khóa `TRÁI` lại bị sụp đổ độ nhạy xuống 22.50% trong đánh giá xác nhận Phase 59? Nguyên nhân ngữ âm học là gì?
**Trả lời:**
- *Cơ sở bằng chứng:* Ma trận nhầm lẫn Track A trong [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json): D2_H1 chỉ nhận diện đúng $9/40$ clip `TRÁI`, có tới $29/40$ clip bị gán vào REJECT và $2$ clip bị nhầm sang `LEN`.
- *Nguyên nhân ngữ âm:*
  1. Từ `TRÁI` bắt đầu bằng phụ âm quặt lưỡi vô thanh $/ʈʂ/$ (hoặc tắc xát vô thanh $/tɕ/$ tùy phương ngữ) kết hợp với âm đệm $/w/$, nguyên âm $/aː/$ và thanh sắc (sắc nổi).
  2. Trong engine tổng hợp VieNeu-TTS (tập huấn luyện), formant chuyển tiếp và năng lượng dải tần cao (4.000–8.000Hz) của phụ âm $/ʈʂ/$ có đặc trưng phổ rất khác so với engine Piper VAIS-1000 (tập kiểm chứng Phase 59).
  3. Khi chuyển sang Piper VAIS-1000, sự sai lệch formant này khiến mô hình coi đoạn âm thanh không khớp với phân phối từ khóa đã học, và với trọng số phạt nặng lớp REJECT, mô hình lập tức đẩy toàn bộ clip vào REJECT.

---

### CÂU HỎI 4: Nếu D2_H1 có nhiều tiến bộ về giảm false alarm như vậy, tại sao nhóm không chọn D2_H1 làm mô hình mặc định mà vẫn giữ legacy v11?
**Trả lời:**
- *Cơ sở bằng chứng:* [`FINAL_ARTIFACT_CATALOG.csv`](file:///d:/trainidentificationaudio/FINAL_ARTIFACT_CATALOG.csv) và Báo cáo tổng kết [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md).
- *Giải thích:* Đây là nguyên tắc **quản trị an toàn phòng ngừa (defensive governance)** trong kỹ thuật phần mềm:
  1. Baseline legacy $v11$ tuy có tỷ lệ false alarm cao hơn trên tập synthetic ($34.22\%$ trên val_v1), nhưng nó giữ được độ nhạy tương đối đồng đều trên cả 4 từ khóa (`TRAI` đạt 78.75%, `PHAI` đạt 96.25%, `LEN` đạt 94.38%, `XUONG` đạt 91.88%).
  2. Ứng viên D2_H1 tuy giảm false alarm xuống $9.22\%$ trên val_v1 và $5.77\%$ trên Piper, nhưng lại bị 'què quặt' nghiêm trọng về độ nhạy (TRAI recall chỉ còn 22.5%, streaming recall chỉ 17.24%).
  3. Thay thế một mô hình mặc định bằng một ứng viên bị sụp đổ hoàn toàn trên một lớp lệnh chính sẽ tạo ra sự thoái lui chức năng (regression) tai hại. Vì vậy, $v11$ được giữ làm default reference model bất biến.

---

### CÂU HỎI 5: Nguồn gốc của con số 88.59% accuracy từng xuất hiện trong báo cáo lịch sử của v11 là gì, và tại sao con số chính xác lại là 78.05%?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_ALIGNED_PREDICTIONS.csv) và [`PHASE64_VALIDATION_MASTER_METRICS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_MASTER_METRICS.json).
- *Giải thích pháp y:*
  1. Trong tập kiểm định `validation_v1.csv` gồm $1.280$ mẫu ($640$ mẫu REJECT, $160$ mẫu mỗi từ khóa), ứng viên $D_3\_C$ nhận diện đúng $567$ trên $640$ mẫu REJECT, đạt tỷ lệ REJECT recall là đúng $567 / 640 = 88.59375\%$.
  2. Trong quá trình soạn thảo tài liệu lịch sử ở Phase 55, con số $88.59\%$ này đã bị sao chép nhầm vị trí vào ô Overall Accuracy của mô hình $v11$.
  3. Bằng chứng kiểm toán chạy lại trực tiếp từ raw audio cho thấy: Dưới ngưỡng vận hành bị khóa $\theta_{\text{cmd}} = 0.20$, $v11$ đoán đúng $126$ TRAI, $154$ PHAI, $151$ LEN, $147$ XUONG và $421$ REJECT. Tổng số mẫu đúng là đúng $999 / 1.280$, tương ứng độ chính xác chính thức là **$78.046875\%$ ($78.05\%$)**.
  4. Nếu quét tìm ngưỡng tự do (grid sweep từ $0.10$ đến $0.95$), độ chính xác cao nhất mà $v11$ có thể đạt được là $80.55\%$ tại $\theta = 0.28$. Nhưng do ngưỡng đã khóa ở $0.20$, con số công bố trung thực bắt buộc là $78.05\%$.

---

### CÂU HỎI 6: Tại sao trong luồng streaming không quan sát thấy báo động giả nào trên silence (k=0), nhưng nhóm vẫn báo cáo cận trên Poisson là 3.48 lần/phút?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE60_TRACKB_FA_RATE_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE60_TRACKB_FA_RATE_AUDIT.md) và phân phối xác suất Poisson.
- *Giải thích toán học:*
  1. Trong thống kê, điểm ước lượng (point estimate) $\hat{\lambda} = k/T = 0/0.86045 = 0\text{ FA/min}$ chỉ phản ánh kết quả trong mẫu quan sát, không phản ánh độ bất định khi kích thước mẫu nhỏ.
  2. Theo phân phối Poisson cho biến cố hiếm, xác suất quan sát thấy 0 biến cố trong khoảng thời gian $T$ với tốc độ thực tế $\lambda$ là $P(K=0) = e^{-\lambda T}$. Để tìm cận trên khoảng tin cậy một phía 95%, ta giải bất phương trình $e^{-\lambda_{\text{upper}} T} \ge 0.05$, dẫn tới:
     $$\lambda_{\text{upper}} = \frac{-\ln(0.05)}{T} \approx \frac{2.99573}{T}$$
  3. Với $T = 0.86045\text{ phút}$, ta có $\lambda_{\text{upper}} = 2.99573 / 0.86045 = \mathbf{3.48\text{ FA/min}}$ (áp dụng Quy tắc số Ba - Rule of Three xấp xỉ $3 / 0.86045 = 3.49\text{ FA/min}$).
  4. Báo cáo giá trị 3.48 FA/min giúp bảo vệ đề tài khỏi sai lầm ngụy biện rằng hệ thống 'hoàn hảo tuyệt đối' trên khoảng lặng.

---

### CÂU HỎI 7: Tại sao mức chênh lệch +5.24 điểm phần trăm giữa D2_H1 và D3_C có p = 0.0336 nhưng nhóm lại kết luận là 'Không có ý nghĩa thống kê'?
**Trả lời:**
- *Cơ sở bằng chứng:* Văn bản đăng ký trước thiết kế thực nghiệm [`PHASE58_SAMPLE_SIZE_AND_CLUSTER_DESIGN.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE58_SAMPLE_SIZE_AND_CLUSTER_DESIGN.md) Mục 2.1.
- *Giải thích:*
  1. Để kiểm soát tỷ lệ dương tính giả tích lũy (Family-Wise Error Rate) qua hàng chục phép so sánh đa mô hình trong suốt dự án, giao thức kiểm định đã quy định tiên nghiệm (preregistered) mức ý nghĩa thống kê là $\alpha = 0.01$ (thay vì mức thông thường 0.05).
  2. Khi kiểm định ghép cặp có tính cụm (clustered paired $t$-test trên 105 waveform gốc, $df=104$), kết quả thu được $t = 2.1533$ và $p = 0.0336$.
  3. Vì $p = 0.0336 > 0.01$, nên theo chuẩn mực đăng ký trước, giả thuyết không ($H_0$) không bị bác bỏ. Nhóm nghiên cứu kết luận đây chỉ là một xu hướng thực nghiệm (exploratory trend), không được phép tuyên bố là phát hiện có ý nghĩa thống kê khẳng định (confirmatory statistical significance).

---

### CÂU HỎI 8: Tại sao nhóm kiên quyết giữ dung lượng mô hình ở 138.469 tham số mà không mở rộng dung lượng (E2 Capacity Expansion) để tăng độ chính xác?
**Trả lời:**
- *Cơ sở bằng chứng:* Quyết định quản trị `DEFER_E2_CAPACITY` trong [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md) Section 3.
- *Giải thích:*
  1. Triết lý thiết kế của bài toán KWS là phục vụ cho thiết bị cận biên (Edge AI / Microcontrollers) với bộ nhớ SRAM và Flash hạn chế, yêu cầu độ trễ xử lý dưới 5ms và tiêu thụ năng lượng thấp.
  2. Kiến trúc `CorrectedVietnameseKWSNet` với 138.469 tham số tương đương kích thước file ONNX chỉ khoảng 550KB, thời gian suy luận trên CPU là 1.8ms (P95 = 2.12ms), hoàn toàn đáp ứng môi trường nhúng.
  3. Mở rộng tham số (E2) khi chưa giải quyết được vấn đề ranh giới dữ liệu và thiên lệch âm học sẽ chỉ làm tăng nguy cơ học vẹt (overfitting) trên dữ liệu tổng hợp chứ không giải quyết được bản chất của bài toán.

---

### CÂU HỎI 9: Ngưỡng theta_cmd = 0.20 có vai trò gì? Tại sao không dùng chuẩn argmax 5 lớp thông thường?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE64_VALIDATION_RECONCILED_METRICS.csv`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE64_VALIDATION_RECONCILED_METRICS.csv).
- *Giải thích:*
  1. Trong bài toán KWS 5 lớp, lớp REJECT chiếm phần lớn thời lượng thực tế và có phân phối âm học rất rộng. Khi tính softmax trên cả 5 lớp, logit của REJECT thường có xu hướng chiếm ưu thế tuyệt đối.
  2. Nếu dùng chuẩn argmax thông thường, từ khóa cần phải có xác suất $> 0.50$ mới kích hoạt được. Điều này dẫn tới việc mô hình bỏ sót rất nhiều khẩu lệnh khi âm lượng nhỏ hoặc có tiếng ồn nền nhẹ.
  3. Chính sách $\theta_{\text{cmd}} = 0.20$ là cơ chế override: chỉ cần tổng hợp đặc trưng của một trong 4 từ khóa đạt độ tự tin từ 20%, hệ thống sẽ ưu tiên chọn từ khóa đó. Điều này giúp đẩy độ nhạy nhận diện lệnh lên tối đa, sau đó nhóm dùng VAD và lớp REJECT được huấn luyện có trọng số để kiểm soát báo động giả.

---

### CÂU HỎI 10: Tại sao việc đưa thêm mẫu Edge-TTS vào huấn luyện ở D2 không giúp mô hình tổng quát hóa tốt sang Piper VAIS-1000?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE62_SOURCE_AND_PROTOCOL_SCOPE_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE62_SOURCE_AND_PROTOCOL_SCOPE_AUDIT.md).
- *Giải thích:*
  1. Edge-TTS và Piper VAIS-1000 sử dụng hai kiến trúc tổng hợp giọng nói và vocoder hoàn toàn khác nhau. Edge-TTS dựa trên mô hình nơ-ron đám mây của Microsoft, trong khi Piper VAIS-1000 là mô hình VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) chạy cục bộ.
  2. Việc bổ sung Edge-TTS vào tập huấn luyện D2 chỉ giúp mạng học thêm các đặc trưng phổ đặc thù của 2 giọng đọc cụ thể trong Edge-TTS (`vi-VN-HoaiMyNeural` và `vi-VN-NamMinhNeural`).
  3. Khi gặp giọng đọc của Piper VAIS-1000 (giọng đọc khác, tốc độ phát âm khác, các dị biệt về pha và đường bao năng lượng tần số cao), mạng nơ-ron không nhận diện được các đặc trưng quen thuộc, dẫn tới hiện tượng trôi dạt phân phối (distribution shift).

---

### CÂU HỎI 11: Tại sao hệ thống sử dụng biểu đồ phổ Log-Mel mà không dùng MFCC truyền thống hay các mô hình trích xuất đặc trưng tự học từ sóng thô (như Wav2Vec)?
**Trả lời:**
- *Giải thích:*
  1. **So với MFCC:** Phép biến đổi DCT (Discrete Cosine Transform) trong MFCC thực hiện nén giải tương quan (decorrelation), làm mất mát một phần thông tin cấu trúc không gian cục bộ (local spatial relationships) trên trục thời gian - tần số. Bộ lọc tích chập (Conv2D) trong CNN khai thác các bộ lọc cục bộ cực kỳ hiệu quả trên biểu đồ Log-Mel 2 chiều liên tục.
  2. **So với Wav2Vec / HuBERT:** Các mô hình tự học sóng thô như Wav2Vec 2.0 có kích thước từ 95 triệu đến hơn 300 triệu tham số, đòi hỏi tài nguyên phần cứng lớn và độ trễ tính toán lên tới hàng trăm mili-giây, hoàn toàn bất khả thi cho bài toán KWS thời gian thực trên vi điều khiển nhúng. Log-Mel frontend chỉ tốn chưa tới 1ms tính toán trên CPU.

---

### CÂU HỎI 12: Bộ SpeechBandRatioVAD hoạt động như thế nào và có ưu điểm gì so với WebRTC VAD?
**Trả lời:**
- *Cơ sở bằng chứng:* Mã nguồn [`vad.py`](file:///d:/trainidentificationaudio/vad.py).
- *Giải thích:*
  1. `SpeechBandRatioVAD` tính tỷ số giữa năng lượng trong dải tần số tiếng nói của con người (300Hz đến 3.400Hz) so với tổng năng lượng toàn dải phổ (0 đến 8.000Hz).
  2. Khung âm thanh chỉ được coi là có tiếng nói khi tỷ lệ năng lượng này $\ge 0.65$ kết hợp với mức năng lượng tuyệt đối vượt ngưỡng nhiễu nền.
  3. Ưu điểm: Nó loại bỏ hiệu quả các loại nhiễu tần số thấp (tiếng rung cơ học, quạt gió) và nhiễu tần số rất cao mà không cần duy trì mô hình xác suất Gaussian phức tạp như WebRTC VAD, giảm thiểu tối đa footprint bộ nhớ.

---

### CÂU HỎI 13: Làm thế nào nhóm nghiên cứu chứng minh được rằng hoàn toàn không có sự rò rỉ dữ liệu (data leakage) giữa tập huấn luyện và các tập kiểm thử?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE71_RELEASE_CHECKLIST.md`](file:///d:/trainidentificationaudio/PHASE71_RELEASE_CHECKLIST.md) Mục 4.
- *Giải thích pháp y:*
  1. Nhóm đã thực hiện thuật toán quét giao cắt tập hợp (set intersection) trên tên file, mã băm nội dung waveform và metadata giữa tập huấn luyện ($5.440$ mẫu trong `D2_matched_train_v1.csv`) với:
     - Tập Phase 43 (MMS-TTS, 480 mẫu): Giao tập = $0$ file ($0.00\%$).
     - Tập Phase 51 (gTTS, 240 mẫu): Giao tập = $0$ file ($0.00\%$).
     - Tập Phase 59 (Piper VAIS-1000 Track A & B, 470 mẫu): Giao tập = $0$ file ($0.00\%$).
  2. Các tập kiểm thử được lưu trữ trong thư mục cách ly riêng biệt và không có bất kỳ quy trình trích xuất đặc trưng nào từ các tập này được đưa vào quá trình chuẩn hóa (z-score/mean/std) của tập huấn luyện.

---

### CÂU HỎI 14: Sự khác nhau cơ bản giữa 4-Command Macro-F1 và 5-Class Macro-F1 là gì? Tại sao phải phân biệt hai chỉ số này?
**Trả lời:**
- *Giải thích:*
  1. **4-Command Macro-F1:** Là trung bình cộng không trọng số của F1-score trên đúng 4 lớp từ khóa: `TRAI`, `PHAI`, `LEN`, `XUONG`. Chỉ số này phản ánh độ chính xác và độ nhạy chuyên biệt trên các khẩu lệnh điều khiển thực tế.
  2. **5-Class Macro-F1:** Tính trung bình cộng của cả 5 lớp, bao gồm cả lớp `REJECT`.
  3. **Tầm quan trọng của việc tách biệt:** Lớp REJECT có số lượng mẫu rất lớn ($640$ mẫu trong val_v1, $260$ mẫu trong Phase 59) và F1 của lớp REJECT thường rất cao (trên $80\%–90\%$). Nếu chỉ nhìn vào 5-class Macro-F1, chỉ số cao của lớp REJECT sẽ che giấu đi sự suy giảm nghiêm trọng của các lớp từ khóa bên trong. Điển hình ở Phase 59, 5-class Macro-F1 của D2_H1 là $59.65\%$, nhưng nếu bỏ REJECT ra thì 4-command Macro-F1 chỉ còn $53.76\%$, và riêng từ `TRAI` chỉ đạt F1 là $32.73\%$.

---

### CÂU HỎI 15: Nếu muốn đưa hệ thống này ra triển khai thực tế trong tương lai, quy trình bắt buộc phải gồm những bước nào?
**Trả lời:**
- *Cơ sở bằng chứng:* [`RESEARCH_FINAL_REPORT.md`](file:///d:/trainidentificationaudio/RESEARCH_FINAL_REPORT.md) Section 13.
- *Quy trình 5 bước bắt buộc:*
  1. **Thu thập tập dữ liệu tiếng nói tự nhiên đa vùng miền:** Tối thiểu 50 người nói đại diện cho giọng Bắc, Trung, Nam với tỷ lệ cân bằng giới tính.
  2. **Thử nghiệm trên phần cứng microphone thực tế:** Đo lường tác động của đáp ứng tần số micro, bộ tiền khuếch đại (preamp) và thuật toán AGC (Automatic Gain Control).
  3. **Đăng ký trước (Pre-registration):** Khóa toàn bộ tham số mô hình, ngưỡng VAD và ngưỡng quyết định trước khi chạy kiểm thử mù (blind evaluation).
  4. **Kiểm tra độ trễ và tiêu thụ điện năng:** Đo lường thực tế trên chip nhúng (như STM32H7, ESP32-S3 hoặc Raspberry Pi).
  5. **Hội đồng nghiệm thu độc lập:** Đánh giá và cấp phép gỡ bỏ nhãn `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`.

---

### CÂU HỎI 16: Tại sao ứng viên D3_C đạt độ nhạy trong streaming cao hơn D2_H1 (32.76% so với 17.24%) nhưng vẫn bị đánh trượt (FAIL)?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json).
- *Giải thích:*
  1. D3_C được thiết kế với giả thuyết cân bằng độ nhạy (giảm trọng số phạt lớp REJECT từ $1.25\times$ xuống $1.0\times$ và tăng trọng số từ khóa lên $1.05\times$). Nhờ đó, trong streaming, D3_C bớt rụt rè hơn và bắt được $19/58$ lệnh ($32.76\%$), đạt Event F1 $31.40\%$.
  2. Tuy nhiên, D3_C bị đánh trượt vì nó **không đạt các ngưỡng sàn tối thiểu đã đăng ký**:
     - Tiêu chí đăng ký trước yêu cầu Macro-F1 isolated $\ge 60\%$, nhưng D3_C chỉ đạt $41.23\%$.
     - Tiêu chí yêu cầu độ nhạy tối thiểu trên từng lớp lệnh $\ge 40\%$, nhưng trên từ khóa `TRAI`, D3_C bị sụp đổ gần như hoàn toàn, chỉ nhận diện đúng $1$ trong số $40$ clip ($2.50\%$).
  3. Do đó, D3_C thất bại trong việc chứng minh khả năng tổng quát hóa cân bằng.

---

### CÂU HỎI 17: Bộ dữ liệu Tone Probes (60 clip) trong Phase 59 có ý nghĩa gì và kết quả cho thấy điều gì?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE59_FINAL_EVALUATION_RESULTS.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE59_FINAL_EVALUATION_RESULTS.json).
- *Giải thích:*
  1. Tiếng Việt là ngôn ngữ có thanh điệu (tonal language). Các từ phủ định có thể chỉ khác từ khóa ở dấu thanh (ví dụ: `trại`, `trải` so với `trái`; `phải` so với `phái`, `phai`; `lên` so với `lèn`, `lén`).
  2. Bộ Tone Probes gồm 60 clip âm tiết có ngữ âm tương đồng chỉ khác thanh điệu để kiểm tra xem mô hình có bị nhầm lẫn thanh điệu hay không.
  3. Kết quả: D2_H1 chỉ kích hoạt nhầm 4/60 clip (Tone Probe FAR = $6.67\%$), trong khi v11 bị nhầm tới 38/60 clip ($63.33\%$). Điều này chứng minh rằng việc huấn luyện với hard negatives ở D2 giúp mô hình phân biệt thanh điệu tốt hơn, nhưng cái giá phải trả vẫn là sự suy giảm độ nhạy trên từ khóa chính.

---

### CÂU HỎI 18: Tại sao nhóm lại thu hẹp tuyên bố độc lập (narrowed scope) của bộ nhiễu ESC-50 Fold 3?
**Trả lời:**
- *Cơ sở bằng chứng:* [`PHASE62_SOURCE_AND_PROTOCOL_SCOPE_AUDIT.md`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE62_SOURCE_AND_PROTOCOL_SCOPE_AUDIT.md).
- *Giải thích:*
  1. Ban đầu, các báo cáo mô tả ESC-50 Fold 3 là một 'nguồn nhiễu độc lập hoàn toàn'.
  2. Tuy nhiên, kiểm toán chi tiết cho thấy: Mặc dù các đoạn sóng âm (waveforms) trong Fold 3 hoàn toàn không trùng lặp vật lý với Fold 1 và Fold 2 (hệ số tương quan tối đa $r_{\max} = 0.0947 < 0.15$), nhưng các nhóm danh mục ngữ nghĩa (semantic categories như tiếng vỗ tay, tiếng ho, tiếng bước chân) vẫn giữ nguyên mã danh mục như trong tập huấn luyện.
  3. Để bảo đảm tính trung thực học thuật, nhóm đã thu hẹp tuyên bố: ESC-50 Fold 3 là tập nhiễu **độc lập về dạng sóng (waveform-disjoint)** chứ **không phải là họ danh mục nhiễu hoàn toàn mới**.

---

### CÂU HỎI 19: Tại sao nhóm không điều chỉnh lại ngưỡng theta_cmd linh hoạt cho từng từ khóa thay vì dùng chung một ngưỡng 0.20?
**Trả lời:**
- *Giải thích:*
  1. Việc tối ưu ngưỡng riêng cho từng lớp (per-class threshold tuning) là một hướng đi khả thi trên lý thuyết.
  2. Tuy nhiên, trong khuôn khổ nghiên cứu này, việc tinh chỉnh ngưỡng sau khi đã quan sát kết quả kiểm chứng (post-hoc threshold tuning) bị **nghiêm cấm bởi quy chuẩn quản trị khoa học**, vì nó biến một phép thử độc lập thành một bài toán khớp dữ liệu (overfitting to test set).
  3. Nếu nhóm tự ý thay đổi ngưỡng để kéo độ nhạy của từ `TRAI` lên, tính khách quan của phép thử Phase 59 sẽ bị hủy bỏ hoàn toàn. Ngưỡng $\theta_{\text{cmd}} = 0.20$ được bảo toàn để phản ánh trung thực năng lực cố hữu của mô hình.

---

### CÂU HỎI 20: Cổng đóng nghiên cứu `PASS_WITH_DOCUMENTED_REPORT_CORRECTIONS` có ý nghĩa chính xác là gì đối với đề tài?
**Trả lời:**
- *Giải thích:*
  1. **PASS:** Xác nhận toàn bộ quy trình nghiên cứu, mã nguồn, checkpoint mô hình, các ma trận dự đoán và tính toán thống kê đã được tái lập thành công $100\%$, đạt tiêu chuẩn cao nhất về tính tái lập khoa học (reproducibility).
  2. **WITH DOCUMENTED REPORT CORRECTIONS:** Thể hiện tinh thần phê phán và liêm chính học thuật: Tất cả các sai sót trong văn bản lịch sử (như nhầm lẫn 88.59%, công thức Poisson chưa chuẩn, xấp xỉ thời lượng streaming) đã được công khai minh bạch, đối chiếu nguồn gốc và sửa đổi dứt điểm trong bộ tài liệu nghiệm thu.
  3. Cổng này khẳng định đề tài đã hoàn thành xuất sắc sứ mệnh nghiên cứu học thuật khám phá ranh giới kỹ thuật, đồng thời thiết lập nền móng quản trị vững chắc cho bất kỳ nghiên cứu ứng dụng nào tiếp nối trong tương lai.
