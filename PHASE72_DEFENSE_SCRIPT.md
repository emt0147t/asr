# PHASE 72: DEFENSE PACKAGE — PRESENTATION SCRIPT & TECHNICAL TALK TRACK
## Complete 8–10 Minute Vietnamese Presentation Script, Technical Glossary, Slide Checklist, and Presentation Governance

**Dự án:** Vietnamese Keyword Spotting (KWS CNN)  
**Repository:** `D:\trainidentificationaudio`  
**Ngày hoàn thiện:** 26/10/2026  
**Cổng quản trị nghiên cứu:** `PASS_WITH_DOCUMENTED_REPORT_CORRECTIONS`  
**Mô hình mặc định tham chiếu:** Legacy $v11$ (138.469 tham số, $\theta_{\text{cmd}} = 0.20$)  

---

## MANDATORY GOVERNANCE DECLARATIONS

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

> [!CRITICAL]
> **NGUYÊN TẮC BẢO VỆ CỐT LÕI (CORE DEFENSE TENET):**  
> *"Khả năng từ chối mạnh tiếng nói ngoài tập lệnh (Strong OOV Rejection) KHÔNG đồng nghĩa với việc mô hình nhận diện tốt từ khóa (Strong Keyword Recall)."*  
> Người thuyết trình phải luôn trung thực về ranh giới dữ liệu hoàn toàn là tổng hợp (synthetic-only), sự sụp đổ độ nhạy trên lệnh `trái`, độ bất định thống kê theo phân phối Poisson và việc tuyệt đối chưa thử nghiệm trên giọng nói người thật.

---

## 1. Coherent 8–10 Minute Vietnamese Talk Track (Kịch bản thuyết trình bảo vệ)

### PHẦN 1: MỞ ĐẦU, BỐI CẢNH VÀ RANH GIỚI NGHIÊN CỨU (0:00 – 1:30)
> *"Kính thưa Hội đồng và các thầy cô,*
> 
> *Hôm nay, em xin trình bày kết quả nghiên cứu về bài toán 'Nhận diện từ khóa giọng nói tiếng Việt (Vietnamese Keyword Spotting - KWS)' sử dụng mạng nơ-ron tích chập (CNN) dung lượng nhỏ gọn 138.469 tham số.*
> 
> *Trước khi đi vào chi tiết kỹ thuật, em xin phép tuyên bố rõ ràng ranh giới nghiên cứu và quản trị khoa học của đề tài:*
> 1. *Toàn bộ quá trình huấn luyện, tối ưu và kiểm thử trong đề tài này được thực hiện hoàn toàn trên **dữ liệu âm thanh tổng hợp (synthetic TTS)** và các bộ nhiễu môi trường chuẩn hóa.*
> 2. *Đề tài **chưa từng thu thập giọng nói người thật** và **chưa từng thử nghiệm trên microphone vật lý thời gian thực**.*
> 3. *Vì vậy, nghiên cứu này **không tuyên bố một hệ thống sẵn sàng triển khai thực tế**, mà đóng vai trò là một khảo sát khoa học chuyên sâu về độ bền vững âm học, ranh giới dịch chuyển miền tổng hợp, và sự đánh đổi giữa việc giảm báo động giả với độ nhạy nhận diện lệnh.*
> 4. *Mô hình baseline legacy v11 vẫn được giữ nguyên làm mô hình tham chiếu mặc định an toàn cho toàn bộ repository."*

---

### PHẦN 2: BÀI TOÁN 5 LỚP, PIPELINE HỆ THỐNG VÀ NGƯỠNG VẬN HÀNH (1:30 – 3:30)
> *"Kính thưa Hội đồng,*
> 
> *Bài toán của chúng em là nhận diện 4 từ khóa điều hướng: `TRÁI`, `PHẢI`, `LÊN`, `XUỐNG`, cùng một lớp thứ năm cực kỳ quan trọng là lớp `REJECT`.*
> 
> *Lớp `REJECT` bao gồm 3 nhóm âm thanh phủ định:*
> - *Khoảng lặng và tiếng thở (Silence/Breath).*
> - *Tiếng ồn môi trường tự nhiên (Environmental Noise từ ESC-50).*
> - *Tiếng nói ngoài tập lệnh (Out-of-Vocabulary - OOV speech bao gồm cả câu đàm thoại dài và các âm tiết có ngữ âm tương đồng như 'bên phải', 'đi lên').*
> 
> *Pipeline xử lý của hệ thống gồm:*
> 1. ***Frontend Log-Mel:** Chuyển đổi tín hiệu 16kHz thành biểu đồ phổ 40 mel filterbanks với độ dài cửa sổ 25ms (400 samples) và bước nhảy 10ms (160 samples).*
> 2. ***Voice Activity Detection (SpeechBandRatioVAD):** Bộ phát hiện tiếng nói dựa trên tỷ lệ năng lượng dải tần số 300–3.400Hz nhằm lọc bỏ khoảng lặng trước khi kích hoạt mạng CNN.*
> 3. ***Mạng CorrectedVietnameseKWSNet:** Gồm 4 khối Conv2D, chuẩn hóa BatchNorm, MaxPool2D, Adaptive Average Pooling và 2 tầng Fully Connected, tổng cộng 138.469 tham số — được cố định dung lượng chặt chẽ để hướng tới các thiết bị nhúng.*
> 4. ***Chính sách quyết định khóa theta_cmd = 0.20:** Thay vì dùng argmax 5 lớp thông thường vốn dễ bị logit REJECT áp đảo, hệ thống áp dụng cơ chế override: nếu xác suất của bất kỳ từ khóa nào trong 4 lệnh đạt từ 0.20 trở lên, mô hình sẽ chọn từ khóa có xác suất cao nhất; ngược lại sẽ phân loại vào REJECT."*

---

### PHẦN 3: HÀNH TRÌNH THỰC NGHIỆM VÀ DỊCH CHUYỂN MIỀN ÂM HỌC (3:30 – 5:30)
> *"Trong suốt 69 giai đoạn nghiên cứu, mô hình đã được thử thách qua 5 engine tổng hợp tiếng nói khác nhau:*
> - *VieNeu-TTS: Dùng cho tập huấn luyện và kiểm định nội miền (In-domain).*
> - *Microsoft Edge-TTS: Thử nghiệm bổ sung mẫu và kiểm tra Phase 28.*
> - *Meta MMS-TTS: Thử nghiệm dịch chuyển miền âm học Phase 43.*
> - *Google Translate TTS (gTTS): Thử nghiệm Phase 51.*
> - *Piper VAIS-1000: Bộ kiểm chứng xác nhận độc lập Phase 59.*
> 
> *Kết quả cho thấy một hiện tượng khoa học rõ nét: **Mô hình học sâu đạt kết quả rất cao trên tập nội miền, nhưng suy giảm đáng kể khi chuyển giao sang một engine tổng hợp chưa từng thấy**.*
> 
> *Cụ thể, trên tập kiểm định VieNeu-TTS, ứng viên Phase 28 D2_H1 đạt độ chính xác 93.91% và Macro-F1 93.88%. Nhưng khi đưa sang engine Piper VAIS-1000 ở Phase 59, độ chính xác tổng thể giảm xuống 74.52% và 4-command Macro-F1 chỉ còn 53.76%.*
> 
> *Điều này khẳng định rằng: Việc tăng cường dữ liệu tổng hợp (như đưa Edge-TTS vào D2) có thể giúp cải thiện một phần OOV, nhưng cũng tạo ra thiên lệch (bias) âm học mới chứ không tạo ra sự tổng quát hóa tuyệt đối."*

---

### PHẦN 4: ĐÁNH GIÁ PHASE 59 — NGHỊCH LÝ GIỮA TỪ CHỐI VÀ ĐỘ NHẠY LỆNH (5:30 – 7:30)
> *"Kính thưa Hội đồng, đây chính là **kết luận khoa học quan trọng nhất** của đề tài:*
> 
> *Trong đánh giá xác nhận Phase 59 trên bộ âm thanh độc lập Piper VAIS-1000 kết hợp với nhiễu ESC-50 Fold 3:*
> 
> ***Mặt đạt được:** Ứng viên D2_H1 thể hiện khả năng giảm báo động giả (False Alarm) rất ấn tượng. Tỷ lệ báo động giả trên các clip OOV cô lập chỉ là **5.77%** (so với 64.23% của v11). Trên luồng âm thanh liên tục Track B, tỷ lệ báo động giả trên tiếng nói ngoài tập lệnh chỉ là **1.85 lần/phút**.*
> 
> ***Mặt hạn chế nghiêm trọng:** Để đạt được tỷ lệ báo động giả thấp như vậy, D2_H1 đã bị rơi vào trạng thái **từ chối quá mức (over-rejection)**.*
> - *Trên tập cô lập Track A, độ nhạy của D2_H1 với từ khóa `TRÁI` bị sụp đổ nặng nề, chỉ đạt **22.50%** (29 trong tổng số 40 clip có từ `TRÁI` bị mô hình gán nhầm vào REJECT).*
> - *Đặc biệt, trong đánh giá luồng liên tục Track B (50 chuỗi âm thanh chứa 58 khẩu lệnh thực tế), D2_H1 đã **bỏ sót tới 41 khẩu lệnh**, độ nhạy nhận diện lệnh (Command Recall) chỉ đạt **17.24%**, và Event F1 chỉ đạt **19.23%**.*
> 
> *Như vậy, bài học thực tế là: **Một mô hình có tỷ lệ false alarm cực thấp trên lý thuyết có thể trở nên vô dụng trong thực tế nếu nó bỏ qua hơn 80% khẩu lệnh của người dùng**. Độ chính xác isolated cao không phản ánh đúng năng lực nhận diện liên tục."*

---

### PHẦN 5: ĐỐI CHIẾU SỐ LIỆU VÀ Ý NGHĨA THỐNG KÊ (7:30 – 9:00)
> *"Để đảm bảo tính nghiêm cẩn khoa học cao nhất, nhóm nghiên cứu đã tiến hành kiểm tra pháp y (forensic audit) toàn bộ số liệu:*
> 
> 1. ***Đính chính con số độ chính xác của v11:** Trong một số tài liệu lịch sử trước đây, độ chính xác của v11 từng bị ghi nhầm là 88.59%. Kiểm toán pháp y trên 1.280 dòng dự đoán gốc cho thấy: con số 88.59% thực chất là tỷ lệ REJECT recall của mô hình D3_C (567/640) bị sao chép nhầm. Độ chính xác thực tế của v11 dưới ngưỡng khóa theta=0.20 là **78.05%** (999/1280). Mức 80.55% chỉ là điểm cao nhất khi quét thử nghiệm ngưỡng tại theta=0.28.*
> 
> 2. ***Độ bất định thống kê theo phân phối Poisson:** Đối với luồng âm thanh silence (0.86 phút), dù không có sự kiện báo động giả nào xảy ra (k=0), nhưng theo cận trên Poisson một phía 95% (Rule of Three), cận trên thực tế là **3.48 lần/phút**. Tương tự với OOV (0.54 phút, k=1), cận trên là **8.76 lần/phút**. Điều này cho thấy tập streaming hiện tại còn quá ngắn để khẳng định chắc chắn về tỷ lệ false alarm dài hạn.*
> 
> 3. ***Ý nghĩa thống kê giữa D2_H1 và D3_C:** Khi thực hiện kiểm định t-test ghép cặp có tính đến cụm (clustered paired t-test trên 105 waveform gốc), mức chênh lệch +5.24 điểm phần trăm giữa D2_H1 và D3_C cho giá trị p = 0.0336. So với ngưỡng ý nghĩa thống kê đã đăng ký trước là alpha = 0.01, **chênh lệch này KHÔNG đạt ý nghĩa thống kê**. Đây chỉ là một xu hướng mang tính khám phá."*

---

### PHẦN 6: KẾT LUẬN VÀ BÀN GIAO QUẢN TRỊ (9:00 – 10:00)
> *"Kính thưa Hội đồng,*
> 
> *Căn cứ trên toàn bộ kết quả pháp y và thực nghiệm, nhóm nghiên cứu đưa ra quyết định quản trị cuối cùng:*
> - *Lựa chọn Cổng đóng nghiên cứu: **`PASS_WITH_DOCUMENTED_REPORT_CORRECTIONS`**.*
> - *Tiếp tục duy trì mô hình **legacy v11 làm baseline tham chiếu mặc định** của repository.*
> - *Không phê duyệt bất kỳ mô hình candidate nào để triển khai thực tế.*
> - *Toàn bộ artifact, checkpoint, script tái lập và bảng đối soát SHA-256 đã được niêm phong bất biến trong repository.*
> 
> *Để có thể tiến tới việc ứng dụng thực tế trong tương lai, hệ thống bắt buộc phải trải qua: thu thập tập dữ liệu tiếng nói người thật có gắn nhãn độc lập trên nhiều vùng miền (Bắc, Trung, Nam), ghi âm kiểm thử trên microphone vật lý trong môi trường ồn thực tế, và đăng ký trước quy trình đánh giá mù.*
> 
> *Em xin chân thành cảm ơn Quý Thầy Cô trong Hội đồng đã chú ý lắng nghe. Em rất mong nhận được các câu hỏi và ý kiến đóng góp từ Thầy Cô!"*

---

## 2. Short Technical Explanations (Định nghĩa kỹ thuật ngắn gọn, súc tích)

| Khái niệm kỹ thuật | Định nghĩa ngắn gọn & Chuẩn xác | Vai trò trong dự án |
| :--- | :--- | :--- |
| **CNN (Convolutional Neural Network)** | Mạng nơ-ron học sâu sử dụng các bộ lọc tích chập (kernels) trượt trên ma trận dữ liệu để trích xuất các đặc trưng không gian và thời gian bất biến. | Kiến trúc `CorrectedVietnameseKWSNet` gồm 4 khối Conv2D (138.469 tham số), trích xuất formant và đường bao phổ từ spectrogram. |
| **Log-Mel Spectrogram Frontend** | Kỹ thuật biểu diễn âm thanh trong miền thời gian - tần số bằng cách áp dụng biến đổi Fourier (STFT), lọc qua dải ngân hàng bộ lọc Mel (mô phỏng thính giác người) và nén logarit. | Biến đổi sóng âm 16kHz thành ma trận $40 \times 101$ đặc trưng (40 mel bins, 100 khung thời gian/giây). |
| **Voice Activity Detection (VAD)** | Thuật toán phân biệt đoạn âm thanh có chứa tiếng nói người so với khoảng lặng hoặc tiếng ồn nền. | Module `SpeechBandRatioVAD` tính tỷ lệ năng lượng dải tần số đàm thoại (300–3.400Hz) so với toàn dải, kích hoạt khi tỷ lệ $\ge 0.65$. |
| **Lớp REJECT (Phủ định)** | Lớp đầu ra thứ 5 trong mạng KWS đại diện cho tất cả các âm thanh không phải là 4 từ khóa mục tiêu (khoảng lặng, tiếng ồn, tiếng nói ngoài từ khóa). | Cho phép mô hình từ chối phát hiện khi người dùng nói chuyện bình thường hoặc khi môi trường xung quanh có tiếng động lớn. |
| **4-Command Macro-F1 vs 5-Class Macro-F1** | Trung bình cộng không trọng số của F1-score trên 4 lớp từ khóa chính (4-cmd) hoặc trên toàn bộ 5 lớp bao gồm cả REJECT (5-class). | 4-cmd Macro-F1 đo độ chính xác chuyên biệt trên các khẩu lệnh; 5-class Macro-F1 tính cả năng lực từ chối của lớp REJECT. |
| **OOV FAR (Out-of-Vocabulary False Alarm Rate)** | Tỷ lệ phần trăm các đoạn âm thanh phủ định (không chứa từ khóa) bị mô hình kích hoạt nhầm thành một trong 4 từ khóa. | Đo lường mức độ gây phiền nhiễu của hệ thống khi có người nói chuyện bình thường xung quanh thiết bị. |
| **Exact Poisson Rate Bound (Rule of Three)** | Công thức thống kê tính cận trên khoảng tin cậy 95% cho tốc độ xảy ra sự kiện hiếm: $\text{UCB}_{95\%} = -\ln(0.05)/T \approx 3/T$ khi quan sát thấy 0 sự kiện trong thời gian $T$. | Định lượng mức độ bất định khoa học của tỷ lệ báo động giả trong luồng streaming thời lượng ngắn (0.86 phút). |

---

## 3. Slide-by-Slide Presentation Checklist (Bảng kiểm tra từng slide)

```
+-------+----------------------------------+------------------------------------+---------------------------------------+
| Slide | Chủ đề slide                     | Số liệu / Hình ảnh bắt buộc chiếu  | Cảnh báo: Điều TUYỆT ĐỐI KHÔNG nói    |
+-------+----------------------------------+------------------------------------+---------------------------------------+
| S-01  | Tiêu đề & Ranh giới quản trị     | 4 nhãn quản trị bắt buộc           | KHÔNG nói "Hệ thống đã sẵn sàng chạy"  |
| S-02  | Kiến trúc hệ thống & Tham số     | 138.469 tham số, 4 blocks Conv2D   | KHÔNG nói mô hình dùng Transformer    |
| S-03  | Frontend Log-Mel & VAD           | 40 mels, 16kHz, VAD Band-Ratio     | KHÔNG nói dùng MFCC truyền thống      |
| S-04  | Chính sách quyết định 5 lớp      | Ngưỡng theta_cmd = 0.20 override   | KHÔNG nói dùng softmax argmax thuần   |
| S-05  | Tập dữ liệu & Hành trình 5 TTS   | VieNeu, Edge, MMS, gTTS, Piper     | KHÔNG nói Piper là engine hoàn toàn lạ|
| S-06  | Tập kiểm định nội miền (N=1.280) | v11: 78.05% (999/1280), D2: 93.91% | KHÔNG nói v11 đạt 88.59% accuracy     |
| S-07  | Đánh giá xác nhận Piper Phase 59 | D2_H1: OOV FAR 5.77%               | KHÔNG che giấu tỷ lệ recall của TRAI  |
| S-08  | Sụp đổ độ nhạy: Lệnh 'TRÁI'      | TRAI recall: 22.50% (9/40 đúng)    | KHÔNG tuyên bố D2_H1 vượt trội toàn bộ|
| S-09  | Đánh giá luồng liên tục Track B  | Cmd Recall: 17.24%, Event F1: 19.2%| KHÔNG tuyên bố streaming đã thành công|
| S-10  | Kết luận & Điều kiện tương lai   | Gate: PASS_WITH_DOCUMENTED_...     | KHÔNG hứa hẹn hiệu năng trên mic thật |
+-------+----------------------------------+------------------------------------+---------------------------------------+
```

---

## 4. List of Forbidden Claims (Danh sách các tuyên bố bị cấm)

Người thuyết trình **TUYỆT ĐỐI KHÔNG ĐƯỢC PHÁT BIỂU** các câu sau trong buổi bảo vệ:

1. ❌ *"Mô hình D2_H1 đã giải quyết triệt để bài toán nhận diện giọng nói tiếng Việt."*  
   *(Sự thật: D2_H1 bị sụp đổ độ nhạy trên từ `trái` [chỉ đạt 22.5%] và bỏ sót 82.8% lệnh trong luồng streaming).*
2. ❌ *"Mô hình đã sẵn sàng để tích hợp vào phần cứng hoặc robot thực tế."*  
   *(Sự thật: Hệ thống mang nhãn `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`).*
3. ❌ *"Baseline v11 đạt độ chính xác 88.59%."*  
   *(Sự thật: 88.59% là lỗi sao chép nhầm từ REJECT recall của D3_C. Độ chính xác thực tế của v11 là 78.05%).*
4. ❌ *"Hệ thống không bao giờ bị báo động giả trong môi trường im lặng vì đo được 0 sự kiện."*  
   *(Sự thật: Do thời lượng chỉ có 0.86 phút nên theo phân phối Poisson, cận trên báo động giả vẫn có thể lên tới 3.48 lần/phút).*
5. ❌ *"Mô hình đã hoạt động tốt trên giọng nói người thật."*  
   *(Sự thật: 100% dữ liệu thử nghiệm là giọng đọc máy tổng hợp từ các phần mềm TTS).*
6. ❌ *"Sự khác biệt giữa D2_H1 và D3_C là có ý nghĩa thống kê."*  
   *(Sự thật: Giá trị p = 0.0336 lớn hơn ngưỡng quy định trước alpha = 0.01 nên không có ý nghĩa thống kê).*
7. ❌ *"Chúng em đã tối ưu ngưỡng theta_cmd = 0.28 để đạt độ chính xác cao nhất 80.55% cho v11."*  
   *(Sự thật: Ngưỡng 0.20 được khóa bất biến theo giao thức quản trị; 80.55% chỉ là quan sát thăm dò).*
