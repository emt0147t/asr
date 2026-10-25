# Tổng kết cuối cùng — Nghiên cứu Vietnamese Keyword Spotting (KWS CNN)

**Phạm vi:** Giai đoạn tăng dataset lên 8.000 audio đến đóng gói Phase 67–69  
**Repository tham chiếu:** `D:\trainidentificationaudio`  
**Trạng thái theo báo cáo người dùng cung cấp:** `PASS_WITH_DOCUMENTED_RESEARCH_LIMITATIONS`  
**Model mặc định:** legacy v11, không thay đổi  
**Ngưỡng vận hành được khóa:** `theta_cmd = 0.20`  
**Độ lớn mô hình:** 138.469 tham số (`CorrectedVietnameseKWSNet`, E2 được hoãn)

> Tài liệu này tổng hợp các báo cáo đã được chia sẻ trong cuộc trò chuyện. Các số liệu là số liệu theo báo cáo của repository; chúng chưa được chạy lại độc lập từ repository trong môi trường biên soạn tài liệu này.

## 1. Kết luận điều hành

Từ mốc dataset v12 gồm 8.000 audio, dự án đã phát triển từ baseline KWS thành một pipeline nghiên cứu có nhiều checkpoint, frontend Log-Mel, VAD, inference ONNX, các tập đánh giá streaming, kiểm tra nguồn gốc dữ liệu, so sánh liên nguồn TTS, tái lập metric và hồ sơ quản lý artifact.

Kết quả lớn nhất không phải là một mô hình đã sẵn sàng triển khai, mà là nhận diện và đo lường rõ trade-off giữa hai mục tiêu:

- Từ chối tiếng nói ngoài tập lệnh (OOV) và giảm false alarm.
- Giữ độ nhạy đủ cao để phát hiện đúng bốn lệnh `TRAI`, `PHAI`, `LEN`, `XUONG`, đặc biệt trong âm thanh liên tục.

D2_H1 có kết quả tốt trong profile ưu tiên giảm false alarm trên bộ đánh giá tổng hợp Piper Phase 59, nhưng khả năng nhận diện liên tục còn yếu. Vì vậy, **không candidate nào được xác nhận sẵn sàng triển khai** và v11 vẫn là model mặc định tham chiếu.

## 2. Các chặng chính

### 2.1 Dataset 8.000 audio và baseline

Báo cáo lịch sử mô tả dataset v12 gồm:

| Thành phần | Số lượng |
|---|---:|
| Isolated utterances | 5.600 |
| — clean | 1.680 |
| — noisy | 3.920 |
| Specialized negatives | 2.400 |
| — FLEURS | 1.000 |
| — environmental noise | 800 |
| — silence | 600 |
| **Tổng** | **8.000** |

Số lượng file không đồng nghĩa số lượng mẫu độc lập thống kê: các phiên bản SNR/noise có thể cùng xuất phát từ một waveform, còn nhiều câu tổng hợp có thể cùng nguồn văn bản/voice. Các giai đoạn audit sau đã bổ sung theo dõi source ID, synthesis instance, noise provenance và cluster.

### 2.2 Tích hợp pipeline và streaming

Pipeline phát triển gồm frontend Log-Mel, model CNN, ONNX inference, VAD `SpeechBandRatioVAD`, quyết định đủ 5 lớp có `REJECT`, và quy tắc event matching cho streaming. Các lượt audit cũng phát hiện khác biệt frontend và lỗi phân đoạn/gộp sự kiện có ảnh hưởng đáng kể đến metric streaming.

### 2.3 Domain augmentation, hard negatives và candidate

Các điều kiện D0/D2 × H0/H1 qua nhiều seed được thử nghiệm trên cùng kiến trúc 138.469 tham số. D2 bổ sung một tỷ lệ mẫu Edge-TTS; H1 thay đổi cách lấy mẫu/nhấn mạnh hard negatives. Kết quả cho thấy một số cải thiện OOV, nhưng cũng bộc lộ xu hướng từ chối quá mức đối với một số lệnh hợp lệ.

Các candidate chính được giữ nguyên, không ghi đè lịch sử:
- v11: model tham chiếu mặc định.
- E1: baseline cải tiến.
- Phase 28 D2_H1: ứng viên ưu tiên low-false-alarm.
- Phase 26 D0_H1 seed44: ứng viên độ nhạy cao hơn trong một số đánh giá.
- D0_H1 seed42 và D3_A/B/C: các candidate/đối chứng phát triển.

### 2.4 Đánh giá qua nhiều nguồn tổng hợp

Các nguồn đã xuất hiện trong lịch sử dự án gồm VieNeu-TTS, Edge-TTS, MMS-TTS, gTTS và Piper VAIS-1000; các bộ nhiễu được xây dựng từ ESC-50. Các kết quả phải được đọc theo đúng phạm vi:

- Edge-TTS Phase 27/28: D2 được huấn luyện với các mẫu Edge-TTS từ hai voice cũng xuất hiện trong đánh giá; vì vậy đây không phải bằng chứng unseen-voice/unseen-generator.
- MMS-TTS Phase 43: thử thách dịch chuyển miền âm học, đặc biệt có lỗi nhận diện `TRAI`.
- gTTS Phase 51: một TTS service path khác, nhưng không đủ căn cứ để tuyên bố độc lập hoàn toàn về kiến trúc hoặc tập huấn luyện bên trong.
- Piper VAIS-1000 Phase 59: checkpoint/voice và nguồn dữ liệu tổng hợp chưa phơi nhiễm trong train/test trước đó theo audit được báo cáo; Piper runtime đã từng được khảo sát bằng checkpoint khác, nên diễn giải thận trọng là chuyển giao sang checkpoint/voice/corpus khác trong pipeline Piper/VITS.

## 3. Validation metrics cuối theo Phase 64–66

Các số liệu dưới đây là bảng validation đã được báo cáo là tái tính từ một prediction ledger chung trên `validation_v1.csv` (`N=1.280`, gồm 640 mẫu REJECT/OOV và 160 mẫu cho mỗi lớp lệnh).

| Model | Correct | Accuracy | 4-cmd Macro-F1 | 5-class Macro-F1 | Min command recall | OOV FAR | REJECT precision | REJECT recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 28 D2_H1 | 1.202 | 93,91% | 93,88% | 93,91% | 95,62% | 9,22% (59/640) | 97,48% | 90,78% |
| Phase 28 D0_H1 seed42 | 1.198 | 93,59% | 93,78% | 93,63% | 96,88% | 11,56% (74/640) | 99,12% | 88,44% |
| Phase 26 D0_H1 seed44 | 1.192 | 93,12% | 93,22% | 93,19% | 95,00% | 10,78% (69/640) | 97,27% | 89,22% |
| D3_C | 1.191 | 93,05% | 93,22% | 93,17% | 95,00% | 11,41% (73/640) | 97,76% | 88,59% |
| D3_A | 1.189 | 92,89% | 92,96% | 92,96% | 95,62% | 12,19% (78/640) | 98,77% | 87,81% |
| E1 | 1.177 | 91,95% | 92,22% | 92,14% | 95,62% | 13,91% (89/640) | 98,39% | 86,09% |
| D3_B | 1.164 | 90,94% | 91,43% | 91,25% | 91,88% | 15,00% (96/640) | 96,80% | 85,00% |
| v11 | 999 | 78,05% | 79,45% | 78,86% | 78,75% | 34,22% (219/640) | 91,32% | 65,78% |

**Lỗi báo cáo đã truy nguyên:** con số 88,59% từng xuất hiện như accuracy của v11 thực ra là REJECT recall của D3_C (`567/640`). Accuracy v11 dưới ngưỡng vận hành khóa `theta_cmd=0.20` là 78,05% (`999/1280`) theo prediction ledger được báo cáo.

Con số 80,55% của v11 được mô tả đúng hơn là mức accuracy cao nhất quan sát được trong một threshold sweep rời rạc tại `theta=0.28`; không nên gọi là cực đại lý thuyết nếu không đánh giá toàn bộ breakpoint quyết định có thể có. Threshold vận hành vẫn là 0,20.

## 4. Đánh giá xác nhận Phase 59 trên Piper VAIS-1000 + ESC-50 Fold 3

### 4.1 Track A — 420 clip chính

Tone probes là một tập riêng gồm 60 clip và không được cộng vào mẫu số 420 của Track A chính.

| Model | Accuracy | 4-cmd Macro-F1 | 5-class Macro-F1 | Target accuracy | OOV FAR | Tone probe FAR |
|---|---:|---:|---:|---:|---:|---:|
| D2_H1 | 74,52% (313/420) | 53,76% | 59,65% | 42,50% (68/160) | 5,77% (15/260) | 6,67% (4/60) |
| Phase 26 | 70,95% (298/420) | 39,02% | 47,13% | 40,00% (64/160) | 10,00% (26/260) | 3,33% (2/60) |
| D3_C | 69,29% (291/420) | 41,23% | 48,79% | 40,62% (65/160) | 13,08% (34/260) | 5,00% (3/60) |
| E1 | 63,33% (266/420) | 46,96% | 52,09% | 56,88% (91/160) | 32,69% (85/260) | 25,00% (15/60) |
| v11 | 43,33% (182/420) | 41,72% | 42,89% | 55,62% (89/160) | 64,23% (167/260) | 63,33% (38/60) |

### 4.2 Track B — 50 chuỗi streaming

Tổng thời lượng theo manifest là 218,201 giây (khoảng 3,6367 phút), với 58 sự kiện ground truth. Metric false-alarm rate phải luôn đi kèm tử số và thời lượng của cohort được chấm.

| Model | Exact sequence match | Command event F1 | Command recall | Command precision |
|---|---:|---:|---:|---:|
| D3_C | 50,0% (25/50) | 31,40% | 32,76% | 30,16% |
| D2_H1 | 46,0% (23/50) | 19,23% | 17,24% | 21,74% |
| Phase 26 | 44,0% (22/50) | 28,57% | 22,41% | 39,39% |
| E1 | 44,0% (22/50) | 30,08% | 34,48% | 26,67% |
| v11 | 26,0% (13/50) | 9,95% | 17,24% | 6,99% |

Cohort silence: `k=0`, `T=0,86045 min`, point estimate `0 FA/min`; cận trên Poisson chính xác một phía 95% khoảng `3,48 FA/min`.  
Cohort conversational OOV: `k=1`, `T=0,54155 min`, point estimate khoảng `1,85 FA/min`; cận trên Poisson một phía 95% khoảng `8,76 FA/min`.

Các cận trên này giả định mô hình Poisson phù hợp với quá trình sự kiện. Thời lượng streaming còn ngắn, vì vậy chúng không đảm bảo mức false-alarm rate trong vận hành thực tế.

## 5. Kết luận kỹ thuật

1. **D2_H1 là ứng viên mạnh cho mục tiêu giảm false alarm trong các điều kiện synthetic đã đăng ký.** Trên Track A Phase 59, OOV FAR tổng hợp là 5,77%, thấp hơn rõ so với v11 (64,23%).
2. **D2_H1 vẫn bỏ sót nhiều lệnh trong streaming.** Track A `TRAI` recall là 22,5%; Track B command recall 17,24% và event F1 19,23%. Không được dùng thành công của low-FA profile để kết luận KWS liên tục đã đạt yêu cầu.
3. **Không có model thắng toàn diện.** D2_H1 ưu tiên từ chối OOV; Phase 26 và D3_C có ưu/nhược riêng theo từng metric. Các candidate chưa được xác nhận trên giọng nói tự nhiên.
4. **Các phép thử là synthetic-only.** Cross-TTS/voice/corpus đánh giá độ bền trước một số dịch chuyển miền tổng hợp, nhưng không thay thế dữ liệu lời nói tự nhiên.
5. **Giới hạn provenance cần diễn đạt chính xác.** ESC-50 Fold 3 dùng waveform khác với các fold trước trong audit được báo cáo, nhưng một số category vẫn giống. Piper VAIS-1000 là checkpoint/corpus khác trong pipeline Piper/VITS; không nên gọi đó là engine family độc lập hoàn toàn.

## 6. Trạng thái quản trị cuối cùng

Theo báo cáo Phase 67–69, final closeout gate là `PASS_WITH_DOCUMENTED_RESEARCH_LIMITATIONS`.

Các nhãn bắt buộc giữ nguyên:

```text
EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
NOT_VALIDATED_ON_NATURAL_SPEECH
NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
NOT_VALIDATED_PENDING_DATA
```

- v11 vẫn là model mặc định.
- `theta_cmd=0.20` vẫn được giữ cố định.
- Không có thu thập giọng nói người thật hoặc đánh giá microphone vật lý.
- Không có model nào được phê duyệt triển khai.

## 7. Đề xuất kết thúc công việc hiện tại

- Giữ các report phase và prediction ledgers lịch sử ở trạng thái bất biến.
- Dùng `RESEARCH_FINAL_REPORT.md`, `REPRODUCIBILITY_GUIDE.md`, `FINAL_ARTIFACT_CATALOG.csv`, `FINAL_CLAIM_EVIDENCE_TABLE.csv` và `FINAL_SHA256_MANIFEST.json` làm hồ sơ gốc theo báo cáo đã tạo.
- Lưu một Git commit/tag cuối cùng sau khi kiểm tra trạng thái repo thực tế; không khẳng định repository clean nếu chưa kiểm tra trong môi trường repo.
- Chuyển trọng tâm sang báo cáo/slide bảo vệ. Chỉ bắt đầu nghiên cứu tiếp theo khi có giả thuyết mới, tiêu chí đo lường đăng ký trước và một tập kiểm chứng mới phù hợp.
