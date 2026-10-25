# Báo Cáo Kiểm Định Ngoại Tuyến Hồ Sơ B Thử Nghiệm
## (Phase 19 — Controlled Offline Profile B Validation)

> **Mã báo cáo:** `PHASE19_CONTROLLED_OFFLINE_PROFILE_VALIDATION_20261024`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm lập báo cáo:** 2026-10-24  
> **Giai đoạn trước:** Phase 18 — E1 ONNX Export & Controlled Profile B Integration (`PASS_TO_CONTROLLED_OFFLINE_PROFILE_VALIDATION`)  
> **Quyết định cổng nghiệm thu (Final Gate Decision):** **`PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`**  
> **Các nhãn quản trị bắt buộc:**
> - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
> - `NOT_VALIDATED_ON_NATURAL_SPEECH`
> - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
> - `NOT_VALIDATED_PENDING_DATA`
> - `OBSERVED_TEST_REPRODUCTION_NOT_INDEPENDENT_VALIDATION`

---

## 1. Tóm Tắt Chấp Hành & Phạm Vi Kiểm Định Ngoại Tuyến

Phase 19 thực thi kiểm định ngoại tuyến toàn diện cho đường ống tích hợp `experimental_e1_speechband` (Profile B) đã được triển khai ở Phase 18:
1. **Không kích hoạt microphone vật lý, không thu âm người mới:** Toàn bộ quá trình kiểm thử được thực hiện $100\%$ bằng việc tái hiện luồng âm thanh kỹ thuật số ngoại tuyến từ các băng ghi âm đã được phê duyệt.
2. **Bảo tồn $100\%$ hành vi của Hồ sơ mặc định (`legacy_v11`):** Xác minh không có bất kỳ sự suy thoái nào đối với mô hình v11 đóng băng, bộ phân đoạn mặc định, và ngưỡng quyết định canonical $\theta_{\text{cmd}} = 0.20$.
3. **Kiểm tra độ tin cậy khi khởi động & Kiểm thử ca lỗi (Negative Tests):** Xác minh cơ chế fail-fast tại biên khởi tạo; ngăn chặn mọi trường hợp âm thầm hạ cấp (silent fallback) về v11 khi gặp lỗi tệp hoặc sai mã băm.
4. **Tái hiện luồng trên tập phát triển (`vad_dev_corpus`):** Đo lường chi tiết hành vi tích hợp end-to-end trên 60 chuỗi âm thanh phát triển đa lệnh có nhiễu.
5. **Tái hiện độc lập trên tập kiểm thử đã xem (`vad_test_corpus`):**
   - Tái hiện chính xác $100\%$ kết quả đối chuẩn của Phase 15/16 trên công cụ đánh giá PyTorch gốc: Độ chính xác chuỗi đạt **$31/60 = 51.67\%$**, $6/24$ kích hoạt ảo OOV, F1 lệnh $0.8796$.
   - Đo lường thực tế trên đường ống tích hợp ONNX Runtime (`LiveKWSPipeline`): Độ chính xác chuỗi đạt **$32/60 = 53.33\%$**, $7/24$ kích hoạt ảo OOV, F1 lệnh $0.8659$.
   - Truy nguyên và tài liệu hóa chi tiết nguyên nhân sai lệch do khác biệt tính toán Log-Mel giữa PyTorch STFT và NumPy filterbank.
6. **Đánh giá tính khả dụng của tập kiểm thử độc lập (Gate E):** Nhận định khách quan rằng kho dữ liệu hiện tại chưa có tập kiểm thử luồng thứ ba được đóng băng trước, do đó hệ thống dừng ở mức nghiệm thu tái hiện âm thanh đã ghi (`PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`).

---

## 2. Bảng Đối Soát Mật Mã Cổ Vật & Môi Trường Thực Thi

### 2.1 Môi Trường & Phiên Bản Công Cụ
- **Hệ điều hành:** Windows 11 (win32, x86_64)
- **Python Runtime:** Python `3.11.9`
- **PyTorch:** `2.6.0+cu124`
- **ONNX:** `1.23.2`
- **ONNX Runtime:** `1.31.0`
- **Git Commit HEAD:** `471f1af` (`master`)

### 2.2 Đối Soát Mã Băm SHA-256 Byte-for-Byte

| Tên Cổ Vật / Tệp Dữ Liệu | Đường Dẫn Lưu Trữ | SHA-256 Xác Minh | Trạng Thái Toàn Vẹn |
| :--- | :--- | :---: | :---: |
| **v11 Checkpoint** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **100% Khớp Nguyên Bản** |
| **v11 ONNX** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **100% Khớp Nguyên Bản** |
| **E1 Checkpoint** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **100% Khớp Nguyên Bản** |
| **E1 ONNX (Exported)** | `experiments/vieneu_kws_research_v12/onnx/vietnamese_kws_v12_e1.onnx` | `20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8` | **100% Khớp Nguyên Bản** |
| **Selected VAD Config** | `experiments/vieneu_kws_research_v12/configs/selected_phase15_vad_config.json` | `af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3` | **100% Khớp Nguyên Bản** |
| **VAD Dev Manifest** | `experiments/vieneu_kws_research_v12/manifests/vad_dev_corpus_manifest.csv` | `1ebf11f8c7048e162e04d24df92b40e75b026c7059d9a9af5afcbab1274587c3` | **100% Khớp Nguyên Bản** |
| **VAD Test Manifest** | `experiments/vieneu_kws_research_v12/manifests/vad_test_corpus_manifest.csv` | `d0342e3f80c1307801c81637ba5b02e002c598aded35577b525afc106a8b5876` | **100% Khớp Nguyên Bản** |

### 2.3 Tái Kiểm Tra Tính Tương Đồng Số Học E1 ONNX vs PyTorch
Thực thi lại kịch bản `export_e1_onnx.py` để lấy bằng chứng thực thi sống:
- Sai số logit tuyệt đối lớn nhất: **$9.54 \times 10^{-6} \le 1.0 \times 10^{-5}$**
- Sai số xác suất tuyệt đối lớn nhất: **$8.34 \times 10^{-7} \le 1.0 \times 10^{-5}$**
- Tỷ lệ đồng thuận nhãn: **$10/10$ ($100\%$)**

---

## 3. Kết Quả Chi Tiết Từng Cổng Kiểm Định (Gate Audits)

### 3.1 Gate A: Kiểm Định Hồi Quy Hồ Sơ Mặc Định (Default Profile Regression)
- **Lệnh thực thi 1:** `.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/run_v11_regression_fixtures.py --verify`
  - Kết quả: **Exit code 0**. $18/18$ fixtures khớp $100\%$ quyết định và xác suất so với baseline tiền Phase 18 (`v11_pre_change_baseline.json`).
- **Lệnh thực thi 2:** `.venv\Scripts\python.exe -m pytest tests/test_vieneu_research_v13_4.py -v`
  - Kết quả: **Exit code 0**. $5/5$ tests passed ($100\%$).
- **Xác minh hành vi khởi động mặc định:**
  - Khởi chạy không tham số (`python live_mic_app.py`) tự động chọn profile `legacy_v11`.
  - Backend mặc định: `onnx` (nạp `vietnamese_kws_v11.onnx`, SHA-256: `54298c...`).
  - VAD mặc định: `UtteranceSegmenter` (khoảng lặng kết thúc $300\text{ ms}$, ngưỡng năng lượng $-48\text{ dB}$).
  - Quy tắc quyết định 5 lớp và ngưỡng $\theta_{\text{cmd}} = 0.20$ được bảo toàn tuyệt đối.
- **Kết luận Gate A:** **`PASS`** (Không có bất kỳ sự suy thoái nào).

### 3.2 Gate B: Khởi Động Hồ Sơ Thử Nghiệm & Kiểm Thử Ca Lỗi (Negative Tests)
Được kiểm tra thông qua bộ kiểm thử tự động `experiments/vieneu_kws_research_v12/tests/test_profile_b_negative_cases.py`:
- **Tổng số ca kiểm thử:** $7/7$ PASSED ($100\%$).
  1. `test_classes_order_and_threshold`: Bảo tồn thứ tự 5 lớp `['TRAI', 'PHAI', 'LEN', 'XUONG', 'REJECT']` và ngưỡng $0.20$.
  2. `test_reject_argmax_participation`: Xác minh lớp `REJECT` tham gia bình đẳng vào argmax; khi xác suất REJECT cao nhất hoặc khi xác suất lệnh $< 0.20$, hệ thống trả về nhãn `REJECT`.
  3. `test_model_output_dimensions`: Xác minh kích thước đầu ra mô hình ONNX đạt chuẩn $(B, 5)$ trên các kích thước batch động.
  4. `test_cli_negative_missing_e1_onnx`: Khi thiếu tệp ONNX của E1, chương trình báo lỗi ra stderr và dừng với exit code `1`, không âm thầm hạ cấp về v11.
  5. `test_cli_negative_corrupted_model_hash`: Khi tệp ONNX bị sai lệch mã băm, chương trình lập tức dừng với exit code `1`.
  6. `test_cli_negative_missing_vad_config`: Khi thiếu tệp cấu hình VAD, chương trình lập tức dừng với exit code `1`.
  7. `test_cli_negative_invalid_profile`: Khi truyền tên profile không hợp lệ, `argparse` từ chối và thoát với exit code khác `0`.
- **Kết luận Gate B:** **`PASS`**

### 3.3 Gate C: Tái Hiện Luồng Âm Thanh Tập Phát Triển (Gate C Development Replay)
Thực thi kiểm tra toàn diện trên 60 chuỗi âm thanh đã gộp của `vad_dev_corpus` thông qua đường ống `LiveKWSPipeline`:

| Chỉ Số Đánh Giá (Metric) | Kết Quả Thực Nghiệm Gate C | Ghi Chú Kỹ Thuật |
| :--- | :---: | :--- |
| **Tổng số chuỗi kiểm thử** | 60 chuỗi ($192$ lệnh chuẩn) | 6 mức khoảng cách $\times$ 5 điều kiện nhiễu |
| **Độ chính xác cả chuỗi (Exact-Sequence)** | **$32 / 60$ ($53.33\%$)** | Khớp hoàn hảo thứ tự và số lượng lệnh |
| **Số lệnh nhận diện đúng (Matched)** | **$161 / 192$ ($83.85\%$)** | $IoU \ge 0.20$ hoặc overlap $\ge 80\text{ ms}$ |
| **Số lệnh bị bỏ sót (Missed)** | $28 / 192$ ($14.58\%$) | Chủ yếu rơi vào khoảng cách cực ngắn $100\text{ ms}$ |
| **Số lệnh bị nhận nhầm lớp (Substitutions)**| $3 / 192$ ($1.56\%$) | Nhận nhầm giữa các lệnh hợp lệ |
| **Kích hoạt ảo trên từ OOV (OOV FA)** | **$6$ lần** | Tần số $3.43\text{ lần/phút OOV}$ |
| **Kích hoạt ảo trên khoảng lặng/nhiễu** | **$0$ lần** | Không có kích hoạt ảo trên nền nhiễu |
| **Kích hoạt trùng lặp (Duplicate triggers)**| **$0$ lần** | Cơ chế re-arm không sinh sự kiện kép |
| **Độ chuẩn xác lệnh (Command Precision)** | **$94.71\%$** | $161 / (161 + 3 + 6)$ |
| **Độ bao phủ lệnh (Command Recall)** | **$83.85\%$** | $161 / 192$ |
| **F1-Score lệnh** | **$0.8895$** | Trung bình điều hòa Precision & Recall |
| **Độ trễ trung vị (P50 Latency)** | **$287.6\text{ ms}$** | Tính từ khi kết thúc phát âm |
| **Độ trễ phân vị 95 (P95 Latency)** | **$460.0\text{ ms}$** | Thỏa mãn yêu cầu phản hồi thời gian thực |

- **Phân rã theo khoảng cách nghỉ giữa các lệnh (Gap Breakdown):**
  - Gap $100\text{ ms}$: $5/10$ chuỗi chuẩn xác, $25/32$ lệnh bắt được, $1$ FA.
  - Gap $150\text{ ms}$: $4/10$ chuỗi chuẩn xác, $27/32$ lệnh bắt được, $2$ FA.
  - Gap $200\text{ ms}$: $5/10$ chuỗi chuẩn xác, $28/32$ lệnh bắt được, $1$ FA.
  - Gap $250\text{ ms}$: $5/10$ chuỗi chuẩn xác, $27/32$ lệnh bắt được, $1$ FA.
  - Gap $300\text{ ms}$: $5/10$ chuỗi chuẩn xác, $24/32$ lệnh bắt được, $1$ FA.
  - Gap $400\text{ ms}$: $8/10$ chuỗi chuẩn xác, $30/32$ lệnh bắt được, $0$ FA.
- **Kết luận Gate C:** **`PASS`**

---

### 3.4 Gate D: Tái Hiện Độc Lập Trên Tập Kiểm Thử Đã Quan Sát (Gate D Observed Test Reproduction)

> [!IMPORTANT]
> **NHÃN ĐÁNH GIÁ BẮT BUỘC:**  
> **`OBSERVED_TEST_REPRODUCTION_NOT_INDEPENDENT_VALIDATION`**  
> Tập kiểm thử `vad_test_corpus` đã được quan sát trong Phase 15 và Phase 16. Kết quả dưới đây được sử dụng để kiểm chứng tính tái hiện của phần mềm, tuyệt đối không được dùng làm căn cứ tinh chỉnh tham số hoặc tuyên bố độ chính xác trên dữ liệu mới.

#### Bảng Đối Soát Chi Tiết Giữa Phase 16 Và Phase 19:

| Chỉ Số Đánh Giá (Metric) | Phase 16 (PyTorch Offline) | Phase 19 (ONNX Pipeline) | Tình Trạng Sai Khác | Nguyên Nhân Kỹ Thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Độ chính xác cả chuỗi (Exact-Sequence)** | **$31 / 60$ ($51.67\%$)** | **$32 / 60$ ($53.33\%$)** | Lệch $+1$ chuỗi | Do 3 chuỗi cải thiện và 2 chuỗi suy giảm ở vùng biên xác suất |
| **Tổng số lệnh mục tiêu (Target Events)** | $192$ | $192$ | **Khớp 100%** | Phản ánh đúng 60 chuỗi kiểm thử |
| **Số lệnh nhận đúng (Matched Commands)** | **$157$** | **$155$** | Lệch $-2$ lệnh | 2 lệnh ở điều kiện SNR 10dB rơi xuống dưới ngưỡng $0.20$ |
| **Số lệnh bỏ sót (Missed Commands)** | **$33$** | **$33$** | **Khớp 100%** | Hành vi phân đoạn VAD giống hệt nhau |
| **Số lệnh nhận sai lớp (Substitutions)**| **$2$** | **$4$** | Lệch $+2$ lệnh | Do 2 từ ở SNR 10dB bị lệch nhẹ phổ log-mel |
| **Kích hoạt ảo trên OOV (OOV FA)** | **$6 / 24$** | **$7 / 24$** | Lệch $+1$ lỗi | Phân đoạn OOV tại chuỗi 0018 bắt dính từ `Pháo`/`Phép` |
| **Kích hoạt ảo trên Im Lặng / Nhiễu** | **$0$** | **$0$** | **Khớp 100%** | Hoàn toàn không kích hoạt ảo trên nền nhiễu |
| **Kích hoạt lặp lại (Duplicates)** | **$0$** | **$0$** | **Khớp 100%** | Không sinh sự kiện trùng lặp |
| **Độ chuẩn xác lệnh (Command Precision)**| **$95.15\%$** | **$93.37\%$** | Lệch $-1.78\%$ | Mẫu số tăng do thêm 1 OOV FA và 2 substitutions |
| **Độ bao phủ lệnh (Command Recall)** | **$81.77\%$** | **$80.73\%$** | Lệch $-1.04\%$ | $155/192$ thay vì $157/192$ |
| **F1-Score lệnh** | **$0.8796$** | **$0.8659$** | Lệch $-0.0137$ | Tương thích chặt chẽ với chuẩn Phase 16 |
| **Độ trễ phân vị 95 (P95 Latency)** | **$517.6\text{ ms}$** | **$497.6\text{ ms}$** | Nhanh hơn $20\text{ ms}$ | Tốc độ thực thi ONNX Runtime nhanh hơn PyTorch |

#### Phân Tích Nguyên Nhân Sai Lệch Số Học Giữa PyTorch Và ONNX Runtime:
Khi phân tích sâu 11 chuỗi có sự dịch chuyển dự đoán giữa Phase 16 và Phase 19, nguyên nhân gốc rễ đã được làm sáng tỏ:
1. **Phân đoạn âm thanh là Bitwise Identical:** Bộ phân đoạn `SpeechBandRatioSegmenterAdapter` trả về các mốc thời gian `start_sec` và `end_sec` hoàn toàn trùng khớp $100\%$ với lớp gốc `SpeechBandRatioVAD` trên toàn bộ 60 chuỗi kiểm thử.
2. **Khác biệt tính toán Mel Filterbank:**
   - Trong Phase 15/16, kịch bản `eval_stage_b_streaming_kws.py` gọi lớp `KWSEvaluator`, tính Log-Mel qua thư viện C++ của PyTorch `torchaudio.transforms.MelSpectrogram`.
   - Trong Phase 18/19, `LiveKWSPipeline` chạy độc lập bằng pure-NumPy frontend `LogMelFrontendNumPy` và ONNX Runtime CPU execution provider.
   - Dù sai số trung bình giữa hai frontend là $\le 10^{-5}$ trên các fixture sạch, trên các đoạn âm thanh bị pha trộn nhiễu nặng (điều kiện SNR 10dB và SNR 5dB), sai khác làm tròn dấu phẩy động ở các bin tần số biên dẫn tới xác suất đầu ra của mô hình dao động nhẹ quanh ngưỡng cắt quyết định $\theta_{\text{cmd}} = 0.20$.
3. **Kiểm chứng ngược bằng PyTorch:**  
   Khi cho chạy lại mô hình PyTorch với `torchaudio` trên cùng luồng âm thanh đó, kết quả tái hiện **chính xác $100\%$ từng con số của Phase 16** (độ chính xác chuỗi đạt $31/60 = 51.67\%$).
- **Kết luận Gate D:** **`PASS`** (Tái hiện thành công, có tài liệu giải trình nguyên nhân số học đầy đủ).

---

## 4. Gate E: Đánh Giá Tính Khả Dụng Của Tập Kiểm Thử Độc Lập

### 4.1 Kiểm Toán Dữ Liệu Gốc & Ranh Giới Người Nói (Speaker Provenance Audit)
Kiểm tra toàn bộ các manifest chính thức trong kho lưu trữ `experiments/vieneu_kws_research_v12/manifests/`:
- Tập Huấn Luyện (`train_manifest.csv`): 700 giọng nói / người nói độc lập.
- Tập Phát Triển (`val_manifest.csv`): 164 giọng nói độc lập.
- Tập Kiểm Thử Đóng Băng (`test_manifest.csv`): 168 giọng nói độc lập.
- **Ranh giới cô lập người nói:** Không có bất kỳ sự giao thoa giọng nói nào giữa Train, Val và Test (Overlap = 0).

### 4.2 Tình Trạng Dữ Liệu Luồng Đa Lệnh (Sequential Streaming Corpus Status)
- Toàn bộ 60 chuỗi của `vad_dev_corpus` được trích từ tập `val`.
- Toàn bộ 60 chuỗi của `vad_test_corpus` được trích từ tập `test` và **đã bị quan sát trong Phase 15, Phase 16 và Phase 19**.
- Mặc dù tập `test_manifest.csv` còn lại một số mẫu đơn lẻ chưa ghép chuỗi, việc tạo ra một tập kiểm thử luồng mới đòi hỏi phải lập công thức ghép chuỗi mới, chọn khoảng nghỉ, chọn mức nhiễu, cố định mã băm và phê duyệt độc lập trước khi mở kết quả.
- Hiện tại trong repository **chưa có một tập kiểm thử luồng thứ ba nào đã được đóng băng từ trước**.

### 4.3 Tuyên Bố Quản Trị Khoa Học Bắt Buộc:
Theo đúng điều khoản quy định tại Mục 6 của chỉ thị Phase 19:
> *"The old vad_test_corpus cannot serve as a new blind test merely because its manifest hash remains unchanged. If an appropriate new corpus is unavailable, report: `BLOCKED_PENDING_INDEPENDENT_EVALUATION_CORPUS`. Continue to present the existing corpus results as reproduction only."*

Do đó, Phase 19 **từ chối tuyên bố hệ thống đã qua kiểm định mù độc lập mới**. Kết quả kiểm thử trên chỉ được ghi nhận dưới danh nghĩa tái hiện kỹ thuật (reproduction evidence).

---

## 5. Rủi Ro Kỹ Thuật Chưa Được Xử Lý (Unresolved Integration & Reliability Risks)

1. **Độ nhạy kích hoạt ảo trên từ OOV thanh sắc:**  
   Vấn đề nhận nhầm các từ OOV mang thanh sắc (`ghế`, `Pháo`, `Phép`, `Khóa`, `Cứu`, `Hủy`) vẫn tồn tại trong điều kiện nhiễu SNR 10dB (ghi nhận 7 lỗi trên 24 từ OOV kiểm thử).
2. **Khoảng cách lệnh cực ngắn ($< 150\text{ ms}$):**  
   Ở khoảng cách $100\text{ ms}$, tỷ lệ nhận diện chuỗi hoàn hảo chỉ đạt $30\%$ đến $50\%$ do đuôi âm thanh của lệnh trước bị bộ phân đoạn VAD cắt dính vào vùng đệm (preroll) của lệnh sau.
3. **Chưa có kiểm định trên tiếng nói tự nhiên đời thực:**  
   Tất cả các kết quả trên đều dựa trên dữ liệu tổng hợp/bán tự nhiên có kiểm soát. Tuyệt đối không được coi đây là hệ thống sẵn sàng triển khai thực địa (production-ready).

---

## 6. Quyết Định Cổng Nghiệm Thu Cuối Cùng (Final Gate Decision)

Căn cứ vào kết quả kiểm định 5 cổng:
- **Gate A (Default Regression):** `PASS` (Bảo tồn $100\%$ v11).
- **Gate B (Startup & Negative Cases):** `PASS` (Fail-fast chuẩn xác, cấm silent fallback).
- **Gate C (Development Replay):** `PASS` (Tái hiện luồng trên `vad_dev_corpus` trơn tru).
- **Gate D (Observed Test Reproduction):** `PASS` (Tái hiện được $51.67\%$ trên PyTorch và $53.33\%$ trên ONNX Runtime, giải trình rõ sai số Mel filterbank).
- **Gate E (Independent Corpus Eligibility):** `BLOCKED_PENDING_INDEPENDENT_EVALUATION_CORPUS` (Không có tập kiểm thử mù thứ 3).

Theo đúng danh mục quyết định tại Mục 8 của chỉ thị:

# **`PASS_TO_RECORDED_AUDIO_REPRODUCTION_ONLY`**

*(Đường ống tích hợp hoạt động ổn định và tái hiện được toàn bộ kết quả đã công bố; nhưng dừng lại ở mức nghiệm thu tái hiện âm thanh đã ghi sẵn do chưa có tập dữ liệu kiểm thử độc lập mới).*
