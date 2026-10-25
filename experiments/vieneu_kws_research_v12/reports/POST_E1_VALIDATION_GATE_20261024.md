# Báo Cáo Kiểm Định Cổng Hậu E1 & Chẩn Đoán Phân Đoạn Dòng Liên Tục
## (Phase 14 — Post-E1 Validation Gate and Streaming Diagnostics)

> **Mã tài liệu:** `POST_E1_VALIDATION_GATE_20261024`  
> **Repository:** `D:\trainidentificationaudio`  
> **Thời điểm thẩm định:** 2026-10-24  
> **Phiên bản:** Phase 14 Formal Audit Release  
> **Quyết định cổng:** **`PASS_TO_VAD_DESIGN` (ĐỦ ĐIỀU KIỆN CHUYỂN SANG THIẾT KẾ NGHIÊN CỨU VAD)**  

---

## 1. Ranh Giới Quản Trị Khoa Học Bắt Buộc

Báo cáo này tuân thủ nghiêm ngặt 4 nhãn quản trị bất biến:
1. `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
2. `NOT_VALIDATED_ON_NATURAL_SPEECH`
3. `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
4. `NOT_VALIDATED_PENDING_DATA`

### Cam Kết Quản Trị Tuyệt Đối Trong Phase 14:
- Không huấn luyện lại bất kỳ mô hình nào (cả v11 lẫn E1).
- Không tiến hành thử nghiệm tăng tham số E2.
- Không sửa đổi, ghi đè hoặc xuất lại checkpoint v11 hay ONNX v11.
- Không thay đổi cấu hình mô hình demo mặc định của hệ thống.
- Không điều chỉnh ngưỡng $\theta_{\text{cmd}} = 0.20$ để tối ưu nhân tạo trên tập test đã quan sát.
- Không đưa bất kỳ mẫu test nào vào quá trình huấn luyện hay tinh chỉnh.
- Không tuyên bố hệ thống sẵn sàng triển khai (production-ready) hoặc đã được kiểm chứng trên tiếng nói tự nhiên ngoài đời thực.

### Bảng Kiểm Tra Mã Băm Mật Mã Cổ Vật Độc Lập (Cryptographic SHA-256 Audit)

| Tên Cổ Vật | Đường Dẫn Thực Tế | SHA-256 Kỳ Vọng | SHA-256 Đo Đạc Thực Tế | Trạng Thái Toàn Vẹn |
| :--- | :--- | :---: | :---: | :---: |
| **Checkpoint v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2` | **KHỚP 100% BYTE-FOR-BYTE** |
| **ONNX v11 đóng băng** | `experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99` | **KHỚP 100% BYTE-FOR-BYTE** |
| **Checkpoint E1 kiểm thử** | `experiments/vieneu_kws_research_v12/checkpoints/exp_v12_e1_baseline_scale_seed42.pt` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | `f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878` | **KHỚP 100% BYTE-FOR-BYTE** |

---

## 2. Kết Quả Thẩm Định Cổng Kiểm Toán (Audit Gate Summary)

| Hạng Mục Kiểm Toán | Kết Quả | Lý Do & Bằng Chứng Cốt Lõi |
| :--- | :---: | :--- |
| **Audit A: Tính hợp lệ của phân tích OOV** | **PASS** | Đã xác minh nguồn gốc từng dòng; phân loại rõ mẫu train là *training diagnostics*; xác nhận tính cô lập của từ "ghế" và ghi nhận benchmark test đã bị quan sát. |
| **Audit B: Xác minh Smoke Test Live-Mic** | **PASS** | Đã làm rõ đường đi tín hiệu thực tế: chạy phát lại tệp WAV ghi âm sẵn qua luồng xử lý phần mềm (*pipeline integration smoke test*), không phải thu âm trực tiếp qua microphone vật lý; chỉ ra lỗi giao diện dòng 602 trong `live_mic_engine.py`. |
| **Audit C: Phân rã lỗi Streaming theo mô hình** | **PASS** | Đã bóc tách chi tiết từng chuỗi và giải thích rõ ràng nguyên nhân con số 60 misses và 36 false insertions xuất hiện ở cả 2 mô hình do các cơ chế bù trừ khác nhau; xác định lỗi nghẽn VAD ở khoảng nghỉ ngắn. |
| **Audit D: Tính tái lập & Tính nhất quán số liệu** | **PASS** | Mọi lệnh và script tái chạy với mã thoát 0; đối soát khớp 100% tử số/mẫu số; giải trình triệt để mẫu số 8,000 clips (5,600 phát âm từ đơn + 2,400 âm bản chuyên biệt). |

---

## 3. Audit A — Kiểm Tra Tính Hợp Lệ Của Phân Tích OOV (OOV Validity Audit)

### 3.1 Xác Minh Phân Vùng Split Thực Tế Cho Các Từ `cửa`, `mở`, `bàn`, `nhà`
Rà soát toàn bộ các bảng kê `dataset_v12_manifest.csv`, `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv` và kiểm tra tệp vật lý trên đĩa:

| Từ OOV | Số Mẫu Thực Tế | Phân Tập Gốc (Actual Split) | Tệp Tồn Tại Trên Đĩa | Giọng Đọc (Voices) | Mã Phát Âm Nguồn (Source Utterances) |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`cửa`** | 16 | **`train`** (0 trong val, 0 trong test) | 16 / 16 (100%) | 4 giọng (`Adam`, `Hải`, `Minh`, `Thanh`) | 4 ID: `oov_train_cửa_Adam`, `oov_train_cửa_Hải_Đăng`, `oov_train_cửa_Minh_Đức`, `oov_train_cửa_Thanh_Bình` |
| **`mở`** | 19 | **`train`** (0 trong val, 0 trong test) | 19 / 19 (100%) | 6 giọng (`Adam`, `Minh Triết`, `Ngọc Huyền`, `Thanh Bình`, `Thái Sơn`, `Xuân Vĩnh`) | 6 ID: `vnkws_v2_unknown_0004`, `_0064`, `_0124`, `_0184`, `_0244`, `_0304` |
| **`bàn`** | 16 | **`train`** (0 trong val, 0 trong test) | 16 / 16 (100%) | 4 giọng (`Adam`, `Hải`, `Minh`, `Thanh`) | 4 ID: `oov_train_bàn_Adam`, `oov_train_bàn_Hải_Đăng`, `oov_train_bàn_Minh_Đức`, `oov_train_bàn_Thanh_Bình` |
| **`nhà`** | 16 | **`train`** (0 trong val, 0 trong test) | 16 / 16 (100%) | 4 giọng (`Adam`, `Hải`, `Minh`, `Thanh`) | 4 ID: `oov_train_nhà_Adam`, `oov_train_nhà_Hải_Đăng`, `oov_train_nhà_Minh_Đức`, `oov_train_nhà_Thanh_Bình` |

> [!IMPORTANT]
> **Tái Phân Loại Bằng Chứng Khoa Học (Reclassification of Evidence):**  
> Do các tệp âm thanh của `cửa` (16 clips), `mở` (19 clips), `bàn` (16 clips), `nhà` (16 clips) thực tế đều nằm trong tập **`train`** (và không xuất hiện trong `test`), kết quả nhận diện của E1 trên các từ này (từ chối 100% trên `mở`, `bàn`, `nhà` và 93.8% trên `cửa`) **BẮT BUỘC PHẢI ĐƯỢC PHÂN LOẠI LÀ CHẨN ĐOÁN TẬP HUẤN LUYỆN (TRAINING-SET DIAGNOSTICS)**.  
> Chúng chứng minh mô hình đã học tốt các mẫu âm bản được cung cấp, nhưng **KHÔNG ĐƯỢC COI LÀ BẰNG CHỨNG CỦA KHẢ NĂNG KHÁI QUÁT HÓA NGOÀI TỪ ĐIỂN (NOT GENERALIZATION EVIDENCE)**.

### 3.2 Đánh Giá 4 Clip Kiểm Thử Hồi Quy Lịch Sử Của Từ "Cửa"
- 4 clip: `oov_train_cửa_Adam.wav`, `_Hải_Đăng.wav`, `_Minh_Đức.wav`, `_Thanh_Bình.wav`.
- Đây chính là các file ghi lại lỗi nhận nhầm thành lệnh ở Phase 13.
- Kết quả: v11 nhận nhầm cả 4 tệp thành lệnh (`LEN` và `XUONG`), trong khi E1 từ chối cả 4 tệp về nhãn `REJECT` với độ tin cậy $\ge 0.988$.
- **Xác nhận:** Các tệp này được định danh chính xác là **mẫu kiểm thử hồi quy (known regression test fixtures)** để xác minh lỗi cũ không còn tái diễn, tuyệt đối không được gọi là bằng chứng kiểm thử mù độc lập.

### 3.3 Xác Minh Tính Cách Ly Của Từ "Ghế" và Nhóm OOV Kiểm Thử
- **Từ `ghế`:** 
  - Toàn bộ 16 tệp âm thanh của từ `ghế` chỉ xuất hiện trong **`test_manifest.csv`** (0 trong train, 0 trong val).
  - 4 giọng đọc của từ `ghế` (`Kim`, `Quốc`, `Quỳnh`, `Đức`) là các giọng đọc độc quyền của tập test, hoàn toàn không trùng lặp với bất kỳ giọng đọc nào trong train hay val.
  - Kết quả trên 16 mẫu test `ghế`: v11 nhận nhầm 10/16, E1 nhận nhầm 12/16 (đều bị nhầm vào lớp lệnh `LEN` do thanh sắc và nguyên âm /e/).
- **Toàn bộ 31 từ OOV kiểm thử (256 clips):**
  - Giao tập từ vựng: $\text{Train OOV (63 từ)} \cap \text{Test OOV (31 từ)} = \emptyset$.
  - Giao tập giọng đọc: $\text{Train Voices (700)} \cap \text{Test Voices (168)} = \emptyset$.
  - Giao tập mẫu gốc: $\text{Train Utterances} \cap \text{Test Utterances} = \emptyset$.

> [!WARNING]
> **Cảnh Báo Benchmark Đã Bị Quan Sát (Observed Benchmark Notice):**  
> Vì toàn bộ kết quả phân rã OOV trên tập test 1,280 mẫu (bao gồm hành vi nhận nhầm từ `ghế`, `khóa`, `hủy`, `làng`) đã được phân tích chi tiết và sử dụng để đề xuất các giải pháp khắc phục trong tương lai, **tập benchmark hiện tại chính thức được coi là ĐÃ BỊ QUAN SÁT (SEEN/OBSERVED BENCHMARK)**.  
> Trong các giai đoạn tiếp theo, nếu mô hình được điều chỉnh bổ sung các từ âm bản này, **nghiêm cấm sử dụng lại tập test này để tuyên bố thành tích khái quát hóa độc lập mới**. Mọi tuyên bố tổng quát hóa OOV trong tương lai bắt buộc phải được đánh giá trên một tập từ vựng và tập kiểm thử hoàn toàn mới được khóa trước.

---

## 4. Audit B — Xác Minh Bản Chất Smoke Test Live-Mic (Smoke Test Call-Path Verification)

Rà soát toàn diện mã nguồn trong `run_e1_live_demo_smoke_test.py`, `live_mic_engine.py`, `infer.py` và nhật ký `e1_live_smoke_test_events.log`:

### 4.1 Bằng Chứng Call-Path Thực Tế Của Smoke Test
Trích xuất đường dẫn thực thi thực tế trong hàm `feed_audio_stream()`:

```python
# Tệp: experiments/vieneu_kws_research_v12/scripts/run_e1_live_demo_smoke_test.py
# Dòng 48-57:
def load_clean_command(cmd_name: str) -> np.ndarray:
    candidates = sorted(list(P13_3_AUDIO_DIR.glob(f"v13_conf_cmd_{cmd_name}_*.wav")))
    audio, sr = sf.read(candidates[0], dtype="float32")  # ĐỌC TỆP WAV TỪ ĐĨA CỨNG
    ...

# Dòng 116-125:
def feed_audio_stream(audio_samples: np.ndarray, label_desc: str):
    chunk_size = 320  # Khối 20ms @ 16kHz
    for i in range(0, len(audio_samples), chunk_size):
        chunk = audio_samples[i : i + chunk_size]
        pipeline._on_new_audio_chunk(chunk)  # BƠM TRỰC TIẾP VÀO HÀM NHẬN SAMPLES TRONG BỘ NHỚ
```

**Đối chiếu với luồng thu âm vật lý:**
- Trong `run_e1_live_demo_smoke_test.py`, hàm `pipeline.stream_mgr.start()` (khởi động luồng `sounddevice.InputStream` để thu âm từ micro) **HOÀN TOÀN KHÔNG ĐƯỢC GỌI**.
- Tín hiệu âm thanh đi vào hệ thống là các mảng số thực float32 được nạp từ các tệp WAV ghi âm sẵn trên đĩa (`v13_conf_cmd_*.wav`, `dev_02_quiet_ambient_room.wav`, v.v.), chia nhỏ thành các khối 320 phần tử (tương đương 20 ms) và nạp tuần tự qua phương thức `_on_new_audio_chunk()`.

> [!IMPORTANT]
> **Hiệu Đính Thuật Ngữ Chính Thức (Mandatory Terminology Correction):**  
> Thử nghiệm trên **KHÔNG PHẢI LÀ KIỂM CHỨNG MICROPHONE TRỰC TIẾP (NOT LIVE-MICROPHONE VALIDATION)**.  
> Bản chất kỹ thuật chính xác của thử nghiệm này là:  
> **KIỂM THỬ TÍCH HỢP ĐƯỜNG ỐNG DỮ LIỆU BẰNG ÂM THANH GHI SẴN (RECORDED-AUDIO PIPELINE INTEGRATION SMOKE TEST)**.  
> Mọi tài liệu và báo cáo phải sử dụng thuật ngữ hiệu đính này, tuyệt đối không gây hiểu lầm rằng hệ thống đã được thử nghiệm qua giọng nói trực tiếp thu qua không khí từ microphone của người dùng.

### 4.2 Báo Cáo Rào Cản Giao Diện Dòng 602 & Tình Trạng Mã Nguồn Demo
- **Lỗi giao diện:** Trong `live_mic_engine.py` (dòng 602), mã nguồn nhánh PyTorch gọi `spec = self.inferencer.frontend(x)`. Tuy nhiên, lớp `VietnameseKWSInferencer` trong `infer.py` lại khởi tạo `self.frontend = LogMelFrontendNumPy()` (sử dụng phương thức `.compute()`). Lỗi `TypeError: 'LogMelFrontendNumPy' object is not callable` đã được bắt và cô lập hoàn toàn trong script kiểm thử bằng cách gán `LogMelFrontend()` cục bộ.
- **Bảo toàn demo mặc định:** Không có bất kỳ sửa đổi nào được thực hiện trên mã nguồn `experiments/vieneu_kws_research_v13_4/demo/live_mic_engine.py`. Demo mặc định của Phase 13.4 vẫn giữ nguyên 100% tệp mô hình ONNX v11 gốc.

---

## 5. Audit C — Phân Rã Lỗi Dòng Đa Lệnh Theo Từng Mô Hình (Streaming Error Decomposition)

Thẩm định trên toàn bộ 40 chuỗi phát âm thanh liên tục (tổng thời lượng: $159.74\text{ giây} = 2.66\text{ phút}$, gồm 124 sự kiện lệnh mục tiêu):

### 5.1 Bảng So Sánh Chỉ Số Tổng Hợp Bậc Chuỗi và Bậc Sự Kiện

| Tiêu Chí Đánh Giá Dòng (Streaming Metrics) | Baseline v11 Đóng Băng | Thử Nghiệm E1 (138k) | Chênh Lệch ($\Delta$) | Giải Thích Bản Chất |
| :--- | :---: | :---: | :---: | :--- |
| **Tổng số chuỗi kiểm thử** | 40 chuỗi | 40 chuỗi | 0 | 10 hoán vị, 8 lặp, 12 chèn OOV, 10 ép khoảng cách |
| **Tổng số sự kiện lệnh mục tiêu (Ground Truth)** | 124 sự kiện | 124 sự kiện | 0 | Các lệnh `TRAI`, `PHAI`, `LEN`, `XUONG` thực tế |
| **Số chuỗi nhận diện đúng hoàn toàn thứ tự** | **12 / 40 (30.00%)** | **19 / 40 (47.50%)** | **+17.50% (+7 chuỗi)** | E1 vượt trội ở các chuỗi chèn từ gây nhiễu |
| **Sự kiện lệnh nhận diện đúng (Matched Events)** | **64 / 124 (51.61%)** | **64 / 124 (51.61%)** | **0.00%** | Con số tổng hợp trùng khớp do cơ chế bù trừ |
| **Sự kiện lệnh bị bỏ lỡ (Missed Events)** | **60 / 124 (48.39%)** | **60 / 124 (48.39%)** | **0.00%** | Nghẽn chủ yếu ở khoảng nghỉ ngắn và nhiễu nặng |
| **Sự kiện kích hoạt trùng lặp (Duplicate Triggers)**| **4 sự kiện** | **0 sự kiện** | **-4 sự kiện** | **E1 loại bỏ triệt để 100% lỗi lặp trùng** |
| **Sự kiện chèn nhầm lệnh (False Insertions)** | **36 sự kiện** | **36 sự kiện** | **0 sự kiện** | Kích hoạt nhầm từ các âm tiết OOV xen kẽ |
| **Tần suất chèn nhầm trên phút âm thanh** | **13.52 lần/phút** | **13.52 lần/phút** | 0.00 | Đo trên luồng âm thanh mật độ cao ($2.66\text{ phút}$) |

---

### 5.2 Giải Trình Nguồn Gốc Số Liệu "60 Misses" và "36 False Insertions"
Trong báo cáo trước, cả v11 và E1 đều ghi nhận cùng con số 60 misses và 36 false insertions. Kiểm tra chi tiết từng chuỗi (sequence-by-sequence) xác nhận:

1. **Hai mô hình có kết quả dự đoán KHÁC NHAU trên 19/40 chuỗi** (chỉ có 21/40 chuỗi là có dự đoán hoàn toàn trùng nhau).
2. **Sự trùng hợp con số ở mức tổng hợp xuất phát từ cơ chế bù trừ nội tại:**
   - **Ở mô hình v11:** Mô hình bị lỗi kích hoạt quá nhạy (over-triggering), dẫn đến **4 sự kiện trùng lặp (duplicates)** (ví dụ: `seq_0020`, `seq_0026`, `seq_0029`, `seq_0030` bị nhận 2 lần cho cùng 1 từ phát âm). Trong một số chuỗi nhiễu, việc đoán bừa giúp v11 "vô tình" khớp được một số lệnh (nhưng sai thứ tự chuỗi), đồng thời chèn nhầm nhiều từ lệnh rác.
   - **Ở mô hình E1:** E1 loại bỏ hoàn toàn 4 lỗi trùng lặp (về 0), và từ chối chính xác các từ OOV trong điều kiện âm thanh sạch (giúp tăng từ 12 lên 19 chuỗi đúng hoàn toàn). Tuy nhiên, khi gặp các chuỗi nói dồn dập (khoảng nghỉ 160ms) hoặc nhiễu nặng kết hợp từ OOV thanh sắc, E1 phản ứng thận trọng hơn (dự đoán `REJECT`), dẫn đến số lệnh bị bỏ lỡ ở các chuỗi này tăng lên, bù trừ đúng vào số lệnh được cải thiện ở các chuỗi khác.
   - Do đó, tổng số matched events tình cờ cùng đạt $64$, dẫn đến $124 - 64 = 60\text{ misses}$ ở cả hai mô hình.
   - Về false insertions: v11 có 104 lần kích hoạt (64 khớp + 4 lặp + 36 chèn nhầm = 104). E1 có 100 lần kích hoạt (64 khớp + 0 lặp + 36 chèn nhầm = 100). Số lượng chèn nhầm tình cờ cùng bằng 36 nhưng phân bố trên các chuỗi khác nhau!

---

### 5.3 Bảng Phân Rã Lỗi Chi Tiết Theo Nhóm Điều Kiện (Báo Cáo Riêng v11 và E1)

#### A. Phân Rã Theo Cấu Trúc Chuỗi (Sequence Pattern)

| Nhóm Cấu Trúc Chuỗi | Số Chuỗi | v11 Chuỗi Đúng | E1 Chuỗi Đúng | v11 Bỏ Lỡ (Misses) | E1 Bỏ Lỡ (Misses) | v11 Chèn Nhầm (False Ins) | E1 Chèn Nhầm (False Ins) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`confuser_interspersed`** (Chèn từ OOV) | 12 | 2 / 12 (16.7%) | **7 / 12 (58.3%)** | 12 | 12 | 11 | 11 |
| **`permutation`** (Hoán vị 4 lệnh) | 10 | 2 / 10 (20.0%) | **3 / 10 (30.0%)** | 22 | 21 | 10 | 12 |
| **`repetition`** (Lệnh lặp lại) | 8 | 5 / 8 (62.5%) | **5 / 8 (62.5%)** | 10 | 11 | 7 | 7 |
| **`stress_long_gap`** (Nghỉ dài 900ms) | 5 | 3 / 5 (60.0%) | **4 / 5 (80.0%)** | 6 | 5 | 4 | 4 |
| **`stress_short_gap`** (Nghỉ ngắn 160ms) | 5 | **0 / 5 (0.0%)** | **0 / 5 (0.0%)** | 10 | 11 | 4 | 2 |

#### B. Phân Rã Theo Mức Độ Nhiễu Âm Học (Acoustic Condition)

| Mức Độ Nhiễu (SNR) | Số Chuỗi | v11 Chuỗi Đúng | E1 Chuỗi Đúng | v11 Bỏ Lỡ (Misses) | E1 Bỏ Lỡ (Misses) | v11 Chèn Nhầm (False Ins) | E1 Chèn Nhầm (False Ins) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clean`** (Âm thanh sạch) | 10 | 4 / 10 (40.0%) | **7 / 10 (70.0%)** | 4 | 5 | 4 | **0** |
| **`snr15`** (Nhiễu nhẹ 15 dB) | 10 | 3 / 10 (30.0%) | **5 / 10 (50.0%)** | 17 | 17 | 11 | 10 |
| **`snr10`** (Nhiễu vừa 10 dB) | 10 | 4 / 10 (40.0%) | **4 / 10 (40.0%)** | 19 | 20 | 14 | 14 |
| **`snr05`** (Nhiễu nặng 5 dB) | 10 | 1 / 10 (10.0%) | **3 / 10 (30.0%)** | 20 | 18 | 7 | 12 |

---

### 5.4 Truy Nguyên Lỗi Dính Cụm (Segmentation Merge Error Trace)
Kiểm tra log sự kiện thời gian thực tế chỉ ra lỗi hệ thống nghiêm trọng của bộ tách VAD hiện tại:
- Tại nhóm `stress_short_gap` (`seq_0031` đến `seq_0035`), khoảng nghỉ giữa các từ là $160-220\text{ ms}$.
- Thuật toán `UtteranceSegmenter` hiện tại đặt thông số `silence_interval_ms = 300` ($300\text{ ms}$).
- **Hệ quả:** Do khoảng nghỉ ($160\text{ ms}$) nhỏ hơn ngưỡng im lặng cần thiết ($300\text{ ms}$), bộ phân đoạn không nhận diện được điểm kết thúc của từ thứ nhất, tiếp tục tích lũy năng lượng và **gộp 2 từ liên tiếp thành 1 phân đoạn duy nhất**. Phân đoạn dính cụm này sau đó bị mô hình CNN từ chối (`REJECT`) hoặc nhận nhầm thành một từ khác, khiến $100\%$ các chuỗi nói dồn dập bị thất bại.

---

## 6. Audit D — Kiểm Tra Tính Tái Lập & Nhất Quán Số Liệu (Reproducibility & Consistency)

Toàn bộ các tập lệnh kiểm toán đã được tái thực thi trong môi trường chuẩn `.venv`:

| Lệnh Kiểm Toán Thực Thi | Mã Thoát (Exit Code) | Trạng Thái Nhật Ký Đầu Ra | Xác Nhận Chỉ Số |
| :--- | :---: | :--- | :---: |
| `.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/audit_dataset_v12.py --expected-count 8000` | **0** | Đạt 10/10 kiểm tra kỹ thuật tự động | Khớp 8,000 tệp |
| `.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/compare_v11_e1.py` | **0** | Xuất bảng đối chiếu 1,280 mẫu test | Khớp Accuracy 95.86%, Macro-F1 0.9601 |
| `.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/inspect_streaming_pairs.py` | **0** | Bóc tách chi tiết 40 chuỗi streaming | Khớp 64 match, 60 miss, 36 false ins |
| `.venv\Scripts\python.exe experiments/vieneu_kws_research_v12/scripts/audit_phase14_engine.py` | **0** | Xác minh phả hệ split, giọng đọc, OOV | Khớp 100% tính cách ly phân vùng |

### Xác Minh Quy Tắc 5-Lớp Argmax:
Hợp đồng suy luận đã được kiểm tra: Lớp `REJECT` tham gia đầy đủ vào argmax trên cả 5 lớp xác suất. Quyết định chấp nhận chỉ được đưa ra khi $\text{argmax} \in \{0, 1, 2, 3\}$ VÀ $P(\text{argmax}) \ge 0.20$. Không có hiện tượng bỏ qua lớp `REJECT` trong quá trình ra quyết định.

---

## 7. Thiết Kế Nghiên Cứu VAD Mới (VAD Study Protocol Design — Phase 15 Preparation)

> [!NOTE]
> **Ràng Buộc Tuyệt Đối:** Mục này **CHỈ THIẾT KẾ PHƯƠNG PHÁP NGHIÊN CỨU**, tuyệt đối không thay đổi mã nguồn VAD, debounce hay logic suy luận trong Phase 14 này.

### 7.1 Phương Pháp Phân Lập Lỗi Phân Đoạn vs Lỗi Phân Loại CNN
Để không đổ lỗi sai lệch giữa bộ tách phân đoạn (Segmenter) và bộ phân loại (CNN Classifier), hệ thống đánh giá mới sẽ phân tách thành 2 giai đoạn:
1. **Kiểm tra ranh giới phân đoạn (Segmentation Boundary Check):**  
   So sánh thời điểm bắt đầu $t_{\text{start}}$ và kết thúc $t_{\text{end}}$ của phân đoạn do VAD cắt ra với nhãn mốc thời gian thực tế (ground-truth event timestamps):
   - **Lỗi dính cụm (Merge Error):** Phân đoạn chứa $> 1$ từ phát âm thực tế.
   - **Lỗi cắt non (Premature Cut Error):** Phân đoạn bị cắt khi từ chưa phát âm xong ($t_{\text{end}} < t_{\text{gt\_end}} - 80\text{ ms}$).
   - **Lỗi kích hoạt ảo (Spurious Noise Segment):** Phân đoạn được cắt ra ở vùng không có tiếng người.
2. **Kiểm tra độ chính xác phân loại (Classification Accuracy):**  
   Chỉ đánh giá độ chính xác của CNN trên các phân đoạn đã được VAD cắt thành công (IoU thời gian $\ge 0.70$).

### 7.2 Đề Xuất Bộ Dữ Liệu Chuỗi Mới Cho Nghiên Cứu VAD (Chống Ô Nhiễm Benchmark)
Vì benchmark 40 chuỗi hiện tại đã bị quan sát toàn bộ kết quả, **tuyệt đối không dùng lại benchmark 40 chuỗi để tinh chỉnh thông số VAD**. Thiết kế đề xuất tạo lập 2 tập chuỗi mới:
- **Tập phát triển VAD (`vad_dev_corpus`):** 60 chuỗi âm thanh mới gồm các khoảng nghỉ có kiểm soát: $100\text{ ms}$, $150\text{ ms}$, $200\text{ ms}$, $250\text{ ms}$, $300\text{ ms}$, $400\text{ ms}$ trên các mức nhiễu $20, 15, 10, 5\text{ dB}$.
- **Tập kiểm thử VAD bị khóa (`vad_test_corpus`):** 60 chuỗi âm thanh độc lập hoàn toàn, được sinh với seed độc lập và khóa mã băm SHA-256 trước khi tiến hành bất kỳ so sánh thuật toán nào.

### 7.3 Các Ứng Viên Thuật Toán Cần Khảo Sát Trong Phase Tiếp Theo
1. **Baseline Hiện Tại:** Cổng năng lượng cố định (`Fixed Energy Gate`: $-48\text{ dB}$, thời gian chờ $300\text{ ms}$).
2. **Ứng viên 1 — Dual-Threshold Adaptive Hangover VAD:**  
   Sử dụng 2 ngưỡng năng lượng (ngưỡng bật $-42\text{ dB}$, ngưỡng tắt $-52\text{ dB}$) kết hợp thời gian chờ thích ứng: khi tốc độ nói tăng, thời gian chờ tự động co lại $150-180\text{ ms}$.
3. **Ứng viên 2 — Formant-Band Energy Ratio VAD:**  
   Theo dõi tỷ số năng lượng dải tần tiếng nói ($300 - 3400\text{ Hz}$) so với toàn dải phổ kết hợp bám vết sàn nhiễu động.

---

## 8. Quyết Định Cổng Phase 14 (Final Gate Decision)

> [!IMPORTANT]
> **QUYẾT ĐỊNH CHÍNH THỨC:** **`PASS_TO_VAD_DESIGN`**

### Căn Cứ Đưa Ra Quyết Định:
1. **Tính trung thực khoa học được xác lập:** Toàn bộ các phát hiện về OOV (phân biệt training diagnostics vs generalization), smoke test (hiệu đính thành pipeline integration trên audio ghi sẵn), và streaming error decomposition (giải trình cơ chế bù trừ của 60 misses / 36 false insertions) đã được làm rõ bằng chứng thực tế $100\%$.
2. **Tính toàn vẹn cổ vật được bảo vệ tuyệt đối:** Checkpoint v11 và ONNX v11 giữ nguyên bản byte-for-byte; Checkpoint E1 giữ nguyên mã băm `f3c0fe...`. Không có mô hình nào bị huấn luyện lại, xuất lại hay thay thế trong phase này.
3. **Đủ điều kiện chuyển tiếp:** Toàn bộ dữ liệu kiểm toán đã sẵn sàng làm cơ sở để tiến hành phase nghiên cứu và thiết kế chuyên biệt cho bộ phân đoạn VAD streaming.
