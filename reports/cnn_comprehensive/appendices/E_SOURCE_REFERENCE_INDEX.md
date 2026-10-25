# Phụ Lục E: Chỉ Mục Tham Chiếu Nguồn (Source Reference Index)

Phụ lục này tổng hợp chỉ mục dẫn xuất tới các báo cáo kỹ thuật bản lề, các tệp mã nguồn chính thức và các tài liệu hợp đồng trong kho mã nguồn:

## 1. Các Báo Cáo Kỹ Thuật Bản Lề (Key Milestone Reports)
1. **[PHASE13_4_FINAL_CNN_RELEASE.md](file:///d:/trainidentificationaudio/reports/PHASE13_4_FINAL_CNN_RELEASE.md)**: Báo cáo đóng gói phát hành ứng viên đông kết cuối cùng, kiểm định đẳng cấu ONNX và xuất mã băm SHA-256.
2. **[PHASE13_3_CONFIRMATORY_EVALUATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_3_CONFIRMATORY_EVALUATION.md)**: Báo cáo thực chứng cuộc đánh giá xác nhận độc lập trên 1.000 clips của 8 giọng đọc mới lạ.
3. **[PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md](file:///d:/trainidentificationaudio/reports/PHASE13_2_1_FRONTEND_COMPATIBILITY_AUDIT.md)**: Báo cáo thẩm định giải quyết dứt điểm bất đồng $N_{\text{fft}} = 400$ vs $512$.
4. **[PHASE13_2_CONFIRMATORY_PREPARATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_2_CONFIRMATORY_PREPARATION.md)**: Báo cáo đăng ký trước giao thức, phân chia giọng đọc và danh mục từ vựng.
5. **[PHASE13_1_PREPROCESSING_REMEDIATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_1_PREPROCESSING_REMEDIATION.md)**: Báo cáo khắc phục lỗi khuếch đại tín hiệu im lặng và phê duyệt Sửa đổi 13.1-A.
6. **[PHASE13_CONFIRMATORY_EVALUATION.md](file:///d:/trainidentificationaudio/reports/PHASE13_CONFIRMATORY_EVALUATION.md)**: Báo cáo ghi nhận kiểm toán dừng khẩn cấp đợt đánh giá Phase 13 do phát hiện lỗi tiền xử lý.
7. **[PHASE12_CNN_FINALIZATION.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v12/reports/PHASE12_CNN_FINALIZATION.md)**: Báo cáo đóng gói ứng viên v12 ban đầu.
8. **[PHASE11_CNN_COMPLETION_CAMPAIGN.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v11/reports/PHASE11_CNN_COMPLETION_CAMPAIGN.md)**: Báo cáo chiến dịch hoàn thiện CNN v11, hàm mất mát Hard-Negative Weighted Cross Entropy.
9. **[PHASE92_CROSS_PLATFORM_PARITY.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/reports/PHASE92_CROSS_PLATFORM_PARITY.md)**: Báo cáo đối chuẩn chéo PyTorch - ONNX trên baseline `variant_w10`.
10. **[PHASE91_EDGE_DEPLOYMENT.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/reports/PHASE91_CNN_DEPLOYMENT.md)**: Báo cáo khởi tạo khung triển khai di động và C++.
11. **[PHASE90_HUMAN_LISTENING_STUDY.md](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/reports/PHASE90_HUMAN_LISTENING_VALIDATION.md)**: Báo cáo thử nghiệm nghe mù 120 clips trên người.

## 2. Các Tệp Mã Nguồn Chính Thức Được Chuẩn Y (Authoritative Source Code)
- **Mã Suy luận Đóng gói 13.4:** [`experiments/vieneu_kws_research_v13_4/bundle/infer.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/infer.py)
- **Kiến trúc Mạng `CorrectedVietnameseKWSNet`:** [`experiments/vieneu_kws_research_v10/deployment/export_model.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/deployment/export_model.py)
- **Frontend Trích xuất Log-Mel ($N_{\text{fft}}=512$):** [`experiments/vieneu_kws_research_v10/parity/portable_frontend.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v10/parity/portable_frontend.py)
- **Mã Huấn luyện Phase 11 Gốc:** [`experiments/vieneu_kws_research_v11/train_v11.py`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v11/train_v11.py)
- **Hợp đồng Tiền xử lý Chính thức:** [`experiments/vieneu_kws_research_v13_4/bundle/preprocessing_contract.json`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/preprocessing_contract.json)

## 3. Các Tệp Trọng Số Mô Hình Đông Kết (Frozen Weights)
- **File Trọng số PyTorch:** [`experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/checkpoint/variant_v11_hard_negative_weighted_seed42.pt)
- **File Mô hình ONNX:** [`experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx`](file:///d:/trainidentificationaudio/experiments/vieneu_kws_research_v13_4/bundle/onnx/vietnamese_kws_v11.onnx)
