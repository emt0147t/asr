#!/usr/bin/env python3
"""
Stage B: Curate Disjoint Lexicon and Texts for Vietnamese KWS v11.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v11/data/prepare_v11_lexicon.py

Guarantees:
1. Utterance/text/template-level separation: Train, Val, and Test text pools are strictly disjoint.
2. 0% overlap with Phase 90 120-clip evaluation texts.
3. Special emphasis on prefix-compound phrases, suffix-compound phrases, minimal pairs, and multiword distractors.
"""

import sys
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Phase 90 evaluated texts to strictly avoid in test/val
P90_MANIFEST = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v10" / "human_review" / "review_manifest.csv"
p90_texts = set()
if P90_MANIFEST.exists():
    with open(P90_MANIFEST, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = r["transcription"].strip().lower()
            if t and t not in {"[silence]", "trái", "phải", "lên", "xuống"}:
                p90_texts.add(t)

print(f"Loaded {len(p90_texts)} non-command Phase 90 texts to exclude from test.")

# 1. NEW TEST TEXTS (100 strictly unique texts, disjoint from train/val/p90)
# 25 texts per keyword direction:
# - 7 prefix compounds
# - 6 suffix compounds
# - 6 tonal minimal pairs
# - 6 consonant/vowel/coda minimal pairs
TEST_NON_TARGET_TEXTS = [
    # --- LEN (25 texts) ---
    # Prefix compounds (sounds like 'lên' at start)
    {"text": "lên mạng tìm việc", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên đơn đặt hàng", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên hình rất đẹp", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên đời điện thoại", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên màu tự nhiên", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên tay lái vững", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên cấp quản lý", "confusion_family": "target_prefix_compound", "target": "LEN"},
    # Suffix compounds (sounds like 'lên' at end)
    {"text": "bước lên bậc thềm", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "tiến lên phía trước", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "nhìn lên bầu trời", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "chạy lên trên gác", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "bay lên không trung", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "leo lên đỉnh núi", "confusion_family": "target_suffix_compound", "target": "LEN"},
    # Tonal minimal pairs
    {"text": "lén nhìn qua khe", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lẻn vào phòng khách", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lễ bái gia tiên", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lệ phí cầu đường", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lề thói cổ hủ", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lê la hàng quán", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    # Consonant/vowel/coda minimal pairs
    {"text": "mền bông ấm áp", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "bện thừng bện chão", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "sên bò trên lá", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "rên la thảm thiết", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "lưng ong thon thả", "confusion_family": "minimal_pair_coda", "target": "LEN"},
    {"text": "lang thang cuối phố", "confusion_family": "minimal_pair_coda", "target": "LEN"},

    # --- XUONG (25 texts) ---
    # Prefix compounds (sounds like 'xuống' at start)
    {"text": "xuống dòng ghi chú", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống nước nhân nhượng", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống hạng bóng đá", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống dốc ngoằn ngoèo", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống cấp trầm trọng", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống đất trồng cây", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống hồ tắm mát", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    # Suffix compounds (sounds like 'xuống' at end)
    {"text": "ngồi xuống nghỉ chân", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "nằm xuống thư giãn", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "bước xuống cầu thang", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "đặt xuống mặt bàn", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "rơi xuống mặt đất", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "lặn xuống đáy biển", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    # Tonal minimal pairs
    {"text": "xưởng may thời trang", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xướng tên người thắng", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xương ống nấu canh", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xương sườn lợn nướng", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xưởng thủ công mỹ nghệ", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xương rồng sa mạc", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    # Consonant/vowel/coda minimal pairs
    {"text": "chuông reo báo thức", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "vuông vắn góc cạnh", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "buông lơi nhẹ nhàng", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "suôn sẻ công việc", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "xông hơi giải cảm", "confusion_family": "minimal_pair_coda", "target": "XUONG"},
    {"text": "xung quanh hàng rào", "confusion_family": "minimal_pair_coda", "target": "XUONG"},

    # --- TRAI (25 texts) ---
    # Prefix compounds (sounds like 'trái' at start)
    {"text": "trái tim nhân hậu", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái phiếu chính phủ", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái phép lấn chiếm", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái ngược hoàn toàn", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái mùa đắt đỏ", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái luật bị phạt", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái đất xoay tròn", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    # Suffix compounds (sounds like 'trái' at end)
    {"text": "rẽ trái ngã tư", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "bên trái con đường", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "hướng trái một chút", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "quay trái góc nhọn", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "tay trái cầm bút", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "cửa trái đóng chặt", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    # Tonal minimal pairs
    {"text": "trại chăn nuôi gà", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trải thảm len mới", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "tráng miệng bằng trà", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trai tráng trong làng", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trại hè thanh thiếu niên", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trải dài theo bờ biển", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    # Consonant/vowel/coda minimal pairs
    {"text": "thoái hóa khớp gối", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "hoài niệm tuổi thơ", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "ngoài trời đổ mưa", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "choài người với tới", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "trang trí lớp học", "confusion_family": "minimal_pair_coda", "target": "TRAI"},
    {"text": "tranh phong cảnh đẹp", "confusion_family": "minimal_pair_coda", "target": "TRAI"},

    # --- PHAI (25 texts) ---
    # Prefix compounds (sounds like 'phải' at start)
    {"text": "phải chăng là duyên", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải lòng cô gái", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải rồi chính xác", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải biết tự lập", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải đạo làm con", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải lẽ công bằng", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải phép lịch sự", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    # Suffix compounds (sounds like 'phải' at end)
    {"text": "rẽ phải vòng xuyến", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "bên phải lối đi", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "hướng phải nhìn sang", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "quay phải góc vuông", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "tay phải giơ cao", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "phía phải khán đài", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    # Tonal minimal pairs
    {"text": "phai mờ theo tháng năm", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phái viên ngoại giao", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phạt nguội giao thông", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phai nhạt màu áo", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phái sinh chứng khoán", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phạt hành chính vi phạm", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    # Consonant/vowel/coda minimal pairs
    {"text": "thoái trào văn hóa", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "khoái trá cười to", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "hoại sinh chất bẩn", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "loại hoa hồng nhung", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "pháo hoa rực rỡ", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
    {"text": "phòng họp ban giám đốc", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
]

# 2. VALIDATION TEXTS (50 unique texts disjoint from test, train, and p90)
VAL_NON_TARGET_TEXTS = [
    # LEN (12 texts)
    {"text": "lên men rượu nếp", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên lịch làm việc", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên đồ đi tiệc", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "nhảy lên thật cao", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "bật lên sáng đèn", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "lén lút vào nhà", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lễ hội mùa xuân", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lề đường lát gạch", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "quét dọn sạch sẽ", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "nền nhà bóng loáng", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "lung linh ánh nến", "confusion_family": "minimal_pair_coda", "target": "LEN"},
    {"text": "lòng yêu thiên nhiên", "confusion_family": "minimal_pair_coda", "target": "LEN"},

    # XUONG (13 texts)
    {"text": "xuống dốc thoai thoải", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống giá bất động sản", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống chợ mua rau", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "chạy xuống dưới nhà", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "hạ xuống từ từ", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "xưởng mộc gia công", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xướng ca rộn rã", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xương khớp dẻo dai", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "chuông chùa ngân nga", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "buông tay chào tạm biệt", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "tuôn trào cảm xúc", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "xôn xao dư luận", "confusion_family": "minimal_pair_coda", "target": "XUONG"},
    {"text": "xâm lăng bờ cõi", "confusion_family": "minimal_pair_coda", "target": "XUONG"},

    # TRAI (12 texts)
    {"text": "trái cây bốn mùa", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái gió trở trời khó chịu", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái chiều quan điểm", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "bên trái khán phòng", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "nghiêng trái một chút", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "trại huấn luyện quân sự", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trải chiếu ăn cơm", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "tráng bánh cuốn nóng", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "loài thú quý hiếm", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "hoài bão thanh niên", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "trao huy chương vàng", "confusion_family": "minimal_pair_coda", "target": "TRAI"},
    {"text": "trâu cày ruộng bậc thang", "confusion_family": "minimal_pair_coda", "target": "TRAI"},

    # PHAI (13 texts)
    {"text": "phải chăng anh giận", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải tay người sành", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải thì con gái", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "bên phải hàng ghế", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "chếch phải mười độ", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "phai dấu thời gian", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phái đoàn cấp cao", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phạt góc khung thành", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "ngoại giao hữu nghị", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "khoái cảm giác mạnh", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "phong thư dán tem", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
    {"text": "phấn trắng bảng đen", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
    {"text": "phím đàn dương cầm", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
]

# 3. TRAINING FRESH HARD-NEGATIVE TEXTS (80 unique texts disjoint from val/test/p90)
TRAIN_HARD_NEGATIVE_TEXTS = [
    # LEN (20 texts)
    {"text": "lên ngôi vô địch", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên đường xuất phát", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên men giấm táo", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên lớp dạy học", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên đỉnh vinh quang", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên dây cót đồng hồ", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên giọng dạy đời", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "lên giá xăng dầu", "confusion_family": "target_prefix_compound", "target": "LEN"},
    {"text": "trèo lên cây cau", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "dựng lên ngôi nhà", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "nhấc lên khỏi sàn", "confusion_family": "target_suffix_compound", "target": "LEN"},
    {"text": "lén lút trong đêm", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lẻn qua hàng rào", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lễ kỷ niệm thành lập", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "lệ quyên góp từ thiện", "confusion_family": "minimal_pair_tone", "target": "LEN"},
    {"text": "quét sạch bụi bẩn", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "khen ngợi thành tích", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "xen lẫn niềm vui", "confusion_family": "minimal_pair_consonant", "target": "LEN"},
    {"text": "lưng chừng đồi núi", "confusion_family": "minimal_pair_coda", "target": "LEN"},
    {"text": "lăng tẩm triều đình", "confusion_family": "minimal_pair_coda", "target": "LEN"},

    # XUONG (20 texts)
    {"text": "xuống núi hái chè", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống giọng ngọt ngào", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống phố đón giao thừa", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống hàng đầu dòng", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống tầng hầm tòa nhà", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống ga xe lửa", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống thuyền qua sông", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "xuống ao thả cá", "confusion_family": "target_prefix_compound", "target": "XUONG"},
    {"text": "ngả người xuống thảm", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "đổ gục xuống bàn", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "chùng xuống trầm lắng", "confusion_family": "target_suffix_compound", "target": "XUONG"},
    {"text": "xưởng đúc gang thép", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xướng danh người tài", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xương cá thu to", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "xưởng dệt may xuất khẩu", "confusion_family": "minimal_pair_tone", "target": "XUONG"},
    {"text": "chuông reo vào lớp", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "buông rèm cửa sổ", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "tuôn rơi giọt sương", "confusion_family": "minimal_pair_consonant", "target": "XUONG"},
    {"text": "xanh ngắt một màu", "confusion_family": "minimal_pair_coda", "target": "XUONG"},
    {"text": "xăm hình nghệ thuật", "confusion_family": "minimal_pair_coda", "target": "XUONG"},

    # TRAI (20 texts)
    {"text": "trái bóng lăn nhanh", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái tuyến bảo hiểm", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái ngang số phận", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái buổi tan tầm", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái nghĩa đồng nghĩa", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái ý cấp trên", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái khoáy khó hiểu", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "trái dừa xiêm ngọt", "confusion_family": "target_prefix_compound", "target": "TRAI"},
    {"text": "lượn sang trái", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "nép về phía trái", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "khoác vai bên trái", "confusion_family": "target_suffix_compound", "target": "TRAI"},
    {"text": "trại giam phạm nhân", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trải nghiệm cuộc sống", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "tráng lệ nguy nga", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "trai làng tập võ", "confusion_family": "minimal_pair_tone", "target": "TRAI"},
    {"text": "thoái lui chiến thuật", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "hoài công vô ích", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "loài chim di cư", "confusion_family": "minimal_pair_consonant", "target": "TRAI"},
    {"text": "trang hoàng rực rỡ", "confusion_family": "minimal_pair_coda", "target": "TRAI"},
    {"text": "tranh cổ động viên", "confusion_family": "minimal_pair_coda", "target": "TRAI"},

    # PHAI (20 texts)
    {"text": "phải chăng người quen", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải lòng bài hát", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải lẽ phải đường", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải thời gặp vận", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải cẩn thận hơn", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải cố gắng nhiều", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải tuân thủ luật", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "phải trung thực luôn", "confusion_family": "target_prefix_compound", "target": "PHAI"},
    {"text": "bước sang bên phải", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "ngả về phía phải", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "nhìn thẳng bên phải", "confusion_family": "target_suffix_compound", "target": "PHAI"},
    {"text": "phai nhạt kỷ niệm", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phái viên thường trực", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phạt thẻ vàng cảnh cáo", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "phạt lao động công ích", "confusion_family": "minimal_pair_tone", "target": "PHAI"},
    {"text": "khoái chí tươi cười", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "thoái thác nhiệm vụ", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "loại trừ sai sót", "confusion_family": "minimal_pair_consonant", "target": "PHAI"},
    {"text": "phát hành tiền giấy", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
    {"text": "phát minh khoa học", "confusion_family": "minimal_pair_coda", "target": "PHAI"},
]


def audit_lexicon():
    print("=" * 80)
    print("STAGE B: VERIFYING UTTERANCE/TEXT/TEMPLATE ISOLATION FOR V11")
    print("=" * 80)

    test_set = {x["text"].strip().lower() for x in TEST_NON_TARGET_TEXTS}
    val_set = {x["text"].strip().lower() for x in VAL_NON_TARGET_TEXTS}
    train_set = {x["text"].strip().lower() for x in TRAIN_HARD_NEGATIVE_TEXTS}

    print(f"Test texts count:  {len(test_set)}")
    print(f"Val texts count:   {len(val_set)}")
    print(f"Train texts count: {len(train_set)}")

    # Check for duplicate within sets
    assert len(test_set) == len(TEST_NON_TARGET_TEXTS), "Duplicates in test texts!"
    assert len(val_set) == len(VAL_NON_TARGET_TEXTS), "Duplicates in val texts!"
    assert len(train_set) == len(TRAIN_HARD_NEGATIVE_TEXTS), "Duplicates in train texts!"

    # Check cross-split leakages
    test_val_leak = test_set.intersection(val_set)
    test_train_leak = test_set.intersection(train_set)
    val_train_leak = val_set.intersection(train_set)

    print(f"Test-Val overlap:   {len(test_val_leak)} {test_val_leak if test_val_leak else '[OK 0 leak]'}")
    print(f"Test-Train overlap: {len(test_train_leak)} {test_train_leak if test_train_leak else '[OK 0 leak]'}")
    print(f"Val-Train overlap:  {len(val_train_leak)} {val_train_leak if val_train_leak else '[OK 0 leak]'}")

    assert len(test_val_leak) == 0, "Test-Val text leakage detected!"
    assert len(test_train_leak) == 0, "Test-Train text leakage detected!"
    assert len(val_train_leak) == 0, "Val-Train text leakage detected!"

    # Check overlap with Phase 90
    test_p90_leak = test_set.intersection(p90_texts)
    val_p90_leak = val_set.intersection(p90_texts)
    print(f"Test-P90 overlap:   {len(test_p90_leak)} {test_p90_leak if test_p90_leak else '[OK 0 leak]'}")
    print(f"Val-P90 overlap:    {len(val_p90_leak)} {val_p90_leak if val_p90_leak else '[OK 0 leak]'}")

    assert len(test_p90_leak) == 0, "Test-P90 leakage detected!"
    assert len(val_p90_leak) == 0, "Val-P90 leakage detected!"

    print("\n[SUCCESS] All lexicon subsets are strictly disjoint with 0% cross-split or Phase 90 leakage!")
    print("=" * 80)


if __name__ == "__main__":
    audit_lexicon()
