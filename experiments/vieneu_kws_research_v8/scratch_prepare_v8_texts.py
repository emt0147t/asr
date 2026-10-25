import os
import sys

BASE_DIR = r"D:\trainidentificationaudio"
V6_EXP = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6")

# Load historical texts
hist_file = os.path.join(V6_EXP, "historical_texts.txt")
historical_set = set()
with open(hist_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().lower()
        if line:
            historical_set.add(line)

# Load v6 texts
sys.path.insert(0, V6_EXP)
from build_dataset_v6 import NON_TARGET_TEXTS
v6_set = set()
for split in ["train", "val", "test"]:
    for item in NON_TARGET_TEXTS[split]:
        v6_set.add(item["text"].strip().lower())

all_used_texts = historical_set.union(v6_set)
print(f"Total existing texts to avoid: {len(all_used_texts)}")

# Proposed 80 new test texts for Phase 74
NEW_TEST_TEXTS = [
    # 1. Minimal pair tone (16 texts: 4 per keyword)
    # TRAI (target: TRAI)
    {"text": "trại giống", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone nặng breeding farm"},
    {"text": "tráng gương phẳng", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone sắc mirror"},
    {"text": "trại nuôi tôm", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone nặng shrimp farm"},
    {"text": "trải nghiệm hay", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone hỏi experience"},
    # PHAI (target: PHAI)
    {"text": "phái viên quốc tế", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone sắc envoy"},
    {"text": "phai hương", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone ngang fading scent"},
    {"text": "phái nữ", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone sắc female"},
    {"text": "phạt nguội", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone nặng traffic fine"},
    # LEN (target: LEN)
    {"text": "lên chức vụ", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Promotion verb"},
    {"text": "lén nhìn trộm", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone sắc peeking"},
    {"text": "lên kế hoạch tuần", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Weekly planning"},
    {"text": "lẻn vào vườn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Sneaking verb"},
    # XUONG (target: XUONG)
    {"text": "xuống dốc đứng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Steep slope descent"},
    {"text": "xưởng may công nghiệp", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Garment factory"},
    {"text": "xướng âm điệu", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Solfege singing"},
    {"text": "xưởng mộc gia đình", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Carpentry workshop"},

    # 2. Minimal pair consonant (20 texts: 5 per keyword)
    # TRAI (target: TRAI)
    {"text": "loài thú quý", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial /l/ species"},
    {"text": "ngoài sân vườn", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial /ŋ/ outside"},
    {"text": "hoài bão lớn", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial /h/ ambition"},
    {"text": "soài cánh", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial /s/ wing flap"},
    {"text": "choài người", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial /tɕ/ reaching"},
    # PHAI (target: PHAI)
    {"text": "thoái trào", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /tʰ/ decline"},
    {"text": "ngoại giao đoàn", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /ŋ/ diplomatic"},
    {"text": "khoái cảm", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /kʰ/ sensation"},
    {"text": "hoại sinh", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /h/ saprophytic"},
    {"text": "loại hoa lan", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /l/ orchid species"},
    # LEN (target: LEN)
    {"text": "quét dọn nhà", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /kʷ/ cleaning"},
    {"text": "bện dây thừng", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /b/ braiding"},
    {"text": "mền ấm", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /m/ blanket"},
    {"text": "sên bò chậm", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /s/ snail"},
    {"text": "rên la đau", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /r/ groaning"},
    # XUONG (target: XUONG)
    {"text": "chuông chùa ngân", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial /tɕ/ bell"},
    {"text": "vuông vắn đẹp", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial /v/ square"},
    {"text": "buông tay ra", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial /b/ release"},
    {"text": "suông sẻ việc", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial /s/ smooth"},
    {"text": "tuông trào lệ", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Initial /t/ flowing"},

    # 3. Minimal pair vowel / coda (20 texts: 5 per keyword)
    # TRAI
    {"text": "tranh đông hồ", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Coda /ŋ/ painting"},
    {"text": "trang sách cũ", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Coda /ŋ/ page"},
    {"text": "trao giải thưởng", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Diphthong /aw/ award"},
    {"text": "trâu cày ruộng", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Diphthong /əw/ buffalo"},
    {"text": "trâm cài tóc", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Coda /m/ hairpin"},
    # PHAI
    {"text": "pháo hoa đêm", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Diphthong /aw/ fireworks"},
    {"text": "phong thư kín", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Coda /ŋ/ envelope"},
    {"text": "phòng họp số hai", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Coda /ŋ/ meeting room"},
    {"text": "phấn trắng viết", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Coda /n/ chalk"},
    {"text": "phím đàn piano", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Coda /m/ key"},
    # LEN
    {"text": "lưng ong đẹp", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Coda /ŋ/ waist"},
    {"text": "lung linh sắc", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Coda /ŋ/ glittering"},
    {"text": "lòng yêu nước", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Coda /ŋ/ heart"},
    {"text": "lang thang phố", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Coda /ŋ/ wandering"},
    {"text": "lăng tẩm cổ", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Coda /ŋ/ mausoleum"},
    # XUONG
    {"text": "xông hơi ấm", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Coda /ŋ/ sauna"},
    {"text": "xôn xao phố", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Coda /n/ bustling"},
    {"text": "xung quanh nhà", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Coda /ŋ/ surrounding"},
    {"text": "xanh ngắt trời", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Coda /ɲ/ blue"},
    {"text": "xâm lăng cấm", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Coda /m/ invade"},

    # 4. Phonetic phrase contexts & conversational compounds (24 texts: 6 per keyword)
    # TRAI
    {"text": "trái nghĩa từ", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Antonym noun"},
    {"text": "trải rộng đồng", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Spread verb"},
    {"text": "trại giam giữ", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Camp noun"},
    {"text": "trái gió trở trời", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Weather idiom"},
    {"text": "trải thảm đỏ", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Red carpet verb"},
    {"text": "trại huấn luyện viên", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Training camp"},
    # PHAI
    {"text": "phạt góc trận đấu", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Corner kick"},
    {"text": "phát hành sách mới", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Publish book"},
    {"text": "phát biểu ý kiến", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Speech verb"},
    {"text": "phân phát gạo", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Distribute rice"},
    {"text": "phát triển kinh tế", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Economic growth"},
    {"text": "phạt vi phạm hành chính", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Administrative fine"},
    # LEN
    {"text": "lên lịch họp trực tuyến", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Online meeting schedule"},
    {"text": "lên men sữa chua", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Yogurt fermentation"},
    {"text": "lên tiếng bảo vệ", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Speak up verb"},
    {"text": "lên sóng truyền hình", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Broadcast verb"},
    {"text": "lên dây đàn ghita", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Tune guitar"},
    {"text": "lên dốc cao nguyên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Highland climb"},
    # XUONG
    {"text": "xuống hầm gửi xe", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Basement parking descent"},
    {"text": "xuống phố đi bộ", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Walking street descent"},
    {"text": "xuống xe buýt nhanh", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Rapid bus exit"},
    {"text": "xuống ga tàu điện", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Metro station exit"},
    {"text": "xương ống nấu canh", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Soup bone noun"},
    {"text": "xưởng sửa chữa ô tô", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Car repair garage"}
]

print(f"Total new test texts: {len(NEW_TEST_TEXTS)}")
new_set = {x["text"].strip().lower() for x in NEW_TEST_TEXTS}
assert len(new_set) == 80, f"Expected 80 unique texts, got {len(new_set)}"

overlap = new_set.intersection(all_used_texts)
print(f"Overlap with all previous texts: {len(overlap)}")
if overlap:
    print(f"Overlapping texts: {overlap}")
    sys.exit(1)
print("SUCCESS: 0 overlap with any historical, v6, or earlier dataset texts!")
