import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5")
DATA_DIR = os.path.join(EXP_DIR, "data")
AUDIO_RAW_DIR = os.path.join(DATA_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(DATA_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

SPLIT_VOICE_ALLOCATION = {
    "train": [
        "Adam", "Adam bựa", "Hải Đăng", "Minh Triết", "Minh Đức", "Mỹ Duyên",
        "Ngọc Huyền", "Ngọc Linh", "Quang Sơn", "Thanh Bình", "Thiền Tâm Đức",
        "Thiện Minh", "Thái Sơn", "Thục Đoan", "Trúc Ly", "Xuân Vĩnh", "Đoan Trang"
    ],
    "val": ["Mai Anh", "Ngọc Trân", "Phạm Tuyên", "Thùy Dung"],
    "test": ["Kim Thanh", "Quốc Tuấn", "Quỳnh Anh", "Đức Trí"]
}

# -----------------------------------------------------------------------------
# 200 Strictly Disjoint Non-Target Vietnamese Texts Across Splits
# Train: 80 texts, Val: 40 texts, Test: 80 texts
# -----------------------------------------------------------------------------
NON_TARGET_TEXTS = {
    "train": [
        # Minimal pair tone (16)
        {"text": "trài", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone huyền vs sắc"},
        {"text": "trải", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone hỏi vs sắc"},
        {"text": "trại", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone nặng vs sắc"},
        {"text": "tráng", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone sắc with velar coda"},
        {"text": "phài", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone huyền vs hỏi"},
        {"text": "phái", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone sắc vs hỏi"},
        {"text": "phại", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone nặng vs hỏi"},
        {"text": "phàn", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone huyền with nasal coda"},
        {"text": "lèn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone huyền vs ngang"},
        {"text": "lén", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone sắc vs ngang"},
        {"text": "lẻn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone hỏi vs ngang"},
        {"text": "lẹn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone nặng vs ngang"},
        {"text": "xuồng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone huyền vs sắc"},
        {"text": "xuổng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone hỏi vs sắc"},
        {"text": "xướng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone sắc with vowel /ɯə/"},
        {"text": "xưởng mộc", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Compound tone hỏi"},

        # Minimal pair consonant (20)
        {"text": "chái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial affricate /tɕ/"},
        {"text": "sái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial sibilant /s/"},
        {"text": "thải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial aspirate /tʰ/"},
        {"text": "thái", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /tʰ/ with sắc"},
        {"text": "khải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial velar aspirate /kʰ/"},
        {"text": "nén", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial nasal /n/"},
        {"text": "khen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /kʰ/"},
        {"text": "chen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /tɕ/"},
        {"text": "đen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial voiced /ɗ/"},
        {"text": "sen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Initial /s/"},
        {"text": "sương", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Sibilant /s/"},
        {"text": "sườn", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Alveolar /n/ coda"},
        {"text": "hướng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Glottal /h/"},
        {"text": "buông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Bilabial /b/"},
        {"text": "chuông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Affricate /tɕ/"},
        {"text": "vuông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Labiodental /v/"},
        {"text": "luồng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Liquid /l/"},
        {"text": "phát", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Stop /t/ coda"},
        {"text": "phạt", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Tone nặng + /t/"},
        {"text": "pháp", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Bilabial /p/ coda"},

        # Phrase contexts & compounds (20)
        {"text": "trái tim", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Heart compound"},
        {"text": "trái phiếu", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Bond compound"},
        {"text": "bên trái", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Locative phrase"},
        {"text": "rẽ trái", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Direction verb"},
        {"text": "phải chăng", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Modal phrase"},
        {"text": "bên phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Locative phrase"},
        {"text": "rẽ phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Direction verb"},
        {"text": "phân phái", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Faction compound"},
        {"text": "đi lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Motion verb"},
        {"text": "lên đường", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Travel compound"},
        {"text": "leo lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Climbing verb"},
        {"text": "đứng lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Posture verb"},
        {"text": "đi xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Motion verb"},
        {"text": "xuống dốc", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Descent phrase"},
        {"text": "trượt xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Sliding verb"},
        {"text": "ngồi xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Posture verb"},
        {"text": "lớp học", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "School compound"},
        {"text": "lực lượng", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Force compound"},
        {"text": "trát tường", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Plaster compound"},
        {"text": "trách móc", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Blame verb"},

        # General Neutral Speech (24)
        {"text": "bật", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "tắt", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "mở", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "đóng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "chạy", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "dừng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "xem", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Sensory verb"},
        {"text": "tìm", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Search verb"},
        {"text": "chọn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Selection verb"},
        {"text": "hủy", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Cancel verb"},
        {"text": "quay lại", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Return verb"},
        {"text": "tiếp tục", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Continue verb"},
        {"text": "âm lượng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Volume noun"},
        {"text": "nhiệt độ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Temperature noun"},
        {"text": "quạt gió", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Fan noun"},
        {"text": "đèn trần", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Light noun"},
        {"text": "ti vi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "TV noun"},
        {"text": "máy giặt", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Appliance noun"},
        {"text": "điều hòa", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "AC noun"},
        {"text": "tủ lạnh", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Fridge noun"},
        {"text": "phòng khách", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Room noun"},
        {"text": "nhà bếp", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Kitchen noun"},
        {"text": "thời tiết", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Weather noun"},
        {"text": "hôm nay", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Time noun"}
    ],

    "val": [
        # 40 Disjoint Validation Texts
        {"text": "trái cây", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Fruit compound"},
        {"text": "buôn bán", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Trade verb"},
        {"text": "chài lưới", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Affricate initial"},
        {"text": "sài gòn", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Proper noun"},
        {"text": "phôi pha", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone ngang"},
        {"text": "giải phóng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Verb compound"},
        {"text": "thảm hại", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Adjective"},
        {"text": "lái xe", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Driving verb"},
        {"text": "bái phục", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Respect verb"},
        {"text": "gái quê", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Noun phrase"},
        {"text": "dài ngoằng", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Adjective phrase"},
        {"text": "mài dao", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Sharpen verb"},
        {"text": "bài học", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Lesson noun"},
        {"text": "bải hoải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Tired adjective"},
        {"text": "hải sản", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Seafood noun"},
        {"text": "tải trọng", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Load noun"},
        {"text": "ngải cứu", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Herb noun"},
        {"text": "gãi đầu", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Scratch verb"},
        {"text": "vải vóc", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Fabric noun"},
        {"text": "nem chua", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Food noun"},
        {"text": "men say", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Yeast noun"},
        {"text": "sen hồng", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Lotus noun"},
        {"text": "quen thuộc", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Familiar adj"},
        {"text": "xen kẽ", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Alternate verb"},
        {"text": "xem xét", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Consider verb"},
        {"text": "tìm kiếm", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Search verb"},
        {"text": "chọn lọc", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Filter verb"},
        {"text": "bắt đầu", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Start verb"},
        {"text": "kết thúc", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "End verb"},
        {"text": "cửa sổ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Window noun"},
        {"text": "ánh sáng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Light noun"},
        {"text": "thông báo", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Notice noun"},
        {"text": "bản đồ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Map noun"},
        {"text": "tin nhắn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Message noun"},
        {"text": "cuộc gọi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Call noun"},
        {"text": "trợ lý", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Assistant noun"},
        {"text": "lịch hẹn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Calendar noun"},
        {"text": "đồng hồ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Clock noun"},
        {"text": "báo thức", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Alarm noun"},
        {"text": "nhắc nhở", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Reminder noun"}
    ],

    "test": [
        # 80 Disjoint Locked Test Texts
        {"text": "trải nghiệm", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Experience noun"},
        {"text": "trại nuôi", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Farm noun"},
        {"text": "trái mùa", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Offseason adj"},
        {"text": "trải dài", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Stretch verb"},
        {"text": "con trai", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Son noun"},
        {"text": "phai mờ", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Fade verb"},
        {"text": "phái đoàn", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Delegation noun"},
        {"text": "thời gian phai", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Phrase context"},
        {"text": "lèn đá", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Ram verb"},
        {"text": "lén lút", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Sneaky adj"},
        {"text": "lẻn vào", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Sneak verb"},
        {"text": "áo len", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Sweater noun"},
        {"text": "cuộn len", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Yarn roll noun"},
        {"text": "xuồng ba lá", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Canoe noun"},
        {"text": "xướng ca", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Singing verb"},
        {"text": "dòng xuồng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Boat row noun"},
        {"text": "nhà xưởng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Workshop noun"},
        {"text": "xương khớp", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Bone noun"},
        {"text": "lái tàu", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Steer ship verb"},
        {"text": "bái lạy", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Bow verb"},
        {"text": "gái đảm", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Capable girl"},
        {"text": "đường dài", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Long road noun"},
        {"text": "mài dũa", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Refine verb"},
        {"text": "bài ca", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Song noun"},
        {"text": "bải buôi", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Casual speech"},
        {"text": "hải đảo", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Island noun"},
        {"text": "tải về", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Download verb"},
        {"text": "ngải đắng", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Bitter herb"},
        {"text": "gãi ngứa", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Scratch itch"},
        {"text": "vải thiều", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Lychee noun"},
        {"text": "nem rán", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Fried roll noun"},
        {"text": "men bia", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Beer yeast noun"},
        {"text": "sen trắng", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "White lotus"},
        {"text": "làm quen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Befriend verb"},
        {"text": "đan xen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Intertwine verb"},
        {"text": "chén trà", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Tea cup noun"},
        {"text": "bèn hỏi", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Ask verb"},
        {"text": "nghẽn mạng", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Traffic jam noun"},
        {"text": "tròn trĩnh", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Plump adj"},
        {"text": "chớm nở", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Bloom verb"},
        {"text": "đơm hoa", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Blossom verb"},
        {"text": "vương giả", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Royal adj"},
        {"text": "lưỡng lự", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Hesitant adj"},
        {"text": "chải chuốt", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Groom verb"},
        {"text": "bãi cát", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Sand beach noun"},
        {"text": "mãi mãi", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Forever adv"},
        {"text": "thong thả", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Leisurely adv"},
        {"text": "vội vã", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Hurried adv"},
        {"text": "nhẹ nhàng", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Gentle adv"},
        {"text": "sâu lắng", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Deep adj"},
        {"text": "tươi vui", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Joyful adj"},
        {"text": "rộn ràng", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Bustling adj"},
        {"text": "êm đềm", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Peaceful adj"},
        {"text": "ấm áp", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Warm adj"},
        {"text": "trong trẻo", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Clear adj"},
        {"text": "dịu dàng", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Sweet adj"},
        {"text": "trầm ấm", "confusion_family": "minimal_pair_vowel_coda", "target": "NONE", "rationale": "Deep warm adj"},
        {"text": "xin chào bạn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Greeting sentence"},
        {"text": "tạm biệt nhé", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Farewell sentence"},
        {"text": "cảm ơn nhiều", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Thanks sentence"},
        {"text": "làm ơn giúp tôi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Request sentence"},
        {"text": "bây giờ là mấy giờ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Question sentence"},
        {"text": "thời tiết thế nào", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Question sentence"},
        {"text": "bật đèn phòng ngủ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Device command"},
        {"text": "tăng độ sáng lên chút", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Device command"},
        {"text": "giảm âm thanh đi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Device command"},
        {"text": "mở nhạc acoustic", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Media command"},
        {"text": "tắt máy lạnh", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Device command"},
        {"text": "kiểm tra email", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "App command"},
        {"text": "gọi điện cho mẹ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Phone command"},
        {"text": "dẫn đường về nhà", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Nav command"},
        {"text": "khóa cửa chính", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Security command"},
        {"text": "chụp một bức ảnh", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Camera command"},
        {"text": "ghi chú lại việc này", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Note command"},
        {"text": "lưu tài liệu", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Doc command"},
        {"text": "gửi tin nhắn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Message command"},
        {"text": "mở ứng dụng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "App command"},
        {"text": "khởi động lại hệ thống", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "System command"},
        {"text": "hẹn giờ mười lăm phút", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Timer command"},
        {"text": "hủy bỏ lịch trình", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Cancel command"}
    ]
}

TARGET_COMMAND_TEXTS = {
    "TRAI": "Trái",
    "PHAI": "Phải",
    "LEN": "Lên",
    "XUONG": "Xuống"
}

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def detect_speech_bounds(audio_16k: np.ndarray, sr: int = 16000):
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = (len(audio_16k) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return 0.0, len(audio_16k) / sr

    frames = np.array([audio_16k[i * hop_len : i * hop_len + frame_len] for i in range(num_frames)])
    energy = np.sum(frames ** 2, axis=1)
    max_e = np.max(energy) if len(energy) > 0 else 0
    if max_e < 1e-7:
        return 0.0, len(audio_16k) / sr

    thresh = max_e * 0.01
    active = np.where(energy > thresh)[0]
    if len(active) == 0:
        return 0.0, len(audio_16k) / sr

    onset_s = round(float(active[0] * hop_len) / sr, 3)
    offset_s = round(float((active[-1] * hop_len) + frame_len) / sr, 3)
    return onset_s, round(offset_s - onset_s, 3)

def format_to_1s(audio: np.ndarray, orig_sr: int, target_sr: int = 16000):
    if orig_sr != target_sr:
        num_target_samples = int(round(len(audio) * float(target_sr) / orig_sr))
        audio = signal.resample(audio, num_target_samples)

    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 10.0 ** (-1.0 / 20.0)
        audio = audio * (target_peak / peak)

    target_samples = 16000
    if len(audio) < target_samples:
        pad_total = target_samples - len(audio)
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        audio = np.pad(audio, (pad_front, pad_back), mode="constant", constant_values=0.0)
    elif len(audio) > target_samples:
        frame_len = int(0.025 * target_sr)
        hop_len = int(0.010 * target_sr)
        num_frames = (len(audio) - frame_len) // hop_len + 1
        energies = [np.sum(audio[i * hop_len : i * hop_len + frame_len] ** 2) for i in range(num_frames)]
        center_frame = int(np.argmax(energies))
        center_sample = center_frame * hop_len + frame_len // 2
        start = max(0, center_sample - target_samples // 2)
        if start + target_samples > len(audio):
            start = len(audio) - target_samples
        audio = audio[start : start + target_samples]

    onset_s, dur_s = detect_speech_bounds(audio, target_sr)
    return audio.astype(np.float32), onset_s, dur_s

def generate_synthetic_background(duration_samples: int, family: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if family == "near_silence":
        audio = rng.normal(0, 0.0008, duration_samples)
    elif family == "hvac_hum":
        t = np.arange(duration_samples) / 16000.0
        hum = 0.012 * np.sin(2 * np.pi * 50.0 * t) + 0.006 * np.sin(2 * np.pi * 100.0 * t)
        rumble = rng.normal(0, 0.008, duration_samples)
        b, a = signal.butter(3, 300.0 / 8000.0, btype="lowpass")
        rumble_filt = signal.lfilter(b, a, rumble)
        audio = hum + rumble_filt
    elif family == "office_ambient":
        raw = rng.normal(0, 0.012, duration_samples)
        b, a = signal.butter(2, [100.0 / 8000.0, 1200.0 / 8000.0], btype="bandpass")
        audio = signal.lfilter(b, a, raw)
    elif family == "airflow_pink":
        white = rng.normal(0, 0.015, duration_samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        audio = signal.lfilter(b, a, white)
    else:
        audio = rng.normal(0, 0.002, duration_samples)
    peak = np.max(np.abs(audio))
    if peak > 0.08:
        audio = audio * (0.08 / peak)
    return audio.astype(np.float32)

def main():
    print("=" * 80)
    print("Phase 56: Building Fresh Bounded-Utterance Dataset v5")
    print("=" * 80)

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Verify text disjointness
    t_train = {x["text"] for x in NON_TARGET_TEXTS["train"]}
    t_val = {x["text"] for x in NON_TARGET_TEXTS["val"]}
    t_test = {x["text"] for x in NON_TARGET_TEXTS["test"]}

    assert len(t_train.intersection(t_val)) == 0, "Train-Val text overlap detected!"
    assert len(t_train.intersection(t_test)) == 0, "Train-Test text overlap detected!"
    assert len(t_val.intersection(t_test)) == 0, "Val-Test text overlap detected!"
    print(f"Verified Strict Text Disjointness: {len(t_train)} Train, {len(t_val)} Val, {len(t_test)} Test texts (0 overlap).")

    # 2. Initialize TTS engine
    print("\n--- Initializing VieNeu TTS Engine (v3turbo) ---")
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")

    records = []

    # -------------------------------------------------------------
    # Part A: Target Commands
    # Train: 17 voices * 4 commands * 10 = 680 clips
    # Val:    4 voices * 4 commands * 10 = 160 clips
    # Test:   4 voices * 4 commands * 10 = 160 clips (FRESH SYNTHESIS!)
    # Total commands = 1,000 clips
    # -------------------------------------------------------------
    print("\n--- Generating Target Commands ---")
    label_map = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3}

    for split in ["train", "val", "test"]:
        voices = SPLIT_VOICE_ALLOCATION[split]
        for kw, text in TARGET_COMMAND_TEXTS.items():
            cid = label_map[kw]
            # 10 repetitions per voice per command
            rep_per_voice = 10
            for v_idx, vid in enumerate(voices):
                for r in range(rep_per_voice):
                    sid = f"v5_cmd_{split}_{kw.lower()}_{v_idx:02d}_{r:02d}"
                    raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                    kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                    # Vary temperature slightly to create acoustic variation
                    temp = round(0.70 + 0.02 * (r % 5), 2)
                    top_p = round(0.90 + 0.01 * (r % 5), 2)

                    if not os.path.exists(kws_path):
                        raw_audio = tts.infer(text, voice=vid, temperature=temp, top_p=top_p, repetition_penalty=1.20)
                        kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                        sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                        sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                    else:
                        kws_audio, _ = sf.read(kws_path, dtype="float32")
                        onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

                    kws_h = compute_sha256(kws_path)
                    records.append({
                        "sample_id": sid,
                        "dataset_version": "research_v5.0_bounded",
                        "class_label": kw,
                        "class_id": cid,
                        "bounded_label": kw,
                        "bounded_class_id": cid,
                        "text": text,
                        "voice_id": vid,
                        "split": split,
                        "seed": 500000 + len(records),
                        "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                        "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                        "kws_onset_sec": onset_s,
                        "kws_active_speech_duration_sec": dur_s,
                        "kws_sha256": kws_h,
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "num_samples": 16000,
                        "is_target_command": True,
                        "confusion_family": "target_command",
                        "target_keyword": kw,
                        "linguistic_rationale": f"Target keyword {kw}"
                    })
        print(f"Synthesized/Checked target commands for {split.upper()} split.")

    # -------------------------------------------------------------
    # Part B: Non-Target UNKNOWN Corpus
    # Train: 340 clips across 17 voices (cycling 80 train texts)
    # Val:   120 clips across 4 voices (30 per voice, cycling 40 val texts)
    # Test:  480 clips across 4 voices (120 per voice, cycling 80 test texts)
    # Total UNKNOWN = 940 clips
    # -------------------------------------------------------------
    print("\n--- Generating Non-Target UNKNOWN Corpus ---")
    for split, target_count in [("train", 340), ("val", 120), ("test", 480)]:
        voices = SPLIT_VOICE_ALLOCATION[split]
        texts = NON_TARGET_TEXTS[split]
        for i in range(target_count):
            vid = voices[i % len(voices)]
            t_info = texts[i % len(texts)]
            sid = f"v5_unk_{split}_{i+1:04d}"
            raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
            kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

            temp = round(0.72 + 0.02 * (i % 5), 2)
            top_p = round(0.91 + 0.01 * (i % 5), 2)

            if not os.path.exists(kws_path):
                raw_audio = tts.infer(t_info["text"], voice=vid, temperature=temp, top_p=top_p, repetition_penalty=1.20)
                kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
            else:
                kws_audio, _ = sf.read(kws_path, dtype="float32")
                onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

            kws_h = compute_sha256(kws_path)
            records.append({
                "sample_id": sid,
                "dataset_version": "research_v5.0_bounded",
                "class_label": "UNKNOWN",
                "class_id": 4,
                "bounded_label": "REJECT",
                "bounded_class_id": 4,
                "text": t_info["text"],
                "voice_id": vid,
                "split": split,
                "seed": 500000 + len(records),
                "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                "kws_onset_sec": onset_s,
                "kws_active_speech_duration_sec": dur_s,
                "kws_sha256": kws_h,
                "sample_rate": 16000,
                "num_channels": 1,
                "num_samples": 16000,
                "is_target_command": False,
                "confusion_family": t_info["confusion_family"],
                "target_keyword": t_info["target"],
                "linguistic_rationale": t_info["rationale"]
            })
        print(f"Synthesized/Checked non-target UNKNOWN for {split.upper()} split ({target_count} clips).")

    # -------------------------------------------------------------
    # Part C: SILENCE Corpus
    # Train: 340 clips (85 per condition)
    # Val:    80 clips (20 per condition)
    # Test:   80 clips (20 per condition)
    # Total SILENCE = 500 clips
    # -------------------------------------------------------------
    print("\n--- Generating SILENCE Corpus Across Noise Profiles ---")
    sil_conditions = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]
    for split, target_count in [("train", 340), ("val", 80), ("test", 80)]:
        per_cond = target_count // len(sil_conditions)
        for c_idx, cond in enumerate(sil_conditions):
            for i in range(per_cond):
                sid = f"v5_sil_{split}_{cond}_{i+1:03d}"
                raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                bg_seed = 500000 + len(records)
                audio_16k = generate_synthetic_background(16000, cond, bg_seed)
                # Resample to 48k for raw storage parity
                audio_48k = signal.resample(audio_16k, 48000).astype(np.float32)

                if not os.path.exists(kws_path):
                    sf.write(raw_path, audio_48k, 48000, subtype="PCM_16")
                    sf.write(kws_path, audio_16k, 16000, subtype="PCM_16")

                kws_h = compute_sha256(kws_path)
                records.append({
                    "sample_id": sid,
                    "dataset_version": "research_v5.0_bounded",
                    "class_label": "SILENCE",
                    "class_id": 5,
                    "bounded_label": "REJECT",
                    "bounded_class_id": 4,
                    "text": "[SILENCE]",
                    "voice_id": cond,
                    "split": split,
                    "seed": bg_seed,
                    "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                    "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                    "kws_onset_sec": 0.0,
                    "kws_active_speech_duration_sec": 1.0,
                    "kws_sha256": kws_h,
                    "sample_rate": 16000,
                    "num_channels": 1,
                    "num_samples": 16000,
                    "is_target_command": False,
                    "confusion_family": cond,
                    "target_keyword": "NONE",
                    "linguistic_rationale": f"Synthetic background noise condition {cond}"
                })
        print(f"Generated SILENCE for {split.upper()} split ({target_count} clips).")

    # -------------------------------------------------------------
    # Part D: Write Master Manifest & Split Manifests
    # -------------------------------------------------------------
    df_all = pd.DataFrame(records)
    master_path = os.path.join(DATA_DIR, "manifest.csv")
    df_all.to_csv(master_path, index=False, encoding="utf-8")
    print(f"\nWrote Master Manifest: {master_path} ({len(df_all)} records)")

    for split in ["train", "val", "test"]:
        split_df = df_all[df_all["split"] == split]
        split_path = os.path.join(SPLITS_DIR, f"{split}_manifest.csv")
        split_df.to_csv(split_path, index=False, encoding="utf-8")
        print(f"Wrote {split.upper()} Manifest: {split_path} ({len(split_df)} records)")

    print("\nDataset v5 construction complete!")

if __name__ == "__main__":
    main()
