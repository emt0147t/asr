"""
Phase 46: Fresh Long-Horizon Synthetic Challenge Benchmark Generator.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v3/

Generates:
1. Disjoint text inventories across 5 linguistic confusion families (60 train, 30 val, 120 test).
2. 480 unique synthetic test non-target audio files using VieNeu-TTS v3turbo on 4 test voices.
3. 340 synthetic train non-target files on 17 train voices, 120 val non-target files on 4 val voices.
4. Balanced SILENCE audio files across 4 synthetic acoustic noise conditions.
5. 1,000 verified target command audio files incorporated from historical 6-class baseline.
6. Continuous streaming challenge files:
   - val_stream_continuous_commands.wav (~600s, 160 targets)
   - val_stream_continuous_nontarget.wav (~900s, 120 distractors)
   - test_stream_continuous_commands.wav (~600s, 160 targets)
   - test_stream_continuous_nontarget_3h.wav (EXACTLY 10,800s = 3.00 hours, 960 events, 2 passes of 480 clips)
7. Manifests and split files with SHA-256 checksums and ground truth JSONs.
"""

import os
import sys
import json
import time
import hashlib
import shutil
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")
AUDIO_RAW_DIR = os.path.join(DATA_DIR, "audio", "raw_48k")
AUDIO_KWS_DIR = os.path.join(DATA_DIR, "audio", "kws_16k")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
STREAMS_DIR = os.path.join(DATA_DIR, "continuous_streams")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

HISTORICAL_V2_DATA = os.path.join(BASE_DIR, "experiments", "synthetic_kws_dataset_vieneu_6class_v2")

# 25 Voices strictly allocated
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
# 210 Strictly Disjoint Non-Target Vietnamese Texts Across 5 Confusion Families
# Train: 60 texts, Val: 30 texts, Test: 120 texts
# -----------------------------------------------------------------------------
HARD_NEGATIVE_TEXTS = {
    "train": [
        # TRAI (12)
        {"text": "trai", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone ngang vs sắc"},
        {"text": "trài", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone huyền vs sắc"},
        {"text": "trải", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone hỏi vs sắc"},
        {"text": "trại", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone nặng vs sắc"},
        {"text": "chái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial affricate /tɕ/ vs /ʈ/"},
        {"text": "sái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial sibilant /s/ vs /ʈ/"},
        {"text": "trát", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Final stop /t/ vs glide /j/"},
        {"text": "tráo", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Nucleus /aːw/ vs /aːj/"},
        {"text": "trách", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Final stop /c/ vs glide /j/"},
        {"text": "trang", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Final velar nasal /ŋ/"},
        {"text": "trái cây", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Compound phrase context"},
        {"text": "bên trái", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Locative phrase context"},
        # PHAI (12)
        {"text": "phai", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone ngang vs hỏi"},
        {"text": "phài", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone huyền vs hỏi"},
        {"text": "phái", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone sắc vs hỏi"},
        {"text": "phại", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone nặng vs hỏi"},
        {"text": "thải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Aspirate dental /tʰ/ vs labiodental /f/"},
        {"text": "thái", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Dental /tʰ/ with tone sắc"},
        {"text": "khải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Velar aspirate /kʰ/ vs /f/"},
        {"text": "phát", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Final stop /t/ vs glide /j/"},
        {"text": "phạt", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Stop /t/ with tone nặng"},
        {"text": "pháp", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Bilabial stop /p/"},
        {"text": "phải chăng", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Modal phrase context"},
        {"text": "bên phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Directional phrase context"},
        # LEN (12)
        {"text": "lèn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone huyền vs ngang"},
        {"text": "lén", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone sắc vs ngang"},
        {"text": "lẻn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone hỏi vs ngang"},
        {"text": "lẹn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone nặng vs ngang"},
        {"text": "nén", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Nasal /n/ vs lateral /l/"},
        {"text": "khen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Velar /kʰ/ vs /l/"},
        {"text": "chen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Affricate /tɕ/ vs /l/"},
        {"text": "đen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Voiced stop /ɗ/ vs /l/"},
        {"text": "lớp", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Vowel /əː/ + stop /p/"},
        {"text": "lực", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Vowel /ɯ/ + velar /k/"},
        {"text": "đi lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Motion phrase context"},
        {"text": "lên đường", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Compound phrase context"},
        # XUONG (12)
        {"text": "xuồng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone huyền vs sắc"},
        {"text": "xuổng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone hỏi vs sắc"},
        {"text": "xướng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Vowel /ɯə/ vs /uə/"},
        {"text": "sương", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Sibilant /s/ vs /s/"},
        {"text": "sườn", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Alveolar nasal /n/ coda"},
        {"text": "hướng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Glottal /h/ vs /s/"},
        {"text": "buông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Bilabial /b/ vs /s/"},
        {"text": "chuông", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Affricate /tɕ/ vs /s/"},
        {"text": "xuất", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Final stop /t/ vs /ŋ/"},
        {"text": "xong", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Short vowel /awŋ/"},
        {"text": "đi xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Motion phrase context"},
        {"text": "xuống dốc", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Directional phrase context"},
        # Neutral (12)
        {"text": "bật", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common monosyllable"},
        {"text": "tắt", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common monosyllable"},
        {"text": "mở", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common monosyllable"},
        {"text": "đóng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common monosyllable"},
        {"text": "chạy", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "dừng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "nghe", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Sensory verb"},
        {"text": "nói", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Communication verb"},
        {"text": "hát", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Vocal verb"},
        {"text": "đọc", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "viết", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "xin chào", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Greeting phrase"}
    ],
    "val": [
        # 30 Disjoint Validation Texts
        {"text": "trãi", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone ngã vs sắc"},
        {"text": "mái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Initial nasal /m/ vs /ʈ/"},
        {"text": "tròn", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Rounded vowel /ɔn/"},
        {"text": "trại giam", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Disyllabic context"},
        {"text": "phãi", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone ngã vs hỏi"},
        {"text": "mải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Initial /m/ vs /f/"},
        {"text": "phay", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Vowel /a/ vs /aː/"},
        {"text": "không phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Negation phrase context"},
        {"text": "lẽn", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone ngã vs ngang"},
        {"text": "quên", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Velar labialized /kʷ/"},
        {"text": "lưng", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Vowel /ɯ/ + velar /ŋ/"},
        {"text": "tiến lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Directional verb context"},
        {"text": "xuỗng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone ngã vs sắc"},
        {"text": "hưởng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Glottal /h/ vs /s/"},
        {"text": "xanh", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Front vowel /e/ + palatal /ɲ/"},
        {"text": "quê hương", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Compound context"},
        {"text": "xem", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "tìm", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Common verb"},
        {"text": "chọn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Selection verb"},
        {"text": "bắt đầu", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Process verb"},
        {"text": "kết thúc", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Process verb"},
        {"text": "cửa sổ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Noun phrase"},
        {"text": "âm lượng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Control noun"},
        {"text": "ánh sáng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Control noun"},
        {"text": "nhiệt độ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Measurement noun"},
        {"text": "thông báo", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "System noun"},
        {"text": "bản đồ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Navigation noun"},
        {"text": "tin nhắn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Communication noun"},
        {"text": "cuộc gọi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Communication noun"},
        {"text": "trợ lý", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Assistant noun"}
    ],
    "test": [
        # 120 Disjoint Locked Test Texts
        # Tonal Minimal Pairs (20)
        {"text": "trãi nghiệm", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Compound /ʈaːj˦ˀ˥/"},
        {"text": "trại nuôi", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Compound /ʈaːj˨˩ˀ/"},
        {"text": "trái mùa", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Compound with sắc"},
        {"text": "trải dài", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Compound with hỏi"},
        {"text": "con trai", "confusion_family": "minimal_pair_tone", "target": "TRAI", "rationale": "Tone ngang context"},
        {"text": "phai mờ", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone ngang context"},
        {"text": "phái đoàn", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Tone sắc context"},
        {"text": "phôi phai", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Double tone ngang"},
        {"text": "thời gian phai", "confusion_family": "minimal_pair_tone", "target": "PHAI", "rationale": "Phrase final phai"},
        {"text": "lèn đá", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone huyền context"},
        {"text": "lén lút", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone sắc context"},
        {"text": "lẻn vào", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone hỏi context"},
        {"text": "sợi len", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone ngang context"},
        {"text": "áo len", "confusion_family": "minimal_pair_tone", "target": "LEN", "rationale": "Tone ngang context"},
        {"text": "xưởng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone hỏi /sɨəŋ˧˩˧/"},
        {"text": "xuồng ba lá", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone huyền context"},
        {"text": "xướng ca", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone sắc context"},
        {"text": "dòng xuồng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone huyền context"},
        {"text": "nhà xưởng", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone hỏi context"},
        {"text": "xương khớp", "confusion_family": "minimal_pair_tone", "target": "XUONG", "rationale": "Tone ngang /sɨəŋ/"},
        # Consonant Minimal Pairs (24)
        {"text": "lái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Liquid /l/ vs /ʈ/"},
        {"text": "bái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Stop /b/ vs /ʈ/"},
        {"text": "gái", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Velar /ɣ/ vs /ʈ/"},
        {"text": "dài", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Glide /z/ vs /ʈ/"},
        {"text": "mài", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Nasal /m/ vs /ʈ/"},
        {"text": "bài", "confusion_family": "minimal_pair_consonant", "target": "TRAI", "rationale": "Stop /b/ vs /ʈ/"},
        {"text": "bải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Bilabial /b/ vs /f/"},
        {"text": "hải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Glottal /h/ vs /f/"},
        {"text": "tải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Alveolar /t/ vs /f/"},
        {"text": "ngải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Velar nasal /ŋ/ vs /f/"},
        {"text": "gãi", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Velar /ɣ/ vs /f/"},
        {"text": "vải", "confusion_family": "minimal_pair_consonant", "target": "PHAI", "rationale": "Labiodental /v/ vs /f/"},
        {"text": "nem", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Nasal /n/ + bilabial /m/"},
        {"text": "men", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Nasal /m/ vs /l/"},
        {"text": "sen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Sibilant /s/ vs /l/"},
        {"text": "quen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Labialized /kʷ/ vs /l/"},
        {"text": "xen", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Sibilant /s/ vs /l/"},
        {"text": "ren", "confusion_family": "minimal_pair_consonant", "target": "LEN", "rationale": "Retroflex /ʐ/ vs /l/"},
        {"text": "luồng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Lateral /l/ vs /s/"},
        {"text": "dưỡng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Glide /z/ vs /s/"},
        {"text": "tường", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Dental /t/ vs /s/"},
        {"text": "lưỡng", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Lateral /l/ vs /s/"},
        {"text": "vương", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Labial /v/ vs /s/"},
        {"text": "nhường", "confusion_family": "minimal_pair_consonant", "target": "XUONG", "rationale": "Palatal /ɲ/ vs /s/"},
        # Vowel / Coda Minimal Pairs (24)
        {"text": "trước", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Diphthong /ɯə/ + stop /k/"},
        {"text": "trắng", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Vowel /ă/ + velar /ŋ/"},
        {"text": "tra", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Open vowel without coda"},
        {"text": "trâu", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Nucleus /əw/"},
        {"text": "trĩ", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "High vowel /i/"},
        {"text": "trụ", "confusion_family": "minimal_pair_vowel_coda", "target": "TRAI", "rationale": "Back vowel /u/"},
        {"text": "pháo", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Nucleus /aːw/ + stop /w/"},
        {"text": "phần", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Central /ə/ + nasal /n/"},
        {"text": "pha", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Open vowel without coda"},
        {"text": "phú", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "High back /u/"},
        {"text": "phi", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "High front /i/"},
        {"text": "phở", "confusion_family": "minimal_pair_vowel_coda", "target": "PHAI", "rationale": "Central vowel /əː/"},
        {"text": "leng", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Velar nasal /ŋ/ coda"},
        {"text": "lòng", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Back vowel /awŋ/"},
        {"text": "lệnh", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Palatal nasal /ɲ/ + nặng"},
        {"text": "lao", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Nucleus /aːw/"},
        {"text": "lo", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "Open /ɔ/"},
        {"text": "lu", "confusion_family": "minimal_pair_vowel_coda", "target": "LEN", "rationale": "High back /u/"},
        {"text": "xốp", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Mid vowel /o/ + stop /p/"},
        {"text": "xúc", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Back /u/ + stop /k/"},
        {"text": "xuyên", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Diphthong /iə/ + alveolar /n/"},
        {"text": "xứ", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "High central /ɯ/"},
        {"text": "xe", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Front vowel /ɛ/"},
        {"text": "xoài", "confusion_family": "minimal_pair_vowel_coda", "target": "XUONG", "rationale": "Glide /waːj/"},
        # Phrase Context Embeddings (24)
        {"text": "lái xe", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Automotive phrase"},
        {"text": "trái tim", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Anatomical phrase"},
        {"text": "trái phiếu", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Financial phrase"},
        {"text": "trái ngược", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Adverbial phrase"},
        {"text": "tay trái", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Body part phrase"},
        {"text": "hướng trái", "confusion_family": "phonetic_phrase_context", "target": "TRAI", "rationale": "Directional phrase"},
        {"text": "phần mềm", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Technical phrase"},
        {"text": "phải biết", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Modal phrase"},
        {"text": "phải lòng", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Idiomatic phrase"},
        {"text": "bắt buộc phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Obligation phrase"},
        {"text": "tay phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Body part phrase"},
        {"text": "hướng phải", "confusion_family": "phonetic_phrase_context", "target": "PHAI", "rationale": "Directional phrase"},
        {"text": "bước lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Motion verb phrase"},
        {"text": "lên lớp", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Educational phrase"},
        {"text": "lên kế hoạch", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Planning phrase"},
        {"text": "nổi lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Emergence phrase"},
        {"text": "bay lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Ascent phrase"},
        {"text": "nhảy lên", "confusion_family": "phonetic_phrase_context", "target": "LEN", "rationale": "Physical verb phrase"},
        {"text": "đi xuống phố", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Urban movement phrase"},
        {"text": "hạ xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Descent phrase"},
        {"text": "chảy xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Fluid verb phrase"},
        {"text": "rơi xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Gravity verb phrase"},
        {"text": "bước xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Step descent phrase"},
        {"text": "nhìn xuống", "confusion_family": "phonetic_phrase_context", "target": "XUONG", "rationale": "Visual verb phrase"},
        # Neutral Conversational Speech (28)
        {"text": "hỏi", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Inquiry verb"},
        {"text": "lưu", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Storage verb"},
        {"text": "xóa", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Deletion verb"},
        {"text": "cảm ơn", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Courtesy phrase"},
        {"text": "hẹn gặp lại", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Farewell phrase"},
        {"text": "thời tiết hôm nay", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Query phrase"},
        {"text": "tin tức buổi sáng", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Query phrase"},
        {"text": "báo thức", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Device feature"},
        {"text": "máy tính", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Hardware noun"},
        {"text": "điện thoại", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Device noun"},
        {"text": "tìm kiếm", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Action verb"},
        {"text": "kết nối", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Network verb"},
        {"text": "lịch làm việc", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Schedule noun"},
        {"text": "ghi chú", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Productivity noun"},
        {"text": "hủy bỏ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Command verb"},
        {"text": "đồng ý", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Confirmation verb"},
        {"text": "từ chối", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Rejection verb"},
        {"text": "trở lại", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Navigation verb"},
        {"text": "tiếp tục", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Continuation verb"},
        {"text": "hoàn thành", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Completion verb"},
        {"text": "cài đặt", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Settings noun"},
        {"text": "hệ thống", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "System noun"},
        {"text": "trang chủ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Home noun"},
        {"text": "tài khoản", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Account noun"},
        {"text": "bảo mật", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Security noun"},
        {"text": "thông minh", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Adjective"},
        {"text": "công nghệ", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Domain noun"},
        {"text": "phát triển", "confusion_family": "neutral_speech", "target": "NONE", "rationale": "Abstract verb"}
    ]
}


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
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
    print("Phase 46: Building Vietnamese KWS Benchmark v3 (Fresh Long-Horizon 3-Hour Test)")
    print("=" * 80)

    os.makedirs(AUDIO_RAW_DIR, exist_ok=True)
    os.makedirs(AUDIO_KWS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(STREAMS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Verify strict text disjointness
    train_texts = {x["text"] for x in HARD_NEGATIVE_TEXTS["train"]}
    val_texts = {x["text"] for x in HARD_NEGATIVE_TEXTS["val"]}
    test_texts = {x["text"] for x in HARD_NEGATIVE_TEXTS["test"]}

    assert len(train_texts.intersection(val_texts)) == 0, "Train-Val text overlap detected!"
    assert len(train_texts.intersection(test_texts)) == 0, "Train-Test text overlap detected!"
    assert len(val_texts.intersection(test_texts)) == 0, "Val-Test text overlap detected!"
    print(f"Verified Text Disjointness: {len(train_texts)} Train, {len(val_texts)} Val, {len(test_texts)} Test (0 overlap).")

    # 2. Initialize Vieneu TTS engine
    print("\n--- Initializing VieNeu TTS Engine ---")
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")

    records = []
    text_inventory_records = []
    text_counter = 1

    # -----------------------------------------------------------------
    # Part A: Target Commands (1,000 utterances from historical v2)
    # -----------------------------------------------------------------
    print("\n--- Incorporating 1,000 Approved Target Commands ---")
    hist_manifest = pd.read_csv(os.path.join(HISTORICAL_V2_DATA, "manifest.csv"))
    target_hist = hist_manifest[hist_manifest["is_target_command"] == True].copy()
    assert len(target_hist) == 1000, f"Expected 1,000 target commands, found {len(target_hist)}"

    for _, row in target_hist.iterrows():
        src_kws = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
        src_raw = os.path.join(BASE_DIR, row["raw_wav_path"]) if not os.path.isabs(row["raw_wav_path"]) else row["raw_wav_path"]

        sid = row["sample_id"]
        dst_kws = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")
        dst_raw = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")

        if not os.path.exists(dst_kws):
            shutil.copy2(src_kws, dst_kws)
        if not os.path.exists(dst_raw):
            shutil.copy2(src_raw, dst_raw)

        rec = row.to_dict()
        rec["dataset_version"] = "research_v3.0"
        rec["kws_wav_path"] = os.path.relpath(dst_kws, BASE_DIR).replace("\\", "/")
        rec["raw_wav_path"] = os.path.relpath(dst_raw, BASE_DIR).replace("\\", "/")
        rec["confusion_family"] = "target_command"
        records.append(rec)

    # -----------------------------------------------------------------
    # Part B: Synthesize Hard-Negative UNKNOWN Corpus
    # -----------------------------------------------------------------
    print("\n--- Synthesizing Hard-Negative Corpus Across Splits ---")
    # Train: 60 texts. 17 voices -> 340 utterances (20 per voice)
    # Val: 30 texts. 4 voices -> 120 utterances (each voice speaks all 30 texts)
    # Test: 120 texts. 4 voices -> 480 utterances (each voice speaks all 120 texts)

    for split in ["train", "val", "test"]:
        voice_list = SPLIT_VOICE_ALLOCATION[split]
        text_list = HARD_NEGATIVE_TEXTS[split]
        print(f"\nProcessing {split.upper()} split: {len(text_list)} texts across {len(voice_list)} voices...")

        if split == "train":
            target_count = 340
            t_idx = 0
            for i in range(target_count):
                vid = voice_list[i % len(voice_list)]
                t_info = text_list[t_idx % len(text_list)]
                t_idx += 1

                sid = f"v3_unk_{split}_{i+1:04d}"
                raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                if not os.path.exists(kws_path):
                    raw_audio = tts.infer(t_info["text"], voice=vid, temperature=0.75, top_p=0.92, repetition_penalty=1.20)
                    kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                    sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                    sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                else:
                    kws_audio, _ = sf.read(kws_path, dtype="float32")
                    onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

                kws_h = compute_sha256(kws_path)
                rec = {
                    "sample_id": sid,
                    "dataset_version": "research_v3.0",
                    "class_label": "UNKNOWN",
                    "class_id": 4,
                    "text": t_info["text"],
                    "voice_id": vid,
                    "split": split,
                    "seed": 500_000 + i,
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
                }
                records.append(rec)
                text_inventory_records.append({
                    "text_id": f"txt_{split}_{text_counter:04d}",
                    "text": t_info["text"],
                    "split": split,
                    "intended_label": "UNKNOWN",
                    "confusion_family": t_info["confusion_family"],
                    "source": "VieNeu-TTS v3turbo",
                    "tts_voice_id": vid
                })
                text_counter += 1

        elif split in ["val", "test"]:
            # All voices speak all texts in the split
            idx = 0
            for vid in voice_list:
                for t_info in text_list:
                    idx += 1
                    sid = f"v3_unk_{split}_{idx:04d}"
                    raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")
                    kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")

                    if not os.path.exists(kws_path):
                        raw_audio = tts.infer(t_info["text"], voice=vid, temperature=0.75, top_p=0.92, repetition_penalty=1.20)
                        kws_audio, onset_s, dur_s = format_to_1s(raw_audio, 48000, 16000)
                        sf.write(raw_path, raw_audio, 48000, subtype="PCM_16")
                        sf.write(kws_path, kws_audio, 16000, subtype="PCM_16")
                    else:
                        kws_audio, _ = sf.read(kws_path, dtype="float32")
                        onset_s, dur_s = detect_speech_bounds(kws_audio, 16000)

                    kws_h = compute_sha256(kws_path)
                    rec = {
                        "sample_id": sid,
                        "dataset_version": "research_v3.0",
                        "class_label": "UNKNOWN",
                        "class_id": 4,
                        "text": t_info["text"],
                        "voice_id": vid,
                        "split": split,
                        "seed": 500_000 + idx,
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
                    }
                    records.append(rec)
                    text_inventory_records.append({
                        "text_id": f"txt_{split}_{text_counter:04d}",
                        "text": t_info["text"],
                        "split": split,
                        "intended_label": "UNKNOWN",
                        "confusion_family": t_info["confusion_family"],
                        "source": "VieNeu-TTS v3turbo",
                        "tts_voice_id": vid
                    })
                    text_counter += 1
            print(f"Synthesized {idx} utterances for {split.upper()}.")

    # -----------------------------------------------------------------
    # Part C: Synthesize Background SILENCE Samples
    # -----------------------------------------------------------------
    print("\n--- Generating Background SILENCE Samples ---")
    silence_counts = {"train": 340, "val": 80, "test": 80}
    silence_families = ["near_silence", "hvac_hum", "office_ambient", "airflow_pink"]

    for split, count in silence_counts.items():
        for i in range(count):
            fam = silence_families[i % len(silence_families)]
            sid = f"v3_sil_{split}_{i+1:04d}"
            kws_path = os.path.join(AUDIO_KWS_DIR, f"{sid}_kws16k.wav")
            raw_path = os.path.join(AUDIO_RAW_DIR, f"{sid}_raw48k.wav")

            if not os.path.exists(kws_path):
                audio_16k = generate_synthetic_background(16000, fam, seed=600_000 + i)
                sf.write(kws_path, audio_16k, 16000, subtype="PCM_16")
                audio_48k = signal.resample(audio_16k, 48000).astype(np.float32)
                sf.write(raw_path, audio_48k, 48000, subtype="PCM_16")

            rec = {
                "sample_id": sid,
                "dataset_version": "research_v3.0",
                "class_label": "SILENCE",
                "class_id": 5,
                "text": "[SILENCE]",
                "voice_id": fam,
                "split": split,
                "seed": 600_000 + i,
                "raw_wav_path": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
                "kws_wav_path": os.path.relpath(kws_path, BASE_DIR).replace("\\", "/"),
                "kws_onset_sec": 0.0,
                "kws_active_speech_duration_sec": 0.0,
                "kws_sha256": compute_sha256(kws_path),
                "sample_rate": 16000,
                "num_channels": 1,
                "num_samples": 16000,
                "is_target_command": False,
                "confusion_family": "ambient_silence",
                "target_keyword": "NONE",
                "linguistic_rationale": f"Synthetic ambient background: {fam}"
            }
            records.append(rec)

    # Write Manifests
    df_all = pd.DataFrame(records)
    master_manifest_p = os.path.join(DATA_DIR, "manifest.csv")
    df_all.to_csv(master_manifest_p, index=False, encoding="utf-8")

    for split in ["train", "val", "test"]:
        sub_df = df_all[df_all["split"] == split].copy()
        split_p = os.path.join(SPLITS_DIR, f"{split}_manifest.csv")
        sub_df.to_csv(split_p, index=False, encoding="utf-8")
        print(f"Saved {split_p} ({len(sub_df)} rows).")

    text_inv_df = pd.DataFrame(text_inventory_records)
    text_inv_p = os.path.join(DATA_DIR, "text_inventory.csv")
    text_inv_df.to_csv(text_inv_p, index=False, encoding="utf-8")
    print(f"Saved {text_inv_p} ({len(text_inv_df)} entries).")

    # -----------------------------------------------------------------
    # Part D: Build Continuous Streaming Challenge Benchmark
    # -----------------------------------------------------------------
    print("\n--- Assembling Continuous Challenge Streams ---")

    # 1. Validation Commands Stream (160 targets)
    build_stream(
        stream_name="val_stream_continuous_commands",
        samples_df=df_all[(df_all["split"] == "val") & (df_all["is_target_command"] == True)],
        bg_family="office_ambient",
        min_gap_sec=1.5,
        max_gap_sec=3.0,
        seed=701,
        target_exact_dur_sec=None
    )

    # 2. Validation Non-Target Stream (120 distractors)
    build_stream(
        stream_name="val_stream_continuous_nontarget",
        samples_df=df_all[(df_all["split"] == "val") & (df_all["class_label"] == "UNKNOWN")],
        bg_family="hvac_hum",
        min_gap_sec=2.0,
        max_gap_sec=4.0,
        seed=702,
        target_exact_dur_sec=None
    )

    # 3. Test Commands Stream (160 targets)
    build_stream(
        stream_name="test_stream_continuous_commands",
        samples_df=df_all[(df_all["split"] == "test") & (df_all["is_target_command"] == True)],
        bg_family="office_ambient",
        min_gap_sec=1.5,
        max_gap_sec=3.0,
        seed=703,
        target_exact_dur_sec=None
    )

    # 4. Locked Long-Horizon Test Non-Target Stream (EXACTLY 3.00 HOURS = 10,800.0s)
    # Uses 480 test non-target clips across 2 passes (960 events) with realistic background noise
    build_stream(
        stream_name="test_stream_continuous_nontarget_3h",
        samples_df=df_all[(df_all["split"] == "test") & (df_all["class_label"] == "UNKNOWN")],
        bg_family="airflow_pink",
        min_gap_sec=5.0,
        max_gap_sec=14.0,
        seed=704,
        target_exact_dur_sec=10800.0,  # Exactly 3.0 hours!
        num_passes=2
    )

    print("\nPhase 46 dataset and continuous benchmark generation complete!")


def build_stream(stream_name: str, samples_df: pd.DataFrame, bg_family: str,
                 min_gap_sec: float, max_gap_sec: float, seed: int,
                 target_exact_dur_sec: float = None, num_passes: int = 1):
    rng = np.random.RandomState(seed)
    sr = 16000

    items = samples_df.to_dict(orient="records")
    full_event_list = []
    for p in range(num_passes):
        p_items = list(items)
        if p > 0:
            rng.shuffle(p_items)
        full_event_list.extend(p_items)

    print(f"\nBuilding {stream_name} ({len(full_event_list)} speech events, {num_passes} passes)...")

    # Collect audio arrays
    speech_segments = []
    total_speech_samples = 0
    for it in full_event_list:
        kws_p = os.path.join(BASE_DIR, it["kws_wav_path"])
        aud, _ = sf.read(kws_p, dtype="float32")
        speech_segments.append((it, aud))
        total_speech_samples += len(aud)

    num_gaps = len(speech_segments) + 1
    if target_exact_dur_sec is not None:
        target_total_samples = int(target_exact_dur_sec * sr)
        needed_gap_samples = target_total_samples - total_speech_samples
        assert needed_gap_samples > 0, "Speech duration exceeds target duration!"
        # Allocate random gap lengths that sum exactly to needed_gap_samples
        raw_gaps = rng.uniform(min_gap_sec, max_gap_sec, num_gaps)
        scaled_gaps = (raw_gaps / np.sum(raw_gaps)) * needed_gap_samples
        gap_samples = [int(round(g)) for g in scaled_gaps]
        # Adjust rounding difference on final gap
        gap_samples[-1] += (needed_gap_samples - sum(gap_samples))
    else:
        gap_samples = [int(rng.uniform(min_gap_sec, max_gap_sec) * sr) for _ in range(num_gaps)]

    # Assemble stream
    stream_blocks = []
    ground_truth = []
    current_sample = 0

    # Initial gap
    init_bg = generate_synthetic_background(gap_samples[0], bg_family, seed=seed)
    stream_blocks.append(init_bg)
    current_sample += len(init_bg)

    total_active_speech_sec = 0.0

    for idx, (meta, aud) in enumerate(speech_segments):
        onset_s = round(current_sample / sr, 3)
        end_s = round((current_sample + len(aud)) / sr, 3)
        dur_s = round(len(aud) / sr, 3)
        total_active_speech_sec += dur_s

        # Mix with low-level background
        local_bg = generate_synthetic_background(len(aud), bg_family, seed=seed + idx + 1)
        mixed = aud + 0.15 * local_bg

        stream_blocks.append(mixed)
        current_sample += len(mixed)

        ground_truth.append({
            "event_index": idx + 1,
            "sample_id": meta["sample_id"],
            "event_type": "target_command" if meta["is_target_command"] else "non_target_distractor",
            "class_label": meta["class_label"],
            "text": meta["text"],
            "voice_id": meta["voice_id"],
            "confusion_family": meta["confusion_family"],
            "target_keyword": meta["target_keyword"],
            "onset_sec": onset_s,
            "end_time_sec": end_s,
            "active_duration_sec": dur_s,
            "acoustic_condition": bg_family
        })

        # Inter-utterance gap
        gap_aud = generate_synthetic_background(gap_samples[idx + 1], bg_family, seed=seed + idx + 50_000)
        stream_blocks.append(gap_aud)
        current_sample += len(gap_aud)

    full_audio = np.concatenate(stream_blocks)
    actual_dur_sec = round(len(full_audio) / sr, 3)
    actual_dur_hr = round(actual_dur_sec / 3600.0, 4)
    speech_active_hr = round(total_active_speech_sec / 3600.0, 4)
    active_ratio = round((total_active_speech_sec / actual_dur_sec) * 100.0, 2)

    wav_out = os.path.join(STREAMS_DIR, f"{stream_name}.wav")
    sf.write(wav_out, full_audio, sr, subtype="PCM_16")

    gt_out = os.path.join(STREAMS_DIR, f"{stream_name}_ground_truth.json")
    with open(gt_out, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    meta_out = os.path.join(STREAMS_DIR, f"{stream_name}_metadata.json")
    stream_meta = {
        "stream_name": stream_name,
        "wav_file": os.path.basename(wav_out),
        "ground_truth_file": os.path.basename(gt_out),
        "duration_seconds": actual_dur_sec,
        "duration_hours": actual_dur_hr,
        "active_speech_duration_seconds": round(total_active_speech_sec, 3),
        "active_speech_duration_hours": speech_active_hr,
        "active_speech_percentage": active_ratio,
        "num_events": len(ground_truth),
        "background_family": bg_family,
        "num_passes": num_passes,
        "sha256": compute_sha256(wav_out)
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(stream_meta, f, indent=2)

    print(f"Saved {wav_out}: {actual_dur_sec}s ({actual_dur_hr} hr). Active speech: {speech_active_hr} hr ({active_ratio}%).")


if __name__ == "__main__":
    main()
