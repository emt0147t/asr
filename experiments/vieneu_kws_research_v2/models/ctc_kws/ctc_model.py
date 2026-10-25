"""
Character-Level CTC Keyword Spotting Architecture and Tokenizer for Vietnamese.
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v2/models/ctc_kws/

Implements:
1. VietnameseCharTokenizer:
   - Unicode NFC normalization.
   - Comprehensive alphabet covering all tone marks and diacritics.
   - Blank token = 0, Space = 1, Characters = 2..N.
   - Greedy CTC sequence collapse and word boundary matching.
2. VietnameseCTCKWSNet (CRNN):
   - 2D CNN frequency reducer (40 mels -> 1 channel, preserving time T=101).
   - 2-layer Bidirectional GRU (128-dim temporal sequence encoder).
   - Linear projection head with log-softmax.
   - PyTorch nn.CTCLoss compatible.
"""

import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# Authoritative Vietnamese character alphabet (lowercase, Unicode NFC)
VIETNAMESE_ALPHABET = [
    # Blank and space
    "<blank>", " ",
    # Basic Latin letters
    "a", "b", "c", "d", "e", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y",
    # Vietnamese modified vowels & consonants
    "ă", "â", "đ", "ê", "ô", "ơ", "ư",
    # Vowels with sắc (acute)
    "á", "ắ", "ấ", "é", "ế", "í", "ó", "ố", "ớ", "ú", "ứ", "ý",
    # Vowels with huyền (grave)
    "à", "ằ", "ầ", "è", "ề", "ì", "ò", "ồ", "ờ", "ù", "ừ", "ỳ",
    # Vowels with hỏi (hook above)
    "ả", "ẳ", "ẩ", "ẻ", "ể", "ỉ", "ỏ", "ổ", "ở", "ủ", "ử", "ỷ",
    # Vowels with ngã (tilde)
    "ã", "ẵ", "ẫ", "ẽ", "ễ", "ĩ", "õ", "ỗ", "ỡ", "ũ", "ữ", "ỹ",
    # Vowels with nặng (dot below)
    "ạ", "ặ", "ậ", "ẹ", "ệ", "ị", "ọ", "ộ", "ợ", "ụ", "ự", "ỵ"
]


class VietnameseCharTokenizer:
    def __init__(self):
        self.vocab = VIETNAMESE_ALPHABET
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.blank_id = 0
        self.space_id = 1
        self.vocab_size = len(self.vocab)

    def normalize_text(self, text: str) -> str:
        """Normalizes text with Unicode NFC and strips extraneous whitespace."""
        if not text or text == "[SILENCE]" or text.startswith("["):
            return ""
        norm = unicodedata.normalize("NFC", text.strip().lower())
        # Remove punctuation
        norm = re.sub(r"[^\w\s]", "", norm)
        return norm

    def encode(self, text: str) -> list:
        """Converts normalized text into list of integer token IDs."""
        norm = self.normalize_text(text)
        tokens = []
        for ch in norm:
            if ch in self.char_to_id:
                tokens.append(self.char_to_id[ch])
            elif ch.isspace():
                tokens.append(self.space_id)
        return tokens

    def decode(self, token_ids: list) -> str:
        """Greedy CTC collapse: collapses adjacent duplicates and strips blank (0)."""
        collapsed = []
        prev = None
        for tid in token_ids:
            if tid != prev:
                if tid != self.blank_id:
                    collapsed.append(tid)
                prev = tid

        chars = [self.id_to_char.get(tid, "") for tid in collapsed]
        return "".join(chars).strip()

    def match_keyword(self, decoded_text: str):
        """
        Matches decoded text against target keywords:
        Returns: (matched_class_label, is_exact_match)
        """
        norm = unicodedata.normalize("NFC", decoded_text.strip().lower())
        words = norm.split()

        target_map = {
            "trái": "TRAI",
            "phải": "PHAI",
            "lên": "LEN",
            "xuống": "XUONG"
        }

        # Exact whole-string match
        if norm in target_map:
            return target_map[norm], True

        # Standalone word match within short phrase
        for word in words:
            if word in target_map:
                # If phrase contains other words (e.g. "bên phải" or "tiến lên"), mark as context
                return target_map[word], False

        return None, False


class VietnameseCTCKWSNet(nn.Module):
    """
    Compact Streaming-Capable CRNN for Character-Level Vietnamese KWS with CTC.
    Input: Log-Mel Spectrogram (Batch, 1, 40, T) where T=101 (1.0 second audio)
    Output: Log-Softmax Posteriors (T, Batch, Vocab_Size) for CTCLoss
    """
    def __init__(self, vocab_size: int = len(VIETNAMESE_ALPHABET), hidden_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size

        # Frequency-reduction 2D CNN frontend (preserves time T=101)
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 40 -> 20
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 20 -> 10
            nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))    # 10 -> 1
        )

        # 2-layer Bidirectional GRU sequence backbone
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Output projection head
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, 1, 40, T) or (Batch, 40, T)
        Returns:
            log_probs: Tensor of shape (T, Batch, Vocab_Size) for CTCLoss
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 1. 2D CNN frontend: (Batch, 1, 40, T) -> (Batch, 64, 1, T)
        feat = self.frontend(x)
        # Squeeze frequency dimension: (Batch, 64, T)
        feat = feat.squeeze(2)
        # Transpose to (Batch, T, 64) for GRU
        feat = feat.transpose(1, 2)

        # 2. Recurrent temporal encoder: (Batch, T, 128)
        gru_out, _ = self.gru(feat)

        # 3. Dense projection: (Batch, T, Vocab_Size)
        logits = self.fc(gru_out)
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to (T, Batch, Vocab_Size) for CTCLoss
        log_probs_t = log_probs.transpose(0, 1)
        return log_probs_t


def get_ctc_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
