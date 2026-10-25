"""
Phase 42: Training Candidate B (Character-Level CTC Keyword Spotter).
Workspace: D:\trainidentificationaudio
Target: experiments/vieneu_kws_research_v2/models/ctc_kws/

Implements:
1. CTCKWSDataset:
   - Log-Mel spectrograms (16 kHz, 40 mels, 1.0s -> T=101).
   - Dynamic character token targets using VietnameseCharTokenizer.
   - Silence / background waveforms aligned with zero-length targets (blank-only).
   - Time shift, gain, and noise data augmentation during training.
2. Training loop with PyTorch CTCLoss:
   - Evaluates greedy CTC decoding accuracy and exact keyword match on validation split.
   - Operates across independent seeds: 42, 43, 44.
   - Saves best validation checkpoints.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from experiments.vieneu_kws_research_v2.models.ctc_kws.ctc_model import (
    VietnameseCharTokenizer,
    VietnameseCTCKWSNet
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUT_DIR = os.path.join(EXP_DIR, "models", "ctc_kws")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CTCKWSDataset(Dataset):
    def __init__(self, manifest_csv: str, tokenizer: VietnameseCharTokenizer, is_train: bool = True, augment: bool = True):
        self.df = pd.read_csv(manifest_csv)
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.augment = augment and is_train
        self.extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40, win_length=400, hop_length=160)

        self.audio_list = []
        self.target_tokens = []
        self.target_texts = []
        self.class_labels = []

        for _, row in self.df.iterrows():
            p = os.path.join(BASE_DIR, row["kws_wav_path"]) if not os.path.isabs(row["kws_wav_path"]) else row["kws_wav_path"]
            wav, _ = sf.read(p, dtype="float32")
            if len(wav) < 16000:
                wav = np.pad(wav, (0, 16000 - len(wav)))
            elif len(wav) > 16000:
                wav = wav[:16000]

            txt = str(row["text"])
            tokens = self.tokenizer.encode(txt)

            self.audio_list.append(wav)
            self.target_tokens.append(tokens)
            self.target_texts.append(txt)
            self.class_labels.append(row["class_label"])

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio = self.audio_list[idx].copy()
        tokens = self.target_tokens[idx]
        clbl = self.class_labels[idx]

        if self.augment:
            shift = random.randint(-400, 400)
            if shift > 0:
                audio = np.pad(audio[:-shift], (shift, 0))
            elif shift < 0:
                audio = np.pad(audio[-shift:], (0, -shift))
            gain = random.uniform(0.85, 1.15)
            audio = np.clip(audio * gain, -0.98, 0.98)

        t_audio = torch.from_numpy(audio).unsqueeze(0).float()
        with torch.no_grad():
            feat = self.extractor(t_audio).squeeze(0)  # (1, 40, 101)

        return {
            "feat": feat,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "target_len": len(tokens),
            "text": self.target_texts[idx],
            "class_label": clbl
        }


def ctc_collate_fn(batch):
    feats = torch.stack([b["feat"] for b in batch])
    target_lens = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    input_lens = torch.full((len(batch),), 101, dtype=torch.long)

    # Concatenate targets for CTCLoss
    all_tokens = []
    for b in batch:
        all_tokens.extend(b["tokens"].tolist())
    targets = torch.tensor(all_tokens, dtype=torch.long)

    texts = [b["text"] for b in batch]
    labels = [b["class_label"] for b in batch]

    return feats, targets, input_lens, target_lens, texts, labels


def train_ctc_model(seed: int, device):
    set_seed(seed)
    print(f"\n--- Training Candidate B: Character-Level CTC KWS Seed {seed} ---")

    tokenizer = VietnameseCharTokenizer()
    train_ds = CTCKWSDataset(os.path.join(SPLITS_DIR, "train_manifest.csv"), tokenizer, is_train=True, augment=True)
    val_ds = CTCKWSDataset(os.path.join(SPLITS_DIR, "val_manifest.csv"), tokenizer, is_train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=ctc_collate_fn)

    model = VietnameseCTCKWSNet(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    best_val_score = -1.0
    best_weights_path = os.path.join(CHECKPOINTS_DIR, f"best_ctc_seed{seed}.pt")

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for feats, targets, in_lens, tgt_lens, _, _ in train_loader:
            feats = feats.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            optimizer.zero_grad()
            log_probs = model(feats)  # (T=101, B, Vocab)
            loss = ctc_loss(log_probs, targets, in_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(in_lens)
        scheduler.step()

        # Validation evaluation: keyword decoding accuracy on command targets
        model.eval()
        correct_keywords = 0
        total_command_targets = 0
        with torch.no_grad():
            for feats, _, _, _, texts, labels in val_loader:
                feats = feats.to(device)
                log_probs = model(feats)  # (101, B, Vocab)
                # Greedy decode
                preds = torch.argmax(log_probs, dim=-1).cpu().numpy()  # (101, B)
                for b in range(feats.size(0)):
                    seq = preds[:, b].tolist()
                    decoded_str = tokenizer.decode(seq)
                    gt_label = labels[b]
                    if gt_label in ["TRAI", "PHAI", "LEN", "XUONG"]:
                        total_command_targets += 1
                        matched_kw, exact = tokenizer.match_keyword(decoded_str)
                        if matched_kw == gt_label:
                            correct_keywords += 1

        val_kw_acc = correct_keywords / total_command_targets if total_command_targets > 0 else 0.0

        if val_kw_acc > best_val_score:
            best_val_score = val_kw_acc
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state": model.state_dict(),
                "val_keyword_accuracy": val_kw_acc,
                "architecture": "VietnameseCTCKWSNet",
                "vocab_size": tokenizer.vocab_size
            }, best_weights_path)

        if epoch % 5 == 0 or epoch == 30:
            print(f"CTC Epoch {epoch:02d}/30 - Train Loss: {total_loss/len(train_ds):.4f} - Val Keyword Acc: {val_kw_acc*100:.2f}% (Best: {best_val_score*100:.2f}%)")

    print(f"Candidate B Seed {seed} Best Val Keyword Acc: {best_val_score*100:.2f}% saved to {best_weights_path}")
    return best_weights_path


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for CTC:", device)

    seeds = [42, 43, 44]
    results = {}
    for seed in seeds:
        ckpt_p = train_ctc_model(seed, device)
        results[f"seed{seed}"] = {
            "ctc_checkpoint": os.path.relpath(ckpt_p, BASE_DIR).replace("\\", "/")
        }

    summary_file = os.path.join(LOGS_DIR, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll CTC training completed. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
