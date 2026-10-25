import os
import sys
import json
import csv
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

# Base feature extraction matching project standard
class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=20.0,
            f_max=8000.0
        )

    def forward(self, x):
        # x: (B, 16000)
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1) # (B, 1, 40, 101)
        return log_mel


class VietnameseBoundedKWSNet(nn.Module):
    """
    Compact Discriminative Bounded-Utterance Keyword Spotting Network.
    5 Output classes:
      0 = TRAI
      1 = PHAI
      2 = LEN
      3 = XUONG
      4 = REJECT (Silence, neutral speech, hard-negative distractors)
    """
    def __init__(self, num_classes=5, dropout_p=0.25):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv block 2 (stride 2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv block 3 (residual block)
        self.conv3a = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(64)

        # Conv block 4 (downsample frequency, preserve time)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # Frequency pooling -> (B, 128, 1, T)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Temporal Attentive Pooling
        self.attn_proj = nn.Linear(128, 64)
        self.attn_score = nn.Linear(64, 1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 40, 101)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        
        # Residual connection in block 3
        res = h
        h = F.relu(self.bn3a(self.conv3a(h)))
        h = self.bn3b(self.conv3b(h))
        h = F.relu(h + res)

        h = F.relu(self.bn4(self.conv4(h))) # (B, 128, 10, 51)
        h = self.freq_pool(h).squeeze(2)   # (B, 128, 51)
        h = h.transpose(1, 2)              # (B, 51, 128)

        # Temporal Attention
        attn_weights = F.softmax(self.attn_score(torch.tanh(self.attn_proj(h))), dim=1) # (B, 51, 1)
        pooled = torch.sum(attn_weights * h, dim=1)                                     # (B, 128)

        logits = self.classifier(pooled)   # (B, 5)
        return logits


class BoundedUtteranceDataset(Dataset):
    def __init__(self, manifest_path, is_training=False, target_length=16000):
        self.manifest_path = manifest_path
        self.is_training = is_training
        self.target_length = target_length
        self.rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav_path = row["kws_wav_path"]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(r"D:\trainidentificationaudio", wav_path)

        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected sample rate 16000, got {sr}")

        # Fix length to 16000 samples (1.0s)
        if len(audio) < self.target_length:
            pad = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode="constant")
        elif len(audio) > self.target_length:
            audio = audio[:self.target_length]

        class_id = int(row["bounded_class_id"])

        # Augmentation during training
        if self.is_training:
            # 1. Random time shift (-80ms to +80ms, i.e. -1280 to +1280 samples)
            shift = random.randint(-1280, 1280)
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode="constant")[:self.target_length]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode="constant")[-shift:self.target_length - shift]

            # 2. Random gain (-3dB to +3dB)
            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = audio * gain

            # 3. Additive noise (SNR 12 to 30 dB)
            if random.random() < 0.6:
                noise = np.random.randn(*audio.shape).astype(np.float32)
                # Compute signal RMS
                sig_rms = np.sqrt(np.mean(audio**2) + 1e-9)
                snr_db = random.uniform(12.0, 30.0)
                noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
                audio = audio + noise * noise_rms

        # Peak normalization safety
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        return torch.from_numpy(audio).float(), class_id


def train_seed(seed=42, epochs=25, batch_size=32, lr=1e-3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_dir = r"D:\trainidentificationaudio"
    data_dir = os.path.join(base_dir, "experiments", "vieneu_kws_research_v4", "data", "splits")
    train_manifest = os.path.join(data_dir, "train_manifest.csv")
    val_manifest = os.path.join(data_dir, "val_manifest.csv")

    train_ds = BoundedUtteranceDataset(train_manifest, is_training=True)
    val_ds = BoundedUtteranceDataset(val_manifest, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    model = VietnameseBoundedKWSNet(num_classes=5, dropout_p=0.25).to(device)

    # Loss function with mild label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights = None
    history = []

    print(f"\n=== Training Candidate 3 (VietnameseBoundedKWSNet) Seed {seed} on {device} ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for audios, labels in train_loader:
            audios, labels = audios.to(device), labels.to(device)
            specs = frontend(audios) # (B, 1, 40, 101)

            # SpecAugment on specs during training
            if random.random() < 0.5:
                # Frequency masking
                f_mask = random.randint(1, 8)
                f_start = random.randint(0, 40 - f_mask)
                specs[:, :, f_start:f_start+f_mask, :] = 0.0
            if random.random() < 0.5:
                # Time masking
                t_mask = random.randint(1, 15)
                t_start = random.randint(0, 101 - t_mask)
                specs[:, :, :, t_start:t_start+t_mask] = 0.0

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for audios, labels in val_loader:
                audios, labels = audios.to(device), labels.to(device)
                specs = frontend(audios)
                logits = model(specs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

    out_ckpt_dir = os.path.join(base_dir, "experiments", "vieneu_kws_research_v4", "models", "bounded_utterance", "checkpoints")
    out_log_dir = os.path.join(base_dir, "experiments", "vieneu_kws_research_v4", "models", "bounded_utterance", "logs")
    os.makedirs(out_ckpt_dir, exist_ok=True)
    os.makedirs(out_log_dir, exist_ok=True)

    ckpt_path = os.path.join(out_ckpt_dir, f"bounded_kws_net_seed{seed}.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": best_weights,
        "best_val_acc": best_val_acc,
        "seed": seed,
        "num_classes": 5
    }, ckpt_path)

    log_path = os.path.join(out_log_dir, f"training_log_seed{seed}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "best_val_acc": best_val_acc,
            "history": history
        }, f, indent=2)

    print(f"Seed {seed} finished. Best Val Acc: {best_val_acc*100:.2f}%. Saved to {ckpt_path}")
    return best_val_acc


if __name__ == "__main__":
    seeds = [42, 43, 44]
    for s in seeds:
        train_seed(seed=s, epochs=25, batch_size=32, lr=1e-3)
