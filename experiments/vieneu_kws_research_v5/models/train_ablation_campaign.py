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

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5")
DATA_DIR = os.path.join(EXP_DIR, "data", "splits")
MODELS_DIR = os.path.join(EXP_DIR, "models")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
BOUNDED_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]

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
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)
        return log_mel


class VietnameseBoundedKWSNet(nn.Module):
    """
    Compact Discriminative Bounded-Utterance Keyword Spotting Network.
    5 Output classes: [TRAI, PHAI, LEN, XUONG, REJECT]
    Parameter count: ~183,500 params
    """
    def __init__(self, num_classes=5, dropout_p=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.attn_proj = nn.Linear(128, 64)
        self.attn_score = nn.Linear(64, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))

        res = h
        h = F.relu(self.bn3a(self.conv3a(h)))
        h = self.bn3b(self.conv3b(h))
        h = F.relu(h + res)

        h = F.relu(self.bn4(self.conv4(h)))
        h = self.freq_pool(h).squeeze(2)
        h = h.transpose(1, 2)

        attn_weights = F.softmax(self.attn_score(torch.tanh(self.attn_proj(h))), dim=1)
        pooled = torch.sum(attn_weights * h, dim=1)
        logits = self.classifier(pooled)
        return logits


class BoundedDatasetV5(Dataset):
    def __init__(self, manifest_path, is_training=False, target_length=16000, apply_hard_negative_weight=False):
        self.manifest_path = manifest_path
        self.is_training = is_training
        self.target_length = target_length
        self.apply_hard_negative_weight = apply_hard_negative_weight
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
            wav_path = os.path.join(BASE_DIR, wav_path)

        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected 16kHz, got {sr}")

        if len(audio) < self.target_length:
            pad = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode="constant")
        elif len(audio) > self.target_length:
            audio = audio[:self.target_length]

        class_id = int(row["bounded_class_id"])
        
        # Hard-negative weight assignment
        sample_weight = 1.0
        if self.apply_hard_negative_weight:
            fam = row.get("confusion_family", "")
            if fam in ["minimal_pair_tone", "minimal_pair_consonant", "minimal_pair_vowel_coda", "phonetic_phrase_context"]:
                sample_weight = 1.5  # Elevate hard-negative loss penalty
            elif class_id < 4:
                sample_weight = 1.25  # Maintain command recall preservation

        if self.is_training:
            # Augmentations
            # 1. Random time shift (-80ms to +80ms)
            shift = random.randint(-1280, 1280)
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode="constant")[:self.target_length]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode="constant")[-shift:self.target_length - shift]

            # 2. Gain variation (-3dB to +3dB)
            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = audio * gain

            # 3. Additive noise (SNR 12 to 28 dB)
            if random.random() < 0.6:
                noise = np.random.randn(*audio.shape).astype(np.float32)
                sig_rms = np.sqrt(np.mean(audio**2) + 1e-9)
                snr_db = random.uniform(12.0, 28.0)
                noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
                audio = audio + noise * noise_rms

        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        return torch.from_numpy(audio).float(), class_id, float(sample_weight)


def train_model(variant_name, seed, epochs=25, batch_size=32, lr=1e-3, use_ablation_weights=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_manifest = os.path.join(DATA_DIR, "train_manifest.csv")
    val_manifest = os.path.join(DATA_DIR, "val_manifest.csv")

    train_ds = BoundedDatasetV5(train_manifest, is_training=True, apply_hard_negative_weight=use_ablation_weights)
    val_ds = BoundedDatasetV5(val_manifest, is_training=False, apply_hard_negative_weight=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    model = VietnameseBoundedKWSNet(num_classes=5, dropout_p=0.25).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_score = -1.0
    best_weights = None
    best_metrics = None
    history = []

    print(f"\n=== Training [{variant_name}] Seed {seed} on {device} (Ablation Weights={use_ablation_weights}) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for audios, labels, weights in train_loader:
            audios, labels, weights = audios.to(device), labels.to(device), weights.to(device)
            specs = frontend(audios)

            # SpecAugment
            if random.random() < 0.5:
                f_mask = random.randint(1, 8)
                f_start = random.randint(0, 40 - f_mask)
                specs[:, :, f_start:f_start+f_mask, :] = 0.0
            if random.random() < 0.5:
                t_mask = random.randint(1, 15)
                t_start = random.randint(0, 101 - t_mask)
                specs[:, :, :, t_start:t_start+t_mask] = 0.0

            optimizer.zero_grad()
            logits = model(specs)

            if use_ablation_weights:
                # Weighted cross-entropy with sample weights
                loss_per_sample = F.cross_entropy(logits, labels, label_smoothing=0.05, reduction="none")
                loss = torch.mean(loss_per_sample * weights)
            else:
                loss = F.cross_entropy(logits, labels, label_smoothing=0.05)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        scheduler.step()
        train_loss = total_loss / total_samples

        # Validation evaluation
        model.eval()
        val_cmd_hits = 0
        val_cmd_total = 0
        val_nt_rejects = 0
        val_nt_total = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for audios, labels, _ in val_loader:
                audios, labels = audios.to(device), labels.to(device)
                specs = frontend(audios)
                logits = model(specs)
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                    if l < 4:
                        val_cmd_total += 1
                        if p == l:
                            val_cmd_hits += 1
                    else:
                        val_nt_total += 1
                        if p == 4:
                            val_nt_rejects += 1

        val_recall = val_cmd_hits / val_cmd_total if val_cmd_total > 0 else 0.0
        val_far = (val_nt_total - val_nt_rejects) / val_nt_total if val_nt_total > 0 else 0.0
        val_acc = val_correct / val_total

        # Selection score: Recall - 2.5 * FAR
        selection_score = val_recall - 2.5 * val_far

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_recall": round(val_recall, 4),
            "val_far": round(val_far, 4),
            "selection_score": round(selection_score, 4)
        })

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "val_recall": round(val_recall, 4),
                "val_far": round(val_far, 4),
                "val_acc": round(val_acc, 4),
                "selection_score": round(selection_score, 4)
            }

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Recall: {val_recall*100:.2f}% | Val FAR: {val_far*100:.2f}%")

    # Save checkpoint and logs
    out_dir = os.path.join(MODELS_DIR, variant_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"model_seed{seed}.pt")
    torch.save({
        "variant": variant_name,
        "seed": seed,
        "num_classes": 5,
        "model_architecture": "VietnameseBoundedKWSNet",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_state_dict": best_weights,
        "best_metrics": best_metrics,
        "use_ablation_weights": use_ablation_weights
    }, ckpt_path)

    log_path = os.path.join(log_dir, f"training_log_seed{seed}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "variant": variant_name,
            "seed": seed,
            "best_metrics": best_metrics,
            "history": history
        }, f, indent=2)

    print(f"Saved {variant_name} Seed {seed} checkpoint to {ckpt_path} (Best Val Recall: {best_metrics['val_recall']*100:.2f}%, FAR: {best_metrics['val_far']*100:.2f}%)")
    return best_metrics


def run_ablation():
    seeds = [42, 43, 44]
    
    # 1. Candidate C: Standard Bounded Classifier
    print("\n" + "=" * 80)
    print("TRAINING CANDIDATE C: Standard Bounded Utterance Classifier")
    print("=" * 80)
    for s in seeds:
        train_model("candidate_c_standard", seed=s, epochs=25, batch_size=32, lr=1e-3, use_ablation_weights=False)

    # 2. Candidate D: Controlled Hard-Negative & Command-Preserving Ablation
    print("\n" + "=" * 80)
    print("TRAINING CANDIDATE D: Hard-Negative Weighted Bounded Classifier")
    print("=" * 80)
    for s in seeds:
        train_model("candidate_d_ablation", seed=s, epochs=25, batch_size=32, lr=1e-3, use_ablation_weights=True)

    print("\nAll model training runs completed successfully!")

if __name__ == "__main__":
    run_ablation()
