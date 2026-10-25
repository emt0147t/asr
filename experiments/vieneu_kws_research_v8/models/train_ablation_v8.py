import os
import sys
import json
import csv
import random
import time
import hashlib
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
sys.path.insert(0, BASE_DIR)

V6_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6")
V8_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8")
DATA_DIR = os.path.join(V6_EXP_DIR, "data", "splits")

MODELS_DIR = os.path.join(V8_EXP_DIR, "models")
CKPT_DIR = os.path.join(MODELS_DIR, "checkpoints")
REPORTS_DIR = os.path.join(V8_EXP_DIR, "reports")

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
BOUNDED_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]


from experiments.vieneu_kws_research_v7.models.train_ablation import (
    LogMelFrontend,
    CorrectedVietnameseKWSNet
)


class BoundedDatasetV8(Dataset):
    def __init__(self, manifest_path, is_training=False, target_length=16000, w_hn=1.2, w_cmd=2.0, w_sil=0.5):
        self.manifest_path = manifest_path
        self.is_training = is_training
        self.target_length = target_length
        self.w_hn = w_hn
        self.w_cmd = w_cmd
        self.w_sil = w_sil
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
        fam = row.get("confusion_family", "")

        # Sample weight assignment
        if class_id < 4:
            sample_weight = self.w_cmd
        elif fam in ["minimal_pair_tone", "minimal_pair_consonant", "minimal_pair_vowel_coda", "phonetic_phrase_context"]:
            sample_weight = self.w_hn
        else:
            sample_weight = self.w_sil

        if self.is_training:
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
            if random.random() < 0.5:
                noise = np.random.randn(*audio.shape).astype(np.float32)
                sig_rms = np.sqrt(np.mean(audio**2) + 1e-9)
                snr_db = random.uniform(12.0, 28.0)
                noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
                audio = audio + noise * noise_rms

        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        return torch.from_numpy(audio).float(), class_id, float(sample_weight)


def compute_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def train_ablation_model(variant_name, w_hn, seed, epochs=25, batch_size=32, lr=1e-3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_manifest = os.path.join(DATA_DIR, "train_manifest.csv")
    val_manifest = os.path.join(DATA_DIR, "val_manifest.csv")

    train_ds = BoundedDatasetV8(train_manifest, is_training=True, w_hn=w_hn, w_cmd=2.0, w_sil=0.5)
    val_ds = BoundedDatasetV8(val_manifest, is_training=False, w_hn=w_hn, w_cmd=2.0, w_sil=0.5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_score = -1.0
    best_weights = None
    best_metrics = None
    best_epoch = 0

    print(f"\n=== Training [{variant_name}] (w_hn={w_hn}) Seed {seed} on {device} ===")
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

            loss_per_sample = F.cross_entropy(logits, labels, label_smoothing=0.05, reduction="none")
            loss = torch.mean(loss_per_sample * weights)

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

        per_cmd_hits = {c: 0 for c in range(4)}
        per_cmd_total = {c: 0 for c in range(4)}

        with torch.no_grad():
            for audios, labels, _ in val_loader:
                audios, labels = audios.to(device), labels.to(device)
                specs = frontend(audios)
                logits = model(specs)
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                for p, t in zip(preds, labels):
                    p_item, t_item = p.item(), t.item()
                    if t_item < 4:
                        val_cmd_total += 1
                        per_cmd_total[t_item] += 1
                        if p_item == t_item:
                            val_cmd_hits += 1
                            per_cmd_hits[t_item] += 1
                    else:
                        val_nt_total += 1
                        if p_item == 4:
                            val_nt_rejects += 1

        val_macro_rec = sum(per_cmd_hits[c] / per_cmd_total[c] for c in range(4)) / 4.0
        val_far = (val_nt_total - val_nt_rejects) / val_nt_total
        val_acc = val_correct / val_total

        # Selection metric: prioritize recall >= 95% and FAR <= 5%
        score = val_macro_rec * 0.6 + (1.0 - val_far) * 0.4
        if val_macro_rec >= 0.95 and val_far <= 0.05:
            score += 1.0

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_cmd_recall": round(val_macro_rec, 4),
                "val_nt_far": round(val_far, 4),
                "per_cmd_recall": {TARGET_CLASSES[c]: round(per_cmd_hits[c]/per_cmd_total[c], 4) for c in range(4)},
                "score": round(score, 4)
            }

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Cmd Rec: {val_macro_rec*100:.2f}% | NT FAR: {val_far*100:.2f}%")

    ckpt_path = os.path.join(CKPT_DIR, f"{variant_name}_seed{seed}.pt")
    torch.save({
        "variant": variant_name,
        "w_hn": w_hn,
        "seed": seed,
        "epoch": best_epoch,
        "metrics": best_metrics,
        "model_state_dict": best_weights
    }, ckpt_path)

    sha = compute_file_sha256(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path} (Best Epoch {best_epoch}, Val Recall: {best_metrics['val_cmd_recall']*100:.2f}%, FAR: {best_metrics['val_nt_far']*100:.2f}%, SHA: {sha[:12]})")
    best_metrics["checkpoint_path"] = ckpt_path
    best_metrics["checkpoint_sha256"] = sha
    return best_metrics


def run_phase72_training_ablation():
    print("=" * 80)
    print("Phase 72: Controlled Hard-Negative Weight Ablation Campaign")
    print("=" * 80)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Predeclared ablation grid
    # Rationale:
    # Baseline Candidate B uses w_hn = 1.2 (achieved 99.17% recall, 5.89% FAR).
    # w_hn = 1.0 tests a milder penalty on hard negatives.
    # w_hn = 1.4 provides exact class-mass parity between hard negatives (244 * 1.4 = 341.6) and per-command mass (170 * 2.0 = 340.0).
    # w_hn = 1.6 tests stronger penalty against confusable phonemes without overwhelming command gradients (total REJECT mass 608.4 remains < total command mass 1360.0).
    ablation_grid = [
        {"variant": "variant_w10", "w_hn": 1.0, "desc": "Mild Hard-Negative (w_hn=1.0)"},
        {"variant": "variant_w12", "w_hn": 1.2, "desc": "Baseline Candidate B Re-implementation (w_hn=1.2)"},
        {"variant": "variant_w14", "w_hn": 1.4, "desc": "Balanced Hard-Negative Parity (w_hn=1.4)"},
        {"variant": "variant_w16", "w_hn": 1.6, "desc": "Stronger Hard-Negative Suppression (w_hn=1.6)"}
    ]

    seeds = [42, 43, 44]
    training_summary = {}

    for item in ablation_grid:
        var_name = item["variant"]
        w_val = item["w_hn"]
        training_summary[var_name] = {
            "w_hn": w_val,
            "desc": item["desc"],
            "seeds": {}
        }
        for s in seeds:
            res = train_ablation_model(var_name, w_val, s, epochs=25, batch_size=32, lr=1e-3)
            training_summary[var_name]["seeds"][f"seed{s}"] = res

    # Save training summary JSON
    sum_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\nWrote Training Summary: {sum_path}")

    # Write report
    write_phase72_report(training_summary, ablation_grid)


def write_phase72_report(training_summary, ablation_grid):
    seeds = [42, 43, 44]
    lines = [
        "# Phase 72 — Controlled Hard-Negative Weight Ablation Report",
        "",
        "**Workspace:** `D:\\trainidentificationaudio`  ",
        "**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  ",
        "**Model Architecture:** `CorrectedVietnameseKWSNet` (138,469 parameters, global average pooling)  ",
        "**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Predeclared Rationale",
        "",
        "Phase 72 executes a controlled ablation on the hard-negative loss weight $w_{\\text{hn}}$ across seeds 42, 43, and 44, holding all other components strictly fixed (command weight $w_{\\text{cmd}}=2.0$, silence weight $w_{\\text{silence}}=0.5$, architecture, AdamW optimizer, Cosine Annealing scheduler, 25 epochs).",
        "",
        "### Predeclared Ablation Grid & Mathematical Rationale:",
        "- **`variant_w10` ($w_{\\text{hn}} = 1.0$):** Mild hard-negative weight. Tests whether reducing hard-negative pressure further boosts recall on difficult keywords like `PHAI`.",
        "- **`variant_w12` ($w_{\\text{hn}} = 1.2$):** Candidate B reference re-implementation. Reconstructs baseline behavior under fresh v8 pipeline.",
        "- **`variant_w14` ($w_{\\text{hn}} = 1.4$):** Balanced class-mass parity. On 244 hard-negative training samples, $244 \\times 1.4 = 341.6$, perfectly balancing the mass of an individual command class ($170 \\times 2.0 = 340.0$).",
        "- **`variant_w16` ($w_{\\text{hn}} = 1.6$):** Stronger hard-negative suppression. Evaluates whether stronger gradient pressure on confusable minimal pairs reduces FAR towards the $\\le 5.0\\%$ target while preserving total REJECT mass ($608.4$) well below total command mass ($1,360.0$).",
        "",
        "---",
        "",
        "## 2. Dataset Composition Audit (Training Split)",
        "",
        "| Partition / Class | Samples | Loss Weight $w_i$ | Total Loss Mass | Mass Ratio vs Command Class (340.0) |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| **TRAI** | 170 | 2.0 | 340.0 | $1.00\\times$ |",
        "| **PHAI** | 170 | 2.0 | 340.0 | $1.00\\times$ |",
        "| **LEN** | 170 | 2.0 | 340.0 | $1.00\\times$ |",
        "| **XUONG** | 170 | 2.0 | 340.0 | $1.00\\times$ |",
        "| **Hard Negatives (Minimal Pairs)** | 244 | $w_{\\text{hn}} \\in [1.0, 1.6]$ | 244.0 to 390.4 | $0.72\\times$ to $1.15\\times$ |",
        "| **Plain Silence & Ambient** | 436 | 0.5 | 218.0 | $0.64\\times$ |",
        "| **Total Commands** | 680 | 2.0 | 1,360.0 | $4.00\\times$ |",
        "| **Total REJECT** | 680 | Mixed | 462.0 to 608.4 | $1.36\\times$ to $1.79\\times$ |",
        "",
        "---",
        "",
        "## 3. Training Convergence & Best Validation Epoch Performance",
        "",
        "| Variant | $w_{\\text{hn}}$ | Seed | Best Epoch | Train Loss | Val Acc | Val Cmd Recall | Val Non-Target FAR | TRAI | PHAI | LEN | XUONG | Checkpoint SHA-256 (prefix) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for item in ablation_grid:
        var = item["variant"]
        w = item["w_hn"]
        for s in seeds:
            res = training_summary[var]["seeds"][f"seed{s}"]
            pk = res["per_cmd_recall"]
            sha_pref = res["checkpoint_sha256"][:12]
            lines.append(
                f"| **{var}** | {w:.1f} | {s} | {res['epoch']:02d} | {res['train_loss']:.4f} | {res['val_acc']*100:.2f}% | **{res['val_cmd_recall']*100:.2f}%** | **{res['val_nt_far']*100:.2f}%** | {pk['TRAI']*100:.1f}% | {pk['PHAI']*100:.1f}% | {pk['LEN']*100:.1f}% | {pk['XUONG']*100:.1f}% | `{sha_pref}` |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Cross-Seed Validation Comparison",
        "",
        "| Variant Descriptor | Weight $w_{\\text{hn}}$ | Mean Val Recall $\\pm$ Std | Mean Val FAR $\\pm$ Std | Mean Val 5-Class Accuracy | Best Operating Candidate |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for item in ablation_grid:
        var = item["variant"]
        w = item["w_hn"]
        recs = [training_summary[var]["seeds"][f"seed{s}"]["val_cmd_recall"] for s in seeds]
        fars = [training_summary[var]["seeds"][f"seed{s}"]["val_nt_far"] for s in seeds]
        accs = [training_summary[var]["seeds"][f"seed{s}"]["val_acc"] for s in seeds]
        m_rec = float(np.mean(recs))
        s_rec = float(np.std(recs, ddof=1))
        m_far = float(np.mean(fars))
        s_far = float(np.std(fars, ddof=1))
        m_acc = float(np.mean(accs))
        lines.append(
            f"| **{item['desc']}** | {w:.1f} | **{m_rec*100:.2f}% $\\pm$ {s_rec*100:.2f}%** | **{m_far*100:.2f}% $\\pm$ {s_far*100:.2f}%** | {m_acc*100:.2f}% | `{var}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Next Steps for Phase 73",
        "",
        "- All 12 checkpoints and metadata hashes are permanently archived in `experiments/vieneu_kws_research_v8/models/checkpoints/`.",
        "- In Phase 73, these variants alongside historical Cascade v1 and Candidate B will undergo robust validation calibration across clean and synthetic noise/gain conditions to select the top finalists for locked test benchmarking."
    ])

    rep_path = os.path.join(REPORTS_DIR, "PHASE72_WEIGHT_ABLATION.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {rep_path}")


if __name__ == "__main__":
    run_phase72_training_ablation()
