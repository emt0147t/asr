"""
Canonical Prediction Generator for Synthetic Vietnamese KWS V4 Integrity Audit.
Executes fresh forward inference on frozen checkpoints and manifests without modification.
Produces versioned, row-level canonical prediction records.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def generate_all_canonical_predictions():
    base_dir = Path("experiments/synthetic_kws_v4/audit")
    base_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "Arm A (Control)": Path("experiments/synthetic_kws_v4/checkpoints/arm_a_control.pt"),
        "Arm B (Onset-Aware, No Resample)": Path("experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt"),
        "Arm C (Onset-Aware + Resample)": Path("experiments/synthetic_kws_v4/checkpoints/arm_c_onset_with_resample.pt"),
        "V3 Champion TC-ResNet-14": Path("experiments/synthetic_kws_v3/checkpoints/best_tcresnet14_v3.pt")
    }

    partitions = {
        "hist_144": Path("experiments/synthetic_kws_baseline/test_manifest.csv"),
        "hist_216": Path("experiments/synthetic_kws_v2/test_manifest_6class.csv"),
        "v2_fresh_300": Path("experiments/synthetic_kws_v2/fresh_holdout_manifest_6class.csv"),
        "v3_fresh_300": Path("experiments/synthetic_kws_v3/fresh_holdout_manifest_6class.csv"),
        "v4_fresh_300": Path("experiments/synthetic_kws_v4/fresh_holdout_manifest_6class.csv"),
        "val_baseline_216": Path("experiments/synthetic_kws_v3/val_manifest_6class.csv"),
        "val_onset_216": Path("experiments/synthetic_kws_v4/val_manifest_onset_aware.csv")
    }

    # Verify all exist
    for cname, cp in checkpoints.items():
        assert cp.exists(), f"Missing checkpoint: {cp}"
    for pname, pp in partitions.items():
        assert pp.exists(), f"Missing partition manifest: {pp}"

    device = torch.device("cpu")
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)

    rows = []

    for arm_name, ckpt_path in checkpoints.items():
        ckpt_sha = sha256_file(ckpt_path)
        print(f"Loading checkpoint {arm_name} ({ckpt_path})...")
        model = VietnameseTCResNet(num_classes=6)
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_dict["model_state_dict"])
        model.eval()

        for part_id, part_path in partitions.items():
            part_sha = sha256_file(part_path)
            df = pd.read_csv(part_path)
            print(f"  Inference: {arm_name} on {part_id} ({len(df)} rows)...")

            for idx, r in df.iterrows():
                rec_id = r["recording_id"]
                spk_id = r["speaker_id"]
                true_cid = int(r["class_id"])
                true_label = str(r["class_label"])
                wav_path = Path(r["processed_path"])

                audio, sr = sf.read(str(wav_path), dtype="float32")
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                else:
                    audio = audio[:16000]

                inp = torch.from_numpy(audio).float()
                with torch.no_grad():
                    log_mel = extractor(inp)
                    logits = model(log_mel)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_cid = int(np.argmax(probs))

                pred_label = CLASS_NAMES_6[pred_cid]
                is_correct = bool(true_cid == pred_cid)

                rows.append({
                    "model_arm": arm_name,
                    "checkpoint_path": str(ckpt_path).replace("\\", "/"),
                    "checkpoint_sha256": ckpt_sha,
                    "partition_id": part_id,
                    "manifest_path": str(part_path).replace("\\", "/"),
                    "manifest_sha256": part_sha,
                    "recording_id": rec_id,
                    "speaker_id": spk_id,
                    "processed_path": str(wav_path).replace("\\", "/"),
                    "true_class_id": true_cid,
                    "true_class_label": true_label,
                    "pred_class_id": pred_cid,
                    "pred_class_label": pred_label,
                    "correct": is_correct,
                    "prob_trai": float(probs[0]),
                    "prob_phai": float(probs[1]),
                    "prob_len": float(probs[2]),
                    "prob_xuong": float(probs[3]),
                    "prob_unknown": float(probs[4]),
                    "prob_silence": float(probs[5])
                })

    df_out = pd.DataFrame(rows)
    csv_out = base_dir / "canonical_predictions.csv"
    json_out = base_dir / "canonical_predictions.json"

    df_out.to_csv(csv_out, index=False)
    print(f"Saved canonical predictions CSV: {csv_out} ({len(df_out)} rows)")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved canonical predictions JSON: {json_out}")


if __name__ == "__main__":
    generate_all_canonical_predictions()
