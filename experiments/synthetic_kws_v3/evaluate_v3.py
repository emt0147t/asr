"""
Comprehensive Evaluation Pipeline for Synthetic Vietnamese KWS V3.
Evaluates frozen models across:
  1. Historical Diagnostic Holdout (SPK_CALIB_018..020, 216 clips)
  2. V2 Diagnostic Fresh Holdout (SPK_CALIB_021..025, 300 clips)
  3. V3 Untouched Fresh Synthetic-Seed Holdout (SPK_CALIB_026..030, 300 clips)

Measures:
  - Per-class precision, recall, F1, support
  - Macro-F1 and Overall Accuracy
  - TRAI and PHAI recall explicitly
  - Command miss rate (target commands predicted as UNKNOWN, SILENCE, or wrong command)
  - False command activation rate on UNKNOWN and SILENCE
  - Disaggregated CPU latency (features, model forward, end-to-end)
"""

import sys
import os
import time
import json
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet, VietnameseTemporalTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
TARGET_IDS = {0, 1, 2, 3}


def evaluate_manifest(model, manifest_csv: Path, partition_name: str, device: torch.device = torch.device("cpu")):
    df = pd.read_csv(manifest_csv)
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    model.eval()
    
    y_true, y_pred, y_probs = [], [], []
    lat_feat, lat_model, lat_e2e = [], [], []
    
    for idx, row in df.iterrows():
        wav_p = Path(row["processed_path"])
        label = int(row["class_id"])
        audio, _ = sf.read(str(wav_p), dtype="float32")
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        else:
            audio = audio[:16000]
            
        t_start = time.perf_counter()
        inp = torch.from_numpy(audio).float()
        
        t_feat0 = time.perf_counter()
        with torch.no_grad():
            log_mel = extractor(inp)
        t_feat1 = time.perf_counter()
        
        t_mod0 = time.perf_counter()
        with torch.no_grad():
            logits = model(log_mel)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred = int(np.argmax(probs))
        t_mod1 = time.perf_counter()
        t_end = time.perf_counter()
        
        lat_feat.append((t_feat1 - t_feat0) * 1000.0)
        lat_model.append((t_mod1 - t_mod0) * 1000.0)
        lat_e2e.append((t_end - t_start) * 1000.0)
        
        y_true.append(label)
        y_pred.append(pred)
        y_probs.append(probs)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lat_feat = np.array(lat_feat)
    lat_model = np.array(lat_model)
    lat_e2e = np.array(lat_e2e)
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(6)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    
    # Target command miss rate
    target_mask = np.isin(y_true, list(TARGET_IDS))
    target_total = int(target_mask.sum())
    target_correct = int((y_true[target_mask] == y_pred[target_mask]).sum())
    target_misses = target_total - target_correct
    target_miss_rate = (target_misses / target_total * 100.0) if target_total > 0 else 0.0
    
    # False activations on negatives (UNKNOWN and SILENCE triggering a keyword)
    unknown_mask = (y_true == 4)
    unknown_total = int(unknown_mask.sum())
    unknown_fa_count = int(np.isin(y_pred[unknown_mask], list(TARGET_IDS)).sum()) if unknown_total > 0 else 0
    unknown_fa_rate = (unknown_fa_count / unknown_total * 100.0) if unknown_total > 0 else 0.0
    
    silence_mask = (y_true == 5)
    silence_total = int(silence_mask.sum())
    silence_fa_count = int(np.isin(y_pred[silence_mask], list(TARGET_IDS)).sum()) if silence_total > 0 else 0
    silence_fa_rate = (silence_fa_count / silence_total * 100.0) if silence_total > 0 else 0.0
    
    # TRAI / PHAI confusion
    trai_mask = (y_true == 0)
    trai_total = int(trai_mask.sum())
    trai_as_phai = int((y_pred[trai_mask] == 1).sum()) if trai_total > 0 else 0
    trai_as_unknown = int((y_pred[trai_mask] == 4).sum()) if trai_total > 0 else 0
    
    phai_mask = (y_true == 1)
    phai_total = int(phai_mask.sum())
    phai_as_trai = int((y_pred[phai_mask] == 0).sum()) if phai_total > 0 else 0
    phai_as_unknown = int((y_pred[phai_mask] == 4).sum()) if phai_total > 0 else 0
    phai_as_len = int((y_pred[phai_mask] == 2).sum()) if phai_total > 0 else 0
    
    return {
        "partition_name": partition_name,
        "sample_count": len(y_true),
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "f1": [float(f) for f in f1],
        "support": [int(s) for s in support],
        "confusion_matrix": cm.tolist(),
        "command_misses": {
            "target_total": target_total,
            "target_correct": target_correct,
            "target_miss_count": target_misses,
            "target_miss_rate_pct": float(target_miss_rate)
        },
        "false_activations": {
            "unknown_total": unknown_total,
            "unknown_fa_count": unknown_fa_count,
            "unknown_fa_rate_pct": float(unknown_fa_rate),
            "silence_total": silence_total,
            "silence_fa_count": silence_fa_count,
            "silence_fa_rate_pct": float(silence_fa_rate)
        },
        "trai_phai_specifics": {
            "trai_total": trai_total,
            "trai_recall": float(recall[0]),
            "trai_as_phai": trai_as_phai,
            "trai_as_unknown": trai_as_unknown,
            "phai_total": phai_total,
            "phai_recall": float(recall[1]),
            "phai_as_trai": phai_as_trai,
            "phai_as_unknown": phai_as_unknown,
            "phai_as_len": phai_as_len
        },
        "latency_feat_ms": {
            "mean": float(np.mean(lat_feat)),
            "p50": float(np.percentile(lat_feat, 50)),
            "p90": float(np.percentile(lat_feat, 90)),
            "p99": float(np.percentile(lat_feat, 99))
        },
        "latency_model_ms": {
            "mean": float(np.mean(lat_model)),
            "p50": float(np.percentile(lat_model, 50)),
            "p90": float(np.percentile(lat_model, 90)),
            "p99": float(np.percentile(lat_model, 99))
        },
        "latency_e2e_ms": {
            "mean": float(np.mean(lat_e2e)),
            "p50": float(np.percentile(lat_e2e, 50)),
            "p90": float(np.percentile(lat_e2e, 90)),
            "p99": float(np.percentile(lat_e2e, 99))
        }
    }


def run_full_evaluation():
    v3_dir = Path("experiments/synthetic_kws_v3")
    ckpt_dir = v3_dir / "checkpoints"
    
    # Model candidate checkpoints
    ckpt_tc = ckpt_dir / "best_tcresnet14_v3.pt"
    ckpt_temp = ckpt_dir / "best_temporal_tcresnet14_v3.pt"
    
    device = torch.device("cpu")
    
    models = {}
    if ckpt_tc.exists():
        m1 = VietnameseTCResNet(num_classes=6)
        c1 = torch.load(ckpt_tc, map_location=device)
        m1.load_state_dict(c1["model_state_dict"])
        models["TC-ResNet-14-V3"] = (m1, ckpt_tc)
        
    if ckpt_temp.exists():
        m2 = VietnameseTemporalTCResNet(num_classes=6)
        c2 = torch.load(ckpt_temp, map_location=device)
        m2.load_state_dict(c2["model_state_dict"])
        models["Temporal-TC-ResNet-14-V3"] = (m2, ckpt_temp)
        
    assert len(models) > 0, "No trained checkpoints found!"
    
    # Partitions to evaluate
    partitions = [
        ("Validation (SPK 015-017, 216 clips)", v3_dir / "val_manifest_6class.csv"),
        ("Historical Test (SPK 018-020, 216 clips [Diagnostic])", Path("experiments/synthetic_kws_v2/test_manifest_6class.csv")),
        ("V2 Fresh Holdout (SPK 021-025, 300 clips [Diagnostic])", Path("experiments/synthetic_kws_v2/fresh_holdout_manifest_6class.csv")),
        ("V3 Fresh Holdout (SPK 026-030, 300 clips [Untouched])", v3_dir / "fresh_holdout_manifest_6class.csv")
    ]
    
    all_results = {}
    
    for m_label, (m_obj, c_path) in models.items():
        param_count = sum(p.numel() for p in m_obj.parameters() if p.requires_grad)
        ckpt_size_kb = c_path.stat().st_size / 1024.0
        print(f"\n=======================================================")
        print(f"EVALUATING MODEL: {m_label}")
        print(f"Parameters: {param_count:,} | Checkpoint: {ckpt_size_kb:.1f} KB")
        print(f"=======================================================")
        
        m_results = {
            "model_name": m_label,
            "param_count": param_count,
            "ckpt_size_kb": float(ckpt_size_kb),
            "partitions": {}
        }
        
        for part_name, part_csv in partitions:
            if not part_csv.exists():
                print(f"Skipping {part_name}: manifest {part_csv} does not exist yet.")
                continue
            res = evaluate_manifest(m_obj, part_csv, part_name, device=device)
            m_results["partitions"][part_name] = res
            
            print(f"\n--- {part_name} ---")
            print(f"  Accuracy: {res['overall_accuracy']*100:.2f}% | Macro-F1: {res['macro_f1']:.4f}")
            print(f"  TRAI Recall: {res['trai_phai_specifics']['trai_recall']*100:.1f}% | PHAI Recall: {res['trai_phai_specifics']['phai_recall']*100:.1f}%")
            print(f"  Command Miss Rate: {res['command_misses']['target_miss_rate_pct']:.2f}% ({res['command_misses']['target_miss_count']}/{res['command_misses']['target_total']})")
            print(f"  False Activation Rate on UNKNOWN: {res['false_activations']['unknown_fa_rate_pct']:.2f}%")
            print(f"  False Activation Rate on SILENCE: {res['false_activations']['silence_fa_rate_pct']:.2f}%")
            print(f"  CPU Latency (E2E): Mean={res['latency_e2e_ms']['mean']:.2f}ms, P50={res['latency_e2e_ms']['p50']:.2f}ms, P90={res['latency_e2e_ms']['p90']:.2f}ms")
            
        all_results[m_label] = m_results
        
    out_eval = v3_dir / "evaluation_results_v3.json"
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll evaluation results saved to {out_eval}")


if __name__ == "__main__":
    run_full_evaluation()
