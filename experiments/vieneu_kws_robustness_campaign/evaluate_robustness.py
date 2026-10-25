"""
Phase 30: Controlled Synthetic-Noise Robustness Benchmark.
Evaluates the three 6-class CNN checkpoints (seeds 42, 43, 44) across:
  - Clean baseline (0 dB noise added / original)
  - 4 noise types: White, Pink, HVAC, Babble
  - 5 SNRs: 20, 15, 10, 5, 0 dB
Evaluates on both validation and held-out test splits.

Outputs:
  - Robustness degradation metrics (accuracy, macro F1, per-class recall)
  - UNKNOWN false activation rate into target commands vs SNR
  - SILENCE out-of-condition behavior vs SNR
  - Per-seed and multi-seed summary statistics (mean +/- std)
  - Full per-sample prediction CSVs saved to predictions/
  - Detailed summary JSON: predictions/robustness_results.json
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.models.kws_cnn import VietnameseKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(CAMPAIGN_DIR, "predictions")

LABEL_TO_ID = {
    "TRAI": 0,
    "PHAI": 1,
    "LEN": 2,
    "XUONG": 3,
    "UNKNOWN": 4,
    "SILENCE": 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]

def load_audio_batch(wav_paths, target_len=16000):
    batch = []
    for p in wav_paths:
        full_p = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
        data, sr = sf.read(full_p, dtype="float32")
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        elif len(data) > target_len:
            data = data[:target_len]
        batch.append(data)
    return np.array(batch, dtype=np.float32)

def evaluate_subset(model, extractor, device, df_subset, batch_size=64):
    """Evaluates a DataFrame subset and returns probabilities and predictions."""
    model.eval()
    all_probs = []
    
    n_samples = len(df_subset)
    path_col = "corrupted_wav_path" if "corrupted_wav_path" in df_subset.columns else "kws_wav_path"
    wav_paths = df_subset[path_col].tolist()
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_paths = wav_paths[i:i + batch_size]
            audio_batch = load_audio_batch(batch_paths)
            audio_t = torch.from_numpy(audio_batch).float()
            # Extract features: (B, 1, 40, 101)
            feats = extractor(audio_t).to(device)
            logits = model(feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            
    if all_probs:
        all_probs = np.vstack(all_probs)
    else:
        all_probs = np.zeros((0, 6), dtype=np.float32)
        
    return all_probs

def compute_metrics(true_labels, pred_labels, probs):
    y_true = np.array([LABEL_TO_ID[l] for l in true_labels])
    y_pred = np.array([LABEL_TO_ID[l] for l in pred_labels])
    
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    
    # Per-class recall
    class_recalls = {}
    class_f1s = {}
    for c_id, c_name in ID_TO_LABEL.items():
        total_c = np.sum(y_true == c_id)
        if total_c > 0:
            rec = float(cm[c_id, c_id] / total_c)
        else:
            rec = 0.0
        class_recalls[c_name] = rec
        
        # F1 for class
        tp = cm[c_id, c_id]
        fp = np.sum(cm[:, c_id]) - tp
        fn = np.sum(cm[c_id, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        class_f1s[c_name] = float(f1)
        
    # UNKNOWN false activation rate into target commands
    unknown_mask = (y_true == LABEL_TO_ID["UNKNOWN"])
    if np.sum(unknown_mask) > 0:
        unknown_preds = y_pred[unknown_mask]
        target_ids = [LABEL_TO_ID[k] for k in TARGET_CLASSES]
        unknown_target_fa = int(np.sum(np.isin(unknown_preds, target_ids)))
        unknown_fa_rate = float(unknown_target_fa / np.sum(unknown_mask))
    else:
        unknown_target_fa = 0
        unknown_fa_rate = 0.0
        
    # SILENCE false activation rate into target commands
    silence_mask = (y_true == LABEL_TO_ID["SILENCE"])
    if np.sum(silence_mask) > 0:
        silence_preds = y_pred[silence_mask]
        target_ids = [LABEL_TO_ID[k] for k in TARGET_CLASSES]
        silence_target_fa = int(np.sum(np.isin(silence_preds, target_ids)))
        silence_fa_rate = float(silence_target_fa / np.sum(silence_mask))
    else:
        silence_target_fa = 0
        silence_fa_rate = 0.0
        
    # Target keyword accuracy (recall on target classes)
    target_mask = np.isin(y_true, [LABEL_TO_ID[k] for k in TARGET_CLASSES])
    if np.sum(target_mask) > 0:
        target_acc = float(accuracy_score(y_true[target_mask], y_pred[target_mask]))
    else:
        target_acc = 0.0
        
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "per_class_recall": class_recalls,
        "per_class_f1": class_f1s,
        "target_accuracy": target_acc,
        "unknown_target_fa_count": unknown_target_fa,
        "unknown_target_fa_rate": unknown_fa_rate,
        "silence_target_fa_count": silence_target_fa,
        "silence_target_fa_rate": silence_fa_rate,
        "confusion_matrix": cm.tolist()
    }

def run_evaluation():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running robustness evaluation on device: {device}")
    
    extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
    )
    
    seeds = [42, 43, 44]
    checkpoints = {
        s: os.path.join(BASE_DIR, f"experiments/vieneu_cnn_6class/checkpoints/best_checkpoint_seed{s}.pt")
        for s in seeds
    }
    
    clean_val_path = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_6class/splits/val_manifest.csv")
    clean_test_path = os.path.join(BASE_DIR, "experiments/synthetic_kws_dataset_vieneu_6class/splits/test_manifest.csv")
    corrupted_val_path = os.path.join(CAMPAIGN_DIR, "corrupted_val_manifest.csv")
    corrupted_test_path = os.path.join(CAMPAIGN_DIR, "corrupted_test_manifest.csv")
    
    df_clean_val = pd.read_csv(clean_val_path)
    df_clean_test = pd.read_csv(clean_test_path)
    df_corrupted_val = pd.read_csv(corrupted_val_path)
    df_corrupted_test = pd.read_csv(corrupted_test_path)
    
    noise_types = ["white", "pink", "hvac", "babble"]
    snr_levels = [20, 15, 10, 5, 0]
    
    results = {
        "seeds": seeds,
        "noise_types": noise_types,
        "snr_levels": snr_levels,
        "per_seed": {},
        "summary": {}
    }
    
    for seed in seeds:
        ckpt_path = checkpoints[seed]
        print(f"\n=======================================================")
        print(f"Loading Checkpoint Seed {seed}: {ckpt_path}")
        print(f"=======================================================")
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = VietnameseKWSNet(num_classes=6)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        
        seed_results = {
            "val": {"clean": None, "conditions": {}},
            "test": {"clean": None, "conditions": {}}
        }
        
        # 1. Clean Validation
        val_clean_probs = evaluate_subset(model, extractor, device, df_clean_val)
        val_clean_preds = [CLASS_NAMES[p] for p in np.argmax(val_clean_probs, axis=1)]
        seed_results["val"]["clean"] = compute_metrics(df_clean_val["class_label"].tolist(), val_clean_preds, val_clean_probs)
        
        # 2. Corrupted Validation
        for n_type in noise_types:
            for snr in snr_levels:
                cond_key = f"{n_type}_{snr}db"
                sub_df = df_corrupted_val[(df_corrupted_val["noise_type"] == n_type) & (df_corrupted_val["target_snr_db"] == snr)]
                probs = evaluate_subset(model, extractor, device, sub_df)
                preds = [CLASS_NAMES[p] for p in np.argmax(probs, axis=1)]
                seed_results["val"]["conditions"][cond_key] = compute_metrics(sub_df["class_label"].tolist(), preds, probs)
                
        # 3. Clean Test
        test_clean_probs = evaluate_subset(model, extractor, device, df_clean_test)
        test_clean_preds = [CLASS_NAMES[p] for p in np.argmax(test_clean_probs, axis=1)]
        seed_results["test"]["clean"] = compute_metrics(df_clean_test["class_label"].tolist(), test_clean_preds, test_clean_probs)
        
        # Save test clean prediction dataframe
        test_pred_rows = []
        for i, row in df_clean_test.iterrows():
            p_dist = test_clean_probs[i]
            p_lbl = test_clean_preds[i]
            test_pred_rows.append({
                "sample_id": row["sample_id"],
                "condition": "clean",
                "noise_type": "none",
                "snr_db": "inf",
                "true_label": row["class_label"],
                "pred_label": p_lbl,
                "confidence": float(np.max(p_dist)),
                "p_TRAI": float(p_dist[0]),
                "p_PHAI": float(p_dist[1]),
                "p_LEN": float(p_dist[2]),
                "p_XUONG": float(p_dist[3]),
                "p_UNKNOWN": float(p_dist[4]),
                "p_SILENCE": float(p_dist[5]),
                "is_correct": int(p_lbl == row["class_label"])
            })
            
        # 4. Corrupted Test
        for n_type in noise_types:
            for snr in snr_levels:
                cond_key = f"{n_type}_{snr}db"
                sub_df = df_corrupted_test[(df_corrupted_test["noise_type"] == n_type) & (df_corrupted_test["target_snr_db"] == snr)]
                probs = evaluate_subset(model, extractor, device, sub_df)
                preds = [CLASS_NAMES[p] for p in np.argmax(probs, axis=1)]
                metrics = compute_metrics(sub_df["class_label"].tolist(), preds, probs)
                seed_results["test"]["conditions"][cond_key] = metrics
                
                print(f"Seed {seed} | Test {cond_key:15s} | Acc: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']*100:.2f}% | UNK FA: {metrics['unknown_target_fa_rate']*100:.1f}%")
                
                # Append rows
                for i, (_, row) in enumerate(sub_df.iterrows()):
                    p_dist = probs[i]
                    p_lbl = preds[i]
                    test_pred_rows.append({
                        "sample_id": row["sample_id"],
                        "condition": cond_key,
                        "noise_type": n_type,
                        "snr_db": snr,
                        "true_label": row["class_label"],
                        "pred_label": p_lbl,
                        "confidence": float(np.max(p_dist)),
                        "p_TRAI": float(p_dist[0]),
                        "p_PHAI": float(p_dist[1]),
                        "p_LEN": float(p_dist[2]),
                        "p_XUONG": float(p_dist[3]),
                        "p_UNKNOWN": float(p_dist[4]),
                        "p_SILENCE": float(p_dist[5]),
                        "is_correct": int(p_lbl == row["class_label"])
                    })
                    
        # Save per-seed test predictions CSV
        df_seed_preds = pd.DataFrame(test_pred_rows)
        seed_preds_path = os.path.join(PREDICTIONS_DIR, f"test_robustness_predictions_seed{seed}.csv")
        df_seed_preds.to_csv(seed_preds_path, index=False)
        print(f"Saved test predictions for seed {seed} ({len(df_seed_preds)} rows) to: {seed_preds_path}")
        
        results["per_seed"][str(seed)] = seed_results

    # Compute Summary Statistics across seeds (Mean & Std)
    summary_test = {"clean": {}, "conditions": {}}
    
    # Clean test summary
    clean_accs = [results["per_seed"][str(s)]["test"]["clean"]["accuracy"] for s in seeds]
    clean_f1s = [results["per_seed"][str(s)]["test"]["clean"]["macro_f1"] for s in seeds]
    clean_unk_fa = [results["per_seed"][str(s)]["test"]["clean"]["unknown_target_fa_rate"] for s in seeds]
    summary_test["clean"] = {
        "accuracy_mean": float(np.mean(clean_accs)),
        "accuracy_std": float(np.std(clean_accs)),
        "macro_f1_mean": float(np.mean(clean_f1s)),
        "macro_f1_std": float(np.std(clean_f1s)),
        "unknown_target_fa_rate_mean": float(np.mean(clean_unk_fa)),
        "unknown_target_fa_rate_std": float(np.std(clean_unk_fa))
    }
    
    # Corrupted test summary
    for n_type in noise_types:
        for snr in snr_levels:
            cond_key = f"{n_type}_{snr}db"
            accs = [results["per_seed"][str(s)]["test"]["conditions"][cond_key]["accuracy"] for s in seeds]
            f1s = [results["per_seed"][str(s)]["test"]["conditions"][cond_key]["macro_f1"] for s in seeds]
            unk_fas = [results["per_seed"][str(s)]["test"]["conditions"][cond_key]["unknown_target_fa_rate"] for s in seeds]
            sil_fas = [results["per_seed"][str(s)]["test"]["conditions"][cond_key]["silence_target_fa_rate"] for s in seeds]
            tgt_accs = [results["per_seed"][str(s)]["test"]["conditions"][cond_key]["target_accuracy"] for s in seeds]
            
            # Per class recalls mean & std
            class_rec_means = {}
            for c_name in CLASS_NAMES:
                c_recs = [results["per_seed"][str(s)]["test"]["conditions"][cond_key]["per_class_recall"][c_name] for s in seeds]
                class_rec_means[c_name] = {
                    "mean": float(np.mean(c_recs)),
                    "std": float(np.std(c_recs))
                }
                
            summary_test["conditions"][cond_key] = {
                "noise_type": n_type,
                "snr_db": snr,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "target_accuracy_mean": float(np.mean(tgt_accs)),
                "target_accuracy_std": float(np.std(tgt_accs)),
                "unknown_target_fa_rate_mean": float(np.mean(unk_fas)),
                "unknown_target_fa_rate_std": float(np.std(unk_fas)),
                "silence_target_fa_rate_mean": float(np.mean(sil_fas)),
                "silence_target_fa_rate_std": float(np.std(sil_fas)),
                "per_class_recall": class_rec_means
            }
            
    results["summary"]["test"] = summary_test
    
    # Save overall summary JSON
    results_json_path = os.path.join(PREDICTIONS_DIR, "robustness_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved overall robustness evaluation results to: {results_json_path}")
    
    # Print Quick Summary Table
    print("\n" + "=" * 80)
    print("ROBUSTNESS BENCHMARK SUMMARY (Held-out Test, Mean ± Std over 3 seeds)")
    print("=" * 80)
    print(f"{'Condition':<16} | {'Accuracy (%)':<15} | {'Macro F1 (%)':<15} | {'Target Acc (%)':<15} | {'UNK FA Rate (%)':<15}")
    print("-" * 80)
    print(f"{'Clean':<16} | {summary_test['clean']['accuracy_mean']*100:5.2f} ± {summary_test['clean']['accuracy_std']*100:4.2f}   | {summary_test['clean']['macro_f1_mean']*100:5.2f} ± {summary_test['clean']['macro_f1_std']*100:4.2f}   | {'---':<15} | {summary_test['clean']['unknown_target_fa_rate_mean']*100:5.2f} ± {summary_test['clean']['unknown_target_fa_rate_std']*100:4.2f}")
    print("-" * 80)
    for n_type in noise_types:
        for snr in snr_levels:
            cond_key = f"{n_type}_{snr}db"
            m = summary_test["conditions"][cond_key]
            print(f"{cond_key:<16} | {m['accuracy_mean']*100:5.2f} ± {m['accuracy_std']*100:4.2f}   | {m['macro_f1_mean']*100:5.2f} ± {m['macro_f1_std']*100:4.2f}   | {m['target_accuracy_mean']*100:5.2f} ± {m['target_accuracy_std']*100:4.2f}   | {m['unknown_target_fa_rate_mean']*100:5.2f} ± {m['unknown_target_fa_rate_std']*100:4.2f}")
        print("-" * 80)

if __name__ == "__main__":
    run_evaluation()
