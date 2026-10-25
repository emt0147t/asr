"""
Phase 31: Validation-Only Confidence-Threshold Calibration and Test Evaluation.
Performs grid search over candidate confidence thresholds theta in [0.50, 0.99]
STRICTLY on validation data (clean validation + corrupted validation).

Objective:
  Maximize validation macro-F1 subject to:
    - Target command recall >= 95% (or best feasible Pareto tradeoff >= 90%)
    - False-command activation rate on UNKNOWN <= 5%
    - False-command activation rate on SILENCE <= 5%

Outputs:
  - calibration/calibration_seed{42,43,44}.json: full calibration curves & diagnostics
  - calibration/selected_thresholds.json: frozen selected threshold per seed
  - calibration/test_calibration_comparison.json: held-out test evaluation BEFORE vs AFTER calibration
  - predictions/calibrated_test_predictions_seed{42,43,44}.csv: per-sample test predictions under calibrated policy
"""

import os
import sys
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
from experiments.vieneu_kws_robustness_campaign.confidence_policy import (
    apply_confidence_policy,
    TARGET_COMMAND_INDICES,
    UNKNOWN_INDEX,
    SILENCE_INDEX
)

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(CAMPAIGN_DIR, "calibration")
PREDICTIONS_DIR = os.path.join(CAMPAIGN_DIR, "predictions")

LABEL_TO_ID = {"TRAI": 0, "PHAI": 1, "LEN": 2, "XUONG": 3, "UNKNOWN": 4, "SILENCE": 5}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]

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

def compute_model_probabilities(model, extractor, device, df, batch_size=64):
    """Computes softmax probability distributions for a DataFrame."""
    model.eval()
    all_probs = []
    path_col = "corrupted_wav_path" if "corrupted_wav_path" in df.columns else "kws_wav_path"
    wav_paths = df[path_col].tolist()
    
    with torch.no_grad():
        for i in range(0, len(wav_paths), batch_size):
            batch_paths = wav_paths[i:i + batch_size]
            audio_batch = load_audio_batch(batch_paths)
            audio_t = torch.from_numpy(audio_batch).float()
            feats = extractor(audio_t).to(device)
            logits = model(feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            
    if all_probs:
        return np.vstack(all_probs)
    return np.zeros((0, 6), dtype=np.float32)

def evaluate_threshold_policy(probs, targets, threshold):
    """Evaluates metrics given probability matrix and confidence threshold."""
    preds, confs, rejs = apply_confidence_policy(probs, threshold)
    
    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(targets, preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(targets, preds, average="macro", zero_division=0))
    
    cm = confusion_matrix(targets, preds, labels=list(range(6)))
    
    # Per-class recall
    recalls = {}
    for idx, cname in enumerate(CLASS_NAMES):
        c_mask = (targets == idx)
        if np.sum(c_mask) > 0:
            recalls[cname] = float(np.sum((targets == idx) & (preds == idx)) / np.sum(c_mask))
        else:
            recalls[cname] = 0.0
            
    # Target command recall
    target_mask = np.isin(targets, list(TARGET_COMMAND_INDICES))
    if np.sum(target_mask) > 0:
        target_rec = float(np.sum(target_mask & (preds == targets)) / np.sum(target_mask))
    else:
        target_rec = 0.0
        
    # UNKNOWN false command activation rate (UNKNOWN classified as one of the 4 target commands)
    unk_mask = (targets == UNKNOWN_INDEX)
    if np.sum(unk_mask) > 0:
        unk_fa_count = int(np.sum(unk_mask & np.isin(preds, list(TARGET_COMMAND_INDICES))))
        unk_fa_rate = float(unk_fa_count / np.sum(unk_mask))
    else:
        unk_fa_count = 0
        unk_fa_rate = 0.0
        
    # SILENCE false command activation rate
    sil_mask = (targets == SILENCE_INDEX)
    if np.sum(sil_mask) > 0:
        sil_fa_count = int(np.sum(sil_mask & np.isin(preds, list(TARGET_COMMAND_INDICES))))
        sil_fa_rate = float(sil_fa_count / np.sum(sil_mask))
    else:
        sil_fa_count = 0
        sil_fa_rate = 0.0
        
    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "target_recall": target_rec,
        "unknown_recall": recalls["UNKNOWN"],
        "unknown_false_activation_rate": unk_fa_rate,
        "unknown_false_activation_count": unk_fa_count,
        "silence_recall": recalls["SILENCE"],
        "silence_false_activation_rate": sil_fa_rate,
        "silence_false_activation_count": sil_fa_count,
        "rejection_rate": float(np.mean(rejs)),
        "rejection_count": int(np.sum(rejs)),
        "per_class_recall": recalls,
        "confusion_matrix": cm.tolist()
    }

def calibrate_seed(seed: int, model, extractor, device, df_val_clean, df_val_corr, config):
    print(f"\n=======================================================")
    print(f"Phase 31: Calibrating Threshold for Seed {seed}")
    print(f"=======================================================")
    
    val_clean_targets = np.array([LABEL_TO_ID[l] for l in df_val_clean["class_label"]])
    val_corr_targets = np.array([LABEL_TO_ID[l] for l in df_val_corr["class_label"]])
    
    print("Computing validation probabilities (clean & corrupted)...")
    val_clean_probs = compute_model_probabilities(model, extractor, device, df_val_clean)
    val_corr_probs = compute_model_probabilities(model, extractor, device, df_val_corr)
    
    candidate_thresholds = config["phase31_calibration"]["candidate_thresholds"]
    min_tgt_rec = config["phase31_calibration"]["constraint_min_target_recall"]
    max_fa_unk = config["phase31_calibration"]["constraint_max_false_activation_unknown"]
    max_fa_sil = config["phase31_calibration"]["constraint_max_false_activation_silence"]
    
    results = []
    feasible_entries = []
    
    for theta in candidate_thresholds:
        clean_res = evaluate_threshold_policy(val_clean_probs, val_clean_targets, theta)
        corr_res = evaluate_threshold_policy(val_corr_probs, val_corr_targets, theta)
        
        # Validation score: clean macro F1 with regularizing weight on corrupted F1
        val_score = 0.7 * clean_res["macro_f1"] + 0.3 * corr_res["macro_f1"]
        
        # Check strict feasibility on clean validation
        is_feasible = (
            clean_res["target_recall"] >= min_tgt_rec and
            clean_res["unknown_false_activation_rate"] <= max_fa_unk and
            clean_res["silence_false_activation_rate"] <= max_fa_sil
        )
        
        entry = {
            "threshold": theta,
            "val_score": float(val_score),
            "clean": clean_res,
            "corrupted": corr_res,
            "is_feasible": is_feasible
        }
        results.append(entry)
        if is_feasible:
            feasible_entries.append(entry)
            
    if feasible_entries:
        # Choose feasible threshold that maximizes validation score
        best_entry = max(feasible_entries, key=lambda x: x["val_score"])
        best_theta = best_entry["threshold"]
        rationale = f"Strict constraints met (Target Rec >= {min_tgt_rec*100}%, UNK FA <= {max_fa_unk*100}%). Maximizes validation Macro-F1."
    else:
        # If strict constraint cannot be met, choose Pareto-optimal compromise:
        # Maintain Target Recall >= 90% while minimizing UNK FA rate
        compromise_pool = [e for e in results if e["clean"]["target_recall"] >= 0.90]
        if compromise_pool:
            # Sort by lowest unknown false activation rate, tie-break on macro-F1
            best_entry = min(compromise_pool, key=lambda x: (x["clean"]["unknown_false_activation_rate"], -x["clean"]["macro_f1"]))
            best_theta = best_entry["threshold"]
            rationale = (f"Strict constraints unachievable due to synthetic UNKNOWN phoneme overlap. "
                         f"Selected Pareto compromise (Target Rec {best_entry['clean']['target_recall']*100:.1f}% >= 90%, "
                         f"UNK FA reduced to {best_entry['clean']['unknown_false_activation_rate']*100:.1f}%).")
        else:
            best_theta = 0.85
            rationale = "Fallback default threshold theta=0.85."
            best_entry = next((e for e in results if np.isclose(e["threshold"], 0.85)), results[len(results)//2])
            
    print(f"Selected Validation Threshold for Seed {seed}: theta = {best_theta:.2f}")
    print(f"  Rationale: {rationale}")
    print(f"  Clean Val Target Recall: {best_entry['clean']['target_recall']*100:.2f}% | UNK FA Rate: {best_entry['clean']['unknown_false_activation_rate']*100:.2f}% | Macro F1: {best_entry['clean']['macro_f1']*100:.2f}%")
    
    # Save seed calibration curve
    out_file = os.path.join(CALIBRATION_DIR, f"calibration_seed{seed}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "selected_threshold": best_theta,
            "is_strictly_feasible": bool(feasible_entries),
            "selection_rationale": rationale,
            "constraints": {
                "min_target_recall": min_tgt_rec,
                "max_fa_unknown": max_fa_unk,
                "max_fa_silence": max_fa_sil
            },
            "candidate_evaluations": results
        }, f, indent=2)
        
    return best_theta, results

def main():
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    cfg_path = os.path.join(CAMPAIGN_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, win_length=400, hop_length=160, f_min=20.0, f_max=8000.0
    )
    
    clean_val_manifest = os.path.join(BASE_DIR, config["dataset"]["val_manifest"])
    clean_test_manifest = os.path.join(BASE_DIR, config["dataset"]["test_manifest"])
    corr_val_manifest = os.path.join(CAMPAIGN_DIR, "corrupted_val_manifest.csv")
    corr_test_manifest = os.path.join(CAMPAIGN_DIR, "corrupted_test_manifest.csv")
    
    df_val_clean = pd.read_csv(clean_val_manifest)
    df_val_corr = pd.read_csv(corr_val_manifest)
    df_test_clean = pd.read_csv(clean_test_manifest)
    df_test_corr = pd.read_csv(corr_test_manifest)
    
    seeds = [42, 43, 44]
    selected_thresholds = {}
    
    test_clean_targets = np.array([LABEL_TO_ID[l] for l in df_test_clean["class_label"]])
    test_corr_targets = np.array([LABEL_TO_ID[l] for l in df_test_corr["class_label"]])
    
    test_comparison_results = {}
    
    for seed in seeds:
        ckpt_path = os.path.join(BASE_DIR, config["checkpoints"][f"seed{seed}"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = VietnameseKWSNet(num_classes=6)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        
        # 1. Validation Calibration
        best_theta, _ = calibrate_seed(seed, model, extractor, device, df_val_clean, df_val_corr, config)
        selected_thresholds[f"seed{seed}"] = best_theta
        
        # 2. Held-Out Test Evaluation Before vs After Calibration
        print(f"\n--- Evaluating Held-Out Test Split for Seed {seed} (Uncalibrated vs Calibrated theta={best_theta:.2f}) ---")
        test_clean_probs = compute_model_probabilities(model, extractor, device, df_test_clean)
        test_corr_probs = compute_model_probabilities(model, extractor, device, df_test_corr)
        
        # Baseline (uncalibrated argmax -> theta = 0.0)
        uncal_clean = evaluate_threshold_policy(test_clean_probs, test_clean_targets, threshold=0.0)
        uncal_corr = evaluate_threshold_policy(test_corr_probs, test_corr_targets, threshold=0.0)
        
        # Calibrated
        cal_clean = evaluate_threshold_policy(test_clean_probs, test_clean_targets, threshold=best_theta)
        cal_corr = evaluate_threshold_policy(test_corr_probs, test_corr_targets, threshold=best_theta)
        
        print(f"Clean Test  | Before: Acc {uncal_clean['accuracy']*100:.2f}%, F1 {uncal_clean['macro_f1']*100:.2f}%, UNK FA {uncal_clean['unknown_false_activation_rate']*100:.1f}%, Rej {uncal_clean['rejection_rate']*100:.1f}%")
        print(f"Clean Test  | After : Acc {cal_clean['accuracy']*100:.2f}%, F1 {cal_clean['macro_f1']*100:.2f}%, UNK FA {cal_clean['unknown_false_activation_rate']*100:.1f}%, Rej {cal_clean['rejection_rate']*100:.1f}%")
        
        test_comparison_results[f"seed{seed}"] = {
            "selected_threshold": best_theta,
            "clean_test": {
                "uncalibrated": uncal_clean,
                "calibrated": cal_clean
            },
            "corrupted_test": {
                "uncalibrated": uncal_corr,
                "calibrated": cal_corr
            }
        }
        
        # Save calibrated test predictions CSV
        cal_preds, confs, rejs = apply_confidence_policy(test_clean_probs, best_theta)
        cal_pred_rows = []
        for i, row in df_test_clean.iterrows():
            orig_p = int(np.argmax(test_clean_probs[i]))
            cal_p = int(cal_preds[i])
            cal_pred_rows.append({
                "sample_id": row["sample_id"],
                "voice_id": row["voice_id"],
                "true_label": row["class_label"],
                "uncalibrated_pred": CLASS_NAMES[orig_p],
                "calibrated_pred": CLASS_NAMES[cal_p],
                "confidence": float(confs[i]),
                "threshold": best_theta,
                "is_rejected_to_unknown": int(rejs[i]),
                "uncalibrated_correct": int(orig_p == LABEL_TO_ID[row["class_label"]]),
                "calibrated_correct": int(cal_p == LABEL_TO_ID[row["class_label"]]),
                "p_TRAI": float(test_clean_probs[i, 0]),
                "p_PHAI": float(test_clean_probs[i, 1]),
                "p_LEN": float(test_clean_probs[i, 2]),
                "p_XUONG": float(test_clean_probs[i, 3]),
                "p_UNKNOWN": float(test_clean_probs[i, 4]),
                "p_SILENCE": float(test_clean_probs[i, 5])
            })
        df_cal_preds = pd.DataFrame(cal_pred_rows)
        cal_preds_path = os.path.join(PREDICTIONS_DIR, f"calibrated_test_predictions_seed{seed}.csv")
        df_cal_preds.to_csv(cal_preds_path, index=False)
        print(f"Saved calibrated predictions to: {cal_preds_path}")
        
    # Save frozen thresholds
    summary_file = os.path.join(CALIBRATION_DIR, "selected_thresholds.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(selected_thresholds, f, indent=2)
    print(f"\nSaved all frozen calibration thresholds to: {summary_file}")
    
    # Save test comparison summary
    comparison_file = os.path.join(CALIBRATION_DIR, "test_calibration_comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(test_comparison_results, f, indent=2)
    print(f"Saved test calibration comparison to: {comparison_file}")
    
    # Print Multi-seed Comparison Table
    print("\n" + "=" * 90)
    print("PHASE 31: CONFIDENCE THRESHOLD CALIBRATION — HELD-OUT TEST SUMMARY")
    print("=" * 90)
    print(f"{'Seed':<8} | {'Theta':<6} | {'Clean Acc (%)':<15} | {'Clean Macro-F1 (%)':<20} | {'UNK FA Rate (%)':<18} | {'Target Rec (%)':<15}")
    print(f"{'':8} | {'':6} | {'Uncal -> Cal':<15} | {'Uncal -> Cal':<20} | {'Uncal -> Cal':<18} | {'Uncal -> Cal':<15}")
    print("-" * 90)
    for s in seeds:
        k = f"seed{s}"
        theta = test_comparison_results[k]["selected_threshold"]
        u = test_comparison_results[k]["clean_test"]["uncalibrated"]
        c = test_comparison_results[k]["clean_test"]["calibrated"]
        print(f"{k:<8} | {theta:<6.2f} | {u['accuracy']*100:5.2f} -> {c['accuracy']*100:5.2f}   | {u['macro_f1']*100:5.2f} -> {c['macro_f1']*100:5.2f}        | {u['unknown_false_activation_rate']*100:5.1f} -> {c['unknown_false_activation_rate']*100:5.1f}       | {u['target_recall']*100:5.2f} -> {c['target_recall']*100:5.2f}")
    print("-" * 90)

if __name__ == "__main__":
    main()
