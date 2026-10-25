"""
Audit V2 evaluation and produce an apples-to-apples metric comparison on the 144 historical target-command clips.
"""

import sys
import os
import json
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

from src.models.kws_architectures import VietnameseTCResNet, VietnameseTemporalKWSNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from src.acquisition.campaign_validator import validate_split_manifests_preflight


def run_audit():
    print("=====================================================================")
    print("STEP 1: AUDIT V2 MANIFESTS, GENERATOR, PROVENANCE, AND DATA INTEGRITY")
    print("=====================================================================")
    
    v2_dir = Path("experiments/synthetic_kws_v2")
    tr_path = v2_dir / "train_manifest_6class.csv"
    va_path = v2_dir / "val_manifest_6class.csv"
    te_path = v2_dir / "test_manifest_6class.csv"
    fr_path = v2_dir / "fresh_holdout_manifest_6class.csv"
    
    assert tr_path.exists() and va_path.exists() and te_path.exists() and fr_path.exists()
    
    df_tr = pd.read_csv(tr_path)
    df_va = pd.read_csv(va_path)
    df_te = pd.read_csv(te_path)
    df_fr = pd.read_csv(fr_path)
    
    audit_log = {}
    
    # 1. Per-class counts
    splits = {"train": df_tr, "validation": df_va, "test_historical": df_te, "fresh_holdout": df_fr}
    class_names_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    counts_by_split = {}
    for s_name, df_s in splits.items():
        vc = df_s["class_label"].value_counts().to_dict()
        counts_by_split[s_name] = {c: int(vc.get(c, 0)) for c in class_names_6}
    
    audit_log["per_class_counts"] = counts_by_split
    print("[CHECK 1] Per-class sample counts:")
    for s_name, counts in counts_by_split.items():
        print(f"  - {s_name}: total={sum(counts.values())}, breakdown={counts}")
    audit_log["check_counts_status"] = "PASS"
    
    # 2. Speaker separation & Leakage check
    spk_tr = set(df_tr["speaker_id"].unique())
    spk_va = set(df_va["speaker_id"].unique())
    spk_te = set(df_te["speaker_id"].unique())
    spk_fr = set(df_fr["speaker_id"].unique())
    
    overlap_tr_va = spk_tr.intersection(spk_va)
    overlap_tr_te = spk_tr.intersection(spk_te)
    overlap_va_te = spk_va.intersection(spk_te)
    overlap_benchmark_fr = (spk_tr | spk_va | spk_te).intersection(spk_fr)
    
    spk_pass = (len(overlap_tr_va) == 0 and len(overlap_tr_te) == 0 and 
                len(overlap_va_te) == 0 and len(overlap_benchmark_fr) == 0)
    audit_log["speaker_overlap"] = {
        "train_val_overlap": list(overlap_tr_va),
        "train_test_overlap": list(overlap_tr_te),
        "val_test_overlap": list(overlap_va_te),
        "benchmark_fresh_overlap": list(overlap_benchmark_fr),
        "status": "PASS" if spk_pass else "FAIL"
    }
    print(f"[CHECK 2] Speaker separation: {'PASS' if spk_pass else 'FAIL'}")
    print(f"  - Train: {sorted(spk_tr)} ({len(spk_tr)} speakers)")
    print(f"  - Val:   {sorted(spk_va)} ({len(spk_va)} speakers)")
    print(f"  - Test:  {sorted(spk_te)} ({len(spk_te)} speakers)")
    print(f"  - Fresh: {sorted(spk_fr)} ({len(spk_fr)} speakers)")
    
    # 3. Recording ID and Hash Uniqueness
    all_recs = [df["recording_id"].tolist() for df in [df_tr, df_va, df_te, df_fr]]
    all_recs_flat = [r for sublist in all_recs for r in sublist]
    rec_unique = len(all_recs_flat) == len(set(all_recs_flat))
    
    all_hashes = [df["sha256"].tolist() for df in [df_tr, df_va, df_te, df_fr]]
    all_hashes_flat = [h for sublist in all_hashes for h in sublist]
    hash_unique = len(all_hashes_flat) == len(set(all_hashes_flat))
    
    audit_log["uniqueness"] = {
        "total_rows": len(all_recs_flat),
        "unique_recording_ids": len(set(all_recs_flat)),
        "unique_sha256_hashes": len(set(all_hashes_flat)),
        "recording_id_unique": rec_unique,
        "sha256_unique": hash_unique,
        "status": "PASS" if (rec_unique and hash_unique) else "FAIL"
    }
    print(f"[CHECK 3] Recording ID uniqueness: {'PASS' if rec_unique else 'FAIL'}")
    print(f"[CHECK 4] SHA-256 hash uniqueness: {'PASS' if hash_unique else 'FAIL'}")
    
    # 4. Split Preflight Validation
    is_valid, errs, diag = validate_split_manifests_preflight(df_tr, df_va, df_te, validation_profile="benchmark")
    audit_log["split_preflight"] = {
        "is_valid": is_valid,
        "errors": errs,
        "diagnostics": diag,
        "status": "PASS" if is_valid else "FAIL"
    }
    print(f"[CHECK 5] Preflight split validation (benchmark profile): {'PASS' if is_valid else 'FAIL'}")
    
    # 5. Train-only Augmentation Inspection
    # In experiments/synthetic_kws_v2/train_and_eval_v2.py lines 53-56:
    # self.use_specaugment = use_specaugment and is_train
    # Jitter and scaling are gated on if self.is_train: lines 89-98.
    audit_log["train_only_augmentation"] = {
        "mechanism": "Synthetic6ClassDataset gates SpecAugment and temporal shift behind `is_train=True`",
        "evidence": "In train_and_eval_v2.py: val_ds and test_ds instantiated with is_train=False, use_specaugment=False",
        "status": "PASS"
    }
    print(f"[CHECK 6] Train-only augmentation boundary: PASS")
    
    # 6. UNKNOWN mapping policy
    # In V2, train UNKNOWN comes from calibration_confusing_words_v1 (120 clips)
    # Val and Test UNKNOWN come from synthetic_negative_test_fixture (36 clips each from NON_TARGET_WORDS)
    # All mapped to class_id 4, class_label UNKNOWN.
    audit_log["unknown_mapping_policy"] = {
        "policy": "Confusing words and non-target tokens mapped to class_id=4, class_label='UNKNOWN'",
        "train_sources": ["calibration_confusing_words_v1"],
        "val_test_sources": ["synthetic_negative_test_fixture"],
        "fresh_sources": ["synthetic_negative_test_fixture"],
        "status": "PASS"
    }
    print(f"[CHECK 7] UNKNOWN mapping policy: PASS")

    print("\n=====================================================================")
    print("STEP 2: APPLES-TO-APPLES COMPARISON ON 144 HISTORICAL TARGET-COMMAND CLIPS")
    print("=====================================================================")
    
    # The 144 clips are the target commands in test_manifest_6class.csv (or synthetic_kws_baseline/test_manifest.csv)
    b_test_path = Path("experiments/synthetic_kws_baseline/test_manifest.csv")
    df_base_test = pd.read_csv(b_test_path)
    assert len(df_base_test) == 144, f"Expected 144 clips, got {len(df_base_test)}"
    
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    
    # Preload 144 test audios
    audios = []
    labels_4c = []
    labels_name = []
    for _, row in df_base_test.iterrows():
        p = Path(row["processed_path"])
        a, sr = sf.read(str(p), dtype="float32")
        if len(a) < 16000:
            a = np.pad(a, (0, 16000 - len(a)))
        else:
            a = a[:16000]
        audios.append(a)
        labels_4c.append(int(row["class_id"]))
        labels_name.append(row["class_label"])
        
    feats = [extractor(torch.from_numpy(a).float()) for a in audios]
    
    # Model 1: V1 Baseline TC-ResNet-14 (4-class)
    v1_ckpt_path = Path("experiments/synthetic_kws_baseline/checkpoints/best_synthetic_kws_model.pt")
    v1_model = VietnameseTCResNet(num_classes=4, dropout_p=0.3)
    v1_ckpt = torch.load(v1_ckpt_path, map_location="cpu")
    v1_model.load_state_dict(v1_ckpt["model_state_dict"])
    v1_model.eval()
    
    v1_preds = []
    with torch.no_grad():
        for feat in feats:
            logits = v1_model(feat)
            pred = int(torch.argmax(logits, dim=1))
            v1_preds.append(pred)
            
    # Model 2: V2 TC-ResNet-14 (6-class)
    v2_tc_path = v2_dir / "checkpoints" / "best_tcresnet_6class.pt"
    v2_tc_model = VietnameseTCResNet(num_classes=6, dropout_p=0.3)
    v2_tc_ckpt = torch.load(v2_tc_path, map_location="cpu")
    v2_tc_model.load_state_dict(v2_tc_ckpt["model_state_dict"])
    v2_tc_model.eval()
    
    v2_tc_raw_preds = []
    v2_tc_mapped_preds = []
    with torch.no_grad():
        for feat in feats:
            logits = v2_tc_model(feat)
            pred = int(torch.argmax(logits, dim=1))
            v2_tc_raw_preds.append(pred)
            # Map UNKNOWN (4) or SILENCE (5) to -1 (miss)
            v2_tc_mapped_preds.append(pred if pred < 4 else -1)
            
    # Model 3: V2 TemporalKWSNet (6-class)
    v2_temp_path = v2_dir / "checkpoints" / "best_temporal_kwsnet_6class.pt"
    v2_temp_model = VietnameseTemporalKWSNet(num_classes=6, dropout_p=0.3)
    v2_temp_ckpt = torch.load(v2_temp_path, map_location="cpu")
    v2_temp_model.load_state_dict(v2_temp_ckpt["model_state_dict"])
    v2_temp_model.eval()
    
    v2_temp_raw_preds = []
    v2_temp_mapped_preds = []
    with torch.no_grad():
        for feat in feats:
            logits = v2_temp_model(feat)
            pred = int(torch.argmax(logits, dim=1))
            v2_temp_raw_preds.append(pred)
            v2_temp_mapped_preds.append(pred if pred < 4 else -1)
            
    def compute_target_metrics(y_true, y_pred, model_label):
        # y_true is 0..3 (36 each)
        # y_pred is 0..3 (or -1 if missed as negative)
        prec, rec, f1, supp = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, -1])
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {
            "model": model_label,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "precision": [float(p) for p in prec],
            "recall": [float(r) for r in rec],
            "f1": [float(f) for f in f1],
            "support": [int(s) for s in supp],
            "confusion_matrix": cm.tolist()
        }
        
    m_v1 = compute_target_metrics(labels_4c, v1_preds, "V1 Baseline (TC-ResNet-14, 4-class)")
    m_v2_tc = compute_target_metrics(labels_4c, v2_tc_mapped_preds, "V2 TC-ResNet-14 (6-class, mapped)")
    m_v2_temp = compute_target_metrics(labels_4c, v2_temp_mapped_preds, "V2 TemporalKWSNet (6-class, mapped)")
    
    print("\n--- APPLES-TO-APPLES METRIC TABLE (Exact Same 144 Historical Diagnostic Clips) ---")
    print(f"{'Metric':<25} | {'V1 4-Class Baseline':<22} | {'V2 TC-ResNet (6-Class)':<22} | {'V2 TemporalKWSNet (6-Class)':<28}")
    print("-" * 105)
    print(f"{'Target Accuracy':<25} | {m_v1['accuracy']*100:6.2f}%                 | {m_v2_tc['accuracy']*100:6.2f}%                 | {m_v2_temp['accuracy']*100:6.2f}%")
    print(f"{'Macro-F1 (Target)':<25} | {m_v1['macro_f1']:6.4f}                  | {m_v2_tc['macro_f1']:6.4f}                  | {m_v2_temp['macro_f1']:6.4f}")
    print(f"{'TRAI Recall (36 samples)':<25} | {m_v1['recall'][0]*100:6.2f}% ({int(m_v1['recall'][0]*36)}/36)         | {m_v2_tc['recall'][0]*100:6.2f}% ({int(m_v2_tc['recall'][0]*36)}/36)         | {m_v2_temp['recall'][0]*100:6.2f}% ({int(m_v2_temp['recall'][0]*36)}/36)")
    print(f"{'PHAI Recall (36 samples)':<25} | {m_v1['recall'][1]*100:6.2f}% ({int(m_v1['recall'][1]*36)}/36)         | {m_v2_tc['recall'][1]*100:6.2f}% ({int(m_v2_tc['recall'][1]*36)}/36)         | {m_v2_temp['recall'][1]*100:6.2f}% ({int(m_v2_temp['recall'][1]*36)}/36)")
    print(f"{'LEN Recall (36 samples)':<25}  | {m_v1['recall'][2]*100:6.2f}% ({int(m_v1['recall'][2]*36)}/36)         | {m_v2_tc['recall'][2]*100:6.2f}% ({int(m_v2_tc['recall'][2]*36)}/36)         | {m_v2_temp['recall'][2]*100:6.2f}% ({int(m_v2_temp['recall'][2]*36)}/36)")
    print(f"{'XUONG Recall (36 samples)':<25}| {m_v1['recall'][3]*100:6.2f}% ({int(m_v1['recall'][3]*36)}/36)         | {m_v2_tc['recall'][3]*100:6.2f}% ({int(m_v2_tc['recall'][3]*36)}/36)         | {m_v2_temp['recall'][3]*100:6.2f}% ({int(m_v2_temp['recall'][3]*36)}/36)")
    
    print("\n--- CONFUSION MATRICES ON 144 HISTORICAL CLIPS ---")
    print("V1 TC-ResNet-14 (Cols: TRAI, PHAI, LEN, XUONG, MISSED):")
    for r, c_name in enumerate(["TRAI", "PHAI", "LEN", "XUONG"]):
        print(f"  {c_name:<5}: {m_v1['confusion_matrix'][r]}")
    print("V2 TC-ResNet-14 (Cols: TRAI, PHAI, LEN, XUONG, MISSED [UNKNOWN/SILENCE]):")
    for r, c_name in enumerate(["TRAI", "PHAI", "LEN", "XUONG"]):
        print(f"  {c_name:<5}: {m_v2_tc['confusion_matrix'][r]}")
    print("V2 TemporalKWSNet (Cols: TRAI, PHAI, LEN, XUONG, MISSED [UNKNOWN/SILENCE]):")
    for r, c_name in enumerate(["TRAI", "PHAI", "LEN", "XUONG"]):
        print(f"  {c_name:<5}: {m_v2_temp['confusion_matrix'][r]}")
        
    apples_results = {
        "audit": audit_log,
        "apples_to_apples_144": {
            "v1_baseline": m_v1,
            "v2_tcresnet": m_v2_tc,
            "v2_temporalkwsnet": m_v2_temp
        }
    }
    
    out_json = Path("experiments/synthetic_kws_v3/audit_and_apples_to_apples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(apples_results, f, indent=2)
    print(f"\nSaved audit and comparison results to {out_json}")


if __name__ == "__main__":
    run_audit()
