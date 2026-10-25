"""
Audit and Recomputation of Diagnostic Probes
Workspace: D:\\trainidentificationaudio

Recomputes without test-guided feature selection:
1. Six-class accuracy for each diagnostic (metadata, strict nuisance, spectral/acoustic)
2. Four-target-only accuracy for TRAI, PHAI, LEN, XUONG
3. Target-present versus target-absent binary metric
4. Relevant class counts, confusion matrices, and the proper chance-level reference for each task
"""

import sys
import json
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

rc_dir = Path("experiments/synthetic_kws_phase13_3_16/dataset_release_candidate")
train_df = pd.read_csv(rc_dir / "splits" / "train.csv")
val_df = pd.read_csv(rc_dir / "splits" / "val.csv")
test_df = pd.read_csv(rc_dir / "splits" / "test.csv")

def extract_features(split_df):
    records = []
    for _, r in split_df.iterrows():
        wav_path = rc_dir / "audio" / f"{r['recording_id']}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        full_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
        peak = float(np.max(np.abs(audio))) + 1e-9
        crest = float(20.0 * np.log10(peak / full_rms))
        f25_samples = int(0.025 * sr)
        f25_rms = float(np.sqrt(np.mean(audio[:f25_samples] ** 2))) + 1e-9
        onset_samples = max(int(float(r['onset_offset_sec']) * sr), f25_samples)
        pre_onset_rms = float(np.sqrt(np.mean(audio[:onset_samples] ** 2))) + 1e-9
        
        fft_vals = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        sum_mag = np.sum(fft_vals) + 1e-9
        spectral_centroid = float(np.sum(freqs * fft_vals) / sum_mag)
        hf_energy = np.sum(fft_vals[freqs >= 2000.0] ** 2)
        tot_energy = np.sum(fft_vals ** 2) + 1e-9
        hf_energy_ratio = float(hf_energy / tot_energy)
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
        
        records.append({
            "recording_id": r["recording_id"],
            "class_label": r["class_label"],
            "split": r["split"],
            "is_target": r["class_label"] in ["TRAI", "PHAI", "LEN", "XUONG"],
            "meta_duration_sec": float(r["duration_sec"]),
            "meta_onset_offset_sec": float(r["onset_offset_sec"]),
            "meta_target_speech_rms_dbfs": float(r["target_speech_rms_dbfs"]),
            "meta_background_rms_dbfs": float(r["background_rms_dbfs"]),
            "meta_gain_factor": float(r["gain_factor"]),
            "full_clip_rms": full_rms,
            "peak_amplitude": peak,
            "crest_factor_db": crest,
            "first_25ms_rms": f25_rms,
            "pre_onset_rms": pre_onset_rms,
            "spectral_centroid_hz": spectral_centroid,
            "hf_energy_ratio": hf_energy_ratio,
            "zcr": zcr
        })
    return pd.DataFrame(records)

def evaluate_nuisance_probes():
    print("Extracting features for train (2100), val (450), test (450)...")
    tr = extract_features(train_df)
    va = extract_features(val_df)
    te = extract_features(test_df)

    feature_sets = {
        "metadata_only": ["meta_duration_sec", "meta_onset_offset_sec", "meta_target_speech_rms_dbfs", "meta_background_rms_dbfs", "meta_gain_factor"],
        "strict_nuisance": ["full_clip_rms", "peak_amplitude", "crest_factor_db", "first_25ms_rms", "pre_onset_rms"],
        "spectral_phonetic": ["spectral_centroid_hz", "hf_energy_ratio", "zcr"]
    }

    results = {}

    # Task 1: 6-Class Classification (TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE)
    print("\n=======================================================")
    print("TASK 1: 6-Class Classification (Chance = 16.67%)")
    print("=======================================================")
    task1_res = {}
    classes_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
    for name, feats in feature_sets.items():
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(tr[feats], tr["class_label"])
        val_acc = float(pipe.score(va[feats], va["class_label"]))
        test_acc = float(pipe.score(te[feats], te["class_label"]))
        preds = pipe.predict(te[feats])
        cm = confusion_matrix(te["class_label"], preds, labels=classes_6)
        p, r, f1, _ = precision_recall_fscore_support(te["class_label"], preds, average="macro", zero_division=0)
        
        task1_res[name] = {
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm.tolist()
        }
        print(f"\n{name.upper()}: Val Acc = {val_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}%, F1 = {f1:.4f}")
        print("Confusion Matrix (rows=true, cols=pred: TRAI, PHAI, LEN, XUONG, UNKNOWN, SILENCE):")
        print(cm)
    results["task1_six_class"] = task1_res

    # Task 2: 4-Target Only Classification (TRAI, PHAI, LEN, XUONG)
    print("\n=======================================================")
    print("TASK 2: 4-Target Keyword Classification (Chance = 25.00%)")
    print("=======================================================")
    task2_res = {}
    classes_4 = ["TRAI", "PHAI", "LEN", "XUONG"]
    tr_4 = tr[tr["is_target"]].copy()
    va_4 = va[va["is_target"]].copy()
    te_4 = te[te["is_target"]].copy()
    
    print(f"Target sample counts: Train={len(tr_4)} (350/class), Val={len(va_4)} (75/class), Test={len(te_4)} (75/class)")
    
    for name, feats in feature_sets.items():
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(tr_4[feats], tr_4["class_label"])
        val_acc = float(pipe.score(va_4[feats], va_4["class_label"]))
        test_acc = float(pipe.score(te_4[feats], te_4["class_label"]))
        preds = pipe.predict(te_4[feats])
        cm = confusion_matrix(te_4["class_label"], preds, labels=classes_4)
        p, r, f1, _ = precision_recall_fscore_support(te_4["class_label"], preds, average="macro", zero_division=0)
        
        task2_res[name] = {
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm.tolist()
        }
        print(f"\n{name.upper()}: Val Acc = {val_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}%, F1 = {f1:.4f}")
        print("Confusion Matrix (rows=true, cols=pred: TRAI, PHAI, LEN, XUONG):")
        print(cm)
    results["task2_four_target_only"] = task2_res

    # Task 3: Target-Present vs Target-Absent Binary Classification
    print("\n=======================================================")
    print("TASK 3: Target-Present vs Target-Absent Binary Classification")
    print("=======================================================")
    # Target-Present (TRAI, PHAI, LEN, XUONG) = True (300 in test)
    # Target-Absent (UNKNOWN, SILENCE) = False (150 in test)
    # Balanced prior accuracy baseline = 50.00%, Majority class baseline = 66.67%
    task3_res = {}
    for name, feats in feature_sets.items():
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        pipe.fit(tr[feats], tr["is_target"])
        val_acc = float(pipe.score(va[feats], va["is_target"]))
        test_acc = float(pipe.score(te[feats], te["is_target"]))
        preds = pipe.predict(te[feats])
        cm = confusion_matrix(te["is_target"], preds, labels=[False, True])
        p, r, f1, _ = precision_recall_fscore_support(te["is_target"], preds, average="binary", zero_division=0)
        
        task3_res[name] = {
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": cm.tolist()
        }
        print(f"\n{name.upper()}: Val Acc = {val_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}%, Prec = {p:.4f}, Rec = {r:.4f}, F1 = {f1:.4f}")
        print("Confusion Matrix (rows=true, cols=pred: Non-Target [150], Target [300]):")
        print(cm)
    results["task3_target_present_binary"] = task3_res

    out_json = Path("experiments/synthetic_kws_phase20_22/phase20_audit/nuisance_probe_audit.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive nuisance probe audit to: {out_json}")

if __name__ == "__main__":
    evaluate_nuisance_probes()
