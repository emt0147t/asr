"""
Controlled 2x2 Alignment and Onset Diagnostic for Synthetic Vietnamese KWS V4.
Separates acoustic onset parameterization (Factor A) from temporal alignment (Factor B).

Diagnostic Conditions:
  Condition A1-B2: Baseline Generator, Center-Aligned (Native historical test clips)
  Condition A1-B1: Baseline Generator, Left-Aligned (Active speech shifted to sample 0)
  Condition A2-B1: Onset-Aware Generator, Left-Aligned (Native V4 fresh holdout clips)
  Condition A2-B2: Onset-Aware Generator, Center-Aligned (Active speech shifted to center)

Evaluates all 4 frozen checkpoints:
  - Arm A (Control)
  - Arm B (Onset-Aware, No Resample)
  - Arm C (Onset-Aware + Resample)
  - V3 Champion TC-ResNet-14

Outputs:
  - Deterministically transformed WAV files in audit/diagnostic_audio/
  - Versioned diagnostic manifests in audit/
  - Complete confusion matrices and acoustic measurements in audit/alignment_diagnostic_results.json
"""

import sys
import os
import json
import hashlib
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.fft import rfft, rfftfreq

from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
TARGET_IDS = {0, 1, 2, 3}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def measure_audio_acoustics(audio: np.ndarray, sr: int = 16000) -> dict:
    """Measures physical waveform properties consistently."""
    abs_a = np.abs(audio)
    act = np.where(abs_a >= 0.005)[0]
    
    first_act = int(act[0]) if len(act) > 0 else -1
    last_act = int(act[-1]) if len(act) > 0 else -1
    act_dur_sec = ((last_act - first_act + 1) / sr) if (first_act >= 0 and last_act >= first_act) else 0.0
    
    # Samples 0 to 399 (0.0 to 25.0 ms)
    c_samples = audio[:400]
    c_rms = float(np.sqrt(np.mean(c_samples**2)))
    c_max = float(np.max(np.abs(c_samples)))
    c_frac_under_005 = float(np.mean(np.abs(c_samples) < 0.005))
    
    # Peak amplitude of entire clip
    peak_amp = float(np.max(abs_a))
    rms_total = float(np.sqrt(np.mean(audio**2)))
    
    # HF energy ratio (>3.5 kHz) in first 50ms of active speech
    if first_act >= 0:
        win_len = min(800, len(audio) - first_act)
        speech_win = audio[first_act : first_act + win_len]
    else:
        speech_win = audio[:800]
        
    spec = np.abs(rfft(speech_win * np.hanning(len(speech_win))))
    freqs = rfftfreq(len(speech_win), 1.0 / sr)
    hf_energy = float(np.sum(spec[freqs >= 3500.0]**2))
    tot_energy = float(np.sum(spec**2) + 1e-9)
    hf_ratio = hf_energy / tot_energy
    
    return {
        "onset_sample_idx": first_act,
        "onset_time_ms": (first_act / sr * 1000.0) if first_act >= 0 else -1.0,
        "active_duration_sec": act_dur_sec,
        "samples_0_to_399_rms": c_rms,
        "samples_0_to_399_max_abs": c_max,
        "samples_0_to_399_frac_under_005": c_frac_under_005,
        "peak_amplitude": peak_amp,
        "rms_total": rms_total,
        "hf_ratio_active_50ms": hf_ratio
    }


def prepare_diagnostic_datasets():
    audit_dir = Path("experiments/synthetic_kws_v4/audit")
    diag_audio_dir = audit_dir / "diagnostic_audio"
    diag_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Source 1: Baseline Historical 144 Target Clips
    hist_man_path = Path("experiments/synthetic_kws_baseline/test_manifest.csv")
    df_hist = pd.read_csv(hist_man_path)
    
    # Source 2: Onset-Aware V4 Fresh Holdout 300 Clips
    v4_man_path = Path("experiments/synthetic_kws_v4/fresh_holdout_manifest_6class.csv")
    df_v4 = pd.read_csv(v4_man_path)
    
    conditions = {}
    
    # -------------------------------------------------------------
    # Condition A1-B2: Baseline Generator, Center-Aligned (Native)
    # -------------------------------------------------------------
    c_a1_b2_records = []
    for idx, r in df_hist.iterrows():
        src_p = Path(r["processed_path"])
        a, sr = sf.read(str(src_p), dtype="float32")
        c_a1_b2_records.append({
            "condition_id": "A1-B2",
            "condition_name": "Baseline Onset, Center-Aligned (Native)",
            "recording_id": r["recording_id"],
            "speaker_id": r["speaker_id"],
            "class_id": int(r["class_id"]),
            "class_label": r["class_label"],
            "processed_path": str(src_p).replace("\\", "/"),
            "sha256": sha256_file(src_p),
            **measure_audio_acoustics(a, sr)
        })
    df_a1_b2 = pd.DataFrame(c_a1_b2_records)
    man_a1_b2 = audit_dir / "manifest_diag_A1_B2_baseline_centered.csv"
    df_a1_b2.to_csv(man_a1_b2, index=False)
    conditions["A1-B2"] = (man_a1_b2, df_a1_b2)
    
    # -------------------------------------------------------------
    # Condition A1-B1: Baseline Generator, Left-Aligned (Transformed)
    # Trim leading silence to place speech onset at sample 0
    # -------------------------------------------------------------
    dir_a1_b1 = diag_audio_dir / "A1_B1_baseline_left"
    dir_a1_b1.mkdir(parents=True, exist_ok=True)
    c_a1_b1_records = []
    for idx, r in df_hist.iterrows():
        src_p = Path(r["processed_path"])
        a, sr = sf.read(str(src_p), dtype="float32")
        
        act = np.where(np.abs(a) >= 0.005)[0]
        first_idx = act[0] if len(act) > 0 else 0
        speech = a[first_idx:]
        # Left align: speech starts at 0, right-padded with zeros to 16,000
        a_left = np.pad(speech, (0, max(0, 16000 - len(speech))))[:16000]
        
        out_p = dir_a1_b1 / f"left_{src_p.name}"
        sf.write(str(out_p), a_left, sr, subtype="PCM_16")
        
        c_a1_b1_records.append({
            "condition_id": "A1-B1",
            "condition_name": "Baseline Onset, Left-Aligned (Shifted)",
            "recording_id": f"left_{r['recording_id']}",
            "speaker_id": r["speaker_id"],
            "class_id": int(r["class_id"]),
            "class_label": r["class_label"],
            "processed_path": str(out_p).replace("\\", "/"),
            "sha256": sha256_file(out_p),
            **measure_audio_acoustics(a_left, sr)
        })
    df_a1_b1 = pd.DataFrame(c_a1_b1_records)
    man_a1_b1 = audit_dir / "manifest_diag_A1_B1_baseline_left.csv"
    df_a1_b1.to_csv(man_a1_b1, index=False)
    conditions["A1-B1"] = (man_a1_b1, df_a1_b1)
    
    # -------------------------------------------------------------
    # Condition A2-B1: Onset-Aware Generator, Left-Aligned (Native)
    # -------------------------------------------------------------
    c_a2_b1_records = []
    for idx, r in df_v4.iterrows():
        src_p = Path(r["processed_path"])
        a, sr = sf.read(str(src_p), dtype="float32")
        c_a2_b1_records.append({
            "condition_id": "A2-B1",
            "condition_name": "Onset-Aware, Left-Aligned (Native)",
            "recording_id": r["recording_id"],
            "speaker_id": r["speaker_id"],
            "class_id": int(r["class_id"]),
            "class_label": r["class_label"],
            "processed_path": str(src_p).replace("\\", "/"),
            "sha256": sha256_file(src_p),
            **measure_audio_acoustics(a, sr)
        })
    df_a2_b1 = pd.DataFrame(c_a2_b1_records)
    man_a2_b1 = audit_dir / "manifest_diag_A2_B1_onset_left.csv"
    df_a2_b1.to_csv(man_a2_b1, index=False)
    conditions["A2-B1"] = (man_a2_b1, df_a2_b1)
    
    # -------------------------------------------------------------
    # Condition A2-B2: Onset-Aware Generator, Center-Aligned (Transformed)
    # Center the active speech window within the 1-second buffer
    # -------------------------------------------------------------
    dir_a2_b2 = diag_audio_dir / "A2_B2_onset_centered"
    dir_a2_b2.mkdir(parents=True, exist_ok=True)
    c_a2_b2_records = []
    for idx, r in df_v4.iterrows():
        src_p = Path(r["processed_path"])
        a, sr = sf.read(str(src_p), dtype="float32")
        
        # In V4, audio was right-padded
        act = np.where(np.abs(a) >= 0.005)[0]
        last_idx = act[-1] if len(act) > 0 else 8000
        speech = a[:last_idx + 1] # Includes the initial closure gap
        
        pad_tot = 16000 - len(speech)
        pad_l = pad_tot // 2
        pad_r = pad_tot - pad_l
        a_center = np.pad(speech, (pad_l, pad_r))[:16000]
        
        out_p = dir_a2_b2 / f"center_{src_p.name}"
        sf.write(str(out_p), a_center, sr, subtype="PCM_16")
        
        c_a2_b2_records.append({
            "condition_id": "A2-B2",
            "condition_name": "Onset-Aware, Center-Aligned (Shifted)",
            "recording_id": f"center_{r['recording_id']}",
            "speaker_id": r["speaker_id"],
            "class_id": int(r["class_id"]),
            "class_label": r["class_label"],
            "processed_path": str(out_p).replace("\\", "/"),
            "sha256": sha256_file(out_p),
            **measure_audio_acoustics(a_center, sr)
        })
    df_a2_b2 = pd.DataFrame(c_a2_b2_records)
    man_a2_b2 = audit_dir / "manifest_diag_A2_B2_onset_centered.csv"
    df_a2_b2.to_csv(man_a2_b2, index=False)
    conditions["A2-B2"] = (man_a2_b2, df_a2_b2)
    
    return conditions


def evaluate_diagnostic():
    conditions = prepare_diagnostic_datasets()
    
    device = torch.device("cpu")
    extractor = LogMelFeatureExtractor(sample_rate=16000, n_mels=40)
    
    checkpoints = {
        "Arm A (Control)": Path("experiments/synthetic_kws_v4/checkpoints/arm_a_control.pt"),
        "Arm B (Onset-Aware, No Resample)": Path("experiments/synthetic_kws_v4/checkpoints/arm_b_onset_no_resample.pt"),
        "Arm C (Onset-Aware + Resample)": Path("experiments/synthetic_kws_v4/checkpoints/arm_c_onset_with_resample.pt"),
        "V3 Champion TC-ResNet-14": Path("experiments/synthetic_kws_v3/checkpoints/best_tcresnet14_v3.pt")
    }
    
    results = {}
    
    for arm_name, ckpt_p in checkpoints.items():
        print(f"\nEvaluating Diagnostic for: {arm_name}")
        model = VietnameseTCResNet(num_classes=6)
        ckpt_dict = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(ckpt_dict["model_state_dict"])
        model.eval()
        
        arm_res = {
            "checkpoint_path": str(ckpt_p).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(ckpt_p),
            "conditions": {}
        }
        
        for cond_id in ["A1-B2", "A1-B1", "A2-B1", "A2-B2"]:
            man_path, df = conditions[cond_id]
            cm = np.zeros((6, 6), dtype=int)
            confidences = []
            pred_records = []
            
            for idx, r in df.iterrows():
                true_cid = int(r["class_id"])
                wav_p = Path(r["processed_path"])
                audio, sr = sf.read(str(wav_p), dtype="float32")
                inp = torch.from_numpy(audio[:16000]).float()
                
                with torch.no_grad():
                    log_mel = extractor(inp)
                    logits = model(log_mel)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_cid = int(np.argmax(probs))
                    conf = float(probs[pred_cid])
                    
                cm[true_cid, pred_cid] += 1
                confidences.append(conf)
                pred_records.append({
                    "recording_id": r["recording_id"],
                    "true_class_id": true_cid,
                    "pred_class_id": pred_cid,
                    "confidence": conf,
                    "prob_trai": float(probs[0]),
                    "prob_phai": float(probs[1])
                })
                
            row_sums = cm.sum(axis=1)
            recalls = [(cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0 for i in range(6)]
            precisions = [(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() > 0 else 0.0 for i in range(6)]
            f1s = [
                (2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))
                if (precisions[i] + recalls[i]) > 0 else 0.0
                for i in range(6)
            ]
            
            total_samples = int(row_sums.sum())
            acc = float(np.trace(cm) / total_samples)
            macro_f1 = float(np.mean(f1s))
            
            # Acoustic summary for this condition
            acoustics = {
                "onset_sample_idx_mean": float(df["onset_sample_idx"].mean()),
                "onset_time_ms_mean": float(df["onset_time_ms"].mean()),
                "active_duration_sec_mean": float(df["active_duration_sec"].mean()),
                "samples_0_to_399_rms_mean": float(df["samples_0_to_399_rms"].mean()),
                "samples_0_to_399_max_abs_mean": float(df["samples_0_to_399_max_abs"].mean()),
                "samples_0_to_399_frac_under_005_mean": float(df["samples_0_to_399_frac_under_005"].mean()),
                "peak_amplitude_mean": float(df["peak_amplitude"].mean()),
                "hf_ratio_active_50ms_mean": float(df["hf_ratio_active_50ms"].mean())
            }
            
            cond_summary = {
                "condition_id": cond_id,
                "condition_name": df["condition_name"].iloc[0],
                "manifest_path": str(man_path).replace("\\", "/"),
                "manifest_sha256": sha256_file(man_path),
                "samples_count": total_samples,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "trai_recall": float(recalls[0]),
                "phai_recall": float(recalls[1]),
                "len_recall": float(recalls[2]),
                "xuong_recall": float(recalls[3]),
                "unknown_recall": float(recalls[4]),
                "silence_recall": float(recalls[5]),
                "confusion_matrix": cm.tolist(),
                "support": [int(s) for s in row_sums],
                "mean_confidence": float(np.mean(confidences)),
                "acoustics": acoustics
            }
            arm_res["conditions"][cond_id] = cond_summary
            
            print(f"  [{cond_id}] Acc: {acc*100:5.2f}% | TRAI Rec: {recalls[0]*100:5.1f}% | PHAI Rec: {recalls[1]*100:5.1f}% | Mean Conf: {np.mean(confidences):.3f}")
            
        results[arm_name] = arm_res
        
    out_json = Path("experiments/synthetic_kws_v4/audit/alignment_diagnostic_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved 2x2 alignment diagnostic results to: {out_json}")


if __name__ == "__main__":
    evaluate_diagnostic()
