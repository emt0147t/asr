"""
Phase 21: Held-out Synthetic Speaker Profile Generator and Evaluator
Workspace: D:\\trainidentificationaudio

Generates an extended evaluation dataset of 288 synthetic clips across 24 newly sampled
held-out speaker profiles (disjoint from train/val/test), evaluates the frozen TC-ResNet-14
model without retraining, and records profile-level generalization metrics.
"""

import os
import sys
import io
import time
import json
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("experiments/synthetic_kws_v5_2_candidate").resolve()))
from experiments.synthetic_kws_v3.models_v3 import VietnameseTCResNet
from src.preprocessing.feature_extractor import LogMelFeatureExtractor
from config import (
    V5_2VoiceProfile,
    V5_2AcousticConfig,
    V5_2NoiseConfig,
    UNKNOWN_PILOT_VOCABULARY_V5_2,
    TARGET_WORDS_SPEC_V5_2
)
from generator import (
    generate_v5_2_speech_clip,
    generate_v5_2_silence_clip,
    SAMPLE_RATE,
    TOTAL_SAMPLES
)

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES_6)}

# 24 newly sampled held-out speaker profiles spanning diverse physical acoustics
# Disjoint from SPK_01..SPK_10 and SPK_TEST_01..02
HELDOUT_24_SPEAKER_PROFILES = [
    # High Female / Child Profiles (6)
    V5_2VoiceProfile("SPK_HELD_01_HIGH_FEMALE_1", 0.84, 240.0, 0.48, -9.5, 0.005),
    V5_2VoiceProfile("SPK_HELD_02_HIGH_FEMALE_2", 0.86, 235.0, 0.52, -10.0, 0.006),
    V5_2VoiceProfile("SPK_HELD_03_CHILD_VOICE",    0.82, 255.0, 0.44, -8.5, 0.004),
    V5_2VoiceProfile("SPK_HELD_04_FEMALE_BREATHY", 0.88, 225.0, 0.65, -13.5, 0.009),
    V5_2VoiceProfile("SPK_HELD_05_FEMALE_TENSE",   0.85, 230.0, 0.42, -7.5, 0.005),
    V5_2VoiceProfile("SPK_HELD_06_FEMALE_RESONANT",0.89, 218.0, 0.56, -11.0, 0.007),

    # Mid Female Profiles (6)
    V5_2VoiceProfile("SPK_HELD_07_MID_FEMALE_1",   0.92, 205.0, 0.54, -10.5, 0.006),
    V5_2VoiceProfile("SPK_HELD_08_MID_FEMALE_2",   0.95, 190.0, 0.58, -11.5, 0.007),
    V5_2VoiceProfile("SPK_HELD_09_MID_FEMALE_SOFT",0.93, 198.0, 0.62, -12.5, 0.008),
    V5_2VoiceProfile("SPK_HELD_10_MID_FEMALE_BRT", 0.91, 210.0, 0.49, -9.0, 0.005),
    V5_2VoiceProfile("SPK_HELD_11_FEMALE_NASAL",   0.94, 185.0, 0.57, -11.0, 0.007),
    V5_2VoiceProfile("SPK_HELD_12_FEMALE_DEEP",    0.97, 175.0, 0.60, -12.0, 0.008),

    # Mid Male Profiles (6)
    V5_2VoiceProfile("SPK_HELD_13_MID_MALE_1",     1.02, 145.0, 0.55, -11.0, 0.007),
    V5_2VoiceProfile("SPK_HELD_14_MID_MALE_2",     1.05, 138.0, 0.58, -12.0, 0.007),
    V5_2VoiceProfile("SPK_HELD_15_MALE_BREATHY",   1.03, 150.0, 0.66, -14.0, 0.010),
    V5_2VoiceProfile("SPK_HELD_16_MALE_TENSE",     1.01, 155.0, 0.44, -8.0, 0.005),
    V5_2VoiceProfile("SPK_HELD_17_MALE_GRAVELLY",  1.06, 130.0, 0.63, -12.5, 0.011),
    V5_2VoiceProfile("SPK_HELD_18_MALE_SMOOTH",    1.04, 142.0, 0.52, -10.0, 0.006),

    # Deep Male / Resonant Profiles (6)
    V5_2VoiceProfile("SPK_HELD_19_DEEP_MALE_1",    1.10, 115.0, 0.62, -13.0, 0.008),
    V5_2VoiceProfile("SPK_HELD_20_DEEP_MALE_2",    1.12, 108.0, 0.65, -13.5, 0.009),
    V5_2VoiceProfile("SPK_HELD_21_BASS_VOICE",     1.16, 98.0,  0.68, -14.5, 0.010),
    V5_2VoiceProfile("SPK_HELD_22_MALE_RESONANT",  1.08, 122.0, 0.56, -11.5, 0.007),
    V5_2VoiceProfile("SPK_HELD_23_DEEP_TIGHT",     1.11, 112.0, 0.46, -9.0, 0.006),
    V5_2VoiceProfile("SPK_HELD_24_EXTRA_LOW",      1.18, 92.0,  0.70, -15.0, 0.011),
]

def generate_heldout_dataset(output_dir: Path) -> Tuple[pd.DataFrame, Path]:
    audio_dir = output_dir / "heldout_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = output_dir / "phase21_heldout_manifest.csv"
    
    print(f"Generating extended held-out dataset across {len(HELDOUT_24_SPEAKER_PROFILES)} speaker profiles...")
    rng = np.random.RandomState(700000)
    
    records = []
    clip_idx = 0
    
    # 24 profiles x 6 classes x 2 clips = 288 clips
    for spk in HELDOUT_24_SPEAKER_PROFILES:
        # Generate 2 clips per class for this profile
        # 1. Target keywords: TRAI, PHAI, LEN, XUONG (2 each = 8 clips)
        for c_label in ["TRAI", "PHAI", "LEN", "XUONG"]:
            spec = TARGET_WORDS_SPEC_V5_2[c_label]
            word = spec["word"]
            subclass = "target_keyword"
            
            for rep in range(2):
                clip_idx += 1
                rec_id = f"p21_heldout_{spk.voice_id.lower()}_{c_label.lower()}_{rep+1:02d}"
                seed = int(710000 + clip_idx * 17)
                noise_seed = int(720000 + clip_idx * 23)
                dur = float(rng.uniform(0.38, 0.52))
                onset = float(rng.uniform(0.08, 0.20))
                speech_rms = float(rng.uniform(-16.5, -14.2))
                bg_rms = float(rng.uniform(-40.0, -32.0))
                noise_type = rng.choice(["room_tone", "hum", "dither"])
                
                cfg = V5_2AcousticConfig(
                    sample_id=rec_id,
                    class_label=c_label,
                    acoustic_subclass=subclass,
                    intended_text=word,
                    seed=seed,
                    voice_profile=spk,
                    duration_sec=dur,
                    onset_offset_sec=onset,
                    target_speech_rms_dbfs=speech_rms,
                    noise_type=noise_type,
                    noise_seed=noise_seed,
                    background_rms_dbfs=bg_rms
                )
                audio, gain, snr = generate_v5_2_speech_clip(cfg)
                
                wav_p = audio_dir / f"{rec_id}.wav"
                sf.write(str(wav_p), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
                sha = hashlib.sha256(wav_p.read_bytes()).hexdigest()
                
                records.append({
                    "recording_id": rec_id,
                    "class_label": c_label,
                    "acoustic_subclass": subclass,
                    "intended_text": word,
                    "simulated_speaker_id": spk.voice_id,
                    "vtl_scale": spk.vtl_scale,
                    "f0_base": spk.f0_base,
                    "open_quotient": spk.open_quotient,
                    "spectral_tilt_db": spk.spectral_tilt_db,
                    "jitter_pct": spk.jitter_pct,
                    "duration_sec": dur,
                    "onset_offset_sec": onset,
                    "target_speech_rms_dbfs": speech_rms,
                    "background_rms_dbfs": bg_rms,
                    "noise_type": noise_type,
                    "noise_seed": noise_seed,
                    "seed": seed,
                    "gain_factor": round(gain, 4),
                    "effective_snr_db": round(snr, 2),
                    "sha256": sha
                })

        # 2. UNKNOWN class (2 clips)
        for rep in range(2):
            clip_idx += 1
            vocab_entry = UNKNOWN_PILOT_VOCABULARY_V5_2[rng.randint(0, len(UNKNOWN_PILOT_VOCABULARY_V5_2))]
            word = vocab_entry[0]
            subclass = vocab_entry[2]
            
            rec_id = f"p21_heldout_{spk.voice_id.lower()}_unknown_{rep+1:02d}"
            seed = int(730000 + clip_idx * 17)
            noise_seed = int(740000 + clip_idx * 23)
            dur = float(rng.uniform(0.35, 0.55))
            onset = float(rng.uniform(0.08, 0.22))
            speech_rms = float(rng.uniform(-16.5, -14.2))
            bg_rms = float(rng.uniform(-40.0, -32.0))
            noise_type = rng.choice(["room_tone", "hum", "dither"])
            
            cfg = V5_2AcousticConfig(
                sample_id=rec_id,
                class_label="UNKNOWN",
                acoustic_subclass=subclass,
                intended_text=word,
                seed=seed,
                voice_profile=spk,
                duration_sec=dur,
                onset_offset_sec=onset,
                target_speech_rms_dbfs=speech_rms,
                noise_type=noise_type,
                noise_seed=noise_seed,
                background_rms_dbfs=bg_rms
            )
            audio, gain, snr = generate_v5_2_speech_clip(cfg)
            
            wav_p = audio_dir / f"{rec_id}.wav"
            sf.write(str(wav_p), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            sha = hashlib.sha256(wav_p.read_bytes()).hexdigest()
            
            records.append({
                "recording_id": rec_id,
                "class_label": "UNKNOWN",
                "acoustic_subclass": subclass,
                "intended_text": word,
                "simulated_speaker_id": spk.voice_id,
                "vtl_scale": spk.vtl_scale,
                "f0_base": spk.f0_base,
                "open_quotient": spk.open_quotient,
                "spectral_tilt_db": spk.spectral_tilt_db,
                "jitter_pct": spk.jitter_pct,
                "duration_sec": dur,
                "onset_offset_sec": onset,
                "target_speech_rms_dbfs": speech_rms,
                "background_rms_dbfs": bg_rms,
                "noise_type": noise_type,
                "noise_seed": noise_seed,
                "seed": seed,
                "gain_factor": round(gain, 4),
                "effective_snr_db": round(snr, 2),
                "sha256": sha
            })

        # 3. SILENCE class (2 clips)
        for rep in range(2):
            clip_idx += 1
            sil_subclass = rng.choice(["ambient_room_tone", "ambient_low_freq_hum", "ambient_digital_zero"])
            rec_id = f"p21_heldout_{spk.voice_id.lower()}_silence_{rep+1:02d}"
            seed = int(750000 + clip_idx * 17)
            
            cfg_sil = V5_2NoiseConfig(
                sample_id=rec_id,
                class_label="SILENCE",
                acoustic_subclass=sil_subclass,
                seed=seed
            )
            audio = generate_v5_2_silence_clip(cfg_sil)
            
            wav_p = audio_dir / f"{rec_id}.wav"
            sf.write(str(wav_p), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            sha = hashlib.sha256(wav_p.read_bytes()).hexdigest()
            
            records.append({
                "recording_id": rec_id,
                "class_label": "SILENCE",
                "acoustic_subclass": sil_subclass,
                "intended_text": "<silence>",
                "simulated_speaker_id": "NONE_AMBIENT",
                "vtl_scale": 1.0,
                "f0_base": 0.0,
                "open_quotient": 0.0,
                "spectral_tilt_db": 0.0,
                "jitter_pct": 0.0,
                "duration_sec": 1.0,
                "onset_offset_sec": 0.0,
                "target_speech_rms_dbfs": -99.0,
                "background_rms_dbfs": round(float(20.0 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)), 2),
                "noise_type": sil_subclass.replace("ambient_", ""),
                "noise_seed": seed,
                "seed": seed,
                "gain_factor": 1.0,
                "effective_snr_db": -99.0,
                "sha256": sha
            })

    df = pd.DataFrame(records)
    df.to_csv(manifest_csv, index=False)
    print(f"Generated {len(df)} held-out clips across {len(HELDOUT_24_SPEAKER_PROFILES)} profiles.")
    print(f"Manifest written to: {manifest_csv}")
    return df, manifest_csv

def evaluate_heldout_dataset(manifest_df: pd.DataFrame, base_dir: Path, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    print(f"\nEvaluating frozen TC-ResNet-14 checkpoint on 288 held-out clips...")
    model = VietnameseTCResNet(num_classes=6, dropout_p=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    fe = LogMelFeatureExtractor().to(device)
    audio_dir = base_dir / "heldout_audio"
    
    y_true, y_pred, y_probs = [], [], []
    records = []
    
    with torch.no_grad():
        for _, row in manifest_df.iterrows():
            wav_p = audio_dir / f"{row['recording_id']}.wav"
            data, sr = sf.read(str(wav_p), dtype="float32")
            if len(data) < 16000: data = np.pad(data, (0, 16000 - len(data)))
            else: data = data[:16000]
            
            wave_t = torch.from_numpy(data).unsqueeze(0).to(device)
            feat = fe(wave_t)
            logits = model(feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            true_idx = CLASS_TO_IDX[row["class_label"]]
            
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs)
            
            records.append({
                "recording_id": row["recording_id"],
                "speaker_id": row["simulated_speaker_id"],
                "class_label": row["class_label"],
                "pred_label": CLASS_NAMES_6[pred_idx],
                "confidence": float(probs[pred_idx]),
                "correct": (pred_idx == true_idx)
            })
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    
    # Per-class metrics
    per_class = {}
    p_c, r_c, f_c, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for c_idx, c_name in enumerate(CLASS_NAMES_6):
        per_class[c_name] = {
            "precision": float(p_c[c_idx]),
            "recall": float(r_c[c_idx]),
            "f1": float(f_c[c_idx]),
            "support": int(np.sum(y_true == c_idx))
        }
        
    # Per-speaker profile accuracy
    rec_df = pd.DataFrame(records)
    spk_acc = {}
    for spk_id, grp in rec_df.groupby("speaker_id"):
        spk_acc[spk_id] = {
            "accuracy": float(grp["correct"].mean()),
            "total_clips": len(grp),
            "correct_clips": int(grp["correct"].sum())
        }
        
    # Wilson score interval 95%
    n = len(y_true)
    z = 1.95996
    k = int(np.sum(y_true == y_pred))
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * (n**2))))) / denom
    wilson_ci = [round(float(max(0.0, center - margin)), 4), round(float(min(1.0, center + margin)), 4)]
    
    res = {
        "total_clips_evaluated": n,
        "correct_predictions": k,
        "overall_accuracy": round(acc, 4),
        "macro_f1": round(float(f1), 4),
        "wilson_score_95_ci": wilson_ci,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "per_speaker_profile_metrics": spk_acc,
        "trai_phai_confusion": {
            "trai_as_phai": int(cm[0, 1]),
            "phai_as_trai": int(cm[1, 0])
        },
        "false_alarm_unknown": int(np.sum(cm[4, :4])),
        "false_alarm_silence": int(np.sum(cm[5, :4]))
    }
    
    print(f"Overall Held-out Accuracy: {acc*100:.2f}% ({k}/{n})")
    print(f"Macro F1: {f1:.4f}")
    print(f"Wilson 95% CI: [{wilson_ci[0]*100:.2f}%, {wilson_ci[1]*100:.2f}%]")
    print(f"TRAI vs PHAI Confusion: TRAI->PHAI: {cm[0, 1]}, PHAI->TRAI: {cm[1, 0]}")
    print(f"Non-target False Alarms: UNKNOWN->Target: {res['false_alarm_unknown']}, SILENCE->Target: {res['false_alarm_silence']}")
    return res

if __name__ == "__main__":
    out_dir = Path("experiments/synthetic_kws_phase20_22/phase21_robustness")
    ckpt_path = Path("experiments/synthetic_kws_phase13_3_16/model_evaluation/checkpoints/v5_2_tcresnet14_seed_42.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df, manifest_p = generate_heldout_dataset(out_dir)
    res = evaluate_heldout_dataset(df, out_dir, ckpt_path, device)
    
    eval_json = out_dir / "phase21_heldout_eval.json"
    with open(eval_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved held-out evaluation results to {eval_json}")
