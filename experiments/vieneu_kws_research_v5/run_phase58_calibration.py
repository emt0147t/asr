import os
import sys
import json
import csv
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"D:\trainidentificationaudio"
sys.path.insert(0, BASE_DIR)
EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v5")
DATA_DIR = os.path.join(EXP_DIR, "data", "splits")
MODELS_DIR = os.path.join(EXP_DIR, "models")
CALIB_DIR = os.path.join(EXP_DIR, "calibration")
PRED_DIR = os.path.join(EXP_DIR, "predictions")

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v5.models.train_ablation_campaign import (
    VietnameseBoundedKWSNet,
    LogMelFrontend
)

TARGET_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG"]
BOUNDED_CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]

def load_audio_clip(wav_path, target_length=16000, snr_db=None, gain_db=None, seed=42):
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(BASE_DIR, wav_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}")

    if len(audio) < target_length:
        pad = target_length - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    elif len(audio) > target_length:
        audio = audio[:target_length]

    # Apply controlled robustness conditions if requested
    if gain_db is not None:
        gain = 10.0 ** (gain_db / 20.0)
        audio = audio * gain

    if snr_db is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 1.0, len(audio)).astype(np.float32)
        sig_rms = np.sqrt(np.mean(audio**2) + 1e-9)
        noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
        audio = audio + noise * noise_rms

    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def load_candidate_models(variant, seed, device):
    if variant == "variant_a_cascade_v1":
        s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{seed}.pt")
        s2_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1", "checkpoints", f"best_verifier_seed{seed}.pt")
        m1 = VietnameseKWSNet(num_classes=6).to(device)
        m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
        m1.eval()

        m2 = TemporalKWSVerifier(num_classes=5).to(device)
        m2.load_state_dict(torch.load(s2_ckpt, map_location=device)["model_state_dict"])
        m2.eval()
        return m1, m2

    elif variant == "variant_b_stage1_cnn":
        s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{seed}.pt")
        m1 = VietnameseKWSNet(num_classes=6).to(device)
        m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
        m1.eval()
        return m1, None

    elif variant == "variant_c_standard":
        c_ckpt = os.path.join(MODELS_DIR, "candidate_c_standard", "checkpoints", f"model_seed{seed}.pt")
        m = VietnameseBoundedKWSNet(num_classes=5).to(device)
        m.load_state_dict(torch.load(c_ckpt, map_location=device)["model_state_dict"])
        m.eval()
        return m, None

    elif variant == "variant_d_ablation":
        d_ckpt = os.path.join(MODELS_DIR, "candidate_d_ablation", "checkpoints", f"model_seed{seed}.pt")
        m = VietnameseBoundedKWSNet(num_classes=5).to(device)
        m.load_state_dict(torch.load(d_ckpt, map_location=device)["model_state_dict"])
        m.eval()
        return m, None
    else:
        raise ValueError(f"Unknown variant {variant}")


def evaluate_val_scores(variant, seed, val_rows, device, frontend, snr_db=None, gain_db=None):
    m1, m2 = load_candidate_models(variant, seed, device)
    raw_results = []

    for idx, r in enumerate(val_rows):
        audio = load_audio_clip(r["kws_wav_path"], snr_db=snr_db, gain_db=gain_db, seed=42 + idx)
        t_audio = torch.from_numpy(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            spec = frontend(t_audio)

            if variant == "variant_a_cascade_v1":
                p1 = F.softmax(m1(spec), dim=1).cpu().numpy()[0]
                top_cmd_idx = int(np.argmax(p1[:4]))
                s1_top = float(p1[:4].max())

                crop = np.pad(audio, (0, 3200), mode="constant")
                crop_t = torch.from_numpy(crop).unsqueeze(0).to(device)
                crop_spec = frontend(crop_t)
                p2 = F.softmax(m2(crop_spec), dim=1).cpu().numpy()[0]
                raw_results.append({
                    "sample_id": r["sample_id"],
                    "true_label": r["bounded_label"],
                    "top_cmd_idx": top_cmd_idx,
                    "s1_top": s1_top,
                    "s1_p_unk": float(p1[4]),
                    "s1_p_sil": float(p1[5]),
                    "s2_top": float(p2[top_cmd_idx]),
                    "s2_rej": float(p2[4])
                })

            elif variant == "variant_b_stage1_cnn":
                p = F.softmax(m1(spec), dim=1).cpu().numpy()[0]
                top_cmd_idx = int(np.argmax(p[:4]))
                top_cmd_prob = float(p[:4].max())
                raw_results.append({
                    "sample_id": r["sample_id"],
                    "true_label": r["bounded_label"],
                    "top_cmd_idx": top_cmd_idx,
                    "top_cmd_prob": top_cmd_prob,
                    "p_unk": float(p[4]),
                    "p_sil": float(p[5])
                })

            elif variant in ["variant_c_standard", "variant_d_ablation"]:
                p = F.softmax(m1(spec), dim=1).cpu().numpy()[0]
                top_cmd_idx = int(np.argmax(p[:4]))
                top_cmd_prob = float(p[:4].max())
                raw_results.append({
                    "sample_id": r["sample_id"],
                    "true_label": r["bounded_label"],
                    "top_cmd_idx": top_cmd_idx,
                    "top_cmd_prob": top_cmd_prob,
                    "rej_prob": float(p[4])
                })

    return raw_results


def score_variant_preds(variant, raw_results, params):
    preds = []
    for r in raw_results:
        true_lbl = r["true_label"]
        pred_lbl = "REJECT"

        if variant == "variant_a_cascade_v1":
            th1 = params["theta_s1"]
            th2 = params["theta_s2"]
            if r["s1_top"] >= th1 and r["s1_top"] > r["s1_p_unk"] and r["s1_top"] > r["s1_p_sil"]:
                if r["s2_top"] >= th2 and r["s2_top"] > r["s2_rej"]:
                    pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        elif variant == "variant_b_stage1_cnn":
            th = params["theta_cmd"]
            if r["top_cmd_prob"] >= th and r["top_cmd_prob"] > r["p_unk"] and r["top_cmd_prob"] > r["p_sil"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        elif variant in ["variant_c_standard", "variant_d_ablation"]:
            th = params["theta_reject"]
            if r["top_cmd_prob"] >= th and r["top_cmd_prob"] > r["rej_prob"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        preds.append({"true_label": true_lbl, "pred_label": pred_lbl})

    # Metrics
    cmd_hits = 0
    cmd_total = 0
    kw_hits = {kw: 0 for kw in TARGET_CLASSES}
    kw_total = {kw: 0 for kw in TARGET_CLASSES}

    nt_rejects = 0
    nt_total = 0

    for p in preds:
        t = p["true_label"]
        pr = p["pred_label"]
        if t in TARGET_CLASSES:
            cmd_total += 1
            kw_total[t] += 1
            if pr == t:
                cmd_hits += 1
                kw_hits[t] += 1
        elif t == "REJECT":
            nt_total += 1
            if pr == "REJECT":
                nt_rejects += 1

    macro_recall = cmd_hits / cmd_total if cmd_total > 0 else 0.0
    far = (nt_total - nt_rejects) / nt_total if nt_total > 0 else 0.0
    tnr = nt_rejects / nt_total if nt_total > 0 else 0.0
    per_kw = {kw: (kw_hits[kw] / kw_total[kw] if kw_total[kw] > 0 else 0.0) for kw in TARGET_CLASSES}
    all_kws_pass = all(per_kw[kw] >= 0.95 for kw in TARGET_CLASSES)
    meets_strict_criteria = (all_kws_pass and far <= 0.05)

    return {
        "macro_recall": round(macro_recall, 4),
        "false_rejection_rate": round(1.0 - macro_recall, 4),
        "false_acceptance_rate": round(far, 4),
        "rejection_rate": round(tnr, 4),
        "per_keyword_recall": {k: round(v, 4) for k, v in per_kw.items()},
        "all_keywords_ge_95": all_kws_pass,
        "meets_strict_criteria": meets_strict_criteria
    }


def run_calibration_and_robustness():
    print("=" * 80)
    print("Phase 58: Validation Calibration & Robustness Evaluation")
    print("=" * 80)
    os.makedirs(CALIB_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    val_manifest = os.path.join(DATA_DIR, "val_manifest.csv")
    val_rows = []
    with open(val_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val_rows.append(r)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)

    variants = ["variant_a_cascade_v1", "variant_b_stage1_cnn", "variant_c_standard", "variant_d_ablation"]
    seeds = [42, 43, 44]

    selected_configs = {}
    validation_summary = {}

    any_meets_strict = False

    for var in variants:
        selected_configs[var] = {}
        validation_summary[var] = {}

        for seed in seeds:
            # 1. Clean Validation Grid Search
            raw_val = evaluate_val_scores(var, seed, val_rows, device, frontend)

            best_params = None
            best_score = -999.0
            best_metrics = None

            if var == "variant_a_cascade_v1":
                for th1 in np.arange(0.30, 0.85, 0.05):
                    for th2 in np.arange(0.20, 0.85, 0.05):
                        params = {"theta_s1": round(float(th1), 2), "theta_s2": round(float(th2), 2)}
                        m = score_variant_preds(var, raw_val, params)
                        if m["meets_strict_criteria"]:
                            any_meets_strict = True
                        score = m["macro_recall"] - 2.5 * m["false_acceptance_rate"]
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_metrics = m

            elif var == "variant_b_stage1_cnn":
                for th in np.arange(0.30, 0.95, 0.05):
                    params = {"theta_cmd": round(float(th), 2)}
                    m = score_variant_preds(var, raw_val, params)
                    if m["meets_strict_criteria"]:
                        any_meets_strict = True
                    score = m["macro_recall"] - 2.5 * m["false_acceptance_rate"]
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = m

            elif var in ["variant_c_standard", "variant_d_ablation"]:
                for th in np.arange(0.20, 0.95, 0.05):
                    params = {"theta_reject": round(float(th), 2)}
                    m = score_variant_preds(var, raw_val, params)
                    if m["meets_strict_criteria"]:
                        any_meets_strict = True
                    score = m["macro_recall"] - 2.5 * m["false_acceptance_rate"]
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = m

            # 2. Robustness Conditions Evaluation using selected best_params
            # Condition 1: Additive noise (SNR = 20 dB)
            raw_noise20 = evaluate_val_scores(var, seed, val_rows, device, frontend, snr_db=20.0)
            m_noise20 = score_variant_preds(var, raw_noise20, best_params)

            # Condition 2: Additive noise (SNR = 15 dB)
            raw_noise15 = evaluate_val_scores(var, seed, val_rows, device, frontend, snr_db=15.0)
            m_noise15 = score_variant_preds(var, raw_noise15, best_params)

            # Condition 3: Gain perturbation (-3 dB)
            raw_gain_neg3 = evaluate_val_scores(var, seed, val_rows, device, frontend, gain_db=-3.0)
            m_gain_neg3 = score_variant_preds(var, raw_gain_neg3, best_params)

            selected_configs[var][f"seed{seed}"] = best_params
            validation_summary[var][f"seed{seed}"] = {
                "calibrated_params": best_params,
                "clean_metrics": best_metrics,
                "robustness": {
                    "snr_20db": m_noise20,
                    "snr_15db": m_noise15,
                    "gain_neg3db": m_gain_neg3
                }
            }
            print(f"[{var}] Seed {seed}: Calibrated {best_params} -> Clean Recall: {best_metrics['macro_recall']*100:.2f}%, FAR: {best_metrics['false_acceptance_rate']*100:.2f}%, Per-KW: {best_metrics['per_keyword_recall']}, Strict Pass: {best_metrics['meets_strict_criteria']}")

    # Mean validation summary across seeds
    for var in variants:
        recalls = [validation_summary[var][f"seed{s}"]["clean_metrics"]["macro_recall"] for s in seeds]
        fars = [validation_summary[var][f"seed{s}"]["clean_metrics"]["false_acceptance_rate"] for s in seeds]
        validation_summary[var]["mean_summary"] = {
            "mean_macro_recall": round(float(np.mean(recalls)), 4),
            "mean_false_acceptance_rate": round(float(np.mean(fars)), 4),
            "mean_selection_score": round(float(np.mean(recalls) - 2.5 * np.mean(fars)), 4)
        }
        print(f"\n===> {var} VALIDATION SUMMARY: Mean Recall: {np.mean(recalls)*100:.2f}%, Mean FAR: {np.mean(fars)*100:.2f}%")

    print("\n" + "=" * 80)
    if not any_meets_strict:
        print("DECISION: NO_FEASIBLE_VALIDATION_OPERATING_POINT")
        print("Reason: No configuration simultaneously achieved Recall >= 95% for EACH keyword and FAR <= 5.0%.")
    else:
        print("DECISION: STRICT_VALIDATION_OPERATING_POINT_FOUND")
    print("=" * 80)

    # Save to files
    calib_path = os.path.join(CALIB_DIR, "selected_configs.json")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(selected_configs, f, indent=2)

    val_sum_path = os.path.join(PRED_DIR, "validation_summary.json")
    with open(val_sum_path, "w", encoding="utf-8") as f:
        json.dump({
            "formal_validation_verdict": "NO_FEASIBLE_VALIDATION_OPERATING_POINT" if not any_meets_strict else "FEASIBLE_OPERATING_POINT",
            "predeclared_criteria": {
                "per_keyword_recall_threshold": 0.95,
                "false_acceptance_rate_ceiling": 0.05,
                "latency_ceiling_ms": 50.0
            },
            "summary_by_variant": validation_summary
        }, f, indent=2)

    print(f"Calibration configs saved to {calib_path}")
    print(f"Validation summary saved to {val_sum_path}")

if __name__ == "__main__":
    run_calibration_and_robustness()
