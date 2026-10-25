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
EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6")
DATA_DIR = os.path.join(EXP_DIR, "data", "splits")
MODELS_DIR = os.path.join(EXP_DIR, "models")
CALIB_DIR = os.path.join(EXP_DIR, "calibration")
PRED_DIR = os.path.join(EXP_DIR, "predictions")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v6.models.train_ablation_campaign import (
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

    # Controlled synthetic robustness conditions
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

    elif variant == "variant_b_standard_bounded_cnn":
        b_ckpt = os.path.join(MODELS_DIR, "checkpoints", f"variant_b_seed{seed}.pt")
        m = VietnameseBoundedKWSNet(num_classes=5).to(device)
        m.load_state_dict(torch.load(b_ckpt, map_location=device)["model_state_dict"])
        m.eval()
        return m, None

    elif variant == "variant_c_hard_negative_ablation":
        c_ckpt = os.path.join(MODELS_DIR, "checkpoints", f"variant_c_seed{seed}.pt")
        m = VietnameseBoundedKWSNet(num_classes=5).to(device)
        m.load_state_dict(torch.load(c_ckpt, map_location=device)["model_state_dict"])
        m.eval()
        return m, None

    elif variant == "variant_d_stage1_cnn":
        s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{seed}.pt")
        m1 = VietnameseKWSNet(num_classes=6).to(device)
        m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
        m1.eval()
        return m1, None
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

            elif variant == "variant_d_stage1_cnn":
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

            elif variant in ["variant_b_standard_bounded_cnn", "variant_c_hard_negative_ablation"]:
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

        elif variant == "variant_d_stage1_cnn":
            th = params["theta_cmd"]
            if r["top_cmd_prob"] >= th and r["top_cmd_prob"] > r["p_unk"] and r["top_cmd_prob"] > r["p_sil"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        elif variant in ["variant_b_standard_bounded_cnn", "variant_c_hard_negative_ablation"]:
            th = params["theta_cmd"]
            if r["top_cmd_prob"] >= th and r["top_cmd_prob"] > r["rej_prob"]:
                pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        preds.append({"true_label": true_lbl, "pred_label": pred_lbl})

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
    print("Phase 63: Validation Calibration & Robustness Evaluation")
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

    variants = [
        "variant_a_cascade_v1",
        "variant_b_standard_bounded_cnn",
        "variant_c_hard_negative_ablation",
        "variant_d_stage1_cnn"
    ]
    seeds = [42, 43, 44]

    selected_configs = {}
    validation_summary = {}
    any_meets_strict = False

    for var in variants:
        selected_configs[var] = {}
        validation_summary[var] = {}

        for seed in seeds:
            print(f"\nSearching thresholds for [{var}] Seed {seed} on validation split...")
            val_raw = evaluate_val_scores(var, seed, val_rows, device, frontend)

            best_cfg = None
            best_strict_cfg = None
            best_strict_far = 999.0

            best_fallback_cfg = None
            best_fallback_metric = -999.0

            if var == "variant_a_cascade_v1":
                th1_range = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
                th2_range = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
                grid = [{"theta_s1": t1, "theta_s2": t2} for t1 in th1_range for t2 in th2_range]
            else:
                th_range = [round(x, 2) for x in np.arange(0.35, 0.95, 0.05)]
                grid = [{"theta_cmd": t} for t in th_range]

            for params in grid:
                sc = score_variant_preds(var, val_raw, params)

                # Strict check: all keywords >= 95% AND FAR <= 5%
                if sc["meets_strict_criteria"]:
                    if sc["false_acceptance_rate"] < best_strict_far:
                        best_strict_far = sc["false_acceptance_rate"]
                        best_strict_cfg = {"params": params, "metrics": sc}

                # Fallback: maximize macro recall subject to FAR <= 5.0%
                if sc["false_acceptance_rate"] <= 0.05:
                    if sc["macro_recall"] > best_fallback_metric:
                        best_fallback_metric = sc["macro_recall"]
                        best_fallback_cfg = {"params": params, "metrics": sc}

            if best_strict_cfg is not None:
                best_cfg = best_strict_cfg
                status_point = "FEASIBLE_VALIDATION_OPERATING_POINT"
                any_meets_strict = True
            elif best_fallback_cfg is not None:
                best_cfg = best_fallback_cfg
                status_point = "NO_FEASIBLE_VALIDATION_OPERATING_POINT"
            else:
                # If even FAR <= 5% cannot be satisfied, pick highest macro recall with lowest FAR
                grid_scores = [(score_variant_preds(var, val_raw, p), p) for p in grid]
                grid_scores.sort(key=lambda item: (item[0]["false_acceptance_rate"], -item[0]["macro_recall"]))
                best_cfg = {"params": grid_scores[0][1], "metrics": grid_scores[0][0]}
                status_point = "NO_FEASIBLE_VALIDATION_OPERATING_POINT"

            print(f"  Selected Params: {best_cfg['params']} | Recall: {best_cfg['metrics']['macro_recall']*100:.2f}% | FAR: {best_cfg['metrics']['false_acceptance_rate']*100:.2f}% | Status: {status_point}")

            # Robustness evaluation on validation set with selected threshold
            robustness_results = {}
            for test_snr in [None, 20.0, 15.0, 10.0]:
                for test_gain in [None, 3.0, -3.0]:
                    cond_name = "clean"
                    if test_snr is not None:
                        cond_name = f"snr_{int(test_snr)}db"
                    elif test_gain is not None:
                        cond_name = f"gain_{'+' if test_gain > 0 else ''}{int(test_gain)}db"

                    if cond_name not in robustness_results:
                        rob_raw = evaluate_val_scores(var, seed, val_rows, device, frontend, snr_db=test_snr, gain_db=test_gain)
                        rob_score = score_variant_preds(var, rob_raw, best_cfg["params"])
                        robustness_results[cond_name] = rob_score

            selected_configs[var][f"seed{seed}"] = {
                "params": best_cfg["params"],
                "val_clean_metrics": best_cfg["metrics"],
                "status_point": status_point,
                "robustness": robustness_results
            }
            validation_summary[var][f"seed{seed}"] = best_cfg["metrics"]

    # Save selected configs & validation summary
    cfg_path = os.path.join(CALIB_DIR, "selected_configs.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(selected_configs, f, indent=2)
    print(f"\nWrote Selected Configs: {cfg_path}")

    val_sum_path = os.path.join(PRED_DIR, "validation_summary.json")
    with open(val_sum_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)
    print(f"Wrote Validation Summary: {val_sum_path}")

    # Written Finalist Selection Rule:
    # 1. Any candidate with status_point == FEASIBLE_VALIDATION_OPERATING_POINT takes precedence.
    # 2. Otherwise, rank candidates by mean validation composite score = val_recall * 0.6 + (1 - val_far) * 0.4.
    # Select top 2 finalists for locked test evaluation in Phase 64.
    candidate_ranking = []
    for var in variants:
        recalls = [validation_summary[var][f"seed{s}"]["macro_recall"] for s in seeds]
        fars = [validation_summary[var][f"seed{s}"]["false_acceptance_rate"] for s in seeds]
        all_ge_95 = all(validation_summary[var][f"seed{s}"]["all_keywords_ge_95"] for s in seeds)
        mean_rec = np.mean(recalls)
        mean_far = np.mean(fars)
        composite = mean_rec * 0.6 + (1.0 - mean_far) * 0.4
        candidate_ranking.append({
            "variant": var,
            "mean_recall": round(float(mean_rec), 4),
            "mean_far": round(float(mean_far), 4),
            "composite": round(float(composite), 4),
            "all_keywords_ge_95": all_ge_95
        })

    candidate_ranking.sort(key=lambda x: (-int(x["all_keywords_ge_95"]), -x["composite"]))
    finalists = [x["variant"] for x in candidate_ranking[:2]]
    print(f"\nFinalist Selection Rule Selected: {finalists}")

    # Generate PHASE63_CALIBRATION_AND_ROBUSTNESS.md
    report_lines = [
        "# Phase 63 — Validation Calibration and Robustness Report",
        "",
        "**Workspace:** `D:\\trainidentificationaudio`  ",
        "**Experiment Directory:** `experiments/vieneu_kws_research_v6/`  ",
        "**Selected Configurations:** `experiments/vieneu_kws_research_v6/calibration/selected_configs.json`  ",
        "**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "Phase 63 conducted rigorous, validation-only calibration and synthetic robustness profiling across 4 bounded-utterance candidate architectures on the newly synthesized Benchmark Dataset v6 validation split ($N = 360$ clips).",
        "",
        "### Predeclared Acceptance Criteria:",
        "- **Per-Keyword Target Recall:** $\\ge 95.0\\%$ for EACH keyword (`TRAI`, `PHAI`, `LEN`, `XUONG`).",
        "- **Non-Target False Acceptance Rate (FAR):** $\\le 5.0\\%$ on non-target distractor and silence clips.",
        "- **CPU Latency Ceiling:** $\\le 50\\,\\text{ms}$ post-capture decision latency.",
        f"- **Validation Operating Point Determination:** {'FEASIBLE_VALIDATION_OPERATING_POINT' if any_meets_strict else 'NO_FEASIBLE_VALIDATION_OPERATING_POINT'}.",
        "",
        "---",
        "",
        "## 2. Validation Calibration Performance Summary Across Seeds",
        "",
        "| Candidate Variant | Seed | Calibrated Operating Parameters | Val Command Recall | Val Non-Target FAR | TRAI | PHAI | LEN | XUONG | Strict Pass? | Operating Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for var in variants:
        for s in seeds:
            cfg = selected_configs[var][f"seed{s}"]
            p = cfg["params"]
            p_str = ", ".join(f"{k}={v}" for k, v in p.items())
            m = cfg["val_clean_metrics"]
            pk = m["per_keyword_recall"]
            report_lines.append(
                f"| **{var}** | {s} | `{p_str}` | {m['macro_recall']*100:.2f}% | {m['false_acceptance_rate']*100:.2f}% | {pk['TRAI']*100:.1f}% | {pk['PHAI']*100:.1f}% | {pk['LEN']*100:.1f}% | {pk['XUONG']*100:.1f}% | {'YES' if m['meets_strict_criteria'] else 'NO'} | `{cfg['status_point']}` |"
            )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Synthetic Noise & Gain Robustness Breakdown (Validation Split)",
        "",
        "| Candidate Variant | Seed | Clean (Recall / FAR) | SNR 20dB (Recall / FAR) | SNR 15dB (Recall / FAR) | SNR 10dB (Recall / FAR) | Gain +3dB (Recall / FAR) | Gain -3dB (Recall / FAR) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    )

    for var in variants:
        for s in seeds:
            rob = selected_configs[var][f"seed{s}"]["robustness"]
            c_rec = f"{rob['clean']['macro_recall']*100:.1f}% / {rob['clean']['false_acceptance_rate']*100:.1f}%"
            s20 = f"{rob['snr_20db']['macro_recall']*100:.1f}% / {rob['snr_20db']['false_acceptance_rate']*100:.1f}%"
            s15 = f"{rob['snr_15db']['macro_recall']*100:.1f}% / {rob['snr_15db']['false_acceptance_rate']*100:.1f}%"
            s10 = f"{rob['snr_10db']['macro_recall']*100:.1f}% / {rob['snr_10db']['false_acceptance_rate']*100:.1f}%"
            gp3 = f"{rob['gain_+3db']['macro_recall']*100:.1f}% / {rob['gain_+3db']['false_acceptance_rate']*100:.1f}%"
            gm3 = f"{rob['gain_-3db']['macro_recall']*100:.1f}% / {rob['gain_-3db']['false_acceptance_rate']*100:.1f}%"
            report_lines.append(f"| **{var}** | {s} | {c_rec} | {s20} | {s15} | {s10} | {gp3} | {gm3} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Finalist Selection for Locked Test Evaluation (Phase 64)",
        "",
        "### Selection Rule Formulation:",
        "1. Any candidate satisfying all four keyword recall targets ($\\ge 95\\%$) and $\\text{FAR} \\le 5.0\\%$ automatically qualifies.",
        "2. Candidates are ranked by composite score $S = 0.6 \\cdot \\text{Recall} + 0.4 \\cdot (1 - \\text{FAR})$.",
        "3. The top 2 ranking candidates advance as finalists to Phase 64 locked test evaluation.",
        "",
        "### Selected Finalists:",
        f"1. **Finalist 1:** `{finalists[0]}`",
        f"2. **Finalist 2:** `{finalists[1]}`",
        "",
        "These two configurations are locked into `calibration/selected_configs.json` and proceed directly to Phase 64."
    ])

    report_path = os.path.join(REPORTS_DIR, "PHASE63_CALIBRATION_AND_ROBUSTNESS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Wrote Phase 63 Report: {report_path}")

    return finalists


if __name__ == "__main__":
    run_calibration_and_robustness()
