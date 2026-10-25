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

V6_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v6")
V7_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v7")
V8_EXP_DIR = os.path.join(BASE_DIR, "experiments", "vieneu_kws_research_v8")

DATA_DIR = os.path.join(V6_EXP_DIR, "data", "splits")
V7_MODELS_DIR = os.path.join(V7_EXP_DIR, "models", "checkpoints")
V8_MODELS_DIR = os.path.join(V8_EXP_DIR, "models", "checkpoints")

CALIB_DIR = os.path.join(V8_EXP_DIR, "calibration")
PRED_DIR = os.path.join(V8_EXP_DIR, "predictions")
REPORTS_DIR = os.path.join(V8_EXP_DIR, "reports")

from src.models.kws_cnn import VietnameseKWSNet
from experiments.vieneu_kws_cascade_v1.train_verifier import TemporalKWSVerifier
from experiments.vieneu_kws_research_v8.models.train_ablation_v8 import (
    CorrectedVietnameseKWSNet,
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
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    elif len(audio) > target_length:
        audio = audio[:target_length]

    if gain_db is not None:
        audio = audio * (10.0 ** (gain_db / 20.0))

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
    if variant == "cascade_v1":
        s1_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_v2", "checkpoints", f"best_checkpoint_seed{seed}.pt")
        s2_ckpt = os.path.join(BASE_DIR, "experiments", "vieneu_kws_cascade_v1", "checkpoints", f"best_verifier_seed{seed}.pt")
        m1 = VietnameseKWSNet(num_classes=6).to(device)
        m1.load_state_dict(torch.load(s1_ckpt, map_location=device)["model_state_dict"])
        m1.eval()

        m2 = TemporalKWSVerifier(num_classes=5).to(device)
        m2.load_state_dict(torch.load(s2_ckpt, map_location=device)["model_state_dict"])
        m2.eval()
        return m1, m2

    elif variant == "candidate_b_v7_baseline":
        ckpt_path = os.path.join(V7_MODELS_DIR, f"candidate_b_seed{seed}.pt")
        m = CorrectedVietnameseKWSNet(num_classes=5).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        m.eval()
        return m, None

    elif variant in ["variant_w10", "variant_w12", "variant_w14", "variant_w16"]:
        ckpt_path = os.path.join(V8_MODELS_DIR, f"{variant}_seed{seed}.pt")
        m = CorrectedVietnameseKWSNet(num_classes=5).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
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

            if variant == "cascade_v1":
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

            else:
                p = F.softmax(m1(spec), dim=1).cpu().numpy()[0]
                top_cmd_idx = int(np.argmax(p[:4]))
                top_cmd_prob = float(p[:4].max())
                raw_results.append({
                    "sample_id": r["sample_id"],
                    "true_label": r["bounded_label"],
                    "top_cmd_idx": top_cmd_idx,
                    "top_cmd_prob": top_cmd_prob,
                    "rej_prob": float(p[4]),
                    "all_probs": [float(x) for x in p]
                })

    return raw_results


def score_variant_preds(variant, raw_results, params):
    preds = []
    for r in raw_results:
        true_lbl = r["true_label"]
        pred_lbl = "REJECT"

        if variant == "cascade_v1":
            th1 = params["theta_s1"]
            th2 = params["theta_s2"]
            if r["s1_top"] >= th1 and r["s1_top"] > r["s1_p_unk"] and r["s1_top"] > r["s1_p_sil"]:
                if r["s2_top"] >= th2 and r["s2_top"] > r["s2_rej"]:
                    pred_lbl = TARGET_CLASSES[r["top_cmd_idx"]]

        else:
            th = params["theta_cmd"]
            if r["top_cmd_prob"] >= th:
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
        "raw_false_acceptance_count": nt_total - nt_rejects,
        "rejection_rate": round(tnr, 4),
        "per_keyword_recall": {k: round(v, 4) for k, v in per_kw.items()},
        "all_keywords_ge_95": all_kws_pass,
        "meets_strict_criteria": meets_strict_criteria
    }


def run_calibration():
    print("=" * 80)
    print("Phase 73: Robust Calibration and Finalist Selection")
    print("=" * 80)

    os.makedirs(CALIB_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    val_manifest = os.path.join(DATA_DIR, "val_manifest.csv")
    val_rows = []
    with open(val_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val_rows.append(r)

    device = torch.device("cpu")
    torch.set_num_threads(4)
    frontend = LogMelFrontend().to(device)

    variants = [
        "cascade_v1",
        "candidate_b_v7_baseline",
        "variant_w10",
        "variant_w12",
        "variant_w14",
        "variant_w16"
    ]
    seeds = [42, 43, 44]

    selected_configs = {}
    validation_summary = {}

    for var in variants:
        selected_configs[var] = {}
        validation_summary[var] = {}

        for seed in seeds:
            print(f"\nCalibrating [{var}] Seed {seed} on validation split...")
            val_raw = evaluate_val_scores(var, seed, val_rows, device, frontend)

            best_strict_cfg = None
            best_strict_far = 999.0
            best_fallback_cfg = None
            best_fallback_metric = -999.0

            if var == "cascade_v1":
                th1_range = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
                th2_range = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
                grid = [{"theta_s1": t1, "theta_s2": t2} for t1 in th1_range for t2 in th2_range]
            else:
                th_range = [round(x, 2) for x in np.arange(0.25, 0.95, 0.05)]
                grid = [{"theta_cmd": t} for t in th_range]

            for params in grid:
                sc = score_variant_preds(var, val_raw, params)

                # Strict criteria: all 4 keywords >= 95% AND FAR <= 5%
                if sc["meets_strict_criteria"]:
                    if sc["false_acceptance_rate"] < best_strict_far:
                        best_strict_far = sc["false_acceptance_rate"]
                        best_strict_cfg = {"params": params, "metrics": sc}

                # Fallback: maximize macro recall subject to FAR <= 5%
                if sc["false_acceptance_rate"] <= 0.05:
                    if sc["macro_recall"] > best_fallback_metric:
                        best_fallback_metric = sc["macro_recall"]
                        best_fallback_cfg = {"params": params, "metrics": sc}

            if best_strict_cfg is not None:
                best_cfg = best_strict_cfg
                status_point = "FEASIBLE_VALIDATION_OPERATING_POINT"
            elif best_fallback_cfg is not None:
                best_cfg = best_fallback_cfg
                status_point = "NO_FEASIBLE_VALIDATION_OPERATING_POINT"
            else:
                grid_scores = [(score_variant_preds(var, val_raw, p), p) for p in grid]
                grid_scores.sort(key=lambda item: (item[0]["false_acceptance_rate"], -item[0]["macro_recall"]))
                best_cfg = {"params": grid_scores[0][1], "metrics": grid_scores[0][0]}
                status_point = "NO_FEASIBLE_VALIDATION_OPERATING_POINT"

            print(f"  Selected: {best_cfg['params']} | Recall: {best_cfg['metrics']['macro_recall']*100:.2f}% | FAR: {best_cfg['metrics']['false_acceptance_rate']*100:.2f}% | Status: {status_point}")

            # Robustness sweep across conditions
            robustness_results = {}
            for test_snr, test_gain, cond_name in [
                (None, None, "clean"),
                (20.0, None, "snr_20db"),
                (15.0, None, "snr_15db"),
                (10.0, None, "snr_10db"),
                (None, 3.0, "gain_+3db"),
                (None, -3.0, "gain_-3db"),
            ]:
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

    # Save outputs
    cfg_path = os.path.join(CALIB_DIR, "selected_configs.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(selected_configs, f, indent=2)
    print(f"\nWrote Selected Configs: {cfg_path}")

    val_sum_path = os.path.join(PRED_DIR, "validation_summary.json")
    with open(val_sum_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)
    print(f"Wrote Validation Summary: {val_sum_path}")

    # Written Finalist Selection Rule:
    # 1. Any candidate with ALL seeds achieving FEASIBLE_VALIDATION_OPERATING_POINT takes precedence.
    # 2. Otherwise rank candidates by composite score S = 0.6 * Mean Recall + 0.4 * (1 - Mean FAR).
    # 3. Select top 2 finalists to advance to Phase 74 locked test evaluation.
    ranking = []
    for var in variants:
        recalls = [validation_summary[var][f"seed{s}"]["macro_recall"] for s in seeds]
        fars = [validation_summary[var][f"seed{s}"]["false_acceptance_rate"] for s in seeds]
        all_ge_95 = all(validation_summary[var][f"seed{s}"]["all_keywords_ge_95"] for s in seeds)
        all_strict = all(validation_summary[var][f"seed{s}"]["meets_strict_criteria"] for s in seeds)
        mean_rec = float(np.mean(recalls))
        mean_far = float(np.mean(fars))
        comp = mean_rec * 0.6 + (1.0 - mean_far) * 0.4
        ranking.append({
            "variant": var,
            "mean_recall": round(mean_rec, 4),
            "mean_far": round(mean_far, 4),
            "composite": round(float(comp), 4),
            "all_strict_pass": all_strict,
            "all_keywords_ge_95": all_ge_95
        })

    ranking.sort(key=lambda x: (-int(x["all_strict_pass"]), -int(x["all_keywords_ge_95"]), -x["composite"]))
    finalists = [x["variant"] for x in ranking[:2]]
    print(f"\nFinalist Selection Rule Selected: {finalists}")

    write_calibration_report(selected_configs, validation_summary, finalists, ranking)
    return finalists


def write_calibration_report(selected_configs, val_summary, finalists, ranking):
    seeds = [42, 43, 44]
    variants = list(selected_configs.keys())

    lines = [
        "# Phase 73 — Robust Calibration and Finalist Selection Report",
        "",
        "**Workspace:** `D:\\trainidentificationaudio`  ",
        "**Experiment Directory:** `experiments/vieneu_kws_research_v8/`  ",
        "**Status Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Written Selection Rule",
        "",
        "Phase 73 compared six candidate systems on the 360-clip validation partition across clean and synthetic robustness conditions:",
        "1. **`cascade_v1`:** Frozen historical reference baseline.",
        "2. **`candidate_b_v7_baseline`:** Historical Candidate B baseline ($w_{\\text{hn}} = 1.2$).",
        "3. **`variant_w10`:** Fresh ablation variant ($w_{\\text{hn}} = 1.0$).",
        "4. **`variant_w12`:** Fresh ablation variant ($w_{\\text{hn}} = 1.2$).",
        "5. **`variant_w14`:** Fresh ablation variant ($w_{\\text{hn}} = 1.4$, class mass parity).",
        "6. **`variant_w16`:** Fresh ablation variant ($w_{\\text{hn}} = 1.6$, stronger suppression).",
        "",
        "### Written Finalist Selection Rule:",
        "1. Prefer candidates where all seeds simultaneously satisfy $\\ge 95.0\\%$ recall for every keyword and $\\le 5.0\\%$ non-target FAR on validation.",
        "2. If no candidate satisfies this strict condition across all seeds, rank candidates by composite score $S = 0.6 \\cdot \\text{Mean Recall} + 0.4 \\cdot (1 - \\text{Mean FAR})$.",
        "3. Select at most two finalists to advance to locked test benchmarking in Phase 74 while honestly retaining the failure status.",
        "",
        "---",
        "",
        "## 2. Validation Calibration Performance Across Candidates",
        "",
        "| Candidate Architecture | Seed | Operating Threshold | Val Recall | Val FAR | Raw FA (out of 200) | TRAI | PHAI | LEN | XUONG | Strict Pass? | Operating Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for var in variants:
        for s in seeds:
            cfg = selected_configs[var][f"seed{s}"]
            p = cfg["params"]
            p_str = ", ".join(f"{k}={v}" for k, v in p.items())
            m = cfg["val_clean_metrics"]
            pk = m["per_keyword_recall"]
            lines.append(
                f"| **{var}** | {s} | `{p_str}` | {m['macro_recall']*100:.2f}% | {m['false_acceptance_rate']*100:.2f}% | {m['raw_false_acceptance_count']}/200 | {pk['TRAI']*100:.1f}% | {pk['PHAI']*100:.1f}% | {pk['LEN']*100:.1f}% | {pk['XUONG']*100:.1f}% | {'YES' if m['meets_strict_criteria'] else 'NO'} | `{cfg['status_point']}` |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Candidate Ranking and Finalist Selection",
        "",
        "| Rank | Candidate Architecture | Mean Val Recall | Mean Val FAR | Composite Score $S$ | All Seeds Strict Pass? | Advance to Phase 74? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for idx, r in enumerate(ranking, 1):
        adv = "YES (Finalist)" if r["variant"] in finalists else "NO"
        lines.append(
            f"| {idx} | **{r['variant']}** | {r['mean_recall']*100:.2f}% | {r['mean_far']*100:.2f}% | {r['composite']:.4f} | {'YES' if r['all_strict_pass'] else 'NO'} | **{adv}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Frozen Finalist Configurations for Phase 74",
        "",
        f"- **Finalist 1:** `{finalists[0]}`",
        f"- **Finalist 2:** `{finalists[1]}`",
        "",
        "All parameters and model checkpoints are frozen into `calibration/selected_configs.json` and proceed without modification into Phase 74."
    ])

    rep_path = os.path.join(REPORTS_DIR, "PHASE73_CALIBRATION.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {rep_path}")


if __name__ == "__main__":
    run_calibration()
