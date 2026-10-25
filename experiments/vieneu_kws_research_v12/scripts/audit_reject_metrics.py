#!/usr/bin/env python3
"""
Phase 26: Exhaustive REJECT Confusion & Error Audit
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/audit_reject_metrics.py
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "training_phase25"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
from train_phase25_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, FactorialKWSDataset, CLASSES, CLASS_TO_IDX

CONDITIONS = ["D0_H0", "D0_H1", "D1_H0", "D1_H1"]
SEEDS = [42, 43, 44]


def main():
    print("=" * 80)
    print("PHASE 26: EVALUATING REJECT CONFUSION & ERROR DYNAMICS")
    print("=" * 80)

    val_manifest = MANIFESTS_DIR / "validation_v1.csv"
    val_dataset = FactorialKWSDataset(val_manifest, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    val_df = pd.read_csv(val_manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = LogMelFrontend().to(device)
    frontend.eval()

    all_results = {}

    for cond in CONDITIONS:
        all_results[cond] = {}
        for seed in SEEDS:
            ckpt_path = TRAIN_DIR / f"{cond}_seed{seed}" / "best_model.pt"
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            preds_list = []
            targets_list = []

            with torch.no_grad():
                for waveforms, targets in val_loader:
                    waveforms = waveforms.to(device)
                    specs = frontend(waveforms)
                    logits = model(specs)
                    probs = F.softmax(logits, dim=-1)

                    target_probs = probs[:, :4]
                    max_p, max_idx = torch.max(target_probs, dim=-1)
                    preds = torch.where(max_p >= 0.20, max_idx, torch.tensor(4, device=device))

                    preds_list.extend(preds.cpu().numpy().tolist())
                    targets_list.extend(targets.numpy().tolist())

            y_true = np.array(targets_list)
            y_pred = np.array(preds_list)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

            # REJECT row is index 4
            # cm[4, 0:4] are false alarms on REJECT (predicted as target keyword)
            reject_total = int(np.sum(cm[4, :]))  # 640
            reject_true_reject = int(cm[4, 4])
            reject_fa_total = int(np.sum(cm[4, :4]))
            reject_fa_rate = reject_fa_total / reject_total

            # Keyword rows are 0:4
            # cm[0:4, 4] are false rejections (keywords predicted as REJECT)
            kw_total = int(np.sum(cm[:4, :])) # 640
            kw_miss_total = int(np.sum(cm[:4, 4]))
            kw_miss_rate = kw_miss_total / kw_total

            # Analyze reject subgroup errors
            reject_mask = (y_true == 4)
            reject_subgroups = val_df.loc[reject_mask, "reject_subgroup"].values
            reject_preds = y_pred[reject_mask]

            subgroup_fa = {}
            for sg in ["oov_word", "conversation", "environmental_noise", "silence_non_speech"]:
                sg_mask = (reject_subgroups == sg)
                sg_total = int(np.sum(sg_mask))
                sg_fa = int(np.sum(reject_preds[sg_mask] != 4))
                subgroup_fa[sg] = {
                    "total": sg_total,
                    "fa_count": sg_fa,
                    "fa_rate": sg_fa / sg_total if sg_total > 0 else 0.0
                }

            all_results[cond][f"seed{seed}"] = {
                "confusion_matrix": cm.tolist(),
                "reject_total": reject_total,
                "reject_true_reject": reject_true_reject,
                "reject_fa_total": reject_fa_total,
                "reject_fa_rate": reject_fa_rate,
                "kw_total": kw_total,
                "kw_miss_total": kw_miss_total,
                "kw_miss_rate": kw_miss_rate,
                "subgroup_fa": subgroup_fa
            }

            print(f"[{cond}_seed{seed}] REJECT FA: {reject_fa_total}/{reject_total} ({reject_fa_rate*100:.2f}%) | KW Miss: {kw_miss_total}/{kw_total} ({kw_miss_rate*100:.2f}%) | OOV FA: {subgroup_fa['oov_word']['fa_count']}/{subgroup_fa['oov_word']['total']} ({subgroup_fa['oov_word']['fa_rate']*100:.2f}%)")

    # Aggregate by condition
    cond_reject_summary = {}
    print("\n--- Summary by Condition (Mean across 3 seeds) ---")
    for cond in CONDITIONS:
        fa_rates = [all_results[cond][f"seed{s}"]["reject_fa_rate"] for s in SEEDS]
        miss_rates = [all_results[cond][f"seed{s}"]["kw_miss_rate"] for s in SEEDS]
        oov_fa_rates = [all_results[cond][f"seed{s}"]["subgroup_fa"]["oov_word"]["fa_rate"] for s in SEEDS]
        conv_fa_rates = [all_results[cond][f"seed{s}"]["subgroup_fa"]["conversation"]["fa_rate"] for s in SEEDS]

        cond_reject_summary[cond] = {
            "mean_reject_fa_rate": float(np.mean(fa_rates)),
            "std_reject_fa_rate": float(np.std(fa_rates, ddof=1)),
            "mean_kw_miss_rate": float(np.mean(miss_rates)),
            "std_kw_miss_rate": float(np.std(miss_rates, ddof=1)),
            "mean_oov_fa_rate": float(np.mean(oov_fa_rates)),
            "std_oov_fa_rate": float(np.std(oov_fa_rates, ddof=1)),
            "mean_conv_fa_rate": float(np.mean(conv_fa_rates)),
            "std_conv_fa_rate": float(np.std(conv_fa_rates, ddof=1))
        }

        print(f"  {cond:<6} | Reject FA: {np.mean(fa_rates)*100:.2f}% ± {np.std(fa_rates, ddof=1)*100:.2f}% | KW Miss: {np.mean(miss_rates)*100:.2f}% ± {np.std(miss_rates, ddof=1)*100:.2f}% | OOV FA: {np.mean(oov_fa_rates)*100:.2f}% | Conv FA: {np.mean(conv_fa_rates)*100:.2f}%")

    out_file = REPORTS_DIR / "phase26_reject_audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "runs": all_results,
            "condition_summary": cond_reject_summary
        }, f, indent=2)

    print(f"\nSaved REJECT audit results to {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
