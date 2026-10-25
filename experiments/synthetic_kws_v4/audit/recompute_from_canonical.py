"""
Recompute Metrics from Canonical Predictions Enforcing Mathematical Identities.
Ensures zero hand-calculated or copy-pasted discrepancy in reported metrics.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

CLASS_NAMES_6 = ["TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE"]
TARGET_IDS = {0, 1, 2, 3}


def recompute_metrics():
    pred_csv = Path("experiments/synthetic_kws_v4/audit/canonical_predictions.csv")
    assert pred_csv.exists(), f"Missing {pred_csv}"

    df = pd.read_csv(pred_csv)
    out_dir = Path("experiments/synthetic_kws_v4/audit")

    metrics_by_arm = {}

    for arm_name in sorted(df["model_arm"].unique()):
        arm_df = df[df["model_arm"] == arm_name]
        ckpt_path = arm_df["checkpoint_path"].iloc[0]
        ckpt_sha = arm_df["checkpoint_sha256"].iloc[0]

        arm_entry = {
            "checkpoint_path": ckpt_path,
            "checkpoint_sha256": ckpt_sha,
            "partitions": {}
        }

        for part_id in sorted(arm_df["partition_id"].unique()):
            pdf = arm_df[arm_df["partition_id"] == part_id]
            man_path = pdf["manifest_path"].iloc[0]
            man_sha = pdf["manifest_sha256"].iloc[0]
            n_samples = len(pdf)

            y_true = pdf["true_class_id"].to_numpy()
            y_pred = pdf["pred_class_id"].to_numpy()

            # 1. 6x6 Confusion Matrix
            cm = np.zeros((6, 6), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1

            # 2. Mathematical Identity Check 1: Row sums == class support
            row_sums = cm.sum(axis=1)
            # Mathematical Identity Check 2: Sum of row sums == total samples
            assert int(row_sums.sum()) == n_samples, f"Sum of row sums {row_sums.sum()} != n_samples {n_samples}"

            # 3. Mathematical Identity Check 3: Overall accuracy == trace / total
            trace = int(np.trace(cm))
            acc = trace / n_samples
            assert np.isclose(acc, accuracy_score(y_true, y_pred)), "Accuracy calculation discrepancy"

            # 4. Mathematical Identity Check 4: Class recall == diag / row sum
            recalls = []
            precisions = []
            f1s = []
            for i in range(6):
                rec = (cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
                recalls.append(float(rec))
                col_sum = cm[:, i].sum()
                prec = (cm[i, i] / col_sum) if col_sum > 0 else 0.0
                precisions.append(float(prec))
                if (prec + rec) > 0:
                    f1_val = 2 * (prec * rec) / (prec + rec)
                else:
                    f1_val = 0.0
                f1s.append(float(f1_val))

            macro_f1 = float(np.mean(f1s))

            # Verify that sum(support_i * recall_i) == trace
            weighted_rec_sum = sum(row_sums[i] * recalls[i] for i in range(6))
            assert np.isclose(weighted_rec_sum, trace), f"Weighted recall sum {weighted_rec_sum} != trace {trace}"

            # 5. Mathematical Identity Check 5: Target command miss count
            # Targets are classes 0, 1, 2, 3
            target_support = int(row_sums[:4].sum())
            target_correct = int(np.trace(cm[:4, :4]))
            target_misses = target_support - target_correct
            # Assert target misses == sum of off-diagonal elements in target rows
            off_diag_target_misses = int(cm[:4, :].sum() - np.trace(cm[:4, :4]))
            assert target_misses == off_diag_target_misses, f"Target misses {target_misses} != off-diag {off_diag_target_misses}"
            target_miss_rate_pct = (target_misses / target_support * 100.0) if target_support > 0 else 0.0

            # 6. Negative False Alarms
            unk_total = int(row_sums[4])
            unk_fa_count = int(cm[4, :4].sum())
            unk_fa_rate_pct = (unk_fa_count / unk_total * 100.0) if unk_total > 0 else 0.0

            sil_total = int(row_sums[5])
            sil_fa_count = int(cm[5, :4].sum())
            sil_fa_rate_pct = (sil_fa_count / sil_total * 100.0) if sil_total > 0 else 0.0

            # Confusion details
            trai_as_phai = int(cm[0, 1])
            trai_as_unk = int(cm[0, 4])
            phai_as_trai = int(cm[1, 0])
            phai_as_unk = int(cm[1, 4])

            part_metrics = {
                "manifest_path": man_path,
                "manifest_sha256": man_sha,
                "samples_processed": n_samples,
                "samples_skipped": 0,
                "overall_accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "precision": [float(p) for p in precisions],
                "recall": [float(r) for r in recalls],
                "f1": [float(f) for f in f1s],
                "support": [int(s) for s in row_sums],
                "confusion_matrix": cm.tolist(),
                "target_command_miss_rate_pct": float(target_miss_rate_pct),
                "target_miss_count": target_misses,
                "target_total": target_support,
                "unknown_fa_rate_pct": float(unk_fa_rate_pct),
                "unknown_fa_count": unk_fa_count,
                "unknown_total": unk_total,
                "silence_fa_rate_pct": float(sil_fa_rate_pct),
                "silence_fa_count": sil_fa_count,
                "silence_total": sil_total,
                "trai_recall": float(recalls[0]),
                "phai_recall": float(recalls[1]),
                "len_recall": float(recalls[2]),
                "xuong_recall": float(recalls[3]),
                "trai_as_phai": trai_as_phai,
                "trai_as_unknown": trai_as_unk,
                "phai_as_trai": phai_as_trai,
                "phai_as_unknown": phai_as_unk
            }
            arm_entry["partitions"][part_id] = part_metrics

        metrics_by_arm[arm_name] = arm_entry

    out_json = out_dir / "canonical_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_by_arm, f, indent=2)
    print(f"Saved canonical recomputed metrics to {out_json}")

    # Generate Markdown Table strictly from numbers
    print("\n" + "=" * 80)
    print("CANONICAL VERIFIED MARKDOWN TABLE")
    print("=" * 80)
    headers = [
        "Model / Arm", "Partition", "Count (Skip)", "Accuracy", "Macro-F1",
        "TRAI Rec", "PHAI Rec", "LEN Rec", "XUONG Rec", "UNK FA", "SIL FA", "Command Miss Rate"
    ]
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] * 2 + [":---:"] * (len(headers) - 2)) + " |"]

    partition_display_names = {
        "hist_144": "Hist. 144 Target [Diag]",
        "hist_216": "Hist. 6-Class (216) [Diag]",
        "v2_fresh_300": "V2 Fresh Holdout (300) [Diag]",
        "v3_fresh_300": "V3 Fresh Holdout (300) [Diag]",
        "v4_fresh_300": "V4 Fresh Holdout (300) [Untouched]",
        "val_baseline_216": "Val Baseline Gen (216) [Diag]",
        "val_onset_216": "Val Onset-Aware Gen (216) [Diag]"
    }

    arm_order = [
        "Arm A (Control)",
        "Arm B (Onset-Aware, No Resample)",
        "Arm C (Onset-Aware + Resample)",
        "V3 Champion TC-ResNet-14"
    ]

    for arm in arm_order:
        if arm not in metrics_by_arm:
            continue
        first_part = True
        arm_data = metrics_by_arm[arm]
        for part_key in [
            "hist_144", "hist_216", "v2_fresh_300", "v3_fresh_300",
            "v4_fresh_300", "val_baseline_216", "val_onset_216"
        ]:
            if part_key not in arm_data["partitions"]:
                continue
            p = arm_data["partitions"][part_key]
            part_name = partition_display_names[part_key]
            arm_col = f"**{arm}**" if first_part else ""
            first_part = False

            row = [
                arm_col,
                part_name,
                f"{p['samples_processed']} ({p['samples_skipped']})",
                f"{p['overall_accuracy']*100:.2f}%",
                f"{p['macro_f1']:.4f}",
                f"{p['trai_recall']*100:.1f}%",
                f"{p['phai_recall']*100:.1f}%",
                f"{p['len_recall']*100:.1f}%",
                f"{p['xuong_recall']*100:.1f}%",
                f"{p['unknown_fa_rate_pct']:.1f}%",
                f"{p['silence_fa_rate_pct']:.1f}%",
                f"{p['target_command_miss_rate_pct']:.2f}% ({p['target_miss_count']}/{p['target_total']})"
            ]
            md_lines.append("| " + " | ".join(row) + " |")

    table_md = "\n".join(md_lines)
    print(table_md)
    with open(out_dir / "canonical_table.md", "w", encoding="utf-8") as f:
        f.write(table_md)


if __name__ == "__main__":
    recompute_metrics()
