import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V12_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12"
REPORTS_DIR = V12_DIR / "reports"

with open(REPORTS_DIR / "eval_v11_baseline.json", encoding="utf-8") as f:
    v11_data = json.load(f)
with open(REPORTS_DIR / "eval_e1_results.json", encoding="utf-8") as f:
    e1_data = json.load(f)

v11_cm = np.array(v11_data["offline_evaluation"]["confusion_matrix"])
e1_cm = np.array(e1_data["offline_evaluation"]["confusion_matrix"])

classes = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot v11
im1 = ax1.imshow(v11_cm, interpolation='nearest', cmap=plt.cm.Blues)
ax1.set_title("Frozen Baseline v11 Confusion Matrix\n(Accuracy: 82.11%, Macro-F1: 0.8232)", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
tick_marks = np.arange(len(classes))
ax1.set_xticks(tick_marks)
ax1.set_xticklabels(classes, rotation=45)
ax1.set_yticks(tick_marks)
ax1.set_yticklabels(classes)
ax1.set_ylabel("True Label", fontsize=11)
ax1.set_xlabel("Predicted Label", fontsize=11)

thresh1 = v11_cm.max() / 2.
for i in range(v11_cm.shape[0]):
    for j in range(v11_cm.shape[1]):
        ax1.text(j, i, f"{v11_cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if v11_cm[i, j] > thresh1 else "black",
                 fontweight="bold" if i == j else "normal")

# Plot E1
im2 = ax2.imshow(e1_cm, interpolation='nearest', cmap=plt.cm.Greens)
ax2.set_title("Controlled Experiment E1 Confusion Matrix\n(Accuracy: 95.86%, Macro-F1: 0.9601)", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_xticks(tick_marks)
ax2.set_xticklabels(classes, rotation=45)
ax2.set_yticks(tick_marks)
ax2.set_yticklabels(classes)
ax2.set_ylabel("True Label", fontsize=11)
ax2.set_xlabel("Predicted Label", fontsize=11)

thresh2 = e1_cm.max() / 2.
for i in range(e1_cm.shape[0]):
    for j in range(e1_cm.shape[1]):
        ax2.text(j, i, f"{e1_cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if e1_cm[i, j] > thresh2 else "black",
                 fontweight="bold" if i == j else "normal")

plt.tight_layout()
out_png = REPORTS_DIR / "e1_confusion_matrix.png"
plt.savefig(out_png, dpi=200)
plt.close()
print(f"Confusion matrix plot saved to: {out_png}")
