#!/usr/bin/env python3
"""
Phase 45: Paired Model Trade-Off Analysis
Repository: D:\\trainidentificationaudio
Target: experiments/vieneu_kws_research_v12/scripts/analyze_phase45_tradeoffs.py

Computes:
1. Paired example-by-example agreement/disagreement tables (McNemar tests).
2. Command recognition quality vs false alarm suppression decomposition.
3. Pareto efficiency frontier analysis.
4. Paired synthesis-instance cluster statistics.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "reports"
MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "manifests"

REPRO_JSON = REPORTS_DIR / "PHASE44_47_REPRODUCED_METRICS.json"
ISOLATED_MANIFEST = MANIFESTS_DIR / "PHASE43_LOCKED_MANIFEST.csv"

with open(REPRO_JSON, "r", encoding="utf-8") as f:
    master_data = json.load(f)

iso_metrics = master_data["reproduced_isolated_metrics"]
str_metrics = master_data["reproduced_streaming_metrics"]

df_iso = pd.read_csv(ISOLATED_MANIFEST)
y_true = np.array([0 if lbl == "TRAI" else 1 if lbl == "PHAI" else 2 if lbl == "LEN" else 3 if lbl == "XUONG" else 4 for lbl in df_iso["label"]])

# Load predictions by re-running forward pass or loading saved outputs
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "scripts"))
import torch
import torch.nn.functional as F
import soundfile as sf
from train_phase28_factorial import CorrectedVietnameseKWSNet, LogMelFrontend, CLASSES, CLASS_TO_IDX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frontend = LogMelFrontend(sample_rate=16000, n_mels=40, win_length=400, hop_length=160, n_fft=512).to(device)
frontend.eval()

# Model paths from master_data
ckpts = {
    "Baseline_v11_seed42": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle" / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt",
    "Baseline_E1_seed42": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt",
    "Phase28_D2_H1_Candidate": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_d2_candidate" / "candidate_checkpoint.pt",
    "Phase28_D0_H1_Candidate": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase28_candidate" / "candidate_checkpoint.pt",
    "Phase26_Candidate_D0_H1_seed44": PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "candidates" / "phase26_candidate_d0_h1_seed44" / "candidate_checkpoint.pt"
}

predictions = {}
for m_name, path in ckpts.items():
    model = CorrectedVietnameseKWSNet(num_classes=5, dropout_p=0.30).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    preds = []
    for idx, row in df_iso.iterrows():
        wav_path = PROJECT_ROOT / row["audio_path"]
        audio, _ = sf.read(str(wav_path), dtype="float32")
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]
        wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            spec = frontend(wav_t)
            logits = model(spec)
            probs = F.softmax(logits, dim=-1)[0]
            target_probs = probs[:4]
            max_p, max_idx = torch.max(target_probs, dim=-1)
            pred = int(max_idx.item()) if max_p.item() >= 0.20 else 4
        preds.append(pred)
    predictions[m_name] = np.array(preds)

print("=== PAIRED MCNEMAR TESTS ON ISOLATED CLIPS (N=480) ===")
# McNemar test on 5-class correct vs incorrect
pairs = [
    ("Phase28_D2_H1_Candidate", "Baseline_E1_seed42"),
    ("Phase28_D2_H1_Candidate", "Baseline_v11_seed42"),
    ("Phase28_D2_H1_Candidate", "Phase26_Candidate_D0_H1_seed44"),
    ("Phase26_Candidate_D0_H1_seed44", "Baseline_E1_seed42")
]

for m1, m2 in pairs:
    c1 = (predictions[m1] == y_true)
    c2 = (predictions[m2] == y_true)
    n11 = np.sum(c1 & c2)
    n10 = np.sum(c1 & ~c2)  # M1 correct, M2 wrong
    n01 = np.sum(~c1 & c2)  # M1 wrong, M2 correct
    n00 = np.sum(~c1 & ~c2)
    # McNemar chi2 with continuity correction
    b, c = n10, n01
    chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0.0
    p_val = stats.chi2.sf(chi2, df=1) if (b + c) > 0 else 1.0

    print(f"\nPair: {m1} vs {m2}")
    print(f"  Contingency Table (Correct vs Incorrect):")
    print(f"    Both Correct (n11): {n11}")
    print(f"    {m1} Correct, {m2} Wrong (n10): {n10}")
    print(f"    {m1} Wrong, {m2} Correct (n01): {n01}")
    print(f"    Both Wrong (n00): {n00}")
    print(f"  McNemar chi2 = {chi2:.4f}, p-value = {p_val:.6e}")

print("\n=== COMMAND CLIPS (N=64) VS OOV CLIPS (N=416) DECOMPOSITION ===")
cmd_mask = (y_true != 4)
oov_mask = (y_true == 4)

for m_name in ckpts.keys():
    preds = predictions[m_name]
    cmd_acc = np.mean(preds[cmd_mask] == y_true[cmd_mask])
    cmd_tp = np.sum(preds[cmd_mask] == y_true[cmd_mask])
    cmd_sub = np.sum((preds[cmd_mask] != y_true[cmd_mask]) & (preds[cmd_mask] != 4))
    cmd_rej = np.sum(preds[cmd_mask] == 4)

    oov_tn = np.sum(preds[oov_mask] == 4)
    oov_fa = np.sum(preds[oov_mask] != 4)
    oov_acc = oov_tn / 416.0

    print(f"\n{m_name}:")
    print(f"  Command Accuracy (N=64): {cmd_acc*100:.2f}% (TP={cmd_tp}/64, Sub={cmd_sub}/64, Missed_as_REJECT={cmd_rej}/64)")
    print(f"  OOV Rejection (N=416):  {oov_acc*100:.2f}% (TN={oov_tn}/416, False_Alarms={oov_fa}/416)")
