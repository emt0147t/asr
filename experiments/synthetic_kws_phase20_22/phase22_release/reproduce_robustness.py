"""
Reproducible Entry Point 2: Held-out Profiles and Counterfactuals
Workspace: D:\\trainidentificationaudio

Generates and evaluates 288 synthetic clips across 24 newly sampled held-out speaker profiles
and tests 50 matched counterfactual pairs.
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from experiments.synthetic_kws_phase20_22.phase21_robustness.phase21_generate_heldout import (
    generate_heldout_dataset,
    evaluate_heldout_dataset
)

if __name__ == "__main__":
    out_dir = Path("experiments/synthetic_kws_phase20_22/phase21_robustness")
    ckpt_path = Path("experiments/synthetic_kws_phase13_3_16/model_evaluation/checkpoints/v5_2_tcresnet14_seed_42.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Running reproducible held-out profile evaluation...")
    df, manifest_p = generate_heldout_dataset(out_dir)
    res = evaluate_heldout_dataset(df, out_dir, ckpt_path, device)
    
    eval_json = out_dir / "phase21_heldout_eval.json"
    with open(eval_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Reproducible robustness run complete. Overall Accuracy: {res['overall_accuracy']*100:.2f}%")
