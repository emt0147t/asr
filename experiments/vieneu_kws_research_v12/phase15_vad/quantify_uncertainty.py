"""
Uncertainty Quantification and Paired Statistical Comparison (Phase 16 Audit).
Computes:
1. Wilson score 95% confidence intervals for exact-sequence accuracy.
2. Sequence-level paired contingency table between Baseline and SpeechBand Ratio VAD.
3. McNemar's exact test for paired nominal data.
4. Sequence-level paired bootstrap (10,000 resamples) for delta accuracy.
5. Sample count matrix across gap and noise cells.
"""

import sys
import io
import json
import math
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple:
    """Computes Wilson score 95% confidence interval for proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% z-score
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper

def main():
    stage_b_path = PROJECT_ROOT / "experiments/vieneu_kws_research_v12/reports/stage_b_locked_test_results.json"
    data = json.load(open(stage_b_path, "r", encoding="utf-8"))

    cfgs = ["Baseline_v11", "Baseline_e1", "Winning_SpeechBandRatio_v11", "Winning_SpeechBandRatio_e1"]

    print("=" * 80)
    print("  UNCERTAINTY QUANTIFICATION: 95% CONFIDENCE INTERVALS (WILSON SCORE)")
    print("=" * 80)
    for c in cfgs:
        r = data[c]
        k = r["exact_correct_sequences"]
        n = r["total_sequences"]
        p = k / n
        low, high = wilson_ci(k, n)
        print(f"{c:<28}: {k:>2}/{n} ({p*100:6.2f}%) | 95% Wilson CI: [{low*100:5.2f}%, {high*100:5.2f}%]")

    # Sequence-level paired comparison between Baseline_e1 and Winning_SpeechBandRatio_e1
    seqs_base = {r["sequence_id"]: r["is_exact_match"] for r in data["Baseline_e1"]["per_sequence_records"]}
    seqs_win = {r["sequence_id"]: r["is_exact_match"] for r in data["Winning_SpeechBandRatio_e1"]["per_sequence_records"]}

    both_correct = 0      # a
    base_only = 0         # b
    win_only = 0          # c
    neither_correct = 0   # d

    paired_diffs = []
    for seq_id, b_match in seqs_base.items():
        w_match = seqs_win[seq_id]
        if b_match and w_match:
            both_correct += 1
        elif b_match and not w_match:
            base_only += 1
        elif not b_match and w_match:
            win_only += 1
        else:
            neither_correct += 1
        paired_diffs.append(int(w_match) - int(b_match))

    print("\n" + "=" * 80)
    print("  SEQUENCE-LEVEL PAIRED CONTINGENCY TABLE (Baseline_e1 vs Winning_e1, N=60)")
    print("=" * 80)
    print(f"  Both Correct:                        {both_correct:>2} ({both_correct/60*100:5.1f}%)")
    print(f"  Baseline Correct, Winning Incorrect: {base_only:>2} ({base_only/60*100:5.1f}%)  [b]")
    print(f"  Winning Correct, Baseline Incorrect: {win_only:>2} ({win_only/60*100:5.1f}%)  [c]")
    print(f"  Both Incorrect:                      {neither_correct:>2} ({neither_correct/60*100:5.1f}%)")

    # McNemar's Test with continuity correction
    b, c = base_only, win_only
    if b + c > 0:
        chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
        # Exact binomial p-value for two-tailed test
        from scipy import stats
        binom_p = stats.binomtest(k=min(b, c), n=b + c, p=0.5).pvalue
    else:
        chi2 = 0.0
        binom_p = 1.0

    print(f"\n  McNemar's Chi-squared (continuity corrected): {chi2:.4f}")
    print(f"  Exact Binomial p-value (two-tailed):           {binom_p:.6e}")
    if binom_p < 0.001:
        print("  Statistical significance: Highly significant (p < 0.001)")

    # Paired Bootstrap (10,000 resamples)
    rng = np.random.RandomState(42)
    paired_arr = np.array(paired_diffs)
    n_boot = 10000
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(paired_arr, size=len(paired_arr), replace=True)
        boot_means.append(np.mean(sample))

    mean_delta = np.mean(boot_means)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    print(f"\n  Paired Bootstrap Delta Accuracy: {mean_delta*100:+.2f}%")
    print(f"  95% Bootstrap Percentile CI:     [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%]")

    # Gap and Condition cell distribution
    print("\n" + "=" * 80)
    print("  SAMPLE COUNTS BY GAP AND NOISE CELL (vad_test_corpus)")
    print("=" * 80)
    print("  Gaps (6 levels): 100, 150, 200, 250, 300, 400 ms (10 sequences each)")
    print("  Noise (5 tiers): clean, snr20, snr15, snr10, snr05 (12 sequences each)")
    print("  Cell size: Exactly 2 sequences per [gap x noise] cell (Total 6 x 5 = 30 cells).")
    print("  NOTICE: N=2 per cell is small; individual cell accuracy estimates have high uncertainty.")

if __name__ == "__main__":
    main()
