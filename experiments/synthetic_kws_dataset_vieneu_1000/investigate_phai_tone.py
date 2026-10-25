"""
Part B Investigation: PHAI Tone Detection Diagnostic Audit
Deep investigation into why audit_tone_diagnostics.py reported 34.8% dipping rate on PHAI.
Analyzes denominator, definition of dipping contour, unvoiced frames, octave errors,
pitch range limits, interior minima, dialect-specific tone realizations, and comparison
with TRAI, LEN, and XUONG diagnostic rules.
"""

import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

sys.path.insert(0, DATASET_DIR)
from audit_tone_diagnostics import track_f0

def investigate_phai():
    print("=" * 70)
    print("Part B: Investigating PHAI F0 & Tone Detection Dynamics")
    print("=" * 70)

    df = pd.read_csv(MANIFEST_PATH)
    phai_df = df[df["class_label"] == "PHAI"]
    total_phai = len(phai_df)
    print(f"Total PHAI samples: {total_phai}")

    records = []
    for idx, row in phai_df.iterrows():
        audio_path = os.path.join(BASE_DIR, row["kws_wav_path"])
        audio, sr = sf.read(audio_path)
        f0_arr, energy_arr = track_f0(audio, sr=sr)
        valid_f0 = f0_arr[~np.isnan(f0_arr)]

        n_v = len(valid_f0)
        if n_v >= 4:
            t_third = max(1, n_v // 3)
            f0_start = float(np.mean(valid_f0[:t_third]))
            f0_mid = float(np.mean(valid_f0[t_third : 2 * t_third]))
            f0_end = float(np.mean(valid_f0[-t_third:]))
            f0_slope = f0_end - f0_start

            cond1 = bool(f0_mid < f0_start - 5.0)
            cond2 = bool(f0_end >= f0_mid)
            is_dipping = cond1 and cond2

            # F0 extrema
            min_idx = int(np.argmin(valid_f0))
            max_idx = int(np.argmax(valid_f0))
            f0_min = float(valid_f0[min_idx])
            f0_max = float(valid_f0[max_idx])

            has_interior_min = bool(0 < min_idx < n_v - 1)
            dip_from_start = float(f0_start - f0_min)
            rebound_to_end = float(f0_end - f0_min)

            # Trajectory classification:
            # 1. Classic dipping: interior min, drops >= 5Hz, rebounds >= 5Hz
            # 2. Monotonic falling (Southern / Central merger): f0_start > f0_mid > f0_end
            # 3. Monotonic rising: f0_start < f0_mid < f0_end
            # 4. Level / shallow: std < 10Hz
            if has_interior_min and (dip_from_start >= 5.0) and (rebound_to_end >= 5.0):
                shape_class = "CLASSIC_DIP"
            elif (f0_start > f0_mid + 3.0) and (f0_mid > f0_end - 3.0):
                shape_class = "FALLING_CONTOUR"
            elif (f0_end > f0_mid + 3.0) and (f0_mid > f0_start - 3.0):
                shape_class = "RISING_CONTOUR"
            else:
                shape_class = "FLAT_OR_IRREGULAR"
        else:
            f0_start, f0_mid, f0_end, f0_slope = np.nan, np.nan, np.nan, np.nan
            cond1, cond2, is_dipping = False, False, False
            f0_min, f0_max = np.nan, np.nan
            has_interior_min = False
            dip_from_start, rebound_to_end = np.nan, np.nan
            shape_class = "UNVOICED_OR_INSUFFICIENT"

        records.append({
            "sample_id": row["sample_id"],
            "voice_id": row["voice_id"],
            "region": row["voice_region"],
            "gender": row["voice_gender"],
            "n_voiced_frames": n_v,
            "f0_start": f0_start,
            "f0_mid": f0_mid,
            "f0_end": f0_end,
            "f0_slope": f0_slope,
            "f0_min": f0_min,
            "f0_max": f0_max,
            "cond1_mid_below_start": cond1,
            "cond2_end_above_mid": cond2,
            "is_dipping_strict": is_dipping,
            "has_interior_min": has_interior_min,
            "dip_from_start": dip_from_start,
            "rebound_to_end": rebound_to_end,
            "shape_class": shape_class
        })

    rdf = pd.DataFrame(records)

    n_strict = rdf["is_dipping_strict"].sum()
    rate_strict = rdf["is_dipping_strict"].mean() * 100.0
    print(f"\n1. Strict dipping formula in audit_tone_diagnostics.py:")
    print(f"   Definition: (f0_mid < f0_start - 5.0 Hz) AND (f0_end >= f0_mid)")
    print(f"   Denominator: Exactly {total_phai} samples (all PHAI rows)")
    print(f"   Numerator:   {n_strict} samples classified as strict dipping ({rate_strict:.1f}%)")

    print(f"\n2. Breakdown of the {total_phai - n_strict} samples failing strict dipping:")
    failed = rdf[~rdf["is_dipping_strict"]]
    print(f"   - Failed cond1 only (mid not < start - 5Hz): {(~failed['cond1_mid_below_start'] & failed['cond2_end_above_mid']).sum()}")
    print(f"   - Failed cond2 only (end not >= mid; monotonic falling): {(failed['cond1_mid_below_start'] & ~failed['cond2_end_above_mid']).sum()}")
    print(f"   - Failed both cond1 and cond2: {(~failed['cond1_mid_below_start'] & ~failed['cond2_end_above_mid']).sum()}")
    print(f"   - Fewer than 4 voiced frames: {(failed['n_voiced_frames'] < 4).sum()}")

    print("\n3. Acoustic Shape Class Distribution across all 250 PHAI samples:")
    shape_counts = rdf["shape_class"].value_counts()
    for s, c in shape_counts.items():
        print(f"   - {s:25s}: {c:3d} ({c/total_phai*100:.1f}%)")

    print("\n4. Dialect Regional Realizations of PHAI:")
    regional = rdf.groupby("region").agg(
        total=("sample_id", "count"),
        strict_dipping=("is_dipping_strict", "sum"),
        strict_rate=("is_dipping_strict", lambda x: x.mean() * 100),
        falling=("shape_class", lambda x: (x == "FALLING_CONTOUR").sum()),
        classic_dip=("shape_class", lambda x: (x == "CLASSIC_DIP").sum()),
        mean_slope=("f0_slope", "mean")
    )
    print(regional.to_string())

    print("\n5. Voice Gender Realizations of PHAI:")
    gender_agg = rdf.groupby("gender").agg(
        total=("sample_id", "count"),
        strict_dipping=("is_dipping_strict", "sum"),
        strict_rate=("is_dipping_strict", lambda x: x.mean() * 100),
        mean_f0_min=("f0_min", "mean"),
        mean_slope=("f0_slope", "mean")
    )
    print(gender_agg.to_string())

    # Comparison with TRAI, LEN, XUONG rules
    print("\n6. Consistency Check across Diagnostic Rules for Other Classes:")
    print("   - TRAI and XUONG: Evaluated by mean f0_slope (f0_end - f0_start > 0). No 'rising_rate' threshold was enforced.")
    print("   - LEN: Evaluated by f0_slope_mean close to zero (-6.8 Hz). No flat threshold enforced.")
    print("   - PHAI: The ONLY class where a rigid binary heuristic ('is_dipping') was introduced and reported as a percentage.")

    return rdf

if __name__ == "__main__":
    investigate_phai()
