"""
Prepares derived manifests for the 4-class synthetic KWS baseline smoke test.
Filters strictly records where source == 'calibration_acoustic_generator_v1'.
Verifies that all 1,000 files exist and that speaker-independent splits are preserved.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import pandas as pd
from src.acquisition.campaign_validator import validate_split_manifests_preflight, validate_audio_file, compute_file_sha256

def prepare_splits(output_dir: Path = Path("experiments/synthetic_kws_baseline")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = Path("data/manifests/source_registry_v2.csv")
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry: {registry_path}")
        
    df = pd.read_csv(registry_path)
    calib = df[df["source"] == "calibration_acoustic_generator_v1"].copy()
    
    if len(calib) != 1000:
        raise ValueError(f"Expected exactly 1,000 calibration records, got {len(calib)}")
        
    # Verify exactly 250 per class
    counts = calib["class_label"].value_counts().to_dict()
    for cmd in ("TRAI", "PHAI", "LEN", "XUONG"):
        if counts.get(cmd, 0) != 250:
            raise ValueError(f"Expected 250 records for {cmd}, got {counts.get(cmd, 0)}")
            
    # Verify all files exist on disk
    for idx, row in calib.iterrows():
        p = Path(row["processed_path"])
        if not p.exists():
            raise FileNotFoundError(f"Missing audio file: {p}")
            
    # Separate by split
    df_train = calib[calib["split"] == "train"].copy()
    df_val = calib[calib["split"] == "validation"].copy()
    df_test = calib[calib["split"] == "test"].copy()
    
    # Preflight split validation
    is_valid, errs, details = validate_split_manifests_preflight(
        df_train, df_val, df_test, validation_profile="benchmark"
    )
    if not is_valid:
        raise ValueError(f"Preflight validation failed on synthetic splits: {errs}")
        
    train_path = output_dir / "train_manifest.csv"
    val_path = output_dir / "val_manifest.csv"
    test_path = output_dir / "test_manifest.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Successfully generated synthetic manifests in {output_dir}:")
    print(f"  Train: {len(df_train)} records -> {train_path}")
    print(f"  Val:   {len(df_val)} records -> {val_path}")
    print(f"  Test:  {len(df_test)} records -> {test_path}")
    print(f"  Speaker disjointness: {details['speaker_status']} (Train: {details['train_speaker_count']}, Val: {details['val_speaker_count']}, Test: {details['test_speaker_count']} speakers)")
    
    return train_path, val_path, test_path

if __name__ == "__main__":
    prepare_splits()
