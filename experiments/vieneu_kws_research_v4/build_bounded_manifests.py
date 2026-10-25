import os
import csv

def build_manifests():
    base_dir = r"D:\trainidentificationaudio"
    src_manifest_path = os.path.join(base_dir, "experiments", "vieneu_kws_research_v3", "data", "manifest.csv")
    out_dir = os.path.join(base_dir, "experiments", "vieneu_kws_research_v4", "data")
    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    label_map = {
        "TRAI": 0,
        "PHAI": 1,
        "LEN": 2,
        "XUONG": 3,
        "UNKNOWN": 4,
        "SILENCE": 4
    }

    train_rows = []
    val_rows = []
    test_rows = []
    all_rows = []

    with open(src_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        if "bounded_label" not in fieldnames:
            fieldnames.extend(["bounded_label", "bounded_class_id"])

        for row in reader:
            lbl = row["class_label"]
            if lbl in ["TRAI", "PHAI", "LEN", "XUONG"]:
                bounded_label = lbl
                bounded_class_id = label_map[lbl]
            else:
                bounded_label = "REJECT"
                bounded_class_id = 4
            
            row["dataset_version"] = "research_v4.0_bounded"
            row["bounded_label"] = bounded_label
            row["bounded_class_id"] = bounded_class_id

            split = row["split"]
            all_rows.append(row)
            if split == "train":
                train_rows.append(row)
            elif split == "val":
                val_rows.append(row)
            elif split == "test":
                test_rows.append(row)

    # Write master manifest
    out_master = os.path.join(out_dir, "manifest.csv")
    with open(out_master, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write split manifests
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        split_path = os.path.join(splits_dir, f"{split_name}_manifest.csv")
        with open(split_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {split_path}")

    print(f"Master manifest written with {len(all_rows)} rows to {out_master}")

if __name__ == "__main__":
    build_manifests()
