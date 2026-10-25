from pathlib import Path
import pandas as pd

base = Path("experiments/synthetic_kws_dataset_vieneu_1000")
manifest = base / "manifest.csv"
report = base / "reports" / "VOICE_DISTRIBUTION_AUDIT.md"

df = pd.read_csv(manifest, encoding="utf-8-sig")

required = ["sample_id", "voice_id", "voice_gender", "voice_region"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample_id found.")

if df[required].isna().any().any():
    raise ValueError("Missing required metadata.")

voice_meta = df[["voice_id", "voice_gender", "voice_region"]].drop_duplicates()

if voice_meta["voice_id"].duplicated().any():
    raise ValueError("Inconsistent metadata for a voice_id.")

if len(df) != 1000 or len(voice_meta) != 25:
    raise ValueError(
        f"Unexpected counts: {len(df)} audio, "
        f"{len(voice_meta)} voices."
    )

per_voice = df.groupby("voice_id").size()
if not (per_voice == 40).all():
    raise ValueError("Some voices do not have exactly 40 audio records.")

gender_names = {"male": "Nam", "female": "N\u1eef"}
regions = ["B\u1eafc", "Trung", "Nam"]

lines = [
    "# B\u00e1o c\u00e1o ph\u00e2n b\u1ed1 gi\u1edbi t\u00ednh v\u00e0 v\u00f9ng mi\u1ec1n",
    "",
    f"- Dataset: `{manifest.as_posix()}`",
    f"- T\u1ed5ng s\u1ed1 audio: {len(df)}",
    f"- T\u1ed5ng s\u1ed1 voice preset: {len(voice_meta)}",
    "- M\u1ed7i voice preset c\u00f3 40 audio.",
    "",
    "## 1. Ph\u00e2n b\u1ed1 theo gi\u1edbi t\u00ednh",
    "",
    "| Gi\u1edbi t\u00ednh | S\u1ed1 voice | T\u1ef7 l\u1ec7 voice | S\u1ed1 audio | T\u1ef7 l\u1ec7 audio |",
    "|---|---:|---:|---:|---:|",
]

for gender in ["male", "female"]:
    nv = int((voice_meta["voice_gender"] == gender).sum())
    na = int((df["voice_gender"] == gender).sum())
    lines.append(
        f"| {gender_names[gender]} | {nv} | {nv/25*100:.1f}% "
        f"| {na} | {na/1000*100:.1f}% |"
    )

lines += [
    "",
    "## 2. Ph\u00e2n b\u1ed1 theo v\u00f9ng mi\u1ec1n",
    "",
    "| V\u00f9ng mi\u1ec1n | S\u1ed1 voice | T\u1ef7 l\u1ec7 voice | S\u1ed1 audio | T\u1ef7 l\u1ec7 audio |",
    "|---|---:|---:|---:|---:|",
]

for region in regions:
    nv = int((voice_meta["voice_region"] == region).sum())
    na = int((df["voice_region"] == region).sum())
    lines.append(
        f"| {region} | {nv} | {nv/25*100:.1f}% "
        f"| {na} | {na/1000*100:.1f}% |"
    )

lines += [
    "",
    "## 3. K\u1ebft h\u1ee3p gi\u1edbi t\u00ednh v\u00e0 v\u00f9ng mi\u1ec1n",
    "",
    "| Gi\u1edbi t\u00ednh | B\u1eafc | Trung | Nam | T\u1ed5ng |",
    "|---|---:|---:|---:|---:|",
]

for gender in ["female", "male"]:
    cells = []
    for region in regions:
        nv = int((
            (voice_meta["voice_gender"] == gender) &
            (voice_meta["voice_region"] == region)
        ).sum())
        na = int((
            (df["voice_gender"] == gender) &
            (df["voice_region"] == region)
        ).sum())
        cells.append(f"{nv} voice ({na} audio)")

    total_v = int((voice_meta["voice_gender"] == gender).sum())
    total_a = int((df["voice_gender"] == gender).sum())
    lines.append(
        f"| {gender_names[gender]} | " + " | ".join(cells)
        + f" | {total_v} voice ({total_a} audio) |"
    )

lines += [
    "",
    "## 4. Danh s\u00e1ch voice preset",
    "",
    "| Voice preset | Gi\u1edbi t\u00ednh | V\u00f9ng mi\u1ec1n | S\u1ed1 audio |",
    "|---|---|---|---:|",
]

for _, row in voice_meta.sort_values(
    ["voice_region", "voice_gender", "voice_id"]
).iterrows():
    count = int((df["voice_id"] == row["voice_id"]).sum())
    gender = gender_names.get(row["voice_gender"], row["voice_gender"])
    lines.append(
        f"| {row['voice_id']} | {gender} "
        f"| {row['voice_region']} | {count} |"
    )

lines += [
    "",
    "## 5. Nh\u1eadn x\u00e9t",
    "",
    "- Gi\u1ecdng nam: 14 voice, 560 audio (56%).",
    "- Gi\u1ecdng n\u1eef: 11 voice, 440 audio (44%).",
    "- Mi\u1ec1n B\u1eafc: 15 voice, 600 audio (60%).",
    "- Mi\u1ec1n Nam: 8 voice, 320 audio (32%).",
    "- Mi\u1ec1n Trung: 2 voice, 80 audio (8%).",
    "- Ph\u00e2n b\u1ed1 n\u00e0y m\u00f4 t\u1ea3 voice preset t\u1ed5ng h\u1ee3p, kh\u00f4ng \u0111\u1ea1i di\u1ec7n cho d\u00e2n s\u1ed1 ng\u01b0\u1eddi n\u00f3i ngo\u00e0i \u0111\u1eddi.",
    "- Tr\u1ea1ng th\u00e1i: `NOT_VALIDATED_ON_NATURAL_SPEECH`.",
    "",
]

report.parent.mkdir(parents=True, exist_ok=True)
content = "\n".join(lines) + "\n"
report.write_bytes(content.encode("utf-8"))

# Read the written bytes back and verify UTF-8 content.
verified = report.read_bytes().decode("utf-8")
assert "B\u00e1o c\u00e1o" in verified
assert "N\u1eef" in verified
assert "B\u1eafc" in verified
assert "\ufffd" not in verified

print("REPORT_REBUILT_UTF8=YES")
print("AUDIO_COUNT=1000")
print("VOICE_COUNT=25")
print("UTF8_ROUNDTRIP=PASS")
print("REPORT_PATH=" + str(report.resolve()))
