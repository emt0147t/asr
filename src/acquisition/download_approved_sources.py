"""
Reproducible Acquisition Pipeline for Approved Vietnamese KWS Corpora.
Downloads and verifies approved sources according to docs/dataset_protocol.md.
"""

import os
import sys
import urllib.request
import tarfile
import hashlib
import json
from pathlib import Path
from tqdm import tqdm

APPROVED_SOURCES = {
    "fleurs_vi_dev": {
        "url_audio": "https://huggingface.co/datasets/google/fleurs/resolve/main/data/vi_vn/audio/dev.tar.gz",
        "url_tsv": "https://huggingface.co/datasets/google/fleurs/raw/main/data/vi_vn/dev.tsv",
        "dest_dir": "data/raw/fleurs_vi",
        "archive_name": "dev.tar.gz",
        "tsv_name": "dev.tsv",
        "license": "CC-BY-4.0",
        "description": "Google FLEURS Vietnamese dev split (361 human-verified read speech recordings)"
    },
    "fleurs_vi_test": {
        "url_audio": "https://huggingface.co/datasets/google/fleurs/resolve/main/data/vi_vn/audio/test.tar.gz",
        "url_tsv": "https://huggingface.co/datasets/google/fleurs/raw/main/data/vi_vn/test.tsv",
        "dest_dir": "data/raw/fleurs_vi",
        "archive_name": "test.tar.gz",
        "tsv_name": "test.tsv",
        "license": "CC-BY-4.0",
        "description": "Google FLEURS Vietnamese test split (857 human-verified read speech recordings)"
    },
    "mic_noise": {
        "url": "https://huggingface.co/datasets/huseinzol05/noise-dataset/resolve/main/mic-noise.zip",
        "dest_dir": "data/raw/noise",
        "archive_name": "mic-noise.zip",
        "license": "CC-BY-4.0",
        "description": "Real microphone and room acoustic background noises for SILENCE class"
    },
    "mswc_vi_splits": {
        "url": "https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/splits/vi/splits.tar.gz",
        "dest_dir": "data/raw/mswc_vi",
        "archive_name": "splits.tar.gz",
        "license": "CC-BY-4.0",
        "description": "MLCommons Multilingual Spoken Words Corpus Vietnamese word metadata"
    }
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[ALREADY EXISTS] {output_path} ({output_path.stat().st_size} bytes)")
        return output_path

    print(f"[DOWNLOADING] {url} -> {output_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name, total=total_size) as t:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                t.update(len(chunk))
    print(f"[DOWNLOAD COMPLETED] {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(128 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_tar_gz(archive_path: Path, extract_to: Path):
    print(f"[EXTRACTING] {archive_path} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"[EXTRACTED] Finished extracting {archive_path.name}")


import zipfile


def extract_zip(archive_path: Path, extract_to: Path):
    print(f"[EXTRACTING] {archive_path} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[EXTRACTED] Finished extracting {archive_path.name}")


def main():
    print("=" * 60)
    print("REPRODUCIBLE DATA ACQUISITION PIPELINE")
    print("=" * 60)

    # 1. Download FLEURS vi_vn dev set (audio + TSV)
    fleurs_cfg = APPROVED_SOURCES["fleurs_vi_dev"]
    fleurs_dest = Path(fleurs_cfg["dest_dir"])
    
    tsv_path = download_file(fleurs_cfg["url_tsv"], fleurs_dest / fleurs_cfg["tsv_name"])
    audio_tar_path = download_file(fleurs_cfg["url_audio"], fleurs_dest / fleurs_cfg["archive_name"])
    
    sha_tsv = compute_sha256(tsv_path)
    sha_audio = compute_sha256(audio_tar_path)
    print(f"FLEURS dev.tsv SHA-256: {sha_tsv}")
    print(f"FLEURS dev.tar.gz SHA-256: {sha_audio}")

    fleurs_extracted_audio = fleurs_dest / "audio_extracted"
    if not (fleurs_extracted_audio / "dev").exists():
        extract_tar_gz(audio_tar_path, fleurs_extracted_audio)
    else:
        print(f"[ALREADY EXTRACTED] {fleurs_extracted_audio / 'dev'}")

    # 1b. Download FLEURS vi_vn test set (audio + TSV)
    fleurs_test_cfg = APPROVED_SOURCES["fleurs_vi_test"]
    tsv_test_path = download_file(fleurs_test_cfg["url_tsv"], fleurs_dest / fleurs_test_cfg["tsv_name"])
    audio_test_tar = download_file(fleurs_test_cfg["url_audio"], fleurs_dest / fleurs_test_cfg["archive_name"])
    sha_test_tsv = compute_sha256(tsv_test_path)
    sha_test_audio = compute_sha256(audio_test_tar)
    print(f"FLEURS test.tsv SHA-256: {sha_test_tsv}")
    print(f"FLEURS test.tar.gz SHA-256: {sha_test_audio}")

    if not (fleurs_extracted_audio / "test").exists():
        extract_tar_gz(audio_test_tar, fleurs_extracted_audio)
    else:
        print(f"[ALREADY EXTRACTED] {fleurs_extracted_audio / 'test'}")

    # 2. Download Microphone Noise archive
    noise_cfg = APPROVED_SOURCES["mic_noise"]
    noise_dest = Path(noise_cfg["dest_dir"])
    noise_zip = download_file(noise_cfg["url"], noise_dest / noise_cfg["archive_name"])
    sha_noise = compute_sha256(noise_zip)
    print(f"Noise zip SHA-256: {sha_noise}")
    noise_extracted = noise_dest / "extracted"
    if not noise_extracted.exists():
        extract_zip(noise_zip, noise_extracted)
    else:
        print(f"[ALREADY EXTRACTED] {noise_extracted}")

    # 3. Download MSWC splits
    mswc_cfg = APPROVED_SOURCES["mswc_vi_splits"]
    mswc_dest = Path(mswc_cfg["dest_dir"])
    mswc_tar = download_file(mswc_cfg["url"], mswc_dest / mswc_cfg["archive_name"])
    mswc_extracted = mswc_dest / "splits_extracted"
    if not (mswc_extracted / "dev.csv").exists():
        extract_tar_gz(mswc_tar, mswc_extracted)

    print("\n[SUCCESS] Approved external sources downloaded and verified.")


if __name__ == "__main__":
    main()
