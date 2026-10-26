"""
Download and Extract ASVspoof 2021 Deepfake (DF) Dataset and Keys
Source: Zenodo (https://zenodo.org/records/4835108) & Official ASVspoof Consortium
"""

import os
import sys
import hashlib
import tarfile
import argparse
import requests
from tqdm import tqdm

KEYS_URL = "https://raw.githubusercontent.com/asvspoof-challenge/2021/main/eval-package/keys/DF/CM/trial_metadata.txt"
KEYS_TAR_URL = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"

ZENODO_FILES = {
    "ASVspoof2021_DF_eval_part00.tar.gz": {
        "url": "https://zenodo.org/api/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz/content",
        "md5": "4f2cbae07cf3ede2a1dde3b8d2ee55ea",
        "size": 8632617637,
    },
    "ASVspoof2021_DF_eval_part01.tar.gz": {
        "url": "https://zenodo.org/api/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz/content",
        "md5": "1578c89ab433c2b60b1dce93bdf8fbec",
        "size": 8617565989,
    },
    "ASVspoof2021_DF_eval_part02.tar.gz": {
        "url": "https://zenodo.org/api/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz/content",
        "md5": "5497a35f0126e94a1d7a7d26db57b4f7",
        "size": 8610543209,
    },
    "ASVspoof2021_DF_eval_part03.tar.gz": {
        "url": "https://zenodo.org/api/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz/content",
        "md5": "42b7512ba2943e98a32a53c9608cf83c",
        "size": 8652686077,
    },
}


def compute_md5(file_path, chunk_size=8192 * 1024):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url, dest_path, expected_bytes=None):
    temp_path = dest_path + ".part"
    initial_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    headers = {}
    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"
        print(f"Resuming download from byte {initial_bytes}...")

    mode = "ab" if initial_bytes > 0 else "wb"

    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP Error {r.status_code} while downloading {url}")

        total_length = int(r.headers.get("content-length", 0)) + initial_bytes
        if expected_bytes:
            total_length = expected_bytes

        with open(temp_path, mode) as f, tqdm(
            desc=os.path.basename(dest_path),
            initial=initial_bytes,
            total=total_length,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    os.replace(temp_path, dest_path)
    print(f"Download completed: {dest_path}")


def download_keys(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    keys_txt_path = os.path.join(dest_dir, "trial_metadata.txt")
    if os.path.exists(keys_txt_path) and os.path.getsize(keys_txt_path) > 1000:
        print(f"ASVspoof 2021 DF keys already present at: {keys_txt_path}")
        return

    print("Downloading official ASVspoof 2021 DF trial metadata keys...")
    try:
        r = requests.get(KEYS_URL, timeout=30)
        if r.status_code == 200:
            with open(keys_txt_path, "wb") as f:
                f.write(r.content)
            print(f"Keys successfully saved to {keys_txt_path} ({len(r.content)} bytes)")
            return
    except Exception as e:
        print(f"GitHub keys download failed: {e}. Falling back to tar.gz...")

    # Fallback to tar.gz
    tar_dest = os.path.join(dest_dir, "DF-keys-full.tar.gz")
    download_file(KEYS_TAR_URL, tar_dest)
    with tarfile.open(tar_dest, "r:gz") as tar:
        tar.extractall(dest_dir)
    print("Extracted evaluation keys from tar.gz.")


def extract_tar(tar_path, extract_to):
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)
    print(f"Extraction complete for {os.path.basename(tar_path)}")


def main():
    parser = argparse.ArgumentParser(description="Download & Extract ASVspoof 2021 DF dataset.")
    parser.add_argument("--data_root", type=str, default=os.getenv("DEEPSHIELD_DATA_ROOT", r"C:\DeepShieldData"), help="Data root directory")
    parser.add_argument("--keys_only", action="store_true", help="Download official trial metadata keys only")
    parser.add_argument("--parts", nargs="+", default=["part00", "part01", "part02", "part03"], help="Subset of parts to download")
    parser.add_argument("--skip_md5", action="store_true", help="Skip MD5 check")
    args = parser.parse_args()

    df_dir = os.path.join(args.data_root, "ASVspoof2021_DF")
    os.makedirs(df_dir, exist_ok=True)

    # 1. Always ensure keys are downloaded
    keys_dir = os.path.join(df_dir, "keys")
    download_keys(keys_dir)

    if args.keys_only:
        print("Keys-only mode specified. Finished.")
        return

    # 2. Download and extract specified parts
    for fname, meta in ZENODO_FILES.items():
        part_name = fname.split("_")[-1].replace(".tar.gz", "")
        if part_name not in args.parts:
            continue

        dest_archive = os.path.join(df_dir, fname)
        if not os.path.exists(dest_archive) or os.path.getsize(dest_archive) != meta["size"]:
            print(f"\nDownloading {fname} ({meta['size'] / (1024**3):.2f} GB)...")
            download_file(meta["url"], dest_archive, expected_bytes=meta["size"])
        else:
            print(f"\n{fname} already present ({os.path.getsize(dest_archive)} bytes).")

        if not args.skip_md5 and os.path.exists(dest_archive):
            print(f"Verifying MD5 for {fname}...")
            actual_md5 = compute_md5(dest_archive)
            print(f"Computed MD5: {actual_md5}")
            print(f"Expected MD5: {meta['md5']}")
            if actual_md5 != meta["md5"]:
                raise ValueError(f"MD5 mismatch for {fname}! Got {actual_md5}, expected {meta['md5']}")
            print("MD5 matches perfectly!")

        extract_tar(dest_archive, df_dir)

    print("\nASVspoof 2021 DF processing complete.")


if __name__ == "__main__":
    main()
