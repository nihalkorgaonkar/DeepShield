"""
Parallel Multi-Worker Downloader for ASVspoof 2019 LA
Source: Edinburgh DataShare (https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip)
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
EXPECTED_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
EXPECTED_BYTES = 7640952520
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB


def compute_md5(file_path, chunk_size=8192 * 1024):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_chunk(url, dest_path, chunk_idx, start_byte, end_byte, max_retries=5):
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code in (200, 206):
                content = r.content
                expected_len = end_byte - start_byte + 1
                if len(content) == expected_len:
                    with open(dest_path, "r+b") as f:
                        f.seek(start_byte)
                        f.write(content)
                    return chunk_idx, len(content)
        except Exception as e:
            time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to download chunk {chunk_idx} [{start_byte}-{end_byte}] after {max_retries} retries.")


def parallel_download(url, dest_path, total_bytes, workers=16):
    progress_file = dest_path + ".progress.json"
    
    # Pre-allocate file if not exists
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) != total_bytes:
        print(f"Allocating {total_bytes / (1024**3):.2f} GB file on disk...")
        with open(dest_path, "wb") as f:
            f.seek(total_bytes - 1)
            f.write(b"\0")

    completed_chunks = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                completed_chunks = set(json.load(f))
            print(f"Resuming download: {len(completed_chunks)} chunks already downloaded.")
        except Exception:
            completed_chunks = set()

    chunks = []
    chunk_idx = 0
    start = 0
    while start < total_bytes:
        end = min(start + CHUNK_SIZE - 1, total_bytes - 1)
        if chunk_idx not in completed_chunks:
            chunks.append((chunk_idx, start, end))
        chunk_idx += 1
        start = end + 1

    total_chunks = chunk_idx
    print(f"Total Chunks: {total_chunks} (Remaining: {len(chunks)}, Workers: {workers})")

    initial_bytes = (total_chunks - len(chunks)) * CHUNK_SIZE
    with tqdm(
        total=total_bytes,
        initial=initial_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc="Parallel LA.zip",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_chunk, url, dest_path, c_idx, s, e): c_idx
                for c_idx, s, e in chunks
            }
            save_counter = 0
            for future in as_completed(futures):
                c_idx, n_bytes = future.result()
                completed_chunks.add(c_idx)
                pbar.update(n_bytes)
                save_counter += 1
                if save_counter % 20 == 0:
                    with open(progress_file, "w") as f:
                        json.dump(list(completed_chunks), f)

    if os.path.exists(progress_file):
        os.remove(progress_file)
    print("Parallel download completed.")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting files"):
            zf.extract(member, extract_to)
    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Download & Extract ASVspoof 2019 LA dataset.")
    parser.add_argument("--data_root", type=str, default=os.getenv("DEEPSHIELD_DATA_ROOT", r"C:\DeepShieldData"), help="Data destination directory")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent download workers")
    parser.add_argument("--skip_download", action="store_true", help="Skip download if zip already present")
    parser.add_argument("--skip_md5", action="store_true", help="Skip MD5 check")
    parser.add_argument("--skip_extract", action="store_true", help="Skip extraction")
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    zip_dest = os.path.join(args.data_root, "LA.zip")
    extract_dest = os.path.join(args.data_root, "ASVspoof2019_LA")
    os.makedirs(extract_dest, exist_ok=True)

    if not args.skip_download:
        if not os.path.exists(zip_dest) or os.path.getsize(zip_dest) != EXPECTED_BYTES:
            print(f"Starting parallel download of LA.zip ({EXPECTED_BYTES / (1024**3):.2f} GB) to {zip_dest}...")
            parallel_download(URL, zip_dest, EXPECTED_BYTES, workers=args.workers)
        else:
            print(f"{zip_dest} already downloaded ({os.path.getsize(zip_dest)} bytes).")

    if not args.skip_md5 and os.path.exists(zip_dest):
        print("Verifying MD5 checksum...")
        calculated_md5 = compute_md5(zip_dest)
        print(f"Calculated MD5: {calculated_md5}")
        print(f"Expected MD5:   {EXPECTED_MD5}")
        if calculated_md5 != EXPECTED_MD5:
            raise ValueError(f"MD5 Checksum Mismatch! Got {calculated_md5}, expected {EXPECTED_MD5}")
        print("MD5 Checksum matches perfectly!")

    if not args.skip_extract:
        extract_zip(zip_dest, extract_dest)
        print("ASVspoof 2019 LA extraction verified.")


if __name__ == "__main__":
    main()
