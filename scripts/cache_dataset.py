"""
DeepShield Authoritative Multi-Chunk Dataset Caching Pipeline
Directly invokes authoritative AudioPreprocessor and SpectrogramExtractor modules.
Extracts:
1. 16 kHz mono audio
2. 30 dB energy-based silence trimming
3. 0.95 peak normalization
4. 4.0-second sliding chunks with 2.0-second hop (50% overlap, 100% temporal coverage)
5. 128-mel log-mel spectrogram (n_fft=1024, win=400, hop=160, 0-8000Hz)
6. Bicubic interpolation to 128x128 (or 224x224 for Exp E)
7. 3-channel broadcast with ImageNet normalization
8. Hierarchical chunk serialization and manifest.jsonl generation with zero label leakage.
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor


def _process_utterance_for_cache(item):
    """
    Worker function: takes an utterance item, loads audio, runs authoritative
    preprocessing, extracts multi-chunk spectrograms, saves them, and returns manifest entries.
    """
    audio_path, utt_id, label, speaker_id, env_id, attack_id, split, utt_out_dir, image_size, spec_kwargs = item
    os.makedirs(utt_out_dir, exist_ok=True)

    try:
        ap = AudioPreprocessor(
            sample_rate=16000,
            mono=True,
            chunk_duration=4.0,
            hop_duration=2.0,
            trim_silence=True,
            top_db=30.0,
            normalize_peak=True,
            target_peak=0.95,
        )
        se = SpectrogramExtractor(image_size=image_size, **spec_kwargs)

        wf = ap.load_audio(audio_path)
        wf_proc = ap.process_waveform(wf)
        chunks = ap.chunk_waveform(wf_proc)

        chunk_manifests = []
        for c_idx, chunk in enumerate(chunks):
            chunk_file = os.path.join(utt_out_dir, f"chunk_{c_idx:03d}.pt")
            if not os.path.exists(chunk_file):
                spec = se(chunk)
                torch.save(spec.float(), chunk_file)

            chunk_manifests.append({
                "utterance_id": utt_id,
                "chunk_idx": c_idx,
                "total_chunks": len(chunks),
                "chunk_file": os.path.abspath(chunk_file),
                "label": int(label),
                "speaker_id": speaker_id,
                "environment_id": env_id,
                "attack_id": attack_id,
                "dataset_name": "ASVspoof2019_LA",
                "split": split,
            })
        return chunk_manifests
    except Exception as e:
        print(f"Error processing {utt_id}: {e}", file=sys.stderr)
        return []


def cache_split(data_root, split, cache_root, image_size=(128, 128), num_workers=8):
    split_map = {
        "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
        "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
        "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
    }
    proto_name, folder_name = split_map[split]

    proto_path = None
    for p in [
        os.path.join(data_root, "LA", "ASVspoof2019_LA_cm_protocols", proto_name),
        os.path.join(data_root, "ASVspoof2019_LA_cm_protocols", proto_name),
        os.path.join(data_root, proto_name),
    ]:
        if os.path.exists(p):
            proto_path = p
            break

    audio_dir = None
    for ad in [
        os.path.join(data_root, "LA", folder_name, "flac"),
        os.path.join(data_root, folder_name, "flac"),
        os.path.join(data_root, "LA", folder_name),
    ]:
        if os.path.exists(ad) and len(os.listdir(ad)) > 100:
            audio_dir = ad
            break

    split_cache_dir = os.path.join(cache_root, f"{image_size[0]}x{image_size[1]}", split)
    os.makedirs(split_cache_dir, exist_ok=True)

    items = []
    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id, utt_id, env_id, attack_id, key = parts[0], parts[1], parts[2], parts[3], parts[4]
                label = 0 if key.lower() == "bonafide" else 1
                audio_path = os.path.join(audio_dir, f"{utt_id}.flac")
                utt_out_dir = os.path.join(split_cache_dir, utt_id)
                items.append((
                    audio_path,
                    utt_id,
                    label,
                    speaker_id,
                    env_id,
                    attack_id,
                    split,
                    utt_out_dir,
                    image_size,
                    {},
                ))

    print(f"\n=======================================================")
    print(f"Caching ASVspoof 2019 LA [{split.upper()}] ({image_size[0]}x{image_size[1]})")
    print(f"Total Utterances: {len(items)}")
    print(f"Destination: {split_cache_dir}")
    print(f"=======================================================")

    manifest_entries = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_process_utterance_for_cache, items, chunksize=32), total=len(items), desc=f"Caching {split}"):
            manifest_entries.extend(result)

    # Write manifest.jsonl
    manifest_path = os.path.join(split_cache_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    # Write cache_metadata.json
    metadata = {
        "source_dataset": "ASVspoof 2019 Logical Access",
        "split": split,
        "preprocessing_version": "1.0-locked",
        "image_size": list(image_size),
        "sample_rate": 16000,
        "chunk_duration_sec": 4.0,
        "hop_duration_sec": 2.0,
        "trim_silence": True,
        "top_db": 30.0,
        "normalize_peak": True,
        "target_peak": 0.95,
        "n_fft": 1024,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 128,
        "fmin": 0.0,
        "fmax": 8000.0,
        "channels": 3,
        "imagenet_norm": True,
        "n_utterances": len(items),
        "n_chunks": len(manifest_entries),
        "manifest_file": os.path.abspath(manifest_path),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_path = os.path.join(split_cache_dir, "cache_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Completed {split}: {len(items)} utterances -> {len(manifest_entries)} chunks.")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Authoritative Multi-Chunk Dataset Caching Pipeline")
    parser.add_argument("--data_root", type=str, default=r"C:\DeepShieldData\ASVspoof2019_LA")
    parser.add_argument("--cache_root", type=str, default=r"C:\DeepShieldData\cache_spec")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in args.splits:
        cache_split(
            data_root=args.data_root,
            split=split,
            cache_root=args.cache_root,
            image_size=(args.image_size, args.image_size),
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
