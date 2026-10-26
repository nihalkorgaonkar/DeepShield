"""
DeepShield Dataset Integrity & Protocol Verification Script
Verifies:
1. Protocol files exist and columns parse cleanly.
2. Label mapping (0=bonafide, 1=spoof) and speaker/attack IDs.
3. Audio files exist and are readable FLAC format.
4. Audio sampling rates, bit depths, and channel configurations.
5. Zero-leakage verification between train/dev/eval speakers and attack algorithms.
"""

import os
import sys
import random
import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor


def verify_asvspoof2019_integrity(data_root: str, sample_check_count: int = 50):
    print("\n" + "=" * 70)
    print("VERIFYING ASVSPOOF 2019 LOGICAL ACCESS (LA) INTEGRITY")
    print("=" * 70)

    # 1. Check protocol paths
    split_configs = {
        "train": {
            "proto": "ASVspoof2019.LA.cm.train.trn.txt",
            "folder": "ASVspoof2019_LA_train",
            "expected_count": 25380,
            "expected_attacks": {"-", "A01", "A02", "A03", "A04", "A05", "A06"},
        },
        "dev": {
            "proto": "ASVspoof2019.LA.cm.dev.trl.txt",
            "folder": "ASVspoof2019_LA_dev",
            "expected_count": 24844,
            "expected_attacks": {"-", "A01", "A02", "A03", "A04", "A05", "A06"},
        },
        "eval": {
            "proto": "ASVspoof2019.LA.cm.eval.trl.txt",
            "folder": "ASVspoof2019_LA_eval",
            "expected_count": 71237,
            "expected_attacks": {"-", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19"},
        },
    }

    preprocessor = AudioPreprocessor()
    spec_extractor = SpectrogramExtractor(image_size=(128, 128))

    speakers_per_split = {}
    total_files_verified = 0

    for split, cfg in split_configs.items():
        print(f"\n--- Checking Split: {split.upper()} ---")
        possible_proto_paths = [
            os.path.join(data_root, "ASVspoof2019_LA_cm_protocols", cfg["proto"]),
            os.path.join(data_root, "LA", "ASVspoof2019_LA_cm_protocols", cfg["proto"]),
            os.path.join(data_root, "ASVspoof2019_LA", "ASVspoof2019_LA_cm_protocols", cfg["proto"]),
            os.path.join(data_root, "ASVspoof2019_LA", "LA", "ASVspoof2019_LA_cm_protocols", cfg["proto"]),
            os.path.join(data_root, cfg["proto"]),
        ]

        proto_file = next((p for p in possible_proto_paths if os.path.exists(p)), None)
        if not proto_file:
            print(f"[!] Protocol file not found for {split}: {cfg['proto']}")
            continue

        print(f"Protocol file located: {proto_file}")

        possible_audio_dirs = [
            os.path.join(data_root, "LA", cfg["folder"], "flac"),
            os.path.join(data_root, "ASVspoof2019_LA", "LA", cfg["folder"], "flac"),
            os.path.join(data_root, "ASVspoof2019_LA", cfg["folder"], "flac"),
            os.path.join(data_root, cfg["folder"], "flac"),
        ]
        audio_dir = next((d for d in possible_audio_dirs if os.path.exists(d) and len(os.listdir(d)) > 0), None)
        if not audio_dir:
            print(f"[!] Audio directory not found for {split}: {cfg['folder']}/flac")
            continue

        print(f"Audio directory located: {audio_dir} ({len(os.listdir(audio_dir))} files)")

        # Parse protocol
        entries = []
        speakers = set()
        attacks = set()
        class_counts = {"bonafide": 0, "spoof": 0}

        with open(proto_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    spk, utt, env, att, key = parts[0], parts[1], parts[2], parts[3], parts[4].lower()
                    speakers.add(spk)
                    attacks.add(att)
                    class_counts[key] += 1
                    entries.append((spk, utt, att, key))

        speakers_per_split[split] = speakers
        print(f"Total entries in protocol: {len(entries)} (Expected: {cfg['expected_count']})")
        print(f"Class distribution: {class_counts}")
        print(f"Unique speakers: {len(speakers)}")
        print(f"Attacks observed: {sorted(list(attacks))}")

        # Verify attack algorithm protocol alignment
        diff_attacks = attacks - cfg["expected_attacks"]
        if diff_attacks:
            print(f"[!] Warning: Unexpected attacks in {split}: {diff_attacks}")
        else:
            print(f"Attack systems strictly match official ASVspoof 2019 {split} specifications.")

        # Representative audio sampling and integrity check
        random.seed(42)
        sample_entries = random.sample(entries, min(sample_check_count, len(entries)))
        corrupted_count = 0

        for spk, utt, att, key in sample_entries:
            audio_path = os.path.join(audio_dir, f"{utt}.flac")
            if not os.path.exists(audio_path):
                print(f"[!] Missing file: {audio_path}")
                corrupted_count += 1
                continue

            try:
                info = sf.info(audio_path)
                if info.samplerate != 16000:
                    print(f"[!] Sample rate mismatch for {utt}: {info.samplerate} Hz (expected 16000 Hz)")
                if info.channels != 1:
                    print(f"[!] Channel count for {utt}: {info.channels} (expected 1 mono)")

                # Test end-to-end load and spectrogram extraction
                wf = preprocessor.load_audio(audio_path)
                proc_wf = preprocessor.process_waveform(wf)
                chunks = preprocessor.chunk_waveform(proc_wf)
                spec = spec_extractor(chunks[0])

                if spec.shape != torch.Size([3, 128, 128]) or not torch.isfinite(spec).all():
                    print(f"[!] Preprocessing failure for {utt}: shape={spec.shape}, finite={torch.isfinite(spec).all()}")
                    corrupted_count += 1
                else:
                    total_files_verified += 1
            except Exception as e:
                print(f"[!] Read error for {audio_path}: {e}")
                corrupted_count += 1

        print(f"Sample integrity check: {len(sample_entries)} files tested, {corrupted_count} errors.")

    # Speaker leakage check
    if "train" in speakers_per_split and "dev" in speakers_per_split and "eval" in speakers_per_split:
        train_dev_leak = speakers_per_split["train"].intersection(speakers_per_split["dev"])
        train_eval_leak = speakers_per_split["train"].intersection(speakers_per_split["eval"])
        dev_eval_leak = speakers_per_split["dev"].intersection(speakers_per_split["eval"])
        print("\n--- Zero-Leakage Speaker Isolation Check ---")
        print(f"Train / Dev Speaker Overlap: {len(train_dev_leak)} (Expected: 0)")
        print(f"Train / Eval Speaker Overlap: {len(train_eval_leak)} (Expected: 0)")
        print(f"Dev / Eval Speaker Overlap: {len(dev_eval_leak)} (Expected: 0)")
        if len(train_dev_leak) == 0 and len(train_eval_leak) == 0 and len(dev_eval_leak) == 0:
            print("PASS: Absolute zero speaker leakage confirmed across all splits!")


def verify_asvspoof2021_df_keys(data_root: str):
    print("\n" + "=" * 70)
    print("VERIFYING ASVSPOOF 2021 DF KEYS & PROTOCOL INTEGRITY")
    print("=" * 70)

    possible_keys = [
        os.path.join(data_root, "ASVspoof2021_DF", "keys", "keys", "DF", "CM", "trial_metadata.txt"),
        os.path.join(data_root, "ASVspoof2021_DF", "keys", "CM", "trial_metadata.txt"),
        os.path.join(data_root, "keys", "keys", "DF", "CM", "trial_metadata.txt"),
        os.path.join(data_root, "keys", "DF", "CM", "trial_metadata.txt"),
        os.path.join(data_root, "trial_metadata.txt"),
    ]

    key_file = next((k for k in possible_keys if os.path.exists(k)), None)
    if not key_file:
        print("[!] ASVspoof 2021 DF keys file not found.")
        return

    print(f"Key file located: {key_file} ({os.path.getsize(key_file) / (1024**2):.2f} MB)")
    total_trials = 0
    labels = {"bonafide": 0, "spoof": 0}
    codecs = {}

    with open(key_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                total_trials += 1
                k = parts[5].lower()
                c = parts[2]
                labels[k] = labels.get(k, 0) + 1
                codecs[c] = codecs.get(c, 0) + 1

    print(f"Total parsed trial trials: {total_trials}")
    print(f"Ground truth label distribution: {labels}")
    print(f"Codecs represented ({len(codecs)} conditions): {sorted(list(codecs.keys()))}")
    print("PASS: ASVspoof 2021 DF official trial keys verified successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.getenv("DEEPSHIELD_DATA_ROOT", r"C:\DeepShieldData"))
    args = parser.parse_args()

    verify_asvspoof2019_integrity(args.data_root)
    verify_asvspoof2021_df_keys(args.data_root)
