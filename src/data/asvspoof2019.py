"""
DeepShield ASVspoof 2019 Logical Access (LA) Dataset Pipeline
Supports:
- Authoritative multi-chunk processing (4.0s chunks, 2.0s hop = 50% overlap, 30 dB silence trimming, 0.95 peak norm)
- Hierarchical chunk cache loading from manifest.jsonl with zero disk overhead
- Dual dataset operation modes:
  - mode="chunk": for high-throughput chunk-level model training
  - mode="utterance": for multi-chunk probability averaging during evaluation/inference
- Deterministic stratified subsampling (--fraction)
- Inverse class-frequency weighting
"""

from typing import Dict, List, Optional, Tuple, Union
import json
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np

from ..preprocessing.audio import AudioPreprocessor
from ..preprocessing.spectrogram import SpectrogramExtractor


class ASVspoof2019Dataset(Dataset):
    """
    Official ASVspoof 2019 Logical Access PyTorch Dataset.
    Standardized for both EfficientNet-B0 and Vision Transformer architectures.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",  # "train", "dev", or "eval"
        fraction: float = 1.0,
        seed: int = 42,
        image_size: Tuple[int, int] = (128, 128),
        sample_rate: int = 16000,
        chunk_duration: float = 4.0,
        hop_duration: float = 2.0,
        trim_silence: bool = True,
        top_db: float = 30.0,
        normalize_peak: bool = True,
        target_peak: float = 0.95,
        cache_dir: Optional[str] = "C:\\DeepShieldData\\cache_spec",
        mode: str = "chunk",  # "chunk" for training, "utterance" for evaluation
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split.lower()
        self.fraction = min(max(fraction, 0.0), 1.0)
        self.seed = seed
        self.image_size = tuple(image_size)
        self.cache_dir = cache_dir
        self.mode = mode.lower()

        # Authoritative Preprocessors
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            mono=True,
            chunk_duration=chunk_duration,
            hop_duration=hop_duration,
            trim_silence=trim_silence,
            top_db=top_db,
            normalize_peak=normalize_peak,
            target_peak=target_peak,
        )
        self.spec_extractor = SpectrogramExtractor(
            sample_rate=sample_rate,
            image_size=image_size,
            channels=3,
            imagenet_norm=True,
        )

        # Split cache path
        if self.cache_dir:
            self.split_cache_dir = os.path.join(self.cache_dir, f"{self.image_size[0]}x{self.image_size[1]}", self.split)
            self.manifest_path = os.path.join(self.split_cache_dir, "manifest.jsonl")
        else:
            self.split_cache_dir = None
            self.manifest_path = None

        self.entries = []
        self._load_data()

    def _load_data(self):
        """Loads dataset entries either from pre-computed cache manifest or raw protocol files."""
        # Try loading from cache manifest if available
        if self.manifest_path and os.path.exists(self.manifest_path):
            manifest_items = []
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        manifest_items.append(json.loads(line))

            if self.mode == "chunk":
                self.entries = manifest_items
                if self.fraction < 1.0 and len(self.entries) > 0:
                    rng = random.Random(self.seed)
                    bonafide = [e for e in self.entries if e["label"] == 0]
                    spoof = [e for e in self.entries if e["label"] == 1]
                    n_b = max(1, int(len(bonafide) * self.fraction))
                    n_s = max(1, int(len(spoof) * self.fraction))
                    rng.shuffle(bonafide)
                    rng.shuffle(spoof)
                    selected = bonafide[:n_b] + spoof[:n_s]
                    rng.shuffle(selected)
                    self.entries = selected
                return
            else:
                # Utterance mode: group chunks by utterance_id
                utt_dict = {}
                for item in manifest_items:
                    u_id = item["utterance_id"]
                    if u_id not in utt_dict:
                        utt_dict[u_id] = {
                            "utterance_id": u_id,
                            "label": item["label"],
                            "speaker_id": item["speaker_id"],
                            "attack_id": item["attack_id"],
                            "dataset_name": item["dataset_name"],
                            "split": self.split,
                            "chunk_files": [],
                        }
                    utt_dict[u_id]["chunk_files"].append(item["chunk_file"])

                self.entries = list(utt_dict.values())
                if self.fraction < 1.0 and len(self.entries) > 0:
                    rng = random.Random(self.seed)
                    bonafide = [e for e in self.entries if e["label"] == 0]
                    spoof = [e for e in self.entries if e["label"] == 1]
                    n_b = max(1, int(len(bonafide) * self.fraction))
                    n_s = max(1, int(len(spoof) * self.fraction))
                    rng.shuffle(bonafide)
                    rng.shuffle(spoof)
                    selected = bonafide[:n_b] + spoof[:n_s]
                    rng.shuffle(selected)
                    self.entries = selected
                return

        # Fallback: Load from raw protocol files
        self._load_from_raw_protocol()

    def _load_from_raw_protocol(self):
        split_map = {
            "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
            "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
            "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
        }
        if self.split not in split_map:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of: train, dev, eval.")

        proto_name, folder_name = split_map[self.split]
        possible_proto_paths = [
            os.path.join(self.data_root, "ASVspoof2019_LA_cm_protocols", proto_name),
            os.path.join(self.data_root, "LA", "ASVspoof2019_LA_cm_protocols", proto_name),
            os.path.join(self.data_root, "ASVspoof2019_LA", "ASVspoof2019_LA_cm_protocols", proto_name),
            os.path.join(self.data_root, "ASVspoof2019_LA", "LA", "ASVspoof2019_LA_cm_protocols", proto_name),
            os.path.join(self.data_root, proto_name),
        ]
        proto_path = None
        for p in possible_proto_paths:
            if os.path.exists(p):
                proto_path = p
                break

        possible_audio_dirs = [
            os.path.join(self.data_root, "LA", folder_name, "flac"),
            os.path.join(self.data_root, "ASVspoof2019_LA", "LA", folder_name, "flac"),
            os.path.join(self.data_root, "ASVspoof2019_LA", folder_name, "flac"),
            os.path.join(self.data_root, folder_name, "flac"),
            os.path.join(self.data_root, "LA", folder_name),
            os.path.join(self.data_root, folder_name),
        ]
        audio_dir = None
        for ad in possible_audio_dirs:
            if os.path.exists(ad) and os.path.isdir(ad):
                try:
                    if len(os.listdir(ad)) > 100:
                        audio_dir = ad
                        break
                except Exception:
                    pass

        if audio_dir is None:
            audio_dir = possible_audio_dirs[0]

        if proto_path is None or not os.path.exists(proto_path):
            return

        raw_entries = []
        with open(proto_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id, utt_id, env_id, attack_id, key = parts[0], parts[1], parts[2], parts[3], parts[4]
                    label = 0 if key.lower() == "bonafide" else 1
                    audio_path = os.path.join(audio_dir, f"{utt_id}.flac")
                    raw_entries.append({
                        "speaker_id": speaker_id,
                        "utterance_id": utt_id,
                        "environment_id": env_id,
                        "attack_id": attack_id,
                        "key": key.lower(),
                        "label": label,
                        "audio_path": audio_path,
                        "dataset_name": "ASVspoof2019_LA",
                        "split": self.split,
                    })

        if self.fraction < 1.0 and len(raw_entries) > 0:
            rng = random.Random(self.seed)
            bonafide = [e for e in raw_entries if e["label"] == 0]
            spoof = [e for e in raw_entries if e["label"] == 1]
            n_b = max(1, int(len(bonafide) * self.fraction))
            n_s = max(1, int(len(spoof) * self.fraction))
            rng.shuffle(bonafide)
            rng.shuffle(spoof)
            raw_entries = bonafide[:n_b] + spoof[:n_s]
            rng.shuffle(raw_entries)

        self.entries = raw_entries

    def get_class_weights(self) -> torch.Tensor:
        """Computes inverse-frequency class weights for weighted Cross-Entropy Loss."""
        if not self.entries:
            return torch.tensor([1.0, 1.0])
        labels = [e["label"] for e in self.entries]
        n_bonafide = sum(1 for l in labels if l == 0)
        n_spoof = sum(1 for l in labels if l == 1)
        total = len(labels)
        w0 = total / (2.0 * max(n_bonafide, 1))
        w1 = total / (2.0 * max(n_spoof, 1))
        return torch.tensor([w0, w1], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, List[torch.Tensor]]]:
        entry = self.entries[idx]

        # Case 1: Loaded from Chunk-Mode Manifest
        if "chunk_file" in entry:
            try:
                spec = torch.load(entry["chunk_file"], weights_only=True)
            except Exception:
                dummy_chunk = torch.zeros(int(self.audio_preprocessor.chunk_samples))
                spec = self.spec_extractor(dummy_chunk)
            return {
                "spectrogram": spec,
                "label": torch.tensor(entry["label"], dtype=torch.long),
                "utterance_id": entry["utterance_id"],
                "chunk_idx": entry.get("chunk_idx", 0),
                "total_chunks": entry.get("total_chunks", 1),
                "speaker_id": entry.get("speaker_id", ""),
                "attack_id": entry.get("attack_id", ""),
                "dataset_name": entry.get("dataset_name", "ASVspoof2019_LA"),
                "split": entry.get("split", self.split),
            }

        # Case 2: Loaded from Utterance-Mode Manifest (Multiple chunk files)
        if "chunk_files" in entry:
            specs = []
            for cf in entry["chunk_files"]:
                try:
                    s = torch.load(cf, weights_only=True)
                except Exception:
                    s = self.spec_extractor(torch.zeros(int(self.audio_preprocessor.chunk_samples)))
                specs.append(s)
            stacked_specs = torch.stack(specs, dim=0)  # Shape: [N, 3, H, W]
            return {
                "spectrograms": stacked_specs,
                "spectrogram": specs[0],
                "label": torch.tensor(entry["label"], dtype=torch.long),
                "utterance_id": entry["utterance_id"],
                "n_chunks": len(specs),
                "speaker_id": entry.get("speaker_id", ""),
                "attack_id": entry.get("attack_id", ""),
                "dataset_name": entry.get("dataset_name", "ASVspoof2019_LA"),
                "split": entry.get("split", self.split),
            }

        # Case 3: Raw On-the-Fly Processing
        audio_path = entry.get("audio_path", "")
        try:
            waveform = self.audio_preprocessor.load_audio(audio_path)
            processed_wf = self.audio_preprocessor.process_waveform(waveform)
            chunks = self.audio_preprocessor.chunk_waveform(processed_wf)
            specs = [self.spec_extractor(c) for c in chunks]
        except Exception:
            dummy_chunk = torch.zeros(int(self.audio_preprocessor.chunk_samples))
            specs = [self.spec_extractor(dummy_chunk)]

        stacked_specs = torch.stack(specs, dim=0)  # Shape: [N, 3, H, W]
        return {
            "spectrograms": stacked_specs,
            "spectrogram": specs[0],
            "label": torch.tensor(entry["label"], dtype=torch.long),
            "utterance_id": entry["utterance_id"],
            "n_chunks": len(specs),
            "speaker_id": entry.get("speaker_id", ""),
            "attack_id": entry.get("attack_id", ""),
            "dataset_name": entry.get("dataset_name", "ASVspoof2019_LA"),
            "split": entry.get("split", self.split),
        }


def collate_utterances(batch_list):
    """
    Collate function for safe batched utterance evaluation.
    Concatenates variable-length chunks from B utterances into a single contiguous
    tensor [M, 3, H, W] while preserving utterance IDs, labels, and chunk boundaries.
    """
    all_chunks = []
    chunk_counts = []
    labels = []
    utterance_ids = []
    speaker_ids = []
    attack_ids = []

    for item in batch_list:
        specs = item["spectrograms"]  # [N_i, 3, H, W]
        all_chunks.append(specs)
        chunk_counts.append(specs.size(0))
        lbl = item["label"].item() if isinstance(item["label"], torch.Tensor) else item["label"]
        labels.append(lbl)
        utterance_ids.append(item["utterance_id"])
        speaker_ids.append(item.get("speaker_id", ""))
        attack_ids.append(item.get("attack_id", ""))

    batched_chunks = torch.cat(all_chunks, dim=0)  # Shape: [M, 3, H, W]
    return {
        "batched_chunks": batched_chunks,
        "chunk_counts": chunk_counts,
        "labels": torch.tensor(labels, dtype=torch.long),
        "utterance_ids": utterance_ids,
        "speaker_ids": speaker_ids,
        "attack_ids": attack_ids,
    }

