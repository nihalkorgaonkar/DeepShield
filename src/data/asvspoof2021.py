"""
DeepShield ASVspoof 2021 Deepfake (DF) Evaluation Dataset Pipeline
Parses official ASVspoof 2021 DF trial files and keys for cross-dataset generalization evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import Dataset

from ..preprocessing.audio import AudioPreprocessor
from ..preprocessing.spectrogram import SpectrogramExtractor


class ASVspoof2021DFDataset(Dataset):
    """
    Official ASVspoof 2021 Deepfake (DF) Evaluation PyTorch Dataset.
    Used strictly for external cross-codec and cross-transmission generalization benchmarking.
    """

    def __init__(
        self,
        data_root: str,
        keys_path: Optional[str] = None,
        trials_path: Optional[str] = None,
        fraction: float = 1.0,
        seed: int = 42,
        image_size: Tuple[int, int] = (128, 128),
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.data_root = data_root
        self.fraction = min(max(fraction, 0.0), 1.0)
        self.seed = seed
        self.image_size = tuple(image_size)

        self.audio_preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.spec_extractor = SpectrogramExtractor(sample_rate=sample_rate, image_size=image_size)

        self.entries = []
        self._load_keys(keys_path, trials_path)

    def _load_keys(self, keys_path: Optional[str], trials_path: Optional[str]):
        """Parses official 2021 DF evaluation metadata/keys."""
        possible_keys = [
            keys_path,
            os.path.join(self.data_root, "keys", "keys", "DF", "CM", "trial_metadata.txt"),
            os.path.join(self.data_root, "keys", "CM", "trial_metadata.txt"),
            os.path.join(self.data_root, "keys", "trial_metadata.txt"),
            os.path.join(self.data_root, "trial_metadata.txt"),
        ]

        resolved_key = None
        for k in possible_keys:
            if k and os.path.exists(k):
                resolved_key = k
                break

        possible_audio_dirs = [
            os.path.join(self.data_root, "ASVspoof2021_DF_eval", "flac"),
            os.path.join(self.data_root, "flac"),
            os.path.join(self.data_root, "ASVspoof2021_DF_eval"),
            self.data_root,
        ]
        audio_dir = None
        for ad in possible_audio_dirs:
            if os.path.exists(ad):
                audio_dir = ad
                break
        if audio_dir is None:
            audio_dir = possible_audio_dirs[0]

        if resolved_key is None or not os.path.exists(resolved_key):
            return

        with open(resolved_key, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    speaker_id = parts[0]
                    utt_id = parts[1]
                    codec = parts[2]
                    source = parts[3]
                    attack_id = parts[4]
                    key = parts[5].lower()
                    label = 0 if key == "bonafide" else 1
                    audio_path = os.path.join(audio_dir, f"{utt_id}.flac")
                    self.entries.append({
                        "speaker_id": speaker_id,
                        "utterance_id": utt_id,
                        "codec": codec,
                        "source": source,
                        "attack_id": attack_id,
                        "key": key,
                        "label": label,
                        "audio_path": audio_path,
                    })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        entry = self.entries[idx]
        audio_path = entry["audio_path"]

        try:
            waveform = self.audio_preprocessor.load_audio(audio_path)
            processed_wf = self.audio_preprocessor.process_waveform(waveform)
            chunks = self.audio_preprocessor.chunk_waveform(processed_wf)
            spec = self.spec_extractor(chunks[0])
        except Exception:
            dummy_chunk = torch.zeros(int(self.audio_preprocessor.chunk_samples))
            spec = self.spec_extractor(dummy_chunk)

        return {
            "spectrogram": spec,
            "label": torch.tensor(entry["label"], dtype=torch.long),
            "utterance_id": entry["utterance_id"],
            "codec": entry.get("codec", "unknown"),
            "dataset_name": "ASVspoof2021_DF",
            "split": "eval",
        }
