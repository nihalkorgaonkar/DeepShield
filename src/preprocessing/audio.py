"""
DeepShield Audio Preprocessing Module
Implements deterministic, strictly standardized audio loading, resampling,
mono conversion, silence trimming, peak normalization, and chunking.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF


class AudioPreprocessor:
    """Standardized Audio Preprocessor for DeepShield speech deepfake detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        chunk_duration: float = 4.0,
        hop_duration: float = 2.0,
        trim_silence: bool = True,
        top_db: float = 30.0,
        normalize_peak: bool = True,
        target_peak: float = 0.95,
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)
        self.trim_silence = trim_silence
        self.top_db = top_db
        self.normalize_peak = normalize_peak
        self.target_peak = target_peak

    def load_audio(
        self,
        audio_path_or_buffer: Union[str, bytes],
        target_sr: Optional[int] = None,
    ) -> torch.Tensor:
        """Loads audio file/buffer, ensures mono and target sample rate."""
        target_sr = target_sr or self.sample_rate
        try:
            data, sr = sf.read(audio_path_or_buffer, dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
        except Exception:
            waveform, sr = torchaudio.load(audio_path_or_buffer)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1 and self.mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if sample rate does not match target
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # Shape: (num_samples,)

    def trim_lead_trail_silence(
        self,
        waveform: torch.Tensor,
        top_db: Optional[float] = None,
    ) -> torch.Tensor:
        """Trims leading and trailing silence based on energy threshold (top_db)."""
        if not self.trim_silence:
            return waveform

        top_db = top_db or self.top_db
        if waveform.numel() == 0:
            return waveform

        # Compute energy envelope
        abs_wf = torch.abs(waveform)
        max_val = torch.max(abs_wf)
        if max_val <= 1e-7:
            return waveform  # All silence, return original

        threshold = max_val * (10.0 ** (-top_db / 20.0))
        non_silent_indices = torch.where(abs_wf >= threshold)[0]

        if non_silent_indices.numel() == 0:
            return waveform

        start_idx = non_silent_indices[0].item()
        end_idx = non_silent_indices[-1].item() + 1
        return waveform[start_idx:end_idx]

    def peak_normalize(
        self,
        waveform: torch.Tensor,
        target_peak: Optional[float] = None,
    ) -> torch.Tensor:
        """Normalizes waveform peaks to [-target_peak, target_peak]."""
        if not self.normalize_peak:
            return waveform

        target_peak = target_peak or self.target_peak
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-7:
            waveform = waveform * (target_peak / max_val)
        return torch.clamp(waveform, -1.0, 1.0)

    def process_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Applies silence trimming and peak normalization to a raw 1D waveform."""
        wf = self.trim_lead_trail_silence(waveform)
        wf = self.peak_normalize(wf)
        return wf

    def chunk_waveform(
        self,
        waveform: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Splits waveform into fixed-length chunks (chunk_samples = 4.0s @ 16kHz).
        If waveform is shorter than chunk_samples, pads with zero.
        If longer, creates sliding window slices with hop_samples (2.0s = 50% overlap).
        Ensures complete temporal coverage including trailing segments.
        """
        num_samples = waveform.shape[-1]

        # Case 1: Audio is shorter than chunk duration -> Pad to chunk_samples
        if num_samples < self.chunk_samples:
            pad_amount = self.chunk_samples - num_samples
            padded = torch.nn.functional.pad(waveform, (0, pad_amount), mode="constant", value=0.0)
            return [padded]

        # Case 2: Audio is equal or longer -> Extract sliding chunks
        chunks = []
        start = 0
        while start + self.chunk_samples <= num_samples:
            chunks.append(waveform[start : start + self.chunk_samples])
            start += self.hop_samples

        # Ensure trailing audio is covered if residual remains
        if len(chunks) == 0:
            pad_amount = self.chunk_samples - num_samples
            chunks.append(torch.nn.functional.pad(waveform, (0, pad_amount), mode="constant", value=0.0))
        else:
            last_end = (start - self.hop_samples) + self.chunk_samples
            if last_end < num_samples:
                chunks.append(waveform[-self.chunk_samples:])

        return chunks
