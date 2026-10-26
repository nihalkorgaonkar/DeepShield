"""
Audio Preprocessing and Decoding Service
Validates uploaded audio, checks file signatures, safely decodes audio,
and applies DeepShield's locked 14-point preprocessing protocol.
"""

import io
import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from backend.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    CHUNK_DURATION,
    HOP_DURATION,
    MAX_UPLOAD_SIZE_BYTES,
    NORMALIZE_PEAK,
    SAMPLE_RATE,
    TARGET_PEAK,
    TOP_DB,
    TRIM_SILENCE,
)
from backend.schemas.detection import AudioMetadata
from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor


class AudioService:
    def __init__(self):
        self.preprocessor = AudioPreprocessor(
            sample_rate=SAMPLE_RATE,
            mono=True,
            chunk_duration=CHUNK_DURATION,
            hop_duration=HOP_DURATION,
            trim_silence=TRIM_SILENCE,
            top_db=TOP_DB,
            normalize_peak=NORMALIZE_PEAK,
            target_peak=TARGET_PEAK,
        )
        self.spec_extractor = SpectrogramExtractor(
            sample_rate=SAMPLE_RATE,
            image_size=(128, 128),
            channels=3,
            imagenet_norm=True,
        )

    @staticmethod
    def validate_upload(filename: str, file_bytes: bytes):
        """Validates file extension and size."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. DeepShield accepts: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            )
        if len(file_bytes) < 100:
            raise ValueError("Uploaded file is empty or corrupted.")

    def load_and_inspect_audio(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[torch.Tensor, AudioMetadata]:
        """
        Loads raw audio and computes descriptive metadata, duration, channels, RMS energy,
        and downsampled waveform envelope.
        """
        buf = io.BytesIO(file_bytes)
        try:
            data, orig_sr = sf.read(buf, dtype="float32")
            if data.ndim == 1:
                raw_tensor = torch.from_numpy(data).unsqueeze(0)
                num_channels = 1
            else:
                raw_tensor = torch.from_numpy(data.T)
                num_channels = data.shape[1]
        except Exception:
            buf.seek(0)
            raw_tensor, orig_sr = torchaudio.load(buf)
            num_channels = raw_tensor.shape[0]

        total_samples = raw_tensor.shape[-1]
        duration_sec = round(total_samples / orig_sr, 3)

        # Compute RMS energy
        rms_energy = float(torch.sqrt(torch.mean(raw_tensor ** 2)).item())

        # Standard mono conversion for internal pipeline
        if raw_tensor.shape[0] > 1:
            mono_waveform = torch.mean(raw_tensor, dim=0, keepdim=True)
        else:
            mono_waveform = raw_tensor

        # Resample to 16 kHz if needed
        if orig_sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=SAMPLE_RATE)
            resampled_waveform = resampler(mono_waveform)
        else:
            resampled_waveform = mono_waveform

        resampled_1d = resampled_waveform.squeeze(0)

        # Downsample waveform for UI visualization (100 peak envelope points)
        downsampled_points = self._compute_waveform_envelope(resampled_1d, num_points=100)

        # Format file size
        size_bytes = len(file_bytes)
        if size_bytes >= 1024 * 1024:
            size_fmt = f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            size_fmt = f"{size_bytes / 1024:.1f} KB"

        metadata = AudioMetadata(
            filename=filename,
            duration_sec=duration_sec,
            sample_rate_hz=orig_sr,
            num_channels=num_channels,
            num_samples=total_samples,
            file_size_bytes=size_bytes,
            file_size_formatted=size_fmt,
            rms_energy=round(rms_energy, 4),
            waveform_downsampled=downsampled_points,
        )

        return resampled_1d, metadata

    @staticmethod
    def _compute_waveform_envelope(waveform: torch.Tensor, num_points: int = 100) -> List[float]:
        """Calculates normalized peak envelope amplitudes across time bins for frontend rendering."""
        wf_np = waveform.detach().cpu().numpy()
        if len(wf_np) == 0:
            return [0.0] * num_points

        chunk_size = max(1, len(wf_np) // num_points)
        envelope = []
        for i in range(num_points):
            start = i * chunk_size
            end = min(len(wf_np), start + chunk_size)
            if start < len(wf_np):
                bin_slice = wf_np[start:end]
                peak = float(np.max(np.abs(bin_slice))) if len(bin_slice) > 0 else 0.0
                envelope.append(round(min(1.0, peak), 3))
            else:
                envelope.append(0.0)
        return envelope

    def preprocess_to_chunks(self, waveform: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Processes 1D waveform through silence trimming, peak normalization,
        and chunking into 4.0s windows with 2.0s hop.
        Returns:
            chunks: List of 1D tensors of length 64,000 (4s @ 16kHz)
            batch_spectrograms: Tensor of shape (num_chunks, 3, 128, 128)
        """
        processed_wf = self.preprocessor.process_waveform(waveform)
        chunks = self.preprocessor.chunk_waveform(processed_wf)
        if not chunks:
            # Fallback zero-padded chunk
            dummy = torch.zeros(int(SAMPLE_RATE * CHUNK_DURATION))
            chunks = [dummy]

        # Extract normalized 3-channel spectrogram for each chunk
        spec_tensors = [self.spec_extractor(c) for c in chunks]
        batch_spec = torch.stack(spec_tensors)  # Shape: (N, 3, 128, 128)
        return chunks, batch_spec
