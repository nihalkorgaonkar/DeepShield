"""
DeepShield Spectrogram Preprocessing Module
Implements deterministic, standardized Log-Mel Spectrogram extraction,
spatial grid formatting (128x128 / 224x224), 3-channel RGB broadcasting,
and ImageNet normalization.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class SpectrogramExtractor(nn.Module):
    """
    Standardized Log-Mel Spectrogram extractor for DeepShield.
    Converts 1D audio waveform chunks into 3-channel 2D spectrogram tensors.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        image_size: Tuple[int, int] = (128, 128),
        channels: int = 3,
        imagenet_norm: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.image_size = tuple(image_size)
        self.channels = channels
        self.imagenet_norm = imagenet_norm

        # MelSpectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

        # Amplitude to DB conversion
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

        # ImageNet normalization parameters
        self.register_buffer("norm_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Processes 1D or 2D audio waveform chunks into normalized spectrograms.
        Args:
            waveform: Tensor of shape (num_samples,) or (batch_size, num_samples)
        Returns:
            spectrogram: Tensor of shape (channels, height, width) or (batch_size, channels, height, width)
        """
        is_batched = waveform.dim() == 2
        if not is_batched:
            waveform = waveform.unsqueeze(0)  # Shape: (1, num_samples)

        # Compute Mel Spectrogram -> Shape: (batch, n_mels, time_frames)
        mel_spec = self.mel_transform(waveform)

        # Convert power to decibels (log-mel)
        log_mel = self.amplitude_to_db(mel_spec)

        # Ensure numerical stability (no NaN / Inf)
        log_mel = torch.nan_to_num(log_mel, nan=-80.0, posinf=0.0, neginf=-80.0)

        # Min-Max scale per spectrogram to [0.0, 1.0]
        batch_size = log_mel.shape[0]
        min_val = log_mel.reshape(batch_size, -1).min(dim=1)[0].reshape(batch_size, 1, 1, 1)
        max_val = log_mel.reshape(batch_size, -1).max(dim=1)[0].reshape(batch_size, 1, 1, 1)
        range_val = max_val - min_val
        range_val = torch.where(range_val < 1e-6, torch.ones_like(range_val), range_val)

        log_mel_4d = log_mel.unsqueeze(1)  # Shape: (batch, 1, n_mels, time_frames)
        scaled_spec = (log_mel_4d - min_val) / range_val

        # Resize to target image resolution (e.g. 128x128)
        if scaled_spec.shape[-2:] != self.image_size:
            resized_spec = F.interpolate(
                scaled_spec,
                size=self.image_size,
                mode="bicubic",
                align_corners=False,
            )
        else:
            resized_spec = scaled_spec

        # Clamp to [0, 1]
        resized_spec = torch.clamp(resized_spec, 0.0, 1.0)

        # 3-channel broadcast for ImageNet pretrained backbones
        if self.channels == 3:
            spec_rgb = resized_spec.repeat(1, 3, 1, 1)
        else:
            spec_rgb = resized_spec

        # ImageNet normalization
        if self.imagenet_norm:
            norm_mean = self.norm_mean.to(spec_rgb.device, dtype=spec_rgb.dtype)
            norm_std = self.norm_std.to(spec_rgb.device, dtype=spec_rgb.dtype)
            spec_out = (spec_rgb - norm_mean) / norm_std
        else:
            spec_out = spec_rgb

        if not is_batched:
            return spec_out.squeeze(0)  # Shape: (3, H, W)
        return spec_out  # Shape: (batch, 3, H, W)
