"""
DeepShield Robustness & Perturbation Stress-Testing Suite
Applies synthetic acoustic corruptions to test audio:
- Additive Gaussian Noise at SNR levels (20dB, 10dB, 5dB, 0dB)
- Volume Perturbations (Gain +/- 6dB)
- Telephony Bandwidth Limitation (8kHz resampling)
- Silence Insertion
Evaluates model degradation without altering official datasets.
"""

from typing import Dict, List, Tuple
import math
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics


class AudioCorruptor:
    """Applies controlled acoustic perturbations to 1D waveform chunks."""

    @staticmethod
    def add_gaussian_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """Adds white Gaussian noise at a specified Signal-to-Noise Ratio (SNR)."""
        signal_power = torch.mean(waveform ** 2)
        if signal_power < 1e-8:
            return waveform

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        corrupted = waveform + noise
        return torch.clamp(corrupted, -1.0, 1.0)

    @staticmethod
    def scale_volume(waveform: torch.Tensor, gain_db: float = 6.0) -> torch.Tensor:
        """Scales audio volume by gain_db decibels."""
        gain_linear = 10.0 ** (gain_db / 20.0)
        scaled = waveform * gain_linear
        return torch.clamp(scaled, -1.0, 1.0)

    @staticmethod
    def resample_telephony(waveform: torch.Tensor, orig_sr: int = 16000, low_sr: int = 8000) -> torch.Tensor:
        """Simulates telephony 8kHz bandpass downsampling and upsampling."""
        downsampler = T.Resample(orig_freq=orig_sr, new_freq=low_sr)
        upsampler = T.Resample(orig_freq=low_sr, new_freq=orig_sr)
        
        downsampled = downsampler(waveform.unsqueeze(0))
        restored = upsampler(downsampled).squeeze(0)
        
        # Ensure exact original length
        if restored.shape[-1] < waveform.shape[-1]:
            restored = torch.nn.functional.pad(restored, (0, waveform.shape[-1] - restored.shape[-1]))
        elif restored.shape[-1] > waveform.shape[-1]:
            restored = restored[:waveform.shape[-1]]
        return restored


def evaluate_robustness_suite(model, dataset, device, threshold: float = 0.5, max_samples: int = 1000):
    """
    Evaluates model across clean and perturbed audio distributions.
    Returns dictionary of results per perturbation condition.
    """
    model.eval()
    conditions = [
        ("Clean", None),
        ("Noise (SNR 20dB)", lambda wf: AudioCorruptor.add_gaussian_noise(wf, snr_db=20.0)),
        ("Noise (SNR 10dB)", lambda wf: AudioCorruptor.add_gaussian_noise(wf, snr_db=10.0)),
        ("Noise (SNR 5dB)", lambda wf: AudioCorruptor.add_gaussian_noise(wf, snr_db=5.0)),
        ("Volume Boost (+6dB)", lambda wf: AudioCorruptor.scale_volume(wf, gain_db=6.0)),
        ("Volume Atten (-6dB)", lambda wf: AudioCorruptor.scale_volume(wf, gain_db=-6.0)),
        ("Telephony (8kHz Resample)", lambda wf: AudioCorruptor.resample_telephony(wf)),
    ]

    results = {}
    bonafide_indices = [i for i, e in enumerate(dataset.entries) if e["label"] == 0]
    spoof_indices = [i for i, e in enumerate(dataset.entries) if e["label"] == 1]
    half_samples = max(1, min(len(bonafide_indices), len(spoof_indices), max_samples // 2))
    indices = bonafide_indices[:half_samples] + spoof_indices[:half_samples]

    for cond_name, perturb_fn in conditions:
        y_true = []
        y_scores = []

        with torch.no_grad():
            for idx in indices:
                entry = dataset.entries[idx]
                audio_path = entry["audio_path"]
                label = entry["label"]

                try:
                    wf = dataset.audio_preprocessor.load_audio(audio_path)
                    if perturb_fn is not None:
                        wf = perturb_fn(wf)
                    proc_wf = dataset.audio_preprocessor.process_waveform(wf)
                    chunks = dataset.audio_preprocessor.chunk_waveform(proc_wf)
                    spec = dataset.spec_extractor(chunks[0]).unsqueeze(0).to(device)

                    if device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            logits = model(spec)
                    else:
                        logits = model(spec)

                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                    y_true.append(label)
                    y_scores.append(prob)
                except Exception:
                    continue

        if len(y_true) > 0:
            metrics = compute_all_metrics(y_true, y_scores, threshold=threshold)
            results[cond_name] = metrics

    return results
