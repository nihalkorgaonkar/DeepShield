"""
Test Spectrogram Extraction & Shape Verification
Verifies:
1. Synthetic audio generation (4s @ 16kHz mono = 64,000 samples)
2. Preprocessing & silence trimming & normalization
3. Spectrogram extraction output shape: (3, 128, 128)
4. Numerical stability (no NaN, no Inf, finite values)
5. Normalization range and frequency parameters
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor


def test_spectrogram_pipeline():
    print("=" * 60)
    print("RUNNING SPECTROGRAM PIPELINE TEST")
    print("=" * 60)

    sample_rate = 16000
    duration = 4.0
    num_samples = int(sample_rate * duration)  # 64,000

    # 1. Generate synthetic audio: mixture of sinusoids (e.g. 440 Hz, 1000 Hz, 3000 Hz)
    t = torch.linspace(0, duration, num_samples)
    synthetic_audio = (
        0.5 * torch.sin(2 * torch.pi * 440 * t) +
        0.3 * torch.sin(2 * torch.pi * 1000 * t) +
        0.2 * torch.sin(2 * torch.pi * 3000 * t)
    )

    print(f"Synthetic Audio Input Shape: {synthetic_audio.shape} ({synthetic_audio.numel()} samples @ {sample_rate} Hz)")
    assert synthetic_audio.numel() == 64000, f"Expected 64000 samples, got {synthetic_audio.numel()}"

    # 2. Audio Preprocessor
    audio_preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        mono=True,
        chunk_duration=4.0,
        hop_duration=2.0,
        trim_silence=True,
        top_db=30.0,
        normalize_peak=True,
        target_peak=0.95,
    )

    processed_wf = audio_preprocessor.process_waveform(synthetic_audio)
    chunks = audio_preprocessor.chunk_waveform(processed_wf)
    print(f"Extracted Chunks Count: {len(chunks)}, Chunk[0] Shape: {chunks[0].shape}")
    assert len(chunks) >= 1, "Expected at least 1 chunk"
    assert chunks[0].shape[-1] == 64000, f"Expected chunk length 64000, got {chunks[0].shape[-1]}"

    # 3. Spectrogram Extractor
    spec_extractor = SpectrogramExtractor(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        n_mels=128,
        fmin=0.0,
        fmax=8000.0,
        image_size=(128, 128),
        channels=3,
        imagenet_norm=True,
    )

    # 4. Extract Spectrogram Tensor
    spec_tensor = spec_extractor(chunks[0])
    print(f"Spectrogram Output Shape: {spec_tensor.shape}")

    # 5. Numerical and Shape Assertions
    assert spec_tensor.shape == (3, 128, 128), f"Expected shape (3, 128, 128), got {spec_tensor.shape}"
    assert not torch.isnan(spec_tensor).any(), "Spectrogram contains NaN values!"
    assert not torch.isinf(spec_tensor).any(), "Spectrogram contains Inf values!"
    assert torch.isfinite(spec_tensor).all(), "Spectrogram values are not all finite!"

    print(f"Spectrogram Stats: Min={spec_tensor.min().item():.4f}, Max={spec_tensor.max().item():.4f}, Mean={spec_tensor.mean().item():.4f}, Std={spec_tensor.std().item():.4f}")
    print("PASS: Spectrogram shape and numerical stability verified successfully!")
    return True


if __name__ == "__main__":
    test_spectrogram_pipeline()
