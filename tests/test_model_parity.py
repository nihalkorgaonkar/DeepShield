"""
Test Model Parity with Identical Input
Verifies:
1. One identical 128x128 spectrogram input is generated
2. Both EfficientNet-B0 and ViT consume this exact input
3. Both models produce valid [batch_size, 2] logits
4. Zero preprocessing differences between CNN and Transformer architectures
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor
from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield


def test_model_parity():
    print("=" * 60)
    print("RUNNING MODEL PARITY TEST (CNN vs ViT)")
    print("=" * 60)

    # 1. Generate synthetic audio
    sample_rate = 16000
    duration = 4.0
    num_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, num_samples)
    audio = 0.5 * torch.sin(2 * torch.pi * 300 * t) + 0.5 * torch.sin(2 * torch.pi * 1200 * t)

    # 2. Process to 128x128 spectrogram
    preprocessor = AudioPreprocessor(sample_rate=sample_rate)
    spec_extractor = SpectrogramExtractor(sample_rate=sample_rate, image_size=(128, 128))

    chunk = preprocessor.chunk_waveform(audio)[0]
    single_spectrogram = spec_extractor(chunk)  # (3, 128, 128)

    batch_input = single_spectrogram.unsqueeze(0).repeat(4, 1, 1, 1)  # (4, 3, 128, 128)
    print(f"Batch Input Spectrogram Shape: {batch_input.shape}")

    # 3. Instantiate EfficientNet-B0
    print("\n--- Testing EfficientNet-B0 ---")
    efficientnet = EfficientNetDeepShield(num_classes=2, pretrained=True)
    efficientnet.eval()
    with torch.no_grad():
        cnn_out = efficientnet(batch_input)
    print(f"EfficientNet Output Shape: {cnn_out.shape}")
    print(f"EfficientNet Logits:\n{cnn_out}")
    assert cnn_out.shape == (4, 2), f"Expected (4, 2), got {cnn_out.shape}"

    # 4. Instantiate ViT
    print("\n--- Testing Vision Transformer (ViT-B/16 with 128x128 pos interpolation) ---")
    vit = VisionTransformerDeepShield(model_name="vit_base_patch16_224", num_classes=2, pretrained=True, img_size=(128, 128))
    vit.eval()
    with torch.no_grad():
        vit_out = vit(batch_input)
    print(f"ViT Output Shape: {vit_out.shape}")
    print(f"ViT Logits:\n{vit_out}")
    assert vit_out.shape == (4, 2), f"Expected (4, 2), got {vit_out.shape}"

    print("\nPASS: Both EfficientNet-B0 and ViT successfully processed the identical spectrogram!")
    return True


if __name__ == "__main__":
    test_model_parity()
