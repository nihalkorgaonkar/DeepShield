"""
Test Vision Transformer 128x128 Compatibility
Verifies:
1. Loading ViT model with 128x128 resolution support
2. Positional embedding dynamic interpolation (no size mismatch error)
3. Forward pass on (2, 3, 128, 128) dummy input
4. Output shape validation: (2, 2)
5. Visual token count, total parameters, trainable parameters reporting
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.vit_model import VisionTransformerDeepShield


def test_vit_128_compatibility():
    print("=" * 60)
    print("RUNNING VISION TRANSFORMER 128x128 COMPATIBILITY TEST")
    print("=" * 60)

    # 1. Initialize ViT with 128x128 input size and 2-class binary head
    model_name = "vit_base_patch16_224"
    img_size = (128, 128)
    patch_size = 16

    print(f"Instantiating ViT model: {model_name}")
    print(f"Target Input Resolution: {img_size}")
    print(f"Patch Size: {patch_size}x{patch_size}")

    model = VisionTransformerDeepShield(
        model_name=model_name,
        num_classes=2,
        pretrained=True,
        img_size=img_size,
        patch_size=patch_size,
        dropout=0.1,
    )
    model.eval()

    # 2. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Backend Used: {model.backend}")
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    print(f"Expected Spatial Patch Grid: {model.grid_h}x{model.grid_w} = {model.num_patches} patches")
    print(f"Total Visual Sequence Tokens: {model.num_tokens} tokens ({model.num_patches} patch tokens + 1 [CLS] token)")

    # 3. Create dummy input tensor
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1])
    print(f"Input Tensor Shape: {dummy_input.shape}")

    # 4. Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output Tensor Shape: {output.shape}")
    print(f"Output Logits Sample:\n{output}")

    # 5. Assertions
    assert output.shape == (batch_size, 2), f"Expected output shape ({batch_size}, 2), got {output.shape}"
    assert not torch.isnan(output).any(), "ViT output contains NaN values!"
    assert not torch.isinf(output).any(), "ViT output contains Inf values!"

    print("PASS: ViT 128x128 positional interpolation and forward pass verified successfully!")
    return True


if __name__ == "__main__":
    test_vit_128_compatibility()
