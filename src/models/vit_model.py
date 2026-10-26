"""
DeepShield Vision Transformer (ViT) Model
Implements Vision Transformer with exact 128x128 input support via
2D bicubic positional embedding interpolation and binary classification head.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

import torchvision.models as models


class VisionTransformerDeepShield(nn.Module):
    """
    Vision Transformer for speech deepfake detection.
    Consumes (B, 3, 128, 128) or (B, 3, 224, 224) log-mel spectrograms.
    Positional embeddings are adapted to (128, 128) resolution (8x8 patch grid)
    via 2D bicubic interpolation without modifying the input audio representation.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 16,
        dropout: float = 0.1,
        backend: str = "torchvision",
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.dropout = dropout

        # Compute grid size
        self.grid_h = self.img_size[0] // patch_size
        self.grid_w = self.img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.num_tokens = self.num_patches + 1  # Patch tokens + [CLS]

        if backend == "timm" and HAS_TIMM:
            # Use timm's native dynamic_img_size positional interpolation
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                img_size=self.img_size,
                dynamic_img_size=True,
                drop_rate=dropout,
            )
            self.backend = "timm"
        else:
            # Use torchvision ViT-B/16 with explicit 2D bicubic positional interpolation
            if pretrained:
                weights = models.ViT_B_16_Weights.DEFAULT
            else:
                weights = None
            self.vit = models.vit_b_16(weights=weights)
            
            # Interpolate positional embeddings from 14x14 (224/16) to grid_h x grid_w (8x8)
            self._interpolate_torchvision_pos_embed()

            # Update head
            in_features = self.vit.heads.head.in_features
            self.vit.heads.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features=in_features, out_features=num_classes),
            )
            self.backend = "torchvision"

    def _interpolate_torchvision_pos_embed(self):
        """Resamples positional embeddings for torchvision ViT model."""
        pos_embed = self.vit.encoder.pos_embedding  # Shape: (1, 197, 768)
        cls_pos = pos_embed[:, :1, :]               # Shape: (1, 1, 768)
        patch_pos = pos_embed[:, 1:, :]             # Shape: (1, 196, 768)

        embed_dim = patch_pos.shape[-1]
        orig_grid = int(math.sqrt(patch_pos.shape[1]))  # 14 for 224x224 / 16
        
        # Reshape to 2D spatial feature map: (1, 768, 14, 14)
        patch_pos_2d = patch_pos.reshape(1, orig_grid, orig_grid, embed_dim).permute(0, 3, 1, 2)
        
        # Interpolate to target grid: (1, 768, grid_h, grid_w) e.g. (1, 768, 8, 8)
        interpolated_patch_pos = F.interpolate(
            patch_pos_2d,
            size=(self.grid_h, self.grid_w),
            mode="bicubic",
            align_corners=False,
        )
        
        # Flatten back: (1, 64, 768)
        interpolated_patch_pos = interpolated_patch_pos.permute(0, 2, 3, 1).reshape(1, self.num_patches, embed_dim)
        
        # Recombine with CLS token position embedding: (1, 65, 768)
        new_pos_embed = torch.cat([cls_pos, interpolated_patch_pos], dim=1)
        self.vit.encoder.pos_embedding = nn.Parameter(new_pos_embed)
        self.vit.image_size = self.img_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 3, 128, 128)
        Returns:
            logits: Tensor of shape (batch_size, 2)
        """
        return self.vit(x)

    def get_attention_weights(self, x: torch.Tensor):
        """Extracts self-attention weights across transformer layers for explainability."""
        if hasattr(self.vit, "get_intermediate_layers"):
            return self.vit.get_intermediate_layers(x)
        return None
