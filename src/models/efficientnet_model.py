"""
DeepShield EfficientNet-B0 Baseline Model
Implements ImageNet-pretrained EfficientNet-B0 with binary classification head Linear(1280, 2).
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetDeepShield(nn.Module):
    """
    EfficientNet-B0 baseline classifier for speech deepfake detection.
    Consumes (B, 3, 128, 128) or (B, 3, 224, 224) log-mel spectrograms.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load weights
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace classification head: original has in_features = 1280
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 3, H, W)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature map before global pooling (for Grad-CAM)."""
        return self.backbone.features(x)
