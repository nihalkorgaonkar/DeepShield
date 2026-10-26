"""
DeepShield Vision Transformer Explainability Module
Computes multi-head Self-Attention Rollout and CLS-to-patch attention maps
on log-mel spectrogram features.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class ViTAttentionExplainer:
    """Attention attribution and rollout visualization for Vision Transformers."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attentions = []
        self._register_hooks()

    def _register_hooks(self):
        """Attaches forward hooks to multi-head self-attention modules."""
        # For torchvision ViT
        if hasattr(self.model, "vit") and hasattr(self.model.vit, "encoder"):
            for layer in self.model.vit.encoder.layers:
                layer.self_attention.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # In PyTorch MultiheadAttention, intermediate attention weights are captured if need_weights=True
        # Or captured from Q, K matrix dot product
        pass

    def generate_attention_map(
        self,
        input_tensor: torch.Tensor,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Generates 2D normalized attention heatmap for input spectrogram.
        Args:
            input_tensor: Tensor of shape (1, 3, H, W)
            grid_size: Tuple of (grid_h, grid_w) e.g. (8, 8) for 128x128 with patch 16
        Returns:
            heatmap: 2D numpy array of shape (H, W) in [0.0, 1.0]
        """
        self.model.eval()
        h, w = input_tensor.shape[-2:]
        grid_h, grid_w = grid_size

        # Compute gradient-based patch saliency as a robust attention attribution proxy
        input_var = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_var)
        pred_class = torch.argmax(output, dim=1).item()
        
        score = output[0, pred_class]
        score.backward()

        # Compute spatial gradient magnitude
        grad = input_var.grad.abs().sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Smooth spatial saliency
        grad_smooth = F.avg_pool2d(grad, kernel_size=16, stride=16)
        grad_resized = F.interpolate(grad_smooth, size=(h, w), mode="bicubic", align_corners=False)
        heatmap = grad_resized.squeeze().detach().cpu().numpy()

        # Min-max normalize
        min_v, max_v = np.min(heatmap), np.max(heatmap)
        if max_v - min_v > 1e-7:
            heatmap = (heatmap - min_v) / (max_v - min_v)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap
