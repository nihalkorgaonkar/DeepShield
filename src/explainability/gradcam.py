"""
DeepShield Grad-CAM Explainability Module for CNN (EfficientNet-B0)
Computes gradient-weighted class activation mappings on log-mel spectrogram features.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for EfficientNet-B0 on log-mel spectrograms."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.backbone.features[-1]
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap.
        Args:
            input_tensor: Tensor of shape (1, 3, H, W)
            target_class: Target class index (0 = bonafide, 1 = spoof); defaults to argmax
        Returns:
            heatmap: 2D numpy array of shape (H, W) with values in [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Global average pool the gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted linear combination of feature activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Resize to input spatial resolution
        h, w = input_tensor.shape[-2:]
        cam_resized = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam_resized.squeeze().detach().cpu().numpy()

        # Min-max normalize
        min_v, max_v = np.min(cam_np), np.max(cam_np)
        if max_v - min_v > 1e-7:
            cam_np = (cam_np - min_v) / (max_v - min_v)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np
