"""
DeepShield Explainability & Saliency Map Generator
Produces comparative visual explanations:
1. EfficientNet-B0: Gradient-weighted Class Activation Mapping (Grad-CAM).
2. Vision Transformer (ViT-B/16): Attention Saliency Rollout Heatmaps.
Generates publication-quality figures comparing authentic and synthetic speech representations.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield
from src.data.asvspoof2019 import ASVspoof2019Dataset
from src.explainability.gradcam import GradCAM
from src.explainability.vit_attention import ViTAttentionExplainer


def main():
    parser = argparse.ArgumentParser(description="Generate DeepShield Explainability Visualizations")
    parser.add_argument("--data_root", type=str, default="C:\\DeepShieldData\\ASVspoof2019_LA")
    parser.add_argument("--cache_dir", type=str, default="C:\\DeepShieldData\\cache_spec")
    parser.add_argument("--eff_ckpt", type=str, default="models/checkpoints/best_efficientnet_b0.pt")
    parser.add_argument("--vit_ckpt", type=str, default="models/checkpoints/best_vit_b16.pt")
    parser.add_argument("--output_figure", type=str, default="outputs/figures/explainability_comparison.png")
    parser.add_argument("--output_json", type=str, default="outputs/figures/explainability_summary.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Explainability suite on device: {device}")

    # Load models
    eff_model = EfficientNetDeepShield(num_classes=2, pretrained=False)
    if os.path.exists(args.eff_ckpt):
        d = torch.load(args.eff_ckpt, map_location="cpu")
        eff_model.load_state_dict(d["model_state_dict"] if "model_state_dict" in d else d)
    eff_model = eff_model.to(device).eval()

    vit_model = VisionTransformerDeepShield(img_size=(128, 128), pretrained=False)
    if os.path.exists(args.vit_ckpt):
        d = torch.load(args.vit_ckpt, map_location="cpu")
        vit_model.load_state_dict(d["model_state_dict"] if "model_state_dict" in d else d)
    vit_model = vit_model.to(device).eval()

    # Load dev dataset in chunk mode (128x128)
    dataset = ASVspoof2019Dataset(
        data_root=args.data_root,
        split="dev",
        fraction=1.0,
        image_size=(128, 128),
        cache_dir=args.cache_dir,
        mode="chunk",
    )

    # Pick 2 bonafide and 2 spoof samples
    bonafide_samples = []
    spoof_samples = []
    for item in dataset:
        if item["label"].item() == 0 and len(bonafide_samples) < 2:
            bonafide_samples.append(item)
        elif item["label"].item() == 1 and len(spoof_samples) < 2:
            spoof_samples.append(item)
        if len(bonafide_samples) >= 2 and len(spoof_samples) >= 2:
            break

    selected_samples = [("Bona Fide (Authentic)", s) for s in bonafide_samples] + [("Spoofed (Synthetic)", s) for s in spoof_samples]

    eff_explainer = GradCAM(eff_model)
    vit_explainer = ViTAttentionExplainer(vit_model)

    os.makedirs(os.path.dirname(args.output_figure), exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), dpi=300)

    summary_records = []

    for idx, (label_str, sample) in enumerate(selected_samples):
        spec = sample["spectrogram"].unsqueeze(0).to(device)  # (1, 3, 128, 128)

        # 1. Base log-mel spectrogram
        spec_img = sample["spectrogram"][0].cpu().numpy()

        # 2. EfficientNet Grad-CAM
        eff_cam = eff_explainer.generate_heatmap(spec, target_class=sample["label"].item())

        # 3. ViT Attention Map
        vit_att = vit_explainer.generate_attention_map(spec, grid_size=(8, 8))

        # Plot Spectrogram
        ax_spec = axes[idx, 0]
        im0 = ax_spec.imshow(spec_img, cmap="viridis", origin="lower", aspect="auto")
        ax_spec.set_title(f"{label_str}\nInput Log-Mel Spectrogram")
        ax_spec.set_ylabel("Frequency (Mel)")
        if idx == 3:
            ax_spec.set_xlabel("Time Chunks")

        # Plot Grad-CAM
        ax_cam = axes[idx, 1]
        im1 = ax_cam.imshow(spec_img, cmap="gray", origin="lower", aspect="auto")
        ax_cam.imshow(eff_cam, cmap="jet", alpha=0.55, origin="lower", aspect="auto")
        ax_cam.set_title(f"CNN (EfficientNet-B0)\nGrad-CAM Saliency")
        if idx == 3:
            ax_cam.set_xlabel("Time Chunks")

        # Plot ViT Attention
        ax_vit = axes[idx, 2]
        im2 = ax_vit.imshow(spec_img, cmap="gray", origin="lower", aspect="auto")
        ax_vit.imshow(vit_att, cmap="magma", alpha=0.55, origin="lower", aspect="auto")
        ax_vit.set_title(f"ViT-B/16\nAttention Rollout Saliency")
        if idx == 3:
            ax_vit.set_xlabel("Time Chunks")

        summary_records.append({
            "sample_index": idx,
            "label": label_str,
            "gradcam_mean_intensity": float(np.mean(eff_cam)),
            "vit_attention_mean_intensity": float(np.mean(vit_att)),
        })

    plt.tight_layout()
    plt.savefig(args.output_figure, dpi=300)
    plt.close()
    print(f"Saved Explainability Figure to: {args.output_figure}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"Saved Explainability Metadata to: {args.output_json}")


if __name__ == "__main__":
    main()
