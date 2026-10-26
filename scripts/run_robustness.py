"""
DeepShield Robustness Evaluation Runner
Evaluates EfficientNet-B0 and Vision Transformer on acoustic corruptions
(Additive Gaussian Noise, Volume Scaling, Telephony Resampling) using the development partition.
Saves results to results/robustness_results.json and generates comparison plot.
"""

import argparse
import json
import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield
from src.data.asvspoof2019 import ASVspoof2019Dataset
from src.evaluation.robustness import evaluate_robustness_suite


def main():
    parser = argparse.ArgumentParser(description="Run DeepShield Robustness Stress-Testing Suite")
    parser.add_argument("--data_root", type=str, default="C:\\DeepShieldData\\ASVspoof2019_LA")
    parser.add_argument("--eff_ckpt", type=str, default="models/checkpoints/best_efficientnet_b0.pt")
    parser.add_argument("--vit_ckpt", type=str, default="models/checkpoints/best_vit_b16.pt")
    parser.add_argument("--eff_threshold_file", type=str, default="results/thresholds.json")
    parser.add_argument("--vit_threshold_file", type=str, default="results/thresholds_vit.json")
    parser.add_argument("--max_samples", type=int, default=500, help="Samples per condition for stress test")
    parser.add_argument("--output_json", type=str, default="results/robustness_results.json")
    parser.add_argument("--output_plot", type=str, default="outputs/figures/robustness_comparison.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Robustness Suite on device: {device}")

    # Load frozen thresholds
    eff_threshold = 0.967773
    if os.path.exists(args.eff_threshold_file):
        try:
            with open(args.eff_threshold_file, "r") as f:
                d = json.load(f)
            eff_threshold = d.get("efficientnet_b0", {}).get("threshold", eff_threshold)
        except Exception:
            pass

    vit_threshold = 0.987305
    if os.path.exists(args.vit_threshold_file):
        try:
            with open(args.vit_threshold_file, "r") as f:
                d = json.load(f)
            vit_threshold = d.get("vit_b16", {}).get("threshold", vit_threshold)
        except Exception:
            pass

    print(f"Using Frozen Thresholds -> EfficientNet: {eff_threshold:.6f}, ViT: {vit_threshold:.6f}")

    # Load raw audio dataset on dev split (cache_dir=None to get audio_path)
    print("Loading development partition with raw audio mappings...")
    dataset = ASVspoof2019Dataset(
        data_root=args.data_root,
        split="dev",
        fraction=1.0,
        cache_dir=None,
    )
    print(f"Loaded {len(dataset)} development utterances.")

    all_results = {}

    # 1. Evaluate EfficientNet-B0
    if os.path.exists(args.eff_ckpt):
        print("\nEvaluating EfficientNet-B0 Robustness...")
        eff_model = EfficientNetDeepShield(num_classes=2, pretrained=False)
        eff_data = torch.load(args.eff_ckpt, map_location="cpu")
        eff_model.load_state_dict(eff_data["model_state_dict"] if "model_state_dict" in eff_data else eff_data)
        eff_model = eff_model.to(device)

        eff_res = evaluate_robustness_suite(
            model=eff_model,
            dataset=dataset,
            device=device,
            threshold=eff_threshold,
            max_samples=args.max_samples,
        )
        all_results["EfficientNet-B0"] = eff_res
        del eff_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Evaluate ViT-B/16
    if os.path.exists(args.vit_ckpt):
        print("\nEvaluating ViT-B/16 Robustness...")
        vit_model = VisionTransformerDeepShield(img_size=(128, 128), pretrained=False)
        vit_data = torch.load(args.vit_ckpt, map_location="cpu")
        vit_model.load_state_dict(vit_data["model_state_dict"] if "model_state_dict" in vit_data else vit_data)
        vit_model = vit_model.to(device)

        vit_res = evaluate_robustness_suite(
            model=vit_model,
            dataset=dataset,
            device=device,
            threshold=vit_threshold,
            max_samples=args.max_samples,
        )
        all_results["ViT-B/16"] = vit_res
        del vit_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save JSON results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Robustness Results to {args.output_json}")

    # Generate Plot
    if all_results:
        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        records = []
        for model_name, conds in all_results.items():
            for cond, metrics in conds.items():
                records.append({
                    "Model": model_name,
                    "Condition": cond,
                    "EER (%)": metrics.get("eer", 0.0) * 100,
                    "Accuracy (%)": metrics.get("accuracy", 0.0) * 100,
                })
        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        sns.barplot(data=df, x="Condition", y="EER (%)", hue="Model", ax=ax, palette="tab10")
        ax.set_title("Acoustic Corruption Degradation Profiles (Equal Error Rate)")
        ax.set_ylabel("Equal Error Rate (%)")
        ax.set_xlabel("Perturbation Stress Condition")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300)
        plt.close()
        print(f"Saved Robustness Figure to {args.output_plot}")


if __name__ == "__main__":
    main()
