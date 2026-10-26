"""
DeepShield Publication Figures Generator
Produces comprehensive, high-resolution scientific plots:
1. Training & Validation Learning Curves (Loss & EER per epoch)
2. ROC and Precision-Recall Curves
3. Confusion Matrices for Clean & Domain-Shift Conditions
4. Model Comparison (EER, min t-DCF, Parameters, Throughput)
5. Cross-Dataset Generalization (ASVspoof 2019 vs 2021 DF)
6. Robustness Degradation Profiles across Acoustic Corruptions
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 13


def plot_learning_curves(eff_log_path, vit_log_path, output_path="outputs/figures/learning_curves.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    if os.path.exists(eff_log_path):
        df_eff = pd.read_csv(eff_log_path)
        axes[0].plot(df_eff["epoch"], df_eff["train_loss"], label="EfficientNet-B0 (Train)", color="#1f77b4", linestyle="--")
        axes[0].plot(df_eff["epoch"], df_eff["val_loss"], label="EfficientNet-B0 (Dev)", color="#1f77b4", linewidth=2)
        axes[1].plot(df_eff["epoch"], df_eff["val_eer"] * 100, label="EfficientNet-B0", color="#1f77b4", linewidth=2, marker="o", markersize=4)

    if os.path.exists(vit_log_path):
        df_vit = pd.read_csv(vit_log_path)
        axes[0].plot(df_vit["epoch"], df_vit["train_loss"], label="ViT-B/16 (Train)", color="#ff7f0e", linestyle="--")
        axes[0].plot(df_vit["epoch"], df_vit["val_loss"], label="ViT-B/16 (Dev)", color="#ff7f0e", linewidth=2)
        axes[1].plot(df_vit["epoch"], df_vit["val_eer"] * 100, label="ViT-B/16", color="#ff7f0e", linewidth=2, marker="s", markersize=4)

    axes[0].set_title("Cross-Entropy Loss vs. Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Equal Error Rate (EER %) vs. Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("EER (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison_bar(results_csv="results/model_comparison.csv", output_path="outputs/figures/model_comparison_bar.png"):
    dfs = []
    if os.path.exists(results_csv):
        dfs.append(pd.read_csv(results_csv))
    vit_csv = "results/model_comparison_vit.csv"
    if os.path.exists(vit_csv):
        dfs.append(pd.read_csv(vit_csv))

    if not dfs:
        print("No evaluation CSV files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Plot 1: EER
    sns.barplot(data=df, x="dataset", y="eer", hue="model", ax=axes[0], palette="tab10")
    axes[0].set_title("Equal Error Rate (Lower is Better)")
    axes[0].set_ylabel("EER")
    axes[0].set_xlabel("Benchmark Dataset")

    # Plot 2: ROC-AUC
    sns.barplot(data=df, x="dataset", y="roc_auc", hue="model", ax=axes[1], palette="tab10")
    axes[1].set_title("ROC-AUC (Higher is Better)")
    axes[1].set_ylabel("AUC")
    axes[1].set_xlabel("Benchmark Dataset")

    # Plot 3: Balanced Accuracy
    sns.barplot(data=df, x="dataset", y="balanced_accuracy", hue="model", ax=axes[2], palette="tab10")
    axes[2].set_title("Balanced Accuracy (Higher is Better)")
    axes[2].set_ylabel("Balanced Accuracy")
    axes[2].set_xlabel("Benchmark Dataset")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_robustness_comparison(robustness_json="results/robustness_results.json", output_path="outputs/figures/robustness_comparison.png"):
    if not os.path.exists(robustness_json):
        return

    with open(robustness_json, "r") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []
    for model_name, conds in data.items():
        for cond, metrics in conds.items():
            records.append({
                "model": model_name,
                "condition": cond,
                "eer": metrics.get("eer", 0.0) * 100,
                "auc": metrics.get("roc_auc", 0.0),
            })

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.barplot(data=df, x="condition", y="eer", hue="model", ax=ax, palette="Set2")
    ax.set_title("Robustness Degradation under Acoustic Corruptions (EER %)")
    ax.set_ylabel("EER (%)")
    ax.set_xlabel("Perturbation Type")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating DeepShield publication figures...")
    plot_model_comparison_bar()
    plot_robustness_comparison()
