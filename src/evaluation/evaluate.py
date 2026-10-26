"""
DeepShield Master Evaluation Pipeline
Evaluates EfficientNet-B0 or Vision Transformer models on ASVspoof 2019 LA or ASVspoof 2021 DF.
Applies frozen decision threshold from results/thresholds.json without modification.
Performs chunk-level probability averaging across all overlapping 4.0s chunks per utterance.
Computes comprehensive biometric metrics at the utterance level and exports results to CSV.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield
from src.data.asvspoof2019 import ASVspoof2019Dataset, collate_utterances
from src.data.asvspoof2021 import ASVspoof2021DFDataset
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.threshold import ThresholdOptimizer


def evaluate_model(
    model,
    loader,
    device,
    threshold: float = 0.5,
    model_name: str = "model",
    dataset_name: str = "asvspoof2019",
    split: str = "eval",
):
    model.eval()
    all_labels = []
    all_scores = []
    all_utts = []
    all_attacks = []
    inference_times_ms = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name} on {dataset_name} ({split})"):
            if "batched_chunks" in batch:
                specs = batch["batched_chunks"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                chunk_counts = batch["chunk_counts"]
                batch_utts = batch["utterance_ids"]
                batch_attacks = batch["attack_ids"]

                t0 = time.perf_counter()
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(specs)
                    torch.cuda.synchronize()
                else:
                    logits = model(specs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                inf_per_utt = elapsed_ms / max(len(chunk_counts), 1)
                for _ in range(len(chunk_counts)):
                    inference_times_ms.append(inf_per_utt)

                probs = torch.softmax(logits, dim=1)[:, 1]
                split_probs = torch.split(probs, chunk_counts)
                for u, a, lbl, sp in zip(batch_utts, batch_attacks, labels, split_probs):
                    utt_score = float(sp.mean().item())
                    all_labels.append(int(lbl.item()))
                    all_scores.append(utt_score)
                    all_utts.append(u)
                    all_attacks.append(a)
            else:
                specs = batch["spectrograms"].squeeze(0).to(device)  # Shape: [N, 3, H, W]
                label = batch["label"]
                utt_id = batch["utterance_id"][0] if isinstance(batch["utterance_id"], list) else batch["utterance_id"]
                attack_id = batch.get("attack_id", [""])[0] if isinstance(batch.get("attack_id"), list) else batch.get("attack_id", "")

                t0 = time.perf_counter()
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(specs)
                    torch.cuda.synchronize()
                else:
                    logits = model(specs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                inference_times_ms.append(elapsed_ms)

                # Multi-chunk probability averaging
                probs = torch.softmax(logits, dim=1)[:, 1]
                utt_score = float(probs.mean().item())

                all_labels.append(int(label.item()))
                all_scores.append(utt_score)
                all_utts.append(utt_id)
                all_attacks.append(attack_id)

    metrics = compute_all_metrics(
        y_true=np.array(all_labels),
        y_scores=np.array(all_scores),
        threshold=threshold,
    )

    avg_inf_time = sum(inference_times_ms) / max(len(inference_times_ms), 1)
    total_params = sum(p.numel() for p in model.parameters())

    metrics["model"] = model_name
    metrics["dataset"] = dataset_name
    metrics["split"] = split
    metrics["parameters_m"] = round(total_params / 1e6, 2)
    metrics["inference_time_ms_per_utterance"] = round(avg_inf_time, 2)

    return metrics, all_utts, all_scores, all_labels, all_attacks


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepShield Model on ASVspoof Benchmark")
    parser.add_argument("--model_type", type=str, required=True, choices=["efficientnet", "vit"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="asvspoof2019", choices=["asvspoof2019", "asvspoof2021"])
    parser.add_argument("--data_root", type=str, default="C:\\DeepShieldData\\ASVspoof2019_LA")
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--cache_dir", type=str, default="C:\\DeepShieldData\\cache_spec")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--output_csv", type=str, default="results/model_comparison.csv")
    parser.add_argument("--threshold_file", type=str, default="results/thresholds.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size in utterances for fast evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Safety guard: preserve baseline Experiment A artifacts byte-for-byte
    if args.model_type == "vit":
        if args.output_csv == "results/model_comparison.csv":
            args.output_csv = "results/model_comparison_vit.csv"
            print(f"[PROTECTION] Preserving baseline model_comparison.csv. Redirecting ViT evaluation to: {args.output_csv}")
        if args.threshold_file == "results/thresholds.json" and os.path.exists("results/thresholds_vit.json"):
            args.threshold_file = "results/thresholds_vit.json"
            print(f"[PROTECTION] Preserving baseline thresholds.json. Reading ViT threshold from: {args.threshold_file}")

    device = torch.device(args.device)

    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    if args.model_type == "efficientnet":
        model = EfficientNetDeepShield(num_classes=2, pretrained=False)
        model_name = "efficientnet_b0" if args.seed == 42 else f"efficientnet_b0_seed{args.seed}"
    else:
        model = VisionTransformerDeepShield(model_name="vit_base_patch16_224", num_classes=2, pretrained=False, img_size=(args.img_size, args.img_size))
        if args.img_size == 128:
            model_name = "vit_b16" if args.seed == 42 else f"vit_b16_seed{args.seed}"
        else:
            model_name = f"vit_b16_{args.img_size}" if args.seed == 42 else f"vit_b16_{args.img_size}_seed{args.seed}"

    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)
    model = model.to(device)

    # 2. Load Frozen Decision Threshold
    th_optimizer = ThresholdOptimizer(save_path=args.threshold_file)
    frozen_threshold = th_optimizer.load_threshold(model_name=model_name, default=0.5)
    print(f"Loaded Frozen Threshold for {model_name} from {args.threshold_file}: {frozen_threshold:.6f}")

    # 3. Load Dataset in utterance mode
    if args.dataset == "asvspoof2019":
        dataset = ASVspoof2019Dataset(
            data_root=args.data_root,
            split=args.split,
            fraction=args.fraction,
            image_size=(args.img_size, args.img_size),
            cache_dir=args.cache_dir,
            mode="utterance",
        )
    else:
        dataset = ASVspoof2021DFDataset(data_root=args.data_root, fraction=args.fraction)

    collate_fn = collate_utterances if (args.dataset == "asvspoof2019" and args.batch_size > 1) else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Evaluate
    metrics, utts, scores, labels, attacks = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        threshold=frozen_threshold,
        model_name=model_name,
        dataset_name=f"{args.dataset}_{args.split}",
        split=args.split,
    )

    # 5. Print & Save Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    # Append to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_new = pd.DataFrame([metrics])
    if os.path.exists(args.output_csv):
        df_existing = pd.read_csv(args.output_csv)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(args.output_csv, index=False)
    print(f"\nSaved evaluation metrics row to: {args.output_csv}")


if __name__ == "__main__":
    main()
