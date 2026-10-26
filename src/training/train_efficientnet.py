"""
DeepShield EfficientNet-B0 Baseline Training Pipeline
Implements:
- Class-weighted Cross-Entropy Loss
- AdamW optimizer with Cosine Annealing scheduler
- Development-set validation with multi-chunk probability averaging
- Dev-only threshold optimization saved to results/thresholds.json
- Checkpoint saving: best checkpoint & latest checkpoint with complete state dicts
- Early stopping & JSON training history logging
"""

import argparse
import json
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.data.asvspoof2019 import ASVspoof2019Dataset, collate_utterances
from src.evaluation.metrics import compute_eer
from src.evaluation.threshold import ThresholdOptimizer


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["spectrogram"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_utts = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if "batched_chunks" in batch:
                x = batch["batched_chunks"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                chunk_counts = batch["chunk_counts"]

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(x)
                        chunk_labels = torch.repeat_interleave(labels, torch.tensor(chunk_counts, device=device))
                        loss = criterion(logits, chunk_labels)
                else:
                    logits = model(x)
                    chunk_labels = torch.repeat_interleave(labels, torch.tensor(chunk_counts, device=device))
                    loss = criterion(logits, chunk_labels)

                running_loss += loss.item() * len(chunk_counts)
                total_utts += len(chunk_counts)

                probs = torch.softmax(logits, dim=1)[:, 1]
                split_probs = torch.split(probs, chunk_counts)
                for lbl, sp in zip(labels, split_probs):
                    all_scores.append(float(sp.mean().item()))
                    all_labels.append(int(lbl.item()))
            else:
                specs = batch["spectrograms"].squeeze(0).to(device)
                label = batch["label"].to(device)

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(specs)
                        targets = label.repeat(specs.size(0))
                        loss = criterion(logits, targets)
                else:
                    logits = model(specs)
                    targets = label.repeat(specs.size(0))
                    loss = criterion(logits, targets)

                running_loss += loss.item()
                total_utts += 1

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_scores.append(float(probs.mean().item()))
                all_labels.append(int(label.item()))

    val_loss = running_loss / max(total_utts, 1)
    eer, dev_threshold = compute_eer(np.array(all_labels), np.array(all_scores))
    return val_loss, eer, dev_threshold, all_labels, all_scores


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Baseline on ASVspoof 2019 LA")
    parser.add_argument("--data_root", type=str, default="C:\\DeepShieldData\\ASVspoof2019_LA")
    parser.add_argument("--fraction", type=float, default=1.0, help="Subsample fraction for smoke-testing (e.g. 0.02)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64, help="Validation batch size in utterances")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    parser.add_argument("--cache_dir", type=str, default="C:\\DeepShieldData\\cache_spec")
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"=======================================================")
    print(f"TRAINING: EfficientNet-B0 Baseline on {device} (Seed: {args.seed})")
    print(f"Data Root: {args.data_root} | Fraction: {args.fraction}")
    print(f"Train Batch Size: {args.batch_size} | Val Batch Size: {args.val_batch_size}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"=======================================================")

    # Datasets: chunk-mode for training, utterance-mode for validation
    train_dataset = ASVspoof2019Dataset(
        data_root=args.data_root,
        split="train",
        fraction=args.fraction,
        seed=args.seed,
        cache_dir=args.cache_dir,
        mode="chunk",
    )
    dev_dataset = ASVspoof2019Dataset(
        data_root=args.data_root,
        split="dev",
        fraction=args.fraction,
        seed=args.seed,
        cache_dir=args.cache_dir,
        mode="utterance",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    # Safe batched validation with collate_utterances
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_utterances,
        pin_memory=(device.type == "cuda"),
    )

    # Class-weighted loss
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Class Weights: Bonafide={class_weights[0].item():.4f}, Spoof={class_weights[1].item():.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = EfficientNetDeepShield(num_classes=2, pretrained=True, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    threshold_optimizer = ThresholdOptimizer(criterion="min_eer", save_path="results/thresholds.json")

    best_eer = float("inf")
    best_epoch = 0
    patience = args.patience
    patience_counter = 0

    if args.seed == 42:
        best_ckpt_path = os.path.join(args.save_dir, "best_efficientnet_b0.pt")
        latest_ckpt_path = os.path.join(args.save_dir, "latest_efficientnet_b0.pt")
        history_path = "results/training_history_efficientnet_b0.json"
        model_key = "efficientnet_b0"
    else:
        best_ckpt_path = os.path.join(args.save_dir, f"best_efficientnet_b0_seed{args.seed}.pt")
        latest_ckpt_path = os.path.join(args.save_dir, f"latest_efficientnet_b0_seed{args.seed}.pt")
        history_path = f"results/training_history_efficientnet_b0_seed{args.seed}.json"
        model_key = f"efficientnet_b0_seed{args.seed}"

    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_eer, dev_threshold, y_true, y_scores = validate(model, dev_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "dev_loss": round(val_loss, 6),
            "dev_eer": round(val_eer, 6),
            "dev_threshold": round(dev_threshold, 6),
            "elapsed_seconds": round(elapsed, 2),
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Dev Loss: {val_loss:.4f} EER: {val_eer*100:.2f}% (Th: {dev_threshold:.4f})")

        # Always save latest checkpoint with complete states for faithful resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "rng_state": torch.get_rng_state(),
            "best_eer": best_eer,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "model_name": model_key,
            "seed": args.seed,
            "history": history,
        }, latest_ckpt_path)

        if val_eer < best_eer:
            best_eer = val_eer
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "dev_eer": best_eer,
                "dev_threshold": dev_threshold,
                "model_name": model_key,
                "seed": args.seed,
            }, best_ckpt_path)

            # Freeze dev threshold
            threshold_optimizer.optimize_threshold(
                y_dev_true=y_true,
                y_dev_scores=y_scores,
                model_name=model_key,
                seed=args.seed,
            )
            print(f"  --> Saved new best checkpoint (Dev EER: {best_eer*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Best Dev EER: {best_eer*100:.2f}% at epoch {best_epoch}).")
                break

    # Save training history JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved complete training history to: {history_path}")
    print(f"Training Complete. Best Model Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
