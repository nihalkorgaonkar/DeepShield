"""
DeepShield Validation Performance Benchmark & Equivalence Test
Compares:
1. Unbatched single-utterance baseline (batch_size=1)
2. Safe batched utterance inference with variable multi-chunk concatenation
Verifies:
- Numerical equivalence of predicted scores within floating-point tolerance (< 1e-5)
- Throughput (utterances/sec)
- VRAM footprint
- Estimated full-validation time for all 24,844 development utterances
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.data.asvspoof2019 import ASVspoof2019Dataset


def collate_utterances(batch_list):
    """
    Safely batches variable numbers of multi-chunks per utterance into a single
    contiguous tensor [M, 3, 128, 128] while tracking individual chunk boundaries.
    """
    all_chunks = []
    chunk_counts = []
    labels = []
    utterance_ids = []

    for item in batch_list:
        specs = item["spectrograms"]  # [N_i, 3, 128, 128]
        all_chunks.append(specs)
        chunk_counts.append(specs.size(0))
        labels.append(item["label"].item() if isinstance(item["label"], torch.Tensor) else item["label"])
        utterance_ids.append(item["utterance_id"])

    batched_chunks = torch.cat(all_chunks, dim=0)  # [M, 3, 128, 128]
    return {
        "batched_chunks": batched_chunks,
        "chunk_counts": chunk_counts,
        "labels": torch.tensor(labels, dtype=torch.long),
        "utterance_ids": utterance_ids,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "models/checkpoints/backup_interrupted_run/best_efficientnet_b0_epoch11.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = EfficientNetDeepShield(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 495 development utterances (fraction=0.02)
    dev_ds = ASVspoof2019Dataset(
        data_root=r"C:\DeepShieldData\ASVspoof2019_LA",
        split="dev",
        fraction=0.02,
        seed=42,
        mode="utterance",
    )
    n_samples = len(dev_ds)

    print(f"Loaded {n_samples} dev utterances for performance benchmark.")

    # -------------------------------------------------------------
    # 1. Unbatched Baseline (Current Implementation: batch_size=1)
    # -------------------------------------------------------------
    loader_unbatched = DataLoader(dev_ds, batch_size=1, shuffle=False, num_workers=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scores_unbatched = []

    with torch.no_grad():
        for b in loader_unbatched:
            specs = b["spectrograms"].squeeze(0).to(device)
            with torch.amp.autocast("cuda"):
                logits = model(specs)
                probs = torch.softmax(logits, dim=1)[:, 1]
            scores_unbatched.append(probs.mean().item())

    torch.cuda.synchronize()
    t_unbatched = time.perf_counter() - t0
    throughput_unbatched = n_samples / t_unbatched

    # -------------------------------------------------------------
    # 2. Optimized Batched Inference (batch_size=64 utterances)
    # -------------------------------------------------------------
    for batch_size in [32, 64, 128]:
        loader_batched = DataLoader(
            dev_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_utterances,
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores_batched = []

        with torch.no_grad():
            for b in loader_batched:
                x = b["batched_chunks"].to(device, non_blocking=True)
                chunk_counts = b["chunk_counts"]

                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[:, 1]

                split_probs = torch.split(probs, chunk_counts)
                for p_utt in split_probs:
                    scores_batched.append(p_utt.mean().item())

        torch.cuda.synchronize()
        t_batched = time.perf_counter() - t0
        throughput_batched = n_samples / t_batched
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

        # Numerical Equivalence Check
        diffs = np.abs(np.array(scores_unbatched) - np.array(scores_batched))
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)

        print(f"\n--- Batch Size: {batch_size} Utterances ---")
        print(f"Elapsed Time:       {t_batched:.3f} s")
        print(f"Throughput:         {throughput_batched:.1f} utts/sec (Speedup: {throughput_batched/throughput_unbatched:.2f}x)")
        print(f"Peak VRAM:          {peak_vram:.2f} MB")
        print(f"Max Absolute Diff:  {max_diff:.8e}")
        print(f"Mean Absolute Diff: {mean_diff:.8e}")
        print(f"Equivalence (<1e-5): {max_diff < 1e-5}")
        print(f"Estimated Full Dev (24,844 utts): {24844 / throughput_batched:.1f} s ({24844 / throughput_batched / 60:.2f} min)")

    print(f"\nBaseline Unbatched Throughput: {throughput_unbatched:.1f} utts/sec ({24844 / throughput_unbatched / 60:.2f} min for Full Dev)")


if __name__ == "__main__":
    main()
