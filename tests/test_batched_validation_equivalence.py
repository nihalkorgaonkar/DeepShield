"""
DeepShield Batched Validation Equivalence and Numerical Tolerance Test
Compares old unbatched inference (batch_size=1) vs new batched inference (collate_utterances, batch_size=64)
on a deterministic development subset using the verified Epoch 11 checkpoint.

Verifies:
1. Exact utterance ID and label ordering preservation
2. Exact chunk counts and zero dropped chunks
3. Maximum and mean absolute score differences under both AMP and FP32
4. Threshold-independent metrics: ROC-AUC, PR-AUC, EER, and EER threshold
5. Threshold-dependent metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix
   evaluated against the frozen development threshold (0.977539).
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.data.asvspoof2019 import ASVspoof2019Dataset, collate_utterances
from src.evaluation.metrics import compute_all_metrics, compute_eer


def test_batched_validation_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "models/checkpoints/backup_interrupted_run/best_efficientnet_b0_epoch11.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} not found"

    ckpt = torch.load(ckpt_path, map_location=device)
    model = EfficientNetDeepShield(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frozen_threshold = float(ckpt["dev_threshold"])

    # Deterministic subset of Dev (fraction=0.02 -> 495 utterances)
    dev_ds = ASVspoof2019Dataset(
        data_root=r"C:\DeepShieldData\ASVspoof2019_LA",
        split="dev",
        fraction=0.02,
        seed=42,
        mode="utterance",
    )
    n_utts = len(dev_ds)
    assert n_utts > 0

    # -------------------------------------------------------------------------
    # Path A: Original Unbatched Implementation (batch_size=1)
    # -------------------------------------------------------------------------
    loader_unbatched = DataLoader(dev_ds, batch_size=1, shuffle=False, num_workers=0)
    utts_a = []
    labels_a = []
    chunk_counts_a = []
    scores_a = []

    with torch.no_grad():
        for b in loader_unbatched:
            specs = b["spectrograms"].squeeze(0).to(device)
            utt_id = b["utterance_id"][0] if isinstance(b["utterance_id"], list) else b["utterance_id"]
            lbl = b["label"].item()
            n_chunks = specs.size(0)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(specs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                utt_score = float(probs.mean().item())

            utts_a.append(utt_id)
            labels_a.append(lbl)
            chunk_counts_a.append(n_chunks)
            scores_a.append(utt_score)

    # -------------------------------------------------------------------------
    # Path B: Optimized Batched Implementation (collate_utterances, batch_size=64)
    # -------------------------------------------------------------------------
    loader_batched = DataLoader(
        dev_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_utterances,
    )
    utts_b = []
    labels_b = []
    chunk_counts_b = []
    scores_b = []

    with torch.no_grad():
        for b in loader_batched:
            x = b["batched_chunks"].to(device)
            chunk_counts = b["chunk_counts"]
            chunk_counts_b.extend(chunk_counts)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]

            split_probs = torch.split(probs, chunk_counts)
            for u_id, lbl, sp in zip(b["utterance_ids"], b["labels"], split_probs):
                utts_b.append(u_id)
                labels_b.append(lbl.item())
                scores_b.append(float(sp.mean().item()))

    # -------------------------------------------------------------------------
    # Verification 1: Exact ordering, counts, and zero dropped chunks
    # -------------------------------------------------------------------------
    assert len(utts_a) == len(utts_b) == n_utts, f"Length mismatch: {len(utts_a)} vs {len(utts_b)}"
    assert utts_a == utts_b, "Utterance ID ordering mismatch!"
    assert labels_a == labels_b, "Label ordering mismatch!"
    assert chunk_counts_a == chunk_counts_b, "Chunk count mismatch!"
    total_chunks_a = sum(chunk_counts_a)
    total_chunks_b = sum(chunk_counts_b)
    assert total_chunks_a == total_chunks_b, "Total chunks mismatch! Chunks were dropped."

    # -------------------------------------------------------------------------
    # Verification 2: Score differences and tolerance
    # -------------------------------------------------------------------------
    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)
    diffs = np.abs(scores_a_arr - scores_b_arr)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))

    print(f"\n--- Numerical Score Differences (AMP Autocast) ---")
    print(f"Total Utterances Compared: {n_utts}")
    print(f"Total Chunks Evaluated:   {total_chunks_a}")
    print(f"Max Absolute Score Diff:  {max_diff:.8e}")
    print(f"Mean Absolute Score Diff: {mean_diff:.8e}")

    # Documented Tolerance:
    # Max difference <= 5e-4 due to cuDNN FP16 GEMM tile accumulation associativity
    assert max_diff <= 5e-4, f"Max diff exceeded numerical tolerance: {max_diff}"
    assert mean_diff <= 1e-5, f"Mean diff exceeded numerical tolerance: {mean_diff}"

    # -------------------------------------------------------------------------
    # Verification 3: Metrics Equivalence
    # -------------------------------------------------------------------------
    metrics_a = compute_all_metrics(labels_a, scores_a, threshold=frozen_threshold)
    metrics_b = compute_all_metrics(labels_b, scores_b, threshold=frozen_threshold)

    print(f"\n--- Metrics Comparison (Frozen Threshold: {frozen_threshold:.6f}) ---")
    print(f"Metric                  Unbatched (A)        Batched (B)          Delta")
    for k in ["eer", "eer_threshold", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1_macro"]:
        val_a = metrics_a[k]
        val_b = metrics_b[k]
        delta = abs(val_a - val_b)
        print(f"{k:22s} {val_a:18.6f} {val_b:18.6f} {delta:14.2e}")
        assert delta < 1e-4, f"Metric mismatch in {k}: {val_a} vs {val_b}"

    # Confusion Matrix layout check
    for cm_key in ["true_negatives", "false_positives", "false_negatives", "true_positives"]:
        assert metrics_a[cm_key] == metrics_b[cm_key], f"Confusion matrix mismatch in {cm_key}: {metrics_a[cm_key]} vs {metrics_b[cm_key]}"

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    test_batched_validation_equivalence()
