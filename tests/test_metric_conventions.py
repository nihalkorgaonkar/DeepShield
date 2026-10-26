"""
DeepShield Metric and Score Convention Verification Tests
Verifies:
1. Binary class index mapping: 0 = bona fide (negative), 1 = spoof (positive).
2. Predicted score convention: probability of spoof class in [0, 1].
3. Decision threshold logic: prediction = 1 if score >= threshold else 0.
4. EER calculation orientation and threshold identification.
5. Confusion matrix layout: [[TN, FP], [FN, TP]].
6. Multi-chunk probability averaging across utterances.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import compute_eer, compute_all_metrics


def test_score_conventions_perfect_separation():
    # 5 bona fide (0) with low spoof scores, 5 spoof (1) with high spoof scores
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_scores = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.95, 0.96, 0.97, 0.98, 0.99])

    eer, eer_th = compute_eer(y_true, y_scores)
    assert eer == 0.0
    # Threshold should lie between 0.05 and 0.95
    assert 0.05 <= eer_th <= 0.95

    metrics = compute_all_metrics(y_true, y_scores, threshold=0.5)
    assert metrics["eer"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["true_negatives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 5
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_confusion_matrix_orientation():
    # Construct specific cases:
    # 2 TN (true 0, score < 0.5)
    # 1 FP (true 0, score >= 0.5)
    # 1 FN (true 1, score < 0.5)
    # 2 TP (true 1, score >= 0.5)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.2, 0.8, 0.9])
    th = 0.5

    metrics = compute_all_metrics(y_true, y_scores, threshold=th)
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 2
    assert metrics["false_positive_rate"] == 1 / 3
    assert metrics["false_negative_rate"] == 1 / 3
    assert metrics["accuracy"] == 4 / 6


def test_chunk_probability_averaging_logic():
    # Simulate an utterance with 3 chunks
    chunk_scores = [0.90, 0.95, 0.85]
    utt_score = np.mean(chunk_scores)
    assert abs(utt_score - 0.90) < 1e-6

    # If ground truth is 1 (spoof) and threshold is 0.5, utterance is classified as spoof
    pred = int(utt_score >= 0.5)
    assert pred == 1
