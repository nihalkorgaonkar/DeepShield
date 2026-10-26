"""
DeepShield Threshold Optimization Module
Strict Development-set-only decision threshold optimization.
Freezes decision threshold before evaluation to prevent dataset leakage.
"""

import json
import os
import time
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from .metrics import compute_eer


class ThresholdOptimizer:
    """Optimizes and serializes classification thresholds on the Development split only."""

    def __init__(
        self,
        criterion: str = "min_eer",
        save_path: str = "results/thresholds.json",
    ):
        self.criterion = criterion
        self.save_path = save_path

    def optimize_threshold(
        self,
        y_dev_true: np.ndarray,
        y_dev_scores: np.ndarray,
        model_name: str,
        dataset_name: str = "ASVspoof2019_LA_dev",
        seed: int = 42,
    ) -> float:
        """
        Computes the optimal decision threshold on the Development split.
        Args:
            y_dev_true: Ground truth binary labels (0 = bonafide, 1 = spoof)
            y_dev_scores: Predicted probabilities of spoof class [0, 1]
            model_name: Name of the evaluated model (e.g. 'efficientnet_b0', 'vit_b16')
            dataset_name: Name of the development split
            seed: Random seed for reproducibility
        Returns:
            optimal_threshold: The chosen threshold
        """
        y_dev_true = np.asarray(y_dev_true, dtype=int)
        y_dev_scores = np.asarray(y_dev_scores, dtype=float)

        if self.criterion == "min_eer":
            _, optimal_threshold = compute_eer(y_dev_true, y_dev_scores)
        elif self.criterion == "max_bacc":
            candidate_thresholds = np.linspace(0.01, 0.99, 100)
            best_score = -1.0
            optimal_threshold = 0.5
            for th in candidate_thresholds:
                preds = (y_dev_scores >= th).astype(int)
                score = balanced_accuracy_score(y_dev_true, preds)
                if score > best_score:
                    best_score = score
                    optimal_threshold = float(th)
        elif self.criterion == "max_f1":
            candidate_thresholds = np.linspace(0.01, 0.99, 100)
            best_score = -1.0
            optimal_threshold = 0.5
            for th in candidate_thresholds:
                preds = (y_dev_scores >= th).astype(int)
                score = f1_score(y_dev_true, preds, average="macro", zero_division=0)
                if score > best_score:
                    best_score = score
                    optimal_threshold = float(th)
        else:
            optimal_threshold = 0.5

        # Serialize to results/thresholds.json
        self.save_threshold(
            model_name=model_name,
            dataset_name=dataset_name,
            threshold=optimal_threshold,
            criterion=self.criterion,
            seed=seed,
        )

        return optimal_threshold

    def save_threshold(
        self,
        model_name: str,
        dataset_name: str,
        threshold: float,
        criterion: str,
        seed: int,
    ):
        """Saves or updates threshold entry in JSON."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        data = {}
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}

        data[model_name] = {
            "threshold": round(float(threshold), 6),
            "development_dataset": dataset_name,
            "optimization_criterion": criterion,
            "random_seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_threshold(self, model_name: str, default: float = 0.5) -> float:
        """Loads frozen threshold from results/thresholds.json."""
        if not os.path.exists(self.save_path):
            return default
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get(model_name, {}).get("threshold", default))
        except (json.JSONDecodeError, IOError):
            return default
