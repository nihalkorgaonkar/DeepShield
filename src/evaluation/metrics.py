"""
DeepShield Evaluation Metrics Suite
Computes:
- Equal Error Rate (EER) and EER threshold
- Balanced Accuracy
- Precision, Recall, F1-Score (per-class and macro)
- ROC-AUC and PR-AUC
- Confusion Matrix (TP, TN, FP, FN, FPR, FNR)
"""

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
)


def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the corresponding decision threshold.
    Args:
        y_true: Binary ground truth labels (0 = bonafide, 1 = spoof)
        y_scores: Predicted spoof probabilities or logits
    Returns:
        eer: Equal Error Rate (percentage / ratio)
        eer_threshold: Decision threshold where FPR equals FNR
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FPR and FNR are closest
    diff = np.abs(fpr - fnr)
    if np.all(np.isnan(diff)):
        return 0.5, 0.5
    eer_idx = np.nanargmin(diff)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    return float(eer), float(eer_threshold)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int, dict]]:
    """
    Computes comprehensive biometric and classification metrics.
    Args:
        y_true: Binary ground truth labels (0 = bonafide, 1 = spoof)
        y_scores: Predicted spoof probabilities (in range [0, 1])
        threshold: Decision threshold for discrete classification (pred = 1 if score >= threshold)
    Returns:
        metrics_dict: Dictionary containing all computed evaluation metrics
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= threshold).astype(int)

    # 1. EER
    eer, eer_threshold = compute_eer(y_true, y_scores)

    # 2. Balanced Accuracy & Accuracy
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    # 3. Precision, Recall, F1
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 4. ROC-AUC & PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = float("nan")

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec_curve, prec_curve)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eval_threshold": threshold,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "bonafide_precision": float(prec_per_class[0]),
        "bonafide_recall": float(rec_per_class[0]),
        "bonafide_f1": float(f1_per_class[0]),
        "spoof_precision": float(prec_per_class[1]),
        "spoof_recall": float(rec_per_class[1]),
        "spoof_f1": float(f1_per_class[1]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }
