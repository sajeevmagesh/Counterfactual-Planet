"""Utility helpers for classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(prob: np.ndarray, target: np.ndarray) -> dict:
    """Compute accuracy/F1/AUC along with positive rates."""

    y_true = target.reshape(-1)
    y_prob = prob.reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pos_rate": float(y_true.mean()),
        "pred_pos_rate": float(y_pred.mean()),
    }
