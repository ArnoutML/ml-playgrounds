"""Metric helpers for evaluating the model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a small set of easy-to-read regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def classify_fit(train_rmse: float, val_rmse: float, noise_std: float) -> tuple[str, str]:
    """
    Classify the model behavior into underfitting, good fit, or overfitting.
    """
    gap = val_rmse - train_rmse

    # Heuristic thresholds for this educational demo.
    if train_rmse > 0.25 and noise_std < 0.6:
        return (
            "underfitting",
            "This model may be underfitting: training error is far from the ideal solution, "
            "so the model is likely too simple to capture the true pattern.",
        )

    if gap > 0.07:
        return (
            "overfitting",
            "This model may be overfitting: training error is much lower than validation error, "
            "which suggests the model is fitting noise or idiosyncrasies in the training data.",
        )

    if train_rmse > 0.25 and noise_std > 0.30:
        return (
            "underfitting",
            "This model may be underfitting: training error is far from the ideal solution, "
            "so the model is likely too simple to capture the true pattern.",
        )

    if train_rmse > 0.35 and noise_std > 0.45:
        return (
            "underfitting",
            "This model may be underfitting: training error is far from the ideal solution, "
            "so the model is likely too simple to capture the true pattern.",
        )

    if train_rmse > 0.45 and noise_std > 0.55:
        return (
            "underfitting",
            "This model may be underfitting: training error is far from the ideal solution, "
            "so the model is likely too simple to capture the true pattern.",
        )

    if train_rmse > 0.55 and noise_std > 0.65:
        return (
            "underfitting",
            "This model may be underfitting: training error is far from the ideal solution, "
            "so the model is likely too simple to capture the true pattern.",
        )

    if train_rmse > 0.60 and noise_std > 0.80:
        return (
            "underfitting",
            "This model may be underfitting: training error is far from the ideal solution, "
            "so the model is likely too simple to capture the true pattern.",
        )



    return (
        "good_fit",
        "This model is generalizing reasonably well: training and validation performance are fairly aligned.",
    )