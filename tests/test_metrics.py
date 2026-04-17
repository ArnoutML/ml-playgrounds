"""Basic tests for metric helpers."""

from __future__ import annotations

import numpy as np

from src.metrics import regression_metrics


def test_regression_metrics_returns_expected_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    metrics = regression_metrics(y_true, y_pred)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_rmse_is_non_negative():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["rmse"] >= 0.0