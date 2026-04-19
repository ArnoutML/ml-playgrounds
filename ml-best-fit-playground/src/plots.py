"""Plotly chart builders for the app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.config import PLOT_RESOLUTION, X_MAX, X_MIN
from src.data import get_true_function


def make_curve_plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model,
    function_name: str,
):
    """Build the main curve plot."""
    x_grid = np.linspace(X_MIN, X_MAX, PLOT_RESOLUTION).reshape(-1, 1)

    true_function = get_true_function(function_name)
    y_true = true_function(x_grid.flatten())
    y_pred = model.predict(x_grid)

    fig = go.Figure()

    fig.add_scatter(
        x=x_grid.flatten(),
        y=y_true,
        mode="lines",
        name="True function",
    )

    fig.add_scatter(
        x=x_grid.flatten(),
        y=y_pred,
        mode="lines",
        name="Model fit",
    )

    fig.add_scatter(
        x=x_train.flatten(),
        y=y_train,
        mode="markers",
        name="Train data",
    )

    fig.add_scatter(
        x=x_val.flatten(),
        y=y_val,
        mode="markers",
        name="Validation data",
    )

    fig.update_layout(
        title=f"Overfitting Playground — {function_name}",
        xaxis_title="x",
        yaxis_title="y",
        height=550,
    )

    return fig


def make_error_bar_chart(train_rmse: float, val_rmse: float):
    """Build a simple error comparison chart."""
    fig = go.Figure()

    fig.add_bar(
        x=["Train RMSE", "Validation RMSE"],
        y=[train_rmse, val_rmse],
    )

    fig.update_layout(
        title="Generalization Check",
        yaxis_title="RMSE",
        height=400,
    )

    return fig