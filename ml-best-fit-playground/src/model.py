"""Model construction and prediction logic."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def build_polynomial_model(degree: int) -> Pipeline:
    """Build a polynomial regression pipeline."""
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regressor", LinearRegression()),
        ]
    )


def fit_model(model: Pipeline, x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Fit the provided model."""
    model.fit(x_train, y_train)
    return model


def predict(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Generate predictions for input features."""
    return model.predict(x)