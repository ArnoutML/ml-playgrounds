"""Synthetic data generation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_RANDOM_SEED, X_MIN, X_MAX


def get_true_function(function_name: str):
    """
    Return a callable true function based on the selected preset.
    """
    functions = {
        "Linear": lambda x: 0.8 * x + 0.5,
        "Quadratic": lambda x: 0.35 * x**2 + 0.4 * x - 0.5,
        "Cubic": lambda x: 0.12 * x**3 - 0.3 * x**2 + 0.2 * x + 0.8,
        "Sine": lambda x: np.sin(1.5 * x),
        "Sine + Trend": lambda x: np.sin(x) + 0.25 * x,
        "Wavy Nonlinear": lambda x: np.sin(2.0 * x) + 0.15 * x**2,
    }

    if function_name not in functions:
        raise ValueError(f"Unknown true function: {function_name}")

    return functions[function_name]


def generate_dataset(
    n_samples: int,
    noise_std: float,
    function_name: str,
    test_size: float = 0.35,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset and split it into train/validation sets.
    """
    rng = np.random.default_rng(random_seed)

    x = rng.uniform(X_MIN, X_MAX, size=n_samples)
    x = np.sort(x)

    true_function = get_true_function(function_name)
    y = true_function(x) + rng.normal(0.0, noise_std, size=n_samples)

    x_2d = x.reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_2d,
        y,
        test_size=test_size,
        random_state=random_seed,
    )

    return x_train, x_val, y_train, y_val