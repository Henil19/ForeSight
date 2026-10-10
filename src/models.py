"""Model construction and chronological supervised-data utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Close", "Volume", "Daily_Return", "MA20", "MA50", "MA200",
    "RSI", "MACD", "Volatility",
]


def prepare_supervised_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature rows and their next-row closing-price targets.

    The input is not modified; chronological row order and index labels are
    preserved. The last row is removed because its next close is unavailable.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError("Features are missing required columns: " + ", ".join(missing))
    target = features["Close"].shift(-1).rename("Target")
    valid = target.notna()
    return features.loc[valid, FEATURE_COLUMNS].copy(), target.loc[valid].copy()


def build_linear_regression() -> Pipeline:
    """Build a scaled linear-regression baseline."""
    return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])


def build_random_forest() -> RandomForestRegressor:
    """Build a compact, reproducible random-forest regressor."""
    return RandomForestRegressor(
        n_estimators=100, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=1,
    )


def build_mlp() -> Pipeline:
    """Build a scaled, modest single-hidden-layer neural-network regressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(32,), activation="relu", solver="adam",
            max_iter=300, early_stopping=False, random_state=42,
        )),
    ])


def build_mlp_scaled_target() -> TransformedTargetRegressor:
    """Build the MLP experiment with training-fitted target scaling.

    The feature pipeline and MLP settings are identical to ``build_mlp``.
    TransformedTargetRegressor fits its target transformer on each fit's y and
    inverse-transforms predictions back to the original target units.
    """
    return TransformedTargetRegressor(
        regressor=build_mlp(),
        transformer=StandardScaler(),
    )


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Fit all supported models on caller-provided training data only."""
    models: dict[str, Any] = {
        "Linear Regression": build_linear_regression(),
        "Random Forest": build_random_forest(),
        "MLP": build_mlp_scaled_target(),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split aligned observations without shuffling, keeping the latest rows for test."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a DataFrame and y must be a Series.")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have the same length and identical index.")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError("test_size must be a number strictly between 0 and 1.")
    test_count = int(len(X) * test_size)
    if test_count < 1 or test_count >= len(X):
        raise ValueError("test_size must leave at least one row in both splits.")
    split_at = len(X) - test_count
    return X.iloc[:split_at].copy(), X.iloc[split_at:].copy(), y.iloc[:split_at].copy(), y.iloc[split_at:].copy()
