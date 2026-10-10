"""Metrics and comparison helpers for next-day closing-price models."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_directional_accuracy(
    previous_close: pd.Series, actual: pd.Series, predicted: pd.Series,
) -> float:
    """Return share of matching movement directions (zero matches zero only).

    A predicted or actual unchanged close has direction zero, so it is correct
    only when both actual and predicted movements are unchanged.
    """
    if not (len(previous_close) == len(actual) == len(predicted)):
        raise ValueError("previous_close, actual, and predicted must have equal lengths.")
    if not (previous_close.index.equals(actual.index) and actual.index.equals(predicted.index)):
        raise ValueError("previous_close, actual, and predicted must have identical indexes.")
    if len(actual) == 0:
        raise ValueError("directional accuracy requires at least one observation.")
    actual_direction = np.sign(actual.to_numpy() - previous_close.to_numpy())
    predicted_direction = np.sign(predicted.to_numpy() - previous_close.to_numpy())
    return float(np.mean(actual_direction == predicted_direction))


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series,
    previous_close: pd.Series,
) -> dict[str, float]:
    """Evaluate predictions on held-out data and return full-precision metrics."""
    if not X_test.index.equals(y_test.index) or not y_test.index.equals(previous_close.index):
        raise ValueError("X_test, y_test, and previous_close must have identical indexes.")
    predicted = pd.Series(model.predict(X_test), index=y_test.index)
    return {
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "R2": float(r2_score(y_test, predicted)),
        "Directional_Accuracy": calculate_directional_accuracy(previous_close, y_test, predicted),
    }


def naive_previous_close_predictions(previous_close: pd.Series) -> pd.Series:
    """Return a copy of the previous closes as naive next-close predictions."""
    if not isinstance(previous_close, pd.Series):
        raise TypeError("previous_close must be a pandas Series.")
    return previous_close.copy()


def evaluate_naive_baseline(
    y_test: pd.Series,
    previous_close: pd.Series,
) -> dict[str, float]:
    """Evaluate predicting that the next close equals the previous actual close."""
    if not isinstance(y_test, pd.Series) or not isinstance(previous_close, pd.Series):
        raise TypeError("y_test and previous_close must be pandas Series.")
    if len(y_test) != len(previous_close) or not y_test.index.equals(previous_close.index):
        raise ValueError("y_test and previous_close must have equal lengths and identical indexes.")
    if y_test.empty:
        raise ValueError("Naive baseline evaluation requires non-empty data.")
    predicted = naive_previous_close_predictions(previous_close)
    return {
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "R2": float(r2_score(y_test, predicted)),
        "Directional_Accuracy": calculate_directional_accuracy(previous_close, y_test, predicted),
    }


def compare_models(
    models: Mapping[str, Any], X_test: pd.DataFrame, y_test: pd.Series,
    previous_close: pd.Series, include_naive_baseline: bool = True,
) -> pd.DataFrame:
    """Evaluate models, optionally add a naive baseline, and sort by ascending RMSE."""
    rows = []
    for name, model in models.items():
        rows.append({"Model": name, **evaluate_model(model, X_test, y_test, previous_close)})
    if include_naive_baseline:
        rows.append({"Model": "Naive Baseline", **evaluate_naive_baseline(y_test, previous_close)})
    return pd.DataFrame(
        rows, columns=["Model", "MAE", "RMSE", "R2", "Directional_Accuracy"]
    ).sort_values("RMSE", ascending=True, ignore_index=True)


def calculate_baseline_improvement(
    model_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Calculate relative error gains and directional-accuracy points vs baseline."""
    required = ("MAE", "RMSE", "Directional_Accuracy")
    for label, metrics in (("model_metrics", model_metrics), ("baseline_metrics", baseline_metrics)):
        missing = [key for key in required if key not in metrics]
        if missing:
            raise ValueError(f"{label} is missing required metric keys: " + ", ".join(missing))
    baseline_mae = float(baseline_metrics["MAE"])
    baseline_rmse = float(baseline_metrics["RMSE"])
    if baseline_mae == 0:
        raise ZeroDivisionError("Baseline MAE is zero; percentage improvement is undefined.")
    if baseline_rmse == 0:
        raise ZeroDivisionError("Baseline RMSE is zero; percentage improvement is undefined.")
    return {
        "MAE_Improvement_Percent": (
            (baseline_mae - float(model_metrics["MAE"])) / baseline_mae * 100
        ),
        "RMSE_Improvement_Percent": (
            (baseline_rmse - float(model_metrics["RMSE"])) / baseline_rmse * 100
        ),
        "Directional_Accuracy_Improvement_PercentagePoints": (
            float(model_metrics["Directional_Accuracy"])
            - float(baseline_metrics["Directional_Accuracy"])
        ) * 100,
    }


