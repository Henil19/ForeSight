"""Metrics and comparison helpers for next-day closing-price models."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
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


def walk_forward_validate(
    models: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    previous_close: pd.Series,
    n_splits: int = 5,
    min_train_size: int | None = None,
) -> pd.DataFrame:
    """Evaluate models with chronological, expanding-window validation.

    The input order is used as-is and must already be chronological. Each test
    window follows its training window; training expands to include all earlier
    observations. Fresh clones are fitted independently in each split.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series) or not isinstance(previous_close, pd.Series):
        raise TypeError("y and previous_close must be pandas Series.")
    if X.empty or y.empty or previous_close.empty:
        raise ValueError("X, y, and previous_close must be non-empty.")
    if not X.index.equals(y.index) or not X.index.equals(previous_close.index):
        raise ValueError("X, y, and previous_close must have identical indexes.")
    if not X.index.is_monotonic_increasing or X.index.has_duplicates:
        raise ValueError("Input indexes must be unique and already chronologically ordered.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer.")

    row_count = len(X)
    if min_train_size is None:
        min_train_size = max(2, row_count // (n_splits + 1))
    elif (
        isinstance(min_train_size, bool)
        or not isinstance(min_train_size, (int, np.integer))
        or min_train_size < 2
    ):
        raise ValueError("min_train_size must be an integer of at least 2.")
    if min_train_size >= row_count:
        raise ValueError("min_train_size must be smaller than the dataset size.")
    if row_count - min_train_size < 2 * n_splits:
        raise ValueError(
            "Dataset and min_train_size cannot produce the requested splits "
            "with at least two test observations per split."
        )

    test_windows = np.array_split(np.arange(min_train_size, row_count), n_splits)
    rows: list[dict[str, Any]] = []
    for split_number, test_positions in enumerate(test_windows, start=1):
        test_start = int(test_positions[0])
        test_stop = int(test_positions[-1]) + 1
        train_X = X.iloc[:test_start]
        train_y = y.iloc[:test_start]
        test_X = X.iloc[test_start:test_stop]
        test_y = y.iloc[test_start:test_stop]
        test_previous_close = previous_close.iloc[test_start:test_stop]

        if not train_X.index[-1] < test_X.index[0]:
            raise ValueError("Each test window must occur strictly after its training window.")

        for name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(train_X, train_y)
            rows.append({
                "Split": split_number,
                "Model": name,
                **evaluate_model(fitted_model, test_X, test_y, test_previous_close),
            })
        rows.append({
            "Split": split_number,
            "Model": "Naive Baseline",
            **evaluate_naive_baseline(test_y, test_previous_close),
        })

    return pd.DataFrame(
        rows,
        columns=["Split", "Model", "MAE", "RMSE", "R2", "Directional_Accuracy"],
    )


def summarize_walk_forward_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward metrics by model and sort by mean RMSE."""
    required = {"Model", "MAE", "RMSE", "R2", "Directional_Accuracy"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("Walk-forward results are missing required columns: " + ", ".join(missing))
    if results.empty:
        raise ValueError("Walk-forward results must contain at least one row.")
    summary = (
        results.groupby("Model", as_index=False)
        .agg(
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_R2=("R2", "mean"),
            Mean_Directional_Accuracy=("Directional_Accuracy", "mean"),
            RMSE_Std=("RMSE", "std"),
        )
        .sort_values("Mean_RMSE", ascending=True, ignore_index=True)
    )
    return summary
