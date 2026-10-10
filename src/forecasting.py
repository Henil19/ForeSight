"""Helpers for selecting models and producing next-session forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models import FEATURE_COLUMNS


def select_best_model(comparison: pd.DataFrame) -> str:
    """Return the model name for the row with the lowest RMSE."""
    required = {"Model", "RMSE"}
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError("Model comparison is missing required columns: " + ", ".join(missing))
    if comparison.empty:
        raise ValueError("Model comparison must contain at least one model.")
    rmse = pd.to_numeric(comparison["RMSE"], errors="coerce")
    valid = np.isfinite(rmse.to_numpy(dtype=float))
    if not valid.any():
        raise ValueError("Model comparison must contain at least one finite RMSE value.")
    best_position = int(np.flatnonzero(valid)[np.argmin(rmse.to_numpy(dtype=float)[valid])])
    name = comparison.iloc[best_position]["Model"]
    if pd.isna(name):
        raise ValueError("The best model row must have a model name.")
    return str(name)


def prepare_previous_close_for_test(
    features: pd.DataFrame, X_test: pd.DataFrame,
) -> pd.Series:
    """Align each test feature date with its actual Close at that same date.

    Since the supervised target is Close.shift(-1), this same-date Close is
    the prior close relative to each target. Index order matches ``X_test``.
    """
    if "Close" not in features.columns:
        raise ValueError("features must contain a Close column.")
    missing = X_test.index.difference(features.index)
    if not missing.empty:
        raise ValueError("X_test contains indices not present in features.")
    return features.loc[X_test.index, "Close"].copy()


def forecast_next_trading_day(model: Any, features: pd.DataFrame) -> float:
    """Predict the close after the latest available feature row."""
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError("Features are missing required columns: " + ", ".join(missing))
    if features.empty:
        raise ValueError("features must contain at least one row to forecast.")
    prediction = model.predict(features.loc[[features.index[-1]], FEATURE_COLUMNS])
    if len(prediction) != 1:
        raise ValueError("The model must return exactly one prediction for one feature row.")
    return float(prediction[0])


def get_next_trading_day(last_date: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekday, normalized to midnight.

    This is a Monday-Friday approximation and does not account for exchange
    holidays or market-specific trading calendars.
    """
    date = pd.Timestamp(last_date).normalize()
    if pd.isna(date):
        raise ValueError("last_date must be a valid date.")
    date += pd.Timedelta(days=1)
    while date.weekday() >= 5:
        date += pd.Timedelta(days=1)
    return date
