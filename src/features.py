"""Causal technical and time-series features for historical OHLCV data."""

from __future__ import annotations

import pandas as pd

_INPUT_COLUMNS = ("Close", "Volume")
_FEATURE_COLUMNS = (
    "Close",
    "Volume",
    "Daily_Return",
    "MA20",
    "MA50",
    "MA200",
    "RSI",
    "MACD",
    "Volatility",
)


def _validate_input(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing = [column for column in _INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Input data is missing required columns: " + ", ".join(missing))
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Input data must have a DatetimeIndex.")
    if frame.index.hasnans:
        raise ValueError("Input DatetimeIndex cannot contain NaT values.")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build causal features, dropping initial rows without sufficient history.

    Output rows are chronological. Moving averages and volatility require their
    full windows; RSI uses Wilder's 14-period exponentially smoothed gains and
    losses; MACD uses 12- and 26-period EMAs. Initial warm-up rows containing
    undefined feature values are dropped so the returned feature matrix is
    complete. Each feature uses only the current and prior observations.
    """
    _validate_input(df)
    result = df.loc[:, list(_INPUT_COLUMNS)].copy().sort_index(kind="stable")
    if result.index.has_duplicates:
        result = result.loc[~result.index.duplicated(keep="last")]
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    result["Volume"] = pd.to_numeric(result["Volume"], errors="coerce")

    close = result["Close"]
    daily_return = close.pct_change(fill_method=None)
    result["Daily_Return"] = daily_return
    for window in (20, 50, 200):
        result[f"MA{window}"] = close.rolling(window=window, min_periods=window).mean()

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100)
    result["RSI"] = rsi

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = ema_12 - ema_26
    result["Volatility"] = daily_return.rolling(window=20, min_periods=20).std()

    result = result.replace([float("inf"), float("-inf")], pd.NA)
    # Required by the MA200 warm-up: retain only rows with every feature defined.
    result = result.dropna(subset=list(_FEATURE_COLUMNS))
    return result.loc[:, list(_FEATURE_COLUMNS)]
