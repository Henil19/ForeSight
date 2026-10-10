"""Download and clean historical OHLCV market data."""

from __future__ import annotations

import re

import pandas as pd
import yfinance as yf

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.^=_-]+$")


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the OHLCV column level from yfinance's occasional MultiIndex."""
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    levels = [
        level
        for level in range(frame.columns.nlevels)
        if {str(value).strip().lower() for value in frame.columns.get_level_values(level)}
        .intersection({name.lower() for name in _REQUIRED_COLUMNS})
    ]
    if not levels:
        return frame
    # The field level is the one containing the greatest number of expected fields.
    field_level = max(
        levels,
        key=lambda level: len(
            {str(value).strip().lower() for value in frame.columns.get_level_values(level)}
            .intersection({name.lower() for name in _REQUIRED_COLUMNS})
        ),
    )
    frame = frame.copy()
    frame.columns = frame.columns.get_level_values(field_level)
    return frame


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download and clean one ticker's daily OHLCV data using yfinance.

    Rows with missing or non-numeric OHLCV values are removed. Duplicate dates
    keep the last downloaded row. No data is synthesized when the download fails.
    """
    if not isinstance(ticker, str) or not ticker.strip() or not _TICKER_PATTERN.fullmatch(ticker.strip()):
        raise ValueError("ticker must be a non-empty valid symbol (for example, 'MSFT' or 'BRK-B').")
    if not isinstance(period, str) or not period.strip():
        raise ValueError("period must be a non-empty yfinance period string.")

    symbol = ticker.strip()
    try:
        data = yf.download(symbol, period=period.strip(), progress=False, auto_adjust=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to download market data for {symbol!r}: {exc}") from exc
    if data is None or data.empty:
        raise ValueError(f"No market data was returned for ticker {symbol!r}.")

    data = _flatten_yfinance_columns(data.copy())
    normalized = {str(column).strip().lower(): column for column in data.columns}
    aliases = {"adj close": "Adj Close", "adj_close": "Adj Close"}
    rename: dict[object, str] = {}
    for key, canonical in ((name.lower(), name) for name in _REQUIRED_COLUMNS):
        if key in normalized:
            rename[normalized[key]] = canonical
    for key, canonical in aliases.items():
        if key in normalized:
            rename[normalized[key]] = canonical
    data = data.rename(columns=rename)

    missing = [column for column in _REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Downloaded data is missing required columns: " + ", ".join(missing))

    try:
        index = pd.to_datetime(data.index, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError("Downloaded market data index could not be interpreted as dates.") from exc
    valid_index = ~pd.isna(index)
    data = data.loc[valid_index].copy()
    data.index = pd.DatetimeIndex(index[valid_index], name="Date")
    if data.empty:
        raise ValueError(f"No valid dated market data was returned for ticker {symbol!r}.")

    data = data.sort_index(kind="stable")
    data = data.loc[~data.index.duplicated(keep="last"), list(_REQUIRED_COLUMNS)]
    for column in _REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(_REQUIRED_COLUMNS))
    if data.empty:
        raise ValueError(f"No complete OHLCV rows were returned for ticker {symbol!r}.")
    return data
