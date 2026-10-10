"""Unit tests for market-data cleaning and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src import data
from src.features import create_features


def synthetic_ohlcv(rows: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=rows, freq="B", name="Date")
    close = 80 + np.arange(rows) * 0.17 + np.sin(np.arange(rows) / 5)
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000 + np.arange(rows) * 100,
        },
        index=dates,
    )


def test_download_cleans_order_duplicates_missing_and_multiindex(monkeypatch):
    raw = synthetic_ohlcv(5).iloc[[2, 0, 1, 1, 4]].copy()
    raw.iloc[3, raw.columns.get_loc("Close")] = 123.0  # last duplicate wins
    raw.iloc[4, raw.columns.get_loc("Volume")] = np.nan
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["TEST"]])
    monkeypatch.setattr(data.yf, "download", lambda *args, **kwargs: raw)

    cleaned = data.download_stock_data(" TEST ")

    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert cleaned.loc[raw.index[2], "Close"] == 123.0
    assert len(cleaned) == 3


def test_download_rejects_invalid_ticker_without_network(monkeypatch):
    def should_not_download(*args, **kwargs):
        pytest.fail("download should not be called for invalid ticker")

    monkeypatch.setattr(data.yf, "download", should_not_download)
    with pytest.raises(ValueError, match="ticker"):
        data.download_stock_data("  ")
    with pytest.raises(ValueError, match="ticker"):
        data.download_stock_data("BAD/TICKER")


def test_download_requires_ohlcv_columns(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *args, **kwargs: pd.DataFrame({"Close": [1]}))
    with pytest.raises(ValueError, match="missing required columns"):
        data.download_stock_data("TEST")


def test_download_rejects_empty_response(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="No market data"):
        data.download_stock_data("TEST")


def test_features_include_expected_values_after_warmup():
    source = synthetic_ohlcv()
    features = create_features(source)

    assert list(features.columns) == [
        "Close", "Volume", "Daily_Return", "MA20", "MA50", "MA200",
        "RSI", "MACD", "Volatility",
    ]
    assert isinstance(features.index, pd.DatetimeIndex)
    assert features.index.is_monotonic_increasing
    assert features.index[0] == source.index[199]
    assert features.notna().all().all()
    assert features["RSI"].between(0, 100).all()
    assert features["MACD"].notna().any()
    assert features["Volatility"].gt(0).any()
    assert features["MA200"].iloc[0] == pytest.approx(source["Close"].iloc[:200].mean())


def test_feature_creation_does_not_mutate_input():
    source = synthetic_ohlcv()
    before = source.copy(deep=True)
    create_features(source)
    pd.testing.assert_frame_equal(source, before)


def test_features_use_no_future_values():
    source = synthetic_ohlcv()
    baseline = create_features(source)
    changed = source.copy()
    changed.loc[changed.index[220]:, ["Close", "Volume"]] *= 3
    revised = create_features(changed)

    common_prefix = baseline.index[baseline.index < source.index[220]]
    pd.testing.assert_frame_equal(baseline.loc[common_prefix], revised.loc[common_prefix])


def test_features_validate_required_columns_and_datetime_index():
    with pytest.raises(ValueError, match="required columns"):
        create_features(pd.DataFrame({"Close": [1]}, index=pd.date_range("2024-01-01", periods=1)))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        create_features(pd.DataFrame({"Close": [1], "Volume": [2]}, index=[0]))


def test_features_sort_and_deduplicate_chronologically():
    source = synthetic_ohlcv(205)
    source = pd.concat([source.iloc[::-1], source.iloc[[100]]])
    result = create_features(source)
    assert result.index.is_monotonic_increasing
    assert not result.index.has_duplicates
