"""Run the educational next-trading-day stock-price forecasting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import download_stock_data
from src.evaluation import (
    compare_models,
    summarize_walk_forward_results,
    walk_forward_validate,
)
from src.features import create_features
from src.forecasting import (
    forecast_next_trading_day,
    get_next_trading_day,
    prepare_previous_close_for_test,
    select_best_model,
)
from src.models import (
    build_linear_regression,
    build_mlp_scaled_target,
    build_random_forest,
    chronological_split,
    prepare_supervised_data,
    train_models,
)

TICKER = "AAPL"
PERIOD = "5y"
TEST_SIZE = 0.2
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _select_best_trained_model(comparison: pd.DataFrame, models: dict) -> str:
    """Choose the lowest-RMSE trained model, excluding benchmark-only rows."""
    trained_comparison = comparison.loc[comparison["Model"].isin(models)]
    best_name = select_best_model(trained_comparison)
    if best_name not in models:
        raise ValueError("Selected best model is not present in the trained models.")
    return best_name


def _fit_final_forecast_model(
    model_name: str,
    model_builders: Mapping[str, Callable[[], Any]],
    X: pd.DataFrame,
    y: pd.Series,
) -> Any:
    """Build a fresh selected model and fit it on all labeled observations."""
    if model_name not in model_builders:
        raise ValueError(f"No model builder is registered for {model_name!r}.")
    final_model = model_builders[model_name]()
    final_model.fit(X, y)
    return final_model


def _save_plots(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save historical, technical-indicator, and held-out prediction plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(prices.index, prices["Close"], label="Close")
    axis.set(title=f"{TICKER} Historical Closing Price", xlabel="Date", ylabel="Price ($)")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "historical_close.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    for column in ("Close", "MA20", "MA50", "MA200"):
        axis.plot(features.index, features[column], label=column)
    axis.set(title=f"{TICKER} Closing Price and Moving Averages", xlabel="Date", ylabel="Price ($)")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "technical_indicators.png", dpi=150)
    plt.close(figure)

    # Map each test row to the immediately following date in the feature index.
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("features must have a unique, chronologically ordered index to map target dates.")
    feature_positions = features.index.get_indexer(X_test.index)
    if (feature_positions < 0).any() or (feature_positions >= len(features.index) - 1).any():
        raise ValueError("Every X_test row must map to a feature date with a following chronological feature date.")
    plot_dates = features.index.take(feature_positions + 1)
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(plot_dates, y_test.to_numpy(), label="Actual next-day Close", linewidth=2)
    for name, model in models.items():
        axis.plot(plot_dates, model.predict(X_test), label=f"{name} prediction", alpha=0.85)
    axis.set(title=f"{TICKER} Held-Out Next-Day Predictions", xlabel="Target date", ylabel="Price ($)")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_predictions.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Download data, train and evaluate models, then save a one-step forecast."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = download_stock_data(TICKER, period=PERIOD)
    features = create_features(prices)
    X, y = prepare_supervised_data(features)
    model_builders = {
        "Linear Regression": build_linear_regression,
        "Random Forest": build_random_forest,
        "MLP": build_mlp_scaled_target,
    }
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=TEST_SIZE)
    models = train_models(X_train, y_train)

    previous_close = prepare_previous_close_for_test(features, X_test)
    comparison = compare_models(models, X_test, y_test, previous_close)
    baseline_rmse = comparison.loc[
        comparison["Model"] == "Naive Baseline", "RMSE"
    ]
    if baseline_rmse.empty:
        raise ValueError("Model comparison is missing the Naive Baseline row.")
    baseline_rmse_value = float(baseline_rmse.iloc[0])
    improvement_column = "RMSE_Improvement_vs_Naive_%"
    comparison[improvement_column] = pd.NA
    baseline_rows = comparison["Model"] == "Naive Baseline"
    comparison.loc[baseline_rows, improvement_column] = 0.0
    if baseline_rmse_value != 0:
        ml_rows = comparison["Model"].isin(models)
        comparison.loc[ml_rows, improvement_column] = (
            (baseline_rmse_value - comparison.loc[ml_rows, "RMSE"])
            / baseline_rmse_value
            * 100
        )
    print("\nModel comparison (held-out test set):")
    print(comparison.to_string(index=False))
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    n_splits = 5
    min_train_size = max(100, len(X) // 4)
    if min_train_size >= len(X) or len(X) - min_train_size < 2 * n_splits:
        raise ValueError(
            "Not enough supervised observations for 5 walk-forward folds "
            f"with min_train_size={min_train_size}; each fold needs at least "
            "two test observations."
        )
    walk_forward_models = {name: builder() for name, builder in model_builders.items()}
    previous_close_all = prepare_previous_close_for_test(features, X)
    walk_forward_results = walk_forward_validate(
        walk_forward_models,
        X,
        y,
        previous_close_all,
        n_splits=n_splits,
        min_train_size=min_train_size,
    )
    walk_forward_summary = summarize_walk_forward_results(walk_forward_results)
    walk_forward_results.to_csv(OUTPUT_DIR / "walk_forward_results.csv", index=False)
    walk_forward_summary.to_csv(OUTPUT_DIR / "walk_forward_summary.csv", index=False)
    print("\nWalk-forward validation results:")
    print(walk_forward_results.to_string(index=False))
    print("\nWalk-forward summary:")
    print(walk_forward_summary.to_string(index=False))

    best_name = _select_best_trained_model(comparison, models)
    final_forecast_model = _fit_final_forecast_model(best_name, model_builders, X, y)
    predicted_close = forecast_next_trading_day(final_forecast_model, features)
    next_day = get_next_trading_day(prices.index[-1])
    (OUTPUT_DIR / "forecast.txt").write_text(
        f"Ticker: {TICKER}\n"
        f"Best Trained Model: {best_name}\n"
        f"Next Trading Day (weekday approximation): {next_day.date()}\n"
        f"Predicted Close: ${predicted_close:.2f}\n\n"
        "Educational ML forecast only; not financial advice.\n",
        encoding="utf-8",
    )
    _save_plots(prices, features, models, X_test, y_test)

    print("\nEducational ML forecast (not financial advice):")
    print(f"Ticker: {TICKER}")
    print(f"Best Trained Model: {best_name}")
    print(f"Next Trading Day: {next_day.date()}")
    print(f"Predicted Close: ${predicted_close:.2f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
