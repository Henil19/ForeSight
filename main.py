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
    chronological_train_validation_test_split,
    prepare_supervised_data,
    train_models,
)

TICKER = "AAPL"
PERIOD = "5y"
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
NAIVE_BASELINE_NAME = "Naive Baseline"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _select_best_forecast_candidate(comparison: pd.DataFrame, models: dict) -> str:
    """Select by validation RMSE, allowing the naive baseline to win."""
    baseline_rows = comparison.loc[comparison["Model"] == NAIVE_BASELINE_NAME]
    if len(baseline_rows) != 1:
        raise ValueError("Validation comparison must contain exactly one Naive Baseline row.")
    trained_comparison = comparison.loc[comparison["Model"].isin(models)]
    if trained_comparison.empty:
        raise ValueError("Validation comparison must contain at least one trained model.")
    best_trained_name = select_best_model(trained_comparison)
    if best_trained_name not in models:
        raise ValueError("Selected trained model is not present in the trained models.")
    baseline_rmse = float(baseline_rows.iloc[0]["RMSE"])
    best_trained_rmse = float(
        trained_comparison.loc[trained_comparison["Model"] == best_trained_name, "RMSE"].iloc[0]
    )
    if baseline_rmse <= best_trained_rmse:
        return NAIVE_BASELINE_NAME
    return best_trained_name


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


def _forecast_selected_candidate(
    model_name: str,
    model_builders: Mapping[str, Callable[[], Any]],
    X: pd.DataFrame,
    y: pd.Series,
    features: pd.DataFrame,
) -> tuple[Any | None, float]:
    """Refit the selected candidate on all labels and forecast the next close."""
    if model_name == NAIVE_BASELINE_NAME:
        if features.empty or "Close" not in features.columns:
            raise ValueError("features must contain a latest Close value for the naive forecast.")
        return None, float(features["Close"].iloc[-1])
    model = _fit_final_forecast_model(model_name, model_builders, X, y)
    return model, forecast_next_trading_day(model, features)


def _save_plots(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    previous_close: pd.Series,
) -> None:
    """Save historical, technical-indicator, and final-test prediction plots."""
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
    axis.plot(plot_dates, y_test.to_numpy(), label="Actual next-observation Close", linewidth=2)
    for name, model in models.items():
        axis.plot(plot_dates, model.predict(X_test), label=f"{name} prediction", alpha=0.85)
    axis.plot(plot_dates, previous_close.to_numpy(), label="Naive Baseline", alpha=0.85)
    axis.set(title=f"{TICKER} Final-Test Predictions", xlabel="Target date", ylabel="Price ($)")
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
    (
        X_train,
        X_validation,
        X_final_test,
        y_train,
        y_validation,
        y_final_test,
    ) = chronological_train_validation_test_split(
        X,
        y,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
    )
    models = train_models(X_train, y_train)

    previous_close_validation = prepare_previous_close_for_test(features, X_validation)
    validation_comparison = compare_models(
        models, X_validation, y_validation, previous_close_validation,
    )
    best_name = _select_best_forecast_candidate(validation_comparison, models)
    baseline_rmse_value = float(
        validation_comparison.loc[
            validation_comparison["Model"] == NAIVE_BASELINE_NAME, "RMSE"
        ].iloc[0]
    )
    improvement_column = "RMSE_Improvement_vs_Naive_%"
    validation_comparison[improvement_column] = pd.NA
    baseline_rows = validation_comparison["Model"] == NAIVE_BASELINE_NAME
    validation_comparison.loc[baseline_rows, improvement_column] = 0.0
    if baseline_rmse_value != 0:
        ml_rows = validation_comparison["Model"].isin(models)
        validation_comparison.loc[ml_rows, improvement_column] = (
            (baseline_rmse_value - validation_comparison.loc[ml_rows, "RMSE"])
            / baseline_rmse_value
            * 100
        )
    validation_comparison["Selected for Forecast"] = (
        validation_comparison["Model"] == best_name
    )
    print("\nModel comparison (validation set; selection basis):")
    print(validation_comparison.to_string(index=False))
    validation_comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    X_development = pd.concat([X_train, X_validation])
    y_development = pd.concat([y_train, y_validation])

    n_splits = 5
    min_train_size = max(100, len(X_development) // 4)
    if min_train_size >= len(X_development) or len(X_development) - min_train_size < 2 * n_splits:
        raise ValueError(
            "Not enough supervised observations for 5 walk-forward folds "
            f"with min_train_size={min_train_size}; each fold needs at least "
            "two test observations."
        )
    walk_forward_models = {name: builder() for name, builder in model_builders.items()}
    previous_close_development = prepare_previous_close_for_test(features, X_development)
    walk_forward_results = walk_forward_validate(
        walk_forward_models,
        X_development,
        y_development,
        previous_close_development,
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

    final_test_models: dict[str, Any] = {}
    if best_name != NAIVE_BASELINE_NAME:
        final_test_model = model_builders[best_name]()
        final_test_model.fit(X_development, y_development)
        final_test_models[best_name] = final_test_model
    previous_close_final_test = prepare_previous_close_for_test(features, X_final_test)
    final_test_comparison = compare_models(
        final_test_models,
        X_final_test,
        y_final_test,
        previous_close_final_test,
    )
    final_test_comparison.to_csv(OUTPUT_DIR / "final_test_comparison.csv", index=False)
    print("\nFinal test results (selected candidate and naive baseline; no reselection):")
    print(final_test_comparison.to_string(index=False))

    final_forecast_model, predicted_close = _forecast_selected_candidate(
        best_name, model_builders, X, y, features,
    )
    next_day = get_next_trading_day(prices.index[-1])
    (OUTPUT_DIR / "forecast.txt").write_text(
        f"Ticker: {TICKER}\n"
        f"Selected Forecast Candidate (validation RMSE): {best_name}\n"
        f"Final Test Results: see final_test_comparison.csv\n"
        f"Next Trading Day (weekday approximation): {next_day.date()}\n"
        f"Predicted Close: ${predicted_close:.2f}\n\n"
        "Educational ML forecast only; not financial advice.\n",
        encoding="utf-8",
    )
    _save_plots(
        prices,
        features,
        final_test_models,
        X_final_test,
        y_final_test,
        previous_close_final_test,
    )

    print("\nEducational ML forecast (not financial advice):")
    print(f"Ticker: {TICKER}")
    print(f"Selected Forecast Candidate (validation RMSE): {best_name}")
    if final_forecast_model is None:
        print("Forecast method: Naive Baseline (previous Close)")
    else:
        print(f"Forecast model refit on all labeled observations: {best_name}")
    print(f"Next Trading Day: {next_day.date()}")
    print(f"Predicted Close: ${predicted_close:.2f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
