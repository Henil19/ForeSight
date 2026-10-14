"""Development-only forecasting experiments over a saved AAPL data snapshot.

This module never downloads data and never evaluates the reserved final test.
All candidates use common dates and target observations. Run it with:

    python -m experiments.phase3
"""

from __future__ import annotations

import json
import argparse
import hashlib
import importlib.metadata
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from src.evaluation import calculate_directional_accuracy
from src.features import create_features
from src.models import (
    FEATURE_COLUMNS,
    build_linear_regression,
    build_mlp_scaled_target,
    build_random_forest,
    chronological_train_validation_test_split,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "experiments" / "data" / "aapl_5y_clean.csv"
METADATA_PATH = ROOT / "experiments" / "data" / "aapl_5y_metadata.json"
RESULTS_DIR = ROOT / "experiments" / "results" / "phase3_corrected"
PHASE4_RESULTS_DIR = ROOT / "experiments" / "results" / "phase4_offline"
N_SPLITS = 5
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
MOVING_AVERAGE_WINDOW = 5

BASE_FEATURE_COLUMNS = [
    "Close", "Volume", "Daily_Return", "MA20", "MA50", "MA200",
    "RSI", "MACD", "Volatility",
]
ADDED_FEATURE_COLUMNS = [
    "Return_Lag_1", "Return_Lag_5", "Momentum_5", "Momentum_20",
    "Volume_Change_1",
]


def add_experimental_features(features: pd.DataFrame) -> pd.DataFrame:
    """Append a small causal feature set to the current feature table."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    missing = [name for name in BASE_FEATURE_COLUMNS if name not in features.columns]
    if missing:
        raise ValueError("features are missing required columns: " + ", ".join(missing))
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("features must have a unique chronological index.")

    result = features.copy()
    returns = result["Close"].pct_change(fill_method=None)
    result["Return_Lag_1"] = returns.shift(1)
    result["Return_Lag_5"] = returns.shift(5)
    result["Momentum_5"] = result["Close"].pct_change(5, fill_method=None)
    result["Momentum_20"] = result["Close"].pct_change(20, fill_method=None)
    result["Volume_Change_1"] = result["Volume"].pct_change(fill_method=None)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=BASE_FEATURE_COLUMNS + ADDED_FEATURE_COLUMNS)


def make_experimental_targets(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create comparable next-close and next-return targets indexed at origin t.

    The return target is ``(Close[t+1] / Close[t]) - 1``. The last row has no
    next observation and is omitted from all returned objects.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    required = FEATURE_COLUMNS
    missing = [name for name in required if name not in features.columns]
    if missing:
        raise ValueError("features are missing required columns: " + ", ".join(missing))
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("features must have a unique chronological index.")
    price_target = features["Close"].shift(-1).rename("Target")
    valid = price_target.notna()
    X = features.loc[valid].copy()
    price_target = price_target.loc[valid].copy()
    origin_close = features.loc[X.index, "Close"]
    return_target = (price_target / origin_close - 1.0).rename("Next_Return")
    if not (X.index.equals(price_target.index) and X.index.equals(return_target.index)):
        raise ValueError("Experimental features and targets are not aligned.")
    return X, price_target, return_target


def baseline_price_predictions(
    method: str,
    historical_close: pd.Series,
    origins: pd.Index,
    moving_average_window: int = MOVING_AVERAGE_WINDOW,
) -> pd.Series:
    """Predict next prices from closes available through each forecast origin.

    ``previous_close`` repeats Close[t]. ``drift`` adds the expanding mean
    close change per observation to Close[t]. ``moving_average`` predicts the
    trailing-window mean including Close[t].
    """
    if not isinstance(historical_close, pd.Series):
        raise TypeError("historical_close must be a pandas Series.")
    if historical_close.empty or not historical_close.index.is_monotonic_increasing:
        raise ValueError("historical_close must be non-empty and chronologically ordered.")
    if historical_close.index.has_duplicates:
        raise ValueError("historical_close must have a unique index.")
    if isinstance(moving_average_window, bool) or not isinstance(moving_average_window, int) or moving_average_window < 1:
        raise ValueError("moving_average_window must be a positive integer.")
    missing = origins.difference(historical_close.index)
    if not missing.empty:
        raise ValueError("Every forecast origin must exist in historical_close.")

    values: list[float] = []
    for origin in origins:
        history = historical_close.loc[:origin]
        current = float(history.iloc[-1])
        if method == "previous_close":
            prediction = current
        elif method == "drift":
            intervals = len(history) - 1
            mean_change = (current - float(history.iloc[0])) / intervals if intervals else 0.0
            prediction = current + mean_change
        elif method == "moving_average":
            prediction = float(history.iloc[-moving_average_window:].mean())
        else:
            raise ValueError("method must be previous_close, drift, or moving_average.")
        values.append(float(prediction))
    return pd.Series(values, index=origins, name=method, dtype=float)


def convert_return_predictions_to_prices(
    previous_close: pd.Series,
    predicted_return: pd.Series,
) -> pd.Series:
    """Convert predicted simple returns to next-close prices at matching origins."""
    if not isinstance(previous_close, pd.Series) or not isinstance(predicted_return, pd.Series):
        raise TypeError("previous_close and predicted_return must be pandas Series.")
    if not previous_close.index.equals(predicted_return.index):
        raise ValueError("previous_close and predicted_return must have identical indexes.")
    return (previous_close * (1.0 + predicted_return)).rename("Predicted_Close")


def _metric_row(
    candidate: str,
    split: str,
    actual_price: pd.Series,
    previous_close: pd.Series,
    predicted_price: pd.Series,
    actual_target: pd.Series,
    predicted_target: pd.Series,
    target_kind: str,
    convergence_warning: bool,
) -> dict[str, Any]:
    if not (
        actual_price.index.equals(previous_close.index)
        and actual_price.index.equals(predicted_price.index)
        and actual_price.index.equals(actual_target.index)
        and actual_price.index.equals(predicted_target.index)
    ):
        raise ValueError("Candidate predictions and targets must use identical dates.")
    return {
        "Split": split,
        "Candidate": candidate,
        "Target": target_kind,
        "MAE": float(mean_absolute_error(actual_price, predicted_price)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_price, predicted_price))),
        "R2": float(r2_score(actual_price, predicted_price)),
        "Directional_Accuracy": calculate_directional_accuracy(
            previous_close, actual_price, predicted_price,
        ),
        "Target_MAE": float(mean_absolute_error(actual_target, predicted_target)),
        "Target_RMSE": float(np.sqrt(mean_squared_error(actual_target, predicted_target))),
        "Prediction_Min": float(predicted_price.min()),
        "Prediction_Max": float(predicted_price.max()),
        "ConvergenceWarning": bool(convergence_warning),
    }


def _model_specs() -> list[dict[str, Any]]:
    builders = {
        "Linear Regression": build_linear_regression,
        "Random Forest": build_random_forest,
        "MLP": build_mlp_scaled_target,
    }
    specs: list[dict[str, Any]] = []
    for target in ("price", "return"):
        for feature_set in ("current", "causal_plus"):
            for model_name, builder in builders.items():
                candidate = f"{model_name} [{target}; {feature_set}]"
                specs.append({
                    "candidate": candidate,
                    "model_name": model_name,
                    "target": target,
                    "feature_set": feature_set,
                    "builder": builder,
                })
    return specs


def _predict_candidate(
    spec: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_return: pd.Series,
    X_eval: pd.DataFrame,
    actual_price: pd.Series,
    actual_return: pd.Series,
    previous_close: pd.Series,
) -> tuple[pd.Series, pd.Series, bool]:
    model = spec["builder"]()
    fit_target = y_train if spec["target"] == "price" else y_train_return
    convergence_warning = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, fit_target)
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)

    raw_prediction = pd.Series(model.predict(X_eval), index=X_eval.index, dtype=float)
    if spec["target"] == "return":
        predicted_price = convert_return_predictions_to_prices(previous_close, raw_prediction)
        return predicted_price.rename(spec["candidate"]), raw_prediction, convergence_warning
    return raw_prediction.rename(spec["candidate"]), raw_prediction, convergence_warning


def _prediction_rows(
    split: str,
    candidate: str,
    origins: pd.Index,
    target_dates: pd.Index,
    actual: pd.Series,
    predicted: pd.Series,
) -> list[dict[str, Any]]:
    if not (origins.equals(actual.index) and origins.equals(predicted.index)):
        raise ValueError("Prediction report rows must share identical forecast origins.")
    return [
        {
            "Split": split,
            "Candidate": candidate,
            "Origin_Date": origin,
            "Target_Date": target_date,
            "Target_Units": "Close price (USD)",
            "Actual_Close_USD": float(actual.loc[origin]),
            "Predicted_Close_USD": float(predicted.loc[origin]),
            "Error_USD_Predicted_Minus_Actual": float(predicted.loc[origin] - actual.loc[origin]),
        }
        for origin, target_date in zip(origins, target_dates, strict=True)
    ]


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in ("numpy", "pandas", "scikit-learn", "matplotlib", "yfinance"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _make_phase4_reports(
    validation: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write separate offline development reports; no final-test rows are accepted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if predictions.empty or (predictions["Split"] == "Final Test").any():
        raise ValueError("Phase 4 reports require development predictions only.")
    summary = (
        folds.groupby("Candidate", as_index=False)
        .agg(
            Mean_MAE_USD=("MAE", "mean"),
            Mean_RMSE_USD=("RMSE", "mean"),
            RMSE_Std_USD=("RMSE", "std"),
            Mean_R2=("R2", "mean"),
            Mean_Directional_Accuracy=("Directional_Accuracy", "mean"),
        )
        .sort_values("Mean_RMSE_USD", ignore_index=True)
    )
    metrics = {
        "target_units": "USD per share for price metrics; return metrics are dimensionless fractions.",
        "definitions": {
            "MAE": "Mean absolute error in next-close USD per share.",
            "RMSE": "Square root of mean squared next-close error, in USD per share.",
            "R2": "Coefficient of determination on next-close prices.",
            "Directional_Accuracy": "Fraction of predictions whose sign of movement from Close[t] matches the actual next-close movement.",
            "Target_MAE": "MAE in the candidate's native target units; USD for price targets and fractional return for return targets.",
            "Target_RMSE": "RMSE in the candidate's native target units; USD for price targets and fractional return for return targets.",
            "Error_USD_Predicted_Minus_Actual": "Predicted next close minus actual next close, in USD per share.",
        },
    }
    def export_metric_units(frame: pd.DataFrame) -> pd.DataFrame:
        exported = frame.copy()
        exported["Target_Units"] = np.where(exported["Target"].eq("return"), "fractional return", "USD/share")
        return exported.rename(columns={
            "MAE": "MAE_USD", "RMSE": "RMSE_USD",
            "Prediction_Min": "Prediction_Min_USD", "Prediction_Max": "Prediction_Max_USD",
        })

    validation_export = export_metric_units(validation)
    folds_export = export_metric_units(folds)
    validation_export.to_csv(output_dir / "phase4_validation_comparison.csv", index=False)
    folds_export.to_csv(output_dir / "phase4_fold_results.csv", index=False)
    summary.to_csv(output_dir / "phase4_model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "phase4_development_predictions.csv", index=False)
    (output_dir / "phase4_model_comparison.json").write_text(
        json.dumps({"metrics": metrics, "results": summary.to_dict(orient="records")}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "phase4_fold_results.json").write_text(
        json.dumps({"metrics": metrics, "results": folds_export.to_dict(orient="records")}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "phase4_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8",
    )

    validation_predictions = predictions[predictions["Split"] == "Validation"]
    best = validation.sort_values("RMSE").iloc[0]["Candidate"]
    shown = [best, "Previous Close", "Expanding Drift", f"{MOVING_AVERAGE_WINDOW}-day Moving Average"]
    shown = [name for name in shown if name in set(validation_predictions["Candidate"])]
    selected = validation_predictions[validation_predictions["Candidate"].isin(shown)]
    fig, ax = plt.subplots(figsize=(12, 6))
    actual_rows = selected[selected["Candidate"] == best]
    if not actual_rows.empty:
        ax.plot(actual_rows["Target_Date"], actual_rows["Actual_Close_USD"], label="Actual", color="black", linewidth=2)
    for name in shown:
        rows = selected[selected["Candidate"] == name]
        if not rows.empty:
            ax.plot(rows["Target_Date"], rows["Predicted_Close_USD"], label=name, alpha=0.85)
    ax.set(title="Validation: Actual vs Predicted Close (development only)", ylabel="Close (USD/share)", xlabel="Target date")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in shown:
        rows = selected[selected["Candidate"] == name]
        if not rows.empty:
            ax.plot(rows["Target_Date"], rows["Error_USD_Predicted_Minus_Actual"], label=name, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Validation prediction errors (predicted − actual; development only)", ylabel="Error (USD/share)", xlabel="Target date")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_prediction_errors.png", dpi=150)
    plt.close(fig)

    pivot = folds.pivot(index="Candidate", columns="Split", values="RMSE")
    pivot = pivot.loc[summary["Candidate"]]
    fig, ax = plt.subplots(figsize=(13, 7))
    for candidate, row in pivot.iterrows():
        ax.plot(pivot.columns, row.values, marker="o", alpha=0.8, label=candidate)
    ax.set(title="Development fold RMSE by candidate", ylabel="RMSE (USD/share)", xlabel="Expanding-window fold")
    ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_fold_rmse.png", dpi=150)
    plt.close(fig)


def run_development_experiments(
    results_dir: Path = RESULTS_DIR,
    phase4_reports: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run validation and five-fold development experiments; never score final test."""
    if not DATA_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Run experiments/download_phase3_data.py once to create the local snapshot.")
    results_dir = Path(results_dir)
    file_prefix = "phase4" if phase4_reports else "phase3"
    report_names = (
        f"{file_prefix}_validation_comparison.csv" if phase4_reports else "phase3_validation.csv",
        f"{file_prefix}_fold_results.csv" if phase4_reports else "phase3_development_folds.csv",
        f"{file_prefix}_model_comparison.csv" if phase4_reports else "phase3_development_summary.csv",
        f"{file_prefix}_metadata.json",
    )
    if phase4_reports:
        report_names += (
            "phase4_model_comparison.json", "phase4_fold_results.json",
            "phase4_development_predictions.csv", "phase4_actual_vs_predicted.png",
            "phase4_prediction_errors.png", "phase4_fold_rmse.png",
        )
    result_paths = tuple(results_dir / name for name in report_names)
    if any(path.exists() for path in result_paths):
        raise FileExistsError(f"{file_prefix.title()} result files already exist; refusing to overwrite them.")
    try:
        raw_bytes = DATA_PATH.read_bytes()
        prices = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
        snapshot_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read the saved offline snapshot or its metadata: {exc}") from exc
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    expected_hash = snapshot_metadata.get("clean_sha256")
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Saved cleaned dataset hash does not match snapshot metadata; refusing to run.")
    if prices.empty:
        raise ValueError("Saved cleaned price snapshot is empty.")
    required_price_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_price_columns = [column for column in required_price_columns if column not in prices.columns]
    if missing_price_columns:
        raise ValueError("Saved price snapshot is missing required columns: " + ", ".join(missing_price_columns))
    prices = prices.loc[:, required_price_columns]
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("Saved clean price snapshot contains missing or non-finite OHLCV values.")
    prices.index.name = "Date"
    if prices.index.hasnans or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Saved clean price snapshot must have valid, unique chronological dates.")

    base_features = create_features(prices)
    expanded_features = add_experimental_features(base_features)
    common_base = base_features.loc[expanded_features.index, BASE_FEATURE_COLUMNS]
    X_current, y_price, y_return = make_experimental_targets(common_base)
    X_plus, y_plus_price, y_plus_return = make_experimental_targets(expanded_features)
    if not X_current.index.equals(X_plus.index):
        raise ValueError("Current and added-feature candidates must use identical forecast dates.")
    pd.testing.assert_series_equal(y_price, y_plus_price)
    pd.testing.assert_series_equal(y_return, y_plus_return)

    # Establish split boundaries, then retain only development observations.
    X_train, X_validation, _X_final_test, y_train, y_validation, _y_final_test = (
        chronological_train_validation_test_split(
            X_current, y_price,
            validation_size=VALIDATION_SIZE,
            test_size=TEST_SIZE,
        )
    )
    development_index = X_train.index.append(X_validation.index)
    X_development = X_current.loc[development_index]
    y_development = y_price.loc[development_index]
    return_development = y_return.loc[development_index]
    previous_development = prices.loc[development_index, "Close"]

    # Recompute using the same declared 60/20/20 row counts for both feature sets.
    train_count = len(X_train)
    validation_count = len(X_validation)
    validation_start = train_count
    validation_stop = train_count + validation_count
    X_validation_plus = X_plus.iloc[validation_start:validation_stop]
    if not X_plus.iloc[:train_count].index.equals(X_train.index) or not X_validation_plus.index.equals(X_validation.index):
        raise ValueError("Experimental feature candidates do not share split boundaries.")

    feature_matrices = {"current": X_current, "causal_plus": X_plus}
    close_history = prices["Close"]
    specs = _model_specs()
    validation_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    def target_dates_for(origins: pd.Index) -> pd.Index:
        positions = expanded_features.index.get_indexer(origins)
        if (positions < 0).any() or (positions + 1 >= len(expanded_features.index)).any():
            raise ValueError("A forecast origin cannot be mapped to its next chronological feature date.")
        return expanded_features.index.take(positions + 1)

    validation_origins = X_validation.index
    val_actual_price = y_validation
    val_previous = previous_development.loc[validation_origins]
    val_actual_return = return_development.loc[validation_origins]
    for method, display_name in (
        ("previous_close", "Previous Close"),
        ("drift", "Expanding Drift"),
        ("moving_average", f"{MOVING_AVERAGE_WINDOW}-day Moving Average"),
    ):
        prediction = baseline_price_predictions(method, close_history, validation_origins)
        prediction_rows.extend(_prediction_rows(
            "Validation", display_name, validation_origins, target_dates_for(validation_origins),
            val_actual_price, prediction,
        ))
        validation_rows.append(_metric_row(
            display_name, "Validation", val_actual_price, val_previous, prediction,
            val_actual_price, prediction, "price", False,
        ))

    for spec in specs:
        X_matrix = feature_matrices[spec["feature_set"]]
        Xtr = X_matrix.loc[X_train.index]
        Xva = X_matrix.loc[X_validation.index]
        price_pred, raw_pred, warned = _predict_candidate(
            spec, Xtr, y_train, return_development.loc[y_train.index],
            Xva, val_actual_price, val_actual_return, val_previous,
        )
        actual_target = val_actual_price if spec["target"] == "price" else val_actual_return
        validation_rows.append(_metric_row(
            spec["candidate"], "Validation", val_actual_price, val_previous,
            price_pred, actual_target, raw_pred, spec["target"], warned,
        ))
        prediction_rows.extend(_prediction_rows(
            "Validation", spec["candidate"], validation_origins, target_dates_for(validation_origins),
            val_actual_price, price_pred,
        ))

    min_train_size = max(100, len(X_development) // 4)
    if min_train_size >= len(X_development) or len(X_development) - min_train_size < 2 * N_SPLITS:
        raise ValueError(
            f"Development set cannot produce {N_SPLITS} folds with min_train_size={min_train_size}."
        )
    fold_windows = np.array_split(np.arange(min_train_size, len(X_development)), N_SPLITS)
    for fold_number, test_positions in enumerate(fold_windows, start=1):
        train_stop = int(test_positions[0])
        test_stop = int(test_positions[-1]) + 1
        train_index = X_development.index[:train_stop]
        test_index = X_development.index[test_positions]
        actual_price = y_development.loc[test_index]
        actual_return = return_development.loc[test_index]
        previous_close = previous_development.loc[test_index]
        for method, display_name in (
            ("previous_close", "Previous Close"),
            ("drift", "Expanding Drift"),
            ("moving_average", f"{MOVING_AVERAGE_WINDOW}-day Moving Average"),
        ):
            prediction = baseline_price_predictions(method, close_history, test_index)
            prediction_rows.extend(_prediction_rows(
                f"Fold {fold_number}", display_name, test_index, target_dates_for(test_index),
                actual_price, prediction,
            ))
            fold_rows.append(_metric_row(
                display_name, fold_number, actual_price, previous_close, prediction,
                actual_price, prediction, "price", False,
            ))
        for spec in specs:
            X_matrix = feature_matrices[spec["feature_set"]]
            Xtr = X_matrix.loc[train_index]
            Xte = X_matrix.loc[test_index]
            fold_train_price = y_development.loc[train_index]
            price_pred, raw_pred, warned = _predict_candidate(
                spec, Xtr, fold_train_price, return_development.loc[train_index],
                Xte, actual_price, actual_return, previous_close,
            )
            actual_target = actual_price if spec["target"] == "price" else actual_return
            fold_rows.append(_metric_row(
                spec["candidate"], fold_number, actual_price, previous_close,
                price_pred, actual_target, raw_pred, spec["target"], warned,
            ))
            prediction_rows.extend(_prediction_rows(
                f"Fold {fold_number}", spec["candidate"], test_index, target_dates_for(test_index),
                actual_price, price_pred,
            ))

    validation_results = pd.DataFrame(validation_rows).sort_values("RMSE", ignore_index=True)
    fold_results = pd.DataFrame(fold_rows)
    summary = (
        fold_results.groupby("Candidate", as_index=False)
        .agg(
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            Mean_R2=("R2", "mean"),
            Mean_Directional_Accuracy=("Directional_Accuracy", "mean"),
            Mean_Target_MAE=("Target_MAE", "mean"),
            Mean_Target_RMSE=("Target_RMSE", "mean"),
            Convergence_Warnings=("ConvergenceWarning", "sum"),
        )
        .sort_values("Mean_RMSE", ignore_index=True)
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(prediction_rows)
    split_boundary = {
        "labeled_observations": int(len(X_current)),
        "train": {"count": int(len(X_train)), "first_date": str(X_train.index[0].date()), "last_date": str(X_train.index[-1].date())},
        "validation": {"count": int(len(X_validation)), "first_date": str(X_validation.index[0].date()), "last_date": str(X_validation.index[-1].date())},
        "final_test": {
            "count": int(len(_X_final_test)),
            "first_origin_date": str(_X_final_test.index[0].date()),
            "last_origin_date": str(_X_final_test.index[-1].date()),
            "status": "reserved_unscored",
        },
        "development_folds": [
            {"split": int(number), "train_count": int(test[0]), "test_count": int(len(test)), "train_last_date": str(X_development.index[int(test[0]) - 1].date()), "test_first_date": str(X_development.index[int(test[0])].date()), "test_last_date": str(X_development.index[int(test[-1])].date())}
            for number, test in enumerate(fold_windows, start=1)
        ],
    }
    run_metadata = {
        "run_mode": "offline_saved_snapshot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_metadata": snapshot_metadata,
        "clean_dataset_sha256": actual_hash,
        "data_date_range": {"first": str(prices.index[0].date()), "last": str(prices.index[-1].date())},
        "split_boundaries": split_boundary,
        "random_seeds": {"RandomForestRegressor": 42, "MLPRegressor": 42, "LinearRegression": None},
        "model_configuration": {
            "Linear Regression": "StandardScaler feature pipeline; LinearRegression defaults.",
            "Random Forest": "100 trees, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1.",
            "MLP": "StandardScaler features and training-fold StandardScaler target; 32-unit ReLU hidden layer, Adam, max_iter=300, random_state=42.",
        },
        "software_versions": _software_versions(),
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "walk_forward_splits": N_SPLITS,
        "min_train_size": min_train_size,
        "test_set_used": False,
        "selection_rule_for_exploration": "Validation RMSE, supported by mean development-fold RMSE; final test not scored.",
        "common_supervised_observations": int(len(X_current)),
        "development_observations": int(len(X_development)),
        "validation_observations": int(len(X_validation)),
        "feature_columns_added": ADDED_FEATURE_COLUMNS,
        "return_target": "(Close[t+1] / Close[t]) - 1; price prediction = Close[t] * (1 + predicted return).",
        "drift_baseline": "Close[t] + (Close[t] - first historical Close) / prior observation count, estimated through origin t.",
        "moving_average_baseline": f"Mean of trailing {MOVING_AVERAGE_WINDOW} closes through origin t, including Close[t].",
        "previous_close_baseline": "Predict Close[t] for the next observation.",
    }
    if phase4_reports:
        _make_phase4_reports(validation_results, fold_results, predictions, run_metadata, results_dir)
    else:
        validation_results.to_csv(result_paths[0], index=False)
        fold_results.to_csv(result_paths[1], index=False)
        summary.to_csv(result_paths[2], index=False)
        (results_dir / "phase3_metadata.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8",
        )
    return validation_results, fold_results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline development experiments on the saved AAPL snapshot.")
    parser.add_argument("--offline", action="store_true", help="Use only the saved cleaned snapshot; no network access is performed.")
    parser.add_argument("--output-dir", type=Path, default=PHASE4_RESULTS_DIR, help="New, separate report directory (must not already contain report files).")
    args = parser.parse_args()
    if not args.offline:
        parser.error("--offline is required; this experiment runner never downloads data.")
    validation, folds, summary = run_development_experiments(args.output_dir, phase4_reports=True)
    print("Development validation results:")
    print(validation.to_string(index=False))
    print("\nDevelopment fold results:")
    print(folds.to_string(index=False))
    print("\nDevelopment fold summary:")
    print(summary.to_string(index=False))
    print(f"\nOffline report files: {args.output_dir}")


if __name__ == "__main__":
    main()
