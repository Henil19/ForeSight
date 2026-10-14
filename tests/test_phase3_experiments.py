"""Deterministic unit tests for the isolated Phase 3 experiment helpers."""

import numpy as np
import pandas as pd
import pytest
import json

from experiments.phase3 import (
    ADDED_FEATURE_COLUMNS,
    add_experimental_features,
    baseline_price_predictions,
    convert_return_predictions_to_prices,
    make_experimental_targets,
    run_development_experiments,
    _make_phase4_reports,
)
import experiments.phase3 as phase3
from src.features import create_features


def synthetic_prices(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2022-01-03", periods=rows, freq="B", name="Date")
    step = np.arange(rows, dtype=float)
    close = 100.0 + 0.15 * step + np.sin(step / 4.0)
    volume = 1_000_000.0 + step * 275 + 10_000 * np.sin(step / 8.0)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.9,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def test_added_features_are_causal_and_calculated_as_defined():
    prices = synthetic_prices()
    features = create_features(prices)
    expanded = add_experimental_features(features)
    assert set(ADDED_FEATURE_COLUMNS).issubset(expanded.columns)

    date = expanded.index[25]
    source_position = prices.index.get_loc(date)
    expected_return = prices["Close"].pct_change(fill_method=None)
    assert expanded.loc[date, "Return_Lag_1"] == pytest.approx(expected_return.iloc[source_position - 1])
    assert expanded.loc[date, "Return_Lag_5"] == pytest.approx(expected_return.iloc[source_position - 5])
    assert expanded.loc[date, "Momentum_5"] == pytest.approx(
        prices["Close"].iloc[source_position] / prices["Close"].iloc[source_position - 5] - 1
    )
    assert expanded.loc[date, "Momentum_20"] == pytest.approx(
        prices["Close"].iloc[source_position] / prices["Close"].iloc[source_position - 20] - 1
    )
    assert expanded.loc[date, "Volume_Change_1"] == pytest.approx(
        prices["Volume"].iloc[source_position] / prices["Volume"].iloc[source_position - 1] - 1
    )

    revised = prices.copy()
    revised.loc[revised.index[source_position + 1]:, ["Close", "Volume"]] *= 4
    revised_expanded = add_experimental_features(create_features(revised))
    pd.testing.assert_series_equal(
        expanded.loc[:date, "Return_Lag_1"],
        revised_expanded.loc[:date, "Return_Lag_1"],
    )
    pd.testing.assert_frame_equal(expanded.loc[:date], revised_expanded.loc[:date])


def test_price_and_return_targets_share_origin_dates_and_drop_final_unlabeled_row():
    features = add_experimental_features(create_features(synthetic_prices()))
    X, price_target, return_target = make_experimental_targets(features)

    assert set(ADDED_FEATURE_COLUMNS).issubset(X.columns)
    assert X.index.equals(price_target.index)
    assert X.index.equals(return_target.index)
    assert X.index.equals(features.index[:-1])
    assert features.index[-1] not in X.index
    np.testing.assert_allclose(price_target.to_numpy(), features["Close"].iloc[1:].to_numpy())
    np.testing.assert_allclose(
        return_target.to_numpy(),
        price_target.to_numpy() / features.loc[X.index, "Close"].to_numpy() - 1,
    )


def test_persistence_drift_and_moving_average_baselines_use_only_origin_history():
    index = pd.date_range("2024-01-01", periods=4, freq="B", name="Date")
    close = pd.Series([100.0, 102.0, 103.0, 1000.0], index=index, name="Close")
    origins = index[[2, 3]]

    persistence = baseline_price_predictions("previous_close", close, origins)
    drift = baseline_price_predictions("drift", close, origins)
    moving_average = baseline_price_predictions("moving_average", close, origins)
    assert persistence.tolist() == [103.0, 1000.0]
    assert drift.tolist() == pytest.approx([104.5, 1300.0])
    assert moving_average.tolist() == pytest.approx([305 / 3, 1305 / 4])
    assert persistence.index.equals(origins)
    assert drift.index.equals(origins)
    assert moving_average.index.equals(origins)

    changed_future = close.copy()
    changed_future.iloc[-1] = -5000.0
    for method in ("previous_close", "drift", "moving_average"):
        original_at_third_origin = baseline_price_predictions(method, close, index[[2]]).iloc[0]
        revised_at_third_origin = baseline_price_predictions(method, changed_future, index[[2]]).iloc[0]
        assert revised_at_third_origin == pytest.approx(original_at_third_origin)


def test_baselines_reject_unknown_origins_and_invalid_method():
    index = pd.date_range("2024-01-01", periods=2, freq="B")
    close = pd.Series([10.0, 11.0], index=index)
    with pytest.raises(ValueError, match="forecast origin"):
        baseline_price_predictions("previous_close", close, pd.DatetimeIndex([index[-1] + pd.Timedelta(days=1)]))
    with pytest.raises(ValueError, match="method"):
        baseline_price_predictions("unknown", close, index[:1])


def test_return_forecasts_convert_to_price_on_matching_origin_dates():
    index = pd.date_range("2024-02-01", periods=2, freq="B")
    previous = pd.Series([100.0, 200.0], index=index)
    predicted_return = pd.Series([0.02, -0.01], index=index)
    converted = convert_return_predictions_to_prices(previous, predicted_return)
    assert converted.tolist() == pytest.approx([102.0, 198.0])
    assert converted.index.equals(index)

    with pytest.raises(ValueError, match="identical indexes"):
        convert_return_predictions_to_prices(previous, predicted_return.set_axis(index[::-1]))


def test_offline_runner_writes_metadata_and_separate_development_reports(tmp_path, monkeypatch):
    prices = synthetic_prices(600)
    data_path = tmp_path / "prices.csv"
    metadata_path = tmp_path / "metadata.json"
    prices.to_csv(data_path, index_label="Date")
    metadata_path.write_text(json.dumps({"ticker": "SYNTH", "requested_period": "synthetic"}), encoding="utf-8")
    monkeypatch.setattr(phase3, "DATA_PATH", data_path)
    monkeypatch.setattr(phase3, "METADATA_PATH", metadata_path)

    class MeanEstimator:
        def fit(self, X, y):
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    for name in ("build_linear_regression", "build_random_forest", "build_mlp_scaled_target"):
        monkeypatch.setattr(phase3, name, MeanEstimator)

    output = tmp_path / "offline-report"
    validation, folds, summary = run_development_experiments(output, phase4_reports=True)
    assert "Naive Baseline" not in set(validation["Candidate"])
    assert len(folds["Split"].unique()) == 5
    assert len(summary) == len(folds["Candidate"].unique())
    required_files = {
        "phase4_validation_comparison.csv", "phase4_fold_results.csv",
        "phase4_model_comparison.csv", "phase4_model_comparison.json",
        "phase4_fold_results.json", "phase4_development_predictions.csv",
        "phase4_metadata.json", "phase4_actual_vs_predicted.png",
        "phase4_prediction_errors.png", "phase4_fold_rmse.png",
    }
    assert required_files.issubset({path.name for path in output.iterdir()})
    metadata = json.loads((output / "phase4_metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_mode"] == "offline_saved_snapshot"
    assert len(metadata["clean_dataset_sha256"]) == 64
    assert metadata["split_boundaries"]["final_test"]["status"] == "reserved_unscored"
    assert metadata["walk_forward_splits"] == 5
    report = json.loads((output / "phase4_model_comparison.json").read_text(encoding="utf-8"))
    assert "definitions" in report["metrics"]
    predictions = pd.read_csv(output / "phase4_development_predictions.csv")
    assert set(predictions["Split"]).issubset({"Validation", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"})
    assert (predictions["Target_Date"] > predictions["Origin_Date"]).all()
    exported_folds = pd.read_csv(output / "phase4_fold_results.csv")
    assert {"MAE_USD", "RMSE_USD", "Target_Units"}.issubset(exported_folds.columns)


def test_phase4_report_refuses_final_test_predictions(tmp_path):
    validation = pd.DataFrame({"Candidate": ["Previous Close"], "RMSE": [1.0]})
    folds = pd.DataFrame({
        "Candidate": ["Previous Close"], "Split": [1], "MAE": [1.0], "RMSE": [1.0],
        "R2": [0.0], "Directional_Accuracy": [0.0],
    })
    predictions = pd.DataFrame({"Split": ["Final Test"]})
    with pytest.raises(ValueError, match="development predictions only"):
        _make_phase4_reports(validation, folds, predictions, {}, tmp_path)
