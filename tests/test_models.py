"""Deterministic tests for model preparation, training, and evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from main import _fit_final_forecast_model, _select_best_trained_model
from src.evaluation import (
    calculate_baseline_improvement,
    calculate_directional_accuracy,
    compare_models,
    evaluate_model,
    evaluate_naive_baseline,
    naive_previous_close_predictions,
    summarize_walk_forward_results,
    walk_forward_validate,
)
from src.models import (
    FEATURE_COLUMNS,
    build_linear_regression,
    build_mlp,
    build_mlp_scaled_target,
    build_random_forest,
    chronological_split,
    prepare_supervised_data,
    train_models,
)


def synthetic_features(rows: int = 80) -> pd.DataFrame:
    """Create locally generated, finite feature observations."""
    index = pd.date_range("2023-01-02", periods=rows, freq="B", name="Date")
    step = np.arange(rows, dtype=float)
    close = 100 + step * 0.2 + np.sin(step / 3)
    return pd.DataFrame({
        "Close": close,
        "Volume": 1_000_000 + step * 250,
        "Daily_Return": np.r_[0.0, np.diff(close) / close[:-1]],
        "MA20": close - 1.0,
        "MA50": close - 2.0,
        "MA200": close - 3.0,
        "RSI": 50 + 10 * np.sin(step / 7),
        "MACD": np.sin(step / 5),
        "Volatility": 0.01 + step * 0.00001,
    }, index=index)[FEATURE_COLUMNS]


def test_prepare_supervised_data_shifts_target_and_does_not_mutate_input():
    source = synthetic_features(12)
    before = source.copy(deep=True)
    X, y = prepare_supervised_data(source)

    assert len(X) == len(source) - 1
    assert y.iloc[0] == source["Close"].iloc[1]
    assert y.iloc[-1] == source["Close"].iloc[-1]
    assert X.index.equals(source.index[:-1])
    assert y.name == "Target"
    pd.testing.assert_frame_equal(source, before)


def test_prepare_supervised_data_validates_columns():
    with pytest.raises(ValueError, match="required columns"):
        prepare_supervised_data(pd.DataFrame({"Close": [1.0]}))


def test_chronological_split_preserves_order_and_rejects_invalid_sizes():
    X, y = prepare_supervised_data(synthetic_features())
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.25)
    assert X_train.index.is_monotonic_increasing
    assert X_test.index.is_monotonic_increasing
    assert X_train.index[-1] < X_test.index[0]
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)
    assert X_train.index[-1] < X_test.index[0]  # train is strictly before test

    for invalid in (0, 1, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="test_size"):
            chronological_split(X, y, invalid)


def test_all_models_build_train_and_predict():
    X, y = prepare_supervised_data(synthetic_features())
    X_train, X_test, y_train, _ = chronological_split(X, y)
    assert isinstance(build_linear_regression(), Pipeline)
    assert isinstance(build_random_forest(), RandomForestRegressor)
    assert isinstance(build_mlp(), Pipeline)

    models = train_models(X_train, y_train)
    assert set(models) == {"Linear Regression", "Random Forest", "MLP"}
    assert isinstance(models["MLP"], TransformedTargetRegressor)
    assert models["MLP"].transformer_.mean_[0] == pytest.approx(y_train.mean())
    assert models["MLP"].transformer_.mean_[0] != pytest.approx(y.mean())
    for model in models.values():
        assert len(model.predict(X_test)) == len(X_test)
    mlp = models["MLP"]
    mlp_predictions = mlp.predict(X_test)
    scaled_predictions = mlp.regressor_.predict(X_test)
    inverse_transformed = mlp.transformer_.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()
    np.testing.assert_allclose(mlp_predictions, inverse_transformed)


def test_scaled_target_mlp_fits_training_target_scaler_and_inverse_transforms():
    X, y = prepare_supervised_data(synthetic_features(70))
    X_train, y_train = X.iloc[:45], y.iloc[:45]
    control = build_mlp().named_steps["regressor"]
    assert control.get_params() == build_mlp().named_steps["regressor"].get_params()
    assert control.get_params()["hidden_layer_sizes"] == (32,)
    assert control.get_params()["activation"] == "relu"
    assert control.get_params()["solver"] == "adam"
    assert control.get_params()["max_iter"] == 300
    assert control.get_params()["early_stopping"] is False
    assert control.get_params()["random_state"] == 42

    experimental = build_mlp_scaled_target()
    assert isinstance(experimental, TransformedTargetRegressor)
    assert experimental.regressor.get_params()["regressor"].get_params() == control.get_params()
    experimental.fit(X_train, y_train)

    # The fitted target transformer sees only y_train, not later/external targets.
    assert experimental.transformer_.mean_[0] == pytest.approx(y_train.mean())
    assert experimental.transformer_.mean_[0] != pytest.approx(y.mean())

    predictions = experimental.predict(X.iloc[45:50])
    standardized_predictions = experimental.regressor_.predict(X.iloc[45:50])
    inverse_transformed = experimental.transformer_.inverse_transform(
        standardized_predictions.reshape(-1, 1)
    ).ravel()
    np.testing.assert_allclose(predictions, inverse_transformed)
    assert np.isfinite(predictions).all()


def test_directional_accuracy_hand_example_including_zero_movement():
    index = pd.RangeIndex(4)
    previous = pd.Series([10.0, 10.0, 10.0, 10.0], index=index)
    actual = pd.Series([11.0, 9.0, 10.0, 11.0], index=index)
    predicted = pd.Series([12.0, 11.0, 10.0, 9.0], index=index)
    # Correct: up row 1 and unchanged row 3; incorrect rows 2 and 4.
    assert calculate_directional_accuracy(previous, actual, predicted) == pytest.approx(0.5)


def test_evaluation_metrics_and_model_comparison():
    class FixedPredictor:
        def predict(self, X):
            return np.array([11.0, 9.0, 12.0])

    index = pd.RangeIndex(3)
    X_test = pd.DataFrame({"x": [1, 2, 3]}, index=index)
    y_test = pd.Series([10.0, 10.0, 10.0], index=index)
    previous = pd.Series([9.0, 11.0, 10.0], index=index)
    metrics = evaluate_model(FixedPredictor(), X_test, y_test, previous)
    assert set(metrics) == {"MAE", "RMSE", "R2", "Directional_Accuracy"}
    squared_errors = (np.array([11.0, 9.0, 12.0]) - y_test.to_numpy()) ** 2
    assert metrics["RMSE"] ** 2 == pytest.approx(squared_errors.mean())
    assert all(np.isfinite(value) for value in metrics.values())

    table = compare_models(
        {"fixed": FixedPredictor()}, X_test, y_test, previous,
        include_naive_baseline=False,
    )
    assert len(table) == 1
    assert list(table.columns) == ["Model", "MAE", "RMSE", "R2", "Directional_Accuracy"]
    assert table.iloc[0]["Model"] == "fixed"
    assert np.isfinite(table[["MAE", "RMSE", "R2", "Directional_Accuracy"]].to_numpy()).all()


def test_compare_models_sorts_by_rmse():
    class Predictor:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    index = pd.RangeIndex(4)
    X_test = pd.DataFrame({"x": range(4)}, index=index)
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    previous = pd.Series([1.0, 1.0, 1.0, 1.0], index=index)
    table = compare_models(
        {"worse": Predictor(20), "better": Predictor(2.5)}, X_test, y, previous,
        include_naive_baseline=False,
    )
    assert table["Model"].tolist() == ["better", "worse"]
    assert table["RMSE"].is_monotonic_increasing


def test_naive_previous_close_predictions_preserve_values_index_and_input():
    previous = pd.Series([10.0, 12.0], index=pd.Index(["a", "b"], name="day"))
    before = previous.copy(deep=True)
    predictions = naive_previous_close_predictions(previous)
    pd.testing.assert_series_equal(predictions, previous)
    assert predictions is not previous
    pd.testing.assert_series_equal(previous, before)
    with pytest.raises(TypeError, match="pandas Series"):
        naive_previous_close_predictions([10.0, 12.0])


def test_evaluate_naive_baseline_returns_expected_metrics_and_validates_inputs():
    index = pd.RangeIndex(3)
    actual = pd.Series([10.0, 10.0, 10.0], index=index)
    previous = pd.Series([9.0, 11.0, 10.0], index=index)
    metrics = evaluate_naive_baseline(actual, previous)
    assert metrics["MAE"] == pytest.approx(2 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["R2"] == pytest.approx(0.0)
    assert metrics["Directional_Accuracy"] == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="identical indexes"):
        evaluate_naive_baseline(actual, previous.set_axis([1, 2, 3]))
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_naive_baseline(pd.Series(dtype=float), pd.Series(dtype=float))


def test_compare_models_optionally_includes_naive_baseline():
    class Predictor:
        def predict(self, X):
            return np.full(len(X), 12.0)

    index = pd.RangeIndex(3)
    X_test = pd.DataFrame({"x": [1, 2, 3]}, index=index)
    actual = pd.Series([10.0, 11.0, 12.0], index=index)
    previous = pd.Series([9.0, 10.0, 11.0], index=index)
    models = {"fixed": Predictor(), "also fixed": Predictor()}
    expected_columns = ["Model", "MAE", "RMSE", "R2", "Directional_Accuracy"]

    with_baseline = compare_models(models, X_test, actual, previous)
    without_baseline = compare_models(
        models, X_test, actual, previous, include_naive_baseline=False,
    )
    assert "Naive Baseline" in with_baseline["Model"].tolist()
    assert set(with_baseline.loc[with_baseline["Model"] != "Naive Baseline", "Model"]) == set(models)
    assert set(without_baseline["Model"]) == set(models)
    assert list(with_baseline.columns) == expected_columns
    assert list(without_baseline.columns) == expected_columns
    assert with_baseline["RMSE"].is_monotonic_increasing


def test_calculate_baseline_improvement_percentages_and_validation():
    model = {"MAE": 4.0, "RMSE": 3.0, "Directional_Accuracy": 0.52}
    baseline = {"MAE": 5.0, "RMSE": 4.0, "Directional_Accuracy": 0.50}
    improvement = calculate_baseline_improvement(model, baseline)
    assert improvement["MAE_Improvement_Percent"] == pytest.approx(20.0)
    assert improvement["RMSE_Improvement_Percent"] == pytest.approx(25.0)
    assert improvement["Directional_Accuracy_Improvement_PercentagePoints"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="missing required metric keys"):
        calculate_baseline_improvement({"MAE": 1.0}, baseline)
    with pytest.raises(ValueError, match="missing required metric keys"):
        calculate_baseline_improvement(model, {"MAE": 1.0})
    for metric in ("MAE", "RMSE"):
        zero_baseline = {**baseline, metric: 0.0}
        with pytest.raises(ZeroDivisionError, match=f"Baseline {metric}"):
            calculate_baseline_improvement(model, zero_baseline)


def test_forecast_model_selection_excludes_naive_baseline():
    class TrainedModel:
        pass

    trained_models = {"Linear Regression": TrainedModel(), "MLP": TrainedModel()}
    comparison = pd.DataFrame(
        {
            "Model": ["Naive Baseline", "Linear Regression", "MLP"],
            "RMSE": [0.1, 2.0, 3.0],
        }
    )

    selected = _select_best_trained_model(comparison, trained_models)
    assert selected == "Linear Regression"
    assert selected in trained_models


def test_final_forecast_refit_is_fresh_and_uses_all_labeled_rows():
    instances = []

    class RecordingModel:
        def fit(self, X, y):
            self.fit_index = X.index.copy()
            self.fit_target = y.copy()
            return self

        def predict(self, X):
            return np.zeros(len(X))

    def build_recording_model():
        model = RecordingModel()
        instances.append(model)
        return model

    X, y = prepare_supervised_data(synthetic_features(20))
    holdout_model = build_recording_model()
    holdout_model.fit(X.iloc[:12], y.iloc[:12])

    final_model = _fit_final_forecast_model(
        "Recording", {"Recording": build_recording_model}, X, y,
    )

    assert final_model is not holdout_model
    assert len(instances) == 2
    assert final_model.fit_index.equals(X.index)
    pd.testing.assert_series_equal(final_model.fit_target, y)


def test_walk_forward_validation_uses_ordered_expanding_windows_and_baseline():
    class RecordingRegressor(RegressorMixin, BaseEstimator):
        fitted_indexes = []

        def fit(self, X, y):
            type(self).fitted_indexes.append(X.index.copy())
            self.value_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.value_)

    index = pd.RangeIndex(30)
    X = pd.DataFrame({"signal": np.arange(30, dtype=float)}, index=index)
    y = pd.Series(50 + np.arange(30, dtype=float) * 0.5, index=index)
    previous = y - 0.25
    RecordingRegressor.fitted_indexes = []

    results = walk_forward_validate(
        {"recording": RecordingRegressor()}, X, y, previous,
        n_splits=4, min_train_size=10,
    )

    assert list(results.columns) == [
        "Split", "Model", "MAE", "RMSE", "R2", "Directional_Accuracy",
    ]
    assert results["Split"].nunique() == 4
    assert len(results) == 4 * 2  # one fitted model and one baseline per split
    assert (results["Model"] == "Naive Baseline").sum() == 4
    for split_number, test_positions in enumerate(np.array_split(np.arange(10, 30), 4), start=1):
        train_end = int(test_positions[0])
        test_start = int(test_positions[0])
        test_end = int(test_positions[-1]) + 1
        train_index = RecordingRegressor.fitted_indexes[split_number - 1]
        assert train_index.equals(index[:train_end])
        assert train_index[-1] < index[test_start]
        assert not train_index.intersection(index[test_start:test_end]).size
        split_rows = results.loc[results["Split"] == split_number]
        assert split_rows["Model"].tolist() == ["recording", "Naive Baseline"]
        expected_baseline = evaluate_naive_baseline(
            y.iloc[test_start:test_end], previous.iloc[test_start:test_end],
        )
        baseline_row = split_rows.loc[split_rows["Model"] == "Naive Baseline"].iloc[0]
        for metric, value in expected_baseline.items():
            assert baseline_row[metric] == pytest.approx(value)

    assert len(RecordingRegressor.fitted_indexes[0]) < len(RecordingRegressor.fitted_indexes[1])
    assert len(RecordingRegressor.fitted_indexes[1]) < len(RecordingRegressor.fitted_indexes[2])
    assert len(RecordingRegressor.fitted_indexes[2]) < len(RecordingRegressor.fitted_indexes[3])


def test_walk_forward_summary_aggregates_metrics_and_sorts_by_rmse():
    results = pd.DataFrame({
        "Split": [1, 2, 1, 2],
        "Model": ["higher", "higher", "lower", "lower"],
        "MAE": [3.0, 5.0, 1.0, 3.0],
        "RMSE": [4.0, 6.0, 2.0, 4.0],
        "R2": [0.2, 0.4, 0.6, 0.8],
        "Directional_Accuracy": [0.4, 0.6, 0.8, 1.0],
    })
    summary = summarize_walk_forward_results(results)
    assert summary["Model"].tolist() == ["lower", "higher"]
    assert summary.columns.tolist() == [
        "Model", "Mean_MAE", "Mean_RMSE", "Mean_R2",
        "Mean_Directional_Accuracy", "RMSE_Std",
    ]
    lower = summary.iloc[0]
    assert lower["Mean_MAE"] == pytest.approx(2.0)
    assert lower["Mean_RMSE"] == pytest.approx(3.0)
    assert lower["Mean_R2"] == pytest.approx(0.7)
    assert lower["Mean_Directional_Accuracy"] == pytest.approx(0.9)
    assert lower["RMSE_Std"] == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_walk_forward_validation_rejects_invalid_split_configuration_and_alignment():
    index = pd.RangeIndex(16)
    X = pd.DataFrame({"x": np.arange(16, dtype=float)}, index=index)
    y = pd.Series(np.arange(16, dtype=float), index=index)
    previous = y - 1
    models = {"linear": build_linear_regression()}

    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="n_splits"):
            walk_forward_validate(models, X, y, previous, n_splits=invalid)
    with pytest.raises(ValueError, match="cannot produce"):
        walk_forward_validate(models, X, y, previous, n_splits=5, min_train_size=7)
    with pytest.raises(ValueError, match="at least 2"):
        walk_forward_validate(models, X, y, previous, min_train_size=1)
    with pytest.raises(ValueError, match="identical indexes"):
        walk_forward_validate(models, X, y.set_axis(pd.RangeIndex(1, 17)), previous)
    with pytest.raises(ValueError, match="non-empty"):
        walk_forward_validate(
            models, X.iloc[:0], y.iloc[:0], previous.iloc[:0], n_splits=2,
        )
    with pytest.raises(ValueError, match="chronologically ordered"):
        walk_forward_validate(models, X.iloc[::-1], y.iloc[::-1], previous.iloc[::-1])
