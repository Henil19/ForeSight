# ForeSight

ForeSight is an educational machine-learning project for forecasting the closing price of the next available AAPL observation. It builds causal technical features from Yahoo Finance data, compares regression models with a naive persistence baseline, evaluates models chronologically, and produces a next-observation forecast.

## Overview

The current application is configured for AAPL. Given historical prices and feature values available at time *t*, it predicts `Close` at the next available chronological feature row. That next observation is not guaranteed to be the next calendar day.

ForeSight is an educational project, **not financial advice**. Historical performance does not guarantee future results.

## Pipeline / Architecture

```text
Yahoo Finance data ingestion
    → chronological data cleaning
    → feature engineering
    → supervised target creation
    → chronological holdout evaluation
    → naive persistence baseline
    → five-fold walk-forward validation
    → model comparison
    → best trained-model selection
    → final refit on all labeled observations
    → next-observation forecast
```

## Data

The application downloads daily OHLCV data (Open, High, Low, Close, Volume) from Yahoo Finance using `yfinance`. `main.py` requests AAPL with the rolling `5y` period and `auto_adjust=False`. The period is relative to the run date; the repository does not contain a bundled fixed historical dataset.

Rows are sorted chronologically. Duplicate dates keep the last downloaded row, invalid dates are removed, and rows with missing or non-numeric required OHLCV values are dropped. A live run requires network access to Yahoo Finance. The `Close` and `Volume` fields are used to create features.

## Features

The pipeline creates exactly these nine features. Calculations use current and prior observations only.

| Feature | Calculation |
|---|---|
| `Close` | Closing price for the current observation. |
| `Volume` | Trading volume for the current observation. |
| `Daily_Return` | Percentage change in `Close` from the prior observation. |
| `MA20` | Simple moving average of `Close` over a full 20-observation window. |
| `MA50` | Simple moving average of `Close` over a full 50-observation window. |
| `MA200` | Simple moving average of `Close` over a full 200-observation window. |
| `RSI` | Relative Strength Index from 14-period Wilder-style exponentially smoothed gains and losses. |
| `MACD` | Difference between the 12- and 26-period exponential moving averages of `Close`. |
| `Volatility` | 20-observation rolling standard deviation of daily returns. |

Rows without sufficient history to calculate every feature are removed. The 200-observation moving average determines the longest warm-up requirement.

## Prediction Target

The supervised target is created with:

```python
target = Close.shift(-1)
```

Each feature row predicts the `Close` value of the next available chronological feature row. The final feature row has no following observed close in the dataset, so it has no known supervised target and is excluded from labeled model data. This is next-observation forecasting rather than a guaranteed next-calendar-day forecast.

## Models

### Linear Regression

A scikit-learn pipeline applies `StandardScaler` to the features before fitting `LinearRegression`.

### Random Forest

The production `RandomForestRegressor` uses 100 trees, `max_depth=8`, `min_samples_leaf=2`, and `random_state=42` (`n_jobs=1`).

### MLP

The production MLP scales the features with `StandardScaler` and scales the target with `StandardScaler` through `TransformedTargetRegressor`. The target scaler is fitted on the training targets for each model fit; predictions are inverse-transformed to the original price units.

The underlying `MLPRegressor` has one hidden layer with 32 ReLU units, uses the Adam solver, has `max_iter=300`, and sets `random_state=42`. The separate `build_mlp()` function remains available as a raw-target control implementation; production training uses `build_mlp_scaled_target()`.

## Naive Baseline

The naive baseline predicts the next `Close` as the previous actual `Close` for that target row. It is not trained; it serves as a persistence benchmark and is included in evaluation. It is excluded from final trained-model selection.

For directional accuracy, the baseline always predicts zero movement relative to the previous close. It therefore counts as directionally correct only when the actual close is unchanged. Its directional-accuracy score is near zero by construction and should be interpreted with that caveat.

## Evaluation

### Chronological Holdout

The labeled observations are divided into an 80/20 chronological split with no shuffling. The latest 20% is the holdout test set. Models are fitted only on the earlier training rows and compared on the same held-out rows.

### Metrics

| Metric | Meaning |
|---|---|
| MAE | Mean absolute difference between predicted and actual closes. |
| RMSE | Square root of the mean squared prediction error. |
| R² | Coefficient of determination for predictions versus actual values. |
| Directional Accuracy | Fraction of rows where the predicted movement direction matches the actual movement direction. |

Directional accuracy compares the signs of `(actual Close − previous Close)` and `(predicted Close − previous Close)`. A zero actual movement matches only a zero predicted movement.

## Walk-Forward Validation

The application runs five sequential chronological folds with expanding training windows. In `main.py`, the initial training size is explicitly set to:

```python
min_train_size = max(100, len(X) // 4)
```

Each test window occurs after its training window, and later training windows include additional earlier observations. The evaluation helper clones each supplied estimator and fits a fresh clone for every fold. Preprocessing and target transformations are fitted as part of that fold's model fit, using only the fold's training data. The naive baseline is evaluated separately on each fold using that fold's previous closes.

Walk-forward results are reported separately. The final model is selected by holdout RMSE, not by the walk-forward summary. Walk-forward evaluation provides multiple chronological historical comparisons but does not guarantee future performance.

## Model Selection and Final Forecast

The final trained model is selected by the lowest RMSE on the chronological holdout comparison. Only trained models are eligible; the naive baseline is excluded. The selected model is freshly instantiated and refitted on all available labeled observations before forecasting. The forecast uses the latest available feature row, whose next-observation target is unknown.

The forecast date is a Monday–Friday weekday approximation. Exchange holidays and market-specific trading calendars are not modeled.

## Current Results

The following values are from the results currently stored in `outputs/`. They describe one run and are not guarantees of future performance.

### Holdout Comparison

| Model | MAE | RMSE | R² | Directional Accuracy |
|---|---:|---:|---:|---:|
| Naive Baseline | 3.349005 | 4.703769 | 0.972312 | 0.000000 |
| Linear Regression | 3.360157 | 4.729820 | 0.972004 | 0.521327 |
| MLP | 6.909706 | 8.458911 | 0.910457 | 0.483412 |
| Random Forest | 21.571111 | 27.913769 | 0.024917 | 0.492891 |

### Walk-Forward Summary

| Model | Mean MAE | Mean RMSE | Mean R² | Mean Directional Accuracy | RMSE Std |
|---|---:|---:|---:|---:|---:|
| Naive Baseline | 2.652242 | 3.738021 | 0.957086 | 0.001258 | 1.126652 |
| Linear Regression | 2.818250 | 3.900445 | 0.952714 | 0.487437 | 1.032152 |
| MLP | 5.491379 | 6.805367 | 0.850015 | 0.486084 | 1.713111 |
| Random Forest | 13.585605 | 17.135303 | 0.289863 | 0.492341 | 11.365708 |

In this run, the naive baseline has the lowest mean walk-forward RMSE. Linear Regression is the best trained ML model by mean walk-forward RMSE. The target-scaled MLP improved substantially over the raw-target MLP in the controlled experiment, but remains behind the naive baseline and Linear Regression in these results. Random Forest performs worse than those two approaches in the reported comparisons. The results do not show that the ML models outperform the naive baseline overall.

The stored example forecast is for AAPL: Linear Regression was selected, with a predicted close of **$336.80** and an approximated forecast date of **2026-10-12**. This is a run-specific result, not a prediction guarantee.

## Generated Outputs

The application writes these files under `outputs/`:

| File | Description |
|---|---|
| `outputs/model_comparison.csv` | Holdout metrics for trained models and the naive baseline, including RMSE improvement versus the baseline. |
| `outputs/walk_forward_results.csv` | Metrics for each model and individual walk-forward fold. |
| `outputs/walk_forward_summary.csv` | Mean metrics and RMSE standard deviation across folds. |
| `outputs/forecast.txt` | Selected trained model, forecast date, predicted close, and educational disclaimer. |
| `outputs/historical_close.png` | Historical closing-price plot. |
| `outputs/technical_indicators.png` | Closing price with MA20, MA50, and MA200. |
| `outputs/model_predictions.png` | Actual and predicted close values for holdout target dates. |

## Project Structure

```text
ForeSight/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── outputs/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── forecasting.py
└── tests/
    ├── test_pipeline.py
    └── test_models.py
```

- `main.py` orchestrates data retrieval, model training and evaluation, walk-forward reporting, final refitting, forecasting, and output generation.
- `src/data.py` downloads and cleans historical OHLCV data.
- `src/features.py` creates causal technical features.
- `src/models.py` defines the models, constructs the next-row target, and performs chronological holdout splitting.
- `src/evaluation.py` calculates model and baseline metrics and performs walk-forward validation and summarization.
- `src/forecasting.py` selects the best comparison row, aligns previous closes, forecasts from the latest features, and approximates the next weekday.
- `tests/test_pipeline.py` and `tests/test_models.py` test data handling, features, modeling, evaluation, and forecasting with synthetic or mocked data.

## Installation

Create a virtual environment and activate it in Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the application:

```powershell
python main.py
```

The application requires network access to Yahoo Finance. Generated output files are written to `outputs/`.

## Testing

Run the test suite with:

```powershell
python -m pytest -q
```

## Limitations

- The application is currently configured for AAPL.
- Data is downloaded from a rolling five-year window, not from a bundled fixed dataset.
- The target is the next available observation, not a guaranteed next calendar day.
- The displayed forecast date uses a weekday approximation and does not account for exchange holidays.
- Historical performance does not guarantee future results.
- A previous-close persistence baseline can be difficult to beat for price-level forecasting.
- The baseline's directional accuracy is near zero by construction because the baseline predicts no movement relative to the previous close.
- Technical indicators do not eliminate market uncertainty.

## Disclaimer

ForeSight is for educational purposes only. It is **not financial advice** and is not a trading or investment system. Nothing in the project should be treated as a guarantee of future prices or performance.
