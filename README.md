# ForeSight

![CI](https://github.com/Henil19/ForeSight/actions/workflows/ci.yml/badge.svg)

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
    → chronological train / validation / final-test split
    → candidate training and validation comparison (including naive baseline)
    → five-fold expanding-window validation on development data
    → validation-based candidate selection
    → selected-candidate final-test evaluation
    → final refit on all labeled observations
    → next-observation forecast
```

## Data

The application downloads daily OHLCV data (Open, High, Low, Close, Volume) from Yahoo Finance using `yfinance`. `main.py` requests AAPL with the rolling `5y` period and `auto_adjust=False`. The period is relative to the run date. A separate saved AAPL snapshot is retained under `experiments/data/` in the current working tree for offline development experiments; `main.py` does not use that snapshot. The snapshot is not part of the proposed source-only release.

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

The naive baseline predicts the next `Close` as the previous actual `Close` for that target row. It is not trained; it serves as a persistence benchmark and is included in evaluation. It is not a trained model, but it remains eligible as a forecast candidate. The validation-RMSE rule selects it when it ties or outperforms the best trained model.

For directional accuracy, the baseline always predicts zero movement relative to the previous close. It scores correctly only when the actual close is unchanged, so its score equals the share of unchanged closes and is typically near zero for daily prices. Interpret that score as a consequence of the metric and baseline definition, not as a price-level error measure.

## Evaluation

### Chronological Train, Validation, and Final Test

The labeled observations are divided chronologically into 60% training, 20% validation, and 20% final test data by default. There is no shuffling. Candidate regressors are fitted on the training portion only and compared with the naive baseline on validation data. The validation set determines which candidate is selected; the final test set is reserved for one evaluation of that selected approach. Before final-test evaluation, a selected trained model is freshly refitted using the combined training and validation data. The final test does not determine model selection.

The comparison written to `outputs/model_comparison.csv` reports validation metrics for all trained candidates and the naive baseline. `outputs/final_test_comparison.csv` reports final-test metrics for the selected candidate and the naive baseline.

### Metrics

| Metric | Meaning |
|---|---|
| MAE | Mean absolute difference between predicted and actual closes. |
| RMSE | Square root of the mean squared prediction error. |
| R² | Coefficient of determination for predictions versus actual values. |
| Directional Accuracy | Fraction of rows where the predicted movement direction matches the actual movement direction. |

Directional accuracy compares the signs of `(actual Close − previous Close)` and `(predicted Close − previous Close)`. A zero actual movement matches only a zero predicted movement.

R² can be negative when predictions are worse than predicting the evaluation-set mean. Price-level R² can also look high when prices trend over time, so it should be read alongside RMSE and the previous-close baseline; it is not evidence by itself that a model beats that baseline.

## Walk-Forward Validation

The application runs five sequential chronological folds with expanding training windows on the pre-test development data (training plus validation). In `main.py`, the initial training size is explicitly set to:

```python
min_train_size = max(100, len(X_development) // 4)
```

Each test window occurs after its training window, and later training windows include additional earlier observations. The evaluation helper clones each supplied estimator and fits a fresh clone for every fold. Feature preprocessing and the MLP target transformation are fitted as part of that fold's model fit, using only that fold's training data. The naive baseline is evaluated separately on each fold using that fold's previous closes. The final test observations are not included in walk-forward validation.

Walk-forward results are reported separately and are not used to select the forecasting candidate. This evaluation provides multiple chronological comparisons within development data but does not guarantee future performance.

## Phase 3 Development Experiments

These experiments were run on a saved AAPL Yahoo Finance snapshot, separate from the production outputs. The snapshot, raw download, and provenance are local artifacts under `experiments/data/` and are not included in the public source release. The cleaned data uses the project's existing cleaning: flatten yfinance OHLCV columns when needed, normalize OHLCV names, parse dates, sort, keep the last duplicate date, coerce OHLCV fields to numeric, and drop invalid or incomplete rows.

The added feature candidate uses only information through its forecast origin: `Return_Lag_1` and `Return_Lag_5` are daily returns shifted by one and five observations; `Momentum_5` and `Momentum_20` are `Close[t] / Close[t-k] - 1`; and `Volume_Change_1` is the one-observation percent change in volume. The existing `Volatility` feature is already a 20-observation rolling standard deviation of returns, so no duplicate rolling-volatility feature was added. The five new features were tested together as one small feature set, not tuned individually. All candidates were restricted to their common dates and target rows after the additional 20-observation warm-up.

Three price baselines were compared:

- **Previous Close:** predict `Close[t]` for the next observation.
- **Expanding Drift:** predict `Close[t] + (Close[t] - Close[first]) / (n - 1)`, where the first and current closes and all `n` observed closes are available by origin `t`. The estimate updates using history available through each forecast origin.
- **5-day Moving Average:** predict the mean of the last five closes through and including `Close[t]`.

The return-target candidates predict `(Close[t+1] / Close[t]) - 1`. Their price forecasts are converted back with `Close[t] * (1 + predicted_return)`. Zero predicted return is exactly the previous-close baseline. Return-target metrics use return units, while price and converted-return candidates are evaluated on the same price target and dates.

The development experiments compared the existing models with previous-close persistence, an expanding-drift baseline, and a trailing moving-average baseline. They also examined a small causal feature set and price-versus-return targets using common evaluation dates. Machine-learning candidates did not consistently outperform the simple baselines across development validation and folds. These results are qualitative here; the local CSV/JSON reports are not included in the public source release. The reserved forecasting final test was not used for these development decisions and remains separate.

The local results also include a preliminary run with a target-construction error in its `causal_plus` candidates. That run is invalid and is not used for the findings above. Snapshot files and all generated experiment reports—including preliminary and corrected results—remain local and are not part of the public source release. The downloader refuses to overwrite an existing snapshot, and the runner refuses to overwrite existing result files.

## Offline Reproducibility and Phase 4 Reports

The development experiment runner uses only the saved cleaned snapshot at `experiments/data/aapl_5y_clean.csv`; it never downloads market data. A source-only checkout does not include a snapshot, so offline execution requires a compatible saved snapshot and its metadata at `experiments/data/`. To create a local snapshot, run the downloader once (this step requires network access):

```powershell
python experiments/download_phase3_data.py
```

The downloader refuses to overwrite existing snapshot files. A fresh download is a different snapshot and will not reproduce the saved-snapshot results exactly. With a compatible snapshot in place, run the experiment explicitly in offline mode:

```powershell
python -m experiments.phase3 --offline
```

The default report directory is `experiments/results/phase4_offline/`. Reports and plots are generated locally and are not included in the public source release. A run creates validation and fold CSVs, comparison and fold JSON files, development-only prediction rows, metadata, and plots for actual-versus-predicted values, prediction errors, and fold RMSE. The runner refuses to proceed if its report files already exist, so choose a new empty destination with `--output-dir` before rerunning. It checks the saved data against the SHA-256 in its metadata and stops on a mismatch or malformed data. The Phase 3 corrected results remain separate and are not overwritten.

Phase 4 records development validation and fold metrics, model configuration, random seeds, software versions, and snapshot provenance. Reports retain any MLP convergence-warning flags; convergence is not guaranteed. The offline runner reserves the forecasting final test and does not score or display its results.

The qualitative findings above reflect the verified saved-snapshot development experiments. An offline run generates new local reports; it does not download market data or evaluate the reserved forecasting final test. Exact reproduction of the saved-snapshot analysis requires the same dataset and configuration recorded in the locally generated metadata.

The repository records Yahoo Finance via `yfinance` as the snapshot source, but contains no separate data license or redistribution permission note. Verify the applicable terms before publishing downloaded CSVs or data-derived charts. A source-only release can omit `experiments/data/` and the generated experiment reports; users would then need a compatible local snapshot before using offline experiments. Downloading a fresh snapshot requires network access and will not reproduce the saved-snapshot results exactly.

## Model Selection and Final Forecast

The forecasting candidate is selected by the lowest validation RMSE across Linear Regression, Random Forest, MLP, and the naive baseline. If no trained model has a lower validation RMSE than the baseline, the baseline is selected (ties favor the baseline). The selected candidate's final-test metrics are reported separately and do not change the selection. After final-test evaluation, a selected trained model is freshly instantiated and refitted on all available labeled observations before forecasting. If the baseline is selected, the forecast is the latest observed `Close`. The forecast uses the latest available feature row, whose next-observation target is unknown.

The forecast date is a Monday–Friday weekday approximation. Exchange holidays and market-specific trading calendars are not modeled.

## Development Findings

The saved-snapshot development experiments tested price and return targets, a limited set of causal features, and simple price baselines alongside the machine-learning candidates. The simple baselines remained competitive; machine-learning candidates did not consistently outperform them across validation and expanding-window folds. These are development findings, not a claim of future performance. The reserved forecasting final test remains separate and is not reported here. MLP training can emit convergence warnings, which are retained rather than hidden.

## Generated Outputs

The application generates these files under `outputs/` when run. `.gitignore` excludes the contents of this directory, so they are not included in a source-only checkout and users must generate them locally. `outputs/final_test_comparison.csv` contains the selected candidate's final-test evaluation when the production application runs; this README does not publish those metrics.

| File | Description |
|---|---|
| `outputs/model_comparison.csv` | Validation metrics for all trained candidates and the naive baseline, including RMSE improvement versus the baseline and the selected candidate. |
| `outputs/final_test_comparison.csv` | Final-test metrics for the selected candidate and naive baseline; not used for selection. |
| `outputs/walk_forward_results.csv` | Metrics for each model and individual walk-forward fold. |
| `outputs/walk_forward_summary.csv` | Mean metrics and RMSE standard deviation across folds. |
| `outputs/forecast.txt` | Selected forecast candidate and method, forecast date, predicted close, and educational disclaimer. |
| `outputs/historical_close.png` | Historical closing-price plot. |
| `outputs/technical_indicators.png` | Closing price with MA20, MA50, and MA200. |
| `outputs/model_predictions.png` | Actual final-test closes, selected candidate predictions when trained, and naive baseline predictions. |

## Project Structure

```text
ForeSight/
├── .github/
│   └── workflows/
│       └── ci.yml
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── experiments/                    # source files proposed; data/reports are local artifacts
│   ├── download_phase3_data.py
│   ├── phase3.py
│   ├── data/                       # required locally for offline runs; excluded from source-only release
│   └── results/                    # generated locally; excluded from source-only release
├── outputs/                        # generated by main.py; ignored by Git
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── forecasting.py
└── tests/
    ├── test_pipeline.py
    ├── test_models.py
    └── test_phase3_experiments.py
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
- The production application downloads a rolling five-year window; the separate saved research snapshot supports offline experiments.
- The target is the next available observation, not a guaranteed next calendar day.
- The displayed forecast date uses a weekday approximation and does not account for exchange holidays.
- Historical performance does not guarantee future results.
- A previous-close persistence baseline can be difficult to beat for price-level forecasting.
- The previous-close baseline's directional-accuracy score equals the fraction of unchanged closes; it is not a measure of its price-level accuracy.
- Technical indicators do not eliminate market uncertainty.

## Disclaimer

ForeSight is for educational purposes only. It is **not financial advice** and is not a trading or investment system. Nothing in the project should be treated as a guarantee of future prices or performance.
