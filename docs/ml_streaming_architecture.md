# ML Streaming Feature Engineering and Serving

## Notebook analysis

`bd-ml-filtering.ipynb` contains three logical parts:

1. Price model A/C
   - Model A predicts forward 14-session maximum upside.
   - Model C predicts forward 14-session drawdown risk.
   - Both models use the same price feature set from OHLCV plus SPY market context.
2. Backtest/export logic
   - Ranks candidates by `Pred_A * (1 - Risk_Prob)`.
   - Exports daily top-N candidates and realized outcomes.
3. News momentum model S
   - Builds daily news dynamics: `S_it`, `N_it`, `delta_t`, `delta_S`, `V`, `A`, `Z`, `M`.
   - Applies a cross-sectional scaling factor over the A/C score.

## Batch input schema

Price input:

| Column | Type | Description |
| --- | --- | --- |
| `Datetime` | timestamp/date | Trading timestamp |
| `Symbol` | string | Ticker |
| `Open` | double | Open price |
| `High` | double | High price |
| `Low` | double | Low price |
| `Close` | double | Close price |
| `Volume` | long/double | Trading volume |

News input for Model S:

| Column | Type | Description |
| --- | --- | --- |
| `Datetime` | timestamp | News timestamp |
| `Symbol` | string | Ticker |
| `headline` | string | News headline |
| `polarity` | double | Sentiment polarity when available |

The current Silver news table does not yet contain `polarity`; when absent, the reusable code defaults it to `0`.

## Preprocessing

Price preprocessing:

- Sort by `Symbol`, `Datetime`.
- Drop duplicate symbol/timestamp rows, keeping the latest record.
- Cast OHLCV to numeric.
- Align each symbol against SPY close by timestamp and forward-fill SPY close for market-context features.
- Replace `inf` and `-inf` with null.
- Match notebook behavior by dropping rows with incomplete feature vectors before writing to the ML namespace.

Categorical handling:

- `Symbol` is retained as an identifier and partition key, not encoded as a model feature.
- No categorical model inputs are used in the notebook price feature vector.

Normalization:

- No fitted scaler is used in the notebook.
- Normalization is formula-based: percent returns, price ratios, z-scores, relative strength, rolling volatility ratios, and percentile placeholder.

## Feature selection

The streaming feature vector is ordered by `PRICE_FEATURE_COLUMNS` in [spark_jobs/ml_features.py](../spark_jobs/ml_features.py):

- Return dynamics: `r1`, `r3`, `r5`, `r10`, `r14`, `r20`, `r21`, `cumret_20`, `return_z_20`, `gap_return`.
- Candle/location: `CLV`.
- Trend/momentum: `EMA20_50_spread`, `EMA50_200_spread`, `EMA20_slope`, `EMA50_slope`, `MACD_hist`, `RSI14`, `ROC10`, `ROC20`, `ADX14`.
- Breakout positioning: `dist_20d_high`, `dist_55d_high`, `dist_52w_high`, `BB_pctB`, `BB_width`, `dist_ema20`, `dist_ema50`.
- Volatility/risk: `ATR14_ratio`, `vol20`, `downside_vol20`, `maxdd20`, `realized_vol_10`, `realized_vol_20`, `vol_ratio_20_60`, `true_range_zscore`.
- Volume flow: `RVOL20`, `OBV_slope`, `MFI14`, `dollar_volume_log`, `volume_slope_20`, `OBV_slope_neg`.
- Market context: `RS_vs_SPY_14`, `RS_vs_SPY_30`, `beta_60D`, `SPY_above_EMA50`, `SPY_20d_return`.
- Placeholder: `sector_percentile_20d`.

`BB_%B` from the notebook is stored as `BB_pctB` to avoid special characters in table columns. The model training/serving contract should use the same sanitized name.

## Streaming implementation

Implemented files:

- [spark_jobs/ml_features.py](../spark_jobs/ml_features.py): reusable feature engineering module for batch and streaming.
- [spark_jobs/silver_to_ml_features.py](../spark_jobs/silver_to_ml_features.py): Structured Streaming job from Silver to ML namespace.

Streaming flow:

1. Read stream from `nessie.silver.stock_market`.
2. For each micro-batch, collect changed symbols.
3. Read historical Silver rows for changed symbols plus `SPY`.
4. Compute the same notebook-derived feature logic with `compute_price_features`.
5. Keep only micro-batch keys, so historical rows are used for state but not re-emitted.
6. MERGE into `nessie.ml.stock_price_features` by `Datetime`, `Symbol`, and `feature_version`.

Output table:

`nessie.ml.stock_price_features`

| Column | Type |
| --- | --- |
| `Datetime` | timestamp |
| `Symbol` | string |
| feature columns | double |
| `feature_vector` | array<double> |
| `feature_version` | string |
| `source_batch_id` | long |
| `process_date` | timestamp |

The job also creates `nessie.ml.stock_predictions` as the serving contract for downstream inference.

## Serving architecture for AI Agent

Recommended near-real-time architecture:

1. Data processing
   - Kafka -> Bronze Iceberg: `spark_jobs/kafka_to_bronze.py`.
   - Bronze -> Silver Iceberg: `spark_jobs/bronze_to_silver.py`.
   - Silver -> ML features: `spark_jobs/silver_to_ml_features.py`.

2. Model inference service
   - Runs as a separate service or Spark streaming job.
   - Reads `nessie.ml.stock_price_features` incrementally.
   - Loads model artifacts for Model A, Model C, and optionally Model S.
   - Writes predictions to `nessie.ml.stock_predictions`.
   - Optional Kafka topic: `stock_predictions` for low-latency fanout.

3. Prediction serving layer for AI Agent
   - API service exposes latest predictions by symbol/date/rank.
   - Primary read path can be Kafka for push/streaming or Iceberg for queryable latest state.
   - Agent calls endpoints such as:
     - `GET /predictions/latest?top_n=20`
     - `GET /predictions/{symbol}`
     - `GET /predictions/stream` using SSE/WebSocket for live updates.

Recommended separation:

- Feature engineering service owns deterministic feature generation and ML table writes.
- Model inference service owns model artifact loading, prediction, score composition, and model versioning.
- Agent serving service owns low-latency retrieval, filtering, auth, and response formatting for the AI Agent.
