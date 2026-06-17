# Stock Big Data

Dự án demo lakehouse cho chứng khoán Mỹ: ingest dữ liệu EOD từ yfinance, xử lý bằng Spark, lưu bảng Iceberg trên Nessie/MinIO, chạy ML inference, tạo sentiment/valuation context và phục vụ khuyến nghị đầu tư qua ORCA multi-agent advisory API. Repo cũng có Streamlit UI để xem dashboard, chat và AI stock picks.

> Đây là môi trường dev/local. Không commit API key, token LLM hoặc secret thật vào repo.

## Kiến trúc tổng quan

```text
yfinance / FinBERT
  -> Airflow EOD DAG
  -> Spark feature engineering + ML inference
  -> Iceberg tables trên Nessie catalog + MinIO object storage
  -> ORCA FastAPI + worker đọc dữ liệu bigdata
  -> Streamlit UI / advisory clients
```

Thành phần chính:

- `airflow/dags/us_stock_eod_inference.py`: DAG EOD sau giờ đóng cửa của thị trường Mỹ.
- `airflow/plugins/eod_inference/`: extract, clean, feature, inference, save và ORCA context.
- `spark_jobs/`: Spark jobs và feature contract dùng chung với notebook/model.
- `data/models/`: model artifact local, mặc định `model_a.joblib` và `model_c.joblib`.
- `orca-agent-advisory/`: FastAPI + CrewAI advisory layer.
- `streamlit_app/`: UI mock/local-first cho dashboard, chat và AI stock picks.
- `docs/`: runbook và tài liệu chi tiết hơn.

## Yêu cầu

- Docker Desktop hoặc Docker Compose.
- Python trên host để chạy test/UI local.
- `uv` nếu phát triển ORCA local.
- FinBERT HTTP API đang chạy và set `FINBERT_API_URL`.
- LLM gateway tương thích OpenAI API, mặc định `http://localhost:20128/v1` cho ORCA.

Sentiment phụ thuộc FinBERT. Nếu `FINBERT_API_URL` thiếu hoặc không truy cập được, EOD pipeline sẽ fail fast; dự án không còn lexical sentiment fallback.

## Chạy stack local

```bash
docker compose up -d
docker compose ps
```

Service/URL hay dùng:

| Thành phần      | URL                             |
| --------------- | ------------------------------- |
| Airflow         | <http://localhost:8085>         |
| Spark master UI | <http://localhost:8080>         |
| Spark worker UI | <http://localhost:8084>         |
| MinIO console   | <http://localhost:9001>         |
| Nessie API      | <http://localhost:19120/api/v2> |
| Kafka UI        | <http://localhost:8086>         |
| ORCA API        | <http://localhost:8000>         |

Tài khoản local mặc định:

| Dịch vụ | User      | Password   |
| ------- | --------- | ---------- |
| Airflow | `airflow` | `airflow`  |
| MinIO   | `admin`   | `password` |

## Cấu hình secret và endpoint

Với Docker Compose, ORCA đọc biến môi trường từ `orca-agent-advisory/.env` qua `env_file`. Đặt secret LLM trong file này hoặc trong môi trường runtime của container:

```env
NINEROUTER_KEY=sk-...
```

Nếu chạy ORCA trực tiếp trên host, set biến trong shell hiện tại.

PowerShell:

```powershell
$env:NINEROUTER_KEY="sk-..."
$env:FINBERT_API_URL="https://your-finbert-url"
$env:FINBERT_API_TIMEOUT="10"
```

Bash:

```bash
export NINEROUTER_KEY="sk-..."
export FINBERT_API_URL="https://your-finbert-url"
export FINBERT_API_TIMEOUT="10"
```

Kiểm tra FinBERT:

```bash
curl -H 'ngrok-skip-browser-warning: 1' "$FINBERT_API_URL/health"
```

Expected shape:

```json
{ "status": "ok", "model": "ProsusAI/finbert", "device": "cuda" }
```

## EOD pipeline

DAG `us_stock_eod_inference` gồm các stage:

```text
extract_eod_prices
-> clean_validate_prices
-> engineer_features
-> build_agent_context
-> run_ml_inference
-> save_predictions
```

Lần chạy đầu tiên cần backfill vì feature engineering cần đủ lookback history.

PowerShell:

```powershell
docker compose exec -T `
  -e PYTHONPATH='/opt/airflow/plugins' `
  -e US_STOCK_EOD_DATA_DIR='/tmp/eod_batch' `
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' `
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' `
  -e US_STOCK_SPARK_CORES_MAX='1' `
  -e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA' `
  -e US_STOCK_INITIAL_LOAD='true' `
  -e US_STOCK_BACKFILL_CALENDAR_DAYS='500' `
  -e FINBERT_API_URL='https://your-finbert-url' `
  -e FINBERT_API_TIMEOUT='10' `
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date 2026-05-29
```

Bash:

```bash
docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  -e US_STOCK_EOD_DATA_DIR='/tmp/eod_batch' \
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' \
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' \
  -e US_STOCK_SPARK_CORES_MAX='1' \
  -e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA' \
  -e US_STOCK_INITIAL_LOAD='true' \
  -e US_STOCK_BACKFILL_CALENDAR_DAYS='500' \
  -e FINBERT_API_URL='https://your-finbert-url' \
  -e FINBERT_API_TIMEOUT='10' \
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date 2026-05-29
```

Sau khi đã có history, bỏ 2 flag này để chạy incremental:

```bash
-e US_STOCK_INITIAL_LOAD='true'
-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'
```

Bảng Iceberg được ghi:

| Layer             | Table mặc định                            |
| ----------------- | ----------------------------------------- |
| Raw prices        | `nessie.raw.us_stock_eod_prices`          |
| Curated prices    | `nessie.curated.us_stock_eod_prices`      |
| ML features       | `nessie.ml_ready.stock_price_features`    |
| Predictions       | `nessie.ml_ready.stock_predictions_v2`    |
| Sentiment context | `nessie.ml_ready.stock_sentiment_context` |
| Valuation context | `nessie.ml_ready.stock_valuation_context` |

Manifest output thường có các trường:

```json
{
  "prediction_rows": 3,
  "orca_context_rows": 3,
  "sentiment_rows": 3,
  "valuation_rows": 3,
  "orca_context_includes": [
    "market_features",
    "ml_predictions",
    "risk_snapshot",
    "sentiment_snapshot",
    "valuation_snapshot"
  ]
}
```

## ORCA advisory API

ORCA được khai báo trong `docker-compose.yml` với 2 service:

- `orca-api`: FastAPI endpoint.
- `orca-worker`: background worker cho job queue.

Chạy/rerun ORCA:

```powershell
docker compose up -d --build orca-api orca-worker
```

Compose mặc định cấu hình ORCA đọc Iceberg bằng provider `bigdata`:

```env
ORCA_TOOL_RESULT_PROVIDER=bigdata
ORCA_ML_PREDICTION_TABLE=ml_ready.stock_predictions_v2
ORCA_ML_FEATURE_TABLE=ml_ready.stock_price_features
ORCA_CURATED_PRICE_TABLE=curated.us_stock_eod_prices
ORCA_SENTIMENT_TABLE=ml_ready.stock_sentiment_context
ORCA_VALUATION_TABLE=ml_ready.stock_valuation_context
```

Gọi advisory API:

```powershell
$body = @'
{
  "request_id": "req_live_20260529_001",
  "timestamp": "2026-05-29T23:30:00Z",
  "as_of_timestamp": "2026-05-29T23:30:00Z",
  "user_query": "Should I buy AAPL today?",
  "decision_mode": "single_symbol_advisory",
  "symbols": ["AAPL"],
  "user_context": {
    "risk_tolerance": "MODERATE",
    "investment_horizon": "SHORT_TERM",
    "target_sectors": ["Technology"],
    "excluded_symbols": [],
    "max_single_asset_weight": 40,
    "allow_cash_position": true,
    "custom_constraints": {"avoid_high_volatility": true}
  },
  "metadata": {
    "client": "web",
    "locale": "en-US",
    "as_of_date": "2026-05-29"
  }
}
'@

Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/api/v1/advisory/decision' `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body `
  -TimeoutSec 600
```

Response chính:

```text
symbol
recommendation
confidence
requires_human_review
decision_rationale
supporting_signals
conflicting_signals
risk_warnings
```

ORCA live advisory có thể chậm vì CrewAI chạy nhiều task LLM-backed.

## Streamlit UI

Chạy UI local:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Pages:

- Dashboard
- AI Chat
- AI Stock Picks

AI Stock Picks có thể đọc output local của EOD pipeline, hoặc đọc từ endpoint nếu set `ML_INFERENCE_PICKS_URL`.

## Biến môi trường quan trọng

| Biến                              | Mặc định                                  | Ghi chú                                                  |
| --------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| `US_STOCK_EOD_SYMBOLS`            | default symbol list trong code            | Giới hạn universe để demo nhanh, ví dụ `AAPL,MSFT,NVDA`. |
| `US_STOCK_EOD_DATA_DIR`           | `/opt/airflow/data/eod_batch`             | Thư mục staging trong container.                         |
| `US_STOCK_INITIAL_LOAD`           | `false`                                   | Set `true` cho first backfill.                           |
| `US_STOCK_BACKFILL_CALENDAR_DAYS` | `500`                                     | Số ngày calendar để backfill.                            |
| `US_STOCK_MIN_LOOKBACK_DAYS`      | `260`                                     | Lookback tối thiểu cho feature.                          |
| `US_STOCK_MODEL_A_PATH`           | `/opt/airflow/data/models/model_a.joblib` | Model return/pick.                                       |
| `US_STOCK_MODEL_C_PATH`           | `/opt/airflow/data/models/model_c.joblib` | Risk model optional.                                     |
| `FINBERT_API_URL`                 | set qua env/compose                       | Bắt buộc cho sentiment.                                  |
| `NINEROUTER_KEY`                  | none                                      | Secret cho LLM gateway.                                  |
| `LLM_BASE_URL`                    | `http://host.docker.internal:20128/v1`    | OpenAI-compatible gateway cho ORCA.                      |

## Test và validation

Compile EOD plugins:

```bash
python -m compileall airflow/plugins/eod_inference
```

EOD/agent context test:

```powershell
$env:PYTHONPATH='airflow/plugins'
python -m pytest tests/test_agent_context.py
```

ORCA tests:

```bash
cd orca-agent-advisory
uv run pytest
```

Một test riêng cho Big Data provider:

```bash
cd orca-agent-advisory
uv run --python 3.12 pytest tests/test_bigdata_ml_provider.py
```

## Troubleshooting nhanh

Nếu Spark bị treo hoặc không có resource:

```bash
docker compose restart spark-master spark-worker
```

Nếu gặp lỗi không đủ lookback history:

```text
Not enough lookback history for feature inference. Required 260
```

Chạy lại với:

```bash
-e US_STOCK_INITIAL_LOAD='true'
-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'
```

Nếu Airflow webserver unhealthy nhưng cần chạy command manual, thử restart:

```bash
docker compose restart airflow-webserver airflow-scheduler
```

Nếu ORCA không thấy prediction mới, kiểm tra table name đang dùng là:

```text
ml_ready.stock_predictions_v2
```

## Tài liệu thêm

- `docs/end_to_end_local_runbook.md`: runbook end-to-end local.
- `docs/us_stock_eod_batch_airflow.md`: chi tiết DAG, feature contract và Airflow variables.
- `docs/ml_streaming_architecture.md`: kiến trúc streaming/ML.
- `docs/ai_chat_production_checklist.md`: checklist cho AI chat.
- `orca-agent-advisory/docs/technical_spec.md`: technical spec của ORCA advisory layer.
