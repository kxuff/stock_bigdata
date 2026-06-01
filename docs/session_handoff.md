# Session Handoff — Autonomous ORCA Agent

## Repo

```text
/home/ming/Desktop/stock_bigdata
```

## Branch target

User requested commit + push to:

```text
autonomous-agent
```

## Main work completed

### AI Chat / ORCA UX

- Removed large hero/card UI from `streamlit_app/pages/2_AI_Chat.py`.
- Reworked sidebar from hard command input to default symbol context.
- Changed chat from fixed `single_symbol_advisory` to autonomous `POST /api/v1/agent/query`.
- Removed manual advisory job refresh UX.
- Added advisory job SSE endpoint:

```text
GET /api/v1/advisory/decision-jobs/{job_id}/events
```

- SSE emits `status`, `heartbeat`, `result`, `failure`, `error`.
- Added Streamlit SSE client in `streamlit_app/services/advisory_api.py`.
- Fixed progress display to avoid `None%`.
- Increased readiness/create-job timeouts for Spark/Iceberg cold start.

### Autonomous agent clean architecture

Added clean autonomous agent layer:

```text
AgentQueryRequest
  -> LLM route planner
  -> AgentQueryRouterService validation/fallback
  -> AutonomousAgentService
  -> route-specific service/provider
  -> AgentQueryResponse
```

Key endpoint:

```text
POST /api/v1/agent/query
```

Design principle:

```text
Autonomous UX, controlled typed execution.
```

## Files added

```text
orca-agent-advisory/app/schemas/agent.py
orca-agent-advisory/app/schemas/route_results.py
orca-agent-advisory/app/application/ports/query_router.py
orca-agent-advisory/app/application/ports/market_screen_provider.py
orca-agent-advisory/app/application/ports/streaming_observability_provider.py
orca-agent-advisory/app/application/ports/streaming_alert_provider.py
orca-agent-advisory/app/application/ports/streaming_quality_provider.py
orca-agent-advisory/app/application/services/agent_query_router_service.py
orca-agent-advisory/app/application/use_cases/autonomous_agent_service.py
orca-agent-advisory/app/application/use_cases/route_services.py
orca-agent-advisory/app/application/use_cases/streaming_route_services.py
orca-agent-advisory/app/infrastructure/llm/agent_route_planner.py
orca-agent-advisory/app/infrastructure/bigdata/market_screen_provider.py
orca-agent-advisory/app/infrastructure/bigdata/streaming_providers.py
```

## Files modified for autonomous work

```text
orca-agent-advisory/app/main.py
orca-agent-advisory/app/bootstrap/container.py
orca-agent-advisory/app/schemas/enums.py
streamlit_app/pages/2_AI_Chat.py
streamlit_app/services/advisory_api.py
docs/session_handoff.md
```

## Routes implemented

`AgentRoute` now includes:

```text
single_symbol_advisory
symbol_comparison
watchlist_review
universe_screen
market_brief
portfolio_rebalance
backtest_analysis
data_diagnostics
streaming_pipeline_health
streaming_freshness_check
streaming_alert_review
streaming_symbol_monitor
streaming_feature_drift
streaming_ingestion_lag
streaming_topic_inspection
streaming_quality_incidents
clarification
out_of_scope
```

### Route status

- `single_symbol_advisory`: uses existing `AdvisoryDecisionService`, `ToolResultProvider`, and CrewAI path.
- `symbol_comparison`: reads prediction table and ranks symbols by `final_score`.
- `watchlist_review`: loads symbols and returns review table.
- `universe_screen`: reads latest prediction leaders.
- `market_brief`: uses latest prediction leaders.
- `data_diagnostics`: provider/table diagnostics.
- `portfolio_rebalance`: deterministic planning only; uses holdings from `context.metadata.holdings` or `context.metadata.portfolio`, equal-weight capped at 40%, cash remainder, human review required, no trade execution.
- `backtest_analysis`: returns backtest planning/spec; no yfinance and no Streamlit backend imports in ORCA API.
- `clarification` / `out_of_scope`: immediate safe responses.

## Streaming pipeline support

Mapped streaming flow:

```text
Kafka topics
  -> nessie.bronze.stock_market / stock_market_indicator / stock_news
  -> nessie.silver.stock_market / stock_market_indicator / stock_news_v2
  -> nessie.ml.stock_price_features / stock_predictions
  -> nessie.alert.stock_market_alerts
```

Relevant jobs:

```text
spark_jobs/kafka_to_bronze.py
spark_jobs/bronze_to_silver.py
spark_jobs/silver_to_ml_features.py
spark_jobs/silver_to_alerts.py
```

Streaming provider reads, fail-soft and read-only:

```text
silver.stock_market
silver.stock_market_indicator
silver.stock_news_v2
ml.stock_price_features
ml.stock_predictions
alert.stock_market_alerts
```

Streaming routes implemented:

- `streaming_pipeline_health`
- `streaming_freshness_check`
- `streaming_alert_review`
- `streaming_symbol_monitor`
- `streaming_feature_drift`
- `streaming_ingestion_lag`
- `streaming_topic_inspection` (diagnostic/limitation; no direct Kafka consumer/admin yet)
- `streaming_quality_incidents`

Note: streaming `ml.stock_predictions` table exists, but no scoring writer was found in `spark_jobs`; it may be empty unless another process writes it.

## Validation performed

Backend compile:

```bash
cd /home/ming/Desktop/stock_bigdata/orca-agent-advisory
python -m py_compile app/main.py app/schemas/agent.py app/schemas/route_results.py app/application/ports/query_router.py app/application/ports/market_screen_provider.py app/application/ports/streaming_observability_provider.py app/application/ports/streaming_alert_provider.py app/application/ports/streaming_quality_provider.py app/application/services/agent_query_router_service.py app/infrastructure/llm/agent_route_planner.py app/infrastructure/bigdata/market_screen_provider.py app/infrastructure/bigdata/streaming_providers.py app/application/use_cases/route_services.py app/application/use_cases/streaming_route_services.py app/application/use_cases/autonomous_agent_service.py app/bootstrap/container.py
```

Streamlit compile:

```bash
cd /home/ming/Desktop/stock_bigdata
python -m py_compile streamlit_app/pages/2_AI_Chat.py streamlit_app/services/advisory_api.py
```

API tests:

```bash
cd /home/ming/Desktop/stock_bigdata/orca-agent-advisory
uv run --python 3.12 pytest tests/test_api_decision_endpoint.py
```

Result:

```text
4 passed, 1 warning
```

## Services restarted during session

```bash
docker compose up -d --build orca-api orca-worker
docker compose up -d --build orca-api
```

Health later passed:

```text
GET /healthz -> {"status":"ok"}
GET /api/v1/status -> ready
```

Restart Streamlit after pulling this work:

```bash
streamlit run streamlit_app/app.py
```

## Known limitations / next hardening

```text
- API tests specifically for /api/v1/agent/query route outcomes
- Generic agent jobs/SSE for long-running non-advisory routes
- Route-level audit persistence
- Iceberg-native backtest adapter
- Portfolio provider/table/account model
- Richer joins: predictions + features + alerts + sentiment + valuation
- Direct Kafka topic inspection via Kafka admin/consumer
- More polished route-specific Streamlit renderer components
```

## Commit safety notes

Do not commit unless explicitly intended:

```text
orca-agent-advisory/.env
airflow/logs/dag_processor_manager/dag_processor_manager.log
airflow/logs/scheduler/latest
.slim/
generated codemap files unless requested
```

User explicitly requested this session handoff, so `docs/session_handoff.md` should be committed.

## Resumable Sessions

Reuse only for clear continuation of same thread. Otherwise start fresh.

- fixer: `fix-2` Implement portfolio backtest routes; `fix-1` Implement agent router MVP
  - Context read by `fix-2`: `orca-agent-advisory/app/application/use_cases/autonomous_agent_service.py` (60 lines), `orca-agent-advisory/app/application/use_cases/route_services.py` (54 lines), `streamlit_app/pages/2_AI_Chat.py` (300 lines)
- explorer: `exp-3` Map streaming pipeline usecases; `exp-4` Backtest portfolio feasibility
  - Context read by `exp-3`: `spark_jobs/silver_to_alerts.py` (547 lines), `spark_jobs/codemap.md` (267 lines), `spark_jobs/ml_features.py` (90 lines), `spark_jobs/kafka_to_bronze.py` (298 lines), `spark_jobs/silver_to_ml_features.py` (190 lines), `docs/ml_streaming_architecture.md` (139 lines), `spark_jobs/bronze_to_silver.py` (190 lines), `codemap.md` (45 lines)
- oracle: `ora-1` Full autonomous roadmap

## Recommended next work

1. Add API tests for `/api/v1/agent/query`.
2. Add generic agent jobs/SSE for long-running routes.
3. Add route audit persistence.
4. Improve joins and route renderers.
5. Add Iceberg-native backtest adapter.
6. Add portfolio provider/table/account model.
7. Add direct Kafka inspection provider.
