# ORCA Production Next Steps

## Current state

- Streamlit `AI Chat` đã gọi ORCA API thật, không mock.
- ORCA API đã có live async-ish decision job endpoints:
  - `GET /healthz`
  - `GET /readyz`
  - `GET /api/v1/status`
  - `GET /api/v1/data/readiness`
  - `POST /api/v1/advisory/decision-jobs`
  - `GET /api/v1/advisory/decision-jobs/{job_id}`
  - `GET /api/v1/advisory/decision-jobs/{job_id}/result`
- Runtime tool provider đã chuyển sang BigData-only.
- Runtime sample provider đã bị xoá khỏi `orca-agent-advisory`.
- Live test pass cho:
  - `AAPL`
  - `MSFT`
  - `NVDA`
- BigData freshness đã được refresh cho `market_features`, `ml_predictions`, `risk_snapshot`.
- Sentiment/valuation vẫn thiếu cho `MSFT` và `NVDA` vì chưa có `FINBERT_API_URL`.

## Priority 0 — Fix UI job handling

### Problem

Backend đã async, nhưng Streamlit UI vẫn block bằng polling loop ngắn:

```python
for _ in range(24):
    status_payload = get_decision_job(job_id)
    ...
    sleep(1)
```

Live job thường mất 80-90 giây, nên UI có thể báo:

```text
ORCA job `<job_id>` still running. Try again soon.
```

### Work

- Bỏ `sleep()` polling loop khỏi Streamlit.
- Lưu pending jobs trong `st.session_state`:
  - `job_id`
  - `symbol`
  - `prompt`
  - `created_at`
  - `status`
- Render pending job card.
- Thêm button:
  - `Refresh status`
  - `Fetch result`
  - `Cancel` sau khi backend có cancel endpoint.
- Khi job `succeeded`, fetch result rồi append assistant message.
- Khi job `failed`, append structured error message.

### Acceptance criteria

- Submit prompt không block UI.
- Refresh browser không mất job đang chạy.
- User thấy `queued/running/succeeded/failed` rõ ràng.
- Full job 90s vẫn xem được result trong UI.

## Priority 1 — Add production job store

### Problem

Current job store trong ORCA API là in-memory dict:

```python
_jobs: dict[str, dict[str, Any]] = {}
```

Không production-safe:

- mất job khi container restart
- không multi-worker safe
- không retry
- không backpressure
- không job history thật
- memory grow nếu nhiều jobs

### Work

- Tạo `DecisionJobStore` interface.
- Implement Postgres-backed job table:
  - `job_id`
  - `request_id`
  - `status`
  - `progress_stage`
  - `progress_pct`
  - `request_payload`
  - `result_run_id`
  - `error_code`
  - `error_message`
  - `created_at`
  - `updated_at`
  - `started_at`
  - `completed_at`
- Tạo migration/init SQL.
- Replace in-memory `_jobs` with store.
- Add idempotency key support:
  - request header `Idempotency-Key`
  - unique constraint per user/tenant/key.

### Acceptance criteria

- Job survives API restart.
- Multiple API workers can read same job.
- `GET /decision-jobs/{job_id}` works after restart.

## Priority 2 — Move long work to real worker queue

### Problem

FastAPI `BackgroundTasks` vẫn chạy trong web worker process. Production should not run CrewAI long tasks inside API worker.

### Work

- Add queue system:
  - Redis + RQ, or
  - Celery, or
  - Arq.
- API `POST /decision-jobs` only writes job + enqueues work.
- Worker process executes:
  - BigData provider
  - CrewAI runner
  - decision assembly
  - output store save
  - job status update
- Add retry policy for transient LLM/provider errors.
- Add queue health to `/readyz`.

### Acceptance criteria

- API stays responsive while jobs run.
- Worker restart does not lose queued jobs.
- Failed transient tasks can retry.

## Priority 3 — Fix sentiment/valuation refresh

### Problem

`build_agent_context` currently depends on `FINBERT_API_URL` for sentiment. If missing, sentiment fails and valuation can be skipped too.

### Work

- Run FinBERT API service locally/prod.
- Set:

```text
FINBERT_API_URL=http://host.docker.internal:<port>
FINBERT_API_TIMEOUT=10
```

- Refactor `agent_context.py` so sentiment and valuation are independent:
  - sentiment missing FinBERT => skip sentiment only
  - valuation still builds
  - partial context saved
- Add context status output:
  - `sentiment_status`
  - `valuation_status`
  - `missing_context_reasons`

### Acceptance criteria

- `MSFT` and `NVDA` have `sentiment_snapshot` and `valuation_snapshot` when FinBERT available.
- If FinBERT down, valuation still updates.
- Pipeline does not fail whole refresh due optional context.

## Priority 4 — Add data coverage endpoint

### Problem

UI currently lets user type any symbol. If BigData lacks data, user sees late readiness failure.

### Work

Add endpoint:

```http
GET /api/v1/data/coverage
```

Response:

```json
{
  "as_of_timestamp": "...",
  "symbols": [
    {
      "symbol": "NVDA",
      "ready": true,
      "last_updated_at": "...",
      "tools": {
        "market_features": "SUCCESS",
        "ml_predictions": "SUCCESS",
        "risk_snapshot": "SUCCESS",
        "sentiment_snapshot": "MISSING",
        "valuation_snapshot": "MISSING"
      }
    }
  ]
}
```

Streamlit should use this endpoint to populate symbol selector.

### Acceptance criteria

- UI default symbols come from backend coverage.
- User cannot pick symbol with missing required data unless explicitly forced.

## Priority 5 — Add decision history and audit API

### Problem

ORCA saves decision JSON to output dir, but no production read API.

### Work

Add endpoints:

```http
GET /api/v1/advisory/decisions
GET /api/v1/advisory/decisions/{run_id}
GET /api/v1/advisory/decisions/{run_id}/audit
GET /api/v1/advisory/decisions/{run_id}/request
GET /api/v1/advisory/decisions/{run_id}/tool-results
```

Rules:

- `tool-results` requires privileged scope later.
- Redact raw sensitive data by default.
- Link `job_id -> run_id`.

### Acceptance criteria

- UI can show past decisions.
- User can open audit drawer for any result.
- Decision result remains accessible after job completes.

## Priority 6 — Auth and tenant boundaries

### Problem

Current local API has no auth. Production must protect financial/advisory data.

### Work

- Add auth middleware:
  - local API key for dev, or
  - JWT/OIDC for production.
- Add endpoint:

```http
GET /api/v1/me
```

- Add scopes:
  - `chat:read`
  - `chat:write`
  - `decision:read`
  - `decision:write`
  - `audit:read`
  - `audit:read:full`
  - `admin:read`
- Add identity fields to jobs/decisions:
  - `tenant_id`
  - `user_id`
  - `created_by`

### Acceptance criteria

- Unauthenticated requests fail.
- Users can only read their own jobs/decisions unless scoped otherwise.

## Priority 7 — Streaming progress

### Work

Add SSE endpoint:

```http
GET /api/v1/advisory/decision-jobs/{job_id}/events
```

Initial events:

```text
job.created
job.queued
job.running
tool_results.loaded
agent.market.completed
agent.sentiment.completed
agent.valuation.completed
agent.risk.completed
manager.completed
decision.completed
job.failed
```

### Acceptance criteria

- UI can show live progress without polling every second.
- Token streaming optional later.

## Priority 8 — Stock picks backend

### Problem

`AI_Stock_Picks` is still mock/static.

### Work

Add endpoints:

```http
GET /api/v1/advisory/picks
GET /api/v1/advisory/picks/{symbol}
POST /api/v1/advisory/picks:refresh
```

Rules:

- Picks come from precomputed ML/context tables.
- Do not run one CrewAI decision per row.
- User clicks one pick to generate full decision job.

### Acceptance criteria

- Stock picks page uses backend data.
- Pick list loads fast.
- Full advisory only runs on demand.

## Priority 9 — CI and tests

### Work

- Add tests for new endpoints:
  - health/status
  - data readiness
  - decision job create/status/result
  - failed job error mapping
  - BigData-only config rejects sample
- Add Streamlit client tests for payload building and error formatting.
- Add pipeline tests:
  - initial load date range
  - no FinBERT partial context behavior
  - CLI runner includes context step when configured
- Add GitHub Actions:
  - Python compile
  - unit tests
  - no secret grep
  - Docker build smoke

### Acceptance criteria

- PR cannot merge if sample runtime provider returns.
- PR cannot merge if `LLM_API_KEY` or secret-like values are committed.

## Priority 10 — Production deployment hardening

### Work

- Add Docker healthcheck for `orca-api`:

```yaml
healthcheck:
  test: ["CMD", "curl", "--fail", "http://localhost:8000/healthz"]
```

- Add structured logging.
- Add request IDs and job IDs to logs.
- Add metrics:
  - job duration
  - LLM latency
  - BigData load latency
  - failure count by error code
  - readiness failures by symbol/tool
- Add retention policy for decisions and raw tool results.
- Add rate limits.

## Suggested execution order

1. Fix Streamlit non-blocking job UI.
2. Add Postgres job store.
3. Add real worker queue.
4. Fix FinBERT/sentiment/valuation partial context.
5. Add data coverage endpoint and dynamic UI symbols.
6. Add decision history/audit endpoints.
7. Add auth and tenant boundaries.
8. Add SSE progress.
9. Move stock picks page to backend.
10. Add CI and deployment hardening.

## Short-term demo checklist

Before demo:

```powershell
cd E:\stock_bigdata
docker compose up -d orca-api airflow-webserver airflow-scheduler postgres spark-master spark-worker minio nessie
curl.exe http://127.0.0.1:8000/healthz
curl.exe http://127.0.0.1:8000/api/v1/status
curl.exe "http://127.0.0.1:8000/api/v1/data/readiness?symbols=AAPL&decision_mode=single_symbol_advisory"
curl.exe "http://127.0.0.1:8000/api/v1/data/readiness?symbols=MSFT&decision_mode=single_symbol_advisory"
curl.exe "http://127.0.0.1:8000/api/v1/data/readiness?symbols=NVDA&decision_mode=single_symbol_advisory"
streamlit run streamlit_app/app.py
```

Use symbols:

```text
AAPL, MSFT, NVDA
```

Avoid `LLY` until refreshed.
