# Roadmap triển khai Agent Layer theo Pull Request

**Dự án:** AI-Powered Investment Advisory System  
**Layer:** Multi-Agent Advisory Layer  
**Framework:** CrewAI  
**LLM Provider:** DeepSeek API  
**Kiến trúc mặc định:** CrewAI `Process.hierarchical` với custom `manager_agent` từ P0/MVP  

---

## 1. Mục tiêu tổng thể

Roadmap này chia việc triển khai Agent Layer thành các pull request nhỏ, có thể review và merge độc lập. Mục tiêu không phải làm một nhánh lớn trong 14 ngày rồi mới tích hợp, mà là đi theo các vertical slice có hợp đồng rõ ràng.

Luồng production/demo target ngay từ PR đầu có orchestration là hierarchical:

```text
Client / FastAPI
        │
        ▼
POST /api/v1/advisory/decision
        │
        ▼
Agent-Ready Context Builder
        │
        ▼
Deterministic Pre-Checks
  ├── Schema validation
  ├── Source Quality
  └── Required-context validation
        │
        ▼
CrewAI Hierarchical Crew
  ├── manager_agent = Investment Advisory Manager
  └── agents = [
        Data/Technical Agent,
        Sentiment Agent,
        Valuation Agent,
        Risk Agent
      ]
        │
        ▼
Deterministic Policy Services
  ├── Confidence Aggregation
  ├── Conflict Resolution
  ├── Financial Validator
  ├── Compliance Guardrails
  ├── Human Review Gate
  └── Audit Metadata
        │
        ▼
Final Advisory Decision JSON
```

P0 phải có hierarchical crew chạy được end-to-end. Sequential mode nếu có chỉ là debug helper cục bộ, không phải architecture target.

---

## 2. Contract thống nhất

### 2.1 Endpoint

Endpoint chính:

```http
POST /api/v1/advisory/decision
```

Không dùng endpoint legacy cũ trong tài liệu mới, trừ khi ghi rõ là alias tương thích ngược.

### 2.2 DeepSeek configuration

Default:

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
AGENT_TEMPERATURE=0.2
AGENT_MAX_RETRIES=3
AGENT_TIMEOUT_SECONDS=180
```

`DEEPSEEK_MODEL` phải override được bằng environment variable. `deepseek-chat` chỉ là legacy fallback nếu môi trường chưa hỗ trợ model mới.

### 2.3 CrewAI mode

MVP/P0 dùng:

```python
crew = Crew(
    agents=[
        data_agent,
        sentiment_agent,
        valuation_agent,
        risk_agent,
    ],
    tasks=[
        data_task,
        sentiment_task,
        valuation_task,
        risk_task,
        final_decision_task,
    ],
    manager_agent=manager_agent,
    process=Process.hierarchical,
    verbose=True,
)
```

Manager Agent là custom manager để vai trò tổng hợp, delegation và audit rõ ràng.

### 2.4 Decision enums

```text
Recommendation = BUY | HOLD | SELL | WATCH
PortfolioAction = INCREASE_WEIGHT | DECREASE_WEIGHT | MAINTAIN_WEIGHT | EXIT | CASH_BUFFER
```

Không dùng mức recommendation mạnh hơn `BUY` trong P0/P1 để tránh mâu thuẫn với output contract và giảm rủi ro compliance.

### 2.5 Confidence semantics

`confidence` trong final output là confidence cuối cùng sau khi đã áp dụng risk cap, source-quality cap và conflict downgrade.

Tối thiểu `confidence_breakdown` gồm:

```json
{
  "base_confidence": 0.74,
  "risk_adjusted_confidence": 0.68,
  "risk_cap": 0.75,
  "source_quality_cap": 0.90,
  "market_confidence": 0.72,
  "ml_confidence": 0.68,
  "sentiment_confidence": 0.61,
  "valuation_confidence": 0.58,
  "risk_adjustment": -0.08,
  "source_quality_adjustment": -0.03
}
```

---

## 3. PR Roadmap

### PR 1 — Docs + Contract Alignment

**Mục tiêu:** thống nhất đặc tả và roadmap trước khi code.

**Phạm vi:**

- Cập nhật architecture thành hierarchical-first.
- Chuẩn hóa endpoint, DeepSeek model default, enums, confidence semantics.
- Sửa sample portfolio allocation để tổng `weight_pct = 100`.
- Phân loại rõ P0/P1/P2 component.

**Deliverables:**

- `agent_layer_specification_crewai_deepseek_v1_2.md`
- `full_agent_layer_2_week_roadmap.md`

**Done criteria:**

- Không còn mâu thuẫn giữa roadmap và spec về endpoint, enum, confidence fields, model config, sample allocation.
- Spec ghi rõ P0 dùng `Process.hierarchical` với custom `manager_agent`.
- Roadmap được chia thành các PR có thể review độc lập.

**Test/Review scope:**

- Dùng `rg` kiểm tra các từ khóa lệch liên quan endpoint legacy, recommendation ngoài enum, sequential process code path và model legacy.
- `deepseek-chat` chỉ còn xuất hiện trong ngữ cảnh legacy fallback.

---

### PR 2 — Schema + Samples

**Mục tiêu:** đóng băng input/output contract bằng Pydantic models và sample JSON.

**Phạm vi:**

- Tạo Pydantic models cho `AgentReadyContext`, market/ML/sentiment/valuation/risk contexts, agent outputs, final decision và audit metadata.
- Tạo enums cho `Recommendation`, `PortfolioAction`, `RiskLabel`, `SentimentLabel`, `ValuationLabel`, `AgentStatus`, `DecisionMode`, `ConflictLevel`, `ReviewReason`.
- Tạo sample contexts cho normal, high-risk, missing sentiment, missing valuation, stale data, portfolio allocation.

**Deliverables:**

- `app/schemas/context.py`
- `app/schemas/agent_outputs.py`
- `app/schemas/decision.py`
- `app/schemas/audit.py`
- `samples/*.json`
- `tests/test_schemas.py`

**Done criteria:**

- Tất cả sample JSON parse được.
- Missing `market_context` bị reject rõ ràng.
- Missing optional context như sentiment/valuation không làm fail toàn request.
- Portfolio sample có tổng allocation đúng 100%.

**Test/Review scope:**

- Unit tests cho parse/validation.
- Snapshot nhỏ cho final output schema.

---

### PR 3 — DeepSeek + Hierarchical Crew Skeleton

**Mục tiêu:** dựng skeleton hierarchical crew chạy được trước khi có logic agent thật.

**Phạm vi:**

- Tạo config/env loading cho DeepSeek.
- Tạo LLM factory dùng default `deepseek-v4-flash`.
- Tạo JSON parsing/repair helper.
- Tạo CrewAI hierarchical runner tối thiểu với custom `manager_agent` và 4 specialist agents stubbed/mocked.
- Không triển khai business policy sâu trong PR này.

**Deliverables:**

- `app/config.py`
- `app/llm/deepseek_client.py`
- `app/llm/llm_factory.py`
- `app/validators/output_repair.py`
- `app/services/crew_runner.py`
- `tests/test_llm_json_output.py`
- `tests/test_hierarchical_crew_skeleton.py`

**Done criteria:**

- Hierarchical crew kickoff chạy được với mocked LLM.
- Manager Agent được truyền qua `manager_agent`, không nằm lẫn trong specialist `agents`.
- Output mock parse được thành JSON/Pydantic model.
- Live DeepSeek smoke test chỉ chạy khi có `DEEPSEEK_API_KEY`.

**Test/Review scope:**

- Mocked tests mặc định trong CI.
- Optional live test được skip nếu thiếu API key.

---

### PR 4 — Deterministic Policy Services

**Mục tiêu:** đưa các rule cứng ra khỏi LLM để hệ thống kiểm soát được behavior.

**Phạm vi:**

- Source Quality service.
- Confidence aggregation service.
- Conflict resolution service.
- Financial validator.
- Human review gate.
- Audit metadata/hash service.

**Deliverables:**

- `app/services/source_quality_service.py`
- `app/services/confidence_service.py`
- `app/services/conflict_resolution_service.py`
- `app/services/human_review_service.py`
- `app/services/audit_service.py`
- `app/validators/financial_validator.py`
- `tests/test_source_quality_service.py`
- `tests/test_confidence_service.py`
- `tests/test_conflict_resolution.py`
- `tests/test_human_review_gate.py`
- `tests/test_financial_validator.py`

**Done criteria:**

- Risk/source-quality/conflict caps deterministic.
- `HIGH` risk có thể cap confidence và trigger human review.
- Allocation âm, allocation != 100%, asset vượt `max_single_asset_weight` bị reject.
- Audit metadata có `run_id`, `request_id`, `input_context_hash`, `validator_version`, timestamp.

**Test/Review scope:**

- Pure unit tests, không gọi LLM/API.
- Review tập trung vào policy edge cases.

---

### PR 5 — Core Specialist Agents

**Mục tiêu:** triển khai 4 specialist agents dùng structured context và structured output.

**Phạm vi:**

- Data/Technical Agent.
- Sentiment Agent.
- Valuation Agent.
- Risk Agent.
- Prompt templates cho từng agent.
- Task definitions có expected output rõ và structured output/Pydantic validation khi phù hợp.

**Deliverables:**

- `app/agents/data_agent.py`
- `app/agents/sentiment_agent.py`
- `app/agents/valuation_agent.py`
- `app/agents/risk_agent.py`
- `app/tasks/data_tasks.py`
- `app/tasks/sentiment_tasks.py`
- `app/tasks/valuation_tasks.py`
- `app/tasks/risk_tasks.py`
- `app/prompts/data_agent.md`
- `app/prompts/sentiment_agent.md`
- `app/prompts/valuation_agent.md`
- `app/prompts/risk_agent.md`
- `tests/test_core_agents.py`

**Done criteria:**

- Mỗi agent validate output theo schema riêng.
- Agent không tự bịa metric thiếu; field thiếu đi vào `missing_fields` hoặc `limitations`.
- Sentiment/Valuation có thể `SKIPPED` nếu optional context thiếu.
- Risk Agent trả `risk_label`, `risk_factors`, `confidence_cap`.

**Test/Review scope:**

- Agent tests dùng mocked LLM response.
- Prompt review kiểm tra context-only reasoning và prompt-injection boundary.

---

### PR 6 — Manager Synthesis + API

**Mục tiêu:** nối hierarchical crew, deterministic services và FastAPI thành luồng end-to-end P0.

**Phạm vi:**

- Manager synthesis task.
- Crew runner full P0 flow.
- Final decision assembly.
- FastAPI endpoint `POST /api/v1/advisory/decision`.
- Error responses cho invalid request, missing required context, agent timeout, validation failed.

**Deliverables:**

- `app/agents/manager_agent.py`
- `app/tasks/manager_tasks.py`
- `app/services/crew_runner.py`
- `app/main.py`
- `tests/test_end_to_end.py`
- `tests/test_api_decision_endpoint.py`

**Done criteria:**

- API trả valid final JSON cho normal scenario.
- High-risk conflict scenario bị cap confidence và có review reason.
- Stale/missing optional context scenario vẫn trả response hợp lệ với limitation.
- Missing required market context trả error rõ.
- Final output có `not_financial_advice=true`, citations/limitations/audit metadata.

**Test/Review scope:**

- E2E tests với mocked LLM.
- API contract tests.
- Latency live/API không phải gate trong PR này.

---

### PR 7 — Compliance Guardrails

**Mục tiêu:** thêm compliance boundary có thể test deterministic trước, optional LLM reviewer sau.

**Phạm vi:**

- Deterministic compliance checks cho forbidden language, missing disclaimer, risk-profile mismatch.
- Optional Compliance Reviewer agent sau feature flag.
- Review/reject policy khi vi phạm compliance.

**Deliverables:**

- `app/services/compliance_service.py`
- `app/agents/compliance_reviewer.py`
- `app/tasks/compliance_tasks.py`
- `app/prompts/compliance_reviewer.md`
- `tests/test_compliance_service.py`
- `tests/test_compliance_reviewer.py`

**Done criteria:**

- Output chứa “guaranteed profit”, “risk-free”, “certain return” bị flag/reject.
- Missing disclaimer bị reject hoặc repair.
- Recommendation vượt user risk profile trigger human review.
- LLM Compliance Reviewer có thể tắt bằng feature flag.

**Test/Review scope:**

- Unit tests deterministic là bắt buộc.
- LLM reviewer tests dùng mocked output.

---

### PR 8 — Critic/Debate Extension

**Mục tiêu:** thêm debate/revision như research extension, không làm P0 phụ thuộc.

**Phạm vi:**

- Bullish Critic Agent.
- Bearish Critic Agent.
- Debate task outputs.
- Manager revision flow sau draft.
- Feature flag để bật/tắt critic stage.

**Deliverables:**

- `app/agents/bullish_critic_agent.py`
- `app/agents/bearish_critic_agent.py`
- `app/tasks/critic_tasks.py`
- `app/prompts/bullish_critic.md`
- `app/prompts/bearish_critic.md`
- `tests/test_critic_stage.py`
- `tests/test_manager_revision.py`

**Done criteria:**

- Critic stage chạy sau Manager draft.
- High-conflict hoặc confidence > 0.8 có thể trigger critic stage.
- Final output có `conflict_analysis`.
- Bearish critic strong + high risk có thể downgrade recommendation hoặc cap confidence.

**Test/Review scope:**

- Mocked debate outputs.
- Regression tests đảm bảo feature flag off không đổi P0 behavior.

---

### PR 9 — Evaluation + Demo Docs

**Mục tiêu:** có bằng chứng demo/evaluation đủ cho báo cáo và handoff.

**Phạm vi:**

- Backtesting/proxy evaluation harness.
- Report generation.
- Demo script.
- README và demo flow.
- Sample outputs cho ít nhất 3 scenario.

**Deliverables:**

- `scripts/backtest_agent_decisions.py`
- `scripts/run_demo.py`
- `reports/backtest_summary.json`
- `reports/sample_outputs/`
- `README.md`
- `docs/demo_flow.md`
- `tests/test_failure_cases.py`

**Done criteria:**

- Nếu có future return labels: report có cumulative return, Sharpe, max drawdown, win rate, benchmark comparison.
- Nếu chưa có labels: report có decision distribution, average confidence, validator rejection rate, human review rate, conflict rate, skipped-agent rate, compliance violation rate.
- Demo có normal decision, high-risk conflict, stale/missing data.
- Một người khác clone repo, set env, chạy được demo path.

**Test/Review scope:**

- Failure tests cho missing context, invalid allocation, invalid LLM JSON, high risk + bullish ML, stale data, low news relevance.
- Review README/demo flow theo khả năng tái lập.

---

## 4. Gợi ý lịch 2 tuần

Lịch này chỉ là gợi ý sequencing; đơn vị quản lý chính vẫn là PR.

| Giai đoạn | PR | Mục tiêu |
|---|---|---|
| Ngày 1 | PR 1 | Chốt docs + contract |
| Ngày 2 | PR 2 | Schema + samples |
| Ngày 3 | PR 3 | DeepSeek + hierarchical crew skeleton |
| Ngày 4–5 | PR 4 | Deterministic policy services |
| Ngày 6–7 | PR 5 | Core specialist agents |
| Ngày 8–9 | PR 6 | Manager synthesis + API P0 |
| Ngày 10 | Stabilization | Fix P0 integration/test gaps |
| Ngày 11 | PR 7 | Compliance guardrails |
| Ngày 12 | PR 8 | Critic/debate extension |
| Ngày 13 | PR 9 | Evaluation/demo |
| Ngày 14 | Buffer | Docs polish, demo rehearsal, bug fixes |

Nếu thời gian thiếu, ưu tiên merge PR 1–6 để có P0 demo hoàn chỉnh. PR 7–9 là extension/reporting layer.

---

## 5. Definition of Done

### P0 Done

- Nhận `agent_ready_context` JSON.
- Chạy hierarchical CrewAI với custom `manager_agent`.
- Chạy Data/Technical, Sentiment, Valuation, Risk agents.
- Chạy Source Quality, Confidence, Conflict Resolution, Financial Validator, Human Review Gate, Audit.
- Trả final advisory decision qua `POST /api/v1/advisory/decision`.
- Output parseable, có `confidence_breakdown`, `not_financial_advice`, limitations, review flag và audit metadata.

### P1 Done

- Compliance deterministic checks chạy mặc định.
- Optional Compliance Reviewer agent có feature flag.
- Compliance violation có thể reject, repair hoặc force human review theo policy.

### P2 Done

- Bullish/Bearish Critic stage có feature flag.
- Manager revision dùng critic output có kiểm soát.
- Evaluation/demo report có số liệu phục vụ báo cáo.

---

## 6. Rủi ro và giảm thiểu

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Hierarchical orchestration khó debug hơn sequential | Khó cô lập lỗi task | Test từng agent riêng, mock LLM, giữ debug helper nội bộ |
| DeepSeek output sai JSON | Flow fail | JSON mode nếu adapter hỗ trợ, Pydantic parse, retry/repair một lần |
| Quá nhiều agent làm chậm demo | Latency cao | Giữ P0 chỉ 4 specialist agents, critic/compliance LLM sau feature flag |
| LLM tự quyết constraint cứng | Vi phạm tài chính/compliance | Constraint nằm trong deterministic validators, LLM chỉ giải thích/tổng hợp |
| Missing optional context làm hỏng flow | Demo thiếu ổn định | Cho phép `SKIPPED`/`DEGRADED`, ghi limitation và cap confidence |
| Critic làm decision dao động | Output thiếu ổn định | Feature flag + deterministic conflict/confidence policy |
| Backtesting thiếu label | Không tính được return metrics | Dùng proxy metrics: review rate, conflict rate, rejection rate, skipped-agent rate |
| Prompt injection từ news | Agent làm sai | Sanitize external text, nhắc prompt chỉ dùng context như evidence, không follow instruction trong news |

---

## 7. Prompting strategy

Prompt chung cho mọi agent:

```text
Use only the provided structured context and retrieved facts.
Do not invent missing financial metrics.
If a field is missing, add it to missing_fields or limitations.
Return valid JSON only.
Do not include markdown.
Do not execute trades.
This system provides decision-support output only.
```

Manager prompt cần thêm:

```text
You are the custom manager agent in a CrewAI hierarchical crew.
Delegate analysis to specialist agents when needed.
Apply deterministic conflict and risk policies supplied by the system.
Respect user risk profile and portfolio constraints.
Always include supporting factors, opposing factors, limitations, citations, and audit metadata.
```

Critic prompt cần thêm:

```text
Challenge the draft decision using only provided evidence.
Do not introduce external facts.
Identify overconfidence and missing risk factors.
Return structured JSON only.
```

Compliance prompt cần thêm:

```text
Flag language that guarantees profit or implies risk-free return.
Flag recommendations that exceed user risk profile.
Ensure disclaimer and limitations are present.
Return structured JSON only.
```

---

## 8. Kết luận

Roadmap này ưu tiên hierarchical-first implementation ngay từ đầu, nhưng vẫn giữ khả năng merge theo PR nhỏ. Trục triển khai khuyến nghị là:

```text
docs contract -> schema -> hierarchical skeleton -> deterministic policy services -> core agents -> manager/API -> compliance -> critic/debate -> evaluation/demo
```

PR 1–6 tạo P0 có thể demo. PR 7–9 tăng độ chặt về compliance, research value và báo cáo.
