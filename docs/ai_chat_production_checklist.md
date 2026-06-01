# AI Chat Production Checklist

File: `streamlit_app/pages/2_AI_Chat.py`

Goal: make screen production-usable with small shippable milestones. Push after each milestone.

## Milestone 1 — Safe basics

- [x] Replace multi-symbol input with single `Symbol submitted to ORCA` input.
- [x] Remove primary selector because ORCA chat is single-symbol only.
- [x] Keep single-symbol copy concise in sidebar/control labels.
- [x] Normalize symbol: trim, uppercase, `.` -> `-`.
- [x] Disable starter buttons when no valid primary symbol.
- [x] Show empty-symbol warning before submit.
- [x] Remove disabled `Cancel` button.
- [x] Verify: no valid symbol means no job can be submitted.

## Milestone 2 — Backend health + readiness

- [x] Add sidebar/backend status badge: `Connected`, `Degraded`, `Offline`.
- [x] Check `fetch_health()` and `fetch_status()` on page load or refresh button.
- [x] Disable submit when API offline.
- [x] Render readiness failure as table/card, not raw dict.
- [x] Show failed tool name, status, missing symbols, stale flag.
- [x] Verify: ORCA offline gives clear message and no submit.

## Milestone 3 — Job lifecycle

- [x] Replace manual-only job flow with clear job cards.
- [x] Show status: `queued`, `running`, `completed`, `failed`, `stale`.
- [x] Show created time, last refreshed time, elapsed time.
- [x] Show progress stage and progress percent when available.
- [x] Add `Refresh all` button.
- [x] Auto-fetch result when job status is completed.
- [x] Add `Remove` for failed/stale/completed jobs.
- [x] Verify: completed job result appears without extra guessing.

## Milestone 4 — No prompt leakage

- [x] Stop storing full prompt in `st.query_params`.
- [x] Store only `job_id`, `symbol`, `status`, `created_at`.
- [x] Keep full prompt only in `st.session_state`.
- [x] Add TTL for pending jobs, e.g. 1 hour.
- [x] Mark stale jobs and offer remove/retry.
- [x] Verify: browser URL contains no user prompt.

## Milestone 5 — Decision result cards

- [x] Replace plain markdown result with structured sections.
- [x] Header card: symbol, recommendation badge, confidence.
- [x] Show human-review warning prominently.
- [x] Show summary card.
- [x] Show rationale table/cards: factor, stance, weight, explanation.
- [x] Show risk warnings in amber/red callout.
- [x] Show supporting vs conflicting signals in two columns.
- [x] Put citations/audit in expanders.
- [x] Verify: recommendation, risk, human review visible in <5 seconds.

## Milestone 6 — Error handling

- [x] Classify errors: API offline, readiness fail, job fail, timeout, malformed response.
- [x] Show friendly message plus technical detail in expander.
- [x] Add retry action for failed submit/fetch.
- [x] Validate API response shape before indexing `job["job_id"]`.
- [x] Verify: malformed API response does not crash page.

## Milestone 7 — UX polish

- [x] Add conversation empty state.
- [x] Add `Clear chat` button.
- [x] Add `Copy summary` for completed decision.
- [x] Truncate long prompt text in pending job cards.
- [x] Test narrow screen; stack cards if needed.
- [x] Replace broad `Production mode` text with live status text.
- [x] Verify: page works on laptop/tablet width.

## Milestone 8 — Security + maintainability

- [x] Escape all dynamic HTML values.
- [x] Never render user prompt with `unsafe_allow_html=True`.
- [x] Reduce CSS selectors targeting Streamlit internals.
- [x] Add helper for safe HTML output.
- [x] Add UI smoke checklist for success/offline/failed/stale/no-symbol flows.
- [x] Verify: user input cannot inject HTML.

## Suggested push order

1. Push Milestone 1.
2. Push Milestone 2.
3. Push Milestone 3.
4. Push Milestone 4.
5. Push Milestone 5.
6. Push Milestone 6.
7. Push Milestone 7.
8. Push Milestone 8.

## Done definition

- [x] No empty-symbol submit.
- [x] Backend/data health visible.
- [x] Job lifecycle understandable.
- [x] Completed results render clearly.
- [x] Failed/stale jobs recoverable.
- [x] Prompt not stored in URL.
- [x] Risk and human review prominent.
- [x] Dynamic HTML escaped.

## UI smoke checklist

- Success: submit one symbol, confirm SSE events update status until structured decision appears and URL `orca_jobs` has no prompt.
- Async status: submit prompt, confirm chat shows user message only while job row listens to queued/running SSE status; no manual refresh or queued assistant filler appears.
- Offline: stop ORCA API, refresh backend, confirm submit disabled and clear offline error appears.
- Failed: force failed job or bad job ID, confirm failed status, technical detail, retry/remove actions.
- Stale: load queued/running job older than 1 hour, confirm stale status plus retry/remove.
- No-symbol: clear symbol input, confirm launch buttons disabled and chat submit returns no job.
- Injection: submit `<script>alert(1)</script>`, confirm text renders as text and no unsafe HTML path handles prompt.
