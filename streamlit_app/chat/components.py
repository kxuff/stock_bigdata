"""Render components for ORCA AI Chat."""
from __future__ import annotations

from html import escape
from uuid import uuid4

import streamlit as st


# ── helpers ──────────────────────────────────────────────────────────────────

def _rec_cls(rec: str) -> str:
    return {"BUY": "buy", "SELL": "sell", "HOLD": "hold", "WATCH": "watch"}.get(
        str(rec).upper(), "wait"
    )


def _rec_icon(rec: str) -> str:
    return {"BUY": "↑", "SELL": "↓", "HOLD": "→", "WATCH": "◎"}.get(
        str(rec).upper(), "•"
    )


def _conf_cls(conf) -> str:
    try:
        v = float(conf)
        if v >= 0.65: return "conf-high"
        if v >= 0.40: return "conf-medium"
    except (TypeError, ValueError):
        pass
    return "conf-low"


def _conf_fmt(conf) -> str:
    try:
        return f"{float(conf):.0%}"
    except (TypeError, ValueError):
        return str(conf) if conf else "N/A"


def _signal_chips(signals: list, kind: str) -> str:
    cls = "support" if kind == "support" else "conflict"
    icon = "✦" if kind == "support" else "✕"
    chips = "".join(
        f'<span class="signal-chip {cls}">{icon} {s}</span>'
        for s in (signals or [])[:4]
    )
    return f'<div class="signal-row">{chips}</div>' if chips else ""


def kicker(label: str) -> None:
    st.markdown(f'<div class="orca-kicker">{label}</div>', unsafe_allow_html=True)


def route_tag(route: str, confidence: float | None = None) -> str:
    label = str(route).replace("_", " ").title()
    return f'<span class="route-tag">{label}</span>'


# ── Decision card ─────────────────────────────────────────────────────────────

def render_decision(decision: dict) -> None:
    symbol = decision.get("symbol", "N/A")
    rec    = str(decision.get("recommendation", "WAIT")).upper()
    conf   = decision.get("confidence")
    cls    = _rec_cls(rec)
    icon   = _rec_icon(rec)

    # Header card
    st.markdown(f"""
<div class="orca-card {cls}">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
    <div style="display:flex;align-items:center;gap:0.75rem;">
      <span style="font-size:1.6rem;font-weight:800;color:#e2e8f0;">{symbol}</span>
      <span class="rec-badge {cls}">{icon} {rec}</span>
    </div>
    <span class="conf-pill {_conf_cls(conf)}">{_conf_fmt(conf)} confidence</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if decision.get("requires_human_review"):
        st.markdown(
            '<div class="orca-alert orca-alert-warn">'
            '<span class="orca-alert-icon">⚠️</span>'
            '<div>Human review required before any action.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Summary / advisor note
    summary = decision.get("summary", "")
    if summary:
        st.markdown("**Advisor note**")
        safe_summary = escape(str(summary)).replace("\n", "<br>")
        st.markdown(
            '<div class="orca-alert orca-alert-info">'
            '<span class="orca-alert-icon">💬</span>'
            f'<div>{safe_summary}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Signals / detailed evidence. Keep user-facing answer prose-first; details collapsed.
    supporting = decision.get("supporting_signals") or decision.get("supporting_evidence") or []
    conflicting = decision.get("conflicting_signals") or decision.get("conflicts") or []
    if supporting or conflicting:
        with st.expander("Why this recommendation", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**What supports the call**")
                for s in supporting[:5]:
                    st.markdown(f"<div style='color:#6ee7b7;font-size:0.85rem;margin:.15rem 0'>• {s}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown("**What could change the view**")
                for s in conflicting[:5]:
                    st.markdown(f"<div style='color:#fda4af;font-size:0.85rem;margin:.15rem 0'>• {s}</div>", unsafe_allow_html=True)

    # Rationale table
    rationale = [r for r in (decision.get("decision_rationale") or []) if isinstance(r, dict)]
    if rationale:
        with st.expander("📊 Decision rationale", expanded=False):
            rows = []
            for r in rationale[:6]:
                rows.append(
                    "<tr>"
                    f"<td>{escape(str(r.get('factor', '—')))}</td>"
                    f"<td>{escape(str(r.get('stance', '—')))}</td>"
                    f"<td>{escape(str(r.get('weight', '—')))}</td>"
                    f"<td>{escape(str(r.get('explanation', '')))}</td>"
                    "</tr>"
                )
            st.markdown(
                '<table class="orca-rationale-table">'
                "<thead><tr>"
                "<th>Factor</th><th>Stance</th><th>Weight</th><th>Explanation</th>"
                "</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                "</table>",
                unsafe_allow_html=True,
            )

    # Risk warnings
    warnings = decision.get("risk_warnings") or []
    if warnings:
        with st.expander("⚠️ Risk warnings", expanded=False):
            for w in warnings:
                st.markdown(f"<div style='color:#fda4af;font-size:0.85rem;margin:.15rem 0'>• {w}</div>", unsafe_allow_html=True)

    # Source quality + audit
    sq = decision.get("source_quality") or {}
    if sq:
        with st.expander("Data quality", expanded=False):
            cols = st.columns(4)
            for col, (label, key) in zip(cols, [
                ("Overall", "overall_quality_score"),
                ("Freshness", "freshness_score"),
                ("Relevance", "relevance_score"),
                ("Complete", "completeness_score"),
            ]):
                v = sq.get(key)
                col.metric(label, f"{float(v):.0%}" if v is not None else "—")

            citations = [str(c) for c in (decision.get("data_citations") or []) if c]
            tool_calls = (decision.get("retrieved_tool_audit") or {}).get("tool_calls") or []
            if citations or tool_calls:
                st.markdown("**Source citations**")
                if citations:
                    refs = "".join(f"<li><code>{escape(ref)}</code></li>" for ref in citations[:8])
                    st.markdown(f'<ul class="orca-source-list">{refs}</ul>', unsafe_allow_html=True)

                rows = []
                for t in tool_calls[:8]:
                    source_refs = t.get("source_refs") or []
                    refs = "<br>".join(f"<code>{escape(str(ref))}</code>" for ref in source_refs[:4]) or "—"
                    rows.append(
                        "<tr>"
                        f"<td>{escape(str(t.get('tool', '—')))}</td>"
                        f"<td>{escape(str(t.get('status', '—')))}</td>"
                        f"<td>{refs}</td>"
                        f"<td><code>{escape(str(t.get('result_hash', '—')))}</code></td>"
                        "</tr>"
                    )
                if rows:
                    st.markdown(
                        '<table class="orca-source-table">'
                        "<thead><tr><th>Tool</th><th>Status</th><th>Source refs</th><th>Result hash</th></tr></thead>"
                        f"<tbody>{''.join(rows)}</tbody>"
                        "</table>",
                        unsafe_allow_html=True,
                    )

    # Tool audit
    tool_audit = decision.get("retrieved_tool_audit") or {}
    tool_calls = tool_audit.get("tool_calls") or []
    if tool_calls:
        with st.expander("🔧 Tool audit", expanded=False):
            for t in tool_calls:
                status = t.get("status", "unknown")
                icon_t = "✅" if status == "SUCCESS" else "❌"
                st.markdown(f"`{icon_t} {t.get('tool', '?')}` — {status}")

    # Copy summary
    summary_text = (
        f"{symbol} · {rec} · confidence {_conf_fmt(conf)}\n\n{decision.get('summary', '')}"
    )
    with st.expander("📋 Copy summary", expanded=False):
        st.text_area(
            "Copyable summary",
            summary_text,
            height=90,
            key=f"copy-{decision.get('run_id', uuid4())}",
            label_visibility="collapsed",
        )


# ── Agent response card ───────────────────────────────────────────────────────

def render_agent_response(response: dict) -> None:
    result_type = response.get("result_type")
    result      = response.get("result") or {}
    route       = response.get("route", "unknown")
    conf        = response.get("router_confidence")

    st.markdown(
        f'{route_tag(route, conf)}',
        unsafe_allow_html=True,
    )

    msg = response.get("message")
    if msg:
        st.caption(msg)

    if result_type == "single_symbol_decision":
        render_decision(result)
        return

    _STRUCTURED = {
        "symbol_comparison", "universe_screen", "watchlist_review",
        "market_brief", "top_stocks", "data_diagnostics", "portfolio_rebalance",
        "backtest_analysis", "streaming_pipeline_health",
        "streaming_freshness_check", "streaming_alert_review",
        "streaming_symbol_monitor", "streaming_feature_drift",
        "streaming_ingestion_lag", "streaming_topic_inspection",
        "streaming_quality_incidents",
    }
    if result_type in _STRUCTURED:
        _render_structured(result_type, result)

    actions = response.get("suggested_actions") or []
    if actions:
        st.markdown("**Suggested next steps**")
        for action in actions[:4]:
            st.markdown(f"› {action.get('label', action)}")


# ── Structured result renderers ───────────────────────────────────────────────

def _rows_header(rows: list) -> None:
    scored   = sum(1 for r in rows if r.get("final_score") is not None)
    warnings = sum(len(r.get("warnings") or []) for r in rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(rows))
    c2.metric("Scored", scored)
    c3.metric("Warnings", warnings)


def _warn_expander(rows: list) -> None:
    msgs = [
        f"**{r.get('symbol','?')}**: {w}"
        for r in rows for w in (r.get("warnings") or [])
    ]
    if msgs:
        with st.expander(f"⚠️ {len(msgs)} data warning(s)", expanded=True):
            for m in msgs[:20]:
                st.markdown(f"<span style='color:#fcd34d;font-size:0.84rem'>• {m}</span>", unsafe_allow_html=True)


def _render_structured(result_type: str, result: dict) -> None:  # noqa: C901
    # ── Screen / comparison / watchlist ──
    rows_key = {
        "symbol_comparison": "rows",
        "universe_screen":   "candidates",
        "watchlist_review":  "items",
        "market_brief":      "leaders",
        "top_stocks":        "stocks",
    }.get(result_type)

    if rows_key is not None:
        if result_type == "market_brief" and result.get("summary"):
            st.info(f"📈 {result['summary']}")
        if result_type == "market_brief":
            _render_market_brief_sections(result)
        rows = result.get(rows_key) or []
        if rows:
            _rows_header(rows)
            _warn_expander(rows)
            st.dataframe(_display_rows(result_type, rows), width=900, hide_index=True)
        else:
            render_empty(f"No {result_type.replace('_', ' ')} data returned.")
        if result_type == "universe_screen":
            diagnostics = result.get("diagnostics") or {}
            if diagnostics:
                with st.expander("🔍 Diagnostics"):
                    st.json(diagnostics)
        return

    if result_type == "data_diagnostics":
        st.json(result.get("diagnostics") or result)
        return

    if result_type == "portfolio_rebalance":
        msg = result.get("message", "")
        if msg:
            st.info(f"💬 {msg}")
        changes = result.get("changes") or []
        if changes:
            st.dataframe(changes, width=900, hide_index=True)
        c1, c2 = st.columns(2)
        c1.metric("Cash target %", f"{result.get('cash_target_weight', 0):.2f}%")
        c2.metric("Human review", "Required ✓" if result.get("human_review_required") else "Not required")
        with st.expander("📐 Constraints"):
            st.json(result.get("constraints") or {})
        return

    if result_type == "backtest_analysis":
        status = result.get("status", "planned")
        icon = {"completed": "✅", "planned": "📋", "disabled": "🚫"}.get(status, "•")
        st.info(f"{icon} {result.get('limitation') or 'Backtest service not connected.'}")
        if result.get("suggested_next_action"):
            st.success(f"→ {result['suggested_next_action']}")
        metrics = result.get("metrics") or {}
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k.replace("_", " ").title(), v)
        with st.expander("📄 Backtest spec", expanded=True):
            st.json(result.get("backtest_spec") or {})
        return

    _STREAMING_KEY = {
        "streaming_pipeline_health":    "stages",
        "streaming_freshness_check":    "rows",
        "streaming_alert_review":       "alerts",
        "streaming_feature_drift":      "rows",
        "streaming_ingestion_lag":      "rows",
        "streaming_topic_inspection":   "samples",
        "streaming_quality_incidents":  "incidents",
    }
    if result_type == "streaming_symbol_monitor":
        if result.get("symbol"):
            st.metric("Symbol", result["symbol"])
        fresh = result.get("freshness") or []
        alerts = result.get("alerts") or []
        if fresh:
            st.markdown("**Freshness**")
            st.dataframe(fresh, width=900, hide_index=True)
        if alerts:
            st.markdown("**Active alerts**")
            st.dataframe(alerts, width=900, hide_index=True)
        if not fresh and not alerts:
            st.json(result)
        return

    if result_type in _STREAMING_KEY:
        rows = result.get(_STREAMING_KEY[result_type]) or []
        if rows:
            st.dataframe(rows, width=900, hide_index=True)
        else:
            st.json(result)
        return


def _render_market_brief_sections(result: dict) -> None:
    sections = [
        ("Highlights", "highlights"),
        ("Hot news", "hot_news"),
        ("Risk flags", "risk_flags"),
        ("Watch next", "watch_next"),
    ]
    for title, key in sections:
        items = result.get(key) or []
        if not items:
            continue
        st.markdown(f"**{title}**")
        for item in items[:6]:
            st.markdown(f"› {item}")


def _display_rows(result_type: str, rows: list) -> list:
    if result_type not in {"market_brief", "top_stocks"}:
        return rows
    cols = ["symbol", "Symbol", "final_score", "latest_price", "price", "RSI14", "RVOL20", "risk_prob"]
    labels = {
        "symbol": "Symbol",
        "Symbol": "Symbol",
        "final_score": "ORCA Score",
        "latest_price": "Price",
        "price": "Price",
        "RSI14": "RSI14",
        "RVOL20": "RVOL20",
        "risk_prob": "Risk Prob",
    }
    display = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {}
        for col in cols:
            if col in row and row[col] is not None:
                item[labels[col]] = _fmt_cell(col, row[col])
        display.append(item)
    return display or rows


def _fmt_cell(col: str, value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if col == "final_score" and numeric <= 1:
        return round(numeric * 100, 1)
    if col in {"final_score", "latest_price", "price", "RSI14", "RVOL20", "risk_prob"}:
        return round(numeric, 2)
    return value


# ── Empty state ───────────────────────────────────────────────────────────────

def render_empty(label: str = "No data returned.") -> None:
    st.markdown(f"""
<div class="empty-state">
  <div class="icon">📭</div>
  <p>{label}</p>
</div>
""", unsafe_allow_html=True)


# ── Chat empty state ──────────────────────────────────────────────────────────

def render_chat_empty() -> None:
    st.markdown("""
<div class="empty-state" style="padding:4rem 2rem;">
  <div class="icon">🧠</div>
  <h3>Ask ORCA anything</h3>
  <p>Market brief, compare symbols, watchlist review, advisory decisions, portfolio rebalance — try a quick question below.</p>
</div>
""", unsafe_allow_html=True)


# ── Backend status badge ──────────────────────────────────────────────────────

def render_backend_pill(state: str, error: str | None = None) -> None:
    cfg = {
        "Connected": ("dot-ok",      "🟢 Connected"),
        "Degraded":  ("dot-warn",    "🟡 Degraded"),
        "Offline":   ("dot-offline", "⚫ Offline"),
    }
    dot_cls, label = cfg.get(state, ("dot-offline", f"• {state}"))
    st.markdown(
        f'<span style="font-size:0.82rem;font-weight:600;">'
        f'<span class="status-dot {dot_cls}"></span>{label}</span>',
        unsafe_allow_html=True,
    )
    if error:
        st.caption(error)


# ── Job status row ────────────────────────────────────────────────────────────

STATUS_ICON = {
    "queued":    "🕓",
    "running":   "🔄",
    "completed": "✅",
    "failed":    "❌",
    "stale":     "⚠️",
}


def render_job_row(job: dict, status: str, cols) -> None:
    icon = STATUS_ICON.get(status, "•")
    cols[0].markdown(f"**{job.get('symbol','N/A')}**")
    cols[1].markdown(f"{icon} `{status}`")
    cols[2].caption(_truncate(job.get("prompt"), 52))
    cols[3].caption(job.get("created_at", "—")[:19] if job.get("created_at") else "—")


def _truncate(value: str | None, limit: int = 52) -> str:
    text = value or "—"
    return text if len(text) <= limit else f"{text[:limit-1]}…"
