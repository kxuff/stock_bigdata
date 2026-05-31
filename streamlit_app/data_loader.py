from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from database import column_name, quote_identifier, read_sql, table_sql


INDICATOR_TABLE = "stock_market_indicator"
MARKET_TABLE = "stock_market"
NEWS_TABLE = "stock_news"
ALERT_TABLE = "stock_alerts"


def _c(table: str, name: str) -> str:
    return quote_identifier(column_name(table, name))


@st.cache_data(ttl=60, show_spinner=False)
def load_market_overview(limit: int = 8) -> pd.DataFrame:
    t = table_sql(INDICATOR_TABLE)
    dt = _c(INDICATOR_TABLE, "Datetime")
    indicator = _c(INDICATOR_TABLE, "Indicator")
    close = _c(INDICATOR_TABLE, "Close")
    volume = _c(INDICATOR_TABLE, "Volume")
    sql = f"""
    with latest as (
        select distinct on ({indicator})
            {indicator} as indicator,
            {dt} as latest_time,
            {close}::double precision as close,
            {volume}::double precision as volume
        from {t}
        order by {indicator}, {dt} desc
    ),
    previous as (
        select distinct on (m.{indicator})
            m.{indicator} as indicator,
            m.{close}::double precision as previous_close
        from {t} m
        join latest l on l.indicator = m.{indicator}
        where m.{dt} < l.latest_time and date(m.{dt}) < date(l.latest_time)
        order by m.{indicator}, m.{dt} desc
    )
    select
        l.indicator,
        l.latest_time,
        l.close,
        p.previous_close,
        l.close - p.previous_close as change_value,
        case when p.previous_close is null or p.previous_close = 0 then null
             else (l.close - p.previous_close) / p.previous_close * 100 end as change_pct,
        l.volume
    from latest l
    left join previous p using (indicator)
    order by l.indicator
    limit :limit
    """
    return read_sql(sql, {"limit": limit})


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_market(symbols: tuple[str, ...] | None = None, limit: int = 500) -> pd.DataFrame:
    t = table_sql(MARKET_TABLE)
    dt = _c(MARKET_TABLE, "Datetime")
    symbol = _c(MARKET_TABLE, "Symbol")
    open_ = _c(MARKET_TABLE, "Open")
    high = _c(MARKET_TABLE, "High")
    low = _c(MARKET_TABLE, "Low")
    close = _c(MARKET_TABLE, "Close")
    volume = _c(MARKET_TABLE, "Volume")
    symbol_filter = ""
    params: dict = {"limit": limit}
    if symbols:
        symbol_filter = f"where {symbol} = any(:symbols)"
        params["symbols"] = list(symbols)

    sql = f"""
    with latest as (
        select distinct on ({symbol})
            {dt} as datetime,
            {symbol} as symbol,
            {open_}::double precision as open,
            {high}::double precision as high,
            {low}::double precision as low,
            {close}::double precision as close,
            {volume}::double precision as volume
        from {t}
        {symbol_filter}
        order by {symbol}, {dt} desc
    )
    select *,
        case when open is null or open = 0 then null
             else (close - open) / open * 100 end as day_change_pct
    from latest
    order by symbol
    limit :limit
    """
    return read_sql(sql, params)


@st.cache_data(ttl=60, show_spinner=False)
def load_symbols() -> list[str]:
    df = read_sql(
        f"""
        select distinct {_c(MARKET_TABLE, "Symbol")} as symbol
        from {table_sql(MARKET_TABLE)}
        where {_c(MARKET_TABLE, "Symbol")} is not null
        order by symbol
        limit 2000
        """
    )
    return df["symbol"].dropna().astype(str).tolist()


@st.cache_data(ttl=60, show_spinner=False)
def load_stock_history(symbol: str, start_time: datetime) -> pd.DataFrame:
    sql = f"""
    select
        {_c(MARKET_TABLE, "Datetime")} as datetime,
        {_c(MARKET_TABLE, "Symbol")} as symbol,
        {_c(MARKET_TABLE, "Open")}::double precision as open,
        {_c(MARKET_TABLE, "High")}::double precision as high,
        {_c(MARKET_TABLE, "Low")}::double precision as low,
        {_c(MARKET_TABLE, "Close")}::double precision as close,
        {_c(MARKET_TABLE, "Volume")}::double precision as volume
    from {table_sql(MARKET_TABLE)}
    where {_c(MARKET_TABLE, "Symbol")} = :symbol
      and {_c(MARKET_TABLE, "Datetime")} >= :start_time
    order by {_c(MARKET_TABLE, "Datetime")}
    limit 20000
    """
    return read_sql(sql, {"symbol": symbol, "start_time": start_time})


@st.cache_data(ttl=300, show_spinner=False)
def load_news(symbol: str | None, search: str, limit: int = 200) -> pd.DataFrame:
    where = []
    params: dict = {"limit": limit}
    if symbol and symbol != "All":
        where.append(f"{_c(NEWS_TABLE, 'Symbol')} = :symbol")
        params["symbol"] = symbol
    if search:
        where.append(f"{_c(NEWS_TABLE, 'headline')} ilike :search")
        params["search"] = f"%{search}%"
    where_sql = "where " + " and ".join(where) if where else ""
    sql = f"""
    select
        {_c(NEWS_TABLE, "Datetime")} as datetime,
        {_c(NEWS_TABLE, "Symbol")} as symbol,
        {_c(NEWS_TABLE, "headline")} as headline,
        {_c(NEWS_TABLE, "source")} as source,
        {_c(NEWS_TABLE, "url")} as url,
        {_c(NEWS_TABLE, "event_timestamp")} as event_timestamp,
        {_c(NEWS_TABLE, "image")} as image
    from {table_sql(NEWS_TABLE)}
    {where_sql}
    order by {_c(NEWS_TABLE, "event_timestamp")} desc nulls last,
             {_c(NEWS_TABLE, "Datetime")} desc
    limit :limit
    """
    return read_sql(sql, params)


@st.cache_data(ttl=60, show_spinner=False)
def load_alerts(
    symbols: tuple[str, ...],
    alert_types: tuple[str, ...],
    alert_levels: tuple[str, ...],
    start_date: date | None,
    end_date: date | None,
    limit: int = 500,
) -> pd.DataFrame:
    where = []
    params: dict = {"limit": limit}
    if symbols:
        where.append("symbol = any(:symbols)")
        params["symbols"] = list(symbols)
    if alert_types:
        where.append("alert_type = any(:alert_types)")
        params["alert_types"] = list(alert_types)
    if alert_levels:
        where.append("alert_level = any(:alert_levels)")
        params["alert_levels"] = list(alert_levels)
    if start_date:
        where.append("event_time >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("event_time < :end_date_plus")
        params["end_date_plus"] = end_date + timedelta(days=1)
    where_sql = "where " + " and ".join(where) if where else ""
    sql = f"""
    select symbol, event_time, alert_type, alert_level, feature_name, feature_value,
           threshold_value, message, metadata
    from {table_sql(ALERT_TABLE)}
    {where_sql}
    order by event_time desc
    limit :limit
    """
    return read_sql(sql, params)


@st.cache_data(ttl=300, show_spinner=False)
def load_alert_filter_values() -> tuple[list[str], list[str], list[str]]:
    df = read_sql(
        f"""
        select distinct symbol, alert_type, alert_level
        from {table_sql(ALERT_TABLE)}
        order by symbol, alert_type, alert_level
        limit 5000
        """
    )
    return (
        sorted(df["symbol"].dropna().astype(str).unique().tolist()),
        sorted(df["alert_type"].dropna().astype(str).unique().tolist()),
        sorted(df["alert_level"].dropna().astype(str).unique().tolist()),
    )
