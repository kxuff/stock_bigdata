from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/stock_db"


def _database_url() -> str:
    try:
        return st.secrets["postgres"]["url"]
    except Exception:
        pass
    return os.getenv("STOCK_DB_URL") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or DEFAULT_DB_URL


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return create_engine(_database_url(), pool_pre_ping=True, pool_recycle=1800)


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as connection:
        result = connection.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_sql(table_name: str) -> str:
    return ".".join(quote_identifier(part) for part in table_name.split("."))


@lru_cache(maxsize=64)
def table_columns(table_name: str) -> tuple[str, ...]:
    parts = table_name.split(".")
    schema, table = parts if len(parts) == 2 else ("public", parts[0])
    df = read_sql(
        """
        select column_name
        from information_schema.columns
        where table_schema = :schema and table_name = :table
        order by ordinal_position
        """,
        {"schema": schema, "table": table},
    )
    return tuple(df["column_name"].tolist())


def column_name(table_name: str, candidates: str | Iterable[str]) -> str:
    names = [candidates] if isinstance(candidates, str) else list(candidates)
    lower_map = {column.lower(): column for column in table_columns(table_name)}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"Missing column in {table_name}: one of {names}")
