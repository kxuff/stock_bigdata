from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import psycopg


@dataclass(frozen=True)
class AgentRouteAuditEntry:
    audit_id: str
    request_id: str | None
    job_id: str | None
    tenant_id: str | None
    user_id: str | None
    message_hash: str
    route: str | None
    router_confidence: float | None
    symbols: list[str] = field(default_factory=list)
    status: str = "unknown"
    error_code: str | None = None
    latency_ms: int | None = None
    created_at: str | None = None


class AgentRouteAuditStore:
    def record(self, entry: AgentRouteAuditEntry) -> None:
        raise NotImplementedError


class NoopAgentRouteAuditStore(AgentRouteAuditStore):
    def record(self, entry: AgentRouteAuditEntry) -> None:
        return None


class MemoryAgentRouteAuditStore(AgentRouteAuditStore):
    def __init__(self) -> None:
        self.entries: list[AgentRouteAuditEntry] = []

    def record(self, entry: AgentRouteAuditEntry) -> None:
        self.entries.append(entry)


class PostgresAgentRouteAuditStore(AgentRouteAuditStore):
    def __init__(self, database_url: str, *, table_name: str = "orca_agent_route_audits") -> None:
        self.database_url = database_url
        self.table_name = _safe_table_name(table_name)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    audit_id text PRIMARY KEY,
                    request_id text,
                    job_id text,
                    tenant_id text,
                    user_id text,
                    message_hash text NOT NULL,
                    route text,
                    router_confidence double precision,
                    symbols jsonb NOT NULL DEFAULT '[]'::jsonb,
                    status text NOT NULL,
                    error_code text,
                    latency_ms integer,
                    created_at timestamptz NOT NULL
                )
                """,
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_request_id_idx ON {self.table_name} (request_id)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_job_id_idx ON {self.table_name} (job_id)")

    def record(self, entry: AgentRouteAuditEntry) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table_name} (
                    audit_id, request_id, job_id, tenant_id, user_id, message_hash, route,
                    router_confidence, symbols, status, error_code, latency_ms, created_at
                ) VALUES (
                    %(audit_id)s, %(request_id)s, %(job_id)s, %(tenant_id)s, %(user_id)s, %(message_hash)s,
                    %(route)s, %(router_confidence)s, %(symbols)s, %(status)s, %(error_code)s,
                    %(latency_ms)s, %(created_at)s
                )
                """,
                {
                    "audit_id": entry.audit_id,
                    "request_id": entry.request_id,
                    "job_id": entry.job_id,
                    "tenant_id": entry.tenant_id,
                    "user_id": entry.user_id,
                    "message_hash": entry.message_hash,
                    "route": entry.route,
                    "router_confidence": entry.router_confidence,
                    "symbols": entry.symbols,
                    "status": entry.status,
                    "error_code": entry.error_code,
                    "latency_ms": entry.latency_ms,
                    "created_at": entry.created_at,
                },
            )

    def ping(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")

    def _connect(self):
        return psycopg.connect(self.database_url)


def _safe_table_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or not cleaned.replace("_", "").isalnum():
        raise ValueError("invalid agent route audit table name")
    return cleaned
