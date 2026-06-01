from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import errors
from psycopg.rows import dict_row


JOB_COLUMNS = [
    "job_id",
    "request_id",
    "status",
    "progress_stage",
    "progress_pct",
    "request_payload",
    "request_hash",
    "result_run_id",
    "result_payload",
    "error_code",
    "error_message",
    "error_payload",
    "idempotency_key",
    "tenant_id",
    "user_id",
    "created_by",
    "created_at",
    "updated_at",
    "started_at",
    "completed_at",
]


@dataclass(frozen=True)
class CreateJobResult:
    job: dict[str, Any]
    created: bool


class IdempotencyConflictError(RuntimeError):
    pass


class DecisionJobStore:
    def create_job(
        self,
        job: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        idempotency_key: str | None,
        tenant_id: str,
        user_id: str | None = None,
        created_by: str | None = None,
    ) -> CreateJobResult:
        raise NotImplementedError

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def update_job(self, job_id: str, **updates: Any) -> None:
        raise NotImplementedError


class PostgresDecisionJobStore(DecisionJobStore):
    def __init__(self, database_url: str, *, table_name: str = "orca_decision_jobs") -> None:
        self.database_url = database_url
        self.table_name = _safe_table_name(table_name)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    job_id text PRIMARY KEY,
                    request_id text NOT NULL,
                    status text NOT NULL,
                    progress_stage text,
                    progress_pct integer NOT NULL DEFAULT 0,
                    request_payload jsonb NOT NULL,
                    request_hash text,
                    result_run_id text,
                    result_payload jsonb,
                    error_code text,
                    error_message text,
                    error_payload jsonb,
                    idempotency_key text,
                    tenant_id text NOT NULL DEFAULT 'local',
                    user_id text,
                    created_by text,
                    created_at timestamptz NOT NULL,
                    updated_at timestamptz NOT NULL,
                    started_at timestamptz,
                    completed_at timestamptz
                )
                """,
            )
            cur.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {self.table_name}_idempotency_key_idx
                ON {self.table_name} (tenant_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """,
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_status_idx ON {self.table_name} (status)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_request_id_idx ON {self.table_name} (request_id)")
            cur.execute(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS request_hash text")

    def create_job(
        self,
        job: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        idempotency_key: str | None,
        tenant_id: str,
        user_id: str | None = None,
        created_by: str | None = None,
    ) -> CreateJobResult:
        request_hash = _request_hash(request_payload)
        payload = {
            "job_id": job["job_id"],
            "request_id": job["request_id"],
            "status": job["status"],
            "progress_stage": job.get("progress_stage"),
            "progress_pct": job.get("progress_pct", job.get("progress", 0)),
            "request_payload": json.dumps(request_payload),
            "request_hash": request_hash,
            "result_run_id": job.get("run_id"),
            "result_payload": json.dumps(job.get("result")) if job.get("result") is not None else None,
            "error_code": None,
            "error_message": None,
            "error_payload": json.dumps(job.get("error")) if job.get("error") is not None else None,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "created_by": created_by,
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            if idempotency_key:
                cur.execute(
                    f"""
                    SELECT {", ".join(JOB_COLUMNS)}
                    FROM {self.table_name}
                    WHERE tenant_id = %s AND idempotency_key = %s
                    """,
                    (tenant_id, idempotency_key),
                )
                existing = cur.fetchone()
                if existing is not None:
                    _ensure_same_request(existing, request_hash)
                    return CreateJobResult(job=_row_to_job(existing), created=False)
            try:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} ({", ".join(JOB_COLUMNS)})
                    VALUES ({", ".join(f"%({column})s" for column in JOB_COLUMNS)})
                    ON CONFLICT (job_id) DO NOTHING
                    RETURNING {", ".join(JOB_COLUMNS)}
                    """,
                    payload,
                )
            except errors.UniqueViolation:
                conn.rollback()
                if not idempotency_key:
                    raise
                with self._connect() as retry_conn, retry_conn.cursor(row_factory=dict_row) as retry_cur:
                    retry_cur.execute(
                        f"""
                        SELECT {", ".join(JOB_COLUMNS)}
                        FROM {self.table_name}
                        WHERE tenant_id = %s AND idempotency_key = %s
                        """,
                        (tenant_id, idempotency_key),
                    )
                    existing = retry_cur.fetchone()
                    if existing is None:
                        raise
                    _ensure_same_request(existing, request_hash)
                    return CreateJobResult(job=_row_to_job(existing), created=False)
            inserted = cur.fetchone()
            if inserted is not None:
                return CreateJobResult(job=_row_to_job(inserted), created=True)
            if idempotency_key:
                cur.execute(
                    f"""
                    SELECT {", ".join(JOB_COLUMNS)}
                    FROM {self.table_name}
                    WHERE tenant_id = %s AND idempotency_key = %s
                    """,
                    (tenant_id, idempotency_key),
                )
                existing = cur.fetchone()
                if existing is not None:
                    _ensure_same_request(existing, request_hash)
                    return CreateJobResult(job=_row_to_job(existing), created=False)
            cur.execute(f"SELECT {', '.join(JOB_COLUMNS)} FROM {self.table_name} WHERE job_id = %s", (job["job_id"],))
            existing = cur.fetchone()
            if existing is None:
                raise RuntimeError("decision job insert failed")
            return CreateJobResult(job=_row_to_job(existing), created=False)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {', '.join(JOB_COLUMNS)} FROM {self.table_name} WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
        return _row_to_job(row) if row else None

    def ping(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")

    def update_job(self, job_id: str, **updates: Any) -> None:
        if not updates:
            return
        mapped = _map_updates(updates)
        assignments = ", ".join(f"{column} = %({column})s" for column in mapped)
        mapped["job_id"] = job_id
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE {self.table_name} SET {assignments} WHERE job_id = %(job_id)s", mapped)

    def _connect(self):
        return psycopg.connect(self.database_url)


def _map_updates(updates: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in updates.items():
        if key == "progress":
            mapped["progress_pct"] = value
        elif key == "run_id":
            mapped["result_run_id"] = value
        elif key == "result":
            mapped["result_payload"] = json.dumps(value) if value is not None else None
        elif key == "error":
            mapped["error_payload"] = json.dumps(value) if value is not None else None
            if isinstance(value, dict):
                body = value.get("body") if isinstance(value.get("body"), dict) else {}
                mapped["error_code"] = body.get("error_code")
                mapped["error_message"] = body.get("message")
        elif key == "updated_at":
            mapped["updated_at"] = value
        elif key in {"status", "progress_stage", "progress_pct", "started_at", "completed_at"}:
            mapped[key] = value
    return mapped


def _row_to_job(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "request_id": row["request_id"],
        "status": row["status"],
        "progress": row["progress_pct"],
        "progress_stage": row.get("progress_stage"),
        "run_id": row.get("result_run_id"),
        "error": row.get("error_payload"),
        "result": row.get("result_payload"),
        "created_at": _iso(row.get("created_at")),
        "updated_at": _iso(row.get("updated_at")),
        "started_at": _iso(row.get("started_at")),
        "completed_at": _iso(row.get("completed_at")),
        "idempotency_key": row.get("idempotency_key"),
        "tenant_id": row.get("tenant_id"),
        "user_id": row.get("user_id"),
        "created_by": row.get("created_by"),
    }


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _safe_table_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or not cleaned.replace("_", "").isalnum():
        raise ValueError("invalid decision job table name")
    return cleaned


def _request_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ensure_same_request(row: dict[str, Any], request_hash: str) -> None:
    stored_hash = row.get("request_hash")
    if stored_hash and stored_hash != request_hash:
        raise IdempotencyConflictError("Idempotency-Key already used with different request payload")
