from __future__ import annotations

from typing import Any

from redis import Redis
from rq import Queue

from app.config import AgentSettings


class DecisionJobQueue:
    def __init__(self, settings: AgentSettings) -> None:
        if not settings.redis_url:
            raise ValueError("ORCA_REDIS_URL is required for decision job queue")
        self.redis = Redis.from_url(settings.redis_url)
        self.queue = Queue(settings.decision_job_queue, connection=self.redis)

    def enqueue_decision_job(self, job_id: str, request_payload: dict[str, Any], *, timeout_seconds: int) -> str:
        job = self.queue.enqueue(
            "app.application.jobs.decision_job_runner.run_decision_job",
            job_id,
            request_payload,
            job_timeout=timeout_seconds,
            result_ttl=86400,
            failure_ttl=86400,
        )
        return job.id

    def ping(self) -> None:
        self.redis.ping()
