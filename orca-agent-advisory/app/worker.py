from __future__ import annotations

import sys

from redis import Redis
from rq import Queue, Worker

from app.config import load_settings


def main() -> None:
    settings = load_settings()
    if not settings.redis_url:
        raise RuntimeError("ORCA_REDIS_URL is required to start worker")
    redis = Redis.from_url(settings.redis_url)
    worker = Worker([Queue(settings.decision_job_queue, connection=redis)], connection=redis)
    sys.exit(0 if worker.work(with_scheduler=False) else 1)


if __name__ == "__main__":
    main()
