from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from eod_inference.config import PipelineConfig


def ensure_dirs(config: PipelineConfig) -> None:
    for name in ["staging"]:
        (config.data_dir / name).mkdir(parents=True, exist_ok=True)


def stage_dir(config: PipelineConfig, target_date: date) -> Path:
    path = config.data_dir / "staging" / target_date.strftime("%Y%m%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
