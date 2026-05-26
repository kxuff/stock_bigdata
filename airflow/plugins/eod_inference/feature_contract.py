from __future__ import annotations

import sys
from pathlib import Path


def _load_feature_contract():
    candidate_paths = [
        Path("/opt/airflow/spark_jobs"),
        Path.cwd() / "spark_jobs",
        Path(__file__).resolve().parents[3] / "spark_jobs",
    ]
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from ml_features import PRICE_FEATURE_COLUMNS, compute_price_features

    return PRICE_FEATURE_COLUMNS, compute_price_features


PRICE_FEATURE_COLUMNS, compute_price_features = _load_feature_contract()
