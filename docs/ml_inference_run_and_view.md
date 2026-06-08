# ML Inference Run And View

This runbook shows how to generate ML predictions for a signal date and view the recommendation output locally.

## 1. Start From Repo Root

```bash
cd /Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata
```

Use the project virtualenv for all commands:

```bash
.venv/bin/python --version
```

## 2. Generate Predictions For A Date

Use this command when you want predictions for one signal date. The date in Python uses `YYYY, M, D`.

Example for `2026-06-08`:

```bash
ML_INFERENCE_AUTO_REFRESH=true \
US_STOCK_EOD_DATA_DIR="$PWD/data/eod_batch" \
.venv/bin/python - <<'PY'
from datetime import date
from streamlit_app.services.ml_inference_api import ensure_latest_ml_inference

a = ensure_latest_ml_inference(today=date(2026, 6, 8))

print("expected_signal_date:", a.expected_signal_date)
print("prediction_path:", a.prediction_path)
print("refreshed:", a.refreshed)
print("refresh_error:", a.refresh_error)
PY
```

Expected output shape:

```text
expected_signal_date: 2026-06-08
prediction_path: /Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260608/predictions.parquet
refreshed: True or False
refresh_error: None
```

`refreshed=False` is not an error. It means the prediction file already existed and was reused.

## 3. View Top Recommendations From CLI

Change `run_date` to the date folder you want, using `YYYYMMDD`.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

import pandas as pd

run_date = "20260608"
path = Path(f"data/eod_batch/staging/{run_date}/predictions.parquet")

df = pd.read_parquet(path)
df = df.sort_values("final_score", ascending=False)

columns = ["Datetime", "Symbol", "entry_price", "pred_a", "risk_prob", "final_score"]
print(df[columns].head(20).to_string(index=False))
PY
```

## 4. View Qualified Picks With App Filters

This uses the same normalization and filters as the Streamlit page:

- `Pred_A >= 0.06`
- `Risk_Prob <= 30%`
- sorted by `FinalScore`

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

import pandas as pd

from streamlit_app.services.ml_inference_api import normalize_ml_inference_picks

run_date = "20260608"
path = Path(f"data/eod_batch/staging/{run_date}/predictions.parquet")

picks = normalize_ml_inference_picks(pd.read_parquet(path), limit=20)
print(picks.to_string(index=False))
PY
```

## 5. Run Direct Inference For An Existing Feature Batch

Use this when `features.parquet` and `feature_manifest.json` already exist for the date.

Example for `2026-06-08`:

```bash
US_STOCK_EOD_DATA_DIR="$PWD/data/eod_batch" \
US_STOCK_MODEL_A_PATH="$PWD/data/models/model_a.joblib" \
US_STOCK_MODEL_C_PATH="$PWD/data/models/model_c.joblib" \
.venv/bin/python - <<'PY'
import json
from pathlib import Path

from inference_direct import run_ml_inference_direct

run_date = "20260608"
stage_dir = Path(f"data/eod_batch/staging/{run_date}")

manifest = json.loads((stage_dir / "feature_manifest.json").read_text(encoding="utf-8"))
manifest["feature_batch"] = str((stage_dir / "features.parquet").resolve())

result = run_ml_inference_direct(
    manifest,
    model_a_path=Path("data/models/model_a.joblib"),
    model_c_path=Path("data/models/model_c.joblib"),
)

print(json.dumps(result, indent=2, default=str))
PY
```

## 6. Run The Streamlit App

```bash
.venv/bin/streamlit run streamlit_app/app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Then open the sidebar page:

```text
AI Stock Picks
```

Use:

- `Strategy Backtest` to run historical performance.
- `Recommendations By Date` to choose a signal date and click `Get Recommendation`.

## 7. Generated Files

For a run date like `2026-06-08`, the local batch files are:

```text
data/eod_batch/staging/20260608/features.parquet
data/eod_batch/staging/20260608/feature_manifest.json
data/eod_batch/staging/20260608/predictions.parquet
data/eod_batch/staging/20260608/inference_manifest.json
```

## 8. Troubleshooting

XGBoost GPU warning:

```text
Device is changed from GPU to CPU
```

This is OK for local runs. It means the model can run on CPU because no GPU is available.

Yahoo download DNS errors:

```text
Could not resolve host: guce.yahoo.com
```

This means local feature generation cannot download market prices. If the date already has `features.parquet`, direct inference can still run. If features are missing, retry with internet access.

No features found:

```text
No features found for YYYY-MM-DD
```

Check whether the date folder exists:

```bash
find data/eod_batch/staging -maxdepth 2 -name 'features.parquet' | sort
```

If the folder exists but predictions are wrong date, rerun:

```bash
ML_INFERENCE_AUTO_REFRESH=true \
US_STOCK_EOD_DATA_DIR="$PWD/data/eod_batch" \
.venv/bin/python - <<'PY'
from datetime import date
from streamlit_app.services.ml_inference_api import ensure_latest_ml_inference

a = ensure_latest_ml_inference(today=date(2026, 6, 8))
print(a)
PY
```
