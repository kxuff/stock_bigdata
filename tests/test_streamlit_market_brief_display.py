import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app.chat.components import _display_rows


def test_market_brief_display_rows_keep_only_user_friendly_columns() -> None:
    rows = [
        {
            "Symbol": "DOW",
            "Datetime": "2026-06-10 00:00:00",
            "process_date": "2026-06-11",
            "model_version": "model_a_v1.0",
            "final_score": 0.5221,
            "latest_price": 34.24001,
            "RSI14": 41.8688,
            "RVOL20": 1.0128,
            "risk_prob": 0.038,
        }
    ]

    display = _display_rows("top_stocks", rows)

    assert display == [
        {
            "Symbol": "DOW",
            "ORCA Score": 52.2,
            "Price": 34.24,
            "RSI14": 41.87,
            "RVOL20": 1.01,
            "Risk Prob": 0.04,
        }
    ]
