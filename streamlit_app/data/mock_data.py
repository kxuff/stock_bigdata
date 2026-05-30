from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


KPI_METRICS = [
    {"label": "Portfolio Signal", "value": "Bullish", "delta": "+7.4% conviction"},
    {"label": "Risk Regime", "value": "Moderate", "delta": "VIX -1.8 pts"},
    {"label": "AI Hit Rate", "value": "68%", "delta": "+4% 30d"},
    {"label": "Watchlist Alerts", "value": "12", "delta": "3 high priority"},
]

RECENT_DECISIONS = pd.DataFrame(
    [
        {"time": "09:35", "symbol": "NVDA", "action": "BUY", "confidence": 0.84, "reason": "AI momentum plus earnings revision"},
        {"time": "10:10", "symbol": "MSFT", "action": "HOLD", "confidence": 0.72, "reason": "Quality trend intact"},
        {"time": "11:20", "symbol": "TSLA", "action": "TRIM", "confidence": 0.67, "reason": "Volatility spike and weak breadth"},
        {"time": "13:45", "symbol": "LLY", "action": "BUY", "confidence": 0.79, "reason": "Defensive growth bid"},
    ]
)

TREND_DATA = pd.DataFrame(
    {
        "date": [date.today() - timedelta(days=days) for days in range(13, -1, -1)],
        "AI confidence": [58, 61, 59, 63, 66, 64, 68, 71, 69, 73, 76, 74, 78, 81],
        "Market breadth": [48, 50, 47, 51, 54, 55, 57, 56, 59, 60, 62, 61, 64, 66],
    }
)

TODAYS_BRIEF = [
    "Large-cap tech leads risk-on tape; semis show strongest relative strength.",
    "Credit spreads calm; macro risk remains focused on rate-cut timing.",
    "Model favors profitable growth and healthcare defensives over high-beta cyclicals.",
]

STOCK_PICKS = [
    {
        "symbol": "NVDA",
        "name": "NVIDIA Corp.",
        "sector": "Technology",
        "rating": "Strong Buy",
        "badge": "emerald",
        "score": 94,
        "target": "$1,180",
        "horizon": "2-6 weeks",
        "thesis": "Accelerating AI demand, strong revisions, and durable margin profile.",
        "risk": "Crowded positioning and valuation sensitivity.",
    },
    {
        "symbol": "MSFT",
        "name": "Microsoft Corp.",
        "sector": "Technology",
        "rating": "Buy",
        "badge": "cyan",
        "score": 88,
        "target": "$470",
        "horizon": "1-3 months",
        "thesis": "Cloud and Copilot monetization support quality compounder setup.",
        "risk": "Azure growth deceleration.",
    },
    {
        "symbol": "LLY",
        "name": "Eli Lilly",
        "sector": "Healthcare",
        "rating": "Buy",
        "badge": "emerald",
        "score": 86,
        "target": "$930",
        "horizon": "1-3 months",
        "thesis": "GLP-1 demand and defensive growth remain attractive.",
        "risk": "Supply constraints and policy headlines.",
    },
    {
        "symbol": "TSLA",
        "name": "Tesla Inc.",
        "sector": "Consumer Discretionary",
        "rating": "Watch",
        "badge": "amber",
        "score": 61,
        "target": "$185",
        "horizon": "1-4 weeks",
        "thesis": "Optionality high, but trend and estimate support are mixed.",
        "risk": "Delivery pressure and margin compression.",
    },
]

INITIAL_CHAT_MESSAGES = [
    {"role": "assistant", "content": "Market copilot ready. Ask about signals, risk, or stock picks."},
]


def picks_dataframe() -> pd.DataFrame:
    return pd.DataFrame(STOCK_PICKS)[["symbol", "name", "sector", "rating", "score", "target", "horizon"]]
