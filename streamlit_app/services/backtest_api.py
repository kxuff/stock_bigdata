from __future__ import annotations

import sys
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from services import ml_inference_api


DEFAULT_MIN_PRED_A = float(getattr(ml_inference_api, "MIN_PRED_A", 0.06))
DEFAULT_MAX_RISK_PROB = float(getattr(ml_inference_api, "MAX_RISK_PROB", 0.3))


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models"
DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "TSLA",
]
MARKET_SYMBOL = "SPY"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spark_jobs.ml_features import PRICE_FEATURE_COLUMNS, compute_price_features  # noqa: E402


@dataclass(frozen=True)
class StrategyBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, Any]
    exit_reason_counts: pd.Series


def run_strategy_backtest(
    *,
    test_start: str,
    end_date: str,
    symbols: list[str] | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    min_pred_a: float = DEFAULT_MIN_PRED_A,
    max_risk_prob: float = DEFAULT_MAX_RISK_PROB,
    stop_loss: float = 0.06,
    horizon_days: int = 14,
) -> StrategyBacktestResult:
    tickers = _clean_tickers(symbols or DEFAULT_SYMBOLS)
    if MARKET_SYMBOL not in tickers:
        tickers = [*tickers, MARKET_SYMBOL]

    artifact_a = joblib.load(model_dir / "model_a.joblib")
    artifact_c = joblib.load(model_dir / "model_c.joblib")
    model_a = artifact_a["model"]
    model_c = artifact_c["model"]
    feature_columns = list(artifact_a.get("feature_columns") or PRICE_FEATURE_COLUMNS)

    start_date = (pd.Timestamp(test_start) - pd.DateOffset(days=420)).strftime("%Y-%m-%d")
    raw_prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw_prices.empty:
        raise ValueError("No price data returned from yfinance.")
    if MARKET_SYMBOL not in raw_prices.columns.get_level_values(0):
        raise ValueError(f"Missing {MARKET_SYMBOL} market benchmark data.")

    price_history = _multiindex_prices_to_rows(raw_prices, [ticker for ticker in tickers if ticker != MARKET_SYMBOL])
    spy_close = _symbol_frame(raw_prices, MARKET_SYMBOL)["Close"].dropna()
    prices_by_symbol = {
        ticker: _symbol_frame(raw_prices, ticker).dropna(subset=["Close"])
        for ticker in tickers
        if ticker != MARKET_SYMBOL and ticker in raw_prices.columns.get_level_values(0)
    }

    features = compute_price_features(price_history, spy_close, drop_incomplete=True)
    features = features[pd.to_datetime(features["Datetime"]) >= pd.Timestamp(test_start)].copy()
    if features.empty:
        raise ValueError("Feature engineering produced no rows in the selected test window.")

    x = features.reindex(columns=feature_columns).astype(float)
    features["Pred_A"] = np.asarray(model_a.predict(x), dtype=float)
    features["Risk_Prob"] = _predict_risk(model_c, x)
    features["FinalScore"] = features["Pred_A"] * (1 - features["Risk_Prob"])

    signals = features[
        (features["Pred_A"] >= min_pred_a)
        & (features["Risk_Prob"] <= max_risk_prob)
    ].sort_values(["Datetime", "FinalScore"], ascending=[True, False])

    trades = _execute_trades(
        signals=signals,
        prices_by_symbol=prices_by_symbol,
        stop_loss=stop_loss,
        horizon_days=horizon_days,
    )
    equity_curve = _build_equity_curve(trades, spy_close, test_start, end_date)
    metrics = _build_metrics(trades, equity_curve)
    exit_counts = trades["Exit_Reason"].value_counts() if not trades.empty else pd.Series(dtype=int)
    return StrategyBacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics, exit_reason_counts=exit_counts)


def build_equity_figure(equity_curve: pd.DataFrame, *, min_pred_a: float, max_risk_prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not equity_curve.empty:
        ax.plot(
            equity_curve.index,
            equity_curve["Strategy"] * 100,
            label=f"AI Strategy (Pred >= {min_pred_a:.0%} | Risk <= {max_risk_prob:.0%})",
            color="#2563eb",
            linewidth=2.4,
        )
        ax.plot(
            equity_curve.index,
            equity_curve["SPY"] * 100,
            label="SPY Benchmark",
            color="#f97316",
            linestyle="--",
            linewidth=2,
        )
    ax.set_title("Cumulative Return: AI Strategy vs SPY", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", frameon=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    fig.tight_layout()
    return fig


def _clean_tickers(symbols: list[str]) -> list[str]:
    return sorted({str(symbol).strip().upper().replace(".", "-") for symbol in symbols if str(symbol).strip()})


def _symbol_frame(raw_prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(raw_prices.columns, pd.MultiIndex):
        if symbol not in raw_prices.columns.get_level_values(0):
            return pd.DataFrame()
        frame = raw_prices[symbol].copy()
    else:
        frame = raw_prices.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.sort_index()


def _multiindex_prices_to_rows(raw_prices: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    rows = []
    available = set(raw_prices.columns.get_level_values(0)) if isinstance(raw_prices.columns, pd.MultiIndex) else set(symbols)
    for symbol in symbols:
        if symbol not in available:
            continue
        frame = _symbol_frame(raw_prices, symbol)
        required = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in frame.columns]
        if len(required) < 5:
            continue
        out = frame[required].dropna(subset=["Open", "High", "Low", "Close"]).copy()
        out.insert(0, "Symbol", symbol)
        out.insert(0, "Datetime", out.index)
        rows.append(out.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(columns=["Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(rows, ignore_index=True)


def _predict_risk(model_c: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model_c, "predict_proba"):
        return np.asarray(model_c.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model_c.predict(x), dtype=float)


def _execute_trades(
    *,
    signals: pd.DataFrame,
    prices_by_symbol: dict[str, pd.DataFrame],
    stop_loss: float,
    horizon_days: int,
) -> pd.DataFrame:
    executed = []
    if signals.empty:
        return _empty_trades_frame()

    for _, signal in signals.iterrows():
        date = pd.Timestamp(signal["Datetime"]).normalize()
        symbol = str(signal["Symbol"])
        price_frame = prices_by_symbol.get(symbol)
        if price_frame is None or date not in price_frame.index:
            continue

        entry_price = float(price_frame.loc[date, "Close"])
        target_upside = float(signal["Pred_A"])
        future_window = price_frame.loc[date:].iloc[1 : horizon_days + 1]
        if future_window.empty:
            continue

        trade_return = None
        exit_date = None
        exit_reason = f"Time Exit ({horizon_days}D)"

        for future_date, future_row in future_window.iterrows():
            if future_row["Low"] <= entry_price * (1 - stop_loss):
                trade_return = -stop_loss
                exit_date = future_date
                exit_reason = "Risk Hit (SL)"
                break
            if future_row["High"] >= entry_price * (1 + target_upside):
                trade_return = target_upside
                exit_date = future_date
                exit_reason = "Target Hit (TP)"
                break

        if trade_return is None:
            final_row = future_window.iloc[-1]
            trade_return = (float(final_row["Close"]) - entry_price) / entry_price
            exit_date = future_window.index[-1]

        executed.append(
            {
                "Entry_Date": date.date(),
                "Ticker": symbol,
                "Pred_A_Target": round(target_upside * 100, 2),
                "Risk_Prob_pct": round(float(signal["Risk_Prob"]) * 100, 2),
                "Entry_Price": round(entry_price, 2),
                "Exit_Date": pd.Timestamp(exit_date).date(),
                "Return_pct": round(trade_return * 100, 2),
                "Exit_Reason": exit_reason,
            }
        )

    if not executed:
        return _empty_trades_frame()
    return pd.DataFrame(executed).sort_values(["Entry_Date", "Ticker"]).reset_index(drop=True)


def _build_equity_curve(df_trades: pd.DataFrame, spy_close: pd.Series, test_start: str, end_date: str) -> pd.DataFrame:
    spy_slice = spy_close.loc[test_start:end_date].dropna()
    if spy_slice.empty:
        return pd.DataFrame(columns=["Strategy", "SPY"])

    trading_days = spy_slice.index
    portfolio_daily_returns = pd.Series(0.0, index=trading_days)
    if not df_trades.empty:
        trades = df_trades.copy()
        trades["Entry_Date"] = pd.to_datetime(trades["Entry_Date"])
        trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"])

        daily_slices: dict[pd.Timestamp, list[float]] = {day: [] for day in trading_days}
        for _, trade in trades.iterrows():
            entry_date = pd.Timestamp(trade["Entry_Date"]).normalize()
            exit_date = pd.Timestamp(trade["Exit_Date"]).normalize()
            active_days = trading_days[(trading_days > entry_date) & (trading_days <= exit_date)]
            if active_days.empty:
                active_days = trading_days[trading_days == exit_date]
            if active_days.empty:
                continue

            daily_return = (float(trade["Return_pct"]) / 100) / len(active_days)
            for active_day in active_days:
                daily_slices[active_day].append(daily_return)

        for day, slices in daily_slices.items():
            if slices:
                portfolio_daily_returns.loc[day] = float(np.mean(slices))

    strategy_equity = (1 + portfolio_daily_returns).cumprod() - 1
    spy_equity = (1 + spy_slice.pct_change().fillna(0)).cumprod() - 1
    return pd.DataFrame({"Strategy": strategy_equity, "SPY": spy_equity}, index=trading_days)


def _build_metrics(df_trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, Any]:
    strategy_total_return = 0.0
    spy_total_return = np.nan
    if not equity_curve.empty:
        strategy_total_return = float(equity_curve["Strategy"].iloc[-1] * 100)
        spy_total_return = float(equity_curve["SPY"].iloc[-1] * 100)

    if df_trades.empty:
        return {
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_return_per_trade": np.nan,
            "strategy_total_return": strategy_total_return,
            "raw_trade_return_sum": 0.0,
            "spy_total_return": spy_total_return,
        }

    return {
        "trade_count": int(len(df_trades)),
        "win_rate": float((df_trades["Return_pct"] > 0).mean() * 100),
        "avg_return_per_trade": float(df_trades["Return_pct"].mean()),
        "strategy_total_return": strategy_total_return,
        "raw_trade_return_sum": float(df_trades["Return_pct"].sum()),
        "spy_total_return": spy_total_return,
    }


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Entry_Date",
            "Ticker",
            "Pred_A_Target",
            "Risk_Prob_pct",
            "Entry_Price",
            "Exit_Date",
            "Return_pct",
            "Exit_Reason",
        ]
    )
