"""
ORCA Agent Decision Backtest
────────────────────────────
Chạy ML model (Model A + Model C) retrospectively trên historical price data,
sau đó so sánh với giá thực tế để đánh giá chất lượng decision.

Metrics:
  • Direction accuracy : Pred_A > 0 → giá thực tế có tăng?
  • Risk_Prob precision : Risk_Prob_pct cao → có thực sự sụt ≥6%?
  • FinalScore IC       : Information Coefficient giữa FinalScore rank vs actual return rank
  • Simulated portfolio : Buy top-K by FinalScore, rebalance monthly, vs SPY buy-and-hold

Usage:
  conda run -n base python backtest_orca.py --symbols AAPL,MSFT,NVDA,TSLA,GOOGL \
      --start 2024-01-01 --end 2025-12-31 --horizon 20 --top-k 3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ── inject feature contract ─────────────────────────────────────────────────
for p in [
    Path("/opt/airflow/spark_jobs"),
    Path(__file__).parent / "spark_jobs",
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ml_features import PRICE_FEATURE_COLUMNS, compute_price_features

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "JPM", "V", "JNJ", "NFLX"]
MODEL_A_PATH    = Path("/tmp/orca_models/model_a.joblib")
MODEL_C_PATH    = Path("/tmp/orca_models/model_c.joblib")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV via yfinance, return long-format DataFrame."""
    all_tickers = list(set(symbols + ["SPY"]))
    print(f"  Downloading {len(all_tickers)} tickers from yfinance ({start} → {end})...")
    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    # yfinance multi-ticker → MultiIndex columns
    frames = []
    for sym in all_tickers:
        try:
            df = raw.xs(sym, axis=1, level=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df["Symbol"] = sym
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Datetime"
            df = df.reset_index()
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING + ML INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def build_predictions(
    prices: pd.DataFrame,
    symbols: list[str],
    model_a,
    model_c,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """For each rebalance date, compute features and run models."""

    spy_prices = prices[prices["Symbol"] == "SPY"].copy()
    target_prices = prices[prices["Symbol"].isin(symbols)].copy()

    results = []
    for rd in rebalance_dates:
        # Slice history up to rebalance date (lookback 260 trading days needed)
        cutoff = rd
        hist = target_prices[target_prices["Datetime"] <= cutoff].copy()
        spy_hist = spy_prices[spy_prices["Datetime"] <= cutoff].copy()

        if hist.empty or spy_hist.empty:
            continue

        try:
            features_df = compute_price_features(hist, spy_hist, drop_incomplete=True)
        except Exception as e:
            print(f"  Feature error on {rd.date()}: {e}")
            continue

        # Take latest row per symbol (as-of rebalance date)
        latest = (
            features_df
            .sort_values("Datetime")
            .groupby("Symbol")
            .last()
            .reset_index()
        )
        if latest.empty:
            continue

        X = latest[PRICE_FEATURE_COLUMNS].fillna(0).values
        valid_mask = np.isfinite(X).all(axis=1)
        if valid_mask.sum() == 0:
            continue

        latest_valid = latest[valid_mask].copy()
        X_valid_df = latest[valid_mask][PRICE_FEATURE_COLUMNS].fillna(0).astype(float)

        # Model A: XGBRegressor → raw upside return (e.g. 0.03 = +3%)
        pred_a_raw = np.asarray(model_a.predict(X_valid_df), dtype=float)

        # Model C: XGBClassifier → risk probability (P(drop ≥6%))
        if model_c is not None:
            try:
                risk_prob = np.asarray(model_c.predict_proba(X_valid_df)[:, 1], dtype=float)
            except AttributeError:
                risk_prob = np.clip(np.asarray(model_c.predict(X_valid_df), dtype=float), 0, 1)
        else:
            risk_prob = np.full(len(pred_a_raw), 0.5)

        pred_a = pred_a_raw  # raw return used for direction: pred_a > 0 → bullish

        # FinalScore = pred_a * (1 - risk_prob) — same formula as pipeline
        final_score = pred_a * (1.0 - risk_prob)

        latest_valid = latest_valid.copy()
        latest_valid["rebalance_date"] = rd
        latest_valid["pred_a"]         = pred_a
        latest_valid["risk_prob"]       = risk_prob
        latest_valid["final_score"]     = final_score

        # Entry price = last close on rebalance date
        entry = (
            target_prices[
                (target_prices["Symbol"].isin(latest_valid["Symbol"].values))
                & (target_prices["Datetime"] <= cutoff)
            ]
            .sort_values("Datetime")
            .groupby("Symbol")["Close"]
            .last()
        )
        latest_valid["entry_price"] = latest_valid["Symbol"].map(entry)
        results.append(latest_valid)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ACTUAL RETURN LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

def attach_actual_returns(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Attach actual T+N returns for each horizon."""
    price_pivot = (
        prices[prices["Symbol"] != "SPY"]
        .sort_values("Datetime")
        .pivot_table(index="Datetime", columns="Symbol", values="Close")
        .sort_index()
    )

    for h in horizons:
        col = f"actual_ret_{h}d"
        predictions[col] = np.nan

    for i, row in predictions.iterrows():
        sym   = row["Symbol"]
        rd    = row["rebalance_date"]
        entry = row["entry_price"]

        if sym not in price_pivot.columns or np.isnan(entry) or entry == 0:
            continue

        sym_prices = price_pivot[sym].dropna()
        future = sym_prices[sym_prices.index > rd]

        for h in horizons:
            trading_days_after = future.iloc[h - 1] if len(future) >= h else np.nan
            if not np.isnan(trading_days_after):
                predictions.at[i, f"actual_ret_{h}d"] = (trading_days_after - entry) / entry

    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame, horizons: list[int]) -> dict:
    metrics = {}
    for h in horizons:
        col = f"actual_ret_{h}d"
        sub = df[df[col].notna()].copy()
        if sub.empty:
            continue

        # Direction accuracy: pred_a > 0 (regressor) → actual return > 0 ?
        pred_up    = sub["pred_a"] > 0
        actual_up  = sub[col] > 0
        dir_acc    = (pred_up == actual_up).mean()

        # Risk precision: risk_prob > 0.5 → actual drop ≥ 6% ?
        high_risk  = sub["risk_prob"] > 0.5
        actual_drop = sub[col] <= -0.06
        risk_prec  = (high_risk & actual_drop).sum() / max(high_risk.sum(), 1)
        risk_rec   = (high_risk & actual_drop).sum() / max(actual_drop.sum(), 1)

        # Information Coefficient (Spearman rank correlation)
        ic = sub["final_score"].corr(sub[col], method="spearman")

        # Average return when FinalScore in top-3 vs bottom-3
        sub_ranked  = sub.sort_values(["rebalance_date", "final_score"], ascending=[True, False])
        top3_ret    = sub_ranked.groupby("rebalance_date").head(3)[col].mean()
        bottom3_ret = sub_ranked.groupby("rebalance_date").tail(3)[col].mean()

        metrics[h] = {
            "horizon_days"   : h,
            "sample_count"   : len(sub),
            "direction_acc"  : round(dir_acc * 100, 1),
            "risk_precision" : round(risk_prec * 100, 1),
            "risk_recall"    : round(risk_rec * 100, 1),
            "IC_spearman"    : round(ic, 4) if not np.isnan(ic) else None,
            "top3_avg_ret_pct"   : round(top3_ret * 100, 2),
            "bottom3_avg_ret_pct": round(bottom3_ret * 100, 2),
        }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 5. PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_portfolio(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    top_k: int,
    horizon: int,
) -> pd.DataFrame:
    """Monthly-rebalance: buy top-K by FinalScore, hold horizon trading days."""
    col = f"actual_ret_{horizon}d"
    sub = predictions[predictions[col].notna()].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for rd, grp in sub.groupby("rebalance_date"):
        ranked = grp.sort_values("final_score", ascending=False)
        top_k_picks = ranked.head(top_k)
        portfolio_ret = top_k_picks[col].mean()

        # SPY return for same period
        spy_prices = prices[prices["Symbol"] == "SPY"].sort_values("Datetime")
        spy_at_entry  = spy_prices[spy_prices["Datetime"] <= rd]["Close"].iloc[-1] if not spy_prices[spy_prices["Datetime"] <= rd].empty else np.nan
        spy_future    = spy_prices[spy_prices["Datetime"] > rd]["Close"]
        spy_exit      = spy_future.iloc[horizon - 1] if len(spy_future) >= horizon else np.nan
        spy_ret       = (spy_exit - spy_at_entry) / spy_at_entry if not np.isnan(spy_at_entry) and spy_at_entry != 0 else np.nan

        rows.append({
            "rebalance_date" : rd,
            "picks"          : ", ".join(top_k_picks["Symbol"].tolist()),
            "portfolio_ret"  : round(portfolio_ret * 100, 2),
            "spy_ret"        : round(spy_ret * 100, 2) if not np.isnan(spy_ret) else np.nan,
            "alpha"          : round((portfolio_ret - spy_ret) * 100, 2) if not np.isnan(spy_ret) else np.nan,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ORCA ML model backtest")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start",   default="2024-01-01")
    parser.add_argument("--end",     default="2025-12-31")
    parser.add_argument("--horizons",default="5,10,20", help="Comma-separated hold periods (trading days)")
    parser.add_argument("--top-k",   default=3, type=int)
    parser.add_argument("--model-a", default=str(MODEL_A_PATH))
    parser.add_argument("--model-c", default=str(MODEL_C_PATH))
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    symbols  = [s.strip().upper() for s in args.symbols.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]

    print("=" * 60)
    print("ORCA ML Backtest")
    print(f"  Symbols : {symbols}")
    print(f"  Period  : {args.start} → {args.end}")
    print(f"  Horizons: {horizons} trading days")
    print(f"  Top-K   : {args.top_k}")
    print("=" * 60)

    # Load models
    print("\n[1/5] Loading models...")
    def _load_artifact(path):
        """Load model artifact — may be a dict wrapper or raw model."""
        art = joblib.load(path)
        if isinstance(art, dict):
            return art  # dict with 'model' key
        return {"model": art}

    model_a_art = _load_artifact(args.model_a)
    model_a = model_a_art["model"]
    model_c_path = Path(args.model_c)
    if model_c_path.exists():
        model_c_art = _load_artifact(str(model_c_path))
        model_c = model_c_art["model"]
    else:
        model_c = None
    print(f"  Model A: {Path(args.model_a).name}  ({model_a.__class__.__name__})")
    print(f"  Model C: {model_c_path.name if model_c else 'NOT FOUND (using 0.5 fallback)'}  ({model_c.__class__.__name__ if model_c else '-'})")

    # Load price data
    print("\n[2/5] Loading price data...")
    prices = load_prices(symbols, args.start, args.end)
    print(f"  Loaded {len(prices)} rows, {prices['Symbol'].nunique()} symbols")
    print(f"  Date range: {prices['Datetime'].min().date()} → {prices['Datetime'].max().date()}")

    # Monthly rebalance dates (first trading day of each month)
    all_dates = prices["Datetime"].drop_duplicates().sort_values()
    monthly = all_dates.groupby(all_dates.dt.to_period("M")).first().tolist()
    # Filter: need at least 260 days of history, and leave room for last horizon
    first_valid = all_dates.iloc[259] if len(all_dates) > 259 else all_dates.iloc[-1]
    last_valid  = all_dates.iloc[-max(horizons) - 1] if len(all_dates) > max(horizons) else all_dates.iloc[-1]
    rebalance_dates = [d for d in monthly if first_valid <= d <= last_valid]
    print(f"  Rebalance dates: {len(rebalance_dates)} months")

    # Build predictions
    print("\n[3/5] Running ML inference on historical dates...")
    predictions = build_predictions(prices, symbols, model_a, model_c, rebalance_dates)
    print(f"  Generated {len(predictions)} prediction rows across {predictions['rebalance_date'].nunique()} dates")

    # Attach actual returns
    print("\n[4/5] Attaching actual returns...")
    predictions = attach_actual_returns(predictions, prices, horizons)

    # Save CSV if requested
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions[[
            "rebalance_date", "Symbol", "entry_price",
            "pred_a", "risk_prob", "final_score",
            *[f"actual_ret_{h}d" for h in horizons],
        ]].round(6).to_csv(out_path, index=False)
        print(f"  Saved predictions to {out_path}")

    # Compute metrics
    print("\n[5/5] Computing metrics...")
    metrics = compute_metrics(predictions, horizons)

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    for h, m in metrics.items():
        print(f"\n  ── T+{h}d horizon ({m['sample_count']} samples) ──")
        print(f"    Direction Accuracy  : {m['direction_acc']}%   (Pred_A > 0.5 → actual > 0)")
        print(f"    Risk Precision      : {m['risk_precision']}%   (high risk_prob → drop ≥6%)")
        print(f"    Risk Recall         : {m['risk_recall']}%   (actual drop ≥6% flagged)")
        print(f"    IC (Spearman)       : {m['IC_spearman']}   (FinalScore rank vs actual return rank)")
        print(f"    Top-3 avg return    : {m['top3_avg_ret_pct']}%")
        print(f"    Bottom-3 avg return : {m['bottom3_avg_ret_pct']}%")

    # Portfolio simulation
    print("\n" + "=" * 60)
    print(f"PORTFOLIO SIMULATION  (top-{args.top_k} by FinalScore, T+{horizons[-1]}d hold)")
    print("=" * 60)
    portfolio = simulate_portfolio(predictions, prices, args.top_k, horizons[-1])
    if not portfolio.empty:
        portfolio_mean = portfolio["portfolio_ret"].mean()
        spy_mean       = portfolio["spy_ret"].mean()
        alpha_mean     = portfolio["alpha"].mean()
        win_rate       = (portfolio["alpha"] > 0).mean() * 100

        print(f"\n  Avg portfolio return : {portfolio_mean:.2f}%")
        print(f"  Avg SPY return       : {spy_mean:.2f}%")
        print(f"  Avg alpha vs SPY     : {alpha_mean:+.2f}%")
        print(f"  Win rate vs SPY      : {win_rate:.1f}%  ({(portfolio['alpha']>0).sum()}/{len(portfolio)} months)")

        print(f"\n  {'Date':<12} {'Picks':<35} {'Port%':>6} {'SPY%':>6} {'Alpha':>7}")
        print("  " + "-" * 70)
        for _, row in portfolio.iterrows():
            date_str = str(row["rebalance_date"])[:10]
            alpha_str = f"{row['alpha']:+.2f}%" if not np.isnan(row["alpha"]) else "  N/A"
            spy_str   = f"{row['spy_ret']:.2f}%" if not np.isnan(row["spy_ret"]) else "  N/A"
            print(f"  {date_str:<12} {row['picks']:<35} {row['portfolio_ret']:>5.2f}%  {spy_str:>6}  {alpha_str:>7}")

        # Cumulative return
        valid = portfolio.dropna(subset=["portfolio_ret", "spy_ret"])
        if len(valid) > 1:
            cum_port = (1 + valid["portfolio_ret"] / 100).prod() - 1
            cum_spy  = (1 + valid["spy_ret"] / 100).prod() - 1
            print(f"\n  Cumulative portfolio : {cum_port*100:+.2f}%")
            print(f"  Cumulative SPY       : {cum_spy*100:+.2f}%")
            print(f"  Total alpha          : {(cum_port - cum_spy)*100:+.2f}%")

    print("\n" + "=" * 60)
    print("TOP PREDICTIONS (last rebalance date, by FinalScore)")
    print("=" * 60)
    last_rd = predictions["rebalance_date"].max()
    last_preds = predictions[predictions["rebalance_date"] == last_rd].sort_values("final_score", ascending=False)
    print(f"\n  As of: {last_rd.date()}")
    print(f"  {'Symbol':<8} {'Pred_A%':>8} {'Risk%':>7} {'FinalScore':>11} ", end="")
    for h in horizons:
        print(f"  {'T+'+str(h)+'d':>7}", end="")
    print()
    print("  " + "-" * (40 + 9 * len(horizons)))
    for _, row in last_preds.iterrows():
        ret_parts = ""
        for h in horizons:
            v = row.get(f"actual_ret_{h}d", np.nan)
            ret_parts += f"  {v*100:>6.2f}%" if not np.isnan(v) else "     N/A"
        print(
            f"  {row['Symbol']:<8} {row['pred_a']*100:>7.2f}%  {row['risk_prob']*100:>6.2f}%  {row['final_score']:>11.4f}"
            + ret_parts
        )

    print()


if __name__ == "__main__":
    main()
