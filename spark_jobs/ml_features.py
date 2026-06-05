from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    import ta
except ImportError:  # pragma: no cover - production image should install ta
    ta = None


LOOKBACKS = [1, 3, 5, 10, 14, 20, 21]

PRICE_FEATURE_COLUMNS = [
    "r1",
    "r3",
    "r5",
    "r10",
    "r14",
    "r20",
    "r21",
    "cumret_20",
    "return_z_20",
    "gap_return",
    "CLV",
    "EMA20_50_spread",
    "EMA50_200_spread",
    "EMA20_slope",
    "EMA50_slope",
    "MACD_hist",
    "RSI14",
    "ROC10",
    "ROC20",
    "ADX14",
    "dist_20d_high",
    "dist_55d_high",
    "dist_52w_high",
    "BB_pctB",
    "BB_width",
    "dist_ema20",
    "dist_ema50",
    "ATR14_ratio",
    "vol20",
    "downside_vol20",
    "maxdd20",
    "realized_vol_10",
    "realized_vol_20",
    "vol_ratio_20_60",
    "true_range_zscore",
    "RVOL20",
    "OBV_slope",
    "MFI14",
    "dollar_volume_log",
    "volume_slope_20",
    "OBV_slope_neg",
    "RS_vs_SPY_14",
    "RS_vs_SPY_30",
    "beta_60D",
    "SPY_above_EMA50",
    "SPY_20d_return",
    "sector_percentile_20d",
]

ORCA_CONTEXT_COLUMNS = ["Close", "maxdd90"]

NEWS_FEATURE_COLUMNS = ["S_it", "N_it", "delta_t", "delta_S", "V", "A", "Z", "M"]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _macd_hist(close: pd.Series) -> pd.Series:
    if ta is not None:
        return ta.trend.macd_diff(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    if ta is not None:
        return ta.momentum.rsi(close, window=window)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _roc(close: pd.Series, window: int) -> pd.Series:
    if ta is not None:
        return ta.momentum.roc(close, window=window)
    return close.pct_change(window) * 100


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    if ta is not None:
        return ta.trend.adx(high, low, close, window=window)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    if ta is not None:
        return ta.volatility.average_true_range(high, low, close, window=window)
    return _true_range(high, low, close).rolling(window).mean()


def _bollinger_pband(close: pd.Series, window: int = 20, window_dev: int = 2) -> pd.Series:
    if ta is not None:
        return ta.volatility.BollingerBands(close, window=window, window_dev=window_dev).bollinger_pband()
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - (mean - window_dev * std)) / (2 * window_dev * std).replace(0, np.nan)


def _bollinger_width(close: pd.Series, window: int = 20, window_dev: int = 2) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return 2 * window_dev * std


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if ta is not None:
        return ta.volume.on_balance_volume(close, volume)
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    if ta is not None:
        return ta.volume.money_flow_index(high, low, close, volume, window=window)
    typical = (high + low + close) / 3
    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(1), 0).rolling(window).sum()
    negative = money_flow.where(typical < typical.shift(1), 0).rolling(window).sum()
    ratio = positive / negative.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def compute_price_features(
    price_history: pd.DataFrame,
    spy_history: pd.DataFrame | pd.Series,
    *,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build the same price feature set used by bd-ml-filtering.ipynb.

    Input rows must contain Datetime, Symbol, Open, High, Low, Close, and Volume.
    The returned frame keeps Datetime/Symbol identifiers plus feature columns.
    """
    if price_history.empty:
        return pd.DataFrame(columns=["Datetime", "Symbol", *PRICE_FEATURE_COLUMNS, *ORCA_CONTEXT_COLUMNS])

    price_history = _normalize_price_history_columns(price_history)
    spy_close = _daily_spy_close(spy_history)
    outputs = []

    for symbol, group in price_history.groupby("Symbol", sort=False):
        group = _daily_ohlcv(group)

        close = pd.to_numeric(group["Close"], errors="coerce")
        high = pd.to_numeric(group["High"], errors="coerce")
        low = pd.to_numeric(group["Low"], errors="coerce")
        volume = pd.to_numeric(group["Volume"], errors="coerce")
        aligned_spy = spy_close.reindex(group.index).ffill()

        feat = _build_one_symbol_features(close, high, low, volume, aligned_spy)
        feat.insert(0, "Symbol", symbol)
        feat.insert(0, "Datetime", feat.index)
        outputs.append(feat.reset_index(drop=True))

    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    result = result.replace([np.inf, -np.inf], np.nan)
    if drop_incomplete:
        result = result.dropna(subset=PRICE_FEATURE_COLUMNS)
    return result[["Datetime", "Symbol", *PRICE_FEATURE_COLUMNS, *ORCA_CONTEXT_COLUMNS]]


def _normalize_price_history_columns(price_history: pd.DataFrame) -> pd.DataFrame:
    """Accept Spark/Pandas case drift while keeping the feature contract stable."""
    frame = price_history.copy()
    if "Datetime" not in frame.columns and str(frame.index.name).lower() == "datetime":
        frame = frame.reset_index()

    canonical = ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    frame = _rename_columns_case_insensitive(frame, canonical)

    missing = [name for name in canonical if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing price history columns: {missing}. Available columns: {list(price_history.columns)}")

    return frame


def _rename_columns_case_insensitive(frame: pd.DataFrame, canonical: Iterable[str]) -> pd.DataFrame:
    by_lower = {str(column).lower(): column for column in frame.columns}
    rename_map = {
        by_lower[name.lower()]: name
        for name in canonical
        if name not in frame.columns and name.lower() in by_lower
    }
    return frame.rename(columns=rename_map) if rename_map else frame


def _daily_ohlcv(group: pd.DataFrame) -> pd.DataFrame:
    """Collapse intraday market rows to one trading-day OHLCV bar."""
    group = _normalize_price_history_columns(group)
    group = group.sort_values("Datetime").drop_duplicates(["Datetime"], keep="last").copy()
    group["Datetime"] = pd.to_datetime(group["Datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    group = group.dropna(subset=["Datetime"]).set_index("Datetime")
    symbol = group["Symbol"].iloc[-1] if "Symbol" in group.columns and not group.empty else None
    daily = (
        group.resample("1D")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    if symbol is not None:
        daily["Symbol"] = symbol
    return daily


def _daily_spy_close(spy_history: pd.DataFrame | pd.Series) -> pd.Series:
    """Return SPY daily close from either daily or intraday input."""
    if isinstance(spy_history, pd.DataFrame):
        spy_history = _rename_columns_case_insensitive(spy_history, ["Datetime", "Close"])
        if "Datetime" in spy_history.columns:
            spy = spy_history.copy()
            spy["Datetime"] = pd.to_datetime(spy["Datetime"], utc=True, errors="coerce").dt.tz_localize(None)
            spy = spy.dropna(subset=["Datetime"]).sort_values("Datetime").set_index("Datetime")
            close = pd.to_numeric(spy["Close"], errors="coerce")
        else:
            close = pd.to_numeric(spy_history["Close"], errors="coerce")
            close.index = pd.to_datetime(close.index, utc=True, errors="coerce").tz_localize(None)
    else:
        close = pd.to_numeric(spy_history.copy(), errors="coerce")
        close.index = pd.to_datetime(close.index, utc=True, errors="coerce").tz_localize(None)
    return close.resample("1D").last().dropna()


def _build_one_symbol_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    spy_close: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df["Close"] = close

    for lb in LOOKBACKS:
        df[f"r{lb}"] = close.pct_change(lb)
    df["cumret_20"] = close.pct_change(20)
    roll_std_20 = df["r20"].rolling(20).std().replace(0, np.nan)
    df["return_z_20"] = (df["r20"] - df["r20"].rolling(20).mean()) / roll_std_20
    df["gap_return"] = close.pct_change(1)

    df["CLV"] = _safe_div(close - low, high - low) - 0.5

    # Match the training notebook, which used the pandas default `adjust=True`.
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    close_safe = close.replace(0, np.nan)
    df["EMA20_50_spread"] = (ema20 - ema50) / close_safe
    df["EMA50_200_spread"] = (ema50 - ema200) / close_safe
    df["EMA20_slope"] = ema20.diff(5) / 5
    df["EMA50_slope"] = ema50.diff(5) / 5
    df["MACD_hist"] = _macd_hist(close)
    df["RSI14"] = _rsi(close, window=14)
    df["ROC10"] = _roc(close, window=10)
    df["ROC20"] = _roc(close, window=20)
    df["ADX14"] = _adx(high, low, close, window=14)

    df["dist_20d_high"] = (high.rolling(20).max() - close) / close_safe
    df["dist_55d_high"] = (high.rolling(55).max() - close) / close_safe
    df["dist_52w_high"] = (high.rolling(252).max() - close) / close_safe
    df["BB_pctB"] = _bollinger_pband(close, window=20, window_dev=2)
    df["BB_width"] = _bollinger_width(close, window=20, window_dev=2) / close_safe
    df["dist_ema20"] = (close - ema20) / close_safe
    df["dist_ema50"] = (close - ema50) / close_safe

    returns = close.pct_change()
    atr14_ratio = _atr(high, low, close, window=14) / close_safe
    df["ATR14_ratio"] = atr14_ratio
    df["vol20"] = returns.rolling(20).std()
    df["downside_vol20"] = returns.where(returns < 0, 0).rolling(20).std()
    cummax = close.expanding().max().replace(0, np.nan)
    df["maxdd20"] = ((close - cummax) / cummax).rolling(20).min() * (-1)
    df["maxdd90"] = ((close - cummax) / cummax).rolling(90).min() * (-1)
    df["realized_vol_10"] = returns.rolling(10).std()
    df["realized_vol_20"] = returns.rolling(20).std()
    df["vol_ratio_20_60"] = df["realized_vol_20"] / returns.rolling(60).std().replace(0, np.nan)
    df["true_range_zscore"] = (atr14_ratio - atr14_ratio.rolling(20).mean()) / atr14_ratio.rolling(20).std()

    avg_vol = volume.rolling(20).mean().replace(0, np.nan)
    df["RVOL20"] = volume / avg_vol
    df["OBV_slope"] = _obv(close, volume).diff(5) / 5
    df["MFI14"] = _mfi(high, low, close, volume, window=14)
    df["dollar_volume_log"] = np.log((close * volume).replace(0, 1))
    df["volume_slope_20"] = volume.diff(5) / 5
    df["OBV_slope_neg"] = (df["OBV_slope"] < 0).astype(int)

    rel_strength = close / spy_close.replace(0, np.nan)
    spy_returns = spy_close.pct_change()
    df["RS_vs_SPY_14"] = rel_strength.pct_change(14)
    df["RS_vs_SPY_30"] = rel_strength.pct_change(30)
    df["beta_60D"] = returns.rolling(60).cov(spy_returns) / spy_returns.rolling(60).var().replace(0, np.nan)
    df["SPY_above_EMA50"] = (spy_close > spy_close.ewm(span=50).mean()).astype(int)
    df["SPY_20d_return"] = spy_close.pct_change(20)
    df["sector_percentile_20d"] = 0.5

    return df[[*PRICE_FEATURE_COLUMNS, *ORCA_CONTEXT_COLUMNS]]


# def compute_news_dynamics(news_history: pd.DataFrame, symbols: Iterable[str] | None = None) -> pd.DataFrame:
#     """Compute Model S news momentum features from Silver stock_news rows."""
#     if news_history.empty:
#         return pd.DataFrame(columns=["Symbol", "date", *NEWS_FEATURE_COLUMNS])

#     df = news_history.copy()
#     df["Symbol"] = df["Symbol"].astype(str).str.replace(".US", "", regex=False).str.strip()
#     if symbols is not None:
#         wanted = set(symbols)
#         df = df[df["Symbol"].isin(wanted)]
#     df["date"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_localize(None).dt.floor("D")
#     if "polarity" in df.columns:
#         df["polarity"] = pd.to_numeric(df["polarity"], errors="coerce").fillna(0)
#     else:
#         df["polarity"] = 0.0

#     daily_news = (
#         df.groupby(["Symbol", "date"])
#         .agg(S_it=("polarity", "mean"), N_it=("headline", "count"))
#         .reset_index()
#         .sort_values(["Symbol", "date"])
#     )

#     def calc_dynamics(group: pd.DataFrame) -> pd.DataFrame:
#         group = group.copy()
#         group["prev_date"] = group["date"].shift(1)
#         group["delta_t"] = (group["date"] - group["prev_date"]).dt.days.fillna(1).replace(0, 1)
#         group["S_prev"] = group["S_it"].shift(1).fillna(0)
#         group["delta_S"] = group["S_it"] - group["S_prev"]
#         group["V"] = group["delta_S"] / group["delta_t"]
#         group["V_prev"] = group["V"].shift(1).fillna(0)
#         group["A"] = group["V"] - group["V_prev"]
#         group["avg_N_20"] = group["N_it"].shift(1).rolling(window=20, min_periods=1).mean().fillna(1)
#         group["Z"] = group["N_it"] / group["avg_N_20"]
#         group["M"] = group["delta_S"] * np.log1p(group["Z"]) * (1 + group["A"])
#         return group

#     features = daily_news.groupby("Symbol", group_keys=False).apply(calc_dynamics).reset_index(drop=True)
#     return features[["Symbol", "date", *NEWS_FEATURE_COLUMNS]]
