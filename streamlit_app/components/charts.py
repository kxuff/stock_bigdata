from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_stock_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )
    if df.empty:
        fig.update_layout(template="plotly_dark", height=560, title=f"{symbol} price history unavailable")
        return fig

    colors = ["#22c55e" if close >= open_ else "#ef4444" for open_, close in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=df["datetime"], y=df["volume"], marker_color=colors, name="Volume", opacity=0.72), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=620,
        title=f"{symbol} Candlestick and Volume",
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080d16",
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.14)")
    fig.update_xaxes(showgrid=False)
    return fig
