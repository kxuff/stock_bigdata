from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #05070d;
            --border: rgba(148, 163, 184, 0.18);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #f59e0b;
            --blue: #38bdf8;
        }
        .stApp { background: linear-gradient(180deg, #05070d 0%, #0b1120 46%, #05070d 100%); color: var(--text); }
        [data-testid="stSidebar"] { background: #080d16; border-right: 1px solid var(--border); }
        .market-header {
            padding: 1.35rem 1.5rem; border: 1px solid var(--border); border-radius: 8px;
            background: linear-gradient(135deg, rgba(17,24,39,.96), rgba(8,13,22,.94)); margin-bottom: 1rem;
        }
        .market-header h1 { margin: 0; font-size: 2.15rem; font-weight: 900; letter-spacing: 0; }
        .market-header p { margin: .35rem 0 0; color: var(--muted); }
        .section-title { margin: 1.25rem 0 .65rem; font-size: 1.05rem; font-weight: 850; color: #f8fafc; }
        .kpi-card, .news-card {
            border: 1px solid var(--border); border-radius: 8px;
            background: linear-gradient(180deg, rgba(17,24,39,.94), rgba(8,13,22,.92));
            box-shadow: 0 12px 32px rgba(0,0,0,.24);
        }
        .kpi-card { padding: .95rem 1rem; min-height: 132px; }
        .kpi-name { color: var(--muted); font-size: .76rem; font-weight: 800; text-transform: uppercase; }
        .kpi-price { color: #f8fafc; font-size: 1.45rem; font-weight: 900; margin-top: .45rem; }
        .kpi-change { margin-top: .45rem; font-weight: 850; }
        .up { color: var(--green); } .down { color: var(--red); } .flat { color: var(--muted); }
        .kpi-time { color: #64748b; font-size: .72rem; margin-top: .45rem; }
        .news-card { display: flex; gap: .9rem; padding: .8rem; margin-bottom: .75rem; }
        .news-card img { width: 128px; height: 84px; object-fit: cover; border-radius: 6px; background: #111827; flex: 0 0 auto; }
        .news-body { min-width: 0; }
        .news-title { margin: 0 0 .35rem; color: #f8fafc; font-weight: 850; line-height: 1.32; }
        .news-meta { color: var(--muted); font-size: .8rem; margin-bottom: .4rem; }
        .news-link { color: var(--blue); font-weight: 800; text-decoration: none; }
        .status-line { color: var(--muted); font-size: .84rem; }
        .stButton > button {
            border-radius: 6px; border: 1px solid rgba(56, 189, 248, .32);
            background: #0f172a; color: #e0f2fe; font-weight: 800;
        }
        .stButton > button:hover { border-color: rgba(56, 189, 248, .72); color: #f8fafc; }
        @media (max-width: 760px) {
            .market-header h1 { font-size: 1.55rem; }
            .news-card { display: block; }
            .news-card img { width: 100%; height: 170px; margin-bottom: .65rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
