"""Premium CSS for ORCA AI Chat."""
from __future__ import annotations
import streamlit as st


def inject() -> None:
    st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:          #020617;
  --panel:       rgba(15,23,42,0.85);
  --panel-hover: rgba(15,23,42,0.95);
  --border:      rgba(103,232,249,0.15);
  --border-hi:   rgba(103,232,249,0.45);
  --cyan:        #67e8f9;
  --cyan-dim:    rgba(103,232,249,0.12);
  --emerald:     #6ee7b7;
  --emerald-dim: rgba(110,231,183,0.12);
  --amber:       #fcd34d;
  --amber-dim:   rgba(252,211,77,0.12);
  --rose:        #fda4af;
  --rose-dim:    rgba(253,164,175,0.12);
  --slate:       #94a3b8;
  --text:        #e2e8f0;
  --text-muted:  #64748b;
  --radius:      16px;
  --radius-sm:   10px;
}

.stApp {
  font-family: 'Inter', sans-serif;
  background:
    radial-gradient(ellipse at 10% 5%,  rgba(20,184,166,0.18) 0%, transparent 40%),
    radial-gradient(ellipse at 90% 15%, rgba(34,211,238,0.14) 0%, transparent 40%),
    linear-gradient(160deg, #020617 0%, #06111e 50%, #0f172a 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(2,6,23,0.97) 0%, rgba(8,47,73,0.55) 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
  background: rgba(15,23,42,0.7) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--panel) !important;
  backdrop-filter: blur(12px);
  box-shadow: 0 4px 24px rgba(2,6,23,0.25);
  margin-bottom: 0.75rem;
  transition: border-color 0.2s ease;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMetric"],
[data-testid="stChatMessage"] [data-testid="stExpander"] {
  max-width: calc(100% - 1.25rem);
}
[data-testid="stChatMessage"] [data-testid="stDataFrame"] {
  margin: 0.75rem 0.625rem 1rem 0.625rem;
}
[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] {
  padding: 0 0.625rem;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] {
  margin: 0.75rem 0.625rem;
}
[data-testid="stChatMessage"]:hover {
  border-color: var(--border-hi) !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
  background: rgba(20,184,166,0.08) !important;
  border-color: rgba(103,232,249,0.22) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  border-radius: var(--radius) !important;
  border-color: var(--border) !important;
  background: rgba(2,6,23,0.85) !important;
  backdrop-filter: blur(12px);
  box-shadow: 0 -2px 20px rgba(2,6,23,0.3);
}
[data-testid="stChatInput"] textarea {
  color: var(--text) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-muted) !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  border-radius: 999px !important;
  border: 1px solid var(--border) !important;
  background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(34,211,238,0.08)) !important;
  color: #e0f2fe !important;
  transition: all 0.18s ease !important;
  letter-spacing: 0.01em;
}
.stButton > button:hover {
  transform: translateY(-1px);
  border-color: var(--border-hi) !important;
  background: linear-gradient(135deg, rgba(20,184,166,0.28), rgba(34,211,238,0.16)) !important;
  box-shadow: 0 4px 16px rgba(103,232,249,0.15);
}

/* ── Metric tiles ── */
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.9rem 1.1rem !important;
  transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--border-hi); }
[data-testid="stMetricLabel"] {
  color: var(--text) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.25 !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
  color: var(--text) !important;
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700 !important; font-size: 1.5rem !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important; overflow: hidden; }

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Custom components ── */
.orca-kicker {
  color: var(--cyan);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin: 1.2rem 0 0.5rem;
}
.orca-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 0.75rem;
  backdrop-filter: blur(8px);
}
.orca-card.buy   { border-left: 3px solid #6ee7b7; }
.orca-card.sell  { border-left: 3px solid #fda4af; }
.orca-card.hold  { border-left: 3px solid #fcd34d; }
.orca-card.watch { border-left: 3px solid #67e8f9; }
.orca-card.wait  { border-left: 3px solid #64748b; }

.rec-badge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  font-weight: 800; font-size: 0.82rem; letter-spacing: 0.06em;
}
.rec-badge.buy   { background: var(--emerald-dim); color: var(--emerald); border: 1px solid rgba(110,231,183,0.3); }
.rec-badge.sell  { background: var(--rose-dim);    color: var(--rose);    border: 1px solid rgba(253,164,175,0.3); }
.rec-badge.hold  { background: var(--amber-dim);   color: var(--amber);   border: 1px solid rgba(252,211,77,0.3); }
.rec-badge.watch { background: var(--cyan-dim);    color: var(--cyan);    border: 1px solid rgba(103,232,249,0.3); }
.rec-badge.wait  { background: rgba(100,116,139,0.12); color: var(--slate); border: 1px solid rgba(100,116,139,0.3); }

.conf-pill {
  display: inline-block;
  padding: 0.15rem 0.65rem;
  border-radius: 999px;
  font-size: 0.78rem; font-weight: 700;
}
.conf-high   { background: var(--emerald-dim); color: var(--emerald); }
.conf-medium { background: var(--amber-dim);   color: var(--amber);   }
.conf-low    { background: var(--rose-dim);     color: var(--rose);    }

.signal-row {
  display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0;
}
.signal-chip {
  display: inline-flex; align-items: center; gap: 0.25rem;
  padding: 0.2rem 0.6rem; border-radius: 999px;
  font-size: 0.76rem; font-weight: 600;
}
.signal-chip.support  { background: var(--emerald-dim); color: var(--emerald); }
.signal-chip.conflict { background: var(--rose-dim);    color: var(--rose);    }

.status-dot {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 0.4rem;
}
.dot-ok      { background: #6ee7b7; box-shadow: 0 0 6px #6ee7b7; }
.dot-warn    { background: #fcd34d; box-shadow: 0 0 6px #fcd34d; }
.dot-error   { background: #fda4af; box-shadow: 0 0 6px #fda4af; }
.dot-offline { background: #64748b; }

.route-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; font-weight: 500;
  background: rgba(103,232,249,0.1);
  color: var(--cyan);
  border: 1px solid rgba(103,232,249,0.2);
  padding: 0.1rem 0.5rem; border-radius: 6px;
}

.empty-state {
  text-align: center;
  padding: 3.5rem 2rem;
  color: var(--text-muted);
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-state h3 { color: var(--slate); font-weight: 600; margin-bottom: 0.4rem; }
.empty-state p  { font-size: 0.9rem; max-width: 360px; margin: 0 auto; }

.prompt-chip button {
  border-radius: 999px !important;
  font-size: 0.82rem !important;
  padding: 0.35rem 1rem !important;
}

.job-row {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.6rem 1rem;
  margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)
