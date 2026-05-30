from __future__ import annotations

import streamlit as st

from data.mock_data import STOCK_PICKS, picks_dataframe


st.set_page_config(page_title="AI Stock Picks", page_icon="⭐", layout="wide")

st.markdown("""
<style>
.card{background:#0f172a;border:1px solid #1f2937;border-radius:18px;padding:18px;margin-bottom:12px}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-weight:800;font-size:12px}.emerald{background:#064e3b;color:#6ee7b7}.cyan{background:#164e63;color:#67e8f9}.amber{background:#78350f;color:#fcd34d}.rose{background:#881337;color:#fda4af}
</style>
""", unsafe_allow_html=True)

st.title("⭐ AI Stock Picks")
st.caption("Ranked mock ideas for placeholder UI.")

sectors = sorted({pick["sector"] for pick in STOCK_PICKS})
ratings = sorted({pick["rating"] for pick in STOCK_PICKS})
col1, col2, col3 = st.columns(3)
sector_filter = col1.multiselect("Sector", sectors, default=sectors)
rating_filter = col2.multiselect("Rating", ratings, default=ratings)
min_score = col3.slider("Minimum AI score", 0, 100, 60)

filtered = [p for p in STOCK_PICKS if p["sector"] in sector_filter and p["rating"] in rating_filter and p["score"] >= min_score]

for pick in filtered:
    st.markdown(
        f"""
        <div class='card'>
          <span class='badge {pick['badge']}'>{pick['rating']}</span>
          <h3>{pick['symbol']} · {pick['name']}</h3>
          <p><b>AI score:</b> {pick['score']} · <b>Target:</b> {pick['target']} · <b>Horizon:</b> {pick['horizon']}</p>
          <p>{pick['thesis']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Ranking Table")
ranking = picks_dataframe()
ranking = ranking[ranking["symbol"].isin([p["symbol"] for p in filtered])]
st.dataframe(ranking.sort_values("score", ascending=False), use_container_width=True, hide_index=True)

selected_symbol = st.selectbox("Detail", [p["symbol"] for p in filtered] or ["No picks"])
selected = next((p for p in filtered if p["symbol"] == selected_symbol), None)
if selected:
    st.subheader(f"{selected['symbol']} detail")
    left, right = st.columns(2)
    left.markdown(f"**Thesis:** {selected['thesis']}")
    right.markdown(f"**Key risk:** {selected['risk']}")
