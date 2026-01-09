# MAIN FILE *ONLY* FOR SECTOR PAGE

import streamlit as st
import plotly.graph_objects as go
from utils_sector import load_sector_data, get_momentum_trends

def run_sector_rotation_app():
    st.title("🧠 Sector Rotation & Theme Explorer")
    dm, uni_df, theme_map = load_sector_data()

    if uni_df.empty:
        st.warning("⚠️ No data found! Please click 'Update Data'.")
        if st.button("🔄 Initial Data Sync"):
            # Trigger his update logic here
            pass
        return

    # --- TOP SECTION: FILTERS (Formerly in Sidebar) ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        view_label = st.radio("⏱️ Timeframe", ["5 Days", "10 Days", "20 Days"], horizontal=True)
        show_trails = st.checkbox("Show 3-Day Trails")
    with col2:
        all_themes = sorted(list(theme_map.keys()))
        selected_themes = st.multiselect("Filter Themes", options=all_themes, default=all_themes)
    with col3:
        search_ticker = st.text_input("Find Ticker in Theme", placeholder="e.g. NVDA").upper()

    # --- MAIN DASHBOARD TABS ---
    tab_chart, tab_explorer, tab_trends = st.tabs(["📊 Rotation Chart", "🔎 Theme Details", "🌊 Momentum Trends"])

    with tab_chart:
        # Insert his Plotly RRG Chart Logic here
        st.info("RRG Rotation Chart rendering here...")

    with tab_explorer:
        # Consolidate his "Theme Explorer" logic here
        theme_to_explore = st.selectbox("Select Theme to Inspect", all_themes)
        st.write(f"Showing details for {theme_to_explore}...")

    with tab_trends:
        inc, neut, dec = get_momentum_trends(dm, theme_map)
        c1, c2, c3 = st.columns(3)
        with c1: st.success("📈 Increasing"); st.write(inc)
        with c2: st.info("⚖️ Neutral"); st.write(neut)
        with c3: st.error("🔻 Decreasing"); st.write(dec)
