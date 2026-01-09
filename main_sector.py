import streamlit as st
import pandas as pd
import utils_sector as us

# ==========================================
# UI HELPERS
# ==========================================
def get_ma_signal(price, ma_val):
    """Returns Emoji based on Price vs MA"""
    if pd.isna(ma_val) or ma_val == 0:
        return "⚠️" 
    return "✅" if price > ma_val else "❌"

@st.cache_data(ttl=3600)
def get_universe_cached():
    """Caches the universe loading to prevent re-reading CSV on every rerun"""
    dm = us.SectorDataManager()
    return dm.load_universe()

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_sector_rotation_app(df_global=None):
    st.title("🔄 Sector Rotation")
    
    # 1. Init Data Manager
    dm = us.SectorDataManager()
    
    # 2. Load Universe (Cached)
    uni_df, tickers, theme_map = get_universe_cached()
    
    if uni_df.empty:
        st.warning("⚠️ SECTOR_UNIVERSE secret is missing or empty.")
        return

    # 3. PERFORMANCE OPTIMIZATION: Bulk Load ETF Data
    # Instead of reading files in a loop later, we load all ETFs into memory once.
    etf_tickers = list(theme_map.values())
    etf_data_cache = dm.load_batch_data(etf_tickers)

    # 4. Session State for Controls
    if "sector_view" not in st.session_state: st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state: st.session_state.sector_trails = False
    if "sector_target" not in st.session_state: st.session_state.sector_target = sorted(list(theme_map.keys()))[0] if theme_map else ""
    
    all_themes = sorted(list(theme_map.keys()))
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- MAIN SECTION START ---
    st.subheader("Sector Rotations")

    # GRAPHIC USER GUIDE
    with st.expander("Graphic User Guide", expanded=False):
        st.markdown("""
        **🧮 How It Works**
        * **X-Axis (Trend):** `> 100` = Outperforming SPY.
        * **Y-Axis (Momentum):** `> 100` = Accelerating.
        
        **📊 Quadrant Guide**
        * 🟢 **LEADING:** Strong Trend + Accelerating.
        * 🟡 **WEAKENING:** Strong Trend, losing steam.
        * 🔴 **LAGGING:** Weak Trend + Decelerating.
        * 🔵 **IMPROVING:** Turnarounds (Weak Trend, Accelerating).
        """)

    # CONTROLS
    with st.expander("⚙️ Chart Inputs & Filters", expanded=True):
        c_in_1, c_in_2, c_spacer = st.columns([1.5, 1, 3]) 
        
        with c_in_1:
            st.session_state.sector_view = st.radio(
                "Timeframe Window", ["5 Days", "10 Days", "20 Days"], 
                horizontal=True, key="timeframe_radio"
            )
        
        with c_in_2:
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            st.session_state.sector_trails = st.checkbox("Show 3-Day Trails", value=st.session_state.sector_trails)
            
        # Update Button with Timestamp
        last_update = dm.get_last_updated()
        st.caption(f"📅 Data last updated: {last_update}")
        
        if st.button("🔄 Update Data", use_container_width=True):
            status = st.empty()
            calc = us.SectorAlphaCalculator()
            calc.run_full_update(status)
            st.cache_data.clear() # Clear cache after update
            st.rerun()

        st.divider()

        # SECTORS SHOWN
        st.markdown("**Sectors Shown**")
        btn_col1, btn_col2, _ = st.columns([1, 1, 6])
        with btn_col1:
            if st.button("➕ Add All", use_container_width=True):
                st.session_state.sector_theme_filter_widget = all_themes
                st.rerun()
        with btn_col2:
            if st.button("➖ Remove All", use_container_width=True):
                st.session_state.sector_theme_filter_widget = []
                st.rerun()
        
        sel_themes = st.multiselect(
            "Select Themes", all_themes, 
            key="sector_theme_filter_widget", label_visibility="collapsed"
        )
    
    filtered_map = {k: v for k, v in theme_map.items() if k in sel_themes}
    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    view_key = timeframe_map[st.session_state.sector_view]

    # --- MOMENTUM SCANS ---
    with st.expander("🚀 Momentum Scans", expanded=True):
        inc_mom, neut_mom, dec_mom = [], [], []
        
        for theme, ticker in theme_map.items():
            # Use Cache
            df = etf_data_cache.get(ticker)
            if df is None or df.empty or "RRG_Mom_Short" not in df.columns: continue
            
            last = df.iloc[-1]
            m5 = last.get("RRG_Mom_Short",0)
            m10 = last.get("RRG_Mom_Med",0)
            m20 = last.get("RRG_Mom_Long",0)
            
            shift = m5 - m20
            # Use util function
            setup = us.classify_setup(df)
            icon = setup.split()[0] if setup else ""
            item = {"theme": theme, "shift": shift, "icon": icon}
            
            if m5 > m10 > m20: inc_mom.append(item)
            elif m5 < m10 < m20: dec_mom.append(item)
            else: neut_mom.append(item)

        inc_mom.sort(key=lambda x: x['shift'], reverse=True)
        neut_mom.sort(key=lambda x: x['shift'], reverse=True)
        dec_mom.sort(key=lambda x: x['shift'], reverse=False)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: 
            st.success(f"📈 Increasing ({len(inc_mom)})")
            for i in inc_mom: st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")
        with m_col2:
            st.warning(f"⚖️ Neutral / Mixed ({len(neut_mom)})")
            for i in neut_mom: st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")
        with m_col3:
            st.error(f"🔻 Decreasing ({len(dec_mom)})")
            for i in dec_mom: st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")

    # RRG CHART (Using Utils function)
    chart_placeholder = st.empty()
    with chart_placeholder:
        # Pass the cache into the plotting function
        fig = us.plot_simple_rrg(etf_data_cache, filtered_map, view_key, st.session_state.sector_trails)
        chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
    
    if chart_event and chart_event.selection and chart_event.selection.points:
        point = chart_event.selection.points[0]
        if "customdata" in point:
            st.session_state.sector_target = point["customdata"]
        elif "text" in point:
            st.session_state.sector_target = point["text"]
    
    st.divider()

    # --- ALL THEMES PERFORMANCE ---
    st.subheader("All Themes Performance")
    summary_data = []
    
    for theme in all_themes:
        etf_ticker = theme_map.get(theme)
        if not etf_ticker: continue
        etf_df = etf_data_cache.get(etf_ticker) # Use Cache
        
        if etf_df is None or etf_df.empty: continue
        
        last = etf_df.iloc[-1]
        row = {"Theme": theme}
        
        for p, key in [("5d", "Short"), ("10d", "Med"), ("20d", "Long")]:
            row[f"Status ({p})"] = us.get_quadrant_status(etf_df, key)
            row[f"Rel Perf ({p})"] = last.get(f"RRG_Ratio_{key}", 100) - 100
            row[f"Mom ({p})"] = last.get(f"RRG_Mom_{key}", 100) - 100

        summary_data.append(row)
        
    if summary_data:
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Theme": st.column_config.TextColumn("Theme"), 
                "Rel Perf (5d)": st.column_config.NumberColumn("Rel Perf (5d)", format="%+.2f%%"),
                "Mom (5d)": st.column_config.NumberColumn("Mom (5d)", format="%+.2f"),
                "Rel Perf (10d)": st.column_config.NumberColumn("Rel Perf (10d)", format="%+.2f%%"),
                "Mom (10d)": st.column_config.NumberColumn("Mom (10d)", format="%+.2f"),
                "Rel Perf (20d)": st.column_config.NumberColumn("Rel Perf (20d)", format="%+.2f%%"),
                "Mom (20d)": st.column_config.NumberColumn("Mom (20d)", format="%+.2f"),
            }
        )

    st.markdown("---")

    # --- EXPLORER SECTION ---
    st.subheader(f"🔎 Explorer: {st.session_state.sector_target}")
    
    search_t = st.text_input("Input a ticker to find its theme(s)", placeholder="NVDA...").strip().upper()
    if search_t:
        matches = uni_df[uni_df['Ticker'] == search_t]
        if not matches.empty:
            found = matches['Theme'].unique()
            st.success(f"📍 Found **{search_t}** in: **{', '.join(found)}**")
            if len(found) > 0: st.session_state.sector_target = found[0]
        else:
            st.warning(f"Ticker {search_t} not found.")

    curr_idx = all_themes.index(st.session_state.sector_target) if st.session_state.sector_target in all_themes else 0
    new_target = st.selectbox("Select Theme to View Stocks", all_themes, index=curr_idx)
    if new_target != st.session_state.sector_target:
        st.session_state.sector_target = new_target

    # --- STOCK TABLE (Optimized) ---
    stock_tickers = uni_df[(uni_df['Theme'] == st.session_state.sector_target) & (uni_df['Role'] == 'Stock')]['Ticker'].tolist()
    
    # BATCH LOAD STOCKS FOR THIS SECTOR ONLY
    stock_cache = dm.load_batch_data(stock_tickers)
    
    ranking_data = []
    
    for stock in stock_tickers:
        sdf = stock_cache.get(stock) # Use Cache
        
        if sdf is None or sdf.empty: continue
        
        try:
            # Volume Filter
            avg_vol = sdf['Volume'].tail(20).mean()
            avg_price = sdf['Close'].tail(20).mean()
            if (avg_vol * avg_price) < us.MIN_DOLLAR_VOLUME: continue
            
            last = sdf.iloc[-1]
            
            def safe_get(key, default=0): return last.get(key, default)

            ranking_data.append({
                "Ticker": stock,
                "Price": last['Close'],
                "Alpha 5d": safe_get("True_Alpha_Short"),
                "RVOL 5d": safe_get("RVOL_Short"),
                "Alpha 10d": safe_get("True_Alpha_Med"),
                "RVOL 10d": safe_get("RVOL_Med"),
                "Alpha 20d": safe_get("True_Alpha_Long"),
                "RVOL 20d": safe_get("RVOL_Long"),
                "8 EMA": get_ma_signal(last['Close'], safe_get('EMA_8')),
                "21 EMA": get_ma_signal(last['Close'], safe_get('EMA_21')),
                "50 MA": get_ma_signal(last['Close'], safe_get('SMA_50')),
                "200 MA": get_ma_signal(last['Close'], safe_get('SMA_200'))
            })
        except Exception:
            continue

    if ranking_data:
        df_disp = pd.DataFrame(ranking_data).sort_values(by='Alpha 5d', ascending=False)
        
        def highlight_cells(row):
            styles = pd.Series('', index=row.index)
            color = 'background-color: #d4edda; color: black;'
            if row["Alpha 5d"] > 0 and row["RVOL 5d"] > 1.2:
                styles["Alpha 5d"] = color; styles["RVOL 5d"] = color
            if row["Alpha 10d"] > 0 and row["RVOL 10d"] > 1.2:
                styles["Alpha 10d"] = color; styles["RVOL 10d"] = color
            if row["Alpha 20d"] > 0 and row["RVOL 20d"] > 1.2:
                styles["Alpha 20d"] = color; styles["RVOL 20d"] = color
            return styles

        st.dataframe(
            df_disp.style.apply(highlight_cells, axis=1),
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"), 
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Alpha 5d": st.column_config.NumberColumn("Alpha 5d", format="%+.2f%%"),
                "RVOL 5d": st.column_config.NumberColumn("RVOL 5d", format="%.1fx"),
                "Alpha 10d": st.column_config.NumberColumn("Alpha 10d", format="%+.2f%%"),
                "RVOL 10d": st.column_config.NumberColumn("RVOL 10d", format="%.1fx"),
                "Alpha 20d": st.column_config.NumberColumn("Alpha 20d", format="%+.2f%%"),
                "RVOL 20d": st.column_config.NumberColumn("RVOL 20d", format="%.1fx"),
                "8 EMA": st.column_config.TextColumn("8 EMA", width="small"),
                "21 EMA": st.column_config.TextColumn("21 EMA", width="small"),
                "50 MA": st.column_config.TextColumn("50 MA", width="small"),
                "200 MA": st.column_config.TextColumn("200 MA", width="small")
            }
        )
    else:
        st.info(f"No stocks found for {st.session_state.sector_target} (or filtered by volume).")