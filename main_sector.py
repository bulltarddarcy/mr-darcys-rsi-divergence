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

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_sector_rotation_app(df_global=None):
    st.title("🔄 Sector Rotation")
    
    # 0. Benchmark Control (Session State)
    if "sector_benchmark" not in st.session_state: st.session_state.sector_benchmark = "SPY"

    # 1. Automatic Data Fetch (Cached 10m)
    # UPDATED: Calls the new optimized function 'get_computed_sector_data'
    with st.spinner(f"Calculating Sector Metrics ({st.session_state.sector_benchmark})..."):
        etf_data_cache, missing_tickers, theme_map, uni_df = us.get_computed_sector_data(st.session_state.sector_benchmark)

    if uni_df.empty:
        st.warning("⚠️ SECTOR_UNIVERSE secret is missing or empty.")
        return

    # 2. Check for Missing Data
    if missing_tickers:
        with st.expander(f"⚠️ Missing Data for {len(missing_tickers)} Tickers", expanded=False):
            st.caption("The following tickers were in your Universe list but no matching Parquet file was found in the Ticker Map.")
            st.write(", ".join(missing_tickers))

    # 3. Session State for Controls
    if "sector_view" not in st.session_state: st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state: st.session_state.sector_trails = False
    
    # Ensure target is valid
    all_themes = sorted(list(theme_map.keys()))
    if not all_themes:
        st.error("No valid themes found. Check data sources.")
        return

    if "sector_target" not in st.session_state or st.session_state.sector_target not in all_themes: 
        st.session_state.sector_target = all_themes[0]
    
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- MAIN SECTION START ---
    st.subheader("Rotation Quadrant Graphic")

    # 1. GRAPHIC USER GUIDE
    with st.expander("🗺️ Graphic User Guide", expanded=False):
        st.markdown(f"""
        **🧮 How It Works (The Math)**
        This chart does **not** show price. It shows **Relative Performance** against **{st.session_state.sector_benchmark}**.
        * **X-Axis (Trend):** Are we beating the benchmark?
            * `> 100`: Outperforming {st.session_state.sector_benchmark}.
            * `< 100`: Underperforming {st.session_state.sector_benchmark}.
        * **Y-Axis (Momentum):** How fast is the trend changing?
            * `> 100`: Gaining speed (Acceleration).
            * `< 100`: Losing speed (Deceleration).
        
        *Note: Calculations use Weighted Regression (Today's price weighted 3x vs 20 days ago).*
        
        **📊 Quadrant Guide**
        * 🟢 **LEADING (Top Right):** Strong Trend + Accelerating Momentum. The Winners.
        * 🟡 **WEAKENING (Bottom Right):** Strong Trend, but losing steam. Often a place to take profits.
        * 🔴 **LAGGING (Bottom Left):** Weak Trend + Decelerating. The Losers.
        * 🔵 **IMPROVING (Top Left):** Weak Trend, but Momentum is waking up. "Turnarounds".
        """)

    # CONTROLS
    with st.expander("⚙️ Chart Inputs & Filters", expanded=False):
        col_inputs, col_filters = st.columns([1, 1])
        
        # --- LEFT COLUMN: Timeframe, Trails ---
        with col_inputs:
            # Benchmark Selection
            st.markdown("**Benchmark Ticker**")
            new_benchmark = st.radio(
                "Benchmark",
                ["SPY", "QQQ"],
                horizontal=True,
                index=["SPY", "QQQ"].index(st.session_state.sector_benchmark) if st.session_state.sector_benchmark in ["SPY", "QQQ"] else 0,
                key="sector_benchmark_radio",
                label_visibility="collapsed"
            )
            
            # UPDATED: If changed, rerun to update session state and fetch new data
            if new_benchmark != st.session_state.sector_benchmark:
                st.session_state.sector_benchmark = new_benchmark
                # Note: We do NOT clear cache here anymore because the download cache is separate now
                st.rerun()

            st.markdown("---")
            
            st.markdown("**Timeframe Window**") 
            st.session_state.sector_view = st.radio(
                "Timeframe Window", 
                ["5 Days", "10 Days", "20 Days"], 
                horizontal=True, 
                key="timeframe_radio",
                label_visibility="collapsed"
            )
            
            st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
            st.session_state.sector_trails = st.checkbox("Show 3-Day Trails", value=st.session_state.sector_trails)
            
            # Display Last Data Date
            if st.session_state.sector_benchmark in etf_data_cache and not etf_data_cache[st.session_state.sector_benchmark].empty:
                last_dt = etf_data_cache[st.session_state.sector_benchmark].index[-1].strftime("%Y-%m-%d")
                st.caption(f"📅 Data Date: {last_dt}")

        # --- RIGHT COLUMN: Sector Filters ---
        with col_filters:
            st.markdown("**Sectors Shown**")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("➕ Everything", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = all_themes
                    st.rerun()

            with btn_col2:
                if st.button("⭐ Big 11", use_container_width=True):
                    big_11_list = [
                        "Communications", "Consumer Discretionary", "Consumer Staples", 
                        "Energy", "Financials", "Healthcare", "Industrials", 
                        "Materials", "Real Estate", "Technology", "Utilities"
                    ]
                    valid_themes = [t for t in big_11_list if t in all_themes]
                    st.session_state.sector_theme_filter_widget = valid_themes
                    st.rerun()

            with btn_col3:
                if st.button("➖ Clear", use_container_width=True):
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
    with st.expander("🚀 Momentum Scans", expanded=False):
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

    # RRG CHART
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
    
    st.markdown(f"""
    * **Rel Perf (Relative Performance):** Measures the strength of the trend against {st.session_state.sector_benchmark}. 
      Values positive (>0) indicate outperformance; negative (<0) indicate underperformance.
    * **Mom (Momentum):** Measures the rate of change (velocity) of the trend. 
      Values positive (>0) indicate acceleration; negative (<0) indicate deceleration.
    """)
    
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
    st.subheader(f"🔎 Explorer: Theme Drilldown")
    
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

    # --- STOCK TABLE ---
    # Filter tickers for current theme
    stock_tickers = uni_df[(uni_df['Theme'] == st.session_state.sector_target) & (uni_df['Role'] == 'Stock')]['Ticker'].tolist()
    
    ranking_data = []
    
    for stock in stock_tickers:
        sdf = etf_data_cache.get(stock) # Already fetched in initial batch
        
        if sdf is None or sdf.empty: continue
        
        try:
            # Volume Filter (Assuming min 20 days data available)
            if len(sdf) < 20: continue
            
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