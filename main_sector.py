"""
Sector Rotation App - REFACTORED VERSION
With multi-theme support, smart filters, and comprehensive scoring.
"""

import streamlit as st
import pandas as pd
import utils_sector as us

# ==========================================
# UI HELPERS
# ==========================================
def get_ma_signal(price: float, ma_val: float) -> str:
    """
    Return emoji based on price vs moving average.
    
    Args:
        price: Current price
        ma_val: Moving average value
        
    Returns:
        Emoji indicator
    """
    if pd.isna(ma_val) or ma_val == 0:
        return "⚠️"
    return "✅" if price > ma_val else "❌"

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_sector_rotation_app(df_global=None):
    """
    Main entry point for Sector Rotation application.
    
    Features:
    - RRG quadrant analysis
    - Multi-timeframe views
    - Stock-level alpha analysis
    - Smart pattern filters
    - Comprehensive scoring
    """
    st.title("🔄 Sector Rotation")
    
    # --- 0a. PRE-CALC LAST OPTIONS TRADE ---
    # Create lookup map for speed: Ticker -> Date string
    last_trade_map = {}
    if df_global is not None and not df_global.empty:
        try:
            # Ensure Trade Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_global['Trade Date']):
                df_global['Trade Date'] = pd.to_datetime(df_global['Trade Date'], errors='coerce')
                
            # Group by Symbol, get Max Date
            last_trade_series = df_global.groupby('Symbol')['Trade Date'].max()
            # Convert to dict: {'AAPL': '25 Jan 24', ...}
            last_trade_map = last_trade_series.dt.strftime('%d %b %y').to_dict()
        except Exception:
            pass # Fail silently if global data structure differs

    # --- 0b. BENCHMARK CONTROL ---
    if "sector_benchmark" not in st.session_state:
        st.session_state.sector_benchmark = "SPY"

    # --- 1. DATA FETCH (CACHED) ---
    with st.spinner(f"Syncing Sector Data ({st.session_state.sector_benchmark})..."):
        etf_data_cache, missing_tickers, theme_map, uni_df, stock_themes = \
            us.fetch_and_process_universe(st.session_state.sector_benchmark)

    if uni_df.empty:
        st.warning("⚠️ SECTOR_UNIVERSE secret is missing or empty.")
        return

    # --- 2. MISSING DATA CHECK ---
    if missing_tickers:
        with st.expander(f"⚠️ Missing Data for {len(missing_tickers)} Tickers", expanded=False):
            st.caption("These tickers were in your Universe but not found in the parquet file.")
            st.write(", ".join(missing_tickers))

    # --- 3. SESSION STATE INITIALIZATION ---
    if "sector_view" not in st.session_state:
        st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state:
        st.session_state.sector_trails = False
    
    all_themes = sorted(list(theme_map.keys()))
    if not all_themes:
        st.error("No valid themes found. Check data sources.")
        return

    if "sector_target" not in st.session_state or st.session_state.sector_target not in all_themes:
        st.session_state.sector_target = all_themes[0]
    
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- 4. RRG QUADRANT GRAPHIC ---
    st.subheader("Rotation Quadrant Graphic")

    # User Guide
    with st.expander("🗺️ Graphic User Guide", expanded=False):
        st.markdown(f"""
        **🧮 How It Works (The Math)**
        This chart shows **Relative Performance** against **{st.session_state.sector_benchmark}** (not absolute price).
        
        * **X-Axis (Trend):** Are we beating the benchmark?
            * `> 100`: Outperforming {st.session_state.sector_benchmark}
            * `< 100`: Underperforming {st.session_state.sector_benchmark}
        * **Y-Axis (Momentum):** How fast is the trend changing?
            * `> 100`: Gaining speed (Acceleration)
            * `< 100`: Losing speed (Deceleration)
        
        *Calculations use Weighted Regression (recent days weighted 3x more)*
        
        **📊 Quadrant Guide**
        * 🟢 **LEADING (Top Right):** Strong trend + accelerating. The winners.
        * 🟡 **WEAKENING (Bottom Right):** Strong trend but losing steam. Take profits.
        * 🔴 **LAGGING (Bottom Left):** Weak trend + decelerating. The losers.
        * 🔵 **IMPROVING (Top Left):** Weak trend but momentum building. Turnarounds.
        """)

    # Controls
    with st.expander("⚙️ Chart Inputs & Filters", expanded=False):
        col_inputs, col_filters = st.columns([1, 1])
        
        # --- LEFT: TIMEFRAME & BENCHMARK ---
        with col_inputs:
            st.markdown("**Benchmark Ticker**")
            new_benchmark = st.radio(
                "Benchmark",
                ["SPY", "QQQ"],
                horizontal=True,
                index=["SPY", "QQQ"].index(st.session_state.sector_benchmark) 
                    if st.session_state.sector_benchmark in ["SPY", "QQQ"] else 0,
                key="sector_benchmark_radio",
                label_visibility="collapsed"
            )
            
            if new_benchmark != st.session_state.sector_benchmark:
                st.session_state.sector_benchmark = new_benchmark
                st.cache_data.clear()
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
            st.session_state.sector_trails = st.checkbox(
                "Show 3-Day Trails",
                value=st.session_state.sector_trails
            )
            
            # Display last data date
            if st.session_state.sector_benchmark in etf_data_cache:
                bench_df = etf_data_cache[st.session_state.sector_benchmark]
                if not bench_df.empty:
                    last_dt = bench_df.index[-1].strftime("%Y-%m-%d")
                    st.caption(f"📅 Data Date: {last_dt}")

        # --- RIGHT: SECTOR FILTERS ---
        with col_filters:
            st.markdown("**Sectors Shown**")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("➕ Everything", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = all_themes
                    st.rerun()

            with btn_col2:
                if st.button("⭐ Big 11", use_container_width=True):
                    big_11 = [
                        "Communications", "Consumer Discretionary", "Consumer Staples",
                        "Energy", "Financials", "Healthcare", "Industrials",
                        "Materials", "Real Estate", "Technology", "Utilities"
                    ]
                    valid = [t for t in big_11 if t in all_themes]
                    st.session_state.sector_theme_filter_widget = valid
                    st.rerun()

            with btn_col3:
                if st.button("➖ Clear", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = []
                    st.rerun()
            
            sel_themes = st.multiselect(
                "Select Themes",
                all_themes,
                key="sector_theme_filter_widget",
                label_visibility="collapsed"
            )
    
    filtered_map = {k: v for k, v in theme_map.items() if k in sel_themes}
    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    view_key = timeframe_map[st.session_state.sector_view]

    # --- 5. MOMENTUM SCANS ---
    with st.expander("🚀 Momentum Scans", expanded=False):
        inc_mom, neut_mom, dec_mom = [], [], []
        
        for theme, ticker in theme_map.items():
            df = etf_data_cache.get(ticker)
            if df is None or df.empty or "RRG_Mom_Short" not in df.columns:
                continue
            
            last = df.iloc[-1]
            m5 = last.get("RRG_Mom_Short", 0)
            m10 = last.get("RRG_Mom_Med", 0)
            m20 = last.get("RRG_Mom_Long", 0)
            
            shift = m5 - m20
            setup = us.classify_setup(df)
            icon = setup.split()[0] if setup else ""
            item = {"theme": theme, "shift": shift, "icon": icon}
            
            # Categorize
            if m5 > m10 > m20:
                inc_mom.append(item)
            elif m5 < m10 < m20:
                dec_mom.append(item)
            else:
                neut_mom.append(item)

        # Sort by magnitude
        inc_mom.sort(key=lambda x: x['shift'], reverse=True)
        neut_mom.sort(key=lambda x: x['shift'], reverse=True)
        dec_mom.sort(key=lambda x: x['shift'], reverse=False)

        # Display in columns
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.success(f"📈 Increasing ({len(inc_mom)})")
            for i in inc_mom:
                st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")
        
        with m_col2:
            st.warning(f"⚖️ Neutral / Mixed ({len(neut_mom)})")
            for i in neut_mom:
                st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")
        
        with m_col3:
            st.error(f"🔻 Decreasing ({len(dec_mom)})")
            for i in dec_mom:
                st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")

    # --- 6. RRG CHART ---
    chart_placeholder = st.empty()
    with chart_placeholder:
        fig = us.plot_simple_rrg(etf_data_cache, filtered_map, view_key, st.session_state.sector_trails)
        chart_event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points"
        )
    
    # Handle chart selection
    if chart_event and chart_event.selection and chart_event.selection.points:
        point = chart_event.selection.points[0]
        if "customdata" in point:
            st.session_state.sector_target = point["customdata"]
        elif "text" in point:
            st.session_state.sector_target = point["text"]
    
    st.divider()

    # --- 7. SECTOR LIFECYCLE ANALYSIS ---
    st.subheader("📊 Sector Lifecycle Dashboard")
    
    st.info("💡 **Where to Deploy Capital** - Sectors grouped by lifecycle stage to identify best entries, holdings to keep, and positions to exit")
    
    # Help section
    col_help_theme1, col_help_theme2, col_help_theme3 = st.columns([1, 1, 1])
    with col_help_theme1:
        st.markdown("**🎯 Early Stage:** Fresh momentum - best new entries")
    with col_help_theme2:
        st.markdown("**⚖️ Established:** Mature trends - hold but don't add")
    with col_help_theme3:
        with st.popover("📖 How Lifecycle Works", use_container_width=True):
            st.markdown("""
            ### Sector Lifecycle Stages
            
            **🎯 Early Stage Leadership**
            - Just entered bullish quadrants
            - 2+ timeframes confirming
            - Score 60+
            → **Action:** Best time to enter new positions
            
            **⚖️ Established Leadership** - Strong but been leading for days
            - High score but not fresh
            → **Action:** Hold positions, don't chase
            
            **📉 Topping/Weakening**
            - Was strong, now losing momentum
            - 5d weaker than 20d
            → **Action:** Take profits, exit positions
            
            **🚫 Weak/Lagging**
            - Poor positioning across timeframes
            - Low scores
            → **Action:** Stay away, no allocation
            """)
            st.markdown("---")
            if st.button("📖 View Complete Theme Guide", use_container_width=True):
                st.session_state.show_theme_guide = True
                st.rerun()
    
    # Show full theme guide if requested
    if st.session_state.get('show_theme_guide', False):
        with st.expander("📖 Complete Theme Scoring Guide", expanded=True):
            if st.button("✖️ Close Theme Guide"):
                st.session_state.show_theme_guide = False
                st.rerun()
            
            try:
                with open("THEME_SCORING_GUIDE.md", "r") as f:
                    st.markdown(f.read())
            except FileNotFoundError:
                st.error("THEME_SCORING_GUIDE.md not found. Please ensure it's in the repo root directory.")
    
    # Get lifecycle-based theme summary
    categories = us.get_actionable_theme_summary(etf_data_cache, theme_map)
    
    # --- EARLY STAGE: Best new entries ---
    if categories['early_stage']:
        st.success(f"🎯 **EARLY STAGE LEADERSHIP** ({len(categories['early_stage'])} sectors)")
        
        early_data = []
        for theme_info in categories['early_stage']:
            # Format momentum trend
            s5, s10, s20 = theme_info['score_5d'], theme_info['score_10d'], theme_info['score_20d']
            if s5 > s10 > s20:
                momentum_trend = f"🚀 {s5:.0f} > {s10:.0f} > {s20:.0f}"
            else:
                momentum_trend = f"➡️ {s5:.0f} ≈ {s10:.0f}"
            
            early_data.append({
                "Sector": theme_info['theme'],
                "Score": theme_info['consensus_score'],
                "Grade": theme_info['grade'],
                "Momentum Trend": momentum_trend,
                "Stage": theme_info['freshness_detail'],
                "5d": theme_info['tf_5d'],
                "10d": theme_info['tf_10d'],
                "20d": theme_info['tf_20d'],
                "Why Selected": theme_info['reason']
            })
        
        st.dataframe(
            pd.DataFrame(early_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Grade": st.column_config.TextColumn("Grade", width="small"),
                "Stage": st.column_config.TextColumn("Stage", width="small"),
                "Why Selected": st.column_config.TextColumn("Why Selected", width="large"),
            }
        )
        st.caption("✅ **Trading Action:** Fresh momentum building - best time to initiate new swing positions. High risk/reward.")
        
        with st.expander("📖 Why These Are 'Early Stage'"):
            st.markdown("""
            **Selection Criteria (ALL must be true):**
            
            1. ✅ **Fresh Entry:** Day 1-3 in current quadrant
               - *Why:* Early = better risk/reward than chasing
            
            2. ✅ **Multi-Timeframe Confirmation:** 2+ timeframes bullish (Leading or Improving)
               - *Why:* Need confirmation across timeframes for swing trades
            
            3. ✅ **Quality Score:** 60+ points
               - *Why:* Filters out weak setups
            
            4. ✅ **Momentum Accelerating or Stable:** 5d ≥ 10d ≥ 20d scores
               - *Why:* Want building momentum, not declining
               - *Example:* Score trend 78 > 75 > 71 = accelerating ✓
               - *Example:* Score trend 72 < 75 < 78 = declining ✗
            """)
    else:
        st.info("🎯 **EARLY STAGE LEADERSHIP** - No sectors currently showing fresh momentum buildup")
    
    # --- ESTABLISHED: Hold but don't chase ---
    if categories['established']:
        st.info(f"⚖️ **ESTABLISHED LEADERSHIP** ({len(categories['established'])} sectors)")
        
        established_data = []
        for theme_info in categories['established']:
            # Format momentum trend
            s5, s10, s20 = theme_info['score_5d'], theme_info['score_10d'], theme_info['score_20d']
            if s5 < s10 or s5 < s20:
                momentum_trend = f"📉 {s5:.0f} < {s10:.0f}"
            elif s5 > s10 > s20:
                momentum_trend = f"🚀 {s5:.0f} > {s10:.0f} > {s20:.0f}"
            else:
                momentum_trend = f"➡️ {s5:.0f} ≈ {s10:.0f}"
            
            established_data.append({
                "Sector": theme_info['theme'],
                "Score": theme_info['consensus_score'],
                "Grade": theme_info['grade'],
                "Momentum Trend": momentum_trend,
                "Stage": theme_info['freshness_detail'],
                "5d": theme_info['tf_5d'],
                "10d": theme_info['tf_10d'],
                "20d": theme_info['tf_20d'],
                "Why Selected": theme_info['reason']
            })
        
        st.dataframe(
            pd.DataFrame(established_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Grade": st.column_config.TextColumn("Grade", width="small"),
                "Stage": st.column_config.TextColumn("Stage", width="small"),
                "Why Selected": st.column_config.TextColumn("Why Selected", width="large"),
            }
        )
        st.caption("⚖️ **Trading Action:** Mature uptrends - hold existing positions but avoid chasing. Look to Early Stage for new entries instead.")
        
        with st.expander("📖 Why These Are 'Established'"):
            st.markdown("""
            **Selection Criteria (ALL must be true):**
            
            1. ✅ **High Score:** 65+ points
               - *Why:* Still strong positioning
            
            2. ✅ **Multi-Timeframe Confirmation:** 2+ timeframes bullish
               - *Why:* Trend still intact
            
            3. ✅ **NOT Fresh:** Day 4+ in current quadrant
               - *Why:* Been running for a while - late to enter
            
            **Note:** May show declining momentum (score 82 → 79 → 75) but still strong overall.
            This is normal for mature trends. Hold but don't add.
            """)
    else:
        st.info("⚖️ **ESTABLISHED LEADERSHIP** - No sectors in mature leadership phase")
    
    # --- TOPPING: Take profits ---
    if categories['topping']:
        st.warning(f"📉 **TOPPING / WEAKENING** ({len(categories['topping'])} sectors)")
        
        topping_data = []
        for theme_info in categories['topping']:
            # Format momentum trend
            s5, s10, s20 = theme_info['score_5d'], theme_info['score_10d'], theme_info['score_20d']
            momentum_trend = f"📉 {s5:.0f} < {s10:.0f} or {s20:.0f}"
            
            topping_data.append({
                "Sector": theme_info['theme'],
                "Score": theme_info['consensus_score'],
                "Grade": theme_info['grade'],
                "Momentum Trend": momentum_trend,
                "Stage": theme_info['freshness_detail'],
                "5d": theme_info['tf_5d'],
                "10d": theme_info['tf_10d'],
                "20d": theme_info['tf_20d'],
                "Why Selected": theme_info['reason']
            })
        
        st.dataframe(
            pd.DataFrame(topping_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Grade": st.column_config.TextColumn("Grade", width="small"),
                "Stage": st.column_config.TextColumn("Stage", width="small"),
                "Why Selected": st.column_config.TextColumn("Why Selected", width="large"),
            }
        )
        st.caption("📉 **Trading Action:** Losing momentum - exit positions, take profits. Don't fight the rotation.")
        
        with st.expander("📖 Why These Are 'Topping'"):
            st.markdown("""
            **Selection Criteria (ANY can trigger):**
            
            1. ⚠️ **Momentum Declining:** 5-day score < 10-day or 20-day score
               - *Why:* Recent momentum weaker than past = losing steam
               - *Example:* Scores 68 < 72 < 75 = declining trend
            
            2. ⚠️ **5-Day Weakening:** Short-term moved to Weakening quadrant
               - *Why:* Early warning sign of reversal
            
            3. ⚠️ **Mixed Signals:** Was bullish on 20d but not on 5d
               - *Why:* Short-term turning negative
            
            **These are EXIT signals.** Don't wait for it to become fully weak.
            Take profits while you still can!
            """)
    else:
        st.success("✅ No sectors currently showing topping behavior")
    
    # --- WEAK: Avoid ---
    if categories['weak']:
        with st.expander(f"🚫 **WEAK / LAGGING** ({len(categories['weak'])} sectors)", expanded=False):
            weak_data = []
            for theme_info in categories['weak']:
                weak_data.append({
                    "Sector": theme_info['theme'],
                    "Score": theme_info['consensus_score'],
                    "Grade": theme_info['grade'],
                    "5d": theme_info['tf_5d'],
                    "10d": theme_info['tf_10d'],
                    "20d": theme_info['tf_20d'],
                    "Why Weak": theme_info['reason']
                })
            
            st.dataframe(
                pd.DataFrame(weak_data),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                    "Grade": st.column_config.TextColumn("Grade", width="small"),
                    "Why Weak": st.column_config.TextColumn("Why Weak", width="large"),
                }
            )
            st.caption("🚫 **Trading Action:** No allocation - stay away until lifecycle improves")
            
            st.markdown("""
            **Why These Are 'Weak':**
            - Low score (<40), OR
            - Fewer than 2 timeframes bullish, OR
            - All showing Lagging
            """)
    

    st.markdown("---")

    # --- 8. STOCK EXPLORER ---
    st.subheader(f"🔎 Explorer: Theme Drilldown")
    
    # Search functionality
    search_t = st.text_input(
        "Input a ticker to find its theme(s)",
        placeholder="NVDA..."
    ).strip().upper()
    
    if search_t:
        matches = uni_df[uni_df['Ticker'] == search_t]
        if not matches.empty:
            found = matches['Theme'].unique()
            st.success(f"📍 Found **{search_t}** in: **{', '.join(found)}**")
            if len(found) > 0:
                st.session_state.sector_target = found[0]
        else:
            st.warning(f"Ticker {search_t} not found.")

    # Theme selector with immediate update
    curr_idx = all_themes.index(st.session_state.sector_target) \
        if st.session_state.sector_target in all_themes else 0
    
    def update_theme():
        st.session_state.sector_target = st.session_state.theme_selector
    
    new_target = st.selectbox(
        "Select Theme to View Stocks", 
        all_themes, 
        index=curr_idx,
        key="theme_selector",
        on_change=update_theme
    )

    st.markdown("---")

    # --- 9. STOCK ANALYSIS WITH SCORING HELP ---
    st.subheader(f"📊 {st.session_state.sector_target} - Stock Analysis")
    
    # Help section - MORE PROMINENT
    st.info("💡 **Stocks are ranked by comprehensive score:** Alpha Performance (40%) + Volume Confirmation (20%) + Technical Position (20%) + Theme Alignment (20%)")
    
    col_help1, col_help2, col_help3 = st.columns([1, 1, 1])
    with col_help1:
        st.markdown("**📊 Grades:** A (80+) • B (70-79) • C (60-69) • D/F (Avoid)")
    with col_help2:
        st.markdown("**🎯 Patterns:** 🚀 Breakout • 💎 Dip Buy • ⚠️ Fading")
    with col_help3:
        with st.popover("📖 How Scoring Works", use_container_width=True):
            st.markdown("""
            ### Quick Reference
            
            **Score Breakdown:**
            - 40 pts: Alpha (beating sector?)
            - 20 pts: Volume (institutions buying?)
            - 20 pts: Technicals (uptrend?)
            - 20 pts: Theme Alignment (sector strong?)
            
            **Pattern Bonuses:**
            - 🚀 Breakout: +10 pts
            - 💎 Dip Buy: +5 pts
            - 📈 Bullish Divergence: +5 pts
            - 📉 Bearish Divergence: -10 pts
            """)
            
            st.markdown("---")
            
            if st.button("📖 View Complete Guide", use_container_width=True):
                st.session_state.show_full_guide = True
                st.rerun()

    # Show full guide if requested
    if st.session_state.get('show_full_guide', False):
        with st.expander("📖 Complete Scoring & Pattern Guide", expanded=True):
            if st.button("✖️ Close Guide"):
                st.session_state.show_full_guide = False
                st.rerun()
            
            try:
                with open("SCORING_GUIDE.md", "r") as f:
                    st.markdown(f.read())
            except FileNotFoundError:
                st.error("SCORING_GUIDE.md not found. Please ensure it's in the repo root directory.")
    
    # Get theme ETF for quadrant status
    theme_etf_ticker = theme_map.get(st.session_state.sector_target)
    theme_df = etf_data_cache.get(theme_etf_ticker)
    theme_quadrant = us.get_quadrant_status(theme_df, 'Short') if theme_df is not None else "N/A"
    
    # Filter stocks for current theme
    stock_tickers = uni_df[
        (uni_df['Theme'] == st.session_state.sector_target) & 
        (uni_df['Role'] == 'Stock')
    ]['Ticker'].tolist()
    
    if not stock_tickers:
        st.info(f"No stocks found for {st.session_state.sector_target}")
        return
    
    # Build ranking data with all new features
    ranking_data = []
    
    with st.spinner(f"Analyzing {len(stock_tickers)} stocks..."):
        for stock in stock_tickers:
            sdf = etf_data_cache.get(stock)
            
            if sdf is None or sdf.empty:
                continue
            
            try:
                # Volume filter
                if len(sdf) < 20:
                    continue
                
                avg_vol = sdf['Volume'].tail(20).mean()
                avg_price = sdf['Close'].tail(20).mean()
                
                if (avg_vol * avg_price) < us.MIN_DOLLAR_VOLUME:
                    continue
                
                last = sdf.iloc[-1]
                
                # Get theme-specific alpha columns
                alpha_5d = last.get(f"Alpha_Short_{st.session_state.sector_target}", 0)
                alpha_10d = last.get(f"Alpha_Med_{st.session_state.sector_target}", 0)
                alpha_20d = last.get(f"Alpha_Long_{st.session_state.sector_target}", 0)
                beta = last.get(f"Beta_{st.session_state.sector_target}", 1.0)
                
                # Pattern detection
                breakout = us.detect_breakout_candidates(sdf, st.session_state.sector_target)
                dip_buy = us.detect_dip_buy_candidates(sdf, st.session_state.sector_target)
                fading = us.detect_fading_candidates(sdf, st.session_state.sector_target)
                
                # UPDATED: Use Shared Divergence Logic (passed ticker)
                divergence = us.detect_relative_strength_divergence(sdf, stock)
                
                # Comprehensive score
                score_data = us.calculate_comprehensive_stock_score(
                    sdf,
                    st.session_state.sector_target,
                    theme_quadrant
                )
                
                # Determine pattern label
                pattern = ""
                if breakout:
                    pattern = f"🚀 Breakout ({breakout['strength']:.0f})"
                elif dip_buy:
                    pattern = "💎 Dip Buy"
                elif fading:
                    pattern = "⚠️ Fading"
                
                # Divergence indicator
                div_label = ""
                if divergence == 'bullish_divergence':
                    div_label = "📈 Bull Div"
                elif divergence == 'bearish_divergence':
                    div_label = "📉 Bear Div"
                
                # LOOKUP LAST TRADE
                last_trade_date = last_trade_map.get(stock, "—")
                
                ranking_data.append({
                    "Ticker": stock,
                    "Score": score_data['total_score'] if score_data else 0,
                    "Grade": score_data['grade'] if score_data else 'F',
                    "Price": last['Close'],
                    "Beta": beta,
                    "Alpha 5d": alpha_5d,
                    "Alpha 10d": alpha_10d,
                    "Alpha 20d": alpha_20d,
                    "RVOL 5d": last.get('RVOL_Short', 0),
                    "RVOL 10d": last.get('RVOL_Med', 0),
                    "RVOL 20d": last.get('RVOL_Long', 0),
                    "Pattern": pattern,
                    "Divergence": div_label,
                    "Last Option": last_trade_date, # NEW COLUMN
                    "8 EMA": us.get_ma_signal(last['Close'], last.get('Ema8', 0)),
                    "21 EMA": us.get_ma_signal(last['Close'], last.get('Ema21', 0)),
                    "50 MA": us.get_ma_signal(last['Close'], last.get('Sma50', 0)),
                    "200 MA": us.get_ma_signal(last['Close'], last.get('Sma200', 0)),
                    # Hidden columns for filtering
                    "_breakout": breakout is not None,
                    "_dip_buy": dip_buy,
                    "_fading": fading
                })
                
            except Exception as e:
                # st.error(f"Error processing {stock}: {e}")
                continue

    if not ranking_data:
        st.info(f"No stocks found for {st.session_state.sector_target} (or filtered by volume).")
        return
    
    df_ranked = pd.DataFrame(ranking_data).sort_values(by='Score', ascending=False)
    
    # --- 10. TABBED DISPLAY WITH SMART FILTERS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 All Stocks",
        "🚀 Breakouts",
        "💎 Dip Buys",
        "⚠️ Faders"
    ])
    
    # Display columns (excluding hidden filter columns)
    display_cols = [c for c in df_ranked.columns if not c.startswith('_')]
    
    with tab1:
        st.caption(f"Showing {len(df_ranked)} stocks sorted by comprehensive score")
        
        # Highlight function
        def highlight_top_scores(row):
            styles = pd.Series('', index=row.index)
            score = row.get('Score', 0)
            
            if score >= 80:
                styles['Score'] = 'background-color: #d4edda; color: #155724; font-weight: bold;'
                styles['Grade'] = 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif score >= 70:
                styles['Score'] = 'background-color: #cce5ff; color: #004085;'
                styles['Grade'] = 'background-color: #cce5ff; color: #004085;'
            
            # Highlight alpha columns
            for col in ['Alpha 5d', 'Alpha 10d', 'Alpha 20d']:
                if col in row.index:
                    alpha = row[col]
                    if alpha > 2.0:
                        styles[col] = 'background-color: #d4edda; color: #155724;'
                    elif alpha < -2.0:
                        styles[col] = 'background-color: #f8d7da; color: #721c24;'
            
            return styles
        
        st.dataframe(
            df_ranked[display_cols].style.apply(highlight_top_scores, axis=1),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Grade": st.column_config.TextColumn("Grade", width="small"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                "Alpha 5d": st.column_config.NumberColumn("Alpha 5d", format="%+.2f%%"),
                "Alpha 10d": st.column_config.NumberColumn("Alpha 10d", format="%+.2f%%"),
                "Alpha 20d": st.column_config.NumberColumn("Alpha 20d", format="%+.2f%%"),
                "RVOL 5d": st.column_config.NumberColumn("RVOL 5d", format="%.1fx"),
                "RVOL 10d": st.column_config.NumberColumn("RVOL 10d", format="%.1fx"),
                "RVOL 20d": st.column_config.NumberColumn("RVOL 20d", format="%.1fx"),
                "Divergence": st.column_config.TextColumn("Divergence", width="small"),
                "Last Option": st.column_config.TextColumn("Last Option", width="medium"),
            }
        )
    
    with tab2:
        breakouts = df_ranked[df_ranked['_breakout'] == True]
        
        if not breakouts.empty:
            st.success(f"🚀 Found {len(breakouts)} breakout candidates")
            st.caption("Stocks transitioning from underperformance to outperformance with volume confirmation")
            
            st.dataframe(
                breakouts[display_cols],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                    "Alpha 5d": st.column_config.NumberColumn("Alpha 5d", format="%+.2f%%"),
                    "RVOL 5d": st.column_config.NumberColumn("RVOL 5d", format="%.1fx"),
                }
            )
        else:
            st.info("No breakout patterns detected currently")
    
    with tab3:
        dip_buys = df_ranked[df_ranked['_dip_buy'] == True]
        
        if not dip_buys.empty:
            st.success(f"💎 Found {len(dip_buys)} dip buy opportunities")
            st.caption("Stocks that were outperforming but pulled back to average - potential buy-the-dip setups")
            
            st.dataframe(
                dip_buys[display_cols],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No dip buy setups currently")
    
    with tab4:
        faders = df_ranked[df_ranked['_fading'] == True]
        
        if not faders.empty:
            st.warning(f"⚠️ {len(faders)} stocks showing weakness")
            st.caption("Stocks that were very strong but alpha is declining - consider taking profits")
            
            st.dataframe(
                faders[display_cols],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success("✅ No concerning faders detected")