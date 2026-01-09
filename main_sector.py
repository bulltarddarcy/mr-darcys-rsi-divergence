import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils_sector as us

# ==========================================
# UI HELPERS
# ==========================================
def classify_setup(df):
    """Classifies the setup (J-Hook, Bull Flag, Rocket)"""
    if df is None or df.empty: return None
    last = df.iloc[-1]
    if "RRG_Mom_Short" not in last or "RRG_Mom_Long" not in last: return None

    m5 = last["RRG_Mom_Short"]
    m10 = last.get("RRG_Mom_Med", 0)
    m20 = last["RRG_Mom_Long"]
    ratio_20 = last.get("RRG_Ratio_Long", 100)

    if m20 < 100 and m5 > 100 and m5 > (m20 + 2): return "🪝 J-Hook"
    if ratio_20 > 100 and m5 > 100 and m5 > m10: return "🚩 Bull Flag"
    if m5 > m10 and m10 > m20 and m20 > 100: return "🚀 Rocket"
    return None 

def get_quadrant_status(df, timeframe_key):
    """Returns just the icon and label for the table"""
    if df is None or df.empty: return "N/A"
    
    col_r = f"RRG_Ratio_{timeframe_key}"
    col_m = f"RRG_Mom_{timeframe_key}"
    
    if col_r not in df.columns: return "N/A"
    
    r = df[col_r].iloc[-1]
    m = df[col_m].iloc[-1]
    
    if r >= 100 and m >= 100: return "🟢 Leading"
    elif r < 100 and m >= 100: return "🔵 Improving"
    elif r < 100 and m < 100: return "🔴 Lagging"
    else: return "🟡 Weakening"

def get_ma_signal(price, ma_val):
    """Returns Emoji based on Price vs MA"""
    if pd.isna(ma_val) or ma_val == 0:
        return "⚠️" # Not enough data
    return "✅" if price > ma_val else "❌"

def plot_simple_rrg(dm, target_map, view_key, show_trails):
    fig = go.Figure()
    all_x, all_y = [], []
    
    for theme, ticker in target_map.items():
        df = dm.load_ticker_data(ticker)
        if df is None or df.empty: continue
        col_x, col_y = f"RRG_Ratio_{view_key}", f"RRG_Mom_{view_key}"
        if col_x not in df.columns: continue
        
        data_slice = df.tail(3) if show_trails else df.tail(1)
        if data_slice.empty: continue

        x_vals = data_slice[col_x].tolist()
        y_vals = data_slice[col_y].tolist()
        all_x.extend(x_vals); all_y.extend(y_vals)
        
        last_x, last_y = x_vals[-1], y_vals[-1]
        if last_x > 100 and last_y > 100: color = '#00CC96' 
        elif last_x < 100 and last_y > 100: color = '#636EFA'
        elif last_x > 100 and last_y < 100: color = '#FFA15A'
        else: color = '#EF553B'
        
        n = len(x_vals)
        sizes = [8] * (n - 1) + [15]
        opacities = [0.4] * (n - 1) + [1.0]
        texts = [""] * (n - 1) + [theme]
        custom_data = [theme] * n

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers+text', name=theme, text=texts,
            customdata=custom_data, textposition="top center",
            marker=dict(size=sizes, color=color, opacity=opacities, line=dict(width=1, color='white')),
            line=dict(color=color, width=1 if show_trails else 0, shape='spline', smoothing=1.3),
            hoverinfo='text+name',
            hovertext=[f"{theme}<br>Trend: {x:.1f}<br>Mom: {y:.1f}" for x,y in zip(x_vals, y_vals)]
        ))

    # Scaling
    if all_x and all_y:
        limit_x = max(max([abs(x - 100) for x in all_x]) * 1.1, 2.0)
        limit_y = max(max([abs(y - 100) for y in all_y]) * 1.1, 2.0)
        x_range = [100 - limit_x, 100 + limit_x]
        y_range = [100 - limit_y, 100 + limit_y]
    else:
        x_range, y_range = [98, 102], [98, 102]
        limit_x, limit_y = 2, 2

    fig.add_hline(y=100, line_width=1, line_color="gray", line_dash="dash")
    fig.add_vline(x=100, line_width=1, line_color="gray", line_dash="dash")
    
    lbl_x, lbl_y = limit_x * 0.5, limit_y * 0.5
    
    def add_hud_label(x, y, text, color):
        fig.add_annotation(
            x=x, y=y, text=f"<b>{text}</b>", showarrow=False, 
            font=dict(color=color, size=20)
        )

    add_hud_label(100+lbl_x, 100+lbl_y, "LEADING", "rgba(0, 255, 0, 0.7)")
    add_hud_label(100-lbl_x, 100+lbl_y, "IMPROVING", "rgba(0, 100, 255, 0.7)")
    add_hud_label(100+lbl_x, 100-lbl_y, "WEAKENING", "rgba(255, 165, 0, 0.7)")
    add_hud_label(100-lbl_x, 100-lbl_y, "LAGGING", "rgba(255, 0, 0, 0.7)")

    fig.update_layout(
        xaxis=dict(title="Relative Trend", showgrid=False, range=x_range, constrain='domain'),
        yaxis=dict(title="Relative Momentum", showgrid=False, range=y_range),
        height=750, showlegend=False, template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_sector_rotation_app(df_global=None):
    st.title("🔄 Sector Rotation")
    
    # 1. Init Data
    dm = us.SectorDataManager()
    uni_df, tickers, theme_map = dm.load_universe()
    
    if uni_df.empty:
        st.warning("⚠️ SECTOR_UNIVERSE secret is missing or empty.")
        return

    # 2. Session State for Controls
    if "sector_view" not in st.session_state: st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state: st.session_state.sector_trails = False
    if "sector_target" not in st.session_state: st.session_state.sector_target = sorted(list(theme_map.keys()))[0] if theme_map else ""
    
    # Ensure filter list is init for the widget
    all_themes = sorted(list(theme_map.keys()))
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- MAIN SECTION START ---
    st.subheader("Sector Rotations")

    # 1. GRAPHIC USER GUIDE (Moved Here)
    with st.expander("Graphic User Guide", expanded=False):
        st.markdown("""
        **🧮 How It Works (The Math)**
        This chart does **not** show price. It shows **Relative Performance** against the S&P 500 (SPY).
        * **X-Axis (Trend):** Are we beating the market?
            * `> 100`: Outperforming the S&P 500.
            * `< 100`: Underperforming the S&P 500.
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

    # 2. CONTROLS
    with st.expander("⚙️ Chart Inputs & Filters", expanded=True):
        
        # --- INPUT ROW 1: Toggles & Trails ---
        c_in_1, c_in_2, c_spacer = st.columns([1.5, 1, 3]) 
        
        timeframe_help = """
        ⏱️ Timeframe Definitions:
        • 5 Days (Short): Tactical View (~1 Week). Highly sensitive.
        • 10 Days (Med): Balanced View (~2 Weeks).
        • 20 Days (Long): Strategic View (~1 Month). Primary trend.
        """
        trails_help = "Shows the trailing path of the last 3 days to visualize velocity and direction changes (e.g. J-Hooks)."

        with c_in_1:
            st.session_state.sector_view = st.radio(
                "Timeframe Window", ["5 Days", "10 Days", "20 Days"], 
                horizontal=True, 
                key="timeframe_radio",
                help=timeframe_help
            )
        
        with c_in_2:
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            st.session_state.sector_trails = st.checkbox(
                "Show 3-Day Trails", 
                value=st.session_state.sector_trails,
                help=trails_help
            )
            
        # --- INPUT ROW 2: Update Button ---
        if st.button("🔄 Update Data", use_container_width=True):
            status = st.empty()
            calc = us.SectorAlphaCalculator()
            calc.run_full_update(status)
            st.rerun()

        st.divider()

        # --- SECTORS SHOWN ---
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
            "Select Themes", 
            all_themes, 
            key="sector_theme_filter_widget", 
            label_visibility="collapsed"
        )
    
    filtered_map = {k: v for k, v in theme_map.items() if k in sel_themes}
    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    view_key = timeframe_map[st.session_state.sector_view]

    # --- MOMENTUM SCANS ---
    with st.expander("🚀 Momentum Scans", expanded=True):
        
        with st.expander("ℹ️ Legend & Setup Key", expanded=False):
             st.markdown("""
             **Category Guide**
             * **📈 Increasing:** 5-Day Momentum > 20-Day Momentum. The move is accelerating.
             * **🔻 Decreasing:** 5-Day Momentum < 20-Day Momentum. The move is slowing down.
             * **⚖️ Neutral:** Mixed signals.
             
             **Setup Key**
             * **🪝 J-Hook:** Long-term weak, Short-term exploding. (Bottom Fish).
             * **🚩 Bull Flag:** Long-term strong, Short-term rested. (Dip Buy).
             * **🚀 Rocket:** Perfect alignment 5>10>20. (Thrust).
             """)

        # --- COLUMNS ---
        inc_mom, neut_mom, dec_mom = [], [], []
        
        for theme, ticker in theme_map.items():
            df = dm.load_ticker_data(ticker)
            if df is None or df.empty or "RRG_Mom_Short" not in df.columns: continue
            last = df.iloc[-1]
            m5, m10, m20 = last.get("RRG_Mom_Short",0), last.get("RRG_Mom_Med",0), last.get("RRG_Mom_Long",0)
            
            # Differential Score (5d - 20d)
            shift = m5 - m20
            
            setup = classify_setup(df)
            icon = setup.split()[0] if setup else ""
            item = {"theme": theme, "shift": shift, "icon": icon}
            
            if m5 > m10 > m20: inc_mom.append(item)
            elif m5 < m10 < m20: dec_mom.append(item)
            else: neut_mom.append(item)

        # Sort Logic
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
        fig = plot_simple_rrg(dm, filtered_map, view_key, st.session_state.sector_trails)
        chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
    
    # Handle Chart Selection
    if chart_event and chart_event.selection and chart_event.selection.points:
        point = chart_event.selection.points[0]
        if "customdata" in point:
            st.session_state.sector_target = point["customdata"]
        elif "text" in point:
            st.session_state.sector_target = point["text"]
    
    st.divider()

    # --- 3. ALL THEMES PERFORMANCE ---
    st.subheader("All Themes Performance")
    
    with st.expander("ℹ️ Legend & Definitions", expanded=False):
        st.markdown("""
        **Metric Definitions**
        * **Rel Perf (Ratio):** Normalized Relative Strength vs SPY. (Above 0 = Outperforming).
        * **MoM (Momentum):** Rate of Change of the Ratio. (Above 0 = Accelerating).
        """)

    summary_data = []
    for theme in all_themes:
        etf_ticker = theme_map.get(theme)
        if not etf_ticker: continue
        etf_df = dm.load_ticker_data(etf_ticker)
        
        if etf_df is None or etf_df.empty: continue
        
        last = etf_df.iloc[-1]
        
        row = {"Theme": theme}
        
        # Calculate Deviation from 100
        
        # Short (5d)
        row["Status (5d)"] = get_quadrant_status(etf_df, "Short")
        row["Rel Perf (5d)"] = last.get("RRG_Ratio_Short", 100) - 100
        row["Mom (5d)"] = last.get("RRG_Mom_Short", 100) - 100
        
        # Med (10d)
        row["Status (10d)"] = get_quadrant_status(etf_df, "Med")
        row["Rel Perf (10d)"] = last.get("RRG_Ratio_Med", 100) - 100
        row["Mom (10d)"] = last.get("RRG_Mom_Med", 100) - 100
        
        # Long (20d)
        row["Status (20d)"] = get_quadrant_status(etf_df, "Long")
        row["Rel Perf (20d)"] = last.get("RRG_Ratio_Long", 100) - 100
        row["Mom (20d)"] = last.get("RRG_Mom_Long", 100) - 100

        summary_data.append(row)
        
    if summary_data:
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Theme": st.column_config.TextColumn("Theme"), 
                "Rel Perf (5d)": st.column_config.NumberColumn("Rel Perf (5d)", format="%+.2f%%", help="Relative Strength vs SPY (5 Days)"),
                "Mom (5d)": st.column_config.NumberColumn("Mom (5d)", format="%+.2f", help="Momentum Velocity (5 Days)"),
                "Rel Perf (10d)": st.column_config.NumberColumn("Rel Perf (10d)", format="%+.2f%%", help="Relative Strength vs SPY (10 Days)"),
                "Mom (10d)": st.column_config.NumberColumn("Mom (10d)", format="%+.2f", help="Momentum Velocity (10 Days)"),
                "Rel Perf (20d)": st.column_config.NumberColumn("Rel Perf (20d)", format="%+.2f%%", help="Relative Strength vs SPY (20 Days)"),
                "Mom (20d)": st.column_config.NumberColumn("Mom (20d)", format="%+.2f", help="Momentum Velocity (20 Days)"),
            }
        )

    st.markdown("---")

    # --- 4. EXPLORER SECTION ---
    st.subheader(f"🔎 Explorer: {st.session_state.sector_target}")
    
    # 1. Ticker Input
    search_t = st.text_input("Not sure which theme? Input a ticker to find its theme(s)", placeholder="NVDA...").strip().upper()
    if search_t:
        matches = uni_df[uni_df['Ticker'] == search_t]
        if not matches.empty:
            found = matches['Theme'].unique()
            st.success(f"📍 Found **{search_t}** in: **{', '.join(found)}**")
            if len(found) > 0: st.session_state.sector_target = found[0]
        else:
            st.warning(f"Ticker {search_t} not found in the Sector Universe.")

    # 2. Dropdown Selector
    curr_idx = all_themes.index(st.session_state.sector_target) if st.session_state.sector_target in all_themes else 0
    new_target = st.selectbox("Select Theme to View Stocks", all_themes, index=curr_idx)
    if new_target != st.session_state.sector_target:
        st.session_state.sector_target = new_target

    # --- STOCK TABLE ---
    stock_tickers = uni_df[(uni_df['Theme'] == st.session_state.sector_target) & (uni_df['Role'] == 'Stock')]['Ticker'].tolist()
    ranking_data = []
    
    for stock in stock_tickers:
        sdf = dm.load_ticker_data(stock)
        
        if sdf is None or sdf.empty: continue
        
        try:
            # Volume Filter
            avg_vol = sdf['Volume'].tail(20).mean()
            avg_price = sdf['Close'].tail(20).mean()
            if (avg_vol * avg_price) < us.MIN_DOLLAR_VOLUME: continue
            
            last = sdf.iloc[-1]
            
            # Helper to safely get value for styling logic
            def safe_get(key, default=0):
                return last.get(key, default)

            ranking_data.append({
                "Ticker": stock,
                "Price": last['Close'],
                # Raw floats for conditional logic
                "Alpha 5d": safe_get("True_Alpha_Short"),
                "RVOL 5d": safe_get("RVOL_Short"),
                "Alpha 10d": safe_get("True_Alpha_Med"),
                "RVOL 10d": safe_get("RVOL_Med"),
                "Alpha 20d": safe_get("True_Alpha_Long"),
                "RVOL 20d": safe_get("RVOL_Long"),
                # MA Emojis
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
            # Create a Series of empty styles with index matching row
            styles = pd.Series('', index=row.index)
            color = 'background-color: #d4edda; color: black;'
            
            # Logic A: 5 Days (Alpha > 0 and RVOL > 1.2)
            if row["Alpha 5d"] > 0 and row["RVOL 5d"] > 1.2:
                styles["Alpha 5d"] = color
                styles["RVOL 5d"] = color
                
            # Logic B: 10 Days
            if row["Alpha 10d"] > 0 and row["RVOL 10d"] > 1.2:
                styles["Alpha 10d"] = color
                styles["RVOL 10d"] = color
                
            # Logic C: 20 Days
            if row["Alpha 20d"] > 0 and row["RVOL 20d"] > 1.2:
                styles["Alpha 20d"] = color
                styles["RVOL 20d"] = color
                
            return styles

        st.dataframe(
            df_disp.style.apply(highlight_cells, axis=1),
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Stock Symbol"), 
                "Price": st.column_config.NumberColumn("Price", format="$%.2f", help="Latest Closing Price"),
                
                "Alpha 5d": st.column_config.NumberColumn("Alpha 5d", format="%+.2f%%", help="5-Day Performance vs Sector ETF"),
                "RVOL 5d": st.column_config.NumberColumn("RVOL 5d", format="%.1fx", help="5-Day Relative Volume (vs 20d avg)"),
                
                "Alpha 10d": st.column_config.NumberColumn("Alpha 10d", format="%+.2f%%", help="10-Day Performance vs Sector ETF"),
                "RVOL 10d": st.column_config.NumberColumn("RVOL 10d", format="%.1fx", help="10-Day Relative Volume"),
                
                "Alpha 20d": st.column_config.NumberColumn("Alpha 20d", format="%+.2f%%", help="20-Day Performance vs Sector ETF"),
                "RVOL 20d": st.column_config.NumberColumn("RVOL 20d", format="%.1fx", help="20-Day Relative Volume"),
                
                "8 EMA": st.column_config.TextColumn("8 EMA", width="small", help="Price > 8 EMA"),
                "21 EMA": st.column_config.TextColumn("21 EMA", width="small", help="Price > 21 EMA"),
                "50 MA": st.column_config.TextColumn("50 MA", width="small", help="Price > 50 SMA"),
                "200 MA": st.column_config.TextColumn("200 MA", width="small", help="Price > 200 SMA (⚠️ = Insufficient History)")
            }
        )
    else:
        st.info(f"No stocks found for {st.session_state.sector_target} (or filtered by volume).")