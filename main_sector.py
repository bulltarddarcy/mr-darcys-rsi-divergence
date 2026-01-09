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

def get_quadrant_label(df, timeframe_key):
    """Returns a formatted string like 'Leading 🟢 (+3.3)'"""
    if df is None or df.empty: return "N/A"
    
    col_r = f"RRG_Ratio_{timeframe_key}"
    col_m = f"RRG_Mom_{timeframe_key}"
    
    if col_r not in df.columns: return "N/A"
    
    r = df[col_r].iloc[-1]
    m = df[col_m].iloc[-1]
    
    if r >= 100 and m >= 100: txt, icn = "Leading", "🟢"
    elif r < 100 and m >= 100: txt, icn = "Improving", "🔵"
    elif r < 100 and m < 100: txt, icn = "Lagging", "🔴"
    else: txt, icn = "Weakening", "🟡"
    
    diff = m - 100
    return f"{icn} {txt} ({diff:+.1f})"

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

    # --- TOP GUIDE ---
    with st.expander("📚 Page Information & User Guide", expanded=False):
        st.markdown("""
        **⏱️ Understanding Timeframes**
        * **5 Days (Short):** The Tactical View. (~1 Week). Highly sensitive. Use this for precise entries (buying dips) or exits.
        * **10 Days (Med):** The Balance. (~2 Weeks). A middle ground that smooths out daily noise but reacts faster than the monthly trend.
        * **20 Days (Long):** The Strategic View. (~1 Month). This is your primary trend. Major moves happen here.

        **🧮 How It Works (The Math)**
        This chart does **not** show price. It shows **Relative Performance** against the S&P 500 (SPY).
        * **X-Axis (Trend):** Are we beating the market?
            * `> 100`: Outperforming the S&P 500.
            * `< 100`: Underperforming the S&P 500.
        * **Y-Axis (Momentum):** How fast is the trend changing?
            * `> 100`: Gaining speed (Acceleration).
            * `< 100`: Losing speed (Deceleration).
        
        *Note: Calculations use Weighted Regression, meaning today's price action is weighted 3x more than data from 20 days ago. This eliminates lag.*

        **📊 How to Read the Quadrants**
        * 🟢 **LEADING (Top Right):** Strong Trend + Accelerating Momentum. The Winners.
        * 🟡 **WEAKENING (Bottom Right):** Strong Trend, but losing steam. Often a place to take profits or wait for a pullback.
        * 🔴 **LAGGING (Bottom Left):** Weak Trend + Decelerating. The Losers. Avoid unless hunting for bottoms.
        * 🔵 **IMPROVING (Top Left):** Weak Trend, but Momentum is waking up. This is where "Turnarounds" happen.

        **🐍 Using Tails (History)**
        * **The Dot:** Represents Today's score.
        * **The Tail:** Shows the path of the last 3 days.
        * **Long Tail:** = High Velocity. The move is strong and decisive.
        * **Short Tail:** = Indecision. The sector is stuck.
        * **J-Hook:** If the tail is in the bottom left but "Hooks" sharply up and right, smart money is rotating in aggressively.
        """)

    # 2. Session State for Controls
    if "sector_view" not in st.session_state: st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state: st.session_state.sector_trails = False
    if "sector_target" not in st.session_state: st.session_state.sector_target = sorted(list(theme_map.keys()))[0] if theme_map else ""
    
    # Ensure filter list is init for the widget
    all_themes = sorted(list(theme_map.keys()))
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # 3. CONTROLS (TWO EXPANDERS SIDE BY SIDE)
    c1, c2 = st.columns(2)
    
    # --- COL 1: CHART INPUTS ---
    with c1:
        with st.expander("⚙️ Chart Inputs", expanded=True):
            st.session_state.sector_view = st.radio(
                "Timeframe Window", ["5 Days", "10 Days", "20 Days"], 
                horizontal=True, key="timeframe_radio"
            )
            
            c_sub1, c_sub2 = st.columns([1, 1])
            with c_sub1:
                st.session_state.sector_trails = st.checkbox("Show 3-Day Trails", value=st.session_state.sector_trails)
            with c_sub2:
                if st.button("🔄 Update Data", use_container_width=True):
                    status = st.empty()
                    calc = us.SectorAlphaCalculator()
                    calc.run_full_update(status)
                    st.rerun()

    # --- COL 2: SECTORS SHOWN ---
    with c2:
        with st.expander("👁️ Sectors Shown", expanded=True):
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("➕ Add All", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = all_themes
                    st.rerun()
            with b_col2:
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
        # Increasing: High to Low (+10 before +2)
        inc_mom.sort(key=lambda x: x['shift'], reverse=True)
        # Neutral: High to Low
        neut_mom.sort(key=lambda x: x['shift'], reverse=True)
        # Decreasing: Low to High (-10 before -2) -> "Increasing order" mathematically
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

        # LEGEND INSIDE EXPANDER
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

    # 4. RRG CHART
    st.subheader("Sector Rotations")
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
    
    # 5. ALL THEMES PERFORMANCE
    st.subheader("All Themes Performance")
    
    summary_data = []
    for theme in all_themes:
        etf_ticker = theme_map.get(theme)
        if not etf_ticker: continue
        etf_df = dm.load_ticker_data(etf_ticker)
        
        row = {"Theme": theme}
        row["Short (5d)"] = get_quadrant_label(etf_df, "Short")
        row["Med (10d)"] = get_quadrant_label(etf_df, "Med")
        row["Long (20d)"] = get_quadrant_label(etf_df, "Long")
        summary_data.append(row)
        
    if summary_data:
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True
        )

    # 6. EXPLORER SECTION (NEW TABLE + STOCK LIST)
    st.subheader(f"🔎 Explorer: {st.session_state.sector_target}")
    
    # Search Bar
    search_c1, search_c2 = st.columns([1, 3])
    with search_c1:
        search_t = st.text_input("Input a ticker to find its theme(s)", placeholder="NVDA...").strip().upper()
        if search_t:
            matches = uni_df[uni_df['Ticker'] == search_t]
            if not matches.empty:
                found = matches['Theme'].unique()
                if len(found) > 0: st.session_state.sector_target = found[0]
    
    with search_c2:
        # Update Target Dropdown
        # FIX: 'all_themes' is used here, ensuring it is defined
        curr_idx = all_themes.index(st.session_state.sector_target) if st.session_state.sector_target in all_themes else 0
        new_target = st.selectbox("Select Theme to View Stocks", all_themes, index=curr_idx)
        if new_target != st.session_state.sector_target:
            st.session_state.sector_target = new_target

    # Stock Table
    stock_tickers = uni_df[(uni_df['Theme'] == st.session_state.sector_target) & (uni_df['Role'] == 'Stock')]['Ticker'].tolist()
    ranking_data = []
    
    for stock in stock_tickers:
        sdf = dm.load_ticker_data(stock)
        
        if sdf is None or sdf.empty: continue
        
        # Volume Filter
        try:
            avg_vol = sdf['Volume'].tail(20).mean()
            avg_price = sdf['Close'].tail(20).mean()
            if (avg_vol * avg_price) < us.MIN_DOLLAR_VOLUME: continue
            
            last = sdf.iloc[-1]
            ranking_data.append({
                "Ticker": stock,
                "Price": last['Close'],
                "Alpha 5d": last.get("True_Alpha_Short", 0),
                "RVOL 5d": last.get("RVOL_Short", 0),
                "Alpha 10d": last.get("True_Alpha_Med", 0),
                "RVOL 10d": last.get("RVOL_Med", 0),
                "Alpha 20d": last.get("True_Alpha_Long", 0),
                "RVOL 20d": last.get("RVOL_Long", 0),
                "8 EMA": "✅" if last['Close'] > last.get('EMA_8', 0) else "❌"
            })
        except Exception:
            continue

    if ranking_data:
        df_disp = pd.DataFrame(ranking_data).sort_values(by='Alpha 5d', ascending=False)
        
        def style_rows(row):
            styles = [''] * len(row)
            if row['Alpha 5d'] > 0 and row['RVOL 5d'] > 1.2:
                 return ['background-color: #d4edda; color: black;'] * len(row)
            return styles

        st.dataframe(
            df_disp.style.apply(style_rows, axis=1).format({
                "Price": "$%.2f", 
                "Alpha 5d": "%+.2f%%", "RVOL 5d": "%.1fx",
                "Alpha 10d": "%+.2f%%", "RVOL 10d": "%.1fx",
                "Alpha 20d": "%+.2f%%", "RVOL 20d": "%.1fx"
            }),
            hide_index=True, use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"), 
                "8 EMA": st.column_config.TextColumn("8 EMA", width="small")
            }
        )
    else:
        st.info(f"No stocks found for {st.session_state.sector_target} (or filtered by volume).")