Here is the complete, updated Python script.
Key Updates Made:
 * Sidebar Note: Added the "Best experience on wide desktop/light mode" alert to the sidebar, positioned above the "Last Updated" date (which I have moved to the sidebar to match your request).
 * Navigation Renaming:
   * "Options Database" is now "Database".
   * "RSI Divergences" page title is updated (menu item remains "RSI Divergences").
 * Pivot Tables Layout: I updated the column definition to st.columns([1, 1, 1]). This is the same structure used in the new Rankings page, which tells Streamlit's layout engine to treat them as equal flexible blocks that automatically stack vertically on mobile devices while remaining side-by-side on desktop.
 * Rankings: Includes the new Smart Money (Conviction/Velocity) section side-by-side with the Bulltard section below it.
You can copy-paste this entire block to replace your existing file.
import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import math
import os
import glob
import streamlit.components.v1 as components
import requests
import time

# --- 1. GLOBAL DATA LOADING & UTILITIES ---
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=65),
    "Strike": st.column_config.TextColumn("Strike", width=95),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=90),
    "Contracts": st.column_config.NumberColumn("Qty", width=60),
    "Dollars": st.column_config.NumberColumn("Dollars", width=110),
}

@st.cache_data(ttl=600, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    want = ["Trade Date","Order Type","Symbol","Strike (Actual)","Strike","Expiry","Contracts","Dollars","Error"]
    keep = [c for c in want if c in df.columns]
    df = df[keep].copy()
    
    for col in ["Order Type", "Symbol", "Strike", "Expiry"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    if "Dollars" in df.columns:
        df["Dollars"] = (df["Dollars"].astype(str)
                         .str.replace("$", "", regex=False)
                         .str.replace(",", "", regex=False))
        df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)

    if "Contracts" in df.columns:
        df["Contracts"] = (df["Contracts"].astype(str)
                           .str.replace(",", "", regex=False))
        df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(0)
    
    if "Trade Date" in df.columns:
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    
    if "Expiry" in df.columns:
        df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
        
    if "Strike (Actual)" in df.columns:
        df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
        
    if "Error" in df.columns:
        df = df[~df["Error"].astype(str).str.upper().isin(["TRUE","1","YES"])]
        
    return df

@st.cache_data(ttl=3600)
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        try:
            return float(t.fast_info['marketCap'])
        except:
            pass
        return float(t.info.get('marketCap', 0))
    except:
        return 0.0

@st.cache_data(ttl=300)
def is_above_ema21(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        h = ticker.history(period="60d")
        if len(h) < 21:
            return True 
        ema21 = h["Close"].ewm(span=21, adjust=False).mean()
        latest_price = h["Close"].iloc[-1]
        latest_ema = ema21.iloc[-1]
        return latest_price > latest_ema
    except:
        return True

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

def highlight_expiry(val):
    try:
        expiry_date = datetime.strptime(val, "%d %b %y").date()
        today = date.today()
        this_fri = today + timedelta(days=(4 - today.weekday()) % 7)
        next_fri = this_fri + timedelta(days=7)
        two_fri = this_fri + timedelta(days=14)
        if expiry_date < today: return "" 
        
        if expiry_date == this_fri: return "background-color: #b7e1cd; color: black;" 
        elif expiry_date == next_fri: return "background-color: #fce8b2; color: black;" 
        elif expiry_date == two_fri: return "background-color: #f4c7c3; color: black;" 
        return ""
    except: return ""

def clean_strike_fmt(val):
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid_dates = df["Trade Date"].dropna()
        if not valid_dates.empty:
            return valid_dates.max().date()
    return date.today() - timedelta(days=1)

def render_page_header(title):
    # Modified to remove the date from here since we moved it to sidebar
    st.markdown(f"<h1 style='margin-bottom: 20px;'>{title}</h1>", unsafe_allow_html=True)

# --- 2. APP MODULES ---

def run_database_app(df):
    render_page_header("📂 Database")
    max_data_date = get_max_trade_date(df)
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        default_ticker = st.session_state.get("db_ticker", "")
        db_ticker = st.text_input("Ticker", value=default_ticker.upper(), key="db_ticker_input").strip().upper()
        st.session_state["db_ticker"] = db_ticker
    with c2: start_date = st.date_input("Trade Start Date", value=max_data_date, key="db_start")
    with c3: end_date = st.date_input("Trade End Date", value=max_data_date, key="db_end")
    with c4:
        exp_range_default = (date.today() + timedelta(days=365))
        db_exp_end = st.date_input("Expiration Range (end)", value=exp_range_default, key="db_exp")
    
    check_cols = st.columns([0.15, 0.15, 0.15, 0.55])
    with check_cols[0]: inc_cb = st.checkbox("Calls Bought", value=True, key="db_inc_cb")
    with check_cols[1]: inc_ps = st.checkbox("Puts Sold", value=True, key="db_inc_ps")
    with check_cols[2]: inc_pb = st.checkbox("Puts Bought", value=True, key="db_inc_pb")
    st.markdown('</div>', unsafe_allow_html=True)
    
    f = df.copy()
    if db_ticker: f = f[f["Symbol"].astype(str).str.upper().eq(db_ticker)]
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    if db_exp_end: f = f[f["Expiry_DT"].dt.date <= db_exp_end]
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed_types = []
    if inc_cb: allowed_types.append("Calls Bought")
    if inc_pb: allowed_types.append("Puts Bought")
    if inc_ps: allowed_types.append("Puts Sold")
    f = f[f[order_type_col].isin(allowed_types)]
    
    if f.empty:
        st.warning("No data found matching these filters.")
        return
        
    f = f.sort_values(by=["Trade Date", "Symbol"], ascending=[False, True])
    display_cols = ["Trade Date", order_type_col, "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]
    f_display = f[display_cols].copy()
    f_display["Trade Date"] = f_display["Trade Date"].dt.strftime("%d %b %y")
    f_display["Expiry"] = pd.to_datetime(f_display["Expiry"]).dt.strftime("%d %b %y")
    
    def highlight_db_order_type(val):
        if val in ["Calls Bought", "Puts Sold"]: return 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
        elif val == "Puts Bought": return 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'
        return ''
        
    st.subheader("Non-Expired Trades")
    st.caption("⚠️ User should check OI to confirm trades are still open")
    st.dataframe(f_display.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}).applymap(highlight_db_order_type, subset=[order_type_col]), use_container_width=True, hide_index=True, height=get_table_height(f_display, max_rows=30))

def run_rankings_app(df):
    render_page_header("🏆 Rankings & Flow")
    max_data_date = get_max_trade_date(df)
    
    # --- 1. FILTERS ---
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c_pad = st.columns([1.2, 1.2, 0.8, 3], gap="small")
    with c1: rank_start = st.date_input("Trade Start Date", value=max_data_date - timedelta(days=14), key="rank_start")
    with c2: rank_end = st.date_input("Trade End Date", value=max_data_date, key="rank_end")
    with c3: limit = st.number_input("Limit", value=15, min_value=1, max_value=200, key="rank_limit")
    st.markdown('</div>', unsafe_allow_html=True)
    
    f = df.copy()
    if rank_start: f = f[f["Trade Date"].dt.date >= rank_start]
    if rank_end: f = f[f["Trade Date"].dt.date <= rank_end]
    
    if f.empty:
        st.warning("No data found matching these dates.")
        return

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f_filtered = f[f[order_type_col].isin(target_types)].copy()
    
    if f_filtered.empty:
        st.warning("No trades found in this range.")
        return

    # --- 2. CALCULATE SMART MONEY METRICS ---
    three_days_ago = pd.to_datetime(max_data_date - timedelta(days=3))
    
    # Total Sentiment Dollars
    dollars = f_filtered.groupby(["Symbol", order_type_col])["Dollars"].sum().unstack(fill_value=0)
    for col in target_types:
        if col not in dollars.columns: dollars[col] = 0
        
    # Recent Flow (3-day) for Velocity calculation
    recent_f = f_filtered[f_filtered["Trade Date"] >= three_days_ago]
    recent_dollars = recent_f.groupby("Symbol")["Dollars"].sum()

    smart_df = pd.DataFrame(index=dollars.index)
    smart_df["Total_D"] = dollars.sum(axis=1)
    smart_df["Bull_D"] = dollars["Calls Bought"] + dollars["Puts Sold"]
    smart_df["Bear_D"] = dollars["Puts Bought"]
    smart_df["Net_D"] = smart_df["Bull_D"] - smart_df["Bear_D"]
    
    # Conviction: How one-sided is the money? (0-100%)
    smart_df["Conviction"] = (smart_df["Net_D"].abs() / smart_df["Total_D"]) * 100
    
    # Velocity: Recent flow vs average daily flow
    avg_daily = smart_df["Total_D"] / 14
    smart_df["Velocity"] = recent_dollars / (avg_daily * 3).replace(0, np.nan)
    smart_df["Velocity"] = smart_df["Velocity"].fillna(0)

    # Prepare Tables
    smart_res = smart_df.reset_index()
    
    # --- 3. RENDER SMART RANKINGS ---
    st.markdown("## 🧠 Smart Money Analysis")
    st.info("""
    **Net Bullish/Bearish:** Ranked by Net Dollars (Conviction). It measures institutional weight rather than just trade count.  
    **Momentum:** Ranked by 'Flow Velocity'. Highlights tickers seeing a sudden surge in volume relative to their 14-day average.
    """)

    # Side-by-side for Desktop, Auto-stacks on Mobile
    smart_col1, smart_col2, smart_col3 = st.columns([1, 1, 1], gap="medium")
    
    smart_config = {
        "Symbol": st.column_config.TextColumn("Sym", width=50),
        "Net_D": st.column_config.NumberColumn("Net $", format="$%.0f"),
        "Conviction": st.column_config.NumberColumn("Conv %", format="%.0f%%"),
        "Velocity": st.column_config.NumberColumn("Vel", format="%.1fx")
    }

    with smart_col1:
        st.markdown("<h3 style='color: #71d28a; font-size: 1.1rem;'>Net Bullish</h3>", unsafe_allow_html=True)
        bull_smart = smart_res[smart_res["Net_D"] > 0].sort_values("Net_D", ascending=False).head(limit)
        st.dataframe(bull_smart[["Symbol", "Net_D", "Conviction"]], use_container_width=True, hide_index=True, column_config=smart_config, height=get_table_height(bull_smart))

    with smart_col2:
        st.markdown("<h3 style='color: #f29ca0; font-size: 1.1rem;'>Net Bearish</h3>", unsafe_allow_html=True)
        bear_smart = smart_res[smart_res["Net_D"] < 0].sort_values("Net_D", ascending=True).head(limit)
        st.dataframe(bear_smart[["Symbol", "Net_D", "Conviction"]], use_container_width=True, hide_index=True, column_config=smart_config, height=get_table_height(bear_smart))

    with smart_col3:
        st.markdown("<h3 style='color: #66b7ff; font-size: 1.1rem;'>Momentum (Velocity)</h3>", unsafe_allow_html=True)
        mom_smart = smart_res.sort_values("Velocity", ascending=False).head(limit)
        st.dataframe(mom_smart[["Symbol", "Velocity", "Total_D"]], use_container_width=True, hide_index=True, column_config={
            "Symbol": "Sym", "Velocity": st.column_config.NumberColumn("Vel", format="%.1fx"), "Total_D": st.column_config.NumberColumn("Total $", format="$%.0f")
        }, height=get_table_height(mom_smart))

    st.markdown("---")

    # --- 4. RENDER BULLTARD RANKINGS (ORIGINAL) ---
    st.markdown("## 💎 Bulltard Rankings")
    st.caption("ℹ️ Ranking tables vary from Bulltard's as he includes expired trades and these do not.")
    
    counts = f_filtered.groupby(["Symbol", order_type_col]).size().unstack(fill_value=0)
    for col in target_types:
        if col not in counts.columns: counts[col] = 0
    
    legacy_df = pd.DataFrame(index=counts.index)
    legacy_df["Score"] = counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"]
    legacy_df["Qty"] = counts.sum(axis=1)
    legacy_df["Dollars"] = smart_df["Net_D"] # Use net dollars from smart calc for sorting
    
    res_legacy = legacy_df.reset_index()
    
    legacy_col_left, legacy_col_right = st.columns(2, gap="large")
    rank_col_config = {
        "Symbol": st.column_config.TextColumn("Sym", width=40),
        "Qty": st.column_config.NumberColumn("Qty", width=40),
        "Dollars": st.column_config.NumberColumn("Dollars", width=90, format="$%.0f"),
        "Score": st.column_config.NumberColumn("Score", width=40),
    }

    with legacy_col_left:
        st.markdown("<h3 style='color: #71d28a; font-size: 1rem;'>Bullish</h3>", unsafe_allow_html=True)
        b_leg = res_legacy.sort_values(by=["Score", "Dollars"], ascending=[False, False]).head(limit)
        st.dataframe(b_leg[["Symbol", "Qty", "Dollars", "Score"]], use_container_width=True, hide_index=True, column_config=rank_col_config, height=get_table_height(b_leg))
        
    with legacy_col_right:
        st.markdown("<h3 style='color: #f29ca0; font-size: 1rem;'>Bearish</h3>", unsafe_allow_html=True)
        r_leg = res_legacy.sort_values(by=["Score", "Dollars"], ascending=[True, True]).head(limit)
        st.dataframe(r_leg[["Symbol", "Qty", "Dollars", "Score"]], use_container_width=True, hide_index=True, column_config=rank_col_config, height=get_table_height(r_leg))


def run_strike_zones_app(df):
    render_page_header("📊 Strike Zones")
    exp_range_default = (date.today() + timedelta(days=365))
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
    with c2: td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
    with c3: td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
    with c4: exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
    
    sc1, sc2, sc3, sc4 = st.columns(4, gap="medium")
    with sc1:
        st.markdown("**View Mode**")
        view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], label_visibility="collapsed")
    with sc2:
        st.markdown("**Zone Width**")
        width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], label_visibility="collapsed")
        if width_mode == "Fixed": 
            fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=10)
        else: fixed_size_choice = 10
    with sc3:
        st.markdown("**Include Order Type**")
        inc_cb = st.checkbox("Calls Bought", value=True)
        inc_ps = st.checkbox("Puts Sold", value=True)
        inc_pb = st.checkbox("Puts Bought", value=True)
    with sc4:
        st.markdown("**Other Options**")
        hide_empty      = st.checkbox("Hide Empty Zones", value=True)
        show_table      = st.checkbox("Show Interactive Data Table", value=True)
    
    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)
        
    f_base = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start: f_base = f_base[f_base["Trade Date"].dt.date >= td_start]
    if td_end: f_base = f_base[f_base["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f_base = f_base[(f_base["Expiry_DT"].dt.date >= today_val) & (f_base["Expiry_DT"].dt.date <= exp_end)]
    order_type_col = "Order Type" if "Order Type" in f_base.columns else "Order type"
    
    allowed_sz_types = []
    if inc_cb: allowed_sz_types.append("Calls Bought")
    if inc_ps: allowed_sz_types.append("Puts Sold")
    if inc_pb: allowed_sz_types.append("Puts Bought")
    
    edit_pool_raw = f_base[f_base[order_type_col].isin(allowed_sz_types)].copy()
    
    if edit_pool_raw.empty:
        st.warning("No trades match current filters.")
        return

    if "Include" not in edit_pool_raw.columns:
        edit_pool_raw.insert(0, "Include", True)
    
    edit_pool_raw["Trade Date Str"] = edit_pool_raw["Trade Date"].dt.strftime("%d %b %y")
    edit_pool_raw["Expiry Str"] = edit_pool_raw["Expiry_DT"].dt.strftime("%d %b %y")

    visual_placeholder = st.container()

    if show_table:
        st.markdown("---")
        st.subheader("Data Table & Selection")
        st.caption("Uncheck rows to remove them from the charts above.")
        
        edited_df = st.data_editor(
            edit_pool_raw[["Include", "Trade Date Str", order_type_col, "Symbol", "Strike", "Expiry Str", "Contracts", "Dollars"]],
            column_config={
                "Include": st.column_config.CheckboxColumn("Include", default=True),
                "Dollars": st.column_config.NumberColumn("Dollars", format="$%d"),
                "Contracts": st.column_config.NumberColumn("Qty"),
                "Trade Date Str": "Trade Date",
                "Expiry Str": "Expiry"
            },
            disabled=["Trade Date Str", order_type_col, "Symbol", "Strike", "Expiry Str", "Contracts", "Dollars"],
            hide_index=True,
            use_container_width=True,
            key="sz_editor"
        )
        f = edit_pool_raw[edited_df["Include"]].copy()
    else:
        f = edit_pool_raw.copy()

    with visual_placeholder:
        if f.empty:
            st.info("No rows selected. Check the 'Include' boxes below.")
        else:
            @st.cache_data(ttl=300)
            def get_stock_indicators(sym: str):
                try:
                    ticker_obj = yf.Ticker(sym)
                    h = ticker_obj.history(period="60d", interval="1d")
                    if len(h) == 0: return None, None, None, None, None
                    close = h["Close"]
                    spot_val = float(close.iloc[-1])
                    ema8  = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
                    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
                    sma200_full = ticker_obj.history(period="2y")["Close"]
                    sma200 = float(sma200_full.rolling(window=200).mean().iloc[-1]) if len(sma200_full) >= 200 else None
                    return spot_val, ema8, ema21, sma200, h
                except: return None, None, None, None, None

            spot, ema8, ema21, sma200, history = get_stock_indicators(ticker)
            if spot is None: spot = 100.0

            def pct_from_spot(x):
                if x is None or np.isnan(x): return "—"
                return f"{(x/spot-1)*100:+.1f}%"
                
            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
            if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
            if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

            f["Signed Dollars"] = f.apply(lambda r: (1 if r[order_type_col] in ("Calls Bought","Puts Sold") else -1) * (r["Dollars"] or 0.0), axis=1)
            fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

            if view_mode == "Price Zones":
                strike_min, strike_max = float(np.nanmin(f["Strike (Actual)"].values)), float(np.nanmax(f["Strike (Actual)"].values))
                if width_mode == "Auto": zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / 12.0)), 100))
                else: zone_w = float(fixed_size_choice)
                
                n_dn, n_up = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w)), int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
                lower_edge = spot - n_dn * zone_w
                total = max(1, n_dn + n_up)
                f["ZoneIdx"] = f["Strike (Actual)"].apply(lambda x: min(total - 1, max(0, int(math.floor((x - lower_edge) / zone_w)))))
                agg = f.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
                zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
                if hide_empty: zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
                
                st.markdown('<div class="zones-panel">', unsafe_allow_html=True)
                for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                    if r["Zone_Low"] + (zone_w/2) > spot:
                        color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, zs["Net_Dollars"].abs().max()))*420))
                        val_str = fmt_neg(r["Net_Dollars"])
                        st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>', unsafe_allow_html=True)
                for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                    if r["Zone_Low"] + (zone_w/2) < spot:
                        color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, zs["Net_Dollars"].abs().max()))*420))
                        val_str = fmt_neg(r["Net_Dollars"])
                        st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                e = f.copy()
                e["Bucket"] = pd.cut((pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days), bins=[0, 7, 30, 90, 180, 10000], labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"], include_lowest=True)
                agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                for _, r in agg.iterrows():
                    color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, agg["Net_Dollars"].abs().max()))*420))
                    val_str = fmt_neg(r["Net_Dollars"])
                    st.markdown(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)

def run_pivot_tables_app(df):
    render_page_header("🎯 Pivot Tables")
    max_data_date = get_max_trade_date(df)
            
    main_col, calc_col = st.columns([1.1, 1], gap="large")

    with main_col:
        st.markdown('<div class="control-box">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 12px; font-size: 1rem;'>🔍 Data Filters</h4>", unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1: 
            td_start = st.date_input("Trade Start Date", value=max_data_date, key="pv_start")
            ticker_filter = st.text_input("Ticker (blank=all)", value="", key="pv_ticker").strip().upper()
            min_mkt_cap = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[st.selectbox("Mkt Cap Min", options=["0B", "10B", "50B", "100B", "200B", "500B", "1T"], index=0, key="pv_mkt_cap")]
        with f2:
            td_end = st.date_input("Trade End Date", value=max_data_date, key="pv_end")
            min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[st.selectbox("Min Dollars", options=["0M", "5M", "10M", "50M", "100M"], index=0, key="pv_notional")]
            ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
        st.markdown('</div>', unsafe_allow_html=True)

    with calc_col:
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 12px; font-size: 1rem; color: #71d28a;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1: c_strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0, format="%.2f", key="calc_strike")
        with c_i2: c_premium = st.number_input("Premium", min_value=0.00, value=2.50, step=0.05, format="%.2f", key="calc_premium")
        with c_i3: c_expiry = st.date_input("Expiration", value=date.today() + timedelta(days=30), key="calc_expiry")
        
        dte = (c_expiry - date.today()).days
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

        st.session_state["calc_out_ann"] = f"{annual_ret:.2f}%"
        st.session_state["calc_out_coc"] = f"{coc_ret:.2f}%"
        st.session_state["calc_out_dte"] = str(max(0, dte))
            
        c_o1, c_o2, c_o3 = st.columns(3)
        with c_o1: st.text_input("Annualised", key="calc_out_ann", disabled=True)
        with c_o2: st.text_input("Cash on Cash", key="calc_out_coc", disabled=True)
        with c_o3: st.text_input("DTE", key="calc_out_dte", disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 5px; margin-bottom: 20px; align-items: center; padding-left: 5px;">
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 12px; height: 12px; border-radius: 2px; background:#b7e1cd"></div> This Fri</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 12px; height: 12px; border-radius: 2px; background:#fce8b2"></div> Next Fri</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 12px; height: 12px; border-radius: 2px; background:#f4c7c3"></div> Two Fri</div>
        <div style="color: #666; font-style: italic; margin-left: 20px;">ℹ️ Market Cap / EMA data can be delayed.</div>
    </div>
    """, unsafe_allow_html=True)
    
    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: 
        st.info("No data found for the selected trade dates.")
        return

    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    cb_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy()
    ps_pool = d_range[d_range[order_type_col] == "Puts Sold"].copy()
    pb_pool = d_range[d_range[order_type_col] == "Puts Bought"].copy()
    
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb_pool['occ'], ps_pool['occ'] = cb_pool.groupby(keys).cumcount(), ps_pool.groupby(keys).cumcount()
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    rr_rows = []
    for idx, row in rr_matches.iterrows():
        rr_rows.append({'Symbol': row['Symbol'], 'Trade Date': row['Trade Date'], 'Expiry_DT': row['Expiry_DT'], 'Contracts': row['Contracts'], 'Dollars': row['Dollars_c'], 'Strike': clean_strike_fmt(row['Strike_c']), 'Pair_ID': idx, 'Pair_Side': 0})
        rr_rows.append({'Symbol': row['Symbol'], 'Trade Date': row['Trade Date'], 'Expiry_DT': row['Expiry_DT'], 'Contracts': row['Contracts'], 'Dollars': row['Dollars_p'], 'Strike': clean_strike_fmt(row['Strike_p']), 'Pair_ID': idx, 'Pair_Side': 1})
    df_rr = pd.DataFrame(rr_rows)

    if not rr_matches.empty:
        match_keys = keys + ['occ']
        def filter_out_matches(pool, matches):
            temp_matches = matches[match_keys].copy()
            temp_matches['_remove'] = True
            merged = pool.merge(temp_matches, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])
        cb_pool = filter_out_matches(cb_pool, rr_matches)
        ps_pool = filter_out_matches(ps_pool, rr_matches)

    def apply_f(data):
        if data.empty: return data
        f = data.copy()
        if ticker_filter: f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
        f = f[f["Dollars"] >= min_notional]
        if not f.empty and min_mkt_cap > 0:
            unique_symbols = f["Symbol"].unique()
            valid_symbols = [s for s in unique_symbols if get_market_cap(s) >= float(min_mkt_cap)]
            f = f[f["Symbol"].isin(valid_symbols)]
        if not f.empty and ema_filter == "Yes":
            unique_symbols = f["Symbol"].unique()
            valid_ema_symbols = [s for s in unique_symbols if is_above_ema21(s)]
            f = f[f["Symbol"].isin(valid_ema_symbols)]
        return f

    df_cb_f, df_ps_f, df_pb_f, df_rr_f = apply_f(cb_pool), apply_f(ps_pool), apply_f(pb_pool), apply_f(df_rr)

    def get_p(data, is_rr=False):
        if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
        sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        if is_rr: piv = data.merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
        else:
            piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().merge(sr, on="Symbol")
            piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
        return piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    # Using [1, 1, 1] allows Streamlit to stack these on mobile automatically
    row1_c1, row1_c2, row1_c3 = st.columns([1, 1, 1], gap="medium") 
    fmt = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}
    
    with row1_c1:
        st.subheader("Calls Bought"); tbl = get_p(df_cb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
    with row1_c2:
        st.subheader("Puts Sold"); tbl = get_p(df_ps_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
    with row1_c3:
        st.subheader("Puts Bought"); tbl = get_p(df_pb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
    
    st.markdown("---")
    st.subheader("Risk Reversals")
    tbl_rr = get_p(df_rr_f, is_rr=True)
    if not tbl_rr.empty: 
        st.dataframe(tbl_rr.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl_rr), column_config=COLUMN_CONFIG_PIVOT)
    else: st.caption("No matched RR pairs found.")

def run_rsi_divergences_app(df):
    render_page_header("📈 RSI Divergences")
    st.markdown("""<style>div.stLinkButton > a { background: linear-gradient(45deg, #ff00ff, #00ffff, #ff0000, #ffff00, #00ff00); background-size: 400% 400%; animation: tie-dye 10s ease infinite; border: none; color: white !important; font-weight: bold; padding: 15px 30px; border-radius: 10px; text-decoration: none; display: inline-block; transition: transform 0.2s; } div.stLinkButton > a:hover { transform: scale(1.05); } @keyframes tie-dye { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }</style>""", unsafe_allow_html=True)
    st.link_button("🌈 🚀 Click to travel to the new RSI Divergence website 🚀 🌈", "https://mr-darcys-rsi-divergence.streamlit.app/")

# --- 3. MAIN EXECUTION ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

st.markdown("""
<style>
    :root{--bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea; --green:#71d28a; --red:#f29ca0; --line:#66b7ff;}
    
    .block-container{padding-top: 2rem !important;}
    
    header[data-testid="stHeader"] {background: transparent !important;}

    .control-box{padding:15px; border-radius:10px; background-color: var(--panel2); border: 1px solid #3a3f45; margin-bottom: 20px;}
    .calc-box {
        padding: 15px; 
        border-radius: 10px; 
        background-color: rgba(113, 210, 138, 0.03); 
        border: 1px solid rgba(113, 210, 138, 0.2); 
        margin-bottom: 20px;
    }

    .st-key-calc_out_ann input, .st-key-calc_out_coc input, .st-key-calc_out_dte input {
        background-color: rgba(113, 210, 138, 0.1) !important;
        color: #71d28a !important;
        border: 1px solid rgba(113, 210, 138, 0.4) !important;
        font-weight: 700 !important;
    }

    .nav-container {
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 10px;
    }
    
    div.stButton > button.nav-btn {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0px !important;
        color: #808495 !important;
        font-weight: 600 !important;
        padding-bottom: 8px !important;
        height: auto !important;
    }
    div.stButton > button.nav-btn:hover {
        color: #ffffff !important;
        border-bottom: 2px solid #555 !important;
    }
    div.stButton > button.nav-active {
        color: #66b7ff !important;
        border-bottom: 2px solid #66b7ff !important;
    }

    .zones-panel{padding:14px 0; border-radius:10px;}
    .zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
    .zone-label{width:100px;font-weight:700; text-align: right;}
    .zone-bar{height:22px;border-radius:6px;min-width:6px}
    .zone-bull{background:linear-gradient(90deg,var(--green),#60c57b)}
    .zone-bear{background:linear-gradient(90deg,var(--red),#e4878d)}
    .zone-value{min-width:220px;font-variant-numeric:tabular-nums}
    .price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
    .price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: var(--line); opacity: 0.6; }
    .price-badge { background: #2b3a45; color: #bfe7ff; border: 1px solid #56b6ff; border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; box-shadow: 0 2px 8px rgba(0,0,0,0.35); white-space: nowrap; margin: 0 12px; z-index: 1; }
    .metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
    .badge{background:#2b3a45;border:1px solid #3b5566;color:#cde8ff;border-radius:18px;padding:6px 10px;font-weight:700}
    .price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
    th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}
</style>
""", unsafe_allow_html=True)

try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)

    if "app_choice" not in st.session_state:
        st.session_state["app_choice"] = "Database"

    # Updated menu item names
    nav_items = ["Database", "Rankings", "Pivot Tables", "Strike Zones", "RSI Divergences"]
    
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1, 1.2, 3]) # Adjusted column ratios for new names
    for i, item in enumerate(nav_items):
        is_active = st.session_state["app_choice"] == item
        btn_key = f"nav_btn_{item}"
        
        btn_class = "nav-active" if is_active else "nav-btn"
        
        if cols[i].button(item, key=btn_key, help=f"Go to {item}", 
                          use_container_width=True):
            st.session_state["app_choice"] = item
            st.rerun()
            
    st.markdown("<hr style='margin-top: -10px; margin-bottom: 25px; opacity: 0.15; height: 1px; border: none; background-color: #555;'>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SIDEBAR UPDATES ---
    with st.sidebar:
        # Added the user experience note as requested
        st.info("💡 **View on Desktop (Light Mode)**\n\nFor the best experience, please use a wide desktop monitor in light mode.")
        
        # Moved the Last Updated date to sidebar underneath the note
        max_date_str = get_max_trade_date(df_global).strftime("%d %b %y")
        st.markdown(f"**Last Data Update:** {max_date_str}")
        st.markdown("---")

    current_choice = st.session_state["app_choice"]
    
    if current_choice == "Database":
        run_database_app(df_global)
    elif current_choice == "Rankings":
        run_rankings_app(df_global)
    elif current_choice == "Pivot Tables":
        run_pivot_tables_app(df_global)
    elif current_choice == "Strike Zones":
        run_strike_zones_app(df_global)
    elif current_choice == "RSI Divergences":
        run_rsi_divergences_app(df_global)
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")

