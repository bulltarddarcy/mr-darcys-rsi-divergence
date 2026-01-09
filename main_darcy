# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
import math
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT UTILS ---
# Import everything we moved to utils.py
from utils_darcy import (
    load_and_clean_data, get_parquet_config, load_parquet_and_clean,
    load_ticker_map, get_stock_indicators, fetch_yahoo_data,
    add_technicals, get_table_height, highlight_expiry,
    parse_periods, find_divergences, prepare_data,
    calculate_optimal_signal_stats, find_whale_confluence,
    analyze_trade_setup, get_market_cap, fetch_market_caps_batch,
    fetch_technicals_batch, clean_strike_fmt, get_max_trade_date,
    calculate_smart_money_score, fetch_history_optimized,
    find_rsi_percentile_signals, is_above_ema21, get_ticker_technicals, 
    VOL_SMA_PERIOD, EMA8_PERIOD, EMA21_PERIOD
)

# --- 1. GLOBAL DATA LOADING & UTILITIES ---

COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

DATA_KEYS_PARQUET = get_parquet_config()

# --- 2. APP MODULES ---

def run_database_app(df):
    st.title("📂 Database")
    max_data_date = get_max_trade_date(df)
    
    if 'saved_db_ticker' not in st.session_state: st.session_state.saved_db_ticker = ""
    if 'saved_db_start' not in st.session_state: st.session_state.saved_db_start = max_data_date
    if 'saved_db_end' not in st.session_state: st.session_state.saved_db_end = max_data_date
    if 'saved_db_exp' not in st.session_state: st.session_state.saved_db_exp = (date.today() + timedelta(days=365))
    if 'saved_db_inc_cb' not in st.session_state: st.session_state.saved_db_inc_cb = True
    if 'saved_db_inc_ps' not in st.session_state: st.session_state.saved_db_inc_ps = True
    if 'saved_db_inc_pb' not in st.session_state: st.session_state.saved_db_inc_pb = True

    def save_db_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        db_ticker = st.text_input("Ticker (blank=all)", value=st.session_state.saved_db_ticker, key="db_ticker_input", on_change=save_db_state, args=("db_ticker_input", "saved_db_ticker")).strip().upper()
    with c2: start_date = st.date_input("Trade Start Date", value=st.session_state.saved_db_start, key="db_start", on_change=save_db_state, args=("db_start", "saved_db_start"))
    with c3: end_date = st.date_input("Trade End Date", value=st.session_state.saved_db_end, key="db_end", on_change=save_db_state, args=("db_end", "saved_db_end"))
    with c4:
        db_exp_end = st.date_input("Expiration Range (end)", value=st.session_state.saved_db_exp, key="db_exp", on_change=save_db_state, args=("db_exp", "saved_db_exp"))
    
    ot1, ot2, ot3, ot_pad = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_inc_cb", on_change=save_db_state, args=("db_inc_cb", "saved_db_inc_cb"))
    with ot2: inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_inc_ps", on_change=save_db_state, args=("db_inc_ps", "saved_db_inc_ps"))
    with ot3: inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_inc_pb", on_change=save_db_state, args=("db_inc_pb", "saved_db_inc_pb"))
    
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
    st.markdown("<br><br><br>", unsafe_allow_html=True)

def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_data_date = get_max_trade_date(df)
    start_default = max_data_date - timedelta(days=14)
    
    if 'saved_rank_start' not in st.session_state: st.session_state.saved_rank_start = start_default
    if 'saved_rank_end' not in st.session_state: st.session_state.saved_rank_end = max_data_date
    if 'saved_rank_limit' not in st.session_state: st.session_state.saved_rank_limit = 20
    if 'saved_rank_mc' not in st.session_state: st.session_state.saved_rank_mc = "10B"
    if 'saved_rank_ema' not in st.session_state: st.session_state.saved_rank_ema = False

    def save_rank_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
    
    c1, c2, c3, c4 = st.columns([1, 1, 0.7, 1.3], gap="small")
    with c1: rank_start = st.date_input("Trade Start Date", value=st.session_state.saved_rank_start, key="rank_start", on_change=save_rank_state, args=("rank_start", "saved_rank_start"))
    with c2: rank_end = st.date_input("Trade End Date", value=st.session_state.saved_rank_end, key="rank_end", on_change=save_rank_state, args=("rank_end", "saved_rank_end"))
    with c3: limit = st.number_input("Limit", value=st.session_state.saved_rank_limit, min_value=1, max_value=200, key="rank_limit", on_change=save_rank_state, args=("rank_limit", "saved_rank_limit"))
    with c4: 
        min_mkt_cap_rank = st.selectbox("Min Market Cap", ["0B", "2B", "10B", "50B", "100B"], index=2, key="rank_mc", on_change=save_rank_state, args=("rank_mc", "saved_rank_mc"))
        filter_ema = st.checkbox("Hide < 8 EMA", value=False, key="rank_ema", on_change=save_rank_state, args=("rank_ema", "saved_rank_ema"))
        
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
        st.warning("No trades found.")
        return

    tab_rank, tab_ideas, tab_vol = st.tabs(["🧠 Smart Money", "💡 Top 3", "🤡 Bulltard"])

    mc_thresh = {"0B":0, "2B":2e9, "10B":1e10, "50B":5e10, "100B":1e11}.get(min_mkt_cap_rank, 1e10)

    top_bulls, top_bears, valid_data = calculate_smart_money_score(df, rank_start, rank_end, mc_thresh, filter_ema, limit)

    with tab_rank:
        if valid_data.empty:
            st.warning("Not enough data for Smart Money scores.")
        else:
            sm_config = {
                "Symbol": st.column_config.TextColumn("Ticker", width=60),
                "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                "Trade_Count": st.column_config.NumberColumn("Qty", width=50),
                "Last Trade": st.column_config.TextColumn("Last", width=70)
            }
            cols_to_show = ["Symbol", "Score", "Trade_Count", "Last Trade"]
            
            sm1, sm2 = st.columns(2, gap="large")
            with sm1:
                st.markdown("<div style='color: #71d28a; font-weight:bold;'>Top Bullish Scores</div>", unsafe_allow_html=True)
                if not top_bulls.empty:
                    st.dataframe(top_bulls[cols_to_show], use_container_width=True, hide_index=True, column_config=sm_config, height=get_table_height(top_bulls, max_rows=100))
            
            with sm2:
                st.markdown("<div style='color: #f29ca0; font-weight:bold;'>Top Bearish Scores</div>", unsafe_allow_html=True)
                if not top_bears.empty:
                    st.dataframe(top_bears[cols_to_show], use_container_width=True, hide_index=True, column_config=sm_config, height=get_table_height(top_bears, max_rows=100))
        st.markdown("<br><br>", unsafe_allow_html=True)

    with tab_ideas:
        if top_bulls.empty:
            st.info("No Bullish candidates found to analyze.")
        else:
            st.caption(f"ℹ️ Analyzing the Top {len(top_bulls)} 'Smart Money' tickers for confluence...")
            st.caption("ℹ️ Strategy: Combines Whale Levels (Global DB), Technicals (EMA), and Historical RSI Backtests to find optimal expirations.")
            
            st.markdown("<span style='color:red;'>⚠️ Note: This methodology is a work in progress and should not be relied upon right now.</span>", unsafe_allow_html=True)
            
            ticker_map = load_ticker_map()
            candidates = []
            
            prog_bar = st.progress(0, text="Analyzing technicals...")
            bull_list = top_bulls["Symbol"].tolist()
            
            batch_results = fetch_technicals_batch(bull_list)
            
            for i, t in enumerate(bull_list):
                prog_bar.progress((i+1)/len(bull_list), text=f"Checking {t}...")
                
                data_tuple = batch_results.get(t)
                t_df = data_tuple[4] if data_tuple else None

                if t_df is None or t_df.empty:
                    t_df = fetch_yahoo_data(t)

                if t_df is not None and not t_df.empty:
                    sm_score = top_bulls[top_bulls["Symbol"]==t]["Score"].iloc[0]
                    
                    tech_score, reasons, suggs = analyze_trade_setup(t, t_df, df)
                    
                    final_conviction = (sm_score / 25.0) + tech_score 
                    
                    candidates.append({
                        "Ticker": t,
                        "Score": final_conviction,
                        "Price": t_df.iloc[-1].get('CLOSE') or t_df.iloc[-1].get('Close'),
                        "Reasons": reasons,
                        "Suggestions": suggs
                    })
            
            prog_bar.empty()
            best_ideas = sorted(candidates, key=lambda x: x['Score'], reverse=True)[:3]
            
            cols = st.columns(3)
            for i, cand in enumerate(best_ideas):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"### #{i+1} {cand['Ticker']}")
                        st.metric("Conviction", f"{cand['Score']:.1f}/10", f"${cand['Price']:.2f}")
                        st.markdown("**Strategy:**")
                        
                        if cand['Suggestions']['Sell Puts']:
                            st.success(f"🛡️ **Sell Put:** {cand['Suggestions']['Sell Puts']}")
                        if cand['Suggestions']['Buy Calls']:
                            st.info(f"🟢 **Buy Call:** {cand['Suggestions']['Buy Calls']}")
                            
                        st.markdown("---")
                        for r in cand['Reasons']:
                            st.caption(f"• {r}")
        st.markdown("<br><br>", unsafe_allow_html=True)

    with tab_vol:
        st.caption("ℹ️ Legacy Methodology: Score = (Calls + Puts Sold) - (Puts Bought).")
        st.caption("ℹ️ Note: These tables differ from Bulltard's because his rankings include expired trades.")
        
        counts = f_filtered.groupby(["Symbol", order_type_col]).size().unstack(fill_value=0)
        
        for col in target_types:
            if col not in counts.columns: counts[col] = 0
            
        scores_df = pd.DataFrame(index=counts.index)
        scores_df["Score"] = counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"]
        scores_df["Trade Count"] = counts.sum(axis=1)
        
        last_trade_series = f_filtered.groupby("Symbol")["Trade Date"].max()
        scores_df["Last Trade"] = last_trade_series.dt.strftime("%d %b %y")
        
        res = scores_df.reset_index()
        if "batch_caps" in locals():
            res["Market Cap"] = res["Symbol"].map(batch_caps)
        else:
            unique_ts = res["Symbol"].unique().tolist()
            res["Market Cap"] = res["Symbol"].map(fetch_market_caps_batch(unique_ts))
            
        res = res[res["Market Cap"] >= mc_thresh]
        
        rank_col_config = {
            "Symbol": st.column_config.TextColumn("Symbol", width=60),
            "Trade Count": st.column_config.NumberColumn("#", width=50),
            "Last Trade": st.column_config.TextColumn("Last Trade", width=90),
            "Score": st.column_config.NumberColumn("Score", width=50),
        }
        
        pre_bull_df = res.sort_values(by=["Score", "Trade Count"], ascending=[False, False])
        pre_bear_df = res.sort_values(by=["Score", "Trade Count"], ascending=[True, False])
        
        def get_filtered_list(source_df, mode="Bull"):
            if not filter_ema:
                return source_df.head(limit)
            
            candidates = source_df.head(limit * 3) 
            final_list = []
            
            needed_tickers = candidates["Symbol"].tolist()
            mini_batch = fetch_technicals_batch(needed_tickers)
            
            for _, r in candidates.iterrows():
                try:
                    s, e8, _, _, _ = mini_batch.get(r["Symbol"], (None,None,None,None,None))
                    if s and e8:
                        if mode == "Bull" and s > e8: final_list.append(r)
                        elif mode == "Bear" and s < e8: final_list.append(r)
                except: pass
                
                if len(final_list) >= limit: break
            
            return pd.DataFrame(final_list)

        bull_df = get_filtered_list(pre_bull_df, "Bull")
        bear_df = get_filtered_list(pre_bear_df, "Bear")
        
        cols_final = ["Symbol", "Trade Count", "Last Trade", "Score"]
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("<div style='color: #71d28a; font-weight:bold;'>Bullish Volume</div>", unsafe_allow_html=True)
            if not bull_df.empty:
                st.dataframe(bull_df[cols_final], use_container_width=True, hide_index=True, column_config=rank_col_config, height=get_table_height(bull_df, max_rows=100))
        with v2:
            st.markdown("<div style='color: #f29ca0; font-weight:bold;'>Bearish Volume</div>", unsafe_allow_html=True)
            if not bear_df.empty:
                st.dataframe(bear_df[cols_final], use_container_width=True, hide_index=True, column_config=rank_col_config, height=get_table_height(bear_df, max_rows=100))
        st.markdown("<br><br>", unsafe_allow_html=True)

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = get_max_trade_date(df)
            
    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    
    if 'saved_pv_start' not in st.session_state: st.session_state.saved_pv_start = max_data_date
    if 'saved_pv_end' not in st.session_state: st.session_state.saved_pv_end = max_data_date
    if 'saved_pv_ticker' not in st.session_state: st.session_state.saved_pv_ticker = ""
    if 'saved_pv_notional' not in st.session_state: st.session_state.saved_pv_notional = "0M"
    if 'saved_pv_mkt_cap' not in st.session_state: st.session_state.saved_pv_mkt_cap = "0B"
    if 'saved_pv_ema' not in st.session_state: st.session_state.saved_pv_ema = "All"
    
    if 'saved_calc_strike' not in st.session_state: st.session_state.saved_calc_strike = 100.0
    if 'saved_calc_premium' not in st.session_state: st.session_state.saved_calc_premium = 2.50
    if 'saved_calc_expiry' not in st.session_state: st.session_state.saved_calc_expiry = date.today() + timedelta(days=30)

    def save_pv_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]

    with col_filters:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>🔍 Filters</h4>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1: 
            td_start = st.date_input("Trade Start Date", value=st.session_state.saved_pv_start, key="pv_start", on_change=save_pv_state, args=("pv_start", "saved_pv_start"))
        with fc2: 
            td_end = st.date_input("Trade End Date", value=st.session_state.saved_pv_end, key="pv_end", on_change=save_pv_state, args=("pv_end", "saved_pv_end"))
        with fc3: 
            ticker_filter = st.text_input("Ticker (blank=all)", value=st.session_state.saved_pv_ticker, key="pv_ticker", on_change=save_pv_state, args=("pv_ticker", "saved_pv_ticker")).strip().upper()
        
        fc4, fc5, fc6 = st.columns(3)
        with fc4: 
            opts_not = ["0M", "5M", "10M", "50M", "100M"]
            curr_not = st.session_state.saved_pv_notional
            idx_not = opts_not.index(curr_not) if curr_not in opts_not else 0
            sel_not = st.selectbox("Min Dollars", options=opts_not, index=idx_not, key="pv_notional", on_change=save_pv_state, args=("pv_notional", "saved_pv_notional"))
            min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[sel_not]
            
        with fc5: 
            opts_mc = ["0B", "10B", "50B", "100B", "200B", "500B", "1T"]
            curr_mc = st.session_state.saved_pv_mkt_cap
            idx_mc = opts_mc.index(curr_mc) if curr_mc in opts_mc else 0
            sel_mc = st.selectbox("Mkt Cap Min", options=opts_mc, index=idx_mc, key="pv_mkt_cap", on_change=save_pv_state, args=("pv_mkt_cap", "saved_pv_mkt_cap"))
            min_mkt_cap = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[sel_mc]
            
        with fc6: 
            opts_ema = ["All", "Yes"]
            curr_ema = st.session_state.saved_pv_ema
            idx_ema = opts_ema.index(curr_ema) if curr_ema in opts_ema else 0
            ema_filter = st.selectbox("Over 21 Day EMA", options=opts_ema, index=idx_ema, key="pv_ema_filter", on_change=save_pv_state, args=("pv_ema_filter", "saved_pv_ema"))

    with col_calculator:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        
        cc1, cc2, cc3 = st.columns(3)
        with cc1: c_strike = st.number_input("Strike Price", min_value=0.01, value=st.session_state.saved_calc_strike, step=1.0, format="%.2f", key="calc_strike", on_change=save_pv_state, args=("calc_strike", "saved_calc_strike"))
        with cc2: c_premium = st.number_input("Premium", min_value=0.00, value=st.session_state.saved_calc_premium, step=0.05, format="%.2f", key="calc_premium", on_change=save_pv_state, args=("calc_premium", "saved_calc_premium"))
        with cc3: c_expiry = st.date_input("Expiration", value=st.session_state.saved_calc_expiry, key="calc_expiry", on_change=save_pv_state, args=("calc_expiry", "saved_calc_expiry"))
        
        dte = (c_expiry - date.today()).days
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

        st.session_state["calc_out_ann"] = f"{annual_ret:.1f}%"
        st.session_state["calc_out_coc"] = f"{coc_ret:.1f}%"
        st.session_state["calc_out_dte"] = str(max(0, dte))

        cc4, cc5, cc6 = st.columns(3)
        with cc4: st.text_input("Annualised Return", key="calc_out_ann")
        with cc5: st.text_input("Cash on Cash Return", key="calc_out_coc")
        with cc6: st.text_input("Days to Expiration", key="calc_out_dte")

    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 10px; margin-bottom: 20px; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#b7e1cd"></div> This Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#fce8b2"></div> Next Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#f4c7c3"></div> Two Fridays</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Market Cap filtering can be buggy. If empty, reset \'Mkt Cap Min\' to 0B.</div>', unsafe_allow_html=True)
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Scroll down to see the Risk Reversals table.</div>', unsafe_allow_html=True)

    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: return

    # --- FIX: Replaced 'f' with 'df' ---
    order_type_col = "Order Type" if "Order Type" in df.columns else "Order type"
    
    cb_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy()
    ps_pool = d_range[d_range[order_type_col] == "Puts Sold"].copy()
    pb_pool = d_range[d_range[order_type_col] == "Puts Bought"].copy()
    
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb_pool['occ'], ps_pool['occ'] = cb_pool.groupby(keys).cumcount(), ps_pool.groupby(keys).cumcount()
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    if not rr_matches.empty:
        rr_c = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_c', 'Strike_c']].copy()
        rr_c.rename(columns={'Dollars_c': 'Dollars', 'Strike_c': 'Strike'}, inplace=True)
        rr_c['Pair_ID'] = rr_matches.index
        rr_c['Pair_Side'] = 0
        
        rr_p = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_p', 'Strike_p']].copy()
        rr_p.rename(columns={'Dollars_p': 'Dollars', 'Strike_p': 'Strike'}, inplace=True)
        rr_p['Pair_ID'] = rr_matches.index
        rr_p['Pair_Side'] = 1
        
        df_rr = pd.concat([rr_c, rr_p])
        df_rr['Strike'] = df_rr['Strike'].apply(clean_strike_fmt)
        
        match_keys = keys + ['occ']
        def filter_out_matches(pool, matches):
            temp_matches = matches[match_keys].copy()
            temp_matches['_remove'] = True
            merged = pool.merge(temp_matches, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])
        cb_pool = filter_out_matches(cb_pool, rr_matches)
        ps_pool = filter_out_matches(ps_pool, rr_matches)
    else:
        df_rr = pd.DataFrame(columns=['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars', 'Strike', 'Pair_ID', 'Pair_Side'])

    def apply_f(data):
        if data.empty: return data
        f = data.copy()
        if ticker_filter: f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
        f = f[f["Dollars"] >= min_notional]
        
        if not f.empty:
            unique_symbols = f["Symbol"].unique()
            valid_symbols = set(unique_symbols)
            
            if min_mkt_cap > 0:
                valid_symbols = {s for s in valid_symbols if get_market_cap(s) >= float(min_mkt_cap)}
            
            if ema_filter == "Yes":
                batch_results = fetch_technicals_batch(list(valid_symbols))
                valid_symbols = {
                    s for s in valid_symbols 
                    if batch_results.get(s, (None, None))[2] is None or 
                    (batch_results[s][0] is not None and batch_results[s][2] is not None and batch_results[s][0] > batch_results[s][2])
                }
            
            f = f[f["Symbol"].isin(valid_symbols)]
            
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
        
        piv["Symbol_Display"] = np.where(piv["Symbol"] == piv["Symbol"].shift(1), "", piv["Symbol"])
        
        return piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    row1_c1, row1_c2, row1_c3 = st.columns(3); fmt = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}
    with row1_c1:
        st.subheader("Calls Bought"); tbl = get_p(df_cb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl, max_rows=50), column_config=COLUMN_CONFIG_PIVOT)
    with row1_c2:
        st.subheader("Puts Sold"); tbl = get_p(df_ps_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl, max_rows=50), column_config=COLUMN_CONFIG_PIVOT)
    with row1_c3:
        st.subheader("Puts Bought"); tbl = get_p(df_pb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl, max_rows=50), column_config=COLUMN_CONFIG_PIVOT)
    
    st.subheader("Risk Reversals")
    tbl_rr = get_p(df_rr_f, is_rr=True)
    if not tbl_rr.empty: 
        st.dataframe(tbl_rr.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl_rr, max_rows=50), column_config=COLUMN_CONFIG_PIVOT)
        st.markdown("<br><br>", unsafe_allow_html=True)
    else: st.caption("No matched RR pairs found.")

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    exp_range_default = (date.today() + timedelta(days=365))
    
    if 'saved_sz_ticker' not in st.session_state: st.session_state.saved_sz_ticker = "AMZN"
    if 'saved_sz_start' not in st.session_state: st.session_state.saved_sz_start = None
    if 'saved_sz_end' not in st.session_state: st.session_state.saved_sz_end = None
    if 'saved_sz_exp' not in st.session_state: st.session_state.saved_sz_exp = exp_range_default
    if 'saved_sz_view' not in st.session_state: st.session_state.saved_sz_view = "Price Zones"
    if 'saved_sz_width_mode' not in st.session_state: st.session_state.saved_sz_width_mode = "Auto"
    if 'saved_sz_fixed' not in st.session_state: st.session_state.saved_sz_fixed = 10
    if 'saved_sz_inc_cb' not in st.session_state: st.session_state.saved_sz_inc_cb = True
    if 'saved_sz_inc_ps' not in st.session_state: st.session_state.saved_sz_inc_ps = True
    if 'saved_sz_inc_pb' not in st.session_state: st.session_state.saved_sz_inc_pb = True

    def save_sz_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
    
    col_settings, col_visuals = st.columns([1, 2.5], gap="large")
    
    with col_settings:
        ticker = st.text_input("Ticker", value=st.session_state.saved_sz_ticker, key="sz_ticker", on_change=save_sz_state, args=("sz_ticker", "saved_sz_ticker")).strip().upper()
        td_start = st.date_input("Trade Date (start)", value=st.session_state.saved_sz_start, key="sz_start", on_change=save_sz_state, args=("sz_start", "saved_sz_start"))
        td_end = st.date_input("Trade Date (end)", value=st.session_state.saved_sz_end, key="sz_end", on_change=save_sz_state, args=("sz_end", "saved_sz_end"))
        exp_end = st.date_input("Exp. Range (end)", value=st.session_state.saved_sz_exp, key="sz_exp", on_change=save_sz_state, args=("sz_exp", "saved_sz_exp"))
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown("**View Mode**")
            view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], index=0 if st.session_state.saved_sz_view == "Price Zones" else 1, label_visibility="collapsed", key="sz_view", on_change=save_sz_state, args=("sz_view", "saved_sz_view"))
            
            st.markdown("**Zone Width**")
            width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], index=0 if st.session_state.saved_sz_width_mode == "Auto" else 1, label_visibility="collapsed", key="sz_width_mode", on_change=save_sz_state, args=("sz_width_mode", "saved_sz_width_mode"))
            if width_mode == "Fixed": 
                fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=st.session_state.saved_sz_fixed, key="sz_fixed", on_change=save_sz_state, args=("sz_fixed", "saved_sz_fixed"))
            else: fixed_size_choice = 10
        
        with c_sub2:
            st.markdown("**Include**")
            inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_sz_inc_cb, key="sz_inc_cb", on_change=save_sz_state, args=("sz_inc_cb", "saved_sz_inc_cb"))
            inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_sz_inc_ps, key="sz_inc_ps", on_change=save_sz_state, args=("sz_inc_ps", "saved_sz_inc_ps"))
            inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_sz_inc_pb, key="sz_inc_pb", on_change=save_sz_state, args=("sz_inc_pb", "saved_sz_inc_pb"))
            
        hide_empty = True
        show_table = True
    
    with col_visuals:
        chart_container = st.container()

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
        with col_visuals:
            st.warning("No trades match current filters.")
        return

    if "Include" not in edit_pool_raw.columns:
        edit_pool_raw.insert(0, "Include", True)
    
    if show_table:
        editor_input = edit_pool_raw[["Include", "Trade Date", order_type_col, "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"]].copy()
        
        editor_input["Dollars"] = pd.to_numeric(editor_input["Dollars"], errors='coerce').fillna(0)
        editor_input["Contracts"] = pd.to_numeric(editor_input["Contracts"], errors='coerce').fillna(0)

        column_configuration = {
            "Include": st.column_config.CheckboxColumn("Include", default=True),
            "Trade Date": st.column_config.DateColumn("Trade Date", format="DD MMM YY"),
            "Expiry_DT": st.column_config.DateColumn("Expiry", format="DD MMM YY"),
            "Dollars": st.column_config.NumberColumn("Dollars", format="$%d"),
            "Contracts": st.column_config.NumberColumn("Qty", format="%d"),
            order_type_col: st.column_config.TextColumn("Order Type"),
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Strike": st.column_config.TextColumn("Strike"),
        }
        
        st.subheader("Data Table & Selection")
        
        edited_df = st.data_editor(
            editor_input,
            column_config=column_configuration,
            disabled=["Trade Date", order_type_col, "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"],
            hide_index=True,
            use_container_width=True,
            key="sz_editor"
        )
        f = edit_pool_raw[edited_df["Include"]].copy()
        st.markdown("<br><br>", unsafe_allow_html=True)
    else:
        f = edit_pool_raw.copy()

    with chart_container:
        if f.empty:
            st.info("No rows selected. Check the 'Include' boxes below.")
        else:
            spot, ema8, ema21, sma200, history = get_stock_indicators(ticker)
            
            if spot is None:
                df_y = fetch_yahoo_data(ticker)
                if df_y is not None and not df_y.empty:
                    try:
                        spot = float(df_y["CLOSE"].iloc[-1])
                        ema8 = float(df_y["CLOSE"].ewm(span=8, adjust=False).mean().iloc[-1])
                        ema21 = float(df_y["CLOSE"].ewm(span=21, adjust=False).mean().iloc[-1])
                        sma200 = float(df_y["CLOSE"].rolling(window=200).mean().iloc[-1]) if len(df_y) >= 200 else None
                    except: 
                        pass

            if spot is None: spot = 100.0

            def pct_from_spot(x):
                if x is None or np.isnan(x): return "—"
                return f"{(x/spot-1)*100:+.1f}%"
            
            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
            if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
            if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

            f["Signed Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
            
            fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

            if view_mode == "Price Zones":
                strike_vals = f["Strike (Actual)"].values
                strike_min, strike_max = float(np.nanmin(strike_vals)), float(np.nanmax(strike_vals))
                if width_mode == "Auto": 
                    denom = 12.0
                    zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / denom)), 100))
                else: zone_w = float(fixed_size_choice)
                
                n_dn = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w))
                n_up = int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
                
                lower_edge = spot - n_dn * zone_w
                total = max(1, n_dn + n_up)
                
                f["ZoneIdx"] = np.clip(
                    np.floor((f["Strike (Actual)"] - lower_edge) / zone_w).astype(int), 
                    0, 
                    total - 1
                )

                agg = f.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                
                zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
                zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
                
                if hide_empty: zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
                
                html_out = ['<div class="zones-panel">']
                
                max_val = max(1.0, zs["Net_Dollars"].abs().max())
                sorted_zs = zs.sort_values("ZoneIdx", ascending=False)
                
                upper_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) > spot]
                lower_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) <= spot]
                
                for _, r in upper_zones.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    pct = (abs(r['Net_Dollars']) / max_val) * 100
                    val_str = fmt_neg(r["Net_Dollars"])
                    html_out.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
                
                html_out.append(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>')
                
                for _, r in lower_zones.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    pct = (abs(r['Net_Dollars']) / max_val) * 100
                    val_str = fmt_neg(r["Net_Dollars"])
                    html_out.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
                
                html_out.append('</div>')
                st.markdown("".join(html_out), unsafe_allow_html=True)
                
            else:
                e = f.copy()
                days_diff = (pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
                
                new_bins = [0, 7, 30, 60, 90, 120, 180, 365, 10000]
                new_labels = ["0-7d", "8-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", ">365d"]
                
                e["Bucket"] = pd.cut(days_diff, bins=new_bins, labels=new_labels, include_lowest=True)
                
                agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                
                max_val = max(1.0, agg["Net_Dollars"].abs().max())
                html_out = []
                for _, r in agg.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    pct = (abs(r['Net_Dollars']) / max_val) * 100
                    val_str = fmt_neg(r["Net_Dollars"])
                    html_out.append(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
                
                st.markdown("".join(html_out), unsafe_allow_html=True)
            
            st.caption("ℹ️ You can exclude individual trades from the graphic by unchecking them in the Data Tables box below.")

def run_price_divergences_app(df_global):
    st.title("📉 Price Divergences")
    
    st.markdown("""
        <style>
        .top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; }
        [data-testid="stDataFrame"] th { font-weight: 900 !important; }
        </style>
        """, unsafe_allow_html=True)

    # --- Session State Init (Divergences Only) ---
    if 'saved_rsi_div_lookback' not in st.session_state: st.session_state.saved_rsi_div_lookback = 90
    if 'saved_rsi_div_source' not in st.session_state: st.session_state.saved_rsi_div_source = "High/Low"
    if 'saved_rsi_div_strict' not in st.session_state: st.session_state.saved_rsi_div_strict = "Yes"
    if 'saved_rsi_div_days_since' not in st.session_state: st.session_state.saved_rsi_div_days_since = 25
    if 'saved_rsi_div_diff' not in st.session_state: st.session_state.saved_rsi_div_diff = 2.0
    
    # History Tab State
    if 'rsi_hist_ticker' not in st.session_state: st.session_state.rsi_hist_ticker = "AMZN"
    if 'rsi_hist_results' not in st.session_state: st.session_state.rsi_hist_results = None
    if 'rsi_hist_last_run_params' not in st.session_state: st.session_state.rsi_hist_last_run_params = {}
    
    # Bulk History State
    if 'rsi_hist_bulk_df' not in st.session_state: st.session_state.rsi_hist_bulk_df = None

    def save_rsi_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
        
    dataset_map = DATA_KEYS_PARQUET
    options = list(dataset_map.keys())
    
    # CSV Hardcoded Defaults
    CSV_PERIODS_DAYS = [5, 21, 63, 126, 252]
    CSV_PERIODS_WEEKS = [4, 13, 26, 52]

    # --- HELPER FUNCTIONS FOR EXPORT ---
    def inject_volume(results_list, data_df):
        """Looks up Volume for P1 and Signal dates and adds to results."""
        if not results_list or data_df is None or data_df.empty:
            return results_list
        
        # 1. Identify Volume Column
        vol_col = next((c for c in data_df.columns if c.strip().upper() == 'VOLUME'), None)
        if not vol_col: return results_list

        # 2. Identify Date Index/Column for Lookup
        lookup = {}
        # Try to create a map: 'YYYY-MM-DD' -> Volume
        try:
            temp_df = data_df.copy()
            date_col = next((c for c in temp_df.columns if 'DATE' in c.upper()), None)
            
            if date_col:
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                temp_df['__date_str'] = temp_df[date_col].dt.strftime('%Y-%m-%d')
                lookup = dict(zip(temp_df['__date_str'], temp_df[vol_col]))
            elif isinstance(temp_df.index, pd.DatetimeIndex):
                # If date is in index
                temp_df['__date_str'] = temp_df.index.strftime('%Y-%m-%d')
                lookup = dict(zip(temp_df['__date_str'], temp_df[vol_col]))
        except:
            return results_list
        
        # 3. Inject
        for row in results_list:
            d1 = row.get('P1_Date_ISO')
            d2 = row.get('Signal_Date_ISO')
            row['Vol1'] = lookup.get(d1, np.nan)
            row['Vol2'] = lookup.get(d2, np.nan)
        
        return results_list

    def process_export_columns(df_in):
        """Renames Ret_XX columns to D_Ret_XX or W_Ret_XX based on timeframe."""
        if df_in.empty: return df_in
        out = df_in.copy()
        
        # Explicit Divergence Type Column (User Request)
        if 'Type' in out.columns:
            out['Divergence Type'] = out['Type']

        # Rename Returns
        cols = out.columns
        ret_cols = [c for c in cols if c.startswith('Ret_')]
        
        for rc in ret_cols:
            # Create D_Ret_5 and W_Ret_5
            d_col_name = f"D_{rc}"
            w_col_name = f"W_{rc}"
            
            # Logic: If row is Daily, put value in D_, else None
            out[d_col_name] = out.apply(lambda x: x[rc] if x.get('Timeframe') == 'Daily' else None, axis=1)
            out[w_col_name] = out.apply(lambda x: x[rc] if x.get('Timeframe') == 'Weekly' else None, axis=1)
        
        # Drop original ambiguous columns
        out = out.drop(columns=ret_cols)
        
        # Reorder for neatness: Put ID stuff first, then Techs, then Vols, then Returns
        first_cols = ['Ticker', 'Divergence Type', 'Timeframe', 'Signal_Date_ISO', 'P1_Date_ISO', 'Price1', 'Price2', 'RSI1', 'RSI2', 'Vol1', 'Vol2']
        existing_first = [c for c in first_cols if c in out.columns]
        other_cols = [c for c in out.columns if c not in existing_first]
        out = out[existing_first + other_cols]
        
        return out

    # --- TABS ---
    tab_div, tab_hist = st.tabs(["📉 Active/Recent Divergences", "📜 Divergences History"])

    # --------------------------------------------------------------------------
    # TAB 1: ACTIVE DIVERGENCES
    # --------------------------------------------------------------------------
    with tab_div:
        data_option_div = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_div_pills")
        
        with st.expander("ℹ️ Page User Guide"):
            c_guide1, c_guide2 = st.columns(2)
            with c_guide1:
                st.markdown("#### ⚙️ Settings & Inputs")
                st.markdown("""
                * **Dataset**: Selects the universe of stocks to scan (e.g., SP500, NASDAQ).
                * **Days Since Signal**: Filters the view to show only signals that were confirmed within this number of past trading days.
                * **Min RSI Delta**: The minimum required difference between the two RSI pivot points.
                * **Strict 50-Cross**: If "Yes", signal is invalid if RSI crossed 50 between pivots.
                """)
            with c_guide2:
                st.markdown("#### 📊 Table Columns")
                st.markdown("""
                * **RSI Δ**: RSI value at first pivot vs second pivot.
                * **Price Δ**: Price at first pivot vs second pivot.
                * **RSI %ile (New)**: The historical percentile rank of the **2nd Pivot (The Signal Candle)**.
                    * **Value**: Shows how rare the signal RSI is (e.g., **5** = Bottom 5% of all history).
                    * **Highlighting**: The cell turns YELLOW if **EITHER** pivot (1 or 2) was historically extreme (<10% or >90%).
                """)
        
        if data_option_div:
            try:
                key = dataset_map[data_option_div]
                master = load_parquet_and_clean(key)
                
                if master is not None and not master.empty:
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    with st.expander(f"View Scanned Tickers ({data_option_div})"):
                        if t_col:
                            unique_tickers = sorted(master[t_col].unique().tolist())
                            st.write(", ".join(unique_tickers))
                        else:
                            st.caption("No ticker column found in dataset.")

                    target_highlight_daily = None
                    highlight_list_weekly = []
                    
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    if date_col_raw:
                        master[date_col_raw] = pd.to_datetime(master[date_col_raw])
                        max_dt_obj = master[date_col_raw].max()
                        target_highlight_daily = max_dt_obj.strftime('%Y-%m-%d')
                        
                        days_to_subtract = max_dt_obj.weekday()
                        current_week_monday = (max_dt_obj - timedelta(days=days_to_subtract))
                        prev_week_monday = current_week_monday - timedelta(days=7)
                        highlight_list_weekly = [current_week_monday.strftime('%Y-%m-%d'), prev_week_monday.strftime('%Y-%m-%d')]
                    
                    # --- INPUTS ---
                    c_d1, c_d2, c_d3, c_d4, c_d5 = st.columns(5)
                    with c_d1: days_since = st.number_input("Days Since Signal", min_value=1, value=st.session_state.saved_rsi_div_days_since, step=1, key="rsi_div_days_since", on_change=save_rsi_state, args=("rsi_div_days_since", "saved_rsi_div_days_since"))
                    with c_d2: div_diff = st.number_input("Min RSI Delta", min_value=0.5, value=st.session_state.saved_rsi_div_diff, step=0.5, key="rsi_div_diff", on_change=save_rsi_state, args=("rsi_div_diff", "saved_rsi_div_diff"))
                    with c_d3: div_lookback = st.number_input("Max Candle Between Pivots", min_value=30, value=st.session_state.saved_rsi_div_lookback, step=5, key="rsi_div_lookback", on_change=save_rsi_state, args=("rsi_div_lookback", "saved_rsi_div_lookback"))
                    with c_d4:
                         curr_strict = st.session_state.saved_rsi_div_strict
                         idx_strict = ["Yes", "No"].index(curr_strict) if curr_strict in ["Yes", "No"] else 0
                         strict_div_str = st.selectbox("Strict 50-Cross Invalidation", ["Yes", "No"], index=idx_strict, key="rsi_div_strict", on_change=save_rsi_state, args=("rsi_div_strict", "saved_rsi_div_strict"))
                         strict_div = (strict_div_str == "Yes")
                    with c_d5:
                         curr_source = st.session_state.saved_rsi_div_source
                         idx_source = ["High/Low", "Close"].index(curr_source) if curr_source in ["High/Low", "Close"] else 0
                         div_source = st.selectbox("Candle Price Methodology", ["High/Low", "Close"], index=idx_source, key="rsi_div_source", on_change=save_rsi_state, args=("rsi_div_source", "saved_rsi_div_source"))
                    
                    raw_results_div = []
                    progress_bar = st.progress(0, text="Scanning Divergences...")
                    grouped = master.groupby(t_col)
                    grouped_list = list(grouped)
                    total_groups = len(grouped_list)
                    
                    for i, (ticker, group) in enumerate(grouped_list):
                        d_d, d_w = prepare_data(group.copy())
                        
                        # --- Process Daily ---
                        if d_d is not None:
                            daily_divs = find_divergences(d_d, ticker, 'Daily', min_n=0, periods_input=CSV_PERIODS_DAYS, optimize_for='PF', lookback_period=div_lookback, price_source=div_source, strict_validation=strict_div, recent_days_filter=days_since, rsi_diff_threshold=div_diff)
                            
                            # Inject RSI Percentiles
                            if daily_divs and 'RSI' in d_d.columns:
                                all_rsi = d_d['RSI'].dropna().values
                                if len(all_rsi) > 0:
                                    for div in daily_divs:
                                        p1 = (all_rsi < div['RSI1']).mean() * 100
                                        p2 = (all_rsi < div['RSI2']).mean() * 100
                                        div['RSI1_Pct'] = p1
                                        div['RSI2_Pct'] = p2
                                        div['Extreme_Flag'] = (p1 < 10 or p2 < 10) if div['Type'] == 'Bullish' else (p1 > 90 or p2 > 90)
                            
                            raw_results_div.extend(daily_divs)
                        
                        # --- Process Weekly ---
                        if d_w is not None: 
                            weekly_divs = find_divergences(d_w, ticker, 'Weekly', min_n=0, periods_input=CSV_PERIODS_WEEKS, optimize_for='PF', lookback_period=div_lookback, price_source=div_source, strict_validation=strict_div, recent_days_filter=days_since, rsi_diff_threshold=div_diff)
                            
                            # Inject RSI Percentiles
                            if weekly_divs and 'RSI' in d_w.columns:
                                all_rsi_w = d_w['RSI'].dropna().values
                                if len(all_rsi_w) > 0:
                                    for div in weekly_divs:
                                        p1 = (all_rsi_w < div['RSI1']).mean() * 100
                                        p2 = (all_rsi_w < div['RSI2']).mean() * 100
                                        div['RSI1_Pct'] = p1
                                        div['RSI2_Pct'] = p2
                                        div['Extreme_Flag'] = (p1 < 10 or p2 < 10) if div['Type'] == 'Bullish' else (p1 > 90 or p2 > 90)

                            raw_results_div.extend(weekly_divs)
                            
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    
                    progress_bar.empty()
                    
                    if raw_results_div:
                        df_all_results = pd.DataFrame(raw_results_div)
                        # --- UI DISPLAY ---
                        res_div_df = df_all_results[df_all_results["Is_Recent"] == True].copy()
                        
                        if res_div_df.empty:
                            st.warning(f"No signals found in the last {days_since} days.")
                        else:
                            res_div_df = res_div_df.sort_values(by='Signal_Date_ISO', ascending=False)
                            consolidated = res_div_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                            
                            for tf in ['Daily', 'Weekly']:
                                targets = highlight_list_weekly if tf == 'Weekly' else ([target_highlight_daily] if target_highlight_daily else [])
                                date_header = "Week Δ" if tf == 'Weekly' else "Day Δ"
                                
                                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                    st.subheader(f"{emoji} {tf} {s_type} Signals")
                                    tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                                    price_header = "Close Price Δ" if div_source == 'Close' else ("Low Price Δ" if s_type == 'Bullish' else "High Price Δ")
                                    
                                    pct_col_title = "RSI Low %ile" if s_type == 'Bullish' else "RSI High %ile"

                                    if not tbl_df.empty:
                                        if 'RSI2_Pct' not in tbl_df.columns: tbl_df['RSI2_Pct'] = 50
                                        if 'Extreme_Flag' not in tbl_df.columns: tbl_df['Extreme_Flag'] = False

                                        def style_div_df(df_in):
                                            def highlight_cells(row):
                                                styles = [''] * len(row)
                                                if row['Signal_Date_ISO'] in targets:
                                                    if 'Date_Display' in df_in.columns:
                                                        idx = df_in.columns.get_loc('Date_Display')
                                                        styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                                
                                                if row.get('Extreme_Flag', False):
                                                    if 'RSI2_Pct' in df_in.columns:
                                                        idx_p = df_in.columns.get_loc('RSI2_Pct')
                                                        styles[idx_p] = 'background-color: rgba(255, 235, 59, 0.25); color: #f57f17; font-weight: bold;'

                                                return styles
                                            return df_in.style.apply(highlight_cells, axis=1)

                                        st.dataframe(
                                            style_div_df(tbl_df),
                                            column_config={
                                                "Ticker": st.column_config.TextColumn("Ticker"),
                                                "Tags": st.column_config.ListColumn("Tags"), 
                                                "Date_Display": st.column_config.TextColumn(date_header),
                                                "RSI_Display": st.column_config.TextColumn("RSI Δ"),
                                                "RSI2_Pct": st.column_config.NumberColumn(pct_col_title, format="%d", help="Percentile rank of the signal RSI relative to ticker history."),
                                                "Price_Display": st.column_config.TextColumn(price_header),
                                                "Last_Close": st.column_config.TextColumn("Last Close"),
                                            },
                                            column_order=["Ticker", "Tags", "Date_Display", "RSI_Display", "Price_Display", "Last_Close", "RSI2_Pct"],
                                            hide_index=True,
                                            use_container_width=True,
                                            height=get_table_height(tbl_df, max_rows=50)
                                        )
                                    else: st.info("No signals.")
                    else: st.warning("No Divergence signals found.")
                else: st.error(f"Failed to load dataset.")
            except Exception as e: st.error(f"Analysis failed: {e}")

    # --------------------------------------------------------------------------
    # TAB 2: DIV HISTORY
    # --------------------------------------------------------------------------
    with tab_hist:
        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        with c_h1:
            hist_ticker_in = st.text_input("Ticker", value=st.session_state.rsi_hist_ticker, key="rsi_hist_ticker_in").strip().upper()
            st.session_state.rsi_hist_ticker = hist_ticker_in
        with c_h2: 
            h_lookback = st.number_input("Max Days Btwn Pivots", min_value=30, value=90, step=5, key="rsi_hist_lookback")
        with c_h3: 
            h_diff = st.number_input("Min RSI Delta", min_value=0.5, value=2.0, step=0.5, key="rsi_hist_diff")
        with c_h4:
            h_strict_str = st.selectbox("50-Cross Inval", ["Yes", "No"], index=0, key="rsi_hist_strict")
            h_strict = (h_strict_str == "Yes")

        c_h5, c_h6, c_h7 = st.columns(3)
        with c_h5: 
            h_source = st.selectbox("Candle Price Method", ["High/Low", "Close"], index=0, key="rsi_hist_source")
        with c_h6: 
            h_per_days = st.text_input("Periods (Days)", value="5, 21, 63, 126, 252", key="rsi_hist_p_days")
        with c_h7: 
            h_per_weeks = st.text_input("Periods (Weeks)", value="4, 13, 26, 52", key="rsi_hist_p_weeks")

        # --- ANALYSIS LOGIC (SINGLE TICKER) ---
        current_params = {"t": hist_ticker_in, "lb": h_lookback, "str": h_strict, "src": h_source, "pd": h_per_days, "pw": h_per_weeks, "diff": h_diff}
        run_hist = False
        if current_params != st.session_state.rsi_hist_last_run_params: run_hist = True
        
        if hist_ticker_in and run_hist:
            with st.spinner(f"Analyzing lifetime history for {hist_ticker_in}..."):
                try:
                    ticker_map = load_ticker_map()
                    df_h = get_ticker_technicals(hist_ticker_in, ticker_map)
                    if df_h is None or df_h.empty: df_h = fetch_yahoo_data(hist_ticker_in)
                    
                    if df_h is not None and not df_h.empty:
                        d_d_h, d_w_h = prepare_data(df_h.copy())
                        
                        raw_results_hist = []
                        p_days_parsed = parse_periods(h_per_days)
                        p_weeks_parsed = parse_periods(h_per_weeks)
                        
                        if d_d_h is not None: 
                            d_daily = find_divergences(d_d_h, hist_ticker_in, 'Daily', min_n=0, periods_input=p_days_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                            d_daily = inject_volume(d_daily, d_d_h) # Inject Vol
                            raw_results_hist.extend(d_daily)
                            
                        if d_w_h is not None: 
                            d_weekly = find_divergences(d_w_h, hist_ticker_in, 'Weekly', min_n=0, periods_input=p_weeks_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                            d_weekly = inject_volume(d_weekly, d_w_h) # Inject Vol
                            raw_results_hist.extend(d_weekly)
                            
                        st.session_state.rsi_hist_results = pd.DataFrame(raw_results_hist)
                        st.session_state.rsi_hist_last_run_params = current_params
                    else:
                        st.error(f"Could not load data for {hist_ticker_in}")
                        st.session_state.rsi_hist_results = pd.DataFrame()
                except Exception as e: st.error(f"Error: {e}")
        
        # ==============================================================================
        # EXISTING TABLES DISPLAY
        # ==============================================================================
        if st.session_state.rsi_hist_results is not None and not st.session_state.rsi_hist_results.empty:
            res_df_h = st.session_state.rsi_hist_results.copy().sort_values(by='Signal_Date_ISO', ascending=False)
            
            for tf in ['Daily', 'Weekly']:
                p_cols_to_show = []
                current_periods = parse_periods(h_per_days if tf == 'Daily' else h_per_weeks)
                for p in current_periods:
                    col_key = f"Ret_{p}"
                    if col_key in res_df_h.columns: p_cols_to_show.append(col_key)

                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {tf} {s_type} History")
                    tbl_df = res_df_h[(res_df_h['Type']==s_type) & (res_df_h['Timeframe']==tf)].copy()
                    
                    if not tbl_df.empty:
                        def style_ret(df_in):
                            def highlight_val(val):
                                if pd.isna(val): return ''
                                color = '#1e7e34' if val > 0 else '#c5221f'
                                return f'color: {color}; font-weight: bold;'
                            style_obj = df_in.style
                            for p_c in p_cols_to_show: style_obj = style_obj.map(highlight_val, subset=[p_c])
                            return style_obj

                        cfg = {
                            "P1_Date_ISO": st.column_config.TextColumn("Date 1", width="medium"),
                            "Signal_Date_ISO": st.column_config.TextColumn("Date 2", width="medium"),
                            "RSI1": st.column_config.NumberColumn("RSI 1", format="%.0f"),
                            "RSI2": st.column_config.NumberColumn("RSI 2", format="%.0f"),
                            "Price1": st.column_config.NumberColumn("Price 1", format="$%.2f"),
                            "Price2": st.column_config.NumberColumn("Price 2", format="$%.2f"),
                        }
                        for p_c in p_cols_to_show:
                            days = p_c.split('_')[1]
                            cfg[p_c] = st.column_config.NumberColumn(f"{days}{'d' if tf=='Daily' else 'w'} %", format="%+.2f%%")
                        
                        cols_base = ["P1_Date_ISO", "Signal_Date_ISO", "RSI1", "RSI2", "Price1", "Price2"]
                        st.dataframe(
                            style_ret(tbl_df[cols_base + p_cols_to_show]),
                            column_config=cfg,
                            hide_index=True,
                            use_container_width=True,
                            height=(min(len(tbl_df), 50) + 1) * 35 
                        )
                    else: st.caption("No signals found.")
        
        # ==============================================================================
        # DOWNLOAD SECTION (MOVED TO BOTTOM)
        # ==============================================================================
        st.divider()
        st.subheader("💾 Data Downloads")
        
        col_dl_1, col_dl_2 = st.columns(2)
        
        # --- BOX 1: Single Ticker Download ---
        with col_dl_1:
            st.markdown(f"**Option 1: {hist_ticker_in} Complete History**")
            st.caption("Download all Daily/Weekly, Bullish/Bearish divergences for this specific ticker.")
            if st.session_state.rsi_hist_results is not None and not st.session_state.rsi_hist_results.empty:
                # Prepare CSV with renamed columns and explicit Types
                export_df = process_export_columns(st.session_state.rsi_hist_results)
                
                csv_single = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⬇️ Download {hist_ticker_in} History (CSV)",
                    data=csv_single,
                    file_name=f"{hist_ticker_in}_Divergence_History.csv",
                    mime="text/csv",
                    key="dl_single_ticker_hist"
                )
            else:
                st.info("Input a ticker above to generate data.")

        # --- BOX 2: Bulk Dataset Download ---
        with col_dl_2:
            st.markdown(f"**Option 2: Bulk Dataset History**")
            st.caption("Scan complete history for EVERY ticker in the selected dataset.")
            
            # Selector specifically for this bulk download action
            bulk_dataset_opt = st.selectbox("Select Dataset", options=options, index=0, key="rsi_hist_bulk_sel", label_visibility="collapsed")
            
            if st.button("🚀 Process Bulk History", key="btn_bulk_hist"):
                try:
                    key = dataset_map[bulk_dataset_opt]
                    master_bulk = load_parquet_and_clean(key)
                    if master_bulk is not None and not master_bulk.empty:
                        t_col_b = next((c for c in master_bulk.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                        
                        if t_col_b:
                            bulk_results = []
                            grouped_bulk = master_bulk.groupby(t_col_b)
                            total_b = len(grouped_bulk)
                            prog_b = st.progress(0, text="Processing Bulk History...")
                            
                            p_days_parsed = parse_periods(h_per_days)
                            p_weeks_parsed = parse_periods(h_per_weeks)
                            
                            for idx, (tkr, grp) in enumerate(grouped_bulk):
                                d_d_b, d_w_b = prepare_data(grp.copy())
                                
                                if d_d_b is not None: 
                                    res_d = find_divergences(d_d_b, tkr, 'Daily', min_n=0, periods_input=p_days_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                                    res_d = inject_volume(res_d, d_d_b) # Inject Vol
                                    bulk_results.extend(res_d)
                                    
                                if d_w_b is not None:
                                    res_w = find_divergences(d_w_b, tkr, 'Weekly', min_n=0, periods_input=p_weeks_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                                    res_w = inject_volume(res_w, d_w_b) # Inject Vol
                                    bulk_results.extend(res_w)
                                
                                if idx % 5 == 0: prog_b.progress((idx+1)/total_b)
                            
                            prog_b.empty()
                            
                            if bulk_results:
                                st.session_state.rsi_hist_bulk_df = pd.DataFrame(bulk_results)
                            else:
                                st.warning("No divergences found in dataset.")
                        else: st.error("Ticker column missing in dataset.")
                except Exception as e: st.error(f"Bulk Process Error: {e}")

            # Show Download Button if data exists in session state
            if st.session_state.rsi_hist_bulk_df is not None and not st.session_state.rsi_hist_bulk_df.empty:
                # Prepare CSV with renamed columns and explicit Types
                bulk_export = process_export_columns(st.session_state.rsi_hist_bulk_df)
                
                csv_bulk = bulk_export.to_csv(index=False).encode('utf-8')
                st.success(f"Ready: {len(bulk_export)} rows generated.")
                st.download_button(
                    label="⬇️ Download Full Dataset History (CSV)",
                    data=csv_bulk,
                    file_name=f"Bulk_Divergence_History_{date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="dl_bulk_ticker_hist"
                )
        st.divider()

def run_rsi_scanner_app(df_global):
    st.title("🤖 RSI Scanner")

    # Session State
    if 'saved_rsi_pct_low' not in st.session_state: st.session_state.saved_rsi_pct_low = 10
    if 'saved_rsi_pct_high' not in st.session_state: st.session_state.saved_rsi_pct_high = 90
    if 'saved_rsi_pct_min_n' not in st.session_state: st.session_state.saved_rsi_pct_min_n = 1
    if 'saved_rsi_pct_periods' not in st.session_state: st.session_state.saved_rsi_pct_periods = "5, 21, 63, 126, 252"

    def save_rsi_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]

    dataset_map = DATA_KEYS_PARQUET
    options = list(dataset_map.keys())

    # --- HELPER: ROBUST DATA LOADING ---
    def load_tech_data(symbol, ticker_map_ref):
        d = get_ticker_technicals(symbol, ticker_map_ref)
        if d is None or d.empty: 
            d = fetch_yahoo_data(symbol)
        
        if d is not None and not d.empty:
            # 1. Normalize all cols to UPPERCASE
            d.columns = [c.strip().upper() for c in d.columns]
            
            # 2. Map 'ADJ CLOSE' or 'Adj Close' to 'CLOSE' if missing
            if 'CLOSE' not in d.columns and 'ADJ CLOSE' in d.columns:
                 d = d.rename(columns={'ADJ CLOSE': 'CLOSE'})
            
            # 3. Last ditch: find any column with 'CLOSE' in it
            if 'CLOSE' not in d.columns:
                possible = [c for c in d.columns if 'CLOSE' in c]
                if possible: d = d.rename(columns={possible[0]: 'CLOSE'})
        return d

    # --- TABS ---
    tab_bot, tab_pct = st.tabs(["🤖 Contextual Backtester", "🔢 RSI Percentiles"])

    # --------------------------------------------------------------------------
    # TAB 1: CONTEXTUAL BACKTESTER
    # --------------------------------------------------------------------------
    with tab_bot:
        # --- LAYOUT: INPUTS ---
        c_left, c_mid, c_right = st.columns([1, 0.2, 3])
        
        with c_left:
            st.markdown("#### 1. Asset & Scope")
            ticker = st.text_input("Ticker", value="NFLX", key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=20, value=10)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=10.0, value=2.0, step=0.5, help="Search for RSI +/- this value.")
            
            # --- NEW: Historical Date Input ---
            use_hist_date = st.checkbox("Test from Past Date", value=False, help="Enable this to see what the tool would have shown on a specific day in the past.")
            if use_hist_date:
                ref_date = st.date_input("Select Reference Date", value=date.today() - timedelta(days=5))
            else:
                ref_date = date.today()
            # ----------------------------------

            # De-duping logic as dropdown
            dedupe_str = st.selectbox("De-dupe Signals", ["Yes", "No"], index=0, help="If Yes (Recommended): Simulates 'One Trade at a Time'. If you buy on Day 1 for a 21-day hold, ignores all other signals until Day 22.\n\nIf No: Counts EVERY signal day as a new trade (can inflate stats during crashes).")
            dedupe_signals = (dedupe_str == "Yes")
            
        with c_right:
            st.markdown("#### 2. Contextual Filters")
            
            # Stacked Filters (Vertical)
            f_sma200 = st.selectbox("Price vs 200 SMA", ["Any", "Above", "Below"], index=0, key="f_sma200")
            f_sma50 = st.selectbox("Price vs 50 SMA", ["Any", "Above", "Below"], index=0, key="f_sma50")

        st.divider()
        
        # --- EXECUTION ---
        if ticker:
            ticker_map = load_ticker_map()
            
            # 1. Fetch MAIN Ticker
            df = load_tech_data(ticker, ticker_map)
            
            if df is None or df.empty:
                st.error(f"Could not retrieve data for {ticker}.")
            else:
                # --- PRE-PROCESS MAIN DATA ---
                date_col = next((c for c in df.columns if 'DATE' in c), None)
                close_col = 'CLOSE'
                
                if not date_col or not close_col:
                    st.error(f"Data format error: Missing DATE or CLOSE columns for {ticker}")
                else:
                    # Date Handling
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(by=date_col).reset_index(drop=True)
                    
                    # Calc RSI if missing
                    if 'RSI' not in df.columns:
                        delta = df[close_col].diff()
                        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                        rs = gain / loss
                        df['RSI'] = 100 - (100 / (1 + rs))

                    # Calc MAs for Filters
                    df['SMA50'] = df[close_col].rolling(50).mean()
                    df['SMA200'] = df[close_col].rolling(200).mean()

                    # --- CRITICAL UPDATE: TIME TRAVEL LOGIC ---
                    # 1. Keep full copy for forward returns (Truth Data)
                    df_full_future = df.copy() 
                    
                    # 2. Filter dataset to end at the Reference Date (Simulation Data)
                    df = df[df[date_col].dt.date <= ref_date].copy().reset_index(drop=True)
                    
                    if df.empty:
                        st.error(f"No data available before {ref_date}.")
                    else:
                        # Trim to Lookback relative to the NEW reference date
                        cutoff_date = df[date_col].max() - timedelta(days=365*lookback_years)
                        df = df[df[date_col] >= cutoff_date].copy().reset_index(drop=True)

                        # --- APPLY FILTERS ---
                        current_row = df.iloc[-1]
                        current_rsi = current_row['RSI']
                        
                        # Guard against NaN RSI (e.g. if date is too early in history)
                        if pd.isna(current_rsi):
                            st.error(f"RSI is NaN on {current_row[date_col].date()}. Need more history.")
                        else:
                            rsi_min, rsi_max = current_rsi - rsi_tol, current_rsi + rsi_tol
                            
                            # Base Filter: RSI Range
                            mask = (df['RSI'] >= rsi_min) & (df['RSI'] <= rsi_max)
                            
                            # Context Filters
                            if f_sma200 == "Above": mask &= (df[close_col] > df['SMA200'])
                            elif f_sma200 == "Below": mask &= (df[close_col] < df['SMA200'])
                            
                            if f_sma50 == "Above": mask &= (df[close_col] > df['SMA50'])
                            elif f_sma50 == "Below": mask &= (df[close_col] < df['SMA50'])
                            
                            # Apply Mask (Exclude the very last row which is our "current reference")
                            matches = df.iloc[:-1][mask[:-1]].copy()
                            
                            # --- PERCENTILE RANK ---
                            rsi_rank = (df['RSI'] < current_rsi).mean() * 100

                            # --- DISPLAY ---
                            ref_date_str = current_row[date_col].strftime('%Y-%m-%d')
                            st.subheader(f"📊 Analysis: {ticker} on {ref_date_str}")
                            
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            sc1.metric(f"Price ({ref_date_str})", f"${current_row[close_col]:.2f}")
                            sc2.metric("Reference RSI", f"{current_rsi:.1f}")
                            sc3.metric("RSI Hist. Rank", f"{rsi_rank:.1f}%", help=f"Percentile Rank: Bottom {rsi_rank:.1f}%")
                            sc4.metric("Total Signals", f"{len(matches)}", help="Raw count of days matching criteria (RSI + Filters) BEFORE De-duping is applied.")
                            
                            if not matches.empty:
                                match_indices = matches.index.values
                                
                                # Re-map match dates to the full future dataset to get accurate forward returns
                                # This ensures that even if we are 'simulating' the past, we use known future data for stats
                                match_dates = matches[date_col].values
                                full_df_idx_map = df_full_future[df_full_future[date_col].isin(match_dates)].index.values
                                
                                full_closes = df_full_future[close_col].values
                                total_len = len(full_closes)
                                
                                results = []
                                trade_log = [] 
                                
                                # 1. Capture Raw Signals
                                for i_raw in match_indices:
                                    trade_log.append({
                                        "Period": "Raw Signal",
                                        "Entry Date": df.iloc[i_raw][date_col].strftime('%Y-%m-%d'),
                                        "Entry Price": df.iloc[i_raw][close_col],
                                        "RSI": df.iloc[i_raw]['RSI'],
                                        "Exit Date": None, "Exit Price": None, "Return %": None, "Max Drawdown %": None
                                    })

                                periods = [5, 10, 21, 42, 63, 126, 252]
                                
                                with st.spinner("Optimizing entry strategies..."):
                                    for p in periods:
                                        lump_returns = []
                                        drawdowns = []
                                        valid_counts = 0
                                        
                                        last_exit_index = -1 
                                        
                                        # Iterate using mapped indices on the FULL dataset
                                        for idx in full_df_idx_map:
                                            if dedupe_signals and idx < last_exit_index: continue
                                            if idx + p >= total_len: continue
                                            
                                            entry_p = full_closes[idx]
                                            exit_p = full_closes[idx + p]
                                            
                                            # Drawdown
                                            period_prices = full_closes[idx+1 : idx+p+1]
                                            
                                            if len(period_prices) > 0:
                                                min_close_during_hold = np.min(period_prices)
                                                dd = (min_close_during_hold - entry_p) / entry_p
                                                drawdowns.append(dd)
                                                dd_val = dd * 100
                                            else:
                                                drawdowns.append(0.0)
                                                dd_val = 0.0
                                                
                                            ret = (exit_p - entry_p) / entry_p
                                            lump_returns.append(ret)
                                            valid_counts += 1
                                            
                                            # LOG TRADE
                                            trade_log.append({
                                                "Period": p,
                                                "Entry Date": df_full_future.iloc[idx][date_col].strftime('%Y-%m-%d'),
                                                "Entry Price": entry_p,
                                                "RSI": df_full_future.iloc[idx]['RSI'],
                                                "Exit Date": df_full_future.iloc[idx + p][date_col].strftime('%Y-%m-%d'),
                                                "Exit Price": exit_p,
                                                "Return %": ret * 100,
                                                "Max Drawdown %": dd_val
                                            })
                                            
                                            if dedupe_signals: last_exit_index = idx + p

                                        if valid_counts == 0: continue

                                        # OPTIMIZE DCA
                                        best_dca_ev = -999.0
                                        best_dca_days = 1
                                        best_dca_wr = 0.0
                                        
                                        lump_mean = np.mean(lump_returns) * 100
                                        lump_wr = np.mean(np.array(lump_returns) > 0) * 100
                                        
                                        best_dca_ev = lump_mean
                                        best_dca_wr = lump_wr
                                        
                                        for d_win in range(2, 11): 
                                            temp_dca_rets = []
                                            last_exit_index_dca = -1
                                            
                                            for idx in full_df_idx_map:
                                                if dedupe_signals and idx < last_exit_index_dca: continue
                                                if idx + d_win >= total_len or idx + p >= total_len: continue
                                                
                                                entries = full_closes[idx : idx + d_win]
                                                if len(entries) < d_win: continue
                                                avg_entry = np.mean(entries)
                                                exit_p = full_closes[idx + p]
                                                temp_dca_rets.append((exit_p - avg_entry) / avg_entry)
                                                
                                                if dedupe_signals: last_exit_index_dca = idx + p
                                            
                                            if temp_dca_rets:
                                                dca_mean = np.mean(temp_dca_rets) * 100
                                                if dca_mean > best_dca_ev:
                                                    best_dca_ev = dca_mean
                                                    best_dca_days = d_win
                                                    best_dca_wr = np.mean(np.array(temp_dca_rets) > 0) * 100

                                        dd_arr = np.array(drawdowns) * 100
                                        
                                        strat_text = "Lump Sum" if best_dca_days == 1 else f"DCA ({best_dca_days}d)"
                                        
                                        res = {
                                            "Days": p,
                                            "Count": valid_counts,
                                            "Lump EV": lump_mean,
                                            "Lump WR": lump_wr,
                                            "Optimal Entry": strat_text,
                                            "Optimal EV": best_dca_ev,
                                            "Optimal WR": best_dca_wr,
                                            "Avg DD": np.mean(dd_arr),
                                            "Median DD": np.median(dd_arr),
                                            "Min DD": np.max(dd_arr),
                                            "Max DD": np.min(dd_arr)
                                        }
                                        results.append(res)
                                        
                                res_df = pd.DataFrame(results)
                                
                                # --- HIGHLIGHTING LOGIC ---
                                def highlight_ev(val):
                                    if pd.isna(val) or val < 10.0: return ''
                                    return 'color: #71d28a; font-weight: bold;'
                                
                                def highlight_wr(val):
                                    if pd.isna(val) or val < 75.0: return ''
                                    return 'color: #71d28a; font-weight: bold;'
                                    
                                def color_dd(val):
                                    if val < -15: return 'color: #c5221f; font-weight: bold;' 
                                    return 'color: #e67e22;'
                                
                                col_order = ["Days", "Count", "Min DD", "Avg DD", "Median DD", "Max DD", "Lump EV", "Lump WR", "Optimal Entry", "Optimal EV", "Optimal WR"]
                                
                                st.dataframe(
                                    res_df[col_order].style
                                    .format({
                                        "Lump EV": "{:+.2f}%", "Lump WR": "{:.1f}%",
                                        "Optimal EV": "{:+.2f}%", "Optimal WR": "{:.1f}%",
                                        "Avg DD": "{:.1f}%", "Median DD": "{:.1f}%", 
                                        "Min DD": "{:.1f}%", "Max DD": "{:.1f}%"
                                    })
                                    .map(highlight_ev, subset=["Lump EV", "Optimal EV"])
                                    .map(highlight_wr, subset=["Lump WR", "Optimal WR"])
                                    .map(color_dd, subset=["Max DD", "Avg DD"]),
                                    column_config={
                                        "Days": st.column_config.NumberColumn("Hold", help="Trading Days held"),
                                        "Lump EV": st.column_config.NumberColumn("Lump EV", help="Avg Return (Lump Sum entry)."),
                                        "Lump WR": st.column_config.NumberColumn("Lump WR", help="Win Rate (Lump Sum entry)."),
                                        "Optimal Entry": st.column_config.TextColumn("Best Entry", help="Strategy (Lump Sum vs DCA) with highest historical EV."),
                                        "Optimal EV": st.column_config.NumberColumn("Best EV", help="Avg Return of the Best Entry strategy."),
                                        "Optimal WR": st.column_config.NumberColumn("Best WR", help="Win Rate of the Best Entry strategy."),
                                        "Min DD": st.column_config.NumberColumn("Lump Min DD", help="Smallest (Best Case) Asset Drawdown."),
                                        "Max DD": st.column_config.NumberColumn("Lump Max DD", help="Largest (Worst Case) Asset Drawdown."),
                                        "Avg DD": st.column_config.NumberColumn("Lump Avg DD", help="Average Asset Drawdown."),
                                        "Median DD": st.column_config.NumberColumn("Lump Med DD", help="Median Asset Drawdown."),
                                    },
                                    use_container_width=True,
                                    hide_index=True,
                                    height=get_table_height(res_df, max_rows=10)
                                )
                                
                                st.markdown("##### 🧠 Strategic Insights")
                                c_i1, c_i2 = st.columns(2)
                                
                                if not res_df.empty:
                                    # 1. Best Hold Recommendation
                                    best_row = res_df.loc[res_df['Optimal EV'].idxmax()]
                                    
                                    with c_i1:
                                        st.success(f"""
                                        **🏆 Best Historical Hold**
                                        If you entered on {ref_date_str} (RSI {current_rsi:.1f}), the best historical strategy was holding for **{best_row['Days']} Days**.
                                        * **Avg Return:** +{best_row['Optimal EV']:.2f}%
                                        * **Win Rate:** {best_row['Optimal WR']:.1f}%
                                        * **Strategy:** {best_row['Optimal Entry']}
                                        """)
                                    
                                    # 2. Download Button (Close Only Logic)
                                    with c_i2:
                                        if trade_log:
                                            trade_log_df = pd.DataFrame(trade_log)
                                            csv = trade_log_df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                label="⬇️ Download Trade History (CSV)",
                                                data=csv,
                                                file_name=f"{ticker}_RSI_Backtest_{ref_date_str}.csv",
                                                mime="text/csv",
                                                help="Downloads log of all trades using Close Prices."
                                            )
                                
                                # --- DRILL DOWN SECTION ---
                                st.divider()
                                st.subheader("🔍 Trade Drill-Down")
                                
                                if trade_log:
                                    df_log = pd.DataFrame(trade_log)
                                    # Get unique periods (excluding "Raw Signal") sorted numerically
                                    unique_periods = sorted([p for p in df_log['Period'].unique() if isinstance(p, int)])
                                    
                                    # Create label map for dropdown
                                    period_opts = [f"{p} Days" for p in unique_periods]
                                    
                                    sel_period_str = st.selectbox("Select Holding Period to Inspect", period_opts, index=len(period_opts)-1) # Default to longest
                                    
                                    if sel_period_str:
                                        # Extract int back from string
                                        sel_p_int = int(sel_period_str.split(" ")[0])
                                        subset = df_log[df_log['Period'] == sel_p_int].copy()
                                        
                                        st.dataframe(
                                            subset[["Entry Date", "Entry Price", "Exit Date", "Exit Price", "Return %", "Max Drawdown %"]].style
                                            .format({
                                                "Entry Price": "${:,.2f}", 
                                                "Exit Price": "${:,.2f}", 
                                                "Return %": "{:+.2f}%", 
                                                "Max Drawdown %": "{:+.2f}%"
                                            })
                                            .map(lambda x: 'color: #c5221f; font-weight: bold;' if x < -15 else '', subset=['Max Drawdown %'])
                                            .map(lambda x: 'color: #71d28a; font-weight: bold;' if x > 0 else 'color: #f29ca0;', subset=['Return %']),
                                            use_container_width=True,
                                            hide_index=True
                                        )

                            else:
                                st.warning(f"No historical matches found for RSI {current_rsi:.1f} (+/- {rsi_tol}). Try widening tolerance.")

    # --------------------------------------------------------------------------
    # TAB 2: RSI PERCENTILES (Unchanged)
    # --------------------------------------------------------------------------
    with tab_pct:
        data_option_pct = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_pct_pills")
        
        if data_option_pct:
            try:
                key = dataset_map[data_option_pct]
                master = load_parquet_and_clean(key)
                if master is not None and not master.empty:
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    with st.expander(f"View Scanned Tickers ({data_option_pct})"):
                        if t_col:
                            unique_tickers = sorted(master[t_col].unique().tolist())
                            st.write(", ".join(unique_tickers))
                        else:
                            st.caption("No ticker column found in dataset.")

                    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                    with c_p1: pct_low = st.number_input("RSI Low (e.g. 10)", min_value=1, max_value=40, value=st.session_state.saved_rsi_pct_low, step=1, key="rsi_pct_low", on_change=save_rsi_state, args=("rsi_pct_low", "saved_rsi_pct_low"))
                    with c_p2: pct_high = st.number_input("RSI High (e.g. 90)", min_value=60, max_value=99, value=st.session_state.saved_rsi_pct_high, step=1, key="rsi_pct_high", on_change=save_rsi_state, args=("rsi_pct_high", "saved_rsi_pct_high"))
                    with c_p3: min_n_pct = st.number_input("Min N", min_value=1, value=st.session_state.saved_rsi_pct_min_n, step=1, key="rsi_pct_min_n", on_change=save_rsi_state, args=("rsi_pct_min_n", "saved_rsi_pct_min_n"))
                    with c_p4: periods_str_pct = st.text_input("Periods", value=st.session_state.saved_rsi_pct_periods, key="rsi_pct_periods", on_change=save_rsi_state, args=("rsi_pct_periods", "saved_rsi_pct_periods"))
            
                    periods_pct = parse_periods(periods_str_pct)
                    raw_results_pct = []
                    
                    prog_bar = st.progress(0, text="Scanning Percentiles...")
                    grouped = master.groupby(t_col)
                    total_groups = len(grouped)
                    
                    for i, (ticker, group) in enumerate(grouped):
                        d_d, _ = prepare_data(group.copy())
                        if d_d is not None:
                            raw_results_pct.extend(find_rsi_percentile_signals(d_d, ticker, pct_low=pct_low/100.0, pct_high=pct_high/100.0, min_n=min_n_pct, timeframe='Daily', periods_input=periods_pct, optimize_for='SQN'))
                        if i % 10 == 0: prog_bar.progress((i+1)/total_groups)
                    
                    prog_bar.empty()
                    
                    if raw_results_pct:
                        df_pct = pd.DataFrame(raw_results_pct)
                        df_pct = df_pct.sort_values(by=['Date_Obj', 'Ticker'], ascending=[False, True])
                        
                        st.dataframe(
                            df_pct,
                            column_config={
                                "Ticker": st.column_config.TextColumn("Ticker"),
                                "Date": st.column_config.TextColumn("Date"),
                                "Action": st.column_config.TextColumn("Signal"),
                                "RSI_Display": st.column_config.TextColumn("RSI Transition"),
                                "Signal_Price": st.column_config.TextColumn("Signal Price"),
                                "Last_Close": st.column_config.TextColumn("Last Close"),
                                "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                                "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                                "EV": st.column_config.NumberColumn("EV", format="%.1f%%"),
                                "EV Target": st.column_config.NumberColumn("EV Target", format="$%.2f"),
                                "SQN": st.column_config.NumberColumn("SQN", format="%.2f")
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=get_table_height(df_pct, max_rows=50)
                        )
                    else: st.info("No percentile signals found.")
                else: st.error("Failed to load data.")
            except Exception as e: st.error(f"Error: {e}")

def run_seasonality_app(df_global):
    st.title("📅 Seasonality")
    
    # --- 0. SESSION STATE INITIALIZATION (Persistence Layer) ---
    # Single Ticker Memory
    if 'seas_single_df' not in st.session_state: st.session_state.seas_single_df = None
    if 'seas_single_last_ticker' not in st.session_state: st.session_state.seas_single_last_ticker = ""
    
    # Scanner Memory
    if 'seas_scan_results' not in st.session_state: st.session_state.seas_scan_results = None
    if 'seas_scan_csvs' not in st.session_state: st.session_state.seas_scan_csvs = None
    if 'seas_scan_active' not in st.session_state: st.session_state.seas_scan_active = False
    
    # --- Helper: Finance Formatting ---
    def fmt_finance(val):
        if pd.isna(val): return ""
        if isinstance(val, str): return val
        if val < 0: return f"({abs(val):.1f}%)"
        return f"{val:.1f}%"

    # Create Tabs
    tab_single, tab_scan = st.tabs(["🔎 Single Ticker Analysis", "🚀 Opportunity Scanner"])
    
    # ==============================================================================
    # TAB 1: SINGLE TICKER ANALYSIS
    # ==============================================================================
    with tab_single:
        with st.expander("ℹ️ Page Notes: Methodology"):
            st.markdown("""
            **📊 Calendar Month Performance**
            * **Year Total:** The **Compounded Return** for that year (Start Price vs End Price), not the sum of months.
            * **Month Average:** The **AVERAGE** return for that specific month across the selected history.
            """)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            # We use the key to maintain the input state, but we manually check for changes below
            ticker_input = st.text_input("Ticker", value="SPY", key="seas_ticker").strip().upper()
            
        if not ticker_input:
            st.info("Please enter a ticker symbol.")
        else:
            ticker_map = load_ticker_map()
            
            # --- PERSISTENCE LOGIC ---
            # Only fetch if the ticker changed OR if we have no data stored yet
            if (ticker_input != st.session_state.seas_single_last_ticker) or (st.session_state.seas_single_df is None):
                with st.spinner(f"Fetching history for {ticker_input}..."):
                    # USES THE IMPORTED FUNCTION FROM UTILS.PY NOW
                    fetched_df = fetch_history_optimized(ticker_input, ticker_map)
                    # Store in session state
                    st.session_state.seas_single_df = fetched_df
                    st.session_state.seas_single_last_ticker = ticker_input
            
            # Use data from memory
            df = st.session_state.seas_single_df

            if df is None or df.empty:
                st.error(f"Could not load data for {ticker_input}. Check the ticker symbol.")
            else:
                df.columns = [c.strip().upper() for c in df.columns]
                date_col = next((c for c in df.columns if 'DATE' in c), None)
                close_col = next((c for c in df.columns if 'CLOSE' in c), None)
                
                if not date_col or not close_col:
                    st.error("Data source format error: Missing Date or Close columns.")
                else:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col).sort_index()
                    
                    # Resample to Monthly Returns
                    df_monthly = df[close_col].resample('M').last()
                    df_pct = df_monthly.pct_change() * 100
                    
                    season_df = pd.DataFrame({
                        'Pct': df_pct,
                        'Year': df_pct.index.year,
                        'Month': df_pct.index.month
                    }).dropna()

                    today = date.today()
                    current_year = today.year
                    current_month = today.month
                    
                    hist_df = season_df[season_df['Year'] < current_year].copy()
                    curr_df = season_df[season_df['Year'] == current_year].copy()
                    
                    if hist_df.empty:
                        st.warning("Not enough historical full-year data available.")
                    else:
                        min_avail_year = int(hist_df['Year'].min())
                        max_avail_year = int(hist_df['Year'].max())
                        
                        with c2:
                            start_year = st.number_input("Start Year (History)", min_value=min_avail_year, max_value=max_avail_year, value=max_avail_year-10 if max_avail_year-10 >= min_avail_year else min_avail_year, key="seas_start")
                        with c3:
                            end_year = st.number_input("End Year (History)", min_value=start_year, max_value=max_avail_year, value=max_avail_year, key="seas_end")

                        mask = (hist_df['Year'] >= start_year) & (hist_df['Year'] <= end_year)
                        hist_filtered = hist_df[mask].copy()
                        
                        if hist_filtered.empty:
                            st.warning("No data in selected date range.")
                        else:
                            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                            
                            # --- STATS ---
                            avg_stats = hist_filtered.groupby('Month')['Pct'].mean().reindex(range(1, 13), fill_value=0)
                            win_rates = hist_filtered.groupby('Month')['Pct'].apply(lambda x: (x > 0).mean() * 100).reindex(range(1, 13), fill_value=0)

                            # --- OUTLOOK ---
                            cur_val = curr_df.groupby('Month')['Pct'].sum().reindex(range(1, 13)).get(current_month, 0.0)
                            if pd.isna(cur_val): cur_val = 0.0
                            
                            hist_avg = avg_stats.get(current_month, 0.0)
                            diff = cur_val - hist_avg
                            if diff > 0:
                                context_str = f"Outperforming Hist Avg of {fmt_finance(hist_avg)}"
                            else:
                                context_str = f"Underperforming Hist Avg of {fmt_finance(hist_avg)}"
                            
                            cur_color = "#71d28a" if cur_val > 0 else "#f29ca0"

                            idx_next = (current_month % 12) + 1
                            idx_next_2 = ((current_month + 1) % 12) + 1
                            nm_name = month_names[idx_next-1]
                            nnm_name = month_names[idx_next_2-1]
                            nm_avg = avg_stats.get(idx_next, 0.0)
                            nm_wr = win_rates.get(idx_next, 0.0)
                            nnm_avg = avg_stats.get(idx_next_2, 0.0)

                            if nm_avg >= 1.5 and nm_wr >= 65:
                                positioning = "🚀 <b>Strong Bullish.</b> Historically a standout month."
                            elif nm_avg > 0 and nm_wr >= 50:
                                positioning = "↗️ <b>Mildly Bullish.</b> Positive bias, moderate conviction."
                            elif nm_avg < 0 and nm_avg > -1.0:
                                positioning = "⚠️ <b>Choppy/Weak.</b> Historically drags or trends flat."
                            else:
                                positioning = "🐻 <b>Bearish.</b> Historically a weak month."

                            trend_vs = "improves" if nnm_avg > nm_avg else "weakens"
                            
                            st.markdown(f"""
                            <div style="background-color: rgba(128,128,128,0.05); border-left: 5px solid #66b7ff; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
                                <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 8px; color: #444;">🤖 Seasonal Outlook</div>
                                <div style="margin-bottom: 4px;">• <b>Current ({month_names[current_month-1]}):</b> <span style="color:{cur_color}; font-weight:bold;">{fmt_finance(cur_val)}</span>. {context_str}.</div>
                                <div style="margin-bottom: 4px;">• <b>Next Month ({nm_name}):</b> {positioning} (Avg: {fmt_finance(nm_avg)}, Win Rate: {nm_wr:.1f}%)</div>
                                <div>• <b>Following ({nnm_name}):</b> Seasonality {trend_vs} to an average of <b>{fmt_finance(nnm_avg)}</b>.</div>
                            </div>
                            """, unsafe_allow_html=True)

                            col_chart1, col_chart2 = st.columns(2, gap="medium")

                            # --- CHART 1: Performance (Line) ---
                            with col_chart1:
                                st.subheader(f"📈 Performance Tracking")
                                hist_cumsum = avg_stats.cumsum()
                                line_data_hist = pd.DataFrame({
                                    'Month': range(1, 13), 'MonthName': month_names,
                                    'Value': hist_cumsum.values, 'Type': f'Avg ({start_year}-{end_year})'
                                })

                                curr_monthly_stats = curr_df.groupby('Month')['Pct'].sum().reindex(range(1, 13)) 
                                curr_cumsum = curr_monthly_stats.cumsum()
                                valid_curr_indices = curr_monthly_stats.dropna().index
                                
                                line_data_curr = pd.DataFrame({
                                    'Month': valid_curr_indices,
                                    'MonthName': [month_names[i-1] for i in valid_curr_indices],
                                    'Value': curr_cumsum.loc[valid_curr_indices].values,
                                    'Type': f'Current Year ({current_year})'
                                })
                                combined_line_data = pd.concat([line_data_hist, line_data_curr])
                                combined_line_data['Label'] = combined_line_data['Value'].apply(fmt_finance)

                                line_base = alt.Chart(combined_line_data).encode(
                                    x=alt.X('MonthName', sort=month_names, title='Month'),
                                    y=alt.Y('Value', title='Cumulative Return (%)'),
                                    color=alt.Color('Type', legend=alt.Legend(orient='bottom', title=None))
                                )
                                st.altair_chart((line_base.mark_line(point=True) + line_base.mark_text(dy=-10, fontSize=12, fontWeight='bold').encode(text='Label')).properties(height=350).configure_axis(labelFontSize=11, titleFontSize=13), use_container_width=True)

                            # --- CHART 2: Monthly Returns (Bar) ---
                            with col_chart2:
                                st.subheader(f"📊 Monthly Returns")
                                hist_bar_data = pd.DataFrame({'Month': range(1, 13), 'MonthName': month_names, 'Value': avg_stats.values, 'Type': 'Historical Avg'})
                                
                                completed_curr_df = curr_df[curr_df['Month'] < current_month].copy()
                                curr_bar_data = pd.DataFrame()
                                if not completed_curr_df.empty:
                                    curr_vals = completed_curr_df.groupby('Month')['Pct'].mean()
                                    curr_bar_data = pd.DataFrame({'Month': curr_vals.index, 'MonthName': [month_names[i-1] for i in curr_vals.index], 'Value': curr_vals.values, 'Type': f'{current_year} Actual'})
                                
                                combined_bar_data = pd.concat([hist_bar_data, curr_bar_data])
                                combined_bar_data['Label'] = combined_bar_data['Value'].apply(fmt_finance)

                                combined_bar_data['LabelY'] = combined_bar_data['Value'].apply(lambda x: max(0, x))

                                base = alt.Chart(combined_bar_data).encode(x=alt.X('MonthName', sort=month_names, title=None))
                                bars = base.mark_bar().encode(
                                    y=alt.Y('Value', title='Return (%)'), xOffset='Type',
                                    color=alt.condition(alt.datum.Value > 0, alt.value("#71d28a"), alt.value("#f29ca0"))
                                )
                                
                                text = base.mark_text(
                                    dy=-10, fontSize=11, fontWeight='bold', color='black'
                                ).encode(
                                    y=alt.Y('LabelY'),
                                    xOffset='Type', 
                                    text='Label'
                                )

                                st.altair_chart((bars + text).properties(height=350).configure_axis(labelFontSize=11, titleFontSize=13), use_container_width=True)

                            # --- CARDS ---
                            st.markdown("##### 🎯 Historical Win Rate & Expectancy")
                            cols = st.columns(6); cols2 = st.columns(6)
                            for i in range(12):
                                mn = month_names[i]
                                wr = win_rates.loc[i+1]
                                avg = avg_stats.loc[i+1]
                                border_color = "#71d28a" if avg > 0 else "#f29ca0"
                                target_col = cols[i] if i < 6 else cols2[i-6]
                                target_col.markdown(f"""<div style="background-color: rgba(128,128,128,0.05); border-radius: 8px; padding: 8px 5px; text-align: center; margin-bottom: 10px; border-bottom: 3px solid {border_color};"><div style="font-size: 0.85rem; font-weight: bold; color: #555;">{mn}</div><div style="font-size: 0.75rem; color: #888; margin-top:2px;">Win Rate</div><div style="font-size: 1.0rem; font-weight: 700;">{wr:.1f}%</div><div style="font-size: 0.75rem; color: #888; margin-top:2px;">Avg Rtn</div><div style="font-size: 0.9rem; font-weight: 600; color: {'#1f7a1f' if avg > 0 else '#a11f1f'};">{fmt_finance(avg)}</div></div>""", unsafe_allow_html=True)

                            # --- HEATMAP ---
                            st.markdown("---"); st.subheader("🗓️ Monthly Returns Heatmap")
                            pivot_hist = hist_filtered.pivot(index='Year', columns='Month', values='Pct')
                            if not completed_curr_df.empty:
                                pivot_curr = completed_curr_df.pivot(index='Year', columns='Month', values='Pct')
                                full_pivot = pd.concat([pivot_curr, pivot_hist])
                            else: full_pivot = pivot_hist

                            full_pivot.columns = [month_names[c-1] for c in full_pivot.columns]
                            for m in month_names:
                                if m not in full_pivot.columns: full_pivot[m] = np.nan
                            full_pivot = full_pivot[month_names].sort_index(ascending=False)
                            
                            full_pivot["Year Total"] = full_pivot.apply(
                                lambda x: ((1 + x/100).prod(skipna=True) - 1) * 100 if x.notna().any() else np.nan, 
                                axis=1
                            )
                            
                            avg_row = full_pivot[month_names].mean(axis=0)
                            avg_row["Year Total"] = full_pivot["Year Total"].mean()
                            avg_row.name = "Month Average"
                            
                            full_pivot = pd.concat([full_pivot, avg_row.to_frame().T])

                            def color_map(val):
                                if pd.isna(val): return ""
                                if val == 0: return "color: #888;"
                                color = "#1f7a1f" if val > 0 else "#a11f1f"
                                bg_color = "rgba(113, 210, 138, 0.2)" if val > 0 else "rgba(242, 156, 160, 0.2)"
                                return f'background-color: {bg_color}; color: {color}; font-weight: 500;'
                            
                            heatmap_config = {c: st.column_config.Column(width="small") for c in full_pivot.columns}
                            
                            st.dataframe(
                                full_pivot.style.format(fmt_finance).applymap(color_map), 
                                use_container_width=True, 
                                height=(len(full_pivot)+1)*35+3,
                                column_config=heatmap_config
                            )

    # ==============================================================================
    # TAB 2: OPPORTUNITY SCANNER
    # ==============================================================================
    with tab_scan:
        with st.expander("ℹ️ Page Notes: Methodology & Metrics"):
            st.markdown("""
            **🚀 Rolling Forward Returns**
            * **Methodology**: Scans history for dates matching the Start Date (+/- 3 days) and calculates performance for future periods.
            * **Ranking Logic**: Tickers ranked by **EV** (High to Low).
            * **Mean Reversion (Arbitrage)**: Looks for tickers with **Positive Seasonality** (Forward 21d EV > 3%) but **Negative Recent Performance** (Trailing 21d < -3%).
            * **Anomaly Detection**: Includes `Hist. Trailing 21d` to help you see if the recent drop is normal for this time of year or a true anomaly.
            """)

        st.subheader("🚀 High-EV Seasonality Scanner")
        
        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            scan_date = st.date_input("Start Date for Scan", value=date.today(), key="seas_scan_date")
        with sc2:
            min_mc_scan = st.selectbox("Min Market Cap", ["0B", "2B", "10B", "50B", "100B"], index=2, key="seas_scan_mc")
            mc_thresh_val = {"0B":0, "2B":2e9, "10B":1e10, "50B":5e10, "100B":1e11}.get(min_mc_scan, 1e10)
        with sc3:
            scan_lookback = st.number_input("Lookback Years", min_value=5, max_value=20, value=10, key="seas_scan_lb")
            
        start_scan = st.button("Run Scanner")
        
        # --- SCANNER LOGIC ---
        if start_scan:
            ticker_map = load_ticker_map()
            if not ticker_map:
                st.error("No TICKER_MAP found in secrets.")
            else:
                all_tickers = [k for k in ticker_map.keys() if not k.upper().endswith('_PARQUET')]
                results = []
                all_csv_rows = { "21d": [], "42d": [], "63d": [], "126d": [] }
                
                st.write(f"Filtering {len(all_tickers)} tickers by Market Cap...")
                
                valid_tickers = []
                
                def check_filters(t):
                    mc = get_market_cap(t)
                    if mc < mc_thresh_val: return None
                    return t

                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = {executor.submit(check_filters, t): t for t in all_tickers}
                    for future in as_completed(futures):
                        res = future.result()
                        if res: valid_tickers.append(res)
                
                st.write(f"Scanning {len(valid_tickers)} tickers for high EV opportunities...")
                progress_bar = st.progress(0)
                
                def calc_forward_returns(ticker_sym):
                    try:
                        # USES THE IMPORTED FUNCTION FROM UTILS.PY NOW
                        d_df = fetch_history_optimized(ticker_sym, ticker_map)
                        if d_df is None or d_df.empty: return None, None
                        
                        d_df.columns = [c.strip().upper() for c in d_df.columns]
                        date_c = next((c for c in d_df.columns if 'DATE' in c), None)
                        close_c = next((c for c in d_df.columns if 'CLOSE' in c), None)
                        if not date_c or not close_c: return None, None
                        
                        d_df[date_c] = pd.to_datetime(d_df[date_c])
                        d_df = d_df.sort_values(date_c).reset_index(drop=True)
                        
                        cutoff = pd.to_datetime(date.today()) - timedelta(days=scan_lookback*365)
                        d_df_hist = d_df[d_df[date_c] >= cutoff].copy()
                        d_df_hist = d_df_hist.reset_index(drop=True)
                        if len(d_df_hist) < 252: return None, None
                        
                        # --- Calculate Recent Performance (Last 21 days) ---
                        recent_perf = 0.0
                        if len(d_df) > 21:
                            last_p = d_df[close_c].iloc[-1]
                            prev_p = d_df[close_c].iloc[-22] 
                            recent_perf = ((last_p - prev_p) / prev_p) * 100
                        
                        target_doy = scan_date.timetuple().tm_yday
                        d_df_hist['DOY'] = d_df_hist[date_c].dt.dayofyear
                        
                        # +/- 3 Day Window
                        matches = d_df_hist[(d_df_hist['DOY'] >= target_doy - 3) & (d_df_hist['DOY'] <= target_doy + 3)].copy()
                        matches['Year'] = matches[date_c].dt.year
                        matches = matches.drop_duplicates(subset=['Year'])
                        curr_y = date.today().year
                        matches = matches[matches['Year'] < curr_y]
                        
                        if len(matches) < 3: return None, None
                        
                        stats_row = {'Ticker': ticker_sym, 'N': len(matches), 'Recent_21d': recent_perf}
                        
                        # --- CALC LAG RETURNS (HISTORICAL TRAILING 21D) ---
                        hist_lag_returns = []
                        for idx in matches.index:
                            if idx >= 21:
                                p_now = d_df_hist.loc[idx, close_c]
                                p_prev = d_df_hist.loc[idx - 21, close_c]
                                hist_lag_returns.append((p_now - p_prev) / p_prev)
                        
                        stats_row['Hist_Lag_21d'] = (np.mean(hist_lag_returns) * 100) if hist_lag_returns else 0.0
                        
                        periods = {"21d": 21, "42d": 42, "63d": 63, "126d": 126}
                        
                        ticker_csv_rows = {k: [] for k in periods.keys()}
                        
                        for p_name, trading_days in periods.items():
                            returns = []
                            for idx in matches.index:
                                entry_p = d_df_hist.loc[idx, close_c]
                                exit_idx = idx + trading_days
                                if exit_idx < len(d_df_hist):
                                    exit_p = d_df_hist.loc[exit_idx, close_c]
                                    ret = (exit_p - entry_p) / entry_p
                                    returns.append(ret)
                                    
                                    ticker_csv_rows[p_name].append({
                                        "Ticker": ticker_sym,
                                        "Start Date": d_df_hist.loc[idx, date_c].date(),
                                        "Entry Price": entry_p,
                                        "Exit Date": d_df_hist.loc[exit_idx, date_c].date(),
                                        "Exit Price": exit_p,
                                        "Return (%)": ret * 100
                                    })
                                        
                            if returns:
                                returns_arr = np.array(returns)
                                avg_ret = np.mean(returns_arr) * 100
                                win_r = np.mean(returns_arr > 0) * 100
                                std_dev = np.std(returns_arr) * 100
                                sharpe = avg_ret / std_dev if std_dev > 0.1 else 0.0
                            else:
                                avg_ret = 0.0; win_r = 0.0; sharpe = 0.0
                                
                            stats_row[f"{p_name}_EV"] = avg_ret
                            stats_row[f"{p_name}_WR"] = win_r
                            stats_row[f"{p_name}_Sharpe"] = sharpe
                            
                        return stats_row, ticker_csv_rows
                    except Exception:
                        return None, None

                with ThreadPoolExecutor(max_workers=20) as executor: 
                    futures = {executor.submit(calc_forward_returns, t): t for t in valid_tickers}
                    completed = 0
                    for future in as_completed(futures):
                        res_stats, res_details = future.result()
                        if res_stats:
                            results.append(res_stats)
                        if res_details:
                            for k in all_csv_rows.keys():
                                if res_details[k]:
                                    all_csv_rows[k].extend(res_details[k])
                        completed += 1
                        if completed % 5 == 0: progress_bar.progress(completed / len(valid_tickers))
                
                progress_bar.empty()
                
                if not results:
                    st.warning("No opportunities found.")
                    st.session_state.seas_scan_results = None
                else:
                    # --- SAVE RESULTS TO SESSION STATE ---
                    st.session_state.seas_scan_results = pd.DataFrame(results)
                    st.session_state.seas_scan_csvs = all_csv_rows
                    st.session_state.seas_scan_active = True

        # --- DISPLAY RESULTS (From Session State) ---
        if st.session_state.seas_scan_active and st.session_state.seas_scan_results is not None:
            res_df = st.session_state.seas_scan_results
            all_csv_rows = st.session_state.seas_scan_csvs
            
            st.write("---")
            
            def highlight_ev(val):
                if pd.isna(val): return ""
                color = "#1f7a1f" if val > 0 else "#a11f1f"
                bg = "rgba(113, 210, 138, 0.25)" if val > 0 else "rgba(242, 156, 160, 0.25)"
                return f'background-color: {bg}; color: {color}; font-weight: bold;'
                
            def color_sharpe(val):
                if pd.isna(val): return ""
                if val < 1.0: return "background-color: #ffccbc; color: black"
                if val < 2.0: return "background-color: #fff9c4; color: black"
                if val < 3.0: return "background-color: #c8e6c9; color: black"
                return "background-color: #81c784; color: black"

            # --- 1. STANDARD TABLES ---
            st.subheader(f"🗓️ Forward Returns (from {scan_date.strftime('%d %b')})")
            
            c_scan1, c_scan2 = st.columns(2)
            c_scan3, c_scan4 = st.columns(2)
            fixed_height = 738

            for col_obj, p_label, sort_col, sharpe_col, p_key in [
                (c_scan1, "**+21 Trading Days**", "21d_EV", "21d_Sharpe", "21d"),
                (c_scan2, "**+42 Trading Days**", "42d_EV", "42d_Sharpe", "42d"),
                (c_scan3, "**+63 Trading Days**", "63d_EV", "63d_Sharpe", "63d"),
                (c_scan4, "**+126 Trading Days**", "126d_EV", "126d_Sharpe", "126d")
            ]:
                with col_obj:
                    st.markdown(p_label)
                    
                    if all_csv_rows[p_key]:
                        df_details = pd.DataFrame(all_csv_rows[p_key])
                        df_details = df_details.sort_values(by=["Ticker", "Start Date"])
                        csv_data = df_details.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"💾 Download CSV",
                            data=csv_data,
                            file_name=f"seasonality_{p_key}_inputs_{scan_date.strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key=f"dl_btn_{p_key}"
                        )

                    top_df = res_df.sort_values(by=sort_col, ascending=False).head(20)
                    
                    st.dataframe(
                        top_df[['Ticker', sort_col, sort_col.replace('EV','WR'), sharpe_col]].style
                        .format({
                            sort_col: fmt_finance, 
                            sort_col.replace('EV','WR'): "{:.1f}%",
                            sharpe_col: "{:.2f}"
                        })
                        .applymap(highlight_ev, subset=[sort_col])
                        .applymap(color_sharpe, subset=[sharpe_col]),
                        use_container_width=True, hide_index=True, height=fixed_height,
                        column_config={
                            sharpe_col: st.column_config.NumberColumn("Sharpe", help="Consistency Score (EV / StdDev). >2 is very consistent.")
                        }
                    )

            # --- 2. UPDATED ARBITRAGE TABLE ---
            st.write("---")
            arb_df = res_df[
                (res_df['21d_EV'] > 3.0) & 
                (res_df['Recent_21d'] < -3.0)
            ].copy()
            
            if not arb_df.empty:
                st.subheader("💎 Arbitrage / Catch-Up Candidates")
                st.caption("Stocks with strong historical seasonality (EV > 3%) that are currently beaten down (Recent < -3%).")
                st.caption("Use 'Hist. Trailing 21d' to determine if the recent drop is normal seasonality (e.g. usually drops) or an anomaly (usually rises).")
                
                arb_df['Anomaly_Score'] = arb_df['Hist_Lag_21d'] - arb_df['Recent_21d']
                arb_display = arb_df.sort_values(by='Anomaly_Score', ascending=False).head(15)
                
                st.dataframe(
                    arb_display[['Ticker', 'Recent_21d', 'Hist_Lag_21d', '21d_EV', '21d_WR']].style
                    .format({
                        'Recent_21d': fmt_finance, 
                        'Hist_Lag_21d': fmt_finance,
                        '21d_EV': fmt_finance, 
                        '21d_WR': "{:.1f}%"
                    })
                    .applymap(lambda x: 'color: #d32f2f; font-weight:bold;', subset=['Recent_21d'])
                    .applymap(lambda x: 'color: #2e7d32; font-weight:bold;', subset=['21d_EV']),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width=None),
                        "Recent_21d": st.column_config.TextColumn("Recent 21d (Actual)", help="How the stock performed in the last 21 days."),
                        "Hist_Lag_21d": st.column_config.TextColumn("Hist. Trailing 21d (Avg)", help="How the stock USUALLY performs during this trailing 21 day period."),
                        "21d_EV": st.column_config.TextColumn("Hist. Forward 21d (EV)", help="How the stock usually performs in the NEXT 21 days."),
                        "21d_WR": st.column_config.TextColumn("Win Rate", help="Frequency of positive returns in the forward period.")
                    }
                )

def run_ema_distance_app(df_global):
    # Helper function defined inside scope to prevent "not defined" errors
    def run_backtest(signal_series, price_data, low_data, lookforward=30, drawdown_thresh=-0.08):
        idxs = signal_series[signal_series].index
        if len(idxs) == 0: return 0, 0, 0
        hits = 0
        days_to_dd = []
        closes = price_data.values
        lows = low_data.values
        is_signal = signal_series.values
        n = len(closes)
        for i in range(n):
            if not is_signal[i]: continue
            if i + lookforward >= n: continue 
            entry_price = closes[i]
            future_window = lows[i+1 : i+1+lookforward]
            min_future = np.min(future_window)
            dd = (min_future - entry_price) / entry_price
            if dd <= drawdown_thresh:
                hits += 1
                target_price = entry_price * (1 + drawdown_thresh)
                hit_indices = np.where(future_window <= target_price)[0]
                if len(hit_indices) > 0:
                    days_to_dd.append(hit_indices[0] + 1)
        hit_rate = (hits / len(idxs)) * 100 if len(idxs) > 0 else 0
        median_days = np.median(days_to_dd) if days_to_dd else 0
        return len(idxs), hit_rate, median_days

    st.title("📏 EMA Distance Analysis")

    # 1. Input Section
    col_in1, col_in2, _ = st.columns([1, 1, 2])
    with col_in1:
        ticker = st.text_input("Ticker", value="QQQ").upper().strip()
    with col_in2:
        years_back = st.number_input("Years to Analyze", min_value=1, max_value=20, value=10, step=1)
    
    if not ticker:
        st.warning("Please enter a ticker.")
        return

    # Percentage formatting to 1 decimal place
    def fmt_pct(val):
        if pd.isna(val): return ""
        if val < 0: return f"({abs(val):.1f}%)"
        return f"{val:.1f}%"

    # 2. Data Fetching
    with st.spinner(f"Crunching data for {ticker}..."):
        try:
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(period=f"{years_back}y")
            if df is None or df.empty:
                st.error(f"Could not fetch data for {ticker}.")
                return
            df = df.reset_index()
            df.columns = [c.upper() for c in df.columns]
            
            # Defining date_col to prevent "not defined" error
            date_col = next((c for c in df.columns if 'DATE' in c), "DATE")
            close_col = 'CLOSE' if 'CLOSE' in df.columns else 'Close'
            low_col = 'LOW' if 'LOW' in df.columns else 'Low'
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

    # Calculations strictly using Close prices
    df['EMA_8'] = df[close_col].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df[close_col].ewm(span=21, adjust=False).mean()
    df['SMA_50'] = df[close_col].rolling(window=50).mean()
    df['SMA_100'] = df[close_col].rolling(window=100).mean()
    df['SMA_200'] = df[close_col].rolling(window=200).mean()
    
    # Distance Gaps
    df['Dist_8'] = ((df[close_col] - df['EMA_8']) / df['EMA_8']) * 100
    df['Dist_21'] = ((df[close_col] - df['EMA_21']) / df['EMA_21']) * 100
    df['Dist_50'] = ((df[close_col] - df['SMA_50']) / df['SMA_50']) * 100
    df['Dist_100'] = ((df[close_col] - df['SMA_100']) / df['SMA_100']) * 100
    df['Dist_200'] = ((df[close_col] - df['SMA_200']) / df['SMA_200']) * 100
    df_clean = df.dropna(subset=['EMA_8', 'EMA_21', 'SMA_50', 'SMA_100', 'SMA_200']).copy()
    
    # Current Distance for Reference Line
    current_dist_50 = df_clean['Dist_50'].iloc[-1]

    # --- TABLE 1: Main Stats ---
    st.subheader(f"{ticker} vs Moving Avgs & Percentiles")

    with st.expander("ℹ️ Table User Guide"):
        st.markdown(f"""
            **1. Key Metrics Tracked.**
            The app calculates the percentage distance (the "Gap") between the current price and five different moving averages:
            * 8-day and 21-day EMA: Short-term momentum and "swing" levels.
            * 50-day, 100-day, and 200-day SMA: Medium to long-term trend baselines.

            **2. The "Rubber Band" Logic (Percentiles).**
            Rather than just showing the current gap, the app looks at 10 years of history for that specific ticker to see how rare the current gap is. It calculates:
            * **p50 (Median):** The typical distance from the average.
            * **p70/p80 (Uptrend):** These levels generally occur in strong uptrends.
            * **p90/p95 (Extremes):** The levels reached only 10% or 5% of the time historically.

            **3. Visual Highlighting System.**
            The table uses a traffic-light system to categorize the current price action:
            * 🟢 **Buy Zone (Green):** Triggered if the Gap is ≤ p50 (Median) AND price is > 8-EMA. Suggests a "pullback to the mean" in an uptrend.
            * 🟡 **Warning Zone (Yellow):** Triggered if the gap is between p50 and p90. Price is extending but not yet extreme.
            * 🔴 **Sell/Trim Zone (Red):** Triggered if the gap is $\ge$ p90. Price is statistically over-extended.

            **4. Data Sources.**
            All Close Prices are sourced directly from Yahoo Finance.
                    """)

    stats_data = []
    thresholds = {} 
    current_price = df_clean[close_col].iloc[-1]
    current_ema8 = df_clean['EMA_8'].iloc[-1]
    
    metrics = [
        ("Close vs 8-EMA", df_clean['EMA_8'], df_clean['Dist_8']),
        ("Close vs 21-EMA", df_clean['EMA_21'], df_clean['Dist_21']),
        ("Close vs 50-SMA", df_clean['SMA_50'], df_clean['Dist_50']),
        ("Close vs 100-SMA", df_clean['SMA_100'], df_clean['Dist_100']),
        ("Close vs 200-SMA", df_clean['SMA_200'], df_clean['Dist_200']),
    ]
    
    for label, ma_series, dist_series in metrics:
        p_vals = np.percentile(dist_series, [50, 70, 80, 90, 95])
        thresholds[dist_series.name] = { 'p80': p_vals[2], 'p90': p_vals[3] }
        stats_data.append({
            "Metric": label, "Price": current_price, "MA Level": ma_series.iloc[-1],
            "Gap": dist_series.iloc[-1], "Avg": dist_series.mean(),
            "p50": p_vals[0], "p70": p_vals[1], "p80": p_vals[2], "p90": p_vals[3], "p95": p_vals[4]
        })

    df_stats = pd.DataFrame(stats_data)

    def color_combined(row):
        styles = [''] * len(row)
        gap, p50, p90 = row['Gap'], row['p50'], row['p90']
        idx_gap = df_stats.columns.get_loc("Gap")
        if gap >= p90: styles[idx_gap] = 'background-color: #fce8e6; color: #c5221f; font-weight: bold;'
        elif gap <= p50 and (current_price > current_ema8): styles[idx_gap] = 'background-color: #e6f4ea; color: #1e7e34; font-weight: bold;'
        elif gap > p50 and gap < p90: styles[idx_gap] = 'background-color: #fff8e1; color: #d68f00;'
        return styles

    st.dataframe(
        df_stats.style.apply(color_combined, axis=1).format(fmt_pct, subset=["Gap", "Avg", "p50", "p70", "p80", "p90", "p95"]),
        use_container_width=True, hide_index=True,
        column_config={"Price": st.column_config.NumberColumn("Price", format="$%.2f"), "MA Level": st.column_config.NumberColumn("MA Level", format="$%.2f")}
    )
    
    # --- TABLE 2: Combo Over-Extension Signals ---
    st.subheader("Combo Over-Extension Signals")

    t8_90 = thresholds['Dist_8']['p90']
    t21_80 = thresholds['Dist_21']['p80']
    t50_80 = thresholds['Dist_50']['p80']
    
    m_d = (df_clean['Dist_8'] >= t8_90) & (df_clean['Dist_21'] >= t21_80)
    m_fs = (df_clean['Dist_8'] >= t8_90) & (df_clean['Dist_50'] >= t50_80)
    m_t = (df_clean['Dist_8'] >= t8_90) & (df_clean['Dist_21'] >= t21_80) & (df_clean['Dist_50'] >= t50_80)
    
    res_d = run_backtest(m_d, df_clean[close_col], df_clean[low_col])
    res_fs = run_backtest(m_fs, df_clean[close_col], df_clean[low_col])
    res_t = run_backtest(m_t, df_clean[close_col], df_clean[low_col])

    # Status Emoji Logic
    d_active = "✅" if bool(m_d.iloc[-1]) else "❌"
    fs_active = "✅" if bool(m_fs.iloc[-1]) else "❌"
    t_active = "✅" if bool(m_t.iloc[-1]) else "❌"

    # Updated column headers for Draw Down
    combo_rows = [
        {
            "Combo Rule": "Double EMA", 
            "Triggers": "(8-EMA Gap ≥ p90), (21-EMA Gap ≥ p80)",
            "Occurrences": res_d[0], "Hit Rate (>=8% Draw Down)": res_d[1], "Median Days to Draw Down": f"{int(res_d[2])} days", 
            "Active Today?": d_active, "raw_status": bool(m_d.iloc[-1])
        },
        {
            "Combo Rule": "Fast vs Swing", 
            "Triggers": "(8-EMA gap ≥ p90), (50-SMA gap ≥ p80)",
            "Occurrences": res_fs[0], "Hit Rate (>=8% Draw Down)": res_fs[1], "Median Days to Draw Down": f"{int(res_fs[2])} days", 
            "Active Today?": fs_active, "raw_status": bool(m_fs.iloc[-1])
        },
        {
            "Combo Rule": "Triple Stack", 
            "Triggers": "(8-EMA gap ≥ p90), (50-SMA gap ≥ p80), (21-EMA gap ≥ p80)",
            "Occurrences": res_t[0], "Hit Rate (>=8% Draw Down)": res_t[1], "Median Days to Draw Down": f"{int(res_t[2])} days", 
            "Active Today?": t_active, "raw_status": bool(m_t.iloc[-1])
        }
    ]
    
    df_combo = pd.DataFrame(combo_rows)

    # Style: Bold row only if raw_status is True
    def style_combo(row):
        return ['font-weight: bold; color: #c5221f;' if row['raw_status'] else ''] * len(row)

    st.dataframe(
        df_combo.style.apply(style_combo, axis=1).format({"Hit Rate (>=8% Draw Down)": "{:.1f}%"}),
        use_container_width=True, hide_index=True,
        column_config={
            "Triggers": st.column_config.TextColumn("Trigger Conditions", width="large"),
            "Active Today?": st.column_config.TextColumn("Active Today?", width="small")
        },
        column_order=["Combo Rule", "Triggers", "Occurrences", "Hit Rate (>=8% Draw Down)", "Median Days to Draw Down", "Active Today?"]
    )

    # --- CHART: Visualization ---
    st.subheader("Visualizing the % Distance from 50 SMA")
    
    chart_data = pd.DataFrame({
        'Date': pd.to_datetime(df_clean[date_col]), 
        'Distance (%)': df_clean['Dist_50']
    })
    
    # --- MODIFIED: Show last 10 years (3650 days) instead of 2 years (730 days) ---
    chart_data = chart_data[chart_data['Date'] >= (chart_data['Date'].max() - timedelta(days=3650))]

    # Base bar chart
    bars = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Date:T', title=None), 
        y=alt.Y('Distance (%)', title='% Dist from 50 SMA'),
        color=alt.condition(alt.datum['Distance (%)'] > 0, alt.value("#71d28a"), alt.value("#f29ca0")),
        tooltip=['Date', 'Distance (%)']
    )

    # Horizontal Rule representing Current % Distance
    rule = alt.Chart(pd.DataFrame({'y': [current_dist_50]})).mark_rule(
        color='#333', 
        strokeDash=[5, 5], 
        strokeWidth=2
    ).encode(y='y:Q')

    # Combined Chart
    final_chart = (bars + rule).properties(height=300).interactive()
    st.altair_chart(final_chart, use_container_width=True)
