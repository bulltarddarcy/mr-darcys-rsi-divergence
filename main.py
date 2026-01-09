# --- IMPORTS ---
import streamlit as st
import pandas as pd
from datetime import date

# --- MODULE IMPORTS ---
import main_darcy
import main_sector
import utils_darcy as ud  # For global data loading & health checks

# --- 0. PAGE CONFIGURATION ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# --- 1. CSS STYLING ---
st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{
    flex-grow: 1; 
    position: relative; 
    height: 24px; 
    background-color: rgba(0,0,0,0.03);
    border-radius: 4px;
    overflow: hidden;
}
.zone-bar{
    position: absolute;
    left: 0; 
    top: 0; 
    bottom: 0; 
    z-index: 1;
    border-radius: 3px;
    opacity: 0.65;
}
.zone-bull{background-color: #71d28a;}
.zone-bear{background-color: #f29ca0;}
.zone-value{
    position: absolute;
    right: 8px;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    z-index: 2;
    font-size: 12px; 
    font-weight: 700;
    color: #1f1f1f;
    white-space: nowrap;
    text-shadow: 0 0 4px rgba(255,255,255,0.8);
}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }
.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; white-space: nowrap; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}
.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}
.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }
</style>""", unsafe_allow_html=True)

# --- 2. GLOBAL DATA LOADING ---
try:
    sheet_url = st.secrets["GSHEET_URL"]
    # Load Main DB using the Darcy Utils loader
    df_global = ud.load_and_clean_data(sheet_url)
    
    # 2a. Database Date (from Google Sheet)
    if not df_global.empty and "Trade Date" in df_global.columns:
        db_date = df_global["Trade Date"].max().strftime("%d %b %y")
    else:
        db_date = "No Data"
    
    # 2b. Price History Date (Check max date in PARQUET_SP100)
    price_date = "Syncing..."
    try:
        # Check combined SP100 parquet file using Darcy Utils
        df_sp100_check = ud.load_parquet_and_clean("PARQUET_SP100")
        
        if df_sp100_check is not None and not df_sp100_check.empty:
            date_col_check = next((c for c in df_sp100_check.columns if 'DATE' in c.upper()), None)
            if date_col_check:
                price_date = pd.to_datetime(df_sp100_check[date_col_check]).max().strftime("%d %b %y")
            else:
                price_date = "Date Error"
        else:
            price_date = "Read Error"
    except Exception:
        price_date = "Offline"

    # --- 3. NAVIGATION SETUP ---
    pg = st.navigation([
        # DARCY APPS
        st.Page(lambda: main_darcy.run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: main_darcy.run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: main_darcy.run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: main_darcy.run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: main_darcy.run_price_divergences_app(df_global), title="Price Divergences", icon="📉", url_path="price_divergences"),
        st.Page(lambda: main_darcy.run_rsi_scanner_app(df_global), title="RSI Scanner", icon="🤖", url_path="rsi_scanner"),
        st.Page(lambda: main_darcy.run_seasonality_app(df_global), title="Seasonality", icon="📅", url_path="seasonality"),
        st.Page(lambda: main_darcy.run_ema_distance_app(df_global), title="EMA Distance", icon="📏", url_path="ema_distance"),
        
        # SECTOR APP
        st.Page(lambda: main_sector.run_sector_rotation_app(df_global), title="Sector Rotation", icon="🔄", url_path="sector_rotation"),
    ])

    # --- 4. SIDEBAR INFO ---
    st.sidebar.caption("🖥️ Wide monitor & light mode.")
    st.sidebar.caption(f"💾 **JB Database:** {db_date}")
    st.sidebar.caption(f"📈 **Price/RSIs:** {price_date}")
    
    # --- 5. DATA HEALTH CHECK ---
    with st.sidebar.expander("🏥 Data Health Check", expanded=False):
        # A. Check Ticker Map
        tm_key = "URL_TICKER_MAP"
        tm_url = st.secrets.get(tm_key, "")
        if not tm_url:
            st.markdown(f"❌ **Ticker Map**: Secret Missing")
        elif "drive.google.com" not in tm_url:
            st.markdown(f"⚠️ **Ticker Map**: Invalid URL")
        else:
             st.markdown(f"✅ **Ticker Map**: Connected")

        # B. Check Parquet Files (using Darcy Utils config)
        health_config = ud.get_parquet_config()
        all_good = True
        
        for name, key in health_config.items():
            url = st.secrets.get(key, "")
            
            if not url:
                st.markdown(f"❌ **{name}**: Secret Missing")
                all_good = False
            elif "drive.google.com" not in url:
                 st.markdown(f"⚠️ **{name}**: Invalid URL Format")
                 all_good = False
            else:
                status_icon = "✅"
                note = ""
                if "usp=drive_link" in url:
                    status_icon = "⚠️" 
                    note = "(drive_link)"
                st.markdown(f"{status_icon} **{name}**: Linked {note}")
        
        if all_good and tm_url:
            st.caption("All configurations look valid.")
        else:
            st.error("Configuration errors detected.")
    
    # --- 6. RUN ---
    pg.run()
    
    # Global padding
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")