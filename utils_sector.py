import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from utils_shared import get_gdrive_binary_data

# ==========================================
# 1. CONFIGURATION
# ==========================================
TIMEFRAMES = {
    'Short': 5,    # 5 Trading Days
    'Med':   10,   # 10 Trading Days
    'Long':  20    # 20 Trading Days
}
MIN_DOLLAR_VOLUME = 2_000_000
BETA_WINDOW = 60
# Optimization: Only need enough history for Beta (60d) + RRG Smoothing (20d)
MAX_LOOKBACK_DAYS = 150 

# ==========================================
# 2. DATA LOADING (LAYER 1 - RAW IO)
# ==========================================

@st.cache_data(ttl=3600, show_spinner="Downloading Sector Universe...")
def load_universe_config():
    """Loads the universe config from Secrets."""
    secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
    if not secret_val: return pd.DataFrame(), {}

    try:
        if secret_val.strip().startswith("http"):
            if "docs.google.com/spreadsheets" in secret_val:
                file_id = secret_val.split("/d/")[1].split("/")[0]
                url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
            elif "drive.google.com" in secret_val:
                file_id = secret_val.split("/d/")[1].split("/")[0]
                url = f"https://drive.google.com/uc?id={file_id}&export=download"
            else: url = secret_val
            df = pd.read_csv(url)
        else:
            df = pd.read_csv(StringIO(secret_val))
        
        df.columns = [c.strip() for c in df.columns]
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        df['Theme'] = df['Theme'].astype(str).str.strip()
        df['Role'] = df['Role'].astype(str).str.strip().str.title() if 'Role' in df.columns else 'Stock'
        
        etf_rows = df[df['Role'] == 'Etf']
        theme_map = dict(zip(etf_rows['Theme'], etf_rows['Ticker'])) if not etf_rows.empty else {}
        
        return df, theme_map
    except Exception as e:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600, show_spinner="Downloading Master Database...")
def load_raw_sector_data():
    """
    Downloads Parquet and renames pre-calculated columns.
    """
    db_url = st.secrets.get("PARQUET_SECTOR_ROTATION")
    if not db_url: return None

    buffer = get_gdrive_binary_data(db_url)
    if not buffer: return None
    
    try:
        df = pd.read_parquet(buffer)
        
        # 1. Standardize Header
        df.columns = [c.strip().upper() for c in df.columns]
        
        # 2. Map Source Columns -> App Standard
        rename_map = {
            'SYMBOL': 'Ticker', 'TICKER': 'Ticker',
            'DATE': 'Date', 'CLOSE': 'Close', 'ADJ CLOSE': 'Close',
            'HIGH': 'High', 'LOW': 'Low', 'VOLUME': 'Volume',
            # Pre-Calculated Technicals
            'EMA8': 'EMA_8', 
            'EMA21': 'EMA_21',
            'SMA50': 'SMA_50', 
            'SMA200': 'SMA_200',
            'RSI14': 'RSI_14',
            'RSI': 'RSI_14'
        }
        df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        
        return df
    except Exception:
        return None

# ==========================================
# 3. PROCESSING (LAYER 2 - MATH)
# ==========================================

def _calc_rrg_metrics_vectorized(close_df, benchmark_series):
    """
    Vectorized RRG calculation for multiple tickers at once.
    """
    # Align
    ratio_df = close_df.div(benchmark_series, axis=0)
    
    metrics = {}
    
    for label, w in TIMEFRAMES.items():
        # Trend: Ratio vs Moving Average of Ratio
        ma_ratio = ratio_df.rolling(window=w).mean()
        # Normalized Deviation
        metrics[f"RRG_Ratio_{label}"] = ((ratio_df - ma_ratio) / ma_ratio) * 100 + 100
        
        # Momentum: ROC of the Ratio
        metrics[f"RRG_Mom_{label}"] = (ratio_df.pct_change(periods=w) * 100) + 100
        
    return metrics, ratio_df

@st.cache_data(ttl=1200, show_spinner="Computing Sector Metrics...")
def get_computed_sector_data(benchmark_ticker):
    """
    Optimized: Filters history, uses matrix math, and ensures Benchmark is returned.
    """
    master_df = load_raw_sector_data()
    uni_df, theme_map = load_universe_config()
    
    if master_df is None or uni_df.empty: return {}, [], theme_map, uni_df

    # --- OPTIMIZATION 1: DATE SLICING ---
    if 'Date' in master_df.columns:
        max_date = master_df['Date'].max()
        cutoff_date = max_date - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
        master_df = master_df[master_df['Date'] >= cutoff_date].copy()
        master_df.sort_values(['Ticker', 'Date'], inplace=True)

    # --- OPTIMIZATION 2: WIDE FORMAT CONVERSION ---
    df_close = master_df.pivot(index='Date', columns='Ticker', values='Close').ffill()
    df_vol = master_df.pivot(index='Date', columns='Ticker', values='Volume').fillna(0)
    
    if benchmark_ticker not in df_close.columns:
        return {}, [f"{benchmark_ticker} missing"], theme_map, uni_df

    bench_series = df_close[benchmark_ticker]
    df_rets = df_close.pct_change()
    bench_rets = df_rets[benchmark_ticker]
    
    # --- OPTIMIZATION 3: VECTORIZED RVOL ---
    df_vol_avg = df_vol.rolling(20).mean()
    df_rvol = df_vol / df_vol_avg
    
    rvol_metrics = {}
    for k, w in TIMEFRAMES.items():
        rvol_metrics[f"RVOL_{k}"] = df_rvol.rolling(w).mean()

    # --- PROCESS ETFs (RRG) ---
    etf_tickers = list(theme_map.values())
    valid_etfs = [t for t in etf_tickers if t in df_close.columns]
    
    if valid_etfs:
        etf_closes = df_close[valid_etfs]
        rrg_results, _ = _calc_rrg_metrics_vectorized(etf_closes, bench_series)
    else:
        rrg_results = {}

    # --- PROCESS STOCKS (Alpha/Beta) ---
    stocks = uni_df[uni_df['Role'] == 'Stock']
    
    theme_groups = {}
    for _, row in stocks.iterrows():
        t = row['Ticker']
        if t in df_close.columns:
            theme = row['Theme']
            parent = theme_map.get(theme, benchmark_ticker)
            if parent not in df_close.columns: parent = benchmark_ticker
            if parent not in theme_groups: theme_groups[parent] = []
            theme_groups[parent].append(t)

    beta_map = {}
    alpha_map = {} 
    rolling_alpha_maps = {k: {} for k in TIMEFRAMES}

    for parent_etf, stock_list in theme_groups.items():
        if not stock_list: continue
        sector_stock_rets = df_rets[stock_list]
        parent_ret_series = df_rets[parent_etf]
        
        rolling_cov = sector_stock_rets.rolling(BETA_WINDOW).cov(parent_ret_series)
        rolling_var = parent_ret_series.rolling(BETA_WINDOW).var()
        
        betas = rolling_cov.div(rolling_var, axis=0).fillna(1.0)
        expected_ret = betas.multiply(parent_ret_series, axis=0)
        true_alphas = sector_stock_rets - expected_ret
        
        for t in stock_list:
            beta_map[t] = betas[t]
            alpha_map[t] = true_alphas[t]
        
        for k, w in TIMEFRAMES.items():
            rolled = true_alphas.rolling(w).sum() * 100
            for t in stock_list:
                rolling_alpha_maps[k][t] = rolled[t]

    # --- REASSEMBLE DATA CACHE ---
    data_cache = {}
    grouped_master = master_df.groupby('Ticker')
    
    # 1. ETFs
    for etf in valid_etfs:
        if etf not in grouped_master.groups: continue
        d = grouped_master.get_group(etf).set_index('Date').sort_index()
        for k, val_df in rrg_results.items():
            if etf in val_df.columns:
                d[k] = val_df[etf]
        data_cache[etf] = d
        
    # 2. Stocks
    all_processed_stocks = []
    for s_list in theme_groups.values(): all_processed_stocks.extend(s_list)
    
    for stock in all_processed_stocks:
        if stock not in grouped_master.groups: continue
        d = grouped_master.get_group(stock).set_index('Date').sort_index()
        
        if stock in beta_map: d['Beta'] = beta_map[stock]
        if stock in alpha_map: d['True_Alpha_1D'] = alpha_map[stock]
        for k in TIMEFRAMES:
            if stock in rolling_alpha_maps[k]:
                d[f"True_Alpha_{k}"] = rolling_alpha_maps[k][stock]
        if stock in df_rvol.columns:
            d['RVOL'] = df_rvol[stock]
            for k in TIMEFRAMES:
                 d[f"RVOL_{k}"] = rvol_metrics[f"RVOL_{k}"][stock]
        data_cache[stock] = d
        
    # --- CRITICAL FIX: ENSURE BENCHMARK IS IN CACHE ---
    # We explicitly add the Benchmark (SPY/QQQ) to the cache so the UI can download its price data.
    if benchmark_ticker in grouped_master.groups and benchmark_ticker not in data_cache:
        d = grouped_master.get_group(benchmark_ticker).set_index('Date').sort_index()
        data_cache[benchmark_ticker] = d

    return data_cache, [], theme_map, uni_df

# ==========================================
# 4. VISUALIZATION HELPERS
# ==========================================
def classify_setup(df):
    if df is None or df.empty: return None
    last = df.iloc[-1]
    if "RRG_Mom_Short" not in last: return None
    m5 = last["RRG_Mom_Short"]
    m10 = last.get("RRG_Mom_Med", 0)
    m20 = last.get("RRG_Mom_Long", 0)
    r20 = last.get("RRG_Ratio_Long", 100)
    
    if m20 < 100 and m5 > 100 and m5 > (m20 + 2): return "ｪJ-Hook"
    if r20 > 100 and m5 > 100 and m5 > m10: return "圸 Bull Flag"
    if m5 > m10 and m10 > m20 and m20 > 100: return "噫 Rocket"
    return None 

def get_quadrant_status(df, key):
    if df is None or df.empty: return "N/A"
    r = df[f"RRG_Ratio_{key}"].iloc[-1]
    m = df[f"RRG_Mom_{key}"].iloc[-1]
    if r >= 100 and m >= 100: return "泙 Leading"
    elif r < 100 and m >= 100: return "鳩 Improving"
    elif r < 100 and m < 100: return "閥 Lagging"
    return "泯 Weakening"

def plot_simple_rrg(data_cache, target_map, view_key, show_trails):
    fig = go.Figure()
    all_x, all_y = [], []
    col_x, col_y = f"RRG_Ratio_{view_key}", f"RRG_Mom_{view_key}"
    
    for theme, ticker in target_map.items():
        df = data_cache.get(ticker)
        if df is None or df.empty or col_x not in df.columns: continue
        
        sl = df.tail(3) if show_trails else df.tail(1)
        xs, ys = sl[col_x].tolist(), sl[col_y].tolist()
        all_x.extend(xs); all_y.extend(ys)
        
        lx, ly = xs[-1], ys[-1]
        color = '#00CC96' if lx>100 and ly>100 else '#636EFA' if lx<100 and ly>100 else '#FFA15A' if lx>100 and ly<100 else '#EF553B'
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines+markers+text', name=theme, text=[""]*(len(xs)-1)+[theme],
            customdata=[theme]*len(xs), textposition="top center",
            marker=dict(size=[8]*(len(xs)-1)+[15], color=color, line=dict(width=1, color='white')),
            line=dict(color=color, width=1 if show_trails else 0),
            hoverinfo='text+name', hovertext=[f"{theme}<br>T:{x:.1f} M:{y:.1f}" for x,y in zip(xs,ys)]
        ))

    lim = max(max([abs(x-100) for x in all_x]+[2.0])*1.1, 2.0) if all_x else 2.0
    
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    fig.add_vline(x=100, line_dash="dash", line_color="gray")
    
    lbl = lim * 0.5
    for x,y,t,c in [(100+lbl,100+lbl,"LEADING","rgba(0,255,0,0.7)"), (100-lbl,100+lbl,"IMPROVING","rgba(0,100,255,0.7)"),
                    (100+lbl,100-lbl,"WEAKENING","rgba(255,165,0,0.7)"), (100-lbl,100-lbl,"LAGGING","rgba(255,0,0,0.7)")]:
        fig.add_annotation(x=x, y=y, text=f"<b>{t}</b>", showarrow=False, font=dict(color=c, size=20))

    fig.update_layout(
        xaxis=dict(title="Relative Trend", showgrid=False, range=[100-lim, 100+lim]),
        yaxis=dict(title="Relative Momentum", showgrid=False, range=[100-lim, 100+lim]),
        height=750, showlegend=False, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig