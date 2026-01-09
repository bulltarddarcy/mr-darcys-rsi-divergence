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
            df = df.sort_values(['Ticker', 'Date'])
            
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        
        return df
    except Exception:
        return None

# ==========================================
# 3. PROCESSING (LAYER 2 - MATH)
# ==========================================

def _calc_rrg_metrics(close_series, bench_series):
    """Vectorized RRG calculation."""
    ratio = close_series / bench_series
    results = {}
    
    for label, w in TIMEFRAMES.items():
        # Trend: Ratio vs Moving Average of Ratio
        ma_ratio = ratio.rolling(window=w).mean()
        results[f"RRG_Ratio_{label}"] = ((ratio - ma_ratio) / ma_ratio) * 100 + 100
        
        # Momentum: ROC of the Ratio
        results[f"RRG_Mom_{label}"] = (ratio.pct_change(periods=w) * 100) + 100
        
    return pd.DataFrame(results, index=close_series.index)

@st.cache_data(ttl=1200, show_spinner="Calculating Alpha & Beta...")
def get_computed_sector_data(benchmark_ticker):
    """
    Uses pre-calculated technicals from parquet. 
    Only calculates Alpha/Beta/RVOL on the fly.
    """
    master_df = load_raw_sector_data()
    uni_df, theme_map = load_universe_config()
    
    if master_df is None or uni_df.empty: return {}, [], theme_map, uni_df

    # 1. Calculate RVOL (Vectorized)
    # We assume Volume is there, but RVOL isn't pre-calc
    if 'RVOL' not in master_df.columns:
        master_df['Vol_Avg'] = master_df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
        master_df['RVOL'] = master_df['Volume'] / master_df['Vol_Avg']

    # 2. Pivot for Fast Alpha Math (Wide Format)
    df_close_wide = master_df.pivot(index='Date', columns='Ticker', values='Close').ffill()
    df_rets_wide = df_close_wide.pct_change()
    
    if benchmark_ticker not in df_close_wide.columns:
        return {}, [f"{benchmark_ticker} missing"], theme_map, uni_df

    bench_series = df_close_wide[benchmark_ticker]
    data_cache = {}

    # 3. Process ETFs (RRG)
    etf_tickers = list(theme_map.values())
    for etf in etf_tickers:
        if etf not in df_close_wide.columns: continue
        
        subset = master_df[master_df['Ticker'] == etf].copy().set_index('Date').sort_index()
        
        # Calc RRG Metrics vs Benchmark
        aligned_bench = bench_series.loc[subset.index]
        rrg_metrics = _calc_rrg_metrics(subset['Close'], aligned_bench)
        
        data_cache[etf] = pd.concat([subset, rrg_metrics], axis=1)

    # 4. Process Stocks (Alpha/Beta)
    stocks = uni_df[uni_df['Role'] == 'Stock']
    
    # Pre-calculate Rolling RVOLs for display
    for k, w in TIMEFRAMES.items():
        master_df[f"RVOL_{k}"] = master_df.groupby('Ticker')['RVOL'].transform(lambda x: x.rolling(w).mean())

    for _, row in stocks.iterrows():
        stock = row['Ticker']
        theme = row['Theme']
        parent_etf = theme_map.get(theme, benchmark_ticker)
        
        if stock not in df_close_wide.columns: continue
        if parent_etf not in df_close_wide.columns: parent_etf = benchmark_ticker
        
        subset = master_df[master_df['Ticker'] == stock].copy().set_index('Date').sort_index()
        
        # Alpha/Beta Calculation
        stock_rets = df_rets_wide[stock]
        parent_rets = df_rets_wide[parent_etf]
        
        rolling_cov = stock_rets.rolling(BETA_WINDOW).cov(parent_rets)
        rolling_var = parent_rets.rolling(BETA_WINDOW).var()
        beta = (rolling_cov / rolling_var).fillna(1.0)
        
        true_alpha = stock_rets - (parent_rets * beta)
        
        subset['Beta'] = beta
        subset['True_Alpha_1D'] = true_alpha
        
        for k, w in TIMEFRAMES.items():
            subset[f"True_Alpha_{k}"] = true_alpha.rolling(w).sum() * 100
            
        data_cache[stock] = subset

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
    
    if m20 < 100 and m5 > 100 and m5 > (m20 + 2): return "🪝 J-Hook"
    if r20 > 100 and m5 > 100 and m5 > m10: return "🚩 Bull Flag"
    if m5 > m10 and m10 > m20 and m20 > 100: return "🚀 Rocket"
    return None 

def get_quadrant_status(df, key):
    if df is None or df.empty: return "N/A"
    r = df[f"RRG_Ratio_{key}"].iloc[-1]
    m = df[f"RRG_Mom_{key}"].iloc[-1]
    if r >= 100 and m >= 100: return "🟢 Leading"
    elif r < 100 and m >= 100: return "🔵 Improving"
    elif r < 100 and m < 100: return "🔴 Lagging"
    return "🟡 Weakening"

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