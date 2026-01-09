import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from io import StringIO, BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
HISTORY_YEARS = 1             

# Volatility & Normalization
ATR_WINDOW = 20               
AVG_VOLUME_WINDOW = 20        
BETA_WINDOW = 60              

# Timeframes (Trading Days)
TIMEFRAMES = {
    'Short': 5,    # 5 Trading Days
    'Med':   10,   # 10 Trading Days
    'Long':  20    # 20 Trading Days
}

# Filters
MIN_DOLLAR_VOLUME = 2_000_000

# ==========================================
# 2. DATA MANAGER
# ==========================================
class SectorDataManager:
    def __init__(self):
        self.universe = pd.DataFrame()
        self.ticker_map = {}

    def load_universe(self, benchmark_ticker="SPY"):
        """Reads the universe from st.secrets['SECTOR_UNIVERSE']"""
        secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
        if not secret_val:
            st.error("❌ Secret 'SECTOR_UNIVERSE' is missing or empty.")
            return pd.DataFrame(), [], {}
            
        try:
            # Handle Google Sheet Links or Raw CSV
            if secret_val.strip().startswith("http"):
                if "docs.google.com/spreadsheets" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                elif "drive.google.com" in secret_val and "/d/" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://drive.google.com/uc?id={file_id}&export=download"
                else:
                    csv_source = secret_val
                df = pd.read_csv(csv_source)
            else:
                df = pd.read_csv(StringIO(secret_val))
            
            required_cols = ['Ticker', 'Theme']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Universe CSV missing required columns: {required_cols}")
                return pd.DataFrame(), [], {}

            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df['Theme'] = df['Theme'].astype(str).str.strip()
            df['Role'] = df['Role'].astype(str).str.strip().str.title() if 'Role' in df.columns else 'Stock'

            tickers = df['Ticker'].unique().tolist()
            if benchmark_ticker not in tickers:
                tickers.append(benchmark_ticker)
                
            etf_rows = df[df['Role'] == 'Etf']
            theme_map = dict(zip(etf_rows['Theme'], etf_rows['Ticker'])) if not etf_rows.empty else {}
            
            self.universe = df
            return df, tickers, theme_map
            
        except Exception as e:
            st.error(f"Error loading SECTOR_UNIVERSE: {e}")
            return pd.DataFrame(), [], {}

    def load_ticker_map(self):
        """Loads the Global Ticker Map to find Parquet IDs"""
        try:
            url = st.secrets.get("URL_TICKER_MAP")
            if not url: return {}

            buffer = get_gdrive_binary_data(url)
            if buffer:
                df = pd.read_csv(buffer, engine='c')
                # Assume Col 0 = Ticker, Col 1 = File ID
                if len(df.columns) >= 2:
                    # Normalized Key: UPPERCASE and STRIPPED for reliable lookup
                    t_map = dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                                     df.iloc[:, 1].astype(str).str.strip()))
                    self.ticker_map = t_map
                    return t_map
        except Exception:
            pass
        return {}

# ==========================================
# 3. CALCULATOR & PIPELINE
# ==========================================
class SectorAlphaCalculator:
    def process_dataframe(self, df):
        """
        Prepares a raw Parquet dataframe for the app.
        Renames source columns (EMA8 -> EMA_8) and calcs missing metrics (RVOL).
        """
        if df is None or df.empty: return None

        # 1. Normalize Columns (Source Parquet -> App Standard)
        # Source: EMA8, SMA50, Close | App: EMA_8, SMA_50, Close
        col_map = {
            'EMA8': 'EMA_8', 'EMA21': 'EMA_21',
            'SMA50': 'SMA_50', 'SMA100': 'SMA_100', 'SMA200': 'SMA_200',
            'RSI14': 'RSI_14', 'RSI': 'RSI_14'
        }
        # Rename only if they exist
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure 'Close' exists
        if 'Close' not in df.columns and 'CLOSE' in df.columns:
            df['Close'] = df['CLOSE']
        
        # 2. Calculate Missing Technicals (RVOL, ADR)
        # We assume EMA/SMA are already in the parquet, so we skip those to save time.
        
        # Daily Range % (ADR)
        if 'High' in df.columns and 'Low' in df.columns:
            daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
            df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
        
        # RVOL
        if 'Volume' in df.columns:
            avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
            df["RVOL"] = df["Volume"] / avg_vol

            for label, time_window in TIMEFRAMES.items():
                df[f"RVOL_{label}"] = df["RVOL"].rolling(window=time_window).mean()
        
        return df

    def _calc_slope(self, series, window):
        x = np.arange(window)
        # Weighted linear regression for slope
        weights = np.ones(window) if window < 10 else np.linspace(1.0, 3.0, window)
        def slope_func(y): 
            try:
                return np.polyfit(x, y, 1, w=weights)[0]
            except: return 0.0
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(self, df, bench_df):
        if df is None or df.empty or bench_df is None or bench_df.empty: return df
        
        # Align dates
        common_index = df.index.intersection(bench_df.index)
        if common_index.empty: return df
        
        df_aligned = df.loc[common_index].copy()
        bench_aligned = bench_df.loc[common_index].copy()
        
        # Determine Close columns
        asset_col = 'Adj Close' if 'Adj Close' in df_aligned.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        # Relative Ratio
        raw_ratio = df_aligned[asset_col] / bench_aligned[bench_col]
        
        # RRG Logic
        for label, time_window in TIMEFRAMES.items():
            ratio_mean = raw_ratio.rolling(window=time_window).mean()
            
            # Trend (Ratio)
            col_ratio = f"RRG_Ratio_{label}"
            # Normalized around 100
            df_aligned[col_ratio] = ((raw_ratio - ratio_mean) / ratio_mean) * 100 + 100
            
            # Momentum (Velocity of the Ratio)
            raw_slope = self._calc_slope(raw_ratio, time_window)
            velocity = (raw_slope / ratio_mean) * time_window * 100
            
            col_mom = f"RRG_Mom_{label}"
            df_aligned[col_mom] = 100 + velocity
            
        return df_aligned

    def calculate_stock_alpha(self, df, parent_df):
        if df is None or df.empty or parent_df is None or parent_df.empty: return df
        
        common_index = df.index.intersection(parent_df.index)
        if common_index.empty: return df
        
        df_aligned = df.loc[common_index].copy()
        parent_aligned = parent_df.loc[common_index].copy()
        
        # Returns
        df_aligned['Pct_Change'] = df_aligned['Close'].pct_change(fill_method=None)
        parent_aligned['Pct_Change'] = parent_aligned['Close'].pct_change(fill_method=None)
        
        # Beta
        rolling_cov = df_aligned['Pct_Change'].rolling(window=BETA_WINDOW).cov(parent_aligned['Pct_Change'])
        rolling_var = parent_aligned['Pct_Change'].rolling(window=BETA_WINDOW).var()
        df_aligned['Beta'] = (rolling_cov / rolling_var).fillna(1.0)
        
        # Alpha
        df_aligned['Expected_Return'] = parent_aligned['Pct_Change'] * df_aligned['Beta']
        df_aligned['True_Alpha_1D'] = df_aligned['Pct_Change'] - df_aligned['Expected_Return']
        
        for label, time_window in TIMEFRAMES.items():
            col_alpha = f"True_Alpha_{label}"
            df_aligned[col_alpha] = df_aligned['True_Alpha_1D'].fillna(0).rolling(window=time_window).sum() * 100
            
        return df_aligned


# ==========================================
# 4. ORCHESTRATOR (FETCH & PROCESS)
# ==========================================

def _fetch_single_parquet(ticker, file_id):
    """Helper to fetch one parquet file from Drive"""
    if not file_id: return None
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        buffer = get_gdrive_binary_data(url)
        if buffer:
            df = pd.read_parquet(buffer)
            # Normalize Index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            return df
    except Exception:
        pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_process_universe(benchmark_ticker="SPY"):
    """
    Main Data Pipeline:
    1. Loads Universe & Ticker Map
    2. Downloads Benchmark (SPY or QQQ) - Robust Lookup
    3. Downloads All Sector Tickers (Parallel)
    4. Calculates Relative Strength & Alpha
    5. Returns Dictionary of DataFrames and Missing Tickers list
    """
    dm = SectorDataManager()
    uni_df, tickers, theme_map = dm.load_universe(benchmark_ticker)
    t_map = dm.load_ticker_map()
    
    if uni_df.empty or not t_map:
        return {}, [], theme_map, uni_df

    calc = SectorAlphaCalculator()
    data_cache = {}
    missing_tickers = []

    # --- 1. ROBUST BENCHMARK LOOKUP ---
    # Try all likely variations of the name
    possible_keys = [
        benchmark_ticker,                       # SPY
        f"{benchmark_ticker}_PARQUET",          # SPY_PARQUET (User requested)
        f"{benchmark_ticker}.PARQUET",          # SPY.PARQUET
        f"PARQUET_{benchmark_ticker}"           # PARQUET_SPY
    ]
    
    bench_id = None
    for key in possible_keys:
        if key in t_map:
            bench_id = t_map[key]
            break
            
    # Final fallback: Look for key that CONTAINS ticker + PARQUET
    if not bench_id:
        for k, v in t_map.items():
            if benchmark_ticker in k and "PARQUET" in k:
                bench_id = v
                break

    bench_df = _fetch_single_parquet(benchmark_ticker, bench_id)
    if bench_df is None or bench_df.empty:
        # Critical failure if Benchmark is missing, cannot calculate RS/Alpha
        return {}, [f"CRITICAL: {benchmark_ticker} (Benchmark) not found. Tried keys: {possible_keys}"], theme_map, uni_df

    # Process Benchmark
    bench_df = calc.process_dataframe(bench_df)
    data_cache[benchmark_ticker] = bench_df

    # 2. Identify Missing Tickers (Using same robust logic)
    valid_tickers = []
    
    for t in tickers:
        # Skip benchmark as we already have it
        if t == benchmark_ticker: continue
        
        # Robust lookup for universe tickers too
        tid = None
        # Order of preference: Exact -> _PARQUET -> PARQUET_
        if t in t_map: tid = t_map[t]
        elif f"{t}_PARQUET" in t_map: tid = t_map[f"{t}_PARQUET"]
        elif f"PARQUET_{t}" in t_map: tid = t_map[f"PARQUET_{t}"]
        
        if tid:
            valid_tickers.append((t, tid))
        else:
            missing_tickers.append(t)

    # 3. Parallel Download
    raw_dfs = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_ticker = {executor.submit(_fetch_single_parquet, t, fid): t for t, fid in valid_tickers}
        
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    raw_dfs[t] = df
                else:
                    missing_tickers.append(t)
            except:
                missing_tickers.append(t)

    # 4. Processing & Calculations
    # A. Process ETFs (Themes) first for RRG
    etf_tickers = list(theme_map.values())
    
    for theme, etf_ticker in theme_map.items():
        if etf_ticker not in raw_dfs: continue
        
        df = raw_dfs[etf_ticker]
        df = calc.process_dataframe(df) # Standardize cols
        df = calc.calculate_rrg_metrics(df, bench_df) # Calculate RRG vs Benchmark
        
        data_cache[etf_ticker] = df

    # B. Process Individual Stocks (Alpha vs Sector ETF)
    # We need the parent ETF data ready first (done above)
    stocks = uni_df[uni_df['Role'] == 'Stock']
    
    for _, row in stocks.iterrows():
        stock = row['Ticker']
        theme = row['Theme']
        parent_etf = theme_map.get(theme)
        
        if stock not in raw_dfs: continue
        
        s_df = raw_dfs[stock]
        s_df = calc.process_dataframe(s_df)
        
        # Calculate Alpha relative to its Sector ETF (if available), else Benchmark
        parent_df = data_cache.get(parent_etf, bench_df) 
        s_df = calc.calculate_stock_alpha(s_df, parent_df)
        
        data_cache[stock] = s_df

    return data_cache, missing_tickers, theme_map, uni_df

# ==========================================
# 5. VISUALIZATION HELPERS
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

def plot_simple_rrg(data_cache, target_map, view_key, show_trails):
    fig = go.Figure()
    all_x, all_y = [], []
    
    for theme, ticker in target_map.items():
        # Use cache instead of loading from disk
        df = data_cache.get(ticker)
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