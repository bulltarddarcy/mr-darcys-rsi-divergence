import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import json
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
HISTORY_YEARS = 1             
BENCHMARK_TICKER = "SPY"      

# Moving Averages
MA_FAST = 8         # EMA
MA_MEDIUM = 21      # EMA
MA_SLOW = 50        # SMA
MA_BASE = 200       # SMA

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

# Paths
DATA_DIR = Path("sector_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = DATA_DIR / "meta.json"

# ==========================================
# 2. DATA MANAGER
# ==========================================
class SectorDataManager:
    def __init__(self):
        self.data_path = DATA_DIR

    def load_universe(self):
        """Reads the universe from st.secrets['SECTOR_UNIVERSE']"""
        secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
        if not secret_val:
            st.error("❌ Secret 'SECTOR_UNIVERSE' is missing or empty.")
            return pd.DataFrame(), [], {}
            
        try:
            if secret_val.strip().startswith("http"):
                # CASE 1: Google Sheets (Native) -> docs.google.com
                if "docs.google.com/spreadsheets" in secret_val:
                    # Extract ID between /d/ and /
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                
                # CASE 2: Google Drive File (Uploaded CSV) -> drive.google.com
                elif "drive.google.com" in secret_val and "/d/" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://drive.google.com/uc?id={file_id}&export=download"
                
                # CASE 3: Standard Direct URL
                else:
                    csv_source = secret_val
                
                df = pd.read_csv(csv_source)
            else:
                # CASE 4: Raw CSV String inside Secrets
                df = pd.read_csv(StringIO(secret_val))
            
            required_cols = ['Ticker', 'Theme']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Universe CSV missing required columns: {required_cols}")
                return pd.DataFrame(), [], {}

            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df['Theme'] = df['Theme'].astype(str).str.strip()
            df['Role'] = df['Role'].astype(str).str.strip().str.title() if 'Role' in df.columns else 'Stock'

            tickers = df['Ticker'].unique().tolist()
            if BENCHMARK_TICKER not in tickers:
                tickers.append(BENCHMARK_TICKER)
                
            etf_rows = df[df['Role'] == 'Etf']
            theme_map = dict(zip(etf_rows['Theme'], etf_rows['Ticker'])) if not etf_rows.empty else {}
            
            return df, tickers, theme_map
            
        except Exception as e:
            st.error(f"Error loading SECTOR_UNIVERSE: {e}")
            return pd.DataFrame(), [], {}

    def get_file_path(self, ticker):
        return self.data_path / f"{ticker}.parquet"

    def save_ticker_data(self, ticker, df):
        if df is None or df.empty: return
        try:
            df.to_parquet(self.get_file_path(ticker))
        except Exception as e:
            print(f"Failed to save {ticker}: {e}")

    def load_ticker_data(self, ticker):
        path = self.get_file_path(ticker)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def load_batch_data(self, tickers):
        """Optimized loading: Loads multiple tickers into a memory dictionary to reduce I/O."""
        cache = {}
        for t in tickers:
            df = self.load_ticker_data(t)
            if df is not None and not df.empty:
                cache[t] = df
        return cache

    def set_last_updated(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(META_FILE, "w") as f:
            json.dump({"last_updated": now_str}, f)

    def get_last_updated(self):
        if META_FILE.exists():
            try:
                with open(META_FILE, "r") as f:
                    data = json.load(f)
                return data.get("last_updated", "Unknown")
            except:
                return "Unknown"
        return "Never"

# ==========================================
# 3. CALCULATOR & UPDATE ENGINE
# ==========================================
class SectorAlphaCalculator:
    def __init__(self):
        self.dm = SectorDataManager()

    def calculate_technical_indicators(self, df):
        if df.empty: return df
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
             df['Close'] = df['Adj Close']

        df['EMA_8'] = df['Close'].ewm(span=MA_FAST, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=MA_MEDIUM, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=MA_SLOW).mean()
        df['SMA_200'] = df['Close'].rolling(window=MA_BASE).mean()
        
        daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
        df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
        
        avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
        df["RVOL"] = df["Volume"] / avg_vol

        for label, time_window in TIMEFRAMES.items():
            df[f"RVOL_{label}"] = df["RVOL"].rolling(window=time_window).mean()
        
        return df

    def _calc_slope(self, series, window):
        x = np.arange(window)
        weights = np.ones(window) if window < 10 else np.linspace(1.0, 3.0, window)
        def slope_func(y): return np.polyfit(x, y, 1, w=weights)[0]
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(self, df, bench_df):
        if df.empty or bench_df.empty: return df
        common_index = df.index.intersection(bench_df.index)
        df = df.loc[common_index].copy()
        bench = bench_df.loc[common_index].copy()
        
        asset_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        raw_ratio = df[asset_col] / bench[bench_col]
        
        for label, time_window in TIMEFRAMES.items():
            ratio_mean = raw_ratio.rolling(window=time_window).mean()
            col_ratio = f"RRG_Ratio_{label}"
            df[col_ratio] = ((raw_ratio - ratio_mean) / ratio_mean) * 100 + 100
            
            raw_slope = self._calc_slope(raw_ratio, time_window)
            velocity = (raw_slope / ratio_mean) * time_window * 100
            col_mom = f"RRG_Mom_{label}"
            df[col_mom] = 100 + velocity
        return df

    def calculate_stock_alpha(self, df, parent_df):
        if df.empty or parent_df.empty: return df
        common_index = df.index.intersection(parent_df.index)
        df = df.loc[common_index].copy()
        parent = parent_df.loc[common_index].copy()
        
        df['Pct_Change'] = df['Close'].pct_change(fill_method=None)
        parent['Pct_Change'] = parent['Close'].pct_change(fill_method=None)
        
        rolling_cov = df['Pct_Change'].rolling(window=BETA_WINDOW).cov(parent['Pct_Change'])
        rolling_var = parent['Pct_Change'].rolling(window=BETA_WINDOW).var()
        df['Beta'] = (rolling_cov / rolling_var).fillna(1.0)
        
        df['Expected_Return'] = parent['Pct_Change'] * df['Beta']
        df['True_Alpha_1D'] = df['Pct_Change'] - df['Expected_Return']
        
        for label, time_window in TIMEFRAMES.items():
            col_alpha = f"True_Alpha_{label}"
            df[col_alpha] = df['True_Alpha_1D'].fillna(0).rolling(window=time_window).sum() * 100
        return df

    def run_full_update(self, status_placeholder=None):
        uni_df, tickers, theme_map = self.dm.load_universe()
        if not tickers: return

        end_date = datetime.today()
        start_date = end_date - timedelta(days=HISTORY_YEARS * 365)
        
        if status_placeholder: status_placeholder.write(f"⬇️ Downloading {len(tickers)} tickers...")
        
        # INCREASED CHUNK SIZE FOR PERFORMANCE
        chunk_size = 100 
        data_cache = {}
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                data = yf.download(chunk, start=start_date, end=end_date, group_by='ticker', auto_adjust=False, threads=True, progress=False)
                for t in chunk:
                    if len(chunk) == 1: t_df = data
                    else: 
                        try: t_df = data[t]
                        except KeyError: continue
                    if not t_df.empty:
                        t_df.index.name = 'Date'
                        data_cache[t] = t_df
            except Exception: continue

        if status_placeholder: status_placeholder.write("🧮 Calculating Benchmarks...")
        spy_df = data_cache.get(BENCHMARK_TICKER)
        if spy_df is None: 
            st.error("Benchmark SPY download failed.")
            return

        spy_df = self.calculate_technical_indicators(spy_df)
        self.dm.save_ticker_data(BENCHMARK_TICKER, spy_df)

        themes = uni_df[uni_df['Role'] == 'Etf']['Theme'].unique()
        
        for theme in themes:
            etf_ticker = theme_map.get(theme)
            if not etf_ticker or etf_ticker not in data_cache: continue
            
            etf_df = data_cache[etf_ticker]
            etf_df = self.calculate_technical_indicators(etf_df)
            etf_df = self.calculate_rrg_metrics(etf_df, spy_df)
            self.dm.save_ticker_data(etf_ticker, etf_df)
            
            stocks = uni_df[(uni_df['Theme'] == theme) & (uni_df['Role'] == 'Stock')]['Ticker'].tolist()
            for stock in stocks:
                if stock not in data_cache: continue
                s_df = data_cache[stock]
                s_df = self.calculate_technical_indicators(s_df)
                s_df = self.calculate_stock_alpha(s_df, etf_df)
                
                if 'True_Alpha_1D' in s_df.columns and not s_df['True_Alpha_1D'].dropna().empty:
                    self.dm.save_ticker_data(stock, s_df)

        self.dm.set_last_updated()
        if status_placeholder: status_placeholder.write("✅ Update Complete.")

# ==========================================
# 4. VISUALIZATION & SIGNAL LOGIC (MOVED FROM MAIN)
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