import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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

# Paths (Using a temp directory for Cloud compatibility)
DATA_DIR = Path("sector_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. DATA MANAGER
# ==========================================
class SectorDataManager:
    def __init__(self):
        self.data_path = DATA_DIR

    def load_universe(self):
        """
        Reads the universe from st.secrets['SECTOR_UNIVERSE']
        Supports:
        1. Google Drive View Links (e.g. https://drive.google.com/.../view?...)
        2. Direct CSV strings (Raw text)
        """
        secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
        if not secret_val:
            st.error("❌ Secret 'SECTOR_UNIVERSE' is missing or empty.")
            return pd.DataFrame(), [], {}
            
        try:
            # --- DETECT URL VS RAW TEXT ---
            if secret_val.strip().startswith("http"):
                # It is a URL. Check if it's Google Drive.
                if "drive.google.com" in secret_val and "/d/" in secret_val:
                    # Convert 'view' link to 'export=download' link
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://drive.google.com/uc?id={file_id}&export=download"
                else:
                    csv_source = secret_val
                
                # Read from URL
                df = pd.read_csv(csv_source)
            else:
                # It is Raw Text. Read from String.
                df = pd.read_csv(StringIO(secret_val))
            
            # --- CLEANING & FORMATTING ---
            # Ensure columns exist
            required_cols = ['Ticker', 'Theme']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Universe CSV missing required columns: {required_cols}")
                return pd.DataFrame(), [], {}

            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df['Theme'] = df['Theme'].astype(str).str.strip()
            
            if 'Role' in df.columns:
                df['Role'] = df['Role'].astype(str).str.strip().str.title() 
            else:
                # If no Role column, infer ETF vs Stock? 
                # Better to default all to Stock if unsure, or require the column.
                # Here we default to Stock to prevent crash.
                df['Role'] = 'Stock' 

            tickers = df['Ticker'].unique().tolist()
            if BENCHMARK_TICKER not in tickers:
                tickers.append(BENCHMARK_TICKER)
                
            # Create Theme Map (Theme -> ETF Ticker)
            # Find rows where Role is 'Etf'
            etf_rows = df[df['Role'] == 'Etf']
            if not etf_rows.empty:
                theme_map = dict(zip(etf_rows['Theme'], etf_rows['Ticker']))
            else:
                theme_map = {}
            
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

# ==========================================
# 3. CALCULATOR & UPDATE ENGINE
# ==========================================
class SectorAlphaCalculator:
    def __init__(self):
        self.dm = SectorDataManager()

    def calculate_technical_indicators(self, df):
        if df.empty: return df
        
        # Ensure we have close data
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
             df['Close'] = df['Adj Close']

        # EMAs/SMAs
        df['EMA_8'] = df['Close'].ewm(span=MA_FAST, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=MA_MEDIUM, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=MA_SLOW).mean()
        df['SMA_200'] = df['Close'].rolling(window=MA_BASE).mean()
        
        # Volatility & RVOL
        daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
        df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
        
        avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
        df["RVOL"] = df["Volume"] / avg_vol

        for label, time_window in TIMEFRAMES.items():
            df[f"RVOL_{label}"] = df["RVOL"].rolling(window=time_window).mean()
        
        return df

    def _calc_slope(self, series, window):
        # Weighted Regression
        x = np.arange(window)
        if window < 10:
            weights = np.ones(window)
        else:
            weights = np.linspace(1.0, 3.0, window)
        
        def slope_func(y):
            return np.polyfit(x, y, 1, w=weights)[0]
            
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(self, df, bench_df):
        if df.empty or bench_df.empty: return df
        
        asset_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        common_index = df.index.intersection(bench_df.index)
        df = df.loc[common_index].copy()
        bench = bench_df.loc[common_index].copy()
        
        raw_ratio = df[asset_col] / bench[bench_col]
        
        for label, time_window in TIMEFRAMES.items():
            ratio_mean = raw_ratio.rolling(window=time_window).mean()
            
            # X-Axis: Trend
            col_ratio = f"RRG_Ratio_{label}"
            df[col_ratio] = ((raw_ratio - ratio_mean) / ratio_mean) * 100 + 100
            
            # Y-Axis: Momentum
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
        
        chunk_size = 50
        data_cache = {}
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                # auto_adjust=False to allow separation of Price (Close) and Total Return (Adj Close)
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

        if status_placeholder: status_placeholder.write("✅ Update Complete.")