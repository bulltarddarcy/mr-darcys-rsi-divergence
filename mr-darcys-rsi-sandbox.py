import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import math
import requests
import re
import time
import altair as alt
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 0. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# --- 1. GLOBAL DATA LOADING & UTILITIES ---

COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
# RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21

@st.cache_data(ttl=600, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        want = ["Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"]
        
        existing_cols = set(df.columns)
        keep = [c for c in want if c in existing_cols]
        df = df[keep].copy()
        
        str_cols = [c for c in ["Order Type", "Symbol", "Strike", "Expiry"] if c in df.columns]
        for c in str_cols:
            df[c] = df[c].astype(str).str.strip()
        
        if "Dollars" in df.columns:
            if df["Dollars"].dtype == 'object':
                df["Dollars"] = df["Dollars"].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)

        if "Contracts" in df.columns:
            if df["Contracts"].dtype == 'object':
                df["Contracts"] = df["Contracts"].str.replace(',', '', regex=False)
            df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(0)
        
        if "Trade Date" in df.columns:
            df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        
        if "Expiry" in df.columns:
            df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
            
        if "Strike (Actual)" in df.columns:
            df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
            
        if "Error" in df.columns:
            mask = df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
            df = df[~mask]
            
        return df
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        return pd.DataFrame()

def parse_periods(periods_str):
    """Parses a comma-separated string into a list of integers."""
    try:
        # Split by comma, strip whitespace, convert to int, filter valid numbers, sort unique
        p_list = sorted(list(set([int(x.strip()) for x in periods_str.split(',') if x.strip().isdigit()])))
        if not p_list:
            # Fallback if parsing fails, though specific defaults are usually set in session state
            return [5, 21, 63, 126] 
        return p_list
    except:
        return [5, 21, 63, 126]

def add_technicals(df):
    """
    Centralized technical indicator calculation.
    Adds RSI (14), EMA (8, 21), and SMA (200) if they don't exist.
    """
    if df is None or df.empty: return df
    
    # 1. Identify Close Column
    cols = df.columns
    close_col = next((c for c in ['Price', 'Close', 'CLOSE'] if c in cols), None)
    
    if not close_col: return df

    # 2. RSI Calculation
    if not any(x in cols for x in ['RSI', 'RSI_14', 'RSI14']):
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI']
    
    # 3. EMA/SMA Calculation
    if not any(x in cols for x in ['EMA8', 'EMA_8']):
        df['EMA_8'] = df[close_col].ewm(span=8, adjust=False).mean()
        
    if not any(x in cols for x in ['EMA21', 'EMA_21']):
        df['EMA_21'] = df[close_col].ewm(span=21, adjust=False).mean()
        
    if not any(x in cols for x in ['SMA200', 'SMA_200']):
        if len(df) >= 200:
            df['SMA_200'] = df[close_col].rolling(window=200).mean()
            
    return df
    
@st.cache_data(ttl=86400)
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        # fast_info is efficient and doesn't require a full API scrape
        mc = t.fast_info.get('marketCap')
        if mc: return float(mc)
        
        # Fallback: slightly slower but more comprehensive
        info = t.info
        mc = info.get('marketCap')
        if mc: return float(mc)
    except Exception:
        pass
    return 0.0

def fetch_market_caps_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except:
                results[t] = 0.0
    return results

@st.cache_data(ttl=300)
def is_above_ema21(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        h = ticker.history(period="60d")
        if len(h) < 21:
            return True 
        ema21_last = h["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
        latest_price = h["Close"].iloc[-1]
        return latest_price > ema21_last
    except:
        return True

@st.cache_data(ttl=300)
def get_stock_indicators(sym: str):
    try:
        ticker_obj = yf.Ticker(sym)
        h_full = ticker_obj.history(period="2y", interval="1d")
        
        if len(h_full) == 0: return None, None, None, None, None
        
        # Centralized Calc
        h_full = add_technicals(h_full)
        
        sma200 = float(h_full["SMA_200"].iloc[-1]) if "SMA_200" in h_full.columns else None
        
        h_recent = h_full.iloc[-60:].copy() if len(h_full) > 60 else h_full.copy()
        if len(h_recent) == 0: return None, None, None, None, None
        
        spot_val = float(h_recent["Close"].iloc[-1])
        # Handle variations (Helper adds EMA_8, but existing files might have EMA8)
        ema8 = float(h_recent.get("EMA_8", h_recent.get("EMA8")).iloc[-1])
        ema21 = float(h_recent.get("EMA_21", h_recent.get("EMA21")).iloc[-1])
        
        return spot_val, ema8, ema21, sma200, h_full
    except: 
        return None, None, None, None, None

def fetch_technicals_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_stock_indicators, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except:
                results[t] = (None, None, None, None, None)
    return results

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

@st.cache_data(ttl=3600)
def get_expiry_color_map():
    try:
        today = date.today()
        days_ahead = (4 - today.weekday()) % 7
        this_fri = today + timedelta(days=days_ahead)
        next_fri = this_fri + timedelta(days=7)
        two_fri = this_fri + timedelta(days=14)
        
        return {
            this_fri.strftime("%d %b %y"): "background-color: #b7e1cd; color: black;",
            next_fri.strftime("%d %b %y"): "background-color: #fce8b2; color: black;",
            two_fri.strftime("%d %b %y"): "background-color: #f4c7c3; color: black;"
        }
    except:
        return {}

def highlight_expiry(val):
    if not isinstance(val, str): return ""
    color_map = get_expiry_color_map()
    return color_map.get(val, "")

def clean_strike_fmt(val):
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid_dates = df["Trade Date"].dropna()
        if not valid_dates.empty:
            return valid_dates.max().date()
    return date.today() - timedelta(days=1)


def get_confirmed_gdrive_data(url):
    """
    Downloads CSV data from Google Drive with safety timeouts to prevent hanging.
    """
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if not file_id: return None
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        
        # 10-second timeout prevents the "Infinite Spinner" if Drive is slow
        response = session.get(download_url, params={'id': file_id}, stream=True, timeout=10)
        
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match: confirm_token = match.group(1)

        if confirm_token:
            # Second request to bypass the warning screen
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True, timeout=10)
        
        if response.text.strip().startswith("<!DOCTYPE html>"): 
            return "HTML_ERROR"
            
        return StringIO(response.text)
    except requests.exceptions.Timeout:
        # Fails gracefully so the script can move to Yahoo fallback
        print(f"Drive timeout for URL: {url}")
        return None
    except Exception as e:
        print(f"Drive Error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_parquet_config():
    """
    Loads dataset configuration. 
    Priority 1: Text file in Google Drive (URL defined in secrets as URL_PARQUET_LIST)
    Priority 2: String in secrets (PARQUET_CONFIG)
    """
    config = {}
    
    # 1. Try loading from Google Drive Text File
    url_list = st.secrets.get("URL_PARQUET_LIST", "")
    if url_list:
        try:
            buffer = get_confirmed_gdrive_data(url_list)
            if buffer and buffer != "HTML_ERROR":
                content = buffer.getvalue()
                lines = content.strip().split('\n')
                for line in lines:
                    if not line.strip(): continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        name = re.sub(r'\\s*', '', name)
                        key = parts[1].strip()
                        config[name] = key
        except Exception as e:
            print(f"Error loading external config: {e}")

    # 2. Fallback to secrets string
    if not config:
        try:
            raw_config = st.secrets.get("PARQUET_CONFIG", "")
            if raw_config:
                lines = raw_config.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        key = parts[1].strip()
                        config[name] = key
        except Exception:
            pass
    
    if not config:
        st.error("⛔ CRITICAL ERROR: No dataset configuration found. Please check 'URL_PARQUET_LIST' in your secrets.toml.")
        st.stop()
        
    return config

DATA_KEYS_PARQUET = get_parquet_config()


def get_gdrive_binary_data(url):
    """
    Forces Google Drive to provide the raw file content, bypassing 
    previews and handling confirmation for large/protected files.
    """
    try:
        # 1. Extract the unique File ID
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            # Extracts ID from between /d/ and the next slash or question mark
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
            if match:
                file_id = match.group(1)
        
        if not file_id: 
            return None
            
        # 2. Reconstruct the direct download endpoint
        download_url = "https://docs.google.com/uc?export=download&id=" + file_id
        session = requests.Session()
        
        # 3. First attempt to fetch
        response = session.get(download_url, stream=True, timeout=15)
        
        # 4. Check if we received an HTML warning instead of the file
        # Parquet files start with 'PAR1', CSVs don't start with '<!DOCTYPE'
        if response.text.strip().startswith("<!DOCTYPE html>"):
            # It's a warning page. Scrape the confirmation token.
            confirm_token = None
            match = re.search(r'confirm=([0-9A-Za-z-_]+)', response.text)
            if match:
                confirm_token = match.group(1)
                # Re-request with the token
                response = session.get(download_url + "&confirm=" + confirm_token, stream=True, timeout=15)
        
        # 5. Return the raw bytes
        return BytesIO(response.content)
    except Exception as e:
        print(f"G-Drive Sync Error: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="Loading Dataset...")
def load_parquet_and_clean(key):
    # 1. Fetch the URL from secrets based on the key (e.g., PARQUET_SP100)
    if key not in st.secrets:
        st.error(f"Key '{key}' not found in Streamlit Secrets.")
        return None
        
    url = st.secrets[key]
    
    try:
        # 2. Use your existing robust binary helper to handle Google Drive
        # This function handles confirmation tokens for large files (SP500/SP100)
        buffer = get_gdrive_binary_data(url)
        
        if not buffer:
            st.error(f"Could not retrieve data for {key}. Check if the Drive link is public.")
            return None
            
        content = buffer.getvalue()

        # 3. Try Parquet First, then Fallback to CSV
        try:
            df = pd.read_parquet(BytesIO(content))
        except Exception:
            try:
                # Sometimes these files are actually CSVs despite the name
                df = pd.read_csv(BytesIO(content))
            except Exception as e:
                st.error(f"Error reading {key}: {e}")
                return None

        # 4. Standardize Columns
        df.columns = [c.strip() for c in df.columns]
        
        # Look for a Date column
        date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.rename(columns={date_col: 'ChartDate'}).sort_values('ChartDate')
            
        # Ensure Price/Close consistency
        if 'Close' not in df.columns and 'Price' in df.columns:
            df = df.rename(columns={'Price': 'Close'})
        if 'Close' in df.columns and 'Price' not in df.columns:
            df['Price'] = df['Close']
            
        # Ensure Numeric for calculations
        cols_to_numeric = ['Price', 'High', 'Low', 'Open', 'Volume', 'RSI', 'EMA8', 'EMA21', 'VolSMA']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df
        
    except Exception as e:
        st.error(f"Critical error loading {key}: {e}")
        return None


@st.cache_data(ttl=3600)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP")
        if not url: return {}

        buffer = get_confirmed_gdrive_data(url)
        if buffer and buffer != "HTML_ERROR":
            df = pd.read_csv(buffer)
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                              df.iloc[:, 1].astype(str).str.strip()))
    except Exception:
        pass
    return {}

@st.cache_data(ttl=300)
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping:
        return None
    file_id = mapping[ticker]
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    buffer = get_confirmed_gdrive_data(file_url)
    if buffer and buffer != "HTML_ERROR":
        try:
            df = pd.read_csv(buffer)
            df.columns = [c.strip().upper() for c in df.columns]
            return df
        except:
            return None
    return None

def calculate_optimal_signal_stats(history_indices, price_array, current_idx, signal_type='Bullish', timeframe='Daily', periods_input=None, optimize_for='PF'):
    """
    Vectorized calculation of forward returns for multiple periods.
    optimize_for: 'PF' (Profit Factor) or 'SQN' (System Quality Number)
    """
    # 1. Filter for valid historical indices
    hist_arr = np.array(history_indices)
    valid_mask = hist_arr < current_idx
    valid_indices = hist_arr[valid_mask]
    
    if len(valid_indices) == 0:
        return None

    # Handle Periods
    if periods_input is None:
        periods = np.array([5, 21, 63, 126]) # Default trading days
    else:
        periods = np.array(periods_input)

    total_len = len(price_array)
    unit = 'w' if timeframe.lower() == 'weekly' else 'd'

    # 2. Vectorized Exit Index Calculation
    exit_indices_matrix = valid_indices[:, None] + periods[None, :]
    
    # 3. Create Validity Mask
    valid_exits_mask = exit_indices_matrix < total_len
    
    # 4. Fetch Prices Safely
    safe_exit_indices = np.clip(exit_indices_matrix, 0, total_len - 1)
    
    entry_prices = price_array[valid_indices]
    exit_prices_matrix = price_array[safe_exit_indices]
    
    # 5. Calculate Returns Matrix
    raw_returns_matrix = (exit_prices_matrix - entry_prices[:, None]) / entry_prices[:, None]
    
    if signal_type == 'Bearish':
        strat_returns_matrix = -raw_returns_matrix
    else:
        strat_returns_matrix = raw_returns_matrix

    # 6. Calculate Stats per Period
    best_score = -999.0
    best_stats = None
    
    for i, p in enumerate(periods):
        col_mask = valid_exits_mask[:, i]
        period_returns = strat_returns_matrix[col_mask, i]
        
        if len(period_returns) == 0:
            continue
            
        wins = period_returns[period_returns > 0]
        losses = period_returns[period_returns < 0]
        
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        
        if gross_loss == 0:
            pf = 999.0 if gross_win > 0 else 0.0
        else:
            pf = gross_win / gross_loss
            
        n = len(period_returns)
        win_rate = (len(wins) / n) * 100
        avg_ret = np.mean(period_returns) * 100
        
        # --- SQN Calculation ---
        # Standard Deviation requires at least 2 data points generally, 
        # but numpy will return 0.0 for 1 data point (ddof=0 default).
        std_dev = np.std(period_returns)
        
        if std_dev > 0 and n > 0:
            sqn = (np.mean(period_returns) / std_dev) * np.sqrt(n)
        else:
            sqn = 0.0
        
        # Determine Score based on optimization metric
        current_score = pf if optimize_for == 'PF' else sqn
        
        if current_score > best_score:
            best_score = current_score
            best_stats = {
                "Best Period": f"{p}{unit}",
                "Profit Factor": pf,
                "Win Rate": win_rate,
                "EV": avg_ret,
                "N": n,
                "SQN": sqn
            }
            
    return best_stats

# --- NEW HELPERS FOR TOP 3 ---

def get_optimal_rsi_duration(history_df, current_rsi, tolerance=2.0):
    if history_df is None or len(history_df) < 100:
        return 30, "Default (No Hist)"

    history_df = add_technicals(history_df)
    
    close_col = "CLOSE" if "CLOSE" in history_df.columns else "Close"
    rsi_col = "RSI_14" if "RSI_14" in history_df.columns else "RSI"
    
    rsi_vals = history_df[rsi_col].values
    close_vals = history_df[close_col].values
    
    min_rsi = current_rsi - tolerance
    max_rsi = current_rsi + tolerance
    
    mask = (rsi_vals >= min_rsi) & (rsi_vals <= max_rsi)
    match_indices = np.where(mask)[0]
    
    if len(match_indices) < 5:
        return 30, "Default (Low Samples)"
        
    periods = [14, 30, 45, 60]
    best_p = 30
    best_score = -999
    
    total_len = len(close_vals)
    
    for p in periods:
        valid_indices = match_indices[match_indices + p < total_len]
        if len(valid_indices) < 5: continue
        
        entries = close_vals[valid_indices]
        exits = close_vals[valid_indices + p]
        returns = (exits - entries) / entries
        
        win_rate = np.mean(returns > 0)
        avg_ret = np.mean(returns)
        
        score = (win_rate * 2) + avg_ret 
        
        if score > best_score:
            best_score = score
            best_p = p
            
    return best_p, f"RSI Backtest (Optimal {best_p}d)"

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None

    today_dt = pd.to_datetime(date.today())
    f = global_df[
        (global_df["Symbol"].astype(str).str.upper() == ticker) & 
        (global_df["Expiry_DT"] > today_dt)
    ].copy()
    
    if f.empty: return None

    if order_type_filter:
        f = f[f["Order Type"] == order_type_filter]
    else:
        f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
        
    if f.empty: return None
    
    f = f.sort_values(by="Dollars", ascending=False)
    
    whale_trade = f.iloc[0]
    whale_strike = whale_trade["Strike (Actual)"]
    whale_exp = whale_trade["Expiry_DT"]
    whale_dollars = whale_trade["Dollars"]
    whale_type = whale_trade["Order Type"]
    
    if whale_type == "Puts Sold" and whale_strike > current_price:
        otm_puts = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm_puts.empty:
            whale_trade = otm_puts.iloc[0]
            whale_strike = whale_trade["Strike (Actual)"]
            whale_exp = whale_trade["Expiry_DT"]
    
    return {
        "Strike": whale_strike,
        "Expiry": whale_exp.strftime("%d %b"),
        "Dollars": whale_dollars,
        "Type": whale_type
    }

def analyze_trade_setup(ticker, t_df, global_df):
    score = 0
    reasons = []
    suggestions = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    
    if t_df is None or t_df.empty:
        return 0, ["No data"], suggestions

    last = t_df.iloc[-1]
    close = last.get('CLOSE', 0) if 'CLOSE' in last else last.get('Close', 0)
    ema8 = last.get('EMA_8', 0)
    ema21 = last.get('EMA_21', 0)
    sma200 = last.get('SMA_200', 0)
    rsi = last.get('RSI_14', 50)
    
    if close > ema8 and close > ema21:
        score += 2
        reasons.append("Strong Trend (Price > EMA8 & EMA21)")
    elif close > ema21:
        score += 1
        reasons.append("Moderate Trend (Price > EMA21)")
        
    if close > sma200:
        score += 2
        reasons.append("Long-term Bullish (> SMA200)")
        
    if 45 < rsi < 65:
        score += 2
        reasons.append(f"Healthy Momentum (RSI {rsi:.0f})")
    elif rsi >= 70:
        score -= 1
        reasons.append("Overbought (RSI > 70)")
    
    opt_days, opt_reason = 30, "Standard 30d"
    if len(t_df) > 100:
         opt_days, opt_reason = get_optimal_rsi_duration(t_df, rsi)
    
    target_date = date.today() + timedelta(days=opt_days)
    target_date_str = target_date.strftime("%d %b")
    
    put_whale = find_whale_confluence(ticker, global_df, close, "Puts Sold")
    call_whale = find_whale_confluence(ticker, global_df, close, "Calls Bought")
    
    sp_strike = math.floor(ema21) 
    sp_reason = "EMA21 Support"
    sp_exp = target_date_str
    
    if put_whale and put_whale["Strike"] < close:
        sp_strike = put_whale["Strike"]
        sp_reason = f"Whale Tailing (${put_whale['Dollars']/1e6:.1f}M sold)"
        sp_exp = put_whale["Expiry"] 
    elif call_whale:
         sp_exp = call_whale["Expiry"]
         sp_reason = f"EMA21 (Align with Call Whale Exp)"
    
    suggestions['Sell Puts'] = f"Strike ${sp_strike} ({sp_reason}), Exp ~{sp_exp}"

    bc_strike = math.ceil(close)
    bc_reason = "ATM Momentum"
    bc_exp = target_date_str
    
    if call_whale:
        bc_strike = call_whale["Strike"]
        bc_exp = call_whale["Expiry"]
        bc_reason = f"Tailing Call Whale (${call_whale['Dollars']/1e6:.1f}M)"
        
    if close > ema8 or call_whale:
        suggestions['Buy Calls'] = f"Strike ${bc_strike} ({bc_reason}), Exp ~{bc_exp}"
        
    suggestions['Buy Commons'] = f"Entry: ${close:.2f}. Stop Loss: ${ema21:.2f}"
    
    if "RSI Backtest" in opt_reason:
        reasons.append(f"Hist. Optimal Hold: {opt_days} Days")
        
    if put_whale:
        reasons.append(f"Whale: Sold Puts @ ${put_whale['Strike']}")
    if call_whale:
        reasons.append(f"Whale: Bought Calls @ ${call_whale['Strike']}")
    
    return score, reasons, suggestions

def prepare_data(df):
    # Standardize column names (removes spaces, dashes, converts to UPPER)
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # --- BUILD DAILY ---
    d_rsi = next((c for c in cols if c in ['RSI', 'RSI14'] and 'W_' not in c), 'RSI')
    d_ema8 = next((c for c in cols if c == 'EMA8'), 'EMA8')
    d_ema21 = next((c for c in cols if c == 'EMA21'), 'EMA21')

    needed_cols = [close_col, vol_col, high_col, low_col]
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8 in df.columns: needed_cols.append(d_ema8)
    if d_ema21 in df.columns: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    
    rename_dict = {close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low'}
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8 in df_d.columns: rename_dict[d_ema8] = 'EMA8'
    if d_ema21 in df_d.columns: rename_dict[d_ema21] = 'EMA21'
    
    df_d.rename(columns=rename_dict, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    df_d = add_technicals(df_d)
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # --- BUILD WEEKLY ---
    w_close, w_vol = 'W_CLOSE', 'W_VOLUME'
    w_high, w_low = 'W_HIGH', 'W_LOW'
    
    w_rsi_source = next((c for c in cols if c in ['W_RSI', 'W_RSI14']), None)
    w_ema8_source = next((c for c in cols if c in ['W_EMA8']), None)
    w_ema21_source = next((c for c in cols if c in ['W_EMA21']), None)
    
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low]):
        cols_w = [w_close, w_vol, w_high, w_low]
        if w_rsi_source: cols_w.append(w_rsi_source)
        if w_ema8_source: cols_w.append(w_ema8_source)
        if w_ema21_source: cols_w.append(w_ema21_source)
        
        df_w = df[cols_w].copy()
        
        w_rename = {w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low'}
        if w_rsi_source: w_rename[w_rsi_source] = 'RSI'
        if w_ema8_source: w_rename[w_ema8_source] = 'EMA8'
        if w_ema21_source: w_rename[w_ema21_source] = 'EMA21'
        
        df_w.rename(columns=w_rename, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        
        df_w = add_technicals(df_w)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: 
        df_w = None
        
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe, min_n=0, periods_input=None, optimize_for='PF', lookback_period=90, price_source='High/Low', strict_validation=True, recent_days_filter=25, rsi_diff_threshold=2.0):
    divergences = []
    n_rows = len(df_tf)
    
    if n_rows < lookback_period + 1: return divergences
    
    rsi_vals = df_tf['RSI'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    close_vals = df_tf['Price'].values 
    
    # Select Price Arrays based on User Input
    if price_source == 'Close':
        low_vals = close_vals
        high_vals = close_vals
    else:
        low_vals = df_tf['Low'].values
        high_vals = df_tf['High'].values
        
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': 
             return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    
    # PASS 1: VECTORIZED PRE-CHECK
    roll_low_min = pd.Series(low_vals).shift(1).rolling(window=lookback_period).min().values
    roll_high_max = pd.Series(high_vals).shift(1).rolling(window=lookback_period).max().values
    
    is_new_low = (low_vals < roll_low_min)
    is_new_high = (high_vals > roll_high_max)
    
    valid_mask = np.zeros(n_rows, dtype=bool)
    valid_mask[lookback_period:] = True
    
    candidate_indices = np.where(valid_mask & (is_new_low | is_new_high))[0]
    
    potential_signals = [] 

    # PASS 2: SCAN CANDIDATES
    for i in candidate_indices:
        p2_rsi = rsi_vals[i]
        p2_vol = vol_vals[i]
        p2_volsma = vol_sma_vals[i]
        
        lb_start = i - lookback_period
        lb_rsi = rsi_vals[lb_start:i]
        
        is_vol_high = int(p2_vol > (p2_volsma * 1.5)) if not np.isnan(p2_volsma) else 0
        
        # Bullish Divergence
        if is_new_low[i]:
            p1_idx_rel = np.argmin(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            
            if p2_rsi > (p1_rsi + rsi_diff_threshold):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                
                is_valid_structure = True
                if strict_validation and np.any(subset_rsi > 50):
                    is_valid_structure = False
                
                if is_valid_structure: 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi <= p1_rsi): valid = False
                    
                    if valid:
                        potential_signals.append({"index": i, "type": "Bullish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})
        
        # Bearish Divergence
        elif is_new_high[i]:
            p1_idx_rel = np.argmax(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            
            if p2_rsi < (p1_rsi - rsi_diff_threshold):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                
                is_valid_structure = True
                if strict_validation and np.any(subset_rsi < 50):
                    is_valid_structure = False
                    
                if is_valid_structure: 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi >= p1_rsi): valid = False
                    
                    if valid:
                        potential_signals.append({"index": i, "type": "Bearish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})

    # PASS 3: REPORT & METRICS
    display_threshold_idx = n_rows - recent_days_filter
    
    # Pre-calculate indices for stats
    bullish_indices = [x['index'] for x in potential_signals if x['type'] == 'Bullish']
    bearish_indices = [x['index'] for x in potential_signals if x['type'] == 'Bearish']

    for sig in potential_signals:
        i = sig["index"]
        s_type = sig["type"]
        idx_p1_abs = sig["p1_idx"]
        
        # Values
        price_p1 = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
        price_p2 = low_vals[i] if s_type=='Bullish' else high_vals[i]
        vol_p1 = vol_vals[idx_p1_abs]
        vol_p2 = vol_vals[i]
        rsi_p1 = rsi_vals[idx_p1_abs]
        rsi_p2 = rsi_vals[i]
        date_p1 = get_date_str(idx_p1_abs, '%Y-%m-%d')
        date_p2 = get_date_str(i, '%Y-%m-%d')
        
        is_recent = (i >= display_threshold_idx)

        div_obj = {
            'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe,
            'Signal_Date_ISO': date_p2, 
            'P1_Date_ISO': date_p1,
            'RSI1': rsi_p1, 'RSI2': rsi_p2,
            'Price1': price_p1, 'Price2': price_p2,
            'Day1_Volume': vol_p1, 'Day2_Volume': vol_p2,
            'Is_Recent': is_recent
        }

        # Tags
        tags = []
        latest_row = df_tf.iloc[-1]
        last_price = latest_row['Price']
        last_ema8 = latest_row.get('EMA8') 
        last_ema21 = latest_row.get('EMA21')
        def is_valid(val): return val is not None and not pd.isna(val)

        if s_type == 'Bullish':
            if is_valid(last_ema8) and last_price >= last_ema8: tags.append(f"EMA{EMA8_PERIOD}")
            if is_valid(last_ema21) and last_price >= last_ema21: tags.append(f"EMA{EMA21_PERIOD}")
        else: 
            if is_valid(last_ema8) and last_price <= last_ema8: tags.append(f"EMA{EMA8_PERIOD}")
            if is_valid(last_ema21) and last_price <= last_ema21: tags.append(f"EMA{EMA21_PERIOD}")
            
        if sig["vol_high"]: tags.append("V_HI")
        if vol_vals[i] > vol_vals[idx_p1_abs]: tags.append("V_GROW")
        
        # Display Strings
        date_display = f"{get_date_str(idx_p1_abs, '%b %d')} → {get_date_str(i, '%b %d')}"
        rsi_display = f"{int(round(rsi_p1))} {'↗' if rsi_p2 > rsi_p1 else '↘'} {int(round(rsi_p2))}"
        price_display = f"${price_p1:,.2f} ↗ ${price_p2:,.2f}" if price_p2 > price_p1 else f"${price_p1:,.2f} ↘ ${price_p2:,.2f}"

        # Optimization Stats
        hist_list = bullish_indices if s_type == 'Bullish' else bearish_indices
        best_stats = calculate_optimal_signal_stats(hist_list, close_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if best_stats is None: best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
        
        if best_stats["N"] < min_n: continue

        div_obj.update({
            'Tags': tags, 
            'Date_Display': date_display,
            'RSI_Display': rsi_display,
            'Price_Display': price_display, 
            'Last_Close': f"${latest_row['Price']:,.2f}",
            'N': best_stats['N']
        })

        # --- FORWARD PERFORMANCE & RAW CSV DATA ---
        prefix = "Daily" if timeframe == "Daily" else "Weekly"
        
        if periods_input is not None:
            for p in periods_input:
                future_idx = i + p
                col_price = f"{prefix}_Price_After_{p}"
                col_vol = f"{prefix}_Volume_After_{p}"
                col_ret = f"Ret_{p}"
                
                if future_idx < n_rows:
                    f_price = close_vals[future_idx]
                    div_obj[col_price] = f_price
                    div_obj[col_vol] = vol_vals[future_idx]
                    
                    entry = close_vals[i]
                    # Logic: Long return for Bullish, Short return for Bearish
                    if s_type == 'Bullish':
                        ret_pct = (f_price - entry) / entry
                    else:
                        ret_pct = (entry - f_price) / entry 
                        
                    div_obj[col_ret] = ret_pct * 100
                    
                else:
                    div_obj[col_price] = "n/a"
                    div_obj[col_vol] = "n/a"
                    div_obj[col_ret] = np.nan
        
        divergences.append(div_obj)
            
    return divergences

def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    
    st.markdown("""
        <style>
        .top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; }
        .footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; }
        [data-testid="stDataFrame"] th { font-weight: 900 !important; }
        </style>
        """, unsafe_allow_html=True)
    
    # --- Session State Init ---
    if 'saved_rsi_div_lookback' not in st.session_state: st.session_state.saved_rsi_div_lookback = 90
    if 'saved_rsi_div_source' not in st.session_state: st.session_state.saved_rsi_div_source = "High/Low"
    if 'saved_rsi_div_strict' not in st.session_state: st.session_state.saved_rsi_div_strict = "Yes"
    if 'saved_rsi_div_days_since' not in st.session_state: st.session_state.saved_rsi_div_days_since = 25
    if 'saved_rsi_div_diff' not in st.session_state: st.session_state.saved_rsi_div_diff = 2.0
    
    # History Tab State
    if 'rsi_hist_ticker' not in st.session_state: st.session_state.rsi_hist_ticker = "AMZN"
    if 'rsi_hist_results' not in st.session_state: st.session_state.rsi_hist_results = None
    if 'rsi_hist_last_run_params' not in st.session_state: st.session_state.rsi_hist_last_run_params = {}

    # Percentile Tab State
    if 'saved_rsi_pct_low' not in st.session_state: st.session_state.saved_rsi_pct_low = 10
    if 'saved_rsi_pct_high' not in st.session_state: st.session_state.saved_rsi_pct_high = 90
    if 'saved_rsi_pct_min_n' not in st.session_state: st.session_state.saved_rsi_pct_min_n = 1
    if 'saved_rsi_pct_periods' not in st.session_state: st.session_state.saved_rsi_pct_periods = "5, 21, 63, 126, 252"

    def save_rsi_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
        
    dataset_map = DATA_KEYS_PARQUET
    options = list(dataset_map.keys())
    
    # CSV Hardcoded Defaults (Scanning Periods)
    CSV_PERIODS_DAYS = [5, 21, 63, 126, 252]
    CSV_PERIODS_WEEKS = [4, 13, 26, 52]

    # --- TAB NAMES ---
    tab_div, tab_hist, tab_pct, tab_bot = st.tabs(["📉 Divergences", "📉 Divergences History", "🔢 Percentiles (WIP)", "🤖 RSI Backtester"])

    # --------------------------------------------------------------------------
    # TAB 1: DIVERGENCES (SCANNER)
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
                * **Min RSI Delta**: The minimum required difference between the two RSI pivot points. (e.g., A value of 2 means RSI must rise from 30 to at least 32).
                * **Max Candle Between Pivots**: The maximum allowed distance (in trading bars) between the first pivot and the second pivot.
                * **Strict 50-Cross Invalidation**: If set to "Yes", a signal is considered invalid if the RSI line crossed the 50 centerline at any point between the two pivots (suggesting a trend reset).
                * **Candle Price Methodology**: Defines which price data to use for divergence detection. "High/Low" uses candle wicks (standard), "Close" uses candle bodies.
                """)
            with c_guide2:
                st.markdown("#### 📊 Table Columns")
                st.markdown("""
                * **Ticker**: The stock symbol.
                * **Tags**: 
                    * `EMA8`/`EMA21`: Current price is above/below these moving averages.
                    * `V_HI`: Signal candle volume is >1.5x the 30-day average.
                    * `V_GROW`: Volume on the second pivot was higher than the first.
                * **Day Δ / Week Δ**: The date the second pivot (the signal) was confirmed.
                * **RSI Δ**: The RSI value at the first pivot compared to the second pivot.
                * **Price Δ**: The price at the first pivot compared to the second pivot (Low vs Low for Bullish, High vs High for Bearish).
                * **Last Close**: The most recent available closing price for context.
                """)
        
        if data_option_div:
            try:
                key = dataset_map[data_option_div]
                master = load_parquet_and_clean(key)
                
                if master is not None and not master.empty:
                    target_highlight_daily = None
                    highlight_list_weekly = []
                    
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    if date_col_raw:
                        master[date_col_raw] = pd.to_datetime(master[date_col_raw])
                        max_dt_obj = master[date_col_raw].max()
                        
                        # Daily Highlight: Exactly the last available date
                        target_highlight_daily = max_dt_obj.strftime('%Y-%m-%d')
                        
                        # Weekly Highlight: The Monday of the current data week AND the Monday of the previous week
                        days_to_subtract = max_dt_obj.weekday()
                        current_week_monday = (max_dt_obj - timedelta(days=days_to_subtract))
                        prev_week_monday = current_week_monday - timedelta(days=7)
                        
                        highlight_list_weekly = [
                            current_week_monday.strftime('%Y-%m-%d'),
                            prev_week_monday.strftime('%Y-%m-%d')
                        ]
                    
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    all_tickers = sorted(master[t_col].unique())
                    
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                        st.write(", ".join(all_tickers[:100]) + ("..." if len(all_tickers) > 100 else ""))
                    
                    # --- STREAMLINED INPUTS ---
                    c_d1, c_d2, c_d3, c_d4, c_d5 = st.columns(5)
                    
                    with c_d1:
                        days_since = st.number_input("Days Since Signal", min_value=1, value=st.session_state.saved_rsi_div_days_since, step=1, key="rsi_div_days_since", on_change=save_rsi_state, args=("rsi_div_days_since", "saved_rsi_div_days_since"))
                    with c_d2:
                        div_diff = st.number_input("Min RSI Delta", min_value=0.5, value=st.session_state.saved_rsi_div_diff, step=0.5, key="rsi_div_diff", on_change=save_rsi_state, args=("rsi_div_diff", "saved_rsi_div_diff"))
                    with c_d3:
                        div_lookback = st.number_input("Max Candle Between Pivots", min_value=30, value=st.session_state.saved_rsi_div_lookback, step=5, key="rsi_div_lookback", on_change=save_rsi_state, args=("rsi_div_lookback", "saved_rsi_div_lookback"))
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
                        
                        if d_d is not None: 
                            raw_results_div.extend(find_divergences(d_d, ticker, 'Daily', min_n=0, periods_input=CSV_PERIODS_DAYS, optimize_for='PF', lookback_period=div_lookback, price_source=div_source, strict_validation=strict_div, recent_days_filter=days_since, rsi_diff_threshold=div_diff))
                        if d_w is not None: 
                            raw_results_div.extend(find_divergences(d_w, ticker, 'Weekly', min_n=0, periods_input=CSV_PERIODS_WEEKS, optimize_for='PF', lookback_period=div_lookback, price_source=div_source, strict_validation=strict_div, recent_days_filter=days_since, rsi_diff_threshold=div_diff))
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    
                    progress_bar.empty()
                    
                    if raw_results_div:
                        df_all_results = pd.DataFrame(raw_results_div)
                        
                        # --- DOWNLOAD BUTTON ---
                        csv_export_df = df_all_results.rename(columns={"Signal_Date_ISO": "Day2", "P1_Date_ISO": "Day1"})
                        csv_export_df["Type"] = csv_export_df["Timeframe"] + " " + csv_export_df["Type"]
                        file_label = data_option_div.replace(" ", "_") if data_option_div else "dataset"

                        # Sort Dynamic Columns
                        d_p_cols = [c for c in csv_export_df.columns if c.startswith("Daily_Price")]
                        w_p_cols = [c for c in csv_export_df.columns if c.startswith("Weekly_Price")]
                        d_v_cols = [c for c in csv_export_df.columns if c.startswith("Daily_Vol")]
                        w_v_cols = [c for c in csv_export_df.columns if c.startswith("Weekly_Vol")]
                        
                        for lst in [d_p_cols, w_p_cols, d_v_cols, w_v_cols]:
                            try: lst.sort(key=lambda x: int(x.split('_')[-1]))
                            except: lst.sort()

                        final_cols = d_p_cols + w_p_cols + d_v_cols + w_v_cols
                        for c in final_cols:
                            if c not in csv_export_df.columns: csv_export_df[c] = "n/a"
                            else: csv_export_df[c] = csv_export_df[c].fillna("n/a")

                        base_cols = ["Type", "Ticker", "Day1", "Day2", "RSI1", "RSI2", "Price1", "Price2", "Day1_Volume", "Day2_Volume"]
                        valid_export_cols = [c for c in base_cols + final_cols if c in csv_export_df.columns]
                        
                        csv_data = csv_export_df[valid_export_cols].to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download 10Y Signal History (Raw)", data=csv_data, file_name=f"rsi_div_raw_{file_label}.csv", mime="text/csv")

                        # --- UI DISPLAY ---
                        res_div_df = df_all_results[df_all_results["Is_Recent"] == True].copy()
                        
                        if res_div_df.empty:
                            st.warning(f"No signals found in the last {days_since} days.")
                        else:
                            res_div_df = res_div_df.sort_values(by='Signal_Date_ISO', ascending=False)
                            consolidated = res_div_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                            
                            for tf in ['Daily', 'Weekly']:
                                # Highlight logic: Single Date for Daily, List for Weekly
                                if tf == 'Weekly':
                                    targets = highlight_list_weekly
                                else:
                                    targets = [target_highlight_daily] if target_highlight_daily else []
                                    
                                date_header = "Week Δ" if tf == 'Weekly' else "Day Δ"
                                
                                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                    st.subheader(f"{emoji} {tf} {s_type} Signals")
                                    tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                                    
                                    if div_source == 'Close':
                                        price_header = "Close Price Δ"
                                    else:
                                        price_header = "Low Price Δ" if s_type == 'Bullish' else "High Price Δ"

                                    if not tbl_df.empty:
                                        def style_div_df(df_in):
                                            def highlight_row(row):
                                                styles = [''] * len(row)
                                                # Check if the signal date is in our target list
                                                if row['Signal_Date_ISO'] in targets:
                                                    if 'Date_Display' in df_in.columns:
                                                        idx = df_in.columns.get_loc('Date_Display')
                                                        styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                                return styles
                                            return df_in.style.apply(highlight_row, axis=1)

                                        st.dataframe(
                                            style_div_df(tbl_df),
                                            column_config={
                                                "Ticker": st.column_config.TextColumn("Ticker", width=None),
                                                "Tags": st.column_config.ListColumn("Tags", width="medium"), 
                                                "Date_Display": st.column_config.TextColumn(date_header, width="small"),
                                                "RSI_Display": st.column_config.TextColumn("RSI Δ", width="small"),
                                                "Price_Display": st.column_config.TextColumn(price_header, width="medium"),
                                                "Last_Close": st.column_config.TextColumn("Last Close", width="small"),
                                            },
                                            column_order=["Ticker", "Tags", "Date_Display", "RSI_Display", "Price_Display", "Last_Close"],
                                            hide_index=True,
                                            use_container_width=True,
                                            height=get_table_height(tbl_df, max_rows=50)
                                        )
                                        st.markdown("<br>", unsafe_allow_html=True)
                                    else: st.info("No signals.")
                    else: st.warning("No Divergence signals found.")
                else: st.error(f"Failed to load dataset.")
            except Exception as e: st.error(f"Analysis failed: {e}")

    # --------------------------------------------------------------------------
    # TAB 2: DIV HISTORY (SINGLE TICKER + ANALYSIS)
    # --------------------------------------------------------------------------
    with tab_hist:
        
        with st.expander("ℹ️ Page User Guide"):
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.markdown("#### 🧠 Analysis Logic")
                st.markdown("""
                * **Chains vs Single Signals**: Divergences often trigger multiple times in a row as price continues to grind lower (bullish) or higher (bearish). This tool groups these consecutive signals into a single "Chain" or "Event" originating from the same starting pivot (Date 1).
                * **Best Entry Detection**: The analysis compares the first signal in a chain vs the signal that marked the absolute bottom (or top).
                * **Smart Tips**: Based on historical behavior for *this specific stock*, the tool suggests whether you should buy on the first signal, wait for higher volume, or wait for a wider RSI gap.
                """)
            with c_g2:
                st.markdown("#### 📊 Table Columns")
                st.markdown("""
                * **Date 1**: The date of the first pivot point (the setup).
                * **Date 2**: The date of the second pivot point (the trigger).
                * **RSI 1 / RSI 2**: RSI values at the two pivot points.
                * **Price 1 / Price 2**: Price at the two pivot points.
                * **5d % / 21d % etc**: Percentage return if you held the trade for X days/weeks after the signal date.
                """)

        c_h1, c_h2, c_h3, c_h4, c_h5, c_h6, c_h7 = st.columns(7)
        
        with c_h1:
            hist_ticker_in = st.text_input("Ticker", value=st.session_state.rsi_hist_ticker, key="rsi_hist_ticker_in").strip().upper()
            st.session_state.rsi_hist_ticker = hist_ticker_in
        with c_h2:
            h_lookback = st.number_input("Max Candle Between Pivots", min_value=30, value=90, step=5, key="rsi_hist_lookback")
        with c_h3:
            h_diff = st.number_input("Min RSI Delta", min_value=0.5, value=2.0, step=0.5, key="rsi_hist_diff")
        with c_h4:
            h_strict_str = st.selectbox("Strict 50-Cross Invalidation", ["Yes", "No"], index=0, key="rsi_hist_strict")
            h_strict = (h_strict_str == "Yes")
        with c_h5:
            h_source = st.selectbox("Candle Price Methodology", ["High/Low", "Close"], index=0, key="rsi_hist_source")
        with c_h6:
            h_per_days = st.text_input("Periods (Days)", value="5, 21, 63, 126, 252", key="rsi_hist_p_days")
        with c_h7:
            h_per_weeks = st.text_input("Periods (Weeks)", value="4, 13, 26, 52", key="rsi_hist_p_weeks")

        current_params = {
            "t": hist_ticker_in, "lb": h_lookback, "str": h_strict, "src": h_source, 
            "pd": h_per_days, "pw": h_per_weeks, "diff": h_diff
        }
        
        run_hist = False
        if current_params != st.session_state.rsi_hist_last_run_params:
            run_hist = True
        
        if hist_ticker_in and run_hist:
            with st.spinner(f"Analyzing lifetime history for {hist_ticker_in}..."):
                try:
                    ticker_map = load_ticker_map()
                    df_h = get_ticker_technicals(hist_ticker_in, ticker_map)
                    if df_h is None or df_h.empty: df_h = fetch_yahoo_data(hist_ticker_in)
                    
                    if df_h is not None and not df_h.empty:
                        d_d_h, d_w_h = prepare_data(df_h.copy())
                        if d_w_h is None and d_d_h is not None:
                            temp_w = d_d_h.copy().resample('W').agg({'Price': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'}).dropna()
                            temp_w = add_technicals(temp_w)
                            temp_w['VolSMA'] = temp_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
                            temp_w['ChartDate'] = temp_w.index - pd.to_timedelta(temp_w.index.dayofweek, unit='D')
                            d_w_h = temp_w.dropna(subset=['Price', 'RSI'])
                        
                        raw_results_hist = []
                        p_days_parsed = parse_periods(h_per_days)
                        p_weeks_parsed = parse_periods(h_per_weeks)
                        
                        if d_d_h is not None:
                            raw_results_hist.extend(find_divergences(d_d_h, hist_ticker_in, 'Daily', min_n=0, periods_input=p_days_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff))
                        if d_w_h is not None:
                            raw_results_hist.extend(find_divergences(d_w_h, hist_ticker_in, 'Weekly', min_n=0, periods_input=p_weeks_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff))
                            
                        st.session_state.rsi_hist_results = pd.DataFrame(raw_results_hist)
                        st.session_state.rsi_hist_last_run_params = current_params
                    else:
                        st.error(f"Could not load data for {hist_ticker_in}")
                        st.session_state.rsi_hist_results = pd.DataFrame()
                except Exception as e: st.error(f"Error: {e}")
        
        # --- DISPLAY RESULTS ---
        if st.session_state.rsi_hist_results is not None and not st.session_state.rsi_hist_results.empty:
            res_df_h = st.session_state.rsi_hist_results.copy()
            res_df_h = res_df_h.sort_values(by='Signal_Date_ISO', ascending=False)
            
            # --- DOWNLOAD BUTTON ---
            csv_export_h = res_df_h.rename(columns={"Signal_Date_ISO": "Day2", "P1_Date_ISO": "Day1"})
            csv_export_h["Type"] = csv_export_h["Timeframe"] + " " + csv_export_h["Type"]
            
            d_p_cols = [c for c in csv_export_h.columns if c.startswith("Daily_Price")]
            w_p_cols = [c for c in csv_export_h.columns if c.startswith("Weekly_Price")]
            d_v_cols = [c for c in csv_export_h.columns if c.startswith("Daily_Vol")]
            w_v_cols = [c for c in csv_export_h.columns if c.startswith("Weekly_Vol")]
            for lst in [d_p_cols, w_p_cols, d_v_cols, w_v_cols]:
                try: lst.sort(key=lambda x: int(x.split('_')[-1]))
                except: lst.sort()
            
            final_cols_h = d_p_cols + w_p_cols + d_v_cols + w_v_cols
            for c in final_cols_h:
                if c not in csv_export_h.columns: csv_export_h[c] = "n/a"
                else: csv_export_h[c] = csv_export_h[c].fillna("n/a")
                
            base_cols = ["Type", "Ticker", "Day1", "Day2", "RSI1", "RSI2", "Price1", "Price2", "Day1_Volume", "Day2_Volume"]
            valid_export_cols = [c for c in base_cols + final_cols_h if c in csv_export_h.columns]
            
            csv_data_h = csv_export_h[valid_export_cols].to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Download History ({hist_ticker_in})", data=csv_data_h, file_name=f"rsi_div_history_{hist_ticker_in}.csv", mime="text/csv")
            
            st.divider()

            # --- ANALYSIS LOGIC (Chains & Optimization) ---
            def analyze_chains(df_sub):
                if df_sub.empty: return None
                
                # Group by P1_Date (The Start of the Divergence Chain)
                chains = []
                for p1_date, group in df_sub.groupby('P1_Date_ISO'):
                    group = group.sort_values('Signal_Date_ISO') # Sort chronologically
                    
                    first_sig = group.iloc[0]
                    
                    # Determine Best Entry in the Chain
                    # Bullish = Minimum Price2 (Bottom), Bearish = Maximum Price2 (Top)
                    is_bullish = first_sig['Type'] == 'Bullish'
                    
                    if is_bullish:
                        best_entry_idx = group['Price2'].idxmin()
                    else:
                        best_entry_idx = group['Price2'].idxmax()

                    best_sig = group.loc[best_entry_idx]
                    
                    # Volume Analysis
                    vol_first = first_sig.get('Day2_Volume', 0)
                    vol_best = best_sig.get('Day2_Volume', 0)
                    vol_confirmed = False
                    if vol_best > vol_first:
                         vol_confirmed = True

                    # Metrics for this specific chain
                    duration = (pd.to_datetime(group['Signal_Date_ISO'].max()) - pd.to_datetime(group['Signal_Date_ISO'].min())).days
                    price_drawdown = abs(first_sig['Price2'] - best_sig['Price2'])
                    
                    # Forward Returns of the BEST entry
                    ret_cols = [c for c in group.columns if c.startswith('Ret_')]
                    best_rets = {c: best_sig[c] for c in ret_cols}
                    
                    chains.append({
                        'P1': p1_date,
                        'Duration': duration,
                        'Drawdown_Avoided': price_drawdown,
                        'Is_Chain': len(group) > 1,
                        'First_Is_Best': first_sig.name == best_sig.name,
                        'RSI_Gap_Best': abs(best_sig['RSI2'] - best_sig['RSI1']),
                        'RSI_Gap_First': abs(first_sig['RSI2'] - first_sig['RSI1']),
                        'Vol_Confirmed': vol_confirmed,
                        **best_rets
                    })
                
                return pd.DataFrame(chains)

            # --- TABLE GENERATION ---
            for tf in ['Daily', 'Weekly']:
                p_cols_to_show = []
                current_periods = parse_periods(h_per_days if tf == 'Daily' else h_per_weeks)
                for p in current_periods:
                    col_key = f"Ret_{p}"
                    if col_key in res_df_h.columns: p_cols_to_show.append(col_key)

                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {tf} {s_type} History (Lifetime)")
                    
                    # 1. Filter Data
                    tbl_df = res_df_h[(res_df_h['Type']==s_type) & (res_df_h['Timeframe']==tf)].copy()
                    
                    # 2. Run Analysis & Display Recommendations (Under Header, Before Table)
                    if not tbl_df.empty:
                        chains = analyze_chains(tbl_df)
                        if chains is not None and not chains.empty:
                            count = len(chains)
                            avg_dur = chains['Duration'].mean()
                            
                            # Optimal Hold
                            ret_cols = [c for c in chains.columns if c.startswith('Ret_')]
                            
                            if ret_cols:
                                avg_rets = chains[ret_cols].mean()
                                # FIX: Check for Valid Index before replacing string
                                best_idx = avg_rets.idxmax()
                                if pd.notna(best_idx):
                                    best_period = str(best_idx).replace('Ret_', '')
                                    best_val = avg_rets.max()
                                else:
                                    best_period, best_val = "N/A", 0.0
                            else:
                                best_period, best_val = "N/A", 0.0

                            # Dynamic Tip Logic
                            chains_with_drawdown = chains[chains['Is_Chain']] 
                            
                            tip_str = ""
                            if not chains_with_drawdown.empty:
                                vol_confirm_rate = (chains_with_drawdown['Vol_Confirmed'].sum() / len(chains_with_drawdown)) * 100
                                avg_rsi_gap_first = chains_with_drawdown['RSI_Gap_First'].mean()
                                avg_rsi_gap_best = chains_with_drawdown['RSI_Gap_Best'].mean()
                                first_is_best_rate = (chains['First_Is_Best'].sum() / len(chains)) * 100
                                
                                if first_is_best_rate > 60:
                                     tip_str = f"💡 **Tip**: Don't overthink it. The **First Signal** was the best entry {first_is_best_rate:.0f}% of the time."
                                elif vol_confirm_rate > 60:
                                    tip_str = f"💡 **Tip**: Volume is key. The true bottom/top was accompanied by **Higher Volume** than the initial signal {vol_confirm_rate:.0f}% of the time."
                                elif avg_rsi_gap_best > avg_rsi_gap_first + 1.5:
                                     tip_str = f"💡 **Tip**: Patience pays. Wait for the RSI Gap to widen. (Avg Best Gap: **{avg_rsi_gap_best:.1f}** vs Initial: {avg_rsi_gap_first:.1f})"
                                else:
                                     tip_str = f"💡 **Tip**: Complex action. Chains last avg **{avg_dur:.1f} days**. Consider scaling in."
                            else:
                                tip_str = "💡 **Tip**: Signals tend to be singular events (not chains) for this timeframe."

                            # Display Analysis Block
                            st.markdown(f"**Events**: {count} | **Avg Duration**: {avg_dur:.1f}d | **Optimal Hold**: {best_period}d (+{best_val:.1f}%)")
                            st.info(tip_str)

                        # 3. Display Table
                        def style_ret(df_in):
                            def highlight_val(val):
                                if pd.isna(val): return ''
                                color = '#1e7e34' if val > 0 else '#c5221f'
                                return f'color: {color}; font-weight: bold;'
                            
                            style_obj = df_in.style
                            for p_c in p_cols_to_show:
                                style_obj = style_obj.map(highlight_val, subset=[p_c])
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
                        final_cols = cols_base + p_cols_to_show

                        # Height Calculation: Show ~50 rows
                        st.dataframe(
                            style_ret(tbl_df[final_cols]),
                            column_config=cfg,
                            hide_index=True,
                            use_container_width=True,
                            height=(min(len(tbl_df), 50) + 1) * 35 
                        )
                    else:
                        st.caption("No signals found.")
    
    # --------------------------------------------------------------------------
    # TAB 3: PERCENTILES (WIP)
    # --------------------------------------------------------------------------
    with tab_pct:
        data_option_pct = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_pct_pills")
        
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            pct_low = st.number_input("RSI Low (e.g. 10)", min_value=1, max_value=40, value=st.session_state.saved_rsi_pct_low, step=1, key="rsi_pct_low", on_change=save_rsi_state, args=("rsi_pct_low", "saved_rsi_pct_low"))
        with c_p2:
            pct_high = st.number_input("RSI High (e.g. 90)", min_value=60, max_value=99, value=st.session_state.saved_rsi_pct_high, step=1, key="rsi_pct_high", on_change=save_rsi_state, args=("rsi_pct_high", "saved_rsi_pct_high"))
        with c_p3:
            min_n_pct = st.number_input("Min N", min_value=1, value=st.session_state.saved_rsi_pct_min_n, step=1, key="rsi_pct_min_n", on_change=save_rsi_state, args=("rsi_pct_min_n", "saved_rsi_pct_min_n"))
        with c_p4:
            periods_str_pct = st.text_input("Periods", value=st.session_state.saved_rsi_pct_periods, key="rsi_pct_periods", on_change=save_rsi_state, args=("rsi_pct_periods", "saved_rsi_pct_periods"))
            
        periods_pct = parse_periods(periods_str_pct)
        
        if data_option_pct:
            try:
                key = dataset_map[data_option_pct]
                master = load_parquet_and_clean(key)
                if master is not None and not master.empty:
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
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
                                "Best Period": st.column_config.TextColumn("Best Period"),
                                "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                                "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                                "EV": st.column_config.NumberColumn("EV", format="%.1f%%"),
                                "EV Target": st.column_config.NumberColumn("EV Target", format="$%.2f"),
                                "N": st.column_config.NumberColumn("N"),
                                "SQN": st.column_config.NumberColumn("SQN", format="%.2f")
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=get_table_height(df_pct, max_rows=50)
                        )
                    else:
                        st.info("No percentile signals found.")
                else:
                    st.error("Failed to load data.")
            except Exception as e:
                st.error(f"Error: {e}")

    # --------------------------------------------------------------------------
    # TAB 4: BACKTESTER (Optimized)
    # --------------------------------------------------------------------------
    with tab_bot:
        
        with st.expander("ℹ️ Page User Guide"):
            #col_guide_a, col_guide_b = st.columns(2)
            #with col_guide_a:
            st.markdown("#### 🤔 What is this tool?")
            st.markdown("This tool scans the **entire price history** (e.g., last 10 years) of a stock to find every instance where the RSI matched your criteria. It then 'simulates' a trade to see what would have happened if you bought and held.")
            st.markdown("#### 🎯 Why use it?")
            st.markdown("To validate your trade ideas. If a stock hits RSI 30, does it *actually* bounce historically? Or does it typically keep crashing? This tool gives you the hard data so you don't trade blind.")
            st.markdown("#### 📊 How to Read the Data")
            st.markdown("""
            * **Profit Factor**: The most critical metric. Calculated as `Total Gains / Total Losses`.
                * **< 1.0**: Losing strategy.
                * **> 1.5**: Strong strategy.
                * **> 2.0**: Excellent.
            * **Win Rate**: How often the trade makes money (e.g. 65% = 6.5 wins out of 10).
            * **EV (Expected Value)**: Represents the average profit or loss per trade expressed as a percentage. It's calculated by dividing the total P&L by the number of trades. A positive EV indicates a profitable system on average, while negative EV suggests losses. This metric answers the fundamental question: "How much do I make (or lose) per trade on average?" However, EV alone doesn't account for consistency or risk - a high EV could come from a few large wins masking many small losses.
                * Calculated as `Total P&L / Number of Trades`.                
            * **SQN (System Quality Number)**: Measures the quality and reliability of a trading system by evaluating how consistently it generates returns relative to their variability. It combines the average return, standard deviation, and sample size into a single metric that indicates whether your edge is statistically significant. SQN is essentially a Sharpe Ratio scaled by the square root of the number of trades. Unlike EV, SQN accounts for both the magnitude and consistency of returns.
                * Calculated as `(Average Return / Standard Deviation) x √(Number of Trades)`.
                * **< 1.6**: Hard to trade / Volatile.
                * **1.6 - 3.0**: Good / Average.
                * **> 3.0**: Excellent / Holy Grail.
            * **Count**: Sample size. (Higher is better).
            """)

        c_left, c_right = st.columns([1, 6])
        
        with c_left:
            ticker = st.text_input("Ticker", value="NFLX", help="Enter a symbol (e.g., TSLA, NVDA)", key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=10, value=10)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            
            # --- OPTIMIZATION: Checkbox to enable heavy SQN math ---
            calc_sqn = st.checkbox("Calc SQN", value=False, help="Enable to calculate System Quality Number (slower).")
            
            rsi_metric_container = st.empty()
        
        if ticker:
            ticker_map = load_ticker_map()
            
            with st.spinner(f"Crunching numbers for {ticker}..."):
                df = get_ticker_technicals(ticker, ticker_map)
                
                if df is None or df.empty:
                    df = fetch_yahoo_data(ticker)
                
                if df is None or df.empty:
                    st.error(f"Sorry, data could not be retrieved for {ticker} (neither via Drive nor Yahoo Finance).")
                else:
                    df.columns = [c.strip().upper() for c in df.columns]
                    
                    date_col = next((c for c in df.columns if 'DATE' in c), None)
                    close_col = next((c for c in df.columns if 'CLOSE' in c), None)
                    rsi_priority = ['RSI14', 'RSI', 'RSI_14']
                    rsi_col = next((c for c in rsi_priority if c in df.columns), None)
                    
                    if not rsi_col:
                        rsi_col = next((c for c in df.columns if 'RSI' in c and 'W_' not in c), None)

                    if not all([date_col, close_col]):
                        st.error("Data source missing Date or Close columns.")
                    else:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.sort_values(by=date_col).reset_index(drop=True)

                        if not rsi_col:
                            delta = df[close_col].diff()
                            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                            rs = gain / loss
                            df['RSI'] = 100 - (100 / (1 + rs))
                            rsi_col = 'RSI'

                        cutoff_date = df[date_col].max() - timedelta(days=365*lookback_years)
                        df = df[df[date_col] >= cutoff_date].copy().reset_index(drop=True) 

                        current_row = df.iloc[-1]
                        current_rsi = current_row[rsi_col]
                        
                        rsi_metric_container.markdown(f"""<div style="margin-top: 10px; font-size: 0.9rem; color: #666;">Current RSI</div><div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 15px;">{current_rsi:.2f}</div>""", unsafe_allow_html=True)
                        
                        rsi_min = current_rsi - rsi_tol
                        rsi_max = current_rsi + rsi_tol
                        
                        hist_df = df.iloc[:-1].copy()
                        matches = hist_df[(hist_df[rsi_col] >= rsi_min) & (hist_df[rsi_col] <= rsi_max)].copy()
                        
                        full_close = df[close_col].values
                        match_indices = matches.index.values
                        total_len = len(full_close)

                        results = []
                        periods = [1, 5, 10, 21, 42, 63, 126, 252]
                        
                        for p in periods:
                            valid_indices = match_indices[match_indices + p < total_len]
                            
                            if len(valid_indices) == 0:
                                results.append({"Days": p, "Win Rate": np.nan, "EV": np.nan, "Count": 0, "Profit Factor": np.nan, "SQN": 0.0})
                                continue
                                
                            entry_prices = full_close[valid_indices]
                            exit_prices = full_close[valid_indices + p]
                            
                            returns = (exit_prices - entry_prices) / entry_prices
                            
                            wins = returns[returns > 0]
                            losses = returns[returns < 0]
                            gross_win = np.sum(wins)
                            gross_loss = np.abs(np.sum(losses))
                            
                            pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
                            
                            win_rate = np.mean(returns > 0) * 100
                            avg_ret = np.mean(returns) * 100
                            count = len(valid_indices)

                            # --- SQN CALCULATION (OPTIMIZED) ---
                            sqn = 0.0
                            if calc_sqn:
                                std_dev = np.std(returns) * 100
                                if std_dev > 0 and count > 0:
                                    sqn = (avg_ret / std_dev) * np.sqrt(count)
                            
                            results.append({
                                "Days": p, 
                                "Profit Factor": pf, 
                                "Win Rate": win_rate, 
                                "EV": avg_ret, 
                                "SQN": sqn,
                                "Count": count
                            })

                        res_df = pd.DataFrame(results)

                        with c_right:
                            if matches.empty:
                                st.warning(f"No historical periods found where RSI was between {rsi_min:.2f} and {rsi_max:.2f}.")
                            else:
                                def highlight_best(row):
                                    days = row['Days']
                                    if days <= 20: threshold = 30
                                    elif days <= 60: threshold = 20
                                    else: threshold = 10
                                    
                                    condition = (row['Count'] >= threshold) and (row['Win Rate'] > 75)
                                    color = 'background-color: rgba(144, 238, 144, 0.2)' if condition else ''
                                    return [color] * len(row)

                                def highlight_ret(val):
                                    if val is None or pd.isna(val): return ''
                                    if not isinstance(val, (int, float)): return ''
                                    color = '#71d28a' if val > 0 else '#f29ca0'
                                    return f'color: {color}; font-weight: bold;'
                                
                                format_func = lambda x: f"{x:+.2f}%" if pd.notnull(x) else "—"
                                format_wr = lambda x: f"{x:.1f}%" if pd.notnull(x) else "—"
                                format_pf = lambda x: f"{x:.2f}" if pd.notnull(x) else "—"
                                format_sqn = lambda x: f"{x:.1f}" if pd.notnull(x) else "—"

                                st.dataframe(
                                    res_df.style
                                    .format({"Win Rate": format_wr, "EV": format_func, "Profit Factor": format_pf, "SQN": format_sqn})
                                    .map(highlight_ret, subset=["EV"])
                                    .apply(highlight_best, axis=1)
                                    .set_table_styles([dict(selector="th", props=[("font-weight", "bold"), ("background-color", "#f0f2f6")])]),
                                    use_container_width=False, 
                                    column_config={
                                        "Days": st.column_config.NumberColumn("Days"),
                                        "Profit Factor": st.column_config.NumberColumn("Profit Factor"),
                                        "Win Rate": st.column_config.TextColumn("Win Rate"),
                                        "EV": st.column_config.TextColumn("EV"),
                                        "SQN": st.column_config.NumberColumn("SQN"),
                                        "Count": st.column_config.NumberColumn("Count")
                                    },
                                    hide_index=True,
                                    height=get_table_height(res_df, max_rows=50)
                                )

                        st.markdown("<br><br><br>", unsafe_allow_html=True)

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, min_n=1, filter_date=None, timeframe='Daily', periods_input=None, optimize_for='SQN'):
    signals = []
    if len(df) < 200: return signals
    
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(pct_low)
    p90 = hist_df['RSI'].quantile(pct_high)
    
    rsi_series = hist_df['RSI']
    rsi_vals = rsi_series.values 
    price_vals = hist_df['Price'].values
    
    # 1. Identify ALL Signal Indices
    prev_rsi = rsi_series.shift(1)
    
    bull_mask = (prev_rsi < p10) & (rsi_series >= (p10 + 1.0))
    bear_mask = (prev_rsi > p90) & (rsi_series <= (p90 - 1.0))
    
    bullish_signal_indices = np.where(bull_mask)[0].tolist()
    bearish_signal_indices = np.where(bear_mask)[0].tolist()
            
    # 2. Filter and Optimize
    latest_close = df['Price'].iloc[-1] 
    all_indices = sorted(bullish_signal_indices + bearish_signal_indices)
    
    for i in all_indices:
        curr_row = hist_df.iloc[i]
        curr_date = curr_row.name.date()
        
        if filter_date and curr_date < filter_date:
            continue
            
        is_bullish = i in bullish_signal_indices
        s_type = 'Bullish' if is_bullish else 'Bearish'
        thresh_val = p10 if is_bullish else p90
        curr_rsi_val = rsi_vals[i]
        
        hist_list = bullish_signal_indices if is_bullish else bearish_signal_indices
        best_stats = calculate_optimal_signal_stats(hist_list, price_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0, "SQN": 0.0}
             
        if best_stats["N"] < min_n:
            continue
            
        rsi_disp = f"{thresh_val:.0f} ↗ {curr_rsi_val:.0f}" if is_bullish else f"{thresh_val:.0f} ↘ {curr_rsi_val:.0f}"
        action_str = "Leaving Low" if is_bullish else "Leaving High"
        
        # --- NEW EV TARGET CALCULATION ---
        ev_val = best_stats['EV']
        sig_close = curr_row['Price']
        
        # New Rule: If N=0, EV Target is 0
        if best_stats['N'] == 0:
            ev_price = 0.0
        else:
            if is_bullish:
                ev_price = sig_close * (1 + (ev_val / 100.0))
            else:
                ev_price = sig_close * (1 - (ev_val / 100.0))

        signals.append({
            'Ticker': ticker,
            'Date': curr_row.name.strftime('%b %d'),
            'Date_Obj': curr_date,
            'Action': action_str,
            'RSI_Display': rsi_disp,
            'Signal_Price': f"${sig_close:,.2f}",
            'Last_Close': f"${latest_close:,.2f}", 
            'Signal_Type': s_type,
            'Best Period': best_stats['Best Period'],
            'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'],
            'EV': best_stats['EV'],
            'EV Target': ev_price,
            'N': best_stats['N'],
            'SQN': best_stats.get('SQN', 0.0)
        })
            
    return signals

@st.cache_data(ttl=3600)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        
        df = df.reset_index()
        date_col_name = df.columns[0]
        df = df.rename(columns={date_col_name: "DATE"})
        
        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"])
        if df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
            
        df = df.rename(columns={"Close": "CLOSE", "Volume": "VOLUME", "High": "HIGH", "Low": "LOW", "Open": "OPEN"})
        df.columns = [c.upper() for c in df.columns]
        
        # Centralized Calc
        df = add_technicals(df)
        
        return df
    except Exception:
        return None

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

@st.cache_data(ttl=600, show_spinner="Crunching Smart Money Data...")
def calculate_smart_money_score(df, start_d, end_d, mc_thresh, filter_ema, limit):
    f = df.copy()
    if start_d: f = f[f["Trade Date"].dt.date >= start_d]
    if end_d: f = f[f["Trade Date"].dt.date <= end_d]
    
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f_filtered = f[f[order_type_col].isin(target_types)].copy()
    
    if f_filtered.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    f_filtered["Signed_Dollars"] = np.where(
        f_filtered[order_type_col].isin(["Calls Bought", "Puts Sold"]), 
        f_filtered["Dollars"], -f_filtered["Dollars"]
    )
    
    smart_stats = f_filtered.groupby("Symbol").agg(
        Signed_Dollars=("Signed_Dollars", "sum"),
        Trade_Count=("Symbol", "count"),
        Last_Trade=("Trade Date", "max")
    ).reset_index()
    
    smart_stats.rename(columns={"Signed_Dollars": "Net Sentiment ($)"}, inplace=True)
    
    unique_tickers = smart_stats["Symbol"].unique().tolist()
    batch_caps = fetch_market_caps_batch(unique_tickers)
    smart_stats["Market Cap"] = smart_stats["Symbol"].map(batch_caps)
    
    valid_data = smart_stats[smart_stats["Market Cap"] >= mc_thresh].copy()
    
    unique_dates = sorted(f_filtered["Trade Date"].unique())
    recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
    f_momentum = f_filtered[f_filtered["Trade Date"].isin(recent_dates)]
    mom_stats = f_momentum.groupby("Symbol")["Signed_Dollars"].sum().reset_index()
    mom_stats.rename(columns={"Signed_Dollars": "Momentum ($)"}, inplace=True)
    
    valid_data = valid_data.merge(mom_stats, on="Symbol", how="left").fillna(0)
    
    top_bulls = pd.DataFrame()
    top_bears = pd.DataFrame()

    if not valid_data.empty:
        valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
        
        def normalize(series):
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn) if (mx != mn) else 0

        # Base Scores
        b_flow_norm = normalize(valid_data["Net Sentiment ($)"].clip(lower=0))
        b_imp_norm = normalize(valid_data["Impact"].clip(lower=0))
        b_mom_norm = normalize(valid_data["Momentum ($)"].clip(lower=0))
        valid_data["Base_Score_Bull"] = (0.35 * b_flow_norm) + (0.30 * b_imp_norm) + (0.35 * b_mom_norm)
        
        br_flow_norm = normalize(-valid_data["Net Sentiment ($)"].clip(upper=0))
        br_imp_norm = normalize(-valid_data["Impact"].clip(upper=0))
        br_mom_norm = normalize(-valid_data["Momentum ($)"].clip(upper=0))
        valid_data["Base_Score_Bear"] = (0.35 * br_flow_norm) + (0.30 * br_imp_norm) + (0.35 * br_mom_norm)
        
        valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")
        
        candidates_bull = valid_data.sort_values(by="Base_Score_Bull", ascending=False).head(limit * 3).copy()
        candidates_bear = valid_data.sort_values(by="Base_Score_Bear", ascending=False).head(limit * 3).copy()
        
        all_tickers_to_fetch = set(candidates_bull["Symbol"]).union(set(candidates_bear["Symbol"]))
        batch_techs = fetch_technicals_batch(list(all_tickers_to_fetch)) if filter_ema else {}

        def check_ema_filter_fast(ticker, mode="Bull"):
            if not filter_ema: return True, "—"
            s, e8, _, _, _ = batch_techs.get(ticker, (None, None, None, None, None))
            if not s or not e8: return False, "—"
            if mode == "Bull":
                return (s > e8), ("✅ >EMA8" if s > e8 else "⚠️ <EMA8")
            else:
                return (s < e8), ("✅ <EMA8" if s < e8 else "⚠️ >EMA8")
        
        bull_results = []
        for idx, row in candidates_bull.iterrows():
            passes, trend_s = check_ema_filter_fast(row["Symbol"], "Bull")
            if passes:
                row["Score"] = row["Base_Score_Bull"] * 100
                row["Trend"] = trend_s
                bull_results.append(row)
        top_bulls = pd.DataFrame(bull_results).head(limit)
        
        bear_results = []
        for idx, row in candidates_bear.iterrows():
            passes, trend_s = check_ema_filter_fast(row["Symbol"], "Bear")
            if passes:
                row["Score"] = row["Base_Score_Bear"] * 100
                row["Trend"] = trend_s
                bear_results.append(row)
        top_bears = pd.DataFrame(bear_results).head(limit)
        
    return top_bulls, top_bears, valid_data

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

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_history_optimized(ticker_sym, t_map):
    """
    Optimized fetcher with multi-level caching and strict timeouts.
    Priority: Parquet (Drive) -> CSV (Drive) -> Yahoo Finance.
    """
    # 1. Try Parquet from Drive (Highest Priority)
    pq_key = f"{ticker_sym}_PARQUET"
    if pq_key in t_map:
        try:
            file_id = t_map[pq_key]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            buffer = get_gdrive_binary_data(url) 
            if buffer:
                df = pd.read_parquet(buffer, engine='pyarrow')
                return df.reset_index() if 'DATE' in str(df.index.name).upper() else df
        except Exception:
            pass 

    # 2. Try CSV from Drive (Medium Priority)
    if ticker_sym in t_map:
        try:
            df = get_ticker_technicals(ticker_sym, t_map)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    # 3. Fallback to Yahoo Finance (Lowest Priority)
    try:
        return fetch_yahoo_data(ticker_sym)
    except Exception:
        return None

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
    
    # --- Helper: Optimized Data Fetching (Parquet > CSV > Yahoo) ---
    def fetch_history_optimized(ticker_sym, t_map):
        pq_key = f"{ticker_sym}_PARQUET"
        if pq_key in t_map:
            try:
                file_id = t_map[pq_key]
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                buffer = get_gdrive_binary_data(url)
                if buffer:
                    df = pd.read_parquet(buffer, engine='pyarrow')
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                    elif df.index.name and 'DATE' in df.index.name.upper():
                        df = df.reset_index()
                    elif 'Date' not in df.columns and 'DATE' not in df.columns:
                        df = df.reset_index()
                    return df
            except Exception:
                pass 
        if ticker_sym in t_map:
            return get_ticker_technicals(ticker_sym, t_map)
        return fetch_yahoo_data(ticker_sym)

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
    st.markdown("ℹ️ Historical Close Prices sourced from Yahoo Finance")

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
**1. Key Metrics Tracked**
The app calculates the percentage distance (the "Gap") between the current price and five different moving averages:
* 8-day and 21-day EMA: Short-term momentum and "swing" levels.
* 50-day, 100-day, and 200-day SMA: Medium to long-term trend baselines.

**2. The "Rubber Band" Logic (Percentiles)**
Rather than just showing the current gap, the app looks at 10 years of history for that specific ticker to see how rare the current gap is. It calculates:
* **p50 (Median):** The typical distance from the average.
* **p70/p80 (Uptrend):** These levels generally occur in strong uptrends.
* **p90/p95 (Extremes):** The levels reached only 10% or 5% of the time historically.

**3. Visual Highlighting System**
The table uses a traffic-light system to categorize the current price action:
* 🟢 **Buy Zone (Green):** Triggered if the Gap is ≤ p50 (Median) AND price is > 8-EMA. Suggests a "pullback to the mean" in an uptrend.
* 🟡 **Warning Zone (Yellow):** Triggered if the gap is between p50 and p90. Price is extending but not yet extreme.
* 🔴 **Sell/Trim Zone (Red):** Triggered if the gap is $\ge$ p90. Price is statistically over-extended.
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
    st.subheader("🔥 Combo Over-Extension Signals")

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
        return ['font-weight: bold;' if row['raw_status'] else ''] * len(row)

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
    chart_data = chart_data[chart_data['Date'] >= (chart_data['Date'].max() - timedelta(days=730))]

    # Base bar chart
    bars = alt.Chart(chart_data).mark_bar().encode(
        x='Date:T', 
        y=alt.Y('Distance (%)', title='% Dist from 50 SMA'),
        color=alt.condition(alt.datum['Distance (%)'] > 0, alt.value("#1e7e34"), alt.value("#c5221f"))
    )

    # Horizontal Rule representing Current % Distance
    rule = alt.Chart(pd.DataFrame({'y': [current_dist_50]})).mark_rule(
        color='#333', 
        strokeDash=[5, 5], 
        strokeWidth=2
    ).encode(y='y:Q')

    # Combined Chart
    final_chart = (bars + rule).properties(height=300)
    st.altair_chart(final_chart, use_container_width=True)

# --- CSS STYLING ---
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

# --- APP INITIALIZATION & NAVIGATION ---
try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)
    
    # 1. Database Date (from Google Sheet)
    db_date = df_global["Trade Date"].max().strftime("%d %b %y")
    
    # 2. Price History Date (fetch AAPL as proxy for latest market data)
    price_date = "Syncing..."
    try:
        df_aapl = fetch_yahoo_data("AAPL")
        if df_aapl is not None and not df_aapl.empty:
            price_date = pd.to_datetime(df_aapl['DATE']).max().strftime("%d %b %y")
        else:
            price_date = "Offline"
    except Exception:
        price_date = "Offline"

    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: run_rsi_scanner_app(df_global), title="RSI Scanner", icon="📈", url_path="rsi_scanner"),
        st.Page(lambda: run_seasonality_app(df_global), title="Seasonality", icon="📅", url_path="seasonality"),
        st.Page(lambda: run_ema_distance_app(df_global), title="EMA Distance", icon="📏", url_path="ema_distance"),
    ])

    st.sidebar.caption("🖥️ Everything is best viewed with a wide desktop monitor in light mode.")
    st.sidebar.caption(f"💾 **Database:** {db_date}")
    st.sidebar.caption(f"📈 **Price History:** {price_date}")
    
    pg.run()
    
    # Global padding at the bottom of the page
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
