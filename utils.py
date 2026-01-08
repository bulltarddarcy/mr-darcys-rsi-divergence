# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import requests
import re
import time
from scipy.signal import argrelextrema
from datetime import date, timedelta, datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# --- DATA LOADERS & GOOGLE DRIVE UTILS ---

def get_gdrive_binary_data(url):
    """
    Robust Google Drive downloader.
    Automatically bypasses 'Virus Scan' warnings by extracting the confirmation token from the HTML.
    """
    try:
        # 1. Extract ID
        match = re.search(r'/d/([a-zA-Z0-9_-]{25,})', url)
        if not match:
            match = re.search(r'id=([a-zA-Z0-9_-]{25,})', url)
            
        if not match:
            st.error(f"Invalid Google Drive URL: {url}")
            return None
            
        file_id = match.group(1)
        download_url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        
        # 2. First Attempt
        response = session.get(download_url, params={'id': file_id}, stream=True, timeout=60)
        
        # 3. Check for "Virus Scan" HTML Page
        # If we get HTML instead of binary data, we need to find the 'confirm' token
        if "text/html" in response.headers.get("Content-Type", "").lower():
            content = response.text
            
            # Look for the 'confirm=...' pattern in the HTML (often in the 'Download anyway' link)
            # It usually looks like: &confirm=xxxx or ?confirm=xxxx
            token_match = re.search(r'confirm=([a-zA-Z0-9_]+)', content)
            
            if token_match:
                token = token_match.group(1)
                # Retry with the confirmation token
                params = {'id': file_id, 'confirm': token}
                response = session.get(download_url, params=params, stream=True, timeout=60)
            else:
                # If we can't find a token in the HTML, check cookies (older method)
                for key, value in session.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'confirm': value}
                        response = session.get(download_url, params=params, stream=True, timeout=60)
                        break

        # 4. Final Validation
        if response.status_code == 200:
            # Check the first chunk to ensure it's not still HTML
            try:
                chunk = next(response.iter_content(chunk_size=100), b"")
                if chunk.strip().startswith(b"<!DOCTYPE"):
                    st.error(f"Download Error: Google Drive returned an HTML warning page for {file_id}. The file might be too large or restricted.")
                    return None
                return BytesIO(chunk + response.raw.read())
            except StopIteration:
                return None
                
        st.error(f"Download Error: HTTP {response.status_code}")
        return None

    except Exception as e:
        st.error(f"Download Exception: {e}")
        return None

@st.cache_data(ttl=3600)
def get_parquet_config():
    """
    Bulletproof loader for dataset configuration.
    Uses PARQUET_CONFIG in Streamlit Secrets as the single source of truth.
    """
    config = {}
    
    try:
        # 1. Fetch the raw configuration string from Secrets
        raw_config = st.secrets.get("PARQUET_CONFIG", "")
        
        if not raw_config:
            st.error("⛔ CRITICAL ERROR: 'PARQUET_CONFIG' not found in your Streamlit Secrets.")
            st.stop()
            
        # 2. Parse lines and clean whitespaces/empty lines
        lines = [line.strip() for line in raw_config.strip().split('\n') if line.strip()]
        
        for line in lines:
            # Split by comma and ensure we have at least 2 parts
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                display_name = parts[0]
                secret_key = parts[1]
                
                # Verify that the secret_key actually exists in st.secrets
                if secret_key in st.secrets:
                    config[display_name] = secret_key
                else:
                    # Helpful warning in logs if a specific key is missing
                    print(f"Warning: Secret key '{secret_key}' for dataset '{display_name}' is missing.")

    except Exception as e:
        st.error(f"Failed to parse PARQUET_CONFIG: {e}")
    
    if not config:
        st.error("⛔ CRITICAL ERROR: No valid datasets could be mapped. Check your Secrets formatting.")
        st.stop()
        
    return config


@st.cache_data(ttl=3600, show_spinner="Loading Dataset...")
def load_parquet_and_clean(key):
    # Ensure key has no hidden whitespace
    clean_key = key.strip()
    
    if clean_key not in st.secrets:
        st.error(f"Secret Key '{clean_key}' not found in secrets.toml")
        return None
        
    url = st.secrets[clean_key]
    
    try:
        buffer = get_gdrive_binary_data(url)
        # If buffer is None, the downloader has already printed the error.
        if not buffer:
            return None
            
        content = buffer.getvalue()
        
        # Try Parquet, Fallback to CSV
        try:
            df = pd.read_parquet(BytesIO(content))
            
            # --- CRITICAL FIX: Check for Date in Index ---
            # If Date is the index (common in financial data), move it to a column.
            if isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
            elif df.index.name and 'DATE' in str(df.index.name).upper():
                df.reset_index(inplace=True)
                
        except Exception:
            try:
                df = pd.read_csv(BytesIO(content))
            except Exception as e:
                st.error(f"Data format error for {clean_key} (Not valid Parquet or CSV): {e}")
                return None

        # --- Standard Cleaning Logic ---
        df.columns = [str(c).strip() for c in df.columns]
        
        # 1. Find Date Column (Case Insensitive)
        date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
        
        # Fallback: Check if 'index' column exists and holds dates
        if not date_col and 'index' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['index']) or pd.api.types.is_object_dtype(df['index']):
                 date_col = 'index'

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.rename(columns={date_col: 'ChartDate'}).sort_values('ChartDate')
        else:
            # If we still can't find a date column, print the columns to help debug
            st.error(f"Error {clean_key}: Could not identify Date column. Found: {list(df.columns)}")
            return None
        
        # 2. Find Price/Close Column
        if 'Price' not in df.columns:
            close_col = next((c for c in df.columns if c.upper() == 'CLOSE'), None)
            if close_col:
                df['Price'] = df[close_col]
            
        return df
    except Exception as e:
        st.error(f"Fatal error processing {clean_key}: {e}")
        return None


@st.cache_data(ttl=3600)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP")
        if not url: return {}

        # FIXED: Use the function that actually exists in your script
        buffer = get_gdrive_binary_data(url)
        
        # Note: get_gdrive_binary_data returns None on failure, not "HTML_ERROR"
        if buffer:
            df = pd.read_csv(buffer)
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                              df.iloc[:, 1].astype(str).str.strip()))
    except Exception as e:
        # Optional: Print the error to see it in logs next time
        print(f"Error loading ticker map: {e}")
        pass
    return {}

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

# --- MATH & TECHNICAL ANALYSIS ---

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

# --- HELPERS ---

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

def find_divergences(df, ticker, timeframe, min_n=1, periods_input=None, lookback_period=60, 
                     price_source='High/Low', strict_validation=True, recent_days_filter=30, 
                     rsi_diff_threshold=1.0, optimize_for='PF', mode="Standard (Reversal)"):
    """
    Scans for RSI Divergences (Standard and Hidden).
    mode: "Standard (Reversal)", "Hidden (Continuation)", or "Both"
    """
    if df is None or len(df) < lookback_period:
        return []

    # --- 1. ROBUST COLUMN DETECTION ---
    # Create a map of {UPPERCASE_NAME: actual_name}
    # This allows us to find 'CLOSE', 'Close', 'close' without renaming
    col_map = {c.strip().upper(): c for c in df.columns}
    
    # Identify the actual column names in this specific DataFrame
    close_col = col_map.get('CLOSE', col_map.get('ADJ CLOSE'))
    high_col  = col_map.get('HIGH', close_col) # Fallback to Close if High missing
    low_col   = col_map.get('LOW', close_col)  # Fallback to Close if Low missing
    rsi_col   = col_map.get('RSI')
    date_col  = col_map.get('DATE')

    # Vital Check: If we can't find Close or RSI, we can't run analysis
    if not close_col or not rsi_col:
        return []

    # --- 2. SETUP DATA ARRAYS ---
    # Use the detected column names to extract data
    if price_source == 'Close':
        price_high = df[close_col].values
        price_low  = df[close_col].values
    else:
        price_high = df[high_col].values
        price_low  = df[low_col].values

    rsi = df[rsi_col].values
    
    # Handle Dates (Use index if Date col missing)
    if date_col:
        dates = df[date_col].values
    else:
        dates = df.index.values

    # --- 3. FIND PEAKS/VALLEYS ---
    safe_order = max(1, int(min_n))
    peaks = argrelextrema(rsi, np.greater, order=safe_order)[0]
    valleys = argrelextrema(rsi, np.less, order=safe_order)[0]
    
    results = []
    
    # Helper: Strict 50-Cross Check
    def check_strict(idx_start, idx_end, is_bullish):
        if idx_end - idx_start <= 1: return True
        mid_rsi = rsi[idx_start+1 : idx_end]
        if len(mid_rsi) == 0: return True
        
        if is_bullish: return not (mid_rsi > 50).any()
        else: return not (mid_rsi < 50).any()

    # --- 4. SCAN BULLISH DIVERGENCES (VALLEYS) ---
    if len(valleys) >= 2:
        for i in range(len(valleys)-1, 0, -1):
            idx_curr = valleys[i]
            
            for j in range(i-1, -1, -1):
                idx_prev = valleys[j]
                if (idx_curr - idx_prev) > lookback_period: break 
                
                rsi_curr = rsi[idx_curr]; rsi_prev = rsi[idx_prev]
                p_curr = price_low[idx_curr]; p_prev = price_low[idx_prev]
                
                if abs(rsi_curr - rsi_prev) < rsi_diff_threshold: continue
                    
                div_type_found = None
                
                # A) STANDARD BULLISH
                if p_curr < p_prev and rsi_curr > rsi_prev:
                    if "Standard" in mode or mode == "Both":
                        div_type_found = "Bullish (Standard)"

                # B) HIDDEN BULLISH
                elif p_curr > p_prev and rsi_curr < rsi_prev:
                    if "Hidden" in mode or mode == "Both":
                        div_type_found = "Bullish (Hidden)"
                
                if div_type_found:
                    if strict_validation and "Standard" in div_type_found:
                        if not check_strict(idx_prev, idx_curr, is_bullish=True):
                            continue
                    
                    # FIX: Use 'close_col' variable instead of hardcoded 'Close'
                    last_close_val = df[close_col].iloc[-1]
                    
                    results.append({
                        "Ticker": ticker,
                        "Type": div_type_found,
                        "Timeframe": timeframe,
                        "P1_Date_ISO": pd.to_datetime(dates[idx_prev]).strftime('%Y-%m-%d'),
                        "Signal_Date_ISO": pd.to_datetime(dates[idx_curr]).strftime('%Y-%m-%d'),
                        "Date_Display": f"{(pd.to_datetime(dates[idx_curr]) - pd.to_datetime(dates[idx_prev])).days}d",
                        "Price1": p_prev, "Price2": p_curr,
                        "RSI1": rsi_prev, "RSI2": rsi_curr,
                        "RSI_Display": f"{rsi_prev:.0f} → {rsi_curr:.0f}",
                        "Price_Display": f"${p_prev:.2f} → ${p_curr:.2f}",
                        "Last_Close": f"${last_close_val:.2f}", # Safe reference
                        "Is_Recent": (len(df) - idx_curr) <= recent_days_filter,
                        "Idx_Signal": idx_curr 
                    })
                    break 

    # --- 5. SCAN BEARISH DIVERGENCES (PEAKS) ---
    if len(peaks) >= 2:
        for i in range(len(peaks)-1, 0, -1):
            idx_curr = peaks[i]
            
            for j in range(i-1, -1, -1):
                idx_prev = peaks[j]
                if (idx_curr - idx_prev) > lookback_period: break
                
                rsi_curr = rsi[idx_curr]; rsi_prev = rsi[idx_prev]
                p_curr = price_high[idx_curr]; p_prev = price_high[idx_prev]
                
                if abs(rsi_curr - rsi_prev) < rsi_diff_threshold: continue

                div_type_found = None

                # A) STANDARD BEARISH
                if p_curr > p_prev and rsi_curr < rsi_prev:
                    if "Standard" in mode or mode == "Both":
                        div_type_found = "Bearish (Standard)"

                # B) HIDDEN BEARISH
                elif p_curr < p_prev and rsi_curr > rsi_prev:
                    if "Hidden" in mode or mode == "Both":
                        div_type_found = "Bearish (Hidden)"
                
                if div_type_found:
                    if strict_validation and "Standard" in div_type_found:
                        if not check_strict(idx_prev, idx_curr, is_bullish=False):
                            continue

                    # FIX: Use 'close_col' variable instead of hardcoded 'Close'
                    last_close_val = df[close_col].iloc[-1]

                    results.append({
                        "Ticker": ticker,
                        "Type": div_type_found,
                        "Timeframe": timeframe,
                        "P1_Date_ISO": pd.to_datetime(dates[idx_prev]).strftime('%Y-%m-%d'),
                        "Signal_Date_ISO": pd.to_datetime(dates[idx_curr]).strftime('%Y-%m-%d'),
                        "Date_Display": f"{(pd.to_datetime(dates[idx_curr]) - pd.to_datetime(dates[idx_prev])).days}d",
                        "Price1": p_prev, "Price2": p_curr,
                        "RSI1": rsi_prev, "RSI2": rsi_curr,
                        "RSI_Display": f"{rsi_prev:.0f} → {rsi_curr:.0f}",
                        "Price_Display": f"${p_prev:.2f} → ${p_curr:.2f}",
                        "Last_Close": f"${last_close_val:.2f}", # Safe reference
                        "Is_Recent": (len(df) - idx_curr) <= recent_days_filter,
                        "Idx_Signal": idx_curr
                    })
                    break

    # --- 6. CALCULATE FORWARD RETURNS ---
    if periods_input and results:
        # FIX: Use 'close_col' variable to get array
        closes = df[close_col].values
        max_idx = len(closes) - 1
        for res in results:
            idx = res['Idx_Signal']
            for p in periods_input:
                try:
                    days = int(p) 
                    if idx + days <= max_idx:
                        ret = (closes[idx + days] - closes[idx]) / closes[idx] * 100
                        res[f"Ret_{p}"] = ret
                    else:
                        res[f"Ret_{p}"] = None
                except:
                    res[f"Ret_{p}"] = None

    return results

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
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping:
        return None
    
    file_id = mapping[ticker]
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # FIXED: Changed from 'get_confirmed_gdrive_data' to 'get_gdrive_binary_data'
    buffer = get_gdrive_binary_data(file_url)
    
    if buffer:
        try:
            df = pd.read_csv(buffer)
            
            # 1. Clean Column Names
            df.columns = [c.strip().upper() for c in df.columns]
            
            # 2. Ensure we can identify the Date column (Column A or named "DATE")
            # If the first column is named something weird, we rename it to DATE
            if "DATE" not in df.columns:
                first_col = df.columns[0]
                df.rename(columns={first_col: "DATE"}, inplace=True)

            return df
        except Exception as e:
            print(f"Error parsing CSV for {ticker}: {e}")
            return None
    return None

