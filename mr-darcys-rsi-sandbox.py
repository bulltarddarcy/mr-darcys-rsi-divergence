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
from io import StringIO, BytesIO
import altair as alt
import google.generativeai as genai
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
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21
EV_LOOKBACK_YEARS = 3
URL_TICKER_MAP_DEFAULT = "https://drive.google.com/file/d/1MlVp6yF7FZjTdRFMpYCxgF-ezyKvO4gG/view?usp=sharing"

# --- DATASET KEYS (PARQUET) ---
DATA_KEYS_PARQUET = {
    "Darcy List": "PARQUET_DARCY",
    "NQ100": "PARQUET_NQ100",
    "SP100": "PARQUET_SP100",
    "Sectors": "PARQUET_SECTORS",
    "Macro": "PARQUET_MACRO",
    "MidCaps": "PARQUET_MIDCAP"
}

# --- OPTIMIZED NETWORK UTILS ---

def _download_from_gdrive(url: str, as_binary: bool = False):
    """Unified helper to download content from Google Drive with confirmation token handling."""
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if not file_id: return None
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        # Check for confirmation token (for large files)
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            # Check content for confirmation link text
            chunk = response.content[:1024].decode('utf-8', errors='ignore')
            match = re.search(r'confirm=([0-9A-Za-z_]+)', chunk)
            if match: confirm_token = match.group(1)

        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
            
        if not as_binary:
            text_content = response.text
            if text_content.strip().startswith("<!DOCTYPE html>"): return "HTML_ERROR"
            return StringIO(text_content)
        else:
            return BytesIO(response.content)

    except Exception as e:
        # Silently fail or log if needed, user sees "No Data" later
        return None

def get_confirmed_gdrive_data(url):
    return _download_from_gdrive(url, as_binary=False)

def get_gdrive_binary_data(url):
    return _download_from_gdrive(url, as_binary=True)

@st.cache_data(ttl=600, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    try:
        # Load straight from csv
        df = pd.read_csv(url)
        want = ["Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"]
        
        # Filter cols efficiently
        df = df[df.columns.intersection(want)].copy()
        
        # Vectorized string cleaning
        str_cols = ["Order Type", "Symbol", "Strike", "Expiry"]
        for c in str_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        # Numeric cleanup
        if "Dollars" in df.columns and df["Dollars"].dtype == 'object':
            df["Dollars"] = df["Dollars"].str.replace(r'[$,]', '', regex=True)
        df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)

        if "Contracts" in df.columns and df["Contracts"].dtype == 'object':
            df["Contracts"] = df["Contracts"].str.replace(',', '', regex=False)
        df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(0)
        
        # Date conversion
        if "Trade Date" in df.columns:
            df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        if "Expiry" in df.columns:
            df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
        if "Strike (Actual)" in df.columns:
            df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
            
        # Filter errors
        if "Error" in df.columns:
            # Drop error rows
            df = df[~df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})]
            
        return df
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_market_cap(symbol: str) -> float:
    for attempt in range(3):
        try:
            t = yf.Ticker(symbol)
            mc = t.fast_info.get('marketCap')
            if mc: return float(mc)
            info = t.info
            mc = info.get('marketCap')
            if mc: return float(mc)
        except Exception:
            time.sleep(0.1)
    return 0.0

def fetch_market_caps_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            results[t] = future.result() if not future.exception() else 0.0
    return results

@st.cache_data(ttl=300)
def get_stock_indicators(sym: str):
    try:
        ticker_obj = yf.Ticker(sym)
        # Fetching 2y is okay for SMA200, but we can optimize if only daily needed
        h_full = ticker_obj.history(period="2y", interval="1d")
        
        if h_full.empty: return None, None, None, None, None
        
        # Efficient selection
        close = h_full["Close"]
        sma200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else None
        
        # For EMA we only need enough data to stabilize, 60 is okay
        h_recent = close.iloc[-60:] if len(close) > 60 else close
        if h_recent.empty: return None, None, None, None, None
        
        spot_val = float(h_recent.iloc[-1])
        ema8  = float(h_recent.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = float(h_recent.ewm(span=21, adjust=False).mean().iloc[-1])
        
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
    if row_count == 0: return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

@st.cache_data(ttl=3600)
def get_expiry_color_map():
    try:
        today = date.today()
        days_ahead = (4 - today.weekday()) % 7
        this_fri = today + timedelta(days=days_ahead)
        return {
            this_fri.strftime("%d %b %y"): "background-color: #b7e1cd; color: black;",
            (this_fri + timedelta(days=7)).strftime("%d %b %y"): "background-color: #fce8b2; color: black;",
            (this_fri + timedelta(days=14)).strftime("%d %b %y"): "background-color: #f4c7c3; color: black;"
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
        if f.is_integer(): return str(int(f))
        return str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid_dates = df["Trade Date"].dropna()
        if not valid_dates.empty:
            return valid_dates.max().date()
    return date.today() - timedelta(days=1)

@st.cache_data(ttl=900)
def load_parquet_and_clean(url_key):
    """
    Loads Parquet data and normalizes column names (EMA_8 -> EMA8).
    Includes optimization to standardize columns once upon load.
    """
    url = st.secrets.get(url_key)
    if not url: return None
    
    try:
        buffer = _download_from_gdrive(url, as_binary=True)
        if not buffer: return None
        
        df = pd.read_parquet(buffer, engine='pyarrow')
        
        # Standardize column names immediately (Upper & Strip)
        # This prevents doing it repeatedly inside the loop for every ticker
        df.columns = [c.strip().upper().replace(' ', '').replace('-', '') for c in df.columns]
        
        # Rename logic to match script expectations
        rename_map = {
            "EMA_8": "EMA8", "EMA8": "EMA8",
            "EMA_21": "EMA21", "EMA21": "EMA21",
            "RSI_14": "RSI", "RSI": "RSI",
            "W_EMA_8": "W_EMA8", "W_EMA8": "W_EMA8",
            "W_EMA_21": "W_EMA21", "W_EMA21": "W_EMA21",
            "W_RSI_14": "W_RSI", "W_RSI": "W_RSI"
        }
        
        # Only apply renaming to columns that actually exist
        # We look for the standardized keys in the map
        final_rename = {}
        for col in df.columns:
            if col in rename_map:
                final_rename[col] = rename_map[col]
            # Handle variations like "EMA_8" becoming "EMA_8" after upper/strip 
            # (Wait, replace('-','') handles underscores? No, replace('_', '') was not in original)
            # Let's fix the underscore issue:
            # The original code did: replace('-', '').upper(). 
            # It did NOT remove underscores.
        
        # To be safe and compatible with the original logic's mapping:
        # We do the mapping manually based on the specific keys we know
        
        # Re-apply the specific renames for the known columns
        # Since we just uppercased everything, we need uppercase keys
        manual_map = {
            "EMA_8": "EMA8", "EMA8": "EMA8",
            "EMA_21": "EMA21", "EMA21": "EMA21",
            "RSI_14": "RSI", "RSI14": "RSI", "RSI": "RSI",
            "W_EMA_8": "W_EMA8", "W_EMA8": "W_EMA8",
            "W_EMA_21": "W_EMA21", "W_EMA21": "W_EMA21",
            "W_RSI_14": "W_RSI", "W_RSI14": "W_RSI", "W_RSI": "W_RSI"
        }
        
        df.rename(columns=manual_map, inplace=True)
        
        # Ensure Date format
        date_cols = [c for c in df.columns if "DATE" in c]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            
        return df
    except Exception as e:
        st.error(f"Error loading {url_key}: {e}")
        return None

@st.cache_data(ttl=3600)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP", URL_TICKER_MAP_DEFAULT)
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
    if not mapping or ticker not in mapping: return None
    file_id = mapping[ticker]
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    buffer = get_confirmed_gdrive_data(file_url)
    if buffer and buffer != "HTML_ERROR":
        try:
            df = pd.read_csv(buffer)
            df.columns = [c.strip().upper() for c in df.columns]
            return df
        except: return None
    return None

def calculate_optimal_signal_stats_vectorized(history_indices, fwd_ret_dict, current_idx, signal_type='Bullish', timeframe='Daily'):
    """
    Optimized version of stats calculator.
    Uses pre-calculated forward returns arrays (fwd_ret_dict) to avoid O(N^2) loops.
    """
    # 1. Filter indices to only look at history
    # valid_hist_indices is a list of integers
    valid_hist_indices = [idx for idx in history_indices if idx < current_idx]
    
    if not valid_hist_indices: return None

    periods = [10, 30, 60, 90, 180]
    best_pf = -1.0
    best_stats = None
    
    unit = 'w' if timeframe.lower() == 'weekly' else 'd'

    # Convert to numpy array for fast indexing
    indices_arr = np.array(valid_hist_indices)

    for p in periods:
        # Get pre-calculated returns for these indices
        # This is O(K) where K is number of history signals, instead of O(K) calculation
        if p not in fwd_ret_dict: continue
        
        # Extract returns
        raw_rets = fwd_ret_dict[p][indices_arr]
        
        # Filter out NaNs (which occur if signal was too close to end of data)
        raw_rets = raw_rets[~np.isnan(raw_rets)]
        
        if len(raw_rets) == 0: continue
            
        # Flip returns for Bearish signals (Shorting logic: Price Drop = Positive Return)
        if signal_type == 'Bearish':
            strat_rets = -raw_rets
        else:
            strat_rets = raw_rets
            
        n = len(strat_rets)
        wins = strat_rets[strat_rets > 0]
        losses = strat_rets[strat_rets < 0]
        
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        
        if gross_loss == 0:
            pf = 999.0 if gross_win > 0 else 0.0
        else:
            pf = gross_win / gross_loss
            
        win_rate = (len(wins) / n) * 100
        avg_ret = np.mean(strat_rets) * 100
        
        if pf > best_pf:
            best_pf = pf
            best_stats = {
                "Best Period": f"{p}{unit}",
                "Profit Factor": pf,
                "Win Rate": win_rate,
                "EV": avg_ret,
                "N": n
            }
            
    return best_stats

# --- NEW HELPERS FOR TOP 3 ---
def get_optimal_rsi_duration(history_df, current_rsi, tolerance=2.0):
    # (Kept unchanged, legacy logic for top 3)
    if history_df is None or len(history_df) < 100: return 30, "Default (No Hist)"
    close_col = "CLOSE" if "CLOSE" in history_df.columns else "Close"
    
    if "RSI_14" not in history_df.columns and "RSI" not in history_df.columns:
         delta = history_df[close_col].diff()
         gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
         loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
         rs = gain / loss
         history_df["RSI"] = 100 - (100 / (1 + rs))
    
    rsi_col = "RSI_14" if "RSI_14" in history_df.columns else "RSI"
    rsi_vals = history_df[rsi_col].values
    close_vals = history_df[close_col].values
    
    min_rsi, max_rsi = current_rsi - tolerance, current_rsi + tolerance
    mask = (rsi_vals >= min_rsi) & (rsi_vals <= max_rsi)
    match_indices = np.where(mask)[0]
    
    if len(match_indices) < 5: return 30, "Default (Low Samples)"
        
    periods, best_p, best_score = [14, 30, 45, 60], 30, -999
    total_len = len(close_vals)
    
    for p in periods:
        valid_indices = match_indices[match_indices + p < total_len]
        if len(valid_indices) < 5: continue
        entries = close_vals[valid_indices]
        exits = close_vals[valid_indices + p]
        returns = (exits - entries) / entries
        score = (np.mean(returns > 0) * 2) + np.mean(returns)
        if score > best_score:
            best_score, best_p = score, p
            
    return best_p, f"RSI Backtest (Optimal {best_p}d)"

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None
    today_dt = pd.to_datetime(date.today())
    f = global_df[(global_df["Symbol"].astype(str).str.upper() == ticker) & (global_df["Expiry_DT"] > today_dt)].copy()
    if f.empty: return None

    if order_type_filter: f = f[f["Order Type"] == order_type_filter]
    else: f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
    if f.empty: return None
    
    f = f.sort_values(by="Dollars", ascending=False)
    wt = f.iloc[0]
    
    # Simple logic: if biggest trade is Put Sold but OTM (below price), use it.
    if wt["Order Type"] == "Puts Sold" and wt["Strike (Actual)"] > current_price:
        otm_puts = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm_puts.empty: wt = otm_puts.iloc[0]
    
    return {"Strike": wt["Strike (Actual)"], "Expiry": wt["Expiry_DT"].strftime("%d %b"), "Dollars": wt["Dollars"], "Type": wt["Order Type"]}

def analyze_trade_setup(ticker, t_df, global_df):
    score, reasons = 0, []
    suggestions = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    if t_df is None or t_df.empty: return 0, ["No data"], suggestions

    last = t_df.iloc[-1]
    close = last.get('CLOSE', 0) or last.get('Close', 0)
    ema8 = last.get('EMA_8', 0)
    ema21 = last.get('EMA_21', 0)
    sma200 = last.get('SMA_200', 0)
    rsi = last.get('RSI_14', 50)
    
    if close > ema8 and close > ema21: score += 2; reasons.append("Strong Trend (Price > EMA8 & EMA21)")
    elif close > ema21: score += 1; reasons.append("Moderate Trend (Price > EMA21)")
        
    if close > sma200: score += 2; reasons.append("Long-term Bullish (> SMA200)")
    if 45 < rsi < 65: score += 2; reasons.append(f"Healthy Momentum (RSI {rsi:.0f})")
    elif rsi >= 70: score -= 1; reasons.append("Overbought (RSI > 70)")
    
    opt_days, opt_reason = (30, "Standard 30d") if len(t_df) < 100 else get_optimal_rsi_duration(t_df, rsi)
    target_date_str = (date.today() + timedelta(days=opt_days)).strftime("%d %b")
    
    put_whale = find_whale_confluence(ticker, global_df, close, "Puts Sold")
    call_whale = find_whale_confluence(ticker, global_df, close, "Calls Bought")
    
    sp_strike = math.floor(ema21) 
    sp_reason, sp_exp = "EMA21 Support", target_date_str
    
    if put_whale and put_whale["Strike"] < close:
        sp_strike, sp_reason, sp_exp = put_whale["Strike"], f"Whale Tailing (${put_whale['Dollars']/1e6:.1f}M sold)", put_whale["Expiry"] 
    elif call_whale:
         sp_exp, sp_reason = call_whale["Expiry"], f"EMA21 (Align with Call Whale Exp)"
    suggestions['Sell Puts'] = f"Strike ${sp_strike} ({sp_reason}), Exp ~{sp_exp}"

    bc_strike, bc_reason, bc_exp = math.ceil(close), "ATM Momentum", target_date_str
    if call_whale:
        bc_strike, bc_exp, bc_reason = call_whale["Strike"], call_whale["Expiry"], f"Tailing Call Whale (${call_whale['Dollars']/1e6:.1f}M)"
        
    if close > ema8 or call_whale: suggestions['Buy Calls'] = f"Strike ${bc_strike} ({bc_reason}), Exp ~{bc_exp}"
    suggestions['Buy Commons'] = f"Entry: ${close:.2f}. Stop Loss: ${ema21:.2f}"
    
    if "RSI Backtest" in opt_reason: reasons.append(f"Hist. Optimal Hold: {opt_days} Days")
    if put_whale: reasons.append(f"Whale: Sold Puts @ ${put_whale['Strike']}")
    if call_whale: reasons.append(f"Whale: Bought Calls @ ${call_whale['Strike']}")
    
    return score, reasons, suggestions

def prepare_data(df):
    """
    Prepares dataframes and PRE-CALCULATES forward returns for vectorization.
    """
    # Note: Columns are already cleaned in load_parquet_and_clean
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    # Identify RSI and EMA columns
    d_rsi = next((c for c in cols if 'RSI' in c and 'W_' not in c), 'RSI_14')
    d_ema8 = next((c for c in cols if c == 'EMA8'), None)
    d_ema21 = next((c for c in cols if c == 'EMA21'), None)
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Build Daily Dataframe
    needed_cols = [close_col, vol_col, high_col, low_col]
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8: needed_cols.append(d_ema8)
    if d_ema21: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    
    rename_dict = {close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low'}
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8: rename_dict[d_ema8] = 'EMA8'
    if d_ema21: rename_dict[d_ema21] = 'EMA21'
    
    df_d.rename(columns=rename_dict, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    if 'RSI' not in df_d.columns:
        delta = df_d['Price'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df_d['RSI'] = 100 - (100 / (1 + rs))
        
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    # --- VECTORIZED PRE-CALCULATION FOR DAILY ---
    # Calculate all forward returns at once: O(N) instead of O(N^2) loops later
    periods = [10, 30, 60, 90, 180]
    for p in periods:
        # Shift(-p) brings the future price to the current row
        # Return = (Future - Current) / Current
        df_d[f'FwdRet_{p}'] = (df_d['Price'].shift(-p) - df_d['Price']) / df_d['Price']

    # Build Weekly Dataframe
    w_close, w_vol = 'W_CLOSE', 'W_VOLUME'
    w_high, w_low = 'W_HIGH', 'W_LOW'
    w_ema8 = next((c for c in cols if c == 'W_EMA8'), 'W_EMA_8')
    w_ema21 = next((c for c in cols if c == 'W_EMA21'), 'W_EMA_21')
    actual_w_rsi = next((c for c in cols if c in ['W_RSI', 'W_RSI_14']), None)
    
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low]) and actual_w_rsi:
        cols_w = [w_close, w_vol, w_high, w_low, actual_w_rsi]
        if w_ema8 in df.columns: cols_w.append(w_ema8)
        if w_ema21 in df.columns: cols_w.append(w_ema21)
        
        df_w = df[cols_w].copy()
        w_rename = {w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low', actual_w_rsi: 'RSI'}
        if w_ema8 in df.columns: w_rename[w_ema8] = 'EMA8'
        if w_ema21 in df.columns: w_rename[w_ema21] = 'EMA21'
        
        df_w.rename(columns=w_rename, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        df_w = df_w.dropna(subset=['Price', 'RSI'])

        # --- VECTORIZED PRE-CALCULATION FOR WEEKLY ---
        for p in periods:
            df_w[f'FwdRet_{p}'] = (df_w['Price'].shift(-p) - df_w['Price']) / df_w['Price']
    else: 
        df_w = None
        
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe, min_n=0):
    divergences = []
    n_rows = len(df_tf)
    if n_rows < DIVERGENCE_LOOKBACK + 1: return divergences
    
    # Pre-fetch columns as numpy arrays for speed
    rsi_vals = df_tf['RSI'].values
    low_vals = df_tf['Low'].values
    high_vals = df_tf['High'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    
    # Grab the pre-calculated return arrays
    # Dict mapping period -> numpy array
    fwd_ret_dict = {p: df_tf[f'FwdRet_{p}'].values for p in [10, 30, 60, 90, 180]}

    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': 
             return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    
    bullish_signal_indices = []
    bearish_signal_indices = []
    potential_signals = [] 

    start_search = DIVERGENCE_LOOKBACK
    
    # Optimization: Could vectorize this scanning loop, but logic is complex.
    # Python loop is acceptable here, bottleneck was mostly the stats calculation.
    for i in range(start_search, n_rows):
        p2_rsi, p2_low, p2_high = rsi_vals[i], low_vals[i], high_vals[i]
        
        lb_start = i - DIVERGENCE_LOOKBACK
        lb_rsi = rsi_vals[lb_start:i]
        
        # Bullish Logic
        if p2_low < np.min(low_vals[lb_start:i]):
            p1_idx_rel = np.argmin(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            if p2_rsi > (p1_rsi + RSI_DIFF_THRESHOLD):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                if not np.any(subset_rsi > 50): 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi <= p1_rsi): valid = False
                    if valid:
                        is_vol_high = int(vol_vals[i] > (vol_sma_vals[i] * 1.5)) if not np.isnan(vol_sma_vals[i]) else 0
                        bullish_signal_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bullish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})
        
        # Bearish Logic
        elif p2_high > np.max(high_vals[lb_start:i]):
            p1_idx_rel = np.argmax(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            if p2_rsi < (p1_rsi - RSI_DIFF_THRESHOLD):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                if not np.any(subset_rsi < 50): 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi >= p1_rsi): valid = False
                    if valid:
                        is_vol_high = int(vol_vals[i] > (vol_sma_vals[i] * 1.5)) if not np.isnan(vol_sma_vals[i]) else 0
                        bearish_signal_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bearish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})

    display_threshold_idx = n_rows - SIGNAL_LOOKBACK_PERIOD
    
    for sig in potential_signals:
        i = sig["index"]
        if i < display_threshold_idx: continue

        s_type = sig["type"]
        idx_p1_abs = sig["p1_idx"]
        
        tags = []
        row_at_sig = df_tf.iloc[i] 
        curr_price = row_at_sig['Price']
        
        ema8_val = row_at_sig.get('EMA8') 
        ema21_val = row_at_sig.get('EMA21')

        if s_type == 'Bullish':
            if ema8_val is not None and curr_price >= ema8_val: tags.append(f"EMA{EMA8_PERIOD}")
            if ema21_val is not None and curr_price >= ema21_val: tags.append(f"EMA{EMA21_PERIOD}")
        else:
            if ema8_val is not None and curr_price <= ema8_val: tags.append(f"EMA{EMA8_PERIOD}")
            if ema21_val is not None and curr_price <= ema21_val: tags.append(f"EMA{EMA21_PERIOD}")
        
        if sig["vol_high"]: tags.append("VOL_HIGH")
        if vol_vals[i] > vol_vals[idx_p1_abs]: tags.append("VOL_GROW")
        
        hist_list = bullish_signal_indices if s_type == 'Bullish' else bearish_signal_indices
        
        # --- OPTIMIZATION CALL ---
        # Pass fwd_ret_dict instead of calculating on the fly
        best_stats = calculate_optimal_signal_stats_vectorized(hist_list, fwd_ret_dict, i, signal_type=s_type, timeframe=timeframe)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
        
        if best_stats["N"] < min_n: continue

        price_p1 = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
        price_p2 = low_vals[i] if s_type=='Bullish' else high_vals[i]
            
        divergences.append({
            'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 
            'Tags': tags, 'Signal_Date_ISO': get_date_str(i, '%Y-%m-%d'), 
            'Date_Display': f"{get_date_str(idx_p1_abs, '%b %d')} → {get_date_str(i, '%b %d')}",
            'RSI_Display': f"{int(round(rsi_vals[idx_p1_abs]))} {'↗' if rsi_vals[i] > rsi_vals[idx_p1_abs] else '↘'} {int(round(rsi_vals[i]))}", 
            'Price_Display': f"${price_p1:,.2f} ↗ ${price_p2:,.2f}" if price_p2 > price_p1 else f"${price_p1:,.2f} ↘ ${price_p2:,.2f}", 
            'Last_Close': f"${df_tf.iloc[-1]['Price']:,.2f}", 
            'Best Period': best_stats['Best Period'], 'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'], 'EV': best_stats['EV'], 'N': best_stats['N']
        })
            
    return divergences

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, min_n=1, filter_date=None, timeframe='Daily'):
    signals = []
    if len(df) < 200: return signals
    
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(pct_low)
    p90 = hist_df['RSI'].quantile(pct_high)
    
    rsi_vals = hist_df['RSI'].values
    # Get vector arrays from the filtered history DF
    fwd_ret_dict = {p: hist_df[f'FwdRet_{p}'].values for p in [10, 30, 60, 90, 180]}
    
    # 1. Identify ALL Signal Indices
    bullish_signal_indices = []
    bearish_signal_indices = []
    
    for i in range(1, len(hist_df)):
        prev_rsi = rsi_vals[i-1]
        curr_rsi = rsi_vals[i]
        if prev_rsi < p10 and curr_rsi >= (p10 + 1.0): bullish_signal_indices.append(i)
        elif prev_rsi > p90 and curr_rsi <= (p90 - 1.0): bearish_signal_indices.append(i)
            
    # 2. Filter and Optimize
    latest_close = df['Price'].iloc[-1] 
    all_indices = sorted(bullish_signal_indices + bearish_signal_indices)
    
    for i in all_indices:
        curr_row = hist_df.iloc[i]
        curr_date = curr_row.name.date()
        
        if filter_date and curr_date < filter_date: continue
            
        is_bullish = i in bullish_signal_indices
        s_type = 'Bullish' if is_bullish else 'Bearish'
        thresh_val = p10 if is_bullish else p90
        curr_rsi_val = rsi_vals[i]
        
        hist_list = bullish_signal_indices if is_bullish else bearish_signal_indices
        
        # Optimized call
        best_stats = calculate_optimal_signal_stats_vectorized(hist_list, fwd_ret_dict, i, signal_type=s_type, timeframe=timeframe)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
             
        if best_stats["N"] < min_n: continue
            
        rsi_disp = f"{thresh_val:.0f} ↗ {curr_rsi_val:.0f}" if is_bullish else f"{thresh_val:.0f} ↘ {curr_rsi_val:.0f}"
        
        signals.append({
            'Ticker': ticker,
            'Date': curr_row.name.strftime('%b %d'),
            'Date_Obj': curr_date,
            'Action': "Leaving Low" if is_bullish else "Leaving High",
            'RSI_Display': rsi_disp,
            'Signal_Price': f"${curr_row['Price']:,.2f}",
            'Last_Close': f"${latest_close:,.2f}", 
            'Signal_Type': s_type,
            'Best Period': best_stats['Best Period'],
            'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'],
            'EV': best_stats['EV'],
            'N': best_stats['N']
        })
            
    return signals

@st.cache_data(ttl=3600)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        df = df.reset_index().rename(columns={df.index.name: "DATE"}).rename(columns={"Date": "DATE", "Close": "CLOSE", "Volume": "VOLUME", "High": "HIGH", "Low": "LOW", "Open": "OPEN"})
        
        if "DATE" not in df.columns and "index" in df.columns: df.rename(columns={"index": "DATE"}, inplace=True)

        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"])

        if df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
            
        df.columns = [c.upper() for c in df.columns]
        
        delta = df["CLOSE"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI_14"] = df["RSI"]
        
        return df
    except Exception: return None

# --- 2. APP MODULES ---

def run_database_app(df):
    st.title("📂 Database")
    max_data_date = get_max_trade_date(df)
    
    # State init
    for k, v in {
        'saved_db_ticker': "", 'saved_db_start': max_data_date, 'saved_db_end': max_data_date,
        'saved_db_exp': (date.today() + timedelta(days=365)), 
        'saved_db_inc_cb': True, 'saved_db_inc_ps': True, 'saved_db_inc_pb': True
    }.items():
        if k not in st.session_state: st.session_state[k] = v

    def save_db_state(key, saved_key): st.session_state[saved_key] = st.session_state[key]
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: db_ticker = st.text_input("Ticker (blank=all)", value=st.session_state.saved_db_ticker, key="db_ticker_input", on_change=save_db_state, args=("db_ticker_input", "saved_db_ticker")).strip().upper()
    with c2: start_date = st.date_input("Trade Start Date", value=st.session_state.saved_db_start, key="db_start", on_change=save_db_state, args=("db_start", "saved_db_start"))
    with c3: end_date = st.date_input("Trade End Date", value=st.session_state.saved_db_end, key="db_end", on_change=save_db_state, args=("db_end", "saved_db_end"))
    with c4: db_exp_end = st.date_input("Expiration Range (end)", value=st.session_state.saved_db_exp, key="db_exp", on_change=save_db_state, args=("db_exp", "saved_db_exp"))
    
    ot1, ot2, ot3, _ = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_inc_cb", on_change=save_db_state, args=("db_inc_cb", "saved_db_inc_cb"))
    with ot2: inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_inc_ps", on_change=save_db_state, args=("db_inc_ps", "saved_db_inc_ps"))
    with ot3: inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_inc_pb", on_change=save_db_state, args=("db_inc_pb", "saved_db_inc_pb"))
    
    f = df.copy()
    if db_ticker: f = f[f["Symbol"].astype(str).str.upper().eq(db_ticker)]
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    if db_exp_end: f = f[f["Expiry_DT"].dt.date <= db_exp_end]
    
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed_types = [t for t, inc in [("Calls Bought", inc_cb), ("Puts Bought", inc_pb), ("Puts Sold", inc_ps)] if inc]
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
    f_filtered = f[f[order_type_col].isin(["Calls Bought", "Puts Sold", "Puts Bought"])].copy()
    if f_filtered.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    f_filtered["Signed_Dollars"] = np.where(f_filtered[order_type_col].isin(["Calls Bought", "Puts Sold"]), f_filtered["Dollars"], -f_filtered["Dollars"])
    
    smart_stats = f_filtered.groupby("Symbol").agg(
        Signed_Dollars=("Signed_Dollars", "sum"),
        Trade_Count=("Symbol", "count"),
        Last_Trade=("Trade Date", "max")
    ).reset_index().rename(columns={"Signed_Dollars": "Net Sentiment ($)"})
    
    batch_caps = fetch_market_caps_batch(smart_stats["Symbol"].unique())
    smart_stats["Market Cap"] = smart_stats["Symbol"].map(batch_caps)
    
    valid_data = smart_stats[smart_stats["Market Cap"] >= mc_thresh].copy()
    
    unique_dates = sorted(f_filtered["Trade Date"].unique())
    recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
    mom_stats = f_filtered[f_filtered["Trade Date"].isin(recent_dates)].groupby("Symbol")["Signed_Dollars"].sum().reset_index().rename(columns={"Signed_Dollars": "Momentum ($)"})
    
    valid_data = valid_data.merge(mom_stats, on="Symbol", how="left").fillna(0)
    top_bulls, top_bears = pd.DataFrame(), pd.DataFrame()

    if not valid_data.empty:
        valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
        def normalize(series):
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn) if (mx != mn) else 0

        # Base Scores
        for prefix, sign in [("Bull", 1), ("Bear", -1)]:
            flow = normalize((sign * valid_data["Net Sentiment ($)"]).clip(lower=0))
            imp = normalize((sign * valid_data["Impact"]).clip(lower=0))
            mom = normalize((sign * valid_data["Momentum ($)"]).clip(lower=0))
            valid_data[f"Base_Score_{prefix}"] = (0.35 * flow) + (0.30 * imp) + (0.35 * mom)
        
        valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")
        
        candidates_bull = valid_data.sort_values(by="Base_Score_Bull", ascending=False).head(limit * 3)
        candidates_bear = valid_data.sort_values(by="Base_Score_Bear", ascending=False).head(limit * 3)
        
        batch_techs = fetch_technicals_batch(list(set(candidates_bull["Symbol"]).union(set(candidates_bear["Symbol"])))) if filter_ema else {}

        def filter_list(source_df, mode="Bull"):
            results = []
            for _, row in source_df.iterrows():
                if not filter_ema: 
                    row["Score"], row["Trend"] = row[f"Base_Score_{mode}"] * 100, "—"
                    results.append(row)
                else:
                    s, e8, _, _, _ = batch_techs.get(row["Symbol"], (None, None, None, None, None))
                    if s and e8:
                        passes = (s > e8) if mode == "Bull" else (s < e8)
                        if passes:
                            row["Score"] = row[f"Base_Score_{mode}"] * 100
                            row["Trend"] = f"✅ {'>' if mode=='Bull' else '<'}EMA8"
                            results.append(row)
            return pd.DataFrame(results).head(limit)

        top_bulls = filter_list(candidates_bull, "Bull")
        top_bears = filter_list(candidates_bear, "Bear")
        
    return top_bulls, top_bears, valid_data

def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_data_date = get_max_trade_date(df)
    
    # Init State
    for k, v in {'saved_rank_start': max_data_date - timedelta(days=14), 'saved_rank_end': max_data_date,
                 'saved_rank_limit': 20, 'saved_rank_mc': "10B", 'saved_rank_ema': False}.items():
        if k not in st.session_state: st.session_state[k] = v

    def save_rank_state(key, saved_key): st.session_state[saved_key] = st.session_state[key]
    
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
    if f.empty: return st.warning("No data found matching these dates.")

    tab_rank, tab_ideas, tab_vol = st.tabs(["🧠 Smart Money", "💡 Top 3", "🤡 Bulltard"])
    mc_thresh = {"0B":0, "2B":2e9, "10B":1e10, "50B":5e10, "100B":1e11}.get(min_mkt_cap_rank, 1e10)
    top_bulls, top_bears, valid_data = calculate_smart_money_score(df, rank_start, rank_end, mc_thresh, filter_ema, limit)

    sm_config = {"Symbol": st.column_config.TextColumn("Ticker", width=60), "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                 "Trade_Count": st.column_config.NumberColumn("Qty", width=50), "Last Trade": st.column_config.TextColumn("Last", width=70)}
    
    with tab_rank:
        if valid_data.empty: st.warning("Not enough data for Smart Money scores.")
        else:
            sm1, sm2 = st.columns(2, gap="large")
            with sm1:
                st.markdown("<div style='color: #71d28a; font-weight:bold;'>Top Bullish Scores</div>", unsafe_allow_html=True)
                if not top_bulls.empty: st.dataframe(top_bulls[["Symbol", "Score", "Trade_Count", "Last Trade"]], use_container_width=True, hide_index=True, column_config=sm_config, height=get_table_height(top_bulls, max_rows=100))
            with sm2:
                st.markdown("<div style='color: #f29ca0; font-weight:bold;'>Top Bearish Scores</div>", unsafe_allow_html=True)
                if not top_bears.empty: st.dataframe(top_bears[["Symbol", "Score", "Trade_Count", "Last Trade"]], use_container_width=True, hide_index=True, column_config=sm_config, height=get_table_height(top_bears, max_rows=100))

    with tab_ideas:
        if top_bulls.empty: st.info("No Bullish candidates found.")
        else:
            st.caption(f"ℹ️ Analyzing the Top {len(top_bulls)} 'Smart Money' tickers for confluence...")
            st.markdown("<span style='color:red;'>⚠️ Note: This methodology is a work in progress.</span>", unsafe_allow_html=True)
            
            ticker_map = load_ticker_map()
            candidates = []
            prog = st.progress(0, text="Analyzing technicals...")
            bull_list = top_bulls["Symbol"].tolist()
            
            for i, t in enumerate(bull_list):
                prog.progress((i+1)/len(bull_list), text=f"Checking {t}...")
                t_df = get_ticker_technicals(t, ticker_map) or fetch_yahoo_data(t)
                if t_df is not None:
                    sm_score = top_bulls[top_bulls["Symbol"]==t]["Score"].iloc[0]
                    tech_score, reasons, suggs = analyze_trade_setup(t, t_df, df)
                    candidates.append({"Ticker": t, "Score": (sm_score / 25.0) + tech_score, "Price": t_df.iloc[-1].get('CLOSE') or t_df.iloc[-1].get('Close'), "Reasons": reasons, "Suggestions": suggs})
            
            prog.empty()
            for i, cand in enumerate(sorted(candidates, key=lambda x: x['Score'], reverse=True)[:3]):
                with st.columns(3)[i]:
                    with st.container(border=True):
                        st.markdown(f"### #{i+1} {cand['Ticker']}")
                        st.metric("Conviction", f"{cand['Score']:.1f}/10", f"${cand['Price']:.2f}")
                        if cand['Suggestions']['Sell Puts']: st.success(f"🛡️ **Sell Put:** {cand['Suggestions']['Sell Puts']}")
                        if cand['Suggestions']['Buy Calls']: st.info(f"🟢 **Buy Call:** {cand['Suggestions']['Buy Calls']}")
                        st.markdown("---")
                        for r in cand['Reasons']: st.caption(f"• {r}")

    with tab_vol:
        f_filtered = f[f["Order Type"].isin(["Calls Bought", "Puts Sold", "Puts Bought"])].copy()
        counts = f_filtered.groupby(["Symbol", "Order Type"]).size().unstack(fill_value=0)
        for col in ["Calls Bought", "Puts Sold", "Puts Bought"]: 
            if col not in counts.columns: counts[col] = 0
            
        scores = pd.DataFrame({"Score": counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"],
                               "Trade Count": counts.sum(axis=1),
                               "Last Trade": f_filtered.groupby("Symbol")["Trade Date"].max().dt.strftime("%d %b %y")}).reset_index()
        
        batch_caps = fetch_market_caps_batch(scores["Symbol"].unique())
        scores = scores[scores["Symbol"].map(batch_caps) >= mc_thresh]
        
        pre_bull = scores.sort_values(by=["Score", "Trade Count"], ascending=[False, False])
        pre_bear = scores.sort_values(by=["Score", "Trade Count"], ascending=[True, False])

        def get_filtered_list(source, mode="Bull"):
            if not filter_ema: return source.head(limit)
            cands = source.head(limit*3)
            techs = fetch_technicals_batch(cands["Symbol"].tolist())
            final = []
            for _, r in cands.iterrows():
                s, e8, _, _, _ = techs.get(r["Symbol"], (None,None,None,None,None))
                if s and e8 and ((mode=="Bull" and s>e8) or (mode=="Bear" and s<e8)): final.append(r)
            return pd.DataFrame(final).head(limit)

        v1, v2 = st.columns(2)
        rc = {"Symbol": st.column_config.TextColumn("Symbol", width=60), "Trade Count": st.column_config.NumberColumn("#", width=50), 
              "Last Trade": st.column_config.TextColumn("Last Trade", width=90), "Score": st.column_config.NumberColumn("Score", width=50)}
        with v1: 
            st.markdown("<div style='color: #71d28a; font-weight:bold;'>Bullish Volume</div>", unsafe_allow_html=True)
            b = get_filtered_list(pre_bull, "Bull")
            if not b.empty: st.dataframe(b, use_container_width=True, hide_index=True, column_config=rc, height=get_table_height(b, max_rows=100))
        with v2:
            st.markdown("<div style='color: #f29ca0; font-weight:bold;'>Bearish Volume</div>", unsafe_allow_html=True)
            b = get_filtered_list(pre_bear, "Bear")
            if not b.empty: st.dataframe(b, use_container_width=True, hide_index=True, column_config=rc, height=get_table_height(b, max_rows=100))

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    for k, v in {'saved_sz_ticker': "AMZN", 'saved_sz_start': None, 'saved_sz_end': None, 'saved_sz_exp': (date.today() + timedelta(days=365)),
                 'saved_sz_view': "Price Zones", 'saved_sz_width_mode': "Auto", 'saved_sz_fixed': 10,
                 'saved_sz_inc_cb': True, 'saved_sz_inc_ps': True, 'saved_sz_inc_pb': True}.items():
        if k not in st.session_state: st.session_state[k] = v

    def save_sz(key, saved_key): st.session_state[saved_key] = st.session_state[key]
    
    col_settings, col_visuals = st.columns([1, 2.5], gap="large")
    with col_settings:
        ticker = st.text_input("Ticker", value=st.session_state.saved_sz_ticker, key="sz_ticker", on_change=save_sz, args=("sz_ticker", "saved_sz_ticker")).strip().upper()
        td_start = st.date_input("Trade Date (start)", value=st.session_state.saved_sz_start, key="sz_start", on_change=save_sz, args=("sz_start", "saved_sz_start"))
        td_end = st.date_input("Trade Date (end)", value=st.session_state.saved_sz_end, key="sz_end", on_change=save_sz, args=("sz_end", "saved_sz_end"))
        exp_end = st.date_input("Exp. Range (end)", value=st.session_state.saved_sz_exp, key="sz_exp", on_change=save_sz, args=("sz_exp", "saved_sz_exp"))
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], index=0 if st.session_state.saved_sz_view == "Price Zones" else 1, label_visibility="collapsed", key="sz_view", on_change=save_sz, args=("sz_view", "saved_sz_view"))
            width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], index=0 if st.session_state.saved_sz_width_mode == "Auto" else 1, label_visibility="collapsed", key="sz_width_mode", on_change=save_sz, args=("sz_width_mode", "saved_sz_width_mode"))
            fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=st.session_state.saved_sz_fixed, key="sz_fixed", on_change=save_sz, args=("sz_fixed", "saved_sz_fixed")) if width_mode == "Fixed" else 10
        
        with c_sub2:
            inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_sz_inc_cb, key="sz_inc_cb", on_change=save_sz, args=("sz_inc_cb", "saved_sz_inc_cb"))
            inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_sz_inc_ps, key="sz_inc_ps", on_change=save_sz, args=("sz_inc_ps", "saved_sz_inc_ps"))
            inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_sz_inc_pb, key="sz_inc_pb", on_change=save_sz, args=("sz_inc_pb", "saved_sz_inc_pb"))
    
    with col_visuals: chart_container = st.container()

    f_base = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start: f_base = f_base[f_base["Trade Date"].dt.date >= td_start]
    if td_end: f_base = f_base[f_base["Trade Date"].dt.date <= td_end]
    f_base = f_base[(f_base["Expiry_DT"].dt.date >= date.today()) & (f_base["Expiry_DT"].dt.date <= exp_end)]
    
    allowed = [t for t, i in [("Calls Bought", inc_cb), ("Puts Sold", inc_ps), ("Puts Bought", inc_pb)] if i]
    edit_pool_raw = f_base[f_base["Order Type"].isin(allowed)].copy()
    
    if edit_pool_raw.empty:
        with col_visuals: st.warning("No trades match current filters.")
        return

    edit_pool_raw.insert(0, "Include", True)
    edited_df = st.data_editor(
        edit_pool_raw[["Include", "Trade Date", "Order Type", "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"]],
        column_config={"Include": st.column_config.CheckboxColumn("Include", default=True), "Trade Date": st.column_config.DateColumn("Trade Date", format="DD MMM YY"),
                       "Expiry_DT": st.column_config.DateColumn("Expiry", format="DD MMM YY"), "Dollars": st.column_config.NumberColumn("Dollars", format="$%d"),
                       "Contracts": st.column_config.NumberColumn("Qty", format="%d")},
        disabled=["Trade Date", "Order Type", "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"],
        hide_index=True, use_container_width=True, key="sz_editor"
    )
    f = edit_pool_raw[edited_df["Include"]].copy()

    with chart_container:
        if f.empty: st.info("No rows selected.")
        else:
            spot, ema8, ema21, sma200, _ = get_stock_indicators(ticker)
            if spot is None: 
                y = fetch_yahoo_data(ticker)
                if y is not None: spot = y["CLOSE"].iloc[-1]
            if spot is None: spot = 100.0

            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            pct = lambda x: f"{(x/spot-1)*100:+.1f}%" if x else "—"
            if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct(ema8)})</span>')
            if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct(ema21)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

            f["Signed Dollars"] = np.where(f["Order Type"].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
            fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

            if view_mode == "Price Zones":
                s_min, s_max = f["Strike (Actual)"].min(), f["Strike (Actual)"].max()
                zw = float(st.select_slider("Fixed size", options=[1, 5, 10, 25, 50, 100], value=10) if width_mode == "Fixed" else next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, s_max - s_min) / 12.0)), 100))
                
                n_dn = int(math.ceil(max(0.0, (spot - s_min)) / zw))
                lower_edge = spot - n_dn * zw
                f["ZoneIdx"] = np.floor((f["Strike (Actual)"] - lower_edge) / zw).astype(int)

                agg = f.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                zs = pd.DataFrame({"ZoneIdx": range(int(f["ZoneIdx"].max() + 1))})
                zs["Zone_Low"] = lower_edge + zs["ZoneIdx"]*zw
                zs["Zone_High"] = lower_edge + (zs["ZoneIdx"]+1)*zw
                zs = zs.merge(agg, on="ZoneIdx", how="left").fillna(0)
                zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
                
                max_val = max(1.0, zs["Net_Dollars"].abs().max())
                html = ['<div class="zones-panel">']
                
                for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                    if r["Zone_Low"] + zw/2 <= spot and (html[-1] != '</div>' and 'price-divider' not in html[-1]):
                         html.append(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>')
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    html.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{(abs(r["Net_Dollars"])/max_val)*100:.1f}%"></div><div class="zone-value">{fmt_neg(r["Net_Dollars"])} | n={int(r.Trades)}</div></div></div>')
                html.append('</div>')
                st.markdown("".join(html), unsafe_allow_html=True)
            else:
                days = (pd.to_datetime(f["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
                f["Bucket"] = pd.cut(days, bins=[0, 7, 30, 60, 90, 120, 180, 365, 9999], labels=["0-7d", "8-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", ">365d"])
                agg = f.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                max_val = max(1.0, agg["Net_Dollars"].abs().max())
                html = []
                for _, r in agg.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    html.append(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{(abs(r["Net_Dollars"])/max_val)*100:.1f}%"></div><div class="zone-value">{fmt_neg(r["Net_Dollars"])} | n={int(r.Trades)}</div></div></div>')
                st.markdown("".join(html), unsafe_allow_html=True)

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = get_max_trade_date(df)
            
    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    
    # State Init
    for k, v in {'saved_pv_start': max_data_date, 'saved_pv_end': max_data_date, 'saved_pv_ticker': "",
                 'saved_pv_notional': "0M", 'saved_pv_mkt_cap': "0B", 'saved_pv_ema': "All",
                 'saved_calc_strike': 100.0, 'saved_calc_premium': 2.50, 'saved_calc_expiry': date.today() + timedelta(days=30)}.items():
        if k not in st.session_state: st.session_state[k] = v

    def save_pv(key, saved_key): st.session_state[saved_key] = st.session_state[key]

    with col_filters:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>🔍 Filters</h4>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1: td_start = st.date_input("Trade Start Date", value=st.session_state.saved_pv_start, key="pv_start", on_change=save_pv, args=("pv_start", "saved_pv_start"))
        with fc2: td_end = st.date_input("Trade End Date", value=st.session_state.saved_pv_end, key="pv_end", on_change=save_pv, args=("pv_end", "saved_pv_end"))
        with fc3: ticker_filter = st.text_input("Ticker (blank=all)", value=st.session_state.saved_pv_ticker, key="pv_ticker", on_change=save_pv, args=("pv_ticker", "saved_pv_ticker")).strip().upper()
        
        fc4, fc5, fc6 = st.columns(3)
        with fc4: 
            sel_not = st.selectbox("Min Dollars", options=["0M", "5M", "10M", "50M", "100M"], index=0, key="pv_notional", on_change=save_pv, args=("pv_notional", "saved_pv_notional"))
            min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[sel_not]
        with fc5: 
            sel_mc = st.selectbox("Mkt Cap Min", options=["0B", "10B", "50B", "100B", "200B", "500B", "1T"], index=0, key="pv_mkt_cap", on_change=save_pv, args=("pv_mkt_cap", "saved_pv_mkt_cap"))
            min_mkt_cap = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[sel_mc]
        with fc6: ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter", on_change=save_pv, args=("pv_ema_filter", "saved_pv_ema"))

    with col_calculator:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: c_strike = st.number_input("Strike Price", min_value=0.01, value=st.session_state.saved_calc_strike, step=1.0, format="%.2f", key="calc_strike", on_change=save_pv, args=("calc_strike", "saved_calc_strike"))
        with cc2: c_premium = st.number_input("Premium", min_value=0.00, value=st.session_state.saved_calc_premium, step=0.05, format="%.2f", key="calc_premium", on_change=save_pv, args=("calc_premium", "saved_calc_premium"))
        with cc3: c_expiry = st.date_input("Expiration", value=st.session_state.saved_calc_expiry, key="calc_expiry", on_change=save_pv, args=("calc_expiry", "saved_calc_expiry"))
        
        dte = max(0, (c_expiry - date.today()).days)
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0
        
        cc4, cc5, cc6 = st.columns(3)
        cc4.metric("Annualised", f"{annual_ret:.1f}%")
        cc5.metric("Cash on Cash", f"{coc_ret:.1f}%")
        cc6.metric("DTE", str(dte))

    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: return

    # Identify Risk Reversals
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    pools = {t: d_range[d_range["Order Type"] == t].copy() for t in ["Calls Bought", "Puts Sold", "Puts Bought"]}
    
    for p in pools.values(): p['occ'] = p.groupby(keys).cumcount()
    
    rr_matches = pd.merge(pools["Calls Bought"], pools["Puts Sold"], on=keys + ['occ'], suffixes=('_c', '_p'))
    if not rr_matches.empty:
        # Reconstruct RR dataframe
        rr_c = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_c', 'Strike_c']].rename(columns={'Dollars_c': 'Dollars', 'Strike_c': 'Strike'})
        rr_c['Pair_ID'], rr_c['Pair_Side'] = rr_matches.index, 0
        rr_p = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_p', 'Strike_p']].rename(columns={'Dollars_p': 'Dollars', 'Strike_p': 'Strike'})
        rr_p['Pair_ID'], rr_p['Pair_Side'] = rr_matches.index, 1
        df_rr = pd.concat([rr_c, rr_p])
        df_rr['Strike'] = df_rr['Strike'].apply(clean_strike_fmt)
        
        # Filter matches out of original pools
        match_keys = keys + ['occ']
        rr_matches['_remove'] = True
        for k in ["Calls Bought", "Puts Sold"]:
             m = pools[k].merge(rr_matches[match_keys + ['_remove']], on=match_keys, how='left')
             pools[k] = m[m['_remove'].isna()].drop(columns=['_remove'])
    else:
        df_rr = pd.DataFrame(columns=['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars', 'Strike', 'Pair_ID', 'Pair_Side'])

    def apply_f(f):
        if f.empty: return f
        if ticker_filter: f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
        f = f[f["Dollars"] >= min_notional]
        if not f.empty and (min_mkt_cap > 0 or ema_filter == "Yes"):
            valid = set(f["Symbol"].unique())
            if min_mkt_cap > 0:
                caps = fetch_market_caps_batch(list(valid))
                valid = {s for s in valid if caps.get(s, 0) >= min_mkt_cap}
            if ema_filter == "Yes":
                techs = fetch_technicals_batch(list(valid))
                valid = {s for s in valid if techs.get(s, (None, None))[2] is None or (techs[s][0] and techs[s][2] and techs[s][0] > techs[s][2])}
            f = f[f["Symbol"].isin(valid)]
        return f

    df_rr_f = apply_f(df_rr)
    pools = {k: apply_f(v) for k, v in pools.items()}

    def get_p(data, is_rr=False):
        if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
        sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        if is_rr: piv = data.merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
        else: piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        piv["Expiry_Table"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
        piv["Symbol"] = np.where(piv["Symbol"] == piv["Symbol"].shift(1), "", piv["Symbol"])
        return piv[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    cols = st.columns(3)
    fmt = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}
    for i, (name, key) in enumerate([("Calls Bought", "Calls Bought"), ("Puts Sold", "Puts Sold"), ("Puts Bought", "Puts Bought")]):
        with cols[i]:
            st.subheader(name)
            t = get_p(pools[key])
            if not t.empty: st.dataframe(t.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(t, max_rows=50), column_config=COLUMN_CONFIG_PIVOT)
    
    st.subheader("Risk Reversals")
    t_rr = get_p(df_rr_f, is_rr=True)
    if not t_rr.empty: st.dataframe(t_rr.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(t_rr, max_rows=50), column_config=COLUMN_CONFIG_PIVOT)
    else: st.caption("No matched RR pairs found.")

def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    st.markdown("""<style>.top-note { color: #888888; font-size: 14px; margin-bottom: 2px; } .footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; } [data-testid="stDataFrame"] th { font-weight: 900 !important; }</style>""", unsafe_allow_html=True)
    
    dataset_map = DATA_KEYS_PARQUET
    options = list(dataset_map.keys())

    tab_div, tab_pct, tab_bot = st.tabs(["📉 Divergences", "🔢 Percentiles", "🤖 Backtester"])

    with tab_bot:
        st.markdown('<div class="light-note">ℹ️ If this is buggy, just go back to the RSI Divergences tab and back here.</div>', unsafe_allow_html=True)
        c_left, c_right = st.columns([1, 6])
        with c_left:
            ticker = st.text_input("Ticker", value="NFLX", key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=10, value=10)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            rsi_metric_container = st.empty()
        
        if ticker:
            with st.spinner(f"Crunching numbers for {ticker}..."):
                df = get_ticker_technicals(ticker, load_ticker_map()) or fetch_yahoo_data(ticker)
                if df is None or df.empty: st.error("Data not found.")
                else:
                    df = df.sort_values(by="DATE").reset_index(drop=True)
                    if "RSI" not in df.columns: # fallback calc
                         delta = df["CLOSE"].diff()
                         rs = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                         df["RSI"] = 100 - (100 / (1 + rs))
                    
                    df = df[df["DATE"] >= (df["DATE"].max() - timedelta(days=365*lookback_years))].reset_index(drop=True)
                    curr_rsi = df["RSI"].iloc[-1]
                    rsi_metric_container.markdown(f"""<div style="margin-top: 10px; font-size: 0.9rem; color: #666;">Current RSI</div><div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 15px;">{curr_rsi:.2f}</div>""", unsafe_allow_html=True)
                    
                    matches = df[(df["RSI"] >= curr_rsi - rsi_tol) & (df["RSI"] <= curr_rsi + rsi_tol)]
                    full_close = df["CLOSE"].values
                    match_indices = matches.index.values
                    total_len = len(full_close)
                    
                    results = []
                    for p in [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]:
                        valid_indices = match_indices[match_indices + p < total_len]
                        if len(valid_indices) == 0: continue
                        returns = (full_close[valid_indices + p] - full_close[valid_indices]) / full_close[valid_indices]
                        wins, losses = returns[returns > 0], returns[returns < 0]
                        pf = np.sum(wins) / np.abs(np.sum(losses)) if np.sum(losses) != 0 else (999.0 if np.sum(wins) > 0 else 0.0)
                        results.append({"Days": p, "Profit Factor": pf, "Win Rate": np.mean(returns > 0)*100, "EV": np.mean(returns)*100, "Count": len(valid_indices)})

                    with c_right:
                        st.dataframe(pd.DataFrame(results).style.format({"Win Rate": "{:.1f}%", "EV": "{:+.2f}%", "Profit Factor": "{:.2f}"}).map(lambda x: f'color: {"#71d28a" if x>0 else "#f29ca0"}; font-weight: bold;', subset=["EV"]), 
                                     use_container_width=False, hide_index=True)

    with tab_div:
        data_option_div = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_div_pills")
        with st.expander("ℹ️ Page Notes: Divergence Strategy Logic"):
            st.write("Scans for pivots over 25 periods. Optimizes holding period using historical signals.")

        if data_option_div:
            try:
                master = load_parquet_and_clean(dataset_map[data_option_div])
                if master is not None:
                    t_col = next((c for c in master.columns if c in ['TICKER', 'SYMBOL']), None)
                    all_tickers = sorted(master[t_col].unique())
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)})"):
                        sq_div = st.text_input("Filter...", key="rsi_div_ft").upper()
                        min_n_div = st.number_input("Minimum N", 0, value=0, key="rsi_div_n")
                        cols = st.columns(6)
                        for i, t in enumerate([t for t in all_tickers if sq_div in t]): cols[i % 6].write(t)

                    raw_results_div = []
                    progress = st.progress(0, text="Scanning...")
                    groups = list(master.groupby(t_col))
                    
                    # --- OPTIMIZATION: Loop is now lighter due to vectorized stats ---
                    for i, (ticker, group) in enumerate(groups):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None: raw_results_div.extend(find_divergences(d_d, ticker, 'Daily', min_n_div))
                        if d_w is not None: raw_results_div.extend(find_divergences(d_w, ticker, 'Weekly', min_n_div))
                        if i % 10 == 0: progress.progress((i + 1) / len(groups))
                    progress.empty()
                    
                    if raw_results_div:
                        df_res = pd.DataFrame(raw_results_div).sort_values(by='Signal_Date_ISO', ascending=False)
                        for tf in ['Daily', 'Weekly']:
                            for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                st.subheader(f"{emoji} {tf} {s_type}")
                                subset = df_res[(df_res['Type']==s_type) & (df_res['Timeframe']==tf)].groupby('Ticker').head(1)
                                if not subset.empty:
                                    st.dataframe(subset[["Ticker", "Tags", "Date_Display", "RSI_Display", "Price_Display", "Last_Close", "Best Period", "Profit Factor", "Win Rate", "EV", "N"]], hide_index=True, use_container_width=True)
                                else: st.info("No signals.")
                    else: st.warning("No signals found.")
            except Exception as e: st.error(f"Analysis failed: {e}")

    with tab_pct:
        data_option_pct = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_pct_pills")
        
        if data_option_pct:
            try:
                master = load_parquet_and_clean(dataset_map[data_option_pct])
                if master is not None:
                    t_col = next((c for c in master.columns if c in ['TICKER', 'SYMBOL']), None)
                    all_tickers = sorted(master[t_col].unique())
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)})"):
                         sq_pct = st.text_input("Filter...", key="pct_ft").upper()
                         
                    c1, c2, c3, c4 = st.columns(4)
                    in_low = c1.number_input("Low Pct", 1, 49, 10)
                    in_high = c2.number_input("High Pct", 51, 99, 90)
                    show_filter = c3.selectbox("Show", ["Everything", "Leaving High", "Leaving Low"])
                    min_n_pct = c4.number_input("Min N", 1, value=1)
                    
                    raw_results_pct = []
                    progress = st.progress(0, text="Scanning...")
                    groups = list(master.groupby(t_col))
                    
                    for i, (ticker, group) in enumerate(groups):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None: raw_results_pct.extend(find_rsi_percentile_signals(d_d, ticker, in_low/100, in_high/100, min_n_pct, None, 'Daily'))
                        if d_w is not None: raw_results_pct.extend(find_rsi_percentile_signals(d_w, ticker, in_low/100, in_high/100, min_n_pct, None, 'Weekly'))
                        if i % 10 == 0: progress.progress((i + 1) / len(groups))
                    progress.empty()

                    if raw_results_pct:
                        df_res = pd.DataFrame(raw_results_pct).sort_values(by='Date_Obj', ascending=False)
                        if show_filter == "Leaving High": df_res = df_res[df_res['Signal_Type'] == 'Bearish']
                        elif show_filter == "Leaving Low": df_res = df_res[df_res['Signal_Type'] == 'Bullish']
                        
                        st.dataframe(df_res[["Ticker", "Date", "Action", "RSI_Display", "Signal_Price", "Last_Close", "Best Period", "Profit Factor", "Win Rate", "EV", "N"]], hide_index=True, use_container_width=True)
                    else: st.info("No signals found.")
            except Exception as e: st.error(f"Analysis failed: {e}")

st.markdown("""<style>.block-container{padding-top:3.5rem;padding-bottom:1rem;}.zones-panel{padding:14px 0; border-radius:10px;}.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}.zone-wrapper{flex-grow: 1; position: relative; height: 24px; background-color: rgba(0,0,0,0.03); border-radius: 4px; overflow: hidden;}.zone-bar{position: absolute; left: 0; top: 0; bottom: 0; z-index: 1; border-radius: 3px; opacity: 0.65;}.zone-bull{background-color: #71d28a;}.zone-bear{background-color: #f29ca0;}.zone-value{position: absolute; right: 8px; top: 0; bottom: 0; display: flex; align-items: center; z-index: 2; font-size: 12px; font-weight: 700; color: #1f1f1f; white-space: nowrap; text-shadow: 0 0 4px rgba(255,255,255,0.8);}.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; white-space: nowrap; margin: 0 12px; z-index: 1; }.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; } [data-testid="stDataFrame"] th { font-weight: 900 !important; }</style>""", unsafe_allow_html=True)

try:
    df_global = load_and_clean_data(st.secrets["GSHEET_URL"])
    last_updated_date = df_global["Trade Date"].max().strftime("%d %b %y")
    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: run_rsi_scanner_app(df_global), title="RSI Scanner", icon="📈", url_path="rsi_scanner"), 
    ])
    st.sidebar.caption("🖥️ Everything is best viewed with a wide desktop monitor in light mode.")
    st.sidebar.caption(f"📅 **Last Updated:** {last_updated_date}")
    pg.run()
except Exception as e: st.error(f"Error initializing dashboard: {e}")
