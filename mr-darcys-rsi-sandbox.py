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
from io import StringIO
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
# MIN_N_THRESHOLD is now dynamic based on user input
URL_TICKER_MAP_DEFAULT = "https://drive.google.com/file/d/1MlVp6yF7FZjTdRFMpYCxgF-ezyKvO4gG/view?usp=sharing"

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
        
        sma200 = float(h_full["Close"].rolling(window=200).mean().iloc[-1]) if len(h_full) >= 200 else None
        
        h_recent = h_full.iloc[-60:].copy() if len(h_full) > 60 else h_full.copy()
        
        if len(h_recent) == 0: return None, None, None, None, None
        
        close = h_recent["Close"]
        spot_val = float(close.iloc[-1])
        ema8  = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        
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
        
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match: confirm_token = match.group(1)

        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
        
        if response.text.strip().startswith("<!DOCTYPE html>"): return "HTML_ERROR"
            
        return StringIO(response.text)
    except Exception as e:
        return None

def load_dataset_config():
    try:
        if "URL_CONFIG" not in st.secrets:
            return {"Darcy Data": "URL_DARCY", "S&P 100 Data": "URL_SP100"}
        config_url = st.secrets["URL_CONFIG"]
        buffer = get_confirmed_gdrive_data(config_url)
        if buffer and buffer != "HTML_ERROR":
            lines = buffer.getvalue().splitlines()
            config_dict = {}
            for line in lines:
                if ',' in line:
                    name, key = line.split(',')
                    config_dict[name.strip()] = key.strip()
            return config_dict
    except Exception as e:
        st.error(f"Error loading config file: {e}")
    return {"Darcy Data": "URL_DARCY"}

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

def fetch_and_prepare_ai_context(url, name, limit=90):
    try:
        buffer = get_confirmed_gdrive_data(url)
        if buffer and buffer != "HTML_ERROR":
            df = pd.read_csv(buffer)
            df_recent = df.tail(limit)
            csv_str = df_recent.to_csv(index=False)
            return f"\n\n--- DATASET: {name} ---\n{csv_str}"
    except Exception as e:
        return f"\n[Error loading {name}: {e}]"
    return ""

def calculate_optimal_signal_stats(history_indices, price_array, current_idx):
    """
    Looks back at all indices in history_indices that are LESS than current_idx.
    Calculates forward returns for 10, 30, 60, 90, 180 periods.
    Returns the stats for the Optimal Period (Winner based on Profit Factor).
    """
    valid_hist_indices = [idx for idx in history_indices if idx < current_idx]
    
    if not valid_hist_indices:
        return None

    periods = [10, 30, 60, 90, 180]
    best_pf = -1.0
    best_stats = None
    
    total_len = len(price_array)

    for p in periods:
        returns = []
        for idx in valid_hist_indices:
            exit_idx = idx + p
            if exit_idx < total_len:
                entry_p = price_array[idx]
                exit_p = price_array[exit_idx]
                if entry_p > 0:
                    ret = (exit_p - entry_p) / entry_p
                    returns.append(ret)
        
        if not returns:
            continue
            
        returns_np = np.array(returns)
        n = len(returns_np)
        
        wins = returns_np[returns_np > 0]
        losses = returns_np[returns_np < 0]
        
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        
        # Profit Factor: Gross Wins / Gross Losses
        # Handle Division by Zero
        if gross_loss == 0:
            pf = 999.0 if gross_win > 0 else 0.0
        else:
            pf = gross_win / gross_loss
            
        win_rate = (len(wins) / n) * 100
        avg_ret = np.mean(returns_np) * 100
        
        if pf > best_pf:
            best_pf = pf
            best_stats = {
                "Best Period": f"{p}d",
                "Profit Factor": pf,
                "Win Rate": win_rate,
                "EV": avg_ret,
                "N": n
            }
            
    return best_stats

# --- NEW HELPERS FOR TOP 3 ---

def get_optimal_rsi_duration(history_df, current_rsi, tolerance=2.0):
    if history_df is None or len(history_df) < 100:
        return 30, "Default (No Hist)"

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
    # Standardize column names to remove spaces/dashes and make uppercase
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    # Identify RSI and EMA columns for Daily data
    d_rsi = next((c for c in cols if 'RSI' in c and 'W_' not in c), 'RSI_14')
    d_ema8 = 'EMA_8' if 'EMA_8' in cols else None
    d_ema21 = 'EMA_21' if 'EMA_21' in cols else None
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Build Daily Dataframe
    needed_cols = [close_col, vol_col, high_col, low_col]
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8: needed_cols.append(d_ema8)
    if d_ema21: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    
    # Map CSV names to internal script names used by find_divergences
    rename_dict = {close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low'}
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8: rename_dict[d_ema8] = 'EMA8'   # Maps EMA_8 -> EMA8
    if d_ema21: rename_dict[d_ema21] = 'EMA21' # Maps EMA_21 -> EMA21
    
    df_d.rename(columns=rename_dict, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    # Handle missing RSI calculation if needed
    if 'RSI' not in df_d.columns:
        delta = df_d['Price'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df_d['RSI'] = 100 - (100 / (1 + rs))
        
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # Identify Weekly columns
    w_close, w_vol, w_rsi = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_high, w_low = 'W_HIGH', 'W_LOW'
    w_ema8, w_ema21 = 'W_EMA_8', 'W_EMA_21'
    
    # Build Weekly Dataframe
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low, w_rsi]):
        cols_w = [w_close, w_vol, w_high, w_low, w_rsi]
        if w_ema8 in df.columns: cols_w.append(w_ema8)
        if w_ema21 in df.columns: cols_w.append(w_ema21)
        
        df_w = df[cols_w].copy()
        
        # Map Weekly CSV names to internal names
        w_rename = {w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low', w_rsi: 'RSI'}
        if w_ema8 in df.columns: w_rename[w_ema8] = 'EMA8'   # Maps W_EMA_8 -> EMA8
        if w_ema21 in df.columns: w_rename[w_ema21] = 'EMA21' # Maps W_EMA_21 -> EMA21
        
        df_w.rename(columns=w_rename, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: 
        df_w = None
        
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe, min_n=0):
    divergences = []
    n_rows = len(df_tf)
    
    # 1. Setup & Safety Check
    if n_rows < DIVERGENCE_LOOKBACK + 1: return divergences
    
    rsi_vals = df_tf['RSI'].values
    low_vals = df_tf['Low'].values
    high_vals = df_tf['High'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    close_vals = df_tf['Price'].values
    
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': 
             return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    
    # ---------------------------------------------------------
    # PASS 1: SCAN FULL HISTORY
    # ---------------------------------------------------------
    bullish_signal_indices = []
    bearish_signal_indices = []
    potential_signals = [] 

    start_search = DIVERGENCE_LOOKBACK
    
    for i in range(start_search, n_rows):
        p2_rsi = rsi_vals[i]
        p2_low = low_vals[i]
        p2_high = high_vals[i]
        p2_vol = vol_vals[i]
        p2_volsma = vol_sma_vals[i]
        
        lb_start = i - DIVERGENCE_LOOKBACK
        lb_rsi = rsi_vals[lb_start:i]
        lb_low = low_vals[lb_start:i]
        lb_high = high_vals[lb_start:i]
        
        is_vol_high = int(p2_vol > (p2_volsma * 1.5)) if not np.isnan(p2_volsma) else 0
        
        # Bullish Logic
        if p2_low < np.min(lb_low):
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
                        bullish_signal_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bullish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})
        
        # Bearish Logic
        elif p2_high > np.max(lb_high):
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
                        bearish_signal_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bearish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})

    # ---------------------------------------------------------
    # PASS 2: REPORT & BACKTEST
    # ---------------------------------------------------------
    display_threshold_idx = n_rows - SIGNAL_LOOKBACK_PERIOD
    
    for sig in potential_signals:
        i = sig["index"]
        
        # Limit display to the last 25 periods
        if i < display_threshold_idx:
            continue

        s_type = sig["type"]
        idx_p1_abs = sig["p1_idx"]
        latest_p = df_tf.iloc[-1] 
        
        # --- FIXED TAGS GENERATION ---
        tags = []
        row_at_sig = df_tf.iloc[i] 
        curr_price = row_at_sig['Price']
        
        ema8_val = row_at_sig.get('EMA8') 
        ema21_val = row_at_sig.get('EMA21')

        if s_type == 'Bullish':
            if ema8_val is not None and curr_price >= ema8_val: tags.append(f"EMA{EMA8_PERIOD}")
            if ema21_val is not None and curr_price >= ema21_val: tags.append(f"EMA{EMA21_PERIOD}")
        else: # Bearish
            if ema8_val is not None and curr_price <= ema8_val: tags.append(f"EMA{EMA8_PERIOD}")
            if ema21_val is not None and curr_price <= ema21_val: tags.append(f"EMA{EMA21_PERIOD}")
        
        if sig["vol_high"]: tags.append("VOL_HIGH")
        if vol_vals[i] > vol_vals[idx_p1_abs]: tags.append("VOL_GROW")
        
        # Display Data
        sig_date_iso = get_date_str(i, '%Y-%m-%d')
        date_display = f"{get_date_str(idx_p1_abs, '%b %d')} → {get_date_str(i, '%b %d')}"
        rsi_display = f"{int(round(rsi_vals[idx_p1_abs]))} {'↗' if rsi_vals[i] > rsi_vals[idx_p1_abs] else '↘'} {int(round(rsi_vals[i]))}"
        
        price_p1 = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
        price_p2 = low_vals[i] if s_type=='Bullish' else high_vals[i]
        price_display = f"${price_p1:,.2f} ↗ ${price_p2:,.2f}" if price_p2 > price_p1 else f"${price_p1:,.2f} ↘ ${price_p2:,.2f}"

        # Optimal Period Calculation
        hist_list = bullish_signal_indices if s_type == 'Bullish' else bearish_signal_indices
        best_stats = calculate_optimal_signal_stats(hist_list, close_vals, i)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
        
        if best_stats["N"] < min_n:
            continue
            
        divergences.append({
            'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 
            'Tags': tags, 'Signal_Date_ISO': sig_date_iso, 'Date_Display': date_display,
            'RSI_Display': rsi_display, 'Price_Display': price_display, 'Last_Close': f"${latest_p['Price']:,.2f}", 
            'Best Period': best_stats['Best Period'], 'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'], 'EV': best_stats['EV'], 'N': best_stats['N']
        })
            
    return divergences

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, min_n=1, filter_date=None):
    signals = []
    if len(df) < 200: return signals
    
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(pct_low)
    p90 = hist_df['RSI'].quantile(pct_high)
    
    rsi_vals = hist_df['RSI'].values
    price_vals = hist_df['Price'].values
    
    # 1. Identify ALL Signal Indices in the 10-year history
    bullish_signal_indices = []
    bearish_signal_indices = []
    
    # Iterate through history
    # Start from index 1
    for i in range(1, len(hist_df)):
        prev_rsi = rsi_vals[i-1]
        curr_rsi = rsi_vals[i]
        
        if prev_rsi < p10 and curr_rsi >= (p10 + 1.0):
            bullish_signal_indices.append(i)
        elif prev_rsi > p90 and curr_rsi <= (p90 - 1.0):
            bearish_signal_indices.append(i)
            
    # 2. Report signals (filtered by latest date logic usually done outside, but we can do it here)
    # The scan_window logic was: df.iloc[-(periods_to_scan+1):]
    # But user wants "Minimum N to be included" which implies we look at all potential signals
    # that meet the date filter.
    
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
        best_stats = calculate_optimal_signal_stats(hist_list, price_vals, i)
        
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
             
        if best_stats["N"] < min_n:
            continue
            
        rsi_disp = f"{thresh_val:.0f} ↗ {curr_rsi_val:.0f}" if is_bullish else f"{thresh_val:.0f} ↘ {curr_rsi_val:.0f}"
        action_str = "Leaving Low" if is_bullish else "Leaving High"
        
        signals.append({
            'Ticker': ticker,
            'Date': curr_row.name.strftime('%b %d'),
            'Date_Obj': curr_date,
            'Action': action_str,
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
        
        df = df.reset_index()
        
        date_col_name = df.columns[0]
        df = df.rename(columns={date_col_name: "DATE"})
        
        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"])

        if df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
            
        df = df.rename(columns={"Close": "CLOSE", "Volume": "VOLUME", "High": "HIGH", "Low": "LOW", "Open": "OPEN"})
        df.columns = [c.upper() for c in df.columns]
        
        delta = df["CLOSE"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI_14"] = df["RSI"]
        
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
    # Padding at the bottom
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
            
            for i, t in enumerate(bull_list):
                prog_bar.progress((i+1)/len(bull_list), text=f"Checking {t}...")
                t_df = get_ticker_technicals(t, ticker_map)
                
                if t_df is None or t_df.empty:
                    t_df = fetch_yahoo_data(t)

                if t_df is not None:
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

    st.markdown("""
        <style>
            .st-key-calc_out_ann input, .st-key-calc_out_coc input, .st-key-calc_out_dte input {
                background-color: rgba(113, 210, 138, 0.1) !important;
                color: #71d28a !important;
                border: 1px solid #71d28a !important;
                font-weight: 700 !important;
                pointer-events: none !important;
                cursor: default !important;
            }
        </style>
    """, unsafe_allow_html=True)

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

    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    
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
                # Optimization: Batch fetch instead of sequential calls
                batch_results = fetch_technicals_batch(list(valid_symbols))
                valid_symbols = {
                    s for s in valid_symbols 
                    if batch_results.get(s, (None, None))[2] is None or # Retain original behavior (fail open)
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
    else: st.caption("No matched RR pairs found.")

def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    
    st.markdown("""
        <style>
        .top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; }
        .footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; }
        [data-testid="stDataFrame"] th { font-weight: 900 !important; }
        </style>
        """, unsafe_allow_html=True)
    
    dataset_map = load_dataset_config()
    options = list(dataset_map.keys())

    tab_div, tab_pct, tab_bot = st.tabs(["📉 Divergences", "🔢 Percentiles", "🤖 Backtester"])

    with tab_bot:
        st.markdown('<div class="light-note" style="margin-bottom: 15px;">ℹ️ If this is buggy, just go back to the RSI Divergences tab and back here and it will work.</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ Page Notes: Backtester Logic"):
            st.markdown("""
            * **Data Source**: Unlike the Divergences and Percentile tabs (which use limited ~3yr history files), this tab pulls **Complete Price History** via Yahoo Finance or the full Ticker Map file.
            * **Methodology**: Calculates forward returns for all historical periods matching the criteria.
            * **Metrics**:
                * **Profit Factor**: Gross Wins / Gross Losses.
                * **Win Rate**: Percentage of trades that closed positive.
                * **EV**: Average Return % per trade.
            """)

        c_left, c_right = st.columns([1, 6])
        
        with c_left:
            # FIX 1: ADDED KEY TO TEXT INPUT TO PREVENT TAB RESET
            ticker = st.text_input("Ticker", value="NFLX", help="Enter a symbol (e.g., TSLA, NVDA)", key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=10, value=10)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
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
                    rsi_col = next((c for c in df.columns if 'RSI' in c), None)

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
                        periods = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
                        
                        for p in periods:
                            valid_indices = match_indices[match_indices + p < total_len]
                            
                            if len(valid_indices) == 0:
                                results.append({"Days": p, "Win Rate": np.nan, "EV": np.nan, "Count": 0, "Profit Factor": np.nan})
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
                            
                            results.append({
                                "Days": p, 
                                "Win Rate": win_rate, 
                                "EV": avg_ret, 
                                "Count": len(valid_indices),
                                "Profit Factor": pf
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

                                st.dataframe(
                                    res_df.style
                                    .format({"Win Rate": format_wr, "EV": format_func, "Profit Factor": format_pf})
                                    .map(highlight_ret, subset=["EV"])
                                    .apply(highlight_best, axis=1)
                                    .set_table_styles([dict(selector="th", props=[("font-weight", "bold"), ("background-color", "#f0f2f6")])]),
                                    use_container_width=False,
                                    column_config={
                                        "Days": st.column_config.NumberColumn("Days", width=60),
                                        "Profit Factor": st.column_config.NumberColumn("Profit Factor", width=80),
                                        "Win Rate": st.column_config.TextColumn("Win Rate", width=80),
                                        "EV": st.column_config.TextColumn("EV", width=80),
                                        "Count": st.column_config.NumberColumn("Count", width=60)
                                    },
                                    hide_index=True
                                )

                        st.markdown("<br><br><br>", unsafe_allow_html=True)

    with tab_div:
        data_option_div = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_div_pills")
        
        with st.expander("ℹ️ Page Notes: Divergence Strategy Logic"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.markdown('<div class="footer-header">📉 SIGNAL LOGIC</div>', unsafe_allow_html=True)
                st.markdown(f"""
                * **Identification**: Scans for **True Pivots** over a **{SIGNAL_LOOKBACK_PERIOD}-period** window.
                * **Divergence**: 
                    * **Bullish**: Price makes a Lower Low, but RSI makes a Higher Low.
                    * **Bearish**: Price makes a Higher High, but RSI makes a Lower High.
                * **Invalidation**: If RSI crosses the 50 midline between pivots, the setup is reset.
                """)
            with f_col2:
                st.markdown('<div class="footer-header">🔮 SIGNAL-BASED OPTIMIZATION</div>', unsafe_allow_html=True)
                st.markdown(f"""
                * **New Methodology**: Instead of just looking at RSI levels, this tool looks back at **Every Historical Occurrence** of the specific signal type (e.g., Daily Bullish Divergence) for the ticker.
                * **Optimization Loop**: It calculates the forward returns for **10, 30, 60, 90, and 180** trading days for each historical signal.
                * **Selection**: It compares these 5 holding periods and selects the **Optimal Time Period** based on the highest **Profit Factor**.
                * **Data Constraint**: Note that the input files for this specific scanner currently contain ~3 years of price history.
                """)
            with f_col3:
                st.markdown('<div class="footer-header">🏷️ TAGS</div>', unsafe_allow_html=True)
                st.markdown(f"""
                * **EMA{EMA8_PERIOD}**: Bullish (Price > EMA8) or Bearish (Price < EMA8).
                * **EMA{EMA21_PERIOD}**: Bullish (Price > EMA21) or Bearish (Price < EMA21).
                * **VOL_HIGH**: Signal candle volume is > 150% of the 30-day average.
                * **VOL_GROW**: Volume on the second pivot (P2) is higher than the first pivot (P1).
                """)
        
        if data_option_div:
            try:
                target_url = st.secrets[dataset_map[data_option_div]]
                csv_buffer = get_confirmed_gdrive_data(target_url)
                
                if csv_buffer and csv_buffer != "HTML_ERROR":
                    master = pd.read_csv(csv_buffer)
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    if date_col_raw:
                        max_dt_obj = pd.to_datetime(master[date_col_raw]).max()
                        target_highlight_daily = max_dt_obj.strftime('%Y-%m-%d')
                        days_to_subtract = max_dt_obj.weekday() + (7 if max_dt_obj.weekday() < 4 else 0)
                        target_highlight_weekly = (max_dt_obj - timedelta(days=days_to_subtract)).strftime('%Y-%m-%d')
                    
                    all_tickers = sorted(master[t_col].unique())
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                        sq_div = st.text_input("Filter...", key="rsi_div_filter_ticker").upper()
                        min_n_div = st.number_input("Minimum N", min_value=0, value=0, step=1, key="rsi_div_min_n")
                        ft_div = [t for t in all_tickers if sq_div in t]
                        cols = st.columns(6)
                        for i, ticker in enumerate(ft_div): cols[i % 6].write(ticker)

                    raw_results_div = []
                    progress_bar = st.progress(0, text="Scanning Divergences...")
                    grouped = master.groupby(t_col)
                    grouped_list = list(grouped)
                    total_groups = len(grouped_list)
                    
                    for i, (ticker, group) in enumerate(grouped_list):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None: raw_results_div.extend(find_divergences(d_d, ticker, 'Daily', min_n=min_n_div))
                        if d_w is not None: raw_results_div.extend(find_divergences(d_w, ticker, 'Weekly', min_n=min_n_div))
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    
                    progress_bar.empty()
                    
                    if raw_results_div:
                        res_div_df = pd.DataFrame(raw_results_div).sort_values(by='Signal_Date_ISO', ascending=False)
                        consolidated = res_div_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                        
                        for tf in ['Daily', 'Weekly']:
                            target_highlight = target_highlight_weekly if tf == 'Weekly' else target_highlight_daily
                            date_header = "Week Δ" if tf == 'Weekly' else "Day Δ"
                            
                            for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                st.subheader(f"{emoji} {tf} {s_type} Signals")
                                tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                                
                                price_header = "Low Price Δ" if s_type == 'Bullish' else "High Price Δ"
                                
                                if not tbl_df.empty:
                                    def style_div_df(df_in):
                                        def highlight_row(row):
                                            styles = [''] * len(row)
                                            # Highlight Date
                                            if row['Signal_Date_ISO'] == target_highlight:
                                                idx = df_in.columns.get_loc('Date_Display')
                                                styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                            
                                            # Color EV Numeric Cells
                                            if 'EV' in df_in.columns:
                                                val = row['EV']
                                                if pd.notnull(val) and val != 0:
                                                    is_green = val > 0
                                                    bg = 'background-color: #e6f4ea; color: #1e7e34;' if is_green else 'background-color: #fce8e6; color: #c5221f;'
                                                    idx = df_in.columns.get_loc('EV')
                                                    styles[idx] = f'{bg} font-weight: 500;'
                                            return styles
                                        return df_in.style.apply(highlight_row, axis=1)

                                    st.dataframe(
                                        style_div_df(tbl_df),
                                        column_config={
                                            "Ticker": st.column_config.TextColumn("Ticker"),
                                            "Tags": st.column_config.ListColumn("Tags", width="medium"), # ListColumn for Bubbles
                                            "Date_Display": st.column_config.TextColumn(date_header),
                                            "RSI_Display": st.column_config.TextColumn("RSI Δ"),
                                            "Price_Display": st.column_config.TextColumn(price_header),
                                            "Last_Close": st.column_config.TextColumn("Last Close"),
                                            "Best Period": st.column_config.TextColumn("Best Period"),
                                            "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                                            "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                                            "EV": st.column_config.NumberColumn("EV", format="%.1f%%"),
                                            "N": st.column_config.NumberColumn("N"),
                                            # Hide raw cols
                                            "Signal_Date_ISO": None, "Type": None, "Timeframe": None
                                        },
                                        hide_index=True,
                                        use_container_width=True,
                                        height=get_table_height(tbl_df, max_rows=50) # <--- INCREASED HEIGHT
                                    )
                                else: st.info("No signals.")
                    else: st.warning("No Divergence signals found.")
            except Exception as e: st.error(f"Analysis failed: {e}")

    with tab_pct:
        data_option_pct = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_pct_pills")
        
        with st.expander("ℹ️ Page Notes: Percentile Strategy Logic"):
             st.markdown("""
            * **Signal Trigger**: RSI crosses **ABOVE Low Percentile** (Leaving Low) or **BELOW High Percentile** (Leaving High).
            * **Signal-Based Optimization**: Instead of matching RSI values, this backtester finds all historical instances where the stock "Left the Low/High" and calculates performance.
            * **Optimization Loop**: Calculates returns for **10, 30, 60, 90, 180** days and selects the Winner based on **Profit Factor**.
            * **Data Constraint**: Note that the input files for this specific scanner currently contain ~3 years of price history.
            """)
        
        if data_option_pct:
            try:
                target_url = st.secrets[dataset_map[data_option_pct]]
                csv_buffer = get_confirmed_gdrive_data(target_url)
                
                if csv_buffer and csv_buffer != "HTML_ERROR":
                    master = pd.read_csv(csv_buffer)
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    max_date_in_set = None
                    if date_col_raw:
                        max_dt_obj = pd.to_datetime(master[date_col_raw]).max()
                        max_date_in_set = max_dt_obj.date()

                    all_tickers = sorted(master[t_col].unique())
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                        sq_pct = st.text_input("Filter...", key="rsi_pct_filter_ticker").upper()
                        ft_pct = [t for t in all_tickers if sq_pct in t]
                        cols = st.columns(6)
                        for i, ticker in enumerate(ft_pct): cols[i % 6].write(ticker)

                    c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)
                    with c_p1: in_low = st.number_input("RSI Low Percentile (%)", min_value=1, max_value=49, value=10, step=1, key="rsi_pct_low")
                    with c_p2: in_high = st.number_input("RSI High Percentile (%)", min_value=51, max_value=99, value=90, step=1, key="rsi_pct_high")
                    with c_p3: show_filter = st.selectbox("Actions to Show", ["Everything", "Leaving High", "Leaving Low"], index=0, key="rsi_pct_show")
                    
                    if not df_global.empty and "Trade Date" in df_global.columns:
                        ref_date = df_global["Trade Date"].max().date()
                    else:
                        ref_date = date.today()
                    default_start = ref_date - timedelta(days=14)
                    
                    with c_p4: filter_date = st.date_input("Latest Date", value=default_start, key="rsi_pct_date")
                    with c_p5: min_n_pct = st.number_input("Minimum N", min_value=1, value=1, step=1, key="rsi_pct_min_n")

                    raw_results_pct = []
                    progress_bar = st.progress(0, text="Scanning Percentiles...")
                    grouped = master.groupby(t_col)
                    grouped_list = list(grouped)
                    total_groups = len(grouped_list)
                    
                    for i, (ticker, group) in enumerate(grouped_list):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None:
                            raw_results_pct.extend(find_rsi_percentile_signals(d_d, ticker, pct_low=in_low/100.0, pct_high=in_high/100.0, min_n=min_n_pct, filter_date=filter_date))
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    
                    progress_bar.empty()

                    if raw_results_pct:
                        res_pct_df = pd.DataFrame(raw_results_pct).sort_values(by='Date_Obj', ascending=False)
                        
                        if show_filter == "Leaving High":
                            res_pct_df = res_pct_df[res_pct_df['Signal_Type'] == 'Bearish']
                        elif show_filter == "Leaving Low":
                            res_pct_df = res_pct_df[res_pct_df['Signal_Type'] == 'Bullish']
                            
                        # Date filter is now applied inside find_rsi_percentile_signals to handle the optimization loop correctly
                        
                        def style_pct_df(df_in):
                            def highlight_row(row):
                                styles = [''] * len(row)
                                # Highlight Date
                                if row['Date_Obj'] == max_date_in_set:
                                    idx = df_in.columns.get_loc('Date')
                                    styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                
                                # Color EV
                                if 'EV' in df_in.columns:
                                    val = row['EV']
                                    if pd.notnull(val) and val != 0:
                                        is_green = val > 0
                                        bg = 'background-color: #e6f4ea; color: #1e7e34;' if is_green else 'background-color: #fce8e6; color: #c5221f;'
                                        idx = df_in.columns.get_loc('EV')
                                        styles[idx] = f'{bg} font-weight: 500;'
                                return styles
                            return df_in.style.apply(highlight_row, axis=1)

                        st.dataframe(
                            style_pct_df(res_pct_df),
                            column_config={
                                "Ticker": st.column_config.TextColumn("Ticker"),
                                "Date": st.column_config.TextColumn("Date"),
                                "Action": st.column_config.TextColumn("Action"),
                                "RSI_Display": st.column_config.TextColumn("RSI Δ"),
                                "Signal_Price": st.column_config.TextColumn("Signal Close"),
                                "Last_Close": st.column_config.TextColumn("Last Close"), 
                                "Best Period": st.column_config.TextColumn("Best Period"),
                                "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                                "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                                "EV": st.column_config.NumberColumn("EV", format="%.1f%%"),
                                "N": st.column_config.NumberColumn("N"),
                                # Hide raw
                                "Signal_Type": None, "Date_Obj": None
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=get_table_height(res_pct_df, max_rows=50) # <--- INCREASED HEIGHT
                        )

                    else: st.info(f"No Percentile signals found (Crossing {in_low}th/{in_high}th percentile).")

            except Exception as e: st.error(f"Analysis failed: {e}")

def run_trade_ideas_app(df_global):
    st.title("🤖 AI Macro Portfolio Manager")
    st.caption("This module ingests the Darcy, SP100, NQ100, and Macro datasets to generate a comprehensive strategy report.")
    
    if st.button("Run Global Macro Scan"):
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Missing GOOGLE_API_KEY in secrets.toml")
        elif "URL_Prompt" not in st.secrets:
            st.error("Missing URL_Prompt in secrets.toml")
        else:
            try:
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                
                with st.spinner("Step 1/3: Fetching System Prompt..."):
                    prompt_buffer = get_confirmed_gdrive_data(st.secrets["URL_Prompt"])
                    if not prompt_buffer or prompt_buffer == "HTML_ERROR":
                        st.error("Failed to load prompt.")
                        st.stop()
                    system_prompt = prompt_buffer.getvalue()

                with st.spinner("Step 2/3: Ingesting Datasets..."):
                    context_data = ""
                    context_data += fetch_and_prepare_ai_context(st.secrets["URL_DARCY"], "DARCY WATCHLIST", 90)
                    context_data += fetch_and_prepare_ai_context(st.secrets["URL_SP100"], "S&P 100", 90)
                    context_data += fetch_and_prepare_ai_context(st.secrets["URL_NQ100"], "NASDAQ 100", 90)
                    context_data += fetch_and_prepare_ai_context(st.secrets["URL_MACRO"], "MACRO INDICATORS", 90)
                    
                    full_prompt = f"{system_prompt}\n\n==========\nLIVE DATA CONTEXT:\n{context_data}\n=========="

                with st.spinner("Step 3/3: AI Analysis (may take 60s)..."):
                    candidate_models = ["gemini-1.5-pro", "gemini-1.5-flash"]
                    response = None
                    for model_name in candidate_models:
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(full_prompt)
                            break 
                        except Exception:
                            continue 
                    
                    if response:
                        st.success("Analysis Complete!")
                        st.markdown("---")
                        st.markdown(response.text)
                    else:
                        st.error("AI models failed. Check API Quota.")
                    
            except Exception as e:
                st.error(f"AI Pipeline Failed: {e}")

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

/* BACKTESTER BOLD HEADERS */
[data-testid="stDataFrame"] th {
    font-weight: 900 !important;
}
</style>""", unsafe_allow_html=True)

try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)
    last_updated_date = df_global["Trade Date"].max().strftime("%d %b %y")

    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: run_rsi_scanner_app(df_global), title="RSI Scanner", icon="📈", url_path="rsi_scanner"), 
        st.Page(lambda: run_trade_ideas_app(df_global), title="Trade Ideas", icon="💡", url_path="trade_ideas"),
    ])

    st.sidebar.caption("🖥️ Everything is best viewed with a wide desktop monitor in light mode.")
    st.sidebar.caption(f"📅 **Last Updated:** {last_updated_date}")
    
    pg.run()
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
