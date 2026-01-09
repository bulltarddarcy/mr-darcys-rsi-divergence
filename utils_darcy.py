# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import requests
import re
from datetime import date, timedelta
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data, get_table_height

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# REFRESH TIME: 600 seconds = 10 minutes
CACHE_TTL = 600 

# --- DATA LOADERS & GOOGLE DRIVE UTILS ---

@st.cache_data(ttl=CACHE_TTL)
def get_parquet_config():
    config = {}
    try:
        raw_config = st.secrets.get("PARQUET_CONFIG", "")
        if not raw_config:
            st.error("⛔ CRITICAL ERROR: 'PARQUET_CONFIG' not found in Secrets.")
            st.stop()
            
        lines = [line.strip() for line in raw_config.strip().split('\n') if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[1] in st.secrets:
                config[parts[0]] = parts[1]

    except Exception as e:
        st.error(f"Failed to parse PARQUET_CONFIG: {e}")
        
    if not config:
        st.error("⛔ CRITICAL ERROR: No valid datasets mapped.")
        st.stop()
    return config


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading Dataset...")
def load_parquet_and_clean(key):
    clean_key = key.strip()
    if clean_key not in st.secrets: return None
        
    url = st.secrets[clean_key]
    
    try:
        buffer = get_gdrive_binary_data(url)
        if not buffer: return None
            
        content = buffer.getvalue()
        
        # Try Parquet first (faster), then CSV
        try:
            df = pd.read_parquet(BytesIO(content))
            if isinstance(df.index, pd.DatetimeIndex): df.reset_index(inplace=True)
            elif df.index.name and 'DATE' in str(df.index.name).upper(): df.reset_index(inplace=True)
        except Exception:
            try:
                # engine='c' is faster for CSV
                df = pd.read_csv(BytesIO(content), engine='c')
            except Exception:
                return None

        # Standard Cleaning
        df.columns = [str(c).strip() for c in df.columns]
        
        # Optimize Date Finding
        cols_upper = [c.upper() for c in df.columns]
        date_col = next((df.columns[i] for i, c in enumerate(cols_upper) if 'DATE' in c), None)
        
        if not date_col and 'index' in df.columns:
            date_col = 'index'

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.rename(columns={date_col: 'ChartDate'}).sort_values('ChartDate')
        else:
            return None
        
        if 'Price' not in df.columns:
            close_col = next((df.columns[i] for i, c in enumerate(cols_upper) if c == 'CLOSE'), None)
            if close_col: df['Price'] = df[close_col]
            
        return df
    except Exception as e:
        st.error(f"Error processing {clean_key}: {e}")
        return None


@st.cache_data(ttl=CACHE_TTL)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP")
        if not url: return {}

        buffer = get_gdrive_binary_data(url)
        if buffer:
            df = pd.read_csv(buffer, engine='c')
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                              df.iloc[:, 1].astype(str).str.strip()))
    except Exception:
        pass
    return {}

@st.cache_data(ttl=CACHE_TTL, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    try:
        # Use C engine for speed
        df = pd.read_csv(url, engine='c')
        
        # Optimize: Filter columns immediately
        want = {"Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"}
        existing_cols = [c for c in df.columns if c in want]
        df = df[existing_cols]
        
        # Vectorized string cleanup
        for c in ["Order Type", "Symbol", "Strike", "Expiry"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        if "Dollars" in df.columns and df["Dollars"].dtype == 'object':
             df["Dollars"] = pd.to_numeric(df["Dollars"].str.replace(r'[$,]', '', regex=True), errors="coerce").fillna(0.0)

        if "Contracts" in df.columns and df["Contracts"].dtype == 'object':
             df["Contracts"] = pd.to_numeric(df["Contracts"].str.replace(',', '', regex=False), errors="coerce").fillna(0)
        
        if "Trade Date" in df.columns:
            df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        
        if "Expiry" in df.columns:
            df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
            
        if "Strike (Actual)" in df.columns:
            df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
            
        if "Error" in df.columns:
            # Faster boolean masking
            mask = df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
            df = df[~mask]
            
        return df
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        return pd.DataFrame()

# --- MATH & TECHNICAL ANALYSIS ---

def add_technicals(df):
    if df is None or df.empty: return df
    
    # Check if we already have the columns to avoid re-calc overhead
    cols = set(df.columns)
    has_rsi = 'RSI' in cols or 'RSI_14' in cols
    has_ema8 = 'EMA_8' in cols
    has_ema21 = 'EMA_21' in cols
    has_sma200 = 'SMA_200' in cols
    
    if has_rsi and has_ema8 and has_ema21 and has_sma200:
        return df

    # Find close column efficiently
    close_col = next((c for c in ['Price', 'Close', 'CLOSE'] if c in cols), None)
    if not close_col: return df
    
    # Use the series for calc
    close_series = df[close_col]

    if not has_rsi:
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI']
    
    if not has_ema8:
        df['EMA_8'] = close_series.ewm(span=8, adjust=False).mean()
        
    if not has_ema21:
        df['EMA_21'] = close_series.ewm(span=21, adjust=False).mean()
        
    if not has_sma200 and len(df) >= 200:
        df['SMA_200'] = close_series.rolling(window=200).mean()
            
    return df

@st.cache_data(ttl=CACHE_TTL)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
        
        if df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
            
        df.rename(columns={"Close": "CLOSE", "Volume": "VOLUME", "High": "HIGH", "Low": "LOW", "Open": "OPEN"}, inplace=True)
        df.columns = [c.upper() for c in df.columns]
        
        return add_technicals(df)
    except Exception:
        return None

# --- HELPERS ---

def parse_periods(periods_str):
    try:
        return sorted(list(set([int(x.strip()) for x in periods_str.split(',') if x.strip().isdigit()])))
    except:
        return [5, 21, 63, 126]

@st.cache_data(ttl=CACHE_TTL)
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
    return get_expiry_color_map().get(val, "")

def clean_strike_fmt(val):
    try:
        f = float(val)
        if f.is_integer(): return str(int(f))
        return str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid = df["Trade Date"].dropna()
        if not valid.empty: return valid.max().date()
    return date.today() - timedelta(days=1)

@st.cache_data(ttl=CACHE_TTL)
def get_stock_indicators(sym: str):
    try:
        ticker_obj = yf.Ticker(sym)
        h_full = ticker_obj.history(period="2y", interval="1d")
        
        if len(h_full) == 0: return None, None, None, None, None
        
        h_full = add_technicals(h_full)
        
        sma200 = float(h_full["SMA_200"].iloc[-1]) if "SMA_200" in h_full.columns else None
        
        h_recent = h_full.iloc[-60:].copy() if len(h_full) > 60 else h_full.copy()
        if len(h_recent) == 0: return None, None, None, None, None
        
        spot_val = float(h_recent["Close"].iloc[-1])
        ema8 = float(h_recent.get("EMA_8", h_recent.get("EMA8")).iloc[-1])
        ema21 = float(h_recent.get("EMA_21", h_recent.get("EMA21")).iloc[-1])
        
        return spot_val, ema8, ema21, sma200, h_full
    except: 
        return None, None, None, None, None


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


def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    # Fast column lookup
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    
    if not date_col or not close_col: return None, None
    
    # Vectorized Indexing
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df[date_col])
    
    # Only sort if needed (check first 2 elements)
    if len(df) > 1 and df.index[0] > df.index[-1]:
        df = df.sort_index()
    
    # Identify key columns
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    if not all([vol_col, high_col, low_col]): return None, None
    
    # --- BUILD DAILY ---
    d_rsi = next((c for c in cols if c in ['RSI', 'RSI14'] and 'W_' not in c), 'RSI')
    d_ema8 = next((c for c in cols if c == 'EMA8'), 'EMA8')
    d_ema21 = next((c for c in cols if c == 'EMA21'), 'EMA21')

    # Minimal copy
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
    df_d.dropna(subset=['Price', 'RSI'], inplace=True)
    
    # --- BUILD WEEKLY ---
    # Quick check for pre-calculated weekly columns
    w_close = 'W_CLOSE'
    if w_close not in df.columns:
        return df_d, None

    cols_w = [c for c in df.columns if c.startswith('W_')]
    df_w = df[cols_w].copy()
    
    # Fast rename map
    w_map = {c: c.replace('W_', '').replace('CLOSE', 'Price').title() for c in cols_w}
    # Fix specific keys
    for k, v in w_map.items():
        if 'Price' in v: w_map[k] = 'Price'
        if 'Volume' in v: w_map[k] = 'Volume'
        if 'High' in v: w_map[k] = 'High'
        if 'Low' in v: w_map[k] = 'Low'
        if 'Rsi' in v: w_map[k] = 'RSI'
        if 'Ema8' in v: w_map[k] = 'EMA8'
        if 'Ema21' in v: w_map[k] = 'EMA21'
        
    df_w.rename(columns=w_map, inplace=True)
    df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
    
    df_w = add_technicals(df_w)
    df_w.dropna(subset=['Price', 'RSI'], inplace=True)
        
    return df_d, df_w

def calculate_optimal_signal_stats(history_indices, price_array, current_idx, signal_type='Bullish', timeframe='Daily', periods_input=None, optimize_for='PF'):
    hist_arr = np.array(history_indices)
    valid_indices = hist_arr[hist_arr < current_idx]
    
    if len(valid_indices) == 0: return None

    periods = np.array(periods_input) if periods_input else np.array([5, 21, 63, 126])
    total_len = len(price_array)
    unit = 'w' if timeframe.lower() == 'weekly' else 'd'

    # Vectorized Exits
    exit_indices_matrix = valid_indices[:, None] + periods[None, :]
    valid_exits_mask = exit_indices_matrix < total_len
    safe_exit_indices = np.clip(exit_indices_matrix, 0, total_len - 1)
    
    entry_prices = price_array[valid_indices]
    exit_prices_matrix = price_array[safe_exit_indices]
    
    raw_returns = (exit_prices_matrix - entry_prices[:, None]) / entry_prices[:, None]
    if signal_type == 'Bearish': raw_returns = -raw_returns

    best_score = -999.0
    best_stats = None
    
    # Inner loop is small (4-5 periods max)
    for i, p in enumerate(periods):
        col_mask = valid_exits_mask[:, i]
        p_rets = raw_returns[col_mask, i]
        
        if len(p_rets) == 0: continue
            
        wins = p_rets[p_rets > 0]
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(p_rets[p_rets < 0]))
        
        pf = 999.0 if gross_loss == 0 and gross_win > 0 else (gross_win / gross_loss if gross_loss > 0 else 0.0)
        
        n = len(p_rets)
        win_rate = (len(wins) / n) * 100
        avg_ret = np.mean(p_rets) * 100
        std_dev = np.std(p_rets)
        sqn = (np.mean(p_rets) / std_dev) * np.sqrt(n) if std_dev > 0 else 0.0
        
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
    
    close_vals = history_df["CLOSE" if "CLOSE" in history_df.columns else "Close"].values
    rsi_vals = history_df["RSI_14" if "RSI_14" in history_df.columns else "RSI"].values
    
    mask = (rsi_vals >= (current_rsi - tolerance)) & (rsi_vals <= (current_rsi + tolerance))
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
        
        score = (np.mean(returns > 0) * 2) + np.mean(returns)
        if score > best_score:
            best_score = score
            best_p = p
            
    return best_p, f"RSI Backtest (Optimal {best_p}d)"

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None

    # Filter efficiently
    f = global_df[global_df["Symbol"].astype(str).str.upper() == ticker].copy()
    if f.empty: return None

    today_dt = pd.Timestamp.now()
    f = f[f["Expiry_DT"] > today_dt]
    
    if order_type_filter:
        f = f[f["Order Type"] == order_type_filter]
    else:
        f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
        
    if f.empty: return None
    
    f.sort_values(by="Dollars", ascending=False, inplace=True)
    whale = f.iloc[0]
    
    # Logic for Put Sales OTM
    if whale["Order Type"] == "Puts Sold" and whale["Strike (Actual)"] > current_price:
        otm_puts = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm_puts.empty:
            whale = otm_puts.iloc[0]
    
    return {
        "Strike": whale["Strike (Actual)"],
        "Expiry": whale["Expiry_DT"].strftime("%d %b"),
        "Dollars": whale["Dollars"],
        "Type": whale["Order Type"]
    }

def analyze_trade_setup(ticker, t_df, global_df):
    score = 0
    reasons = []
    suggestions = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    
    if t_df is None or t_df.empty:
        return 0, ["No data"], suggestions

    last = t_df.iloc[-1]
    close = last.get('CLOSE', last.get('Close', 0))
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
        
    if put_whale: reasons.append(f"Whale: Sold Puts @ ${put_whale['Strike']}")
    if call_whale: reasons.append(f"Whale: Bought Calls @ ${call_whale['Strike']}")
    
    return score, reasons, suggestions

@st.cache_data(ttl=43200) # Cache for 12 hours
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        # Try fast_info first
        fi = t.fast_info
        mc = fi.get('marketCap')
        if mc: return float(mc)
        
        # Fallback: Manual calculation (More stable)
        shares = fi.get('shares')
        price = fi.get('lastPrice')
        if shares and price:
            return float(shares * price)
    except:
        pass
    return 0.0

def fetch_market_caps_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            results[future_to_ticker[future]] = future.result()
    return results

def fetch_technicals_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_stock_indicators, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            try:
                results[future_to_ticker[future]] = future.result()
            except:
                results[future_to_ticker[future]] = (None, None, None, None, None)
    return results

@st.cache_data(ttl=CACHE_TTL, show_spinner="Crunching Smart Money Data...")
def calculate_smart_money_score(df, start_d, end_d, mc_thresh, filter_ema, limit):
    f = df.copy()
    if start_d: f = f[f["Trade Date"].dt.date >= start_d]
    if end_d: f = f[f["Trade Date"].dt.date <= end_d]
    
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f = f[f[order_type_col].isin(target_types)].copy()
    
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    f["Signed_Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), f["Dollars"], -f["Dollars"])
    
    smart_stats = f.groupby("Symbol").agg(
        Signed_Dollars=("Signed_Dollars", "sum"),
        Trade_Count=("Symbol", "count"),
        Last_Trade=("Trade Date", "max")
    ).reset_index()
    
    smart_stats.rename(columns={"Signed_Dollars": "Net Sentiment ($)"}, inplace=True)
    
    unique_tickers = smart_stats["Symbol"].unique().tolist()
    batch_caps = fetch_market_caps_batch(unique_tickers)
    smart_stats["Market Cap"] = smart_stats["Symbol"].map(batch_caps)
    
    valid_data = smart_stats[smart_stats["Market Cap"] >= mc_thresh].copy()
    
    # Calculate Momentum
    unique_dates = sorted(f["Trade Date"].unique())
    recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
    f_momentum = f[f["Trade Date"].isin(recent_dates)]
    mom_stats = f_momentum.groupby("Symbol")["Signed_Dollars"].sum().reset_index().rename(columns={"Signed_Dollars": "Momentum ($)"})
    
    valid_data = valid_data.merge(mom_stats, on="Symbol", how="left").fillna(0)
    
    top_bulls = pd.DataFrame()
    top_bears = pd.DataFrame()

    if not valid_data.empty:
        valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
        
        # Vectorized Normalization
        def normalize(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx != mn else 0

        # Scores
        valid_data["Base_Score_Bull"] = (0.35 * normalize(valid_data["Net Sentiment ($)"].clip(lower=0))) + \
                                        (0.30 * normalize(valid_data["Impact"].clip(lower=0))) + \
                                        (0.35 * normalize(valid_data["Momentum ($)"].clip(lower=0)))
        
        valid_data["Base_Score_Bear"] = (0.35 * normalize(-valid_data["Net Sentiment ($)"].clip(upper=0))) + \
                                        (0.30 * normalize(-valid_data["Impact"].clip(upper=0))) + \
                                        (0.35 * normalize(-valid_data["Momentum ($)"].clip(upper=0)))
        
        valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")
        
        # Batch Fetch Techs
        candidates_bull = valid_data.sort_values(by="Base_Score_Bull", ascending=False).head(limit * 3)
        candidates_bear = valid_data.sort_values(by="Base_Score_Bear", ascending=False).head(limit * 3)
        
        all_tickers = set(candidates_bull["Symbol"]).union(set(candidates_bear["Symbol"]))
        batch_techs = fetch_technicals_batch(list(all_tickers)) if filter_ema else {}

        # OPTIMIZED: Vectorized Filter Application
        def apply_ema_filter(df, mode="Bull"):
            if not filter_ema:
                df["Score"] = df[f"Base_Score_{mode}"] * 100
                df["Trend"] = "—"
                return df.head(limit)
            
            def check_row(t):
                s, e8, _, _, _ = batch_techs.get(t, (None, None, None, None, None))
                if not s or not e8: return False, "—"
                if mode == "Bull": return (s > e8), ("✅ >EMA8" if s > e8 else "⚠️ <EMA8")
                return (s < e8), ("✅ <EMA8" if s < e8 else "⚠️ >EMA8")
            
            # Use list comprehension (faster than iterrows)
            results = [check_row(t) for t in df["Symbol"]]
            mask = [r[0] for r in results]
            trends = [r[1] for r in results]
            
            filtered = df[mask].copy()
            filtered["Trend"] = [t for i, t in enumerate(trends) if mask[i]]
            filtered["Score"] = filtered[f"Base_Score_{mode}"] * 100
            return filtered.head(limit)

        top_bulls = apply_ema_filter(candidates_bull, "Bull")
        top_bears = apply_ema_filter(candidates_bear, "Bear")
        
    return top_bulls, top_bears, valid_data

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_history_optimized(ticker_sym, t_map):
    pq_key = f"{ticker_sym}_PARQUET"
    # Try Parquet
    if pq_key in t_map:
        try:
            url = f"https://drive.google.com/uc?export=download&id={t_map[pq_key]}"
            buffer = get_gdrive_binary_data(url) 
            if buffer:
                df = pd.read_parquet(buffer, engine='pyarrow')
                return df.reset_index() if 'DATE' in str(df.index.name).upper() else df
        except Exception: pass 

    # Try CSV
    if ticker_sym in t_map:
        try:
            df = get_ticker_technicals(ticker_sym, t_map)
            if df is not None and not df.empty: return df
        except Exception: pass

    # Fallback
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
    
    rsi_vals = hist_df['RSI'].values 
    price_vals = hist_df['Price'].values
    p10 = np.quantile(rsi_vals, pct_low)
    p90 = np.quantile(rsi_vals, pct_high)
    
    # Vectorized Signal Detection
    prev_rsi = np.roll(rsi_vals, 1)
    prev_rsi[0] = rsi_vals[0] # Fix roll artifact
    
    bull_mask = (prev_rsi < p10) & (rsi_vals >= (p10 + 1.0))
    bear_mask = (prev_rsi > p90) & (rsi_vals <= (p90 - 1.0))
    
    bull_indices = np.where(bull_mask)[0]
    bear_indices = np.where(bear_mask)[0]
    all_indices = np.sort(np.concatenate((bull_indices, bear_indices)))
    
    latest_close = df['Price'].iloc[-1] 
    
    # Process Found Signals
    for i in all_indices:
        curr_date = hist_df.index[i].date()
        if filter_date and curr_date < filter_date: continue
            
        is_bullish = i in bull_indices
        s_type = 'Bullish' if is_bullish else 'Bearish'
        
        hist_list = bull_indices if is_bullish else bear_indices
        best_stats = calculate_optimal_signal_stats(hist_list, price_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if not best_stats or best_stats["N"] < min_n: continue
            
        ev_val = best_stats['EV']
        sig_close = price_vals[i]
        ev_price = sig_close * (1 + (ev_val / 100.0)) if is_bullish else sig_close * (1 - (ev_val / 100.0))
        
        thresh = p10 if is_bullish else p90
        curr_rsi = rsi_vals[i]

        signals.append({
            'Ticker': ticker,
            'Date': curr_date.strftime('%b %d'),
            'Date_Obj': curr_date,
            'Action': "Leaving Low" if is_bullish else "Leaving High",
            'RSI_Display': f"{thresh:.0f} {'↗' if is_bullish else '↘'} {curr_rsi:.0f}",
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

@st.cache_data(ttl=CACHE_TTL)
def is_above_ema21(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        h = ticker.history(period="60d")
        if len(h) < 21: return True 
        ema21 = h["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
        return h["Close"].iloc[-1] > ema21
    except:
        return True

@st.cache_data(ttl=CACHE_TTL)
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping: return None
    
    file_url = f"https://drive.google.com/uc?export=download&id={mapping[ticker]}"
    buffer = get_gdrive_binary_data(file_url)
    
    if buffer:
        try:
            # Engine C for speed
            df = pd.read_csv(buffer, engine='c')
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" not in df.columns: df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
            return df
        except Exception:
            return None
    return None