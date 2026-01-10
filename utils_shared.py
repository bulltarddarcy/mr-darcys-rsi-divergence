# utils_shared.py
import streamlit as st
import pandas as pd
import requests
import re
import numpy as np
from io import BytesIO

# --- PERFORMANCE OPTIMIZATION: GLOBAL SESSION ---
GLOBAL_SESSION = requests.Session()

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

def get_gdrive_binary_data(url):
    """
    Robust Google Drive downloader using a global session for speed.
    Handles 'virus scan' confirmation pages and various URL formats.
    """
    try:
        match = re.search(r'/d/([a-zA-Z0-9_-]{25,})', url)
        if not match:
            match = re.search(r'id=([a-zA-Z0-9_-]{25,})', url)
            
        if not match:
            return None
            
        file_id = match.group(1)
        download_url = "https://drive.google.com/uc?export=download"
        
        response = GLOBAL_SESSION.get(download_url, params={'id': file_id}, stream=True, timeout=30)
        
        if "text/html" in response.headers.get("Content-Type", "").lower():
            content = response.text
            token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', content)
            
            if token_match:
                token = token_match.group(1)
                params = {'id': file_id, 'confirm': token}
                response = GLOBAL_SESSION.get(download_url, params=params, stream=True, timeout=30)
            else:
                for key, value in GLOBAL_SESSION.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'confirm': value}
                        response = GLOBAL_SESSION.get(download_url, params=params, stream=True, timeout=30)
                        break

        if response.status_code == 200:
            try:
                chunk = next(response.iter_content(chunk_size=100), b"")
                if chunk.strip().startswith(b"<!DOCTYPE"):
                    return None
                return BytesIO(chunk + response.raw.read())
            except StopIteration:
                return None
                
        return None

    except Exception as e:
        print(f"Download Exception: {e}")
        return None

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0: return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

# --- SHARED TECHNICAL ANALYSIS ---

def add_technicals(df):
    """
    Ensures technical columns exist. 
    Respects pre-populated columns (RSI, EMA, SMA) and skips calculation if found.
    """
    if df is None or df.empty: return df
    
    # Check if we already have the columns to avoid re-calc overhead
    cols = set(df.columns)
    
    # Check for variations of column names typically found in source files
    has_rsi = 'RSI' in cols or 'RSI_14' in cols or 'RSI14' in cols
    has_ema8 = 'EMA_8' in cols or 'EMA8' in cols
    has_ema21 = 'EMA_21' in cols or 'EMA21' in cols
    has_sma200 = 'SMA_200' in cols or 'SMA200' in cols
    
    # If all major indicators exist, skip calculation
    if has_rsi and has_ema8 and has_ema21 and has_sma200:
        return df

    # Find close column efficiently
    close_col = next((c for c in ['Price', 'Close', 'CLOSE', 'Adj Close'] if c in cols), None)
    if not close_col: return df
    
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

def prepare_data(df):
    """
    Standardizes column names and splits into Daily/Weekly dataframes.
    Uses pre-populated W_ columns if available.
    """
    # Normalize columns: remove whitespace, upper case, remove special chars
    df.columns = [str(col).strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    # Find Daily Close (exclude W_ columns)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    
    if not date_col or not close_col: return None, None
    
    # Vectorized Indexing
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df[date_col])
    
    if len(df) > 1 and df.index[0] > df.index[-1]:
        df = df.sort_index()
    
    # Identify key columns
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    # --- BUILD DAILY ---
    d_rsi = next((c for c in cols if c in ['RSI', 'RSI14', 'RSI_14'] and 'W_' not in c), 'RSI')
    d_ema8 = next((c for c in cols if c in ['EMA8', 'EMA_8'] and 'W_' not in c), 'EMA8')
    d_ema21 = next((c for c in cols if c in ['EMA21', 'EMA_21'] and 'W_' not in c), 'EMA21')

    # Minimal copy
    needed_cols = [close_col]
    if vol_col: needed_cols.append(vol_col)
    if high_col: needed_cols.append(high_col)
    if low_col: needed_cols.append(low_col)
    
    # Add techs if they exist
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8 in df.columns: needed_cols.append(d_ema8)
    if d_ema21 in df.columns: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    
    rename_dict = {close_col: 'Price'}
    if vol_col: rename_dict[vol_col] = 'Volume'
    if high_col: rename_dict[high_col] = 'High'
    if low_col: rename_dict[low_col] = 'Low'
    
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8 in df_d.columns: rename_dict[d_ema8] = 'EMA8'
    if d_ema21 in df_d.columns: rename_dict[d_ema21] = 'EMA21'
    
    df_d.rename(columns=rename_dict, inplace=True)
    
    if 'Volume' in df_d.columns:
        df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    else:
        df_d['Volume'] = 0
        df_d['VolSMA'] = 0
    
    # Ensure techs exist (calc if missing, skip if present)
    df_d = add_technicals(df_d)
    
    # --- BUILD WEEKLY ---
    # Quick check for pre-calculated weekly columns
    w_close_candidates = [c for c in df.columns if c.startswith('W_') and 'CLOSE' in c]
    
    if not w_close_candidates:
        # If no pre-calc weekly data, return None (or we could resample, but logic prefers pre-calc)
        return df_d, None

    cols_w = [c for c in df.columns if c.startswith('W_')]
    df_w = df[cols_w].copy()
    
    # Fast rename map
    w_map = {c: c.replace('W_', '').replace('CLOSE', 'Price').title() for c in cols_w}
    for k, v in w_map.items():
        if 'Price' in v: w_map[k] = 'Price'
        if 'Volume' in v: w_map[k] = 'Volume'
        if 'High' in v: w_map[k] = 'High'
        if 'Low' in v: w_map[k] = 'Low'
        if 'Rsi' in v: w_map[k] = 'RSI'
        if 'Ema8' in v: w_map[k] = 'EMA8'
        if 'Ema21' in v: w_map[k] = 'EMA21'
        
    df_w.rename(columns=w_map, inplace=True)
    
    if 'Volume' in df_w.columns:
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    # Use Monday date for ChartDate
    df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
    
    df_w = add_technicals(df_w)
        
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

def find_divergences(df_tf, ticker, timeframe, min_n=0, periods_input=None, optimize_for='PF', lookback_period=90, price_source='High/Low', strict_validation=True, recent_days_filter=25, rsi_diff_threshold=2.0):
    divergences = []
    n_rows = len(df_tf)
    
    if n_rows < lookback_period + 1: return divergences
    
    # Ensure RSI exists (it should via add_technicals, but safety check)
    if 'RSI' not in df_tf.columns:
        if 'RSI14' in df_tf.columns: df_tf['RSI'] = df_tf['RSI14']
        else: return divergences # Cannot run without RSI

    rsi_vals = df_tf['RSI'].values
    vol_vals = df_tf['Volume'].values if 'Volume' in df_tf.columns else np.zeros(n_rows)
    vol_sma_vals = df_tf['VolSMA'].values if 'VolSMA' in df_tf.columns else np.zeros(n_rows)
    close_vals = df_tf['Price'].values 
    
    # Select Price Arrays based on User Input
    if price_source == 'Close':
        low_vals = close_vals
        high_vals = close_vals
    else:
        # Fallback if High/Low missing
        low_vals = df_tf['Low'].values if 'Low' in df_tf.columns else close_vals
        high_vals = df_tf['High'].values if 'High' in df_tf.columns else close_vals
        
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        # For weekly, use specific ChartDate col if available
        if timeframe.lower() == 'weekly' and 'ChartDate' in df_tf.columns: 
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
        
        is_vol_high = int(p2_vol > (p2_volsma * 1.5)) if not np.isnan(p2_volsma) and p2_volsma > 0 else 0
        
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
    # Allow looking back further if we just want "last signal", but usually filter by recent
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

        # Optimization Stats (Optional if calculating history)
        if periods_input is not None:
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
                'N': best_stats['N'],
                'Best Period': best_stats['Best Period'],
                'Profit Factor': best_stats['Profit Factor'],
                'Win Rate': best_stats['Win Rate'],
                'EV': best_stats['EV']
            })

            # --- FORWARD PERFORMANCE & RAW CSV DATA ---
            prefix = "Daily" if timeframe == "Daily" else "Weekly"
            
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
        else:
             # Minimal object for scanners
             div_obj.update({
                'Tags': tags, 
                'Date_Display': date_display,
                'RSI_Display': rsi_display,
                'Price_Display': price_display, 
                'Last_Close': f"${latest_row['Price']:,.2f}"
            })
        
        divergences.append(div_obj)
            
    return divergences