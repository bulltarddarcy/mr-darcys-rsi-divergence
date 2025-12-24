import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
from datetime import datetime

# --- Improved Data Fetching (Bypasses Large File Warnings) ---
def download_gdrive_csv(url):
    """Downloads a CSV from Google Drive, handling virus scan warnings for large files."""
    try:
        # 1. Extract File ID
        file_id = url.split('/')[-2]
        download_url = "https://docs.google.com/uc?export=download"
        
        session = requests.Session()
        # Initial request
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        # Check for Google's 'confirm' token in cookies/HTML (required for >100MB files)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            # If token found, make a second request with confirmation
            params = {'id': file_id, 'confirm': token}
            response = session.get(download_url, params=params, stream=True)
            
        # Return as a string buffer for pandas
        return StringIO(response.text)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

# --- Logic Constants (Synced with Source of Truth) ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Divergence Scanner", layout="wide")
st.title("📈 RSI Price Divergence Scanner")

# Dataset Selection Sidebar
data_option = st.sidebar.selectbox(
    "Select Dataset to Analyze",
    ("Divergences Data", "S&P 500 Data")
)

# Accessing secrets
try:
    if data_option == "Divergences Data":
        DATA_URL = st.secrets["URL_DIVERGENCES"]
    else:
        DATA_URL = st.secrets["URL_SP500"]
except KeyError:
    st.error("Secrets not found. Please ensure URL_DIVERGENCES and URL_SP500 are set in Streamlit Secrets.")
    st.stop()

# --- Core Logic Functions ---

def prepare_data(df):
    """Clean and map columns from the combined master CSV. Sync with SOT logic."""
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    # Column Mappings as defined in SOT
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close_col, w_vol_col, w_rsi_col = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_ema8_col, w_ema21_col = 'W_EMA_8', 'W_EMA_21'
    w_high_col, w_low_col = 'W_HIGH', 'W_LOW'

    if not all([date_col, close_col, vol_col, high_col, low_col]):
        return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Daily Subset
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={
        close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low',
        d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'
    }, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI', 'High', 'Low'])

    # Weekly Subset
    df_w = df[[w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col, w_ema8_col, w_ema21_col]].copy()
    df_w.rename(columns={
        w_close_col: 'Price', w_vol_col: 'Volume', w_high_col: 'High', w_low_col: 'Low',
        w_rsi_col: 'RSI', w_ema8_col: 'EMA8', w_ema21_col: 'EMA21'
    }, inplace=True)
    df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
    df_w = df_w.dropna(subset=['Price', 'RSI', 'High', 'Low'])
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    """Detects RSI divergences including invalidation and volume metrics."""
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences

    def get_date_str(point):
        return df_tf.loc[point.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else point.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5))
        
        # Bullish
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    post_signal_df = df_tf.iloc[i + 1 :]
                    if not (not post_signal_df.empty and (post_signal_df['RSI'] <= p1['RSI']).any()):
                        tags = []
                        if p2['Price'] >= p2['EMA8']: tags.append(f"EMA{EMA_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe,
                            'Tags': ", ".join(tags), 'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"
                        })

        # Bearish
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    post_signal_df = df_tf.iloc[i + 1 :]
                    if not (not post_signal_df.empty and (post_signal_df['RSI'] >= p1['RSI']).any()):
                        tags = []
                        if p2['Price'] <= p2['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe,
                            'Tags': ", ".join(tags), 'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"
                        })
    return divergences

# --- Execution ---

st.info(f"Connecting to data source (handling large file bypass)...")
csv_buffer = download_gdrive_csv(DATA_URL)

if csv_buffer:
    try:
        master = pd.read_csv(csv_buffer)
        t_col = None
        for col in master.columns:
            if col.strip().upper() in ['TICKER', 'SYMBOL', 'SYM', 'CODE']:
                t_col = col
                break
        
        if not t_col:
            st.error(f"Could not find a ticker/symbol column. Available: {', '.join(master.columns)}")
            st.stop()

        all_tickers = master[t_col].unique()
        raw_results = []
        progress_bar = st.progress(0)

        for i, ticker in enumerate(all_tickers):
            df_t = master[master[t_col] == ticker].copy()
            d_d, d_w = prepare_data(df_t)
            if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
            if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
            progress_bar.progress((i + 1) / len(all_tickers))

        if raw_results:
            res_df = pd.DataFrame(raw_results)
            res_df = res_df.sort_values(by='Signal Date', ascending=False)
            consolidated_df = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
            
            for timeframe in ['Daily', 'Weekly']:
                st.markdown(f"## {timeframe} Analysis")
                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.markdown(f"### {emoji} {timeframe} {s_type} Signals")
                    tbl_df = consolidated_df[(consolidated_df['Type'] == s_type) & (consolidated_df['Timeframe'] == timeframe)]
                    if not tbl_df.empty:
                        st.table(tbl_df.drop(columns=['Type', 'Timeframe']))
                    else:
                        st.write(f"No {timeframe.lower()} {s_type.lower()} signals found.")
                st.divider()
        else:
            st.write("No divergences found.")
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
