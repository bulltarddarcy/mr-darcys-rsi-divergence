import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Secrets & Path Configuration ---
def get_gdrive_download_url(url):
    """Converts a standard Google Drive view URL into a direct download URL."""
    try:
        # Extracts file ID from URL: .../file/d/[ID]/view
        file_id = url.split('/')[-2]
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    except Exception:
        return url

# --- Logic Constants ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2

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
        raw_url = st.secrets["URL_DIVERGENCES"]
    else:
        raw_url = st.secrets["URL_SP500"]
    
    DATA_URL = get_gdrive_download_url(raw_url)
except KeyError:
    st.error("Secrets not found. Please ensure URL_DIVERGENCES and URL_SP500 are set.")
    st.stop()

# --- Core Functions ---

def prepare_data(df):
    """Clean and map columns from the combined master CSV."""
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    # Define Column Names
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close_col, w_vol_col, w_rsi_col = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_ema8_col, w_ema21_col = 'W_EMA_8', 'W_EMA_21'
    w_high_col, w_low_col = 'W_HIGH', 'W_LOW'

    if not date_col or not close_col:
        return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # --- Daily Subset ---
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={
        close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low',
        d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'
    }, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI', 'High', 'Low'])

    # --- Weekly Subset ---
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
    """Detects RSI divergences using candle extremes."""
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences

    def get_date_str(point):
        return df_tf.loc[point.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe == 'weekly' else point.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        
        # Bullish (Low)
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    divergences.append({
                        'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe,
                        'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                        'P1 RSI': round(p1['RSI'], 1), 'P2 RSI': round(p2['RSI'], 1),
                        'P1 Price': round(p1['Low'], 2), 'P2 Price': round(p2['Low'], 2)
                    })
        # Bearish (High)
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    divergences.append({
                        'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe,
                        'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                        'P1 RSI': round(p1['RSI'], 1), 'P2 RSI': round(p2['RSI'], 1),
                        'P1 Price': round(p1['High'], 2), 'P2 Price': round(p2['High'], 2)
                    })
    return divergences

# --- Execution ---

st.info(f"Connecting to data source...")
try:
    master = pd.read_csv(DATA_URL)
    t_col = next((c for c in master.columns if 'TICKER' in c.upper()), 'TICKER')
    all_tickers = master[t_col].unique()

    raw_results = []
    progress_bar = st.progress(0)

    for i, ticker in enumerate(all_tickers):
        df_t = master[master[t_col] == ticker].copy()
        d_d, d_w = prepare_data(df_t)
        
        if d_d is not None:
            raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
        if d_w is not None:
            raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
        
        progress_bar.progress((i + 1) / len(all_tickers))

    if raw_results:
        # CONSOLIDATION LOGIC: Keep only the latest signal per ticker/type/timeframe
        res_df = pd.DataFrame(raw_results)
        # Sort by Signal Date descending to ensure the latest is on top
        res_df = res_df.sort_values(by='Signal Date', ascending=False)
        # Drop duplicates based on the unique combination of ticker, type, and timeframe
        consolidated_df = res_df.drop_duplicates(subset=['Ticker', 'Type', 'Timeframe'], keep='first')
        
        # --- Stacked Layout (Full Width) ---
        for timeframe in ['Daily', 'Weekly']:
            st.markdown(f"## {timeframe} Analysis")
            
            # Bullish Table
            st.markdown(f"### 🟢 {timeframe} Bullish Signals")
            bull_df = consolidated_df[(consolidated_df['Type'] == 'Bullish') & (consolidated_df['Timeframe'] == timeframe)]
            if not bull_df.empty:
                st.table(bull_df.drop(columns=['Type', 'Timeframe']))
            else:
                st.write(f"No {timeframe.lower()} bullish signals found.")
            
            # Bearish Table
            st.markdown(f"### 🔴 {timeframe} Bearish Signals")
            bear_df = consolidated_df[(consolidated_df['Type'] == 'Bearish') & (consolidated_df['Timeframe'] == timeframe)]
            if not bear_df.empty:
                st.table(bear_df.drop(columns=['Type', 'Timeframe']))
            else:
                st.write(f"No {timeframe.lower()} bearish signals found.")
            
            st.divider()
    else:
        st.write("No divergences found.")
except Exception as e:
    st.error(f"Error loading data: {e}")
