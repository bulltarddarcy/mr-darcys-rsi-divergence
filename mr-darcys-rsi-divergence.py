import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from datetime import datetime

# --- Secrets & Path Configuration ---
def get_confirmed_gdrive_data(url):
    """Bypasses the 'File too large to scan' warning page using a token handshake."""
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if not file_id:
            return None
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        
        # Pass 1: Get the initial response (might be the HTML warning page)
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match:
                confirm_token = match.group(1)

        # Pass 2: Request with the confirmation token
        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
        
        # Check if the result is still HTML (indicating failure)
        if response.text.strip().startswith("<!DOCTYPE html>"):
            return "HTML_ERROR"
            
        return StringIO(response.text)
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return None

# --- Logic Constants (Synced with Source of Truth: divergence_make_dashboard.py) ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Divergence Scanner", layout="wide")
st.title("📈 RSI Divergence Scanner")

# Updated Dataset Selection
data_option = st.sidebar.selectbox(
    "Select Dataset to Analyze",
    ("Divergences Data", "S&P 100 Data", "NQ 100 Data")
)

# Secrets Mapping for new datasets
try:
    if data_option == "Divergences Data":
        target_url = st.secrets["URL_DIVERGENCES"]
    elif data_option == "S&P 100 Data":
        target_url = st.secrets["URL_SP100"]
    else: # NQ 100 Data
        target_url = st.secrets["URL_NQ100"]
except KeyError as e:
    st.error(f"Secret key {e} not found. Please ensure URL_DIVERGENCES, URL_SP100, and URL_NQ100 are set.")
    st.stop()

# --- Logic Functions ---

def prepare_data(df):
    """Sync with Source of Truth prepare_data logic."""
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    # Identify basic columns
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    # Daily logic columns
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    # Weekly logic columns
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
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    # Weekly Subset
    if all(c in df.columns for c in [w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col]):
        df_w = df[[w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col, w_ema8_col, w_ema21_col]].copy()
        df_w.rename(columns={
            w_close_col: 'Price', w_vol_col: 'Volume', w_high_col: 'High', w_low_col: 'Low',
            w_rsi_col: 'RSI', w_ema8_col: 'EMA8', w_ema21_col: 'EMA21'
        }, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else:
        df_w = None
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    """Detection logic. Sync with SOT logic."""
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences

    def get_date_str(p):
        return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        # Bullish
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                # Invalidation: RSI peaks > 50 between P1 and P2
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    # Post-Signal Invalidation: RSI drops below P1 RSI later
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] <= p1['RSI']).any()):
                        tags = []
                        if 'EMA8' in p2 and p2['Price'] >= p2['EMA8']: tags.append(f"EMA{EMA_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"
                        })

        # Bearish
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                # Invalidation: RSI drops below 50 between P1 and P2
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    # Post-Signal Invalidation: RSI rises above P1 RSI later
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] >= p1['RSI']).any()):
                        tags = []
                        if 'EMA21' in p2 and p2['Price'] <= p2['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"
                        })
    return divergences

# --- App Execution ---

st.info(f"Connecting to {data_option}...")
csv_buffer = get_confirmed_gdrive_data(target_url)

if csv_buffer == "HTML_ERROR":
    st.error("Google Drive is still serving an HTML warning page. Your file may still be too large.")
elif csv_buffer:
    try:
        master = pd.read_csv(csv_buffer)
        
        # Robust Ticker Column Detection
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
        if not t_col:
            st.error(f"Ticker column not found. Available: {list(master.columns)}")
            st.stop()

        raw_results = []
        progress_bar = st.progress(0, text="Scanning tickers...")
        
        grouped = master.groupby(t_col)
        total_groups = len(grouped)
        
        for i, (ticker, group) in enumerate(grouped):
            d_d, d_w = prepare_data(group.copy())
            if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
            if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
            progress_bar.progress((i + 1) / total_groups)

        if raw_results:
            res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
            # NFLX FIX: groupby ensures both Bullish and Bearish show up for a single ticker
            consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
            
            for tf in ['Daily', 'Weekly']:
                st.markdown(f"---")
                st.header(f"📅 {tf} Divergence Analysis")
                
                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {s_type} Signals")
                    tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)]
                    if not tbl_df.empty:
                        st.table(tbl_df.drop(columns=['Type', 'Timeframe']))
                    else:
                        st.write(f"No {tf} {s_type} signals found.")
        else:
            st.warning("No signals detected.")
    except Exception as e:
        st.error(f"Processing Error: {e}")
