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
        st.error(f"Fetch Error: {e}")
        return None

def load_dataset_config():
    """Reads the TXT file from Drive and returns a dictionary {Name: SecretKey}"""
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

# --- Logic Constants ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Divergence Scanner", layout="wide")

st.markdown("""
    <style>
    table { width: 100%; border-collapse: collapse; table-layout: fixed; margin-bottom: 2rem; }
    thead tr th { background-color: #f0f2f6 !important; color: #31333f !important; padding: 12px !important; border-bottom: 2px solid #dee2e6; }
    th:nth-child(1) { width: 10%; } th:nth-child(2) { width: 32%; } th:nth-child(3) { width: 12%; } 
    th:nth-child(4) { width: 12%; } th:nth-child(5) { width: 10%; } th:nth-child(6) { width: 12%; } th:nth-child(7) { width: 12%; }
    tbody tr td { padding: 10px !important; border-bottom: 1px solid #eee; word-wrap: break-word; }
    .align-left { text-align: left !important; }
    .align-center { text-align: center !important; }
    .tag-bubble { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 14px; font-weight: 600; margin: 2px 4px 2px 0; color: white; white-space: nowrap; }
    .grey-note { color: #888888; font-size: 16px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 RSI Divergence Scanner")
st.markdown('<div class="grey-note">ℹ️ See bottom of page for strategy logic and tag explanations.</div>', unsafe_allow_html=True)

# --- Helpers ---
def style_tags(tag_str):
    if not tag_str: return ''
    tags = tag_str.split(", ")
    html_str = ''
    colors = {f"EMA{EMA8_PERIOD}": "#4a90e2", f"EMA{EMA21_PERIOD}": "#9b59b6", "VOL_HIGH": "#e67e22", "V_GROW": "#27ae60"}
    for t in tags:
        color = colors.get(t, "#7f8c8d")
        html_str += f'<span class="tag-bubble" style="background-color: {color};">{t}</span>'
    return html_str

# --- Scanning Logic ---
def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, 'RSI_14', 'EMA_8', 'EMA_21']].copy()
    df_d.rename(columns={close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low', 'RSI_14': 'RSI', 'EMA_8': 'EMA8', 'EMA_21': 'EMA21'}, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    return df_d.dropna(subset=['Price', 'RSI']), None

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences
    latest_p = df_tf.iloc[-1]
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        # Bullish
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD) and not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                post_df = df_tf.iloc[i + 1 :]
                if not (not post_df.empty and (post_df['RSI'] <= p1['RSI']).any()):
                    tags = [t for t, c in [(f"EMA{EMA8_PERIOD}", latest_p['Price'] >= latest_p['EMA8']), (f"EMA{EMA21_PERIOD}", latest_p['Price'] >= latest_p['EMA21']), ("VOL_HIGH", is_vol_high), ("V_GROW", p2['Volume'] > p1['Volume'])] if c]
                    divergences.append({'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'Tags': ", ".join(tags), 'P1 Date': p1.name.strftime('%Y-%m-%d'), 'Signal Date': p2.name.strftime('%Y-%m-%d'), 'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}", 'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"})
        # Bearish
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD) and not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                post_df = df_tf.iloc[i + 1 :]
                if not (not post_df.empty and (post_df['RSI'] >= p1['RSI']).any()):
                    tags = [t for t, c in [(f"EMA{EMA8_PERIOD}", latest_p['Price'] <= latest_p['EMA8']), (f"EMA{EMA21_PERIOD}", latest_p['Price'] <= latest_p['EMA21']), ("VOL_HIGH", is_vol_high), ("V_GROW", p2['Volume'] > p1['Volume'])] if c]
                    divergences.append({'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe, 'Tags': ", ".join(tags), 'P1 Date': p1.name.strftime('%Y-%m-%d'), 'Signal Date': p2.name.strftime('%Y-%m-%d'), 'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}", 'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"})
    return divergences

# --- App Logic ---
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

if data_option:
    target_url = st.secrets[dataset_map[data_option]]
    csv_buffer = get_confirmed_gdrive_data(target_url)
    if csv_buffer and csv_buffer != "HTML_ERROR":
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
        
        raw_results = []
        progress_bar = st.progress(0, text="Scanning...")
        grouped = master.groupby(t_col)
        for i, (ticker, group) in enumerate(grouped):
            d_d, _ = prepare_data(group.copy())
            if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
            progress_bar.progress((i + 1) / len(grouped))

        if raw_results:
            full_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
            all_detected = sorted(full_df['Ticker'].unique())
            
            # --- restored: EXCLUSION SELECTOR ---
            st.divider()
            st.subheader("🎯 Strike Zones: Filter Results")
            selected_tickers = st.multiselect("Include/Exclude specific tickers from the report:", options=all_detected, default=all_detected)
            
            filtered_df = full_df[full_df['Ticker'].isin(selected_tickers)]
            
            for tf in ['Daily']:
                st.divider()
                st.header(f"📅 {tf} Divergence Analysis")
                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {s_type} Signals")
                    tbl_df = filtered_df[(filtered_df['Type']==s_type)].copy()
                    if not tbl_df.empty:
                        html = '<table><thead><tr><th class="align-left">Ticker</th><th class="align-left">Tags</th><th class="align-center">P1 Date</th><th class="align-center">Signal Date</th><th class="align-center">RSI</th><th class="align-left">P1 Price</th><th class="align-left">P2 Price</th></tr></thead><tbody>'
                        for _, r in tbl_df.iterrows():
                            html += f'<tr><td class="align-left"><b>{r["Ticker"]}</b></td><td class="align-left">{style_tags(r["Tags"])}</td><td class="align-center">{r["P1 Date"]}</td><td class="align-center">{r["Signal Date"]}</td><td class="align-center">{r["RSI"]}</td><td class="align-left">{r["P1 Price"]}</td><td class="align-left">{r["P2 Price"]}</td></tr>'
                        st.markdown(html + '</tbody></table>', unsafe_allow_html=True)
                    else: st.write("No signals for current filter.")
        else: st.warning("No signals found.")

        # --- Footer ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Strategy Logic")
            st.markdown(f"""
            * **Signal Window**: Valid signals within the last **{SIGNAL_LOOKBACK_PERIOD} periods**.
            * **Lookback Window**: Searches preceding **{DIVERGENCE_LOOKBACK} periods** for extremes.
            * **Exclusions**: Disqualified if RSI crosses the 50 centerline between P1 and P2.
            * **Breakage**: Invalidated if price or RSI break the extreme established at P1 before a new signal.
            * **Bullish/Bearish**: New price extreme, but RSI is higher/lower than previous extreme.
            """)
        with col2:
            st.subheader("🏷️ Tags Explained")
            st.markdown(f"""
            * **EMA{EMA8_PERIOD} / EMA{EMA21_PERIOD}**: Price currently holding **above** (Bullish) or **below** (Bearish) these levels.
            * **VOL_HIGH**: Volume > 150% of {VOL_SMA_PERIOD}-day average.
            * **V_GROW**: Signal Volume > P1 Volume.
            """)
