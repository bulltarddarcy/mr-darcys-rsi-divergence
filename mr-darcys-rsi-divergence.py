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
    /* Table layout fixed ensures consistent column widths across all tables */
    table { 
        width: 100%; 
        border-collapse: collapse; 
        table-layout: fixed; 
        margin-bottom: 2rem;
    }
    
    /* Header Styling and Alignment */
    thead tr th {
        background-color: #f0f2f6 !important;
        color: #31333f !important;
        padding: 12px !important;
        border-bottom: 2px solid #dee2e6;
    }
    
    /* Specific Fixed Widths for consistency */
    th:nth-child(1) { width: 10%; } /* Ticker */
    th:nth-child(2) { width: 32%; } /* Tags */
    th:nth-child(3) { width: 12%; } /* P1 Date */
    th:nth-child(4) { width: 12%; } /* Signal Date */
    th:nth-child(5) { width: 10%; } /* RSI */
    th:nth-child(6) { width: 12%; } /* P1 Price */
    th:nth-child(7) { width: 12%; } /* P2 Price */
    
    /* Body Cell Padding and wrapping */
    tbody tr td { 
        padding: 10px !important; 
        border-bottom: 1px solid #eee; 
        word-wrap: break-word;
    }

    /* Alignment Rules */
    .align-left { text-align: left !important; }
    .align-center { text-align: center !important; }

    /* Tag Bubble Styling */
    .tag-bubble {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        color: white;
        white-space: nowrap;
    }

    /* Custom Grey Note Styling */
    .grey-note {
        color: #888888;
        font-size: 16px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 RSI Divergence Scanner")

# Replaced st.info with a light grey markdown note
st.markdown('<div class="grey-note">ℹ️ See bottom of page for strategy logic and tag explanations.</div>', unsafe_allow_html=True)

# --- Helpers ---
def style_tags(tag_str):
    if not tag_str: return ''
    tags = tag_str.split(", ")
    html_str = ''
    colors = {
        f"EMA{EMA8_PERIOD}": "#4a90e2", 
        f"EMA{EMA21_PERIOD}": "#9b59b6", 
        "VOL_HIGH": "#e67e22",        
        "V_GROW": "#27ae60"           
    }
    for t in tags:
        color = colors.get(t, "#7f8c8d")
        html_str += f'<span class="tag-bubble" style="background-color: {color};">{t}</span>'
    return html_str

# --- Logic Functions ---
def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close_col, w_vol_col, w_rsi_col = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_ema8_col, w_ema21_col = 'W_EMA_8', 'W_EMA_21'
    w_high_col, w_low_col = 'W_HIGH', 'W_LOW'

    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={
        close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low',
        d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'
    }, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])

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
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences
    latest_p = df_tf.iloc[-1]

    def get_date_str(p):
        return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        # Bullish Divergence
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] <= p1['RSI']).any()):
                        tags = []
                        # Bullish: Above EMA
                        if 'EMA8' in latest_p and latest_p['Price'] >= latest_p['EMA8']: tags.append(f"EMA{EMA8_PERIOD}")
                        if 'EMA21' in latest_p and latest_p['Price'] >= latest_p['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROW")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"
                        })

        # Bearish Divergence
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] >= p1['RSI']).any()):
                        tags = []
                        # Bearish: Below EMA
                        if 'EMA8' in latest_p and latest_p['Price'] <= latest_p['EMA8']: tags.append(f"EMA{EMA8_PERIOD}")
                        if 'EMA21' in latest_p and latest_p['Price'] <= latest_p['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROW")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"
                        })
    return divergences

# --- App Logic ---
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

if not data_option:
    st.warning("Please select a dataset.")
    st.stop()

try:
    secret_key_name = dataset_map[data_option]
    target_url = st.secrets[secret_key_name]
except KeyError:
    st.error("Secret error.")
    st.stop()

csv_buffer = get_confirmed_gdrive_data(target_url)

if csv_buffer and csv_buffer != "HTML_ERROR":
    try:
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
        all_tickers = sorted(master[t_col].unique())
        
        with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
            sq = st.text_input("Filter...").upper()
            ft = [t for t in all_tickers if sq in t]
            cols = st.columns(6)
            for i, ticker in enumerate(ft): cols[i % 6].write(ticker)

        raw_results = []
        progress_bar = st.progress(0, text="Scanning...")
        grouped = master.groupby(t_col)
        for i, (ticker, group) in enumerate(grouped):
            d_d, d_w = prepare_data(group.copy())
            if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
            if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
            progress_bar.progress((i + 1) / len(grouped))

        if raw_results:
            res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
            consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
            
            for tf in ['Daily', 'Weekly']:
                st.divider()
                st.header(f"📅 {tf} Divergence Analysis")
                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {s_type} Signals")
                    tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                    if not tbl_df.empty:
                        display_df = tbl_df.drop(columns=['Type', 'Timeframe'])
                        
                        align_map = {
                            'Ticker': 'align-left', 'Tags': 'align-left',
                            'P1 Price': 'align-left', 'P2 Price': 'align-left',
                            'P1 Date': 'align-center', 'Signal Date': 'align-center', 'RSI': 'align-center'
                        }

                        html = '<table><thead><tr>'
                        for col in display_df.columns:
                            cls = align_map.get(col, 'align-center')
                            html += f'<th class="{cls}">{col}</th>'
                        html += '</tr></thead><tbody>'
                        
                        for _, row in display_df.iterrows():
                            html += '<tr>'
                            html += f'<td class="align-left"><b>{row["Ticker"]}</b></td>'
                            html += f'<td class="align-left">{style_tags(row["Tags"])}</td>'
                            html += f'<td class="align-center">{row["P1 Date"]}</td>'
                            html += f'<td class="align-center">{row["Signal Date"]}</td>'
                            html += f'<td class="align-center">{row["RSI"]}</td>'
                            html += f'<td class="align-left">{row["P1 Price"]}</td>'
                            html += f'<td class="align-left">{row["P2 Price"]}</td>'
                            html += '</tr>'
                        html += '</tbody></table>'
                        
                        st.markdown(html, unsafe_allow_html=True)
                    else: st.write("No signals.")
        else: st.warning("No signals.")

        # --- Footer Logic ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Strategy Logic")
            st.markdown(f"""
            * **Signal Window**: Scans signals within the last **{SIGNAL_LOOKBACK_PERIOD} periods**.
            * **Lookback Window**: Searches preceding **{DIVERGENCE_LOOKBACK} periods** for extremes.
            * **Bullish Divergence**: New price low, but RSI is higher than previous low.
            * **Bearish Divergence**: New price high, but RSI is lower than previous high.
            """)
        with col2:
            st.subheader("🏷️ Tags Explained")
            st.markdown(f"""
            * **EMA{EMA8_PERIOD} / EMA{EMA21_PERIOD}**: Added if current price is holding **above** (Bullish) or **below** (Bearish) these levels.
            * **VOL_HIGH**: Volume > 150% of {VOL_SMA_PERIOD}-day average.
            * **V_GROW**: Signal Volume > P1 Volume.
            """)
    except Exception as e: st.error(f"Error: {e}")
