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
st.set_page_config(page_title="RSI Analysis Pro", layout="wide")

st.markdown("""
    <style>
    /* Table styling */
    table { width: 100%; border-collapse: collapse; margin-bottom: 2rem; font-family: monospace; }
    thead tr th { background-color: #f0f2f6 !important; color: #31333f !important; padding: 10px !important; border-bottom: 2px solid #dee2e6; }
    tbody tr td { padding: 8px !important; border-bottom: 1px solid #eee; }
    
    /* Win Rate Card Styling */
    .analysis-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .pos-val { color: #27ae60; font-weight: bold; }
    .neg-val { color: #eb4d4b; font-weight: bold; }
    .tag-bubble { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 14px; font-weight: 600; margin: 2px 4px; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Navigation Sidebar ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Divergence Scanner", "RSI Forward Win Rates"])

# --- Core Functions ---
def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'

    if not all([date_col, close_col, vol_col, high_col, low_col]): return None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={
        close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low',
        d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'
    }, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    return df_d.dropna(subset=['Price', 'RSI'])

def calculate_win_rates(df, current_rsi, tol=2):
    lower, upper = current_rsi - tol, current_rsi + tol
    matches = df[(df['RSI'] >= lower) & (df['RSI'] <= upper)].index
    
    windows = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
    results = []
    
    for w in windows:
        rets = []
        for idx in matches:
            pos = df.index.get_loc(idx)
            if pos + w < len(df):
                future_p = df.iloc[pos + w]['Price']
                entry_p = df.iloc[pos]['Price']
                rets.append((future_p - entry_p) / entry_p)
        
        if rets:
            win_rate = (sum(1 for r in rets if r > 0) / len(rets)) * 100
            avg_ret = np.mean(rets) * 100
            med_ret = np.median(rets) * 100
            results.append({
                "Days": w, "Win Rate": f"{win_rate:.1f}%",
                "Avg": avg_ret, "Med": med_ret
            })
    return results, len(matches)

# --- App Logic ---
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

if data_option:
    secret_key = dataset_map[data_option]
    target_url = st.secrets[secret_key]
    csv_buffer = get_confirmed_gdrive_data(target_url)

    if csv_buffer and csv_buffer != "HTML_ERROR":
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), 'TICKER')

        if page_selection == "RSI Forward Win Rates":
            st.header("🎯 RSI Forward Win Rates")
            
            # --- TESTING LOCK: NFLX ONLY ---
            target_tickers = ["NFLX"]
            
            for ticker in target_tickers:
                t_df = master[master[t_col] == ticker].copy()
                df_clean = prepare_data(t_df)
                
                if df_clean is not None:
                    curr_rsi = df_clean['RSI'].iloc[-1]
                    stats, sample_size = calculate_win_rates(df_clean, curr_rsi)
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h2 style='margin-top:0;'>RSI Analysis: {ticker}</h2>
                        <p><b>Current RSI: {curr_rsi:.2f}</b> (live as of {df_clean.index[-1].strftime('%Y-%m-%d')})</p>
                        <p style="color: #666;">RSI Range: [{curr_rsi-2:.2f}, {curr_rsi+2:.2f}] | Matching Periods: {sample_size}</p>
                    """, unsafe_allow_html=True)
                    
                    # Split into Short and Long Term
                    for title, data_slice in [("Short-Term Forward Returns", stats[:6]), ("Long-Term Forward Returns", stats[6:])]:
                        st.write(f"### {title}")
                        tbl_html = "<table><thead><tr><th>Days</th><th>Win Rate</th><th>Avg Ret</th><th>Med Ret</th></tr></thead><tbody>"
                        for row in data_slice:
                            a_cls = "pos-val" if row['Avg'] > 0 else "neg-val"
                            m_cls = "pos-val" if row['Med'] > 0 else "neg-val"
                            tbl_html += f"""
                                <tr>
                                    <td>{row['Days']}</td>
                                    <td>{row['Win Rate']}</td>
                                    <td class='{a_cls}'>{row['Avg']:+.2f}%</td>
                                    <td class='{m_cls}'>{row['Med']:+.2f}%</td>
                                </tr>"""
                        tbl_html += "</tbody></table>"
                        st.markdown(tbl_html, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        elif page_selection == "Divergence Scanner":
            st.header("📈 RSI Divergence Scanner")
            # [The original divergence logic goes here]
            st.info("The Divergence Scanner logic is ready to be restored here once the Win Rate page is confirmed.")
