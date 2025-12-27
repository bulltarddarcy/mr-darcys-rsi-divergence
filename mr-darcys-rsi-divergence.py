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
TOLERANCE = 2

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Forward Win Rates", layout="wide")

st.markdown("""
    <style>
    /* Styling for the Win Rate Card and Tables */
    .analysis-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 2rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .stats-table {
        width: 100%;
        border-collapse: collapse;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 14px;
        margin-top: 10px;
    }
    .stats-table th {
        text-align: left;
        color: #6a737d;
        border-bottom: 1px solid #dfe1e4;
        padding: 8px;
        font-weight: 400;
    }
    .stats-table td {
        padding: 8px;
        border-bottom: 1px solid #f6f8fa;
    }
    .pos-ret { color: #28a745; }
    .neg-ret { color: #d73a49; }
    .grey-note {
        color: #888888;
        font-size: 14px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 RSI Forward Win Rates")

# --- Logic Functions ---
def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    d_rsi_col = 'RSI_14'

    if not all([date_col, close_col]): return None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_clean = df[[close_col, d_rsi_col]].copy()
    df_clean.rename(columns={close_col: 'Price', d_rsi_col: 'RSI'}, inplace=True)
    return df_clean.dropna()

def calculate_forward_stats(df, current_rsi):
    lower_bound = current_rsi - TOLERANCE
    upper_bound = current_rsi + TOLERANCE
    
    # Exclude the very last row to avoid data leakage (using historical samples only)
    historical_df = df.iloc[:-1]
    matches = historical_df[(historical_df['RSI'] >= lower_bound) & (historical_df['RSI'] <= upper_bound)].index
    
    windows = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
    results = []
    
    for w in windows:
        forward_returns = []
        for idx in matches:
            pos = df.index.get_loc(idx)
            if pos + w < len(df):
                p_entry = df.iloc[pos]['Price']
                p_exit = df.iloc[pos + w]['Price']
                forward_returns.append((p_exit - p_entry) / p_entry)
        
        if forward_returns:
            win_rate = (sum(1 for r in forward_returns if r > 0) / len(forward_returns)) * 100
            avg_ret = np.mean(forward_returns) * 100
            med_ret = np.median(forward_returns) * 100
            results.append({
                "Days": w,
                "Win Rate": f"{win_rate:.1f}%",
                "Avg Ret": f"{'+' if avg_ret >= 0 else ''}{avg_ret:.2f}%",
                "Med Ret": f"{'+' if med_ret >= 0 else ''}{med_ret:.2f}%",
                "Raw Avg": avg_ret,
                "Raw Med": med_ret
            })
    return results, len(matches)

# --- App Execution ---
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

if not data_option:
    st.stop()

secret_key_name = dataset_map[data_option]
target_url = st.secrets[secret_key_name]
csv_buffer = get_confirmed_gdrive_data(target_url)

if csv_buffer and csv_buffer != "HTML_ERROR":
    master = pd.read_csv(csv_buffer)
    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
    all_tickers = sorted(master[t_col].unique())
    
    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
        sq = st.text_input("Filter...").upper()
        ft = [t for t in all_tickers if sq in t]
        cols = st.columns(6)
        for i, ticker in enumerate(ft): cols[i % 6].write(ticker)

    # --- TESTING LOCK: NFLX ---
    test_ticker = "NFLX"
    if test_ticker in all_tickers:
        group = master[master[t_col] == test_ticker]
        df_clean = prepare_data(group)
        
        if df_clean is not None and not df_clean.empty:
            curr_rsi = df_clean['RSI'].iloc[-1]
            stats, samples = calculate_forward_stats(df_clean, curr_rsi)
            
            # Rendering the UI Box
            st.markdown(f"""
            <div class="analysis-card">
                <h3 style="margin-top:0;">RSI Analysis: {test_ticker}</h3>
                <p><b>Current RSI: {curr_rsi:.2f}</b> <span style="color:#6a737d;">(live as of {df_clean.index[-1].strftime('%Y-%m-%d')})</span></p>
                <p>RSI Range: [{curr_rsi - TOLERANCE:.2f}, {curr_rsi + TOLERANCE:.2f}]<br>
                Matching Periods: {samples}</p>
            """, unsafe_allow_html=True)
            
            # Short-Term Table
            st.write("**Short-Term Forward Returns**")
            st_html = '<table class="stats-table"><tr><th>Days</th><th>Win Rate</th><th>Avg Ret</th><th>Med Ret</th></tr>'
            for row in stats[:6]:
                avg_class = "pos-ret" if row['Raw Avg'] >= 0 else "neg-ret"
                med_class = "pos-ret" if row['Raw Med'] >= 0 else "neg-ret"
                st_html += f"<tr><td>{row['Days']}</td><td>{row['Win Rate']}</td><td class='{avg_class}'>{row['Avg Ret']}</td><td class='{med_class}'>{row['Med Ret']}</td></tr>"
            st_html += "</table>"
            st.markdown(st_html, unsafe_allow_html=True)
            
            # Long-Term Table
            st.write("<br>**Long-Term Forward Returns**", unsafe_allow_html=True)
            lt_html = '<table class="stats-table"><tr><th>Days</th><th>Win Rate</th><th>Avg Ret</th><th>Med Ret</th></tr>'
            for row in stats[6:]:
                avg_class = "pos-ret" if row['Raw Avg'] >= 0 else "neg-ret"
                med_class = "pos-ret" if row['Raw Med'] >= 0 else "neg-ret"
                lt_html += f"<tr><td>{row['Days']}</td><td>{row['Win Rate']}</td><td class='{avg_class}'>{row['Avg Ret']}</td><td class='{med_class}'>{row['Med Ret']}</td></tr>"
            lt_html += "</table>"
            st.markdown(lt_html, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="grey-note">Tolerance: ±{TOLERANCE} | {samples} samples • Today at {datetime.now().strftime('%I:%M %p')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No data available for {test_ticker}.")
    else:
        st.warning(f"{test_ticker} not found in dataset.")
