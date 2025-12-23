import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import io

# --- Page Config ---
st.set_page_config(
    page_title="RSI Divergences Live",
    page_icon="📊",
    layout="wide"
)

# --- Constants from local script (Divergence_make_dashboard.py) ---
RSI_PERIOD = 14
EMA_PERIOD = 8 
EMA21_PERIOD = 21 
VOL_SMA_PERIOD = 30 
DIVERGENCE_LOOKBACK = 180  
SIGNAL_LOOKBACK_PERIOD = 15 

# --- Improved Google Drive Loading Utility ---

def get_drive_file_id(url):
    """Extracts file ID from various Google Drive link formats."""
    if not url: return None
    if "/file/d/" in url:
        return url.split('/d/')[-1].split('/')[0].split('?')[0]
    elif "id=" in url:
        return url.split('id=')[-1].split('&')[0]
    return None

def download_large_drive_file(file_id):
    """Downloads large files from Drive by bypassing the virus scan warning."""
    base_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # First request to check for the 'confirm' token
    response = session.get(base_url, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        # Second request with the confirmation token
        params = {'id': file_id, 'confirm': token}
        response = session.get(base_url, params=params, stream=True)
        
    return response

# --- Technical Indicator Logic (Matches divergence_make_dashboard.py) ---

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf) 
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean().round(2)

@st.cache_data
def process_ticker_indicators(df, timeframe='Daily'):
    df = df.copy()
    if timeframe == 'Weekly':
        df = df.resample('W-FRI', on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Ticker': 'first'
        }).reset_index().dropna()

    df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
    df['EMA8'] = calculate_ema(df['Close'], EMA_PERIOD) 
    df['EMA21'] = calculate_ema(df['Close'], EMA21_PERIOD)
    df['VolSMA'] = df['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    return df.dropna()

def find_divergences(df_timeframe, ticker):
    divergences = {'bullish': [], 'bearish': []}
    if len(df_timeframe) < DIVERGENCE_LOOKBACK + 1:
        return divergences
            
    start_index = max(DIVERGENCE_LOOKBACK, len(df_timeframe) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_index, len(df_timeframe)):
        second_point = df_timeframe.iloc[i]
        lookback_df = df_timeframe.iloc[i - DIVERGENCE_LOOKBACK : i]
        
        min_rsi_idx = lookback_df['RSI'].idxmin() 
        first_point_low = lookback_df.loc[min_rsi_idx]
        max_rsi_idx = lookback_df['RSI'].idxmax()
        first_point_high = lookback_df.loc[max_rsi_idx]

        is_vol_high = second_point['Volume'] > (second_point['VolSMA'] * 1.5)
        v_growth = second_point['Volume'] > first_point_low['Volume']
        
        tags = []
        if is_vol_high: tags.append("VOL_HIGH")
        if v_growth: tags.append("V_GROWTH")

        if second_point['RSI'] > first_point_low['RSI'] and second_point['Close'] < lookback_df['Close'].min():
            current_tags = list(tags)
            if second_point['Close'] >= second_point['EMA8']:
                current_tags.append("EMA8")
            divergences['bullish'].append({
                'Ticker': ticker,
                'Tags': " ".join(current_tags),
                'First Date': first_point_low['Date'].strftime('%Y-%m-%d'),
                'Signal Date': second_point['Date'].strftime('%Y-%m-%d'),
                'RSI': f"{int(first_point_low['RSI'])} → {int(second_point['RSI'])}",
                'Price 1': round(float(first_point_low['Close']), 2),
                'Price 2': round(float(second_point['Close']), 2)
            })

        if second_point['RSI'] < first_point_high['RSI'] and second_point['Close'] > lookback_df['Close'].max():
            current_tags = list(tags)
            if second_point['Close'] <= second_point['EMA21']:
                current_tags.append("EMA21")
            divergences['bearish'].append({
                'Ticker': ticker,
                'Tags': " ".join(current_tags),
                'First Date': first_point_high['Date'].strftime('%Y-%m-%d'),
                'Signal Date': second_point['Date'].strftime('%Y-%m-%d'),
                'RSI': f"{int(first_point_high['RSI'])} → {int(second_point['RSI'])}",
                'Price 1': round(float(first_point_high['Close']), 2),
                'Price 2': round(float(second_point['Close']), 2)
            })
    return divergences

# --- Data Loading ---

@st.cache_data(ttl=3600, show_spinner="Loading large dataset...")
def load_large_data(url):
    file_id = get_drive_file_id(url)
    if not file_id:
        st.error("Could not parse File ID from URL.")
        return pd.DataFrame()
        
    try:
        response = download_large_drive_file(file_id)
        if response.status_code != 200:
            st.error(f"Failed to fetch data. Status: {response.status_code}")
            return pd.DataFrame()
            
        dtype_dict = {
            'Ticker': 'category', 'Close': 'float32', 'Open': 'float32', 
            'High': 'float32', 'Low': 'float32', 'Volume': 'float32'
        }
        
        # Use io.BytesIO to read the binary content directly into pandas
        df = pd.read_csv(io.BytesIO(response.content), dtype=dtype_dict)
        
        # Handle Date parsing
        date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        if date_col != 'Date': df.rename(columns={date_col: 'Date'}, inplace=True)
            
        return df
    except Exception as e:
        st.error(f"Critical load error: {str(e)}")
        return pd.DataFrame()

# --- App UI ---

st.title("📉 RSI Divergences Live")

PRESETS = {
    "Darcy's List": st.secrets.get("URL_DIVERGENCES"),
    "Midcap": st.secrets.get("URL_MIDCAP"),
    "S&P 500": st.secrets.get("URL_SP500")
}

AVAILABLE_DATASETS = {k: v for k, v in PRESETS.items() if v is not None}

if not AVAILABLE_DATASETS:
    st.error("No secrets configured in Streamlit Cloud.")
    st.stop()

st.sidebar.markdown("### Data Configuration")
selected_dataset = st.sidebar.radio("Select Dataset", list(AVAILABLE_DATASETS.keys()), index=0)
timeframe = st.sidebar.radio("RSI Divergence Length", ["Daily", "Weekly"], index=0)
view_mode = st.sidebar.radio("View Mode", ["Summary Dashboard", "Ticker Detail"], index=0)

final_url = AVAILABLE_DATASETS[selected_dataset]

DIV_COLUMN_CONFIG = {
    "Ticker": st.column_config.TextColumn(width="small"),
    "Tags": st.column_config.TextColumn(width="medium"),
    "First Date": st.column_config.TextColumn(width="small"),
    "Signal Date": st.column_config.TextColumn(width="small"),
    "RSI": st.column_config.TextColumn(width="small"),
    "Price 1": st.column_config.NumberColumn(format="$%.2f", width="small"),
    "Price 2": st.column_config.NumberColumn(format="$%.2f", width="small"),
}

if final_url:
    raw_df = load_large_data(final_url)

    if not raw_df.empty:
        if view_mode == "Summary Dashboard":
            st.header(f"System-Wide Scanner: {selected_dataset} ({timeframe})")
            all_bullish, all_bearish = [], []
            unique_tickers = raw_df['Ticker'].unique()
            
            scan_progress = st.progress(0)
            for idx, ticker in enumerate(unique_tickers):
                t_df = raw_df[raw_df['Ticker'] == ticker].sort_values('Date')
                t_df = process_ticker_indicators(t_df, timeframe)
                divs = find_divergences(t_df, ticker)
                all_bullish.extend(divs['bullish'])
                all_bearish.extend(divs['bearish'])
                scan_progress.progress((idx + 1) / len(unique_tickers))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🟢 Bullish Divergences")
                if all_bullish: 
                    df_bull = pd.DataFrame(all_bullish).sort_values('Signal Date', ascending=False)
                    st.dataframe(df_bull, use_container_width=True, hide_index=True, column_config=DIV_COLUMN_CONFIG)
                else: st.write("None detected.")
            with col2:
                st.subheader("🔴 Bearish Divergences")
                if all_bearish: 
                    df_bear = pd.DataFrame(all_bearish).sort_values('Signal Date', ascending=False)
                    st.dataframe(df_bear, use_container_width=True, hide_index=True, column_config=DIV_COLUMN_CONFIG)
                else: st.write("None detected.")
        else:
            ticker_list = sorted(raw_df['Ticker'].unique())
            selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
            ticker_df = raw_df[raw_df['Ticker'] == selected_ticker].sort_values('Date')
            ticker_df = process_ticker_indicators(ticker_df, timeframe)
            
            st.subheader(f"Analysis: {selected_ticker} ({timeframe})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Close'], name="Price", line=dict(color='white')))
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['EMA8'], name="EMA 8", line=dict(color='cyan', dash='dot')))
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['EMA21'], name="EMA 21", line=dict(color='magenta', dash='dot')))
            fig.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['RSI'], name="RSI", line=dict(color='yellow')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=300, title=f"RSI ({timeframe})", template="plotly_dark")
            st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.warning("Data load failed. Check Drive permissions or Secrets configuration.")
else:
    st.info("Select a dataset to begin.")
