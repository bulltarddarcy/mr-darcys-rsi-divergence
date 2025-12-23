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
    """
    Downloads files from Drive, handling the virus scan warning for large files.
    """
    base_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(base_url, params={'id': file_id}, stream=True)
    
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            break
            
    if confirm_token:
        params = {'id': file_id, 'confirm': confirm_token}
        response = session.get(base_url, params=params, stream=True)
        
    return response

# --- Technical Indicator Logic (Calculated BEFORE slicing) ---

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    # Wilder's Smoothing via ewm (com=period-1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf) 
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean().round(2)

@st.cache_data
def process_ticker_indicators(df, timeframe='Daily'):
    """
    FIX: Indicators are calculated on the RAW dataframe first.
    Slicing/Resampling happens AFTER to ensure indicator warm-up parity.
    """
    df = df.copy().sort_values('Date')
    
    # --- STEP 1: Calculate on full raw data ---
    df['RSI'] = calculate_rsi(df['Price'], RSI_PERIOD)
    df['EMA8'] = calculate_ema(df['Price'], EMA_PERIOD) 
    df['EMA21'] = calculate_ema(df['Price'], EMA21_PERIOD)
    df['VolSMA'] = df['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    # --- STEP 2: Resample/Slice if needed ---
    if timeframe == 'Weekly':
        df = df.resample('W-FRI', on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Price': 'last',
            'Volume': 'sum',
            'Ticker': 'first',
            'RSI': 'last',
            'EMA8': 'last',
            'EMA21': 'last',
            'VolSMA': 'last'
        }).reset_index().dropna()
    else:
        df = df.dropna()

    return df

def find_divergences(df_timeframe, ticker):
    divergences = {'bullish': [], 'bearish': []}
    if len(df_timeframe) < DIVERGENCE_LOOKBACK + 1:
        return divergences
            
    start_index = max(DIVERGENCE_LOOKBACK, len(df_timeframe) - SIGNAL_LOOKBACK_PERIOD)
    
    bullish_hits = []
    bearish_hits = []

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

        # Bullish Divergence
        if second_point['RSI'] > first_point_low['RSI'] and second_point['Price'] < lookback_df['Price'].min():
            current_tags = list(tags)
            if second_point['Price'] >= second_point['EMA8']:
                current_tags.append("EMA8")
            bullish_hits.append({
                'Ticker': ticker,
                'Tags': current_tags,
                'First Date': first_point_low['Date'],
                'Signal Date': second_point['Date'],
                'firstRSI': first_point_low['RSI'],
                'secondRSI': second_point['RSI'],
                'Price 1': round(float(first_point_low['Price']), 2),
                'Price 2': round(float(second_point['Price']), 2)
            })

        # Bearish Divergence
        if second_point['RSI'] < first_point_high['RSI'] and second_point['Price'] > lookback_df['Price'].max():
            current_tags = list(tags)
            if second_point['Price'] <= second_point['EMA21']:
                current_tags.append("EMA21")
            bearish_hits.append({
                'Ticker': ticker,
                'Tags': current_tags,
                'First Date': first_point_high['Date'],
                'Signal Date': second_point['Date'],
                'firstRSI': first_point_high['RSI'],
                'secondRSI': second_point['RSI'],
                'Price 1': round(float(first_point_high['Price']), 2),
                'Price 2': round(float(second_point['Price']), 2)
            })
    
    # Consolidation
    if bullish_hits:
        latest_bull = bullish_hits[-1]
        all_tags = set()
        for h in bullish_hits: all_tags.update(h['Tags'])
        divergences['bullish'].append({
            'Ticker': ticker,
            'Tags': " ".join(sorted(list(all_tags))),
            'First Date': min([h['First Date'] for h in bullish_hits]).strftime('%Y-%m-%d'),
            'Signal Date': latest_bull['Signal Date'].strftime('%Y-%m-%d'),
            'RSI': f"{int(latest_bull['firstRSI'])} → {int(latest_bull['secondRSI'])}",
            'Price 1': latest_bull['Price 1'],
            'Price 2': latest_bull['Price 2']
        })

    if bearish_hits:
        latest_bear = bearish_hits[-1]
        all_tags = set()
        for h in bearish_hits: all_tags.update(h['Tags'])
        divergences['bearish'].append({
            'Ticker': ticker,
            'Tags': " ".join(sorted(list(all_tags))),
            'First Date': min([h['First Date'] for h in bearish_hits]).strftime('%Y-%m-%d'),
            'Signal Date': latest_bear['Signal Date'].strftime('%Y-%m-%d'),
            'RSI': f"{int(latest_bear['firstRSI'])} → {int(latest_bear['secondRSI'])}",
            'Price 1': latest_bear['Price 1'],
            'Price 2': latest_bear['Price 2']
        })

    return divergences

# --- Data Loading ---

@st.cache_data(ttl=3600)
def load_large_data(url):
    file_id = get_drive_file_id(url)
    if not file_id: return pd.DataFrame()
    try:
        response = download_large_drive_file(file_id)
        if response.status_code != 200: return pd.DataFrame()
        
        df = pd.read_csv(io.BytesIO(response.content))
        df.columns = [col.strip().upper() for col in df.columns]
        
        column_map = {
            'TICKER': 'Ticker', 'SYMBOL': 'Ticker',
            'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
            'CLOSE': 'Price', 'PRICE': 'Price',
            'VOLUME': 'Volume', 'DATE': 'Date'
        }
        
        df.rename(columns={c: column_map[c] for c in df.columns if c in column_map}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return pd.DataFrame()

# --- App UI ---

st.title("📉 RSI Divergences Live")

PRESETS = {
    "Darcy's List": st.secrets.get("URL_DIVERGENCES"),
    "Midcap": st.secrets.get("URL_MIDCAP"),
    "S&P 500": st.secrets.get("URL_SP500")
}

AVAILABLE_DATASETS = {k: v for k, v in PRESETS.items() if v is not None}

st.sidebar.markdown("### Data Configuration")
selected_dataset = st.sidebar.radio("Select Dataset", list(AVAILABLE_DATASETS.keys()) if AVAILABLE_DATASETS else ["None"])
timeframe = st.sidebar.radio("RSI Divergence Length", ["Daily", "Weekly"])
view_mode = st.sidebar.radio("View Mode", ["Summary Dashboard", "Ticker Detail"])

if AVAILABLE_DATASETS and selected_dataset != "None":
    raw_df = load_large_data(AVAILABLE_DATASETS[selected_dataset])

    if not raw_df.empty:
        if view_mode == "Summary Dashboard":
            st.header(f"Scanner: {selected_dataset} ({timeframe})")
            all_bullish, all_bearish = [], []
            tickers = raw_df['Ticker'].unique()
            
            prog = st.progress(0)
            for idx, ticker in enumerate(tickers):
                t_df = raw_df[raw_df['Ticker'] == ticker]
                t_df = process_ticker_indicators(t_df, timeframe)
                divs = find_divergences(t_df, ticker)
                all_bullish.extend(divs['bullish'])
                all_bearish.extend(divs['bearish'])
                prog.progress((idx + 1) / len(tickers))
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🟢 Bullish")
                if all_bullish: st.dataframe(pd.DataFrame(all_bullish).sort_values('Signal Date', ascending=False), hide_index=True)
                else: st.write("No signals.")
            with c2:
                st.subheader("🔴 Bearish")
                if all_bearish: st.dataframe(pd.DataFrame(all_bearish).sort_values('Signal Date', ascending=False), hide_index=True)
                else: st.write("No signals.")
        else:
            ticker_list = sorted(raw_df['Ticker'].unique())
            sel_ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
            t_df = raw_df[raw_df['Ticker'] == sel_ticker]
            t_df = process_ticker_indicators(t_df, timeframe)
            
            st.subheader(f"Analysis: {sel_ticker} ({timeframe})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['Date'], y=t_df['Price'], name="Price", line=dict(color='white')))
            fig.add_trace(go.Scatter(x=t_df['Date'], y=t_df['EMA8'], name="EMA 8", line=dict(color='cyan', dash='dot')))
            fig.add_trace(go.Scatter(x=t_df['Date'], y=t_df['EMA21'], name="EMA 21", line=dict(color='magenta', dash='dot')))
            fig.update_layout(height=450, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=t_df['Date'], y=t_df['RSI'], name="RSI", line=dict(color='yellow')))
            fig_r.add_hline(y=70, line_dash="dash", line_color="red")
            fig_r.add_hline(y=30, line_dash="dash", line_color="green")
            fig_r.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig_r, use_container_width=True)
