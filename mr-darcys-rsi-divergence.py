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
    Downloads files from Drive, handling the virus scan warning for large files
    by extracting the confirmation token from the session cookies.
    """
    base_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # First attempt to get the confirmation token
    response = session.get(base_url, params={'id': file_id}, stream=True)
    
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            break
            
    if confirm_token:
        # Second request with the confirmation token
        params = {'id': file_id, 'confirm': confirm_token}
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
    
    # Resample if Weekly is selected
    if timeframe == 'Weekly':
        # aggregate Price as 'last' to maintain consistency with original script
        df = df.resample('W-FRI', on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Price': 'last',
            'Volume': 'sum',
            'Ticker': 'first'
        }).reset_index().dropna()

    df['RSI'] = calculate_rsi(df['Price'], RSI_PERIOD)
    df['EMA8'] = calculate_ema(df['Price'], EMA_PERIOD) 
    df['EMA21'] = calculate_ema(df['Price'], EMA21_PERIOD)
    df['VolSMA'] = df['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    return df.dropna()

def find_divergences(df_timeframe, ticker):
    divergences = {'bullish': [], 'bearish': []}
    if len(df_timeframe) < DIVERGENCE_LOOKBACK + 1:
        return divergences
            
    start_index = max(DIVERGENCE_LOOKBACK, len(df_timeframe) - SIGNAL_LOOKBACK_PERIOD)
    
    # Track raw hits before consolidation
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

        # Standard Bullish (logic looks for second_point['Price'])
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

        # Standard Bearish
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
    
    # Consolidation Logic (Matches consolidateSignals in original JS/HTML)
    if bullish_hits:
        latest_bull = bullish_hits[-1]
        earliest_first_date = min([h['First Date'] for h in bullish_hits])
        all_tags = set()
        for h in bullish_hits: all_tags.update(h['Tags'])
        
        divergences['bullish'].append({
            'Ticker': ticker,
            'Tags': " ".join(sorted(list(all_tags))),
            'First Date': earliest_first_date.strftime('%Y-%m-%d'),
            'Signal Date': latest_bull['Signal Date'].strftime('%Y-%m-%d'),
            'RSI': f"{int(latest_bull['firstRSI'])} → {int(latest_bull['secondRSI'])}",
            'Price 1': latest_bull['Price 1'],
            'Price 2': latest_bull['Price 2']
        })

    if bearish_hits:
        latest_bear = bearish_hits[-1]
        earliest_first_date = min([h['First Date'] for h in bearish_hits])
        all_tags = set()
        for h in bearish_hits: all_tags.update(h['Tags'])
        
        divergences['bearish'].append({
            'Ticker': ticker,
            'Tags': " ".join(sorted(list(all_tags))),
            'First Date': earliest_first_date.strftime('%Y-%m-%d'),
            'Signal Date': latest_bear['Signal Date'].strftime('%Y-%m-%d'),
            'RSI': f"{int(latest_bear['firstRSI'])} → {int(latest_bear['secondRSI'])}",
            'Price 1': latest_bear['Price 1'],
            'Price 2': latest_bear['Price 2']
        })

    return divergences

# --- Data Loading ---

@st.cache_data(ttl=3600, show_spinner="Loading dataset...")
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
        
        df = pd.read_csv(io.BytesIO(response.content), dtype=dtype_dict)
        
        # Standardize Columns immediately after loading (Matches prepare_data in original script)
        # 1. Clean column names to handle casing and spaces
        df.columns = [col.strip().upper() for col in df.columns]
        
        # 2. Identify and rename Ticker column robustly
        ticker_col = next((col for col in df.columns if 'TICKER' in col or 'SYMBOL' in col), None)
        if ticker_col:
            df.rename(columns={ticker_col: 'Ticker'}, inplace=True)
            df['Ticker'] = df['Ticker'].astype('category')
        else:
            # If no ticker column, we can't group data properly
            st.error("Missing 'Ticker' or 'Symbol' column in dataset.")
            return pd.DataFrame()
        
        # 3. Identify and rename Close to Price
        close_col = next((col for col in df.columns if 'CLOSE' in col or 'PRICE' in col), None)
        if close_col:
            df.rename(columns={close_col: 'Price'}, inplace=True)
        else:
            st.error("Missing 'Close' or 'Price' column in dataset.")
            return pd.DataFrame()

        # 4. Handle Date parsing
        date_col = next((c for c in df.columns if 'DATE' in c), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        if date_col != 'Date': 
            df.rename(columns={date_col: 'Date'}, inplace=True)
            
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
            
            # Robustly get unique tickers after column normalization
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
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Price'], name="Price", line=dict(color='white')))
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
        st.warning("Data load failed. Ensure the Google Drive files are shared as 'Anyone with the link' and are set to 'Viewer'.")
else:
    st.info("Select a dataset to begin.")
