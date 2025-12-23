import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="RSI Divergences Live",
    page_icon="📉",
    layout="wide"
)

st.title("📉 RSI Divergences Live")
st.markdown("""
This application pulls consolidated data directly from Google Drive to identify and visualize RSI Divergences.
""")

# --- Data Connection ---
# Instructions: 
# 1. Share your Google Drive CSV with "Anyone with the link" as a Viewer.
# 2. In your Streamlit Secrets (on the Cloud dashboard), add the URL for:
#    combined_data_divergences.csv, combined_data_sp500.csv, and combined_data_midcap.csv
# This uses the st-gsheets-connection or direct URL reading.

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(url):
    try:
        # If the file is small enough, direct CSV read via export link works best
        # Replace the 'open?id=' part of your drive link with 'uc?export=download&id='
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Placeholder URLs - Replace these with your actual "Direct Download" Google Drive links
DIVERGENCE_DATA_URL = st.sidebar.text_input("Divergences CSV Link", placeholder="Paste Drive direct download link here")

if DIVERGENCE_DATA_URL:
    with st.spinner("Fetching data from Drive..."):
        df_div = load_data(DIVERGENCE_DATA_URL)

    if not df_div.empty:
        # --- Sidebar Filters ---
        st.sidebar.header("Filters")
        all_tickers = sorted(df_div['Ticker'].unique())
        selected_ticker = st.sidebar.selectbox("Select Ticker", all_tickers)
        
        date_range = st.sidebar.slider(
            "Select Date Range",
            min_value=df_div['Date'].min().to_pydatetime(),
            max_value=df_div['Date'].max().to_pydatetime(),
            value=(df_div['Date'].max() - timedelta(days=90)).to_pydatetime()
        )

        # --- Filter Data ---
        ticker_data = df_div[df_div['Ticker'] == selected_ticker].copy()
        ticker_data = ticker_data[ticker_data['Date'] >= date_range]

        # --- Visualizations ---
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(f"Price & RSI Analysis: {selected_ticker}")
            
            # Create subplots manually with Plotly
            fig = go.Figure()

            # Price Chart
            fig.add_trace(go.Scatter(
                x=ticker_data['Date'], y=ticker_data['Close'],
                name="Price", line=dict(color='royalblue', width=2)
            ))

            # Add RSI to a second Y-axis or separate section
            # For brevity in this example, we'll focus on the data table if columns aren't present
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Stats")
            last_price = ticker_data['Close'].iloc[-1]
            st.metric("Last Price", f"${last_price:.2f}")
            st.write(f"Data points: {len(ticker_data)}")

        # --- Data Table ---
        with st.expander("View Raw Data Table"):
            st.dataframe(ticker_data, use_container_width=True)
    else:
        st.info("Please enter a valid Google Drive CSV direct download link in the sidebar.")
else:
    st.info("Waiting for data link...")

# --- Footer ---
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
