# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import requests
import re
import time
from datetime import date, timedelta, datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# --- DATA LOADERS & GOOGLE DRIVE UTILS ---

def get_gdrive_binary_data(url):
    """
    Robust Google Drive downloader.
    Uses Session Cookies to bypass 'Virus Scan' and 'High Traffic' warnings automatically.
    """
    try:
        # 1. Extract ID
        match = re.search(r'/d/([a-zA-Z0-9_-]{25,})', url)
        if not match:
            match = re.search(r'id=([a-zA-Z0-9_-]{25,})', url)
            
        if not match:
            st.error(f"Invalid Google Drive URL: {url}")
            return None
            
        file_id = match.group(1)
        
        # 2. Standard Download URL
        download_url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        
        # 3. First Attempt: Try to get the file
        # We allow a longer timeout (50s) to handle server lags
        response = session.get(download_url, params={'id': file_id}, stream=True, timeout=50)
        
        # 4. Check for the Warning Token in Cookies
        # Google sets a cookie named 'download_warning_...' when it intercepts the download.
        token = None
        for key, value in session.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # 5. If we found a warning token, we automatically send it back to confirm
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(download_url, params=params, stream=True, timeout=50)
            
        # 6. Final Validation
        if response.status_code == 200:
            # Peek at the first 100 bytes to ensure it's not still an HTML error page
            try:
                chunk = next(response.iter_content(chunk_size=100), b"")
            except StopIteration:
                return None
            
            # If the file content starts with "<!DOCTYPE html", the bypass failed (likely a hard 24hr ban)
            if chunk.strip().startswith(b"<!DOCTYPE html"):
                return None
            
            # Success: Stitch the chunk back together with the rest of the stream
            return BytesIO(chunk + response.raw.read())
            
        return None

    except Exception as e:
        return None

