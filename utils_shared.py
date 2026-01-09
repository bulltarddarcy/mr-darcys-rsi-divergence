# utils_shared.py
import streamlit as st
import pandas as pd
import requests
import re
from io import BytesIO

# --- PERFORMANCE OPTIMIZATION: GLOBAL SESSION ---
# Reusing the session enables HTTP Keep-Alive for faster Drive downloads.
GLOBAL_SESSION = requests.Session()

def get_gdrive_binary_data(url):
    """
    Robust Google Drive downloader using a global session for speed.
    Handles 'virus scan' confirmation pages and various URL formats.
    """
    try:
        # 1. Extract ID
        match = re.search(r'/d/([a-zA-Z0-9_-]{25,})', url)
        if not match:
            match = re.search(r'id=([a-zA-Z0-9_-]{25,})', url)
            
        if not match:
            # It might be a direct link or invalid; return None or let caller handle
            return None
            
        file_id = match.group(1)
        download_url = "https://drive.google.com/uc?export=download"
        
        # 2. First Attempt (Using Global Session)
        response = GLOBAL_SESSION.get(download_url, params={'id': file_id}, stream=True, timeout=30)
        
        # 3. Check for "Virus Scan" HTML Page
        if "text/html" in response.headers.get("Content-Type", "").lower():
            content = response.text
            token_match = re.search(r'confirm=([a-zA-Z0-9_]+)', content)
            
            if token_match:
                token = token_match.group(1)
                params = {'id': file_id, 'confirm': token}
                response = GLOBAL_SESSION.get(download_url, params=params, stream=True, timeout=30)
            else:
                for key, value in GLOBAL_SESSION.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'confirm': value}
                        response = GLOBAL_SESSION.get(download_url, params=params, stream=True, timeout=30)
                        break

        # 4. Final Validation
        if response.status_code == 200:
            try:
                # Peek first chunk to ensure not HTML (error page)
                chunk = next(response.iter_content(chunk_size=100), b"")
                if chunk.strip().startswith(b"<!DOCTYPE"):
                    return None
                return BytesIO(chunk + response.raw.read())
            except StopIteration:
                return None
                
        return None

    except Exception as e:
        print(f"Download Exception: {e}")
        return None

def get_table_height(df, max_rows=30):
    """
    Calculates a dynamic height for Streamlit dataframes based on row count.
    Prevents massive empty whitespace for short tables.
    """
    row_count = len(df)
    if row_count == 0: return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5