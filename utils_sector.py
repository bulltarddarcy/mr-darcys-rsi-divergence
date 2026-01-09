# UTILS **ONLY** FOR SECTOR TRENDS

import streamlit as st
import pandas as pd

# Import Sector specific logic managers
from data_manager import DataManager 
from update_data import update_market_data
from calculator import AlphaCalculator

@st.cache_data
def load_sector_data():
    dm = DataManager()
    try:
        uni_df, tickers, theme_map = dm.load_universe()
        return dm, uni_df, theme_map
    except Exception:
        return None, pd.DataFrame(), {}

def classify_setup(df):
    if df is None or df.empty: return None
    last = df.iloc[-1]
    if "RRG_Mom_Short" not in last or "RRG_Mom_Long" not in last: return None

    m5, m10, m20 = last["RRG_Mom_Short"], last["RRG_Mom_Med"], last["RRG_Mom_Long"]
    ratio_20 = last.get("RRG_Ratio_Long", 100)

    if m20 < 100 and m5 > 100 and m5 > (m20 + 2): return "🪝 J-Hook"
    if ratio_20 > 100 and m5 > 100 and m5 > m10: return "🚩 Bull Flag"
    if m5 > m10 and m10 > m20 and m20 > 100: return "🚀 Rocket"
    return None 

def get_momentum_trends(dm, theme_map):
    inc_mom, neut_mom, dec_mom = [], [], []
    for theme, ticker in theme_map.items():
        df = dm.load_ticker_data(ticker)
        if df is None or df.empty or "RRG_Mom_Short" not in df.columns: continue
        last = df.iloc[-1]
        m5, m10, m20 = last["RRG_Mom_Short"], last["RRG_Mom_Med"], last["RRG_Mom_Long"]
        item = {"theme": theme, "shift": m5 - m20, "m5": m5, "icon": (classify_setup(df) or " ").split()[0]}
        
        if m5 > m10 > m20: inc_mom.append(item)
        elif m5 < m10 < m20: dec_mom.append(item)
        else: neut_mom.append(item)
    
    return sorted(inc_mom, key=lambda x: x['shift'], reverse=True), \
           sorted(neut_mom, key=lambda x: x['m5'], reverse=True), \
           sorted(dec_mom, key=lambda x: x['shift'])
