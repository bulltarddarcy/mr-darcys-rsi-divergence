# utils_sector.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from io import StringIO
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

# Configure logging
logger = logging.getLogger(__name__)

# --- IMPORT SHARED UTILS ---
from utils_shared import (
    get_gdrive_binary_data, 
    add_technicals, 
    prepare_data, 
    find_divergences
)

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
HISTORY_YEARS = 1

# Volatility & Normalization
ATR_WINDOW = 20
AVG_VOLUME_WINDOW = 20
BETA_WINDOW = 60

# Timeframes (Trading Days)
TIMEFRAMES = {
    'Short': 5,
    'Med': 10,
    'Long': 20
}

# Filters
MIN_DOLLAR_VOLUME = 2_000_000

# Regression Weights
WEIGHT_MIN = 1.0
WEIGHT_MAX = 3.0

# Visualization
MARKER_SIZE_TRAIL = 8
MARKER_SIZE_CURRENT = 15
TRAIL_OPACITY = 0.4
CURRENT_OPACITY = 1.0

# Pattern Detection Thresholds
JHOOK_MIN_SHIFT = 2.0
ALPHA_DIP_BUY_THRESHOLD = 2.0
ALPHA_NEUTRAL_RANGE = 0.5
ALPHA_BREAKOUT_THRESHOLD = 1.0
ALPHA_FADING_THRESHOLD = 3.0
RVOL_HIGH_THRESHOLD = 1.3
RVOL_BREAKOUT_THRESHOLD = 1.3

# ==========================================
# 2. DATA MANAGER
# ==========================================
class SectorDataManager:
    """Manages sector universe configuration and ticker mapping."""
    
    def __init__(self):
        self.universe = pd.DataFrame()
        self.ticker_map = {}

    def load_universe(self, benchmark_ticker: str = "SPY") -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
        """
        Load sector universe from secrets.
        """
        secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
        if not secret_val:
            st.error("❌ Secret 'SECTOR_UNIVERSE' is missing or empty.")
            return pd.DataFrame(), [], {}
            
        try:
            # Handle Google Sheet Links or Raw CSV
            if secret_val.strip().startswith("http"):
                if "docs.google.com/spreadsheets" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                elif "drive.google.com" in secret_val and "/d/" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://drive.google.com/uc?id={file_id}&export=download"
                else:
                    csv_source = secret_val
                df = pd.read_csv(csv_source)
            else:
                df = pd.read_csv(StringIO(secret_val))
            
            required_cols = ['Ticker', 'Theme']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Universe CSV missing required columns: {required_cols}")
                return pd.DataFrame(), [], {}

            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df['Theme'] = df['Theme'].astype(str).str.strip()
            df['Role'] = df['Role'].astype(str).str.strip().str.title() if 'Role' in df.columns else 'Stock'

            tickers = df['Ticker'].unique().tolist()
            if benchmark_ticker not in tickers:
                tickers.append(benchmark_ticker)
                
            etf_rows = df[df['Role'] == 'Etf']
            theme_map = dict(zip(etf_rows['Theme'], etf_rows['Ticker'])) if not etf_rows.empty else {}
            
            self.universe = df
            logger.info(f"Loaded universe with {len(tickers)} tickers, {len(theme_map)} themes")
            return df, tickers, theme_map
            
        except Exception as e:
            logger.exception(f"Error loading SECTOR_UNIVERSE: {e}")
            st.error(f"Error loading SECTOR_UNIVERSE: {e}")
            return pd.DataFrame(), [], {}

# ==========================================
# 3. CALCULATOR & PIPELINE
# ==========================================
class SectorAlphaCalculator:
    """Calculates relative performance metrics for sectors and stocks."""
    
    def process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare raw dataframe with technical indicators.
        Uses shared add_technicals to ensure RSI/EMA existence (optimised).
        """
        if df is None or df.empty:
            return None

        # Normalize cols for add_technicals 
        if 'Close' not in df.columns and 'CLOSE' in df.columns:
            df['Close'] = df['CLOSE']
        if 'Volume' not in df.columns and 'VOLUME' in df.columns:
            df['Volume'] = df['VOLUME']
            
        if 'Close' not in df.columns:
            logger.error("No 'Close' column found in dataframe")
            return None

        # --- SHARED TECHNICALS ---
        # This adds RSI, EMA8, EMA21, SMA200 if missing.
        df = add_technicals(df)
        
        # --- SECTOR SPECIFIC ---
        # Calculate RVOL
        if 'Volume' in df.columns:
            avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
            df["RVOL"] = df["Volume"] / avg_vol

            for label, time_window in TIMEFRAMES.items():
                df[f"RVOL_{label}"] = df["RVOL"].rolling(window=time_window).mean()
        
        # Daily Range % (ADR)
        if 'High' in df.columns and 'Low' in df.columns:
            daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
            df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
        
        return df

    def _calc_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate weighted linear regression slope."""
        x = np.arange(window)
        weights = np.ones(window) if window < 10 else np.linspace(WEIGHT_MIN, WEIGHT_MAX, window)
        
        def slope_func(y):
            try:
                return np.polyfit(x, y, 1, w=weights)[0]
            except:
                return 0.0
                
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(self, df: pd.DataFrame, bench_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate Relative Rotation Graph metrics vs benchmark."""
        if df is None or df.empty or bench_df is None or bench_df.empty:
            return None
        
        common_index = df.index.intersection(bench_df.index)
        if common_index.empty: return None
        
        df_aligned = df.loc[common_index].copy()
        bench_aligned = bench_df.loc[common_index].copy()
        
        asset_col = 'Adj Close' if 'Adj Close' in df_aligned.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        raw_ratio = df_aligned[asset_col] / bench_aligned[bench_col]
        
        for label, time_window in TIMEFRAMES.items():
            ratio_mean = raw_ratio.rolling(window=time_window).mean()
            
            # Trend (Ratio) - Normalized around 100
            col_ratio = f"RRG_Ratio_{label}"
            df_aligned[col_ratio] = ((raw_ratio - ratio_mean) / ratio_mean) * 100 + 100
            
            # Momentum (Velocity of the Ratio)
            raw_slope = self._calc_slope(raw_ratio, time_window)
            velocity = (raw_slope / ratio_mean) * time_window * 100
            
            col_mom = f"RRG_Mom_{label}"
            df_aligned[col_mom] = 100 + velocity
        
        # Merge back
        for col in df_aligned.columns:
            if col.startswith('RRG_'):
                df[col] = df_aligned[col]
                
        return df

    def calculate_stock_alpha_multi_theme(self, df: pd.DataFrame, parent_df: pd.DataFrame, theme_suffix: str) -> pd.DataFrame:
        """Calculate beta and alpha with theme-specific column names."""
        if df is None or df.empty or parent_df is None or parent_df.empty:
            return df
        
        common_index = df.index.intersection(parent_df.index)
        if common_index.empty: return df
        
        df_aligned = df.loc[common_index].copy()
        parent_aligned = parent_df.loc[common_index].copy()
        
        if 'Pct_Change' not in df_aligned.columns:
            df_aligned['Pct_Change'] = df_aligned['Close'].pct_change()
        if 'Pct_Change' not in parent_aligned.columns:
            parent_aligned['Pct_Change'] = parent_aligned['Close'].pct_change()
        
        rolling_cov = df_aligned['Pct_Change'].rolling(window=BETA_WINDOW).cov(parent_aligned['Pct_Change'])
        rolling_var = parent_aligned['Pct_Change'].rolling(window=BETA_WINDOW).var()
        
        beta_col = f"Beta_{theme_suffix}"
        df_aligned[beta_col] = np.where(rolling_var > 1e-8, rolling_cov / rolling_var, 1.0)
        
        expected_return_col = f"Expected_Return_{theme_suffix}"
        df_aligned[expected_return_col] = parent_aligned['Pct_Change'] * df_aligned[beta_col]
        
        alpha_1d_col = f"Alpha_1D_{theme_suffix}"
        df_aligned[alpha_1d_col] = df_aligned['Pct_Change'] - df_aligned[expected_return_col]
        
        for label, time_window in TIMEFRAMES.items():
            alpha_col = f"Alpha_{label}_{theme_suffix}"
            df_aligned[alpha_col] = (df_aligned[alpha_1d_col].fillna(0).rolling(window=time_window).sum() * 100)
        
        for col in df_aligned.columns:
            if col.endswith(f"_{theme_suffix}") or col == beta_col:
                df[col] = df_aligned[col]
        
        return df

# ==========================================
# 4. PATTERN DETECTION & SCORING
# ==========================================

def detect_dip_buy_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    if df is None or df.empty: return False
    try:
        last = df.iloc[-1]
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        
        was_strong = alpha_20d > ALPHA_DIP_BUY_THRESHOLD
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_5d <= ALPHA_NEUTRAL_RANGE
        
        price = last.get('Close', 0)
        ema21 = last.get('Ema21', 0)
        trend_intact = price > ema21 if ema21 > 0 else False
        
        return was_strong and now_neutral and trend_intact
    except: return False

def detect_breakout_candidates(df: pd.DataFrame, theme_suffix: str) -> Optional[Dict]:
    if df is None or df.empty: return None
    try:
        last = df.iloc[-1]
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        rvol = last.get('RVOL_Short', 0)
        
        was_lagging = alpha_20d < -ALPHA_BREAKOUT_THRESHOLD
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_10d <= ALPHA_NEUTRAL_RANGE
        now_leading = alpha_5d > ALPHA_BREAKOUT_THRESHOLD
        volume_confirms = rvol > RVOL_BREAKOUT_THRESHOLD
        
        if was_lagging and now_neutral and now_leading and volume_confirms:
            alpha_delta = alpha_5d - alpha_20d
            return {'pattern': 'breakout', 'strength': min(100, (alpha_delta / 5.0) * 100)}
    except: pass
    return None

def detect_fading_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    if df is None or df.empty or len(df) < 10: return False
    try:
        last = df.iloc[-1]
        prev_5 = df.iloc[-6]
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d_now = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_5d_prev = prev_5.get(f"Alpha_Short_{theme_suffix}", 0)
        
        was_very_strong = alpha_20d > ALPHA_FADING_THRESHOLD
        still_positive = 0 < alpha_5d_now < ALPHA_DIP_BUY_THRESHOLD
        declining = alpha_5d_now < alpha_5d_prev
        
        return was_very_strong and still_positive and declining
    except: return False

def detect_relative_strength_divergence(df: pd.DataFrame, ticker: str, lookback: int = 90) -> Optional[str]:
    """
    Detect divergence using the SHARED RSI logic.
    """
    if df is None or df.empty: return None
    try:
        # Prepare data using shared utility
        d_d, _ = prepare_data(df)
        if d_d is None: return None
        
        # Look back 20 days for a 'recent' signal to display
        divs = find_divergences(
            d_d, ticker, 'Daily', 
            min_n=0, 
            periods_input=[5], 
            lookback_period=90, 
            price_source='High/Low', 
            strict_validation=True, 
            recent_days_filter=20, 
            rsi_diff_threshold=2.0
        )
        
        if not divs: return None
        
        last_sig = divs[-1]
        if last_sig['Type'] == 'Bullish': return 'bullish_divergence'
        if last_sig['Type'] == 'Bearish': return 'bearish_divergence'
    except Exception as e:
        logger.error(f"Error in shared divergence detection: {e}")
    
    return None

def calculate_comprehensive_stock_score(df: pd.DataFrame, theme_suffix: str, theme_quadrant: str) -> Optional[Dict]:
    if df is None or df.empty: return None
    try:
        last = df.iloc[-1]
        score_breakdown = {}
        
        # --- FACTOR 1: Alpha Trajectory (40 points) ---
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        
        # Consistency
        if alpha_5d > 0 and alpha_10d > 0 and alpha_20d > 0: score_breakdown['alpha_consistency'] = 15
        elif alpha_5d > 0 and alpha_10d > 0: score_breakdown['alpha_consistency'] = 10
        elif alpha_5d > 0: score_breakdown['alpha_consistency'] = 5
        else: score_breakdown['alpha_consistency'] = 0
        
        # Acceleration
        if alpha_5d > alpha_10d > alpha_20d: score_breakdown['alpha_acceleration'] = 15
        elif alpha_5d > alpha_10d: score_breakdown['alpha_acceleration'] = 10
        elif alpha_5d > alpha_20d: score_breakdown['alpha_acceleration'] = 5
        else: score_breakdown['alpha_acceleration'] = 0
        
        # Magnitude
        max_alpha = max(alpha_5d, alpha_10d, alpha_20d)
        if max_alpha > 5.0: score_breakdown['alpha_magnitude'] = 10
        elif max_alpha > 2.0: score_breakdown['alpha_magnitude'] = 7
        elif max_alpha > 1.0: score_breakdown['alpha_magnitude'] = 4
        else: score_breakdown['alpha_magnitude'] = 0
        
        # --- FACTOR 2: Volume (20 points) ---
        rvol_5d = last.get('RVOL_Short', 0)
        if rvol_5d > 1.5: score_breakdown['volume'] = 20
        elif rvol_5d > 1.2: score_breakdown['volume'] = 15
        else: score_breakdown['volume'] = 0
        
        # --- FACTOR 3: Technicals (20 points) ---
        price = last.get('Close', 0)
        ema8 = last.get('Ema8', 0)
        ema21 = last.get('Ema21', 0)
        sma50 = last.get('Sma50', 0)
        sma200 = last.get('Sma200', 0)
        
        ma_score = 0
        if price > ema8 and ema8 > 0: ma_score += 5
        if price > ema21 and ema21 > 0: ma_score += 5
        if price > sma50 and sma50 > 0: ma_score += 5
        if price > sma200 and sma200 > 0: ma_score += 5
        score_breakdown['technical'] = ma_score
        
        # --- FACTOR 4: Theme (20 points) ---
        if "Leading" in theme_quadrant:
            score_breakdown['theme_sync'] = 20 if alpha_5d > 1.0 else (10 if alpha_5d > 0 else 0)
        elif "Improving" in theme_quadrant:
            score_breakdown['theme_sync'] = 20 if alpha_5d > alpha_20d else 5
        elif "Weakening" in theme_quadrant:
            score_breakdown['theme_sync'] = 0 if alpha_5d < 0 else 10
        else:
            score_breakdown['theme_sync'] = 0
        
        total_score = sum(score_breakdown.values())
        
        # Bonuses
        if detect_breakout_candidates(df, theme_suffix): total_score += 10
        if detect_dip_buy_candidates(df, theme_suffix): total_score += 5
        
        total_score = min(100, max(0, total_score))
        
        return {
            'total_score': total_score,
            'breakdown': score_breakdown,
            'grade': _score_to_grade(total_score)
        }
    except: return None

def _score_to_grade(score: float) -> str:
    if score >= 80: return 'A'
    if score >= 70: return 'B'
    if score >= 60: return 'C'
    if score >= 50: return 'D'
    return 'F'

# ==========================================
# 5. DATA PIPELINE FUNCTIONS
# ==========================================

@st.cache_data(ttl=600)
def fetch_and_process_universe(benchmark_ticker: str = "SPY"):
    """Main data fetching pipeline."""
    mgr = SectorDataManager()
    uni_df, tickers, theme_map = mgr.load_universe(benchmark_ticker)
    
    if uni_df.empty:
        return {}, [], {}, uni_df, {}

    # Get parquet config
    try:
        raw_config = st.secrets.get("PARQUET_CONFIG", "")
        config = {}
        if raw_config:
             lines = [line.strip() for line in raw_config.strip().split('\n') if line.strip()]
             for line in lines:
                 parts = [p.strip() for p in line.split(',')]
                 if len(parts) >= 2 and parts[1] in st.secrets:
                     config[parts[0]] = parts[1]
    except: config = {}
    
    # Load all tickers
    data_cache = {}
    missing_tickers = []
    
    # Pre-fetch benchmark
    bench_df = None
    if benchmark_ticker in config:
        url = st.secrets[config[benchmark_ticker]] if config[benchmark_ticker] in st.secrets else ""
        if url:
            buffer = get_gdrive_binary_data(url)
            if buffer:
                try:
                    df = pd.read_parquet(buffer, engine='pyarrow')
                    if isinstance(df.index, pd.DatetimeIndex): df.reset_index(inplace=True)
                    elif 'DATE' in str(df.index.name).upper(): df.reset_index(inplace=True)
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    # Date fix
                    cols_upper = [c.upper() for c in df.columns]
                    date_col = next((df.columns[i] for i, c in enumerate(cols_upper) if 'DATE' in c), None)
                    if not date_col and 'index' in df.columns: date_col = 'index'
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.rename(columns={date_col: 'ChartDate'}).sort_values('ChartDate').set_index('ChartDate')
                        bench_df = df
                        data_cache[benchmark_ticker] = bench_df
                except: pass

    calc = SectorAlphaCalculator()

    # Load and process components
    for ticker in tickers:
        if ticker == benchmark_ticker: continue
        
        # Determine Parquet Key
        key = f"{ticker}_PARQUET"
        if key not in config:
             missing_tickers.append(ticker)
             continue
             
        url = st.secrets[config[key]]
        buffer = get_gdrive_binary_data(url)
        
        if not buffer:
            missing_tickers.append(ticker)
            continue
            
        try:
            df = pd.read_parquet(buffer, engine='pyarrow')
            if isinstance(df.index, pd.DatetimeIndex): df.reset_index(inplace=True)
            elif 'DATE' in str(df.index.name).upper(): df.reset_index(inplace=True)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Date fix
            cols_upper = [c.upper() for c in df.columns]
            date_col = next((df.columns[i] for i, c in enumerate(cols_upper) if 'DATE' in c), None)
            if not date_col and 'index' in df.columns: date_col = 'index'
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.rename(columns={date_col: 'ChartDate'}).sort_values('ChartDate').set_index('ChartDate')
                
                # Process basic techs
                df = calc.process_dataframe(df)
                
                # Process RRG if benchmark exists
                if bench_df is not None:
                    df = calc.calculate_rrg_metrics(df, bench_df)
                
                # Process Theme Alpha
                # Identify themes for this ticker
                themes = uni_df[uni_df['Ticker'] == ticker]['Theme'].unique()
                for theme in themes:
                    if theme in theme_map:
                        etf_ticker = theme_map[theme]
                        # Only calc alpha if we have the ETF data loaded
                        # (Need to ensure ETFs are loaded first or in second pass)
                        # Simplified: assuming we load everything then do alpha? 
                        # For efficiency in this script we often load ETFs first.
                        pass # Alpha calc needs parent ETF data
                
                data_cache[ticker] = df
            else:
                missing_tickers.append(ticker)
        except:
            missing_tickers.append(ticker)

    # Second pass for Alpha Calculation (Stocks against their Sector ETFs)
    # Ensure ETFs are in cache
    for theme, etf_ticker in theme_map.items():
        if etf_ticker not in data_cache: continue
        etf_df = data_cache[etf_ticker]
        
        # Calculate ETF Alpha vs SPY
        if bench_df is not None:
             etf_df = calc.calculate_stock_alpha_multi_theme(etf_df, bench_df, benchmark_ticker)
             data_cache[etf_ticker] = etf_df

        # Find stocks in this theme
        theme_stocks = uni_df[(uni_df['Theme'] == theme) & (uni_df['Role'] == 'Stock')]['Ticker'].unique()
        
        for stock in theme_stocks:
            if stock in data_cache:
                # Calc alpha against the Sector ETF
                data_cache[stock] = calc.calculate_stock_alpha_multi_theme(data_cache[stock], etf_df, theme)
    
    return data_cache, missing_tickers, theme_map, uni_df, {}

def get_quadrant_status(df: pd.DataFrame, timeframe_label: str) -> str:
    """Determine RRG quadrant text."""
    if df is None or df.empty: return "N/A"
    try:
        last = df.iloc[-1]
        ratio = last.get(f"RRG_Ratio_{timeframe_label}", 100)
        mom = last.get(f"RRG_Mom_{timeframe_label}", 100)
        
        if ratio > 100 and mom > 100: return "Leading 🟢"
        if ratio > 100 and mom < 100: return "Weakening 🟡"
        if ratio < 100 and mom < 100: return "Lagging 🔴"
        if ratio < 100 and mom > 100: return "Improving 🔵"
    except: pass
    return "N/A"

def classify_setup(df):
    """Simple momentum text classification."""
    try:
        status = get_quadrant_status(df, 'Short')
        return status
    except: return ""

def get_actionable_theme_summary(etf_cache: Dict, theme_map: Dict):
    """Classify themes into lifecycle buckets."""
    categories = {
        'early_stage': [], 'established': [], 'topping': [], 'weak': []
    }
    
    for theme, ticker in theme_map.items():
        df = etf_cache.get(ticker)
        if df is None or df.empty: continue
        
        try:
            last = df.iloc[-1]
            
            # Get Scores (normalized momentum)
            s5 = last.get("RRG_Mom_Short", 100)
            s10 = last.get("RRG_Mom_Med", 100)
            s20 = last.get("RRG_Mom_Long", 100)
            
            # Quadrant Check
            q5 = get_quadrant_status(df, 'Short')
            q10 = get_quadrant_status(df, 'Med')
            
            # Logic
            info = {
                'theme': theme,
                'consensus_score': (s5+s10+s20)/3,
                'grade': _score_to_grade((s5+s10+s20)/3 - 40), # Rough scaling
                'tf_5d': q5, 'tf_10d': q10, 'tf_20d': get_quadrant_status(df, 'Long'),
                'score_5d': s5, 'score_10d': s10, 'score_20d': s20
            }
            
            if "Leading" in q5 and "Improving" in q10:
                info['freshness_detail'] = "Breaking Out"
                info['reason'] = "Just moved to Leading from Improving"
                categories['early_stage'].append(info)
            elif "Leading" in q5 and "Leading" in q10:
                info['freshness_detail'] = "Mature"
                info['reason'] = "Strong trend across timeframes"
                categories['established'].append(info)
            elif "Weakening" in q5:
                info['freshness_detail'] = "Topping"
                info['reason'] = "Short term momentum lost"
                categories['topping'].append(info)
            else:
                info['freshness_detail'] = "Weak"
                info['reason'] = "Lagging or Improving only"
                categories['weak'].append(info)
                
        except: continue
        
    return categories

def plot_simple_rrg(data_cache, theme_map, view_key='Short', show_trails=False):
    """Plot Plotly RRG Chart."""
    fig = go.Figure()
    
    ratio_col = f"RRG_Ratio_{view_key}"
    mom_col = f"RRG_Mom_{view_key}"
    
    for theme, ticker in theme_map.items():
        df = data_cache.get(ticker)
        if df is None or df.empty: continue
        
        last = df.iloc[-1]
        
        # Trail
        if show_trails and len(df) > 3:
            trail = df.iloc[-4:-1]
            fig.add_trace(go.Scatter(
                x=trail[ratio_col], y=trail[mom_col],
                mode='lines+markers',
                marker=dict(size=MARKER_SIZE_TRAIL, opacity=TRAIL_OPACITY),
                line=dict(color='gray', width=1),
                name=f"{theme} Trail",
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # Point
        fig.add_trace(go.Scatter(
            x=[last[ratio_col]], y=[last[mom_col]],
            mode='markers+text',
            text=[theme],
            textposition="top center",
            marker=dict(size=MARKER_SIZE_CURRENT, opacity=CURRENT_OPACITY),
            customdata=[theme],
            name=theme
        ))
        
    # Formatting
    fig.add_shape(type="line", x0=96, y0=100, x1=104, y1=100, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=100, y0=96, x1=100, y1=104, line=dict(color="gray", width=1, dash="dash"))
    
    fig.update_layout(
        xaxis=dict(title="Relative Trend (Ratio)", range=[96, 104]),
        yaxis=dict(title="Relative Momentum", range=[96, 104]),
        width=800, height=600,
        showlegend=False,
        title=f"Sector Rotation - {view_key} Term"
    )
    
    # Quadrant Labels
    fig.add_annotation(x=102, y=102, text="LEADING", showarrow=False, font=dict(color="green", size=14))
    fig.add_annotation(x=98, y=102, text="IMPROVING", showarrow=False, font=dict(color="blue", size=14))
    fig.add_annotation(x=102, y=98, text="WEAKENING", showarrow=False, font=dict(color="orange", size=14))
    fig.add_annotation(x=98, y=98, text="LAGGING", showarrow=False, font=dict(color="red", size=14))
    
    return fig