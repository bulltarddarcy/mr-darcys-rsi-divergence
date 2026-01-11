"""
Sector rotation utilities - REFACTORED VERSION
Performance optimizations and multi-theme support added.
"""

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
from utils_shared import get_gdrive_binary_data

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
        
        Args:
            benchmark_ticker: Benchmark to include (SPY or QQQ)
            
        Returns:
            Tuple of (universe_df, all_tickers, theme_to_etf_map)
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
        
        Args:
            df: Raw price dataframe
            
        Returns:
            Processed dataframe with technicals, or None on error
        """
        if df is None or df.empty:
            return None

        # Ensure 'Close' exists
        if 'Close' not in df.columns and 'CLOSE' in df.columns:
            df['Close'] = df['CLOSE']
        
        if 'Close' not in df.columns:
            logger.error("No 'Close' column found in dataframe")
            return None
        
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
        """
        Calculate weighted linear regression slope.
        
        Args:
            series: Time series data
            window: Rolling window size
            
        Returns:
            Series of slopes
        """
        x = np.arange(window)
        weights = np.ones(window) if window < 10 else np.linspace(WEIGHT_MIN, WEIGHT_MAX, window)
        
        def slope_func(y):
            try:
                return np.polyfit(x, y, 1, w=weights)[0]
            except:
                return 0.0
                
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(
        self, 
        df: pd.DataFrame, 
        bench_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Calculate Relative Rotation Graph metrics vs benchmark.
        
        Args:
            df: Asset dataframe with DatetimeIndex and Close column
            bench_df: Benchmark dataframe with same structure
            
        Returns:
            DataFrame with RRG columns added, or None on error
        """
        if df is None or df.empty or bench_df is None or bench_df.empty:
            logger.warning("Empty dataframe passed to calculate_rrg_metrics")
            return None
        
        # Align dates
        common_index = df.index.intersection(bench_df.index)
        
        if common_index.empty:
            logger.error(
                f"No date overlap. Asset: {df.index.min()} to {df.index.max()}, "
                f"Bench: {bench_df.index.min()} to {bench_df.index.max()}"
            )
            return None
        
        if len(common_index) < 50:
            logger.warning(f"Only {len(common_index)} overlapping dates - results may be unreliable")
        
        df_aligned = df.loc[common_index].copy()
        bench_aligned = bench_df.loc[common_index].copy()
        
        # Determine Close columns
        asset_col = 'Adj Close' if 'Adj Close' in df_aligned.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        # Relative Ratio
        raw_ratio = df_aligned[asset_col] / bench_aligned[bench_col]
        
        # RRG Logic for each timeframe
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
        
        # Merge back to original df
        for col in df_aligned.columns:
            if col.startswith('RRG_'):
                df[col] = df_aligned[col]
                
        return df

    def calculate_stock_alpha_multi_theme(
        self, 
        df: pd.DataFrame, 
        parent_df: pd.DataFrame, 
        theme_suffix: str
    ) -> pd.DataFrame:
        """
        Calculate beta and alpha with theme-specific column names.
        
        This allows a stock to have multiple betas (one per theme).
        
        Args:
            df: Stock dataframe
            parent_df: Theme/sector ETF dataframe
            theme_suffix: Theme name for column naming (e.g., "Technology")
            
        Returns:
            DataFrame with added columns:
            - Beta_{theme}
            - Alpha_Short_{theme}
            - Alpha_Med_{theme}
            - Alpha_Long_{theme}
        """
        if df is None or df.empty or parent_df is None or parent_df.empty:
            logger.warning(f"Empty dataframe in alpha calculation for theme {theme_suffix}")
            return df
        
        common_index = df.index.intersection(parent_df.index)
        
        if common_index.empty:
            logger.error(f"No date overlap for theme {theme_suffix}")
            return df
        
        if len(common_index) < BETA_WINDOW:
            logger.warning(
                f"Only {len(common_index)} overlapping dates for {theme_suffix} "
                f"(need {BETA_WINDOW} for beta calculation)"
            )
        
        df_aligned = df.loc[common_index].copy()
        parent_aligned = parent_df.loc[common_index].copy()
        
        # Calculate returns if not already present
        if 'Pct_Change' not in df_aligned.columns:
            df_aligned['Pct_Change'] = df_aligned['Close'].pct_change()
        if 'Pct_Change' not in parent_aligned.columns:
            parent_aligned['Pct_Change'] = parent_aligned['Close'].pct_change()
        
        # Beta calculation with theme-specific name
        rolling_cov = df_aligned['Pct_Change'].rolling(window=BETA_WINDOW).cov(
            parent_aligned['Pct_Change']
        )
        rolling_var = parent_aligned['Pct_Change'].rolling(window=BETA_WINDOW).var()
        
        beta_col = f"Beta_{theme_suffix}"
        df_aligned[beta_col] = np.where(
            rolling_var > 1e-8,  # Avoid division by zero
            rolling_cov / rolling_var,
            1.0  # Default to market beta
        )
        
        # Expected return (CAPM)
        expected_return_col = f"Expected_Return_{theme_suffix}"
        df_aligned[expected_return_col] = parent_aligned['Pct_Change'] * df_aligned[beta_col]
        
        # Alpha (actual - expected)
        alpha_1d_col = f"Alpha_1D_{theme_suffix}"
        df_aligned[alpha_1d_col] = df_aligned['Pct_Change'] - df_aligned[expected_return_col]
        
        # Multi-timeframe cumulative alpha
        for label, time_window in TIMEFRAMES.items():
            alpha_col = f"Alpha_{label}_{theme_suffix}"
            df_aligned[alpha_col] = (
                df_aligned[alpha_1d_col]
                .fillna(0)
                .rolling(window=time_window)
                .sum() * 100
            )
        
        # Merge back to original df (preserving all dates)
        for col in df_aligned.columns:
            if col.endswith(f"_{theme_suffix}") or col == beta_col:
                df[col] = df_aligned[col]
        
        return df

# ==========================================
# 4. PATTERN DETECTION
# ==========================================

def detect_dip_buy_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    """
    Detect stocks that were outperforming but pulled back to average.
    
    Criteria:
    - 20-day alpha was significantly positive
    - 5-day alpha is near zero
    - Stock above 21-EMA (trend intact)
    
    Args:
        df: Stock dataframe with alpha columns
        theme_suffix: Theme name for column lookup
        
    Returns:
        True if stock matches dip buy pattern
    """
    if df is None or df.empty:
        return False
    
    try:
        last = df.iloc[-1]
        
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        
        # Historical outperformance
        was_strong = alpha_20d > ALPHA_DIP_BUY_THRESHOLD
        
        # Recent pullback to average
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_5d <= ALPHA_NEUTRAL_RANGE
        
        # Trend still intact
        price = last.get('Close', 0)
        ema21 = last.get('Ema21', 0)
        trend_intact = price > ema21 if ema21 > 0 else False
        
        return was_strong and now_neutral and trend_intact
    except Exception as e:
        logger.error(f"Error in dip_buy detection: {e}")
        return False

def detect_breakout_candidates(
    df: pd.DataFrame, 
    theme_suffix: str
) -> Optional[Dict]:
    """
    Detect stocks transitioning from underperformance to outperformance.
    
    Criteria:
    - 20-day alpha was negative
    - 10-day alpha neutral
    - 5-day alpha positive
    - Volume increasing
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        
    Returns:
        Dict with pattern details or None
    """
    if df is None or df.empty:
        return None
    
    try:
        last = df.iloc[-1]
        
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        rvol = last.get('RVOL_Short', 0)
        
        # Classic acceleration pattern
        was_lagging = alpha_20d < -ALPHA_BREAKOUT_THRESHOLD
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_10d <= ALPHA_NEUTRAL_RANGE
        now_leading = alpha_5d > ALPHA_BREAKOUT_THRESHOLD
        volume_confirms = rvol > RVOL_BREAKOUT_THRESHOLD
        
        if was_lagging and now_neutral and now_leading and volume_confirms:
            alpha_delta = alpha_5d - alpha_20d
            strength = min(100, (alpha_delta / 5.0) * 100)
            
            return {
                'pattern': 'breakout',
                'strength': strength,
                'alpha_change': alpha_delta
            }
    except Exception as e:
        logger.error(f"Error in breakout detection: {e}")
        
    return None

def detect_fading_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    """
    Detect stocks losing their alpha advantage.
    
    Criteria:
    - 20-day alpha very positive
    - 5-day alpha declining but still positive
    - Alpha slope negative
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        
    Returns:
        True if stock is fading
    """
    if df is None or df.empty or len(df) < 10:
        return False
    
    try:
        last = df.iloc[-1]
        prev_5 = df.iloc[-6]
        
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d_now = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_5d_prev = prev_5.get(f"Alpha_Short_{theme_suffix}", 0)
        
        price = last.get('Close', 0)
        sma50 = last.get('Sma50', 0)
        
        # Was very strong
        was_very_strong = alpha_20d > ALPHA_FADING_THRESHOLD
        
        # Still positive but declining
        still_positive = 0 < alpha_5d_now < ALPHA_DIP_BUY_THRESHOLD
        declining = alpha_5d_now < alpha_5d_prev
        
        # Hasn't broken down yet
        not_broken = price > sma50 if sma50 > 0 else True
        
        return was_very_strong and still_positive and declining and not_broken
    except Exception as e:
        logger.error(f"Error in fader detection: {e}")
        return False

def detect_relative_strength_divergence(
    df: pd.DataFrame, 
    theme_suffix: str, 
    lookback: int = 20
) -> Optional[str]:
    """
    Detect divergence between price and alpha.
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        lookback: Days to look back
        
    Returns:
        'bullish_divergence', 'bearish_divergence', or None
    """
    if df is None or df.empty or len(df) < lookback:
        return None
    
    try:
        recent = df.tail(lookback)
        alpha_col = f"Alpha_Short_{theme_suffix}"
        
        if alpha_col not in recent.columns:
            return None
        
        # Find extremes
        price_high = recent['Close'].max()
        price_low = recent['Close'].min()
        alpha_high = recent[alpha_col].max()
        alpha_low = recent[alpha_col].min()
        
        # Current values
        current_price = recent['Close'].iloc[-1]
        current_alpha = recent[alpha_col].iloc[-1]
        
        # Bearish Divergence: Price high, alpha declining
        if current_price >= price_high * 0.98:  # Near high
            if current_alpha < alpha_high * 0.7:  # Alpha well off highs
                return 'bearish_divergence'
        
        # Bullish Divergence: Price low, alpha improving
        if current_price <= price_low * 1.05:  # Near lows
            if current_alpha > alpha_low * 1.3:  # Alpha recovering
                return 'bullish_divergence'
    except Exception as e:
        logger.error(f"Error in divergence detection: {e}")
    
    return None

def calculate_comprehensive_stock_score(
def calculate_comprehensive_stock_score(
    df: pd.DataFrame, 
    theme_suffix: str, 
    theme_quadrant: str,
    sector_score: float = 70,
    sector_stage: str = "Established"
) -> Optional[Dict]:
    """
    EARLY ENTRY FOCUSED scoring system for stock selection.
    
    GOAL: Catch rotation INTO stocks, not highlight leaders already extended.
    
    Factors:
    1. Setup Quality (40%) - Is this at an inflection point?
    2. Entry Timing (30%) - Is NOW the right time to enter?
    3. Risk/Reward (20%) - How much upside potential?
    4. Sector Context (10%) - Is sector at right stage?
    
    PHILOSOPHY:
    - High scores = Early inflection points (0-3% alpha, just turning)
    - Low scores = Already extended (5%+ alpha, too late)
    - Base scoring + bonuses (not all-or-nothing)
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        theme_quadrant: Current quadrant of theme
        sector_score: Theme consensus score
        sector_stage: Lifecycle stage (Early/Established/Topping/Weak)
        
    Returns:
        Dict with score, breakdown, grade, and category label
    """
    if df is None or df.empty or len(df) < 20:
        return None
    
    try:
        last = df.iloc[-1]
        score_breakdown = {}
        
        # Get alpha values
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        
        # Get price and volume data
        price = last.get('Close', 0)
        ema8 = last.get('Ema8', 0)
        ema21 = last.get('Ema21', 0)
        sma50 = last.get('Sma50', 0)
        sma200 = last.get('Sma200', 0)
        rvol_5d = last.get('RVOL_Short', 0)
        rvol_10d = last.get('RVOL_Med', 0)
        rvol_20d = last.get('RVOL_Long', 0)
        
        # Calculate days positive (cumulative in last 20)
        days_positive = 0
        for i in range(min(20, len(df))):
            row_alpha = df.iloc[-(i+1)].get(f"Alpha_Short_{theme_suffix}", 0)
            if row_alpha > 0:
                days_positive += 1
        
        # --- FACTOR 1: Setup Quality (40 points) ---
        setup_score = 0
        setup_reason = ""
        
        # BASE SCORING: Give points for being positive in decent sector
        if alpha_5d > 0 and sector_score >= 60:
            setup_score += 12
            setup_reason = "Positive in decent sector"
        elif alpha_5d > 0:
            setup_score += 8
            setup_reason = "Positive alpha"
        
        # BONUS 1: Fresh Entry Range (0-3% alpha) = +15 points
        if 0 <= alpha_5d <= 3.0:
            setup_score += 15
            setup_reason = "Fresh entry range (0-3% alpha)"
            
            # Extra bonus if also fresh on days
            if days_positive <= 7:
                setup_score += 5
                setup_reason += ", very fresh (≤7 days)"
        
        # BONUS 2: Improving Trend (alpha accelerating) = +10 points
        if alpha_5d > alpha_10d and alpha_10d > alpha_20d:
            setup_score += 10
            setup_reason += ", accelerating"
        elif alpha_5d > alpha_20d:
            setup_score += 6
            setup_reason += ", improving"
        
        # PENALTY: Extended (>5% alpha OR high alpha + mature) = -15 points
        if alpha_5d > 5.0:
            setup_score -= 15
            setup_reason = "Extended (>5% alpha)"
        elif alpha_5d > 3.5 and days_positive >= 15:
            setup_score -= 10
            setup_reason = "Extended (mature + high alpha)"
        
        score_breakdown['setup_quality'] = max(0, min(40, setup_score))
        score_breakdown['setup_reason'] = setup_reason
        
        # --- FACTOR 2: Entry Timing (30 points) ---
        timing_score = 0
        timing_reason = ""
        
        # Part A: Price Position (15 points)
        if ema21 > 0:
            dist_from_ema21 = (price - ema21) / ema21
        else:
            dist_from_ema21 = 999
        
        if sma50 > 0:
            dist_from_sma50 = (price - sma50) / sma50
        else:
            dist_from_sma50 = 999
        
        # BASE: Give some points for reasonable position
        if price > sma200 and sma200 > 0:
            timing_score += 5
            timing_reason = "Above 200 MA"
        
        # BONUS: Near support = +10 points
        if abs(dist_from_ema21) <= 0.03 and price > sma200 and sma200 > 0:
            timing_score += 10
            timing_reason = "At support (21 EMA)"
        # BONUS: Fresh breakout = +10 points
        elif 0 < dist_from_sma50 <= 0.08 and rvol_5d >= 1.2:
            timing_score += 10
            timing_reason = "Fresh breakout"
        # OK: Neutral zone = +5 points
        elif -0.05 <= dist_from_ema21 <= 0.08:
            timing_score += 5
            timing_reason = "Neutral zone"
        # PENALTY: Extended from entry = -10 points
        elif dist_from_ema21 > 0.12 or dist_from_sma50 > 0.18:
            timing_score -= 10
            timing_reason = "Extended from entry"
        
        score_breakdown['timing_price'] = timing_score
        
        # Part B: Volume Pattern (15 points) - LOWERED THRESHOLDS
        volume_score = 0
        volume_reason = ""
        
        # BASE: Give points for any reasonable volume
        if rvol_5d >= 1.3:
            volume_score += 15
            volume_reason = "Strong volume (1.3x+)"
        elif rvol_5d >= 1.0:
            volume_score += 12
            volume_reason = "Steady volume (1.0-1.3x)"
        elif rvol_5d >= 0.85:
            volume_score += 8
            volume_reason = "Acceptable volume (0.85x+)"
        else:
            volume_score += 3
            volume_reason = "Weak volume (<0.85x)"
        
        # BONUS: Volume building (accelerating) = +3 points
        if rvol_5d > rvol_10d > rvol_20d:
            volume_score += 3
            volume_reason += ", building"
        
        score_breakdown['timing_volume'] = min(15, volume_score)
        score_breakdown['timing_reason'] = timing_reason + "; " + volume_reason
        
        # --- FACTOR 3: Risk/Reward (20 points) ---
        rr_score = 0
        rr_reason = ""
        
        # Part A: Upside Potential (15 points)
        sector_implied_alpha = 3.5 if sector_score >= 80 else (2.5 if sector_score >= 70 else 1.5)
        upside_gap = sector_implied_alpha - alpha_5d
        
        # BASE: Give points for being in strong sector
        if sector_score >= 70:
            rr_score += 5
        
        # BONUS: High catch-up potential
        if upside_gap >= 2.5 and sector_score >= 70:
            rr_score += 10
            rr_reason = f"High upside ({upside_gap:.1f}% room)"
        elif upside_gap >= 0.5:
            rr_score += 7
            rr_reason = "Moderate upside"
        elif upside_gap >= -1.0:
            rr_score += 4
            rr_reason = "Limited upside"
        else:
            rr_reason = "Already ahead of sector"
        
        # Part B: Downside Protection (5 points)
        if price > sma50 > sma200 and sma50 > 0 and sma200 > 0:
            rr_score += 5
        elif price > sma200 and sma200 > 0:
            rr_score += 3
        
        score_breakdown['risk_reward'] = min(20, rr_score)
        score_breakdown['rr_reason'] = rr_reason
        
        # --- FACTOR 4: Sector Context (10 points) ---
        context_score = 0
        context_reason = ""
        
        if sector_stage == "Early" and sector_score >= 70:
            context_score += 10
            context_reason = "Perfect: Early stage sector"
        elif sector_stage == "Established" and sector_score >= 65:
            context_score += 7
            context_reason = "Good: Established sector"
        elif sector_stage == "Established" and sector_score >= 55:
            context_score += 4
            context_reason = "OK: Mature sector"
        elif sector_stage == "Topping" or sector_score < 50:
            context_score -= 10
            context_reason = "Wrong sector: Topping/weak"
        
        score_breakdown['sector_context'] = context_score
        score_breakdown['context_reason'] = context_reason
        
        # --- CALCULATE BASE TOTAL ---
        total_score = (score_breakdown['setup_quality'] + 
                      score_breakdown['timing_price'] + 
                      score_breakdown['timing_volume'] +
                      score_breakdown['risk_reward'] + 
                      score_breakdown['sector_context'])
        
        # --- PATTERN BONUSES/PENALTIES (Partial Credit) ---
        pattern_label = ""
        pattern_score = 0
        
        # FRESH BREAKOUT PATTERN: Check 4 criteria, give partial credit
        fb_matches = 0
        if -2.5 <= alpha_20d <= 0.5: fb_matches += 1
        if 0 <= alpha_5d <= 3.0: fb_matches += 1
        if rvol_5d >= 1.2: fb_matches += 1
        if days_positive <= 8: fb_matches += 1
        
        if fb_matches >= 3:
            pattern_score = 10 if fb_matches == 4 else 7
            pattern_label = "🚀 Fresh Breakout"
            score_breakdown['pattern_bonus'] = pattern_score
        
        # DIP BUY PATTERN: Check 3 criteria, partial credit
        elif alpha_20d >= 2.5 and -0.5 <= alpha_5d <= 2.0:
            db_matches = 1  # Already know alpha criteria match
            if price > ema21 and ema21 > 0: db_matches += 1
            if abs(dist_from_ema21) <= 0.05: db_matches += 1
            
            if db_matches >= 2:
                pattern_score = 8 if db_matches == 3 else 5
                pattern_label = "💎 Dip Buy"
                score_breakdown['pattern_bonus'] = pattern_score
        
        # SECTOR ROTATION: Early sector + turning positive
        elif (sector_stage == "Early" and 
              -1.5 <= alpha_20d <= 2.0 and 
              0.5 <= alpha_5d <= 4.0):
            pattern_score = 12
            pattern_label = "🎯 Sector Rotation"
            score_breakdown['pattern_bonus'] = pattern_score
        
        # EXTENDED WARNING: -12 PENALTY (less harsh than -15)
        extended_count = 0
        if alpha_5d > 5.0: extended_count += 1
        if alpha_5d > 3.5 and days_positive >= 13: extended_count += 1
        if dist_from_ema21 > 0.12: extended_count += 1
        
        if extended_count >= 2:
            pattern_score = -12
            pattern_label = "⚡ Extended"
            score_breakdown['extended_penalty'] = pattern_score
        
        # BEARISH DIVERGENCE: -8 PENALTY
        divergence = detect_relative_strength_divergence(df, theme_suffix)
        if divergence == 'bearish_divergence':
            pattern_score -= 8
            pattern_label = pattern_label + " + 📉" if pattern_label else "📉 Bear Div"
            score_breakdown['divergence_penalty'] = -8
        
        total_score += pattern_score
        
        # Cap score at 0-100
        total_score = min(100, max(0, total_score))
        
        # Determine category label
        if total_score >= 75 and alpha_5d <= 3.5 and days_positive <= 12:
            category = "🎯 Optimal Entry"
        elif total_score >= 70 and alpha_20d >= 3.0 and alpha_5d <= 2.0:
            category = "💎 Pullback Buy"
        elif total_score >= 75 and alpha_5d > 4.0:
            category = "⚖️ Established Winner"
        elif total_score >= 60:
            category = "⚖️ Selective"
        else:
            category = "⚠️ Caution"
        
        return {
            'total_score': total_score,
            'breakdown': score_breakdown,
            'grade': _score_to_grade(total_score),
            'category': category,
            'pattern_label': pattern_label,
            'days_positive': days_positive
        }
    except Exception as e:
        logger.error(f"Error calculating stock score: {e}")
        return None

    if score >= 80: return 'A'
    if score >= 70: return 'B'
    if score >= 60: return 'C'
    if score >= 50: return 'D'
    return 'F'

# ==========================================
# 6. THEME SCORING & ANALYSIS
# ==========================================

def calculate_theme_score(df: pd.DataFrame, view_key: str = 'Short') -> Optional[Dict]:
    """
    Calculate comprehensive score for a theme/sector (0-100 with grade).
    
    Scoring Breakdown:
    - Current Position (40 pts): Quadrant strength
    - Momentum Quality (30 pts): Consistency + acceleration
    - Trajectory (20 pts): 3-day path analysis
    - Stability (10 pts): Smoothness of movement
    
    Args:
        df: Theme dataframe with RRG metrics
        view_key: Timeframe to analyze ('Short', 'Med', 'Long')
        
    Returns:
        Dict with score, grade, components, and actionability
    """
    if df is None or df.empty or len(df) < 4:
        return None
    
    try:
        col_ratio = f"RRG_Ratio_{view_key}"
        col_mom = f"RRG_Mom_{view_key}"
        
        if col_ratio not in df.columns or col_mom not in df.columns:
            return None
        
        # Get last 4 days (today + 3-day trail)
        recent = df.tail(4)
        if len(recent) < 4:
            return None
        
        last = recent.iloc[-1]
        ratio = last[col_ratio]
        mom = last[col_mom]
        
        score_breakdown = {}
        
        # --- FACTOR 1: Current Position (40 points) ---
        # Based on which quadrant and how deep into it
        
        # Quadrant base score
        if ratio >= 100 and mom >= 100:  # Leading
            quadrant_base = 30
            quadrant_name = "🟢 Leading"
        elif ratio < 100 and mom >= 100:  # Improving
            quadrant_base = 20
            quadrant_name = "🔵 Improving"
        elif ratio >= 100 and mom < 100:  # Weakening
            quadrant_base = 15
            quadrant_name = "🟡 Weakening"
        else:  # Lagging
            quadrant_base = 5
            quadrant_name = "🔴 Lagging"
        
        # Distance from center (magnitude bonus)
        distance = abs(ratio - 100) + abs(mom - 100)
        if distance > 10:
            magnitude_bonus = 10
        elif distance > 5:
            magnitude_bonus = 5
        else:
            magnitude_bonus = 0
        
        score_breakdown['position'] = min(40, quadrant_base + magnitude_bonus)
        
        # --- FACTOR 2: Momentum Quality (30 points) ---
        
        # Check all three timeframes
        mom_short = last.get('RRG_Mom_Short', 0)
        mom_med = last.get('RRG_Mom_Med', 0)
        mom_long = last.get('RRG_Mom_Long', 0)
        
        # Consistency (15 points)
        positive_count = sum([mom_short > 100, mom_med > 100, mom_long > 100])
        if positive_count == 3:
            consistency_score = 15
        elif positive_count == 2:
            consistency_score = 10
        elif positive_count == 1:
            consistency_score = 5
        else:
            consistency_score = 0
        
        # Acceleration (15 points)
        if mom_short > mom_med > mom_long:
            acceleration_score = 15
        elif mom_short > mom_med or mom_short > mom_long:
            acceleration_score = 8
        else:
            acceleration_score = 0
        
        score_breakdown['momentum_quality'] = consistency_score + acceleration_score
        
        # --- FACTOR 3: Trajectory (20 points) ---
        # Analyze the 3-day path
        
        trajectory_data = []
        for i in range(4):
            row = recent.iloc[i]
            r = row[col_ratio]
            m = row[col_mom]
            
            if r >= 100 and m >= 100:
                quad = "Leading"
            elif r < 100 and m >= 100:
                quad = "Improving"
            elif r >= 100 and m < 100:
                quad = "Weakening"
            else:
                quad = "Lagging"
            
            trajectory_data.append({
                'quadrant': quad,
                'ratio': r,
                'mom': m
            })
        
        # Improvement trajectory (10 points)
        quadrant_rank = {"Lagging": 0, "Weakening": 1, "Improving": 2, "Leading": 3}
        
        trajectory_path = [quadrant_rank[t['quadrant']] for t in trajectory_data]
        
        if trajectory_path[-1] > trajectory_path[0]:  # Improved over 3 days
            if all(trajectory_path[i] <= trajectory_path[i+1] for i in range(3)):
                trajectory_improvement = 10  # Consistent improvement each day
            else:
                trajectory_improvement = 5  # Improved but not smoothly
        elif trajectory_path[-1] == trajectory_path[0]:
            trajectory_improvement = 3  # Stayed same
        else:
            trajectory_improvement = 0  # Deteriorated
        
        # Momentum acceleration in trail (10 points)
        mom_values = [t['mom'] for t in trajectory_data]
        mom_change = mom_values[-1] - mom_values[0]
        
        if mom_change > 5:
            momentum_acceleration = 10
        elif mom_change > 2:
            momentum_acceleration = 5
        elif mom_change > 0:
            momentum_acceleration = 3
        else:
            momentum_acceleration = 0
        
        score_breakdown['trajectory'] = trajectory_improvement + momentum_acceleration
        
        # --- FACTOR 4: Stability (10 points) ---
        # Reward smooth movement, penalize whipsaws
        
        # Check if bouncing between quadrants
        unique_quadrants = len(set([t['quadrant'] for t in trajectory_data]))
        
        if unique_quadrants == 1:
            stability_score = 10  # Rock solid
        elif unique_quadrants == 2:
            # Check if it's a smooth progression vs whipsaw
            if trajectory_path == sorted(trajectory_path) or trajectory_path == sorted(trajectory_path, reverse=True):
                stability_score = 7  # Smooth transition
            else:
                stability_score = 3  # Some chop
        else:
            stability_score = 0  # Very choppy
        
        score_breakdown['stability'] = stability_score
        
        # --- CALCULATE TOTAL ---
        total_score = sum(score_breakdown.values())
        total_score = min(100, max(0, total_score))
        
        # --- FRESHNESS INDICATOR ---
        # Count how many days it's been in current quadrant
        current_quad = trajectory_data[-1]['quadrant']
        days_in_quadrant = 1
        for i in range(len(trajectory_data) - 2, -1, -1):
            if trajectory_data[i]['quadrant'] == current_quad:
                days_in_quadrant += 1
            else:
                break
        
        if days_in_quadrant <= 2:
            freshness = "🆕 Fresh"
            freshness_detail = f"Day {days_in_quadrant}"
        elif days_in_quadrant == 3:
            freshness = "⭐ Early"
            freshness_detail = f"Day {days_in_quadrant}"
        else:
            freshness = "🕐 Established"
            freshness_detail = f"Day {days_in_quadrant}+"
        
        # --- TRAJECTORY DESCRIPTION ---
        if trajectory_path[-1] > trajectory_path[0]:
            trajectory_label = "🚀 Rising"
        elif trajectory_path[-1] == trajectory_path[0]:
            if stability_score >= 7:
                trajectory_label = "➡️ Stable"
            else:
                trajectory_label = "⚡ Choppy"
        else:
            trajectory_label = "📉 Falling"
        
        return {
            'total_score': total_score,
            'grade': _score_to_grade(total_score),
            'quadrant': quadrant_name,
            'freshness': freshness,
            'freshness_detail': freshness_detail,
            'trajectory_label': trajectory_label,
            'breakdown': score_breakdown,
            'days_in_quadrant': days_in_quadrant,
            'stability': "Smooth" if stability_score >= 7 else "Choppy"
        }
        
    except Exception as e:
        logger.error(f"Error calculating theme score: {e}")
        return None

def calculate_consensus_theme_score(
    df: pd.DataFrame
) -> Optional[Dict]:
    """
    Calculate multi-timeframe consensus score for swing trading.
    
    Combines 5-day, 10-day, and 20-day scores with emphasis on longer timeframes.
    Only recommends themes where multiple timeframes align.
    
    Args:
        df: Theme dataframe with RRG metrics
        
    Returns:
        Dict with consensus score, conviction level, and all timeframe data
    """
    if df is None or df.empty:
        return None
    
    try:
        # Calculate scores for all timeframes
        score_5d = calculate_theme_score(df, 'Short')
        score_10d = calculate_theme_score(df, 'Med')
        score_20d = calculate_theme_score(df, 'Long')
        
        if not all([score_5d, score_10d, score_20d]):
            return None
        
        # Weighted consensus (favoring longer timeframes for swing trading)
        consensus_score = (
            score_5d['total_score'] * 0.20 +  # 20% weight
            score_10d['total_score'] * 0.30 +  # 30% weight  
            score_20d['total_score'] * 0.50    # 50% weight (most important)
        )
        
        # Check alignment across timeframes
        bullish_quadrants = ['🟢 Leading', '🔵 Improving']
        
        bullish_5d = score_5d['quadrant'] in bullish_quadrants
        bullish_10d = score_10d['quadrant'] in bullish_quadrants
        bullish_20d = score_20d['quadrant'] in bullish_quadrants
        
        bullish_count = sum([bullish_5d, bullish_10d, bullish_20d])
        
        # Conviction level
        if bullish_count == 3:
            conviction = "High"
            conviction_emoji = "✅"
            alignment_bonus = 10
        elif bullish_count == 2:
            conviction = "Medium"
            conviction_emoji = "🎯"
            alignment_bonus = 0
        elif bullish_count == 1:
            conviction = "Low"
            conviction_emoji = "⚠️"
            alignment_bonus = -10
        else:
            conviction = "None"
            conviction_emoji = "🚫"
            alignment_bonus = -15
        
        # Apply alignment bonus
        consensus_score = min(100, max(0, consensus_score + alignment_bonus))
        
        return {
            'consensus_score': round(consensus_score, 1),
            'grade': _score_to_grade(consensus_score),
            'conviction': conviction,
            'conviction_emoji': conviction_emoji,
            'timeframes': {
                '5d': {
                    'score': score_5d['total_score'],
                    'quadrant': score_5d['quadrant'],
                    'trajectory': score_5d['trajectory_label']
                },
                '10d': {
                    'score': score_10d['total_score'],
                    'quadrant': score_10d['quadrant'],
                    'trajectory': score_10d['trajectory_label']
                },
                '20d': {
                    'score': score_20d['total_score'],
                    'quadrant': score_20d['quadrant'],
                    'trajectory': score_20d['trajectory_label']
                }
            },
            'freshness': score_20d['freshness'],  # Use 20d for freshness
            'freshness_detail': score_20d['freshness_detail']
        }
        
    except Exception as e:
        logger.error(f"Error calculating consensus score: {e}")
        return None

def get_actionable_theme_summary(
    etf_data_cache: Dict,
    theme_map: Dict
) -> Dict[str, List[Dict]]:
    """
    Group themes by lifecycle stage using momentum trends and score quality.
    
    Categorization Logic (in plain English):
    
    EARLY STAGE (Best new entries):
    - Fresh position (Day 1-3 in current quadrant)
    - 2+ timeframes bullish (Leading or Improving)
    - Score 60+ (decent quality)
    - Momentum accelerating (5d > 10d > 20d) OR at least stable
    
    ESTABLISHED (Hold but don't chase):
    - NOT fresh (Day 4+)
    - High score (65+)
    - 2+ timeframes bullish
    - Even if momentum stable/declining - it's mature
    
    TOPPING (Take profits):
    - Momentum declining (5d < 10d or 5d < 20d) 
    - OR 5-day showing weakness (Weakening quadrant)
    - OR was bullish on 20d but losing on 5d
    - Score typically 40-65 range
    
    WEAK (Avoid):
    - Low score (<40) OR
    - Fewer than 2 timeframes bullish OR
    - All timeframes showing Lagging
    
    Args:
        etf_data_cache: Cache of theme dataframes
        theme_map: Dict mapping theme names to tickers
        
    Returns:
        Dict with themes grouped by lifecycle stage + explanation
    """
    categories = {
        'early_stage': [],        # Fresh bullish - BEST NEW ENTRIES
        'established': [],        # Mature leadership - HOLD
        'topping': [],           # Losing momentum - TAKE PROFITS
        'weak': []               # Lagging - AVOID
    }
    
    for theme, ticker in theme_map.items():
        df = etf_data_cache.get(ticker)
        if df is None or df.empty:
            continue
        
        consensus = calculate_consensus_theme_score(df)
        if not consensus:
            continue
        
        # Extract data
        score = consensus['consensus_score']
        
        # Get scores for each timeframe (not just quadrant!)
        score_5d = consensus['timeframes']['5d']['score']
        score_10d = consensus['timeframes']['10d']['score']
        score_20d = consensus['timeframes']['20d']['score']
        
        # Get quadrants
        quad_5d = consensus['timeframes']['5d']['quadrant']
        quad_10d = consensus['timeframes']['10d']['quadrant']
        quad_20d = consensus['timeframes']['20d']['quadrant']
        
        # Check freshness
        freshness = consensus['freshness']
        is_fresh = freshness in ["🆕 Fresh", "⭐ Early"]
        
        # Check if bullish (Leading or Improving)
        bullish_quads = ['🟢 Leading', '🔵 Improving']
        is_5d_bullish = quad_5d in bullish_quads
        is_10d_bullish = quad_10d in bullish_quads
        is_20d_bullish = quad_20d in bullish_quads
        bullish_count = sum([is_5d_bullish, is_10d_bullish, is_20d_bullish])
        
        # CRITICAL: Check momentum trend (is it accelerating or declining?)
        momentum_accelerating = score_5d > score_10d > score_20d
        momentum_stable = score_5d >= score_10d >= score_20d
        momentum_declining = score_5d < score_10d or score_5d < score_20d
        
        # Determine reason for categorization
        reason = ""
        
        theme_info = {
            'theme': theme,
            'consensus_score': score,
            'grade': consensus['grade'],
            'freshness': consensus['freshness'],
            'freshness_detail': consensus['freshness_detail'],
            'tf_5d': quad_5d,
            'tf_10d': quad_10d,
            'tf_20d': quad_20d,
            'score_5d': score_5d,
            'score_10d': score_10d,
            'score_20d': score_20d
        }
        
        # ═══════════════════════════════════════════════════════
        # CATEGORIZATION LOGIC
        # ═══════════════════════════════════════════════════════
        
        # --- EARLY STAGE: Fresh + Building + Good Score ---
        if (
            is_fresh and 
            bullish_count >= 2 and 
            score >= 60 and
            (momentum_accelerating or momentum_stable)
        ):
            if momentum_accelerating:
                reason = f"Fresh position (Day {consensus['freshness_detail'].split()[-1]}), 2+ timeframes bullish, momentum accelerating ({score_5d:.0f} > {score_10d:.0f} > {score_20d:.0f})"
            else:
                reason = f"Fresh position (Day {consensus['freshness_detail'].split()[-1]}), 2+ timeframes bullish, momentum stable ({score_5d:.0f} ≈ {score_10d:.0f})"
            
            theme_info['reason'] = reason
            categories['early_stage'].append(theme_info)
        
        # --- ESTABLISHED: High score + Not fresh (check this BEFORE topping) ---
        elif (
            bullish_count >= 2 and
            score >= 65 and
            not is_fresh
        ):
            days = consensus['freshness_detail']
            if momentum_declining:
                reason = f"Strong score ({score:.0f}) but momentum declining ({score_5d:.0f} < {score_10d:.0f}), been leading for {days}"
            else:
                reason = f"Strong score ({score:.0f}), established position ({days}), mature trend"
            
            theme_info['reason'] = reason
            categories['established'].append(theme_info)
        
        # --- TOPPING: Momentum declining + Lower score ---
        elif (
            momentum_declining and
            score >= 40 and
            score < 65 and
            (is_20d_bullish or is_10d_bullish)
        ):
            reason = f"Momentum declining ({score_5d:.0f} < {score_10d:.0f} or {score_20d:.0f})"
            if quad_5d == '🟡 Weakening':
                reason += f", 5-day already weakening"
            
            theme_info['reason'] = reason
            categories['topping'].append(theme_info)
        
        # --- WEAK: Everything else ---
        else:
            if score < 40:
                reason = f"Low score ({score:.0f}), weak positioning"
            elif bullish_count < 2:
                reason = f"Only {bullish_count} of 3 timeframes bullish, insufficient confirmation"
            else:
                reason = f"Mixed signals, score {score:.0f} with {bullish_count} bullish timeframes"
            
            theme_info['reason'] = reason
            categories['weak'].append(theme_info)
    
    # Sort each category by score
    for category in categories:
        categories[category].sort(key=lambda x: x['consensus_score'], reverse=True)
    
    return categories
    """
    Calculate comprehensive score for a theme/sector (0-100 with grade).
    
    Scoring Breakdown:
    - Current Position (40 pts): Quadrant strength
    - Momentum Quality (30 pts): Consistency + acceleration
    - Trajectory (20 pts): 3-day path analysis
    - Stability (10 pts): Smoothness of movement
    
    Args:
        df: Theme dataframe with RRG metrics
        view_key: Timeframe to analyze ('Short', 'Med', 'Long')
        
    Returns:
        Dict with score, grade, components, and actionability
    """
    if df is None or df.empty or len(df) < 4:
        return None
    
    try:
        col_ratio = f"RRG_Ratio_{view_key}"
        col_mom = f"RRG_Mom_{view_key}"
        
        if col_ratio not in df.columns or col_mom not in df.columns:
            return None
        
        # Get last 4 days (today + 3-day trail)
        recent = df.tail(4)
        if len(recent) < 4:
            return None
        
        last = recent.iloc[-1]
        ratio = last[col_ratio]
        mom = last[col_mom]
        
        score_breakdown = {}
        
        # --- FACTOR 1: Current Position (40 points) ---
        # Based on which quadrant and how deep into it
        
        # Quadrant base score
        if ratio >= 100 and mom >= 100:  # Leading
            quadrant_base = 30
            quadrant_name = "🟢 Leading"
        elif ratio < 100 and mom >= 100:  # Improving
            quadrant_base = 20
            quadrant_name = "🔵 Improving"
        elif ratio >= 100 and mom < 100:  # Weakening
            quadrant_base = 15
            quadrant_name = "🟡 Weakening"
        else:  # Lagging
            quadrant_base = 5
            quadrant_name = "🔴 Lagging"
        
        # Distance from center (magnitude bonus)
        distance = abs(ratio - 100) + abs(mom - 100)
        if distance > 10:
            magnitude_bonus = 10
        elif distance > 5:
            magnitude_bonus = 5
        else:
            magnitude_bonus = 0
        
        score_breakdown['position'] = min(40, quadrant_base + magnitude_bonus)
        
        # --- FACTOR 2: Momentum Quality (30 points) ---
        
        # Check all three timeframes
        mom_short = last.get('RRG_Mom_Short', 0)
        mom_med = last.get('RRG_Mom_Med', 0)
        mom_long = last.get('RRG_Mom_Long', 0)
        
        # Consistency (15 points)
        positive_count = sum([mom_short > 100, mom_med > 100, mom_long > 100])
        if positive_count == 3:
            consistency_score = 15
        elif positive_count == 2:
            consistency_score = 10
        elif positive_count == 1:
            consistency_score = 5
        else:
            consistency_score = 0
        
        # Acceleration (15 points)
        if mom_short > mom_med > mom_long:
            acceleration_score = 15
        elif mom_short > mom_med or mom_short > mom_long:
            acceleration_score = 8
        else:
            acceleration_score = 0
        
        score_breakdown['momentum_quality'] = consistency_score + acceleration_score
        
        # --- FACTOR 3: Trajectory (20 points) ---
        # Analyze the 3-day path
        
        trajectory_data = []
        for i in range(4):
            row = recent.iloc[i]
            r = row[col_ratio]
            m = row[col_mom]
            
            if r >= 100 and m >= 100:
                quad = "Leading"
            elif r < 100 and m >= 100:
                quad = "Improving"
            elif r >= 100 and m < 100:
                quad = "Weakening"
            else:
                quad = "Lagging"
            
            trajectory_data.append({
                'quadrant': quad,
                'ratio': r,
                'mom': m
            })
        
        # Improvement trajectory (10 points)
        quadrant_rank = {"Lagging": 0, "Weakening": 1, "Improving": 2, "Leading": 3}
        
        trajectory_path = [quadrant_rank[t['quadrant']] for t in trajectory_data]
        
        if trajectory_path[-1] > trajectory_path[0]:  # Improved over 3 days
            if all(trajectory_path[i] <= trajectory_path[i+1] for i in range(3)):
                trajectory_improvement = 10  # Consistent improvement each day
            else:
                trajectory_improvement = 5  # Improved but not smoothly
        elif trajectory_path[-1] == trajectory_path[0]:
            trajectory_improvement = 3  # Stayed same
        else:
            trajectory_improvement = 0  # Deteriorated
        
        # Momentum acceleration in trail (10 points)
        mom_values = [t['mom'] for t in trajectory_data]
        mom_change = mom_values[-1] - mom_values[0]
        
        if mom_change > 5:
            momentum_acceleration = 10
        elif mom_change > 2:
            momentum_acceleration = 5
        elif mom_change > 0:
            momentum_acceleration = 3
        else:
            momentum_acceleration = 0
        
        score_breakdown['trajectory'] = trajectory_improvement + momentum_acceleration
        
        # --- FACTOR 4: Stability (10 points) ---
        # Reward smooth movement, penalize whipsaws
        
        # Check if bouncing between quadrants
        unique_quadrants = len(set([t['quadrant'] for t in trajectory_data]))
        
        if unique_quadrants == 1:
            stability_score = 10  # Rock solid
        elif unique_quadrants == 2:
            # Check if it's a smooth progression vs whipsaw
            if trajectory_path == sorted(trajectory_path) or trajectory_path == sorted(trajectory_path, reverse=True):
                stability_score = 7  # Smooth transition
            else:
                stability_score = 3  # Some chop
        else:
            stability_score = 0  # Very choppy
        
        score_breakdown['stability'] = stability_score
        
        # --- CALCULATE TOTAL ---
        total_score = sum(score_breakdown.values())
        total_score = min(100, max(0, total_score))
        
        # --- FRESHNESS INDICATOR ---
        # Count how many days it's been in current quadrant
        current_quad = trajectory_data[-1]['quadrant']
        days_in_quadrant = 1
        for i in range(len(trajectory_data) - 2, -1, -1):
            if trajectory_data[i]['quadrant'] == current_quad:
                days_in_quadrant += 1
            else:
                break
        
        if days_in_quadrant <= 2:
            freshness = "🆕 Fresh"
            freshness_detail = f"Just entered (Day {days_in_quadrant})"
        elif days_in_quadrant == 3:
            freshness = "⭐ Early"
            freshness_detail = f"Early stage (Day {days_in_quadrant})"
        else:
            freshness = "🕐 Established"
            freshness_detail = f"Established (Day {days_in_quadrant}+)"
        
        # --- ACTIONABILITY ---
        if total_score >= 70 and quadrant_name in ["🟢 Leading", "🔵 Improving"]:
            action = "✅ BUY"
            action_detail = "Strong momentum - actively allocate"
        elif total_score >= 50 and quadrant_name == "🔵 Improving":
            action = "🎯 WATCH"
            action_detail = "Building momentum - prepare to enter"
        elif total_score >= 40 and quadrant_name == "🟢 Leading":
            action = "⚠️ HOLD"
            action_detail = "Still positive but momentum fading"
        elif quadrant_name == "🟡 Weakening":
            action = "⚠️ REDUCE"
            action_detail = "Losing momentum - take profits"
        else:
            action = "🚫 AVOID"
            action_detail = "Weak positioning - stay away"
        
        # --- TRAJECTORY DESCRIPTION ---
        if trajectory_path[-1] > trajectory_path[0]:
            trajectory_label = "🚀 Accelerating"
        elif trajectory_path[-1] == trajectory_path[0]:
            if stability_score >= 7:
                trajectory_label = "➡️ Stable"
            else:
                trajectory_label = "⚡ Choppy"
        else:
            trajectory_label = "📉 Declining"
        
        return {
            'total_score': total_score,
            'grade': _score_to_grade(total_score),
            'quadrant': quadrant_name,
            'freshness': freshness,
            'freshness_detail': freshness_detail,
            'trajectory_label': trajectory_label,
            'action': action,
            'action_detail': action_detail,
            'breakdown': score_breakdown,
            'days_in_quadrant': days_in_quadrant,
            'stability': "Smooth" if stability_score >= 7 else "Choppy"
        }
        
    except Exception as e:
        logger.error(f"Error calculating theme score: {e}")
        return None

# ==========================================
# 5. ORCHESTRATOR (OPTIMIZED)
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_process_universe(
    benchmark_ticker: str = "SPY"
) -> Tuple[Dict, List[str], Dict[str, str], pd.DataFrame, Dict[str, List[str]]]:
    """
    OPTIMIZED data pipeline with multi-theme support.
    
    Key optimizations:
    1. Single parquet download
    2. Vectorized filtering with groupby
    3. Multi-theme beta calculations
    4. Efficient caching
    
    Args:
        benchmark_ticker: SPY or QQQ
        
    Returns:
        Tuple of (data_cache, missing_tickers, theme_map, universe_df, stock_themes_map)
    """
    dm = SectorDataManager()
    uni_df, tickers, theme_map = dm.load_universe(benchmark_ticker)
    
    if uni_df.empty:
        return {}, ["SECTOR_UNIVERSE is empty"], theme_map, uni_df, {}

    # --- 1. DOWNLOAD MASTER DB ---
    db_url = st.secrets.get("PARQUET_SECTOR_ROTATION")
    if not db_url:
        st.error("⛔ Secret 'PARQUET_SECTOR_ROTATION' is missing.")
        return {}, ["PARQUET_SECTOR_ROTATION secret missing"], theme_map, uni_df, {}

    try:
        buffer = get_gdrive_binary_data(db_url)
        if not buffer:
            logger.error("Failed to download sector rotation parquet")
            return {}, ["Failed to download Master DB"], theme_map, uni_df, {}
        
        master_df = pd.read_parquet(buffer)
        logger.info(f"Loaded master parquet with {len(master_df)} rows")
        
    except Exception as e:
        logger.exception(f"Error reading parquet: {e}")
        return {}, [f"Error reading Parquet file: {e}"], theme_map, uni_df, {}

    # --- 2. STANDARDIZE MASTER DB ---
    master_df.columns = [c.strip().title() for c in master_df.columns]
    
    if 'Symbol' in master_df.columns and 'Ticker' not in master_df.columns:
        master_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        
    if 'Date' in master_df.columns:
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        master_df = master_df.set_index('Date').sort_index()
    elif isinstance(master_df.index, pd.DatetimeIndex):
        master_df = master_df.sort_index()
    
    if 'Ticker' in master_df.columns:
        master_df['Ticker'] = master_df['Ticker'].astype(str).str.upper().str.strip()
    else:
        return {}, ["Critical: 'Ticker' column missing in Master DB"], theme_map, uni_df, {}

    calc = SectorAlphaCalculator()
    data_cache = {}
    missing_tickers = []

    # --- 3. PROCESS BENCHMARK ---
    bench_df = master_df[master_df['Ticker'] == benchmark_ticker].copy()
    
    if bench_df.empty:
        logger.error(f"Benchmark {benchmark_ticker} not found in parquet")
        return {}, [f"Benchmark '{benchmark_ticker}' not found"], theme_map, uni_df, {}

    bench_df = calc.process_dataframe(bench_df)
    data_cache[benchmark_ticker] = bench_df
    logger.info(f"Processed benchmark {benchmark_ticker}")

    # --- 4. PROCESS ETFS (OPTIMIZED WITH GROUPBY) ---
    etf_tickers = list(theme_map.values())
    
    if etf_tickers:
        etf_mask = master_df['Ticker'].isin(etf_tickers)
        etf_groups = master_df[etf_mask].groupby('Ticker')
        
        for etf in etf_tickers:
            try:
                df = etf_groups.get_group(etf).copy()
            except KeyError:
                missing_tickers.append(etf)
                logger.warning(f"ETF {etf} not found in parquet")
                continue
                
            df = calc.process_dataframe(df)
            df = calc.calculate_rrg_metrics(df, bench_df)
            data_cache[etf] = df
        
        logger.info(f"Processed {len(etf_tickers)} ETFs")

    # --- 5. PROCESS STOCKS (MULTI-THEME AWARE) ---
    stocks = uni_df[uni_df['Role'] == 'Stock']
    
    # Build stock-to-themes mapping
    stock_themes_map = stocks.groupby('Ticker')['Theme'].apply(list).to_dict()
    logger.info(f"Found {len(stock_themes_map)} unique stocks across themes")
    
    # Get all unique stock tickers
    stock_tickers = list(stock_themes_map.keys())
    
    if stock_tickers:
        # OPTIMIZATION: Filter and group once
        stock_mask = master_df['Ticker'].isin(stock_tickers)
        stock_groups = master_df[stock_mask].groupby('Ticker')
        
        for ticker, themes_list in stock_themes_map.items():
            try:
                df = stock_groups.get_group(ticker).copy()
            except KeyError:
                missing_tickers.append(ticker)
                logger.warning(f"Stock {ticker} not found in parquet")
                continue
            
            # Process base dataframe once
            df = calc.process_dataframe(df)
            
            # Calculate beta/alpha vs EACH theme
            for theme in themes_list:
                parent_etf = theme_map.get(theme)
                parent_df = data_cache.get(parent_etf, bench_df)
                
                if parent_df is not None:
                    df = calc.calculate_stock_alpha_multi_theme(df, parent_df, theme)
            
            data_cache[ticker] = df
        
        logger.info(f"Processed {len(stock_tickers)} stocks with multi-theme support")

    return data_cache, missing_tickers, theme_map, uni_df, stock_themes_map


# ==========================================
# 6. VISUALIZATION & CLASSIFICATION
# ==========================================

def classify_setup(df: pd.DataFrame) -> Optional[str]:
    """
    Classify sector/theme setup pattern.
    
    Patterns:
    - J-Hook: Turnaround from weak to strong
    - Bull Flag: Strong trend with breakout
    - Rocket: Parabolic acceleration
    - Breakdown: Accelerating downward
    - Dead Cat: Weak bounce in downtrend
    - Power Trend: Sustained strength
    - Deteriorating: Losing momentum
    
    Args:
        df: Theme dataframe with RRG metrics
        
    Returns:
        Pattern string or None
    """
    if df is None or df.empty:
        return None
        
    try:
        last = df.iloc[-1]
        
        if "RRG_Mom_Short" not in last or "RRG_Mom_Long" not in last:
            return None

        m5 = last["RRG_Mom_Short"]
        m10 = last.get("RRG_Mom_Med", 0)
        m20 = last["RRG_Mom_Long"]
        ratio_20 = last.get("RRG_Ratio_Long", 100)

        # J-Hook: Turnaround play
        if m20 < 100 and m5 > 100 and m5 > (m20 + JHOOK_MIN_SHIFT):
            return "🪝 J-Hook"
        
        # Bull Flag: Continuation
        if ratio_20 > 100 and m5 > 100 and m5 > m10:
            return "🚩 Bull Flag"
        
        # Rocket: Parabolic
        if m5 > m10 > m20 and m20 > 100:
            return "🚀 Rocket"
        
        # Breakdown: Accelerating down
        if m5 < m10 < m20 and m20 < 100:
            return "💥 Breakdown"
        
        # Dead Cat Bounce: Weak bounce
        if m20 < 100 and m5 > 100 and m5 < (m20 + JHOOK_MIN_SHIFT):
            return "🐱 Dead Cat"
        
        # Power Trend: Sustained strength
        if ratio_20 > 100 and m20 > 100 and m10 > 100 and m5 > 100:
            return "⚡ Power"
        
        # Deteriorating: Losing steam
        if ratio_20 > 100 and m5 < 100 and m5 < m10:
            return "📉 Fading"
            
    except Exception as e:
        logger.error(f"Error in classify_setup: {e}")
    
    return None

def get_quadrant_status(df: pd.DataFrame, timeframe_key: str) -> str:
    """
    Get quadrant status for a theme.
    
    Args:
        df: Theme dataframe with RRG metrics
        timeframe_key: 'Short', 'Med', or 'Long'
        
    Returns:
        Quadrant status string
    """
    if df is None or df.empty:
        return "N/A"
    
    try:
        col_r = f"RRG_Ratio_{timeframe_key}"
        col_m = f"RRG_Mom_{timeframe_key}"
        
        if col_r not in df.columns or col_m not in df.columns:
            return "N/A"
        
        r = df[col_r].iloc[-1]
        m = df[col_m].iloc[-1]
        
        if r >= 100 and m >= 100:
            return "🟢 Leading"
        elif r < 100 and m >= 100:
            return "🔵 Improving"
        elif r < 100 and m < 100:
            return "🔴 Lagging"
        else:
            return "🟡 Weakening"
    except Exception as e:
        logger.error(f"Error getting quadrant status: {e}")
        return "N/A"

def plot_simple_rrg(
    data_cache: Dict, 
    target_map: Dict, 
    view_key: str, 
    show_trails: bool
) -> go.Figure:
    """
    Create Relative Rotation Graph scatter plot.
    
    Args:
        data_cache: Dict of {ticker: DataFrame}
        target_map: Dict of {theme: ticker}
        view_key: 'Short', 'Med', or 'Long'
        show_trails: Whether to show 3-day trails
        
    Returns:
        Plotly Figure object
    """
    # Validate view_key
    valid_keys = ['Short', 'Med', 'Long']
    if view_key not in valid_keys:
        logger.error(f"Invalid view_key: {view_key}")
        raise ValueError(f"view_key must be one of {valid_keys}, got: {view_key}")
    
    if not target_map:
        logger.warning("Empty target_map in plot_simple_rrg")
        fig = go.Figure()
        fig.add_annotation(
            text="No sectors selected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = go.Figure()
    all_x, all_y = [], []
    
    for theme, ticker in target_map.items():
        df = data_cache.get(ticker)
        if df is None or df.empty:
            continue
        
        col_x = f"RRG_Ratio_{view_key}"
        col_y = f"RRG_Mom_{view_key}"
        
        if col_x not in df.columns or col_y not in df.columns:
            continue
        
        data_slice = df.tail(3) if show_trails else df.tail(1)
        if data_slice.empty:
            continue

        x_vals = data_slice[col_x].tolist()
        y_vals = data_slice[col_y].tolist()
        all_x.extend(x_vals)
        all_y.extend(y_vals)
        
        # Determine color based on quadrant
        last_x, last_y = x_vals[-1], y_vals[-1]
        if last_x > 100 and last_y > 100:
            color = '#00CC96'  # Green
        elif last_x < 100 and last_y > 100:
            color = '#636EFA'  # Blue
        elif last_x > 100 and last_y < 100:
            color = '#FFA15A'  # Orange
        else:
            color = '#EF553B'  # Red
        
        # Marker sizes and opacities
        n = len(x_vals)
        sizes = [MARKER_SIZE_TRAIL] * (n - 1) + [MARKER_SIZE_CURRENT]
        opacities = [TRAIL_OPACITY] * (n - 1) + [CURRENT_OPACITY]
        texts = [""] * (n - 1) + [theme]
        custom_data = [theme] * n

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers+text',
            name=theme,
            text=texts,
            customdata=custom_data,
            textposition="top center",
            marker=dict(
                size=sizes,
                color=color,
                opacity=opacities,
                line=dict(width=1, color='white')
            ),
            line=dict(
                color=color,
                width=1 if show_trails else 0,
                shape='spline',
                smoothing=1.3
            ),
            hoverinfo='text+name',
            hovertext=[
                f"{theme}<br>Trend: {x:.1f}<br>Mom: {y:.1f}"
                for x, y in zip(x_vals, y_vals)
            ]
        ))

    # Dynamic scaling
    if all_x and all_y:
        limit_x = max(max([abs(x - 100) for x in all_x]) * 1.1, 2.0)
        limit_y = max(max([abs(y - 100) for y in all_y]) * 1.1, 2.0)
        x_range = [100 - limit_x, 100 + limit_x]
        y_range = [100 - limit_y, 100 + limit_y]
    else:
        x_range, y_range = [98, 102], [98, 102]
        limit_x, limit_y = 2, 2

    # Add crosshairs
    fig.add_hline(y=100, line_width=1, line_color="gray", line_dash="dash")
    fig.add_vline(x=100, line_width=1, line_color="gray", line_dash="dash")
    
    # Add quadrant labels
    lbl_x, lbl_y = limit_x * 0.5, limit_y * 0.5
    
    fig.add_annotation(
        x=100+lbl_x, y=100+lbl_y,
        text="<b>LEADING</b>",
        showarrow=False,
        font=dict(color="rgba(0, 255, 0, 0.7)", size=20)
    )
    fig.add_annotation(
        x=100-lbl_x, y=100+lbl_y,
        text="<b>IMPROVING</b>",
        showarrow=False,
        font=dict(color="rgba(0, 100, 255, 0.7)", size=20)
    )
    fig.add_annotation(
        x=100+lbl_x, y=100-lbl_y,
        text="<b>WEAKENING</b>",
        showarrow=False,
        font=dict(color="rgba(255, 165, 0, 0.7)", size=20)
    )
    fig.add_annotation(
        x=100-lbl_x, y=100-lbl_y,
        text="<b>LAGGING</b>",
        showarrow=False,
        font=dict(color="rgba(255, 0, 0, 0.7)", size=20)
    )

    fig.update_layout(
        xaxis=dict(
            title="Relative Trend",
            showgrid=False,
            range=x_range,
            constrain='domain'
        ),
        yaxis=dict(
            title="Relative Momentum",
            showgrid=False,
            range=y_range
        ),
        height=750,
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig
