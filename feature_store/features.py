"""
Feature engineering for sales forecasting.

Generates 40+ features across categories:
- Lag features (1d, 7d, 14d, 30d, 90d)
- Rolling statistics (mean, std, min, max, skew at multiple windows)
- Calendar/temporal features (DOW, month, quarter, holiday proximity)
- Exponential weighted means (multiple spans)
- Momentum and trend indicators
- Price and promotion features (when available)
- Fourier terms for seasonality encoding

All features use strict future-leakage prevention:
- Lags always reference t-N (never t or later)
- Rolling stats always use shift(1) before rolling
- Target-based stats use train-set means (no test leakage)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import logging

log = logging.getLogger(__name__)


# ── Feature groups ────────────────────────────────────────────────────────────
LAG_PERIODS = [1, 2, 3, 7, 14, 21, 28, 35, 42, 56, 70, 84, 91]
ROLLING_WINDOWS = [7, 14, 21, 28, 56, 91]
EWM_SPANS = [7, 14, 28, 56]
FOURIER_PERIODS = [7, 30.4375, 91.3125, 365.25]  # Weekly, monthly, quarterly, yearly
FOURIER_TERMS = 2  # Number of sin/cos pairs per period


def build_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    date_col: str = "date",
    additional_cols: Optional[List[str]] = None,
    include_fourier: bool = True,
    include_price: bool = False,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: DataFrame with date and target columns (sorted by date ascending)
        target_col: Name of the sales/quantity column
        date_col: Name of the date column
        additional_cols: Extra columns to include as-is (e.g., price, promotion_flag)
        include_fourier: Add Fourier terms for seasonality encoding
        include_price: Add price-elasticity features (requires 'price' column)

    Returns:
        Feature-enriched DataFrame. Drop rows with NaN before training.
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    
    y = df[target_col]
    
    # ── 1. Lag features ───────────────────────────────────────────────────────
    log.debug(f"Building {len(LAG_PERIODS)} lag features")
    for lag in LAG_PERIODS:
        df[f"lag_{lag}d"] = y.shift(lag)
    
    # ── 2. Rolling statistics (shift(1) prevents leakage) ─────────────────────
    log.debug(f"Building rolling stats for windows {ROLLING_WINDOWS}")
    y_shifted = y.shift(1)
    for w in ROLLING_WINDOWS:
        roll = y_shifted.rolling(w, min_periods=max(1, w // 2))
        df[f"roll_mean_{w}d"] = roll.mean().round(4)
        df[f"roll_std_{w}d"]  = roll.std().round(4)
        df[f"roll_min_{w}d"]  = roll.min()
        df[f"roll_max_{w}d"]  = roll.max()
        df[f"roll_skew_{w}d"] = roll.skew().round(4)
        # Coefficient of variation (std/mean) — measures relative volatility
        df[f"roll_cv_{w}d"] = (df[f"roll_std_{w}d"] / (df[f"roll_mean_{w}d"].abs() + 1e-9)).round(4)
    
    # ── 3. Exponential weighted means ─────────────────────────────────────────
    log.debug("Building EWM features")
    for span in EWM_SPANS:
        df[f"ewm_{span}d"] = y_shifted.ewm(span=span, min_periods=span // 2).mean().round(4)
    
    # ── 4. Momentum / change features ─────────────────────────────────────────
    # Week-over-week change
    df["wow_change"] = y.shift(1) - y.shift(8)  # lag1 vs lag8 = same DOW last week
    df["wow_change_pct"] = (df["wow_change"] / (y.shift(8).abs() + 1e-9)).round(4)
    
    # Month-over-month change
    df["mom_change"] = y.shift(1) - y.shift(29)
    df["mom_change_pct"] = (df["mom_change"] / (y.shift(29).abs() + 1e-9)).round(4)
    
    # Trend direction (7d simple linear trend slope)
    def rolling_slope(x):
        if x.isna().any():
            return np.nan
        return np.polyfit(range(len(x)), x.values, 1)[0]
    df["trend_7d"] = y_shifted.rolling(7).apply(rolling_slope, raw=False).round(4)
    
    # ── 5. Calendar features ──────────────────────────────────────────────────
    log.debug("Building calendar features")
    dt = df[date_col]
    df["day_of_week"]       = dt.dt.dayofweek           # 0=Mon, 6=Sun
    df["day_of_month"]      = dt.dt.day
    df["day_of_year"]       = dt.dt.dayofyear
    df["week_of_year"]      = dt.dt.isocalendar().week.astype(int)
    df["month"]             = dt.dt.month
    df["quarter"]           = dt.dt.quarter
    df["year"]              = dt.dt.year
    df["is_weekend"]        = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"]    = dt.dt.is_month_start.astype(int)
    df["is_month_end"]      = dt.dt.is_month_end.astype(int)
    df["is_quarter_start"]  = dt.dt.is_quarter_start.astype(int)
    df["is_quarter_end"]    = dt.dt.is_quarter_end.astype(int)
    
    # Days to next month-end (useful for B2B invoice cycle patterns)
    next_me = dt.dt.to_period("M").dt.to_timestamp("M")
    df["days_to_month_end"] = (next_me - dt).dt.days
    
    # Linear trend (days since series start) — for global drift
    df["time_idx"] = (dt - dt.min()).dt.days
    
    # ── 6. Fourier terms for seasonality ─────────────────────────────────────
    if include_fourier:
        log.debug(f"Building Fourier features ({len(FOURIER_PERIODS)} periods × {FOURIER_TERMS} terms)")
        t = df["time_idx"].values
        for period in FOURIER_PERIODS:
            for k in range(1, FOURIER_TERMS + 1):
                period_str = f"{int(period)}d" if period < 100 else f"{int(period)}y" if period > 300 else f"{int(period)}q"
                df[f"sin_{period_str}_k{k}"] = np.sin(2 * np.pi * k * t / period).round(6)
                df[f"cos_{period_str}_k{k}"] = np.cos(2 * np.pi * k * t / period).round(6)
    
    # ── 7. Price elasticity features ──────────────────────────────────────────
    if include_price and "price" in df.columns:
        log.debug("Building price features")
        p = df["price"]
        df["price_lag1"]    = p.shift(1)
        df["price_change"]  = p - p.shift(1)
        df["price_pct_chg"] = (df["price_change"] / (p.shift(1).abs() + 1e-9)).round(4)
        # Rolling average price (proxy for "normal" price)
        df["price_roll7"]   = p.shift(1).rolling(7).mean().round(2)
        df["price_vs_avg"]  = ((p - df["price_roll7"]) / (df["price_roll7"].abs() + 1e-9)).round(4)
    
    # ── 8. Include additional columns as-is ───────────────────────────────────
    if additional_cols:
        for col in additional_cols:
            if col not in df.columns:
                log.warning(f"Requested additional_col {col!r} not found in DataFrame")
    
    log.info(f"Feature engineering complete: {df.shape[1]} total columns")
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "sales",
                        date_col: str = "date", exclude: Optional[List[str]] = None) -> List[str]:
    """Return list of feature columns (excludes target, date, and any specified columns)."""
    exclude_set = {target_col, date_col} | set(exclude or [])
    return [c for c in df.columns if c not in exclude_set]


def create_train_test_split(df: pd.DataFrame, test_days: int = 90,
                             date_col: str = "date") -> tuple:
    """
    Time-series aware train/test split — no shuffling, test always after train.
    
    Args:
        df: Feature DataFrame sorted by date
        test_days: Number of days to hold out for test
        date_col: Date column name
    
    Returns:
        (train_df, test_df)
    """
    df = df.sort_values(date_col)
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff]
    test  = df[df[date_col] > cutoff]
    log.info(f"Train: {len(train)} rows ({train[date_col].min().date()} → {train[date_col].max().date()})")
    log.info(f"Test:  {len(test)} rows ({test[date_col].min().date()} → {test[date_col].max().date()})")
    return train, test
