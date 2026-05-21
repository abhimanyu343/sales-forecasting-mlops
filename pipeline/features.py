"""
Feature engineering for sales forecasting.
Generates lag features, rolling statistics, and calendar features.
"""
import pandas as pd
import numpy as np
from typing import List


def engineer_features(df: pd.DataFrame, target_col: str = "sales", lags: List[int] = None) -> pd.DataFrame:
    """
    Generate ML-ready features from raw time-series sales data.
    
    Args:
        df: DataFrame with 'date' and sales column
        target_col: Name of the target column
        lags: List of lag periods (default: [1, 7, 14, 30])
    
    Returns:
        Feature-enriched DataFrame
    """
    if lags is None:
        lags = [1, 7, 14, 30]
    
    df = df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    
    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    
    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df[target_col].shift(1).rolling(window).std()
        df[f"rolling_max_{window}"] = df[target_col].shift(1).rolling(window).max()
        df[f"rolling_min_{window}"] = df[target_col].shift(1).rolling(window).min()
    
    # Exponential weighted mean
    df["ewm_7"] = df[target_col].shift(1).ewm(span=7).mean()
    df["ewm_30"] = df[target_col].shift(1).ewm(span=30).mean()
    
    # Trend feature (days since start)
    df["trend"] = (df["date"] - df["date"].min()).dt.days
    
    return df.dropna()


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, 
                      col: str, threshold: float = 0.1) -> dict:
    """
    Detect data drift using population stability index (PSI).
    Triggers retraining if PSI > threshold.
    """
    ref_mean, cur_mean = reference_df[col].mean(), current_df[col].mean()
    psi = abs(cur_mean - ref_mean) / (ref_mean + 1e-9)
    return {
        "col": col,
        "psi": round(psi, 4),
        "drift_detected": psi > threshold,
        "reference_mean": round(ref_mean, 2),
        "current_mean": round(cur_mean, 2)
    }
