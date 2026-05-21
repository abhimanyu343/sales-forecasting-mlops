"""
Model evaluation utilities — walk-forward cross-validation, metrics, comparison.

Walk-forward CV is the correct evaluation methodology for time series:
- Never shuffle data
- Each fold trains on all data up to cutoff, tests on the next window
- Simulates real deployment where you always predict the future
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

log = logging.getLogger(__name__)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — treats over/under-forecast equally."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean forecast bias — positive means over-forecasting."""
    return float(np.mean(y_pred - y_true))

def coverage_80(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of 80% prediction intervals."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          lower: np.ndarray = None, upper: np.ndarray = None) -> Dict[str, float]:
    """Compute the full suite of forecast evaluation metrics."""
    metrics = {
        "mape":  round(mape(y_true, y_pred), 3),
        "smape": round(smape(y_true, y_pred), 3),
        "rmse":  round(rmse(y_true, y_pred), 2),
        "mae":   round(mae(y_true, y_pred), 2),
        "bias":  round(bias(y_true, y_pred), 2),
        "n":     len(y_true),
    }
    if lower is not None and upper is not None:
        metrics["coverage_80"] = round(coverage_80(y_true, lower, upper), 3)
    return metrics


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "sales",
    date_col: str = "date",
    n_folds: int = 3,
    forecast_horizon: int = 30,
    min_train_days: int = 180
) -> Dict[str, Any]:
    """
    Walk-forward cross-validation for time-series models.

    Splits time series into n_folds evaluation windows. For each fold:
    - Train on all data before the fold start
    - Forecast the next forecast_horizon days
    - Compute MAPE on actuals

    Args:
        df: Feature-engineered DataFrame sorted by date
        feature_cols: Columns to use as features for GB models
        target_col: Target column
        date_col: Date column
        n_folds: Number of CV folds
        forecast_horizon: Days to forecast per fold
        min_train_days: Minimum training days required

    Returns:
        Dict with per-fold metrics and aggregate stats
    """
    from models.ensemble import SalesEnsemble

    df = df.sort_values(date_col).reset_index(drop=True)
    total_days = (df[date_col].max() - df[date_col].min()).days
    holdout_total = n_folds * forecast_horizon

    if total_days - holdout_total < min_train_days:
        log.warning(f"Not enough data for {n_folds}-fold CV. Reducing to 2 folds.")
        n_folds = 2

    fold_results = []
    max_date = df[date_col].max()

    for fold in range(n_folds - 1, -1, -1):
        # Each fold's test window ends at max_date - fold * horizon
        test_end   = max_date - pd.Timedelta(days=fold * forecast_horizon)
        test_start = test_end - pd.Timedelta(days=forecast_horizon)
        val_start  = test_start - pd.Timedelta(days=forecast_horizon)

        df_train_cv = df[df[date_col] <= val_start]
        df_val_cv   = df[(df[date_col] > val_start) & (df[date_col] <= test_start)]
        df_test_cv  = df[(df[date_col] > test_start) & (df[date_col] <= test_end)]

        if len(df_train_cv) < min_train_days or len(df_test_cv) < 5:
            log.warning(f"Fold {fold}: skipping (insufficient data)")
            continue

        log.info(f"Fold {n_folds - fold}/{n_folds}: train={len(df_train_cv)} val={len(df_val_cv)} test={len(df_test_cv)}")

        try:
            model = SalesEnsemble(run_name=f"cv_fold_{fold}")
            # Suppress MLflow logging for CV folds
            import mlflow
            with mlflow.start_run(run_name=f"cv_fold_{fold}", nested=True):
                model.fit(df_train_cv, df_val_cv, feature_cols, target_col, date_col)

            y_test = df_test_cv[target_col].values
            forecast = model.predict(
                df[df[date_col] <= test_start],
                horizon=len(df_test_cv)
            )
            y_pred = forecast["forecast"].values[:len(y_test)]
            lower  = forecast["lower_80"].values[:len(y_test)]
            upper  = forecast["upper_80"].values[:len(y_test)]

            fold_m = compute_full_metrics(y_test, y_pred, lower, upper)
            fold_m["fold"] = n_folds - fold
            fold_results.append(fold_m)
            log.info(f"  Fold {n_folds - fold} | MAPE={fold_m['mape']:.2f}% RMSE={fold_m['rmse']:.0f}")

        except Exception as e:
            log.error(f"Fold {fold} failed: {e}", exc_info=True)

    if not fold_results:
        return {"mean_mape": 999.0, "std_mape": 0.0, "folds": []}

    mapes = [r["mape"] for r in fold_results]
    return {
        "mean_mape":   round(np.mean(mapes), 3),
        "std_mape":    round(np.std(mapes), 3),
        "min_mape":    round(np.min(mapes), 3),
        "max_mape":    round(np.max(mapes), 3),
        "n_folds":     len(fold_results),
        "folds":       fold_results,
    }
